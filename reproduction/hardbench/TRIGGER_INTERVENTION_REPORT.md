# 人工单点 Trigger 诊断（trigger_intervention_v1）

日期：2026-09-20

## 结论先行

在 MemGen-always 失败且存在 inference 候选点的 7 道现有难题上，人工选择一个
看起来关键的位置调用一次 memory，**没有提高 exact accuracy**：N（只做 prompt
augmentation）和 H（prompt augmentation + 人工单点 inference augmentation）
都是 `1/7`。

H 改变了 `3/7` 个最终答案，说明注入确实产生了因果扰动；但其中只有 1 题的
数值明显靠近 gold、仍未答对，另外 2 题变得更差。因此这轮小样本不支持
“当前 Weaver 在人工关键点大概率有用”。更准确的判断是：**位置与剂量会影响
结果，但现有 memory 内容和候选点机制还不足以提供稳定正收益。**

## 条件映射

为对应讨论中的 A/B/C，这里使用报告别名；括号中是 raw 文件的内部标签：

| 报告名 | 条件 |
|---|---|
| A | Base zero-shot（`A_base_zero_shot`） |
| B | MemGen 每个候选都注入，最多 3 次（`D_memgen_official`） |
| C | MemGen 在候选点 Bernoulli(0.5) 注入（`R_memgen_random50_inference`） |
| S | Base 固定 3-shot 风格控制（`B_base_3shot`，不计入 A/B/C） |
| N | prompt augmentation 开启，inference 从不注入（`N_memgen_no_inference`） |
| H | prompt augmentation 开启，只在人工点位注入一次（`H_memgen_human_single`） |

## 选择与执行协议

1. 从 `pilot_min6_v1` 中取 B（内部 D）失败的 8 题。
2. `gsm8k_test_1161` 没有 inference 候选点，Trigger 无法行动，因此排除。
3. 对剩余 7 题先运行 N，保存每个候选点之前模型可见的完整前缀。
4. 人工只查看题目和这些前缀，选择每题至多一个候选点；此时不运行评分，
   也不查看 gold。
5. 将选择冻结到 `policies/human_single_v1.json` 后运行 H，再统一评分。
6. 事件审计确认 7 题都只注入一次，且唯一 `decision=1` 与 policy 序号一致。

这是一项 post-hoc 机制诊断，不是无偏 benchmark：题目本身按 B 的失败筛选，
所以 B 的 `0/7` 不能拿来估计总体准确率。

## 人工选择与逐题结果

| sample | 人工候选 | 选择理由（简写） | gold | N | H | H 相对 N |
|---|---:|---|---:|---:|---:|---|
| `gsm8k_test_0710` | 7 | 四类商品小计完成，进入折扣阶段 | 45 | 45 ✓ | 45 ✓ | 不变 |
| `gsm8k_test_0063` | 3 | 两个半年分量已具备，进入年度汇总 | 1596 | 966 | 966 | 不变 |
| `gsm8k_test_0423` | 1 | 最早可用纠偏点；此前已开始误分 savings | 8 | 9.33 | 6 | 更差 |
| `gsm8k_test_1088` | 3 | 工作日周总分钟完成，进入周末加时 | 30 | 26.67 | 26.67 | 不变 |
| `gsm8k_test_0611` | 1 | 在把五个月总 funding 当成月 funding 前尽早干预 | 1,450,000 | 6,003,125 | 1,625,000 | 更接近，仍错 |
| `gsm8k_test_0976` | 1 | 解析跨周六/周日数量关系之前 | 540 | 750 | 750 | 不变 |
| `gsm8k_test_0754` | 1 | 第一次 sticker→button 换算开始时 | 89 | 67 | 40 | 更差 |

## 同题条件对比

| 条件 | exact correct | 说明 |
|---|---:|---|
| A | 2/7 | Base zero-shot |
| S | 2/7 | Base 3-shot 风格控制 |
| B | 0/7 | 这些题就是按 B 失败筛出的，不能作总体比较 |
| C | 1/7 | 随机 50%，每题实际注入 1–3 次 |
| N | 1/7 | inference 0 次 |
| H | 1/7 | 人工选择，inference 恰好 1 次 |

最有信息量的是 N↔H 配对，而不是 B 的绝对分数。`0710` 还说明 always-invoke
会破坏原本正确的轨迹：B 得到 44.1，N/H 都得到 45；但 N 与 H 相同，所以
证据支持的是“避免不合适的早期/重复注入”，而不是“这次人工注入带来了额外
正确性”。

## 计算量

- N：7 题，1492 generated tokens，累计生成延迟 525.9 秒。
- H：7 题，1497 generated tokens，累计生成延迟 609.0 秒。
- 合计模型生成延迟约 18.9 分钟；不含模型加载与审计脚本。

两组已经覆盖全部 7 道可干预的现有失败题。greedy 解码下重复相同 N/H 不会
提供新的独立样本，因此在两小时上限前停止，没有为了耗满预算重复运行。

## 局限与下一步含义

- 候选点由当前 delimiter 机制提出；若干候选落在小数或金额中间，而不是自然的
  推理边界。人工只能“从已有候选中选”，无法在真正理想的位置创建候选。
- 有些错误在第一个 inference 候选之前已经由 prompt augmentation 或初始推理
  形成，Trigger 再准确也未必能修复。
- 当前 H 测的是现成 Weaver 在人工时机的效用，不等同于训练完成后的 Trigger，
  更不能证明所有学习型 Trigger 都无效。
- 下一轮最值得做的不是直接扩大题量，而是在少量代表题上穷举每个单点候选，
  画出“位置→效用”曲线；若多数题根本不存在正收益点，应先改 Weaver/候选生成，
  而不是训练 Trigger 去学习一个不存在的信号。

## 可审计产物

- 冻结策略：`policies/human_single_v1.json`
- 原始生成：`runs/trigger_intervention_v1/raw/`
- 独立评分：`runs/trigger_intervention_v1/scored/`
- 汇总：`runs/trigger_intervention_v1/summary.json`
- H 运行配置包含 policy 绝对路径及 SHA-256，原始记录逐题保存候选事件、选中序号、
  人工理由和实际 augmentation 位置。
