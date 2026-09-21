# Trigger 全部证据综合分析

日期：2026-09-20

## 一句话判断

目前的数据**不支持直接训练一个逐候选、独立二分类的 Trigger**；但它提供了更有
价值的机制证据：memory invocation 的效用是非单调、有历史依赖的，当前 delimiter
候选机制还会在同一个原子计算内部连续提出候选，从而制造高阶干扰。

我们的下一版目标应从“预测这个点要不要 invoke”调整为：

> 在语义完整的推理边界上，由一个带调用历史、预算和 verifier 信号的顺序控制器，
> 决定 `HOLD / INVOKE / STOP`。

这不是复现失败。相反，复现已经足够稳定，能够把问题定位到候选生成、Weaver
可救性、prompt memory 和多次调用交互四个具体层面。

## 证据范围

本分析合并四批正式数据：

| 数据 | 规模 | 目的 |
|---|---:|---|
| `pilot_min6_v1` | 10 题 × 4 条件 = 40 | A/S/B/C 总体对照 |
| `trigger_intervention_v1` | 7 题 × N/H = 14 | 人工单点是否优于不触发 |
| `single_candidate_sweep_v1` | 47 | 三道代表题的全部单点效用曲线 |
| `candidate_interaction_v1` | 4 | `0710` 前三候选的二阶/三阶交互 |

合计 105 条已封存生成记录，累计模型生成延迟 9,739.4 秒，约 162.3 分钟。
所有实验使用同一 checkpoint、greedy decoding 和独立评分。以下以 semantic exact
为主指标，因为 Base/风格控制有未使用 `\boxed{}` 但数值正确的回答；官方格式评分
在 A 和 S 上分别只计 2/10 与 1/10，而 semantic 分别为 3/10 与 4/10。

## 1. 十题试点：MemGen 改善格式，但没有改善总体正确率

| 条件 | 讨论别名 | semantic exact | boxed 格式 | 平均 inference 调用 |
|---|---|---:|---:|---:|
| Base zero-shot | A | 3/10 | 8/10 | — |
| Base 3-shot style | S | 4/10 | 4/10 | — |
| MemGen always | B（raw D） | 2/10 | 10/10 | 2.4 |
| MemGen random50 | C（raw R） | 3/10 | 10/10 | 2.1 |

样本级看，B 相对 A 救回 `0810`，但破坏了 `0063` 和 `0710`，semantic 净变化
为 -1。C 相对 B 只救回 `0710`、没有新增退化，净 +1；但 C 与 A 都是 3/10。
S 的 4/10 反而是本批最高，说明至少在这十题上，单纯 rationale 风格控制不弱于
当前 memory 机制。

样本明细：

| sample | gold | A | S | B always | C random50 |
|---|---:|---:|---:|---:|---:|
| `0063` | 1596 | 1596 ✓ | 1596 ✓ | 854 | 966 |
| `0106` | 72 | 72 ✓ | 72 ✓ | 72 ✓ | 72 ✓ |
| `0423` | 8 | 6 | 6 | 0 | 0 |
| `0611` | 1,450,000 | 812 | 7,000,000 | 1,250,000 | 4,062,500 |
| `0710` | 45 | 45 ✓ | 45 ✓ | 44.1 | 45 ✓ |
| `0754` | 89 | 144 | 90 | 87 | 455 |
| `0810` | 310 | 150 | 310 ✓ | 310 ✓ | 310 ✓ |
| `0976` | 540 | 465 | 465 | 645 | 750 |
| `1088` | 30 | 28.33 | 28.33 | 26.67 | 28.33 |
| `1161` | 170 | 140 | 140 | 140 | 140 |

样本只有 10 个，所有配对差异都不足以做显著性结论；它们应被当作机制发现，
不能当作总体效果估计。

## 2. Prompt memory 与 inference Trigger 不是同一个问题

N 保持 prompt augmentation 开启、关闭全部 inference 调用。在从 B 失败题中选出的
7 道可干预题上，N 与人工 H 都为 1/7；H 改变了 6/7 条完整推理和 3/7 个最终
预测，但 exact recovery 为 0、regression 也为 0。

`0063` 尤其关键：A/S 都正确得到 1596，而 N、H 和 C 都得到 966。也就是说错误
在 inference Trigger 采取行动前已经形成，单靠学习 inference 时机无法修复。
因此后续应分别测：是否做 prompt augmentation，以及生成过程中何时调用 memory；
不能默认 prompt memory 永远开启，只训练后半段 Trigger。

## 3. 单点穷举：输出很敏感，但错误题不存在 exact-positive 点

47 个单点变体中，38 个改变了完整 completion；按最终数值到 gold 的距离，24 个
改善、11 个恶化、12 个持平。然而 7 个 exact-correct 变体全部来自 N 本来就正确
的 `0710`。

两道 N 错题共有 40 个单点候选，exact rescue 为 **0/40**：

- `0611`：36 点产生 15 种预测，人工 #1 数值最接近，但仍错误。
- `0754`：#3 从误差 22 改善到 18，仍没有恢复正确换算。

更重要的是，`0611/#1` 的“改善”来自抵消性错误：模型继续把首五个月总 funding
误读成每月 funding，并写出 `750,000 × 5 = 375,000`。这证明数值距离不能作为
Trigger utility label；否则最明显的假改善会被标成强正例。

## 4. 多点交互：最小破坏集合是 `{2,3}`

`0710` 的 N 和全部七个单点变体都得到 45。对前三个候选做完整子集测试后：

| 调用候选集合 | 最终答案 | 正确 |
|---|---:|---:|
| `{}` | 45 | ✓ |
| `{1}` | 45 | ✓ |
| `{2}` | 45 | ✓ |
| `{3}` | 45 | ✓ |
| `{1,2}` | 45 | ✓ |
| `{1,3}` | 45 | ✓ |
| `{2,3}` | 44.1 | ✗ |
| `{1,2,3}` | 44.1 | ✗ |

三点组合的 token IDs、完整 completion 和 augmentation positions 都与官方 always
逐项相同。最小破坏集合因此严格为 `{2,3}`，候选 1 对最终数值没有必要作用。

候选 2 与 3 不是两个独立推理步骤，而是同一个 cinnamon-roll 计算内部的两个
delimiter 事件：一个前缀停在 `4*2.`，另一个停在 `4*2.5 = $<<4*2.`。单独注入
都安全，同时注入则把 `4 × 2.5` 扭成 9，最终得到 44.1。

这是当前最强的因果证据：

- 错误不是“memory 调用次数 ≥ 2”——`{1,2}`、`{1,3}` 都安全。
- 错误也不是候选 2 或 3 单独有害——两个单点都安全。
- 错误是特定候选组合产生的非加性交互。
- 如果候选器禁止在小数和 `<<...>>` 计算 span 内触发，这次退化会直接消失。

## 5. 对我们项目的判断

### 现在不该做的

- 不应立刻用当前候选做独立 `invoke / skip` 二分类训练。
- 不应把“答案数值更接近”当正标签。
- 不应把 prompt augmentation 固定为永远开启。
- 不应仅比较平均调用次数；`{1,2}` 与 `{2,3}` 次数相同、结果相反。

### 值得继续做的

1. **语义候选过滤。** 只在完整句子、完整推理步骤或完整 calculator span 之后开放
   Trigger；禁止在小数、金额逗号和 `<<...>>` 内部触发。
2. **加入 cooldown/step budget。** 同一原子推理步骤最多调用一次，避免 `0710`
   这类局部重复注入。
3. **改成有状态顺序策略。** 输入至少包含已调用次数、上次调用位置、当前 step、
   上次 memory 摘要；输出可设计为 `HOLD / INVOKE / STOP`。
4. **先测可救性。** 对 N 错题穷举单点或少量组合；只有存在 exact-positive 分支的
   样本才提供正 utility 标签。没有正确分支的题应回流给 Weaver，而不是归咎 Trigger。
5. **使用 verifier 标签。** 以 exact final answer 加关键中间关系为核心，不用单纯
   数值距离；必要时区分“改变了输出”和“真正提高了正确性”。
6. **联合控制 prompt memory。** 增加 prompt-off / prompt-on 的配对条件，区分
   “一开始就被 memory 带偏”和“中途调用造成干扰”。

## 6. 建议的下一阶段实验

先做一个很小但信息密度高的 Gate Sanitation v1：

1. 实现 calculator-span / decimal / currency-aware 候选过滤和同 step cooldown。
2. 在原 10 题上补全 prompt-only N（目前只覆盖 7 道 selected failure）。
3. 比较 A、S、N、原 always、过滤后 always、random50，保持同一 manifest。
4. 对过滤后仍错误的题重新做单点 sweep，统计：
   - N 错题中至少存在一个 exact-positive 点的比例；
   - N 正题被 memory 破坏的比例；
   - 多调用最小破坏集合的比例；
   - paired net gain，而不是格式命中率。

只有当过滤后的候选中出现稳定、可验证的 exact-positive signal，再进入 Trigger
训练。若 exact-positive 仍接近零，优先改 Weaver memory 内容和训练目标。

## 可审计产物

- 机器汇总：`analysis/trigger_evidence_v1.json`
- 试点：`runs/pilot_min6_v1/`
- 人工单点：`runs/trigger_intervention_v1/`
- 单点穷举：`runs/single_candidate_sweep_v1/`
- 多点交互：`runs/candidate_interaction_v1/`
- 各阶段详细报告：`PILOT_REPORT.md`、`TRIGGER_INTERVENTION_REPORT.md`、
  `TRIGGER_SWEEP_REPORT.md`
