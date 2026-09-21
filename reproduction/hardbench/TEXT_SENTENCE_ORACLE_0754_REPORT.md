# Sentence-level Sequential Oracle Report

**结论：句级 sequential oracle 未找到 exact rescue 路径。**

## 实验问题

对 `gsm8k_test_0754`，在每条实时生成轨迹上重新识别论文所述的 delimiter-token sentence-granularity 节点，穷举 invoke/skip 策略。第一次 invoke 改变文本后，后续节点也随新轨迹变化，因此这不是 baseline 固定位置的 subset sweep。Prompt augmentation 始终开启。

边界规则保留论文及发布代码使用的 comma / period / newline delimiter，但排除金额逗号、小数点、未闭合 `<<...>>` 计算区间、空行，以及紧跟未完成算术运算符的标点。普通语义逗号仍是合法节点。

## 覆盖与结果

| 深度 | 配置含义 | 策略数 | 完整到达决策深度 | 唯一完成文本 | exact-correct 策略 | 最佳绝对误差 |
|---:|---|---:|---:|---:|---:|---:|
| 3 | checkpoint 原生最多 3 次 invoke | 8 | 8 | 5 | 0 | 12 |
| 5 | 同权重，诊断上限临时提高到 5 | 32 | 32 | 12 | 0 | 12 |

Gold 为 `89`。3-step no-invoke 预测 `67`；5-step no-invoke 预测 `67`。两条 no-invoke 完成文本完全一致。

### 2^3

- exact-correct policies: `[]`
- 输出相对 no-invoke 改变：5/8
- 最接近 gold 的 policy：`['010']`（绝对误差 12；距离仅作诊断，不当作正确标签）

### 2^5

- exact-correct policies: `[]`
- 其中 checkpoint 原生预算内（invoke 次数 <= 3）：`[]`
- 需要 4–5 次 invoke 的扩展预算路径：`[]`
- 输出相对 no-invoke 改变：26/32
- 最接近 gold 的 policy：`['00010', '00011', '00110', '00111', '01000', '01001', '01010', '01110']`（绝对误差 12；距离仅作诊断，不当作正确标签）

## 能推出什么

在本次严格限定的句级候选规则与深度 3/5 搜索中，即使 oracle 事后知道答案，也没有找到能让当前 Weaver exact-correct 的路径。这明显削弱了“只是 Trigger 没选准时机”的解释，并把排查重点进一步推向 Weaver、prompt memory、checkpoint/论文配置差异。

该结论仍只覆盖一个样本、最多五个动态决策节点；它不是对任意长度策略或任意 candidate generator 的全称证明。

## 完整性检查

- 计划/实际策略：40/40
- 所有已发生的决策前缀均与计划 bit 串一致：True
- 完整到达名义决策深度的策略：40/40（其余策略提前到达生成叶节点）
- 误接纳 numeric/internal delimiter：0
- 原始生成总耗时：3371.5 秒

逐策略预测、边界前缀和调用位置保存在机器可读分析 JSON 与 scored JSONL 中。
