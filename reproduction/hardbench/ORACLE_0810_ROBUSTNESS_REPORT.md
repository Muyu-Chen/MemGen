# `0810` Oracle Rescue Robustness

**结论：发布代码的原 token-ID gate 也能到达该 exact rescue。 首节点 rescue 对后续四个动态决策的所有组合都稳定。**

## 发布版 gate 可达性

| policy | detector | prediction | correct | tokens | actual invokes | horizon |
|---|---|---:|---:|---:|---:|---:|
| `0` | released token-ID | 10737418240 | False | 810 | 0 | True |
| `1` | released token-ID | 310 | True | 205 | 1 | True |

该对照只改变 candidate detection：同一 checkpoint、prompt augmentation、greedy decoding 与 1024-token 上限。若 policy `1` 正确，就能排除“正例只是 decoded suffix 扩展创造的非发布候选”这一替代解释。

## 深度 5 rescue-branch 稳定性

固定首位为 `1`，穷举随后四位，共 16 条动态策略。exact-correct 16/16；完整到达第五决策 16/16；截断 0。

预测分布：`{'310': 16}`。唯一 completion 数：3。

## 解释

这两组不是为了再扩大总体准确率估计，而是验证一个已发现 exact-positive label 的可用性：候选是否与发布实现兼容，以及正例是否会因后续 memory 调用而翻转。它直接决定 `0810` 能否成为后续 Trigger 训练的可信监督样本。

总生成耗时：1523.1 秒；policy-prefix 校验：True。
