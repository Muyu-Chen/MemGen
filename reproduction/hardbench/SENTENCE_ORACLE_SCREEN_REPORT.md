# Depth-3 Sentence Oracle Screen

**结论：7 个非截断 prompt-only 失败样本上，三步 sentence-level oracle 的 exact rescue rate 为 0/7；56 条策略均未得到正确答案。**

## 设计

对冻结十题 pilot 中的 prompt-on / inference-off (`N`) 结果，排除两道已正确题和一条 512-token 截断题 `0810`，在其余 7 个错误样本上穷举前三个动态 sentence-level 决策的 `2^3` invoke/skip 策略。Prompt augmentation 始终开启；每次 invoke 后，后续边界沿新轨迹重新识别。

| sample | gold | N prediction | unique answers | changed completions | best error | exact rescue |
|---|---:|---:|---:|---:|---:|---:|
| `gsm8k_test_0423` | 8 | 9.33 | 3 | 7/8 | 1.33 | 0/8 |
| `gsm8k_test_0063` | 1596 | 966 | 1 | 7/8 | 630 | 0/8 |
| `gsm8k_test_1088` | 30 | 26.67 | 2 | 7/8 | 3.33 | 0/8 |
| `gsm8k_test_0611` | 1450000 | 6003125 | 3 | 7/8 | 2425000 | 0/8 |
| `gsm8k_test_0976` | 540 | 750 | 3 | 6/8 | 105 | 0/8 |
| `gsm8k_test_0754` | 89 | 67 | 5 | 5/8 | 12 | 0/8 |
| `gsm8k_test_1161` | 170 | 140 | 1 | 0/8 | 30 | 0/8 |

## 关键观察

- `0063` 的 8 条策略全部预测 `966`，而 A/S 原先正确预测 `1596`；前三个 inference 调用决策无法逆转 prompt memory 已造成的错误。
- `1161` 的 8 条策略全部预测 `140`，即使修复原 token-id detector 漏掉句末的问题，Weaver 仍未提供可见救援路径。
- `0976` 最佳策略把 `750` 改为 `645`，更接近 gold `540`，但 numeric closeness 不是 exact-positive 标签。
- `0423` 与 `1088` 出现显著新轨迹，却只产生新的错误答案；Weaver 能改变 reasoning，不等于能修复 reasoning。
- `0611` 与 `0754` 已另行完成深度 5 搜索，仍为 0/32 exact rescue。

## 责任切分

在这批样本和当前 checkpoint 上，数据不支持“只要训练一个更会选前三个句级时机的 Trigger 就能得到论文收益”。因为对每个失败样本，拥有答案信息的 oracle 都找不到 exact-positive 三步策略。当前证据把主要排查方向推向 Weaver 能力、prompt augmentation，以及 checkpoint / 论文配置差异。

这仍不是任意深度策略的全称证明：筛选层只覆盖前三个动态决策；只有 `0611/0754` 已扩展到五步。

## 完整性

- 样本：7；策略：56；完整到达三步：56/56
- exact-correct policies：0
- 截断：0；误接纳 numeric/internal delimiter：0
- 本筛选生成耗时：4395.5 秒

逐策略 completion、答案、边界前缀与 augmentation mask 保存在各 run 的 scored JSONL；汇总 JSON 保留所有 source hashes。
