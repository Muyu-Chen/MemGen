# Depth-5 Sentence Oracle Follow-up

**结论：两个深度 3 时曾被 Weaver 推近答案的样本，在完整 `2^5` 动态句级策略搜索中仍然没有 exact rescue（0/64）。**

## 为什么选这两题

`0976` 与 `0423` 是深度 3 筛选中最接近形成正例的两题：调用 Weaver 后数值误差曾缩小，因此它们比输出完全不变的题更适合检验“更多顺序调用是否能跨过最后一步”。Prompt augmentation 始终开启，后续决策点沿每条新轨迹重新识别。

| sample | gold | 00000 | unique answers | 完整到达五步 | exact | 最佳预测 | 最佳误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gsm8k_test_0976` | 540 | 750 | 7 | 32/32 | 0/32 | ['570'] | 30 |
| `gsm8k_test_0423` | 8 | 9.33 | 5 | 8/32 | 0/32 | ['6.67', '9.33'] | 1.33 |

## 观察

- `0976` 的最佳预测从三步搜索的 `645` 改善到 `570`（gold `540`），说明更多调用能继续改变方向，但仍不是 exact-positive。
- `0423` 出现提前终止的分支；这证明第一次调用会改变后续轨迹和可达决策节点，不能把本实验等价成固定 baseline 上选一个 subset。
- 数值更近仅作为诊断信号，评分仍严格使用 GSM8K exact/semantic numeric equality；没有把近似答案算作 rescue。

## 能推出什么

在当前 checkpoint、prompt memory 和句级候选生成规则下，把策略深度从 3 扩到 5 仍未为这两道最有希望的题创造正例。这进一步削弱了“Trigger 只需学会组合更多正确时机”的解释。剩余更合理的排查对象是 Weaver 生成的记忆内容、prompt augmentation 的因果影响，以及发布 checkpoint 与论文配置是否一致。

该结论只覆盖前五个动态决策节点；它不声称穷尽任意长度或其他 delimiter 集合。

## 完整性

- 策略：64/64；exact-correct：0；截断：0
- 完整到达五步：40/64；其余路径提前到达生成叶节点
- 已发生的 policy prefix 全部与计划一致：True
- 误接纳 numeric/internal delimiter：0
- 总生成耗时：6806.8 秒

逐策略的 completion、边界前缀、实际调用次数与答案保存在 scored JSONL；汇总 JSON 记录 source hashes。
