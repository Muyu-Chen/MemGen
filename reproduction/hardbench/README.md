# MemGen HardBench

这个目录提供一个可续跑、可审计的配对评测底座。它的目的不是挑出对
MemGen 有利的题，而是让 Base、风格控制、MemGen 以及后续自有模型在同一批
较难题目上使用同一套输入和评分协议。

## 固定条件

- **A**：Base，官方 GSM8K zero-shot prompt。
- **B**：Base，固定的 3-shot GSM8K rationale 风格控制。
- **D**：MemGen 官方 checkpoint，`trigger_active=false`，checkpoint 自带的
  `max_inference_aug_num=3`。
- **R**：与 D 使用同一 checkpoint；prompt augmentation 仍始终开启，但每个
  inference delimiter 候选点以固定 seed 的 Bernoulli(0.5) 决定是否注入。
  这是随机稀疏／剂量对照，不代表训练后的 Trigger。
- **N**：与 D 使用同一 checkpoint；prompt augmentation 开启，但所有 inference
  候选点都不注入。它用于隔离 prompt memory 与 inference memory 的作用。
- **H**：与 N 相同，但根据预先冻结的 policy 在人工选择的一个候选点注入一次。
  policy 保存候选序号、选择理由和 SHA-256；选择时只看题目与 N 的可见前缀，
  不看 gold 或 N 的评分。

题目来自 GSM8K test，按 gold rationale 中 `<<...>>` 计算标注的数量筛选。
正式试点默认要求至少 6 步（当前 test split 共 87 题）。筛选不使用任何模型
输出，因此不会按 MemGen 表现挑题。最初的链路 smoke test 另保留了 min-4 manifest。

## 原始答案与评分分离

运行脚本只向 `runs/<run>/raw/*.raw.jsonl` 追加原始记录，保存完整 completion、
completion token IDs、prompt、耗时、截断状态和（MemGen）增强位置。运行脚本
不导入评分器，也不会回写已有记录。

`score_results.py` 只读 raw 文件，调用仓库自带的
`data/utils/math_utils.py:compute_score`，并另写 `scored/` 与 `summary.json`。
因此评分规则发生变化时不需要重新生成答案。

## 使用顺序

从 MemGen 源码仓库根目录执行：

```powershell
..\.venv\Scripts\python.exe reproduction\hardbench\prepare_manifest.py

..\.venv\Scripts\python.exe reproduction\hardbench\run_base.py `
  --run-dir reproduction\hardbench\runs\pilot_v1 `
  --max-samples 20 --time-budget-minutes 110 --max-new-tokens 512

..\.venv\Scripts\python.exe reproduction\hardbench\run_memgen.py `
  --run-dir reproduction\hardbench\runs\pilot_v1 `
  --max-samples 20 --time-budget-minutes 110 --max-new-tokens 512

..\.venv\Scripts\python.exe reproduction\hardbench\run_memgen.py `
  --condition random50 --run-dir reproduction\hardbench\runs\pilot_v1 `
  --max-samples 20 --time-budget-minutes 110 --max-new-tokens 512

..\.venv\Scripts\python.exe reproduction\hardbench\score_results.py `
  --run-dir reproduction\hardbench\runs\pilot_v1
```

重复相同命令会根据 `(condition, sample_id)` 跳过已有记录，可以安全续跑。

## 人工单点 Trigger 诊断

`trigger_intervention_v1` 只使用试点中 D 失败且至少有一个 inference 候选点的
7 道题。先运行 N 并在不评分的情况下冻结
`policies/human_single_v1.json`，再运行 H：

```powershell
$sampleIds = @(
  'gsm8k_test_0710', 'gsm8k_test_0063', 'gsm8k_test_0423',
  'gsm8k_test_1088', 'gsm8k_test_0611', 'gsm8k_test_0976',
  'gsm8k_test_0754'
)

..\.venv\Scripts\python.exe reproduction\hardbench\run_memgen.py `
  --condition no_inference `
  --run-dir reproduction\hardbench\runs\trigger_intervention_v1 `
  --sample-ids $sampleIds --max-samples 7

..\.venv\Scripts\python.exe reproduction\hardbench\run_memgen.py `
  --condition scheduled_single `
  --policy-file reproduction\hardbench\policies\human_single_v1.json `
  --run-dir reproduction\hardbench\runs\trigger_intervention_v1 `
  --sample-ids $sampleIds --max-samples 7
```

完整样本列表、人工选择、逐题结果与局限见 `TRIGGER_INTERVENTION_REPORT.md`。

## 单点候选穷举

`single_candidate_sweep_v1` 对三道代表题的 N 轨迹候选点逐一做单次注入：
`0710` 7 点、`0611` 36 点、`0754` 4 点，共 47 条反事实轨迹。计划会先核对
N 原始事件中的候选数；运行器以 `(sample_id, candidate_ordinal)` 续跑，并验证每条
轨迹恰好注入一次：

```powershell
..\.venv\Scripts\python.exe reproduction\hardbench\run_trigger_sweep.py `
  --run-dir reproduction\hardbench\runs\single_candidate_sweep_v1 `
  --time-budget-minutes 110 --max-new-tokens 512

..\.venv\Scripts\python.exe reproduction\hardbench\score_results.py `
  --run-dir reproduction\hardbench\runs\single_candidate_sweep_v1

..\.venv\Scripts\python.exe reproduction\hardbench\analyze_trigger_sweep.py `
  --run-dir reproduction\hardbench\runs\single_candidate_sweep_v1

..\.venv\Scripts\python.exe reproduction\hardbench\plot_trigger_sweep.py `
  --analysis reproduction\hardbench\runs\single_candidate_sweep_v1\sweep_analysis.json `
  --output reproduction\hardbench\runs\single_candidate_sweep_v1\trigger_utility_curve.png
```

结论和逐题解释见 `TRIGGER_SWEEP_REPORT.md`。
