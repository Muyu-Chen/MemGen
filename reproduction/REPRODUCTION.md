# MemGen Reproduction Log

> **最新进展（2026-09-20）**：官方 inference 路径的**执行级全量 trace** 已完成，
> 见 [`wholeProcess/`](./wholeProcess/)。它推翻/修正了本文档与
> `MEMGEN_ARCHITECTURE_EXPLAINED.md` 中的若干描述，改动清单见
> [`wholeProcess/CORRECTIONS_NEEDED.md`](./wholeProcess/CORRECTIONS_NEEDED.md)，
> 已就地改正并在正文标注。受影响最大的是**发现 #6 的一条观察**（见新增的发现 #8）
> 与 **Trigger softmax 数值的配置归属**（见发现 #1 的配置标注）。

## Version Info

- **MemGen commit**: `970cc95af99b5008610e6b281619d181bc9b5ab9`
- **Branch**: main
- **Working tree**: clone 时 clean；**现已包含本地修改**——
  `memgen/model/modeling_memgen.py`（LoRA adapter name remap 修复）与
  `memgen/utils.py`（tensorboard import 改 try/except），详见文末「已修复的 bug」
- **Date cloned**: 2026-09-19

## Environment

- **Python**: 3.11.9
- **venv path**: `E:\MemoryProject\MemGen\Reproduct\.venv`
- **torch**: 2.7.1+cpu (CPU-only, 无 CUDA)
- **transformers**: 4.55.4
- **accelerate**: 1.10.1
- **peft**: 0.17.1
- **trl**: 0.21.0
- **datasets**: 4.0.0
- **huggingface_hub**: 0.36.2

## Machine

- **CPU**: AMD Ryzen 7 8840H
- **GPU**: 无独立 GPU (本次复现全程 CPU)

## Model Paths

- **Base model**: `E:\MemoryProject\MemGen\Reproduct\models\Qwen2.5-1.5B-Instruct`
  - 来源: HuggingFace `Qwen/Qwen2.5-1.5B-Instruct`
  - 大小: 2.9 GB
  - 文件完整: config.json, model.safetensors, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, generation_config.json
- **MemGen checkpoint**: 已下载 (`models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model`)
  - 包含: projs.bin, weaver.bin, trigger.bin, weaver/weaver/adapter_model.safetensors, trigger/trigger/adapter_model.safetensors
  - 配置: prompt_latents_len=8, inference_latents_len=8, max_prompt_aug_num=1, max_inference_aug_num=3
  - LoRA: r=16, lora_alpha=32, target_modules=[q_proj, v_proj]
  - trigger_active=false (weaver-sft 阶段不训练 trigger)

## Smoke Tests Completed

- [x] `import torch` — 2.7.1+cpu
- [x] `import transformers` — 4.55.4
- [x] `import memgen` — 核心模块全部导入成功
- [x] Tokenizer 加载 Qwen2.5-1.5B-Instruct — Qwen2TokenizerFast
- [x] CPU 加载模型 (1543.7M params) 并生成文本 — "Hello! How can I assist you today?"
- [x] **MemGen 完整模型 CPU 测试** (test_cpu.py)
  - 模型加载: 4638.1M 总参数, 6.9M 可训练参数 (LoRA)
  - Forward pass: 12.71s, loss=2.9961
  - Generation: 14 tokens / 6.63s = 2.11 tokens/s
  - 生成结果: "12 + 7 = 19" (正确)
- [x] **A/B 对比测试** (ab_compare.py, 2026-09-19)
  - Base Model (Qwen2.5-1.5B-Instruct) vs MemGen (Base + Weaver LoRA + Latent Memory)
  - LoRA 加载: weaver 112/112 keys matched, trigger 112/112 keys matched
  - 总参数: 4640.3M, 可训练参数 (LoRA + projections): 9.1M
  - 2 道 GSM8K 测试题，输出全部不同 (2/2 DIFFERENT)
  - Base Model: 100 tokens 内只生成笼统步骤，未得出答案
  - MemGen: 两道题均正确解答 (Q1: $18, Q2: \boxed{3})
  - 结论: **MemGen checkpoint 加载成功，weaver latent memory 有效改变了 reasoner 行为**

## Known Issues (CPU 环境限制)

1. **flash_attention_2 不可用**: `modeling_memgen.py:669-671` 硬编码了 `attn_implementation="flash_attention_2"` 和 `torch_dtype=torch.bfloat16`，CPU 上无法使用。测试脚本通过直接加载模型绕过此限制。
2. **正式复现需修改**: 在 GPU 机器上运行时，需要确保 CUDA 环境支持 flash_attention_2，或修改代码使用其他 attention 实现。
3. **LoRA adapter name 不匹配**: 官方 checkpoint 的 LoRA 权重以 adapter_name="default" 保存 (keys: `...lora_A.weight`)，但 `from_pretrained` 使用 adapter_name="weaver"/"trigger" 加载 (keys: `...lora_A.weaver.weight`)。`PeftModel.from_pretrained` 会静默跳过不匹配的 key，导致 LoRA 权重未加载。A/B 测试脚本通过手动 remap key 解决此问题。在 GPU 上通过官方 `from_pretrained` 加载时可能也有此问题，需要验证。

## Official Scripts

- **Train**: `MemGen/scripts/train/qwen2_5_gsm8k_sft.sh`
- **Eval**: `MemGen/scripts/eval/qwen2_5_gsm8k_sft.sh`
- 两者均使用 `accelerate launch` + `configs/zero2.yaml` + `configs/latent_memory/gsm8k.yaml`

## Issues & Notes

1. **tensorboard 缺失**: requirements.txt 未列出但代码需要，已手动安装
2. **torch CPU-only**: 明天正式复现需在有 GPU 的机器上安装 CUDA 版本
3. **MemGen checkpoint 未下载**: eval 脚本中的 `LOAD_MODEL_PATH` 指向 HuggingFace Hub 路径，需明天确认是否需要提前下载
4. **NCCL 相关环境变量**: 官方脚本设置了 NCCL 相关变量，CPU 机器上无影响

## Tomorrow's First Steps

1. 确认 GPU 机器环境 (CUDA / GPU 型号 / 显存)
2. 安装 CUDA 版 torch: `pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128`
3. 下载 MemGen 官方 weaver-sft checkpoint
4. 运行 eval 脚本: `bash scripts/eval/qwen2_5_gsm8k_sft.sh`

## Phase 1: Trigger 三组对照实验 (2026-09-19)

### 命名约定（重要）

本 checkpoint 来自 **weaver-sft 阶段**，该阶段官方配置为：

```bash
# scripts/train/qwen2_5_gsm8k_sft.sh:49,52,53
model.trigger.active False
run.train_weaver True
run.train_trigger False
```

即 **Trigger 在此阶段完全没有被训练**。因此下文一律使用：

- **`checkpoint_trigger`** —— 指 checkpoint 中附带的、**未经 trigger 阶段训练**的 Trigger
  （`trigger.active=True`，跑真实 LoRA + `output_layer`）

**不使用** "trained Trigger" 这一称呼，因为它会误导读者以为 Trigger 经过了训练。
本复现的所有 Trigger 观测结果，**都不能**用于推断经过 `trigger_train` 阶段训练后的 Trigger 行为。

### 实验设计

在 CPU 环境下运行三组对照实验，分离 Weaver 与 Trigger 的贡献：

| 模式 | 实现方式 | 含义 |
|------|---------|------|
| **always_0** | monkey-patch trigger 强制输出 0 | 完全不增强（≈ Base Model + 格式差异） |
| **checkpoint_trigger** | `trigger.active=True`，用 checkpoint 权重 | checkpoint 附带的未训练 Trigger 的真实决策 |
| **always_1** | `trigger.active=False`（`trigger.py:37-40` 硬编码 `logits[...,1]=1.0`） | 总是增强 |

测试集：20 道简单数学题（2+3, 10-4, 3*5 等）

### 关键发现

#### 1. checkpoint_trigger 在全部 candidate 上输出 1

> **⚠️ 配置标注（必读）**：下列 softmax 数值来自 **`trigger.active=True`** 的
> instrumented 运行，即真实跑了 LoRA + `output_layer` 前向。
> **官方 eval 配置是 `active=False`**（`scripts/eval/qwen2_5_gsm8k_sft.sh:19`），
> 此时 `trigger.py:37-40` 直接返回常量 `logits=[0.0, 1.0]`，
> softmax 恒为 **`[0.268941, 0.731059]`**，且 `trigger.model` **零次 forward**。
> 两套数字不可混用——详见下方「发现 #6」与
> [`MEMGEN_ARCHITECTURE_EXPLAINED.md`](./MEMGEN_ARCHITECTURE_EXPLAINED.md) 的「Trigger 的两种模式」。

```
checkpoint_trigger 统计（22 个 candidate，active=True）:
- Decision=0: 0 次 (0%)
- Decision=1: 22 次 (100%)
- Softmax P(augment=1): 0.968 ~ 0.999, 中位数 0.989
```

对照：官方 `active=False` 路径下（`wholeProcess/` 全量 trace，3 次 Trigger 调用）：

```
logits（最后位置）= [0.0, 1.0]      <- 硬编码常量，与输入无关
softmax            = [0.268941, 0.731059]
argmax             = 1
trigger.model forward 次数 = 0      <- module census 实测
```

**准确的结论**：

> weaver-sft checkpoint 中附带的 Trigger，在本 probe 的 22 个 candidate 上全部输出 class 1。

**不能得出的结论**：

> ~~"Trigger 训练后退化为 always-1"~~ —— 该 Trigger 从未被训练，此现象反映的是
> **随机初始化分类头**的行为，与 trigger 训练阶段的收敛性无关。

#### 2. Trigger 权重处于初始化状态的证据

两条相互独立的证据，均可直接复算：

**证据 A：LoRA `lora_B` 全零**

LoRA 的 `lora_B` 矩阵按 PEFT 约定初始化为**全零**（保证训练开始时 LoRA 增量为 0）。
因此 `lora_B` 是否全零可直接判定该 adapter 是否被训练过：

```
weaver/weaver/adapter_model.safetensors:
  lora_A 全零: 0/56      lora_B 全零: 0/56   -> Weaver LoRA 已训练

trigger/trigger/adapter_model.safetensors:
  lora_A 全零: 0/56      lora_B 全零: 56/56  -> Trigger LoRA 完全未训练
```

**证据 B：`output_layer` 处于随机初始化边界**

`trigger.bin` 中的分类头：

```
output_layer.weight  shape (2, 1536)  std=0.01458  absmax=0.02551
output_layer.bias    shape (2,)       std=0.01392  absmax=0.02405
```

`nn.Linear(1536, 2)` 的默认 kaiming-uniform 初始化边界为 $1/\sqrt{1536}=0.02551$。
实测 `absmax` **恰好等于** 0.02551，即权重仍贴着初始化边界，未经梯度更新。

**旁证：Weaver 的非 LoRA 参数也仍是初始值**

```
weaver.bin:
  prompt_query_latents     std=0.998   <- ≈ torch.randn 初始化
  inference_query_latents  std=1.001   <- ≈ torch.randn 初始化
  prompt_latent_ln.weight  全为 1.0    <- LayerNorm 初始化
  prompt_latent_ln.bias    absmax=0.002 <- ≈ 0，LayerNorm 初始化
  prompt_latent_scale      = 1.0       <- 初始化值
```

这说明该 checkpoint 中**唯一被训练过的组件是 Weaver 的 LoRA adapter**。

#### 3. Weaver 改变输出格式

```
always_0 (无增强):
  'To solve the math problem 10-4, we need to subtract 4 from 10.
   10 - 4 = 6
   Therefore, the final answer is 6.'
  格式: 自然语言，分步说明

checkpoint_trigger / always_1 (有增强):
  '10-4=<<10-4=6>>6'
  格式: GSM8K rationale 风格（含 <<...>> 计算器标注），极度简洁
```

**结论**: Weaver 增强把输出从"自然语言解释"切换为"GSM8K 训练数据的 rationale 风格"。
这是**风格迁移**，不是推理能力提升。

#### 4. 简单数学 20 题：自动化准确率 vs 语义准确率

| 模式 | 自动化准确率（`\boxed{}` 提取） | **语义准确率**（人工/正则核对原始输出） | 总增强次数 |
|------|------------------------------|-----------------------------------|-----------|
| always_0 | 3/20 (15%) | **20/20 (100%)** | 0 |
| checkpoint_trigger | 19/20 (95%) | **20/20 (100%)** | 22 |
| always_1 | 19/20 (95%) | **20/20 (100%)** | 22 |

**语义准确率核对方法**：对每题原始输出用正则 `(?<![\d.])GT(?!\d)` 检索 GT 数字，
20 题 × 3 模式共 60 条输出，**语义失败数为 0**。

**关键洞察**:
- always_0 的 15% 纯属格式问题：答案全对，只是不写 `\boxed{}`
- checkpoint_trigger 的 19/20 也是格式问题：easy_02 输出 `'10-4=<<10-4=6>>6'`，
  答案正确但无 `\boxed{}`，提取器返回 null
- 三组语义准确率**完全相同（20/20）**
- checkpoint_trigger ≡ always_1（**字节级完全相同**），符合 Trigger 未经训练、
  等价于硬编码 always-1 的预期
- **对简单题，Weaver 只改变输出格式，不改变推理能力**

**格式差异示例（同一题 easy_01, GT=5）**:
- always_0: `The answer is 5.`
- checkpoint_trigger / always_1: `2+3 is 5.\boxed{5}`

#### 5. GSM8K 难题诊断 (5 题，逐题结果)

**测试集**: GSM8K test 前 5 题

标记说明：`auto` = `\boxed{}` 提取匹配；`sem` = 语义正确（人工核对原始输出的最终答案）

| # | 题目 (简) | GT | Base | Weaver-only | Full MemGen |
|---|----------|-----|------|-------------|-------------|
| 0 | Janet's ducks (eggs) | 18 | ✅/✅ 18 | ✅/✅ 18 | ✅/✅ 18 |
| 1 | Robe fiber (bolts) | 3 | ✅/✅ 3 | ✅/✅ 3 | ✅/✅ 3 |
| 2 | Josh flipping house | 70000 | ✅/✅ 70000 | ❌/❌ 150000 | ❌/❌ 150000 |
| 3 | James sprints | 540 | ❌/**✅** 540 | ✅/✅ 540 | ✅/✅ 540 |
| 4 | Wendi chickens (feed) | 20 | ❌/❌ 0 | ✅/✅ 20 | ✅/✅ 20 |

**汇总**:

| 模式 | 自动化准确率 | **语义准确率** | 总 tokens | 平均延迟/题 |
|------|------------|--------------|----------|------------|
| Base | 3/5 (60%) | **4/5 (80%)** | 1323 | 108.6s |
| Weaver-only | 4/5 (80%) | **4/5 (80%)** | 530 | 59.8s |
| Full MemGen | 4/5 (80%) | **4/5 (80%)** | 530 | 63.0s |

**Q3 是自动化指标的假阴性**：Base 输出
`3 sprints/day * 3 days/week * 60 meters/sprint = 540 meters/week. Therefore, the final answer is 540 meters.`
—— 推理与答案完全正确，只是没写 `\boxed{}`，导致提取器返回 null。

**按语义准确率，真实变化是**：

```
Base:         4/5
Weaver-only:  4/5
Full MemGen:  4/5

净变化: 0
```

配对转移（Base → Weaver）：

```
Q2: correct -> wrong     regression（破坏性干扰）
Q3: wrong   -> correct   recovery（仅格式修复，语义本来就对 -> 实际无变化）
Q4: wrong   -> correct   recovery（真实修复）
```

严格按语义算，Q3 在 Base 下本来就是对的，所以真实的一进一出是：

```
Q2 regression  +  Q4 recovery  =>  净 semantic gain = 0
```

**逐题分析**:
- **Q2 (Josh flipping house)**: Base 正确算出利润 = $200,000 − $130,000 = $70,000。
  Weaver-only / Full MemGen 则算 $200,000 − $50,000 = $150,000，**漏减了初始购房成本 $80,000**。
  这是 Weaver 增强**引入的新错误**（destructive interference）。
- **Q3 (James sprints)**: 纯格式差异，三种模式语义都对。
- **Q4 (Wendi chickens)**: Base 推理混乱（凭空假设"总共 30 cups"，算出 30−40=−10 后输出 0）。
  Weaver 正确算出 3×20−(15+25)=20。这是 Weaver 的**真实修复**。

**准确的结论**：

> Weaver augmentation **materially changes the reasoning trajectory**。
> 在这个 5-example probe 中，它修复了一个 Base failure（Q4），
> 同时引入了一个新的 failure（Q2）；**没有观察到净 semantic-accuracy gain**。

**已删除的错误结论**：

> ~~"Weaver 对难题有实质帮助 (+20%)"~~ —— 该 +20% 完全来自 Q3 的格式假阴性，
> 按语义准确率为 4/5 → 4/5。

> ~~"性能提升 100% 来自 Weaver 架构本身"~~ —— 语义准确率净变化为 0，不存在"提升"。

这个"一进一出"的结果其实**比 +20% 更有研究价值**，因为它正对应
recovery vs destructive interference 这一核心问题：latent memory 并非单调有益，
它可能修好一道题、同时弄坏另一道。

Weaver-only ≡ Full MemGen（**字节级完全相同**），
与 Trigger 未经训练、等价于硬编码 always-1 的事实一致。

#### 6. Trigger 详细日志 (GSM8K 5 题，`active=True`)

此日志由 `trigger_instrumentation.py` 在 `trigger.active=True` 下采集，
因此记录的是**未训练分类头的真实 logits**（而非 `active=False` 的硬编码值）。

| # | 题目 | Trigger 调用次数 | 增强位置 | 所有 P(augment=1) |
|---|------|----------------|---------|------------------|
| 0 | Janet's ducks | 4 | [12, 68, 84] | 0.971, 0.992, 0.968, 0.997 |
| 1 | Robe fiber | 3 | [16, 40] | 0.986, 0.995, 0.861 |
| 2 | Josh house | 4 | [19, 50, 54] | 0.993, 0.886, 0.635, 0.988 |
| 3 | James sprints | 1 | [] | 0.973 |
| 4 | Wendi chickens | 2 | [13] | 0.998, 0.997 |

**观察**:
- 全部 14 次调用决策均为 1（augment），无一例外
- 最低置信度: Q2 第 3 次调用 P=0.635（仍选择 augment）
- Prompt 位置 (i=0) 的置信度普遍较高 (>0.97)
- ~~增强通常发生在句子边界（delimiter 位置）~~ —— **此条已推翻，见下方「发现 #8」**

**解读注意**：这些 logits 来自**随机初始化的 `output_layer`**（见发现 #2 证据 B）。
一个随机线性头作用在 Qwen hidden states 上，恰好在这 14 个样本上都偏向 class 1。
这是**未训练权重的偶然行为**，不构成关于 Trigger 学习动力学的任何证据。

#### 7. always_0 控制验证

**实验目的**: 验证 always_0 的 15% 自动化准确率是格式问题还是能力问题。

**方法**: 用纯 Base Model (Qwen2.5-1.5B-Instruct) 直接跑同样 20 道简单数学题，
与 always_0 逐题对比。

**结果**:
- Base Model (`reasoner.generate()`): 20/20 = 100% 正确
- always_0 (monkey-patch): 语义 20/20 = 100% 正确
- 两者答案一致，都是自然语言格式（无 `\boxed{}`）

**结论**: always_0 的推理能力完好，15% 自动化准确率纯粹是格式匹配问题。

**踩坑记录**：该验证脚本最初因两处问题失败，已在后续脚本中修正——
1. 模型路径含 `=` 字符（`pn=1_pl=8_in=3_il=8`），被 HF Hub 当作 repo id 校验而抛
   `HFValidationError`；改为 `os.path.join(_PROJECT_ROOT, "models/...")` 绝对路径。
2. 误用不存在的 `MemGenForCausalLM`；正确类名是 `MemGenModel`。

#### 8. 增强点不在句子边界，而在「独立成 token 的 delimiter」之后（推翻发现 #6 的一条观察）

来源：`wholeProcess/` 的官方路径全量 trace（GSM8K `test[0]`，111 个生成步）。
证据文件 `wholeProcess/logs/delimiter_check.txt`。

**机制**：`_check_ends_with_delimiter`（`modeling_utils.py:153-175`）**不 decode、不做字符串比较**，
只取最后一个非 pad token 的 **id**，判断是否属于 `_get_delimiter_token_ids`
（`modeling_utils.py:145-151`）预先算好的集合。Qwen2.5 tokenizer 下该集合 = **{11, 13, 198}**
（`,` / `.` / `\n` 各自单独成 token 时的 id）。

**实测**：111 步中最后一个 token 的 id 分布

| id | 字符 | 命中次数 |
|---|---|---|
| 11 | `,` | **2** |
| 13 | `.` | **0** |
| 198 | `\n` | **0** |

而**文本里确实含句号/换行、却因 BPE 合并而漏检**的有 3 步：

```
step  38  id= 624  tok='.Ċ'    <- "." + "\n" 合并成一个 token
step  66  id= 624  tok='.Ċ'    <- 同上
step 105  id=7110  tok='.\'    <- "." + "\" 合并（\boxed 之前）
```

**结论**：

- 本次运行的 2 次 inference augmentation **都发生在逗号之后**，三处真正的句子边界**一次都没触发**。
- 发现 #6 里「增强通常发生在句子边界」这条观察**不成立**：增强点的分布由
  **tokenizer 的合并行为**决定，而不是由语义上的句子结构决定。
  任何与句号粘连的 token（`.\n`、`.\`、`.T`…）都会绕过门控。
- 发现 #6 表格里记录的增强位置（如 Q0 的 `[12, 68, 84]`）应据此重新理解：
  它们是「id 落在 {11,13,198} 的位置」，不是「句子结束的位置」。

> 设计自有模型时，若想让记忆注入真正对齐推理步骤边界，这个判据必须换成显式机制，
> 不能沿用 token-id 集合匹配。

### 架构分析（已由 tensor trace 实证验证）

完整的 MemGen 工作流程解析见 [`MEMGEN_ARCHITECTURE_EXPLAINED.md`](./MEMGEN_ARCHITECTURE_EXPLAINED.md)。
官方 inference 路径的**执行级全量 trace**（module / tensor / parameter 三层，
含 18 条问答与参数普查）见 [`wholeProcess/FULL_EXECUTION_TRACE.md`](./wholeProcess/FULL_EXECUTION_TRACE.md)。

`reproduction/verify_tensor_trace.py` 对 Weaver 模块注册 forward pre-hook，
抓取子模块**实际收到**的张量后做逐元素比对，单题（`"What is 2+3?"`）验证结果 **4/4 PASS**：

| # | 待验证命题 | 结果 | 实测证据 |
|---|-----------|------|---------|
| 1 | Weaver 输入是 Reasoner **input embeddings**，不是 hidden states | PASS | `equals_embedding_table_lookup: True`（与 `embed_weight[input_ids]` 逐元素相同） |
| 2 | 拼接顺序是 `[context, query_latents]` | PASS | `head_is_context: True`, `tail_is_query_latents: True`，layout `[context(40), Q(8)]` |
| 3 | 提取**最后** K 个 hidden states | PASS | `matches_LAST_K: True` |
| 4 | **不是**提取前 K 个 | PASS | `matches_FIRST_K: False` |

trace 实测 shape 链路（prompt_len=40, hidden=1536, K=8）：

```
input_ids                [1, 40]
  -> Embed_R             [1, 40, 1536]   std=0.0249
  -> Proj_R2W            [1, 40, 1536]   std=0.0209
  -> cat([ctx, Q])       [1, 48, 1536]
  -> Weaver forward      [1, 48, 1536]   std=2.6235   <- 量级放大，确属深层 hidden states
  -> 取 [-8:]            [1,  8, 1536]
  -> Proj_W2R            [1,  8, 1536]   std=1.7247
  -> cat([ctx, M])       [1, 48, 1536]   context_retained=True, latent_appended_at_tail=True
  -> Reasoner next token id=17 ('2'),  top5: '2'(0.568) 'The'(0.164) 'Adding'(0.045)
  -> 完整输出: '2+3 is 5.\boxed{5}'   augmentation positions: [0]
```

**四项此前写错、现已修正的理解**：

1. ~~Weaver 读 Reasoner hidden states~~ -> 读的是 **input embeddings**（可见 token 查表结果）
2. ~~concat `[Q, context]`~~ -> 实际是 **`[context, Q]`**（`weaver.py:70`）
3. ~~取前 8 个 hidden states~~ -> 实际取 **最后 8 个**（`weaver.py:92`）
4. 此前遗漏：**每次增强后 KV cache 被清空**，Reasoner 对完整序列重新 forward

第 2、3 点是绑定的：因为 Weaver 是 causal transformer，Q 必须放在末尾才能 attend 到
完整 context；顺序颠倒则 Q 看不到任何 context，机制彻底失效。

> **由此得到的关键定性**：MemGen 的 Weaver 是
> **context-conditioned latent generator**，
> 而**不是** reasoning-hidden-state-conditioned memory updater。
> Weaver 从未接触过 Reasoner 的内部 hidden reasoning state。

### 实验脚本

| 脚本 | 用途 |
|------|------|
| `reproduction/verify_tensor_trace.py` | **单题 tensor trace，实证验证 Weaver 数据流（4/4 PASS）** |
| `reproduction/trigger_three_way_probe.py` | 三组对照实验（always_0 / checkpoint_trigger / always_1） |
| `reproduction/full_memgen_diagnostic.py` | GSM8K 三模式诊断（Base / Weaver-only / Full MemGen） |
| `reproduction/trigger_instrumentation.py` | Trigger 决策 logits 日志 |
| `reproduction/verify_always0_control.py` | always_0 vs 纯 Base Model 控制验证 |
| `reproduction/eval_gsm8k_cpu.py` | CPU 版 GSM8K 评测 |
| `reproduction/sanity_check_bilingual.py` | 中英双语 sanity check |

### 结果数据

| 文件 | 内容 |
|------|------|
| `reproduction/phase1_results/tensor_trace.json` | 单题 tensor trace 全部中间张量统计 |
| `reproduction/phase1_results/trigger_three_way_easy_probe.jsonl` | 三组对照原始输出（20 题 × 3 模式） |
| `reproduction/phase1_results/diagnostic_traces.jsonl` | GSM8K 诊断原始输出（5 题 × 3 模式） |
| `reproduction/phase1_results/trigger_instrumentation.json` | Trigger logits / softmax / aug_mask |

## Bug 修复记录

### 1. LoRA adapter name 不匹配 (严重)

**现象**: MemGen 模型加载后，Weaver 和 Trigger 的 LoRA 权重未生效，生成结果与 Base Model 完全相同。

**原因**: 官方 checkpoint 的 LoRA 权重以 `adapter_name="default"` 保存 (keys: `...lora_A.weight`)，但 `PeftModel.from_pretrained` 使用 `adapter_name="weaver"` / `"trigger"` 加载 (keys: `...lora_A.weaver.weight`)。`PeftModel.from_pretrained` 静默跳过不匹配的 key，导致 LoRA 权重未加载，且不报任何错误。

**修复**: 手动 remap key，将 `lora_A.weight` → `lora_A.weaver.weight`：
```python
def _remap_lora_adapter_key(key, adapter_name):
    key = key.replace("lora_A.weight", f"lora_A.{adapter_name}.weight")
    key = key.replace("lora_B.weight", f"lora_B.{adapter_name}.weight")
    return key
```

**影响**: 如果不修复，A/B 测试会显示"无差异"，误判 MemGen 无效。

**验证证据链**（remap 前后对比，可复算）：

```
remap 前: matched LoRA keys =   0 / 112   -> LoRA 完全未加载
remap 后: matched LoRA keys = 112 / 112   -> 全部加载

加载后行为变化:
  weaver LoRA lora_B 全零数 = 0/56   （确认 adapter 本身是训练过的，非空壳）
  输出行为:  Base 自然语言解释  ->  MemGen rationale 风格 + \boxed{}
  A/B 测试:  2/2 题输出不同
```

即：该 bug 不是"我怀疑"，而是有 key 匹配数 + 权重非零性 + 输出行为变化三重证据。

### 2. flash_attention_2 硬编码

**现象**: CPU 环境无法加载 MemGen 模型。

**原因**: `modeling_memgen.py:669-671` 硬编码了 `attn_implementation="flash_attention_2"` 和 `torch_dtype=torch.bfloat16`。

**修复**: 测试脚本绕过此限制，直接加载模型。正式 GPU 环境需确认 CUDA 支持 flash_attention_2。

### 3. tensorboard 缺失

**现象**: `import memgen` 失败。

**原因**: `memgen/utils.py` 直接 `from torch.utils.tensorboard import SummaryWriter`，但 requirements.txt 未列出 tensorboard。

**修复**: 改为 try-except 可选导入。

## 最终结论

### MemGen 复现结果总结

| 维度 | 结论 |
|------|------|
| **Weaver 机制** | 已完整实证。输入为 Reasoner **input embeddings**（非 hidden states），拼接 `[context, Q]`，取**最后** K 个 hidden states 作为 latent memory |
| **Weaver 是否训练过** | 是。`lora_B` 0/56 全零；非 LoRA 参数（query_latents / LN / scale）仍为初始值 |
| **Weaver 的作用** | 显著改变推理轨迹与输出风格（自然语言 -> GSM8K rationale + `\boxed{}`），并减少输出 token（1323 -> 530） |
| **Weaver 的净准确率增益** | 在 5 题 GSM8K probe 上 **semantic 4/5 -> 4/5，净变化 0**（Q4 recovery，Q2 regression） |
| **Trigger 门控** | 本 checkpoint 的 Trigger **完全未经训练**，`checkpoint_trigger` ≡ `always_1`（字节级相同）。**无法评估** trigger 训练后的真实门控能力 |
| **LoRA 加载** | 有 bug。需手动 remap key（0/112 -> 112/112），否则静默失败 |

### 明确不成立的结论（此前文档写过，现已删除）

1. ~~"Weaver 对难题有实质帮助 (+20%)"~~
   -> 该 +20% 完全来自 Q3 的**格式假阴性**。按语义准确率为 4/5 -> 4/5。
2. ~~"性能提升 100% 来自 Weaver 架构本身"~~
   -> 语义准确率净变化为 0，不存在"提升"可归因。
3. ~~"Trigger 退化为常函数"~~ / ~~"Trained Trigger"~~
   -> 该 Trigger 从未训练（`lora_B` 全零 + `output_layer` 贴初始化边界）。
   观测到的 always-1 是**随机分类头的偶然行为**，不构成训练动力学证据。

### 可以下的结论

> 当前公开 weaver-sft checkpoint 的行为差异来自 **Weaver augmentation**；
> checkpoint 中未经第二阶段训练的 Trigger 在所测样本上没有产生任何 selective behavior。

### 待验证事项

1. **GPU 环境**: 需在 CUDA 环境验证 flash_attention_2 是否正常工作
2. **官方 eval 脚本**: 需确认官方 `from_pretrained` 是否也有 LoRA key 不匹配问题
3. **真正的 Trigger 行为**: 需获取经过 `trigger_train` 阶段的 checkpoint，
   或自行训练 Trigger，才能评估其门控是否有效
4. **更大规模测试**: 当前仅 5 题 GSM8K + 20 题简单数学，
   样本量不足以判定 Weaver 的净增益（5 题下 ±1 题即 ±20%）
5. **recovery vs destructive interference**: Q2/Q4 的一进一出值得在更大样本上量化

---

## Phase 1 完成状态

MemGen reproduction 的**理解性目标已达成**：完整数据流已由 tensor trace 实证验证（4/4 PASS），
组件训练状态已用权重证据确认，性能结论已按语义准确率重新校准。

**2026-09-20 补充**：又完成了一次**官方 inference 路径的执行级全量 trace**
（[`wholeProcess/`](./wholeProcess/)），把理解从「数据流对不对」推进到
「每个 module / tensor / parameter 在这一次前向里到底发生了什么」。三个量化结论：

| 指标 | 实测值 |
|---|---|
| 加载的参数总量 | 4,640,256,516 |
| 本次前向**实际执行**到的 | 2,860,986,370 |
| **加载却从未执行**的 | **1,779,270,146（38.34%）** |
| `requires_grad=True` 的参数量 | 9,113,604 |
| 其中 Trigger LoRA + `output_layer`（带梯度但**零次执行**） | 2,182,146 |
| 单个最大闲置张量 | Weaver `embed_tokens.weight`，233,373,696 |

即：**`requires_grad` 与「是否被用到」两个方向都不相关**——
唯一在做预测的 Reasoner（1,543,714,304 参数、`requires_grad` 全为 False）承担了全部 111 步生成，
而带梯度的 Trigger（base 1.5437 B + LoRA + head）在官方 `active=False` 配置下一次前向都没跑。
判据：「used」= 所属 module 在本次 `generate()` 中执行过 forward，
或该张量对象被 identity 证明被读取（`prompt_query_latents` 这类裸 `nn.Parameter`）。
证据：`wholeProcess/logs/parameter_usage.json`。

按当前计划，不再进行更大规模的性能复现，转入自有模型设计。
自有方向的切入点已由上述架构事实明确：

```
MemGen:   visible context embeddings -> fresh latent -> append
本方向:   H_t^R + M_t -> M_t+1
```

即让 Reasoner 的**内部认知状态**直接更新一个**持久可变的记忆状态**。
MemGen 的 Weaver 从未接触过 Reasoner 的 hidden state，这是真实的结构性差异。

`wholeProcess/` 的 trace 另外暴露出三个同样可切入的空白：

1. **门控与内容脱钩** —— 决定「是否注入」的 Trigger 只看 token id 序列，
   决定「注入什么」的 Weaver 看到的是含 latent 的 embedding 序列，两者视图不同构。
2. **注入点由 tokenizer 决定** —— `.\n` 被 BPE 合并成 id 624 就绕过门控（发现 #8），
   「在推理步骤边界增强」在实现层面并不成立。
3. **append-only、无容量约束** —— latent 只追加不覆写，K=8 个槽位两两余弦 0.72~0.77，
   行 norm 约为 token embedding 的 60 倍，插入后即主导后续表示。
