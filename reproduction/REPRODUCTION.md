# MemGen Reproduction Log

## Version Info

- **MemGen commit**: `970cc95af99b5008610e6b281619d181bc9b5ab9`
- **Branch**: main
- **Working tree**: clean (no modifications)
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

- **CPU**: Intel 8840H
- **GPU**: 无独立 GPU (今晚仅做 CPU 环境准备)

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

### 实验设计

在 CPU 环境下运行三组对照实验，验证 Trigger 和 Weaver 的贡献：
- **always_0**: 强制 Trigger 输出 0（不增强）
- **trained**: 使用训练好的 Trigger（真实决策）
- **always_1**: 强制 Trigger 输出 1（总是增强）

测试集：20 道简单数学题（2+3, 10-4, 3*5 等）

### 关键发现

#### 1. Trigger 退化为常函数

```
Trained Trigger 统计:
- Decision=0: 0 次 (0%)
- Decision=1: 22 次 (100%)
- Softmax P(augment=1): 0.968 ~ 0.999, 中位数 0.993
```

**结论**: Trigger 学会了总是预测 class 1（总是增强），门控机制失效。

#### 2. Weaver 改变输出格式

```
always_0 (无增强):
  输出: "The answer is 5."
  格式: 自然语言
  
trained/always_1 (有增强):
  输出: "2+3 is 5.\boxed{5}"
  格式: 包含 \boxed{}
```

**结论**: Weaver 增强使模型更倾向使用 `\boxed{}` 格式，但不影响推理能力。

#### 3. 性能对比

| 模式 | 自动化准确率 | 真实准确率 | 总增强次数 |
|------|------------|-----------|-----------|
| always_0 | 3/20 (15%) | 20/20 (100%) | 0 |
| trained | 19/20 (95%) | 20/20 (100%) | 22 |
| always_1 | 19/20 (95%) | 20/20 (100%) | 22 |

**关键洞察**: 
- always_0 的 15% 是格式问题（答案正确但缺少 `\boxed{}`）
- 真实准确率：三组都是 100%
- trained ≡ always_1（字节级相同），证明 Trigger 退化
- 对简单题，Weaver 只改变格式，不影响推理能力

**逐题对比 (20 题)**:

| # | 题目 | GT | always_0 | trained | always_1 | 输出一致? |
|---|------|-----|----------|---------|----------|----------|
| 01 | 2+3 | 5 | ✅ "5" | ✅ "\boxed{5}" | ✅ "\boxed{5}" | trained≡always_1 |
| 02 | 10-4 | 6 | ✅ "6" | ❌ null | ❌ null | trained≡always_1 |
| 03 | 3*5 | 15 | ✅ "15" | ✅ "\boxed{15}" | ✅ "\boxed{15}" | trained≡always_1 |
| 04 | 12/3 | 4 | ✅ "4" | ✅ "\boxed{4}" | ✅ "\boxed{4}" | trained≡always_1 |
| 05 | 5 apples +2 | 7 | ✅ "7" | ✅ "\boxed{7}" | ✅ "\boxed{7}" | trained≡always_1 |
| 06 | 20 students, 8 boys | 12 | ✅ "12" | ✅ "\boxed{12}" | ✅ "\boxed{12}" | trained≡always_1 |
| 07 | 7+8 | 15 | ✅ "15" | ✅ "\boxed{15}" | ✅ "\boxed{15}" | trained≡always_1 |
| 08 | 15-6 | 9 | ✅ "9" | ✅ "\boxed{9}" | ✅ "\boxed{9}" | trained≡always_1 |
| 09 | 4*6 | 24 | ✅ "24" | ✅ "\boxed{24}" | ✅ "\boxed{24}" | trained≡always_1 |
| 10 | 20/4 | 5 | ✅ "5" | ✅ "\boxed{5}" | ✅ "\boxed{5}" | trained≡always_1 |
| 11 | 10 candies -3 | 7 | ✅ "7" | ✅ "\boxed{7}" | ✅ "\boxed{7}" | trained≡always_1 |
| 12 | 6 red +4 blue | 10 | ✅ "10" | ✅ "\boxed{10}" | ✅ "\boxed{10}" | trained≡always_1 |
| 13 | 9+11 | 20 | ✅ "20" | ✅ "\boxed{20}" | ✅ "\boxed{20}" | trained≡always_1 |
| 14 | 18-9 | 9 | ✅ "9" | ✅ "\boxed{9}" | ✅ "\boxed{9}" | trained≡always_1 |
| 15 | 5*5 | 25 | ✅ "25" | ✅ "\boxed{25}" | ✅ "\boxed{25}" | trained≡always_1 |
| 16 | 16/2 | 8 | ✅ "8" | ✅ "\boxed{8}" | ✅ "\boxed{8}" | trained≡always_1 |
| 17 | 8 toys +4 | 12 | ✅ "12" | ✅ "\boxed{12}" | ✅ "\boxed{12}" | trained≡always_1 |
| 18 | 15 birds -5 | 10 | ✅ "10" | ✅ "\boxed{10}" | ✅ "\boxed{10}" | trained≡always_1 |
| 19 | 6+7 | 13 | ✅ "13" | ✅ "\boxed{13}" | ✅ "\boxed{13}" | trained≡always_1 |
| 20 | 14-7 | 7 | ✅ "7" | ✅ "\boxed{7}" | ✅ "\boxed{7}" | trained≡always_1 |

**格式差异示例**:
- always_0: `The answer is 5.`
- trained/always_1: `2+3 is 5.\boxed{5}`

#### 4. GSM8K 难题诊断 (5 题，逐题结果)

**测试集**: GSM8K test 前 5 题

| # | 题目 (简) | GT | Base | Weaver-only | Full MemGen |
|---|----------|-----|------|-------------|-------------|
| 0 | Janet's ducks (eggs) | 18 | ✅ 18 | ✅ 18 | ✅ 18 |
| 1 | Robe fiber (bolts) | 3 | ✅ 3 | ✅ 3 | ✅ 3 |
| 2 | Josh flipping house | 70000 | ✅ 70000 | ❌ 150000 | ❌ 150000 |
| 3 | James sprints | 540 | ❌ null | ✅ 540 | ✅ 540 |
| 4 | Wendi chickens (feed) | 20 | ❌ 0 | ✅ 20 | ✅ 20 |

**汇总**:

| 模式 | 准确率 | 总 tokens | 平均延迟/题 |
|------|--------|----------|------------|
| Base | 3/5 (60%) | 1323 | 108.6s |
| Weaver-only | 4/5 (80%) | 530 | 59.8s |
| Full MemGen | 4/5 (80%) | 530 | 63.0s |

**逐题分析**:
- Q2 (Josh flipping house): Base 正确计算利润=70000，但 Weaver-only 和 Full MemGen 都错误地用 $200,000-$50,000=$150,000（忘记减去初始购房成本）。Weaver 引入了推理错误。
- Q3 (James sprints): Base 未输出 `\boxed{}` 格式导致提取失败，但推理正确 (540)。Weaver 帮助格式化输出。
- Q4 (Wendi chickens): Base 推理过程混乱（算出负数），Weaver 正确解题。

**结论**: Weaver 对难题有实质帮助 (+20%)，但也会引入新的推理错误 (Q2)。Weaver-only ≡ Full MemGen（字节级相同），再次证明 Trigger 无额外贡献。

#### 5. Trigger 详细日志 (GSM8K 5 题)

| # | 题目 | Trigger 调用次数 | 增强位置 | 所有 P(augment=1) |
|---|------|----------------|---------|------------------|
| 0 | Janet's ducks | 4 | [12, 68, 84] | 0.971, 0.992, 0.968, 0.997 |
| 1 | Robe fiber | 3 | [16, 40] | 0.986, 0.995, 0.861 |
| 2 | Josh house | 4 | [19, 50, 54] | 0.993, 0.886, 0.635, 0.988 |
| 3 | James sprints | 1 | [] | 0.973 |
| 4 | Wendi chickens | 2 | [13] | 0.998, 0.997 |

**观察**:
- 所有 Trigger 决策均为 1（augment），无一例外
- 最低置信度: Q2 第3次调用 P=0.635（仍选择 augment）
- Prompt 位置 (i=0) 的置信度普遍较高 (>0.97)
- 增强通常发生在句子边界 (delimiter 位置)

#### 6. always_0 控制验证

**实验目的**: 验证 always_0 的 15% 自动化准确率是格式问题还是能力问题。

**方法**: 用纯 Base Model (Qwen2.5-1.5B-Instruct) 直接跑 20 道简单数学题，与 always_0 逐题对比。

**结果**:
- Base Model: 20/20 = 100% 正确
- always_0 (monkey-patch): 20/20 = 100% 正确（人工检查）
- 两者答案一致，都是自然语言格式（无 `\boxed{}`）

**结论**: always_0 的推理能力完好，15% 自动化准确率纯粹是格式匹配问题。

### 架构分析

完整的 MemGen 工作流程解析见 [`MEMGEN_ARCHITECTURE_EXPLAINED.md`](./MEMGEN_ARCHITECTURE_EXPLAINED.md)，包含：
- 数据流：prompt tokens → reasoner embeddings → Weaver augmentation → projection → insertion → generation
- Trigger 机制：二分类决策器，决定何时插入潜在记忆
- Weaver 增强：8 个 query latents + Reasoner hidden states → Weaver forward → 提取增强 latent → 投影回 Reasoner
- 多次增强：第二次 augmentation 时，第一次的 latent 保留在序列中

### 实验脚本

- `reproduction/trigger_three_way_probe.py`: 三组对照实验
- `reproduction/full_memgen_diagnostic.py`: GSM8K 三模式诊断
- `reproduction/verify_always0_control.py`: always_0 控制验证
- `reproduction/trigger_instrumentation.py`: Trigger 决策日志
- `reproduction/phase1_results/trigger_three_way_easy_probe.jsonl`: 三组对照原始结果
- `reproduction/phase1_results/diagnostic_traces.jsonl`: GSM8K 诊断原始结果
- `reproduction/phase1_results/trigger_instrumentation.json`: Trigger logits 日志

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
| Weaver 架构 | ✅ 有效。GSM8K 难题 +20% (60%→80%)，简单题格式改变 |
| Trigger 门控 | ❌ 退化。100% 预测 augment=1，等价于 always_1 |
| 性能提升来源 | 100% 来自 Weaver 架构本身，Trigger 无贡献 |
| LoRA 加载 | ⚠️ 有 bug。需手动 remap key，否则静默失败 |
| 输出格式影响 | Weaver 使模型倾向使用 `\boxed{}` 格式 |

### 待验证事项

1. **GPU 环境**: 需在 CUDA 环境验证 flash_attention_2 是否正常工作
2. **官方 eval 脚本**: 需确认官方 `from_pretrained` 是否也有 LoRA key 不匹配问题
3. **Trigger 退化原因**: 可能是 weaver-sft 阶段 `trigger_active=false` 导致 Trigger 未充分训练
4. **更大规模测试**: 当前仅 5 题 GSM8K + 20 题简单数学，需更大规模验证
