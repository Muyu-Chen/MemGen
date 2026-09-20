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

#### 4. GSM8K 难题诊断 (5 题)

| 模式 | 准确率 |
|------|--------|
| Base Model | 3/5 (60%) |
| Weaver-only | 4/5 (80%) |
| Full MemGen | 4/5 (80%) |

**结论**: Weaver 对难题有实质帮助 (+20%)，性能提升 100% 来自 Weaver 架构本身。

### 架构分析

完整的 MemGen 工作流程解析见 [`MEMGEN_ARCHITECTURE_EXPLAINED.md`](./MEMGEN_ARCHITECTURE_EXPLAINED.md)，包含：
- 数据流：prompt tokens → reasoner embeddings → Weaver augmentation → projection → insertion → generation
- Trigger 机制：二分类决策器，决定何时插入潜在记忆
- Weaver 增强：8 个 query latents + Reasoner hidden states → Weaver forward → 提取增强 latent → 投影回 Reasoner
- 多次增强：第二次 augmentation 时，第一次的 latent 保留在序列中

### 实验脚本

- `reproduction/trigger_three_way_probe.py`: 三组对照实验
- `reproduction/phase1_results/trigger_three_way_easy_probe.jsonl`: 原始结果
- `reproduction/phase1_results/diagnostic_traces.jsonl`: GSM8K 诊断结果
