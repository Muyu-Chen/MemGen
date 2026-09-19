# MemGen 复现

本项目是 [MemGen](https://github.com/bingreeky/MemGen) (ICLR 2026) 的复现 fork。

原始 README: [README-Origin.md](./README-Origin.md)

## 目录结构

```
.
├── memgen/                  # MemGen 源码（来自原始仓库）
├── configs/                 # 训练 & 评估配置
├── scripts/                 # 训练 & 评估脚本
├── main.py                  # 入口文件
├── requirements.txt         # Python 依赖
├── reproduction/            # 复现脚本 & 日志
│   ├── ab_compare.py        # A/B 对比：基础模型 vs MemGen
│   ├── chat_base_model.py   # 交互对话：仅基础模型
│   ├── chat_memgen.py       # 交互对话：MemGen（基础模型 + Weaver）
│   ├── test_cpu.py          # CPU 端到端冒烟测试
│   ├── test_ab.py           # 早期 A/B 测试
│   ├── REPRODUCTION.md      # 完整复现日志
│   ├── SETUP_REPORT.md      # 环境配置报告
│   ├── ENV_NOTES.md         # 环境注意事项
│   └── first-step.md        # 复现计划
└── models/                  # （需单独下载，见下文）
    ├── Qwen2.5-1.5B-Instruct/
    └── memgen-checkpoints/
```

## 环境配置

```bash
# 推荐 Python 3.11+
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
# 额外安装：tensorboard 是必需的但未列在 requirements.txt 中
pip install tensorboard
```

### 核心依赖

| 包 | 版本 |
|---|---|
| torch | 2.7.1（GPU 需 CUDA 版本） |
| transformers | 4.55.4 |
| accelerate | 1.10.1 |
| peft | 0.17.1 |
| trl | 0.21.0 |
| datasets | 4.0.0 |

## 模型下载

### 1. 基础模型

基础模型 `Qwen/Qwen2.5-1.5B-Instruct` 在运行脚本时会从 HuggingFace Hub **自动下载**，无需手动操作。

如需手动下载（如离线使用）：

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
```

如使用本地模型，需修改评估脚本中的 `REASONER_MODEL`、`WEAVER_MODEL`、`TRIGGER_MODEL` 指向本地路径。

### 2. MemGen 检查点

从 [Kana-s/MemGen](https://huggingface.co/Kana-s/MemGen/tree/main) 下载并放置到 `models/memgen-checkpoints/`：

```bash
# 示例：GSM8K weaver-sft 检查点
# 下载地址：https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft

# 最终目录结构应为：
# models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model/
#   ├── projs.bin
#   ├── weaver.bin
#   ├── trigger.bin
#   ├── config.json
#   ├── weaver/weaver/adapter_model.safetensors
#   └── trigger/trigger/adapter_model.safetensors
```

### 可用检查点

| 基础模型 | 数据集 | 方法 | HF 链接 |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | GSM8K | weaver-sft | [链接](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft) |
| Qwen2.5-1.5B-Instruct | GSM8K | weaver-grpo | [链接](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/gsm8k/weaver-grpo) |
| Qwen2.5-1.5B-Instruct | KodCode | weaver-sft | [链接](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/kodcode/weaver-sft) |
| Qwen2.5-1.5B-Instruct | TriviaQA | weaver-sft | [链接](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft) |
| SmolLM3-3B | KodCode | weaver-sft | [链接](https://huggingface.co/Kana-s/MemGen/tree/main/SmolLM3-3B/kodcode/weaver-sft) |
| SmolLM3-3B | TriviaQA | weaver-sft | [链接](https://huggingface.co/Kana-s/MemGen/tree/main/SmolLM3-3B/triviaqa/weaver-sft) |

## 运行评估

```bash
# GSM8K 评估（Qwen2.5-1.5B-Instruct, weaver-sft）
bash scripts/eval/qwen2_5_gsm8k_sft.sh

# 其他数据集/方法 — 见 scripts/eval/
```

修改脚本中的 `CUDA_VISIBLE_DEVICES` 来控制使用的 GPU。

## 交互对话

两个脚本用于直观对比 — 输入问题，获取回答：

```bash
# 先激活虚拟环境
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 1. 仅基础模型（Qwen2.5-1.5B-Instruct，无 MemGen）
python reproduction/chat_base_model.py

# 2. MemGen（基础模型 + Weaver 潜在记忆，GSM8K 检查点）
python reproduction/chat_memgen.py
```

两个脚本均支持 `--max-new-tokens`、`--temperature`、`--top-p` 参数。MemGen 脚本还支持 `--checkpoint` 指定不同检查点。

输入 `clear` 重置对话历史，`quit` 退出。

### 推荐测试题目

以下题目可有效展示 Base Model 与 MemGen 的差异（MemGen 使用 GSM8K 检查点）：

**简单题（Base 可能也能做对）：**
```
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
```

**中等题（Base 容易出错）：**
```
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
```

**复杂题（Base 大概率失败）：**
```
Josh has a piggy bank. He starts with $10 in the bank. Every week, he adds $5 to the piggy bank. After 4 weeks, he takes out half of the money to buy a new game. The next week, he adds $5 again. How much money is in the piggy bank now?
```

**多步推理（最能看出差异）：**
```
There are 20 students in a class. 3/5 of the students are girls. If 2/3 of the girls and all of the boys are on the honor roll, how many students are on the honor roll?
```

建议先跑最后一题，Base Model 大概率给不出完整答案，MemGen 应能正确推理出 **12 人**。

## 已知问题

### LoRA Adapter 名称不匹配（已修复）

**根本原因**：官方检查点保存 LoRA 权重时使用 `adapter_name="default"`（键名以 `.lora_A.weight` 结尾），但 `MemGenModel.__init__` 创建 PeftModel 时使用 `adapter_name="weaver"` / `"trigger"`（期望键名为 `.lora_A.weaver.weight`）。当 `from_pretrained()` 调用 `PeftModel.from_pretrained(..., adapter_name="weaver")` 时，PEFT 因键名不匹配会**静默跳过全部 112 个 LoRA 权重**。

**症状**：模型加载无报错，但 weaver/trigger 的 LoRA 权重仍为随机初始化。输出与基础模型完全一致 — 潜在记忆没有任何效果。

**修复方案**：将 `modeling_memgen.py:from_pretrained()` 中的 `PeftModel.from_pretrained()` 替换为手动加载 safetensors + 键名重映射：

```
.lora_A.weight → .lora_A.weaver.weight  （weaver adapter）
.lora_B.weight → .lora_B.weaver.weight
.lora_A.weight → .lora_A.trigger.weight （trigger adapter）
.lora_B.weight → .lora_B.trigger.weight
```

修复代码位于 `memgen/model/modeling_memgen.py`（函数 `_remap_lora_adapter_key` 和 `_load_lora_adapter_into_model`）。

**验证**：`reproduction/ab_compare.py` 确认修复后 MemGen 输出与基础模型不同（GSM8K 上从通用步骤变为正确解答）。

### flash_attention_2

`modeling_memgen.py` 硬编码了 `attn_implementation="flash_attention_2"` 和 `torch_dtype=torch.bfloat16`，需要：
- NVIDIA Ampere 或更新架构的 GPU
- 已安装 CUDA 兼容的 `flash-attn` 包

CPU 或较旧的 GPU 需要修改 `from_config()` 中的 attention 实现。

## CPU 验证结果

在 Intel 8840H（无 GPU）上使用 2 道 GSM8K 题目测试：

| | 基础模型 | MemGen |
|---|---|---|
| 输出（100 tokens） | 通用步骤，无答案 | 正确解答 |
| Q1 | 仅列出步骤 | $18 |
| Q2 | 仅列出步骤 | \boxed{3} |

**结论**：MemGen 检查点加载成功，Weaver 潜在记忆有效改变了 Reasoner 的行为。

详见 `reproduction/REPRODUCTION.md`。
