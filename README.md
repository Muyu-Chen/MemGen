# MemGen Reproduction

This is a fork of [MemGen](https://github.com/bingreeky/MemGen) (ICLR 2026) for reproduction purposes.

Original README: [README-Origin.md](./README-Origin.md)

## Directory Structure

```
.
├── memgen/                  # MemGen source code (from original repo)
├── configs/                 # Training & eval configs
├── scripts/                 # Training & eval shell scripts
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── reproduction/            # Reproduction scripts & logs
│   ├── ab_compare.py        # A/B comparison: Base Model vs MemGen
│   ├── test_cpu.py          # CPU end-to-end smoke test
│   ├── test_ab.py           # Earlier A/B test variant
│   ├── REPRODUCTION.md      # Full reproduction log
│   ├── SETUP_REPORT.md      # Environment setup report
│   ├── ENV_NOTES.md         # Environment notes & gotchas
│   └── first-step.md        # Reproduction plan
└── models/                  # (download separately, see below)
    ├── Qwen2.5-1.5B-Instruct/
    └── memgen-checkpoints/
```

## Environment Setup

```bash
# Python 3.11+ recommended
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
# Additional: tensorboard is required but not listed in requirements.txt
pip install tensorboard
```

### Key Dependencies

| Package | Version |
|---------|---------|
| torch | 2.7.1 (CUDA version for GPU) |
| transformers | 4.55.4 |
| accelerate | 1.10.1 |
| peft | 0.17.1 |
| trl | 0.21.0 |
| datasets | 4.0.0 |

## Model Download

### 1. Base Model

The base model `Qwen/Qwen2.5-1.5B-Instruct` will be **automatically downloaded** from HuggingFace Hub when running scripts. No manual download needed.

If you prefer to download manually (e.g., for offline use):

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/Qwen2.5-1.5B-Instruct
```

If using a local copy, update `REASONER_MODEL`, `WEAVER_MODEL`, `TRIGGER_MODEL` in the eval scripts to point to the local path.

### 2. MemGen Checkpoints

Download from [Kana-s/MemGen](https://huggingface.co/Kana-s/MemGen/tree/main) and place in `models/memgen-checkpoints/`:

```bash
# Example: GSM8K weaver-sft checkpoint
# Download from: https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft

# Final directory structure should be:
# models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model/
#   ├── projs.bin
#   ├── weaver.bin
#   ├── trigger.bin
#   ├── config.json
#   ├── weaver/weaver/adapter_model.safetensors
#   └── trigger/trigger/adapter_model.safetensors
```

### Available Checkpoints

| Base Model | Dataset | Method | HF Link |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | GSM8K | weaver-sft | [link](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft) |
| Qwen2.5-1.5B-Instruct | GSM8K | weaver-grpo | [link](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/gsm8k/weaver-grpo) |
| Qwen2.5-1.5B-Instruct | KodCode | weaver-sft | [link](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/kodcode/weaver-sft) |
| Qwen2.5-1.5B-Instruct | TriviaQA | weaver-sft | [link](https://huggingface.co/Kana-s/MemGen/tree/main/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft) |
| SmolLM3-3B | KodCode | weaver-sft | [link](https://huggingface.co/Kana-s/MemGen/tree/main/SmolLM3-3B/kodcode/weaver-sft) |
| SmolLM3-3B | TriviaQA | weaver-sft | [link](https://huggingface.co/Kana-s/MemGen/tree/main/SmolLM3-3B/triviaqa/weaver-sft) |

## Running Evaluation

```bash
# GSM8K eval (Qwen2.5-1.5B-Instruct, weaver-sft)
bash scripts/eval/qwen2_5_gsm8k_sft.sh

# Other datasets / methods — see scripts/eval/
```

Edit `CUDA_VISIBLE_DEVICES` in the script to control which GPU(s) to use.

## Known Issues

### LoRA Adapter Name Mismatch

The official checkpoints save LoRA weights with `adapter_name="default"` (keys end in `.lora_A.weight`), but `from_pretrained()` loads with `adapter_name="weaver"` / `"trigger"` (expects `.lora_A.weaver.weight`). This causes `PeftModel.from_pretrained()` to **silently skip all LoRA weights**.

**Workaround**: Manually load the safetensors and remap the keys:

```python
# .lora_A.weight → .lora_A.weaver.weight
# .lora_B.weight → .lora_B.weaver.weight
```

See `reproduction/ab_compare.py` for a working implementation.

### flash_attention_2

`modeling_memgen.py` hardcodes `attn_implementation="flash_attention_2"` and `torch_dtype=torch.bfloat16`. This requires:
- NVIDIA GPU with Ampere or newer architecture
- CUDA-compatible `flash-attn` package installed

For CPU or older GPUs, you need to modify the attention implementation in `from_config()`.

## CPU Validation Results

Tested on Intel 8840H (no GPU) with 2 GSM8K questions:

| | Base Model | MemGen |
|---|---|---|
| Output (100 tokens) | Generic steps, no answer | Correct solutions |
| Q1 | Outline only | $18 |
| Q2 | Outline only | \boxed{3} |

**Conclusion**: MemGen checkpoint loads successfully and weaver latent memory effectively changes reasoner behavior.

See `reproduction/REPRODUCTION.md` for full details.
