# MemGen Setup Report

**日期**: 2026-09-19  
**目标**: 完成正式复现前的环境准备 (CPU-only, 8840H 本地机器)

---

## 基本信息

| 项目 | 值 |
|------|-----|
| MemGen commit | `970cc95af99b5008610e6b281619d181bc9b5ab9` |
| Python 版本 | 3.11.9 |
| venv 路径 | `E:\MemoryProject\MemGen\Reproduct\.venv` |
| torch 版本 | 2.7.1+cpu |
| transformers 版本 | 4.55.4 |

## 依赖安装

- **requirements.txt 安装状态**: 全部成功 (torch 替换为 CPU 版本)
- **额外安装**: tensorboard 2.21.0 (requirements.txt 未列出但代码需要)
- **环境偏差详情**: 见 `ENV_NOTES.md`

## 模型

| 项目 | 状态 |
|------|------|
| Qwen2.5-1.5B-Instruct 下载位置 | `models/Qwen2.5-1.5B-Instruct/` |
| 磁盘占用 | 2.9 GB |
| config.json | ✅ 存在 |
| model.safetensors | ✅ 存在 (3.09 GB) |
| tokenizer.json | ✅ 存在 |
| tokenizer_config.json | ✅ 存在 |
| vocab.json / merges.txt | ✅ 存在 |
| generation_config.json | ✅ 存在 |
| **模型文件完整性** | **完整** |

## CPU Sanity Check

| 测试项 | 结果 |
|--------|------|
| `import torch` | ✅ 2.7.1+cpu |
| `import transformers` | ✅ 4.55.4 |
| `import memgen` (核心模块) | ✅ 全部导入成功 |
| Tokenizer 加载 Qwen2.5-1.5B-Instruct | ✅ Qwen2TokenizerFast |
| CPU 加载模型 + 生成文本 | ✅ 生成 "Hello! How can I assist you today?" |

## 官方脚本

| 脚本 | 路径 | 状态 |
|------|------|------|
| GSM8K SFT 训练 | `MemGen/scripts/train/qwen2_5_gsm8k_sft.sh` | ✅ 存在 |
| GSM8K SFT 评估 | `MemGen/scripts/eval/qwen2_5_gsm8k_sft.sh` | ✅ 存在 |

**训练脚本摘要**: 使用 accelerate + DeepSpeed ZeRO-2，三个模型组件 (reasoner/weaver/trigger) 均指向 Qwen2.5-1.5B-Instruct，weaver SFT 训练，prompt/inference latents 长度各为 8。

**评估脚本摘要**: 加载 checkpoint `MemGen/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model`，batch_size=4，temperature=0.0，max_response_length=1024。

## 明天正式复现前待解决

1. **GPU 环境**: 当前机器无 CUDA，需要在有 GPU 的机器上安装 CUDA 版 torch (`torch==2.7.1+cu128`)
2. **MemGen checkpoint**: 评估脚本引用的 `MemGen/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/...` checkpoint 尚未下载，需从 HuggingFace Hub 获取
3. **DeepSpeed / ZeRO-2**: 官方使用 `configs/zero2.yaml`，需确认 GPU 机器上 DeepSpeed 兼容性
4. **bf16 训练**: 训练脚本设置 `bf16=True`，需确认 GPU 支持 bf16 (Ampere 及以上)
5. **wandb 配置**: 可能需要配置 wandb 登录或设置为 offline 模式

## 文件清单

```
E:\MemoryProject\MemGen\Reproduct\
├── .gitignore              # 排除 venv, models, MemGen 等大文件目录
├── .venv/                  # Python 虚拟环境
├── first-step.md           # 原始任务说明
├── ENV_NOTES.md            # 环境偏差记录
├── REPRODUCTION.md         # 复现日志 (详细)
├── SETUP_REPORT.md         # 本报告
├── models/
│   └── Qwen2.5-1.5B-Instruct/  # Base model (2.9 GB)
└── MemGen/                 # 官方仓库克隆 (未修改)
```
