# Environment Deviations from Official requirements.txt

## PyTorch

- **Official**: `torch==2.7.1+cu128` (CUDA 12.8 build)
- **Installed**: `torch==2.7.1+cpu` (CPU-only build)
- **Reason**: 当前机器 (8840H) 无独立 GPU，仅用于环境准备和 CPU sanity check
- **Install command**: `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu`
- **Impact**: 无法进行 GPU 训练，明天需要在有 GPU 的机器上重新安装 CUDA 版本

## Extra Dependencies (not in requirements.txt)

- **tensorboard**: MemGen 的 `memgen/utils.py` 导入了 `torch.utils.tensorboard.SummaryWriter`，
  但 `tensorboard` 未列在官方 requirements.txt 中。已手动安装 `tensorboard==2.21.0`。
