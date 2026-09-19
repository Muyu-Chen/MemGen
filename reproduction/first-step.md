官方仓库：

`https://github.com/bingreeky/MemGen`

今晚只完成准备工作，不开始正式训练，也不要修改 MemGen 的算法源码。
```
E:\MemoryProject\MemGen\Reproduct>python --version
Python 3.11.9
```
需要完成：

1. 克隆官方 MemGen 仓库，并记录当前 commit hash。
2. 使用 Python 3.11 创建项目内 `.venv`，安装官方 `requirements.txt`。
3. 检查 PyTorch / CUDA / GPU 是否正常，确保至少能够 import MemGen 所需主要依赖。
4. 下载并本地保存：

   * `Qwen/Qwen2.5-1.5B-Instruct`
   * 官方 `Qwen2.5-1.5B-Instruct / GSM8K / weaver-sft` MemGen checkpoint
5. 优先查看并准备官方脚本：

   * `scripts/train/qwen2_5_gsm8k_sft.sh`
   * `scripts/eval/qwen2_5_gsm8k_sft.sh`
6. 修改任何“本地路径配置”之前先判断是否真的需要。如果只是把模型路径指向已经下载的本地模型，可以记录修改方案，但尽量保持官方仓库 working tree clean。
7. 做最小 smoke test：

   * Python 能 import torch / transformers / MemGen 相关模块；
   * CUDA 可见；
   * Qwen tokenizer 能加载；
   * base model 如果显存允许，尝试成功加载一次后退出；
   * 官方 checkpoint 文件完整可见。
8. 可以自行处理明显而低风险的问题，例如：

   * 缺少普通 Python package；
   * HuggingFace 下载工具缺失；
   * cache/path 不存在；
   * 环境变量或权限等简单问题。

9. 如果遇到以下问题，不要为了跑通而擅自修改源码：

   * MemGen 核心实现报错；
   * CUDA / flash-attention / DeepSpeed 等较复杂兼容问题；
   * 官方脚本本身疑似有 bug；
   * 需要改变训练参数；
   * 需要改变模型结构或数据处理逻辑。

   对这些问题保存完整报错和你的分析，明天我再处理。

另外建立一个 `REPRODUCTION.md`，记录：

* MemGen commit hash
* Python / PyTorch / CUDA 版本
* GPU 型号与显存
* venv 路径
* base model 本地路径
* MemGen checkpoint 本地路径
* 已完成的 smoke tests
* 遇到的问题及处理情况
* 明天第一条建议执行的官方 evaluation 命令

最终目标：今晚结束时，代码、环境和模型均已在本地，简单环境问题已排除；明天无需重新下载大文件，可以直接开始官方 GSM8K Weaver-SFT checkpoint evaluation。

不要开始正式训练和复现，停止在能运行模型时即可。
