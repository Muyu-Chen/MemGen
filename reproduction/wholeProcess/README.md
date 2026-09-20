在当前 MemGen 仓库中，有目录：

```text
reproduction/wholeProcess/
```

目标：**完整地、可复现地跑通一次官方 MemGen inference 流程，并从 tensor / module / parameter 层面记录整个 execution trace。**

这不是性能测试，也不是重新评估 benchmark。只选择 **1 道能够触发至少一次 prompt augmentation，并最好再触发至少一次 inference augmentation 的 GSM8K 样本**，把从原始 prompt 到最终输出的全过程逐步拆开。

不要根据 README 或我们之前的说明猜测架构。**以当前仓库实际代码为唯一准则**。如果文档描述和代码不一致，以代码为准，并明确记录差异。

不要修改 MemGen 的算法行为。允许增加 logging、forward hook、debug wrapper、monkey patch 和独立调试脚本，但必须保证原模型计算逻辑不变。所有调试代码都放进 `reproduction/wholeProcess/`，不要污染核心源码。

首先记录当前实验环境，包括：

* Git commit
* 使用的 base model
* 使用的 MemGen checkpoint
* checkpoint 具体路径
* Weaver / Trigger / Reasoner 的模型类型
* prompt latent 数量
* inference latent 数量
* max prompt / inference augmentation 数量
* LoRA 配置
* Trigger 当前 `active` 状态
* generation config
* dtype / device

尤其要明确区分下面三个概念：

```text
loaded parameters
trainable parameters
actually active/used parameters during this inference
```

inference 时即使所有参数 `requires_grad=False`，也要记录哪些模块实际参与 forward。

---

## 1. 先建立整体 module map

根据实际代码列出 MemGen 中所有与本次推理有关的主要组件：

```text
tokenizer
Reasoner
Weaver base model
Weaver LoRA
prompt_query_latents
inference_query_latents
reasoner_to_weaver projection
weaver_to_reasoner projection
Trigger base model
Trigger LoRA
Trigger output_layer
generation loop
KV cache
```

对每一个组件说明：

* 类名
* 源文件
* 关键函数
* 参数量
* checkpoint 权重来源
* 是否加载
* 是否在本次 inference 实际调用
* 是否有 LoRA adapter 生效
* 输入 shape
* 输出 shape

不要把“checkpoint 中存在某个参数文件”等同于“这个模块经过训练”。

---

## 2. 从 raw prompt 开始逐步 trace

对同一道 GSM8K 问题完整记录以下过程。

### Step A — Prompt construction

记录：

```text
raw question
最终 prompt 文本
chat template 是否应用
input_ids
token 数量
attention_mask
position_ids
```

可以只展示部分 token ID，但要保存完整版本到日志文件。

解释：

为什么最终输入长这样？由哪个函数构造？

---

### Step B — Prompt augmentation candidate / Trigger

记录 prompt 结束时：

```text
是否成为 augmentation candidate
为什么成为 candidate
调用了哪个函数
Trigger.active 的值
Trigger 输入 shape
Trigger logits
softmax probability
最终 augmentation decision
```

如果当前官方 eval 使用：

```text
trigger.active = False
```

必须明确说明这在代码中究竟意味着什么，不能写成“关闭 Trigger”。

要指出：

```text
active=False 时是否仍会 augmentation
decision 是如何生成的
```

---

## 3. Weaver augmentation 必须逐行拆开

这是本任务最重要的部分。

实际记录进入 Weaver 前的：

```text
current_inputs_embeds.shape
attention_mask.shape
position_ids.shape
```

然后逐步记录：

### 3.1 Reasoner embedding

明确确认 Weaver 获取的是：

```text
Reasoner input embeddings
```

还是：

```text
Reasoner hidden states
```

必须用实际变量和代码位置证明，不允许根据之前的文档推断。

### 3.2 reasoner_to_weaver projection

记录：

```text
input shape
weight shape
output shape
dtype
```

简要统计数值：

```text
mean
std
norm
min/max
```

不需要 dump 全 tensor。

### 3.3 Query latents

分别记录：

```text
prompt_query_latents
inference_query_latents
```

当前这一步实际使用哪一个。

记录：

```text
original shape
LayerNorm 后 shape
latent_scale
norm before / after
```

### 3.4 Concatenation

必须确认真实顺序究竟是：

```text
[context, query_latents]
```

还是：

```text
[query_latents, context]
```

并记录：

```text
concat 前 shape
concat 后 shape
query latent 的 position_ids
attention mask
```

### 3.5 Weaver forward

记录：

```text
Weaver model
当前激活的 LoRA adapter
LoRA target modules
q_proj / v_proj 中 adapter 是否实际执行
Weaver input shape
final hidden state shape
```

最好使用 hook 验证至少一个带 LoRA 的 `q_proj` / `v_proj` 确实经过 forward。

### 3.6 Latent extraction

必须确认最终取的是：

```text
first K hidden states
```

还是：

```text
last K hidden states
```

记录代码位置和实际 tensor slice。

保存新生成 latent 的：

```text
shape
mean
std
norm
cosine similarity between latent slots（可选）
```

不需要尝试 decode latent。

### 3.7 weaver_to_reasoner projection

记录：

```text
input shape
output shape
projection weight shape
norm before / after
```

---

## 4. Latent 如何真正插回 Reasoner

这是第二个重点。

augmentation 前：

```text
Reasoner context structure
context length
```

augmentation 后：

```text
[原 context][latent]
```

记录：

```text
new inputs_embeds.shape
new attention_mask.shape
new position_ids
latent position range
```

明确回答：

这些 latent 是否拥有真实 token ID？

是否经过 tokenizer？

是否会经过 LM head 直接 decode？

Reasoner 是如何“看到”这些 latent 的？

---

## 5. Reasoner 如何继续生成

记录 augmentation 后第一次 Reasoner forward：

```text
Reasoner input shape
KV cache 是否存在
past_key_values 状态
output logits shape
predicted next token
decoded token
```

至少跟踪几个 token，直到下一次 delimiter / augmentation candidate。

说明：

```text
token-by-token generation
cache 如何更新
latent 是否进入 KV cache
```

---

## 6. 必须完整 trace 第二次 augmentation

如果样本发生 inference augmentation，必须重点记录第二次。

在第二次 Weaver 调用之前，打印当前 logical sequence：

```text
prompt
latent_0
generated reasoning tokens
```

明确验证：

```text
latent_0 是否仍然存在于 current_inputs_embeds
latent_0 是否参与第二次 Weaver 的输入
```

然后记录：

```text
Weaver 第二次输入 shape
inference_query_latents
latent_1 shape
插入后整个 context 结构
```

最终用类似下面的图表示：

```text
Prompt
  ↓
[Prompt]
  ↓ Weaver
[Prompt][M0]
  ↓ Reasoner
[Prompt][M0][R0]
  ↓ Weaver
[Prompt][M0][R0][M1]
  ↓ Reasoner
[Prompt][M0][R0][M1][R1]
...
```

但是这个图必须和真实执行结果一致。

---

## 7. 记录“到底哪些参数在工作”

单独生成一张表，至少包含：

| Component               | Loaded | Trained checkpoint? | Used in this forward? | LoRA active? | requires_grad | Parameter count |
| ----------------------- | ------ | ------------------- | --------------------- | ------------ | ------------- | --------------- |
| Reasoner                |        |                     |                       |              |               |                 |
| Weaver base             |        |                     |                       |              |               |                 |
| Weaver LoRA             |        |                     |                       |              |               |                 |
| Prompt query latents    |        |                     |                       |              |               |                 |
| Inference query latents |        |                     |                       |              |               |                 |
| R→W projection          |        |                     |                       |              |               |                 |
| W→R projection          |        |                     |                       |              |               |                 |
| Trigger base            |        |                     |                       |              |               |                 |
| Trigger LoRA            |        |                     |                       |              |               |                 |
| Trigger head            |        |                     |                       |              |               |                 |

特别注意：

```text
requires_grad=False
```

不代表：

```text
parameter not used
```

要把“是否训练”和“是否参与 inference”彻底分开。

---

## 8. 输出文件

`reproduction/wholeProcess/` 至少包含：

```text
wholeProcess/
├── README.md
├── FULL_EXECUTION_TRACE.md
├── PARAMETER_FLOW.md
├── trace_memgen.py
├── hooks.py
├── logs/
│   ├── full_trace.txt
│   ├── tensor_shapes.json
│   └── parameter_usage.json
└── results/
    └── sample_output.json
```

`FULL_EXECUTION_TRACE.md` 是最重要的最终文档。

它必须做到：**一个没有读过 MemGen 代码的人，只看这份文档，也能准确解释一次 MemGen augmentation 从哪里来、经过哪些 tensor、哪些模型、哪些 projection，最后怎么影响 Reasoner。**

---

## 9. 最终必须回答这些问题

在文档结尾增加：

```text
## Questions Answered
```

明确逐条回答：

1. Weaver 的输入究竟是 Reasoner hidden state，还是 input embeddings？
2. Query latents 在 context 前还是后？
3. 为什么 query latent 能获取 context 信息？
4. 最终取 Weaver 的哪 K 个 hidden states？
5. latent memory 有没有 token ID？
6. latent 是否经过 tokenizer？
7. latent 是否经过 LM head？
8. latent 如何影响 Reasoner 的下一步 token prediction？
9. 第二次 Weaver 调用时，第一次 latent 是否还存在？
10. 第二次 Weaver 是否能看到第一次 latent？
11. 历史 latent 是 append、replace，还是 update？
12. context 长度如何随 augmentation 增长？
13. Trigger 在当前 released checkpoint / eval configuration 中到底做了什么？
14. 本次 inference 中哪些 LoRA adapter 实际参与计算？
15. Reasoner、Weaver、Trigger 是否共享权重，还是三个独立模型实例？
16. 哪些参数来自 base model，哪些来自 checkpoint，哪些是额外 learned parameters？
17. prompt augmentation 和 inference augmentation 的唯一区别是什么？
18. KV cache 在 augmentation 前后如何处理？

---

## 10. 工作原则

* 不做新的 benchmark。
* 不跑完整 GSM8K。
* 不训练模型。
* 不研究 Trigger 的 paper-code mismatch，本任务只记录当前代码实际执行行为。
* 不根据变量名猜意义。
* 每一个重要结论都应尽量附上对应源码文件、函数名和关键代码。
* 如果实际运行结果和 `REPRODUCTION.md` 或 `MEMGEN_ARCHITECTURE_EXPLAINED.md` 不一致，**不要迁就旧文档**，记录真实行为，并在最后列出需要修正的旧描述。
* 不直接修改已有 reproduction 结论；先把发现记录在 `wholeProcess/` 中，最后给我一份 `CORRECTIONS_NEEDED.md`，告诉我其他文档哪些地方应该改。

最终不要只告诉我“运行成功”。我要的是一份**execution-level forensic trace**：能让我真正理解 MemGen 在一次 inference 中，每一步发生了什么、为什么发生、调用了哪个模块、tensor 怎么变化、哪些参数实际参与了计算。
