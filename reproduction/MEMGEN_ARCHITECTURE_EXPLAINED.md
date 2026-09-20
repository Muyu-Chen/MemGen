# MemGen 完整工作流程解析

> **本文档已由单题 tensor trace 实证验证**。所有 shape、拼接顺序、切片位置均来自
> `reproduction/verify_tensor_trace.py` 的真实运行结果，而非仅凭代码阅读推断。
> 验证脚本对 Weaver 模块注册 forward pre-hook，直接抓取子模块实际收到的张量后做逐元素比对。
>
> 验证结论（4/4 PASS）：
> - Weaver 输入是 Reasoner **input embeddings**，不是 hidden states
> - 拼接顺序是 `[context, query_latents]`，不是 `[query, context]`
> - 提取的是**最后** K 个 hidden states，不是前 K 个
> - 此前 append 的 latent 持久保留在序列中，后续 Weaver 能看到

## 核心架构

MemGen = **Reasoner**（主模型）+ **Weaver**（潜在记忆生成器）+ **Trigger**（增强决策器）

三者均为同一份 Qwen2.5-1.5B-Instruct 的 deepcopy，Weaver / Trigger 各挂一套 LoRA。

```
Input: "What is 2+3?"
  -> [Tokenize] input_ids [1, 40]
  -> 1. Reasoner Embedding Lookup（不是 forward!）
        reasoner.get_input_embeddings()(input_ids) -> [1, 40, 1536]
  -> 2. Trigger Decision -> 0 / 1
  -> 3. Weaver Augmentation（若 decision=1）
        a. reasoner_to_weaver(embeds)   -> [1, 40, 1536]
        b. cat([context, Q])            -> [1, 48, 1536]
        c. Weaver forward               -> [1, 48, 1536]
        d. 取 **最后** 8 个 hidden state -> [1,  8, 1536]
        e. weaver_to_reasoner(latent)   -> [1,  8, 1536]
        f. cat([context, latent])       -> [1, 48, 1536]
  -> 4. Reasoner Generation（逐 token，遇 delimiter 再问 Trigger，最多 3 次）
  -> Output: "2+3 is 5.\boxed{5}"
```

---

## 四个最容易搞错的关键点

这四点决定了 Weaver 到底"看到了什么"，是本复现最重要的架构结论。

### 1. Weaver 的输入是 Reasoner 的 input embeddings，不是 hidden states

`modeling_memgen.py:110` 的注释写得很直白：

```python
self.reasoner_to_weaver = nn.Linear(reasoner_hidden_size, weaver_hidden_size)
# map reasoner input embeddings to weaver input embeddings
```

调用链（`modeling_memgen.py:522` -> `:567`）：

```python
inputs_embeds = reasoner.get_input_embeddings()(input_ids)   # 仅查表，无 forward
...
weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)
```

**调用 Weaver 之前没有跑 Reasoner forward。** Weaver 拿不到 Reasoner 的内部推理状态。

trace 实证（step 2）：

```
equals_embedding_table_lookup: True
```

即 `reasoner.get_input_embeddings()(input_ids)` 与 `embed_weight[input_ids]` 逐元素相同，
证明这只是 embedding 查表，不含任何 Transformer 层计算。

> **研究含义**：MemGen 的 Weaver 是一个
> **context-conditioned latent generator**（基于可见 token 生成潜在记忆），
> 而**不是** reasoning-hidden-state-conditioned memory updater（基于内部认知状态更新记忆）。

### 2. 拼接顺序是 [context, query_latents]，不是 [query, context]

`weaver.py:70`：

```python
inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)
```

query latents 拼在**后面**。

trace 实证（step 4，通过 forward pre-hook 抓取 Weaver 实际收到的张量）：

```
context_len: 40, query_latents_K: 8, concatenated_len: 48
head_is_context:        True    <- 前 40 行 == reasoner_to_weaver 的输出
tail_is_query_latents:  True    <- 后 8 行 == LN(prompt_query_latents) * scale
layout: [context(40), query_latents(8)]
```

### 3. 提取的是最后 K 个 hidden states，不是前 K 个

`weaver.py:92`：

```python
latents_hidden_states = hidden_states[:, -latents_num:, :]
```

trace 实证（step 6）：

```
matches_LAST_K:   True
matches_FIRST_K:  False
```

### 4. 为什么"放后面 + 取最后"恰好是对的

因为 Weaver 是 causal transformer：

```
position:   0 ... 39 | 40 41 ... 47
content:  context    | Q1 Q2 ... Q8
                        ^
                   可以 attend 到前面全部 context
```

query latents 位于序列末尾，因此每个 Q_i 都能看到完整 context，
输出的最后 8 个 hidden states 就是"读完 context 之后生成的潜在记忆"。

若按错误顺序 `[Q, context]`，Q 在 causal mask 下将**看不到任何 context**，机制完全失效。
所以这个顺序不是实现细节，而是机制成立的**前提**。

---

## 数学表述

设当前可见 context 为 ctx_t（含此前已 append 的 latent embeddings）：

```
C_t       = Proj_R2W( Embed_R(ctx_t) )
Z_t       = Weaver( [ C_t , Q ] )[ -K: ]
M_t       = Proj_W2R( Z_t )
ctx_t+1   = [ ctx_t , M_t ]
```

其中 Q 是可学习参数（`prompt_query_latents` / `inference_query_latents`），K=8。

---

## 详细步骤（含 trace 真实数值）

以下数值来自 `"What is 2+3?"` 的实际运行：prompt_len=40，hidden_size=1536，K=8。

### Step 1: Prompt Tokenization

```
input_ids shape: [1, 40]
first_10_ids:    [151644, 872, 198, 50, 3948, 279, 6888, 3491, 448, 6169]
last_5_ids:      [151645, 198, 151644, 77091, 198]
decoded_last_5:  ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n']
```

### Step 2: Reasoner Input Embeddings

```python
inputs_embeds = reasoner.get_input_embeddings()(input_ids)
```

```
shape: [1, 40, 1536]
mean: 0.000164   std: 0.024937   absmax: 0.300781
equals_embedding_table_lookup: True    <- 关键证据
```

### Step 3: Trigger Decision

Trigger 在 delimiter 位置被调用（prompt 位置 `i=0` 必定调用一次）。

```python
# modeling_utils.py:280-293
trigger_logits = trigger(input_ids, attention_mask, position_ids)
last_token_logits = trigger_logits[:, -1]     # [B, 2]
next_tokens = self._get_next_token(last_token_logits, ...)
```

**注意 `trigger.py:37-40` 的 `active=False` 分支**：

```python
else:
    logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)
    logits[..., 1] = 1.0     # 硬编码：永远 class 1
```

即 `active=False` 时 Trigger 被**硬编码为总是增强**，softmax 恒为 `[0.269, 0.731]`。
本复现的三组对照实验中，`always_1` 模式正是走这条路径。

### Step 4: Projection Reasoner -> Weaver

```python
weaver_inputs_embeds = self.reasoner_to_weaver(inputs_embeds)
```

```
in_shape:     [1, 40, 1536]
out_shape:    [1, 40, 1536]
weight_shape: [1536, 1536]
mean: -0.000733   std: 0.020924   absmax: 0.080168
```

### Step 5: Query Latents 归一化 + 拼接

```python
# weaver.py:66-70
latents = latent_ln(latents) * latent_scale          # LayerNorm + scale
latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)   # [context, Q]
```

```
concatenated shape: [1, 48, 1536]
layout: [context(40), query_latents(8)]
```

position_ids 也顺延（`weaver.py:77-80`）：

```python
last_position_ids = position_ids.max(dim=1)[0]
latents_position_ids = last_position_ids.unsqueeze(1) + arange(latents_num) + 1
```

### Step 6: Weaver Forward

```
last_hidden_state shape: [1, 48, 1536]
mean: -0.064108   std: 2.623458   absmax: 94.309715
```

注意 std 从输入的 0.021 放大到 2.62 —— 这是 Transformer 深层 hidden states 的正常量级，
也再次说明 Weaver 输出**不在** input embedding 的数值尺度上。

### Step 7: 提取 Latent（最后 K 个）

```python
latents_hidden_states = hidden_states[:, -latents_num:, :]
```

```
extracted shape: [1, 8, 1536]
matches_LAST_K:  True
matches_FIRST_K: False
```

### Step 8: Projection Weaver -> Reasoner

```python
latent_inputs_embeds = self.weaver_to_reasoner(latent_hidden)
```

```
in_shape:  [1, 8, 1536]
out_shape: [1, 8, 1536]
mean: -0.028814   std: 1.724742   absmax: 6.851033
```

### Step 9: Append 回 Reasoner 序列

```python
# modeling_memgen.py:578
candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)
```

```
before_shape: [1, 40, 1536]
after_shape:  [1, 48, 1536]
context_retained:        True    <- 原 context 完整保留
latent_appended_at_tail: True    <- latent 拼在末尾
attention_mask: 全 1（latent 也参与 attention）
```

同时 **KV cache 被清空**（增强分支末尾 `current_cache = None`），
Reasoner 会对完整的 48-token 序列重新 forward。

### Step 10: Reasoner 生成

```
logits shape: [1, 151936]
next_token_id: 17  ->  '2'
top5: [('2', 0.5681), ('The', 0.1643), ('Adding', 0.0453),
       ('We', 0.0236), ('To', 0.0230)]

完整输出: '2+3 is 5.\boxed{5}'
augmentation positions: [0]      <- 本题只触发了 prompt 增强
```

---

## 第二次 Augmentation 时，第一次的 Latent 在哪里？

**答：仍留在 `current_inputs_embeds` 中，且 Weaver 能看到它。**

`modeling_utils.py:318` 的 `_append_one_step` 只做尾部追加：

```python
current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embeds], dim=1)
```

从不删除已有内容。而增强分支（`modeling_memgen.py:564-567`）取的是**整个累积序列**：

```python
candidate_inputs_embeds = current_inputs_embeds[augment_indices]   # 全序列
weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)
```

所以时间线是（以 prompt_len=40 为例）：

```
Prompt 增强 (i=0):
  Weaver 输入:   [prompt(40)]                                        = 40
  Weaver 输出:   latent_1
  Reasoner 看到: [prompt(40), latent_1(8)]                           = 48

生成 12 个 token 后，第二次增强 (i=12):
  Weaver 输入:   [prompt(40), latent_1(8), generated(12)]             = 60  <- latent_1 在里面!
  Weaver 输出:   latent_2
  Reasoner 看到: [prompt(40), latent_1(8), generated(12), latent_2(8)] = 68
```

**含义**：Weaver 的输入包含此前生成的 latent embeddings，
因此它不是"从零生成"，而是**在已有潜在记忆的基础上继续叠加**。
但这些 latent 是以 **embedding 形式**被 Weaver 看到的 ——
Weaver 依然**看不到 Reasoner 处理它们之后产生的 hidden states**。

---

## Trigger 的两种模式

| `active` | 行为 | 代码位置 |
|----------|------|---------|
| `False` | 硬编码 `logits[...,1]=1.0`，**永远增强** | `trigger.py:37-40` |
| `True` | 跑 Trigger LoRA 模型 + `output_layer` 线性头，真实二分类 | `trigger.py:27-35` |

官方 weaver-sft 脚本设置的是 `active False`：

```bash
# scripts/train/qwen2_5_gsm8k_sft.sh:49,52,53
model.trigger.active False
run.train_weaver True
run.train_trigger False
```

**本复现所用 checkpoint 的 Trigger 完全未经训练**，有两条独立证据
（详见 [`REPRODUCTION.md`](./REPRODUCTION.md)）：

1. `trigger/trigger/adapter_model.safetensors` 的 `lora_B` **56/56 全为零**
   （LoRA 的 B 矩阵初始化为零，非零即代表训练过）
2. `trigger.bin` 的 `output_layer.weight` absmax = 0.02551 = 1/sqrt(1536)，
   正好是 `nn.Linear` 随机初始化的边界值

对比之下 **Weaver 的 `lora_B` 0/56 全零**，即 Weaver LoRA 确实训练过。

所以本复现中 "Trigger 总是输出 1" 的现象，反映的是**随机初始化分类头**的行为，
**不能**用来推断经过 `trigger_train` 阶段训练后的 Trigger 表现。

---

## 配置参数

```python
# checkpoint 路径: pn=1_pl=8_in=3_il=8
max_prompt_aug_num    = 1     # prompt 最多增强 1 次
prompt_latents_len    = 8     # 每次 prompt 增强插入 8 个 latent
max_inference_aug_num = 3     # 推理过程最多增强 3 次
inference_latents_len = 8     # 每次 inference 增强插入 8 个 latent

# LoRA（weaver 与 trigger 相同，来源: adapter_config.json 实测）
r              = 16
lora_alpha     = 32
lora_dropout   = 0.1
target_modules = ["q_proj", "v_proj"]

# delimiter（触发 Trigger 检查的符号）
delimiters = [",", ".", "\n"]
```

---

## 总结

MemGen 的实际机制：

1. **Weaver 是 context-conditioned latent generator** —— 输入是 Reasoner 的
   **input embeddings**（可见 token 的查表结果 + 此前 append 的 latent embeddings），
   而**不是** Reasoner 的内部 hidden reasoning state。
2. **拼接为 [context, Q]，提取最后 K 个 hidden states** —— 依赖 causal attention
   让末尾的 query latents 读到完整 context。顺序颠倒则机制失效。
3. **latent 以 embedding 形式 append 回 Reasoner 序列**，并持久保留，
   后续 Weaver 调用能看到它们。
4. **每次增强后清空 KV cache**，Reasoner 对完整序列重新 forward。

### 对我们自己模型设计的启示

MemGen 的信息流是：

```
visible context embeddings -> fresh latent -> append
```

若我们改为：

```
H_t^R + M_t -> M_t+1
```

即用 Reasoner 的**内部认知状态** H_t^R 去更新一个**持久可变的记忆状态** M_t，
那么这与 MemGen 是**真实的结构性差异**，而不只是"也循环调用 Weaver"。

MemGen 的 Weaver 从未接触过 H_t^R —— 这正是可切入的空白。

