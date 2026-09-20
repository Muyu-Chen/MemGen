# MemGen 完整工作流程解析

> **本文档已由两轮实证验证**：
>
> 1. 单题 tensor trace（`reproduction/verify_tensor_trace.py`，4/4 PASS）——
>    对 Weaver 模块注册 forward pre-hook，直接抓取子模块实际收到的张量后做逐元素比对。
> 2. **官方 inference 路径的完整执行级 trace**（`reproduction/wholeProcess/`）——
>    走真实调用链 `runner.evaluate → SingleTurnInteractionManager.run_agent_loop
>    → MemGenModel.generate`，在 module / tensor / parameter 三个层面记录全过程，
>    4 次重跑结果一致。证据见 `wholeProcess/logs/`。
>
> 验证结论（4/4 PASS）：
> - Weaver 输入是 Reasoner **input embeddings**，不是 hidden states
> - 拼接顺序是 `[context, query_latents]`，不是 `[query, context]`
> - 提取的是**最后** K 个 hidden states，不是前 K 个
> - 此前 append 的 latent 持久保留在序列中，后续 Weaver 能看到（逐比特 `max_abs_diff = 0.0`）
>
> 第 2 轮 trace 另外**推翻/修正**了本文档早先的 4 处描述，已在下文就地改正并标注
> 「**修正**」；完整清单见 [`wholeProcess/CORRECTIONS_NEEDED.md`](./wholeProcess/CORRECTIONS_NEEDED.md)。

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
  -> 4. Reasoner Generation（逐 token；每步之后按 token-id 判据筛出 candidate 再问 Trigger，
        inference 增强最多 3 次，达上限后剩余 token 交给 reasoner.generate 一次性生成）
  -> Output: "2+3 is 5.\boxed{5}"
```

> **修正（原写法：「逐 token，遇 delimiter 再问 Trigger，最多 3 次」）**
> 两处不准确，详见下文 [Trigger 门控的真实判据](#trigger-门控的真实判据token-id-精确匹配)
> 与 [达到上限后循环整体跳出](#达到-max_inference_aug_num-后循环整体跳出)：
> 判据不是「遇到 delimiter 字符」而是「最后一个 token 的 **id** 恰好等于 `,`/`.`/`\n`
> 各自单独成 token 时的 id」；且「最多 3 次」之后并非继续逐步生成，而是切到一条完全不同的批量路径。

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

> **修正（原写法：「Trigger 在 delimiter 位置被调用」）**
> 判据是 **token-id 精确匹配**，不是文本层面的 delimiter 匹配；句子边界会因 BPE 合并而系统性漏检。
> 见 [Trigger 门控的真实判据](#trigger-门控的真实判据token-id-精确匹配)。
> prompt 位置 `i=0` 确实必定调用一次（`_should_augment(..., is_prompt=True)` 不做任何筛选）。

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

### 实测时间线（GSM8K `test[0]`，prompt=96 token，K=8）

上面的 40-token 例子是示意；下面是 `wholeProcess/` 那次真实运行的数字：

| 事件 | Weaver 输入行数 | 插入位置 | 之后 Reasoner embeds 长度 |
|---|---|---|---|
| M0（prompt 增强，`i=0`） | 96 | index 96~103 | 104 |
| M1（第 12 个生成 token 后） | 116 | index 116~123 | 124 |
| M2（第 77 个生成 token 后） | 189 | index 189~196 | 197 |

最终序列布局：`prompt[0,96) M0[96,104) tokens[104,116) M1[116,124) tokens[124,189)
M2[189,197) tokens[197,231)`，共 231 行 embedding 对应 207 个 token id，
差值 24 = 3×8 个 latent。持久性为**逐比特**验证：M0 的 8 行在 R2W 调用 #2、#3
时与首次插入时 `max_abs_diff = 0.0`，M1 在 #3 时同样为 `0.0`。

注意 Weaver 输入行数（96/116/189）与同期 **Trigger 输入长度（96/108/173）** 的差 ——
那正是 Trigger 看不到的 latent。见
[执行级 trace 补充的关键事实](#执行级-trace-补充的关键事实) 第 1 条。

---

## Trigger 门控的真实判据（token-id 精确匹配）

**这不是「遇到 delimiter 字符就问 Trigger」，而是「最后一个 token 的 id 是否落在一个固定的 id 集合里」。**

`modeling_memgen.py:114` 定义的是三个**字符串**：

```python
self.delimiters: list[str] = [",", ".", "\n"]
```

但它们只是**原料**。`_get_delimiter_token_ids`（`modeling_utils.py:145-151`）在首次调用时把每个
字符串各自 encode 一次，得到一个 **id 集合**并缓存：

```python
for d in delimiters:
    ids = tokenizer.encode(d, add_special_tokens=False)
    delimiter_token_ids.update(ids)
```

Qwen2.5 tokenizer 下实测该集合 = **{11, 13, 198}**（`,` / `.` / `\n`）。

随后 `_check_ends_with_delimiter`（`modeling_utils.py:153-175`）只做一件事——取出最后一个非 pad
token 的 **id**，判断它是否属于该集合。**全程没有 decode，没有任何字符串比较**：

```python
last_tokens = input_ids[torch.arange(batch_size, device=device), last_positions]
is_delimiter = (last_tokens.unsqueeze(1) == delimiter_tensor).any(dim=1)
```

### 后果：句子边界被系统性漏检

BPE 会把「句号 + 后续字符」合并成**单个 token**，其 id 既不是 13 也不是 198，于是判据失败。
GSM8K `test[0]` 的 111 个生成步实测（证据：`wholeProcess/logs/delimiter_check.txt`）：

| 最后一个 token 的 id | 出现次数 |
|---|---|
| 11 (`,`) | **2** |
| 13 (`.`) | **0** |
| 198 (`\n`) | **0** |

而**文本里确实含句号/换行、却因合并而漏检**的步有 3 个：

```
step  38  id= 624  tok='.Ċ'      <- "." + "\n" 合并
step  66  id= 624  tok='.Ċ'      <- "." + "\n" 合并
step 105  id=7110  tok='.\'      <- "." + "\" 合并（\boxed 前）
```

即：**111 步里只有 2 步成为 augmentation candidate，两次都是逗号；三处真正的句子边界一次都没触发。**
本次运行最终的 2 次 inference augmentation 也正是发生在这两个逗号之后。

> 这条判据的形状意味着：任何与句号粘连的 token（`.\n`、`.\`、`.T` 等）都会绕过门控。
> 增强点的实际分布由 **tokenizer 的合并行为**决定，而不是由语义上的句子结构决定。
> 设计自有模型时，如果想让门控真正对齐「推理步骤边界」，这里必须换成显式判据。

---

## 达到 max_inference_aug_num 后循环整体跳出

`generate()` 的逐步循环里，在增强分支**之后**、Reasoner forward **之前**有一个逃逸出口
（`modeling_memgen.py:610-626`）：

```python
if (sentence_augment_count >= max_augment_num).all():
    generation_config_continue = GenerationConfig(
        do_sample=generation_config.weaver_do_sample,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=False,                 # <- KV cache 关闭
        max_new_tokens=max_new_tokens-i
    )
    generated = reasoner.generate(
        inputs_embeds=current_inputs_embeds,
        attention_mask=current_attention_mask,
        generation_config=generation_config_continue
    )
    current_input_ids = torch.cat([current_input_ids, generated], dim=1)
    break                                # <- 整个逐步循环结束
```

一旦触发，剩余 token 由 `reasoner.generate()` **一次性生成**，之后：

- **不再有逐步 gate 检查**
- **不再有 delimiter 判断**
- **Trigger 不再被调用**
- **KV cache 关闭**（`use_cache=False`，等于对整段序列重新计算）

两个容易忽略的细节：

1. 条件是 `.all()`——batch=1 时单条序列一达上限就立刻跳出；batch>1 时要**等所有序列都达上限**。
   在此之前，已达上限的序列靠 `aug_vector[over_limit] = -100`（`modeling_utils.py:275-276`）
   被排除出 candidate，但仍陪着走逐步循环。
2. **prompt augmentation 不计入该上限**：`modeling_memgen.py:558` 的 `if i != 0` 使得 `i==0`
   那次增强不会让 `sentence_augment_count` 自增。所以「1 次 prompt + 最多 3 次 inference」。

本次 trace 中 `sentence_augment_count` 只到 2（eos 先到），该分支**未被触发**——
`reasoner_forward_bulk_path` 事件数为 0，111 次 Reasoner 调用的 `use_cache` 全为 True
（证据：`wholeProcess/logs/tensor_shapes.json`）。但这条路径真实存在，且会显著改变行为，
不能只用「最多 3 次」概括。

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

推理时**以 checkpoint 自带的 `config.json` 为准**（`MemGenConfig.from_pretrained(ckpt)`），
下列取值即来自该文件，实测打印见 `wholeProcess/logs/full_trace.txt:60-61`。

```python
# checkpoint 路径: pn=1_pl=8_in=3_il=8
max_prompt_aug_num    = 1     # 见下方「修正」：generate() 从不读它
prompt_latents_len    = 8     # 每次 prompt 增强插入 8 个 latent
max_inference_aug_num = 3     # 推理过程最多增强 3 次（达上限后走 bulk 路径）
inference_latents_len = 8     # 每次 inference 增强插入 8 个 latent

# LoRA（weaver 与 trigger 相同，来源: adapter_config.json 实测）
r              = 16
lora_alpha     = 32
lora_dropout   = 0.1
target_modules = ["q_proj", "v_proj"]

# delimiter（只是原料，真正判据是它们各自 encode 出的 id 集合 {11, 13, 198}）
delimiters = [",", ".", "\n"]
```

> **修正（原注释：「`max_prompt_aug_num = 1  # prompt 最多增强 1 次`」）**
> `generate()`（`modeling_memgen.py:498-675`）**从未读取** `config.max_prompt_aug_num`。
> 全仓库它只出现在三处：`_conversational_forward`（`modeling_memgen.py:327-332`，
> **训练期**从多轮对话里随机挑选增强段落数）、`from_config` 的传参（`:683,704`）、
> 以及 `configuration_memgen.py` 的定义本身。
>
> 推理时 prompt 增强只发生一次，真正的原因是它被绑定在 `i == 0` 这**一个循环位置**上
> （`modeling_memgen.py:568` 的 `if i == 0:` 走 `augment_prompt`，其余走 `augment_inference`），
> 与这个配置值无关。把它理解成「推理期的上限」会在设计自有模型时误导。

### yaml 与 checkpoint config 互相矛盾

`configs/latent_memory/gsm8k.yaml` 与 released checkpoint 的 `config.json` /
`adapter_config.json` **不一致**：

| 项 | `gsm8k.yaml` | checkpoint（实测，应以此为准） |
|---|---|---|
| `max_inference_aug_num` | 5 | **3**（eval 脚本 `qwen2_5_gsm8k_sft.sh:24` 也写 3，命令行覆盖 yaml） |
| LoRA `target_modules` | q,k,v,o,gate,up,down（7 个） | **q_proj, v_proj（2 个）** |
| `interaction.batch_size` | 8 | eval 脚本 `qwen2_5_gsm8k_sft.sh:28` 写 **4** |
| `trigger.active` | False | eval 脚本 `:19` 同为 **False**（两者一致） |

若按 yaml 构建模型，会注入 5 组多余 adapter（`lora_B=0`，数值上无害），
但 module map 与参数总量都会和 checkpoint 对不上。本复现一律用
`MemGenConfig.from_pretrained(ckpt)`。

---

## 执行级 trace 补充的关键事实

以下每条都由 `reproduction/wholeProcess/` 的完整运行实测得到（GSM8K `test[0]`，
bf16 / greedy / batch=1，111 个生成步，1 次 prompt + 2 次 inference augmentation），
证据文件标注在每条末尾。早先的文档里没有这些内容。

**1. Trigger 看不到 latent，门控是在「纯 token 视图」上做的。**
`_should_augment` 只用 `current_input_ids` 并自行重算 attention mask
（`modeling_utils.py:270-271`），而 latent 是 embedding、没有 token id。
实测 Trigger 输入长 **96 / 108 / 173**，同期 Reasoner 的 embeds 长
**96→104 / 116→124 / 189→197**，差值恰好等于已插入的 latent 数（8 / 8 / 8）。
即**门控信号与实际注入内容不同构**。〔`tensor_shapes.json` 的 `trigger_calls`〕

**2. Weaver 的 `embed_tokens`（233,373,696 参数）完全不参与计算。**
Weaver 通过 `inputs_embeds=` 被调用，查表环节被跳过。
module census：Weaver base 338 个张量里 **337 个被用到，唯一未用的就是 `embed_tokens.weight`**。
〔`parameter_usage.json`；`key_findings.txt:19-20`〕

**3. Weaver 的 `lm_head` 照样执行，产出 `[1,104,151936]` 的 logits，但无人消费。**
记录字段 `logits_consumed_by_caller = false`。`_augment` 只取 `outputs.hidden_states[-1]`
（`weaver.py:85-92`），词表投影纯属浪费算力——在 CPU 上这是单次 Weaver 调用里最贵的一块。
〔`tensor_shapes.json` → `weaver_calls[*].weaver_output`〕

**4. latent 的数值量级约为 token embedding 的 60 倍。**
`weaver_to_reasoner` 的输出（即真正插入 Reasoner 序列的行）逐行 L2 norm：

| Weaver 调用 | 8 行的 L2 norm | 插入位置 |
|---|---|---|
| #1 (prompt) | 65.11 / 59.65 / 66.64 / 61.56 / 63.66 / 52.48 / 76.93 / 62.67 | index 96 |
| #2 | 59.99 / 60.95 / 52.38 / 55.90 / 54.27 / 65.40 / 59.04 / 56.12 | index 116 |
| #3 | 57.29 / 60.77 / 51.95 / 55.19 / 53.50 / 64.10 / 58.89 / 56.08 | index 189 |

而 token embedding 行：整段 96 行 Frobenius norm = 9.4926 ⇒ 平均每行 ≈ **0.97**
（采样前 12 行落在 0.43~1.21）。

直接后果可见于 `reasoner_to_weaver` 的输入 Frobenius norm：
**9.49 → 181.06 → 244.72**。第二次调用只比第一次多 20 行，norm 却涨了 19 倍——
因为其中 8 行是 M0 的 latent。**latent 一旦插入，就在后续表示中占主导地位。**
〔`tensor_shapes.json` → `events_ordered[3,5,45,47,244,246]`〕

**5. latent 不是任何 token embedding 的近似。**
把插入的 8 行 latent 与全部 151,936 个词表 embedding 逐一求余弦，
**最大值仅 0.098395**，`is_exact_embedding_of_some_token = false`。
同一套方法的**阳性对照**：对 96 行 prompt embedding 做同样检索，
最大余弦 **恰好 1.0**，且检索回的 token id 与原 prompt ids 完全一致。
说明「latent 是某种软 token」这类直觉在数值上不成立。
〔`tensor_shapes.json` → `events_ordered[6].check` 与 `events_ordered[3].vocab_similarity_check`〕

**6. 8 个 latent slot 之间高度相似，并不正交互补。**
实测平均两两余弦：**0.77325 / 0.719533 / 0.726276**（三次调用）。
K=8 提供的有效信息量远小于「8 个独立槽位」的直觉。
〔`tensor_shapes.json` → `weaver_calls[*].latent_cosine_between_slots`〕

**7. latent 占据真实 position id 并进入 KV cache。**
Weaver 为 latent 分配的位置是 `[96, 97, ..., 103]`（紧接 context 之后），
首个生成 token 的 `position_id_of_new_token = 104` 而**不是** 96；
该步 Reasoner 的输入长度 104、`use_cache=false`（cache 刚被清空），
之后 108 步全部 `use_cache=true` 且输入长度为 1。
即 **KV cache 里存着 latent 位的 K/V**，后续每个 token 都会 attend 到它。
〔`tensor_shapes.json` → `token_steps`；`key_findings.txt:187-198`〕

**8. Weaver 返回的 `position_ids` 被 `generate()` 丢弃。**
`modeling_memgen.py:569` 与 `:573` 都用 `_` 接住第三个返回值，
真实位置由 `_generate_position_ids(current_attention_mask)` 在 `:606` 重算。
两者在 batch=1 且无 padding 时结果相同，但 batch>1 带 padding 时是两条独立的路径。

**9. prompt augmentation 不计入 `max_inference_aug_num`。**
`modeling_memgen.py:558` 的 `if i != 0:` 使 `i==0` 那次不自增 `sentence_augment_count`。
所以上限语义是「1 次 prompt + 最多 3 次 inference」。

**10. 官方推理 dtype 确实是 bfloat16。**
`runner.py:224` 显式 `self.model = self.model.to(torch.bfloat16)`，
且 `projs.bin` / `weaver.bin` / `trigger.bin` 内部张量本身就是 bfloat16。
本次 trace 全程 bf16，未做任何 dtype 折衷。

**11. 输出里的 `<<...>>` 是 GSM8K 自带的计算器标注，与 MemGen 机制无关。**
本次 completion：`...16 - 3 = <<16-3=13>>13 eggs left...` → 结尾 `\boxed{18}`。
**gold rationale 里同样带 `<<16-3-4=9>>`**，说明这是数据集标注格式，SFT 让模型学会照抄。
分析输出时应把它当作格式噪声，而不是推理链的一部分。
〔`results/sample_output.json` 的 `completion` / `gold` 字段〕

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
4. **每次增强后清空 KV cache**，Reasoner 对完整序列重新 forward；
   latent 占据真实 position id，其 K/V 留在 cache 里被后续每个 token attend。
5. **门控判据是 token-id 精确匹配 {11, 13, 198}**，不是文本层面的 delimiter；
   BPE 合并导致句子边界系统性漏检（本题 111 步只命中 2 次，都是逗号）。
6. **Trigger 只在纯 token 视图上做决策**，看不到任何已插入的 latent ——
   门控信号与实际注入内容不同构。
7. **`max_inference_aug_num` 达上限后整个逐步循环 `break`**，
   剩余 token 交给 `reasoner.generate(use_cache=False)` 一次性生成，之后不再有任何门控。

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

本次 trace 又暴露出三个同样可切入的空白：

- **门控与内容脱钩**：决定「是否注入」的 Trigger 只看 token，
  而决定「注入什么」的 Weaver 看到的是含 latent 的 embedding 序列。两者视图不同。
- **注入点由 tokenizer 决定**：`.\n` 被合并成 id 624 就绕过门控，
  所谓「在推理步骤边界增强」在实现上并不成立。
- **append-only、从不覆写**：latent 只追加，K=8 个槽位两两余弦 0.72~0.77，
  且行 norm 是 token 的约 60 倍——记忆既不可更新，也没有容量约束。

