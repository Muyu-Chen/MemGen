# MemGen 完整工作流程解析

## 核心架构

MemGen = **Reasoner** (主模型) + **Weaver** (潜在记忆生成器) + **Trigger** (增强决策器)

```
┌─────────────────────────────────────────────────────────────┐
│                      MemGen Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "What is 2+3?"                                      │
│    ↓                                                        │
│  [Tokenize] → prompt_tokens (24 tokens)                     │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Trigger Decision (Prompt)                         │  │
│  │    - 检查是否需要在 prompt 后插入潜在记忆              │  │
│  │    - 输出: augment=1 (插入) 或 0 (不插入)             │  │
│  └──────────────────────────────────────────────────────┘  │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 2. Reasoner Forward (Prompt)                         │  │
│  │    - 处理 prompt tokens                               │  │
│  │    - 输出: hidden_states [1, 24, 1536]                │  │
│  └──────────────────────────────────────────────────────┘  │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3. Weaver Augmentation (if augment=1)                │  │
│  │    a. Project: Reasoner → Weaver space                │  │
│  │       reasoner_to_weaver(hidden_states)               │  │
│  │    b. Concat: [query_latents(8), projected_input]     │  │
│  │    c. Weaver Forward: 生成增强后的潜在记忆             │  │
│  │    d. Extract: 取前 8 个 hidden states 作为 latent    │  │
│  │    e. Project: Weaver → Reasoner space                │  │
│  │       weaver_to_reasoner(latents)                     │  │
│  │    f. Insert: 拼接回 Reasoner 输入                    │  │
│  └──────────────────────────────────────────────────────┘  │
│    ↓                                                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. Reasoner Continues Generation                     │  │
│  │    - 输入: [prompt + latent_tokens]                   │  │
│  │    - 逐 token 生成，每个 delimiter 处检查 Trigger     │  │
│  │    - 可能触发多次 augmentation (最多 3 次)            │  │
│  └──────────────────────────────────────────────────────┘  │
│    ↓                                                        │
│  Output: "2+3 is 5.\boxed{5}"                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 详细步骤解析

### Step 1: Prompt Tokenization

```python
Prompt: "Question: What is 2+3?\n\nPlease reason step by step..."
Tokens: [14582, 25, 3555, 374, 220, 17, 10, 18, 1939, 5501, ...]
Shape: [1, 24]  # batch_size=1, seq_len=24
```

### Step 2: Trigger Decision (Prompt)

```python
# Trigger 检查 prompt，决定是否插入潜在记忆
trigger_logits = trigger(input_ids, attention_mask, position_ids)
# 输出: [1, 24, 2]  # 每个 token 位置的二分类 logits

# 取最后一个 token 的决策
decision_logits = trigger_logits[0, -1]  # [2]
softmax = [0.021, 0.979]  # P(augment=0)=2.1%, P(augment=1)=97.9%
decision = 1  # 决定增强
```

### Step 3: Reasoner Forward (Prompt)

```python
# Reasoner 处理 prompt
outputs = reasoner(input_ids, attention_mask, output_hidden_states=True)
hidden_states = outputs.hidden_states[-1]  # [1, 24, 1536]
```

### Step 4: Weaver Augmentation (核心)

#### 4a. Projection: Reasoner → Weaver

```python
# 将 Reasoner 的 hidden states 投影到 Weaver 空间
weaver_input = reasoner_to_weaver(hidden_states)
# Shape: [1, 24, 1536]  # 线性变换
```

#### 4b. Query Latents (固定学习的参数)

```python
# Prompt augmentation 使用固定的 query latents
query_latents = model.weaver.prompt_query_latents
# Shape: [8, 1536]  # 8 个可学习的 query vectors

# LayerNorm + Scale
query_latents = latent_ln(query_latents) * latent_scale
# Shape: [8, 1536]

# 扩展到 batch
query_latents = query_latents.unsqueeze(0).expand(batch_size, -1, -1)
# Shape: [1, 8, 1536]
```

#### 4c. Concatenation

```python
# 拼接: [query_latents, weaver_input]
weaver_full_input = torch.cat([query_latents, weaver_input], dim=1)
# Shape: [1, 8+24, 1536] = [1, 32, 1536]
```

#### 4d. Weaver Forward

```python
# Weaver (带 LoRA adapter) 处理拼接后的输入
weaver_outputs = weaver(inputs_embeds=weaver_full_input, output_hidden_states=True)
weaver_hidden = weaver_outputs.hidden_states[-1]
# Shape: [1, 32, 1536]
```

#### 4e. Extract Augmented Latents

```python
# 取前 8 个 token 的 hidden states 作为增强后的潜在记忆
augmented_latents = weaver_hidden[:, :8, :]
# Shape: [1, 8, 1536]
```

#### 4f. Projection: Weaver → Reasoner

```python
# 投影回 Reasoner 空间
reasoner_augmented = weaver_to_reasoner(augmented_latents)
# Shape: [1, 8, 1536]
```

#### 4g. Insert Back to Reasoner

```python
# 拼接到 Reasoner 输入: [prompt_embeds, latent_embeds]
reasoner_inputs = torch.cat([prompt_embeds, reasoner_augmented], dim=1)
# Shape: [1, 24+8, 1536] = [1, 32, 1536]

# 同时更新 attention_mask
attention_mask = [1,1,...,1 (24个), 1,1,...,1 (8个)]
```

### Step 5: Reasoner Continues Generation

```python
# Reasoner 现在看到: [prompt + 8个latent tokens]
# 开始逐 token 生成

for each new token:
    # 1. Reasoner 预测下一个 token
    logits = reasoner(current_inputs)
    next_token = argmax(logits)
    
    # 2. 检查是否是 delimiter (, . \n)
    if is_delimiter(next_token):
        # 3. Trigger 决定是否再次增强
        decision = trigger(current_input_ids)
        if decision == 1 and augment_count < max_augments:
            # 4. 执行 inference augmentation (类似 prompt augmentation)
            # 但使用 inference_query_latents 而不是 prompt_query_latents
            latent = weaver.augment_inference(...)
            current_inputs = cat([current_inputs, latent])
    
    # 5. 继续生成
    append(next_token)
```

## 关键发现（从实验数据）

### 1. Trigger 退化为常函数

```
Trained Trigger 统计:
- Decision=0: 0 次 (0%)
- Decision=1: 22 次 (100%)
- Softmax P(augment=1): 0.968 ~ 0.999, 中位数 0.993

结论: Trigger 学会了总是预测 class 1 (总是增强)
```

### 2. Weaver 改变输出格式

```
always_0 (无增强):
  输出: "The answer is 5."
  格式: 自然语言
  
trained/always_1 (有增强):
  输出: "2+3 is 5.\boxed{5}"
  格式: 包含 \boxed{}

结论: Weaver 增强使模型更倾向使用 \boxed{} 格式
```

### 3. 性能提升来源

```
GSM8K (5题, 难题):
  Base:         3/5 = 60%
  Weaver-only:  4/5 = 80%  (+20%)
  Full MemGen:  4/5 = 80%  (+20%)

简单数学 (20题):
  always_0:  20/20 = 100% (真实准确率)
  trained:   20/20 = 100%
  always_1:  20/20 = 100%

结论: 
- Weaver 对难题有实质帮助 (+20%)
- 对简单题只是改变格式，不影响推理能力
- Trigger 的门控决策无额外贡献 (trained ≡ always_1)
```

## 数据流总结

```
Input Tokens (24)
    ↓
Reasoner Embeddings [1, 24, 1536]
    ↓
┌─────────────────────────────────────┐
│ Weaver Augmentation                 │
│                                     │
│ 1. reasoner_to_weaver()             │
│    → [1, 24, 1536]                  │
│                                     │
│ 2. Concat query_latents             │
│    → [1, 8+24, 1536] = [1, 32, 1536]│
│                                     │
│ 3. Weaver Forward (with LoRA)       │
│    → [1, 32, 1536]                  │
│                                     │
│ 4. Extract first 8 tokens           │
│    → [1, 8, 1536]                   │
│                                     │
│ 5. weaver_to_reasoner()             │
│    → [1, 8, 1536]                   │
└─────────────────────────────────────┘
    ↓
Reasoner Input: [prompt(24) + latent(8)] = [1, 32, 1536]
    ↓
Reasoner Generation (逐 token)
    ↓
Output: "2+3 is 5.\boxed{5}"
```

## 第二次 Augmentation 时第一次 Latent 的位置

```
Prompt Augmentation (i=0):
  Input: [prompt(24)]
  After: [prompt(24) + latent1(8)] = [32 tokens]
  
First Inference Augmentation (i=12, 在某个 delimiter):
  Input: [prompt(24) + latent1(8) + generated(12)] = [44 tokens]
  After: [prompt(24) + latent1(8) + generated(12) + latent2(8)] = [52 tokens]
  
关键: latent1 保留在序列中，不会被移除
     Reasoner 可以看到所有的历史: prompt + latent1 + 生成的tokens + latent2
```

## 配置参数

```python
# 从 checkpoint 路径: pn=1_pl=8_in=3_il=8
prompt_latents_num = 8      # Prompt augmentation 插入 8 个 latent tokens
inference_latents_num = 8   # Inference augmentation 插入 8 个 latent tokens
max_prompt_aug_num = 1      # Prompt 最多增强 1 次
max_inference_aug_num = 3   # 推理过程最多增强 3 次

# LoRA 配置
weaver_lora: r=8, alpha=16, target=["q_proj", "v_proj"]
trigger_lora: r=8, alpha=16, target=["q_proj", "v_proj"]
```

## 总结

MemGen 的核心思想是通过 Weaver 生成"潜在记忆"来增强 Reasoner 的推理能力：

1. **Trigger** 决定何时插入潜在记忆（但实际退化为总是插入）
2. **Weaver** 通过 LoRA adapter 处理 Reasoner 的 hidden states，生成增强后的潜在记忆
3. **潜在记忆** 是 8 个连续的 token embeddings，拼接回 Reasoner 输入
4. **Reasoner** 在生成过程中可以看到这些潜在记忆，利用它们辅助推理

实验表明：
- Weaver 确实能提升难题的推理能力（GSM8K +20%）
- 但 Trigger 的门控机制未能发挥作用（退化为常函数）
- 性能提升 100% 来自 Weaver 架构本身，而非 Trigger 的选择性门控
