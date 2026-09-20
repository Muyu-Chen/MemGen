# MemGen 一次完整 inference 的 execution-level forensic trace

本文档只描述**实际执行**发生的事。每个数字都来自 `trace_memgen.py` 的一次真实运行。
凡结构性结论均标注 `文件:行号`。

原始日志：`logs/full_trace.txt`（事件流）、`logs/tensor_shapes.json`（张量统计）、
`logs/parameter_usage.json`（逐参数账目）、`logs/environment.json`、`results/sample_output.json`。

## 0. 本次运行是什么

复现的是官方**评测推理路径**，不是自己搭的简化循环：

```
MemGenRunner.evaluate()                                 memgen/runner.py:223-234
  +-- self.model = self.model.to(torch.bfloat16)           runner.py:225
  +-- _static_evaluate()                                memgen/runner.py:265-287
        +-- tokenizer.apply_chat_template(...)             runner.py:268-274
        +-- SingleTurnInteractionManager.run_agent_loop()  interactions/singleturn_interaction.py:85-119
              +-- MemGenModel.generate()     memgen/model/modeling_memgen.py:498-675
```

### 0.1 环境

| 项 | 值 |
|---|---|
| Git commit | `7e067fa6429f8e8e88eaf9c468f093f6a0bdcae3`（工作树 clean） |
| Python / torch / transformers / peft | 3.11.9 / 2.7.1+cpu / 4.55.4 / 0.17.1 |
| CPU | AMD Ryzen 7 8840H（`AMD64 Family 25 Model 117 Stepping 2`），16 逻辑核，torch 8 线程 |
| 内存 | 27.81 GiB 总；生成后进程 RSS **6660 MiB** |
| device / dtype | `cpu` / **bfloat16**（等于官方 `runner.py:225` 的显式 cast） |
| attn_implementation | `eager`（官方 `modeling_memgen.py:715-717` 硬编码 `flash_attention_2`，本机无 CUDA） |
| batch_size | **1**（官方 gsm8k.yaml 写 8，官方 eval 脚本写 4，见 CORRECTIONS_NEEDED.md） |
| 用时 | `generate()` **18.96 s** / 111 token，约 5.85 token/s |

### 0.2 模型来源与配置权威

| 组件 | 类 | 权重来源 |
|---|---|---|
| Reasoner | `Qwen2ForCausalLM` | `models/Qwen2.5-1.5B-Instruct/`（hub id `Qwen/Qwen2.5-1.5B-Instruct`） |
| Weaver | `MemGenWeaver` 内含 `PeftModel(Qwen2ForCausalLM)` | base + checkpoint 的 LoRA / query latents / projs |
| Trigger | `MemGenTrigger` 内含 `PeftModel(Qwen2ForCausalLM)` | base + checkpoint 的 LoRA / output_layer |
| projections | `nn.Linear(1536,1536)` x2 | checkpoint `projs.bin` |

checkpoint = `models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model`
（即官方 `scripts/eval/qwen2_5_gsm8k_sft.sh:30` 的 `LOAD_MODEL_PATH`）。

**配置取自 checkpoint 自带的 `config.json`（`MemGenConfig.from_pretrained`），不取
`configs/latent_memory/gsm8k.yaml`** —— 两者互相冲突，冲突表见 CORRECTIONS_NEEDED.md。实测生效值：

```
prompt_latents_len = 8      inference_latents_len = 8
max_prompt_aug_num = 1      max_inference_aug_num = 3
trigger_active     = False  torch_dtype = bfloat16
weaver/trigger lora: r=16, alpha=32, dropout=0.1, target_modules=[q_proj, v_proj]
hidden_size=1536, 28 层, vocab=151936, eos=151645, pad=151643
（Qwen 无 pad token，`modeling_memgen.py:128-134` 把 pad 回退成 EOS 专用标记 id 151643。）

### 0.3 generation config（真正被使用的那个）

由 `interactions/base_interaction.py:53-61` 构造，字段取自 `gsm8k.yaml` 的 `run.interaction`：

```json
{"max_new_tokens": 256, "do_sample": false, "temperature": 0.0, "top_p": 1.0, "top_k": 50,
 "pad_token_id": 151643, "eos_token_id": 151645, "use_cache": true,
 "weaver_do_sample": false, "trigger_do_sample": false}
```

`do_sample=False` 且 `temperature=0.0`，使 `_get_next_token`（`modeling_utils.py:76-89`）
走 `torch.argmax` 分支 —— Reasoner 侧和 Trigger 侧**都是贪心**。因此本 trace 完全确定：
连跑 4 次（run2/run3/run4/run5），111 个 token、3 次 Weaver 调用、3 次 Trigger 调用、
输出文本逐字符一致（日志文件本身的字节差异只来自 stdout 编码设置，与模型无关）。

### 0.4 结果概览

| 量 | 值 |
|---|---|
| prompt token 数 | 96（无 padding，batch=1） |
| augmentation 总次数 | **3** = 1 次 prompt augmentation + 2 次 inference augmentation |
| Reasoner forward | **111** 次 = 3 次整段重算（输入 104 / 124 / 197 行）+ 108 次单 token（KV cache 命中） |
| Trigger.forward 被调 | 3 次（gate 每步都跑，共 111 次，只有 3 次成为 candidate） |
| Trigger 内部模型 forward | **0 次** —— `active=False`，见 §3 |
| 最终序列长度 | 231 = 207 个真实 token 位 + 24 个 latent 位 |
| 生成 token 数 | 111，末位为 eos(151645)，正常收敛 |
| 输出 | 最终答案 18，与 gold 18 一致 |

## 1. Module map（本次推理涉及的组件）

| 组件 | 类 / 定义处 | 参数量 | 来源 | 本次是否 forward | LoRA |
|---|---|---|---|---|---|
| `model.reasoner` | `Qwen2ForCausalLM`，`modeling_memgen.py:104` | 1,543,714,304 | base 权重 | **是** 111 次（子模块累计 41,074 次） | 无 |
| `model.weaver` | `MemGenWeaver`，`weaver.py:6` | 1,545,924,098 | 见下两行 | 自身从不被 `__call__`，只调 `augment_prompt` / `augment_inference` | — |
| `model.weaver.model` | `PeftModel(Qwen2ForCausalLM)`，`modeling_memgen.py:100` | 1,545,893,376 | base + `weaver/weaver/adapter_model.safetensors` | **是** 3 次 | **112/112 张量参与** |
| `prompt_query_latents` | `nn.Parameter(8,1536)`，`weaver.py:22` | 12,288 | `weaver.bin` | **是**（第 1 次增强） | — |
| `inference_query_latents` | `nn.Parameter(8,1536)`，`weaver.py:28` | 12,288 | `weaver.bin` | **是**（第 2、3 次增强） | — |
| `prompt_latent_ln` + `prompt_latent_scale` | `weaver.py:34,36` | 3,073 | `weaver.bin` | **是** | — |
| `inference_latent_ln` + `inference_latent_scale` | `weaver.py:35,37` | 3,073 | `weaver.bin` | **是** | — |
| `reasoner_to_weaver` | `nn.Linear(1536,1536)`，`modeling_memgen.py:110` | 2,360,832 | `projs.bin` | **是** 3 次 | — |
| `weaver_to_reasoner` | `nn.Linear(1536,1536)`，`modeling_memgen.py:111` | 2,360,832 | `projs.bin` | **是** 3 次 | — |
| `model.trigger` | `MemGenTrigger`，`modeling_memgen.py:101` | 1,545,896,450 | — | `forward` 调 3 次，**内部子模块 0 次** | — |
| `model.trigger.model` | `PeftModel(Qwen2ForCausalLM)` | 1,545,893,376 | base + `trigger/trigger/adapter_model.safetensors` | **否，0 次** | 已加载，delta 恒零 |
| `model.trigger.output_layer` | `nn.Linear(1536,2)`，`trigger.py:18` | 3,074 | `trigger.bin` | **否，0 次** | — |
| generation loop | `modeling_memgen.py:542-661` | — | — | 111 轮 | — |
| KV cache | `DynamicCache`，`modeling_memgen.py:531,607,628-653` | — | — | 108 次命中，3 次清空 | — |

"是否 forward" 由 `ModuleCensus` 给**每一个** `nn.Module` 挂 forward hook 计票得出，
不是按变量名或注释推断；完整计数在 `logs/parameter_usage.json` 的 `module_call_counts`。

裸 `nn.Parameter`（query latents、latent_scale）不经过任何模块 forward，因此模块计票看不到它们。
改用**对象同一性**证明其被读取：`hooks.py` 在 `MemGenWeaver._augment` 入口处记录
传入的 `latents` / `latent_ln.weight` / `latent_ln.bias` / `latent_scale` 四个对象的 `id()`，
并断言 `latents is weaver.prompt_query_latents`（实测为真，见 `which_latent_bank` 字段）。

每个组件**首次被调用时的输入结构与参数量**（README 第 1 节要求的 in/out shape）由
`ModuleCensus` 一并记录，见 `logs/tensor_shapes.json` → `module_forward_signatures`，
共覆盖 900+ 个被执行的模块；本次三个顶层模型的输入/输出 shape 在 §4、§6 逐处给出。

## 2. Step A：96 个 prompt token 是怎么来的

**raw question**（GSM8K `test` split 第 0 题，官方 `shuffle=False` 顺序）：

> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes
> muffins for her friends every day with four. She sells the remainder at the farmers' market
> daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

gold answer 末段为 `#### 18`。

**构造链（谁做的每一步）**：

1. `data/gsm8k/builder.py:GSM8KBuilder._preprocess` —— 本次**直接调用官方函数**，不手抄模板。
   它把固定指令句 `Solve the math problem with proper reasoning, and make sure to put the
   FINAL ANSWER inside \boxed{}.` 与 `Question: {question}\n` 直接相加。
   两者之间**没有空格**，所以真实文本是 `...}.Question: Janet...`。
2. `modeling_memgen.py:137`（`_postprocess_models`）把 tokenizer 的 chat template
   **强制覆盖**为 `memgen/utils.py:CONVERSATION_TEMPLATE`（ChatML 风格：user 轮渲染成
   起始标记 + role + 换行 + content + 结束标记 + 换行；`add_generation_prompt=True`
   再追加一个 起始标记 + `assistant` + 换行）。
   因此 **chat template 确实被应用了**，且用的是 MemGen 自己覆盖的版本，不是 base model 自带的。
3. `runner.py:268-274` 调 `apply_chat_template(..., add_generation_prompt=True,
   return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True,
   return_dict=True)` 得到 `input_ids` / `attention_mask`。

**实测**：`input_ids` 形状 `[1, 96]`，`attention_mask` 全 1（`n_pad_positions = 0`，
batch=1 无需 padding）。序列以 ChatML 起始标记(id 151644) + `user` + 换行 开头，
以 `?` + 换行 + ChatML 结束标记(151645) + 换行 + 起始标记(151644) + `assistant` + 换行(198) 结尾。
**最后一个 token 是换行(198)，id 恰好在 delimiter 集合里** —— 这直接解释了 §3 的 prompt candidate。

完整 96 个 id 与逐 token 对照：`logs/environment.json` → `prompt_construction.input_ids`
与 `prompt_construction.decoded_tokens_with_ids`；chat template 渲染出的**最终 prompt 全文**：
同文件 → `prompt_construction.final_prompt_text`；
`attention_mask`（96 个 1）与 `position_ids`（0..95，由 `_generate_position_ids` 的
`cumsum(-1)-1` 得到，`modeling_utils.py:91-94`）：`logs/tensor_shapes.json` →
`events_ordered[kind=generate_entry]`。

## 3. Step B：augmentation candidate 是怎么选出来的，Trigger 到底做了什么

### 3.1 谁在每步被调用

`generate()` 主循环每一轮都调 `_should_augment`（`modeling_memgen.py:545-551`）。
本次共调用 **111 次**，但只有 **3 次**返回非 `-100`：

| 事件 seq | 位置 | 该步最后一个 token | 是否 delimiter | decision |
|---|---|---|---|---|
| 3 | i=0（prompt 末尾） | 换行 (198) | —（prompt 分支不看 delimiter） | **1** |
| 43 | 第 12 个生成 token | `,` (11) | 是 | **1** |
| 242 | 第 77 个生成 token | `,` (11) | 是 | **1** |
| 9, 12, … | 其余 108 步 | `Jan` / `et` / … | 否 | -100（不咨询 Trigger） |

两条分支（`modeling_utils.py:263-277`）：

- `is_prompt=True`（只有 i=0）：`aug_vector` 初始化为 `0`，`trigger_indices` 取全部 →
  **prompt 位置无条件咨询 Trigger**，与 delimiter 无关。
- `is_prompt=False`：先置 `-100`，再由 `_check_ends_with_delimiter` 把以 delimiter 结尾的行改成 `0`，
  然后把 `sentence_augment_count >= max_inference_aug_num(=3)` 的行重新置回 `-100`。

### 3.2 delimiter 判据是 token-id 精确匹配，不是文本匹配（重要）

`_check_ends_with_delimiter`（`modeling_utils.py:153-175`）只看序列**最后一个 token 的 id**
是否落在 `self._get_delimiter_token_ids()` 里。实测该集合为 `{11, 13, 198}`，
分别来自 `tokenizer.encode(",")`、`encode(".")`、`encode("\n")`。

本次 111 步里，落在 `{11,13,198}` 的只有 **2 步**（两个 `,`）；id 13 与 198 各出现 **0 次**。
但生成的文本里明明有 3 处句号+换行 —— 它们被 BPE 合并成了**单个 token id 624（内容为句号紧跟换行）**，
不等于 13 也不等于 198，所以**不触发** augmentation。

> 校验脚本：`logs/delimiter_check.txt`。

结论：**"每个 delimiter 处检查 Trigger" 这句话在代码层面是错的**，实际是
"每个 *恰好是 11/13/198 这三个 id 之一* 的 token 之后才检查"。句子边界（句号+换行合并成一个 token）
在 GSM8K 输出上系统性漏检，这是本次只发生 2 次 inference augmentation 的直接原因，
而不是 Trigger 做了保守决策。

### 3.3 Trigger 的 `active=False` 到底是什么意思

`scripts/eval/qwen2_5_gsm8k_sft.sh:20` 设 `TRIGGER_ACTIVE=False`，checkpoint `config.json`
也是 `"trigger_active": false`。`MemGenTrigger.forward`（`trigger.py:26-42`）的实际语义：

```
if self.active:   # 走 LoRA 模型 + output_layer
else:             # 直接构造 zeros(B, L, 2) 并把 logits[..., 1] 置 1.0
```

所以 **`active=False` 不是"关闭 Trigger"，而是"把决策函数硬编码为恒等于 class 1"，即永远增强**。
`trigger.model`（1.54 B 参数）与 `trigger.output_layer` 在这条路径上**一次都没有被调用**
（`ModuleCensus`：`trigger.model` 前缀下被 forward 的模块数 = **0**）。

三次 Trigger 调用的实测输出完全相同：

```
trigger_active_attr        = False
code_path                  = hardcoded_branch
logits_at_last_position    = [0.0, 1.0]
softmax_at_last_position   = [0.268941, 0.731059]     <- 不是 0/1，只是 [0,1] 的 softmax
argmax                     = 1
```

decision 的产生方式：`_should_augment` 取 `trigger_logits[:, -1]`（`modeling_utils.py:285`），
交给 `_get_next_token`（`trigger_do_sample=False` ⇒ `argmax`，`modeling_utils.py:287-293`）。

**因此本 checkpoint 在这条配置下没有任何"门控"可言**：不能据此对 Trigger 的选择能力下任何结论。

### 3.4 Trigger 看不到 latent

三次 Trigger 调用的 `input_ids` 长度是 **96 / 108 / 173**，而同一时刻 Reasoner 的
`inputs_embeds` 长度是 **96(→104) / 116(→124) / 189(→197)**。差值恰为已插入的 latent 数 8 / 8 / 16。

原因：`_should_augment` 只接收 `current_input_ids`，并自己用
`(input_ids != pad_token_id)` 重算 attention mask（`modeling_utils.py:264,270`）。
而 latent 没有 token id，从不进入 `current_input_ids`（`_append_one_step` 只 append 真实 token）。
**所以 Trigger 的输入是纯 token 序列，历史 latent 对它完全不可见**，
它拿到的 position_ids 也和 Reasoner 的不同（108 步位置 vs Reasoner 的 116 行）。

## 4. Weaver augmentation 逐行拆开（第 1 次增强，prompt augmentation）

调用链：`modeling_memgen.py:562-576` → `weaver.augment_prompt`（`weaver.py:96`）→
`MemGenWeaver._augment`（`weaver.py:52`）。

进入前状态（实测）：`current_inputs_embeds [1,96,1536]`、`attention_mask [1,96]`、
`position_ids [1,96]`（值 0..95）。

### 4.1 Weaver 拿到的到底是 input embeddings 还是 hidden states？

**是 input embeddings。** 三重证据：

1. 代码：`modeling_memgen.py:522` `inputs_embeds = reasoner.get_input_embeddings()(input_ids)`，
   `527` 赋给 `current_inputs_embeds`，`562` 切片成 `candidate_inputs_embeds`，
   `567` 送进 `reasoner_to_weaver`。中间**没有任何 Reasoner forward**。
2. 数值：`reasoner_to_weaver` 的输入被逐元素验证等于
   `embedding_table[input_ids]` —— 结果 `equals_embedding_table_lookup_of_prompt_ids = true`。
3. 时序：该次调用时 `reasoner_forwards_completed_before_this_call = 0`。
   **Reasoner 一次都还没跑，不可能有 hidden states 可用。**

数值特征也符合 embedding 而非末层 hidden state：
`mean 8.7e-05, std 0.02472, min -0.1377, max 0.3008, 整体 L2 norm 9.4926`（96 行）。

### 4.2 `reasoner_to_weaver` 投影

`nn.Linear(1536,1536)`，weight 形状 `[1536,1536]`，bias 存在。

| | shape | dtype | mean | std | norm |
|---|---|---|---|---|---|
| 输入 | [1,96,1536] | bfloat16 | 8.7e-05 | 0.02472 | 9.4926 |
| 输出 | [1,96,1536] | bfloat16 | -0.000649 | 0.020747 | 7.9708 |

逐行 L2 norm：输入约 0.43~1.21，输出约 0.62~0.97（被压缩并同质化）。
权重 `std=0.014738`、`absmax=0.026611` —— **超过** `nn.Linear` 初始化的
kaiming-uniform 边界 `1/sqrt(1536)=0.025516`，说明这层被训练过（见 §7）。

### 4.3 Query latents

`prompt_query_latents` 形状 `[8,1536]`（`requires_grad=True`，但推理在 `@torch.no_grad()` 下）。
先 `LayerNorm` 再乘 `latent_scale`：

| 量 | 实测 |
|---|---|
| 原始逐行 norm | 38.69, 40.39, 38.16, 39.28, 39.31, 39.28, 37.65, 40.13 |
| LN+scale 后逐行 norm | 39.19, 39.20, 39.19, 39.19, 39.17, 39.22, 39.19, 39.19（同质化到 √1536=39.192） |
| `latent_scale` | 1.0 |
| `prompt_latent_ln.weight` | **精确全 1**（= LayerNorm 初始值） |
| `prompt_latent_ln.bias` absmax | 0.001999（初始应为 0，说明 bias 被更新过） |

本次增强用的是 **prompt 组**（`which_latent_bank = prompt_query_latents`，由对象同一性判定）。
第 2、3 次用 **inference 组**：其 `inference_latent_ln.weight` **不全为 1**（最大偏离 0.003906），
bias absmax 0.001419。两组参数都来自 `weaver.bin`。

### 4.4 拼接顺序：**[context, query_latents]**

`weaver.py:70`：`inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)` —— context 在前。

实测 Weaver 内层模型真正收到的张量（由 `register_forward_pre_hook` 捕获，不是推断）：

```
inputs_embeds shape      [1, 104, 1536]      (= 96 context + 8 query)
head_equals_projected_context            = true
tail_equals_normed_query_latents         = true
alternative_order_q_first_matches        = false     <- [query, context] 被明确排除
attention_mask shape     [1, 104]，最后 8 位全 1
position_ids  前 3 位    [0, 1, 2]
position_ids  后 8 位    [96, 97, 98, 99, 100, 101, 102, 103]
```

position_ids 由 `weaver.py:77-80` 顺延（`max(position_ids) + arange(K) + 1`），
即 query latent 占据 **96..103** 这 8 个位置。

### 4.5 Weaver forward 与 LoRA 是否真的执行

`weaver.model` 是 `PeftModel(Qwen2ForCausalLM)`，active adapter = `weaver`
（`modeling_memgen.py:823` 的 `set_adapter`）。本次输入走 `inputs_embeds=`，
`given_via_inputs_embeds = true`，**没有传 input_ids**。

LoRA 执行证据（`ModuleCensus` 计数，3 次 Weaver forward 对应）：

```
...layers.0.self_attn.q_proj                 x3
...layers.0.self_attn.q_proj.base_layer      x3
...layers.0.self_attn.q_proj.lora_A.weaver   x3
...layers.0.self_attn.q_proj.lora_B.weaver   x3
...layers.0.self_attn.q_proj.lora_dropout.weaver  x3
```

112/112 个 Weaver LoRA 张量所属模块都被 forward。scaling = alpha/r = 32/16 = **2.0**。
输入是 `inputs_embeds`，所以 Weaver 自己的 `embed_tokens`（233,373,696 参数）**没有被使用**。

Weaver 输出：`hidden_states` 共 **29** 层，末层形状 `[1, 104, 1536]`；
`logits` 形状 `[1, 104, 151936]` 被计算出来了，但 `_augment` 只返回 hidden states 切片，
**这些 logits 无人消费**（`weaver.py:85-92`）。

### 4.6 取哪 K 个 hidden states：**最后 K 个**

`weaver.py:91-92`：`hidden_states = outputs.hidden_states[-1]` 然后
`latents_hidden_states = hidden_states[:, -latents_num:, :]`。

实测切片比对（对同一次真实 forward 的输出做双向验证）：

```
returned shape                     [1, 8, 1536]
equals_final_hidden_last_K   = true
equals_final_hidden_first_K  = false     <- "取前 K 个" 被明确排除
```

取出的 latent 统计：`mean -0.108695, std 2.5446, norm 282.318, min -75.0, max 20.375`；
逐行 norm 103.6 / 94.8 / 101.5 / 94.4 / 94.1 / 79.9 / 123.6 / 101.4；
**8 个 slot 之间平均余弦 0.773**（第 2、3 次分别 0.720、0.726）——
即 8 个 latent 高度相似但彼此可区分，没有正交化约束。

### 4.7 `weaver_to_reasoner` 投影回 Reasoner 空间

`nn.Linear(1536,1536)`，weight `absmax 0.028076`（同样超出初始化边界 0.025516 ⇒ 被训练过）。

| | shape | mean | std | norm |
|---|---|---|---|---|
| 输入（Weaver 空间 latent） | [1,8,1536] | -0.108695 | 2.5446 | 282.318 |
| 输出（Reasoner 空间 latent） | [1,8,1536] | -0.030948 | 1.6306 | 180.779 |

输出逐行 L2 norm：**65.11, 59.65, 66.64, 61.56, 63.66, 52.48, 76.93, 62.67**。

> 量级关键点：同一序列里**真实 token 的 embedding 逐行 norm 约 0.43~1.21**，
> 而插入的 latent 逐行 norm 约 **52~77**，是 token embedding 的 **约 60 倍**。
> 这就是为什么第 2 次 `reasoner_to_weaver` 的输入整体 norm 从 9.49 暴涨到 181.06。
> latent 在数值上完全主导它后面的表示。

## 5. latent 如何真正插回 Reasoner

`modeling_memgen.py:578`：`candidate_inputs_embeds = torch.cat([candidate_inputs_embeds,
latent_inputs_embeds], dim=1)`；`579` 同样扩展 attention_mask；`604-606` 重建
`current_inputs_embeds` / `current_attention_mask` 并 `current_position_ids =
self._generate_position_ids(current_attention_mask)`。

| | augmentation 前 | 后 |
|---|---|---|
| `inputs_embeds` | [1, 96, 1536] | [1, **104**, 1536] |
| `attention_mask` | [1, 96]，全 1 | [1, 104]，全 1（`attention_mask_sum = 104`） |
| `position_ids` | 0..95 | 0..**103** |
| `current_input_ids` | 96 | 仍 **96**（之后才 append 生成 token） |
| latent 占用的下标区间 | — | embeds 的 **[96, 104)** |

### 5.1 latent 有没有 token ID？——没有

- latent 只以向量形式存在于 `current_inputs_embeds`，**从不进入 `current_input_ids`**
  （只有 `_append_one_step` 会把真实 token id append 进去，`modeling_utils.py:314`）。
- 实测每个生成步都保持 `embeds_len - ids_len = 8 / 16 / 24`，即 latent 数；
  最终 `ids_len = 207`（96 prompt + 111 生成）而 `embeds_len = 231`。
- **反向验证**：把插入 Reasoner 空间的 M0（8 个向量）与整个词表 151,936 个 embedding
  逐一算余弦，最大值只有 **0.098395**，`is_exact_embedding_of_some_token = false`。
  作为正对照，同一函数作用在 prompt 的 96 个 embedding 上时每个位置余弦都是 **1.0**
  且 id 与 `input_ids` 完全一致。

### 5.2 有没有经过 tokenizer？——没有

latent 是连续向量，没有任何 `decode` / `encode` 调用涉及它们。tokenizer 只在
Step A（输入）和最后 `batch_decode(completion_ids)`（输出）被使用。

### 5.3 有没有经过 LM head？——经过了，但结果被丢弃

增强后第一次 Reasoner forward 是**整段** 104 行（`past_key_values_none = true`），
所以 `lm_head` 对**全部 104 个位置**（含 8 个 latent 位置）算了 logits：
实测 `reasoner_logits_shape = [1, 104, 151936]`。
但 `_append_one_step` 只取 `reasoner_outputs.logits[:, -1]`（`modeling_utils.py:312`），
即**最后一个 latent 之后的那个位置**的 logits。latent 位置自身的 logits 被计算后直接丢弃。

所以准确表述是：latent 不会被 decode（它没有 id），但它的**位置确实产出了 logits，只是没人读**；
而**下一个真实 token 的预测**读的是"最后一个 latent 作为输入"之后那个位置的输出。

### 5.4 Reasoner 如何"看到" latent

没有特殊注意力机制。latent 就是被 `cat` 进 `inputs_embeds` 的额外序列位置，
`attention_mask` 对应位置为 1，position_ids 连续。标准 causal attention 使其后所有位置
都能 attend 到这 8 个向量。这也是为什么它们会进入 KV cache（见 §6）。

---

## 6. 生成、KV cache、以及第 2、3 次增强

### 6.1 KV cache 的真实行为

`generate()` 用 `use_cache=True` 逐步生成（`modeling_memgen.py:636-643`）。
两种形态，实测分布：

| 输入形状 | 次数 | 含义 |
|---|---|---|
| `[1, 104, 1536]` | 1 | 第 1 次增强后整段重算（Reasoner 调用 #1） |
| `[1, 124, 1536]` | 1 | 第 2 次增强后整段重算（#13） |
| `[1, 197, 1536]` | 1 | 第 3 次增强后整段重算（#78） |
| `[1, 1, 1536]` | **108** | 常规单 token 步，`past_key_values` 非空 |

机制：每次 augmentation 后 `current_cache = None`（`modeling_memgen.py:607`），
下一次 forward 重算整条序列，cache 长度 = 整条长度；随后每步只喂最后 1 个 token，
并靠 `assert current_inputs_embeds.size(1) == current_cache.get_seq_length() + 1`（`629`）对齐。
例：调用 #2 的 `past_kv_seq_len = 104`，`position_ids_last5 = [104]`，`attention_mask_sum = 105`。

**latent 会进入 KV cache**：cache 长度 104 包含了那 8 个 latent 位置的 K/V，
之后 12 个单 token 步全部 attend 在这些 K/V 上。

### 6.2 第 2 次增强（inference augmentation）—— 完整拆解

第 12 个生成 token 是 `,`(id 11) → 下一个循环位置成为 candidate → decision=1。
此时 `sentence_augment_count` 从 `[0]` 变 `[1]`（注意 `modeling_memgen.py:558`：
**i==0 的 prompt 增强不计数**）。

调用前 Reasoner 已累积序列结构（`current_inputs_embeds`，共 116 行）：

```
[0, 96)     prompt tokens
[96, 104)   M0   <- 上一次插入的 latent
[104, 116)  12 个已生成 token（Janet's ducks lay 16 eggs per day,）
```

`reasoner_to_weaver` 第 2 次调用的输入形状 = `[1, 116, 1536]`，整体 norm 181.06。

**持久性验证（直接比对张量，不是看代码猜）**：
把 M0 当初插入时的向量存下来，在第 2、3 次 Weaver 调用的输入里按 `[96,104)` 切片比对：

```
checked_at_r2w_call 2:  M0  still_present_bitwise_equal = true,  max_abs_diff = 0.0
checked_at_r2w_call 3:  M0  still_present_bitwise_equal = true,  max_abs_diff = 0.0
checked_at_r2w_call 3:  M1  still_present_bitwise_equal = true,  max_abs_diff = 0.0
```

所以：**历史 latent 既没被移除也没被覆盖，逐比特原样留在序列里，并且被再次喂进 Weaver**。

第 2 次 Weaver：`inference_query_latents`（K=8）+ 116 行 context → 输入 `[1, 124, 1536]`，
query 的 position_ids = 116..123，取末 8 行 → `weaver_to_reasoner` → M1 插入 `[116,124)`。
M1 逐行 norm 92.25/92.51/79.75/84.04/82.47/104.64/93.70/87.09（Weaver 空间）；
投回 Reasoner 空间后 60.0/61.0/52.4/55.9/54.3/65.4/59.0/56.1。

第 3 次增强完全同构：第 77 个 token 是 `,`，Weaver 输入 `[1, 189, 1536]`
（= 96 prompt + M0(8) + 12 token + M1(8) + 55 token），M2 插入 `[189, 197)`。

### 6.3 真实序列增长时间线

```
Prompt (96 tokens)
  |  i=0: gate candidate(prompt) -> Trigger 恒 1 -> Weaver(prompt bank) -> M0
  v
[P:96][M0:8]                                   = 104 行,  cache=None -> 整段重算
  |  12 个单 token 步 (token 12 = ',')
  v
[P:96][M0:8][T:12]                             = 116 行
  |  inference aug #1 -> Weaver(inference bank) -> M1
  v
[P:96][M0:8][T:12][M1:8]                       = 124 行,  cache=None -> 整段重算
  |  64 个单 token 步 (token 77 = ',')
  v
[P:96][M0:8][T:12][M1:8][T:64]                 = 189 行
  |  inference aug #2 -> M2
  v
...[M2:8]                                      = 197 行,  cache=None -> 整段重算
  |  33 个单 token 步，第 111 个生成 token = eos(151645) -> break (modeling_memgen.py:656)
  v
最终: ids 207 位 / embeds 231 行 (差 24 = 3x8 latent)
```

实测 `final_layout`：M0 = [96,104)、M1 = [116,124)、M2 = [189,197)。

**历史 latent 是 append，不是 replace、也不是 update。** 每次增强只读取历史、
在尾部追加新 block，从不修改已存在的 latent 向量（上面的逐比特相等即为此证）。

### 6.4 没有走到的那条分支

`modeling_memgen.py:610-624`：一旦 `sentence_augment_count >= 3`，代码改为
`reasoner.generate(inputs_embeds=..., use_cache=False)` 一次性生成剩余 token 并 `break`，
**此后不再做任何 delimiter/gate 检查**。本次 count 最大到 2，故该分支未被触发
（`reasoner_forward_bulk_path` 事件数 = 0；111 次 Reasoner 调用的 `use_cache` 全为 True）。
这是一条真实存在的路径，配置成 `max_inference_aug_num=3` 时会在第 3 次增强后接管。

### 6.5 附加实验：latent 真的依赖 context 吗

`trace_memgen.py:context_sensitivity` 直接调用真实 `weaver.augment_prompt`，
latent bank / LN / scale 三次都相同、token 多重集合也相同，只改排列：

```
同输入重复两次      : mean row cosine = 1.0, bitwise_identical = true     (确定性)
同样 token 逆序输入 : mean row cosine = 0.970883, relative L2 diff = 0.245388
```

token 集合完全一样、只是顺序不同，latent 就明显改变 ⇒ **latent 确实是 context 条件的产物**，
而不是 8 个固定的可学习向量。机制上就是 causal attention：query 位于序列末尾
（位置 96..103），能 attend 到前面全部 96 个 context 位置；若把 query 放在开头，
它就 attend 不到任何 context，整个机制失效。

## 7. 最终输出

```
Janet's ducks lay 16 eggs per day, so she has 16 - 3 = <<16-3=13>>13 eggs left after breakfast.
She has 13 - 4 = <<13-4=9>>9 eggs left after baking muffins for her friends.
She sells the 9 eggs for $2 each, so she makes 9 * $2 = $<<9*2=18>>18 every day at the farmers' market.\boxed{18}
```

111 个 token，末位 eos(151645)，答案 18 与 gold 18 一致。
注意输出仍带 GSM8K 的 `<<...>>` 计算器记号 —— 这是 SFT 数据格式泄漏，不是 MemGen 特有行为。

参数账目（loaded / trained / actually-used 三分）与逐组表格见 **PARAMETER_FLOW.md**。
一句话结论：模型共 **4,640,256,516** 参数，本次推理实际参与计算的只有
**2,860,986,370**，**1,779,270,146（38.34%）被加载但一次都没执行** ——
其中几乎全部是 Trigger 的 1.5459 B。

---

## 8. Questions Answered

以下 18 条全部来自本次真实执行，括号内为证据位置。

1. **Weaver 的输入是 Reasoner hidden state 还是 input embeddings？**
   input embeddings。`modeling_memgen.py:522→562→567`；数值上等于 `E[input_ids]`；
   且该时刻 Reasoner forward 计数为 0。（§4.1）
2. **Query latents 在 context 前还是后？**
   后。`weaver.py:70` 的 `cat([inputs_embeds, latents])`；实测 Weaver 收到的
   `[1,104,1536]` 中前 96 行 = 投影后的 context、后 8 行 = 归一化后的 query；
   `[query, context]` 排列实测不匹配。（§4.4）
3. **为什么 query latent 能获取 context 信息？**
   因为它被放在序列**末尾**，causal attention 下可 attend 到全部前文；position_ids 顺延为 96..103，
   attention_mask 对应位为 1。逆序 token 实验（cosine 0.971、相对 L2 差 0.245）
   证明输出确实随 context 排列变化，而非固定向量。（§6.5）
4. **最终取哪 K 个 hidden states？**
   末层的**最后** K=8 个：`weaver.py:91-92`；实测 `equals_final_hidden_last_K=true`、
   `equals_final_hidden_first_K=false`。（§4.6）
5. **latent memory 有没有 token ID？**
   没有。只存在于 `inputs_embeds`，从不进 `current_input_ids`（`modeling_utils.py:314`）；
   与 151,936 个词表 embedding 的最大余弦仅 0.098。（§5.1）
6. **latent 是否经过 tokenizer？**
   否。tokenizer 只用于 Step A 编码与最终 decode 输出文本。（§5.2）
7. **latent 是否经过 LM head？**
   经过但不生效：增强后整段重算使 `lm_head` 对全部 104 行（含 8 个 latent 位）产出 logits
   （实测 `[1,104,151936]`），只有 `[:, -1]` 被读取，latent 位置的 logits 直接丢弃。（§5.3）
8. **latent 如何影响 Reasoner 的下一步 token prediction？**
   作为额外的序列前缀参与 attention：下一个 token 由"最后一个 latent 之后那个位置"的
   hidden state 决定，该位置的 attention 覆盖了全部 8 个 latent；并且 latent 进入 KV cache，
   后续所有单 token 步都 attend 到它们的 K/V。（§5.4、§6.1）
9. **第二次 Weaver 调用时，第一次 latent 是否还存在？**
   存在，逐比特相同（`max_abs_diff = 0.0`）。（§6.2）
10. **第二次 Weaver 是否能看到第一次 latent？**
    能。第 2 次 `reasoner_to_weaver` 输入是 `[1,116,1536]` 的**整段累积序列**，
    其中 `[96,104)` 就是 M0；第 3 次同理看到 M0 与 M1。（§6.2）
11. **历史 latent 是 append、replace 还是 update？**
    **append**。只读取历史、在尾部追加，从不修改已插入向量。（§6.2、§6.3）
12. **context 长度如何随 augmentation 增长？**
    每次 +8：96 → 104 → 116 → **124** → 189 → **197** → … → 231。
    `embeds_len - ids_len` 依次为 8 / 16 / 24。（§6.3）
13. **Trigger 在当前 released checkpoint / eval 配置下做了什么？**
    什么都没判断。`trigger_active=False` 使 `trigger.py:37-40` 直接返回
    `logits[...,1]=1.0` 的常量张量（softmax 后是 `[0.269, 0.731]`，不是 0/1），argmax 恒为 1；
    Trigger 的 1.5459 B 参数与 output_layer **零次 forward**。它的 LoRA delta 56/56 **恒为 0**，
    `output_layer` 权重仍停在 `nn.Linear` 初始化的 kaiming 边界上 —— 即这个 Trigger
    从未被训练过，其"恒等于 1"是硬编码加随机头，**不能**用来推断 trigger-stage 训练后的行为。（§3.3）
14. **本次哪些 LoRA adapter 实际参与计算？**
    只有 `weaver` adapter：112/112 张量所属模块被 forward，scaling=2.0，
    56/56 target module 的 `||B@A||_F` 非零（均值 0.040）。
    `trigger` adapter 既没被 forward、delta 也恒零。（§4.5、§3.3）
15. **Reasoner / Weaver / Trigger 是共享权重还是三个独立实例？**
    三个**完全独立**的实例。官方 `from_config`（`modeling_memgen.py:715-717`）做三次
    `from_pretrained`；实测 226 个同名参数中**共享内存缓冲区的数量 = 0**，
    但未被 LoRA 包装的 `o_proj.weight` 在两侧数值完全相同 —— 值相同是因为同源，张量对象独立。（§0.2）
16. **哪些参数来自 base model，哪些来自 checkpoint，哪些是额外 learned parameters？**
    base model：Reasoner 全部 1.5437 B；Weaver/Trigger 各自非 LoRA 的 1.5437 B。
    checkpoint 追加：Weaver LoRA 2,179,072、Trigger LoRA 2,179,072、
    两个投影各 2,360,832、两组 query latents 各 12,288、两个 LN 各 1,536×2、两个 scale 各 1、
    Trigger `output_layer` 3,074。其中 `query_latents` / `latent_ln` / `latent_scale` /
    两个投影 / `output_layer` 都是 **MemGen 新增的 learned parameters**，base model 里没有对应物。（§7）
17. **prompt augmentation 与 inference augmentation 的唯一区别是什么？**
    三点，且仅此三点：
    (a) 用 `prompt_query_latents`+`prompt_latent_ln`+`prompt_latent_scale` 还是 inference 那组
    （`weaver.py:96-125`，两条路径共用同一个 `_augment`）；
    (b) prompt 那次**不增加** `sentence_augment_count`（`modeling_memgen.py:558` 的 `if i != 0`），
    所以不受 `max_inference_aug_num` 限制；
    (c) 触发时机：prompt 位置无条件咨询 Trigger，inference 位置要求最后一个 token id 命中
    `{11,13,198}`（`modeling_utils.py:263-277`）。
    投影、拼接顺序、取末 K、插回方式**完全相同**，`max_prompt_aug_num` 在 `generate()` 里根本没被读。
18. **KV cache 在 augmentation 前后如何处理？**
    增强分支末尾 `current_cache = None`（`modeling_memgen.py:607`），下一次 Reasoner forward
    重算**整条**序列（实测 3 次：104 / 124 / 197 行），因此 latent 被写入新 cache；
    之后每步 `use_cache=True` 只喂最后一个 token（实测 108 次，形状 `[1,1,1536]`），
    靠 `size(1) == cache_len + 1` 断言对齐（`modeling_memgen.py:629`）。

---

## 9. 复现方式

```bash
# 从 MemGen 源码仓库根目录开始
cd reproduction/wholeProcess
../../../.venv/Scripts/python.exe trace_memgen.py --max-new-tokens 256   # 约 60 s
../../../.venv/Scripts/python.exe fingerprint_checkpoint.py              # 只读 checkpoint 文件
```

默认路径由脚本位置推导：源码仓库是 `../../`，模型目录是仓库同级的 `../../../models/`。可用环境变量覆盖：
`MEMGEN_REPO`、`MEMGEN_BASE_MODEL`、`MEMGEN_CKPT`；
GPU 环境下 base model 的 hub id 是 `Qwen/Qwen2.5-1.5B-Instruct`。

与官方代码的**全部**偏离（其余一律照官方路径执行）：

| 偏离 | 原因 | 影响 |
|---|---|---|
| `attn_implementation`: `flash_attention_2` → `eager` | 本机无 CUDA | 只影响注意力实现与速度，不改数学 |
| `batch_size`: 8/4 → 1 | CPU 运行时 | **有语义影响**：batch=1 时 `merged_inputs_embeds` 的左 padding 分支（`modeling_memgen.py:591-602`）不会执行 |
| `max_new_tokens`: 1024 → 256 | CPU 运行时 | 无影响：模型在第 111 步自行输出 eos |

调试代码全部在本目录内（`hooks.py` 只挂 hook、只包一层"调用原函数再记录"的 wrapper，
未修改任何 MemGen 算法行为；`trace_memgen.py` 结束时 `detach()` 还原所有 patch）。

## 10. 与既有文档的冲突

本 trace 有多处与 `reproduction/REPRODUCTION.md`、
`reproduction/MEMGEN_ARCHITECTURE_EXPLAINED.md` 的描述不一致。
按要求不在这里改动旧结论，汇总在 **CORRECTIONS_NEEDED.md**。



