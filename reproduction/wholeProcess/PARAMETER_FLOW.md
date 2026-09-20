# PARAMETER FLOW —— 到底哪些参数在这次 inference 里工作

三个概念必须分开，本文档逐条给证据：

```
loaded parameters      权重文件里有、被实例化/复制到显存(内存)
trainable parameters   requires_grad == True
actually used          本次 generate() 中, 拥有该张量的模块真的执行了 forward
```

判据说明：
* `used` 由 `hooks.py:ModuleCensus` 给 MemGenModel 下**每一个** `nn.Module`
  挂 forward hook 计票得出；参数按"最近祖先模块"归属。
* 裸 `nn.Parameter`（query latents、latent_scale、latent_ln 的 weight/bias）
  不经过任何模块 `__call__`，改用**对象同一性**证明被读取：
  `MemGenWeaver._augment` 入口处记录传入张量的 `id()`。
* 全程在 `@torch.no_grad()` 下，所以 `requires_grad=True` 只表示"优化器会想更新它"，
  与本次是否执行无关。

## 1. 总表

| Component | Loaded | Trained checkpoint? | Used in this forward? | LoRA active? | requires_grad | Parameter count |
|---|---|---|---|---|---|---|
| **Reasoner**（`Qwen2ForCausalLM`） | 是（base model） | 否，纯 base 权重，无 MemGen 更新 | **是** 111 次 forward，338/338 张量 | 无 adapter | **0**（`fix_model_parameters`，`modeling_memgen.py:125`） | 1,543,714,304 |
| **Weaver base**（非 LoRA） | 是（base model） | 否，与 Reasoner 同源 | **是** 但 **337/338** | — | 0（被 PeftModel 冻结） | 1,543,714,304 |
| **Weaver LoRA**（`q_proj`/`v_proj`，r=16） | 是（`weaver/weaver/adapter_model.safetensors`） | **是**：56/56 个 target module 的 `‖B@A‖_F` 非零，均值 0.040，`lora_B` 非全零 | **是** 3 次 forward，112/112 张量 | **是**，adapter 名 `weaver`，scaling 2.0 | 2,179,072 | 2,179,072 |
| **Prompt query latents**（含 `prompt_latent_ln`+`scale`） | 是（`weaver.bin`） | 混合：LN weight 精确全 1、scale 精确 1.0（=初始值）；LN bias absmax 0.001999 ≠ 0（被更新）；latents 统计上与 `randn` 无法区分 | **是**（仅第 1 次增强，对象同一性证明） | — | 15,361 | 15,361 |
| **Inference query latents**（含 `inference_latent_ln`+`scale`） | 是（`weaver.bin`） | 混合：LN weight 偏离 1 达 0.003906、bias 0.001419（被更新）；scale 精确 1.0 | **是**（第 2、3 次增强） | — | 15,361 | 15,361 |
| **R→W projection** | 是（`projs.bin`） | **是**：`absmax 0.026611 >` kaiming 边界 `0.025516` | **是** 3 次 | — | 2,360,832 | 2,360,832 |
| **W→R projection** | 是（`projs.bin`） | **是**：`absmax 0.028076 >` 边界 | **是** 3 次 | — | 2,360,832 | 2,360,832 |
| **Trigger base**（非 LoRA） | 是（base model，另一次独立加载） | 否 | **否**，0/338 —— 一次都没 forward | — | 0 | 1,543,714,304 |
| **Trigger LoRA**（`q_proj`/`v_proj`，r=16） | 是（`trigger/trigger/adapter_model.safetensors`） | **否，可证明未训练**：56/56 个 module 的 `lora_B` **全零** ⇒ `‖B@A‖_F = 0`（PEFT 的 lora_B 零初始化约定） | **否**，0/112 | 名义上 `set_adapter("trigger")` 被调用，但 `trigger.model` 从未 forward | 2,179,072 | 2,179,072 |
| **Trigger head**（`output_layer`） | 是（`trigger.bin`） | **否**：`absmax 0.025513` 恰等于 kaiming 边界 `0.025516` ⇒ 仍在初始值 | **否**，0/2 | — | 3,074 | 3,074 |

**合计**：4,640,256,516 参数；本次实际参与 **2,860,986,370**；
加载但从未执行 **1,779,270,146（38.34 %）**；`requires_grad=True` 共 **9,113,604**。

## 2. 三个反直觉的点

**(a) "checkpoint 里有这个文件" ≠ "这个模块被训练过"。**
`trigger/trigger/adapter_model.safetensors` 确实存在且被成功加载（key remap 后 112/112 匹配），
但它的 `lora_B` 逐张量全零 —— 按 PEFT 约定这就是从未收到梯度的状态。
`trigger.bin` 的 `output_layer` 权重 absmax 停在 `1/√1536` 的初始化边界上，同样未被更新。

**(b) `requires_grad=True` 不等于"被使用"，`False` 也不等于"没被用"。** 两个方向都成立：
* Trigger LoRA / Trigger head 共 **2,182,146** 个参数 `requires_grad=True`，
  但本次一次都没执行（`active=False` 分支根本不进模型）。
* Reasoner 全部 1.5437 B 参数 `requires_grad=False`（被 `fix_model_parameters` 冻结），
  却是本次唯一在做实际预测的模块。

**(c) Weaver 自己那份 233,373,696 参数的 `embed_tokens` 没被用上。**
因为 Weaver 是通过 `inputs_embeds=` 被调用的（`weaver.py:85-90`），词表 embedding 查表被完全跳过。
它被加载、占内存，但对本次推理零贡献。

## 3. 一次 augmentation 的参数流（按执行顺序）

```
input_ids (96)
  └─ reasoner.model.embed_tokens .................... 查表，得 [1,96,1536]，行 norm ~0.43-1.21
      └─ reasoner_to_weaver(Linear 1536x1536) ....... [1,96,1536] -> [1,96,1536]，行 norm ~0.62-0.97
          ├─ weaver.prompt_latent_ln(prompt_query_latents) * prompt_latent_scale
          │                                                  [8,1536]，行 norm 全部 ~39.19
          └─ cat([context, query]) .................. [1,104,1536]   (context 在前)
              └─ weaver.model = PeftModel(Qwen2ForCausalLM)  x28 层
                   · 每层 q_proj / v_proj = base_layer + lora_A.weaver -> lora_B.weaver * 2.0
                   · embed_tokens 被跳过（走 inputs_embeds）
                   · lm_head 仍执行，产出 [1,104,151936]，无人消费
                    └─ hidden_states[-1][:, -8:, ] ... [1,8,1536]，行 norm 79.9-123.6
                        └─ weaver_to_reasoner ....... [1,8,1536]，行 norm 52.5-76.9  (token 的 ~60 倍)
                            └─ cat 回 current_inputs_embeds -> [1,104,1536]
                                └─ reasoner(...) 整段重算，DynamicCache 长度 104（含 latent 的 K/V）
```

之后 12 个单 token 步：`reasoner(input_embeds[:, -1:], past_key_values=cache)`，
每步只走 Reasoner 的 28 层 + lm_head，embed_tokens 只对那 1 个新 token 查表。

## 4. forward 计票（实测）

| 前缀 | 被 forward 的模块数 | 累计 forward 次数 |
|---|---|---|
| `reasoner.*` | 369 | 40,960 |
| `weaver.model.*` | 594 | 1,782 |
| `reasoner_to_weaver` | 1 | 3 |
| `weaver_to_reasoner` | 1 | 3 |
| `trigger.*` | **1**（只有 `MemGenTrigger` 自身） | 3 |
| `trigger.model.*` | **0** | **0** |
| `trigger.output_layer` | **0** | **0** |

`trigger.*` 那 1 个是 `MemGenTrigger` 本身的 `forward` —— 它被调用 3 次，
但因为 `active=False`，函数体只构造常量张量，没进入任何子模块。

## 5. 训练状态指纹判据

判据本身可复跑：`fingerprint_checkpoint.py`，输出 `logs/checkpoint_fingerprints.json`。
原理：`nn.Linear` 用 `kaiming_uniform_(a=√5)`，权重绝对值上界恰为 `1/√fan_in`；
PEFT 把 `lora_B` 零初始化。于是"是否越过该边界""`lora_B` 是否仍全零"成为可检验事实。

| 张量 | 实测 | 初始参考值 | 判定 |
|---|---|---|---|
| `reasoner_to_weaver.weight` | absmax 0.026611 | 边界 0.025516 | **moved_from_init** |
| `weaver_to_reasoner.weight` | absmax 0.028076 | 边界 0.025516 | **moved_from_init** |
| `trigger.output_layer.weight` | absmax 0.025513 | 边界 0.025516 | **at_init** |
| `weaver` LoRA（56 module） | `‖B@A‖_F` ∈ [0.0130, 0.0744]，均值 0.040 | 0 | **trained** |
| `trigger` LoRA（56 module） | `‖B@A‖_F` 全为 0（`lora_B` 全零） | 0 | **never_trained** |
| `prompt_latent_ln.weight` | 精确全 1 | 全 1 | at_init |
| `prompt_latent_ln.bias` | absmax 0.001999 | 精确 0 | moved |
| `inference_latent_ln.weight` | 最大偏离 1 达 0.003906 | 全 1 | moved |
| `inference_latent_ln.bias` | absmax 0.001419 | 精确 0 | moved |
| `prompt/inference_latent_scale` | 精确 1.0 | 1.0 | at_init |
| `prompt/inference_query_latents` | std 0.9981 / 1.0012，行 norm 39.11 / 39.23 | randn 期望 39.19 | 统计上无法与初始区分 |

**综合**：这个 weaver-sft checkpoint 里，能被证明训练过的只有
Weaver LoRA、两个 projection、以及两个 latent LayerNorm 的部分元素。
Trigger 侧没有任何训练痕迹，而它本来也在 `active=False` 下不参与计算 ——
所以本次 inference 的全部有效学习参数就是 **Weaver LoRA 2,179,072 + 两个投影 4,721,664
+ 两组 latent bank 30,722 ≈ 6.9 M**，占 4.64 B 总参数的 **0.15 %**。


