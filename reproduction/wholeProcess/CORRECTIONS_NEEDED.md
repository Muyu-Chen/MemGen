# CORRECTIONS_NEEDED —— 本次 trace 与既有文档的冲突清单

> ## ✅ 落实状态（2026-09-20 更新）
>
> 本清单的 A 组 6 条与 B 组 10 条**已全部应用**到旧文档，每条下方标注了改动位置。
> 原始清单文字保留不删改，便于追溯；只有 A1 的一处数字描述被发现不够准确，
> 已在原文用「**勘误**」标出并给出实测值。
>
> | 项 | 状态 | 落实到 |
> |---|---|---|
> | A1 | ✅ 已改 | `MEMGEN_ARCHITECTURE_EXPLAINED.md`：流程图第 4 步 + Step 3 + 新增章节「Trigger 门控的真实判据」+ 配置参数节的 `delimiters` 注释 |
> | A2 | ✅ 已改 | `REPRODUCTION.md`：发现 #6 观察项划掉并指向新增的**发现 #8** |
> | A3 | ✅ 已改 | `MEMGEN_ARCHITECTURE_EXPLAINED.md`：配置参数节的「修正」块 |
> | A4 | ✅ 已改 | `MEMGEN_ARCHITECTURE_EXPLAINED.md`：新增章节「达到 max_inference_aug_num 后循环整体跳出」 |
> | A5 | ✅ 已改 | `REPRODUCTION.md`：发现 #1 顶部加配置标注 + `active=False` 实测对照块；发现 #6 标题原已注明 `active=True` |
> | A6 | ✅ 已改 | `MEMGEN_ARCHITECTURE_EXPLAINED.md`：新增小节「yaml 与 checkpoint config 互相矛盾」 |
> | B1–B10 | ✅ 已补 | `MEMGEN_ARCHITECTURE_EXPLAINED.md`：新增章节「执行级 trace 补充的关键事实」（11 条，含 B 组全部 + latent 非词表近似） |
> | C | — | 无需改动，结论仍成立 |
>
> 另外顺带修正了两处与本清单无关但已过期的表述：
> - `REPRODUCTION.md` 头部 `Working tree: clean (no modifications)` →
>   改为注明现已含 `modeling_memgen.py` 与 `memgen/utils.py` 两处本地修改
>   （`git diff --stat 970cc95 HEAD -- memgen/` 实测确认仅这两个文件）。
> - `MEMGEN_ARCHITECTURE_EXPLAINED.md` 头部只提 tensor trace →
>   补充第二轮执行级 trace 作为验证来源。
>
> **本清单中一处行号勘误**：bf16 转换在 `runner.py:224`（不是 225）；
> 旧文档已按 224 写入。

按 README 第 10 节要求：**不直接修改旧文档**，先把需要改的地方列在这里。
每条都给出：旧表述（文件:行号）→ 实测事实 → 证据位置。

证据统一来自本目录 `logs/`（commit `7e067fa`，配置取 checkpoint 自带 `config.json`）。

---

## A. 需要改正的描述

### A1. "遇 delimiter 再问 Trigger" 在代码层面不成立（最重要）　✅ 已落实

> `MEMGEN_ARCHITECTURE_EXPLAINED.md:32`
> `-> 4. Reasoner Generation（逐 token，遇 delimiter 再问 Trigger，最多 3 次）`
> `:169` `Trigger 在 delimiter 位置被调用`
> `:373-374` `# delimiter（触发 Trigger 检查的符号）` / `delimiters = [",", ".", "\n"]`

实测：判据是**最后一个 token 的 id 是否 ∈ {11, 13, 198}**
（`modeling_utils.py:145-175` + `_get_delimiter_token_ids`），是 **token-id 精确匹配，不是文本匹配**。
`delimiters` 这个"符号列表"只是用来**一次性**编码出三个 id 的原料。

本题 111 个生成步里，只有 **2 步**的 token id 落在该集合（都是 `,`=11）；
id 13（句号）与 id 198（换行）各出现 **0 次**。
而生成的文本里有 3 处含句号/换行的位置因 BPE 合并而漏检。

> **勘误（本清单原文写「3 处『句号+换行』被合并成 id 624」，不够准确）**
> `logs/delimiter_check.txt` 实测是 **2 处 id 624（`.\n`）+ 1 处 id 7110（`.\`，在 `\boxed` 之前）**：
> ```
> step  38  id= 624  tok='.Ċ'
> step  66  id= 624  tok='.Ċ'
> step 105  id=7110  tok='.\'
> ```
> 结论方向不变，而且**更强**：漏检不限于「句号+换行」，
> 任何与句号粘连成单 token 的后续字符（`\n`、`\`、`T`…）都会绕过门控。
> 已按准确版本写入 `MEMGEN_ARCHITECTURE_EXPLAINED.md` 与 `REPRODUCTION.md` 发现 #8。

建议改成：*每个 token 之后检查；只有当该 token 的 id 恰好是 `,`/`.`/`\n` 单独成 token 的三个 id
(11/13/198) 之一时，才成为 augmentation candidate。句子边界常因 BPE 合并而漏检。*

### A2. "增强通常发生在句子边界"　✅ 已落实（`REPRODUCTION.md` 发现 #6 → 新增发现 #8）

> `REPRODUCTION.md:317`

实测相反：本次 2 次 inference augmentation 都发生在**逗号**后，3 处句子边界一次都没触发（见 A1）。
建议改为"发生在独立成 token 的 delimiter 之后，实测以逗号为主"。

### A3. "prompt 最多增强 1 次" 的原因写错了　✅ 已落实

> `MEMGEN_ARCHITECTURE_EXPLAINED.md:362`
> `max_prompt_aug_num = 1   # prompt 最多增强 1 次`

`generate()`（`modeling_memgen.py:498-675`）**从未读取** `config.max_prompt_aug_num`。
全文它只出现在 `_conversational_forward`（`modeling_memgen.py:328-332`，训练期多轮选择）
和 `from_config` 的传参里。prompt 增强只发生一次，是因为它绑定在 `i == 0`
这一个循环位置上（`modeling_memgen.py:568`），与这个配置值无关。
把它写成"推理时的上限"会误导后续设计。

### A4. "最多 3 次" 漏掉了循环会整体跳出　✅ 已落实（新增独立章节）

> `MEMGEN_ARCHITECTURE_EXPLAINED.md:32`

`modeling_memgen.py:610-624`：一旦 `sentence_augment_count >= max_inference_aug_num`，
剩余 token 改由 `reasoner.generate(inputs_embeds=..., use_cache=False)` **一次性生成**并 `break`。
也就是说第 3 次增强之后：不再有逐步 gate 检查、不再有 delimiter 判断、**KV cache 关闭**。
本次 count 只到 2，所以该分支未被触发（`reasoner_forward_bulk_path` 事件数 0），
但这条路径真实存在且会显著改变行为。

### A5. Trigger 的 softmax 数值混用了两种配置　✅ 已落实

> `REPRODUCTION.md:127` `Softmax P(augment=1): 0.968 ~ 0.999, 中位数 0.989`
> `REPRODUCTION.md:307-309` 的逐题 softmax 表

这些数字来自 **`trigger_active=True`** 的 instrumented 运行（用随机权重的前向）。
官方 eval 配置是 `active=False`，此时 `trigger.py:37-40` 返回常量
`logits=[0.0, 1.0]`，softmax 恒为 `[0.268941, 0.731059]`，且 `trigger.model` 零次 forward。
两套数字必须按配置分开标注，否则读者会以为"未训练的 Trigger 自信地输出 0.99"，
而实际上 `active=False` 时连模型都没进。

### A6. LoRA 配置与 augmentation 上限的取值来源要写清　✅ 已落实（新增冲突表小节）

`configs/latent_memory/gsm8k.yaml` 与 released checkpoint 的 `config.json` **互相矛盾**：

| 项 | gsm8k.yaml | checkpoint config.json / adapter_config.json |
|---|---|---|
| `max_inference_aug_num` | 5 | **3** |
| LoRA `target_modules` | q,k,v,o,gate,up,down（7 个） | **q_proj, v_proj（2 个）** |
| `interaction.batch_size` | 8 | eval 脚本 `qwen2_5_gsm8k_sft.sh:28` 写 4 |

推理时应以 **checkpoint 自带配置**为准（`MemGenConfig.from_pretrained(ckpt)`）。
旧文档没有说明这个冲突；若按 yaml 构建模型，会注入 5 组随机初始化（`lora_B=0`）的多余 adapter，
数值上无害但 module map 与参数量都对不上。

---

## B. 旧文档缺失、但对设计自有模型很关键的点　✅ 全部 10 条已补入架构文档

1. **Trigger 看不到 latent。** `_should_augment` 只用 `current_input_ids` 并自行重算 mask
   （`modeling_utils.py:264,270`），而 latent 没有 token id。实测 Trigger 输入长 96/108/173，
   同期 Reasoner 的 embeds 长 96→104/116→124/189→197，差值正好是已插入 latent 数。
   即门控是在"纯 token 视图"上做的，与 Reasoner 的实际输入不同构。
2. **Weaver 的 `embed_tokens`（233,373,696 参数）不参与计算**，因为它通过 `inputs_embeds=` 被调用。
3. **Weaver 的 `lm_head` 照样执行并产出 `[1,104,151936]` logits，但无人消费**（纯浪费算力）。
4. **latent 的数值量级约为 token embedding 的 60 倍**（行 norm 52~77 对 0.43~1.21），
   因此它在后续表示中占主导；`reasoner_to_weaver` 输入的整体 norm 从 9.49 跳到 181.06 即由此而来。
5. **8 个 latent slot 之间平均余弦 0.72~0.77**，高度相似而非正交互补。
6. **latent 会进入 KV cache**（cache 长度 104 含 latent 位），且**占据真实 position id**，
   使后续 token 的位置编号整体后移（首个生成 token 的 position = 104 而非 96）。
7. **Weaver 返回的 position_ids 被 `generate()` 丢弃**（`modeling_memgen.py:569,573` 用 `_` 接住），
   实际位置由 `_generate_position_ids(attention_mask)` 重算。
8. **prompt augmentation 不计入 `max_inference_aug_num`**（`modeling_memgen.py:558` 的 `if i != 0`）。
9. **推理 dtype 官方确实是 bfloat16**：`runner.py:225` 显式 `self.model.to(torch.bfloat16)`，
   且 `projs.bin` / `weaver.bin` / `trigger.bin` 内部张量本身就是 bfloat16。
10. **官方 GSM8K 输出仍带 `<<...>>` 计算器记号**，是 SFT 数据格式泄漏，与 MemGen 机制无关。

## C. 仍然成立、无需改动的旧结论

* Weaver 数据流三要素（输入 = input embeddings、拼接 = `[context, Q]`、取**末** K 个）
  —— 本次以真实执行再次确认（`tensor_shapes.json` 的 `concat_layout` / `extraction` 字段）。
* 历史 latent **append**、不移除不覆盖 —— 本次给出逐比特证据（`persistence_checks`，max_abs_diff 0.0）。
* LoRA adapter name 不匹配需手动 remap —— 本次正是走 `modeling_memgen.py:778-835` 的修复版加载器。
* "weaver-sft checkpoint 的 Trigger 未经训练" —— 本次给出更强的证据：
  `lora_B` 56/56 全零 + `output_layer.absmax` 停在初始化边界（`logs/checkpoint_fingerprints.json`）。

