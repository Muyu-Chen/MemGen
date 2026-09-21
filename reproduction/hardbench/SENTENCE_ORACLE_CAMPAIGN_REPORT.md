# Sentence-level Sequential Oracle Campaign

**结论：动态句级 oracle 共找到 4 条 exact-positive 策略。**

## 问题与边界

本实验回答的不是“某个固定 token 点是否有用”，而是：在生成轨迹会被每次 Weaver 调用实时改写的情况下，拥有答案信息的 oracle controller 能否在论文定义的 sentence-granularity delimiter 节点上选出一条成功路径。Prompt augmentation 始终开启。

论文把候选定义为 delimiter-token 集合（举例为 comma、period）并称其为 sentence-granularity。发布代码还包含 newline。本实现保留 comma / period / newline，而排除小数点、数字内部逗号、未闭合 `<<...>>` calculator span、空行和未完成算术运算；它并不把候选武断收窄成“只有完整非空行末”。decoded-token 检测同时修复了发布实现漏掉 `'.\n'` 等 fused token 的问题。

## 总览

| 深度 | 样本 | 策略 | 完整到达名义深度 | exact-correct | rescued samples |
|---:|---:|---:|---:|---:|---:|
| 3 | 8 | 64 | 64 | 4 | 1 |
| 5 | 5 | 160 | 88 | 0 | 0 |
| 合计 | — | 224 | 152 | 4 | — |

### 深度 3：全部 8 个完整 prompt-only 失败样本

| sample | gold | prompt-only | unique answers | changed | horizon | exact | best error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gsm8k_test_0063` | 1596 | 966 | 1 | 7/8 | 8/8 | 0/8 | 630 |
| `gsm8k_test_0423` | 8 | 9.33 | 3 | 7/8 | 8/8 | 0/8 | 1.33 |
| `gsm8k_test_0611` | 1450000 | 6003125 | 3 | 7/8 | 8/8 | 0/8 | 2425000 |
| `gsm8k_test_0754` | 89 | 67 | 5 | 5/8 | 8/8 | 0/8 | 12 |
| `gsm8k_test_0810` | 310 | 10737418240 | 2 | 4/8 | 8/8 | 4/8 | 0 |
| `gsm8k_test_0976` | 540 | 750 | 3 | 6/8 | 8/8 | 0/8 | 105 |
| `gsm8k_test_1088` | 30 | 26.67 | 2 | 7/8 | 8/8 | 0/8 | 3.33 |
| `gsm8k_test_1161` | 170 | 140 | 1 | 0/8 | 8/8 | 0/8 | 30 |

### 深度 5：优先扩展最有信息量的 5 个样本

| sample | gold | prompt-only | unique answers | changed | horizon | exact | best error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gsm8k_test_0423` | 8 | 9.33 | 5 | 31/32 | 8/32 | 0/32 | 1.33 |
| `gsm8k_test_0611` | 1450000 | 6003125 | 7 | 30/32 | 12/32 | 0/32 | 2425000 |
| `gsm8k_test_0754` | 89 | 67 | 8 | 26/32 | 32/32 | 0/32 | 12 |
| `gsm8k_test_0976` | 540 | 750 | 7 | 29/32 | 32/32 | 0/32 | 30 |
| `gsm8k_test_1088` | 30 | 26.67 | 3 | 28/32 | 4/32 | 0/32 | 3.33 |

## 关键机制证据

- **不是候选器根本没给句级机会。** `0754` 的原 token-id gate 只暴露 4 个小数内部伪候选；decoded detector 找到 6 个真实句末后，三步与五步搜索仍无 exact rescue。
- **失败题并不同质。** 先前 `0611/0754` 的 40 个单点候选为 0/40，动态三步与五步也都无正例；但 `0810` 的首个句级 invoke 把 810-token 错误长链缩成 205 tokens，并精确恢复 `310`。
- **Weaver 确实能改变轨迹。** 多数样本产生多个 completion/答案，部分分支甚至提前结束，说明搜索测到了真实的历史依赖交互，而不是静态 subset 重放。
- **改变不等于修复。** `0976` 可从 750 推近到 570（gold 540），但严格 exact 仍失败；`0423` 的 6.67 与 9.33 对 gold 8 等距。
- **前史会改变相同后续动作的作用。** `1088` 的 `011xx` 进入 393–512-token harmful branch（答案 1.56 或截断），而 `111xx` 因最早节点已调用而保持约 133-token 短轨迹；首个调用没有救对答案，却阻止了后续组合退化。
- **prompt memory 是独立风险。** `0063` 在 Base/风格控制下为 1596 正确，prompt-on / inference-off 已变成 966，且所有三步策略仍是 966；Trigger 在生成中途无法撤销已经注入的错误先验。
- **`0810` 的旧结果是截断，不是成功。** 1024-token 复跑在 810 tokens 正常结束，预测 10,737,418,240（gold 310）；本报告用该完整轨迹替换 512-token 截断记录。

## 责任切分

存在 oracle-positive 路径：`gsm8k_test_0810@d3`。因此至少对这些样本，Trigger 的顺序控制确实是可训练瓶颈；但其余样本在已测深度内仍无正例，不能把所有失败统一归因给 Trigger。应把 exact-positive 样本与无正例样本分开处理，后者优先回流给 Weaver / prompt memory。

## 限制与完整性

这不是对任意长度策略的全称证明：三步覆盖全部 8 个完整失败样本，五步覆盖 5 个优先样本；五步诊断还临时把 checkpoint 的最大 inference augmentation 从 3 提高到 5。结论严格限定于当前 checkpoint、prompt、greedy decoding 和候选规则。

- oracle records：224；截断：1；policy prefix 校验：True
- 误接纳 numeric/internal delimiter：0
- oracle 总生成耗时：20541.1 秒
- prompt-only 十题：2/10 correct；完整记录 10/10

机器可读 JSON 保留每条策略、prediction distribution、边界可达性和全部 source hashes。
