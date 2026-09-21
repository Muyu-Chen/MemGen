"""Consolidate the full dynamic sentence-level oracle campaign."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from common import HARD_BENCH_ROOT, portable_path, read_jsonl, sha256_file, utc_now, write_json
from score_results import numeric_value


RUN_ROOT = HARD_BENCH_ROOT / "runs"
OUTPUT = HARD_BENCH_ROOT / "analysis" / "sentence_oracle_campaign_v1.json"
REPORT = HARD_BENCH_ROOT / "SENTENCE_ORACLE_CAMPAIGN_REPORT.md"
ORACLE_SOURCES = [
    RUN_ROOT
    / "sentence_sequential_oracle_v1"
    / "scored"
    / "Z_memgen_sentence_sequential_oracle.scored.jsonl",
    RUN_ROOT
    / "text_sentence_oracle_0754_v1"
    / "scored"
    / "Z2_memgen_text_sentence_oracle.scored.jsonl",
    RUN_ROOT
    / "text_sentence_oracle_depth3_screen_v1"
    / "scored"
    / "Z3_memgen_text_sentence_oracle_screen.scored.jsonl",
    RUN_ROOT
    / "text_sentence_oracle_depth5_followup_v1"
    / "scored"
    / "Z4_memgen_text_sentence_oracle_depth5.scored.jsonl",
    RUN_ROOT
    / "text_sentence_oracle_0810_depth3_v1"
    / "scored"
    / "Z5_memgen_text_sentence_oracle_0810.scored.jsonl",
    RUN_ROOT
    / "text_sentence_oracle_1088_depth5_v1"
    / "scored"
    / "Z6_memgen_text_sentence_oracle_1088.scored.jsonl",
]
N_BASELINE = (
    RUN_ROOT
    / "trigger_intervention_v1"
    / "scored"
    / "N_memgen_no_inference.scored.jsonl"
)
N_0810_LONG = (
    RUN_ROOT
    / "no_inference_long_0810_v1"
    / "scored"
    / "N_memgen_no_inference.scored.jsonl"
)
EXPECTED_SAMPLES = {
    3: {
        "gsm8k_test_0063",
        "gsm8k_test_0423",
        "gsm8k_test_0611",
        "gsm8k_test_0754",
        "gsm8k_test_0810",
        "gsm8k_test_0976",
        "gsm8k_test_1088",
        "gsm8k_test_1161",
    },
    5: {
        "gsm8k_test_0423",
        "gsm8k_test_0611",
        "gsm8k_test_0754",
        "gsm8k_test_0976",
        "gsm8k_test_1088",
    },
}


def as_decimal(value: Decimal | Fraction) -> Decimal:
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return value


def absolute_error(candidate: str | None, gold: str) -> Decimal | None:
    if candidate is None:
        return None
    candidate_value = numeric_value(candidate)
    gold_value = numeric_value(gold)
    if candidate_value is None or gold_value is None:
        return None
    return abs(as_decimal(candidate_value) - as_decimal(gold_value))


def token_hash(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record["completion_token_ids"], separators=(",", ":")).encode()
    ).hexdigest()


def analyze_group(sample_id: str, depth: int, group: list[dict], n_record: dict) -> dict:
    group = sorted(group, key=lambda record: record["policy_string"])
    expected_policies = [f"{value:0{depth}b}" for value in range(2**depth)]
    observed_policies = [record["policy_string"] for record in group]
    if observed_policies != expected_policies:
        raise ValueError(f"incomplete policy set for {sample_id} depth {depth}")
    no_invoke_hash = token_hash(group[0])
    gold = group[0]["scores"]["semantic_gold"]
    variants = []
    for record in group:
        prediction = record["scores"]["semantic_candidate"]
        error = absolute_error(prediction, gold)
        variants.append(
            {
                "policy": record["policy_string"],
                "prediction": prediction,
                "semantic_correct": bool(record["scores"]["semantic_correct"]),
                "absolute_error": format(error, "f") if error is not None else None,
                "completion_changed_vs_no_invoke": token_hash(record) != no_invoke_hash,
                "horizon_reached": record["intervention_reached"],
                "inference_augmentations": record["inference_augmentation_count"],
                "observed_policy_bits": record["observed_policy_bits"],
            }
        )
    numeric = [variant for variant in variants if variant["absolute_error"] is not None]
    best_error = min(Decimal(variant["absolute_error"]) for variant in numeric)
    best = [variant for variant in numeric if Decimal(variant["absolute_error"]) == best_error]
    return {
        "sample_id": sample_id,
        "depth": depth,
        "gold": gold,
        "prompt_only_prediction": n_record["scores"]["semantic_candidate"],
        "prompt_only_truncated": n_record["truncated_at_max_new_tokens"],
        "no_invoke_prediction": variants[0]["prediction"],
        "policy_count": len(variants),
        "full_horizon_count": sum(variant["horizon_reached"] for variant in variants),
        "correct_policy_count": sum(variant["semantic_correct"] for variant in variants),
        "correct_policies": [variant["policy"] for variant in variants if variant["semantic_correct"]],
        "unique_completion_count": len({token_hash(record) for record in group}),
        "unique_prediction_count": len({variant["prediction"] for variant in variants}),
        "prediction_distribution": dict(Counter(variant["prediction"] for variant in variants)),
        "changed_completion_count": sum(variant["completion_changed_vs_no_invoke"] for variant in variants),
        "best_absolute_error": format(best_error, "f"),
        "best_predictions": sorted({variant["prediction"] for variant in best}),
        "best_error_policies": [variant["policy"] for variant in best],
        "variants": variants,
    }


def sample_table(samples: list[dict]) -> list[str]:
    lines = [
        "| sample | gold | prompt-only | unique answers | changed | horizon | exact | best error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in samples:
        lines.append(
            f"| `{sample['sample_id']}` | {sample['gold']} | {sample['prompt_only_prediction']} | "
            f"{sample['unique_prediction_count']} | {sample['changed_completion_count']}/{sample['policy_count']} | "
            f"{sample['full_horizon_count']}/{sample['policy_count']} | "
            f"{sample['correct_policy_count']}/{sample['policy_count']} | {sample['best_absolute_error']} |"
        )
    return lines


def render_report(analysis: dict) -> str:
    d3 = analysis["depths"]["3"]
    d5 = analysis["depths"]["5"]
    exact = analysis["correct_policy_count"]
    if exact:
        headline = f"动态句级 oracle 共找到 {exact} 条 exact-positive 策略。"
    else:
        headline = "动态句级 oracle 在全部 224 条策略中没有找到 exact rescue。"
    lines = [
        "# Sentence-level Sequential Oracle Campaign",
        "",
        f"**结论：{headline}**",
        "",
        "## 问题与边界",
        "",
        "本实验回答的不是“某个固定 token 点是否有用”，而是：在生成轨迹会被每次 Weaver 调用实时改写的情况下，拥有答案信息的 oracle controller 能否在论文定义的 sentence-granularity delimiter 节点上选出一条成功路径。Prompt augmentation 始终开启。",
        "",
        "论文把候选定义为 delimiter-token 集合（举例为 comma、period）并称其为 sentence-granularity。发布代码还包含 newline。本实现保留 comma / period / newline，而排除小数点、数字内部逗号、未闭合 `<<...>>` calculator span、空行和未完成算术运算；它并不把候选武断收窄成“只有完整非空行末”。decoded-token 检测同时修复了发布实现漏掉 `'.\\n'` 等 fused token 的问题。",
        "",
        "## 总览",
        "",
        "| 深度 | 样本 | 策略 | 完整到达名义深度 | exact-correct | rescued samples |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| 3 | {d3['sample_count']} | {d3['policy_count']} | {d3['full_horizon_count']} | {d3['correct_policy_count']} | {d3['rescued_sample_count']} |",
        f"| 5 | {d5['sample_count']} | {d5['policy_count']} | {d5['full_horizon_count']} | {d5['correct_policy_count']} | {d5['rescued_sample_count']} |",
        f"| 合计 | — | {analysis['policy_count']} | {analysis['full_horizon_count']} | {analysis['correct_policy_count']} | — |",
        "",
        "### 深度 3：全部 8 个完整 prompt-only 失败样本",
        "",
        *sample_table(d3["samples"]),
        "",
        "### 深度 5：优先扩展最有信息量的 5 个样本",
        "",
        *sample_table(d5["samples"]),
        "",
        "## 关键机制证据",
        "",
        "- **不是候选器根本没给句级机会。** `0754` 的原 token-id gate 只暴露 4 个小数内部伪候选；decoded detector 找到 6 个真实句末后，三步与五步搜索仍无 exact rescue。",
        "- **失败题并不同质。** 先前 `0611/0754` 的 40 个单点候选为 0/40，动态三步与五步也都无正例；但 `0810` 的首个句级 invoke 把 810-token 错误长链缩成 205 tokens，并精确恢复 `310`。",
        "- **Weaver 确实能改变轨迹。** 多数样本产生多个 completion/答案，部分分支甚至提前结束，说明搜索测到了真实的历史依赖交互，而不是静态 subset 重放。",
        "- **改变不等于修复。** `0976` 可从 750 推近到 570（gold 540），但严格 exact 仍失败；`0423` 的 6.67 与 9.33 对 gold 8 等距。",
        "- **前史会改变相同后续动作的作用。** `1088` 的 `011xx` 进入 393–512-token harmful branch（答案 1.56 或截断），而 `111xx` 因最早节点已调用而保持约 133-token 短轨迹；首个调用没有救对答案，却阻止了后续组合退化。",
        "- **prompt memory 是独立风险。** `0063` 在 Base/风格控制下为 1596 正确，prompt-on / inference-off 已变成 966，且所有三步策略仍是 966；Trigger 在生成中途无法撤销已经注入的错误先验。",
        "- **`0810` 的旧结果是截断，不是成功。** 1024-token 复跑在 810 tokens 正常结束，预测 10,737,418,240（gold 310）；本报告用该完整轨迹替换 512-token 截断记录。",
        "",
        "## 责任切分",
        "",
    ]
    if exact:
        rescued = [
            f"{sample['sample_id']}@d{sample['depth']}"
            for depth in ("3", "5")
            for sample in analysis["depths"][depth]["samples"]
            if sample["correct_policy_count"]
        ]
        lines.extend(
            [
                f"存在 oracle-positive 路径：`{', '.join(rescued)}`。因此至少对这些样本，Trigger 的顺序控制确实是可训练瓶颈；但其余样本在已测深度内仍无正例，不能把所有失败统一归因给 Trigger。应把 exact-positive 样本与无正例样本分开处理，后者优先回流给 Weaver / prompt memory。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "在已测试样本和前五个动态决策节点内，结果不支持“只要把 Trigger 的句级调用时机学好，当前 Weaver 就能恢复正确答案”。oracle 已拥有未来答案信息仍找不到正例，因此把这些失败全部标成 Trigger 训练数据会产生错误归因。",
                "",
                "下一阶段的主变量应是 Weaver 产出的 latent memory 与 prompt augmentation，而不是继续在同一 checkpoint 上扩大单点 gate sweep。Trigger 训练应只使用先证明存在 exact-positive 分支的样本。",
                "",
            ]
        )
    lines.extend(
        [
            "## 限制与完整性",
            "",
            "这不是对任意长度策略的全称证明：三步覆盖全部 8 个完整失败样本，五步覆盖 5 个优先样本；五步诊断还临时把 checkpoint 的最大 inference augmentation 从 3 提高到 5。结论严格限定于当前 checkpoint、prompt、greedy decoding 和候选规则。",
            "",
            f"- oracle records：{analysis['policy_count']}；截断：{analysis['truncated_count']}；policy prefix 校验：{analysis['all_observed_prefixes_match_policies']}",
            f"- 误接纳 numeric/internal delimiter：{analysis['invalid_accepted_boundary_count']}",
            f"- oracle 总生成耗时：{analysis['total_latency_seconds']:.1f} 秒",
            f"- prompt-only 十题：{analysis['prompt_only']['correct']}/10 correct；完整记录 {analysis['prompt_only']['nontruncated']}/10",
            "",
            "机器可读 JSON 保留每条策略、prediction distribution、边界可达性和全部 source hashes。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    sources = []
    records = []
    for path in ORACLE_SOURCES:
        source = read_jsonl(path)
        sources.append({"path": portable_path(path), "sha256": sha256_file(path), "records": len(source)})
        records.extend(source)

    keys = [(record["sample_id"], record["decision_depth"], record["policy_string"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate sample/depth/policy records")

    n_records = {record["sample_id"]: record for record in read_jsonl(N_BASELINE)}
    long_0810 = read_jsonl(N_0810_LONG)
    if len(long_0810) != 1 or long_0810[0]["sample_id"] != "gsm8k_test_0810":
        raise ValueError("unexpected long 0810 baseline")
    n_records["gsm8k_test_0810"] = long_0810[0]

    depths = {}
    for depth, expected_samples in EXPECTED_SAMPLES.items():
        observed_samples = {
            record["sample_id"] for record in records if record["decision_depth"] == depth
        }
        if observed_samples != expected_samples:
            raise ValueError(
                f"depth {depth} sample mismatch: expected {sorted(expected_samples)}, found {sorted(observed_samples)}"
            )
        samples = [
            analyze_group(
                sample_id,
                depth,
                [
                    record
                    for record in records
                    if record["decision_depth"] == depth and record["sample_id"] == sample_id
                ],
                n_records[sample_id],
            )
            for sample_id in sorted(expected_samples)
        ]
        depth_records = [record for record in records if record["decision_depth"] == depth]
        depths[str(depth)] = {
            "sample_count": len(samples),
            "policy_count": len(depth_records),
            "full_horizon_count": sum(record["intervention_reached"] for record in depth_records),
            "correct_policy_count": sum(record["scores"]["semantic_correct"] for record in depth_records),
            "rescued_sample_count": sum(sample["correct_policy_count"] > 0 for sample in samples),
            "samples": samples,
        }

    invalid_reasons = {
        "numeric_comma_or_period",
        "inside_calculator_span",
        "unfinished_arithmetic_before_punctuation",
        "unfinished_arithmetic_before_newline",
    }
    analysis = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "scope": "dynamic sentence-level sequential oracle on all complete prompt-only failures, plus targeted depth-5 extensions",
        "oracle_sources": sources,
        "prompt_only_sources": [
            {"path": portable_path(N_BASELINE), "sha256": sha256_file(N_BASELINE)},
            {"path": portable_path(N_0810_LONG), "sha256": sha256_file(N_0810_LONG)},
        ],
        "policy_count": len(records),
        "full_horizon_count": sum(record["intervention_reached"] for record in records),
        "correct_policy_count": sum(record["scores"]["semantic_correct"] for record in records),
        "truncated_count": sum(record["truncated_at_max_new_tokens"] for record in records),
        "all_observed_prefixes_match_policies": all(
            record["observed_policy_bits"]
            == record["policy_bits"][: len(record["observed_policy_bits"])]
            for record in records
        ),
        "invalid_accepted_boundary_count": sum(
            event.get("policy_bit_index") is not None
            and event.get("boundary_reason") in invalid_reasons
            for record in records
            for event in record["gate_events"]
        ),
        "total_latency_seconds": sum(record["latency_seconds"] for record in records),
        "prompt_only": {
            "sample_count": len(n_records),
            "correct": sum(record["scores"]["semantic_correct"] for record in n_records.values()),
            "nontruncated": sum(not record["truncated_at_max_new_tokens"] for record in n_records.values()),
        },
        "depths": depths,
    }
    if analysis["policy_count"] != 224:
        raise ValueError(f"expected 224 policies, found {analysis['policy_count']}")
    write_json(OUTPUT, analysis)
    REPORT.write_text(render_report(analysis), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                key: analysis[key]
                for key in (
                    "policy_count",
                    "full_horizon_count",
                    "correct_policy_count",
                    "truncated_count",
                    "invalid_accepted_boundary_count",
                )
            },
            indent=2,
        )
    )
    print(f"analysis={OUTPUT}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
