"""Consolidate the depth-3 sentence-oracle screen across prompt-only failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from common import HARD_BENCH_ROOT, portable_path, read_jsonl, sha256_file, utc_now, write_json
from score_results import numeric_value


TARGET_IDS = [
    "gsm8k_test_0423",
    "gsm8k_test_0063",
    "gsm8k_test_1088",
    "gsm8k_test_0611",
    "gsm8k_test_0976",
    "gsm8k_test_0754",
    "gsm8k_test_1161",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=HARD_BENCH_ROOT / "analysis" / "sentence_oracle_screen_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=HARD_BENCH_ROOT / "SENTENCE_ORACLE_SCREEN_REPORT.md",
    )
    return parser.parse_args()


def as_decimal(value: Decimal | Fraction) -> Decimal:
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return value


def error(candidate: str | None, gold: str) -> Decimal | None:
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


def source_paths() -> tuple[list[Path], Path]:
    run_root = HARD_BENCH_ROOT / "runs"
    oracle_paths = [
        run_root
        / "text_sentence_oracle_depth3_screen_v1"
        / "scored"
        / "Z3_memgen_text_sentence_oracle_screen.scored.jsonl",
        run_root
        / "sentence_sequential_oracle_v1"
        / "scored"
        / "Z_memgen_sentence_sequential_oracle.scored.jsonl",
        run_root
        / "text_sentence_oracle_0754_v1"
        / "scored"
        / "Z2_memgen_text_sentence_oracle.scored.jsonl",
    ]
    baseline_path = (
        run_root
        / "trigger_intervention_v1"
        / "scored"
        / "N_memgen_no_inference.scored.jsonl"
    )
    return oracle_paths, baseline_path


def render_report(analysis: dict) -> str:
    lines = [
        "# Depth-3 Sentence Oracle Screen",
        "",
        "**结论：7 个非截断 prompt-only 失败样本上，三步 sentence-level oracle 的 exact rescue rate 为 0/7；56 条策略均未得到正确答案。**",
        "",
        "## 设计",
        "",
        "对冻结十题 pilot 中的 prompt-on / inference-off (`N`) 结果，排除两道已正确题和一条 512-token 截断题 `0810`，在其余 7 个错误样本上穷举前三个动态 sentence-level 决策的 `2^3` invoke/skip 策略。Prompt augmentation 始终开启；每次 invoke 后，后续边界沿新轨迹重新识别。",
        "",
        "| sample | gold | N prediction | unique answers | changed completions | best error | exact rescue |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in analysis["samples"]:
        lines.append(
            f"| `{sample['sample_id']}` | {sample['gold']} | {sample['baseline_prediction']} | "
            f"{sample['unique_prediction_count']} | {sample['changed_completion_count']}/8 | "
            f"{sample['best_absolute_error']} | {sample['correct_policy_count']}/8 |"
        )
    lines.extend(
        [
            "",
            "## 关键观察",
            "",
            "- `0063` 的 8 条策略全部预测 `966`，而 A/S 原先正确预测 `1596`；前三个 inference 调用决策无法逆转 prompt memory 已造成的错误。",
            "- `1161` 的 8 条策略全部预测 `140`，即使修复原 token-id detector 漏掉句末的问题，Weaver 仍未提供可见救援路径。",
            "- `0976` 最佳策略把 `750` 改为 `645`，更接近 gold `540`，但 numeric closeness 不是 exact-positive 标签。",
            "- `0423` 与 `1088` 出现显著新轨迹，却只产生新的错误答案；Weaver 能改变 reasoning，不等于能修复 reasoning。",
            "- `0611` 与 `0754` 已另行完成深度 5 搜索，仍为 0/32 exact rescue。",
            "",
            "## 责任切分",
            "",
            "在这批样本和当前 checkpoint 上，数据不支持“只要训练一个更会选前三个句级时机的 Trigger 就能得到论文收益”。因为对每个失败样本，拥有答案信息的 oracle 都找不到 exact-positive 三步策略。当前证据把主要排查方向推向 Weaver 能力、prompt augmentation，以及 checkpoint / 论文配置差异。",
            "",
            "这仍不是任意深度策略的全称证明：筛选层只覆盖前三个动态决策；只有 `0611/0754` 已扩展到五步。",
            "",
            "## 完整性",
            "",
            f"- 样本：{analysis['sample_count']}；策略：{analysis['policy_count']}；完整到达三步：{analysis['full_horizon_count']}/{analysis['policy_count']}",
            f"- exact-correct policies：{analysis['correct_policy_count']}",
            f"- 截断：{analysis['truncated_count']}；误接纳 numeric/internal delimiter：{analysis['invalid_accepted_boundary_count']}",
            f"- 本筛选生成耗时：{analysis['total_latency_seconds']:.1f} 秒",
            "",
            "逐策略 completion、答案、边界前缀与 augmentation mask 保存在各 run 的 scored JSONL；汇总 JSON 保留所有 source hashes。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    oracle_paths, baseline_path = source_paths()
    baseline_records = {r["sample_id"]: r for r in read_jsonl(baseline_path)}
    records = []
    source_meta = []
    for path in oracle_paths:
        source = read_jsonl(path)
        selected = [
            record
            for record in source
            if record["decision_depth"] == 3 and record["sample_id"] in TARGET_IDS
        ]
        records.extend(selected)
        source_meta.append(
            {"path": portable_path(path), "sha256": sha256_file(path), "selected_records": len(selected)}
        )
    if len(records) != len(TARGET_IDS) * 8:
        raise ValueError(f"expected 56 oracle records, found {len(records)}")

    samples = []
    invalid_reasons = {
        "numeric_comma_or_period",
        "inside_calculator_span",
        "unfinished_arithmetic_before_punctuation",
        "unfinished_arithmetic_before_newline",
    }
    for sample_id in TARGET_IDS:
        group = sorted(
            (record for record in records if record["sample_id"] == sample_id),
            key=lambda record: record["policy_string"],
        )
        if [record["policy_string"] for record in group] != [f"{value:03b}" for value in range(8)]:
            raise ValueError(f"incomplete policies for {sample_id}")
        baseline = baseline_records[sample_id]
        gold = group[0]["scores"]["semantic_gold"]
        baseline_hash = token_hash(group[0])
        variants = []
        for record in group:
            candidate = record["scores"]["semantic_candidate"]
            absolute_error = error(candidate, gold)
            variants.append(
                {
                    "policy": record["policy_string"],
                    "prediction": candidate,
                    "semantic_correct": bool(record["scores"]["semantic_correct"]),
                    "absolute_error": format(absolute_error, "f") if absolute_error is not None else None,
                    "completion_changed_vs_000": token_hash(record) != baseline_hash,
                    "horizon_reached": record["intervention_reached"],
                    "inference_augmentations": record["inference_augmentation_count"],
                    "generated_token_count": record["generated_token_count"],
                }
            )
        numeric_variants = [v for v in variants if v["absolute_error"] is not None]
        best_error = min(Decimal(v["absolute_error"]) for v in numeric_variants)
        samples.append(
            {
                "sample_id": sample_id,
                "gold": gold,
                "baseline_prediction": baseline["scores"]["semantic_candidate"],
                "baseline_truncated": baseline["truncated_at_max_new_tokens"],
                "policy_count": len(group),
                "full_horizon_count": sum(v["horizon_reached"] for v in variants),
                "correct_policy_count": sum(v["semantic_correct"] for v in variants),
                "correct_policies": [v["policy"] for v in variants if v["semantic_correct"]],
                "unique_completion_count": len({token_hash(record) for record in group}),
                "unique_prediction_count": len({v["prediction"] for v in variants}),
                "prediction_distribution": dict(Counter(v["prediction"] for v in variants)),
                "changed_completion_count": sum(v["completion_changed_vs_000"] for v in variants),
                "best_absolute_error": format(best_error, "f"),
                "best_error_policies": [v["policy"] for v in variants if Decimal(v["absolute_error"]) == best_error],
                "variants": variants,
            }
        )

    analysis = {
        "schema_version": 1,
        "analyzed_at": utc_now(),
        "scope": "depth-3 dynamic sentence-level oracle on non-truncated N-wrong pilot samples",
        "baseline_source": {"path": portable_path(baseline_path), "sha256": sha256_file(baseline_path)},
        "oracle_sources": source_meta,
        "excluded": {
            "N_correct": ["gsm8k_test_0710", "gsm8k_test_0106"],
            "N_truncated": ["gsm8k_test_0810"],
        },
        "sample_count": len(samples),
        "policy_count": len(records),
        "full_horizon_count": sum(record["intervention_reached"] for record in records),
        "correct_policy_count": sum(record["scores"]["semantic_correct"] for record in records),
        "rescued_sample_count": sum(sample["correct_policy_count"] > 0 for sample in samples),
        "truncated_count": sum(record["truncated_at_max_new_tokens"] for record in records),
        "invalid_accepted_boundary_count": sum(
            event.get("policy_bit_index") is not None and event.get("boundary_reason") in invalid_reasons
            for record in records
            for event in record["gate_events"]
        ),
        "total_latency_seconds": sum(record["latency_seconds"] for record in records),
        "samples": samples,
    }
    write_json(args.output_json.resolve(), analysis)
    args.report.resolve().write_text(render_report(analysis), encoding="utf-8", newline="\n")
    print(json.dumps({key: analysis[key] for key in ("sample_count", "policy_count", "full_horizon_count", "correct_policy_count", "rescued_sample_count", "truncated_count", "invalid_accepted_boundary_count")}, indent=2))
    print(f"analysis={args.output_json.resolve()}")
    print(f"report={args.report.resolve()}")


if __name__ == "__main__":
    main()
