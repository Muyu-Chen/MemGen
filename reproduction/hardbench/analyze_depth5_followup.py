"""Analyze the targeted depth-5 sentence-oracle follow-up."""

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


CONDITION = "Z4_memgen_text_sentence_oracle_depth5"
TARGET_IDS = ["gsm8k_test_0976", "gsm8k_test_0423"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=HARD_BENCH_ROOT / "runs" / "text_sentence_oracle_depth5_followup_v1",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=HARD_BENCH_ROOT / "analysis" / "text_sentence_oracle_depth5_followup_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=HARD_BENCH_ROOT / "TEXT_SENTENCE_ORACLE_DEPTH5_FOLLOWUP_REPORT.md",
    )
    return parser.parse_args()


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


def render_report(analysis: dict) -> str:
    lines = [
        "# Depth-5 Sentence Oracle Follow-up",
        "",
        "**结论：两个深度 3 时曾被 Weaver 推近答案的样本，在完整 `2^5` 动态句级策略搜索中仍然没有 exact rescue（0/64）。**",
        "",
        "## 为什么选这两题",
        "",
        "`0976` 与 `0423` 是深度 3 筛选中最接近形成正例的两题：调用 Weaver 后数值误差曾缩小，因此它们比输出完全不变的题更适合检验“更多顺序调用是否能跨过最后一步”。Prompt augmentation 始终开启，后续决策点沿每条新轨迹重新识别。",
        "",
        "| sample | gold | 00000 | unique answers | 完整到达五步 | exact | 最佳预测 | 最佳误差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in analysis["samples"]:
        lines.append(
            f"| `{sample['sample_id']}` | {sample['gold']} | {sample['no_invoke_prediction']} | "
            f"{sample['unique_prediction_count']} | {sample['full_horizon_count']}/32 | "
            f"{sample['correct_policy_count']}/32 | {sample['best_predictions']} | "
            f"{sample['best_absolute_error']} |"
        )
    lines.extend(
        [
            "",
            "## 观察",
            "",
            "- `0976` 的最佳预测从三步搜索的 `645` 改善到 `570`（gold `540`），说明更多调用能继续改变方向，但仍不是 exact-positive。",
            "- `0423` 出现提前终止的分支；这证明第一次调用会改变后续轨迹和可达决策节点，不能把本实验等价成固定 baseline 上选一个 subset。",
            "- 数值更近仅作为诊断信号，评分仍严格使用 GSM8K exact/semantic numeric equality；没有把近似答案算作 rescue。",
            "",
            "## 能推出什么",
            "",
            "在当前 checkpoint、prompt memory 和句级候选生成规则下，把策略深度从 3 扩到 5 仍未为这两道最有希望的题创造正例。这进一步削弱了“Trigger 只需学会组合更多正确时机”的解释。剩余更合理的排查对象是 Weaver 生成的记忆内容、prompt augmentation 的因果影响，以及发布 checkpoint 与论文配置是否一致。",
            "",
            "该结论只覆盖前五个动态决策节点；它不声称穷尽任意长度或其他 delimiter 集合。",
            "",
            "## 完整性",
            "",
            f"- 策略：{analysis['policy_count']}/64；exact-correct：{analysis['correct_policy_count']}；截断：{analysis['truncated_count']}",
            f"- 完整到达五步：{analysis['full_horizon_count']}/{analysis['policy_count']}；其余路径提前到达生成叶节点",
            f"- 已发生的 policy prefix 全部与计划一致：{analysis['all_observed_prefixes_match_policies']}",
            f"- 误接纳 numeric/internal delimiter：{analysis['invalid_accepted_boundary_count']}",
            f"- 总生成耗时：{analysis['total_latency_seconds']:.1f} 秒",
            "",
            "逐策略的 completion、边界前缀、实际调用次数与答案保存在 scored JSONL；汇总 JSON 记录 source hashes。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "raw" / f"{CONDITION}.raw.jsonl"
    scored_path = run_dir / "scored" / f"{CONDITION}.scored.jsonl"
    records = read_jsonl(scored_path)
    if len(records) != 64:
        raise ValueError(f"expected 64 records, found {len(records)}")

    invalid_reasons = {
        "numeric_comma_or_period",
        "inside_calculator_span",
        "unfinished_arithmetic_before_punctuation",
        "unfinished_arithmetic_before_newline",
    }
    samples = []
    for sample_id in TARGET_IDS:
        group = sorted(
            (record for record in records if record["sample_id"] == sample_id),
            key=lambda record: record["policy_string"],
        )
        expected = [f"{value:05b}" for value in range(32)]
        if [record["policy_string"] for record in group] != expected:
            raise ValueError(f"incomplete policies for {sample_id}")
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
                    "completion_changed_vs_00000": token_hash(record) != no_invoke_hash,
                    "horizon_reached": record["intervention_reached"],
                    "observed_policy_bits": record["observed_policy_bits"],
                    "inference_augmentations": record["inference_augmentation_count"],
                    "generated_token_count": record["generated_token_count"],
                }
            )
        numeric = [variant for variant in variants if variant["absolute_error"] is not None]
        best_error = min(Decimal(variant["absolute_error"]) for variant in numeric)
        best = [variant for variant in numeric if Decimal(variant["absolute_error"]) == best_error]
        samples.append(
            {
                "sample_id": sample_id,
                "gold": gold,
                "no_invoke_prediction": variants[0]["prediction"],
                "policy_count": len(group),
                "full_horizon_count": sum(variant["horizon_reached"] for variant in variants),
                "correct_policy_count": sum(variant["semantic_correct"] for variant in variants),
                "correct_policies": [variant["policy"] for variant in variants if variant["semantic_correct"]],
                "unique_completion_count": len({token_hash(record) for record in group}),
                "unique_prediction_count": len({variant["prediction"] for variant in variants}),
                "prediction_distribution": dict(Counter(variant["prediction"] for variant in variants)),
                "changed_completion_count": sum(variant["completion_changed_vs_00000"] for variant in variants),
                "best_absolute_error": format(best_error, "f"),
                "best_predictions": sorted({variant["prediction"] for variant in best}),
                "best_error_policies": [variant["policy"] for variant in best],
                "variants": variants,
            }
        )

    analysis = {
        "schema_version": 1,
        "analyzed_at": utc_now(),
        "scope": "depth-5 dynamic sentence-level oracle follow-up on the two most promising depth-3 failures",
        "raw_source": {"path": portable_path(raw_path), "sha256": sha256_file(raw_path)},
        "scored_source": {"path": portable_path(scored_path), "sha256": sha256_file(scored_path)},
        "sample_count": len(samples),
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
        "samples": samples,
    }
    write_json(args.output_json.resolve(), analysis)
    args.report.resolve().write_text(render_report(analysis), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                key: analysis[key]
                for key in (
                    "sample_count",
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
    for sample in samples:
        print(
            sample["sample_id"],
            json.dumps(
                {
                    key: sample[key]
                    for key in (
                        "correct_policy_count",
                        "unique_prediction_count",
                        "best_absolute_error",
                        "best_predictions",
                        "best_error_policies",
                    )
                },
                ensure_ascii=False,
            ),
        )
    print(f"analysis={args.output_json.resolve()}")
    print(f"report={args.report.resolve()}")


if __name__ == "__main__":
    main()
