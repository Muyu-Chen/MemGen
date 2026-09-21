"""Analyze a single-candidate sweep against the no-inference baseline."""

from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from common import HARD_BENCH_ROOT, read_jsonl, sha256_file, utc_now, write_json
from score_results import numeric_value


CONDITION = "X_memgen_single_candidate_sweep"
DEFAULT_BASELINE = (
    HARD_BENCH_ROOT
    / "runs"
    / "trigger_intervention_v1"
    / "scored"
    / "N_memgen_no_inference.scored.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-scored", type=Path, default=DEFAULT_BASELINE)
    return parser.parse_args()


def as_decimal(value: Fraction | Decimal | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return value


def absolute_error(candidate: str | None, gold: str) -> Decimal | None:
    if candidate is None:
        return None
    candidate_value = as_decimal(numeric_value(candidate))
    gold_value = as_decimal(numeric_value(gold))
    if candidate_value is None or gold_value is None:
        return None
    return abs(candidate_value - gold_value)


def distance_relation(
    variant_distance: Decimal | None, baseline_distance: Decimal | None
) -> str:
    if variant_distance is None or baseline_distance is None:
        return "unknown"
    if variant_distance < baseline_distance:
        return "improved"
    if variant_distance > baseline_distance:
        return "worsened"
    return "equal"


def decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    baseline_path = args.baseline_scored.resolve()
    sweep_path = run_dir / "scored" / f"{CONDITION}.scored.jsonl"
    sweep = read_jsonl(sweep_path)
    baseline = {record["sample_id"]: record for record in read_jsonl(baseline_path)}
    if not sweep:
        raise FileNotFoundError(f"scored sweep is missing or empty: {sweep_path}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in sweep:
        grouped.setdefault(record["sample_id"], []).append(record)

    totals = {
        "observed_variants": 0,
        "intervention_reached": 0,
        "correct_variants": 0,
        "changed_vs_baseline": 0,
        "distance_improved": 0,
        "distance_worsened": 0,
        "distance_equal": 0,
        "distance_unknown": 0,
    }
    sample_summaries = []
    for sample_id, records in grouped.items():
        records.sort(key=lambda record: record["selected_candidate_ordinal"])
        if sample_id not in baseline:
            raise ValueError(f"baseline has no record for {sample_id}")
        base = baseline[sample_id]
        gold = base["scores"]["semantic_gold"]
        base_candidate = base["scores"]["semantic_candidate"]
        base_distance = absolute_error(base_candidate, gold)
        variants = []
        for record in records:
            candidate = record["scores"]["semantic_candidate"]
            distance = absolute_error(candidate, gold)
            relation = distance_relation(distance, base_distance)
            changed = record["completion_text"] != base["completion_text"]
            correct = bool(record["scores"]["semantic_correct"])
            totals["observed_variants"] += 1
            totals["intervention_reached"] += bool(record["intervention_reached"])
            totals["correct_variants"] += correct
            totals["changed_vs_baseline"] += changed
            totals[f"distance_{relation}"] += 1
            variants.append(
                {
                    "candidate_ordinal": record["selected_candidate_ordinal"],
                    "prediction": candidate,
                    "semantic_correct": correct,
                    "absolute_error": decimal_text(distance),
                    "distance_relation_vs_baseline": relation,
                    "completion_changed_vs_baseline": changed,
                    "intervention_reached": bool(record["intervention_reached"]),
                    "latency_seconds": record["latency_seconds"],
                    "generated_token_count": record["generated_token_count"],
                }
            )

        known_distances = [
            (absolute_error(item["scores"]["semantic_candidate"], gold), item)
            for item in records
        ]
        known_distances = [pair for pair in known_distances if pair[0] is not None]
        best_distance = min((pair[0] for pair in known_distances), default=None)
        best_ordinals = [
            item["selected_candidate_ordinal"]
            for distance, item in known_distances
            if distance == best_distance
        ]
        sample_summaries.append(
            {
                "sample_id": sample_id,
                "role": records[0]["sweep_plan_role"],
                "gold": gold,
                "baseline_prediction": base_candidate,
                "baseline_correct": bool(base["scores"]["semantic_correct"]),
                "baseline_absolute_error": decimal_text(base_distance),
                "planned_candidate_count": records[0]["planned_candidate_count"],
                "observed_variant_count": len(records),
                "exact_correct_ordinals": [
                    variant["candidate_ordinal"]
                    for variant in variants
                    if variant["semantic_correct"]
                ],
                "best_absolute_error": decimal_text(best_distance),
                "best_candidate_ordinals": best_ordinals,
                "changed_variant_count": sum(
                    variant["completion_changed_vs_baseline"] for variant in variants
                ),
                "unique_predictions": sorted(
                    {variant["prediction"] for variant in variants if variant["prediction"]}
                ),
                "variants": variants,
            }
        )

    output = {
        "schema_version": 1,
        "analyzed_at": utc_now(),
        "condition": CONDITION,
        "sweep_scored": str(sweep_path),
        "sweep_scored_sha256": sha256_file(sweep_path),
        "baseline_scored": str(baseline_path),
        "baseline_scored_sha256": sha256_file(baseline_path),
        "totals": totals,
        "samples": sample_summaries,
    }
    output_path = run_dir / "sweep_analysis.json"
    write_json(output_path, output)
    print(f"analysis={output_path}")
    print(totals)


if __name__ == "__main__":
    main()
