"""Consolidate pilot, human intervention, sweep, and interaction evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common import HARD_BENCH_ROOT, read_jsonl, sha256_file, utc_now, write_json


PILOT = HARD_BENCH_ROOT / "runs" / "pilot_min6_v1" / "scored"
INTERVENTION = HARD_BENCH_ROOT / "runs" / "trigger_intervention_v1" / "scored"
SWEEP_RUN = HARD_BENCH_ROOT / "runs" / "single_candidate_sweep_v1"
INTERACTION_RUN = HARD_BENCH_ROOT / "runs" / "candidate_interaction_v1"
OUTPUT = HARD_BENCH_ROOT / "analysis" / "trigger_evidence_v1.json"

PILOT_FILES = {
    "A_base_zero_shot": PILOT / "A_base_zero_shot.scored.jsonl",
    "S_base_3shot": PILOT / "B_base_3shot.scored.jsonl",
    "B_memgen_always": PILOT / "D_memgen_official.scored.jsonl",
    "C_memgen_random50": PILOT / "R_memgen_random50_inference.scored.jsonl",
}


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(bool(record["scores"]["semantic_correct"]) for record in records)
    augmentations = [
        record["inference_augmentation_count"]
        for record in records
        if "inference_augmentation_count" in record
    ]
    return {
        "n": total,
        "semantic_correct": correct,
        "semantic_accuracy": correct / total if total else None,
        "mean_inference_augmentations": (
            sum(augmentations) / len(augmentations) if augmentations else None
        ),
    }


def compact_result(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": record["scores"]["semantic_candidate"],
        "correct": bool(record["scores"]["semantic_correct"]),
        "inference_augmentations": record.get("inference_augmentation_count"),
    }


def main() -> None:
    sources: list[Path] = []
    pilot_records: dict[str, list[dict[str, Any]]] = {}
    for label, path in PILOT_FILES.items():
        pilot_records[label] = read_jsonl(path)
        sources.append(path)

    pilot_by_id = {
        label: {record["sample_id"]: record for record in records}
        for label, records in pilot_records.items()
    }
    pilot_ids = sorted(pilot_by_id["A_base_zero_shot"])
    pilot_samples = []
    for sample_id in pilot_ids:
        pilot_samples.append(
            {
                "sample_id": sample_id,
                "gold": pilot_by_id["A_base_zero_shot"][sample_id]["scores"][
                    "semantic_gold"
                ],
                "conditions": {
                    label: compact_result(records[sample_id])
                    for label, records in pilot_by_id.items()
                },
            }
        )

    n_path = INTERVENTION / "N_memgen_no_inference.scored.jsonl"
    h_path = INTERVENTION / "H_memgen_human_single.scored.jsonl"
    n_records = read_jsonl(n_path)
    h_records = read_jsonl(h_path)
    sources.extend((n_path, h_path))
    n_by_id = {record["sample_id"]: record for record in n_records}
    h_by_id = {record["sample_id"]: record for record in h_records}
    paired_ids = sorted(set(n_by_id) & set(h_by_id))
    human_changed = sum(
        n_by_id[sample_id]["completion_text"] != h_by_id[sample_id]["completion_text"]
        for sample_id in paired_ids
    )
    human_prediction_changed = sum(
        n_by_id[sample_id]["scores"]["semantic_candidate"]
        != h_by_id[sample_id]["scores"]["semantic_candidate"]
        for sample_id in paired_ids
    )
    human_recovered = sum(
        not n_by_id[sample_id]["scores"]["semantic_correct"]
        and h_by_id[sample_id]["scores"]["semantic_correct"]
        for sample_id in paired_ids
    )
    human_regressed = sum(
        n_by_id[sample_id]["scores"]["semantic_correct"]
        and not h_by_id[sample_id]["scores"]["semantic_correct"]
        for sample_id in paired_ids
    )

    sweep_analysis_path = SWEEP_RUN / "sweep_analysis.json"
    sweep_scored_path = (
        SWEEP_RUN / "scored" / "X_memgen_single_candidate_sweep.scored.jsonl"
    )
    sweep_analysis = json.loads(sweep_analysis_path.read_text(encoding="utf-8"))
    sweep_records = read_jsonl(sweep_scored_path)
    sources.extend((sweep_analysis_path, sweep_scored_path))

    interaction_path = (
        INTERACTION_RUN
        / "scored"
        / "Y_memgen_candidate_interaction.scored.jsonl"
    )
    interaction_records = read_jsonl(interaction_path)
    sources.append(interaction_path)

    target_id = "gsm8k_test_0710"
    baseline = n_by_id[target_id]
    singles = {
        (record["selected_candidate_ordinal"],): record
        for record in sweep_records
        if record["sample_id"] == target_id
        and record["selected_candidate_ordinal"] in {1, 2, 3}
    }
    interactions = {
        tuple(record["selected_candidate_ordinals"]): record
        for record in interaction_records
    }
    truth_records = {(): baseline, **singles, **interactions}
    truth_table = []
    for subset in sorted(truth_records, key=lambda value: (len(value), value)):
        record = truth_records[subset]
        truth_table.append(
            {
                "selected_candidate_ordinals": list(subset),
                "prediction": record["scores"]["semantic_candidate"],
                "correct": bool(record["scores"]["semantic_correct"]),
                "completion_sha256": content_sha256(record["completion_text"]),
            }
        )
    destructive = {
        subset
        for subset, record in truth_records.items()
        if baseline["scores"]["semantic_correct"]
        and not record["scores"]["semantic_correct"]
    }
    minimal_destructive = [
        list(subset)
        for subset in sorted(destructive, key=lambda value: (len(value), value))
        if not any(
            other < set(subset)
            for other in (set(candidate) for candidate in destructive)
        )
    ]
    official = pilot_by_id["B_memgen_always"][target_id]
    triple = interactions[(1, 2, 3)]

    all_generation_records = (
        [record for records in pilot_records.values() for record in records]
        + n_records
        + h_records
        + sweep_records
        + interaction_records
    )
    output = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "scope": {
            "stored_generation_records": len(all_generation_records),
            "total_generation_latency_seconds": sum(
                record["latency_seconds"] for record in all_generation_records
            ),
            "source_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in sources
            ],
        },
        "pilot": {
            "conditions": {
                label: semantic_summary(records)
                for label, records in pilot_records.items()
            },
            "samples": pilot_samples,
        },
        "human_single_intervention": {
            "n_paired": len(paired_ids),
            "baseline": semantic_summary(n_records),
            "human_single": semantic_summary(h_records),
            "completion_changed": human_changed,
            "final_prediction_changed": human_prediction_changed,
            "exact_recovered": human_recovered,
            "exact_regressed": human_regressed,
        },
        "single_candidate_sweep": {
            "totals": sweep_analysis["totals"],
            "samples": [
                {
                    "sample_id": sample["sample_id"],
                    "baseline_correct": sample["baseline_correct"],
                    "planned_candidate_count": sample["planned_candidate_count"],
                    "exact_correct_ordinals": sample["exact_correct_ordinals"],
                    "best_candidate_ordinals": sample["best_candidate_ordinals"],
                    "best_absolute_error": sample["best_absolute_error"],
                    "changed_variant_count": sample["changed_variant_count"],
                }
                for sample in sweep_analysis["samples"]
            ],
            "failed_baseline_variants": sum(
                sample["planned_candidate_count"]
                for sample in sweep_analysis["samples"]
                if not sample["baseline_correct"]
            ),
            "failed_baseline_exact_rescues": sum(
                len(sample["exact_correct_ordinals"])
                for sample in sweep_analysis["samples"]
                if not sample["baseline_correct"]
            ),
        },
        "candidate_interaction": {
            "sample_id": target_id,
            "truth_table": truth_table,
            "minimal_destructive_subsets": minimal_destructive,
            "triple_matches_official_completion": (
                triple["completion_text"] == official["completion_text"]
            ),
            "triple_matches_official_token_ids": (
                triple["completion_token_ids"] == official["completion_token_ids"]
            ),
            "triple_matches_official_augmentation_positions": (
                triple["inference_augmentation_positions"]
                == official["inference_augmentation_positions"]
            ),
        },
    }
    write_json(OUTPUT, output)
    print(f"analysis={OUTPUT}")
    print(f"stored_generation_records={len(all_generation_records)}")
    print(f"minimal_destructive_subsets={minimal_destructive}")


if __name__ == "__main__":
    main()
