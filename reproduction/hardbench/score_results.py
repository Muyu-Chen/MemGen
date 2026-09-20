"""Score immutable raw HardBench outputs with official and semantic metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from common import read_jsonl, sha256_file, utc_now, write_json, write_jsonl
from data.utils.math_utils import (
    compute_score,
    first_boxed_only_string,
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)


NUMBER_PATTERN = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def clean_gold(raw_answer: str) -> str:
    return raw_answer.rsplit("####", 1)[-1].strip().replace(",", "")


def semantic_candidate(completion: str) -> tuple[str | None, str]:
    boxed = last_boxed_only_string(completion)
    if boxed is not None:
        return remove_boxed(boxed).strip(), "last_boxed"
    match = re.search(r"####\s*([^\n]+)", completion)
    if match:
        return match.group(1).strip(), "hash_answer"
    numbers = NUMBER_PATTERN.findall(completion)
    if numbers:
        return numbers[-1].strip(), "last_number"
    return None, "missing"


def numeric_value(value: str) -> Fraction | Decimal | None:
    cleaned = value.strip().replace("$", "").replace(",", "")
    percent = cleaned.endswith("%")
    if percent:
        cleaned = cleaned[:-1]
    try:
        parsed: Fraction | Decimal
        if "/" in cleaned and cleaned.count("/") == 1:
            parsed = Fraction(cleaned)
        else:
            parsed = Decimal(cleaned)
        if percent:
            parsed = parsed / 100
        return parsed
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def semantic_correct(candidate: str | None, gold: str) -> bool:
    if candidate is None:
        return False
    if is_equiv(candidate, gold):
        return True
    candidate_value = numeric_value(candidate)
    gold_value = numeric_value(gold)
    if candidate_value is None or gold_value is None:
        return False
    return candidate_value == gold_value


def exact_mcnemar_p(recovered: int, regressed: int) -> float:
    discordant = recovered + regressed
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(recovered, regressed) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    has_augmentations = any(
        "inference_augmentation_count" in record for record in records
    )
    augmentation_distribution = None
    if has_augmentations:
        augmentation_distribution = {
            str(count): sum(
                record.get("inference_augmentation_count") == count
                for record in records
            )
            for count in sorted(
                {
                    record["inference_augmentation_count"]
                    for record in records
                    if "inference_augmentation_count" in record
                }
            )
        }
    disagreement_count = sum(
        bool(record["scores"]["official_compute_score"])
        != bool(record["scores"]["semantic_correct"])
        for record in records
    )
    return {
        "n": total,
        "official_correct": sum(record["scores"]["official_compute_score"] for record in records),
        "official_accuracy": (
            sum(record["scores"]["official_compute_score"] for record in records) / total
            if total
            else None
        ),
        "semantic_correct": sum(record["scores"]["semantic_correct"] for record in records),
        "semantic_accuracy": (
            sum(record["scores"]["semantic_correct"] for record in records) / total
            if total
            else None
        ),
        "boxed_format_count": sum(record["scores"]["boxed_format_compliant"] for record in records),
        "boxed_format_rate": (
            sum(record["scores"]["boxed_format_compliant"] for record in records) / total
            if total
            else None
        ),
        "official_semantic_disagreement_count": disagreement_count,
        "official_semantic_disagreement_rate": (
            disagreement_count / total if total else None
        ),
        "truncated_count": sum(record["truncated_at_max_new_tokens"] for record in records),
        "mean_latency_seconds": (
            sum(record["latency_seconds"] for record in records) / total if total else None
        ),
        "mean_generated_tokens": (
            sum(record["generated_token_count"] for record in records) / total if total else None
        ),
        "mean_seconds_per_generated_token": (
            sum(record["seconds_per_generated_token"] for record in records) / total
            if total
            else None
        ),
        "mean_inference_augmentations": (
            sum(record.get("inference_augmentation_count", 0) for record in records) / total
            if total and has_augmentations
            else None
        ),
        "inference_augmentation_count_distribution": augmentation_distribution,
    }


def pairwise(
    left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    left_map = {record["sample_id"]: record for record in left}
    right_map = {record["sample_id"]: record for record in right}
    ids = sorted(set(left_map) & set(right_map))
    preserved = recovered = regressed = both_wrong = 0
    for sample_id in ids:
        a = bool(left_map[sample_id]["scores"][metric])
        b = bool(right_map[sample_id]["scores"][metric])
        if a and b:
            preserved += 1
        elif not a and b:
            recovered += 1
        elif a and not b:
            regressed += 1
        else:
            both_wrong += 1
    return {
        "n_paired": len(ids),
        "preserved": preserved,
        "recovered": recovered,
        "regressed": regressed,
        "both_wrong": both_wrong,
        "net_gain": recovered - regressed,
        "mcnemar_exact_p": exact_mcnemar_p(recovered, regressed),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    raw_files = sorted((run_dir / "raw").glob("*.raw.jsonl"))
    if not raw_files:
        raise FileNotFoundError(f"No raw result files found under {run_dir / 'raw'}")

    by_condition: dict[str, list[dict[str, Any]]] = {}
    raw_sources = []
    for raw_file in raw_files:
        raw_records = read_jsonl(raw_file)
        scored_records = []
        for raw in raw_records:
            completion = raw["completion_text"]
            gold = clean_gold(raw["raw_answer"])
            candidate, method = semantic_candidate(completion)
            scored = dict(raw)
            scored["scores"] = {
                "official_compute_score": float(
                    compute_score(completion, raw["official_solution"])
                ),
                "boxed_format_compliant": first_boxed_only_string(completion) is not None,
                "semantic_candidate": candidate,
                "semantic_extraction_method": method,
                "semantic_gold": gold,
                "semantic_correct": semantic_correct(candidate, gold),
            }
            scored_records.append(scored)
        condition = scored_records[0]["condition"] if scored_records else raw_file.stem
        by_condition[condition] = scored_records
        output = run_dir / "scored" / raw_file.name.replace(".raw.jsonl", ".scored.jsonl")
        write_jsonl(output, scored_records)
        raw_sources.append(
            {
                "path": str(raw_file),
                "sha256": sha256_file(raw_file),
                "records": len(raw_records),
                "scored_path": str(output),
            }
        )

    comparisons = {}
    conditions = sorted(by_condition)
    for left_index, left_condition in enumerate(conditions):
        for right_condition in conditions[left_index + 1 :]:
            key = f"{left_condition}__vs__{right_condition}"
            comparisons[key] = {
                "official_compute_score": pairwise(
                    by_condition[left_condition],
                    by_condition[right_condition],
                    "official_compute_score",
                ),
                "semantic_correct": pairwise(
                    by_condition[left_condition],
                    by_condition[right_condition],
                    "semantic_correct",
                ),
            }

    summary = {
        "schema_version": 1,
        "scored_at": utc_now(),
        "raw_sources": raw_sources,
        "conditions": {
            condition: summarize(records) for condition, records in by_condition.items()
        },
        "pairwise": comparisons,
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=2))
    print(f"summary={run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
