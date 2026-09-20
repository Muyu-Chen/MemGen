"""Create a deterministic, model-independent GSM8K hard-subset manifest."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from datasets import load_dataset

from common import (
    DEFAULT_MANIFEST,
    meta_path_for,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from data.gsm8k.builder import GSM8KBuilder


CALCULATOR_PATTERN = re.compile(r"<<.*?>>")


def calculator_steps(answer: str) -> int:
    return len(CALCULATOR_PATTERN.findall(answer))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--few-shot-count", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    meta_path = meta_path_for(output)
    if (output.exists() or meta_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")

    dataset = load_dataset("gsm8k", "main")
    test_rows: list[dict] = []
    for source_index, example in enumerate(dataset["test"]):
        steps = calculator_steps(example["answer"])
        if steps < args.min_steps:
            continue
        processed = GSM8KBuilder._preprocess(example)
        test_rows.append(
            {
                "sample_id": f"gsm8k_test_{source_index:04d}",
                "source_dataset": "gsm8k/main/test",
                "source_index": source_index,
                "selection_calculator_steps": steps,
                "question": example["question"].strip(),
                "raw_answer": example["answer"].strip(),
                "official_solution": processed["solution"],
                "official_prompt_messages": processed["prompt"],
            }
        )

    test_rng = random.Random(args.seed)
    test_rng.shuffle(test_rows)
    selected = test_rows[: args.limit]
    if not selected:
        raise RuntimeError("No examples matched the hard-subset rule")

    train_indices = list(range(len(dataset["train"])))
    shot_rng = random.Random(args.seed + 1)
    shot_rng.shuffle(train_indices)
    few_shot_examples = []
    few_shot_messages: list[dict[str, str]] = []
    for source_index in train_indices[: args.few_shot_count]:
        example = dataset["train"][source_index]
        processed = GSM8KBuilder._preprocess(example)
        few_shot_examples.append(
            {
                "source_dataset": "gsm8k/main/train",
                "source_index": source_index,
                "question": example["question"].strip(),
                "raw_answer": example["answer"].strip(),
            }
        )
        few_shot_messages.extend(processed["prompt"])
        few_shot_messages.extend(processed["completion"])

    write_jsonl(output, selected)
    write_json(
        meta_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "selection": {
                "dataset": "gsm8k/main/test",
                "rule": "count of gold <<...>> calculator annotations >= min_steps",
                "min_steps": args.min_steps,
                "seed": args.seed,
                "eligible_count": len(test_rows),
                "selected_count": len(selected),
                "limit": args.limit,
            },
            "few_shot_examples": few_shot_examples,
            "few_shot_messages": few_shot_messages,
            "manifest_file": output.name,
            "manifest_sha256": sha256_file(output),
        },
    )
    print(f"manifest={output}")
    print(f"eligible={len(test_rows)} selected={len(selected)} min_steps={args.min_steps}")
    print(f"meta={meta_path}")


if __name__ == "__main__":
    main()
