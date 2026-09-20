"""Generate raw A/B Base outputs. This module deliberately does no scoring."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from common import (
    BASE_MODEL_PATH,
    DEFAULT_MANIFEST,
    append_jsonl,
    build_messages,
    completed_sample_ids,
    load_manifest,
    load_tokenizer,
    meta_path_for,
    resolve_torch_dtype,
    sha256_file,
    utc_now,
    write_json,
)


CONDITIONS = ("A_base_zero_shot", "B_base_3shot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--time-budget-minutes", type=float, default=110.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    run_dir = args.run_dir.resolve()
    raw_dir = run_dir / "raw"
    samples = load_manifest(manifest_path, args.max_samples)
    meta = json.loads(meta_path_for(manifest_path).read_text(encoding="utf-8"))

    torch.set_num_threads(args.threads)
    dtype = resolve_torch_dtype(args.dtype)
    tokenizer = load_tokenizer()
    print(f"Loading Base from {BASE_MODEL_PATH} dtype={args.dtype} threads={args.threads}")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=dtype)
    model.eval()
    print(f"Base loaded in {time.perf_counter() - load_started:.1f}s")

    write_json(
        run_dir / "base_run_config.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "runner": "run_base.py",
            "conditions": args.conditions,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "model": str(BASE_MODEL_PATH),
            "dtype": args.dtype,
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "generation": "greedy",
        },
    )

    outputs = {condition: raw_dir / f"{condition}.raw.jsonl" for condition in args.conditions}
    completed = {
        condition: completed_sample_ids(outputs[condition], condition)
        for condition in args.conditions
    }
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    run_started = time.perf_counter()
    completed_pairs = 0
    for sample in samples:
        if (
            completed_pairs > 0
            and time.perf_counter() - run_started >= args.time_budget_minutes * 60
        ):
            print("Time budget reached before starting the next paired sample")
            break

        for condition in args.conditions:
            if sample["sample_id"] in completed[condition]:
                print(f"skip {condition} {sample['sample_id']}")
                continue
            messages = build_messages(sample, condition, meta)
            rendered_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered_prompt, return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[1]
            started = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config,
                )
            elapsed = time.perf_counter() - started
            new_ids = output_ids[0, prompt_len:].cpu()
            token_ids = new_ids.tolist()
            eos_emitted = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
            record = {
                "schema_version": 1,
                "generated_at": utc_now(),
                "condition": condition,
                "sample_id": sample["sample_id"],
                "source_dataset": sample["source_dataset"],
                "source_index": sample["source_index"],
                "selection_calculator_steps": sample["selection_calculator_steps"],
                "question": sample["question"],
                "raw_answer": sample["raw_answer"],
                "official_solution": sample["official_solution"],
                "messages": messages,
                "rendered_prompt": rendered_prompt,
                "prompt_token_count": prompt_len,
                "completion_token_ids": token_ids,
                "completion_text_raw": tokenizer.decode(token_ids, skip_special_tokens=False),
                "completion_text": tokenizer.decode(token_ids, skip_special_tokens=True),
                "generated_token_count": len(token_ids),
                "eos_emitted": eos_emitted,
                "truncated_at_max_new_tokens": len(token_ids) >= args.max_new_tokens and not eos_emitted,
                "latency_seconds": elapsed,
                "seconds_per_generated_token": elapsed / max(1, len(token_ids)),
                "model_path": str(BASE_MODEL_PATH),
                "dtype": args.dtype,
                "threads": args.threads,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
            }
            append_jsonl(outputs[condition], record)
            completed[condition].add(sample["sample_id"])
            print(
                f"saved {condition} {sample['sample_id']} tokens={len(token_ids)} "
                f"latency={elapsed:.1f}s raw={outputs[condition]}"
            )
        completed_pairs += 1


if __name__ == "__main__":
    main()

