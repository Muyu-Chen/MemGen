"""Run fixed multi-candidate intervention subsets for interaction diagnosis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import GenerationConfig

from common import (
    BASE_MODEL_PATH,
    CHECKPOINT_PATH,
    DEFAULT_MANIFEST,
    HARD_BENCH_ROOT,
    append_jsonl,
    build_messages,
    load_manifest,
    load_tokenizer,
    meta_path_for,
    read_jsonl,
    resolve_torch_dtype,
    sha256_file,
    utc_now,
    write_json,
)
from run_memgen import install_candidate_subset_gate, load_model


CONDITION = "Y_memgen_candidate_interaction"
DEFAULT_PLAN = HARD_BENCH_ROOT / "plans" / "candidate_interaction_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan-file", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--time-budget-minutes", type=float, default=30.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def validate_plan(plan: dict, candidate_source: Path) -> None:
    source = {
        record["sample_id"]: record for record in read_jsonl(candidate_source)
    }
    sample_id = plan["sample_id"]
    if sample_id not in source:
        raise ValueError(f"candidate source has no record for {sample_id}")
    candidate_count = sum(
        not event.get("is_prompt", False)
        for event in source[sample_id].get("random_gate_events", [])
    )
    seen_ids: set[str] = set()
    seen_subsets: set[tuple[int, ...]] = set()
    for variant in plan.get("variants", []):
        variant_id = variant["variant_id"]
        selected = variant["selected_candidate_ordinals"]
        selected_tuple = tuple(selected)
        if variant_id in seen_ids:
            raise ValueError(f"duplicate variant_id: {variant_id}")
        if selected_tuple in seen_subsets:
            raise ValueError(f"duplicate selected subset: {selected}")
        if selected != sorted(set(selected)):
            raise ValueError(f"ordinals must be unique and sorted: {selected}")
        if not selected or any(ordinal < 1 or ordinal > candidate_count for ordinal in selected):
            raise ValueError(
                f"selected ordinals out of range 1..{candidate_count}: {selected}"
            )
        seen_ids.add(variant_id)
        seen_subsets.add(selected_tuple)
    if not seen_ids:
        raise ValueError("interaction plan contains no variants")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    plan_path = args.plan_file.resolve()
    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "raw" / f"{CONDITION}.raw.jsonl"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidate_source = (HARD_BENCH_ROOT / plan["candidate_source"]).resolve()
    validate_plan(plan, candidate_source)

    samples = {sample["sample_id"]: sample for sample in load_manifest(manifest_path)}
    if plan["sample_id"] not in samples:
        raise ValueError(f"sample not found in manifest: {plan['sample_id']}")
    sample = samples[plan["sample_id"]]
    meta = json.loads(meta_path_for(manifest_path).read_text(encoding="utf-8"))

    torch.set_num_threads(args.threads)
    dtype = resolve_torch_dtype(args.dtype)
    tokenizer = load_tokenizer()
    print(f"Loading MemGen dtype={args.dtype} threads={args.threads}", flush=True)
    load_started = time.perf_counter()
    model, config = load_model(tokenizer, dtype)
    print(f"MemGen loaded in {time.perf_counter() - load_started:.1f}s", flush=True)

    write_json(
        run_dir / f"{CONDITION}.run_config.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "runner": "run_trigger_interactions.py",
            "condition": CONDITION,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "plan_file": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "candidate_source": str(candidate_source),
            "candidate_source_sha256": sha256_file(candidate_source),
            "planned_variants": len(plan["variants"]),
            "base_model": str(BASE_MODEL_PATH),
            "checkpoint": str(CHECKPOINT_PATH),
            "dtype": args.dtype,
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "generation": "greedy",
            "trigger_active": bool(config.trigger_active),
            "max_prompt_aug_num": int(config.max_prompt_aug_num),
            "max_inference_aug_num": int(config.max_inference_aug_num),
            "gate_policy": "selected_candidate_subset",
        },
    )

    completed = {
        record["interaction_variant_id"]
        for record in read_jsonl(raw_path)
        if record.get("interaction_variant_id")
    }
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generation_config.weaver_do_sample = False
    generation_config.trigger_do_sample = False

    messages = build_messages(sample, "N_memgen_no_inference", meta)
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    run_started = time.perf_counter()
    generated = 0
    for variant in plan["variants"]:
        variant_id = variant["variant_id"]
        selected = variant["selected_candidate_ordinals"]
        if variant_id in completed:
            print(f"skip {variant_id}", flush=True)
            continue
        if generated > 0 and time.perf_counter() - run_started >= args.time_budget_minutes * 60:
            print("Time budget reached before starting the next variant", flush=True)
            break

        original_should_augment, gate_events = install_candidate_subset_gate(
            model, set(selected)
        )
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output_ids, augmentation_mask = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config,
                    return_augmentation_mask=True,
                )
        finally:
            model._should_augment = original_should_augment
        elapsed = time.perf_counter() - started

        new_ids = output_ids[0, prompt_len:].cpu()
        token_ids = new_ids.tolist()
        eos_emitted = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
        mask_values = augmentation_mask[0, : len(token_ids)].cpu().tolist()
        all_aug_positions = [
            index for index, value in enumerate(mask_values) if value == 1
        ]
        inference_aug_positions = [
            position for position in all_aug_positions if position > 0
        ]
        chosen_ordinals = [
            event["candidate_ordinal"]
            for event in gate_events
            if not event.get("is_prompt") and event.get("decision") == 1
        ]
        intervention_reached = (
            chosen_ordinals == selected
            and len(inference_aug_positions) == len(selected)
        )
        record = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "condition": CONDITION,
            "interaction_variant_id": variant_id,
            "selected_candidate_ordinals": selected,
            "interaction_order": len(selected),
            "intervention_reached": intervention_reached,
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
            "checkpoint_path": str(CHECKPOINT_PATH),
            "dtype": args.dtype,
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "checkpoint_trigger_active": False,
            "gate_policy": "selected_candidate_subset",
            "random_gate_events": gate_events,
            "prompt_augmented": bool(mask_values and mask_values[0] == 1),
            "inference_augmentation_count": len(inference_aug_positions),
            "inference_augmentation_positions": inference_aug_positions,
            "all_augmentation_positions": all_aug_positions,
            "augmentation_mask": mask_values,
        }
        append_jsonl(raw_path, record)
        completed.add(variant_id)
        generated += 1
        print(
            f"saved {variant_id} tokens={len(token_ids)} "
            f"augs={len(inference_aug_positions)} reached={intervention_reached} "
            f"latency={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
