"""Generate raw MemGen condition outputs. This module deliberately does no scoring."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from common import (
    BASE_MODEL_PATH,
    CHECKPOINT_PATH,
    DEFAULT_MANIFEST,
    append_jsonl,
    build_messages,
    completed_sample_ids,
    load_manifest,
    load_tokenizer,
    meta_path_for,
    portable_path,
    resolve_torch_dtype,
    sha256_file,
    utc_now,
    write_json,
)
from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel


CONDITION_LABELS = {
    "official": "D_memgen_official",
    "random50": "R_memgen_random50_inference",
    "no_inference": "N_memgen_no_inference",
    "scheduled_single": "H_memgen_human_single",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITION_LABELS), default="official")
    parser.add_argument("--random-probability", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=20260920)
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--sample-ids", nargs="*")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--time-budget-minutes", type=float, default=110.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def install_random_inference_gate(model, sample_seed: int, probability: float):
    """Keep prompt augmentation on; randomize only inference delimiter candidates."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("random probability must be in [0, 1]")
    rng = random.Random(sample_seed)
    events: list[dict] = []
    original = model._should_augment

    def random_should_augment(
        input_ids,
        sentence_augment_count,
        do_sample,
        temperature,
        is_prompt=False,
    ):
        del do_sample, temperature
        batch_size = input_ids.size(0)
        if is_prompt:
            decision = torch.ones(
                (batch_size,), dtype=torch.long, device=input_ids.device
            )
            events.append(
                {
                    "is_prompt": True,
                    "sequence_length": int(input_ids.shape[1]),
                    "decisions": [1] * batch_size,
                }
            )
            return decision

        aug_vector = torch.full(
            (batch_size,), -100, dtype=torch.long, device=input_ids.device
        )
        candidates = model._check_ends_with_delimiter(
            input_ids, model.tokenizer, model.delimiters
        ).squeeze(1)
        candidates &= sentence_augment_count < model.config.max_inference_aug_num
        candidate_indices = candidates.nonzero(as_tuple=True)[0]
        if candidate_indices.numel() > 0:
            decisions = []
            for batch_index in candidate_indices.tolist():
                decision = 1 if rng.random() < probability else 0
                aug_vector[batch_index] = decision
                decisions.append(
                    {"batch_index": int(batch_index), "decision": decision}
                )
            events.append(
                {
                    "is_prompt": False,
                    "sequence_length": int(input_ids.shape[1]),
                    "decisions": decisions,
                }
            )
        return aug_vector

    model._should_augment = random_should_augment
    return original, events


def install_candidate_subset_gate(model, selected_candidate_ordinals: set[int]):
    """Install a prompt-on gate that augments exactly the selected candidate ordinals."""
    if any(not isinstance(ordinal, int) or ordinal < 1 for ordinal in selected_candidate_ordinals):
        raise ValueError("selected candidate ordinals must be positive integers")
    if len(selected_candidate_ordinals) > model.config.max_inference_aug_num:
        raise ValueError(
            "selected candidate count exceeds checkpoint max_inference_aug_num: "
            f"{len(selected_candidate_ordinals)} > {model.config.max_inference_aug_num}"
        )
    events: list[dict] = []
    original = model._should_augment
    candidate_ordinal = 0

    def deterministic_should_augment(
        input_ids,
        sentence_augment_count,
        do_sample,
        temperature,
        is_prompt=False,
    ):
        nonlocal candidate_ordinal
        del do_sample, temperature
        batch_size = input_ids.size(0)
        if is_prompt:
            decision = torch.ones(
                (batch_size,), dtype=torch.long, device=input_ids.device
            )
            events.append(
                {
                    "is_prompt": True,
                    "sequence_length": int(input_ids.shape[1]),
                    "decisions": [1] * batch_size,
                }
            )
            return decision

        aug_vector = torch.full(
            (batch_size,), -100, dtype=torch.long, device=input_ids.device
        )
        candidates = model._check_ends_with_delimiter(
            input_ids, model.tokenizer, model.delimiters
        ).squeeze(1)
        candidates &= sentence_augment_count < model.config.max_inference_aug_num
        candidate_indices = candidates.nonzero(as_tuple=True)[0]
        for batch_index in candidate_indices.tolist():
            candidate_ordinal += 1
            decision = int(candidate_ordinal in selected_candidate_ordinals)
            aug_vector[batch_index] = decision
            visible_prefix = model.tokenizer.decode(
                input_ids[batch_index].detach().cpu().tolist(),
                skip_special_tokens=True,
            )
            events.append(
                {
                    "is_prompt": False,
                    "candidate_ordinal": candidate_ordinal,
                    "sequence_length": int(input_ids.shape[1]),
                    "decision": decision,
                    "visible_prefix": visible_prefix,
                }
            )
        return aug_vector

    model._should_augment = deterministic_should_augment
    return original, events


def install_deterministic_inference_gate(
    model, mode: str, selected_candidate_ordinal: int | None = None
):
    """Install the legacy no-inference or scheduled-single deterministic gate."""
    if mode not in {"no_inference", "scheduled_single"}:
        raise ValueError(f"unsupported deterministic gate mode: {mode}")
    selected: set[int] = set()
    if mode == "scheduled_single" and selected_candidate_ordinal is not None:
        selected.add(selected_candidate_ordinal)
    return install_candidate_subset_gate(model, selected)


def load_model(tokenizer, dtype):
    config = MemGenConfig.from_pretrained(CHECKPOINT_PATH)
    print(
        "checkpoint config: "
        f"trigger_active={config.trigger_active} "
        f"max_prompt_aug_num={config.max_prompt_aug_num} "
        f"max_inference_aug_num={config.max_inference_aug_num}"
    )
    if config.trigger_active is not False or config.max_inference_aug_num != 3:
        raise RuntimeError("Checkpoint configuration is not the frozen official D condition")

    def load_base(role: str):
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=dtype, attn_implementation="eager"
        )
        print(f"loaded {role} in {time.perf_counter() - started:.1f}s")
        return model

    reasoner = load_base("reasoner")
    gc.collect()
    weaver = load_base("weaver")
    gc.collect()
    trigger = load_base("trigger")
    gc.collect()
    model = MemGenModel.from_pretrained(
        CHECKPOINT_PATH,
        config=config,
        base_tokenizer=tokenizer,
        reasoner_base_model=reasoner,
        weaver_base_model=weaver,
        trigger_base_model=trigger,
    )
    model = model.to(dtype)
    model.eval()
    return model, config


def main() -> None:
    args = parse_args()
    condition = CONDITION_LABELS[args.condition]
    manifest_path = args.manifest.resolve()
    run_dir = args.run_dir.resolve()
    raw_path = run_dir / "raw" / f"{condition}.raw.jsonl"
    samples = load_manifest(manifest_path)
    if args.sample_ids:
        requested = set(args.sample_ids)
        samples = [sample for sample in samples if sample["sample_id"] in requested]
        missing = requested - {sample["sample_id"] for sample in samples}
        if missing:
            raise ValueError(f"sample ids not found in manifest: {sorted(missing)}")
    samples = samples[: args.max_samples]
    meta = json.loads(meta_path_for(manifest_path).read_text(encoding="utf-8"))
    policy = None
    if args.condition == "scheduled_single":
        if args.policy_file is None:
            raise ValueError("--policy-file is required for scheduled_single")
        policy = json.loads(args.policy_file.resolve().read_text(encoding="utf-8"))

    torch.set_num_threads(args.threads)
    dtype = resolve_torch_dtype(args.dtype)
    tokenizer = load_tokenizer()
    print(f"Loading MemGen dtype={args.dtype} threads={args.threads}")
    load_started = time.perf_counter()
    model, config = load_model(tokenizer, dtype)
    print(f"MemGen loaded in {time.perf_counter() - load_started:.1f}s")

    write_json(
        run_dir / f"{condition}.run_config.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "runner": "run_memgen.py",
            "condition": condition,
            "manifest": portable_path(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "base_model": portable_path(BASE_MODEL_PATH),
            "checkpoint": portable_path(CHECKPOINT_PATH),
            "dtype": args.dtype,
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "generation": "greedy",
            "trigger_active": bool(config.trigger_active),
            "max_prompt_aug_num": int(config.max_prompt_aug_num),
            "max_inference_aug_num": int(config.max_inference_aug_num),
            "gate_policy": args.condition,
            "random_probability": (
                args.random_probability if args.condition == "random50" else None
            ),
            "random_seed": args.random_seed if args.condition == "random50" else None,
            "policy_file": (
                portable_path(args.policy_file) if args.policy_file is not None else None
            ),
            "policy_sha256": (
                sha256_file(args.policy_file.resolve())
                if args.policy_file is not None
                else None
            ),
        },
    )

    completed = completed_sample_ids(raw_path, condition)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generation_config.weaver_do_sample = False
    generation_config.trigger_do_sample = False

    run_started = time.perf_counter()
    generated = 0
    for sample in samples:
        if sample["sample_id"] in completed:
            print(f"skip {condition} {sample['sample_id']}")
            continue
        if generated > 0 and time.perf_counter() - run_started >= args.time_budget_minutes * 60:
            print("Time budget reached before starting the next sample")
            break

        messages = build_messages(sample, condition, meta)
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        random_gate_seed = None
        random_gate_events = []
        original_should_augment = None
        selected_candidate_ordinal = None
        human_rationale = None
        if args.condition == "random50":
            random_gate_seed = args.random_seed + int(sample["source_index"])
            original_should_augment, random_gate_events = install_random_inference_gate(
                model, random_gate_seed, args.random_probability
            )
        elif args.condition == "no_inference":
            original_should_augment, random_gate_events = install_deterministic_inference_gate(
                model, "no_inference"
            )
        elif args.condition == "scheduled_single":
            selection = policy["selections"].get(sample["sample_id"])
            if selection is None:
                raise ValueError(f"policy has no selection for {sample['sample_id']}")
            selected_candidate_ordinal = selection.get("candidate_ordinal")
            human_rationale = selection.get("rationale")
            original_should_augment, random_gate_events = install_deterministic_inference_gate(
                model, "scheduled_single", selected_candidate_ordinal
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
            if original_should_augment is not None:
                model._should_augment = original_should_augment
        elapsed = time.perf_counter() - started
        new_ids = output_ids[0, prompt_len:].cpu()
        token_ids = new_ids.tolist()
        eos_emitted = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
        mask_values = augmentation_mask[0, : len(token_ids)].cpu().tolist()
        all_aug_positions = [index for index, value in enumerate(mask_values) if value == 1]
        inference_aug_positions = [position for position in all_aug_positions if position > 0]
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
            "model_path": portable_path(BASE_MODEL_PATH),
            "checkpoint_path": portable_path(CHECKPOINT_PATH),
            "dtype": args.dtype,
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "checkpoint_trigger_active": False,
            "gate_policy": args.condition,
            "random_probability": (
                args.random_probability if args.condition == "random50" else None
            ),
            "random_gate_seed": random_gate_seed,
            "random_gate_events": random_gate_events,
            "selected_candidate_ordinal": selected_candidate_ordinal,
            "human_selection_rationale": human_rationale,
            "prompt_augmented": bool(mask_values and mask_values[0] == 1),
            "inference_augmentation_count": len(inference_aug_positions),
            "inference_augmentation_positions": inference_aug_positions,
            "all_augmentation_positions": all_aug_positions,
            "augmentation_mask": mask_values,
        }
        append_jsonl(raw_path, record)
        completed.add(sample["sample_id"])
        generated += 1
        print(
            f"saved {condition} {sample['sample_id']} tokens={len(token_ids)} "
            f"augs={len(inference_aug_positions)} latency={elapsed:.1f}s raw={raw_path}"
        )


if __name__ == "__main__":
    main()
