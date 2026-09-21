"""Exhaust dynamic invoke/skip policies at sentence-level reasoning boundaries."""

from __future__ import annotations

import argparse
import itertools
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
    portable_path,
    read_jsonl,
    resolve_torch_dtype,
    sha256_file,
    utc_now,
    write_json,
)
from run_memgen import load_model


CONDITION = "Z_memgen_sentence_sequential_oracle"
DEFAULT_PLAN = HARD_BENCH_ROOT / "plans" / "sentence_sequential_oracle_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan-file", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition-label", default=CONDITION)
    parser.add_argument("--decoded-boundaries", action="store_true")
    parser.add_argument("--time-budget-minutes", type=float, default=120.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def enumerate_policies(
    depths: list[int], policy_prefixes: list[str] | None = None
) -> list[tuple[int, tuple[int, ...]]]:
    policies: list[tuple[int, tuple[int, ...]]] = []
    for depth in depths:
        for bits in itertools.product((0, 1), repeat=depth):
            policy = "".join(str(bit) for bit in bits)
            if policy_prefixes and not any(policy.startswith(prefix) for prefix in policy_prefixes):
                continue
            policies.append((depth, bits))
    return policies


def classify_sentence_boundary(generated_text: str) -> tuple[bool, str]:
    """Implement the paper/code delimiter gate, excluding obvious numeric internals."""
    if not generated_text:
        return False, "empty_generation"
    without_horizontal_space = generated_text.rstrip(" \t\r")
    if without_horizontal_space.count("<<") > without_horizontal_space.count(">>"):
        return False, "inside_calculator_span"
    if without_horizontal_space.endswith("\n"):
        before_newline = without_horizontal_space[:-1]
        final_line = before_newline.rsplit("\n", 1)[-1].strip()
        if not final_line:
            return False, "blank_line"
        if final_line[-1] in "+-*/=<>$":
            return False, "unfinished_arithmetic_before_newline"
        return True, "paper_delimiter_newline"
    if not without_horizontal_space.endswith((",", ".")):
        return False, "not_sentence_terminal"
    delimiter = without_horizontal_space[-1]
    preceding = without_horizontal_space[-2] if len(without_horizontal_space) >= 2 else ""
    if preceding.isdigit():
        return False, "numeric_comma_or_period"
    if preceding in "+-*/=<>$":
        return False, "unfinished_arithmetic_before_punctuation"
    if delimiter == ",":
        return True, "paper_delimiter_semantic_comma"
    return True, "paper_delimiter_semantic_period"


def install_sentence_policy_gate(
    model,
    tokenizer,
    prompt_token_count: int,
    policy_bits: tuple[int, ...],
    decoded_boundaries: bool = False,
):
    """Apply policy bits to sentence boundaries encountered on the live trajectory."""
    events: list[dict] = []
    original = model._should_augment
    raw_candidate_ordinal = 0
    boundary_ordinal = 0

    def sentence_policy_should_augment(
        input_ids,
        sentence_augment_count,
        do_sample,
        temperature,
        is_prompt=False,
    ):
        nonlocal raw_candidate_ordinal, boundary_ordinal
        del do_sample, temperature
        batch_size = input_ids.size(0)
        if batch_size != 1:
            raise ValueError("sentence oracle runner requires batch size one")
        if is_prompt:
            decision = torch.ones((1,), dtype=torch.long, device=input_ids.device)
            events.append(
                {
                    "is_prompt": True,
                    "sequence_length": int(input_ids.shape[1]),
                    "decision": 1,
                }
            )
            return decision

        augment_vector = torch.full(
            (1,), -100, dtype=torch.long, device=input_ids.device
        )
        generated_ids = input_ids[0, prompt_token_count:].detach().cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if decoded_boundaries:
            last_token_text = tokenizer.decode(
                input_ids[0, -1:].detach().cpu().tolist(), skip_special_tokens=True
            )
            candidate_detected = bool(
                last_token_text.rstrip(" \t\r").endswith((",", ".", "\n"))
            )
            detection_reason = "decoded_last_token_suffix"
        else:
            raw_candidates = model._check_ends_with_delimiter(
                input_ids, model.tokenizer, model.delimiters
            ).squeeze(1)
            raw_candidates &= sentence_augment_count < model.config.max_inference_aug_num
            candidate_detected = bool(raw_candidates[0].item())
            last_token_text = tokenizer.decode(
                input_ids[0, -1:].detach().cpu().tolist(), skip_special_tokens=True
            )
            detection_reason = "released_token_id_equality"
        if not candidate_detected:
            return augment_vector

        raw_candidate_ordinal += 1
        eligible, boundary_reason = classify_sentence_boundary(generated_text)
        event = {
            "is_prompt": False,
            "raw_candidate_ordinal": raw_candidate_ordinal,
            "sequence_length": int(input_ids.shape[1]),
            "eligible_sentence_boundary": eligible,
            "boundary_reason": boundary_reason,
            "candidate_detection": detection_reason,
            "last_token_text": last_token_text,
            "generated_prefix": generated_text,
        }
        if eligible:
            boundary_ordinal += 1
            event["sentence_boundary_ordinal"] = boundary_ordinal
            if boundary_ordinal <= len(policy_bits):
                decision = int(policy_bits[boundary_ordinal - 1])
                event["policy_bit_index"] = boundary_ordinal
                event["policy_bit"] = decision
            else:
                decision = 0
                event["policy_exhausted"] = True
            event["decision"] = decision
            augment_vector[0] = decision
        events.append(event)
        return augment_vector

    model._should_augment = sentence_policy_should_augment
    return original, events


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    plan_path = args.plan_file.resolve()
    run_dir = args.run_dir.resolve()
    condition = args.condition_label
    raw_path = run_dir / "raw" / f"{condition}.raw.jsonl"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    depths = [int(depth) for depth in plan["decision_depths"]]
    if not depths or any(depth < 1 or depth > 5 for depth in depths):
        raise ValueError(f"decision depths must be between 1 and 5, got {depths}")
    policy_prefixes = plan.get("policy_prefixes")
    if policy_prefixes is not None:
        if not policy_prefixes or any(
            not prefix or any(bit not in "01" for bit in prefix)
            for prefix in policy_prefixes
        ):
            raise ValueError(f"invalid policy_prefixes: {policy_prefixes}")
        if any(len(prefix) > min(depths) for prefix in policy_prefixes):
            raise ValueError("policy prefix is longer than a requested decision depth")
    policies = enumerate_policies(depths, policy_prefixes)
    plan_sample_ids = plan.get("sample_ids") or [plan["sample_id"]]
    if len(policies) * len(plan_sample_ids) != int(plan["expected_variants"]):
        raise ValueError("plan expected_variants does not match policy enumeration")

    samples = {sample["sample_id"]: sample for sample in load_manifest(manifest_path)}
    missing_sample_ids = set(plan_sample_ids) - set(samples)
    if missing_sample_ids:
        raise ValueError(f"samples not found in manifest: {sorted(missing_sample_ids)}")
    meta = json.loads(meta_path_for(manifest_path).read_text(encoding="utf-8"))

    torch.set_num_threads(args.threads)
    dtype = resolve_torch_dtype(args.dtype)
    tokenizer = load_tokenizer()
    print(f"Loading MemGen dtype={args.dtype} threads={args.threads}", flush=True)
    load_started = time.perf_counter()
    model, config = load_model(tokenizer, dtype)
    native_max_inference_aug_num = int(config.max_inference_aug_num)
    if native_max_inference_aug_num != 3:
        raise RuntimeError("sentence oracle expects the frozen checkpoint limit of three")
    print(f"MemGen loaded in {time.perf_counter() - load_started:.1f}s", flush=True)

    write_json(
        run_dir / f"{condition}.run_config.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "runner": "run_sentence_sequential_oracle.py",
            "condition": condition,
            "manifest": portable_path(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "plan_file": portable_path(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "sample_ids": plan_sample_ids,
            "planned_variants": len(policies) * len(plan_sample_ids),
            "decision_depths": depths,
            "policy_prefixes": policy_prefixes,
            "base_model": portable_path(BASE_MODEL_PATH),
            "checkpoint": portable_path(CHECKPOINT_PATH),
            "dtype": args.dtype,
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "generation": "greedy",
            "trigger_active": bool(config.trigger_active),
            "max_prompt_aug_num": int(config.max_prompt_aug_num),
            "checkpoint_max_inference_aug_num": native_max_inference_aug_num,
            "diagnostic_max_inference_aug_num": max(depths),
            "gate_policy": "dynamic_sentence_boundary_binary_sequence",
            "candidate_detection": (
                "decoded_last_token_suffix"
                if args.decoded_boundaries
                else "released_token_id_equality"
            ),
            "paper_boundary_definition": plan["paper_boundary_definition"],
            "boundary_rule": plan["inference_boundary_rule"],
        },
    )

    completed = {
        record["oracle_variant_id"]
        for record in read_jsonl(raw_path)
        if record.get("oracle_variant_id")
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

    run_started = time.perf_counter()
    generated = 0
    budget_reached = False
    for sample_id in plan_sample_ids:
        sample = samples[sample_id]
        messages = build_messages(sample, "N_memgen_no_inference", meta)
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        for depth, bits in policies:
            policy_string = "".join(str(bit) for bit in bits)
            variant_id = f"{sample_id}__d{depth}_p{policy_string}"
            if variant_id in completed:
                print(f"skip {variant_id}", flush=True)
                continue
            if generated > 0 and time.perf_counter() - run_started >= args.time_budget_minutes * 60:
                print("Time budget reached before starting the next variant", flush=True)
                budget_reached = True
                break

            model.config.max_inference_aug_num = depth
            original_should_augment, gate_events = install_sentence_policy_gate(
                model, tokenizer, prompt_len, bits, args.decoded_boundaries
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
                model.config.max_inference_aug_num = native_max_inference_aug_num
            elapsed = time.perf_counter() - started

            new_ids = output_ids[0, prompt_len:].cpu()
            token_ids = new_ids.tolist()
            eos_emitted = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
            mask_values = augmentation_mask[0, : len(token_ids)].cpu().tolist()
            all_aug_positions = [index for index, value in enumerate(mask_values) if value == 1]
            inference_aug_positions = [position for position in all_aug_positions if position > 0]
            policy_events = [
                event
                for event in gate_events
                if not event.get("is_prompt") and event.get("policy_bit_index") is not None
            ]
            observed_bits = tuple(int(event["policy_bit"]) for event in policy_events)
            intervention_reached = observed_bits == bits

            record = {
                "schema_version": 1,
                "generated_at": utc_now(),
                "condition": condition,
                "oracle_variant_id": variant_id,
                "decision_depth": depth,
                "policy_bits": list(bits),
                "policy_string": policy_string,
                "policy_invocation_count": sum(bits),
                "checkpoint_native_policy": sum(bits) <= native_max_inference_aug_num,
                "intervention_reached": intervention_reached,
                "observed_policy_bits": list(observed_bits),
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
                "checkpoint_max_inference_aug_num": native_max_inference_aug_num,
                "effective_max_inference_aug_num": depth,
                "gate_policy": "dynamic_sentence_boundary_binary_sequence",
                "candidate_detection": (
                    "decoded_last_token_suffix"
                    if args.decoded_boundaries
                    else "released_token_id_equality"
                ),
                "paper_boundary_definition": plan["paper_boundary_definition"],
                "boundary_rule": plan["inference_boundary_rule"],
                "gate_events": gate_events,
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
                f"saved {sample_id} d={depth} policy={policy_string} "
                f"tokens={len(token_ids)} augs={len(inference_aug_positions)} "
                f"reached={intervention_reached} latency={elapsed:.1f}s",
                flush=True,
            )
        if budget_reached:
            break


if __name__ == "__main__":
    main()
