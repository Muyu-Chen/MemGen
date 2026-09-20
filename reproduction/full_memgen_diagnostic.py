"""
Full MemGen Diagnostic: Base / Weaver-only / Full MemGen (Trigger+Weaver)

Runs 5 GSM8K test questions in three modes:
  1. Base model (Qwen2.5-1.5B-Instruct, no augmentation)
  2. Weaver-only (trigger_active=False, always augment)
  3. Full MemGen (trigger_active=True, learned trigger)

Saves JSONL trace with trigger info for each question.
"""
import sys
import os
import copy
import time
import json
import gc
import re

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from safetensors.torch import load_file as safe_load_file

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel, _remap_lora_adapter_key

BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/Qwen2.5-1.5B-Instruct")
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT,
    "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"
)
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "reproduction", "phase1_results")

NUM_QUESTIONS = 5
MAX_NEW_TOKENS = 1024

OFFICIAL_PROMPT = r"Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."

WEAVER_LORA_CONFIG = {
    "r": 16, "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
}
TRIGGER_LORA_CONFIG = {
    "r": 16, "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
}


# ── GSM8K loading ──────────────────────────────────────────────────────────

def load_gsm8k_test(n=NUM_QUESTIONS):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    questions = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        answer_raw = ex["answer"].strip()
        parts = answer_raw.split("\n####")
        rationale = parts[0].strip()
        clean_answer = parts[-1].strip()
        questions.append({
            "id": f"gsm8k_test_{i}",
            "question": ex["question"].strip(),
            "ground_truth": clean_answer,
            "full_solution": answer_raw,
        })
    return questions


# ── Prompt building (official template) ────────────────────────────────────

def build_prompt_messages(question: str) -> list:
    content = OFFICIAL_PROMPT + "\nQuestion: " + question
    return [{"role": "user", "content": content}]


def build_prompt_text(tokenizer, question: str) -> str:
    messages = build_prompt_messages(question)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Answer extraction (official math_utils.py logic) ───────────────────────

def first_boxed_only_string(string):
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[1].split("$")[0]
    idx = string.find("\\boxed")
    if idx < 0:
        idx = string.find("\\fbox")
        if idx < 0:
            return None
    i = idx
    num_left = 0
    while i < len(string):
        if string[i] == "{":
            num_left += 1
        if string[i] == "}":
            num_left -= 1
            if num_left == 0:
                return string[idx: i + 1]
        i += 1
    return None


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        return s[len(left):]
    left = "\\boxed{"
    return s[len(left): -1]


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(",", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = string.replace(" ", "")
    return string


def extract_answer(completion: str):
    boxed = first_boxed_only_string(completion)
    if boxed is None:
        return None
    return remove_boxed(boxed)


def is_correct(completion: str, ground_truth: str) -> bool:
    answer = extract_answer(completion)
    if answer is None:
        return False
    return strip_string(answer) == strip_string(ground_truth)


# ── Model loading ──────────────────────────────────────────────────────────

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    return tokenizer


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    model.eval()
    return model


def load_memgen_model(tokenizer, trigger_active=False):
    config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1, max_inference_aug_num=3,
        prompt_latents_len=8, inference_latents_len=8,
        weaver_lora_config=WEAVER_LORA_CONFIG,
        trigger_active=trigger_active,
        trigger_lora_config=TRIGGER_LORA_CONFIG,
    )

    base_for_clone = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    reasoner = copy.deepcopy(base_for_clone)
    weaver = copy.deepcopy(base_for_clone)
    trigger = copy.deepcopy(base_for_clone)
    del base_for_clone
    gc.collect()

    model = MemGenModel(
        config=config, base_tokenizer=tokenizer,
        reasoner_base_model=reasoner, weaver_base_model=weaver, trigger_base_model=trigger,
    )

    # Load projections
    proj_state = torch.load(os.path.join(CHECKPOINT_PATH, "projs.bin"), map_location="cpu", weights_only=True)
    model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])

    # Load weaver params
    weaver_state = torch.load(os.path.join(CHECKPOINT_PATH, "weaver.bin"), map_location="cpu", weights_only=True)
    model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
    model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
    model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
    model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
    model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])

    # Load trigger output_layer
    trigger_state = torch.load(os.path.join(CHECKPOINT_PATH, "trigger.bin"), map_location="cpu", weights_only=True)
    model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])

    # Load weaver LoRA adapter
    weaver_ckpt_sd = safe_load_file(
        os.path.join(CHECKPOINT_PATH, "weaver", "weaver", "adapter_model.safetensors"), device="cpu"
    )
    weaver_model_sd = model.weaver.model.state_dict()
    weaver_final = {
        _remap_lora_adapter_key(k, "weaver"): v
        for k, v in weaver_ckpt_sd.items()
        if _remap_lora_adapter_key(k, "weaver") in weaver_model_sd
    }
    model.weaver.model.load_state_dict(weaver_final, strict=False)
    model.weaver.model.set_adapter("weaver")

    # Load trigger LoRA adapter
    trigger_ckpt_sd = safe_load_file(
        os.path.join(CHECKPOINT_PATH, "trigger", "trigger", "adapter_model.safetensors"), device="cpu"
    )
    trigger_model_sd = model.trigger.model.state_dict()
    trigger_final = {
        _remap_lora_adapter_key(k, "trigger"): v
        for k, v in trigger_ckpt_sd.items()
        if _remap_lora_adapter_key(k, "trigger") in trigger_model_sd
    }
    model.trigger.model.load_state_dict(trigger_final, strict=False)
    model.trigger.model.set_adapter("trigger")

    model.eval()
    return model


# ── Generation ─────────────────────────────────────────────────────────────

def generate_base(model, tokenizer, question):
    prompt = build_prompt_text(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=gen_config,
        )
    latency = time.time() - t0

    new_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    token_count = new_ids.shape[0]
    return text, latency, token_count


def generate_memgen(model, tokenizer, question):
    prompt = build_prompt_text(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_config.weaver_do_sample = False
    gen_config.trigger_do_sample = False

    t0 = time.time()
    with torch.no_grad():
        output_ids, aug_mask = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=gen_config,
            return_augmentation_mask=True,
        )
    latency = time.time() - t0

    prompt_len = input_ids.shape[1]
    new_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    token_count = new_ids.shape[0]

    # Parse augmentation mask
    aug_mask_seq = aug_mask[0, :token_count]  # (gen_len,)
    trigger_positions = []
    inference_trigger_positions = []
    for pos_idx in range(token_count):
        val = aug_mask_seq[pos_idx].item()
        if val == 1:
            trigger_positions.append(pos_idx)
            if pos_idx > 0:  # position 0 is prompt augmentation, not inference
                inference_trigger_positions.append(pos_idx)

    prompt_augmented = (aug_mask_seq[0].item() == 1) if token_count > 0 else False
    trigger_count = len(inference_trigger_positions)

    return text, latency, token_count, {
        "prompt_augmented": prompt_augmented,
        "trigger_count": trigger_count,
        "trigger_positions": inference_trigger_positions,
        "all_aug_positions": trigger_positions,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_path = os.path.join(OUTPUT_DIR, "diagnostic_traces.jsonl")

    print("=" * 80)
    print("Full MemGen Diagnostic: Base / Weaver-only / Full MemGen")
    print(f"Questions: {NUM_QUESTIONS} | Max tokens: {MAX_NEW_TOKENS}")
    print(f"Output: {results_path}")
    print("=" * 80)

    # Load GSM8K
    print("\nLoading GSM8K test set...")
    questions = load_gsm8k_test(NUM_QUESTIONS)
    print(f"Loaded {len(questions)} questions")
    for q in questions:
        print(f"  [{q['id']}] answer={q['ground_truth']} | {q['question'][:60]}...")

    tokenizer = load_tokenizer()
    all_results = []

    # ── Phase 1: Base Model ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Phase 1/3: Base Model")
    print("=" * 80)

    base_model = load_base_model()

    for q in questions:
        print(f"\n  [{q['id']}] {q['question'][:60]}...")
        text, latency, tokens = generate_base(base_model, tokenizer, q["question"])
        answer = extract_answer(text)
        correct = is_correct(text, q["ground_truth"])
        print(f"    Base: answer={answer} correct={correct} latency={latency:.1f}s tokens={tokens}")

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "mode": "base",
            "output": text,
            "extracted_answer": answer,
            "correct": correct,
            "latency_sec": round(latency, 1),
            "token_count": tokens,
            "trigger_active": None,
            "prompt_augmented": None,
            "trigger_count": None,
            "trigger_positions": None,
        })

    del base_model
    gc.collect()
    print("\n  Base model unloaded.")

    # ── Phase 2 & 3: MemGen (Weaver-only + Full MemGen) ───────────────────
    print("\n" + "=" * 80)
    print("Phase 2/3: MemGen Weaver-only (trigger_active=False)")
    print("=" * 80)

    print("\n  Loading MemGen model...")
    memgen_model = load_memgen_model(tokenizer, trigger_active=False)
    print("  MemGen loaded.")

    # Weaver-only
    for q in questions:
        print(f"\n  [{q['id']}] {q['question'][:60]}...")
        text, latency, tokens, trigger_info = generate_memgen(memgen_model, tokenizer, q["question"])
        answer = extract_answer(text)
        correct = is_correct(text, q["ground_truth"])
        print(f"    Weaver-only: answer={answer} correct={correct} "
              f"trigger_count={trigger_info['trigger_count']} latency={latency:.1f}s tokens={tokens}")

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "mode": "weaver_only",
            "output": text,
            "extracted_answer": answer,
            "correct": correct,
            "latency_sec": round(latency, 1),
            "token_count": tokens,
            "trigger_active": False,
            "prompt_augmented": trigger_info["prompt_augmented"],
            "trigger_count": trigger_info["trigger_count"],
            "trigger_positions": trigger_info["trigger_positions"],
        })

    # ── Phase 3: Full MemGen (trigger_active=True) ─────────────────────────
    print("\n" + "=" * 80)
    print("Phase 3/3: Full MemGen (trigger_active=True)")
    print("=" * 80)

    memgen_model.trigger.active = True
    print(f"  Trigger active set to: {memgen_model.trigger.active}")

    for q in questions:
        print(f"\n  [{q['id']}] {q['question'][:60]}...")
        text, latency, tokens, trigger_info = generate_memgen(memgen_model, tokenizer, q["question"])
        answer = extract_answer(text)
        correct = is_correct(text, q["ground_truth"])
        print(f"    Full MemGen: answer={answer} correct={correct} "
              f"trigger_count={trigger_info['trigger_count']} latency={latency:.1f}s tokens={tokens}")

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "mode": "full_memgen",
            "output": text,
            "extracted_answer": answer,
            "correct": correct,
            "latency_sec": round(latency, 1),
            "token_count": tokens,
            "trigger_active": True,
            "prompt_augmented": trigger_info["prompt_augmented"],
            "trigger_count": trigger_info["trigger_count"],
            "trigger_positions": trigger_info["trigger_positions"],
        })

    # ── Save results ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Saving results...")
    with open(results_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved to: {results_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    for mode in ["base", "weaver_only", "full_memgen"]:
        mode_results = [r for r in all_results if r["mode"] == mode]
        correct_count = sum(1 for r in mode_results if r["correct"])
        total = len(mode_results)
        print(f"  {mode:15s}: {correct_count}/{total} correct")

    # Paired comparison: Base vs Full MemGen
    print("\n  Paired comparison (Base vs Full MemGen):")
    base_results = {r["id"]: r for r in all_results if r["mode"] == "base"}
    memgen_results = {r["id"]: r for r in all_results if r["mode"] == "full_memgen"}

    preserved = recovered = regressed = both_wrong = 0
    for qid in base_results:
        b = base_results[qid]["correct"]
        m = memgen_results[qid]["correct"]
        if b and m:
            preserved += 1
        elif not b and m:
            recovered += 1
        elif b and not m:
            regressed += 1
        else:
            both_wrong += 1

    total = len(base_results)
    base_correct = sum(1 for r in base_results.values() if r["correct"])
    memgen_correct = sum(1 for r in memgen_results.values() if r["correct"])

    print(f"    Base correct & MemGen correct (preserved):  {preserved}")
    print(f"    Base wrong   & MemGen correct (recovered):  {recovered}")
    print(f"    Base correct & MemGen wrong   (regressed):  {regressed}")
    print(f"    Base wrong   & MemGen wrong   (both_wrong): {both_wrong}")
    print()
    print(f"    Base accuracy:      {base_correct}/{total} = {base_correct/total:.1%}")
    print(f"    Full MemGen accuracy: {memgen_correct}/{total} = {memgen_correct/total:.1%}")
    if base_correct < total:
        print(f"    Recovery rate:      {recovered}/{total - base_correct} = {recovered/(total - base_correct):.1%}")
    if base_correct > 0:
        print(f"    Regression rate:    {regressed}/{base_correct} = {regressed/base_correct:.1%}")
        print(f"    Preservation rate:  {preserved}/{base_correct} = {preserved/base_correct:.1%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
