"""
Three-way controlled probe: always_0 / trained / always_1
Tests whether performance comes from Weaver or Trigger gating.
"""
import sys
import os
import copy
import json
import gc
import re
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
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

MAX_NEW_TOKENS = 512
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

# 20 very simple math questions
SIMPLE_QUESTIONS = [
    {"id": "easy_01", "question": "What is 2+3?", "answer": "5"},
    {"id": "easy_02", "question": "What is 10-4?", "answer": "6"},
    {"id": "easy_03", "question": "What is 3*5?", "answer": "15"},
    {"id": "easy_04", "question": "What is 12/3?", "answer": "4"},
    {"id": "easy_05", "question": "John has 5 apples and buys 2 more. How many apples does he have now?", "answer": "7"},
    {"id": "easy_06", "question": "There are 20 students. 8 are boys. How many are girls?", "answer": "12"},
    {"id": "easy_07", "question": "What is 7+8?", "answer": "15"},
    {"id": "easy_08", "question": "What is 15-6?", "answer": "9"},
    {"id": "easy_09", "question": "What is 4*6?", "answer": "24"},
    {"id": "easy_10", "question": "What is 20/4?", "answer": "5"},
    {"id": "easy_11", "question": "Mary has 10 candies. She gives 3 to her friend. How many does she have left?", "answer": "7"},
    {"id": "easy_12", "question": "A box has 6 red balls and 4 blue balls. How many balls in total?", "answer": "10"},
    {"id": "easy_13", "question": "What is 9+11?", "answer": "20"},
    {"id": "easy_14", "question": "What is 18-9?", "answer": "9"},
    {"id": "easy_15", "question": "What is 5*5?", "answer": "25"},
    {"id": "easy_16", "question": "What is 16/2?", "answer": "8"},
    {"id": "easy_17", "question": "Tom has 8 toys. He gets 4 more for his birthday. How many toys does he have now?", "answer": "12"},
    {"id": "easy_18", "question": "There are 15 birds on a tree. 5 fly away. How many are left?", "answer": "10"},
    {"id": "easy_19", "question": "What is 6+7?", "answer": "13"},
    {"id": "easy_20", "question": "What is 14-7?", "answer": "7"},
]


def build_prompt_text(tokenizer, question: str) -> str:
    content = OFFICIAL_PROMPT + "\nQuestion: " + question
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    return tokenizer


def load_memgen_model(tokenizer):
    config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1, max_inference_aug_num=3,
        prompt_latents_len=8, inference_latents_len=8,
        weaver_lora_config=WEAVER_LORA_CONFIG,
        trigger_active=True,
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


# Answer extraction (from official math_utils.py)
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


def generate_with_mode(model, tokenizer, question, mode):
    """
    Generate with specified trigger mode.
    mode: 'always_0', 'trained', 'always_1'
    """
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

    # Setup trigger mode
    trigger_log = []
    original_forward = model.trigger.forward

    if mode == "always_0":
        # Force decision=0 (never augment)
        def forced_forward(**kwargs):
            logits = original_forward(**kwargs)
            # Force class 0
            logits[..., 0] = 10.0
            logits[..., 1] = -10.0
            return logits
        model.trigger.forward = forced_forward
        model.trigger.active = True  # Must be active for forward to be called
    elif mode == "trained":
        # Use real trigger
        model.trigger.active = True
        def logged_forward(**kwargs):
            logits = original_forward(**kwargs)
            last_logits = logits[0, -1, :]
            softmax = F.softmax(last_logits, dim=0)
            decision = torch.argmax(last_logits).item()
            trigger_log.append({
                "logits": last_logits.tolist(),
                "softmax": softmax.tolist(),
                "decision": decision,
            })
            return logits
        model.trigger.forward = logged_forward
    elif mode == "always_1":
        # Force decision=1 (always augment) - equivalent to active=False
        def forced_forward(**kwargs):
            logits = original_forward(**kwargs)
            # Force class 1
            logits[..., 0] = -10.0
            logits[..., 1] = 10.0
            return logits
        model.trigger.forward = forced_forward
        model.trigger.active = True

    with torch.no_grad():
        output_ids, aug_mask = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=gen_config,
            return_augmentation_mask=True,
        )

    # Restore
    model.trigger.forward = original_forward

    prompt_len = input_ids.shape[1]
    new_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    token_count = new_ids.shape[0]
    aug_mask_seq = aug_mask[0, :token_count].tolist()

    # Parse candidates
    candidates = []
    for i, v in enumerate(aug_mask_seq):
        if v != -100:
            token_text = "<prompt>" if i == 0 else tokenizer.decode([output_ids[0, prompt_len + i - 1]])
            candidates.append({
                "position": i,
                "token": repr(token_text)[:15],
                "aug_value": v,
            })

    augmentation_count = sum(1 for v in aug_mask_seq if v == 1)

    return {
        "text": text,
        "augmentation_count": augmentation_count,
        "candidates": candidates,
        "trigger_log": trigger_log,
        "aug_mask": aug_mask_seq,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "trigger_three_way_easy_probe.jsonl")

    print("=" * 80)
    print("Three-way controlled probe: always_0 / trained / always_1")
    print(f"Questions: {len(SIMPLE_QUESTIONS)} | Max tokens: {MAX_NEW_TOKENS}")
    print("=" * 80)

    tokenizer = load_tokenizer()
    print("\nLoading MemGen model...")
    model = load_memgen_model(tokenizer)
    print("Model loaded.")

    all_results = []

    # Run three modes
    for mode in ["always_0", "trained", "always_1"]:
        print(f"\n{'='*80}")
        print(f"Mode: {mode}")
        print(f"{'='*80}")

        for q in SIMPLE_QUESTIONS:
            print(f"  [{q['id']}] {q['question'][:50]}...")
            result = generate_with_mode(model, tokenizer, q["question"], mode)
            answer = extract_answer(result["text"])
            correct = is_correct(result["text"], q["answer"])

            print(f"    answer={answer} correct={correct} aug_count={result['augmentation_count']}")

            all_results.append({
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["answer"],
                "mode": mode,
                "output": result["text"],
                "extracted_answer": answer,
                "correct": correct,
                "augmentation_count": result["augmentation_count"],
                "candidates": result["candidates"],
                "trigger_log": result["trigger_log"],
            })

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved to: {output_path}")

    # Analysis
    print(f"\n{'='*80}")
    print("Analysis")
    print(f"{'='*80}")

    # A. Overall results
    print("\nA. Overall results:")
    print(f"{'mode':<12} {'accuracy':<12} {'total_aug':<12} {'avg_aug/q':<12}")
    print("-" * 50)
    for mode in ["always_0", "trained", "always_1"]:
        mode_results = [r for r in all_results if r["mode"] == mode]
        correct_count = sum(1 for r in mode_results if r["correct"])
        total_aug = sum(r["augmentation_count"] for r in mode_results)
        avg_aug = total_aug / len(mode_results)
        print(f"{mode:<12} {correct_count}/{len(mode_results)} = {correct_count/len(mode_results):.1%}{'':<4} {total_aug:<12} {avg_aug:.2f}")

    # B. Paired transitions
    print("\nB. Paired transitions (baseline: always_0):")
    for target_mode in ["trained", "always_1"]:
        print(f"\n  {target_mode} vs always_0:")
        baseline = {r["id"]: r for r in all_results if r["mode"] == "always_0"}
        target = {r["id"]: r for r in all_results if r["mode"] == target_mode}

        preserved = recovered = regressed = both_wrong = 0
        for qid in baseline:
            b = baseline[qid]["correct"]
            t = target[qid]["correct"]
            if b and t:
                preserved += 1
            elif not b and t:
                recovered += 1
            elif b and not t:
                regressed += 1
            else:
                both_wrong += 1

        print(f"    Preserved:  {preserved}")
        print(f"    Recovered:  {recovered}")
        print(f"    Regressed:  {regressed}")
        print(f"    Both wrong: {both_wrong}")

    # C. Trained trigger statistics
    print("\nC. Trained trigger statistics:")
    trained_results = [r for r in all_results if r["mode"] == "trained"]
    all_candidates = []
    all_decisions = []
    prompt_candidates = []
    inference_candidates = []

    for r in trained_results:
        for i, cand in enumerate(r["candidates"]):
            cand["question_id"] = r["id"]
            all_candidates.append(cand)
            if i == 0:
                prompt_candidates.append(cand)
            else:
                inference_candidates.append(cand)

        for log_entry in r["trigger_log"]:
            all_decisions.append(log_entry["decision"])

    total_cands = len(all_candidates)
    dec_0 = sum(1 for d in all_decisions if d == 0)
    dec_1 = sum(1 for d in all_decisions if d == 1)

    print(f"  Total candidates: {total_cands}")
    print(f"  Prompt candidates: {len(prompt_candidates)}")
    print(f"  Inference candidates: {len(inference_candidates)}")
    print(f"  Decision=0: {dec_0} ({dec_0/len(all_decisions):.1%})")
    print(f"  Decision=1: {dec_1} ({dec_1/len(all_decisions):.1%})")

    # Softmax p1 statistics
    p1_values = [log["softmax"][1] for r in trained_results for log in r["trigger_log"]]
    if p1_values:
        print(f"  p1 min: {min(p1_values):.3f}")
        print(f"  p1 median: {sorted(p1_values)[len(p1_values)//2]:.3f}")
        print(f"  p1 mean: {sum(p1_values)/len(p1_values):.3f}")
        print(f"  p1 max: {max(p1_values):.3f}")

    # List all decision=0 cases
    dec_0_cases = [c for c, d in zip(all_candidates, all_decisions) if d == 0]
    if dec_0_cases:
        print(f"\n  Decision=0 cases ({len(dec_0_cases)}):")
        for c in dec_0_cases:
            print(f"    [{c['question_id']}] pos={c['position']} token={c['token']}")
    else:
        print(f"\n  No decision=0 cases found.")

    # D. Generation path comparison
    print("\nD. Generation path comparison:")
    for q in SIMPLE_QUESTIONS:
        qid = q["id"]
        trained_r = next(r for r in all_results if r["id"] == qid and r["mode"] == "trained")
        always1_r = next(r for r in all_results if r["id"] == qid and r["mode"] == "always_1")

        trained_augs = [c["aug_value"] for c in trained_r["candidates"]]
        always1_augs = [c["aug_value"] for c in always1_r["candidates"]]

        if trained_augs == always1_augs:
            if trained_r["output"] == always1_r["output"]:
                print(f"  [{qid}] Identical decisions & byte-identical output")
            else:
                print(f"  [{qid}] Identical decisions but DIFFERENT output (unexpected!)")
        else:
            # Find first difference
            for i, (t, a) in enumerate(zip(trained_augs, always1_augs)):
                if t != a:
                    print(f"  [{qid}] First diff at candidate {i}: trained={t}, always_1={a}")
                    break

    print("\nDone.")


if __name__ == "__main__":
    main()
