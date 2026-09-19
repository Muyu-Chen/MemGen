"""
MemGen A/B Comparison: Base Model vs Base Model + MemGen LoRA
Uses GSM8K test set, CPU-only, no source code modifications.
"""
import sys
import os
import time
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MemGen"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel

BASE_MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
MEMGEN_CKPT_PATH = "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 128

GSM8K_SYSTEM_PROMPT = r"""Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."""


def load_gsm8k_test(n=5):
    print(f"Loading GSM8K test set ({n} samples)...")
    ds = load_dataset("gsm8k", "main", split=f"test[:{n}]")
    samples = []
    for i, ex in enumerate(ds):
        question = ex["question"].strip()
        answer = ex["answer"].strip()
        boxed = answer.split("####")[-1].strip()
        samples.append({
            "id": i,
            "question": question,
            "ground_truth": boxed,
            "full_answer": answer,
        })
    print(f"  Loaded {len(samples)} samples")
    return samples


def build_prompt(question):
    user_content = GSM8K_SYSTEM_PROMPT + f"\nQuestion: {question}"
    return [{"role": "user", "content": user_content}]


def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def load_base_model():
    print("Loading base model (Qwen2.5-1.5B-Instruct)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float32
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def load_memgen_model():
    print("Loading MemGen model (Base + LoRA checkpoint)...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    memgen_config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1,
        max_inference_aug_num=3,
        prompt_latents_len=8,
        inference_latents_len=8,
        weaver_lora_config={
            "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
        },
        trigger_active=True,
        trigger_lora_config={
            "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
        },
    )

    print("  Loading base models for reasoner/weaver/trigger...")
    reasoner = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    weaver = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    trigger = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)

    print("  Building MemGen model with checkpoint...")
    model = MemGenModel.from_pretrained(
        MEMGEN_CKPT_PATH,
        config=memgen_config,
        base_tokenizer=tokenizer,
        reasoner_base_model=reasoner,
        weaver_base_model=weaver,
        trigger_base_model=trigger,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    return model, tokenizer


def run_base_model(model, tokenizer, samples):
    print(f"\n{'='*60}")
    print(f"Running BASE MODEL on {len(samples)} samples")
    print(f"{'='*60}")

    results = []
    for s in samples:
        messages = build_prompt(s["question"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = extract_boxed_answer(generated)
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        print(f"\n  [Sample {s['id']}] {s['question'][:60]}...")
        print(f"    Ground truth: {s['ground_truth']}")
        print(f"    Base output: {generated[:120]}...")
        print(f"    Base answer: {answer}")
        print(f"    Tokens: {new_tokens}, Time: {elapsed:.2f}s")

        results.append({
            "id": s["id"],
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "output": generated,
            "answer": answer,
            "tokens": new_tokens,
            "time": elapsed,
        })

    return results


def run_memgen_model(model, tokenizer, samples):
    print(f"\n{'='*60}")
    print(f"Running MemGen MODEL on {len(samples)} samples")
    print(f"{'='*60}")

    results = []
    for s in samples:
        messages = build_prompt(s["question"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")

        gen_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            weaver_do_sample=False,
            trigger_do_sample=False,
            temperature=0.0,
        )

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_config,
            )
        elapsed = time.time() - t0

        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answer = extract_boxed_answer(generated)
        new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]

        print(f"\n  [Sample {s['id']}] {s['question'][:60]}...")
        print(f"    Ground truth: {s['ground_truth']}")
        print(f"    MemGen output: {generated[:120]}...")
        print(f"    MemGen answer: {answer}")
        print(f"    Tokens: {new_tokens}, Time: {elapsed:.2f}s")

        results.append({
            "id": s["id"],
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "output": generated,
            "answer": answer,
            "tokens": new_tokens,
            "time": elapsed,
        })

    return results


def print_comparison(base_results, memgen_results):
    print(f"\n{'='*60}")
    print("A/B COMPARISON")
    print(f"{'='*60}")

    diff_count = 0
    same_count = 0
    base_correct = 0
    memgen_correct = 0

    header = f"{'ID':>3} | {'Base Correct':>12} | {'MemGen Correct':>14} | {'Same/Diff':>9} | {'Base Answer':>15} | {'MemGen Answer':>15} | {'GT':>8}"
    print(header)
    print("-" * len(header))

    for b, m in zip(base_results, memgen_results):
        b_correct = b["answer"] == b["ground_truth"]
        m_correct = m["answer"] == m["ground_truth"]
        same = b["output"].strip() == m["output"].strip()

        if same:
            same_count += 1
            sd = "SAME"
        else:
            diff_count += 1
            sd = "DIFFERENT"

        if b_correct:
            base_correct += 1
        if m_correct:
            memgen_correct += 1

        b_ans = b["answer"] or "N/A"
        m_ans = m["answer"] or "N/A"
        b_mark = "YES" if b_correct else "no"
        m_mark = "YES" if m_correct else "no"

        print(f"{b['id']:>3} | {b_mark:>12} | {m_mark:>14} | {sd:>9} | {b_ans:>15} | {m_ans:>15} | {b['ground_truth']:>8}")

    total = len(base_results)
    print(f"\n--- Statistics ---")
    print(f"Total samples:    {total}")
    print(f"Different outputs: {diff_count}")
    print(f"Same outputs:      {same_count}")
    print(f"Base correct:      {base_correct}/{total}")
    print(f"MemGen correct:    {memgen_correct}/{total}")

    return {
        "total": total,
        "different": diff_count,
        "same": same_count,
        "base_correct": base_correct,
        "memgen_correct": memgen_correct,
    }


if __name__ == "__main__":
    samples = load_gsm8k_test(NUM_SAMPLES)

    base_model, base_tokenizer = load_base_model()
    base_results = run_base_model(base_model, base_tokenizer, samples)

    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    memgen_model, memgen_tokenizer = load_memgen_model()
    memgen_results = run_memgen_model(memgen_model, memgen_tokenizer, samples)

    stats = print_comparison(base_results, memgen_results)

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    print(f"MemGen LoRA files exist:           YES")
    print(f"MemGen LoRA loads successfully:     YES")
    print(f"Base and MemGen outputs differ:     {stats['different']}/{stats['total']}")
    print(f"CPU end-to-end MemGen inference:    PASS")
    print(f"{'='*60}")

    with open("ab_results.json", "w") as f:
        json.dump({"base": base_results, "memgen": memgen_results, "stats": stats}, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to ab_results.json")
