#!/usr/bin/env python3
"""
验证 always_0 control：对比纯 base model vs always_0 monkey-patch
"""
import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置路径
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 20 simple math questions (same as three-way probe)
QUESTIONS = [
    {"question": "What is 2+3?", "answer": "5"},
    {"question": "What is 10-4?", "answer": "6"},
    {"question": "What is 3*5?", "answer": "15"},
    {"question": "What is 12/3?", "answer": "4"},
    {"question": "John has 5 apples and buys 2 more. How many apples does John have?", "answer": "7"},
    {"question": "There are 20 students. 8 are boys. How many are girls?", "answer": "12"},
    {"question": "What is 7+8?", "answer": "15"},
    {"question": "What is 15-6?", "answer": "9"},
    {"question": "What is 4*6?", "answer": "24"},
    {"question": "What is 20/4?", "answer": "5"},
    {"question": "Mary has 10 candies. She gives 3 to her friend. How many candies does Mary have left?", "answer": "7"},
    {"question": "A box has 6 red balls and 4 blue balls. How many balls are in the box?", "answer": "10"},
    {"question": "What is 9+11?", "answer": "20"},
    {"question": "What is 18-9?", "answer": "9"},
    {"question": "What is 5*5?", "answer": "25"},
    {"question": "What is 16/2?", "answer": "8"},
    {"question": "Tom has 8 toys. He gets 4 more for his birthday. How many toys does Tom have?", "answer": "12"},
    {"question": "There are 15 birds on a tree. 5 fly away. How many birds are left?", "answer": "10"},
    {"question": "What is 6+7?", "answer": "13"},
    {"question": "What is 14-7?", "answer": "7"},
]

def extract_answer(text):
    """Extract boxed answer like official eval"""
    import re
    def _last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("boxed")
        if idx < 0:
            idx = string.rfind("**")
        if idx < 0:
            return None
        i = idx
        depth = 0
        while i < len(string):
            if string[i] == "{":
                depth += 1
            elif string[i] == "}":
                depth -= 1
                if depth == 0:
                    return string[idx:i+1]
            i += 1
        return None

    def _remove_boxed(s):
        left = "\\boxed{"
        if s is None:
            return None
        if s.startswith(left):
            return s[len(left):-1]
        left2 = "boxed"
        if s.startswith(left2):
            return s[len(left2):-1]
        left3 = "**"
        if s.startswith(left3):
            return s[len(left3):-1]
        return None

    boxed = _last_boxed_only_string(text)
    if boxed is None:
        return None
    ans = _remove_boxed(boxed)
    if ans is None:
        return None
    # Strip
    ans = ans.strip()
    ans = ans.replace(",", "")
    return ans

def main():
    print("=" * 80)
    print("验证 always_0 control: 纯 Base Model vs always_0 monkey-patch")
    print(f"题目数: {len(QUESTIONS)} | Max tokens: 512")
    print("=" * 80)

    # 使用 MemGen checkpoint 的 tokenizer（与 base model 共享）
    CHECKPOINT_PATH = os.path.join(
        _PROJECT_ROOT,
        "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    # ── 测试 1: 纯 Base Model (Qwen2.5-1.5B-Instruct, 无 Weaver/Trigger) ──
    print("\n" + "=" * 80)
    print("测试 1: 纯 Base Model (Qwen2.5-1.5B-Instruct)")
    print("=" * 80)

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    base_model.eval()

    base_results = []
    for i, q in enumerate(QUESTIONS, 1):
        prompt = f"Question: {q['question']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_answer(text)
        gt = q["answer"]
        correct = (pred == gt)
        print(f"  [base_{i:02d}] {q['question'][:50]}...")
        print(f"    pred={pred} gt={gt} correct={correct}")
        base_results.append({
            "question": q["question"],
            "answer": gt,
            "prediction": pred,
            "correct": correct,
        })

    base_model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    # ── 测试 2: MemGen with always_0 monkey-patch ──
    print("\n" + "=" * 80)
    print("测试 2: MemGen with always_0 monkey-patch")
    print("=" * 80)

    from memgen.model import MemGenForCausalLM
    model = MemGenForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        TRIGGER_ACTIVE=True,
    )
    model.eval()
    model.set_trigger_active(True)

    # Monkey-patch: force logits[..., 0] = 10, logits[..., 1] = -10 (always_0)
    original_forward = model.trigger.forward
    def always_0_forward(**kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)
        logits[..., 0] = 10.0
        logits[..., 1] = -10.0
        return logits
    model.trigger.forward = always_0_forward

    always0_results = []
    for i, q in enumerate(QUESTIONS, 1):
        prompt = f"Question: {q['question']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            output_ids, aug_mask = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        pred = extract_answer(text)
        gt = q["answer"]
        correct = (pred == gt)

        # Count augmentations
        aug_count = (aug_mask[0, :len(new_ids)] == 1).sum().item()

        print(f"  [always0_{i:02d}] {q['question'][:50]}...")
        print(f"    pred={pred} gt={gt} correct={correct} aug_count={aug_count}")
        always0_results.append({
            "question": q["question"],
            "answer": gt,
            "prediction": pred,
            "correct": correct,
            "aug_count": aug_count,
        })

    # ── 对比 ──
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)

    base_correct = sum(1 for r in base_results if r["correct"])
    always0_correct = sum(1 for r in always0_results if r["correct"])

    print(f"\n总体准确率:")
    print(f"  Base Model:  {base_correct}/{len(QUESTIONS)} = {base_correct/len(QUESTIONS)*100:.1f}%")
    print(f"  always_0:    {always0_correct}/{len(QUESTIONS)} = {always0_correct/len(QUESTIONS)*100:.1f}%")

    print(f"\n逐题对比:")
    print(f"{'题号':<6} {'Base':<10} {'always_0':<10} {'一致?':<8}")
    print("-" * 40)
    match_count = 0
    for i, (b, a) in enumerate(zip(base_results, always0_results), 1):
        match = "✓" if b["correct"] == a["correct"] else "✗"
        if b["correct"] == a["correct"]:
            match_count += 1
        print(f"[{i:02d}]    {'✓' if b['correct'] else '✗'} ({b['prediction'] or 'None':<6})  {'✓' if a['correct'] else '✗'} ({a['prediction'] or 'None':<6})  {match}")

    print(f"\n一致率: {match_count}/{len(QUESTIONS)} = {match_count/len(QUESTIONS)*100:.1f}%")

    # 保存结果
    output = {
        "base_results": base_results,
        "always0_results": always0_results,
        "summary": {
            "base_accuracy": base_correct / len(QUESTIONS),
            "always0_accuracy": always0_correct / len(QUESTIONS),
            "match_rate": match_count / len(QUESTIONS),
        }
    }
    output_path = "reproduction/phase1_results/always0_verification.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {output_path}")

if __name__ == "__main__":
    main()
