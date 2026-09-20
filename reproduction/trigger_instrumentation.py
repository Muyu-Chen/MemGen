"""
Trigger instrumentation: capture logits/softmax/decision at each augmentation point.
Runs 5 GSM8K questions with TRIGGER_ACTIVE=True only.
"""
import sys
import os
import copy
import json
import gc
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


def load_gsm8k_test(n=NUM_QUESTIONS):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    questions = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        answer_raw = ex["answer"].strip()
        parts = answer_raw.split("\n####")
        clean_answer = parts[-1].strip()
        questions.append({
            "id": f"gsm8k_test_{i}",
            "question": ex["question"].strip(),
            "ground_truth": clean_answer,
        })
    return questions


def build_prompt_messages(question: str) -> list:
    content = OFFICIAL_PROMPT + "\nQuestion: " + question
    return [{"role": "user", "content": content}]


def build_prompt_text(tokenizer, question: str) -> str:
    messages = build_prompt_messages(question)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    return tokenizer


def load_memgen_model(tokenizer, trigger_active=True):
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


def generate_with_trigger_logging(model, tokenizer, question):
    """Generate with trigger logging via monkey-patch."""
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

    # Monkey-patch trigger forward to log calls
    trigger_log = []
    original_forward = model.trigger.forward
    
    def logged_forward(**kwargs):
        # Call original
        logits = original_forward(**kwargs)
        
        # Capture logits at last position
        last_logits = logits[0, -1, :]  # [2]
        softmax = F.softmax(last_logits, dim=0)
        decision = torch.argmax(last_logits).item()
        
        trigger_log.append({
            "logits": last_logits.tolist(),
            "softmax": softmax.tolist(),
            "decision": decision,
        })
        
        return logits
    
    model.trigger.forward = logged_forward
    
    with torch.no_grad():
        output_ids, aug_mask = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=gen_config,
            return_augmentation_mask=True,
        )
    
    # Restore original
    model.trigger.forward = original_forward
    
    prompt_len = input_ids.shape[1]
    new_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    token_count = new_ids.shape[0]
    
    return text, aug_mask[0, :token_count].tolist(), trigger_log, output_ids, input_ids.shape[1]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("Trigger Instrumentation: TRIGGER_ACTIVE=True only")
    print("=" * 80)
    
    questions = load_gsm8k_test(NUM_QUESTIONS)
    tokenizer = load_tokenizer()
    
    print("\nLoading MemGen model with TRIGGER_ACTIVE=True...")
    model = load_memgen_model(tokenizer, trigger_active=True)
    print(f"Model loaded. Trigger active: {model.trigger.active}")
    
    # Check trigger weights
    print("\nTrigger checkpoint inspection:")
    trigger_state = torch.load(os.path.join(CHECKPOINT_PATH, "trigger.bin"), map_location="cpu", weights_only=True)
    print(f"  trigger.bin keys: {list(trigger_state.keys())}")
    output_layer_weight = trigger_state["output_layer"]["weight"]
    output_layer_bias = trigger_state["output_layer"]["bias"]
    print(f"  output_layer.weight shape: {output_layer_weight.shape}")
    print(f"  output_layer.bias shape: {output_layer_bias.shape}")
    print(f"  output_layer.weight sample: {output_layer_weight[0, :5].tolist()}")
    print(f"  output_layer.bias: {output_layer_bias.tolist()}")
    
    trigger_ckpt_sd = safe_load_file(
        os.path.join(CHECKPOINT_PATH, "trigger", "trigger", "adapter_model.safetensors"), device="cpu"
    )
    print(f"  Trigger LoRA adapter keys: {list(trigger_ckpt_sd.keys())[:5]}...")
    
    all_results = []
    
    for q in questions:
        print(f"\n{'='*80}")
        print(f"[{q['id']}] {q['question'][:60]}...")
        print(f"{'='*80}")
        
        text, aug_mask, trigger_log, output_ids, prompt_len = generate_with_trigger_logging(
            model, tokenizer, q["question"]
        )
        
        print(f"Generated {len(aug_mask)} tokens")
        print(f"Trigger was called {len(trigger_log)} times")
        
        # Find candidate positions (where aug_mask != -100)
        candidate_positions = [i for i, v in enumerate(aug_mask) if v != -100]
        print(f"Candidate augmentation points: {len(candidate_positions)}")
        
        print(f"\n{'Call#':<6} {'Pos':<6} {'Token':<15} {'Logits [c0,c1]':<30} {'Softmax [p0,p1]':<30} {'Decision':<10} {'Aug':<6}")
        print("-" * 100)
        
        # Match trigger calls to positions
        # Trigger is called at: position 0 (prompt), then at delimiter positions during generation
        call_idx = 0
        for pos_idx in candidate_positions[:20]:  # Limit for readability
            if pos_idx == 0:
                token_text = "<prompt>"
            else:
                # Get the token at this position
                token_id = output_ids[0, prompt_len + pos_idx - 1]
                token_text = tokenizer.decode([token_id])
                token_text = repr(token_text)[:12]
            
            if call_idx < len(trigger_log):
                log_entry = trigger_log[call_idx]
                logits_str = f"[{log_entry['logits'][0]:.3f}, {log_entry['logits'][1]:.3f}]"
                softmax_str = f"[{log_entry['softmax'][0]:.3f}, {log_entry['softmax'][1]:.3f}]"
                decision = log_entry['decision']
                call_idx += 1
            else:
                logits_str = "N/A"
                softmax_str = "N/A"
                decision = "N/A"
            
            aug_val = aug_mask[pos_idx]
            aug_str = f"{aug_val}"
            
            print(f"{call_idx:<6} {pos_idx:<6} {token_text:<15} {logits_str:<30} {softmax_str:<30} {decision:<10} {aug_str:<6}")
        
        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "num_candidates": len(candidate_positions),
            "trigger_calls": len(trigger_log),
            "trigger_log_sample": trigger_log[:10],
            "aug_mask": aug_mask,
        })
    
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    for r in all_results:
        print(f"  [{r['id']}] {r['num_candidates']} candidates")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "trigger_instrumentation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
