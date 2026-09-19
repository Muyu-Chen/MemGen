"""
Minimal CPU test for MemGen model.
Does NOT modify MemGen source - loads model with CPU-compatible settings directly.
"""
import sys
import os
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel

MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/Qwen2.5-1.5B-Instruct")

def load_model_cpu():
    print("=" * 60)
    print("MemGen CPU Test - Loading model with CPU-compatible settings")
    print("=" * 60)

    model_name = MODEL_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    # MemGen config with minimal settings for CPU test
    memgen_config = MemGenConfig.from_pretrained(
        model_name,
        max_prompt_aug_num=1,
        max_inference_aug_num=1,
        prompt_latents_len=4,
        inference_latents_len=4,
        weaver_lora_config={
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        trigger_active=False,
        trigger_lora_config={
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
    )

    # Load base models on CPU with fp32, NO flash_attention_2
    print("Loading reasoner base model...")
    t0 = time.time()
    reasoner = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    print(f"  Reasoner loaded in {time.time()-t0:.1f}s")

    print("Loading weaver base model...")
    t0 = time.time()
    weaver = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    print(f"  Weaver loaded in {time.time()-t0:.1f}s")

    print("Loading trigger base model...")
    t0 = time.time()
    trigger = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    print(f"  Trigger loaded in {time.time()-t0:.1f}s")

    print("Building MemGenModel...")
    t0 = time.time()
    model = MemGenModel(
        config=memgen_config,
        base_tokenizer=tokenizer,
        reasoner_base_model=reasoner,
        weaver_base_model=weaver,
        trigger_base_model=trigger,
    )
    model.eval()
    print(f"  MemGenModel built in {time.time()-t0:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable params: {trainable_params/1e6:.1f}M")

    return model, tokenizer


def test_forward_simple(model, tokenizer):
    """Minimal forward test with correct label boundary format."""
    print("\n" + "=" * 60)
    print("Test 1: Forward pass (simplified)")
    print("=" * 60)

    # Use a simple single-turn format with exactly one prompt->label boundary
    prompt_text = "Solve: What is 2+3?"
    completion_text = " The answer is 5."

    prompt_messages = [{"role": "user", "content": prompt_text}]
    full_messages = prompt_messages + [{"role": "assistant", "content": completion_text}]

    prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(full_messages, tokenize=True)

    input_ids = torch.tensor([full_ids])
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    prompt_len = len(prompt_ids)
    labels[:, :prompt_len] = -100

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Prompt tokens: {prompt_len}, Total tokens: {input_ids.shape[1]}")
    print(f"  Label boundary at position: {prompt_len}")
    print(f"  Running forward pass...")

    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    elapsed = time.time() - t0

    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Forward pass time: {elapsed:.2f}s")
    print("  PASSED")


def test_generate(model, tokenizer):
    print("\n" + "=" * 60)
    print("Test 2: Generation with weaver augmentation")
    print("=" * 60)

    prompt = "Q: What is 12 + 7?\nA:"
    messages = [{"role": "user", "content": "What is 12 + 7? Think step by step."}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"  Prompt: {text[:80]}...")
    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Generating (max 20 new tokens)...")

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        weaver_do_sample=False,
        trigger_do_sample=False,
        temperature=0.0,
    )

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )
    elapsed = time.time() - t0

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    print(f"  Generated {new_tokens} tokens in {elapsed:.2f}s")
    print(f"  Speed: {new_tokens/elapsed:.2f} tokens/s")
    print(f"  Output: {generated_text[:200]}")
    print("  PASSED")


if __name__ == "__main__":
    model, tokenizer = load_model_cpu()
    test_forward_simple(model, tokenizer)
    test_generate(model, tokenizer)
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
