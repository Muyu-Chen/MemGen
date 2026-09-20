#!/usr/bin/env python3
"""
MemGen 完整流程追踪：单题深度 instrumentation
"""
import sys
import os
import copy
import gc
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel, _remap_lora_adapter_key
from safetensors.torch import load_file as safe_load_file

BASE_MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT,
    "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"
)

WEAVER_LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "task_type": "CAUSAL_LM", "target_modules": ["q_proj", "v_proj"]}
TRIGGER_LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "task_type": "CAUSAL_LM", "target_modules": ["q_proj", "v_proj"]}

# 选择一道简单的题
QUESTION = "What is 2+3?"
ANSWER = "5"

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_tensor(name, tensor, max_vals=5):
    if tensor is None:
        print(f"{name}: None")
        return
    shape = tuple(tensor.shape)
    dtype = tensor.dtype
    device = tensor.device
    print(f"{name}: shape={shape}, dtype={dtype}, device={device}")
    if tensor.numel() > 0:
        flat = tensor.flatten()[:max_vals]
        print(f"  sample values: {flat.tolist()}")

def main():
    print_section("MemGen 完整流程追踪")
    print(f"题目: {QUESTION}")
    print(f"答案: {ANSWER}")
    
    # ── 1. 加载模型和 tokenizer ──
    print_section("1. 加载模型和 Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True, local_files_only=True)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # 准备 prompt
    prompt = f"Question: {QUESTION}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    print(f"\nPrompt text:\n{prompt}")
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"\nPrompt tokens shape: {input_ids.shape}")
    print(f"Prompt token IDs: {input_ids[0].tolist()}")
    print(f"Prompt text (decoded): {tokenizer.decode(input_ids[0])}")
    
    # 加载 MemGen 模型
    print("\n加载 MemGen 模型...")
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
    
    # 加载 weights
    proj_state = torch.load(os.path.join(CHECKPOINT_PATH, "projs.bin"), map_location="cpu", weights_only=True)
    model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])
    
    weaver_state = torch.load(os.path.join(CHECKPOINT_PATH, "weaver.bin"), map_location="cpu", weights_only=True)
    model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
    model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
    model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
    model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
    model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])
    
    trigger_state = torch.load(os.path.join(CHECKPOINT_PATH, "trigger.bin"), map_location="cpu", weights_only=True)
    model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])
    
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
    print("模型加载完成。")
    
    # ── 2. Instrumentation: 重写 generate 方法以追踪每一步 ──
    print_section("2. 开始生成并追踪流程")
    
    # 保存中间结果
    trace_log = []
    
    def log_step(step_name, data):
        trace_log.append({"step": step_name, **data})
        print(f"\n[{step_name}]")
        for k, v in data.items():
            if k != "step":
                if isinstance(v, torch.Tensor):
                    print_tensor(f"  {k}", v)
                else:
                    print(f"  {k}: {v}")
    
    # ── Step 1: Prompt augmentation check ──
    print("\n--- Step 1: Prompt Augmentation Check ---")
    prompt_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    position_ids = attention_mask.long().cumsum(-1) - 1
    
    # Trigger decision for prompt
    with torch.no_grad():
        trigger_logits = model.trigger(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    
    log_step("prompt_trigger", {
        "trigger_logits_shape": trigger_logits.shape,
        "trigger_logits_last": trigger_logits[0, -1].tolist(),
        "softmax": torch.softmax(trigger_logits[0, -1], dim=-1).tolist(),
        "decision": "augment" if trigger_logits[0, -1, 1] > trigger_logits[0, -1, 0] else "no_augment"
    })
    
    # ── Step 2: Reasoner forward (prompt) ──
    print("\n--- Step 2: Reasoner Forward (Prompt) ---")
    with torch.no_grad():
        reasoner_outputs = model.reasoner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
    
    log_step("reasoner_prompt", {
        "output_logits_shape": reasoner_outputs.logits.shape,
        "hidden_states_last_shape": reasoner_outputs.hidden_states[-1].shape,
    })
    
    # ── Step 3: Weaver augmentation (prompt) ──
    print("\n--- Step 3: Weaver Augmentation (Prompt) ---")
    
    # Get reasoner hidden states
    reasoner_hidden = reasoner_outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
    print(f"Reasoner hidden states: {reasoner_hidden.shape}")
    
    # Project to Weaver space
    with torch.no_grad():
        weaver_input = model.reasoner_to_weaver(reasoner_hidden)
    
    print(f"After projection to Weaver space: {weaver_input.shape}")
    
    # Get query latents
    query_latents = model.weaver.prompt_query_latents  # (num_latents, hidden_size)
    print(f"Query latents shape: {query_latents.shape}")
    print(f"Query latents sample: {query_latents[0, :5].tolist()}")
    
    # Apply LayerNorm and scale
    query_latents_normed = model.weaver.prompt_latent_ln(query_latents)
    query_latents_scaled = query_latents_normed * model.weaver.prompt_latent_scale
    print(f"After LN+scale: {query_latents_scaled.shape}")
    
    # Expand to batch
    batch_size = weaver_input.shape[0]
    query_latents_batch = query_latents_scaled.unsqueeze(0).expand(batch_size, -1, -1)
    print(f"Expanded to batch: {query_latents_batch.shape}")
    
    # Concatenate: [query_latents, weaver_input]
    weaver_full_input = torch.cat([query_latents_batch, weaver_input], dim=1)
    print(f"Weaver full input (concat): {weaver_full_input.shape}")
    
    # Create attention mask for Weaver
    weaver_attention_mask = torch.ones(
        batch_size, weaver_full_input.shape[1],
        device=weaver_full_input.device, dtype=attention_mask.dtype
    )
    
    # Weaver forward
    with torch.no_grad():
        weaver_outputs = model.weaver(
            inputs_embeds=weaver_full_input,
            attention_mask=weaver_attention_mask,
            output_hidden_states=True,
        )
    
    weaver_hidden = weaver_outputs.hidden_states[-1]
    print(f"Weaver output hidden states: {weaver_hidden.shape}")
    
    # Extract augmented latents (first 8 tokens)
    augmented_latents = weaver_hidden[:, :query_latents.shape[0], :]
    print(f"Augmented latents (first {query_latents.shape[0]} tokens): {augmented_latents.shape}")
    
    # Project back to Reasoner space
    with torch.no_grad():
        reasoner_augmented = model.weaver_to_reasoner(augmented_latents)
    
    print(f"Projected back to Reasoner space: {reasoner_augmented.shape}")
    
    log_step("weaver_prompt_augmentation", {
        "reasoner_hidden_shape": reasoner_hidden.shape,
        "weaver_input_shape": weaver_input.shape,
        "query_latents_shape": query_latents.shape,
        "weaver_full_input_shape": weaver_full_input.shape,
        "weaver_hidden_shape": weaver_hidden.shape,
        "augmented_latents_shape": augmented_latents.shape,
        "reasoner_augmented_shape": reasoner_augmented.shape,
    })
    
    # ── Step 4: Continue generation ──
    print("\n--- Step 4: Continue Generation (with augmentation) ---")
    
    # For simplicity, just run the full generate and show the result
    with torch.no_grad():
        output_ids, aug_mask = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    
    print(f"\nGenerated text:\n{generated_text}")
    print(f"\nAugmentation mask (first 50 tokens): {aug_mask[0, :50].tolist()}")
    print(f"Total augmentations: {(aug_mask[0, :len(new_ids)] == 1).sum().item()}")
    
    # ── 保存 trace ──
    print_section("3. 保存 Trace")
    output_path = "reproduction/phase1_results/memgen_full_trace.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "question": QUESTION,
            "answer": ANSWER,
            "prompt": prompt,
            "prompt_tokens": input_ids[0].tolist(),
            "generated_text": generated_text,
            "trace_log": trace_log,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"已保存: {output_path}")
    
    print_section("完成")

if __name__ == "__main__":
    main()
