"""
Interactive chat with MemGen (Base + Weaver latent memory).

Loads the MemGen model with a weaver-sft checkpoint and runs interactive
generation.  The weaver injects latent memory tokens into the reasoner's
context at each augmentation point.

Usage:
    python reproduction/chat_memgen.py
    python reproduction/chat_memgen.py --checkpoint models/memgen-checkpoints/.../model
    python reproduction/chat_memgen.py --max-new-tokens 256
"""
import sys
import os
import argparse
import copy
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel
from memgen.model.modeling_memgen import _remap_lora_adapter_key
from safetensors.torch import load_file as safe_load_file

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CHECKPOINT = os.path.join(
    _PROJECT_ROOT,
    "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model",
)


def load_memgen(base_model_name, checkpoint_path):
    print(f"Loading base model: {base_model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    dtype = torch.float32 if not torch.cuda.is_available() else torch.bfloat16
    model_kwargs = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    base_model.eval()
    print(f"  Base model loaded in {time.time()-t0:.1f}s")

    print(f"Loading MemGen checkpoint: {checkpoint_path}")
    t1 = time.time()

    memgen_config = MemGenConfig.from_pretrained(
        base_model_name,
        max_prompt_aug_num=1,
        max_inference_aug_num=3,
        prompt_latents_len=8,
        inference_latents_len=8,
        weaver_lora_config={
            "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
        },
        trigger_active=False,
        trigger_lora_config={
            "r": 16, "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
        },
    )

    print("  Cloning base model for reasoner/weaver/trigger...")
    reasoner = copy.deepcopy(base_model)
    weaver = copy.deepcopy(base_model)
    trigger = copy.deepcopy(base_model)

    model = MemGenModel(
        config=memgen_config,
        base_tokenizer=tokenizer,
        reasoner_base_model=reasoner,
        weaver_base_model=weaver,
        trigger_base_model=trigger,
    )

    # --- Load checkpoint weights ---
    # 1. Projection layers
    proj_state = torch.load(
        os.path.join(checkpoint_path, "projs.bin"), map_location="cpu", weights_only=True
    )
    model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])

    # 2. Weaver state (latents, layer norms, scales)
    weaver_state = torch.load(
        os.path.join(checkpoint_path, "weaver.bin"), map_location="cpu", weights_only=True
    )
    model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
    model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
    model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
    model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
    model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])

    # 3. Trigger state
    trigger_state = torch.load(
        os.path.join(checkpoint_path, "trigger.bin"), map_location="cpu", weights_only=True
    )
    model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])

    # 4. Weaver LoRA adapter (remap keys from adapter_name='default')
    weaver_ckpt_sd = safe_load_file(
        os.path.join(checkpoint_path, "weaver", "weaver", "adapter_model.safetensors"),
        device="cpu",
    )
    weaver_model_sd = model.weaver.model.state_dict()
    weaver_final = {}
    for k, v in weaver_ckpt_sd.items():
        new_key = _remap_lora_adapter_key(k, "weaver")
        if new_key in weaver_model_sd:
            weaver_final[new_key] = v
    missing, _ = model.weaver.model.load_state_dict(weaver_final, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    print(f"  Weaver LoRA: {len(weaver_final)}/{len(weaver_ckpt_sd)} keys loaded"
          + (f", WARNING: {len(lora_missing)} lora keys missing!" if lora_missing else ""))

    # 5. Trigger LoRA adapter
    trigger_ckpt_sd = safe_load_file(
        os.path.join(checkpoint_path, "trigger", "trigger", "adapter_model.safetensors"),
        device="cpu",
    )
    trigger_model_sd = model.trigger.model.state_dict()
    trigger_final = {}
    for k, v in trigger_ckpt_sd.items():
        new_key = _remap_lora_adapter_key(k, "trigger")
        if new_key in trigger_model_sd:
            trigger_final[new_key] = v
    missing, _ = model.trigger.model.load_state_dict(trigger_final, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    print(f"  Trigger LoRA: {len(trigger_final)}/{len(trigger_ckpt_sd)} keys loaded"
          + (f", WARNING: {len(lora_missing)} lora keys missing!" if lora_missing else ""))

    model.eval()
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  MemGen loaded in {time.time()-t1:.1f}s")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total/1e6:.1f}M params, Trainable (LoRA+proj): {trainable/1e6:.1f}M\n")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat with MemGen (Base + Weaver)")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer = load_memgen(args.base_model, args.checkpoint)

    print("=" * 60)
    print("  MemGen Chat (Base + Weaver Latent Memory)")
    print("  Type your question and press Enter.")
    print("  Commands: 'quit'/'exit' to stop, 'clear' to reset history.")
    print("=" * 60)

    messages = []

    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages.clear()
            print("[History cleared]")
            continue

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            weaver_do_sample=False,
            trigger_do_sample=False,
        )

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        elapsed = time.time() - t0

        new_ids = output_ids[0, input_ids.shape[1]:]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True)
        n_tokens = new_ids.shape[0]

        print(f"\n[MemGen] ({n_tokens} tokens, {elapsed:.1f}s)")
        print(reply)

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
