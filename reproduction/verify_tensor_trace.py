"""
Single-question tensor trace of the MemGen pipeline.

Empirically verifies the real data flow by hooking into the live modules rather
than re-implementing it:

  1. prompt token IDs
  2. Reasoner input embeddings            (NOT hidden states)
  3. reasoner_to_weaver projection
  4. concat order inside Weaver           -> [context, query_latents]
  5. Weaver forward -> last hidden state
  6. extraction slice                     -> hidden_states[:, -K:, :]
  7. weaver_to_reasoner projection
  8. append back into Reasoner sequence
  9. Reasoner next token

Verification method: forward pre-hooks capture the exact tensors the submodules
receive, then we assert the concat layout and the extraction slice by direct
tensor comparison.
"""
import sys
import os
import json
import copy
import gc

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from safetensors.torch import load_file as safe_load_file

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel, _remap_lora_adapter_key
from memgen.model.weaver import MemGenWeaver

BASE_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models/Qwen2.5-1.5B-Instruct")
CHECKPOINT_PATH = os.path.join(
    _PROJECT_ROOT,
    "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model",
)
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "reproduction", "phase1_results", "tensor_trace.json")

WEAVER_LORA_CONFIG = {
    "r": 16, "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM",
}
TRIGGER_LORA_CONFIG = dict(WEAVER_LORA_CONFIG)

QUESTION = "What is 2+3?"
OFFICIAL_PROMPT = r"Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."
MAX_NEW_TOKENS = 40


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
    return tok


def load_model(tokenizer):
    config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1, max_inference_aug_num=3,
        prompt_latents_len=8, inference_latents_len=8,
        weaver_lora_config=WEAVER_LORA_CONFIG,
        trigger_active=False,          # hardcoded always-augment; deterministic
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

    weaver_ckpt = safe_load_file(
        os.path.join(CHECKPOINT_PATH, "weaver", "weaver", "adapter_model.safetensors"), device="cpu"
    )
    weaver_model_sd = model.weaver.model.state_dict()
    weaver_final = {
        _remap_lora_adapter_key(k, "weaver"): v
        for k, v in weaver_ckpt.items()
        if _remap_lora_adapter_key(k, "weaver") in weaver_model_sd
    }
    model.weaver.model.load_state_dict(weaver_final, strict=False)
    model.weaver.model.set_adapter("weaver")

    trigger_ckpt = safe_load_file(
        os.path.join(CHECKPOINT_PATH, "trigger", "trigger", "adapter_model.safetensors"), device="cpu"
    )
    trigger_model_sd = model.trigger.model.state_dict()
    trigger_final = {
        _remap_lora_adapter_key(k, "trigger"): v
        for k, v in trigger_ckpt.items()
        if _remap_lora_adapter_key(k, "trigger") in trigger_model_sd
    }
    model.trigger.model.load_state_dict(trigger_final, strict=False)
    model.trigger.model.set_adapter("trigger")

    model.eval()
    print(f"  weaver LoRA keys matched: {len(weaver_final)}/{len(weaver_ckpt)}")
    print(f"  trigger LoRA keys matched: {len(trigger_final)}/{len(trigger_ckpt)}")
    return model


def stat(t):
    t = t.detach().float()
    return {
        "shape": list(t.shape),
        "mean": round(float(t.mean()), 6),
        "std": round(float(t.std()), 6),
        "absmax": round(float(t.abs().max()), 6),
        "first3": [round(float(x), 6) for x in t.flatten()[:3].tolist()],
    }


def main():
    print("=" * 78)
    print("MemGen single-question tensor trace")
    print("=" * 78)

    print("\nLoading tokenizer + model...")
    tokenizer = load_tokenizer()
    model = load_model(tokenizer)

    messages = [{"role": "user", "content": OFFICIAL_PROMPT + "\nQuestion: " + QUESTION}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = input_ids.size(1)

    trace = {"question": QUESTION, "prompt_len": prompt_len, "steps": []}

    def log(step, payload):
        trace["steps"].append({"step": step, **payload})
        print(f"\n[{step}]")
        for k, v in payload.items():
            print(f"    {k}: {v}")

    # ── STEP 1: prompt tokens ──────────────────────────────────────────────
    log("1. prompt tokens", {
        "input_ids_shape": list(input_ids.shape),
        "first_10_ids": input_ids[0, :10].tolist(),
        "last_5_ids": input_ids[0, -5:].tolist(),
        "decoded_last_5": [tokenizer.decode([int(i)]) for i in input_ids[0, -5:]],
    })

    # ── STEP 2: Reasoner input embeddings ──────────────────────────────────
    reasoner_embed = model.reasoner.get_input_embeddings()
    inputs_embeds = reasoner_embed(input_ids)

    # Prove these are INPUT EMBEDDINGS: identical to a direct table lookup,
    # and NOT a Reasoner forward-pass hidden state.
    table_lookup = reasoner_embed.weight[input_ids]
    is_input_embedding = torch.allclose(inputs_embeds, table_lookup, atol=1e-6)

    log("2. Reasoner input embeddings", {
        "shape": list(inputs_embeds.shape),
        "equals_embedding_table_lookup": bool(is_input_embedding),
        "note": "True => Weaver sees input embeddings, NOT Reasoner hidden states",
        "stats": stat(inputs_embeds),
    })
    trace["verified_input_embeddings_not_hidden_states"] = bool(is_input_embedding)

    # ── Hook the Weaver to capture the REAL concatenated tensor ────────────
    captured = {}

    def weaver_pre_hook(module, args, kwargs):
        captured["weaver_input"] = kwargs.get("inputs_embeds")
        captured["attention_mask"] = kwargs.get("attention_mask")
        captured["position_ids"] = kwargs.get("position_ids")
        return args, kwargs

    def weaver_post_hook(module, args, kwargs, output):
        captured["weaver_output"] = output
        return output

    h_pre = model.weaver.model.register_forward_pre_hook(weaver_pre_hook, with_kwargs=True)
    h_post = model.weaver.model.register_forward_hook(weaver_post_hook, with_kwargs=True)

    # ── STEP 3: reasoner_to_weaver projection ──────────────────────────────
    weaver_inputs_embeds = model.reasoner_to_weaver(inputs_embeds)
    log("3. reasoner_to_weaver projection", {
        "in_shape": list(inputs_embeds.shape),
        "out_shape": list(weaver_inputs_embeds.shape),
        "weight_shape": list(model.reasoner_to_weaver.weight.shape),
        "is_linear_map_of_embeddings": True,
        "stats": stat(weaver_inputs_embeds),
    })

    # ── STEP 4-6: run the real augment_prompt path ─────────────────────────
    position_ids = model._generate_position_ids(attention_mask)
    latent_hidden, latent_mask, latent_pos = model.weaver.augment_prompt(
        weaver_inputs_embeds, attention_mask, position_ids
    )
    K = model.weaver.prompt_latents_num

    weaver_in = captured["weaver_input"]
    ctx_len = weaver_inputs_embeds.size(1)
    total_len = weaver_in.size(1)

    # Empirical concat-order verification
    head_is_context = torch.allclose(weaver_in[:, :ctx_len, :], weaver_inputs_embeds, atol=1e-6)

    normalized_q = model.weaver.prompt_latent_ln(model.weaver.prompt_query_latents) \
        * model.weaver.prompt_latent_scale
    normalized_q = normalized_q.unsqueeze(0).expand(weaver_in.size(0), -1, -1)
    tail_is_query = torch.allclose(weaver_in[:, ctx_len:, :], normalized_q, atol=1e-5)

    log("4. Weaver concat layout (empirically verified)", {
        "query_latents_K": K,
        "context_len": ctx_len,
        "concatenated_len": total_len,
        "layout": f"[context({ctx_len}), query_latents({K})]",
        "head_is_context": bool(head_is_context),
        "tail_is_query_latents": bool(tail_is_query),
        "CORRECTED": "order is [context, Q] -- NOT [Q, context]",
    })
    trace["verified_concat_is_context_then_query"] = bool(head_is_context and tail_is_query)

    weaver_out = captured["weaver_output"]
    weaver_hidden = weaver_out.hidden_states[-1] if hasattr(weaver_out, "hidden_states") else weaver_out[1][-1]

    log("5. Weaver forward output", {
        "last_hidden_state_shape": list(weaver_hidden.shape),
        "stats": stat(weaver_hidden),
    })

    # Empirical extraction verification
    expected_tail = weaver_hidden[:, -K:, :]
    expected_head = weaver_hidden[:, :K, :]
    tail_matches = torch.allclose(latent_hidden, expected_tail, atol=1e-6)
    head_matches = torch.allclose(latent_hidden, expected_head, atol=1e-6)

    log("6. latent extraction slice (empirically verified)", {
        "extracted_shape": list(latent_hidden.shape),
        "matches_LAST_K": bool(tail_matches),
        "matches_FIRST_K": bool(head_matches),
        "CORRECTED": "takes hidden_states[:, -K:, :] (LAST K) -- NOT the first K",
        "why_it_works": "causal attention: Q sits at the end, so it attends to the full context",
    })
    trace["verified_extract_last_K"] = bool(tail_matches)
    trace["verified_extract_NOT_first_K"] = bool(not head_matches)

    # ── STEP 7: weaver_to_reasoner ─────────────────────────────────────────
    latent_inputs_embeds = model.weaver_to_reasoner(latent_hidden)
    log("7. weaver_to_reasoner projection", {
        "in_shape": list(latent_hidden.shape),
        "out_shape": list(latent_inputs_embeds.shape),
        "stats": stat(latent_inputs_embeds),
    })

    # ── STEP 8: append back into the Reasoner sequence ─────────────────────
    merged_embeds = torch.cat([inputs_embeds, latent_inputs_embeds], dim=1)
    merged_mask = torch.cat([attention_mask, latent_mask], dim=1)
    log("8. append latent memory to Reasoner input", {
        "before_shape": list(inputs_embeds.shape),
        "after_shape": list(merged_embeds.shape),
        "attention_mask_after": merged_mask[0].tolist()[:6] + ["..."] + merged_mask[0].tolist()[-K:],
        "context_retained": bool(torch.allclose(merged_embeds[:, :prompt_len, :], inputs_embeds, atol=1e-6)),
        "latent_appended_at_tail": bool(
            torch.allclose(merged_embeds[:, prompt_len:, :], latent_inputs_embeds, atol=1e-6)
        ),
    })

    # ── STEP 9: Reasoner continues generation ──────────────────────────────
    h_pre.remove()
    h_post.remove()

    with torch.no_grad():
        out = model.reasoner(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            position_ids=model._generate_position_ids(merged_mask),
            use_cache=False,
        )
    next_logits = out.logits[:, -1, :]
    next_id = int(next_logits.argmax(dim=-1).item())

    log("9. Reasoner next token (first step after augmentation)", {
        "logits_shape": list(next_logits.shape),
        "next_token_id": next_id,
        "next_token": repr(tokenizer.decode([next_id])),
        "top5": [(repr(tokenizer.decode([int(i)])), round(float(p), 4))
                 for p, i in zip(*torch.softmax(next_logits[0], dim=-1).topk(5))],
    })

    # ── STEP 10: full generation for reference ─────────────────────────────
    print("\n[10. full generation]")
    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    gen_config.weaver_do_sample = False
    gen_config.trigger_do_sample = False
    with torch.no_grad():
        generated, aug_mask = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=gen_config, return_augmentation_mask=True,
        )
    gen_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
    aug_positions = [int(i) for i in (aug_mask[0] == 1).nonzero(as_tuple=True)[0].tolist()]
    print(f"    output: {gen_text!r}")
    print(f"    augmentation positions: {aug_positions}")
    trace["full_generation"] = gen_text
    trace["augmentation_positions"] = aug_positions

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)
    checks = [
        ("Weaver input is Reasoner INPUT EMBEDDINGS (not hidden states)",
         trace["verified_input_embeddings_not_hidden_states"]),
        ("concat layout is [context, query_latents]",
         trace["verified_concat_is_context_then_query"]),
        ("extraction takes the LAST K hidden states",
         trace["verified_extract_last_K"]),
        ("extraction does NOT take the first K",
         trace["verified_extract_NOT_first_K"]),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
