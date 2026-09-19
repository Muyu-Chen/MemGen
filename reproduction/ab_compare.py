"""
A/B Comparison: Base Model (Qwen2.5-1.5B-Instruct) vs MemGen (Base + Weaver LoRA)
on GSM8K test questions.

Goal: confirm that loading MemGen checkpoint changes model behavior
(weaver inserts latent memory tokens into the reasoning stream).

Does NOT modify MemGen source code.
"""
import sys
import os
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MemGen"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel

BASE_MODEL_PATH = "models/Qwen2.5-1.5B-Instruct"
CHECKPOINT_PATH = "models/memgen-checkpoints/Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8/model"

# 2 GSM8K test questions (from the official test set) — kept small for CPU speed
GSM8K_QUESTIONS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "18"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "3"
    },
]

PROMPT_TEMPLATE = (
    "Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \\boxed{{}}.\n"
    "Question: {question}"
)


def _remap_adapter_key(key: str, adapter_name: str) -> str:
    """Remap LoRA checkpoint key from adapter_name='default' to named adapter.

    Checkpoint keys: ...lora_A.weight / ...lora_B.weight
    Model expects:   ...lora_A.<adapter_name>.weight / ...lora_B.<adapter_name>.weight
    """
    for lora_part in [".lora_A.weight", ".lora_B.weight"]:
        if key.endswith(lora_part):
            # .lora_A.weight -> .lora_A.<adapter_name>.weight
            middle = lora_part.replace(".weight", "")  # ".lora_A"
            return key[: -len(lora_part)] + f"{middle}.{adapter_name}.weight"
    return key


def build_prompt(tokenizer, question: str) -> str:
    content = PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text


def load_base_model_and_tokenizer():
    print("=" * 70)
    print("Loading base model (Qwen2.5-1.5B-Instruct) on CPU, fp32")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float32
    )
    model.eval()
    print(f"  Base model loaded in {time.time()-t0:.1f}s")
    print(f"  Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, tokenizer


def run_base_model(model, tokenizer, questions, max_new_tokens=100):
    print("\n" + "=" * 70)
    print("Model A: Base Qwen2.5-1.5B-Instruct (no MemGen)")
    print("=" * 70)

    results = []
    for i, q in enumerate(questions):
        prompt_text = build_prompt(tokenizer, q["question"])
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        elapsed = time.time() - t0

        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_tokens = generated_ids.shape[0]

        print(f"\n  Q{i+1}: {q['question'][:60]}...")
        print(f"  Ground truth: {q['answer']}")
        print(f"  Generated ({new_tokens} tokens, {elapsed:.1f}s):")
        print(f"  {generated_text[:300]}")

        results.append({
            "question": q["question"],
            "answer": q["answer"],
            "generated": generated_text,
            "tokens": new_tokens,
            "time": elapsed,
        })

    return results


def load_memgen_model(base_model, tokenizer):
    print("\n" + "=" * 70)
    print("Loading MemGen model with checkpoint (Base + Weaver LoRA)")
    print("=" * 70)

    # Read checkpoint config to get the right MemGen settings
    memgen_config = MemGenConfig.from_pretrained(
        BASE_MODEL_PATH,
        max_prompt_aug_num=1,
        max_inference_aug_num=3,
        prompt_latents_len=8,
        inference_latents_len=8,
        weaver_lora_config={
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        trigger_active=False,
        trigger_lora_config={
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
    )

    # Clone base model for reasoner, weaver, trigger
    print("  Cloning base model for reasoner/weaver/trigger...")
    t0 = time.time()
    reasoner = copy.deepcopy(base_model)
    weaver = copy.deepcopy(base_model)
    trigger = copy.deepcopy(base_model)
    print(f"  Cloned in {time.time()-t0:.1f}s")

    # Build MemGenModel
    print("  Building MemGenModel...")
    t0 = time.time()
    model = MemGenModel(
        config=memgen_config,
        base_tokenizer=tokenizer,
        reasoner_base_model=reasoner,
        weaver_base_model=weaver,
        trigger_base_model=trigger,
    )
    print(f"  MemGenModel built in {time.time()-t0:.1f}s")

    # Load checkpoint weights
    print(f"  Loading checkpoint from {CHECKPOINT_PATH}...")
    t0 = time.time()

    # 1. Load projection layers
    proj_path = os.path.join(CHECKPOINT_PATH, "projs.bin")
    proj_state = torch.load(proj_path, map_location="cpu", weights_only=True)
    model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
    model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])
    print(f"    Projection layers loaded")

    # 2. Load weaver state (query latents, layer norm, scale)
    weaver_path = os.path.join(CHECKPOINT_PATH, "weaver.bin")
    weaver_state = torch.load(weaver_path, map_location="cpu", weights_only=True)
    model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
    model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
    model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
    model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
    model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
    model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])
    print(f"    Weaver state loaded")

    # 3. Load trigger state
    trigger_path = os.path.join(CHECKPOINT_PATH, "trigger.bin")
    trigger_state = torch.load(trigger_path, map_location="cpu", weights_only=True)
    model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])
    print(f"    Trigger state loaded")

    # 4. Load weaver LoRA adapter weights
    # Checkpoint was saved with adapter_name="default" (keys: ...lora_A.weight),
    # but MemGenModel.__init__ created PeftModel with adapter_name="weaver" (keys: ...lora_A.weaver.weight).
    # We remap the keys and load into the existing PeftModel.
    from safetensors.torch import load_file as safe_load_file

    weaver_adapter_path = os.path.join(CHECKPOINT_PATH, "weaver", "weaver", "adapter_model.safetensors")
    weaver_ckpt_sd = safe_load_file(weaver_adapter_path, device="cpu")

    # Build target state dict: only LoRA keys, remapped to correct adapter name
    weaver_model_sd = model.weaver.model.state_dict()
    weaver_final = {}
    for k, v in weaver_ckpt_sd.items():
        new_key = _remap_adapter_key(k, "weaver")
        if new_key in weaver_model_sd:
            weaver_final[new_key] = v

    missing, unexpected = model.weaver.model.load_state_dict(weaver_final, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    print(f"    Weaver LoRA adapter loaded: {len(weaver_final)}/{len(weaver_ckpt_sd)} keys matched, "
          f"{len(lora_missing)} lora keys missing")
    if lora_missing:
        print(f"    WARNING: Missing LoRA keys: {lora_missing[:3]}...")

    # 5. Load trigger LoRA adapter weights
    trigger_adapter_path = os.path.join(CHECKPOINT_PATH, "trigger", "trigger", "adapter_model.safetensors")
    trigger_ckpt_sd = safe_load_file(trigger_adapter_path, device="cpu")

    trigger_model_sd = model.trigger.model.state_dict()
    trigger_final = {}
    for k, v in trigger_ckpt_sd.items():
        new_key = _remap_adapter_key(k, "trigger")
        if new_key in trigger_model_sd:
            trigger_final[new_key] = v

    missing, unexpected = model.trigger.model.load_state_dict(trigger_final, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    print(f"    Trigger LoRA adapter loaded: {len(trigger_final)}/{len(trigger_ckpt_sd)} keys matched, "
          f"{len(lora_missing)} lora keys missing")
    if lora_missing:
        print(f"    WARNING: Missing LoRA keys: {lora_missing[:3]}...")

    model.eval()
    print(f"  Checkpoint loaded in {time.time()-t0:.1f}s")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.1f}M")
    print(f"  Trainable params (LoRA + projections): {trainable_params/1e6:.1f}M")

    return model


def run_memgen_model(model, tokenizer, questions, max_new_tokens=100):
    print("\n" + "=" * 70)
    print("Model B: MemGen (Base + Weaver LoRA + Latent Memory)")
    print("=" * 70)

    results = []
    for i, q in enumerate(questions):
        prompt_text = build_prompt(tokenizer, q["question"])
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
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

        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        new_tokens = generated_ids.shape[0]

        print(f"\n  Q{i+1}: {q['question'][:60]}...")
        print(f"  Ground truth: {q['answer']}")
        print(f"  Generated ({new_tokens} tokens, {elapsed:.1f}s):")
        print(f"  {generated_text[:300]}")

        results.append({
            "question": q["question"],
            "answer": q["answer"],
            "generated": generated_text,
            "tokens": new_tokens,
            "time": elapsed,
        })

    return results


def compare_results(base_results, memgen_results):
    print("\n" + "=" * 70)
    print("A/B Comparison Summary")
    print("=" * 70)

    same_count = 0
    diff_count = 0

    for i, (a, b) in enumerate(zip(base_results, memgen_results)):
        a_text = a["generated"].strip()
        b_text = b["generated"].strip()
        is_same = a_text == b_text

        if is_same:
            same_count += 1
        else:
            diff_count += 1

        status = "SAME" if is_same else "DIFFERENT"
        print(f"\n  Q{i+1}: [{status}]")
        print(f"    Ground truth: {a['answer']}")
        print(f"    Base  ({a['tokens']:3d} tokens, {a['time']:.1f}s): {a_text[:120]}...")
        print(f"    MemGen ({b['tokens']:3d} tokens, {b['time']:.1f}s): {b_text[:120]}...")

    print(f"\n  Summary: {diff_count} different, {same_count} same out of {len(base_results)} questions")
    print(f"  Conclusion: {'MemGen LoRA loading confirmed - outputs differ from base model' if diff_count > 0 else 'WARNING: outputs identical - checkpoint may not be loaded correctly'}")


if __name__ == "__main__":
    total_start = time.time()

    # 1. Load base model
    base_model, tokenizer = load_base_model_and_tokenizer()

    # 2. Run base model (Model A)
    base_results = run_base_model(base_model, tokenizer, GSM8K_QUESTIONS)

    # 3. Load MemGen model with checkpoint (Model B)
    memgen_model = load_memgen_model(base_model, tokenizer)

    # Free base model memory
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 4. Run MemGen model (Model B)
    memgen_results = run_memgen_model(memgen_model, tokenizer, GSM8K_QUESTIONS)

    # 5. Compare
    compare_results(base_results, memgen_results)

    total_time = time.time() - total_start
    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 70)
