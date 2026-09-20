#!/usr/bin/env python
"""Run ONE official MemGen inference and record it at tensor / module / parameter level.

Faithful to the repository's real inference path:

    MemGenRunner.evaluate()            memgen/runner.py:223-234   (model.to(bfloat16))
      -> _static_evaluate()            memgen/runner.py:265-287   (apply_chat_template + run_agent_loop)
        -> SingleTurnInteractionManager.run_agent_loop()           interactions/singleturn_interaction.py:85-119
          -> MemGenModel.generate()    memgen/model/modeling_memgen.py:498-675

Config comes from the released checkpoint's own config.json (memgen/model/configuration_memgen.py),
NOT from configs/latent_memory/gsm8k.yaml, because the two disagree (see CORRECTIONS_NEEDED.md).

Nothing about the algorithm is reimplemented here; hooks.py only observes.

Usage (CPU):
    ../.venv/Scripts/python.exe trace_memgen.py --max-new-tokens 256
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))          # .../Reproduct
REPO = os.environ.get("MEMGEN_REPO", os.path.join(ROOT, "MemGen"))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

BASE_MODEL = os.environ.get(
    "MEMGEN_BASE_MODEL", os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
CKPT = os.environ.get(
    "MEMGEN_CKPT",
    os.path.join(ROOT, "models", "memgen-checkpoints", "Qwen2.5-1.5B-Instruct",
                 "gsm8k", "weaver-sft", "pn=1_pl=8_in=3_il=8", "model"))

LOGS = os.path.join(HERE, "logs")
RESULTS = os.path.join(HERE, "results")

_trace_lines = []


def log(msg=""):
    print(msg, flush=True)
    _trace_lines.append(str(msg))


def jdump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    log(f"[written] {path}")


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #

def collect_environment(extra=None):
    import torch
    import transformers
    import peft

    def git(*a):
        try:
            return subprocess.run(["git"] + list(a), cwd=REPO, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception as e:
            return f"<unavailable: {e}>"

    try:
        import psutil
        vm = psutil.virtual_memory()
        ram = {"total_gib": round(vm.total / 2**30, 2), "available_gib": round(vm.available / 2**30, 2)}
    except Exception:
        ram = None

    env = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "repo_path": REPO,
            "commit": git("rev-parse", "HEAD"),
            "commit_subject": git("log", "-1", "--pretty=%s"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_dirty": git("status", "--porcelain") != "",
            "working_tree_changes": git("status", "--porcelain").splitlines(),
            "remotes": git("remote", "-v").splitlines(),
        },
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "cpu_name": _cpu_name(),
            "physical_threads": os.cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "ram": ram,
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "platform": platform.platform(),
        },
        "model_sources": {
            "base_model_local_path": BASE_MODEL,
            "base_model_hub_id_in_official_yaml": "Qwen/Qwen2.5-1.5B-Instruct",
            "checkpoint_local_path": CKPT,
            "checkpoint_hub_repo": "Kana-s/MemGen",
            "config_authority": "checkpoint config.json (MemGenConfig.from_pretrained)",
        },
    }
    if extra:
        env.update(extra)
    return env


def _cpu_name():
    try:
        import psutil
        return os.environ.get("PROCESSOR_IDENTIFIER", psutil.cpu_freq() and "unknown")
    except Exception:
        return os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


# --------------------------------------------------------------------------- #
# model construction  (mirrors MemGenModel.from_config + runner.evaluate)
# --------------------------------------------------------------------------- #

def build_model(dtype, attn_impl):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from memgen.model.configuration_memgen import MemGenConfig
    from memgen.model.modeling_memgen import MemGenModel

    log("== build: MemGenConfig.from_pretrained(checkpoint) ==")
    cfg = MemGenConfig.from_pretrained(CKPT)
    for k in ("prompt_latents_len", "inference_latents_len", "max_prompt_aug_num",
              "max_inference_aug_num", "trigger_active"):
        log(f"   config.{k} = {getattr(cfg, k)!r}")
    log(f"   config.weaver_lora_config = {cfg.weaver_lora_config}")
    log(f"   config.trigger_lora_config = {cfg.trigger_lora_config}")
    log(f"   config.torch_dtype(json) = {getattr(cfg, 'torch_dtype', None)}")

    log("== build: tokenizer (official from_config uses the BASE model tokenizer) ==")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def load(tag):
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=dtype, attn_implementation=attn_impl)
        log(f"   {tag}: {type(m).__name__} loaded in {time.time()-t0:.1f}s "
            f"params={sum(p.numel() for p in m.parameters()):,}")
        return m

    log("== build: three INDEPENDENT base-model instances "
        "(official from_config does exactly three from_pretrained calls, modeling_memgen.py:715-717) ==")
    reasoner = load("reasoner")
    gc.collect()
    weaver = load("weaver")
    gc.collect()
    trigger = load("trigger")
    gc.collect()

    log("== build: MemGenModel.from_pretrained (official loader) ==")
    model = MemGenModel.from_pretrained(
        CKPT, config=cfg, base_tokenizer=tokenizer,
        reasoner_base_model=reasoner, weaver_base_model=weaver, trigger_base_model=trigger)

    # official: memgen/runner.py:225  ->  self.model = self.model.to(torch.bfloat16)
    model = model.to(dtype)
    model.eval()
    gc.collect()
    return model, cfg, tokenizer


def share_report(model):
    """Are Reasoner / Weaver / Trigger the same objects, or independent instances?"""
    import torch
    r, w, t = model.reasoner, model.weaver.model, model.trigger.model

    def strip_prefix(d):
        out = {}
        for k, v in d.items():
            k2 = k[len("base_model.model."):] if k.startswith("base_model.model.") else k
            out[k2] = v
        return out

    rp = dict(r.named_parameters())
    wp = strip_prefix(dict(w.named_parameters()))
    tp = strip_prefix(dict(t.named_parameters()))
    common_w = set(rp) & set(wp)
    common_t = set(rp) & set(tp)
    same_w = sum(1 for k in common_w if rp[k].data_ptr() == wp[k].data_ptr())
    same_t = sum(1 for k in common_t if rp[k].data_ptr() == tp[k].data_ptr())
    probe = "model.layers.0.self_attn.o_proj.weight"  # not a LoRA target: name unwrapped
    out = {
        "reasoner_class": type(r).__name__,
        "weaver_is_reasoner_object": w is r,
        "trigger_is_reasoner_object": t is r,
        "reasoner_param_names_matched_in_weaver": len(common_w),
        "of_which_share_same_memory_buffer": same_w,
        "reasoner_param_names_matched_in_trigger": len(common_t),
        "of_which_share_same_memory_buffer": same_t,
        "value_identical_at_probe": (bool(torch.equal(rp[probe], wp[probe]))
                                     if probe in rp and probe in wp else None),
        "probe": probe,
    }
    out["conclusion"] = (
        "three INDEPENDENT parameter sets (no shared storage); values coincide because "
        "each was loaded from the same base checkpoint"
        if same_w == 0 and same_t == 0 else "partial weight sharing - inspect")
    return out


# --------------------------------------------------------------------------- #
# prompt construction (runner.py:268-274 + data/gsm8k/builder.py)
# --------------------------------------------------------------------------- #

def official_prompt(question):
    try:
        from data.gsm8k.builder import GSM8KBuilder
        return GSM8KBuilder._preprocess({"question": question, "answer": ""})["prompt"], \
            "data/gsm8k/builder.py:GSM8KBuilder._preprocess (official code)"
    except Exception as e:
        fmt = r"""Solve the math problem with proper reasoning, and make sure to put the FINAL ANSWER inside \boxed{}."""
        return [{"role": "user",
                 "content": fmt + "Question: " + question.strip() + "\n"}], \
            f"inlined copy of data/gsm8k/builder.py (import failed: {e})"


def tokenize_prompt(model, tokenizer, prompt_msgs, pad_side_ok):
    import torch
    kwargs = dict(add_generation_prompt=True, return_tensors="pt", padding=True,
                  add_special_tokens=True, return_dict=True)
    try:
        enc = tokenizer.apply_chat_template(prompt_msgs, padding_side="left", **kwargs)
        how = "apply_chat_template(padding_side='left') as in runner.py:268-275"
    except TypeError as e:
        pad_side_ok["error"] = str(e)
        tokenizer.padding_side = "left"
        enc = tokenizer.apply_chat_template(prompt_msgs, **kwargs)
        how = "tokenizer.padding_side='left' then apply_chat_template (padding_side kwarg rejected)"
    return enc["input_ids"], enc["attention_mask"], how, enc


# --------------------------------------------------------------------------- #
# parameter accounting
# --------------------------------------------------------------------------- #

GROUPS = [
    ("Reasoner (all)", lambda n: n.startswith("reasoner.")),
    ("Weaver base (non-LoRA)", lambda n: n.startswith("weaver.model.") and "lora_" not in n),
    ("Weaver LoRA", lambda n: n.startswith("weaver.model.") and "lora_" in n),
    ("Weaver prompt_query_latents", lambda n: n.startswith("weaver.prompt_query_latents")
     or n.startswith("weaver.prompt_latent_ln") or n.startswith("weaver.prompt_latent_scale")),
    ("Weaver inference_query_latents", lambda n: n.startswith("weaver.inference_query_latents")
     or n.startswith("weaver.inference_latent_ln") or n.startswith("weaver.inference_latent_scale")),
    ("reasoner_to_weaver projection", lambda n: n.startswith("reasoner_to_weaver.")),
    ("weaver_to_reasoner projection", lambda n: n.startswith("weaver_to_reasoner.")),
    ("Trigger base (non-LoRA)", lambda n: n.startswith("trigger.model.") and "lora_" not in n),
    ("Trigger LoRA", lambda n: n.startswith("trigger.model.") and "lora_" in n),
    ("Trigger output_layer (head)", lambda n: n.startswith("trigger.output_layer.")),
]

CKPT_SOURCES = {
    "projs.bin": ["reasoner_to_weaver.", "weaver_to_reasoner."],
    "weaver.bin": ["weaver.prompt_query_latents", "weaver.inference_query_latents",
                   "weaver.prompt_latent_ln", "weaver.inference_latent_ln",
                   "weaver.prompt_latent_scale", "weaver.inference_latent_scale"],
    "trigger.bin": ["trigger.output_layer."],
    "weaver/weaver/adapter_model.safetensors": ["weaver.model."],
    "trigger/trigger/adapter_model.safetensors": ["trigger.model."],
}


def classify_group(name):
    for label, fn in GROUPS:
        if fn(name):
            return label
    return "other"


def owner_module(name):
    return name.rsplit(".", 1)[0] if "." in name else ""


def checkpoint_source(name):
    hits = []
    if name.startswith(("reasoner.", "weaver.model.", "trigger.model.")) and "lora_" not in name:
        hits.append("base model safetensors (Qwen2.5-1.5B-Instruct)")
    if name.startswith("weaver.model.") and "lora_" in name:
        hits.append("weaver/weaver/adapter_model.safetensors")
    if name.startswith("trigger.model.") and "lora_" in name:
        hits.append("trigger/trigger/adapter_model.safetensors")
    for f, prefixes in CKPT_SOURCES.items():
        if f.startswith(("projs", "weaver.bin", "trigger.bin")) and any(name.startswith(p) for p in prefixes):
            hits.append(f)
    return hits or ["created by nn.Module.__init__ (never written to checkpoint)"]


def parameter_report(model, census, used_ids=frozenset()):
    rows, agg = [], {}
    for name, p in model.named_parameters():
        group = classify_group(name)
        owner = owner_module(name)
        by_module = owner in census.counts
        by_identity = id(p) in used_ids
        used = by_module or by_identity
        src = checkpoint_source(name)
        from_ckpt = "base model" not in src[0]
        rec = {
            "name": name,
            "group": group,
            "shape": list(p.shape),
            "numel": p.numel(),
            "dtype": str(p.dtype).replace("torch.", ""),
            "requires_grad": bool(p.requires_grad),
            "loaded_from": src,
            "came_from_memgen_checkpoint": from_ckpt,
            "owner_module": owner,
            "owner_module_invoked": by_module,
            "owner_module_call_count": census.counts.get(owner, 0),
            "proven_read_by_identity": by_identity,
            "used_in_this_inference": used,
        }
        rows.append(rec)
        a = agg.setdefault(group, {
            "n_param_tensors": 0, "total_numel": 0,
            "requires_grad_numel": 0, "requires_grad_tensors": 0,
            "used_tensors": 0, "used_numel": 0,
            "unused_tensors": 0, "unused_numel": 0,
            "from_memgen_checkpoint_tensors": 0, "from_memgen_checkpoint_numel": 0,
            "dtypes": set(), "unused_examples": [],
        })
        a["n_param_tensors"] += 1
        a["total_numel"] += p.numel()
        a["dtypes"].add(str(p.dtype).replace("torch.", ""))
        if p.requires_grad:
            a["requires_grad_numel"] += p.numel()
            a["requires_grad_tensors"] += 1
        if from_ckpt:
            a["from_memgen_checkpoint_tensors"] += 1
            a["from_memgen_checkpoint_numel"] += p.numel()
        if used:
            a["used_tensors"] += 1
            a["used_numel"] += p.numel()
        else:
            a["unused_tensors"] += 1
            a["unused_numel"] += p.numel()
            if len(a["unused_examples"]) < 6:
                a["unused_examples"].append(name)
    for a in agg.values():
        a["dtypes"] = sorted(a["dtypes"])
    total = sum(a["total_numel"] for a in agg.values())
    unused = sum(a["unused_numel"] for a in agg.values())
    summary = {
        "total_parameters": total,
        "parameters_used_in_this_inference": total - unused,
        "parameters_loaded_but_unused": unused,
        "pct_unused": round(100.0 * unused / total, 2),
        "note": ("'used' = owning module executed a forward during this generate() call, "
                 "or the tensor object was proven read by identity (bare nn.Parameter such as "
                 "the query-latent banks). requires_grad is irrelevant to this determination."),
    }
    return rows, agg, summary


# --------------------------------------------------------------------------- #
# module map (README section 1)
# --------------------------------------------------------------------------- #

def module_map(model, census):
    def info(prefix, cls_hint=None):
        mod = model
        for part in [p for p in prefix.split(".") if p]:
            mod = getattr(mod, part, None)
            if mod is None:
                break
        calls = {k: v for k, v in census.counts.items() if k.startswith(prefix)}
        return {
            "attribute_path": prefix or "<MemGenModel>",
            "class": type(mod).__name__ if mod is not None else None,
            "source_file": None,
            "n_submodules_invoked": len(calls),
            "total_submodule_forward_calls": sum(calls.values()),
            "n_parameters": (sum(p.numel() for p in mod.parameters())
                             if hasattr(mod, "parameters") and mod is not None else 0),
            "directly_invoked": prefix in census.counts,
            "direct_call_count": census.counts.get(prefix, 0),
        }

    entries = [
        ("tokenizer", "AutoTokenizer / ChatML template overridden by memgen/utils.py:CONVERSATION_TEMPLATE"),
        ("reasoner", "memgen/model/modeling_memgen.py:104"),
        ("weaver", "memgen/model/weaver.py:MemGenWeaver"),
        ("weaver.model", "PeftModel wrapping the base LM (modeling_memgen.py:100)"),
        ("reasoner_to_weaver", "modeling_memgen.py:110 nn.Linear"),
        ("weaver_to_reasoner", "modeling_memgen.py:111 nn.Linear"),
        ("trigger", "memgen/model/trigger.py:MemGenTrigger"),
        ("trigger.model", "PeftModel wrapping the base LM (modeling_memgen.py:101)"),
        ("trigger.output_layer", "trigger.py:18 nn.Linear(hidden,2)"),
    ]
    out = []
    for path, note in entries:
        d = info(path, note)
        d["defined_at"] = note
        if path == "weaver":
            d["extra_parameters"] = {
                "prompt_query_latents": list(model.weaver.prompt_query_latents.shape),
                "inference_query_latents": list(model.weaver.inference_query_latents.shape),
                "prompt_latent_ln": "nn.LayerNorm(1536)",
                "inference_latent_ln": "nn.LayerNorm(1536)",
                "prompt_latent_scale": list(model.weaver.prompt_latent_scale.shape),
                "inference_latent_scale": list(model.weaver.inference_latent_scale.shape),
            }
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def context_sensitivity(model, input_ids):
    """Does the Weaver latent actually depend on the context it is fed?

    Runs the real MemGenWeaver.augment_prompt three times. The query-latent bank,
    its LayerNorm and its scale are the same parameters in every call, and all
    calls use the same token multiset and sequence length, so any difference in
    the returned latent is attributable to the arrangement of the context.
    """
    import torch
    E = model.reasoner.get_input_embeddings()

    def latent_for(ids):
        emb = E(ids)
        am = torch.ones((1, ids.shape[1]), dtype=torch.long)
        pi = model._generate_position_ids(am)
        return model.weaver.augment_prompt(model.reasoner_to_weaver(emb), am, pi)[0]

    def cmp(a, b):
        an = torch.nn.functional.normalize(a.float(), dim=-1)
        bn = torch.nn.functional.normalize(b.float(), dim=-1)
        return {
            "mean_row_cosine": round(float((an * bn).sum(-1).mean()), 6),
            "relative_l2_difference": round(
                float((a.float() - b.float()).norm() / a.float().norm()), 6),
            "latent_norm_a": round(float(a.float().norm()), 3),
            "latent_norm_b": round(float(b.float().norm()), 3),
        }

    base = latent_for(input_ids)
    repeat = latent_for(input_ids)
    reversed_ = latent_for(input_ids.flip(1))
    return {
        "same_input_repeat": {**cmp(base, repeat),
                              "bitwise_identical": bool(torch.equal(base, repeat))},
        "same_tokens_reversed_order": cmp(base, reversed_),
        "conclusion": "latent is context-conditioned"
        if cmp(base, reversed_)["mean_row_cosine"] < 0.99
        else "latent appears NOT to depend on context order - investigate",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="index into GSM8K test split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="bfloat16 == official (runner.py:225); float32 only for debugging")
    ap.add_argument("--attn", default="eager",
                    help="official hardcodes flash_attention_2 (modeling_memgen.py:715); unavailable on CPU")
    ap.add_argument("--dry-run", action="store_true", help="build + record env, skip generation")
    args = ap.parse_args()

    import torch
    import numpy as np
    np.random.seed(42)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    env = collect_environment(extra={
        "inference_choices": {
            "dtype_requested": args.dtype,
            "attn_implementation_requested": args.attn,
            "batch_size": 1,
            "max_new_tokens": args.max_new_tokens,
            "official_max_response_length_in_yaml": 1024,
            "deviations_from_official_code": [
                "attn_implementation: flash_attention_2 -> eager (no CUDA on this host)",
                "batch_size: 8 (interaction.batch_size in gsm8k.yaml) -> 1 (CPU runtime)",
                f"max_new_tokens: 1024 -> {args.max_new_tokens} (CPU runtime)" if args.max_new_tokens != 1024 else "none",
            ],
        }
    })
    log(f"== environment ==\n{json.dumps(env, indent=2, ensure_ascii=False)}")

    t0 = time.time()
    model, cfg, tokenizer = build_model(dtype, args.attn)
    log(f"[timing] model construction: {time.time()-t0:.1f}s  rss={__import__('hooks').rss_mib()} MiB")

    # ---- dataset sample -------------------------------------------------
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")[args.split]
    ex = ds[args.sample]
    prompt_msgs, prompt_origin = official_prompt(ex["question"])
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True)

    log("\n== Step A: prompt construction ==")
    log(f"  raw question: {ex['question']!r}")
    log(f"  prompt builder: {prompt_origin}")
    log(f"  user content  : {prompt_msgs[0]['content']!r}")
    log(f"  chat template source: memgen/model/modeling_memgen.py:137 sets tokenizer.chat_template "
        f"= CONVERSATION_TEMPLATE (memgen/utils.py)")
    log(f"  final prompt text:\n{prompt_text}")

    pad_note = {}
    ids, amask, tok_how, enc_full = tokenize_prompt(model, tokenizer, [prompt_msgs], pad_note)
    log(f"  tokenization: {tok_how}")
    log(f"  input_ids shape {list(ids.shape)}, attention_mask sum {int(amask.sum())}")

    # ---- tracing ---------------------------------------------------------
    from hooks import MemGenTracer, ModuleCensus, lora_delta_census

    census = ModuleCensus(model).attach()
    tracer = MemGenTracer(model).attach()

    shares = share_report(model)
    lora_weaver = lora_delta_census(model.weaver.model)
    lora_trigger = lora_delta_census(model.trigger.model)

    # ---- official generation config (base_interaction.py:53-61) ---------
    from transformers import GenerationConfig
    from interactions.base_interaction import InteractionConfig
    from interactions.singleturn_interaction import SingleTurnInteractionManager
    from interactions.base_interaction import InteractionDataProto

    ic = InteractionConfig(
        max_turns=1, max_start_length=1024, max_prompt_length=4096,
        max_response_length=args.max_new_tokens, max_obs_length=512,
        temperature=0.0, batch_size=1, output_dir=RESULTS,
        weaver_do_sample=False, trigger_do_sample=False)
    mgr = SingleTurnInteractionManager(tokenizer, model, ic, is_validation=True)
    gconf = mgr.generation_config
    log("\n== generation config actually used ==")
    log("  " + json.dumps({
        "max_new_tokens": gconf.max_new_tokens, "do_sample": gconf.do_sample,
        "temperature": gconf.temperature, "eos_token_id": gconf.eos_token_id,
        "pad_token_id": gconf.pad_token_id, "use_cache": gconf.use_cache,
        "weaver_do_sample": gconf.weaver_do_sample,
        "trigger_do_sample": gconf.trigger_do_sample,
    }))

    if args.dry_run:
        log("[dry-run] stopping before generation")
        return

    gb = InteractionDataProto()
    gb.batch["input_ids"] = ids
    gb.batch["attention_mask"] = amask
    gb.no_tensor_batch["initial_prompts"] = [prompt_msgs]

    log("\n== Step B onward: MemGenModel.generate ==")
    t1 = time.time()
    out = mgr.run_agent_loop(gb)
    dt = time.time() - t1

    resp_ids = out.batch["responses"][0]
    completion = tokenizer.decode(resp_ids, skip_special_tokens=True)
    log(f"\n[timing] generate: {dt:.1f}s for {len(tracer.token_steps)} traced steps")
    log(f"completion (skip_special_tokens=True):\n{completion}")

    census.detach()
    tracer.detach()

    log("\n== supplementary: is the Weaver latent context-conditioned? (real augment_prompt) ==")
    sens = context_sensitivity(model, ids)
    log(json.dumps(sens, indent=2))

    # ---- assemble --------------------------------------------------------
    rows, agg, param_summary = parameter_report(model, census, tracer.used_param_ids)
    augs = tracer.augmentation_log()
    layout = tracer.final_layout()
    env["_param_summary"] = param_summary
    env["context_sensitivity_test"] = sens

    env["runtime_realised"] = {
        "reasoner_class": type(model.reasoner).__name__,
        "weaver_class": type(model.weaver).__name__,
        "weaver_inner_model_class": type(model.weaver.model).__name__,
        "trigger_class": type(model.trigger).__name__,
        "trigger_inner_model_class": type(model.trigger.model).__name__,
        "trigger_active": bool(model.trigger.active),
        "dtype_of_reasoner_weight": str(model.reasoner.get_input_embeddings().weight.dtype),
        "dtype_of_query_latents": str(model.weaver.prompt_query_latents.dtype),
        "model_parameter_total": sum(p.numel() for p in model.parameters()),
        "model_parameter_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "prompt_token_count": int(ids.shape[1]),
        "generate_wall_seconds": round(dt, 2),
        "rss_mib_after_generate": __import__("hooks").rss_mib(),
    }
    env["three_model_instances"] = shares
    env["lora_delta_census"] = {
        "weaver": {k: v for k, v in lora_weaver.items() if k != "modules"},
        "trigger": {k: v for k, v in lora_trigger.items() if k != "modules"},
    }
    env["prompt_construction"] = {
        "raw_question": ex["question"],
        "gold_answer": ex["answer"],
        "user_content": prompt_msgs[0]["content"],
        "builder": prompt_origin,
        "chat_template": tokenizer.chat_template,
        "final_prompt_text": prompt_text,
        "tokenizer_call": tok_how,
        "padding_note": pad_note or None,
        "input_ids": ids[0].tolist(),
        "attention_mask": amask[0].tolist(),
        "decoded_tokens_with_ids": [
            {"i": i, "id": int(t), "tok": repr(tokenizer.decode([int(t)]))}
            for i, t in enumerate(ids[0].tolist())],
    }

    shapes = {
        "note": "every entry produced by the real execution; see hooks.py for the capture points",
        "events_ordered": tracer.events,
        "weaver_calls": tracer.weaver_calls,
        "trigger_calls": tracer.trigger_calls,
        "reasoner_calls": tracer.reasoner_calls,
        "token_steps": tracer.token_steps,
        "persistence_checks": tracer.persistence_checks,
        "augmentation_log": augs,
        "final_layout": layout,
        "module_forward_signatures": census.signatures,
    }
    param_usage = {
        "summary": param_summary,
        "per_parameter": rows,
        "per_group": agg,
        "module_map": module_map(model, census),
        "module_call_counts": census.counts,
        "lora_modules": {"weaver": lora_weaver["modules"], "trigger": lora_trigger["modules"]},
    }
    sample_out = {
        "sample_index": args.sample, "split": args.split,
        "question": ex["question"], "gold": ex["answer"],
        "prompt_text": prompt_text,
        "completion_token_ids": resp_ids.tolist(),
        "completion": completion,
        "n_generated_tokens_traced": len(tracer.token_steps),
        "n_prompt_augmentations": sum(1 for a in augs if a["is_prompt_augmentation"]),
        "n_inference_augmentations": sum(1 for a in augs if not a["is_prompt_augmentation"]),
        "augmentation_positions": [i for i, s in enumerate(tracer.token_steps)],
        "eos_emitted": bool(any(s["generated_token_id"] == tokenizer.eos_token_id
                                for s in tracer.token_steps)),
    }

    jdump(os.path.join(LOGS, "environment.json"), env)
    jdump(os.path.join(LOGS, "tensor_shapes.json"), shapes)
    jdump(os.path.join(LOGS, "parameter_usage.json"), param_usage)
    jdump(os.path.join(RESULTS, "sample_output.json"), sample_out)

    # ---- human readable full trace --------------------------------------
    render_full_trace(os.path.join(LOGS, "full_trace.txt"), env, shapes, agg, sample_out, augs, layout)
    log("\nDONE")
    log(json.dumps({"prompt_aug": sample_out["n_prompt_augmentations"],
                    "inference_aug": sample_out["n_inference_augmentations"],
                    "tokens": sample_out["n_generated_tokens_traced"],
                    "weaver_calls": len(tracer.weaver_calls),
                    "trigger_calls": len(tracer.trigger_calls),
                    "reasoner_calls": len(tracer.reasoner_calls),
                    "eos": sample_out["eos_emitted"]}))


def render_full_trace(path, env, shapes, agg, sample_out, augs, layout):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_trace_lines))
        f.write("\n\n")
        f.write("=" * 78 + "\nEXECUTION EVENT STREAM (ordered)\n" + "=" * 78 + "\n")
        for ev in shapes["events_ordered"]:
            k = ev["kind"]
            if k == "token_step":
                f.write(f"[{ev['seq']:4d}] token_step   #{ev['n']:3d} "
                        f"tok={ev['generated_token_text']!r:20s} ids_len={ev['cumulative_input_ids_len']} "
                        f"emb_len={ev['cumulative_embeds_len']} latents_so_far={ev['embeds_minus_ids']} "
                        f"pos_id={ev['position_id_of_new_token']} cache={ev['reasoner_used_cache']}\n")
            elif k == "should_augment":
                f.write(f"[{ev['seq']:4d}] gate           is_prompt={ev['is_prompt_position']} "
                        f"last_tok={ev['last_token_text']!r} delimiter={ev['last_token_is_delimiter']} "
                        f"-> {ev['decision_value']} ({ev['decision_meaning']})\n")
            elif k == "trigger_forward":
                f.write(f"[{ev['seq']:4d}] trigger        #{ev['n']} active={ev['trigger_active_attr']} "
                        f"logits={ev['logits_at_last_position']} softmax={ev['softmax_at_last_position']} "
                        f"argmax={ev['argmax']} path={ev['code_path']}\n")
            elif k in ("reasoner_to_weaver", "weaver_to_reasoner"):
                i, o = ev["input"], ev["output"]
                f.write(f"[{ev['seq']:4d}] {k:18s} {i['shape']} {i['dtype']} -> {o['shape']} "
                        f"mean={i['mean']}/{o['mean']} std={i['std']}/{o['std']} norm={i['norm']}/{o['norm']}\n")
            elif k == "weaver_augment":
                f.write(f"[{ev['seq']:4d}] weaver_augment #{ev['n']} bank={ev['which_latent_bank']} "
                        f"K={ev['K']} ctx_len={ev['context_len fed_to_weaver']} "
                        f"weaver_in={ev['weaver_model_inputs']['inputs_embeds_shape']} "
                        f"concat={ev['concat_layout']['order']} "
                        f"extract_lastK={ev['extraction']['equals_final_hidden_last_K']} "
                        f"extract_firstK={ev['extraction']['equals_final_hidden_first_K']} "
                        f"latent_norm={ev['latent_stats']['norm']}\n")
            elif k.startswith("reasoner_forward"):
                f.write(f"[{ev['seq']:4d}] {k:26s} #{ev['n']:3d} in={ev['inputs_embeds_shape']} "
                        f"via={ev['given_via']} attn_sum={ev['attention_mask_sum']} "
                        f"pos_last5={ev['position_ids_last5']} use_cache={ev['use_cache']} "
                        f"past_kv={ev['past_kv_seq_len'] if ev['past_kv_seq_len'] is not None else 'None'}\n")
            else:
                f.write(f"[{ev['seq']:4d}] {k}: "
                        f"{json.dumps({kk: vv for kk, vv in ev.items() if kk not in ('seq','kind')})[:900]}\n")
        f.write("\n" + "=" * 78 + "\nAUGMENTATION SUMMARY\n" + "=" * 78 + "\n")
        for a in augs:
            w = a["weaver_call"] or {}
            f.write(
                f"aug#{a['augmentation_index']} prompt={a['is_prompt_augmentation']} "
                f"bank={w.get('which_latent_bank')} ctx_len_fed_to_weaver="
                f"{w.get('context_len fed_to_weaver')} weaver_in="
                f"{w.get('weaver_model_inputs', {}).get('inputs_embeds_shape')} "
                f"concat={w.get('concat_layout', {}).get('order')} "
                f"extract_lastK={w.get('extraction', {}).get('equals_final_hidden_last_K')}\n")
        f.write("\n" + "=" * 78 + "\nPERSISTENCE CHECKS (are earlier latents still in the sequence?)\n"
                + "=" * 78 + "\n")
        for c in shapes["persistence_checks"]:
            f.write(json.dumps(c) + "\n")
        f.write("\n" + "=" * 78 + "\nFINAL SEQUENCE LAYOUT (positions in current_inputs_embeds)\n"
                + "=" * 78 + "\n")
        for s in layout:
            f.write(json.dumps(s) + "\n")
        f.write("\n" + "=" * 78 + "\nPARAMETER GROUPS\n" + "=" * 78 + "\n")
        for g, a in agg.items():
            f.write(f"{g:36s} tensors={a['n_param_tensors']:4d} numel={a['total_numel']:12,} "
                    f"requires_grad={a['requires_grad_numel']:10,} used={a['used_tensors']:4d} "
                    f"used_numel={a['used_numel']:12,} UNUSED={a['unused_tensors']:4d}/"
                    f"{a['unused_numel']:12,} from_ckpt={a['from_memgen_checkpoint_tensors']:4d} "
                    f"dtypes={a['dtypes']}\n")
            if a["unused_examples"]:
                f.write(f"{'':36s}   unused examples: {a['unused_examples']}\n")
        f.write(f"\nSUMMARY: {json.dumps(env['_param_summary'])}\n")
        f.write("\n" + "=" * 78 + "\nRESULT\n" + "=" * 78 + "\n")
        f.write(sample_out["completion"] + "\n")
    print(f"[written] {path}")


if __name__ == "__main__":
    main()
