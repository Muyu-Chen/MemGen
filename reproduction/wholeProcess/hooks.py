"""Non-invasive execution tracer for MemGen inference.

Everything here *observes*: forward hooks, forward pre-hooks, and thin wrappers
that call the original function and then record. No MemGen algorithm code is
modified, replaced, or re-implemented.

Design rule: every number in the logs must come from the real execution, so we
attach to the real modules rather than recomputing anything for comparison.
"""

import gc
import json
import os

import torch
import torch.nn as nn

from memgen.model.modeling_memgen import MemGenModel
from memgen.model.modeling_utils import MemGenGenerationMixin
from memgen.model.weaver import MemGenWeaver
from memgen.model.trigger import MemGenTrigger


# --------------------------------------------------------------------------- #
# tensor statistics (shapes + summary numbers, never full dumps)
# --------------------------------------------------------------------------- #

def tensor_stats(t, name=None, extra=None):
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return {"non_tensor_type": type(t).__name__}
    tf = t.detach().float()
    out = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "numel": int(t.numel()),
        "requires_grad": bool(t.requires_grad),
        "data_ptr": t.data_ptr(),
        "mean": round(float(tf.mean()), 6) if t.numel() else None,
        "std": round(float(tf.std()), 6) if t.numel() > 1 else None,
        "norm": round(float(tf.norm()), 6) if t.numel() else None,
        "min": round(float(tf.min()), 6) if t.numel() else None,
        "max": round(float(tf.max()), 6) if t.numel() else None,
    }
    if t.numel() and t.ndim >= 2 and t.shape[-1] > 1:
        out["per_row_l2_norm"] = [round(float(v), 6) for v in tf.norm(dim=-1).reshape(-1)[:12]]
    if extra:
        out.update(extra)
    return out


def tensor_stats_list(ts, name=None):
    return [tensor_stats(t, f"{name}[{i}]") for i, t in enumerate(ts)]


def cosine_between_slots(t):
    tf = t.detach().float().reshape(-1, t.shape[-1])
    if tf.shape[0] < 2:
        return None
    n = torch.nn.functional.normalize(tf, dim=-1)
    sim = n @ n.T
    off = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return round(float(sim[off].mean()), 6)


def close(a, b, atol=1e-6, rtol=1e-5):
    """dtype-safe equality check (bf16 vs fp32 comparisons otherwise raise)."""
    if a is None or b is None:
        return False
    if a.shape != b.shape:
        return False
    return bool(torch.allclose(a.detach().float(), b.detach().float(), atol=atol, rtol=rtol))


def rss_mib():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 2**20, 1)
    except Exception:
        return None


def _desc(x, depth=0):
    if depth > 4:
        return "..."
    if isinstance(x, torch.Tensor):
        return {"tensor": list(x.shape), "dtype": str(x.dtype).replace("torch.", "")}
    if isinstance(x, dict):
        return {str(k): _desc(v, depth + 1) for k, v in list(x.items())[:14]}
    if isinstance(x, (list, tuple)):
        return [_desc(v, depth + 1) for v in x[:8]]
    cn = x.__class__.__name__
    if cn in ("DynamicCache", "Cache", "EncoderDecoderCache"):
        return {"cache_class": cn,
                "seq_len": int(x.get_seq_length()) if hasattr(x, "get_seq_length") else None,
                "n_layers": len(getattr(x, "key_cache", []))}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return {"object": cn}


# --------------------------------------------------------------------------- #
# 1. module-call census -> "which modules actually ran during this inference"
# --------------------------------------------------------------------------- #

class ModuleCensus:
    """Counts forward invocations of every nn.Module reachable from `root`.

    Ownership mapping: a parameter is "used in this inference" iff the nearest
    ancestor module that owns it was invoked at least once. Measured, not inferred.
    """

    def __init__(self, root: nn.Module):
        self.root = root
        self.counts = {}
        self.signatures = {}
        self._handles = []

    def attach(self):
        for name, mod in self.root.named_modules():
            if not name:
                continue
            self._handles.append(mod.register_forward_hook(self._make(name)))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make(self, name):
        def hook(mod, args, output):
            self.counts[name] = self.counts.get(name, 0) + 1
            if name not in self.signatures:
                self.signatures[name] = {
                    "class": type(mod).__name__,
                    "n_params": sum(p.numel() for p in mod.parameters(recurse=True)),
                    "input": _desc(args[0] if len(args) == 1 else (args if len(args) else None)),
                }
        return hook

    def ran(self, prefix):
        """True if any module whose name starts with `prefix` was called."""
        return any(k.startswith(prefix) for k in self.counts)

    def calls(self, prefix):
        return {k: v for k, v in self.counts.items() if k.startswith(prefix)}


# --------------------------------------------------------------------------- #
# 2. LoRA delta census (parameter-level evidence of trained vs untrained)
# --------------------------------------------------------------------------- #

def lora_delta_census(peft_model: nn.Module):
    """For every LoRA target module report ||B@A|| and scaling; zero => no effect."""
    rows = {}
    for name, mod in peft_model.named_modules():
        cls = type(mod).__name__
        if cls not in ("Linear",):
            continue
        if not hasattr(mod, "lora_A"):
            continue
        for adapter, loraA in mod.lora_A.items():
            loraB = mod.lora_B[adapter]
            A = loraA.weight.detach().float()
            B = loraB.weight.detach().float()
            delta = B @ A
            try:
                scaling = mod.scaling[adapter]
            except Exception:
                scaling = None
            key = name
            rows[key] = {
                "adapter": adapter,
                "r": int(A.shape[0]),
                "lora_A_shape": list(A.shape),
                "lora_B_shape": list(B.shape),
                "scaling": float(scaling) if scaling is not None else None,
                "A_absmax": round(float(A.abs().max()), 6),
                "B_absmax": round(float(B.abs().max()), 6),
                "delta_frobenius_norm": round(float(delta.norm()), 6),
                "delta_is_all_zero": bool(delta.abs().max().item() == 0.0),
            }
    nz = [k for k, v in rows.items() if not v["delta_is_all_zero"]]
    return {
        "n_lora_target_modules": len(rows),
        "n_modules_with_nonzero_delta": len(nz),
        "n_modules_with_zero_delta": len(rows) - len(nz),
        "all_zero_delta": len(nz) == 0,
        "example_zero_delta_module": next((k for k, v in rows.items() if v["delta_is_all_zero"]), None),
        "example_nonzero_delta_module": nz[0] if nz else None,
        "modules": rows,
    }


def nearest_vocabulary_embedding(vec, embedding_table, k=3):
    """Is this vector the embedding of some real token? Answers 'do latents have token IDs'."""
    with torch.no_grad():
        v = vec.detach().float()
        v = v.reshape(-1, v.shape[-1])
        E = embedding_table.detach().float()
        vn = torch.nn.functional.normalize(v, dim=-1)
        En = torch.nn.functional.normalize(E, dim=-1)
        sim = vn @ En.T                       # [n_latent, vocab]
        top = sim.topk(k, dim=-1)
        return {
            "per_latent_best_cos_sim": [round(float(x), 6) for x in top.values[:, 0]],
            "per_latent_best_token_id": [int(x) for x in top.indices[:, 0]],
            "max_cos_sim_over_all": round(float(sim.max()), 6),
            "is_exact_embedding_of_some_token": bool(
                (sim.max() > 0.9999).item()),
        }


# --------------------------------------------------------------------------- #
# 3. the tracer: wraps the four functions that own the whole inference
# --------------------------------------------------------------------------- #

class MemGenTracer:
    def __init__(self, model: MemGenModel):
        self.model = model
        self.events = []
        self.weaver_calls = []
        self.trigger_calls = []
        self.reasoner_calls = []
        self.augmentations = []      # one entry per accepted augmentation (prompt or inference)
        self.token_steps = []
        self.persistence_checks = []
        self.used_param_ids = set()  # id() of Parameters proven read by the traced code
        self._seq = 0
        self._handles = []
        self._patched = []
        self._r2w_n = 0
        self._w2r_n = 0
        self._pending = None         # state between _should_augment and the augmentation
        self._last_w2r_out = None    # latent in Reasoner space, awaiting insertion
        self._last_r2w_in_len = None # sequence length fed to Weaver just before insertion
        self._latent_spans = []      # (label, start, end, tensor) inside current_inputs_embeds
        self._active_span = None
        self._reasoner_now = {}
        self.prompt_len = None

    # -- registration ------------------------------------------------------ #

    def record(self, kind, payload):
        self._seq += 1
        self.events.append({"seq": self._seq, "kind": kind, **payload})

    def attach(self):
        tr = self
        model = self.model
        tokenizer = model.tokenizer

        # --- MemGenModel.generate (entry args only) ---
        orig_generate = MemGenModel.generate

        def generate(self_m, input_ids, attention_mask, generation_config=None,
                     return_augmentation_mask=False, **kw):
            tr._on_generate_entry(input_ids, attention_mask, generation_config)
            out = orig_generate(self_m, input_ids, attention_mask,
                                generation_config=generation_config,
                                return_augmentation_mask=return_augmentation_mask, **kw)
            ids = out[0] if isinstance(out, tuple) else out
            tr.record("generate_exit", {"output_input_ids_shape": list(ids.shape)})
            return out

        # --- _should_augment (Trigger gate) ---
        orig_should = MemGenGenerationMixin._should_augment

        def _should_augment(self_m, input_ids, sentence_augment_count=None,
                            do_sample=False, temperature=0.0, is_prompt=False):
            decision = orig_should(self_m, input_ids,
                                   sentence_augment_count=sentence_augment_count,
                                   do_sample=do_sample, temperature=temperature,
                                   is_prompt=is_prompt)
            tr._on_should_augment(input_ids, sentence_augment_count, do_sample,
                                 temperature, is_prompt, decision)
            return decision

        # --- Weaver._augment (tag which latent bank was used) ---
        orig_augment = MemGenWeaver._augment

        def _augment(self_w, latents, latent_ln, latent_scale,
                     inputs_embeds, attention_mask, position_ids):
            ctx = tr._on_weaver_entry(self_w, latents, latent_ln, latent_scale,
                                      inputs_embeds, attention_mask, position_ids)
            out = orig_augment(self_w, latents, latent_ln, latent_scale,
                               inputs_embeds, attention_mask, position_ids)
            tr._on_weaver_exit(ctx, out)
            return out

        # --- Trigger.forward ---
        orig_trigger = MemGenTrigger.forward

        def _trigger_forward(self_t, input_ids, attention_mask, position_ids):
            logits = orig_trigger(self_t, input_ids, attention_mask, position_ids)
            tr._on_trigger(self_t, input_ids, attention_mask, position_ids, logits)
            return logits

        MemGenModel.generate = generate
        MemGenGenerationMixin._should_augment = _should_augment
        MemGenWeaver._augment = _augment
        MemGenTrigger.forward = _trigger_forward
        self._patched = [
            (MemGenModel, "generate", orig_generate),
            (MemGenGenerationMixin, "_should_augment", orig_should),
            (MemGenWeaver, "_augment", orig_augment),
            (MemGenTrigger, "forward", orig_trigger),
        ]

        # --- projections ---
        self._handles.append(model.reasoner_to_weaver.register_forward_pre_hook(
            self._hook_r2w_pre, with_kwargs=True))
        self._handles.append(model.reasoner_to_weaver.register_forward_hook(
            self._hook_r2w_post, with_kwargs=True))
        self._handles.append(model.weaver_to_reasoner.register_forward_hook(
            self._hook_w2r, with_kwargs=True))

        # --- reasoner (sees cache-invalidation pattern directly) ---
        self._handles.append(model.reasoner.register_forward_pre_hook(
            self._hook_reasoner_pre, with_kwargs=True))

        # --- weaver inner model: exact concatenated input + full hidden states ---
        self._handles.append(model.weaver.model.register_forward_pre_hook(
            self._hook_weaver_model_pre, with_kwargs=True))
        self._handles.append(model.weaver.model.register_forward_hook(
            self._hook_weaver_model_post, with_kwargs=True))

        # --- token-by-token generation via the real _append_one_step ---
        orig_append = MemGenGenerationMixin._append_one_step

        def _append_one_step(self_m, outputs, cur_emb, cur_am, cur_pi, cur_ids,
                             do_sample=False, temperature=0.0):
            res = orig_append(self_m, outputs, cur_emb, cur_am, cur_pi, cur_ids,
                              do_sample=do_sample, temperature=temperature)
            tr._on_append(outputs, cur_emb, res, do_sample, temperature)
            return res

        MemGenGenerationMixin._append_one_step = _append_one_step
        self._patched.append(
            (MemGenGenerationMixin, "_append_one_step", orig_append))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for owner, name, orig in self._patched:
            setattr(owner, name, orig)
        self._patched.clear()

    # -- generate / gate --------------------------------------------------- #

    def _on_generate_entry(self, input_ids, attention_mask, gc_):
        m = self.model
        tok = m.tokenizer
        row = input_ids[0]
        pad_id = tok.pad_token_id
        real = row[row != pad_id]
        self.prompt_len = int(row.numel())
        self._gen_input_ids = input_ids[0].tolist()
        self.record("generate_entry", {
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "n_pad_positions": int((row == pad_id).sum()),
            "pad_token_id": pad_id,
            "eos_token_id": tok.eos_token_id,
            "real_token_count": int(real.numel()),
            "full_input_ids": input_ids[0].tolist(),
            "full_attention_mask": attention_mask[0].tolist(),
            "first10_ids": row[:10].tolist(),
            "first10_decoded": [repr(tok.decode([int(i)])) for i in row[:10].tolist()],
            "last8_ids": row[-8:].tolist(),
            "last8_decoded": [repr(tok.decode([int(i)])) for i in row[-8:].tolist()],
            "max_inference_aug_num": m.config.max_inference_aug_num,
            "prompt_latents_len": m.config.prompt_latents_len,
            "inference_latents_len": m.config.inference_latents_len,
            "max_prompt_aug_num_in_config": m.config.max_prompt_aug_num,
            "delimiters": list(m.delimiters),
            "delimiter_token_ids": sorted(m._get_delimiter_token_ids(tok, m.delimiters)),
            "trigger_active": bool(m.trigger.active),
            "generation_config": {
                "class": type(gc_).__name__,
                "max_new_tokens": getattr(gc_, "max_new_tokens", None),
                "do_sample": getattr(gc_, "do_sample", None),
                "temperature": getattr(gc_, "temperature", None),
                "top_p": getattr(gc_, "top_p", None),
                "top_k": getattr(gc_, "top_k", None),
                "pad_token_id": getattr(gc_, "pad_token_id", None),
                "eos_token_id": getattr(gc_, "eos_token_id", None),
                "use_cache": getattr(gc_, "use_cache", None),
                "weaver_do_sample": getattr(gc_, "weaver_do_sample", "<absent>"),
                "trigger_do_sample": getattr(gc_, "trigger_do_sample", "<absent>"),
            },
        })

    def _on_should_augment(self, input_ids, count, do_sample, temperature,
                           is_prompt, decision):
        m = self.model
        tok = m.tokenizer
        d = int(decision[0].item())
        last = int(input_ids[0, -1].item())
        payload = {
            "is_prompt_position": bool(is_prompt),
            "token_seq_len": int(input_ids.shape[1]),
            "last_token_id": last,
            "last_token_text": repr(tok.decode([last])),
            "last_token_is_delimiter": last in m._get_delimiter_token_ids(tok, m.delimiters),
            "sentence_augment_count": count.flatten().tolist(),
            "max_augment_num": m.config.max_inference_aug_num,
            "do_sample": bool(do_sample),
            "temperature": temperature,
            "decision_value": d,
            "decision_meaning": {-100: "not_candidate_trigger_not_called",
                                 0: "candidate_rejected_no_insert",
                                 1: "candidate_accepted_insert"}[d],
        }
        self.record("should_augment", payload)
        self._pending = payload

    # -- trigger ----------------------------------------------------------- #

    def _on_trigger(self, trigger, input_ids, attention_mask, position_ids, logits):
        last_logits = logits[0, -1].detach().float()
        probs = torch.softmax(last_logits, dim=-1)
        entry = {
            "n": len(self.trigger_calls) + 1,
            "trigger_active_attr": bool(trigger.active),
            "input_ids_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "position_ids_shape": list(position_ids.shape),
            "position_ids_last3": position_ids[0, -3:].tolist(),
            "logits_shape": list(logits.shape),
            "logits_at_last_position": [round(float(v), 6) for v in last_logits.tolist()],
            "softmax_at_last_position": [round(float(v), 6) for v in probs.tolist()],
            "argmax": int(last_logits.argmax()),
            "code_path": "active_branch" if trigger.active else "hardcoded_branch",
            "owning_gate": "trigger.model(PeftModel) was invoked"
            if trigger.active else "trigger.model NOT invoked (zeros + logits[...,1]=1)",
        }
        self.trigger_calls.append(entry)
        self.record("trigger_forward", entry)

    # -- projections ------------------------------------------------------- #

    def _hook_r2w_pre(self, mod, args, kwargs):
        x = args[0] if args else kwargs.get("input")
        self._r2w_in = x.detach()
        self._last_r2w_in_len = int(x.shape[1])

        # ---- persistence check: are previously inserted latents still in the
        # sequence, and do they feed into this Weaver call? (README section 6)
        for label, s, e, vec in self._latent_spans:
            chk = {"checked_at_r2w_call": self._r2w_n + 1,
                   "looking_for": label,
                   "weaver_input_len": int(x.shape[1]),
                   "expected_slice": [s, e]}
            if e <= x.shape[1]:
                sl = x[0, s:e, :]
                ref = vec[0, : sl.shape[1]]
                chk["slice_shape"] = list(sl.shape)
                chk["still_present_bitwise_equal"] = bool(
                    torch.equal(sl.contiguous(), ref.contiguous()))
                chk["still_present_allclose"] = close(sl, ref)
                chk["max_abs_diff"] = round(
                    float((sl.float() - ref.float()).abs().max()), 8)
                chk["fed_into_reasoner_to_weaver"] = True
            else:
                chk["still_present_bitwise_equal"] = False
                chk["note"] = "slice beyond current sequence length"
            self.persistence_checks.append(chk)
        return None

    def _hook_r2w_post(self, mod, args, kwargs, output):
        self._r2w_n += 1
        emb_table = self.model.reasoner.get_input_embeddings().weight
        entry = {
            "n": self._r2w_n,
            "input": tensor_stats(self._r2w_in, "reasoner_side_current_inputs_embeds"),
            "weight": tensor_stats(mod.weight.detach(), "reasoner_to_weaver.weight"),
            "output": tensor_stats(output.detach(), "weaver_side_inputs_embeds"),
            "weight_absmax": round(float(mod.weight.detach().float().abs().max()), 6),
        }
        if self._r2w_n == 1 and self.prompt_len is not None:
            # At the first augmentation the Reasoner has not run yet, so the only
            # possible source for these vectors is the embedding table lookup.
            ids = torch.tensor([self._gen_input_ids], device=self._r2w_in.device)
            E = emb_table[ids[0]]
            entry["equals_embedding_table_lookup_of_prompt_ids"] = close(
                self._r2w_in[0, : self.prompt_len], E)
            entry["reasoner_forwards_completed_before_this_call"] = len(self.reasoner_calls)
            entry["vocab_similarity_check"] = nearest_vocabulary_embedding(
                self._r2w_in[0, : self.prompt_len], emb_table)
        self.record("reasoner_to_weaver", entry)
        self._r2w_out = output.detach()

    def _hook_w2r(self, mod, args, kwargs, output):
        x = args[0] if args else kwargs.get("input")
        self._w2r_n += 1
        self.record("weaver_to_reasoner", {
            "n": self._w2r_n,
            "input": tensor_stats(x.detach(), "weaver_hidden_latents"),
            "weight": tensor_stats(mod.weight.detach(), "weaver_to_reasoner.weight"),
            "weight_absmax": round(float(mod.weight.detach().float().abs().max()), 6),
            "output": tensor_stats(output.detach(), "latent_inputs_embeds_in_reasoner_space"),
            "will_be_inserted_at_embed_index": self._last_r2w_in_len,
        })
        self._last_w2r_out = output.detach()
        if self._w2r_n == 1:
            # Negative control for "do latents have token IDs?": compare the vectors
            # actually inserted into the Reasoner against every vocabulary embedding.
            self.record("latent_vs_vocabulary", {
                "which": "M0 (first inserted latent block, in Reasoner space)",
                "latent_shape": list(output.shape),
                "check": nearest_vocabulary_embedding(
                    output, self.model.reasoner.get_input_embeddings().weight),
            })
        self._active_span = (
            f"M{self._w2r_n - 1}",
            self._last_r2w_in_len,
            self._last_r2w_in_len + output.shape[1],
            output.detach(),
        )
        return None

    # -- weaver ------------------------------------------------------------ #

    def _on_weaver_entry(self, weaver, latents, latent_ln, latent_scale,
                         inputs_embeds, attention_mask, position_ids):
        which = ("prompt_query_latents" if latents is weaver.prompt_query_latents
                 else "inference_query_latents" if latents is weaver.inference_query_latents
                 else "unknown")
        # Identity-based proof of use for parameters that are read directly
        # (not through a module forward), so the module census cannot see them.
        self.used_param_ids.update({
            id(latents), id(latent_ln.weight), id(latent_ln.bias), id(latent_scale),
        })
        ctx = {
            "n": len(self.weaver_calls) + 1,
            "which": which,
            "which_ln": ("prompt_latent_ln" if latent_ln is weaver.prompt_latent_ln
                         else "inference_latent_ln"),
            "which_scale": ("prompt_latent_scale" if latent_scale is weaver.prompt_latent_scale
                            else "inference_latent_scale"),
            "K": int(latents.size(0)),
            "context_len": int(inputs_embeds.shape[1]),
            "inputs_embeds": inputs_embeds.detach(),
            "attention_mask": attention_mask.detach(),
            "position_ids": position_ids.detach(),
            "latents_raw": latents.detach(),
            "latent_ln": latent_ln,
            "latent_scale": latent_scale.detach(),
        }
        return ctx

    def _hook_weaver_model_pre(self, mod, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None and args:
            emb = args[0]
        self._wm_in_emb = emb.detach()
        self._wm_in_am = kwargs.get("attention_mask").detach()
        self._wm_in_pi = kwargs.get("position_ids").detach()
        self._wm_input_ids_given = "input_ids" in kwargs
        return None

    def _hook_weaver_model_post(self, mod, args, kwargs, output):
        self._wm_hidden = output.hidden_states[-1].detach()
        self._wm_all_hidden = [h.detach() for h in output.hidden_states] \
            if output.hidden_states else None
        self._wm_logits = None if output.logits is None else output.logits.detach()
        return None

    def _on_weaver_exit(self, ctx, out):
        latents_hidden, latents_mask, latents_pos = out
        emb = ctx["inputs_embeds"]
        K = ctx["K"]
        L = ctx["context_len"]
        hidden = self._wm_hidden

        # the exact tensors the Weaver inner model received
        wm_emb = self._wm_in_emb
        ln = ctx["latent_ln"]
        scaled = (ln(ctx["latents_raw"]) * ctx["latent_scale"])
        B = emb.shape[0]
        scaled_b = scaled.unsqueeze(0).repeat(B, 1, 1)

        entry = {
            "n": ctx["n"],
            "which_latent_bank": ctx["which"],
            "which_latent_ln": ctx["which_ln"],
            "which_latent_scale": ctx["which_scale"],
            "K": K,
            "context_len fed_to_weaver": L,
            "latent_bank_param": tensor_stats(ctx["latents_raw"], ctx["which"]),
            "latent_ln_weight_all_ones": close(
                ln.weight.detach(), torch.ones_like(ln.weight.detach())),
            "latent_ln_bias_absmax": round(
                float(ln.bias.detach().float().abs().max()), 8),
            "latent_scale_value": round(float(ctx["latent_scale"]), 6),
            "latent_norm_before_ln": [round(float(v), 6)
                                      for v in ctx["latents_raw"].float().norm(dim=-1)],
            "latent_norm_after_ln_scale": [round(float(v), 6)
                                           for v in scaled.float().norm(dim=-1)],
            "weaver_model_inputs": {
                "given_via_inputs_embeds": not self._wm_input_ids_given,
                "inputs_embeds_shape": list(wm_emb.shape),
                "expected_shape": [B, L + K, emb.shape[-1]],
                "attention_mask_shape": list(self._wm_in_am.shape),
                "position_ids_shape": list(self._wm_in_pi.shape),
                "position_ids_last_K": self._wm_in_pi[0, -K:].tolist(),
                "position_ids_first3": self._wm_in_pi[0, :3].tolist(),
                "attention_mask_last_K": self._wm_in_am[0, -K:].tolist(),
            },
            "concat_layout": {
                "order": "[context, query_latents]"
                if close(wm_emb[:, :L, :], emb) and close(wm_emb[:, L:, :], scaled_b, atol=1e-5)
                else "NEITHER_[context,query]_NOR_[query,context]",
                "head_equals_projected_context": close(wm_emb[:, :L, :], emb),
                "tail_equals_normed_query_latents": close(wm_emb[:, L:, :], scaled_b, atol=1e-5),
                "alternative_order_q_first_matches": close(wm_emb[:, :K, :], scaled_b, atol=1e-5)
                and close(wm_emb[:, K:, :], emb),
            },
            "weaver_output": {
                "hidden_states_layers": len(self._wm_all_hidden) if self._wm_all_hidden else None,
                "final_hidden_shape": list(hidden.shape),
                "logits_shape": list(self._wm_logits.shape) if self._wm_logits is not None else None,
                "logits_consumed_by_caller": False,
            },
            "extraction": {
                "returned_shape": list(latents_hidden.shape),
                "equals_final_hidden_last_K": close(latents_hidden, hidden[:, -K:, :]),
                "equals_final_hidden_first_K": close(latents_hidden, hidden[:, :K, :]),
                "code": "weaver.py:91-92 hidden_states[:, -latents_num:, :]",
            },
            "latent_stats": tensor_stats(latents_hidden, "weaver_hidden_latents"),
            "latent_cosine_between_slots": cosine_between_slots(latents_hidden),
            "returned_latents_mask": latents_mask[0].tolist() if latents_mask.ndim > 1
            else latents_mask.tolist(),
            "returned_latent_position_ids": latents_pos[0].tolist(),
        }
        self.weaver_calls.append(entry)
        self.record("weaver_augment", {k: v for k, v in entry.items()})

    # -- reasoner ---------------------------------------------------------- #

    def _hook_reasoner_pre(self, mod, args, kwargs):
        n = len(self.reasoner_calls) + 1
        emb = kwargs.get("inputs_embeds")
        pkv = kwargs.get("past_key_values")
        entry = {
            "n": n,
            "given_via": "inputs_embeds" if emb is not None else "input_ids",
            "inputs_embeds_shape": list(emb.shape) if emb is not None else None,
            "attention_mask_shape": list(kwargs["attention_mask"].shape)
            if kwargs.get("attention_mask") is not None else None,
            "attention_mask_sum": int(kwargs["attention_mask"].sum())
            if kwargs.get("attention_mask") is not None else None,
            "position_ids_shape": list(kwargs["position_ids"].shape)
            if kwargs.get("position_ids") is not None else None,
            "position_ids_last5": kwargs["position_ids"][0, -5:].tolist()
            if kwargs.get("position_ids") is not None else None,
            "use_cache": kwargs.get("use_cache"),
            "past_key_values_none": pkv is None,
            "past_kv_seq_len": int(pkv.get_seq_length()) if pkv is not None else None,
        }
        self.reasoner_calls.append(entry)
        self._reasoner_now = entry
        # discriminator: the bulk escape hatch at modeling_memgen.py:612-624 is the
        # only Reasoner call built with use_cache=False.
        in_bulk = kwargs.get("use_cache") is False
        self._in_bulk = in_bulk
        tag = "reasoner_forward_bulk_path" if in_bulk else "reasoner_forward"
        if len(self.reasoner_calls) <= 30 or not entry["past_key_values_none"]:
            self.record(tag, entry)
        return None

    # -- token step -------------------------------------------------------- #

    def _on_append(self, outputs, cur_emb_before, res, do_sample, temperature):
        new_emb, new_am, new_pi, new_ids = res
        tok = self.model.tokenizer
        tid = int(new_ids[0, -1].item())

        # commit the span of the latent block that was inserted just before this step
        committed = None
        if self._active_span is not None:
            label, s, e, vec = self._active_span
            assert e <= new_emb.shape[1], "latent span exceeds accumulated length"
            self._latent_spans.append(self._active_span)
            committed = {"label": label, "start": s, "end": e}
            self._active_span = None

        step = {
            "n": len(self.token_steps) + 1,
            "generated_token_id": tid,
            "generated_token_text": repr(tok.decode([tid])),
            "cumulative_input_ids_len": int(new_ids.shape[1]),
            "cumulative_embeds_len": int(new_emb.shape[1]),
            "embeds_minus_ids": int(new_emb.shape[1]) - int(new_ids.shape[1]),
            "position_id_of_new_token": int(new_pi[0, -1].item()),
            "reasoner_logits_shape": list(outputs.logits.shape),
            "do_sample": bool(do_sample),
            "was_reasoner_input_len": (self._reasoner_now["inputs_embeds_shape"] or [None])[1]
            if self._reasoner_now.get("inputs_embeds_shape") else None,
            "reasoner_used_cache": not self._reasoner_now.get("past_key_values_none", True),
            "latent_block_inserted_before_this_step": committed,
        }
        self.token_steps.append(step)
        self.record("token_step", step)


    # -- derived summaries ------------------------------------------------- #

    def augmentation_log(self):
        """Pair each accepted gate decision with the Weaver call it caused."""
        augs = []
        wi = 0
        for ev in self.events:
            if ev["kind"] == "should_augment" and ev["decision_value"] == 1:
                wi += 1
                augs.append({
                    "augmentation_index": wi - 1,
                    "is_prompt_augmentation": ev["is_prompt_position"],
                    "gate_event_seq": ev["seq"],
                    "trigger_called": True,
                    "trigger_entry": self.trigger_calls[len(self.trigger_calls) - 1]
                    if self.trigger_calls else None,
                    "weaver_call": self.weaver_calls[wi - 1] if wi - 1 < len(self.weaver_calls) else None,
                    "reasoner_call_after": self.reasoner_calls[len(self.reasoner_calls) - 1]
                    if self.reasoner_calls else None,
                })
        return augs

    def final_layout(self):
        """Reconstruct the accumulated Reasoner sequence as labelled segments."""
        segs = [{"segment": "prompt_tokens", "start": 0, "end": self.prompt_len}]
        for label, s, e, _ in self._latent_spans:
            segs.append({"segment": label, "kind": "latent", "start": s, "end": e,
                         "length": e - s})
        return sorted(segs, key=lambda x: x["start"])
