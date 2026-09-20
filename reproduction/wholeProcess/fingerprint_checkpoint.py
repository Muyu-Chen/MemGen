"""Fingerprint the released MemGen checkpoint: which tensors still sit at their
PyTorch initialisation value, and which clearly moved during training.

'Present in a checkpoint file' is NOT evidence of training. nn.Linear uses
kaiming_uniform_ with bound 1/sqrt(fan_in), and PEFT initialises lora_B to zero,
so both leave a checkable signature.

Run:  ../../.venv/Scripts/python.exe fingerprint_checkpoint.py
"""

import json
import math
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
CKPT = os.path.join(ROOT, "models", "memgen-checkpoints", "Qwen2.5-1.5B-Instruct",
                    "gsm8k", "weaver-sft", "pn=1_pl=8_in=3_il=8", "model")
HIDDEN = 1536


def linear_init_bound(fan_in):
    return 1.0 / math.sqrt(fan_in)


def probe_linear(tag, sd, fan_in):
    bound = linear_init_bound(fan_in)
    w, b = sd["weight"].float(), sd["bias"].float()
    return {
        "tensor": tag,
        "weight_shape": list(sd["weight"].shape),
        "weight_absmax": round(float(w.abs().max()), 6),
        "expected_init_absmax_bound": round(bound, 6),
        "weight_std": round(float(w.std()), 6),
        "expected_init_std": round(bound / math.sqrt(3), 6),
        "bias_absmax": round(float(b.abs().max()), 6),
        "exceeds_init_support": bool(float(w.abs().max()) > bound + 1e-5
                                     or float(b.abs().max()) > bound + 1e-5),
        "verdict": "moved_from_init" if float(w.abs().max()) > bound + 1e-5 else "at_init",
    }


def main():
    out = {"checkpoint_path": CKPT, "hidden_size": HIDDEN}

    proj = torch.load(os.path.join(CKPT, "projs.bin"), map_location="cpu", weights_only=True)
    out["projections"] = [probe_linear(k, proj[k], HIDDEN)
                          for k in ("reasoner_to_weaver", "weaver_to_reasoner")]

    trg = torch.load(os.path.join(CKPT, "trigger.bin"), map_location="cpu", weights_only=True)
    out["trigger_head"] = probe_linear("output_layer", trg["output_layer"], HIDDEN)

    wv = torch.load(os.path.join(CKPT, "weaver.bin"), map_location="cpu", weights_only=True)
    ln, lat, scl = {}, {}, {}
    for k, v in wv.items():
        if not torch.is_tensor(v):        # latent_ln entries are nested state_dicts
            continue
        v = v.float()
        if "query_latents" in k:
            lat[k] = {
                "shape": list(wv[k].shape), "std": round(float(v.std()), 6),
                "mean": round(float(v.mean()), 6),
                "row_norm_mean": round(float(v.norm(dim=-1).mean()), 3),
                "expected_row_norm_for_randn": round(math.sqrt(HIDDEN), 3),
                "note": "nn.Parameter(torch.randn(K, hidden)) => indistinguishable from init "
                        "by these statistics alone",
            }
        if "latent_scale" in k:
            scl[k] = {"value": round(float(v), 8), "init_value": 1.0,
                      "at_init": bool(float(v) == 1.0)}
    for k in ("prompt_latent_ln", "inference_latent_ln"):
        W, B = wv[k]["weight"].float(), wv[k]["bias"].float()
        ln[k] = {
            "weight_equals_one_exactly": bool(torch.all(W == 1.0)),
            "weight_max_deviation_from_one": round(float((W - 1).abs().max()), 6),
            "bias_absmax": round(float(B.abs().max()), 6),
            "init_values": "weight=1, bias=0 (nn.LayerNorm default)",
            "verdict": ("moved_from_init" if float(B.abs().max()) > 0
                        or float((W - 1).abs().max()) > 0 else "at_init"),
        }
    out["weaver_latent_ln"] = ln
    out["weaver_query_latents"] = lat
    out["weaver_latent_scale"] = scl

    from safetensors.torch import load_file
    lora = {}
    for comp in ("weaver", "trigger"):
        sd = load_file(os.path.join(CKPT, comp, comp, "adapter_model.safetensors"))
        bsz = [float(v.float().abs().max()) for k, v in sd.items() if "lora_B" in k]
        asz = [float(v.float().abs().max()) for k, v in sd.items() if "lora_A" in k]
        lora[comp] = {
            "n_tensors": len(sd),
            "n_lora_B_tensors": len(bsz),
            "lora_B_all_zero": bool(max(bsz) == 0.0),
            "lora_B_absmax": round(max(bsz), 8),
            "lora_A_absmax_mean": round(sum(asz) / len(asz), 6),
            "init_convention": "PEFT zero-inits lora_B => all-zero lora_B means the adapter "
                               "never received a gradient update",
            "verdict": "never_trained" if max(bsz) == 0.0 else "trained",
        }
    out["lora_adapters"] = lora

    path = os.path.join(HERE, "logs", "checkpoint_fingerprints.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
