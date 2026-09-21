"""Analyze released-gate reachability and depth-5 stability of the 0810 rescue."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from common import HARD_BENCH_ROOT, portable_path, read_jsonl, sha256_file, utc_now, write_json


RUN_ROOT = HARD_BENCH_ROOT / "runs"
RELEASED = (
    RUN_ROOT
    / "released_gate_0810_depth1_v1"
    / "scored"
    / "Z7_memgen_released_gate_0810.scored.jsonl"
)
STABILITY = (
    RUN_ROOT
    / "text_sentence_oracle_0810_depth5_rescue_branch_v1"
    / "scored"
    / "Z8_memgen_0810_rescue_stability.scored.jsonl"
)
OUTPUT = HARD_BENCH_ROOT / "analysis" / "oracle_0810_robustness_v1.json"
REPORT = HARD_BENCH_ROOT / "ORACLE_0810_ROBUSTNESS_REPORT.md"


def token_hash(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record["completion_token_ids"], separators=(",", ":")).encode()
    ).hexdigest()


def render_report(analysis: dict) -> str:
    released = analysis["released_gate"]
    stability = analysis["decoded_depth5_rescue_branch"]
    if released["invoke_correct"]:
        reachability = "发布代码的原 token-ID gate 也能到达该 exact rescue。"
    else:
        reachability = "发布代码的原 token-ID gate 未复现 decoded detector 找到的 exact rescue。"
    if stability["correct_policy_count"] == stability["policy_count"]:
        stability_text = "首节点 rescue 对后续四个动态决策的所有组合都稳定。"
    else:
        stability_text = (
            f"首节点 rescue 在 {stability['regressed_policy_count']}/"
            f"{stability['policy_count']} 条后续策略中被破坏。"
        )
    lines = [
        "# `0810` Oracle Rescue Robustness",
        "",
        f"**结论：{reachability} {stability_text}**",
        "",
        "## 发布版 gate 可达性",
        "",
        "| policy | detector | prediction | correct | tokens | actual invokes | horizon |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in released["variants"]:
        lines.append(
            f"| `{variant['policy']}` | released token-ID | {variant['prediction']} | "
            f"{variant['correct']} | {variant['generated_token_count']} | "
            f"{variant['inference_augmentations']} | {variant['horizon_reached']} |"
        )
    lines.extend(
        [
            "",
            "该对照只改变 candidate detection：同一 checkpoint、prompt augmentation、greedy decoding 与 1024-token 上限。若 policy `1` 正确，就能排除“正例只是 decoded suffix 扩展创造的非发布候选”这一替代解释。",
            "",
            "## 深度 5 rescue-branch 稳定性",
            "",
            f"固定首位为 `1`，穷举随后四位，共 {stability['policy_count']} 条动态策略。exact-correct {stability['correct_policy_count']}/{stability['policy_count']}；完整到达第五决策 {stability['full_horizon_count']}/{stability['policy_count']}；截断 {stability['truncated_count']}。",
            "",
            f"预测分布：`{stability['prediction_distribution']}`。唯一 completion 数：{stability['unique_completion_count']}。",
            "",
            "## 解释",
            "",
            "这两组不是为了再扩大总体准确率估计，而是验证一个已发现 exact-positive label 的可用性：候选是否与发布实现兼容，以及正例是否会因后续 memory 调用而翻转。它直接决定 `0810` 能否成为后续 Trigger 训练的可信监督样本。",
            "",
            f"总生成耗时：{analysis['total_latency_seconds']:.1f} 秒；policy-prefix 校验：{analysis['all_observed_prefixes_match_policies']}。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    released_records = sorted(read_jsonl(RELEASED), key=lambda record: record["policy_string"])
    stability_records = sorted(read_jsonl(STABILITY), key=lambda record: record["policy_string"])
    if [record["policy_string"] for record in released_records] != ["0", "1"]:
        raise ValueError("released-gate run is incomplete")
    expected_stability = [f"{value:05b}" for value in range(16, 32)]
    if [record["policy_string"] for record in stability_records] != expected_stability:
        raise ValueError("depth-5 rescue branch is incomplete")

    released_variants = [
        {
            "policy": record["policy_string"],
            "prediction": record["scores"]["semantic_candidate"],
            "correct": bool(record["scores"]["semantic_correct"]),
            "generated_token_count": record["generated_token_count"],
            "inference_augmentations": record["inference_augmentation_count"],
            "horizon_reached": record["intervention_reached"],
            "boundary_prefixes": [
                event["generated_prefix"]
                for event in record["gate_events"]
                if event.get("policy_bit_index") is not None
            ],
        }
        for record in released_records
    ]
    stability_predictions = [
        record["scores"]["semantic_candidate"] for record in stability_records
    ]
    stability = {
        "policy_count": len(stability_records),
        "correct_policy_count": sum(
            record["scores"]["semantic_correct"] for record in stability_records
        ),
        "regressed_policy_count": sum(
            not record["scores"]["semantic_correct"] for record in stability_records
        ),
        "correct_policies": [
            record["policy_string"]
            for record in stability_records
            if record["scores"]["semantic_correct"]
        ],
        "full_horizon_count": sum(
            record["intervention_reached"] for record in stability_records
        ),
        "truncated_count": sum(
            record["truncated_at_max_new_tokens"] for record in stability_records
        ),
        "unique_completion_count": len(
            {token_hash(record) for record in stability_records}
        ),
        "prediction_distribution": dict(Counter(stability_predictions)),
    }
    analysis = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "sample_id": "gsm8k_test_0810",
        "gold": released_records[0]["scores"]["semantic_gold"],
        "sources": [
            {"path": portable_path(RELEASED), "sha256": sha256_file(RELEASED)},
            {"path": portable_path(STABILITY), "sha256": sha256_file(STABILITY)},
        ],
        "released_gate": {
            "skip_correct": released_variants[0]["correct"],
            "invoke_correct": released_variants[1]["correct"],
            "completion_changed": token_hash(released_records[0])
            != token_hash(released_records[1]),
            "variants": released_variants,
        },
        "decoded_depth5_rescue_branch": stability,
        "total_latency_seconds": sum(
            record["latency_seconds"]
            for record in released_records + stability_records
        ),
        "all_observed_prefixes_match_policies": all(
            record["observed_policy_bits"]
            == record["policy_bits"][: len(record["observed_policy_bits"])]
            for record in released_records + stability_records
        ),
    }
    write_json(OUTPUT, analysis)
    REPORT.write_text(render_report(analysis), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "released_gate_invoke_correct": analysis["released_gate"]["invoke_correct"],
                "stability_correct": stability["correct_policy_count"],
                "stability_total": stability["policy_count"],
                "stability_regressed": stability["regressed_policy_count"],
            },
            indent=2,
        )
    )
    print(f"analysis={OUTPUT}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
