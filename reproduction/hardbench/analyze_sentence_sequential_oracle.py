"""Analyze the sentence-granularity sequential oracle experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from common import HARD_BENCH_ROOT, portable_path, read_jsonl, sha256_file, utc_now, write_json
from score_results import numeric_value


CONDITION = "Z_memgen_sentence_sequential_oracle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--condition", default=CONDITION)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=HARD_BENCH_ROOT / "analysis" / "sentence_sequential_oracle_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=HARD_BENCH_ROOT / "SENTENCE_SEQUENTIAL_ORACLE_REPORT.md",
    )
    return parser.parse_args()


def absolute_error(candidate: str | None, gold: str) -> Decimal | None:
    if candidate is None:
        return None
    candidate_value = numeric_value(candidate)
    gold_value = numeric_value(gold)
    if candidate_value is None or gold_value is None:
        return None
    def as_decimal(value: Decimal | Fraction) -> Decimal:
        if isinstance(value, Fraction):
            return Decimal(value.numerator) / Decimal(value.denominator)
        return value

    return abs(as_decimal(candidate_value) - as_decimal(gold_value))


def decimal_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def completion_hash(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record["completion_token_ids"], separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def policy_summary(record: dict, baseline_hash: str) -> dict:
    score = record["scores"]
    error = absolute_error(score["semantic_candidate"], score["semantic_gold"])
    boundaries = [
        event
        for event in record["gate_events"]
        if not event.get("is_prompt") and event.get("policy_bit_index") is not None
    ]
    return {
        "policy": record["policy_string"],
        "invocations": record["policy_invocation_count"],
        "checkpoint_native_policy": record["checkpoint_native_policy"],
        "decision_horizon_reached": record["intervention_reached"],
        "observed_policy_bits": record["observed_policy_bits"],
        "actual_inference_augmentations": record["inference_augmentation_count"],
        "prediction": score["semantic_candidate"],
        "semantic_correct": bool(score["semantic_correct"]),
        "absolute_error": decimal_string(error),
        "completion_changed_vs_no_invoke": completion_hash(record) != baseline_hash,
        "generated_token_count": record["generated_token_count"],
        "latency_seconds": record["latency_seconds"],
        "boundary_reasons": [event["boundary_reason"] for event in boundaries],
        "boundary_prefixes": [event["generated_prefix"] for event in boundaries],
    }


def analyze_depth(records: list[dict], depth: int) -> dict:
    depth_records = sorted(
        (record for record in records if record["decision_depth"] == depth),
        key=lambda record: record["policy_string"],
    )
    expected = 2**depth
    if len(depth_records) != expected:
        raise ValueError(f"depth {depth}: expected {expected} records, found {len(depth_records)}")
    policies = [record["policy_string"] for record in depth_records]
    if len(set(policies)) != expected:
        raise ValueError(f"depth {depth}: duplicate policies")
    baseline_policy = "0" * depth
    baseline = next(record for record in depth_records if record["policy_string"] == baseline_policy)
    baseline_hash = completion_hash(baseline)
    variants = [policy_summary(record, baseline_hash) for record in depth_records]
    correct = [variant for variant in variants if variant["semantic_correct"]]
    reached = [variant for variant in variants if variant["decision_horizon_reached"]]
    errors = [
        (Decimal(variant["absolute_error"]), variant)
        for variant in variants
        if variant["absolute_error"] is not None
    ]
    best_error = min((error for error, _ in errors), default=None)
    best = [variant for error, variant in errors if error == best_error]
    by_invocation_count = {}
    for invocation_count in range(depth + 1):
        group = [v for v in variants if v["invocations"] == invocation_count]
        by_invocation_count[str(invocation_count)] = {
            "policies": len(group),
            "correct": sum(v["semantic_correct"] for v in group),
            "horizon_reached": sum(v["decision_horizon_reached"] for v in group),
        }
    return {
        "depth": depth,
        "expected_policies": expected,
        "observed_policies": len(variants),
        "decision_horizon_reached": len(reached),
        "unique_completion_count": len({completion_hash(record) for record in depth_records}),
        "unique_prediction_count": len({v["prediction"] for v in variants}),
        "prediction_distribution": dict(sorted(Counter(v["prediction"] for v in variants).items(), key=lambda item: str(item[0]))),
        "correct_policy_count": len(correct),
        "correct_policies": [v["policy"] for v in correct],
        "minimum_invocations_for_correct": min((v["invocations"] for v in correct), default=None),
        "checkpoint_native_correct_policies": [v["policy"] for v in correct if v["checkpoint_native_policy"]],
        "expanded_budget_correct_policies": [v["policy"] for v in correct if not v["checkpoint_native_policy"]],
        "no_invoke_prediction": variants[0]["prediction"],
        "no_invoke_correct": variants[0]["semantic_correct"],
        "changed_vs_no_invoke_count": sum(v["completion_changed_vs_no_invoke"] for v in variants),
        "best_absolute_error": decimal_string(best_error),
        "best_error_policies": [v["policy"] for v in best],
        "by_invocation_count": by_invocation_count,
        "variants": variants,
    }


def render_report(analysis: dict) -> str:
    d3 = analysis["depths"]["3"]
    d5 = analysis["depths"]["5"]
    exact_any = d3["correct_policy_count"] + d5["correct_policy_count"] > 0
    if exact_any:
        headline = "句级 sequential oracle 找到了 exact rescue 路径。"
    else:
        headline = "句级 sequential oracle 未找到 exact rescue 路径。"
    lines = [
        "# Sentence-level Sequential Oracle Report",
        "",
        f"**结论：{headline}**",
        "",
        "## 实验问题",
        "",
        f"对 `{analysis['sample_id']}`，在每条实时生成轨迹上重新识别论文所述的 delimiter-token sentence-granularity 节点，穷举 invoke/skip 策略。第一次 invoke 改变文本后，后续节点也随新轨迹变化，因此这不是 baseline 固定位置的 subset sweep。Prompt augmentation 始终开启。",
        "",
        "边界规则保留论文及发布代码使用的 comma / period / newline delimiter，但排除金额逗号、小数点、未闭合 `<<...>>` 计算区间、空行，以及紧跟未完成算术运算符的标点。普通语义逗号仍是合法节点。",
        "",
        "## 覆盖与结果",
        "",
        "| 深度 | 配置含义 | 策略数 | 完整到达决策深度 | 唯一完成文本 | exact-correct 策略 | 最佳绝对误差 |",
        "|---:|---|---:|---:|---:|---:|---:|",
        f"| 3 | checkpoint 原生最多 3 次 invoke | {d3['observed_policies']} | {d3['decision_horizon_reached']} | {d3['unique_completion_count']} | {d3['correct_policy_count']} | {d3['best_absolute_error']} |",
        f"| 5 | 同权重，诊断上限临时提高到 5 | {d5['observed_policies']} | {d5['decision_horizon_reached']} | {d5['unique_completion_count']} | {d5['correct_policy_count']} | {d5['best_absolute_error']} |",
        "",
        f"Gold 为 `{analysis['gold']}`。3-step no-invoke 预测 `{d3['no_invoke_prediction']}`；5-step no-invoke 预测 `{d5['no_invoke_prediction']}`。两条 no-invoke 完成文本" + ("完全一致。" if analysis["cross_depth_no_invoke_identical"] else "不一致，需要把它视为配置敏感性信号。"),
        "",
        "### 2^3",
        "",
        f"- exact-correct policies: `{d3['correct_policies']}`",
        f"- 输出相对 no-invoke 改变：{d3['changed_vs_no_invoke_count']}/{d3['observed_policies']}",
        f"- 最接近 gold 的 policy：`{d3['best_error_policies']}`（绝对误差 {d3['best_absolute_error']}；距离仅作诊断，不当作正确标签）",
        "",
        "### 2^5",
        "",
        f"- exact-correct policies: `{d5['correct_policies']}`",
        f"- 其中 checkpoint 原生预算内（invoke 次数 <= 3）：`{d5['checkpoint_native_correct_policies']}`",
        f"- 需要 4–5 次 invoke 的扩展预算路径：`{d5['expanded_budget_correct_policies']}`",
        f"- 输出相对 no-invoke 改变：{d5['changed_vs_no_invoke_count']}/{d5['observed_policies']}",
        f"- 最接近 gold 的 policy：`{d5['best_error_policies']}`（绝对误差 {d5['best_absolute_error']}；距离仅作诊断，不当作正确标签）",
        "",
        "## 能推出什么",
        "",
    ]
    if exact_any:
        lines.extend(
            [
                "至少存在一条多步、轨迹依赖的 Weaver 调用路径可救回该题，因此此前 0/36 单点结果不能把责任直接归到 Weaver；Trigger 的顺序控制能力仍可能是关键瓶颈。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "在本次严格限定的句级候选规则与深度 3/5 搜索中，即使 oracle 事后知道答案，也没有找到能让当前 Weaver exact-correct 的路径。这明显削弱了“只是 Trigger 没选准时机”的解释，并把排查重点进一步推向 Weaver、prompt memory、checkpoint/论文配置差异。",
                "",
                "该结论仍只覆盖一个样本、最多五个动态决策节点；它不是对任意长度策略或任意 candidate generator 的全称证明。",
                "",
            ]
        )
    lines.extend(
        [
            "## 完整性检查",
            "",
            f"- 计划/实际策略：{analysis['expected_total_policies']}/{analysis['observed_total_policies']}",
        f"- 所有已发生的决策前缀均与计划 bit 串一致：{analysis['all_observed_prefixes_match_policies']}",
        f"- 完整到达名义决策深度的策略：{analysis['full_horizon_policy_count']}/{analysis['observed_total_policies']}（其余策略提前到达生成叶节点）",
            f"- 误接纳 numeric/internal delimiter：{analysis['invalid_accepted_boundary_count']}",
            f"- 原始生成总耗时：{analysis['total_latency_seconds']:.1f} 秒",
            "",
            "逐策略预测、边界前缀和调用位置保存在机器可读分析 JSON 与 scored JSONL 中。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    condition = args.condition
    raw_path = run_dir / "raw" / f"{condition}.raw.jsonl"
    scored_path = run_dir / "scored" / f"{condition}.scored.jsonl"
    records = read_jsonl(scored_path)
    if not records:
        raise FileNotFoundError(f"missing scored records: {scored_path}")
    if any(record["condition"] != condition for record in records):
        raise ValueError("unexpected condition in scored records")

    d3 = analyze_depth(records, 3)
    d5 = analyze_depth(records, 5)
    invalid_accepted = [
        event
        for record in records
        for event in record["gate_events"]
        if event.get("policy_bit_index") is not None
        and event.get("boundary_reason")
        in {
            "numeric_comma_or_period",
            "inside_calculator_span",
            "unfinished_arithmetic_before_punctuation",
            "unfinished_arithmetic_before_newline",
        }
    ]
    d3_zero = next(r for r in records if r["decision_depth"] == 3 and r["policy_string"] == "000")
    d5_zero = next(r for r in records if r["decision_depth"] == 5 and r["policy_string"] == "00000")
    analysis = {
        "schema_version": 1,
        "analyzed_at": utc_now(),
        "condition": condition,
        "sample_id": records[0]["sample_id"],
        "gold": records[0]["scores"]["semantic_gold"],
        "raw_path": portable_path(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "scored_path": portable_path(scored_path),
        "scored_sha256": sha256_file(scored_path),
        "expected_total_policies": 40,
        "observed_total_policies": len(records),
        "all_policy_horizons_reached": all(record["intervention_reached"] for record in records),
        "full_horizon_policy_count": sum(record["intervention_reached"] for record in records),
        "all_observed_prefixes_match_policies": all(
            record["observed_policy_bits"]
            == record["policy_bits"][: len(record["observed_policy_bits"])]
            for record in records
        ),
        "invalid_accepted_boundary_count": len(invalid_accepted),
        "cross_depth_no_invoke_identical": d3_zero["completion_token_ids"] == d5_zero["completion_token_ids"],
        "total_latency_seconds": sum(record["latency_seconds"] for record in records),
        "depths": {"3": d3, "5": d5},
    }
    write_json(args.output_json.resolve(), analysis)
    args.report.resolve().write_text(render_report(analysis), encoding="utf-8", newline="\n")
    print(json.dumps({key: analysis[key] for key in ("observed_total_policies", "full_horizon_policy_count", "all_observed_prefixes_match_policies", "invalid_accepted_boundary_count", "cross_depth_no_invoke_identical")}, indent=2))
    print(json.dumps({depth: {key: value[key] for key in ("correct_policy_count", "correct_policies", "best_absolute_error", "best_error_policies")} for depth, value in analysis["depths"].items()}, indent=2))
    print(f"analysis={args.output_json.resolve()}")
    print(f"report={args.report.resolve()}")


if __name__ == "__main__":
    main()
