#!/usr/bin/env python3
"""Aggregate the single frozen PM V1.5 V5.1 system confirmation.

This script is deliberately analysis-only.  It joins the blinded human exports
to the private key after review, reconstructs the response selected by each
precommitted policy, and applies the gate frozen before confirmation outcomes.
It never changes a feature, threshold, resource, response, or policy decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUALITY = Path(
    "/home/tokkio/.codex/attachments/9661e32a-1c9c-461a-b117-6a450fa633ae/pasted-text.txt"
)
DEFAULT_RISK = Path(
    "/home/tokkio/.codex/attachments/baa9c7c1-7f17-445b-869e-048e626dd8e4/pasted-text.txt"
)
DEFAULT_REVIEW_DIR = ROOT / "outputs/pm_v1_5_v5_1_confirmation_review_v1_candidate"
DEFAULT_PLAN_DIR = ROOT / "outputs/pm_v1_5_v5_1_confirmation_plan_v1"
DEFAULT_EXECUTION_DIR = ROOT / "outputs/pm_v1_5_v5_1_confirmation_execution_v1"
DEFAULT_CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json"
DEFAULT_OUT_DIR = ROOT / "outputs/pm_v1_5_v5_1_confirmation_analysis_v1"

POLICIES = ("always_off", "component_fixed_high", "transparent_rule", "learned_pm")
COMPARATORS = ("component_fixed_high", "always_off", "transparent_rule")
CRITICAL_CATEGORIES = {"fabricated_recall", "internal_resource_label_exposure"}
BOOTSTRAP_REPLICATES = 50_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_group[row["counterfactual_group_id"]].append(float(row[value_key]))
    groups = sorted(by_group)
    group_means = np.asarray([np.mean(by_group[group]) for group in groups], dtype=float)
    if len(groups) != 64:
        raise ValueError(f"expected 64 counterfactual groups, found {len(groups)}")
    # All confirmation groups have the same number of seed-level units, so the
    # mean of resampled group means is the cluster-resampled observation mean.
    draws = rng.integers(0, len(groups), size=(replicates, len(groups)))
    sampled = group_means[draws].mean(axis=1)
    return percentile_ci(sampled)


def selected_arm(policy_decisions: dict[str, bool], policy: str) -> str:
    return "ON" if bool(policy_decisions[policy]) else "OFF"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    quality_rows = read_jsonl(args.quality)
    risk_rows = read_jsonl(args.risk)
    quality_key_rows = read_jsonl(args.review_dir / "private_quality_key.jsonl")
    risk_key_rows = read_jsonl(args.review_dir / "private_risk_key.jsonl")
    binding_rows = read_jsonl(args.plan_dir / "state_policy_bindings_private.jsonl")
    outcome_rows = read_jsonl(args.execution_dir / "outcomes_ordered_private.jsonl")
    contract = json.loads(args.contract.read_text())

    if len(quality_rows) != 256 or len(risk_rows) != 512:
        raise ValueError(f"incomplete review: quality={len(quality_rows)}, risk={len(risk_rows)}")
    if len(quality_key_rows) != 256 or len(risk_key_rows) != 512:
        raise ValueError("private review key has unexpected size")
    if len(binding_rows) != 128 or len(outcome_rows) != 512:
        raise ValueError("plan or execution has unexpected size")

    quality = keyed(quality_rows, "blind_item_id")
    quality_key = keyed(quality_key_rows, "blind_item_id")
    risk = keyed(risk_rows, "risk_item_id")
    risk_key = keyed(risk_key_rows, "risk_item_id")
    outcomes = keyed(outcome_rows, "call_id")

    if set(quality) != set(quality_key) or set(risk) != set(risk_key):
        raise ValueError("human export IDs do not exactly match the frozen private key")
    if Counter(row["quality_preference"] for row in quality_rows).keys() - {"A", "B", "tie"}:
        raise ValueError("quality export contains an invalid preference")
    if Counter(row["any_material_risk"] for row in risk_rows).keys() - {"yes", "no"}:
        raise ValueError("risk export contains an invalid decision")
    if any(not row["itt_row_valid"] for row in outcome_rows):
        raise ValueError("confirmation contains an invalid ITT row")

    # One row per seed-level ON/OFF pair.
    seed_rows: list[dict[str, Any]] = []
    for blind_id, key in quality_key.items():
        review = quality[blind_id]
        preference = review["quality_preference"]
        if preference == "tie":
            on_minus_off = 0.0
        else:
            preferred_arm = key["response_a_arm"] if preference == "A" else key["response_b_arm"]
            on_minus_off = 1.0 if preferred_arm == "ON" else -1.0

        calls = {
            key["response_a_arm"]: outcomes[key["response_a_call_id"]],
            key["response_b_arm"]: outcomes[key["response_b_call_id"]],
        }
        if set(calls) != {"ON", "OFF"}:
            raise ValueError(f"quality pair lacks ON/OFF arms: {blind_id}")

        risks_for_arm: dict[str, dict[str, Any]] = {}
        for risk_id, risk_key_row in risk_key.items():
            if risk_key_row["call_id"] in {key["response_a_call_id"], key["response_b_call_id"]}:
                risks_for_arm[risk_key_row["arm"]] = risk[risk_id]
        if set(risks_for_arm) != {"ON", "OFF"}:
            raise ValueError(f"quality pair lacks matching risk rows: {blind_id}")

        row: dict[str, Any] = {
            "blind_item_id": blind_id,
            "state_id": key["state_id"],
            "component": key["component"],
            "counterfactual_group_id": key["counterfactual_group_id"],
            "seed": key["seed"],
            "on_minus_off_quality": on_minus_off,
        }
        for policy in POLICIES:
            arm = selected_arm(key["policy_decisions"], policy)
            outcome = calls[arm]
            risk_review = risks_for_arm[arm]
            row[f"{policy}_arm"] = arm
            row[f"{policy}_quality_vs_off"] = on_minus_off if arm == "ON" else 0.0
            row[f"{policy}_risk"] = 1.0 if risk_review["any_material_risk"] == "yes" else 0.0
            row[f"{policy}_prompt_tokens"] = float(outcome["usage"]["prompt_tokens"])
            row[f"{policy}_total_tokens"] = float(outcome["usage"]["total_tokens"])
            row[f"{policy}_fallback"] = 1.0 if outcome["fallback_used"] else 0.0
            row[f"{policy}_critical_events"] = sum(
                category in CRITICAL_CATEGORIES for category in risk_review["selected_categories"]
            )
        seed_rows.append(row)

    if len(seed_rows) != 256:
        raise ValueError(f"expected 256 reconstructed seed rows, found {len(seed_rows)}")

    # Ensure the private policy binding and the review key carry identical frozen decisions.
    bindings = {(row["state_id"], row["component"]): row for row in binding_rows}
    for row in seed_rows:
        binding = bindings[(row["state_id"], row["component"])]
        key = quality_key[row["blind_item_id"]]
        if binding["policy_decisions"] != key["policy_decisions"]:
            raise ValueError("policy decision changed between freeze and review packet")

    protocol_seed = int.from_bytes(
        hashlib.sha256(b"pm-v1.5-v5.1-confirmation-cluster-bootstrap-v1").digest()[:8], "big"
    )
    rng = np.random.default_rng(protocol_seed)

    policy_summary: dict[str, Any] = {}
    for policy in POLICIES:
        selected_on = sum(row[f"{policy}_arm"] == "ON" for row in seed_rows)
        policy_summary[policy] = {
            "selected_on_seed_rows": selected_on,
            "selected_on_fraction": selected_on / len(seed_rows),
            "quality_mean_vs_always_off": float(
                np.mean([row[f"{policy}_quality_vs_off"] for row in seed_rows])
            ),
            "material_risk_count": int(sum(row[f"{policy}_risk"] for row in seed_rows)),
            "material_risk_rate": float(np.mean([row[f"{policy}_risk"] for row in seed_rows])),
            "mean_prompt_tokens": float(np.mean([row[f"{policy}_prompt_tokens"] for row in seed_rows])),
            "mean_total_tokens": float(np.mean([row[f"{policy}_total_tokens"] for row in seed_rows])),
            "fallback_count": int(sum(row[f"{policy}_fallback"] for row in seed_rows)),
            "critical_grounding_event_count": int(
                sum(row[f"{policy}_critical_events"] for row in seed_rows)
            ),
        }

    comparisons: dict[str, Any] = {}
    for comparator in COMPARATORS:
        comparison_rows: list[dict[str, Any]] = []
        for row in seed_rows:
            learned_arm = row["learned_pm_arm"]
            comparator_arm = row[f"{comparator}_arm"]
            if learned_arm == comparator_arm:
                quality_diff = 0.0
            elif learned_arm == "ON":
                quality_diff = row["on_minus_off_quality"]
            else:
                quality_diff = -row["on_minus_off_quality"]
            comparison_rows.append(
                {
                    "counterfactual_group_id": row["counterfactual_group_id"],
                    "quality_diff": quality_diff,
                    "risk_diff": row["learned_pm_risk"] - row[f"{comparator}_risk"],
                    "prompt_diff": row["learned_pm_prompt_tokens"]
                    - row[f"{comparator}_prompt_tokens"],
                    "total_diff": row["learned_pm_total_tokens"]
                    - row[f"{comparator}_total_tokens"],
                }
            )
        quality_values = [row["quality_diff"] for row in comparison_rows]
        risk_values = [row["risk_diff"] for row in comparison_rows]
        comparisons[comparator] = {
            "quality_better": quality_values.count(1.0),
            "quality_worse": quality_values.count(-1.0),
            "quality_tie_or_same_arm": quality_values.count(0.0),
            "quality_mean_difference": float(np.mean(quality_values)),
            "quality_cluster_bootstrap_95_ci": cluster_bootstrap(
                comparison_rows, "quality_diff", rng=rng
            ),
            "material_risk_rate_difference": float(np.mean(risk_values)),
            "risk_cluster_bootstrap_95_ci": cluster_bootstrap(
                comparison_rows, "risk_diff", rng=rng
            ),
            "mean_prompt_token_difference": float(
                np.mean([row["prompt_diff"] for row in comparison_rows])
            ),
            "mean_total_token_difference": float(
                np.mean([row["total_diff"] for row in comparison_rows])
            ),
        }

    gates = contract["single_use_system_gate"]
    fixed = comparisons["component_fixed_high"]
    off = comparisons["always_off"]
    rule = comparisons["transparent_rule"]
    learned_summary = policy_summary["learned_pm"]
    fixed_summary = policy_summary["component_fixed_high"]
    prompt_reduction = 1.0 - (
        learned_summary["mean_prompt_tokens"] / fixed_summary["mean_prompt_tokens"]
    )
    activity = sum(row["policy_decisions"]["learned_pm"] for row in binding_rows) / len(binding_rows)

    strict_rule_improvement = any(
        (
            rule["quality_mean_difference"] > 0,
            rule["material_risk_rate_difference"] < 0,
            rule["mean_prompt_token_difference"] < 0,
        )
    )
    gate_results = {
        "quality_vs_fixed_high_lower_ci": {
            "value": fixed["quality_cluster_bootstrap_95_ci"][0],
            "threshold": gates["quality_vs_fixed_high"]["lower_95_ci_min"],
            "pass": fixed["quality_cluster_bootstrap_95_ci"][0]
            >= gates["quality_vs_fixed_high"]["lower_95_ci_min"],
        },
        "quality_vs_always_off_point": {
            "value": off["quality_mean_difference"],
            "threshold": gates["quality_vs_always_off"]["point_estimate_min"],
            "pass": off["quality_mean_difference"]
            >= gates["quality_vs_always_off"]["point_estimate_min"],
        },
        "quality_vs_always_off_lower_ci": {
            "value": off["quality_cluster_bootstrap_95_ci"][0],
            "threshold": gates["quality_vs_always_off"]["lower_95_ci_min"],
            "pass": off["quality_cluster_bootstrap_95_ci"][0]
            >= gates["quality_vs_always_off"]["lower_95_ci_min"],
        },
        "risk_vs_fixed_high_point": {
            "value": fixed["material_risk_rate_difference"],
            "threshold": gates["risk_vs_fixed_high"]["point_difference_max"],
            "pass": fixed["material_risk_rate_difference"]
            <= gates["risk_vs_fixed_high"]["point_difference_max"],
        },
        "risk_vs_fixed_high_upper_ci": {
            "value": fixed["risk_cluster_bootstrap_95_ci"][1],
            "threshold": gates["risk_vs_fixed_high"]["upper_95_ci_max"],
            "pass": fixed["risk_cluster_bootstrap_95_ci"][1]
            <= gates["risk_vs_fixed_high"]["upper_95_ci_max"],
        },
        "prompt_cost_reduction_vs_fixed_high": {
            "value": prompt_reduction,
            "threshold": gates["prompt_cost_vs_fixed_high"]["relative_reduction_min"],
            "pass": prompt_reduction
            >= gates["prompt_cost_vs_fixed_high"]["relative_reduction_min"],
        },
        "quality_vs_transparent_rule_lower_ci": {
            "value": rule["quality_cluster_bootstrap_95_ci"][0],
            "threshold": gates["transparent_rule_comparison"]["quality_lower_95_ci_min"],
            "pass": rule["quality_cluster_bootstrap_95_ci"][0]
            >= gates["transparent_rule_comparison"]["quality_lower_95_ci_min"],
        },
        "risk_vs_transparent_rule_point": {
            "value": rule["material_risk_rate_difference"],
            "threshold": gates["transparent_rule_comparison"]["risk_point_difference_max"],
            "pass": rule["material_risk_rate_difference"]
            <= gates["transparent_rule_comparison"]["risk_point_difference_max"],
        },
        "prompt_tokens_vs_transparent_rule_point": {
            "value": rule["mean_prompt_token_difference"],
            "threshold": gates["transparent_rule_comparison"]["prompt_token_point_difference_max"],
            "pass": rule["mean_prompt_token_difference"]
            <= gates["transparent_rule_comparison"]["prompt_token_point_difference_max"],
        },
        "strict_point_improvement_vs_transparent_rule": {
            "value": strict_rule_improvement,
            "threshold": True,
            "pass": strict_rule_improvement,
        },
        "nondegenerate_policy_activity": {
            "value": activity,
            "threshold": [
                gates["nondegenerate_policy"]["requested_on_fraction_min"],
                gates["nondegenerate_policy"]["requested_on_fraction_max"],
            ],
            "pass": gates["nondegenerate_policy"]["requested_on_fraction_min"]
            <= activity
            <= gates["nondegenerate_policy"]["requested_on_fraction_max"],
        },
        "critical_grounding_count": {
            "value": learned_summary["critical_grounding_event_count"],
            "threshold": gates["critical_grounding_failures"]["learned_count_max"],
            "pass": learned_summary["critical_grounding_event_count"]
            <= gates["critical_grounding_failures"]["learned_count_max"],
        },
        "critical_grounding_not_above_fixed_high": {
            "value": [
                learned_summary["critical_grounding_event_count"],
                fixed_summary["critical_grounding_event_count"],
            ],
            "threshold": "learned<=fixed_high",
            "pass": learned_summary["critical_grounding_event_count"]
            <= fixed_summary["critical_grounding_event_count"],
        },
    }
    overall_pass = all(result["pass"] for result in gate_results.values())

    component_summary: dict[str, Any] = {}
    for component in ("MP", "MS", "ME", "RS"):
        subset = [row for row in seed_rows if row["component"] == component]
        component_summary[component] = {
            "n_seed_rows": len(subset),
            "learned_on_fraction": float(np.mean([row["learned_pm_arm"] == "ON" for row in subset])),
            "learned_quality_mean_vs_always_off": float(
                np.mean([row["learned_pm_quality_vs_off"] for row in subset])
            ),
            "learned_material_risk_rate": float(np.mean([row["learned_pm_risk"] for row in subset])),
            "fixed_high_material_risk_rate": float(
                np.mean([row["component_fixed_high_risk"] for row in subset])
            ),
            "learned_mean_prompt_tokens": float(
                np.mean([row["learned_pm_prompt_tokens"] for row in subset])
            ),
            "fixed_high_mean_prompt_tokens": float(
                np.mean([row["component_fixed_high_prompt_tokens"] for row in subset])
            ),
        }

    result = {
        "protocol": "pm-v1.5-v5.1-confirmation-analysis-v1",
        "status": "PASS_SINGLE_USE_SYSTEM_GATE" if overall_pass else "FAIL_SINGLE_USE_SYSTEM_GATE",
        "method_or_policy_changed_after_confirmation": False,
        "review_data_quality": {
            "quality_rows": len(quality_rows),
            "risk_rows": len(risk_rows),
            "quality_unique_ids": len(quality),
            "risk_unique_ids": len(risk),
            "quality_uncertain": 0,
            "risk_uncertain": 0,
            "quality_annotators": dict(Counter(row["annotator_id"] for row in quality_rows)),
            "risk_annotators": dict(Counter(row["annotator_id"] for row in risk_rows)),
        },
        "policy_summary": policy_summary,
        "learned_comparisons": comparisons,
        "component_diagnostic_not_a_gate": component_summary,
        "bootstrap": {
            "method": "percentile cluster bootstrap over counterfactual_group_id",
            "groups": 64,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": protocol_seed,
        },
        "gate_results": gate_results,
        "all_checks_required": True,
        "overall_pass": overall_pass,
        "input_sha256": {
            "quality_human_export": sha256(args.quality),
            "risk_human_export": sha256(args.risk),
            "private_quality_key": sha256(args.review_dir / "private_quality_key.jsonl"),
            "private_risk_key": sha256(args.review_dir / "private_risk_key.jsonl"),
            "policy_bindings": sha256(args.plan_dir / "state_policy_bindings_private.jsonl"),
            "execution_outcomes": sha256(args.execution_dir / "outcomes_ordered_private.jsonl"),
            "frozen_contract": sha256(args.contract),
        },
        "stopping_rule_applied": contract["stopping_rule"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "confirmation_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (args.out_dir / "verified_quality_human_export.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in quality_rows) + "\n"
    )
    (args.out_dir / "verified_risk_human_export.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in risk_rows) + "\n"
    )
    print(json.dumps({
        "protocol": result["protocol"],
        "status": result["status"],
        "overall_pass": overall_pass,
        "failed_gates": [name for name, value in gate_results.items() if not value["pass"]],
        "output": str(args.out_dir / "confirmation_analysis.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
