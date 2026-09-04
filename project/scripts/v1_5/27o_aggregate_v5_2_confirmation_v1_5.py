#!/usr/bin/env python3
"""Replay four frozen policies and apply the V5.2 single-use system gate.

Inputs must be the final resolved primary quality and risk exports.  Exact
public-surface duplicates are expanded through the private representative map.
This analysis never changes a response, feature, model, threshold, resource,
policy decision, or executor.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
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
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    rng: np.random.Generator,
) -> list[float]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_group[str(row["counterfactual_group_id"])].append(float(row[value_key]))
    groups = sorted(by_group)
    if len(groups) != 64:
        raise ValueError(f"expected 64 counterfactual groups, found {len(groups)}")
    group_means = np.asarray([np.mean(by_group[group]) for group in groups], dtype=float)
    draws = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    return percentile_ci(group_means[draws].mean(axis=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--risk", type=Path, required=True)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_review_v1_candidate",
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_execution_v1",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_analysis_v1",
    )
    args = parser.parse_args()

    quality_rows = read_jsonl(args.quality)
    risk_rows = read_jsonl(args.risk)
    quality_key_rows = read_jsonl(args.review_dir / "private_quality_key.jsonl")
    risk_key_rows = read_jsonl(args.review_dir / "private_risk_key.jsonl")
    binding_rows = read_jsonl(args.plan_dir / "state_policy_bindings_private.jsonl")
    outcome_rows = read_jsonl(args.execution_dir / "outcomes_ordered_private.jsonl")
    contract = json.loads(args.contract.read_text())

    if len(quality_key_rows) != 256 or len(risk_key_rows) != 512:
        raise ValueError("private review key has unexpected size")
    if len(binding_rows) != 128 or len(outcome_rows) != 512:
        raise ValueError("frozen plan or execution has unexpected size")
    quality = keyed(quality_rows, "blind_item_id")
    risk = keyed(risk_rows, "risk_item_id")
    quality_key = keyed(quality_key_rows, "blind_item_id")
    risk_key = keyed(risk_key_rows, "risk_item_id")
    outcomes = keyed(outcome_rows, "call_id")
    bindings = keyed(binding_rows, "state_id")

    expected_quality_representatives = {
        str(row["manual_representative_id"]) for row in quality_key_rows
    }
    expected_risk_representatives = {
        str(row["manual_representative_id"]) for row in risk_key_rows
    }
    if set(quality) != expected_quality_representatives:
        raise ValueError("quality export IDs do not match frozen manual representatives")
    if set(risk) != expected_risk_representatives:
        raise ValueError("risk export IDs do not match frozen manual representatives")
    quality_values = {str(row.get("quality_preference", "")) for row in quality_rows}
    risk_values = {str(row.get("any_material_risk", "")) for row in risk_rows}
    if quality_values - {"A", "B", "tie"}:
        raise ValueError(f"quality export has invalid or unresolved labels: {quality_values}")
    if risk_values - {"yes", "no"}:
        raise ValueError(f"risk export has invalid or unresolved labels: {risk_values}")
    if any(not bool(row["itt_row_valid"]) for row in outcome_rows):
        raise ValueError("confirmation contains invalid ITT rows")

    risk_by_call: dict[str, dict[str, Any]] = {}
    for key_row in risk_key_rows:
        review = risk[str(key_row["manual_representative_id"])]
        call_id = str(key_row["call_id"])
        if call_id in risk_by_call:
            raise ValueError(f"duplicate risk call identity: {call_id}")
        risk_by_call[call_id] = review

    seed_rows: list[dict[str, Any]] = []
    for blind_id, key_row in quality_key.items():
        review = quality[str(key_row["manual_representative_id"])]
        preference = str(review["quality_preference"])
        if preference == "tie":
            on_minus_off = 0.0
        else:
            preferred_arm = (
                str(key_row["response_a_arm"])
                if preference == "A"
                else str(key_row["response_b_arm"])
            )
            on_minus_off = 1.0 if preferred_arm == "ON" else -1.0
        calls = {
            str(key_row["response_a_arm"]): outcomes[str(key_row["response_a_call_id"])],
            str(key_row["response_b_arm"]): outcomes[str(key_row["response_b_call_id"])],
        }
        if set(calls) != {"ON", "OFF"}:
            raise ValueError(f"quality pair lacks ON/OFF arms: {blind_id}")
        row: dict[str, Any] = {
            "blind_item_id": blind_id,
            "state_id": str(key_row["state_id"]),
            "component": str(key_row["component"]),
            "counterfactual_group_id": str(key_row["counterfactual_group_id"]),
            "seed": int(key_row["seed"]),
            "on_minus_off_quality": on_minus_off,
        }
        policy_decisions = dict(key_row["policy_decisions"])
        if policy_decisions != bindings[row["state_id"]]["policy_decisions"]:
            raise ValueError(f"frozen policy decision changed: {row['state_id']}")
        for policy in POLICIES:
            arm = "ON" if bool(policy_decisions[policy]) else "OFF"
            outcome = calls[arm]
            risk_review = risk_by_call[str(outcome["call_id"])]
            row[f"{policy}_arm"] = arm
            row[f"{policy}_quality_vs_off"] = on_minus_off if arm == "ON" else 0.0
            row[f"{policy}_risk"] = (
                1.0 if str(risk_review["any_material_risk"]) == "yes" else 0.0
            )
            row[f"{policy}_prompt_tokens"] = float(outcome["usage"]["prompt_tokens"])
            row[f"{policy}_total_tokens"] = float(outcome["usage"]["total_tokens"])
            row[f"{policy}_critical_events"] = float(
                sum(
                    category in CRITICAL_CATEGORIES
                    for category in risk_review.get("selected_categories", [])
                )
            )
        seed_rows.append(row)
    if len(seed_rows) != 256:
        raise ValueError(f"expected 256 reconstructed seed rows, found {len(seed_rows)}")

    bootstrap_seed = int.from_bytes(
        hashlib.sha256(b"pm-v1.5-v5.2-confirmation-cluster-bootstrap-v1").digest()[:8],
        "big",
    )
    rng = np.random.default_rng(bootstrap_seed)
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
            "material_risk_rate": float(
                np.mean([row[f"{policy}_risk"] for row in seed_rows])
            ),
            "mean_prompt_tokens": float(
                np.mean([row[f"{policy}_prompt_tokens"] for row in seed_rows])
            ),
            "mean_total_tokens": float(
                np.mean([row[f"{policy}_total_tokens"] for row in seed_rows])
            ),
            "critical_grounding_event_count": int(
                sum(row[f"{policy}_critical_events"] for row in seed_rows)
            ),
        }

    comparisons: dict[str, Any] = {}
    for comparator in COMPARATORS:
        comparison_rows: list[dict[str, Any]] = []
        for row in seed_rows:
            learned_arm = str(row["learned_pm_arm"])
            comparator_arm = str(row[f"{comparator}_arm"])
            if learned_arm == comparator_arm:
                quality_diff = 0.0
            elif learned_arm == "ON":
                quality_diff = float(row["on_minus_off_quality"])
            else:
                quality_diff = -float(row["on_minus_off_quality"])
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
        quality_diff_values = [row["quality_diff"] for row in comparison_rows]
        risk_diff_values = [row["risk_diff"] for row in comparison_rows]
        comparisons[comparator] = {
            "quality_better": quality_diff_values.count(1.0),
            "quality_worse": quality_diff_values.count(-1.0),
            "quality_tie_or_same_arm": quality_diff_values.count(0.0),
            "quality_mean_difference": float(np.mean(quality_diff_values)),
            "quality_cluster_bootstrap_95_ci": cluster_bootstrap(
                comparison_rows, "quality_diff", rng=rng
            ),
            "material_risk_rate_difference": float(np.mean(risk_diff_values)),
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

    gates = dict(contract["single_use_system_gate"])
    fixed = comparisons["component_fixed_high"]
    off = comparisons["always_off"]
    rule = comparisons["transparent_rule"]
    learned = policy_summary["learned_pm"]
    fixed_summary = policy_summary["component_fixed_high"]
    prompt_reduction = 1.0 - learned["mean_prompt_tokens"] / fixed_summary["mean_prompt_tokens"]
    requested_on_fraction = sum(
        bool(row["policy_decisions"]["learned_pm"]) for row in binding_rows
    ) / len(binding_rows)
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
            "threshold": gates["quality_vs_fixed_high_lower_95_ci_min"],
            "pass": fixed["quality_cluster_bootstrap_95_ci"][0]
            >= gates["quality_vs_fixed_high_lower_95_ci_min"],
        },
        "quality_vs_always_off_point": {
            "value": off["quality_mean_difference"],
            "threshold": gates["quality_vs_always_off_point_min"],
            "pass": off["quality_mean_difference"] >= gates["quality_vs_always_off_point_min"],
        },
        "quality_vs_always_off_lower_ci": {
            "value": off["quality_cluster_bootstrap_95_ci"][0],
            "threshold": gates["quality_vs_always_off_lower_95_ci_min"],
            "pass": off["quality_cluster_bootstrap_95_ci"][0]
            >= gates["quality_vs_always_off_lower_95_ci_min"],
        },
        "risk_vs_fixed_high_point": {
            "value": fixed["material_risk_rate_difference"],
            "threshold": gates["risk_vs_fixed_high_point_difference_max"],
            "pass": fixed["material_risk_rate_difference"]
            <= gates["risk_vs_fixed_high_point_difference_max"],
        },
        "risk_vs_fixed_high_upper_ci": {
            "value": fixed["risk_cluster_bootstrap_95_ci"][1],
            "threshold": gates["risk_vs_fixed_high_upper_95_ci_max"],
            "pass": fixed["risk_cluster_bootstrap_95_ci"][1]
            <= gates["risk_vs_fixed_high_upper_95_ci_max"],
        },
        "prompt_cost_reduction_vs_fixed_high": {
            "value": prompt_reduction,
            "threshold": gates["prompt_token_relative_reduction_vs_fixed_high_min"],
            "pass": prompt_reduction
            >= gates["prompt_token_relative_reduction_vs_fixed_high_min"],
        },
        "quality_vs_transparent_rule_lower_ci": {
            "value": rule["quality_cluster_bootstrap_95_ci"][0],
            "threshold": gates["transparent_rule_quality_lower_95_ci_min"],
            "pass": rule["quality_cluster_bootstrap_95_ci"][0]
            >= gates["transparent_rule_quality_lower_95_ci_min"],
        },
        "risk_vs_transparent_rule_point": {
            "value": rule["material_risk_rate_difference"],
            "threshold": gates["transparent_rule_risk_point_difference_max"],
            "pass": rule["material_risk_rate_difference"]
            <= gates["transparent_rule_risk_point_difference_max"],
        },
        "prompt_tokens_vs_transparent_rule_point": {
            "value": rule["mean_prompt_token_difference"],
            "threshold": gates["transparent_rule_prompt_token_point_difference_max"],
            "pass": rule["mean_prompt_token_difference"]
            <= gates["transparent_rule_prompt_token_point_difference_max"],
        },
        "strict_point_improvement_vs_transparent_rule": {
            "value": strict_rule_improvement,
            "threshold": True,
            "pass": strict_rule_improvement,
        },
        "nondegenerate_policy_activity": {
            "value": requested_on_fraction,
            "threshold": [
                gates["learned_requested_on_fraction_min"],
                gates["learned_requested_on_fraction_max"],
            ],
            "pass": gates["learned_requested_on_fraction_min"]
            <= requested_on_fraction
            <= gates["learned_requested_on_fraction_max"],
        },
        "critical_grounding_count": {
            "value": learned["critical_grounding_event_count"],
            "threshold": gates["critical_grounding_learned_count_max"],
            "pass": learned["critical_grounding_event_count"]
            <= gates["critical_grounding_learned_count_max"],
        },
        "critical_grounding_not_above_fixed_high": {
            "value": [
                learned["critical_grounding_event_count"],
                fixed_summary["critical_grounding_event_count"],
            ],
            "threshold": "learned<=fixed_high",
            "pass": learned["critical_grounding_event_count"]
            <= fixed_summary["critical_grounding_event_count"],
        },
    }
    overall_pass = all(result["pass"] for result in gate_results.values())

    component_summary: dict[str, Any] = {}
    for component in ("MP", "MS", "ME", "RS"):
        subset = [row for row in seed_rows if row["component"] == component]
        component_summary[component] = {
            "n_seed_rows": len(subset),
            "learned_on_fraction": float(
                np.mean([row["learned_pm_arm"] == "ON" for row in subset])
            ),
            "learned_quality_mean_vs_always_off": float(
                np.mean([row["learned_pm_quality_vs_off"] for row in subset])
            ),
            "learned_material_risk_rate": float(
                np.mean([row["learned_pm_risk"] for row in subset])
            ),
            "fixed_high_material_risk_rate": float(
                np.mean([row["component_fixed_high_risk"] for row in subset])
            ),
        }

    result = {
        "protocol": "pm-v1.5-v5.2-content-disjoint-confirmation-analysis-v1",
        "status": (
            "PASS_SINGLE_USE_V5_2_SYSTEM_GATE"
            if overall_pass
            else "FAIL_SINGLE_USE_V5_2_SYSTEM_GATE"
        ),
        "overall_pass": overall_pass,
        "method_or_policy_changed_after_confirmation": False,
        "review_data_quality": {
            "manual_quality_rows": len(quality_rows),
            "manual_risk_rows": len(risk_rows),
            "expanded_quality_rows": len(quality_key_rows),
            "expanded_risk_rows": len(risk_key_rows),
            "quality_annotators": dict(Counter(str(row["annotator_id"]) for row in quality_rows)),
            "risk_annotators": dict(Counter(str(row["annotator_id"]) for row in risk_rows)),
        },
        "policy_summary": policy_summary,
        "learned_comparisons": comparisons,
        "component_diagnostic_not_a_gate": component_summary,
        "bootstrap": {
            "method": "percentile cluster bootstrap over counterfactual_group_id",
            "groups": 64,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": bootstrap_seed,
        },
        "gate_results": gate_results,
        "all_checks_required": True,
        "input_sha256": {
            "quality_final_export": sha256(args.quality),
            "risk_final_export": sha256(args.risk),
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
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "status": result["status"],
                "overall_pass": overall_pass,
                "failed_gates": [
                    name for name, value in gate_results.items() if not value["pass"]
                ],
                "output": str(args.out_dir / "confirmation_analysis.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
