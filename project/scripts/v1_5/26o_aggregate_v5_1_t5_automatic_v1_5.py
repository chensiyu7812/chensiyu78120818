#!/usr/bin/env python3
"""Aggregate objective T5 coverage, routing, fallback and token-cost metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.1-t5-automatic-evaluation-v1"
PLAN_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_plan_v1"
EXECUTION_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_execution_v1"
SURFACE_REPORT = ROOT / "outputs/pm_v1_5_v5_1_t5_surfaces_v1/surface_gate_report.json"
POLICIES = (
    "always_off",
    "fixed_high_eligible",
    "transparent_rule",
    "cost_matched_fixed",
    "learned_pm",
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build(*, out_dir: Path) -> dict[str, Any]:
    manifest = read_json(PLAN_DIR / "freeze_manifest.json")
    summary = read_json(EXECUTION_DIR / "execution_summary.json")
    surface = read_json(SURFACE_REPORT)
    if manifest.get("status") != "READY_FOR_SINGLE_T5_GENERATION":
        raise RuntimeError("T5 plan is not frozen")
    if summary.get("status") != "COMPLETE_AWAITING_FROZEN_AUTOMATIC_JUDGE_AND_HUMAN_EVALUATION":
        raise RuntimeError("T5 generation is incomplete")
    if summary["call_plan_sha256"] != manifest["call_plan_sha256"]:
        raise RuntimeError("T5 execution does not bind the frozen call plan")

    bindings = rows(PLAN_DIR / "policy_bindings_private.jsonl")
    calls = rows(PLAN_DIR / "call_plan_private.jsonl")
    outcomes = rows(EXECUTION_DIR / "outcomes_ordered_private.jsonl")
    if len(calls) != len(outcomes):
        raise RuntimeError("T5 call/outcome count mismatch")
    outcome_by_key = {
        (str(row["state_id"]), str(row["seed_hex"]), str(row["requested_action_id"])): row
        for row in outcomes
    }
    if len(outcome_by_key) != len(outcomes):
        raise RuntimeError("T5 outcome keys are not unique")

    policy_rows: list[dict[str, Any]] = []
    for binding in bindings:
        for seed_index, seed_hex in enumerate(binding["generation_seed_hex"], start=1):
            for policy in POLICIES:
                route = binding["policy_routes"][policy]
                key = (
                    str(binding["state_id"]),
                    str(seed_hex),
                    str(route["action_id"]),
                )
                outcome = outcome_by_key[key]
                policy_rows.append(
                    {
                        "protocol": PROTOCOL,
                        "domain": binding["domain"],
                        "partition": binding["partition"],
                        "state_id": binding["state_id"],
                        "group_id_private_analysis_only": binding["group_id_private_analysis_only"],
                        "user_id_private_analysis_only": binding["user_id_private_analysis_only"],
                        "target_component_private_analysis_only": binding["target_component_private_analysis_only"],
                        "seed_index": seed_index,
                        "seed_hex": seed_hex,
                        "policy": policy,
                        "requested_bits_before_feasibility": route["requested_bits_before_feasibility"],
                        "feasible_bits": route["feasible_bits"],
                        "requested_action_id": route["action_id"],
                        "realized_action_id": outcome["realized_action_id"],
                        "fallback_used": outcome["fallback_used"],
                        "guard_errors": outcome["guard_errors"],
                        "prompt_tokens": int(outcome["usage"]["prompt_tokens"]),
                        "completion_tokens": int(outcome["usage"]["completion_tokens"]),
                        "total_tokens": int(outcome["usage"]["total_tokens"]),
                        "final_response_sha256": __import__("hashlib").sha256(
                            str(outcome["final_response"]).encode("utf-8")
                        ).hexdigest(),
                        "policy_alias_reused_same_action_response": sum(
                            str(other["action_id"]) == str(route["action_id"])
                            for other in binding["policy_routes"].values()
                        ) > 1,
                    }
                )

    aggregate: dict[str, Any] = {}
    for partition in sorted({str(row["partition"]) for row in policy_rows}):
        partition_rows = [row for row in policy_rows if row["partition"] == partition]
        by_policy: dict[str, Any] = {}
        for policy in POLICIES:
            selected = [row for row in partition_rows if row["policy"] == policy]
            states = len({str(row["state_id"]) for row in selected})
            by_policy[policy] = {
                "state_seed_rows": len(selected),
                "states": states,
                "requested_action_distribution": dict(Counter(str(row["requested_action_id"]) for row in selected)),
                "realized_action_distribution": dict(Counter(str(row["realized_action_id"]) for row in selected)),
                "fallback_count": sum(bool(row["fallback_used"]) for row in selected),
                "fallback_rate": mean([float(bool(row["fallback_used"])) for row in selected]),
                "mean_prompt_tokens": mean([float(row["prompt_tokens"]) for row in selected]),
                "mean_completion_tokens": mean([float(row["completion_tokens"]) for row in selected]),
                "mean_total_tokens": mean([float(row["total_tokens"]) for row in selected]),
                "component_requested_on_fraction_before_feasibility": {
                    component: mean([
                        float(row["requested_bits_before_feasibility"][component])
                        for row in selected
                    ])
                    for component in ("MP", "MS", "ME", "RS")
                },
                "component_feasible_on_fraction": {
                    component: mean([
                        float(row["feasible_bits"][component]) for row in selected
                    ])
                    for component in ("MP", "MS", "ME", "RS")
                },
            }
        learned = by_policy["learned_pm"]
        comparisons = {}
        for comparator in POLICIES[:-1]:
            base = by_policy[comparator]
            comparisons[comparator] = {
                "prompt_token_difference_learned_minus_comparator": learned["mean_prompt_tokens"] - base["mean_prompt_tokens"],
                "prompt_token_relative_reduction": (
                    (base["mean_prompt_tokens"] - learned["mean_prompt_tokens"])
                    / max(1.0, base["mean_prompt_tokens"])
                ),
                "total_token_difference_learned_minus_comparator": learned["mean_total_tokens"] - base["mean_total_tokens"],
                "fallback_rate_difference": learned["fallback_rate"] - base["fallback_rate"],
            }
        aggregate[partition] = {
            "policies": by_policy,
            "learned_comparisons": comparisons,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = out_dir / "policy_outcomes_private.jsonl"
    write_jsonl(policy_path, policy_rows)
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_OBJECTIVE_T5_METRICS",
        "states": len(bindings),
        "policy_state_seed_rows": len(policy_rows),
        "unique_generator_calls": len(outcomes),
        "coverage": surface["coverage"],
        "routing_cost_fallback": aggregate,
        "responsibility_boundary": {
            "candidate_absent_or_unsupported_subtype": "retrieval/transport coverage",
            "head_requested_but_infeasible": "feasibility projection, not generator",
            "requested_and_feasible_but_guard_fallback": "Step2/generator execution",
            "final_quality_or_material_risk": "human primary and independent LLM sensitivity, not inferred here",
        },
        "input_sha256": {
            "freeze_manifest": sha256_file(PLAN_DIR / "freeze_manifest.json"),
            "policy_bindings": sha256_file(PLAN_DIR / "policy_bindings_private.jsonl"),
            "call_plan": sha256_file(PLAN_DIR / "call_plan_private.jsonl"),
            "execution_summary": sha256_file(EXECUTION_DIR / "execution_summary.json"),
            "ordered_outcomes": sha256_file(EXECUTION_DIR / "outcomes_ordered_private.jsonl"),
            "surface_report": sha256_file(SURFACE_REPORT),
        },
        "policy_outcomes_sha256": sha256_file(policy_path),
        "method_changed_from_t5_outcomes": False,
    }
    write_json(out_dir / "automatic_report.json", report)
    return report


def main() -> None:
    report = build(
        out_dir=ROOT / "outputs/pm_v1_5_v5_1_t5_automatic_v1"
    )
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "status": report["status"],
                "states": report["states"],
                "unique_generator_calls": report["unique_generator_calls"],
            }
        )
    )


if __name__ == "__main__":
    main()
