#!/usr/bin/env python3
"""Mechanically audit all V1.5b policies against the shared 16-action compiler."""

from __future__ import annotations

from itertools import product
from pathlib import Path

from metacom_pm.contracts import ALL_ACTION_IDS
from metacom_pm.io import read_json, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5b_policy_runtime import (
    COMPONENTS,
    ComponentOpportunity,
    compile_component_bits,
    route_policy,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-final-policy-compiler-audit-v1"
POLICIES = (
    "always_off",
    "fixed_high_resource",
    "transparent_rule",
    "learned_pm_full",
    "learned_pm_conservative",
    "cost_matched_fixed",
)


def _observations(bits: dict[str, bool]) -> dict[str, ComponentOpportunity]:
    return {
        component: ComponentOpportunity(
            component=component,
            candidate_present=True,
            hard_gate_pass=True,
            transparent_rule_on=bool(bits[component]),
            learned_in_support=True,
            learned_on=bool(bits[component]),
            learned_probability=0.8 if bits[component] else 0.2,
            decision_reason="MECHANICAL_AUDIT_HEAD_ON"
            if bits[component]
            else "MECHANICAL_AUDIT_HEAD_OFF",
        )
        for component in COMPONENTS
    }


def build(*, root: Path = ROOT) -> dict:
    out_dir = root / "outputs/pm_v1_5b_final_policy_compiler_audit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_report_path = (
        root
        / "outputs/pm_v1_5b_internal_cost_matched_fixed_v1/cost_match_report.json"
    )
    cost_report = read_json(cost_report_path)
    if cost_report["status"] != "PASS_INTERNAL_COST_MATCHED_FIXED_FROZEN":
        raise RuntimeError("internal cost-matched fixed action is not frozen")
    cost_matched_action = str(cost_report["selected_action_id"])
    rows = []
    compiled_actions = set()
    for values in product((False, True), repeat=4):
        bits = dict(zip(COMPONENTS, values, strict=True))
        compiled = compile_component_bits(bits)
        compiled_actions.add(compiled)
        observations = _observations(bits)
        routes = {
            policy: route_policy(
                policy=policy,
                opportunities=observations,
                cost_matched_action_id=cost_matched_action,
            )
            for policy in POLICIES
        }
        rows.append(
            {
                "protocol": PROTOCOL,
                "input_bits": bits,
                "canonical_action_id": compiled,
                "policy_routes": {
                    name: {
                        "action_id": route.action_id,
                        "requested_bits": dict(route.requested_bits),
                        "reasons": dict(route.reasons),
                    }
                    for name, route in routes.items()
                },
            }
        )
    rows_path = out_dir / "mechanical_routes.jsonl"
    write_jsonl(rows_path, rows)
    checks = {
        "all_16_actions_reachable": compiled_actions == set(ALL_ACTION_IDS),
        "always_off_always_m0_r0": all(
            row["policy_routes"]["always_off"]["action_id"] == "M0+R0"
            for row in rows
        ),
        "full_head_preserves_all_four_input_bits": all(
            row["policy_routes"]["learned_pm_full"]["requested_bits"]
            == row["input_bits"]
            for row in rows
        ),
        "conservative_masks_only_mp_ms": all(
            not row["policy_routes"]["learned_pm_conservative"]["requested_bits"]["MP"]
            and not row["policy_routes"]["learned_pm_conservative"]["requested_bits"]["MS"]
            and row["policy_routes"]["learned_pm_conservative"]["requested_bits"]["ME"]
            == row["input_bits"]["ME"]
            and row["policy_routes"]["learned_pm_conservative"]["requested_bits"]["RS"]
            == row["input_bits"]["RS"]
            for row in rows
        ),
        "cost_matched_fixed_preserves_frozen_action_when_eligible": all(
            row["policy_routes"]["cost_matched_fixed"]["action_id"]
            == cost_matched_action
            for row in rows
        ),
    }
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_SHARED_16_ACTION_COMPILER"
        if all(checks.values())
        else "FAIL_POLICY_COMPILER",
        "checks": checks,
        "policy_names": list(POLICIES),
        "mechanical_input_patterns": len(rows),
        "canonical_actions_reached": sorted(compiled_actions),
        "cost_matched_action_id": cost_matched_action,
        "cost_matched_action_is_frozen_from_internal_tokens_only": True,
        "external_quality_risk_or_outcome_read": False,
        "cost_match_report_sha256": sha256_file(cost_report_path),
        "routes_sha256": sha256_file(rows_path),
    }
    write_json(out_dir / "compiler_audit_report.json", report)
    return report


def main() -> None:
    print(build())


if __name__ == "__main__":
    main()
