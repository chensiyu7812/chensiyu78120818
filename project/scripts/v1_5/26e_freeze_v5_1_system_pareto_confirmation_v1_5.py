#!/usr/bin/env python3
"""Freeze the one-shot V5.1 system-level Pareto confirmation contract.

This script deliberately does not read confirmation, sealed, or external
outcomes.  It preserves the formal V5 per-head result and registers a separate
system-level decision rule that matches the paper's actual QRC claim.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.1-system-pareto-confirmation-freeze-v1"

FIT_REPORT = ROOT / "outputs/pm_v1_5_v5_fit_training_result_v1/fit_report.json"
PARETO_REPORT = ROOT / "outputs/pm_v1_5_v5_oof_pareto_diagnostic_v1/diagnostic_report.json"
BLUEPRINT = ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_1_system_pareto_confirmation_freeze_v1"

EXPECTED_HASHES = {
    "fit_report": "65833df7c2004955c59e63ed3f0467678e9598014623e4f925b8c6c9d4c35531",
    "pareto_report": "dc8f7cf94eb9cf9298a9abc4c443eb09fe4395a6e159b1143ee7d3bbb6dae0c8",
    "blueprint": "c601c7f16bad56c054d5e73c4a37f1fd22b8d329c35c128d06d6473bfc44e421",
}


def _confirmation_rows() -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in iter_jsonl(BLUEPRINT)
        if row.get("split") == "FRESH_CONFIRMATION"
    ]


def main() -> None:
    actual_hashes = {
        "fit_report": sha256_file(FIT_REPORT),
        "pareto_report": sha256_file(PARETO_REPORT),
        "blueprint": sha256_file(BLUEPRINT),
    }
    if actual_hashes != EXPECTED_HASHES:
        raise RuntimeError(
            "frozen V5 inputs changed; refuse to create a post-outcome contract: "
            f"{actual_hashes}"
        )

    fit = read_json(FIT_REPORT)
    pareto = read_json(PARETO_REPORT)
    rows = _confirmation_rows()
    groups = {str(row["counterfactual_group_id"]) for row in rows}
    by_component = Counter(str(row["target_component"]) for row in rows)
    if len(rows) != 128 or len(groups) != 64:
        raise RuntimeError("confirmation blueprint must contain 128 units in 64 groups")
    if by_component != Counter({"MP": 32, "MS": 32, "ME": 32, "RS": 32}):
        raise RuntimeError(f"unexpected component distribution: {by_component}")
    if any(row.get("paired_response_outcomes") is not None for row in rows):
        raise RuntimeError("confirmation outcomes are already populated; freeze refused")
    if any(row.get("external_text_or_outcome_read") is not False for row in rows):
        raise RuntimeError("confirmation blueprint is not outcome-blind")

    final_heads = {
        component: {
            "feature_names": head["feature_names"],
            "scaler_mean": head["final_model"]["scaler_mean"],
            "scaler_scale": head["final_model"]["scaler_scale"],
            "coefficients": head["final_model"]["coefficients"],
            "intercept": head["final_model"]["intercept"],
            "threshold": head["final_model"]["threshold"],
        }
        for component, head in fit["heads"].items()
    }
    contract = {
        "protocol": PROTOCOL,
        "status": "FROZEN_BEFORE_CONFIRMATION_OUTCOMES",
        "research_estimand": (
            "Whether the frozen learned pre-injection policy reduces material "
            "interaction-and-grounding risk and context cost while preserving "
            "immediate response quality relative to fixed-high and transparent rules."
        ),
        "decision_correction": {
            "preserve_v5_head_result": fit["status"],
            "why_system_confirmation_is_allowed": (
                "Per-head balanced-accuracy promotion was a diagnostic gate, not the "
                "paper's asymmetric quality-risk-cost utility. The frozen grouped-OOF "
                "policy exhibits a non-degenerate development Pareto signal."
            ),
            "not_allowed": [
                "relabel V5 head qualification as passed",
                "change a feature, model, threshold, resource, executor, or generator",
                "use confirmation, sealed, or external outcomes for repair",
                "create another small human-review development packet",
            ],
        },
        "frozen_policy": {
            "model_family": "per-component standardized L2 logistic regression",
            "joint_action": "independent MP/MS/ME/RS bits projected to the legal 16-action interface",
            "heads": final_heads,
        },
        "confirmation_population": {
            "split": "FRESH_CONFIRMATION",
            "state_candidate_units": len(rows),
            "counterfactual_groups": len(groups),
            "units_per_component": dict(sorted(by_component.items())),
            "generation_seeds_per_unit": 2,
            "review_unit": "same-state blinded ON/OFF pair; both arms receive quality and risk review",
        },
        "pre_execution_data_gate": {
            "timing": "after actual Rank-1 materialization and before any confirmation API call",
            "required": [
                "exactly 128 units, 64 counterfactual groups, and 32 units per component",
                "no counterfactual_group_id shared with EFFECT_FIT",
                "zero canonical current-state-plus-candidate duplicates across EFFECT_FIT and FRESH_CONFIRMATION",
                "paired counterfactual units remain in one statistical group",
                "actual Rank-1 candidate is non-empty, owner/time bound, and produced by the same frozen retriever",
                "construction benefit enrichment, split identity, topic prefix length, and outcome are absent from model inputs",
                "visible cue phrases are not perfectly predictive of ON/OFF outcome or candidate subtype",
            ],
            "if_failed": (
                "repair only the outcome-blind construction/materialization bug before any "
                "confirmation call; never use generated replies or labels to select the repair"
            ),
        },
        "comparators": [
            "always_off",
            "component_fixed_high",
            "transparent_rule",
            "learned_pm",
        ],
        "metrics": {
            "quality_score": "better=+1, tie=0, worse=-1 for learned minus comparator",
            "quality_uncertainty": "95% percentile bootstrap clustered by counterfactual_group_id",
            "material_risk": "named interaction-and-grounding event rate; both selected arms reviewed",
            "risk_uncertainty": "95% percentile bootstrap clustered by counterfactual_group_id",
            "cost": "reported prompt and total tokens on the realized selected arm",
            "policy_activity": "fraction of state-candidate units requested ON",
        },
        "single_use_system_gate": {
            "quality_vs_fixed_high": {
                "lower_95_ci_min": -0.05,
                "meaning": "noninferior immediate response quality",
            },
            "quality_vs_always_off": {
                "point_estimate_min": 0.0,
                "lower_95_ci_min": -0.05,
                "meaning": "resources retain non-negative value rather than being universally useless",
            },
            "risk_vs_fixed_high": {
                "point_difference_max": 0.0,
                "upper_95_ci_max": 0.05,
                "meaning": "directionally lower and statistically noninferior material risk",
            },
            "prompt_cost_vs_fixed_high": {
                "relative_reduction_min": 0.05,
            },
            "transparent_rule_comparison": {
                "quality_lower_95_ci_min": -0.05,
                "risk_point_difference_max": 0.0,
                "prompt_token_point_difference_max": 0.0,
                "at_least_one_strict_point_improvement": True,
            },
            "nondegenerate_policy": {
                "requested_on_fraction_min": 0.10,
                "requested_on_fraction_max": 0.90,
            },
            "critical_grounding_failures": {
                "categories": ["fabricated_recall", "internal_resource_label_exposure"],
                "learned_count_max": 2,
                "learned_count_must_not_exceed_fixed_high": True,
            },
            "all_checks_required": True,
        },
        "development_evidence_not_confirmation": pareto["headline"],
        "stopping_rule": {
            "pass": "freeze the complete 16-action runtime and proceed once to sealed/internal/external comparison",
            "fail": "report a bounded or negative V1.5 result; do not repair on this confirmation set",
        },
        "input_sha256": actual_hashes,
        "confirmation_or_external_outcomes_read": False,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CONTRACT, contract)
    write_json(
        OUT_DIR / "freeze_report.json",
        {
            "protocol": PROTOCOL,
            "status": contract["status"],
            "contract_path": str(CONTRACT.relative_to(ROOT)),
            "confirmation_population": contract["confirmation_population"],
            "frozen_policy_thresholds": {
                component: head["threshold"] for component, head in final_heads.items()
            },
            "development_status": pareto["status"],
            "confirmation_or_external_outcomes_read": False,
            "post_outcome_model_or_threshold_change": False,
        },
    )
    print(
        {
            "protocol": PROTOCOL,
            "status": contract["status"],
            "confirmation_units": len(rows),
            "confirmation_groups": len(groups),
            "outcomes_read": False,
        }
    )


if __name__ == "__main__":
    main()
