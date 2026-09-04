#!/usr/bin/env python3
"""Fail-closed preflight for the corrected factorized PM Step-1 target."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-step1-factorized-causal-preflight-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _check(
    check_id: str,
    passed: bool,
    evidence: Any,
    failure_action: str,
    *,
    gate: str,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "gate": gate,
        "passed": bool(passed),
        "evidence": evidence,
        "failure_action": failure_action,
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    contract_path = (
        root
        / "data/pm_v1_5_contracts"
        / "pm_step1_factorized_training_correction_v1.json"
    )
    model_path = (
        root / "outputs/pm_v1_5_transparent_pm_rs_v1/pm_rs_model.json"
    )
    labels_path = (
        root
        / "outputs/pm_v1_5_rs_six_card_labels_v1"
        / "pm_rs_training_labels.jsonl"
    )
    label_report_path = (
        root
        / "outputs/pm_v1_5_rs_six_card_labels_v1/label_report.json"
    )
    bge_path = (
        root
        / "outputs/pm_v1_5_transparent_pm_rs_v1"
        / "bge_feature_diagnostic.json"
    )
    opportunity_path = (
        root
        / "outputs/pm_v1_5_rs_opportunity_layer_v1"
        / "opportunity_report.json"
    )
    review_contract_path = (
        root
        / "data/pm_v1_5_contracts"
        / "rs_corrected_supplement_review_v1.json"
    )
    wave2_plan_path = (
        root
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v2_confirmation"
        / "plan_report.json"
    )
    wave2_outcomes_path = (
        root
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v2_confirmation_execution"
        / "generation_outcomes.jsonl"
    )
    old_freeze_path = (
        root
        / "outputs/pm_v1_5_pm_rs_wave2_confirmation_freeze_v1"
        / "confirmation_contract.json"
    )

    required = [
        contract_path,
        model_path,
        labels_path,
        label_report_path,
        bge_path,
        wave2_plan_path,
        old_freeze_path,
        review_contract_path,
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        return {
            "protocol": PROTOCOL,
            "status": "BLOCKED_MISSING_REQUIRED_EVIDENCE",
            "missing": missing,
            "checks": [],
        }

    contract = read_json(contract_path)
    model = read_json(model_path)
    label_report = read_json(label_report_path)
    bge = read_json(bge_path)
    opportunity = (
        read_json(opportunity_path) if opportunity_path.exists() else None
    )
    wave2_plan = read_json(wave2_plan_path)
    old_freeze = read_json(old_freeze_path)
    labels = _rows(labels_path)

    n_groups = len({row["user_id"] for row in labels})
    n_features = len(model["feature_names"])
    max_features = n_groups // 5
    oof = model["oof"]["metrics"]

    by_move: dict[str, Counter[int]] = defaultdict(Counter)
    for row in labels:
        by_move[str(row["selected_strategy_family"])][int(row["target_y"])] += 1
    per_move_cells = {
        move: {
            "nonpositive": counts[0],
            "positive": counts[1],
            "total": counts[0] + counts[1],
        }
        for move, counts in sorted(by_move.items())
    }
    min_direction = int(
        contract["minimum_data_and_capacity_guards"][
            "minimum_positive_groups_per_binary_head"
        ]
    )
    per_move_bidirectional = all(
        cell["positive"] > 0 and cell["nonpositive"] > 0
        for cell in per_move_cells.values()
    )

    risk_events: Counter[str] = Counter()
    risk_applicable_responses = 2 * len(labels)
    for row in labels:
        for risk in row.get("r0_material_risks", []):
            risk_events[str(risk)] += 1
        for risk in row.get("rs_material_risks", []):
            risk_events[str(risk)] += 1
    risk_support = {
        risk: {
            "events": count,
            "non_events": risk_applicable_responses - count,
            "learnable": (
                count >= min_direction
                and risk_applicable_responses - count >= min_direction
            ),
        }
        for risk, count in sorted(risk_events.items())
    }
    atomic_risk_heads_ready = bool(risk_support) and all(
        item["learnable"] for item in risk_support.values()
    )

    checks = [
        _check(
            "contract_frozen_before_wave2_outcomes",
            (
                contract["status"]
                == "FROZEN_BEFORE_ANY_WAVE2_RESPONSE_OUTCOME"
                and not wave2_outcomes_path.exists()
            ),
            {
                "contract_status": contract["status"],
                "wave2_outcomes_present": wave2_outcomes_path.exists(),
            },
            "Do not generate wave2 outcomes under an unfrozen or contaminated contract.",
            gate="data_collection",
        ),
        _check(
            "full_step1_opportunity_layer_exists",
            (
                opportunity is not None
                and opportunity["status"]
                == "AUDITED_TRANSPARENT_OPPORTUNITY_BASELINE_READY"
            ),
            {
                "expected_artifact": str(opportunity_path.relative_to(root)),
                "present": opportunity_path.exists(),
                "status": (
                    opportunity.get("status") if opportunity is not None else None
                ),
            },
            "Build and audit the outcome-blind opportunity layer before calling this a full Step1 PM.",
            gate="data_collection",
        ),
        _check(
            "corrected_atomic_risk_review_contract_frozen",
            read_json(review_contract_path)["status"]
            == "FROZEN_BEFORE_SUPPLEMENT_SELECTION_AND_OUTCOMES",
            {
                "path": str(review_contract_path.relative_to(root)),
                "status": read_json(review_contract_path)["status"],
                "multi_select": read_json(review_contract_path)["risk"][
                    "stage2_if_yes"
                ]["multi_select"],
            },
            "Freeze the two-stage multi-select risk review before selecting or generating supplement outcomes.",
            gate="data_collection",
        ),
        _check(
            "wave1_has_both_quality_effect_directions",
            (
                int(model["class_counts"]["RS"]) >= min_direction
                and int(model["class_counts"]["R0"]) >= min_direction
                and per_move_bidirectional
            ),
            {
                "overall": model["class_counts"],
                "minimum_overall_each_direction": min_direction,
                "per_move_cells": per_move_cells,
                "per_move_requirement": "at least one observed direction each; move-specific heads are not authorized",
            },
            "Do not expand an effect pilot with a degenerate overall target or one-way-only move strata.",
            gate="data_collection",
        ),
        _check(
            "fresh_pool_can_supply_minimal_16_group_supplement",
            (
                int(wave2_plan["independent_dialogue_groups"]) >= 16
                and int(wave2_plan["planned_pairs"]) >= 16
            ),
            {
                "available_independent_groups": wave2_plan[
                    "independent_dialogue_groups"
                ],
                "prepared_pairs": wave2_plan["planned_pairs"],
                "maximum_corrected_supplement_pairs": 16,
            },
            "Prepare no more than 16 fresh groups; do not execute the old 32-pair confirmation wholesale.",
            gate="data_collection",
        ),
        _check(
            "effect_pilot_has_minimum_independent_groups",
            n_groups
            >= int(
                contract["minimum_data_and_capacity_guards"][
                    "clean_effect_groups_per_component_pilot"
                ]["minimum"]
            ),
            {
                "independent_groups": n_groups,
                "minimum": contract["minimum_data_and_capacity_guards"][
                    "clean_effect_groups_per_component_pilot"
                ]["minimum"],
            },
            "Add only pre-registered clean, independent effect groups; do not pool incompatible historical rounds.",
            gate="model_promotion",
        ),
        _check(
            "primary_feature_dimension_within_capacity",
            n_features <= max_features,
            {
                "features": n_features,
                "independent_groups": n_groups,
                "maximum_features": max_features,
            },
            "Replace the 22-feature candidate with a pre-outcome feature set no larger than floor(groups/5).",
            gate="model_promotion",
        ),
        _check(
            "primary_model_beats_prevalence_on_proper_scores",
            (
                float(oof["oof_log_loss"])
                < float(oof["prevalence_only_log_loss"])
                and float(oof["oof_brier"])
                < float(oof["prevalence_only_brier"])
            ),
            {
                "model_log_loss": oof["oof_log_loss"],
                "prevalence_log_loss": oof["prevalence_only_log_loss"],
                "model_brier": oof["oof_brier"],
                "prevalence_brier": oof["prevalence_only_brier"],
            },
            "Do not confirm or promote a candidate that is worse than the prevalence-only baseline.",
            gate="model_promotion",
        ),
        _check(
            "quality_risk_cost_are_not_collapsed_into_one_training_head",
            False,
            {
                "current_label_decision_order": label_report["decision_order"],
                "current_model_scope": model["scope"],
                "correction": "quality head, atomic risk heads, deterministic cost, then selector",
            },
            "Train and report the separate heads; keep the composite action only for end-to-end auditing.",
            gate="model_promotion",
        ),
        _check(
            "atomic_risk_heads_have_event_and_nonevent_support",
            atomic_risk_heads_ready,
            {
                "minimum_each_direction": min_direction,
                "applicable_response_upper_bound": risk_applicable_responses,
                "observed": risk_support,
            },
            "Use deterministic risk guards and report INSUFFICIENT_EVIDENCE until each atomic head has enough events and non-events.",
            gate="model_promotion",
        ),
        _check(
            "baai_not_promoted_after_negative_diagnostic",
            not bool(bge["bge_feature_promotion_gate_passed"]),
            {
                "promotion_gate": bge["bge_feature_promotion_gate_passed"],
                "decision": bge["decision"],
            },
            "Keep BAAI out of the V1.5 RS ranker and PM head.",
            gate="data_collection",
        ),
        _check(
            "old_wave2_freeze_is_secondary_only",
            (
                old_freeze["status"]
                == "FROZEN_BEFORE_WAVE2_GENERATION_AND_LABELS"
                and contract["rs_specific_corrections"]["wave2_role"].startswith(
                    "No response generation"
                )
            ),
            {
                "old_freeze_status": old_freeze["status"],
                "old_frozen_actions": old_freeze["frozen_action_counts"],
                "new_role": "historical secondary prospective diagnostic only",
            },
            "Never use the old freeze as the primary authorization for wave2 generation.",
            gate="data_collection",
        ),
        _check(
            "wave2_plan_does_not_claim_execution_authority",
            (
                wave2_plan["status"]
                == "READY_FOR_NO_PM_DIRECT_EFFECT_GENERATION"
                and not wave2_outcomes_path.exists()
            ),
            {
                "plan_status": wave2_plan["status"],
                "planned_calls": wave2_plan["planned_generation_calls"],
                "outcomes_present": wave2_outcomes_path.exists(),
            },
            "Treat the existing plan as a prepared pool only; create a fresh identity after all scientific gates pass.",
            gate="data_collection",
        ),
    ]
    collection_failed = [
        item["check_id"]
        for item in checks
        if item["gate"] == "data_collection" and not item["passed"]
    ]
    promotion_failed = [
        item["check_id"]
        for item in checks
        if item["gate"] == "model_promotion" and not item["passed"]
    ]
    collection_status = (
        "READY_TO_PREPARE_CORRECTED_16_PAIR_SUPPLEMENT"
        if not collection_failed
        else "BLOCKED_BEFORE_CORRECTED_SUPPLEMENT_PREPARATION"
    )
    promotion_status = (
        "READY_FOR_MODEL_CONFIRMATION"
        if not promotion_failed
        else "BLOCKED_BEFORE_MODEL_CONFIRMATION"
    )
    return {
        "protocol": PROTOCOL,
        "status": (
            "READY_TO_PREPARE_CORRECTED_SUPPLEMENT_MODEL_PROMOTION_BLOCKED"
            if not collection_failed and promotion_failed
            else (
                "READY_FOR_MODEL_CONFIRMATION"
                if not collection_failed and not promotion_failed
                else "BLOCKED_BEFORE_CORRECTED_SUPPLEMENT_PREPARATION"
            )
        ),
        "decision": (
            "Do not run the prepared 64 wave2 calls. Prepare at most 16 "
            "fresh supplement pairs under a new identity; no paid execution "
            "is authorized by this report."
        ),
        "data_collection_gate": {
            "status": collection_status,
            "failed_checks": collection_failed,
        },
        "model_promotion_gate": {
            "status": promotion_status,
            "failed_checks": promotion_failed,
        },
        "failed_checks": collection_failed + promotion_failed,
        "checks": checks,
        "next_scientific_actions": [
            "Freeze a six-or-fewer-feature primary representation before any new outcomes.",
            "Build an outcome-blind RS opportunity layer and report it separately.",
            "Prepare at most 16 additional clean effect groups to reach 48; do not run the old 32-pair confirmation wholesale.",
            "Train quality and applicable atomic-risk heads separately; keep cost deterministic.",
            "If corrected grouped OOF still does not beat prevalence/rule baselines, report individualized RS routing as not learned and stop RS expansion for V1.5."
        ],
        "lineage": {
            "contract_sha256": sha256_file(contract_path),
            "wave1_model_sha256": sha256_file(model_path),
            "wave1_labels_sha256": sha256_file(labels_path),
            "bge_diagnostic_sha256": sha256_file(bge_path),
            "wave2_plan_sha256": sha256_file(wave2_plan_path),
            "old_wave2_freeze_sha256": sha256_file(old_freeze_path),
            "opportunity_report_sha256": (
                sha256_file(opportunity_path)
                if opportunity_path.exists()
                else None
            ),
            "corrected_review_contract_sha256": sha256_file(
                review_contract_path
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_step1_factorized_preflight_v1"
            / "preflight_report.json"
        ),
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless all corrected scientific gates pass.",
    )
    args = parser.parse_args()
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(report)
    if (
        args.require_ready
        and report["model_promotion_gate"]["status"]
        != "READY_FOR_MODEL_CONFIRMATION"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
