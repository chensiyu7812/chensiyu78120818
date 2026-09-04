#!/usr/bin/env python3
"""Freeze the sole content-disjoint V5.2 internal confirmation identity.

This stage is deliberately zero-API and outcome-blind.  It creates one new
synthetic, within-design examination surface after the V5.2 FIT result was
frozen.  The logic families and resource stack are unchanged; only ordinary
non-clinical topic content and deterministic IDs/seeds are new.

The script also records why the old FRESH/SEALED identities cannot be reused:
both already have saved generated outcomes.  Rerunning this script is allowed
only when it reproduces byte-identical frozen artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import normalize_for_hash
from metacom_pm.v1_5_v3_candidate_materialization import materialize_v3_candidates
from metacom_pm.v1_5_v3_effect_blueprint import (
    EFFECT_PAIRS,
    HISTORY_SCALES,
    PROTOCOL as BLUEPRINT_PROTOCOL,
    SOURCE_CATALOG_TARGETS,
)
from metacom_pm.v1_5_v3_state_realization import realize_v3_states
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-content-disjoint-confirmation-freeze-v1"
SPLIT = "V5_2_CONTENT_DISJOINT_CONFIRMATION"
TOPICS = (
    "commute_disruption",
    "volunteer_commitment",
    "household_coordination",
    "hobby_group_tension",
    "course_project",
    "seasonal_routine_change",
    "shared_space_noise",
    "appointment_planning_nonurgent",
)

OLD_BLUEPRINT = (
    ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
)
OLD_CANDIDATES = (
    ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
)
STRATEGY_BANK = (
    ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
)
FIT_REPORT = ROOT / "outputs/pm_v1_5_v5_2_fit_final_training_v1/final_fit_report.json"
PARETO_REPORT = (
    ROOT / "outputs/pm_v1_5_v5_2_oof_pareto_diagnostic_v1/diagnostic_report.json"
)
V52_EXECUTOR_CONTRACT = (
    ROOT / "data/pm_v1_5_contracts/v5_2_locked_executor_repair_v1.json"
)
OLD_EXECUTIONS = {
    "EFFECT_FIT": ROOT
    / "outputs/pm_v1_5_v5_2_locked_fit_execution_v2/outcomes_private.jsonl",
    "FRESH_CONFIRMATION": ROOT
    / "outputs/pm_v1_5_v5_1_confirmation_execution_v1/outcomes_private.jsonl",
    "SEALED_INTERNAL_TEST": ROOT
    / "outputs/pm_v1_5_v5_1_t5_execution_v1/outcomes_private.jsonl",
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def candidate_requirements(component: str, subtype: str) -> dict[str, Any]:
    return {
        "subtype": subtype,
        "same_owner": True,
        "time_valid": True,
        "goal_function_compatible": True,
        "boundary_burden_compatible": True,
        "currently_redundant": False,
        "specific_increment_present": True,
        "must_be_rediscovered_by_formal_retriever": True,
        "component_specific": {
            "MP": "stable_preference_or_relevant_incremental_profile",
            "MS": "strictly_prior_session_goal_distinction_or_thread",
            "ME": "past_action_or_choice_plus_result_or_mechanism",
            "RS": "bank_atomic_move_with_no_card_external_advice",
        }[component],
    }


def build_blueprint() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for component_index, component in enumerate(COMPONENTS):
        for pair_index, pair in enumerate(EFFECT_PAIRS[component]):
            for ordinal in range(4):
                topic = TOPICS[(ordinal + 2 * pair_index + component_index) % len(TOPICS)]
                scale = HISTORY_SCALES[ordinal % len(HISTORY_SCALES)]
                group_id = "v52confirm_cf_" + stable_hex(
                    PROTOCOL, component, pair["pair"], ordinal, n=20
                )
                surface_family = "v52confirm_surface_" + stable_hex(
                    PROTOCOL, component, pair["pair"], ordinal, n=20
                )
                for enrichment, logic_family in (
                    ("HIGH", pair["high"]),
                    ("LOW_OR_NEUTRAL", pair["low"]),
                ):
                    row_id = "v52confirm_" + stable_hex(
                        PROTOCOL, component, logic_family, ordinal, n=24
                    )
                    result.append(
                        {
                            "protocol": BLUEPRINT_PROTOCOL,
                            "blueprint_row_id": row_id,
                            "user_id": "user_" + row_id,
                            "group_id": "group_" + row_id,
                            "track": "COMPONENT_EFFECT",
                            "split": SPLIT,
                            "target_component": component,
                            "content_origin": "SYNTHETIC_INTERNAL_V5_2_ONE_SHOT_CONTENT_DISJOINT",
                            "external_text_or_outcome_read": False,
                            "construction_intent_is_gold": False,
                            "construction_intent_is_model_input": False,
                            "exact_rank1_candidate": None,
                            "human_eligibility_gold": None,
                            "step2_qualification": None,
                            "paired_response_outcomes": None,
                            "worth_opening_gold": None,
                            "logic_family": logic_family,
                            "logic_pair": pair["pair"],
                            "surface_family": surface_family,
                            "counterfactual_group_id": group_id,
                            "topic_family": topic,
                            "history_scale": scale,
                            "source_catalog_size_target": SOURCE_CATALOG_TARGETS[component][scale],
                            "explicit_current_goal": pair["goal"],
                            "candidate_subtype_target": pair["subtype"],
                            "required_candidate_function": pair["function"],
                            "external_construct_scope": pair["external_scope"],
                            "private_benefit_enrichment": enrichment,
                            "intended_eligibility_for_construction_only": "ELIGIBLE",
                            "candidate_requirements": candidate_requirements(
                                component, pair["subtype"]
                            ),
                            "effect_treatments": {
                                "control": "M0+R0",
                                "treatment": {
                                    "MP": "MP+R0",
                                    "MS": "MS+R0",
                                    "ME": "ME+R0",
                                    "RS": "M0+RS",
                                }[component],
                                "generation_seeds": [
                                    stable_hex(PROTOCOL, row_id, "seed-a", n=16),
                                    stable_hex(PROTOCOL, row_id, "seed-b", n=16),
                                ],
                                "single_component_change_only": True,
                            },
                        }
                    )
    return result


def decision_signature(row: Mapping[str, Any]) -> str:
    payload = {
        "component": str(row["target_component_private_not_model_input"]),
        "visible_dialogue": [
            {
                "role": str(turn["role"]),
                "content": normalize_for_hash(str(turn["content"])),
            }
            for turn in row["visible_dialogue"]
        ],
        "current_user_text": normalize_for_hash(str(row["current_user_text"])),
        "candidate": normalize_for_hash(
            str(row["exact_rank1_candidate"]["candidate_text"])
        ),
    }
    return sha256_text(canonical_json(payload))


def visible_signature(row: Mapping[str, Any]) -> str:
    payload = {
        "visible_dialogue": [
            normalize_for_hash(str(turn["content"])) for turn in row["visible_dialogue"]
        ],
        "current_user_text": normalize_for_hash(str(row["current_user_text"])),
    }
    return sha256_text(canonical_json(payload))


def _old_exposure_counts(old_blueprint: list[dict[str, Any]]) -> dict[str, Any]:
    split_by_state = {str(row["blueprint_row_id"]): str(row["split"]) for row in old_blueprint}
    evidence: dict[str, Any] = {}
    for split, path in OLD_EXECUTIONS.items():
        saved = rows(path)
        state_ids = {
            str(row["state_id"])
            for row in saved
            if split_by_state.get(str(row["state_id"])) == split
        }
        expected = {state for state, value in split_by_state.items() if value == split}
        evidence[split] = {
            "saved_rows": len(saved),
            "expected_states": len(expected),
            "exposed_states": len(state_ids),
            "all_states_exposed": state_ids == expected,
            "source": str(path.relative_to(ROOT)),
            "source_sha256": sha256_file(path),
        }
    return evidence


def _write_once_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    if path.exists():
        old = path.read_text(encoding="utf-8")
        candidate = "".join(canonical_json(value) + "\n" for value in values)
        if old != candidate:
            raise RuntimeError(f"one-shot frozen artifact would change: {path}")
        return
    write_jsonl(path, values)


def _write_once_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"one-shot frozen artifact would change: {path}")
        return
    write_json(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_v5_2_confirmation_v1/private",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_freeze_v1",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.2 requires {FORMAL_PYTHON}; got {sys.executable}")
    required = [
        OLD_BLUEPRINT,
        OLD_CANDIDATES,
        STRATEGY_BANK,
        FIT_REPORT,
        PARETO_REPORT,
        V52_EXECUTOR_CONTRACT,
        *OLD_EXECUTIONS.values(),
    ]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required frozen input missing: {path}")

    fit = read_json(FIT_REPORT)
    pareto = read_json(PARETO_REPORT)
    executor = read_json(V52_EXECUTOR_CONTRACT)
    if executor.get("status") != "FROZEN_BEFORE_V5_2_FIT_GENERATION":
        raise RuntimeError("V5.2 executor contract is not frozen")
    if fit.get("post_label_executor_or_method_change_allowed") is not False:
        raise RuntimeError("V5.2 FIT does not prohibit post-label method changes")

    blueprint = build_blueprint()
    states = realize_v3_states(blueprint)
    candidate_rows, candidate_report = materialize_v3_candidates(
        states=states,
        blueprint_rows=blueprint,
        strategy_cards=rows(STRATEGY_BANK),
    )
    if candidate_report["status"] != "PASS":
        raise RuntimeError(candidate_report)

    old_blueprint = rows(OLD_BLUEPRINT)
    old_candidates = rows(OLD_CANDIDATES)
    old_ids = {str(row["state_id"]) for row in old_candidates}
    new_ids = {str(row["state_id"]) for row in candidate_rows}
    old_groups = {str(row["counterfactual_group_id"]) for row in old_blueprint}
    new_groups = {str(row["counterfactual_group_id"]) for row in blueprint}
    old_decisions = {decision_signature(row) for row in old_candidates}
    new_decisions = [decision_signature(row) for row in candidate_rows]
    old_visible = {visible_signature(row) for row in old_candidates}
    new_visible = [visible_signature(row) for row in candidate_rows]
    component_counts = Counter(
        str(row["target_component_private_not_model_input"]) for row in candidate_rows
    )
    logic_counts = Counter(str(row["logic_family"]) for row in blueprint)
    scale_counts = Counter(str(row["history_scale"]) for row in blueprint)
    topic_counts = Counter(str(row["topic_family"]) for row in blueprint)
    enrichment_counts = Counter(str(row["private_benefit_enrichment"]) for row in blueprint)
    group_members: dict[str, list[str]] = defaultdict(list)
    for row in blueprint:
        group_members[str(row["counterfactual_group_id"])].append(
            str(row["blueprint_row_id"])
        )

    old_exposure = _old_exposure_counts(old_blueprint)
    checks = {
        "128_units_64_two_member_groups": len(candidate_rows) == 128
        and len(group_members) == 64
        and all(len(value) == 2 for value in group_members.values()),
        "32_units_per_component": component_counts
        == Counter({component: 32 for component in COMPONENTS}),
        "four_per_logic_family": len(logic_counts) == 32
        and all(value == 4 for value in logic_counts.values()),
        "history_scale_balanced": scale_counts
        == Counter({scale: 32 for scale in HISTORY_SCALES}),
        "new_topic_balanced": topic_counts == Counter({topic: 16 for topic in TOPICS}),
        "private_enrichment_balanced": enrichment_counts
        == Counter({"HIGH": 64, "LOW_OR_NEUTRAL": 64}),
        "state_ids_disjoint_from_all_old_internal_splits": not bool(old_ids & new_ids),
        "counterfactual_groups_disjoint_from_all_old_internal_splits": not bool(
            old_groups & new_groups
        ),
        "complete_decision_surfaces_unique_within_confirmation": len(new_decisions)
        == len(set(new_decisions)),
        "complete_decision_surfaces_disjoint_from_all_old_internal_splits": not bool(
            old_decisions & set(new_decisions)
        ),
        "visible_surfaces_disjoint_from_all_old_internal_splits": not bool(
            old_visible & set(new_visible)
        ),
        "actual_rank1_present_and_same_stack": all(
            bool(row["exact_rank1_candidate"]["candidate_present"])
            and int(row["exact_rank1_candidate"]["selected_rank"]) == 1
            and row["response_or_outcome_read"] is False
            for row in candidate_rows
        ),
        "old_effect_fit_fresh_and_sealed_all_previously_exposed": all(
            value["all_states_exposed"] for value in old_exposure.values()
        ),
        "no_api_human_external_or_response_outcomes_read": True,
    }
    if not all(checks.values()):
        raise RuntimeError({key: value for key, value in checks.items() if not value})

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blueprint_path = args.data_dir / "construction_blueprint.jsonl"
    states_path = args.data_dir / "raw_states.jsonl"
    candidates_path = args.data_dir / "candidate_rows_private.jsonl"
    _write_once_jsonl(blueprint_path, blueprint)
    _write_once_jsonl(states_path, states)
    _write_once_jsonl(candidates_path, candidate_rows)

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
        "status": "FROZEN_BEFORE_ANY_V5_2_CONFIRMATION_GENERATION",
        "research_estimand": (
            "Whether the frozen learned 16-action pre-injection policy preserves "
            "immediate support quality while reducing material interaction-and-grounding "
            "risk and context cost relative to fixed-high and transparent rules."
        ),
        "identity": {
            "split": SPLIT,
            "state_candidate_units": 128,
            "counterfactual_groups": 64,
            "units_per_component": dict(sorted(component_counts.items())),
            "generation_seeds_per_unit": 2,
            "content_scope": "synthetic_nonclinical_within_design_confirmation",
            "topics": list(TOPICS),
            "seed_derivation": "protocol_and_state_id_sha256; no seed search",
        },
        "frozen_method": {
            "executor": executor["protocol"],
            "generator_and_generation_config": "same as V5.2 FIT",
            "strategy_bank": "same frozen 80-card V4 bank",
            "retriever_and_candidate_compiler": "same as V5.2 FIT",
            "policy_model_family": "four standardized L2 logistic heads",
            "heads": final_heads,
            "joint_action": "four bits projected to the existing legal 16-action interface",
        },
        "comparators": [
            "always_off",
            "component_fixed_high",
            "transparent_rule",
            "learned_pm",
        ],
        "measurement": {
            "quality": "same-state blinded material A/B preference; tie retained",
            "material_risk": "named interaction-and-grounding event rate on every selected arm",
            "cost": "reported prompt and total tokens of the realized selected arm",
            "uncertainty": "95% percentile bootstrap clustered by counterfactual_group_id",
            "human_review": (
                "one complete primary quality/risk panel; a protocol-hashed 20% overlap "
                "goes to an independent reviewer; disagreements adjudicated once"
            ),
            "llm_judge": (
                "pre-frozen independent robustness analysis only; never replaces human "
                "primary labels and never changes the policy"
            ),
        },
        "single_use_system_gate": {
            "quality_vs_fixed_high_lower_95_ci_min": -0.05,
            "quality_vs_always_off_point_min": 0.0,
            "quality_vs_always_off_lower_95_ci_min": -0.05,
            "risk_vs_fixed_high_point_difference_max": 0.0,
            "risk_vs_fixed_high_upper_95_ci_max": 0.05,
            "prompt_token_relative_reduction_vs_fixed_high_min": 0.05,
            "transparent_rule_quality_lower_95_ci_min": -0.05,
            "transparent_rule_risk_point_difference_max": 0.0,
            "transparent_rule_prompt_token_point_difference_max": 0.0,
            "transparent_rule_requires_one_strict_point_improvement": True,
            "learned_requested_on_fraction_min": 0.10,
            "learned_requested_on_fraction_max": 0.90,
            "critical_grounding_categories": [
                "fabricated_recall",
                "internal_resource_label_exposure",
            ],
            "critical_grounding_learned_count_max": 2,
            "critical_grounding_not_above_fixed_high": True,
            "all_checks_required_for_full_pass": True,
        },
        "stopping_rule": {
            "pass": "freeze the result and run corrected ESConv/EvoEmo only as external replication",
            "fail": (
                "report bounded/negative V1.5 confirmation; do not create a second "
                "confirmation set and do not modify the method on this evidence"
            ),
        },
        "external_evaluation_boundary": {
            "ESConv": "corrected RS-focused replication; no longitudinal-memory claim",
            "EvoEmo": "same per-user compiler with that user's strictly prior history; repaired replication",
            "old_external_outputs": "historical/diagnostic only, never V5.2 untouched confirmation",
        },
        "input_sha256": {
            str(FIT_REPORT.relative_to(ROOT)): sha256_file(FIT_REPORT),
            str(PARETO_REPORT.relative_to(ROOT)): sha256_file(PARETO_REPORT),
            str(V52_EXECUTOR_CONTRACT.relative_to(ROOT)): sha256_file(
                V52_EXECUTOR_CONTRACT
            ),
            str(OLD_BLUEPRINT.relative_to(ROOT)): sha256_file(OLD_BLUEPRINT),
            str(OLD_CANDIDATES.relative_to(ROOT)): sha256_file(OLD_CANDIDATES),
            str(STRATEGY_BANK.relative_to(ROOT)): sha256_file(STRATEGY_BANK),
        },
        "frozen_artifact_sha256": {
            str(blueprint_path.relative_to(ROOT)): sha256_file(blueprint_path),
            str(states_path.relative_to(ROOT)): sha256_file(states_path),
            str(candidates_path.relative_to(ROOT)): sha256_file(candidates_path),
        },
        "development_evidence_not_confirmation": pareto["headline"],
        "confirmation_external_or_human_outcomes_read": False,
        "post_confirmation_method_change_allowed": False,
    }
    _write_once_json(args.contract, contract)
    quality_report = {
        "protocol": PROTOCOL,
        "status": "PASS_V5_2_CONTENT_DISJOINT_CONFIRMATION_IDENTITY_FROZEN",
        "dataset_grain": "state_candidate_unit; paired units share one counterfactual group",
        "checks": checks,
        "component_counts": dict(sorted(component_counts.items())),
        "logic_family_counts": dict(sorted(logic_counts.items())),
        "history_scale_counts": dict(sorted(scale_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "private_enrichment_counts": dict(sorted(enrichment_counts.items())),
        "historical_exposure_audit": old_exposure,
        "candidate_materialization": deepcopy(candidate_report),
        "contract_path": str(args.contract.relative_to(ROOT)),
        "contract_sha256": sha256_file(args.contract),
        "api_calls": 0,
        "human_labels_read": 0,
        "external_outcomes_read": 0,
        "generated_responses": 0,
        "next_step": "MATERIALIZE_FROZEN_FEATURES_POLICY_DECISIONS_AND_LOCKED_ON_OFF_CALL_PLAN",
    }
    report_path = args.out_dir / "freeze_and_data_quality_report.json"
    _write_once_json(report_path, quality_report)
    print(
        json.dumps(
            {
                "status": quality_report["status"],
                "states": len(candidate_rows),
                "groups": len(group_members),
                "old_internal_splits_reusable": False,
                "api_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
