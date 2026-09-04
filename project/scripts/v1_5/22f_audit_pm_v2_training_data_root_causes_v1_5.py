#!/usr/bin/env python3
"""Build a zero-API, train-only PM-v1.5 root-cause evidence report."""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json
from metacom_pm.v1_5_training_data_root_cause import (
    ROOT_CAUSE_AUDIT_PROTOCOL,
    matching_target_treatment_rows,
    require_learnable_transfer_training_contract,
    summarize_memory_treatment_composition,
)


ROOT = Path(__file__).resolve().parents[2]


def _load_train_prefix(
    *,
    states_path: Path,
    contexts_path: Path,
    outcomes_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    states = list(iter_jsonl(states_path))
    train_states = [row for row in states if row.get("split") == "train"]
    if states[: len(train_states)] != train_states:
        raise RuntimeError("train states must be an exact prefix before sealed splits")
    train_ids = {str(row["state_id"]) for row in train_states}
    if len(train_ids) != len(train_states):
        raise RuntimeError("train state IDs are not unique")

    contexts: list[dict[str, Any]] = []
    with contexts_path.open("r", encoding="utf-8") as handle:
        for _ in range(len(train_states)):
            line = handle.readline()
            if not line:
                raise RuntimeError("evaluator contexts ended before train prefix")
            import json

            contexts.append(json.loads(line))
    contexts_by_state = {str(row["state_id"]): row for row in contexts}
    if set(contexts_by_state) != train_ids:
        raise RuntimeError("train evaluator-context prefix does not match train states")

    expected_actions = sum(len(row["allowed_actions"]) for row in train_states)
    action_rows: list[dict[str, Any]] = []
    with outcomes_path.open("r", encoding="utf-8") as handle:
        for _ in range(expected_actions):
            line = handle.readline()
            if not line:
                raise RuntimeError("action outcomes ended before train prefix")
            import json

            action_rows.append(json.loads(line))
    if any(str(row["state_id"]) not in train_ids for row in action_rows):
        raise RuntimeError("train action prefix crossed into a sealed split")
    counts = collections.Counter(str(row["state_id"]) for row in action_rows)
    expected_by_state = {
        str(row["state_id"]): len(row["allowed_actions"]) for row in train_states
    }
    if counts != expected_by_state:
        raise RuntimeError("train action prefix lacks exact state-action coverage")
    return train_states, contexts_by_state, action_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate"
        / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--contexts",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate"
        / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--action-outcomes",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_action_sweep_v8_19_2_continuation_v2_dry_run"
        / "action_outcomes.jsonl",
    )
    parser.add_argument(
        "--production-uptake",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_response_mechanism_pilot_v1_uptake_execution_candidate"
        / "uptake_report.json",
    )
    parser.add_argument(
        "--oracle-uptake",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_oracle_memory_upper_bound_pilot_v1_execution_candidate"
        / "oracle_memory_uptake_report.json",
    )
    parser.add_argument(
        "--multisource-uptake",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_multisource_decomposition_pilot_v1_execution_candidate"
        / "multisource_uptake_report.json",
    )
    parser.add_argument(
        "--oracle-quality",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_oracle_memory_quality_execution_candidate"
        / "summary.json",
    )
    parser.add_argument(
        "--factorized-signal",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_factorized_weak_signal_audit_v1"
        / "factorized_weak_signal_report.json",
    )
    parser.add_argument(
        "--candidate-viability",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_dual_domain_weak_fit_only_candidate_v4_audit"
        / "candidate_viability_report.json",
    )
    parser.add_argument(
        "--judge-qualification",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strict_judge_bakeoff_v1_semantic_aggregation_repro_a"
        / "qualification_report.json",
    )
    parser.add_argument(
        "--external-support",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_merged_ood_audit_v8_19_2_v3_production_shape.json",
    )
    parser.add_argument(
        "--redesign-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/learnable_transfer_training_data_v2.json",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_training_data_root_cause_audit_v1.json",
    )
    args = parser.parse_args()

    train_states, contexts, action_rows = _load_train_prefix(
        states_path=args.states,
        contexts_path=args.contexts,
        outcomes_path=args.action_outcomes,
    )
    target_rows = matching_target_treatment_rows(
        action_rows=action_rows,
        evaluator_contexts_by_state=contexts,
    )
    composition = summarize_memory_treatment_composition(
        action_rows=target_rows,
        evaluator_contexts_by_state=contexts,
    )
    production = read_json(args.production_uptake)
    oracle = read_json(args.oracle_uptake)
    multisource = read_json(args.multisource_uptake)
    oracle_quality = read_json(args.oracle_quality)
    factorized = read_json(args.factorized_signal)
    viability = read_json(args.candidate_viability)
    judge_qualification = read_json(args.judge_qualification)
    external = read_json(args.external_support)
    redesign_contract = require_learnable_transfer_training_contract(
        args.redesign_contract
    )

    target_regimes = composition["by_regime"]
    positive_regimes = ("profile_needed", "summary_needed", "event_needed")
    positive_rows = sum(target_regimes[name]["rows"] for name in positive_regimes)
    positive_mixed = sum(
        target_regimes[name]["rows_with_mixed_helpful_and_nonhelpful"]
        for name in positive_regimes
    )
    esconv_support = external["merged_synthetic_plus_esconv_auxiliary"][
        "esconv_test"
    ]
    evoemo_support = external["merged_synthetic_plus_esconv_auxiliary"][
        "evoemo_formal_turn_states"
    ]

    report = {
        "protocol": ROOT_CAUSE_AUDIT_PROTOCOL,
        "status": "TRAINING_DATA_REDESIGN_REQUIRED_BEFORE_REFIT",
        "api_calls_made": 0,
        "scope": {
            "development_train_only": True,
            "internal_outcomes_opened": False,
            "external_outcomes_opened": False,
            "external_observable_states_only": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "states": args.states,
                "contexts": args.contexts,
                "action_outcomes": args.action_outcomes,
                "production_uptake": args.production_uptake,
                "oracle_uptake": args.oracle_uptake,
                "multisource_uptake": args.multisource_uptake,
                "oracle_quality": args.oracle_quality,
                "factorized_signal": args.factorized_signal,
                "candidate_viability": args.candidate_viability,
                "judge_qualification": args.judge_qualification,
                "external_support": args.external_support,
                "redesign_contract": args.redesign_contract,
            }.items()
        },
        "training_grain": {
            "train_states": len(train_states),
            "train_action_rows": len(action_rows),
            "target_treatment_rows": len(target_rows),
        },
        "retrieval_treatment_composition": {
            **composition,
            "positive_single_source_target_rows": positive_rows,
            "positive_single_source_rows_mixed_with_nonhelpful": positive_mixed,
            "positive_single_source_mixed_rate": positive_mixed / positive_rows,
        },
        "mechanism_evidence": {
            "production_top1_helpful_positive_pairs": production["helpful"][
                "positive_bge_pairs"
            ],
            "production_top1_helpful_pairs": production["helpful"]["pairs"],
            "oracle_helpful_positive_pairs": oracle["helpful"]["positive_pairs"],
            "oracle_helpful_pairs": oracle["helpful"]["pairs"],
            "oracle_harmful_nonpositive_pairs": oracle["harmful"][
                "nonpositive_pairs"
            ],
            "oracle_harmful_pairs": oracle["harmful"]["pairs"],
            "multisource_clear_interference_states": multisource["summary"][
                "clear_multisource_interference_states"
            ],
            "multisource_states": multisource["summary"]["state_count"],
        },
        "measurement_evidence": {
            "automatic_gold_status": judge_qualification["status"],
            "oracle_quality_diagnostic": oracle_quality["aggregate"],
            "factorized_judge_effects": {
                component: values["consensus_counts"]
                for component, values in factorized["longitudinal_synthetic"][
                    "by_component"
                ].items()
            },
        },
        "model_evidence": {
            "previous_candidate_status": viability["status"],
            "calibration_actions": {
                domain: values["action_viability"]["metrics"][
                    "action_distribution"
                ]
                for domain, values in viability["domains"].items()
            },
        },
        "external_observable_support": {
            "esconv_test": {
                "states": esconv_support["n_states"],
                "severe_ood_rate": esconv_support["severe_rate"],
            },
            "evoemo_formal_v3": {
                "states": evoemo_support["n_states"],
                "formal_result": evoemo_support["formal_result"],
                "severe_ood_rate": evoemo_support["severe_rate"],
                "top_metadata_dimensions": evoemo_support[
                    "top_metadata_dimensions_by_mean_contribution"
                ][:10],
            },
        },
        "root_cause_attribution": {
            "development_surface_integrity": {
                "finding": "not_the_current_primary_failure_after_v8_19_2_repairs",
                "severity": "not_primary",
            },
            "generator_capacity": {
                "finding": "not_globally_broken_oracle_helpful_memory_is_absorbed",
                "severity": "not_primary",
            },
            "retrieval_and_evidence_composition": {
                "finding": "positive source treatments are confounded by nonhelpful items and multisource interference",
                "severity": "primary_confirmed",
            },
            "judge_and_rubric": {
                "finding": "useful_as_abstaining_weak_sources_but_not_supported_as_automatic_gold",
                "severity": "contributing_confirmed",
            },
            "label_target_construction": {
                "finding": "sixteen_way_noisy_absolute_argmax_creates_winners_curse_pseudo_oracles",
                "severity": "primary_confirmed",
            },
            "model_family": {
                "finding": "old_HGB_collapse_is_downstream_and_does_not_identify_capacity_until_labels_are_repaired",
                "severity": "not_yet_identifiable",
            },
            "external_transfer_support": {
                "finding": "ESConv support is present after auxiliary states but EvoEmo age/catalog metadata support is absent",
                "severity": "primary_confirmed",
            },
        },
        "reuse_decision": {
            "existing_action_outcomes": "retain_as_noisy_auxiliary_and_cost_lineage_not_primary_targets",
            "existing_judge_results": "retain_as_separate_labeling_functions_with_abstention",
            "existing_internal_outcomes": "remain_sealed",
            "new_clean_train_contrasts_required": True,
            "new_external_shape_train_augmentation_required": True,
        },
        "required_next_contract": {
            "protocol": redesign_contract["protocol"],
            "status": redesign_contract["status"],
            "contract_sha256": redesign_contract["contract_sha256"],
        },
        "claim_boundary": (
            "This report localizes training-data and measurement failures. It "
            "does not show that a redesigned PM will pass calibration or external "
            "evaluation."
        ),
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_path, report)
    print(args.out_path)


if __name__ == "__main__":
    main()
