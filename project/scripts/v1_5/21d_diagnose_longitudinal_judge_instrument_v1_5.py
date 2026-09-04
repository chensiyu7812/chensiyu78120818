from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from metacom_pm.contracts import ActionOutcome
from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.pm_v2_audit import audit_pilot_label_value_feasibility
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_features import PMV2FeatureBuilder
from metacom_pm.pm_v2_contracts import ResponseDimensions
from metacom_pm.pm_v2_judging import (
    JudgeResult,
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_action_label,
    composite_spec_from_config,
    dimension_applicability_by_action,
    dimensions_inapplicable_to_every_action,
    labeling_settings_from_config,
    validate_action_applicable_risk_signal,
    validate_judge_table,
    validate_raw_judge_family_health,
    validate_raw_judge_family_subgroup_health,
)
from metacom_pm.pm_v2_model import (
    applicable_risk_fields,
    estimated_action_cost_profile,
)
from metacom_pm.config import load_config


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-longitudinal-judge-instrument-root-cause-v1"
MINIMUM_NONZERO_OBSERVATIONS = 10


def _family_utility(
    row: dict[str, Any],
    *,
    action_id: str,
    composite_spec: Any,
    normalized_cost: float,
    risk_weight: float,
    cost_weight: float,
) -> float:
    response_output = ResponseJudgeOutput.model_validate(row["response"])
    response = ResponseDimensions(
        **{
            name: float(getattr(response_output, name))
            for name in ResponseDimensions.model_fields
        }
    )
    risk = RiskJudgeOutput.model_validate(row["risk"])
    applicable = applicable_risk_fields(action_id)
    risk_score = max(float(getattr(risk, name)) / 3.0 for name in applicable)
    return float(
        composite_spec.score(response)
        - risk_weight * risk_score
        - cost_weight * normalized_cost
    )


def _cross_family_transfer(
    *,
    raw_rows: list[dict[str, Any]],
    states: list[Any],
    composite_spec: Any,
    risk_weight: float,
    cost_weight: float,
) -> dict[str, Any]:
    state_by_id = {state.state_id: state for state in states}
    families = sorted({str(row["judge_family"]) for row in raw_rows})
    actions = sorted({str(row["action_id"]) for row in raw_rows})
    raw_by_key = {
        (str(row["state_id"]), str(row["action_id"]), str(row["judge_family"])): row
        for row in raw_rows
    }
    builder = PMV2FeatureBuilder()
    utilities: dict[str, dict[str, dict[str, float]]] = {
        family: {} for family in families
    }
    for state_id, state in sorted(state_by_id.items()):
        cost_profile = estimated_action_cost_profile(builder, state)
        if not all((state_id, action, family) in raw_by_key for action in actions for family in families):
            continue
        for family in families:
            utilities[family][state_id] = {
                action: _family_utility(
                    raw_by_key[(state_id, action, family)],
                    action_id=action,
                    composite_spec=composite_spec,
                    normalized_cost=float(
                        cost_profile[action]["normalized_estimated_resource_cost"]
                    ),
                    risk_weight=risk_weight,
                    cost_weight=cost_weight,
                )
                for action in actions
            }
    state_ids = sorted(set.intersection(*(set(rows) for rows in utilities.values())))
    selected = {
        family: {
            state_id: min(
                actions,
                key=lambda action: (-utilities[family][state_id][action], action),
            )
            for state_id in state_ids
        }
        for family in families
    }
    pair_reports = []
    for selector in families:
        for evaluator in families:
            if selector == evaluator:
                continue
            fixed_means = {
                action: float(
                    np.mean(
                        [utilities[evaluator][state_id][action] for state_id in state_ids]
                    )
                )
                for action in actions
            }
            best_fixed = min(
                actions, key=lambda action: (-fixed_means[action], action)
            )
            transferred = float(
                np.mean(
                    [
                        utilities[evaluator][state_id][selected[selector][state_id]]
                        for state_id in state_ids
                    ]
                )
            )
            pair_reports.append(
                {
                    "selector_family": selector,
                    "evaluator_family": evaluator,
                    "selected_policy_mean": transferred,
                    "evaluator_best_fixed_action": best_fixed,
                    "evaluator_best_fixed_mean": fixed_means[best_fixed],
                    "selected_minus_evaluator_best_fixed": (
                        transferred - fixed_means[best_fixed]
                    ),
                }
            )
    oracle_agreement = None
    correlations = []
    if len(families) == 2 and state_ids:
        left, right = families
        oracle_agreement = float(
            np.mean(
                [
                    selected[left][state_id] == selected[right][state_id]
                    for state_id in state_ids
                ]
            )
        )
        for state_id in state_ids:
            a = np.asarray(
                [utilities[left][state_id][action] for action in actions], dtype=float
            )
            b = np.asarray(
                [utilities[right][state_id][action] for action in actions], dtype=float
            )
            if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12:
                correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "states": len(state_ids),
        "families": families,
        "actions": actions,
        "family_oracle_action_exact_agreement_rate": oracle_agreement,
        "within_state_family_utility_correlation_mean": (
            float(np.mean(correlations)) if correlations else None
        ),
        "within_state_family_utility_correlation_median": (
            float(np.median(correlations)) if correlations else None
        ),
        "cross_family_policy_transfer": pair_reports,
        "interpretation": (
            "diagnostic only; negative transfer versus the evaluator family's "
            "best fixed action is evidence against judge-specific adaptive labels"
        ),
    }


def diagnose(
    *,
    config_path: Path,
    states_path: Path,
    evaluator_contexts_path: Path,
    outcomes_path: Path,
    raw_results_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    composite_spec = composite_spec_from_config(config)
    labeling = labeling_settings_from_config(config)
    all_states = load_states(states_path)
    states = [
        state for state in all_states if str(state.split.value) in {"train", "calibration"}
    ]
    state_by_id = {state.state_id: state for state in states}
    evaluator_index = load_evaluator_context_index(
        evaluator_contexts_path,
        states=all_states,
        require_exact=True,
    )
    outcomes = [
        ActionOutcome.model_validate(row)
        for row in iter_jsonl(outcomes_path)
        if str(row["state_id"]) in state_by_id
    ]
    outcome_by_key = {(row.state_id, row.action_id): row for row in outcomes}
    raw_rows = [
        dict(row)
        for row in iter_jsonl(raw_results_path)
        if str(row["state_id"]) in state_by_id
    ]
    families = sorted({str(row["judge_family"]) for row in raw_rows})
    actions = sorted({str(row["action_id"]) for row in raw_rows})
    expected_raw_keys = {
        (state.state_id, action, family)
        for state in states
        for action in actions
        for family in families
    }
    raw_by_key = {
        (str(row["state_id"]), str(row["action_id"]), str(row["judge_family"])): row
        for row in raw_rows
    }
    if set(raw_by_key) != expected_raw_keys:
        raise RuntimeError(
            "root-cause diagnostic requires an exact two-family state-action matrix"
        )
    if set(outcome_by_key) != {
        (state.state_id, action) for state in states for action in actions
    }:
        raise RuntimeError(
            "root-cause diagnostic requires an exact state-action outcome matrix"
        )
    canonical_rows = [
        {
            "judge_family": row["judge_family"],
            "action_id": row["action_id"],
            "response": row["response"],
            "risk": row["risk"],
        }
        for row in raw_rows
    ]
    applicability = dimension_applicability_by_action(actions)
    globally_inapplicable = dimensions_inapplicable_to_every_action(actions)
    raw_common = {
        "expected_families": families,
        "duplicate_exact_match_rate": labeling["duplicate_exact_match_rate"],
        "maximum_absolute_dimension_correlation": labeling[
            "maximum_absolute_dimension_correlation"
        ],
        "composite_support_exact_match_rate": labeling[
            "composite_support_exact_match_rate"
        ],
        "maximum_absolute_composite_support_correlation": labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        "reject_constant_response_dimensions": labeling[
            "reject_constant_response_dimensions"
        ],
        "reject_constant_risk_dimensions": labeling[
            "reject_constant_risk_dimensions"
        ],
        "minimum_nonzero_observations": MINIMUM_NONZERO_OBSERVATIONS,
        "split_correlation_by_sign": True,
        "composite_spec": composite_spec,
        "raise_on_failure": False,
    }
    raw_global = validate_raw_judge_family_health(
        canonical_rows,
        inapplicable_risk_dimensions=globally_inapplicable,
        inapplicable_risk_dimensions_by_action=applicability,
        **raw_common,
    )
    raw_by_action = validate_raw_judge_family_subgroup_health(
        canonical_rows,
        subgroup_key="action_id",
        expected_subgroups=actions,
        **raw_common,
    )
    risk_signal = validate_action_applicable_risk_signal(
        canonical_rows,
        expected_actions=actions,
        expected_families=families,
        minimum_signal_rate=labeling[
            "minimum_action_applicable_risk_signal_rate"
        ],
        minimum_distinct_values=labeling[
            "minimum_action_applicable_risk_distinct_values"
        ],
        raise_on_failure=False,
    )
    labels = []
    for state in states:
        for action in actions:
            outcome = outcome_by_key[(state.state_id, action)]
            results = []
            for family in families:
                row = raw_by_key[(state.state_id, action, family)]
                results.append(
                    JudgeResult(
                        family=family,
                        model=str(row["judge_model"]),
                        response=ResponseJudgeOutput.model_validate(row["response"]),
                        risk=RiskJudgeOutput.model_validate(row["risk"]),
                        response_request_hash=str(row["response_request_hash"]),
                        risk_request_hash=str(row["risk_request_hash"]),
                    )
                )
            labels.append(
                build_action_label(
                    state=state,
                    action_id=action,
                    observed_input_tokens=outcome.cost.total_input_tokens,
                    retrieval_calls=outcome.cost.retrieval_calls,
                    results=results,
                    composite_spec=composite_spec,
                    minimum_families=labeling["minimum_families"],
                    reliable_mad_threshold=labeling["reliable_mad_threshold"],
                )
            )
    quality_gate = validate_judge_table(
        labels,
        minimum_families=labeling["minimum_families"],
        minimum_reliable_rate=labeling["minimum_reliable_rate"],
        reliable_mad_threshold=labeling["reliable_mad_threshold"],
        minimum_low_mad_coverage_per_dimension=labeling[
            "minimum_low_mad_coverage_per_dimension"
        ],
        minimum_low_mad_coverage_per_action_dimension=labeling[
            "minimum_low_mad_coverage_per_action_dimension"
        ],
        duplicate_exact_match_rate=labeling["duplicate_exact_match_rate"],
        maximum_absolute_dimension_correlation=labeling[
            "maximum_absolute_dimension_correlation"
        ],
        composite_support_exact_match_rate=labeling[
            "composite_support_exact_match_rate"
        ],
        maximum_absolute_composite_support_correlation=labeling[
            "maximum_absolute_composite_support_correlation"
        ],
        reject_constant_response_dimensions=labeling[
            "reject_constant_response_dimensions"
        ],
        reject_constant_risk_dimensions=labeling[
            "reject_constant_risk_dimensions"
        ],
        inapplicable_risk_dimensions=globally_inapplicable,
        inapplicable_risk_dimensions_by_action=applicability,
        minimum_nonzero_observations=MINIMUM_NONZERO_OBSERVATIONS,
        split_correlation_by_sign=True,
        composite_spec=composite_spec,
        raise_on_failure=False,
    )
    feasibility = audit_pilot_label_value_feasibility(
        states,
        labels,
        evaluator_contexts=evaluator_index,
        composite_spec=composite_spec,
        pilot_actions=actions,
        thresholds=dict(
            config["development_judging"]["compatibility_pilot"][
                "label_value_feasibility"
            ]
        ),
        risk_weight=float(config["selection"]["risk_weight"]),
        cost_weight=float(config["selection"]["cost_weight"]),
    )
    raw_status = (
        "PASS"
        if raw_global["status"] == raw_by_action["status"] == risk_signal["status"] == "PASS"
        else "FAIL"
    )
    supported = bool(
        raw_status == "PASS"
        and quality_gate["status"] == "PASS"
        and feasibility["status"] == "PASS"
    )
    return {
        "protocol": PROTOCOL,
        "status": (
            "SUPPORTED_FOR_LONGITUDINAL_PM_TRAINING"
            if supported
            else "LONGITUDINAL_SUPERVISION_NOT_SUPPORTED"
        ),
        "training_labels_created": False,
        "source_lineage": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "states": {"path": str(states_path), "sha256": sha256_file(states_path)},
            "evaluator_contexts": {
                "path": str(evaluator_contexts_path),
                "sha256": sha256_file(evaluator_contexts_path),
            },
            "outcomes": {
                "path": str(outcomes_path),
                "sha256": sha256_file(outcomes_path),
            },
            "raw_results": {
                "path": str(raw_results_path),
                "sha256": sha256_file(raw_results_path),
            },
        },
        "matrix": {
            "states": len(states),
            "actions": len(actions),
            "families": len(families),
            "raw_rows": len(raw_rows),
            "aggregated_state_action_rows": len(labels),
            "split_counts": dict(
                sorted(Counter(str(state.split.value) for state in states).items())
            ),
        },
        "raw_family_quality_gate": {
            "status": raw_status,
            "global": raw_global,
            "family_by_action": raw_by_action,
            "action_applicable_risk_signal": risk_signal,
            "composite_support_exact_match_semantics": (
                "failure occurs when rate is greater than or equal to the "
                "maximum threshold; low rates are healthy"
            ),
        },
        "aggregated_label_quality_gate": quality_gate,
        "preregistered_label_value_feasibility": feasibility,
        "cross_family_transfer": _cross_family_transfer(
            raw_rows=raw_rows,
            states=states,
            composite_spec=composite_spec,
            risk_weight=float(config["selection"]["risk_weight"]),
            cost_weight=float(config["selection"]["cost_weight"]),
        ),
        "decision_rule": {
            "requires_raw_family_gate_pass": True,
            "requires_aggregated_label_gate_pass": True,
            "requires_preregistered_label_value_feasibility_pass": True,
            "thresholds_were_not_changed_after_observing_results": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--evaluator-contexts", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        config_path=args.config,
        states_path=args.states,
        evaluator_contexts_path=args.evaluator_contexts,
        outcomes_path=args.outcomes,
        raw_results_path=args.raw_results,
    )
    write_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
