#!/usr/bin/env python3
"""PM-v1.5 thin wrapper for `22_train_pm_v2.py`.

The original script hard-checks `pm_config.get("version") == "pm-v2.2"`.
PM-v1.5 uses its own honestly-versioned config (`configs/pm_v1_5.yaml`,
`version: pm-v1.5` -- not disguised as "pm-v2.2"), so this fork only widens
that one check. Everything else is identical to the original: no gates were
touched here (this script has none), no Evidence Filter or bank logic lives
in the training step itself.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

from metacom_pm.config import load_config
from metacom_pm.pm_v2_audit import (
    EXPECTED_REGIME_CHECKS,
    _regime_pass,
    audit_training_labels,
)
from metacom_pm.pm_v2_contracts import ActionLabel, CompositeSpec, PMV2Split
from metacom_pm.pm_v2_data import (
    EvaluatorContextIndex,
    audit_cross_split_near_duplicates,
    load_evaluator_context_index,
    load_states,
    validate_split_manifests,
)
from metacom_pm.pm_v2_model import (
    PMV2Model,
    RESPONSE_FIELDS,
    RISK_FIELDS,
    SelectionConfig,
    applicable_risk_fields,
    calibrate_uncertainty_multiplier,
    composite_weights_digest,
    cost_calibration_diagnostics,
    decision_fallback_kind,
    estimated_action_cost_profile,
    evaluate_cost_diagnostics,
    evaluate_policy,
    evaluate_prediction_coverage,
    tune_selection_config,
)
from metacom_pm.io import iter_jsonl, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]


def complete_reliable_action_matrix_subset(
    states,
    labels,
    *,
    split_name,
    gate_cfg,
    low_mad_threshold,
):
    """Gate the complete aggregated matrix without whole-row reliability deletion.

    All legal action medians must be present for every state. Per-dimension MAD is
    then audited globally and for every action/dimension cell; ``label_reliable``
    remains an all-row quality diagnostic and never controls retention.
    """

    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states) or not state_map:
        raise ValueError(f"{split_name} matrix subset requires unique non-empty states")
    label_keys = [(label.state_id, label.action_id) for label in labels]
    if len(label_keys) != len(set(label_keys)):
        raise ValueError(f"{split_name} matrix subset received duplicate labels")
    unknown = sorted({label.state_id for label in labels} - set(state_map))
    if unknown:
        raise ValueError(f"{split_name} matrix labels reference unknown states: {unknown[:10]}")
    by_state = defaultdict(list)
    for label in labels:
        by_state[label.state_id].append(label)

    if not np.isfinite(low_mad_threshold) or low_mad_threshold < 0.0:
        raise ValueError("low_mad_threshold must be finite and non-negative")
    complete_ids: set[str] = set()
    incomplete: list[dict[str, object]] = []
    expected_action_labels = 0
    for state_id, state in sorted(state_map.items()):
        expected = set(state.allowed_actions)
        state_labels = by_state.get(state_id, [])
        observed = {label.action_id for label in state_labels}
        expected_action_labels += len(expected)
        illegal = sorted(observed - expected)
        if illegal:
            raise ValueError(
                f"{split_name} state {state_id} has illegal action labels: {illegal}"
            )
        if observed == expected:
            complete_ids.add(state_id)
        else:
            incomplete.append(
                {
                    "state_id": state_id,
                    "missing_actions": sorted(expected - observed),
                }
            )
    # Missing a state/action is a failed sweep, not a license for complete-case
    # deletion. This keeps the estimand equal to the preregistered state matrix.
    retained_states = list(states) if not incomplete else []
    retained_labels = list(labels) if not incomplete else []
    original_users = {state.user_id for state in states}
    retained_users = {state.user_id for state in retained_states}
    original_families = {state.semantic_family for state in states}
    retained_families = {state.semantic_family for state in retained_states}

    def rate(numerator, denominator):
        return float(numerator / max(denominator, 1))

    reliable_legal_action_labels = sum(label.label_reliable for label in labels)
    dimension_names = [
        *[f"response.{name}" for name in RESPONSE_FIELDS],
        *[f"risk.{name}" for name in RISK_FIELDS],
    ]
    dimension_low_mad_coverage = {
        name: float(
            np.mean(
                [
                    float(label.dimension_mad[name]) <= low_mad_threshold
                    for label in labels
                ]
            )
        )
        if labels
        else 0.0
        for name in dimension_names
    }
    action_ids = sorted({action for state in states for action in state.allowed_actions})
    action_dimension_low_mad_coverage: dict[str, dict[str, float]] = {}
    for action_id in action_ids:
        action_labels = [label for label in labels if label.action_id == action_id]
        action_dimension_low_mad_coverage[action_id] = {
            name: float(
                np.mean(
                    [
                        float(label.dimension_mad[name]) <= low_mad_threshold
                        for label in action_labels
                    ]
                )
            )
            if action_labels
            else 0.0
            for name in dimension_names
        }
    minimum_dimension_coverage = min(dimension_low_mad_coverage.values(), default=0.0)
    minimum_action_dimension_coverage = min(
        (
            value
            for action_rows in action_dimension_low_mad_coverage.values()
            for value in action_rows.values()
        ),
        default=0.0,
    )
    metrics = {
        "original_states": len(states),
        "retained_states": len(retained_states),
        "state_retention_rate": rate(len(retained_states), len(states)),
        "original_users": len(original_users),
        "retained_users": len(retained_users),
        "user_retention_rate": rate(len(retained_users), len(original_users)),
        "original_semantic_families": len(original_families),
        "retained_semantic_families": len(retained_families),
        "semantic_family_retention_rate": rate(
            len(retained_families), len(original_families)
        ),
        "expected_action_labels": expected_action_labels,
        "observed_legal_action_labels": len(labels),
        "complete_action_matrix_rate": rate(
            len(complete_ids), len(states)
        ),
        "reliable_legal_action_labels": reliable_legal_action_labels,
        "reliable_action_label_coverage_rate": rate(
            reliable_legal_action_labels, expected_action_labels
        ),
        "low_mad_threshold": float(low_mad_threshold),
        "dimension_low_mad_coverage": dimension_low_mad_coverage,
        "minimum_dimension_low_mad_coverage": minimum_dimension_coverage,
        "action_dimension_low_mad_coverage": action_dimension_low_mad_coverage,
        "minimum_action_dimension_low_mad_coverage": (
            minimum_action_dimension_coverage
        ),
    }
    checks = {
        "complete_action_matrix": not incomplete
        and len(labels) == expected_action_labels,
        "retained_states": metrics["retained_states"]
        >= int(gate_cfg["minimum_retained_states"]),
        "state_retention_rate": metrics["state_retention_rate"]
        >= float(gate_cfg["minimum_state_retention_rate"]),
        "retained_users": metrics["retained_users"]
        >= int(gate_cfg["minimum_retained_users"]),
        "user_retention_rate": metrics["user_retention_rate"]
        >= float(gate_cfg["minimum_user_retention_rate"]),
        "retained_semantic_families": metrics["retained_semantic_families"]
        >= int(gate_cfg["minimum_retained_semantic_families"]),
        "semantic_family_retention_rate": metrics["semantic_family_retention_rate"]
        >= float(gate_cfg["minimum_semantic_family_retention_rate"]),
        "dimension_low_mad_coverage": minimum_dimension_coverage
        >= float(gate_cfg["minimum_low_mad_coverage_per_dimension"]),
        "action_dimension_low_mad_coverage": minimum_action_dimension_coverage
        >= float(gate_cfg["minimum_low_mad_coverage_per_action_dimension"]),
    }
    report = {
        "split": split_name,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "metrics": metrics,
        "thresholds": dict(gate_cfg),
        "checks": checks,
        "incomplete_state_count": len(incomplete),
        "incomplete_states": incomplete,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"{split_name} complete aggregated action-matrix gate failed: {report}"
        )
    expected_retained_labels = sum(
        len(state.allowed_actions) for state in retained_states
    )
    if len(retained_labels) != expected_retained_labels:
        raise RuntimeError(
            f"{split_name} retained action matrix is not exact: "
            f"expected={expected_retained_labels}, observed={len(retained_labels)}"
        )
    return retained_states, retained_labels, report


def fixed_action_metrics(model, states, labels, action_id):
    label_map = {(label.state_id, label.action_id): label for label in labels}
    config = model.selection_config
    spec = config.composite_spec
    rows = []
    for state in states:
        if action_id not in state.allowed_actions:
            return None
        label = label_map.get((state.state_id, action_id))
        if label is None:
            return None
        if label.user_id != state.user_id:
            raise ValueError(f"fixed-action label user mismatch for state {state.state_id}")
        quality = spec.score(label.response)
        risk = max(
            float(getattr(label.risk, name)) / 3.0
            for name in applicable_risk_fields(action_id)
        )
        cost_profile = estimated_action_cost_profile(model.feature_builder, state)[action_id]
        normalized_estimated_cost = cost_profile[
            "normalized_estimated_resource_cost"
        ]
        realized_utility = (
            quality
            - config.risk_weight * risk
            - config.cost_weight * normalized_estimated_cost
        )
        rows.append(
            {
                "state_id": state.state_id,
                "user_id": state.user_id,
                "quality": quality,
                "response_dimensions": {
                    name: float(getattr(label.response, name)) for name in RESPONSE_FIELDS
                },
                "risk": risk,
                "realized_utility": realized_utility,
                "estimated_resource_cost": cost_profile["estimated_resource_cost"],
                "normalized_estimated_resource_cost": normalized_estimated_cost,
                "observed_input_tokens": float(label.observed_input_tokens),
                "cost": float(label.observed_input_tokens),
            }
        )
    if not rows:
        return None
    mean_quality = float(np.mean([row["quality"] for row in rows]))
    mean_risk = float(np.mean([row["risk"] for row in rows]))
    mean_observed = float(np.mean([row["observed_input_tokens"] for row in rows]))
    mean_estimated = float(np.mean([row["estimated_resource_cost"] for row in rows]))
    mean_normalized_estimated = float(
        np.mean([row["normalized_estimated_resource_cost"] for row in rows])
    )
    utility = float(np.mean([row["realized_utility"] for row in rows]))
    return {
        "action_id": action_id,
        "n": len(rows),
        "mean_quality": mean_quality,
        "mean_response_dimensions": {
            name: float(np.mean([row["response_dimensions"][name] for row in rows]))
            for name in RESPONSE_FIELDS
        },
        "mean_risk": mean_risk,
        "mean_realized_utility": utility,
        "mean_estimated_resource_cost": mean_estimated,
        "mean_normalized_estimated_resource_cost": mean_normalized_estimated,
        "mean_observed_input_tokens": mean_observed,
        "estimated_vs_observed_cost": cost_calibration_diagnostics(
            [row["estimated_resource_cost"] for row in rows],
            [row["observed_input_tokens"] for row in rows],
        ),
        "mean_cost": mean_observed,
        "utility": float(utility),
        "rows": rows,
    }


def policy_regime_alignment(
    model,
    states,
    evaluator_contexts: EvaluatorContextIndex,
    *,
    confidence_level: float = 0.95,
):
    if not 0.5 < float(confidence_level) < 1.0:
        raise ValueError("regime alignment confidence_level must be in (0.5, 1.0)")
    evaluator_by_state = evaluator_contexts.require_states(states, exact=False)
    by_regime = defaultdict(list)
    actions = []
    learned_actions = []
    fallback_counts = Counter()
    for state in states:
        decision = model.choose(state)
        action = decision.chosen_action
        actions.append(action)
        fallback_kind = decision_fallback_kind(decision)
        if fallback_kind is None:
            learned_actions.append(action)
        else:
            fallback_counts[fallback_kind] += 1
        regime = str(evaluator_by_state[state.state_id]["regime"])
        quality_values = sorted(
            (prediction.quality_mean, action_id)
            for action_id, prediction in decision.predictions.items()
        )
        quality_gap = (
            float(quality_values[-1][0] - quality_values[-2][0])
            if len(quality_values) > 1
            else 0.0
        )
        # A conservative fallback is not evidence of learned regime alignment.
        by_regime[regime].append(
            (
                str(state.user_id),
                bool(
                    fallback_kind is None
                    and _regime_pass(
                        regime,
                        action,
                        quality_gap,
                        evaluator_by_state[state.state_id].get(
                            "needed_memory_sources"
                        )
                        or [],
                    )
                ),
            )
        )
    counts = Counter(actions)
    learned_counts = Counter(learned_actions)
    if learned_actions:
        probabilities = np.asarray(list(learned_counts.values()), dtype=float) / len(
            learned_actions
        )
        learned_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        maximum_learned_share = max(learned_counts.values()) / len(learned_actions)
    else:
        learned_entropy = 0.0
        maximum_learned_share = 1.0
    expected_regimes = tuple(sorted(EXPECTED_REGIME_CHECKS))
    regime_counts = {
        regime: len(by_regime.get(regime, [])) for regime in expected_regimes
    }
    duplicate_user_regimes = {
        regime: sorted(
            user_id
            for user_id, count in Counter(
                user_id for user_id, _ in by_regime.get(regime, [])
            ).items()
            if count != 1
        )
        for regime in expected_regimes
    }
    duplicate_user_regimes = {
        regime: users for regime, users in duplicate_user_regimes.items() if users
    }
    if duplicate_user_regimes:
        raise RuntimeError(
            "internal regime alignment requires one Bernoulli block per user and "
            f"regime; duplicates={duplicate_user_regimes}"
        )
    regime_pass_rate = {
        regime: (
            float(np.mean([passed for _, passed in by_regime[regime]]))
            if by_regime.get(regime)
            else 0.0
        )
        for regime in expected_regimes
    }
    critical_value = NormalDist().inv_cdf(0.5 + float(confidence_level) / 2.0)

    def wilson_lower_bound(successes: int, total: int) -> float:
        if total <= 0:
            return 0.0
        proportion = successes / total
        denominator = 1.0 + critical_value**2 / total
        center = proportion + critical_value**2 / (2.0 * total)
        margin = critical_value * np.sqrt(
            proportion * (1.0 - proportion) / total
            + critical_value**2 / (4.0 * total**2)
        )
        return float(max(0.0, (center - margin) / denominator))

    regime_wilson_lower_bound = {
        regime: wilson_lower_bound(
            sum(bool(passed) for _, passed in by_regime.get(regime, [])),
            len(by_regime.get(regime, [])),
        )
        for regime in expected_regimes
    }
    unexpected_regimes = sorted(set(by_regime) - set(expected_regimes))
    missing_regimes = [
        regime for regime in expected_regimes if regime_counts[regime] == 0
    ]
    return {
        "action_distribution": dict(counts),
        "learned_action_distribution": dict(learned_counts),
        "fallback_distribution": dict(fallback_counts),
        "learned_decisions": len(learned_actions),
        "fallback_decisions": len(actions) - len(learned_actions),
        # These legacy names now deliberately describe learned, nonfallback
        # decisions; fallback selections cannot satisfy diversity gates.
        "distinct_actions": len(learned_counts),
        "maximum_action_share": float(maximum_learned_share),
        "action_entropy_bits": learned_entropy,
        "regime_state_counts": regime_counts,
        "regime_user_block_counts": regime_counts,
        "regime_pass_rate": regime_pass_rate,
        "regime_wilson_confidence_level": float(confidence_level),
        "regime_wilson_lower_bound": regime_wilson_lower_bound,
        "missing_regimes": missing_regimes,
        "unexpected_regimes": unexpected_regimes,
        "minimum_regime_pass_rate": float(min(regime_pass_rate.values())),
        "minimum_regime_wilson_lower_bound": float(
            min(regime_wilson_lower_bound.values())
        ),
        "mean_regime_pass_rate": (
            float(
                np.mean(
                    [
                        passed
                        for values in by_regime.values()
                        for _, passed in values
                    ]
                )
            )
            if by_regime
            else 0.0
        ),
    }


def paired_user_cluster_bootstrap(
    pm_rows,
    fixed_rows,
    *,
    replicates,
    confidence_level,
    seed,
):
    """Paired state deltas with users as the bootstrap resampling unit."""

    replicates = int(replicates)
    confidence_level = float(confidence_level)
    if replicates < 100:
        raise ValueError("paired user bootstrap requires at least 100 replicates")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("paired bootstrap confidence_level must be in (0.5, 1.0)")

    def keyed(rows, name):
        result = {}
        for row in rows:
            state_id = str(row["state_id"])
            if state_id in result:
                raise ValueError(f"duplicate {name} bootstrap state_id: {state_id}")
            result[state_id] = row
        return result

    pm_map = keyed(pm_rows, "PM")
    fixed_map = keyed(fixed_rows, "fixed")
    if set(pm_map) != set(fixed_map) or not pm_map:
        raise ValueError("paired bootstrap requires identical non-empty state IDs")
    metric_names = ("emotional_support", "quality", "risk", "utility")
    by_user = defaultdict(list)
    for state_id in sorted(pm_map):
        pm_row = pm_map[state_id]
        fixed_row = fixed_map[state_id]
        if pm_row["user_id"] != fixed_row["user_id"]:
            raise ValueError(f"paired bootstrap user mismatch for state {state_id}")
        deltas = {
            "emotional_support": (
                pm_row["response_dimensions"]["emotional_support"]
                - fixed_row["response_dimensions"]["emotional_support"]
            ),
            "quality": pm_row["quality"] - fixed_row["quality"],
            "risk": pm_row["risk"] - fixed_row["risk"],
            "utility": pm_row["realized_utility"] - fixed_row["realized_utility"],
        }
        by_user[str(pm_row["user_id"])].append(
            np.asarray([deltas[name] for name in metric_names], dtype=float)
        )
    users = sorted(by_user)
    if len(users) < 3:
        raise ValueError("paired user bootstrap requires at least three users")
    user_sums = np.vstack(
        [np.sum(np.vstack(by_user[user]), axis=0) for user in users]
    )
    user_counts = np.asarray([len(by_user[user]) for user in users], dtype=float)
    point = np.sum(user_sums, axis=0) / np.sum(user_counts)
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(users), size=(replicates, len(users)))
    bootstrap_sums = np.sum(user_sums[sampled], axis=1)
    bootstrap_counts = np.sum(user_counts[sampled], axis=1)
    bootstrap_means = bootstrap_sums / bootstrap_counts[:, None]
    alpha = 1.0 - confidence_level
    lower = np.quantile(bootstrap_means, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap_means, 1.0 - alpha / 2.0, axis=0)
    return {
        "n_states": len(pm_map),
        "n_users": len(users),
        "cluster_key": "user_id",
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": int(seed),
        "delta_direction": "PM_minus_cost_matched_fixed",
        "metrics": {
            name: {
                "mean_delta": float(point[index]),
                "ci_lower": float(lower[index]),
                "ci_upper": float(upper[index]),
            }
            for index, name in enumerate(metric_names)
        },
    }


def build_internal_reportability_assessment(
    *,
    internal,
    internal_cost_matched,
    calibration_cost_matched,
    internal_alignment,
    internal_coverage,
    internal_cost_diagnostics,
    paired_bootstrap,
    gate_cfg,
):
    """Build the fail-closed internal gate from observed selected-action labels."""

    quality_delta = internal["mean_quality"] - internal_cost_matched["mean_quality"]
    response_dimension_deltas = {
        name: (
            internal["mean_response_dimensions"][name]
            - internal_cost_matched["mean_response_dimensions"][name]
        )
        for name in RESPONSE_FIELDS
    }
    risk_delta = internal["mean_risk"] - internal_cost_matched["mean_risk"]
    pm_observed_tokens = float(internal["mean_observed_input_tokens"])
    fixed_observed_tokens = float(
        internal_cost_matched["mean_observed_input_tokens"]
    )
    if pm_observed_tokens <= 0.0 or fixed_observed_tokens <= 0.0:
        raise RuntimeError(
            "internal cost matching requires positive observed input-token means"
        )
    baseline_to_pm_cost_ratio = fixed_observed_tokens / pm_observed_tokens
    cost_matched_relative_deviation = abs(baseline_to_pm_cost_ratio - 1.0)
    calibration_cost_matched_relative_deviation = float(
        calibration_cost_matched["calibration_observed_token_relative_deviation"]
    )
    internal_pm_utility = internal["mean_realized_utility"]
    internal_fixed_utility = internal_cost_matched["mean_realized_utility"]
    utility_delta = internal_pm_utility - internal_fixed_utility
    estimated_observed_cost_spearman = internal_cost_diagnostics[
        "all_legal_actions"
    ]["spearman_correlation"]
    bootstrap_metrics = paired_bootstrap["metrics"]
    expected_bootstrap_points = {
        "emotional_support": response_dimension_deltas["emotional_support"],
        "quality": quality_delta,
        "risk": risk_delta,
        "utility": utility_delta,
    }
    mismatched_points = {
        name: {
            "summary": float(expected),
            "bootstrap": float(bootstrap_metrics[name]["mean_delta"]),
        }
        for name, expected in expected_bootstrap_points.items()
        if abs(float(bootstrap_metrics[name]["mean_delta"]) - float(expected)) > 1e-9
    }
    if mismatched_points:
        raise RuntimeError(
            f"paired bootstrap point estimates do not match summaries: {mismatched_points}"
        )
    common_checks = {
        "calibration_fixed_frontier_has_eligible_challenger": bool(
            calibration_cost_matched.get("cost_frontier_eligible", False)
        ),
        "risk_vs_cost_matched": bootstrap_metrics["risk"]["ci_upper"]
        <= float(gate_cfg["maximum_risk_delta_vs_cost_matched"]),
        "observed_token_cost_match": cost_matched_relative_deviation
        <= float(gate_cfg["maximum_cost_matched_relative_deviation"]),
        "calibration_observed_token_cost_match": (
            calibration_cost_matched_relative_deviation
            <= float(gate_cfg["maximum_cost_matched_relative_deviation"])
        ),
        "estimated_observed_cost_spearman": (
            estimated_observed_cost_spearman is not None
            and float(estimated_observed_cost_spearman)
            >= float(gate_cfg["minimum_estimated_observed_cost_spearman"])
        ),
        "learned_distinct_actions": internal_alignment["distinct_actions"]
        >= int(gate_cfg["minimum_distinct_actions"]),
        "learned_maximum_action_share": internal_alignment["maximum_action_share"]
        <= float(gate_cfg["maximum_learned_action_share"]),
        "learned_action_entropy": internal_alignment["action_entropy_bits"]
        >= float(gate_cfg["minimum_learned_action_entropy_bits"]),
        "learned_m0_rate": internal["learned_m0_rate"]
        >= float(gate_cfg["minimum_m0_rate"]),
        "learned_r0_rate": internal["learned_r0_rate"]
        >= float(gate_cfg["minimum_r0_rate"]),
        "nonfallback_m0_r0_rate": internal["nonfallback_m0_r0_rate"]
        >= float(gate_cfg["minimum_nonfallback_m0_r0_rate"]),
        "severe_ood_fallback_rate": internal["severe_ood_fallback_rate"]
        <= float(gate_cfg["maximum_severe_ood_fallback_rate"]),
        "no_feasible_fallback_rate": internal["no_feasible_fallback_rate"]
        <= float(gate_cfg["maximum_no_feasible_fallback_rate"]),
        "quality_interval_coverage": internal_coverage["quality"][
            "coverage_lower_bound"
        ]
        >= float(gate_cfg["minimum_quality_interval_coverage"]),
        "response_interval_coverage": internal_coverage[
            "minimum_response_dimension_coverage_lower_bound"
        ]
        >= float(gate_cfg["minimum_response_interval_coverage"]),
        "risk_interval_coverage": internal_coverage[
            "minimum_risk_dimension_coverage_lower_bound"
        ]
        >= float(gate_cfg["minimum_risk_interval_coverage"]),
        "mean_regime_alignment": internal_alignment["mean_regime_pass_rate"]
        >= float(gate_cfg["minimum_mean_regime_pass_rate"]),
        "each_regime_user_block_wilson": internal_alignment[
            "minimum_regime_wilson_lower_bound"
        ]
        >= float(gate_cfg["minimum_each_regime_wilson_lower_bound"]),
        "regime_wilson_confidence_contract": float(
            internal_alignment["regime_wilson_confidence_level"]
        )
        == float(gate_cfg["regime_wilson_confidence_level"]),
    }
    deployment_tradeoff_checks = {
        "quality_vs_cost_matched": bootstrap_metrics["quality"]["ci_lower"]
        >= float(gate_cfg["minimum_quality_delta_vs_cost_matched"]),
        "emotional_support_vs_cost_matched": bootstrap_metrics[
            "emotional_support"
        ]["ci_lower"]
        >= float(gate_cfg["minimum_emotional_support_delta_vs_cost_matched"]),
        "utility_vs_cost_matched": bootstrap_metrics["utility"]["ci_lower"]
        >= float(gate_cfg["minimum_utility_delta_vs_cost_matched"]),
    }
    learned_advantage_checks = {
        "advantage_quality_vs_cost_matched": bootstrap_metrics["quality"][
            "ci_lower"
        ]
        > float(gate_cfg["minimum_advantage_quality_delta_vs_cost_matched"]),
        "advantage_emotional_support_vs_cost_matched": bootstrap_metrics[
            "emotional_support"
        ]["ci_lower"]
        > float(
            gate_cfg[
                "minimum_advantage_emotional_support_delta_vs_cost_matched"
            ]
        ),
        "advantage_utility_vs_cost_matched": bootstrap_metrics["utility"][
            "ci_lower"
        ]
        > float(gate_cfg["minimum_advantage_utility_delta_vs_cost_matched"]),
    }
    deployment_tradeoff_reportable = all(
        [*common_checks.values(), *deployment_tradeoff_checks.values()]
    )
    learned_routing_advantage_verified = all(
        [*common_checks.values(), *learned_advantage_checks.values()]
    )
    require_advantage = bool(
        gate_cfg["require_learned_routing_advantage_before_external"]
    )
    reportable = (
        learned_routing_advantage_verified
        if require_advantage
        else deployment_tradeoff_reportable
    )
    checks = {
        **common_checks,
        **deployment_tradeoff_checks,
        **(
            learned_advantage_checks
            if require_advantage
            else {}
        ),
    }
    return {
        "reportable": reportable,
        "deployment_tradeoff_reportable": deployment_tradeoff_reportable,
        "learned_routing_advantage_verified": (
            learned_routing_advantage_verified
        ),
        "require_learned_routing_advantage_before_external": require_advantage,
        "checks": checks,
        "common_checks": common_checks,
        "deployment_tradeoff_checks": deployment_tradeoff_checks,
        "learned_routing_advantage_checks": learned_advantage_checks,
        "paired_user_cluster_bootstrap": paired_bootstrap,
        "deltas": {
            "quality": float(quality_delta),
            "response_dimensions": {
                name: float(value) for name, value in response_dimension_deltas.items()
            },
            "risk": float(risk_delta),
            "baseline_to_pm_observed_token_ratio": float(
                baseline_to_pm_cost_ratio
            ),
            "observed_token_relative_deviation": float(
                cost_matched_relative_deviation
            ),
            "calibration_observed_token_relative_deviation": float(
                calibration_cost_matched_relative_deviation
            ),
            "utility": float(utility_delta),
            "pm_utility": float(internal_pm_utility),
            "fixed_utility": float(internal_fixed_utility),
            "estimated_observed_cost_spearman": (
                None
                if estimated_observed_cost_spearman is None
                else float(estimated_observed_cost_spearman)
            ),
        },
    }


def selection_from_file(config):
    quality = config["quality_composite"]
    composite = CompositeSpec(
        version=str(quality["version"]),
        weights={str(key): float(value) for key, value in quality["weights"].items()},
    )
    selection = dict(config["selection"])
    return SelectionConfig(**selection, composite_spec=composite)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl")
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "evaluator_contexts.jsonl",
    )
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs" / "pm_v1_5_judging" / "action_labels.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "pm_v1_5_model")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--allow-nonreportable", action="store_true")
    args = parser.parse_args()

    pm_config = load_config(args.pm_v2_config)
    if pm_config.get("version") != "pm-v1.5":
        raise ValueError("PM-v1.5 training requires a pm-v1.5 config")
    model_cfg = pm_config["model"]
    feature_cfg = pm_config["features"]
    labeling_cfg = pm_config["labeling"]
    grid_cfg = pm_config["calibration_grid"]
    audit_cfg = pm_config["data_label_gate"]
    gate_cfg = pm_config["internal_reportability_gate"]
    uncertainty_cfg = pm_config["uncertainty_calibration"]
    reliable_matrix_cfg = pm_config["reliable_action_matrix_gate"]
    ood_cfg = dict(pm_config["ood_calibration"])
    expected_ood_keys = {
        "protocol",
        "calibration_split_only",
        "semantic_false_positive_quantile",
        "metadata_false_positive_quantile",
        "maximum_joint_in_distribution_fallback_rate",
        "minimum_semantic_challenge_detection_rate",
        "minimum_metadata_challenge_detection_rate",
        "claim_boundary",
    }
    if set(ood_cfg) != expected_ood_keys:
        raise ValueError("ood_calibration keys do not match the frozen contract")
    if (
        ood_cfg["protocol"] != "pm-v2-ood-calibration-v1"
        or ood_cfg["calibration_split_only"] is not True
        or ood_cfg["claim_boundary"]
        != "plumbing_gate_not_real_world_ood_coverage"
    ):
        raise ValueError("invalid PM-v2 OOD calibration protocol")
    initial_selection = selection_from_file(pm_config)
    preregistered_z = [
        float(value) for value in uncertainty_cfg["z_candidates"]
    ]
    if (
        len(preregistered_z) != 1
        or abs(preregistered_z[0] - initial_selection.uncertainty_z) > 1e-12
    ):
        raise ValueError(
            "uncertainty_calibration must contain exactly the same single z as "
            "selection.uncertainty_z"
        )

    states = load_states(args.states)
    evaluator_contexts = load_evaluator_context_index(
        args.evaluator_contexts,
        states=states,
        require_exact=True,
    )
    labels = [ActionLabel.model_validate(row) for row in iter_jsonl(args.labels)]
    expected_weights_hash = composite_weights_digest(initial_selection.composite_spec)
    bad_label_hashes = sorted(
        {
            label.composite_weights_sha256
            for label in labels
            if label.composite_weights_sha256 != expected_weights_hash
        }
    )
    if bad_label_hashes:
        raise RuntimeError(
            "action labels do not match configured composite weights hash: "
            f"expected={expected_weights_hash}, observed={bad_label_hashes[:5]}"
        )
    states_by_split = {
        split: [state for state in states if state.split is split]
        for split in (PMV2Split.TRAIN, PMV2Split.CALIBRATION, PMV2Split.INTERNAL_TEST)
    }
    if any(not rows for rows in states_by_split.values()):
        raise RuntimeError("PM-v2 requires non-empty train, calibration, and internal-test splits")
    split_manifest = validate_split_manifests(states_by_split)
    near_duplicate_cfg = pm_config["splits"]["near_duplicate_audit"]
    if near_duplicate_cfg.get("method") != (
        "fixed_hash_word_1_2_and_char_3_5_cosine"
    ):
        raise ValueError("unsupported PM-v2 near-duplicate audit method")
    cross_split_near_duplicate_audit = audit_cross_split_near_duplicates(
        states_by_split,
        maximum_word_hash_cosine=float(
            near_duplicate_cfg["maximum_word_hash_cosine"]
        ),
        maximum_char_hash_cosine=float(
            near_duplicate_cfg["maximum_char_hash_cosine"]
        ),
        report_top_pairs=int(near_duplicate_cfg["report_top_pairs"]),
    )
    state_ids_by_split = {
        split: {state.state_id for state in rows}
        for split, rows in states_by_split.items()
    }
    labels_by_split = {
        split: [label for label in labels if label.state_id in state_ids]
        for split, state_ids in state_ids_by_split.items()
    }
    reliable_matrix_report = {}
    for split in (PMV2Split.TRAIN, PMV2Split.CALIBRATION, PMV2Split.INTERNAL_TEST):
        retained_states, retained_labels, subset_report = (
            complete_reliable_action_matrix_subset(
                states_by_split[split],
                labels_by_split[split],
                split_name=split.value,
                gate_cfg=reliable_matrix_cfg[split.value],
                low_mad_threshold=float(labeling_cfg["reliable_mad_threshold"]),
            )
        )
        states_by_split[split] = retained_states
        labels_by_split[split] = retained_labels
        reliable_matrix_report[split.value] = subset_report
    data_label_audit = audit_training_labels(
        states_by_split[PMV2Split.TRAIN] + states_by_split[PMV2Split.CALIBRATION],
        labels_by_split[PMV2Split.TRAIN] + labels_by_split[PMV2Split.CALIBRATION],
        evaluator_contexts=evaluator_contexts,
        composite_spec=initial_selection.composite_spec,
        risk_weight=initial_selection.risk_weight,
        cost_weight=initial_selection.cost_weight,
        maximum_single_action_share=float(audit_cfg["maximum_single_action_share"]),
        minimum_m0_share=float(audit_cfg["minimum_m0_share"]),
        minimum_r0_share=float(audit_cfg["minimum_r0_share"]),
        minimum_rs_share=float(audit_cfg["minimum_rs_share"]),
        minimum_distinct_actions=int(audit_cfg["minimum_distinct_actions"]),
        minimum_regime_pass_rate=float(audit_cfg["minimum_regime_pass_rate"]),
        minimum_reliable_rate=float(audit_cfg["minimum_reliable_rate"]),
    )
    model = PMV2Model.train(
        states_by_split[PMV2Split.TRAIN],
        labels_by_split[PMV2Split.TRAIN],
        selection_config=initial_selection,
        n_models=int(model_cfg["bootstrap_models"]),
        seed=args.seed,
        dimension_mad_scale=float(labeling_cfg["reliable_mad_threshold"]),
        bootstrap_group_key=str(model_cfg.get("group_bootstrap_key", "user_id")),
        use_precomputed_embeddings=bool(feature_cfg["optional_precomputed_semantic_embedding"]),
        word_features=int(feature_cfg["word_hash_features"]),
        char_features=int(feature_cfg["char_hash_features"]),
    )
    ood_calibration = model.feature_builder.calibrate_ood(
        states_by_split[PMV2Split.CALIBRATION],
        semantic_false_positive_quantile=float(
            ood_cfg["semantic_false_positive_quantile"]
        ),
        metadata_false_positive_quantile=float(
            ood_cfg["metadata_false_positive_quantile"]
        ),
        maximum_joint_in_distribution_fallback_rate=float(
            ood_cfg["maximum_joint_in_distribution_fallback_rate"]
        ),
        minimum_semantic_challenge_detection_rate=float(
            ood_cfg["minimum_semantic_challenge_detection_rate"]
        ),
        minimum_metadata_challenge_detection_rate=float(
            ood_cfg["minimum_metadata_challenge_detection_rate"]
        ),
    )
    uncertainty_calibration = calibrate_uncertainty_multiplier(
        model,
        states_by_split[PMV2Split.CALIBRATION],
        labels_by_split[PMV2Split.CALIBRATION],
        z_candidates=preregistered_z,
        target_coverage=float(uncertainty_cfg["target_coverage"]),
        minimum_quality_coverage_lower_bound=float(
            uncertainty_cfg["minimum_quality_coverage_lower_bound"]
        ),
        minimum_response_coverage_lower_bound=float(
            uncertainty_cfg["minimum_response_coverage_lower_bound"]
        ),
        minimum_risk_coverage_lower_bound=float(
            uncertainty_cfg["minimum_risk_coverage_lower_bound"]
        ),
        coverage_confidence_level=float(uncertainty_cfg["confidence_level"]),
    )
    tuning = tune_selection_config(
        model,
        states_by_split[PMV2Split.CALIBRATION],
        labels_by_split[PMV2Split.CALIBRATION],
        cost_weights=[float(value) for value in grid_cfg["cost_weights"]],
        risk_weights=[float(value) for value in grid_cfg["risk_weights"]],
        resource_gains=[float(value) for value in grid_cfg["resource_gains"]],
        strategy_gains=[float(value) for value in grid_cfg["strategy_gains"]],
        max_risks=[float(value) for value in grid_cfg["max_risks"]],
        minimum_quality=float(grid_cfg["minimum_quality"]),
        objective_risk_weight=float(grid_cfg["objective_risk_weight"]),
        objective_cost_weight=float(grid_cfg["objective_cost_weight"]),
        objective_version=str(grid_cfg["objective_version"]),
    )
    calibration_pm = evaluate_policy(
        model,
        states_by_split[PMV2Split.CALIBRATION],
        labels_by_split[PMV2Split.CALIBRATION],
    )
    calibration_coverage = evaluate_prediction_coverage(
        model,
        states_by_split[PMV2Split.CALIBRATION],
        labels_by_split[PMV2Split.CALIBRATION],
        confidence_level=float(uncertainty_cfg["confidence_level"]),
    )
    calibration_cost_diagnostics = evaluate_cost_diagnostics(
        model,
        states_by_split[PMV2Split.CALIBRATION],
        labels_by_split[PMV2Split.CALIBRATION],
    )
    common_actions = sorted(
        set.intersection(
            *(set(state.allowed_actions) for state in states_by_split[PMV2Split.CALIBRATION])
        )
    )
    if not common_actions:
        raise RuntimeError("calibration split has no common fixed action")
    calibration_fixed = [
        fixed_action_metrics(
            model,
            states_by_split[PMV2Split.CALIBRATION],
            labels_by_split[PMV2Split.CALIBRATION],
            action_id,
        )
        for action_id in common_actions
    ]
    calibration_fixed = [row for row in calibration_fixed if row is not None]
    calibration_pm_tokens = float(calibration_pm["mean_observed_input_tokens"])
    maximum_frontier_deviation = float(
        gate_cfg["maximum_cost_matched_relative_deviation"]
    )
    for row in calibration_fixed:
        row["calibration_pm_mean_observed_input_tokens"] = calibration_pm_tokens
        row["calibration_observed_token_relative_deviation"] = abs(
            float(row["mean_observed_input_tokens"])
            / max(calibration_pm_tokens, 1.0)
            - 1.0
        )
        row["cost_frontier_eligible"] = (
            row["calibration_observed_token_relative_deviation"]
            <= maximum_frontier_deviation
        )
    eligible_fixed_frontier = [
        row for row in calibration_fixed if row["cost_frontier_eligible"]
    ]
    frontier_pool = eligible_fixed_frontier or calibration_fixed
    cost_matched = max(
        frontier_pool,
        key=lambda row: (
            row["utility"],
            row["mean_quality"],
            row["mean_response_dimensions"]["emotional_support"],
            -row["calibration_observed_token_relative_deviation"],
            -row["mean_observed_input_tokens"],
            row["action_id"],
        ),
    )
    cost_matched["cost_matching_basis"] = (
        "strongest_calibration_fixed_within_observed_token_band"
    )
    cost_matched["cost_frontier_status"] = (
        "ELIGIBLE_CHALLENGER_SELECTED"
        if eligible_fixed_frontier
        else "NO_ELIGIBLE_CHALLENGER_STRONGEST_FIXED_DIAGNOSTIC_ONLY"
    )
    cost_matched["maximum_cost_matched_relative_deviation"] = (
        maximum_frontier_deviation
    )
    cost_matched["eligible_fixed_frontier_actions"] = sorted(
        row["action_id"] for row in eligible_fixed_frontier
    )
    cost_matched["eligible_fixed_frontier_size"] = len(eligible_fixed_frontier)
    cost_matched["me_r0_in_eligible_frontier"] = any(
        row["action_id"] == "ME+R0" for row in eligible_fixed_frontier
    )
    best_fixed = max(
        calibration_fixed,
        key=lambda row: (
            row["utility"],
            row["mean_quality"],
            -row["mean_normalized_estimated_resource_cost"],
        ),
    )

    internal = evaluate_policy(
        model,
        states_by_split[PMV2Split.INTERNAL_TEST],
        labels_by_split[PMV2Split.INTERNAL_TEST],
    )
    internal_coverage = evaluate_prediction_coverage(
        model,
        states_by_split[PMV2Split.INTERNAL_TEST],
        labels_by_split[PMV2Split.INTERNAL_TEST],
        confidence_level=float(uncertainty_cfg["confidence_level"]),
    )
    internal_cost_diagnostics = evaluate_cost_diagnostics(
        model,
        states_by_split[PMV2Split.INTERNAL_TEST],
        labels_by_split[PMV2Split.INTERNAL_TEST],
    )
    internal_cost_matched = fixed_action_metrics(
        model,
        states_by_split[PMV2Split.INTERNAL_TEST],
        labels_by_split[PMV2Split.INTERNAL_TEST],
        cost_matched["action_id"],
    )
    internal_best_fixed = fixed_action_metrics(
        model,
        states_by_split[PMV2Split.INTERNAL_TEST],
        labels_by_split[PMV2Split.INTERNAL_TEST],
        best_fixed["action_id"],
    )
    if internal_cost_matched is None or internal_best_fixed is None:
        raise RuntimeError("calibration-selected fixed action is not legal on internal test")
    internal_alignment = policy_regime_alignment(
        model,
        states_by_split[PMV2Split.INTERNAL_TEST],
        evaluator_contexts,
        confidence_level=float(gate_cfg["regime_wilson_confidence_level"]),
    )

    internal_paired_bootstrap = paired_user_cluster_bootstrap(
        internal["rows"],
        internal_cost_matched["rows"],
        replicates=int(gate_cfg["paired_bootstrap_replicates"]),
        confidence_level=float(gate_cfg["paired_bootstrap_confidence_level"]),
        seed=int(gate_cfg["paired_bootstrap_seed"]),
    )
    assessment = build_internal_reportability_assessment(
        internal=internal,
        internal_cost_matched=internal_cost_matched,
        calibration_cost_matched=cost_matched,
        internal_alignment=internal_alignment,
        internal_coverage=internal_coverage,
        internal_cost_diagnostics=internal_cost_diagnostics,
        paired_bootstrap=internal_paired_bootstrap,
        gate_cfg=gate_cfg,
    )
    reportability_checks = assessment["checks"]
    reportable = assessment["reportable"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "pm_v1_5.joblib"
    model.save(checkpoint)
    report = {
        "status": "COMPLETE" if reportable else "NONREPORTABLE",
        "format_version": model.format_version,
        "pm_v2_config": str(args.pm_v2_config),
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "states": str(args.states),
        "states_sha256": sha256_file(args.states),
        "labels": str(args.labels),
        "labels_sha256": sha256_file(args.labels),
        "evaluator_contexts": str(args.evaluator_contexts),
        "evaluator_contexts_sha256": evaluator_contexts.source_sha256,
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training": model.training_report,
        "split_manifest": split_manifest.model_dump(mode="json"),
        "cross_split_near_duplicate_audit": cross_split_near_duplicate_audit,
        "composite_weights_sha256": expected_weights_hash,
        "reliable_action_matrix_subsets": reliable_matrix_report,
        "data_label_audit": {key: value for key, value in data_label_audit.items() if key != "rows"},
        "uncertainty_calibration": uncertainty_calibration,
        "ood_calibration": ood_calibration,
        "calibration": tuning,
        "calibration_pm": {key: value for key, value in calibration_pm.items() if key != "rows"},
        "calibration_prediction_coverage": calibration_coverage,
        "calibration_cost_diagnostics": calibration_cost_diagnostics,
        "calibration_fixed_frontier": [
            {key: value for key, value in row.items() if key != "rows"}
            for row in sorted(calibration_fixed, key=lambda row: row["action_id"])
        ],
        "calibration_cost_matched_fixed": {
            key: value for key, value in cost_matched.items() if key != "rows"
        },
        "calibration_best_fixed": {
            key: value for key, value in best_fixed.items() if key != "rows"
        },
        "internal_test": {key: value for key, value in internal.items() if key != "rows"},
        "internal_prediction_coverage": internal_coverage,
        "internal_cost_diagnostics": internal_cost_diagnostics,
        "internal_cost_matched_fixed": {
            key: value for key, value in internal_cost_matched.items() if key != "rows"
        },
        "internal_best_fixed": {
            key: value for key, value in internal_best_fixed.items() if key != "rows"
        },
        "internal_policy_regime_alignment": internal_alignment,
        "internal_paired_user_cluster_bootstrap": internal_paired_bootstrap,
        "internal_deltas_vs_cost_matched": assessment["deltas"],
        "deployment_tradeoff_reportable": assessment[
            "deployment_tradeoff_reportable"
        ],
        "learned_routing_advantage_verified": assessment[
            "learned_routing_advantage_verified"
        ],
        "require_learned_routing_advantage_before_external": assessment[
            "require_learned_routing_advantage_before_external"
        ],
        "deployment_tradeoff_checks": assessment[
            "deployment_tradeoff_checks"
        ],
        "learned_routing_advantage_checks": assessment[
            "learned_routing_advantage_checks"
        ],
        "reportability_thresholds": gate_cfg,
        "reportability_checks": reportability_checks,
        "selection_config": model.selection_config.model_dump(mode="json"),
        "selection_config_hash": model.selection_config.digest(),
    }
    write_json(args.out_dir / "training_report.json", report)
    print(report)
    if not reportable and not args.allow_nonreportable:
        raise RuntimeError(
            "PM-v2 failed the frozen internal reportability gate. "
            "Do not run external generation. See training_report.json."
        )


if __name__ == "__main__":
    main()
