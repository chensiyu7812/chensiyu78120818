from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from .contracts import MemorySource, StrategyMode, parse_action_id
from .pm_v2_contracts import ActionLabel, CompositeSpec, PMV2State, ResourceNeedRegime
from .pm_v2_data import EvaluatorContextIndex
from .pm_v2_features import PMV2FeatureBuilder
from .pm_v2_model import (
    applicable_risk_fields,
    composite_weights_digest,
    estimated_action_cost_profile,
)


EXPECTED_REGIME_CHECKS = {
    ResourceNeedRegime.CONTEXT_ONLY.value: "m0",
    ResourceNeedRegime.PROFILE_NEEDED.value: "mp",
    ResourceNeedRegime.SUMMARY_NEEDED.value: "ms",
    ResourceNeedRegime.EVENT_NEEDED.value: "me",
    ResourceNeedRegime.MULTI_SOURCE_NEEDED.value: "multi",
    ResourceNeedRegime.MEMORY_HARMFUL.value: "m0",
    ResourceNeedRegime.STRATEGY_HELPFUL.value: "rs",
    ResourceNeedRegime.STRATEGY_HARMFUL.value: "r0",
    ResourceNeedRegime.AMBIGUOUS.value: "ambiguous",
}


def _quality(label: ActionLabel, spec: CompositeSpec) -> float:
    return spec.score(label.response)


def _risk(label: ActionLabel) -> float:
    return max(
        float(getattr(label.risk, name)) / 3.0
        for name in applicable_risk_fields(label.action_id)
    )


def _regime_pass(
    regime: str,
    action_id: str,
    quality_gap: float,
    needed_memory_sources: Sequence[str] | None = None,
) -> bool:
    sources, strategy = parse_action_id(action_id)
    check = EXPECTED_REGIME_CHECKS.get(regime)
    needed = frozenset(
        source if isinstance(source, MemorySource) else MemorySource(str(source))
        for source in (needed_memory_sources or [])
    )
    if check == "m0":
        return not sources and not needed
    if check == "mp":
        return sources == needed == frozenset({MemorySource.MP})
    if check == "ms":
        return sources == needed == frozenset({MemorySource.MS})
    if check == "me":
        return sources == needed == frozenset({MemorySource.ME})
    if check == "multi":
        if needed:
            return len(needed) >= 2 and sources == needed
        # Legacy/test-only fallback: never let an always-full MPMSME policy
        # masquerade as calibrated when the exact complementary pair is absent.
        return len(sources) == 2
    if check == "rs":
        return strategy is StrategyMode.RS and sources == needed
    if check == "r0":
        return strategy is StrategyMode.R0 and sources == needed
    if check == "ambiguous":
        source_ok = sources == needed if needed else len(sources) <= 1
        return source_ok and quality_gap <= 0.05
    return False


def audit_training_labels(
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    *,
    evaluator_contexts: EvaluatorContextIndex,
    composite_spec: CompositeSpec | None = None,
    risk_weight: float = 0.25,
    cost_weight: float = 0.10,
    maximum_single_action_share: float = 0.35,
    minimum_m0_share: float = 0.10,
    minimum_r0_share: float = 0.15,
    minimum_rs_share: float = 0.15,
    minimum_distinct_actions: int = 6,
    minimum_regime_pass_rate: float = 0.60,
    minimum_reliable_rate: float = 0.80,
) -> dict[str, Any]:
    """Fail closed when generated labels do not contain learnable policy diversity."""

    if risk_weight < 0.0 or cost_weight < 0.0:
        raise ValueError("audit risk_weight and cost_weight must be non-negative")

    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states):
        raise RuntimeError("PM-v2 audit received duplicate state_id values")
    if not state_map:
        raise RuntimeError("PM-v2 audit requires at least one state")
    evaluator_by_state = evaluator_contexts.require_states(states, exact=False)
    label_keys = [(label.state_id, label.action_id) for label in labels]
    if len(label_keys) != len(set(label_keys)):
        raise RuntimeError("PM-v2 audit received duplicate state-action labels")
    unknown_states = sorted({label.state_id for label in labels} - set(state_map))
    if unknown_states:
        raise RuntimeError(
            f"PM-v2 audit labels reference unknown states: {unknown_states[:5]}"
        )
    by_state: dict[str, list[ActionLabel]] = defaultdict(list)
    for label in labels:
        if label.state_id in state_map:
            by_state[label.state_id].append(label)
    missing_states = sorted(set(state_map) - set(by_state))
    incomplete: list[dict[str, Any]] = []
    for state_id, state in state_map.items():
        observed = {label.action_id for label in by_state.get(state_id, [])}
        expected = set(state.allowed_actions)
        if observed != expected:
            incomplete.append(
                {
                    "state_id": state_id,
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                }
            )
    if missing_states or incomplete:
        raise RuntimeError(
            "PM-v2 label table is incomplete: "
            f"missing_states={missing_states[:5]}, incomplete={incomplete[:5]}"
        )
    spec = composite_spec or CompositeSpec()
    expected_weights_hash = composite_weights_digest(spec)
    bad_weight_hashes = sorted(
        {
            label.composite_weights_sha256
            for label in labels
            if label.composite_weights_sha256 != expected_weights_hash
        }
    )
    if bad_weight_hashes:
        raise RuntimeError(
            "PM-v2 audit labels do not match composite weights hash: "
            f"expected={expected_weights_hash}, observed={bad_weight_hashes[:5]}"
        )
    reliable_rate = float(np.mean([label.label_reliable for label in labels]))
    cost_builder = PMV2FeatureBuilder()
    rows: list[dict[str, Any]] = []
    best_actions: list[str] = []
    regime_stats: dict[str, list[bool]] = defaultdict(list)
    for state_id, state_labels in sorted(by_state.items()):
        state = state_map[state_id]
        cost_profile = estimated_action_cost_profile(cost_builder, state)
        scored = []
        for label in state_labels:
            quality = _quality(label, spec)
            risk = _risk(label)
            normalized_estimated_cost = cost_profile[label.action_id][
                "normalized_estimated_resource_cost"
            ]
            utility = (
                quality
                - risk_weight * risk
                - cost_weight * normalized_estimated_cost
            )
            scored.append(
                (
                    utility,
                    quality,
                    -risk,
                    -cost_profile[label.action_id]["estimated_resource_cost"],
                    label.action_id,
                )
            )
        ranked = sorted(scored, reverse=True)
        best = ranked[0]
        best_action = best[-1]
        best_actions.append(best_action)
        quality_ranking = sorted((row[1], row[-1]) for row in ranked)[::-1]
        quality_gap = (
            float(quality_ranking[0][0] - quality_ranking[1][0])
            if len(quality_ranking) > 1
            else 0.0
        )
        regime = str(evaluator_by_state[state_id]["regime"])
        passed = _regime_pass(
            regime,
            best_action,
            quality_gap,
            evaluator_by_state[state_id].get("needed_memory_sources") or [],
        )
        regime_stats[regime].append(passed)
        rows.append(
            {
                "state_id": state_id,
                "regime": regime,
                "best_action": best_action,
                "best_utility": float(best[0]),
                "top_quality_gap": quality_gap,
                "regime_expectation_passed": passed,
            }
        )
    counts = Counter(best_actions)
    n = max(len(best_actions), 1)
    m0_share = float(np.mean([not bool(parse_action_id(action)[0]) for action in best_actions]))
    r0_share = float(
        np.mean([parse_action_id(action)[1] is StrategyMode.R0 for action in best_actions])
    )
    rs_share = 1.0 - r0_share
    maximum_share = max(counts.values(), default=0) / n
    regime_summary = {
        regime: {
            "n": len(values),
            "pass_rate": float(np.mean(values)) if values else 0.0,
        }
        for regime, values in sorted(regime_stats.items())
    }
    failures: list[str] = []
    # ``label_reliable`` is the conjunction of all response/risk MAD checks.
    # Its probability shrinks with the number of heads, so it is diagnostic only;
    # script 22 applies the statistically meaningful per-dimension gates.
    if len(counts) < minimum_distinct_actions:
        failures.append("distinct_oracle_actions")
    if maximum_share > maximum_single_action_share:
        failures.append("single_action_dominance")
    if m0_share < minimum_m0_share:
        failures.append("m0_oracle_coverage")
    if r0_share < minimum_r0_share:
        failures.append("r0_oracle_coverage")
    if rs_share < minimum_rs_share:
        failures.append("rs_oracle_coverage")
    required_regimes = set(EXPECTED_REGIME_CHECKS)
    missing_regimes = sorted(required_regimes - set(regime_summary))
    if missing_regimes:
        failures.append("missing_regimes")
    low_regimes = [
        regime
        for regime, data in regime_summary.items()
        if regime in required_regimes and data["pass_rate"] < minimum_regime_pass_rate
    ]
    if low_regimes:
        failures.append("regime_alignment")
    report = {
        "status": "FAIL" if failures else "PASS",
        "n_states": len(by_state),
        "n_labels": len(labels),
        "evaluator_contexts_sha256": evaluator_contexts.source_sha256,
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
        "reliable_label_rate": reliable_rate,
        "reliable_label_rate_below_diagnostic_reference": (
            reliable_rate < minimum_reliable_rate
        ),
        "oracle_action_distribution": dict(counts),
        "oracle_distinct_actions": len(counts),
        "maximum_single_action_share": maximum_share,
        "m0_oracle_share": m0_share,
        "r0_oracle_share": r0_share,
        "rs_oracle_share": rs_share,
        "regime_summary": regime_summary,
        "missing_regimes": missing_regimes,
        "low_pass_regimes": low_regimes,
        "thresholds": {
            "maximum_single_action_share": maximum_single_action_share,
            "minimum_m0_share": minimum_m0_share,
            "minimum_r0_share": minimum_r0_share,
            "minimum_rs_share": minimum_rs_share,
            "minimum_distinct_actions": minimum_distinct_actions,
            "minimum_regime_pass_rate": minimum_regime_pass_rate,
            "minimum_reliable_rate": minimum_reliable_rate,
            "risk_weight": risk_weight,
            "cost_weight": cost_weight,
            "composite_spec": spec.model_dump(mode="json"),
            "composite_weights_sha256": expected_weights_hash,
            "cost_basis": "within_state_normalized_estimated_resource_cost",
        },
        "failures": failures,
        "rows": rows,
    }
    if failures:
        raise RuntimeError("PM-v2 data/label diversity gate failed: " + str(report))
    return report


def audit_pilot_label_value_feasibility(
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    *,
    evaluator_contexts: EvaluatorContextIndex,
    composite_spec: CompositeSpec,
    pilot_actions: Sequence[str],
    thresholds: Mapping[str, float | int],
    risk_weight: float,
    cost_weight: float,
) -> dict[str, Any]:
    """No-API check that a compatibility pilot contains routing signal."""

    expected_thresholds = {
        "minimum_distinct_oracle_actions",
        "maximum_oracle_action_share",
        "minimum_m0_oracle_share",
        "minimum_r0_oracle_share",
        "minimum_rs_oracle_share",
        "minimum_distinct_oracle_actions_per_regime",
        "minimum_mean_within_state_quality_range",
        "minimum_mean_within_state_quality_variance",
        "minimum_strategy_directional_separation",
        "minimum_memory_directional_separation",
        "minimum_oracle_utility_headroom",
        "minimum_oracle_quality_headroom",
        "minimum_oracle_emotional_support_headroom",
    }
    if set(thresholds) != expected_thresholds:
        raise RuntimeError("pilot label-value feasibility thresholds changed")
    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states) or not state_map:
        raise RuntimeError("pilot feasibility requires unique non-empty states")
    evaluator_by_state = evaluator_contexts.require_states(states, exact=False)
    expected_actions = [str(value) for value in pilot_actions]
    if len(expected_actions) != len(set(expected_actions)):
        raise RuntimeError("pilot feasibility action list contains duplicates")
    by_state: dict[str, dict[str, ActionLabel]] = defaultdict(dict)
    for label in labels:
        if label.state_id not in state_map:
            raise RuntimeError("pilot feasibility label references an unknown state")
        if label.action_id in by_state[label.state_id]:
            raise RuntimeError("pilot feasibility label matrix contains duplicates")
        by_state[label.state_id][label.action_id] = label
    expected_action_set = set(expected_actions)
    incomplete = {
        state_id: sorted(expected_action_set ^ set(by_state.get(state_id, {})))
        for state_id in state_map
        if set(by_state.get(state_id, {})) != expected_action_set
    }
    if incomplete:
        raise RuntimeError(
            "pilot feasibility requires the exact state-action matrix: "
            + str(incomplete)
        )

    builder = PMV2FeatureBuilder()
    rows: list[dict[str, Any]] = []
    oracle_actions: list[str] = []
    oracle_by_regime: dict[str, list[str]] = defaultdict(list)
    strategy_effect_by_regime: dict[str, list[float]] = defaultdict(list)
    memory_effect_by_regime: dict[str, list[float]] = defaultdict(list)
    quality_ranges: list[float] = []
    quality_variances: list[float] = []
    utility_matrix: dict[str, list[float]] = defaultdict(list)
    quality_matrix: dict[str, list[float]] = defaultdict(list)
    support_matrix: dict[str, list[float]] = defaultdict(list)
    for state_id in sorted(state_map):
        state = state_map[state_id]
        cost_profile = estimated_action_cost_profile(builder, state)
        quality_by_action: dict[str, float] = {}
        utility_by_action: dict[str, float] = {}
        for action_id in expected_actions:
            label = by_state[state_id][action_id]
            quality = _quality(label, composite_spec)
            utility = (
                quality
                - float(risk_weight) * _risk(label)
                - float(cost_weight)
                * cost_profile[action_id]["normalized_estimated_resource_cost"]
            )
            quality_by_action[action_id] = float(quality)
            utility_by_action[action_id] = float(utility)
            utility_matrix[action_id].append(float(utility))
            quality_matrix[action_id].append(float(quality))
            support_matrix[action_id].append(
                (float(label.response.emotional_support) - 1.0) / 4.0
            )
        oracle = min(
            expected_actions,
            key=lambda action: (-utility_by_action[action], action),
        )
        oracle_actions.append(oracle)
        regime = str(evaluator_by_state[state_id]["regime"])
        oracle_by_regime[regime].append(oracle)
        qualities = np.asarray(list(quality_by_action.values()), dtype=float)
        quality_ranges.append(float(np.max(qualities) - np.min(qualities)))
        quality_variances.append(float(np.var(qualities)))

        paired_strategy_effects = []
        by_sources: dict[frozenset[MemorySource], dict[StrategyMode, float]] = defaultdict(dict)
        for action_id, utility in utility_by_action.items():
            sources, mode = parse_action_id(action_id)
            by_sources[sources][mode] = utility
        for modes in by_sources.values():
            if StrategyMode.R0 in modes and StrategyMode.RS in modes:
                paired_strategy_effects.append(
                    modes[StrategyMode.RS] - modes[StrategyMode.R0]
                )
        strategy_effect_by_regime[regime].append(
            float(np.mean(paired_strategy_effects))
            if paired_strategy_effects
            else 0.0
        )

        needed = frozenset(
            MemorySource(str(value))
            for value in evaluator_by_state[state_id].get(
                "needed_memory_sources", []
            )
        )
        m0_best = max(
            utility
            for action_id, utility in utility_by_action.items()
            if not parse_action_id(action_id)[0]
        )
        if needed:
            matching = [
                utility
                for action_id, utility in utility_by_action.items()
                if parse_action_id(action_id)[0] == needed
            ]
            if not matching and len(needed) >= 2:
                matching = [
                    utility
                    for action_id, utility in utility_by_action.items()
                    if len(parse_action_id(action_id)[0]) >= 2
                ]
            memory_effect = max(matching) - m0_best if matching else 0.0
        else:
            memory_utilities = [
                utility
                for action_id, utility in utility_by_action.items()
                if parse_action_id(action_id)[0]
            ]
            memory_effect = max(memory_utilities) - m0_best
        memory_effect_by_regime[regime].append(float(memory_effect))
        rows.append(
            {
                "state_id": state_id,
                "regime": regime,
                "oracle_action": oracle,
                "quality_range": quality_ranges[-1],
                "quality_variance": quality_variances[-1],
                "strategy_rs_minus_r0": strategy_effect_by_regime[regime][-1],
                "memory_minus_m0": memory_effect_by_regime[regime][-1],
            }
        )

    counts = Counter(oracle_actions)
    n_states = len(oracle_actions)
    m0_share = float(
        np.mean([not parse_action_id(action)[0] for action in oracle_actions])
    )
    r0_share = float(
        np.mean(
            [
                parse_action_id(action)[1] is StrategyMode.R0
                for action in oracle_actions
            ]
        )
    )
    rs_share = 1.0 - r0_share
    distinct_by_regime = {
        regime: len(set(actions))
        for regime, actions in sorted(oracle_by_regime.items())
    }

    def regime_mean(values: Mapping[str, Sequence[float]], regime: str) -> float:
        rows_for_regime = list(values.get(regime) or [])
        return float(np.mean(rows_for_regime)) if rows_for_regime else 0.0

    strategy_helpful = regime_mean(
        strategy_effect_by_regime, ResourceNeedRegime.STRATEGY_HELPFUL.value
    )
    strategy_harmful = regime_mean(
        strategy_effect_by_regime, ResourceNeedRegime.STRATEGY_HARMFUL.value
    )
    memory_harmful = regime_mean(
        memory_effect_by_regime, ResourceNeedRegime.MEMORY_HARMFUL.value
    )
    memory_helpful_regimes = (
        ResourceNeedRegime.PROFILE_NEEDED.value,
        ResourceNeedRegime.SUMMARY_NEEDED.value,
        ResourceNeedRegime.EVENT_NEEDED.value,
        ResourceNeedRegime.MULTI_SOURCE_NEEDED.value,
    )
    memory_helpful = float(
        np.mean(
            [regime_mean(memory_effect_by_regime, regime) for regime in memory_helpful_regimes]
        )
    )

    def utility_oracle_policy_headroom(
        matrix: Mapping[str, Sequence[float]],
    ) -> dict[str, Any]:
        fixed_means = {
            action: float(np.mean(matrix[action])) for action in expected_actions
        }
        best_fixed_action = min(
            expected_actions,
            key=lambda action: (-fixed_means[action], action),
        )
        adaptive_mean = float(
            np.mean(
                [
                    matrix[oracle_actions[index]][index]
                    for index in range(n_states)
                ]
            )
        )
        best_fixed_mean = fixed_means[best_fixed_action]
        return {
            "utility_oracle_policy_mean": adaptive_mean,
            "best_fixed_action": best_fixed_action,
            "best_fixed_mean": best_fixed_mean,
            "oracle_minus_best_fixed": adaptive_mean - best_fixed_mean,
            "fixed_action_means": fixed_means,
        }

    oracle_headroom = {
        "utility": utility_oracle_policy_headroom(utility_matrix),
        "quality": utility_oracle_policy_headroom(quality_matrix),
        "emotional_support": utility_oracle_policy_headroom(support_matrix),
    }
    checks = {
        "distinct_oracle_actions": len(counts)
        >= int(thresholds["minimum_distinct_oracle_actions"]),
        "maximum_oracle_action_share": (
            max(counts.values(), default=0) / max(n_states, 1)
            <= float(thresholds["maximum_oracle_action_share"])
        ),
        "m0_oracle_share": m0_share
        >= float(thresholds["minimum_m0_oracle_share"]),
        "r0_oracle_share": r0_share
        >= float(thresholds["minimum_r0_oracle_share"]),
        "rs_oracle_share": rs_share
        >= float(thresholds["minimum_rs_oracle_share"]),
        "per_regime_oracle_diversity": all(
            value
            >= int(thresholds["minimum_distinct_oracle_actions_per_regime"])
            for value in distinct_by_regime.values()
        )
        and set(distinct_by_regime) == set(EXPECTED_REGIME_CHECKS),
        "within_state_quality_range": float(np.mean(quality_ranges))
        >= float(thresholds["minimum_mean_within_state_quality_range"]),
        "within_state_quality_variance": float(np.mean(quality_variances))
        >= float(thresholds["minimum_mean_within_state_quality_variance"]),
        "strategy_helpful_separation": strategy_helpful
        >= float(thresholds["minimum_strategy_directional_separation"]),
        "strategy_harmful_separation": strategy_harmful
        <= -float(thresholds["minimum_strategy_directional_separation"]),
        "memory_helpful_separation": memory_helpful
        >= float(thresholds["minimum_memory_directional_separation"]),
        "memory_harmful_separation": memory_harmful
        <= -float(thresholds["minimum_memory_directional_separation"]),
        "oracle_utility_headroom": oracle_headroom["utility"][
            "oracle_minus_best_fixed"
        ]
        >= float(thresholds["minimum_oracle_utility_headroom"]),
        "oracle_quality_headroom": oracle_headroom["quality"][
            "oracle_minus_best_fixed"
        ]
        >= float(thresholds["minimum_oracle_quality_headroom"]),
        "oracle_emotional_support_headroom": oracle_headroom[
            "emotional_support"
        ]["oracle_minus_best_fixed"]
        >= float(
            thresholds["minimum_oracle_emotional_support_headroom"]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "thresholds": dict(thresholds),
        "n_states": n_states,
        "n_labels": len(labels),
        "oracle_action_distribution": dict(counts),
        "oracle_distinct_actions": len(counts),
        "maximum_oracle_action_share": max(counts.values(), default=0)
        / max(n_states, 1),
        "m0_oracle_share": m0_share,
        "r0_oracle_share": r0_share,
        "rs_oracle_share": rs_share,
        "distinct_oracle_actions_by_regime": distinct_by_regime,
        "mean_within_state_quality_range": float(np.mean(quality_ranges)),
        "mean_within_state_quality_variance": float(np.mean(quality_variances)),
        "strategy_helpful_rs_minus_r0": strategy_helpful,
        "strategy_harmful_rs_minus_r0": strategy_harmful,
        "memory_helpful_memory_minus_m0": memory_helpful,
        "memory_harmful_memory_minus_m0": memory_harmful,
        "oracle_headroom": oracle_headroom,
        "oracle_headroom_interpretation": (
            "same frozen-utility per-state oracle policy evaluated on utility, "
            "quality, and emotional support; optimistic upper bound only, not "
            "learned-routing evidence"
        ),
        "rows": rows,
    }


def audit_deployable_feature_observability(
    states: Sequence[PMV2State],
    *,
    evaluator_contexts: EvaluatorContextIndex,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Train-only grouped-CV check for signal in deployable PM features.

    This is a necessary observability diagnostic, not evidence that the final
    action-value learner will route well or generalize externally.
    """

    expected_keys = {
        "action_id",
        "cv_folds",
        "classifier_c",
        "classifier_max_iter",
        "random_seed",
        "minimum_train_states",
        "minimum_regime_macro_f1",
        "minimum_needed_sources_macro_f1",
        "minimum_memory_need_macro_f1",
        "minimum_strategy_direction_macro_f1",
        "minimum_memory_direction_macro_f1",
    }
    if set(settings) != expected_keys:
        raise RuntimeError("deployable-feature observability settings changed")
    action_id = str(settings["action_id"])
    if action_id != "M0+R0":
        raise RuntimeError("observability must use fixed M0+R0 features")
    train_states = sorted(
        [state for state in states if state.split.value == "train"],
        key=lambda state: state.state_id,
    )
    if len(train_states) < int(settings["minimum_train_states"]):
        raise RuntimeError("insufficient train states for observability audit")
    if any(action_id not in state.allowed_actions for state in train_states):
        raise RuntimeError("M0+R0 is not legal for every observability state")
    contexts = evaluator_contexts.require_states(train_states, exact=False)
    groups = np.asarray([state.user_id for state in train_states], dtype=object)
    folds = int(settings["cv_folds"])
    if folds < 2 or len(set(groups)) < folds:
        raise RuntimeError("insufficient user groups for observability CV")

    regime = np.asarray(
        [str(contexts[state.state_id]["regime"]) for state in train_states],
        dtype=object,
    )
    needed_sources = np.asarray(
        [
            "+".join(sorted(contexts[state.state_id]["needed_memory_sources"]))
            or "M0"
            for state in train_states
        ],
        dtype=object,
    )
    memory_need = np.asarray(
        ["memory" if value != "M0" else "no_memory" for value in needed_sources],
        dtype=object,
    )
    strategy_mask = np.isin(
        regime,
        [
            ResourceNeedRegime.STRATEGY_HELPFUL.value,
            ResourceNeedRegime.STRATEGY_HARMFUL.value,
        ],
    )
    memory_direction_mask = np.isin(
        regime,
        [
            ResourceNeedRegime.PROFILE_NEEDED.value,
            ResourceNeedRegime.SUMMARY_NEEDED.value,
            ResourceNeedRegime.EVENT_NEEDED.value,
            ResourceNeedRegime.MULTI_SOURCE_NEEDED.value,
            ResourceNeedRegime.MEMORY_HARMFUL.value,
        ],
    )
    target_specs = {
        "regime": (regime, np.ones(len(train_states), dtype=bool)),
        "needed_sources": (
            needed_sources,
            np.ones(len(train_states), dtype=bool),
        ),
        "memory_need": (memory_need, np.ones(len(train_states), dtype=bool)),
        "strategy_direction": (
            np.asarray(
                [
                    "helpful"
                    if value == ResourceNeedRegime.STRATEGY_HELPFUL.value
                    else "harmful"
                    for value in regime
                ],
                dtype=object,
            ),
            strategy_mask,
        ),
        "memory_direction": (
            np.asarray(
                [
                    "harmful"
                    if value == ResourceNeedRegime.MEMORY_HARMFUL.value
                    else "helpful"
                    for value in regime
                ],
                dtype=object,
            ),
            memory_direction_mask,
        ),
    }
    predictions = {
        name: np.empty(len(train_states), dtype=object) for name in target_specs
    }
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=folds)
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(train_states, groups=groups)
    ):
        fold_train = [train_states[index] for index in train_index]
        fold_validation = [train_states[index] for index in validation_index]
        builder = PMV2FeatureBuilder().fit(fold_train)
        x_train = builder.transform([(state, action_id) for state in fold_train])
        x_validation = builder.transform(
            [(state, action_id) for state in fold_validation]
        )
        fold_metrics: dict[str, float] = {}
        for name, (target, target_mask) in target_specs.items():
            selected_train_positions = np.flatnonzero(target_mask[train_index])
            selected_validation_positions = np.flatnonzero(
                target_mask[validation_index]
            )
            selected_train_index = train_index[selected_train_positions]
            selected_validation_index = validation_index[
                selected_validation_positions
            ]
            train_target = target[selected_train_index]
            all_classes = sorted(set(target[target_mask]))
            if set(train_target) != set(all_classes):
                raise RuntimeError(
                    f"observability fold {fold_index} lacks {name} classes"
                )
            classifier = LogisticRegression(
                C=float(settings["classifier_c"]),
                max_iter=int(settings["classifier_max_iter"]),
                class_weight="balanced",
                random_state=int(settings["random_seed"]),
            )
            classifier.fit(x_train[selected_train_positions], train_target)
            predicted = classifier.predict(
                x_validation[selected_validation_positions]
            )
            predictions[name][selected_validation_index] = predicted
            fold_metrics[name + "_macro_f1"] = float(
                f1_score(
                    target[selected_validation_index],
                    predicted,
                    labels=all_classes,
                    average="macro",
                    zero_division=0,
                )
            )
        fold_rows.append(
            {
                "fold": fold_index,
                "train_users": len(set(groups[train_index])),
                "validation_users": len(set(groups[validation_index])),
                "train_states": len(train_index),
                "validation_states": len(validation_index),
                "metrics": fold_metrics,
            }
        )

    metrics = {
        name + "_macro_f1": float(
            f1_score(
                target[target_mask],
                predictions[name][target_mask],
                labels=sorted(set(target[target_mask])),
                average="macro",
                zero_division=0,
            )
        )
        for name, (target, target_mask) in target_specs.items()
    }
    checks = {
        "regime_macro_f1": metrics["regime_macro_f1"]
        >= float(settings["minimum_regime_macro_f1"]),
        "needed_sources_macro_f1": metrics["needed_sources_macro_f1"]
        >= float(settings["minimum_needed_sources_macro_f1"]),
        "memory_need_macro_f1": metrics["memory_need_macro_f1"]
        >= float(settings["minimum_memory_need_macro_f1"]),
        "strategy_direction_macro_f1": metrics[
            "strategy_direction_macro_f1"
        ]
        >= float(settings["minimum_strategy_direction_macro_f1"]),
        "memory_direction_macro_f1": metrics["memory_direction_macro_f1"]
        >= float(settings["minimum_memory_direction_macro_f1"]),
    }
    builder_contract = PMV2FeatureBuilder().fit(train_states)
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "settings": dict(settings),
        "train_only": True,
        "n_states": len(train_states),
        "n_users": len(set(groups)),
        "class_counts": {
            name: dict(Counter(target[target_mask]))
            for name, (target, target_mask) in target_specs.items()
        },
        "metrics": metrics,
        "folds": fold_rows,
        "feature_builder_config_sha256": builder_contract.config_hash(),
        "interpretation": (
            "necessary deployable-feature observability diagnostic only; "
            "not learned-routing or external-performance evidence"
        ),
    }
