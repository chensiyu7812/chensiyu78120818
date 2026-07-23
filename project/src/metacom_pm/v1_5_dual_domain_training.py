from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import MemorySource
from .io import canonical_json, sha256_text
from .pm_v2_contracts import ActionLabel, PMV2Split, PMV2State
from .pm_v2_judging import (
    DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL,
    dimension_applicability_by_action,
    dimensions_inapplicable_to_every_action,
)
from .pm_v2_model import (
    MAD_ADJUSTED_CONSERVATIVE_UTILITY_LAMBDA,
    MAD_ADJUSTED_CONSERVATIVE_UTILITY_PROTOCOL,
    RESPONSE_FIELDS,
    applicable_risk_fields,
    estimated_action_cost_profile,
    evaluate_policy,
    evaluate_prediction_coverage,
    mad_adjusted_conservative_utility,
)


DUAL_DOMAIN_TRAINING_PROTOCOL = "pm-v1.5-dual-domain-training-input-v1"
LONGITUDINAL_DOMAIN = "longitudinal_synthetic"
ESCONV_AUXILIARY_DOMAIN = "esconv_auxiliary"
ESCONV_AUXILIARY_FAMILY = "esconv_auxiliary_strategy_routing"
ESCONV_AUXILIARY_ACTIONS = frozenset({"M0+R0", "M0+RS"})
DUAL_DOMAIN_GATE_PROTOCOL = "pm-v1.5-dual-domain-independent-internal-gates-v1"


def training_domain_for_state(state: PMV2State) -> str:
    if state.semantic_family == ESCONV_AUXILIARY_FAMILY:
        if state.provenance.get("adapted_from_runtime_state") is not True:
            raise ValueError(
                f"ESConv auxiliary state {state.state_id} lacks runtime provenance"
            )
        return ESCONV_AUXILIARY_DOMAIN
    return LONGITUDINAL_DOMAIN


def _state_universe_sha256(states: Sequence[PMV2State]) -> str:
    return sha256_text(canonical_json(sorted(state.state_id for state in states)))


def _label_universe_sha256(labels: Sequence[ActionLabel]) -> str:
    return sha256_text(
        canonical_json(
            sorted((label.state_id, label.action_id) for label in labels)
        )
    )


def _validate_unique_states(
    states: Sequence[PMV2State], *, domain: str
) -> dict[str, PMV2State]:
    by_id = {state.state_id: state for state in states}
    if not states or len(by_id) != len(states):
        raise ValueError(f"{domain} states must be unique and non-empty")
    card_ids = [state.card_id for state in states]
    if len(card_ids) != len(set(card_ids)):
        raise ValueError(f"{domain} card ids must be unique")
    return by_id


def _validate_split_group_isolation(
    states: Sequence[PMV2State], *, domain: str
) -> dict[str, Any]:
    splits_by_user: dict[str, set[str]] = {}
    for state in states:
        splits_by_user.setdefault(state.user_id, set()).add(state.split.value)
    leaked = sorted(
        user_id for user_id, splits in splits_by_user.items() if len(splits) != 1
    )
    if leaked:
        raise ValueError(f"{domain} users/dialogues cross splits: {leaked[:10]}")
    training_splits = (
        PMV2Split.TRAIN,
        PMV2Split.CALIBRATION,
        PMV2Split.INTERNAL_TEST,
    )
    unexpected = sorted(
        state.state_id for state in states if state.split not in training_splits
    )
    if unexpected:
        raise ValueError(f"{domain} contains non-training splits: {unexpected[:10]}")
    return {
        "state_counts": {
            split.value: sum(state.split is split for state in states)
            for split in training_splits
        },
        "group_counts": {
            split.value: len(
                {state.user_id for state in states if state.split is split}
            )
            for split in training_splits
        },
    }


def _validate_exact_labels(
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    *,
    domain: str,
) -> None:
    state_map = {state.state_id: state for state in states}
    keys = [(label.state_id, label.action_id) for label in labels]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{domain} labels contain duplicate state/action rows")
    expected = {
        (state.state_id, action_id)
        for state in states
        if state.split is not PMV2Split.INTERNAL_TEST
        for action_id in state.allowed_actions
    }
    if set(keys) != expected:
        raise ValueError(
            f"{domain} train/calibration labels are not the exact legal matrix; "
            f"missing={sorted(expected - set(keys))[:10]}, "
            f"extra={sorted(set(keys) - expected)[:10]}"
        )
    for label in labels:
        state = state_map[label.state_id]
        if label.user_id != state.user_id or label.card_id != state.card_id:
            raise ValueError(f"{domain} label identity mismatch for {label.state_id}")


def validate_internal_domain_labels(
    *,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    domain: str,
) -> tuple[list[PMV2State], list[ActionLabel]]:
    """Validate one internal domain after its independent ledger is spent.

    This deliberately accepts only INTERNAL_TEST states.  The function is kept
    separate from :func:`validate_dual_domain_training_inputs` so callers cannot
    accidentally deserialize internal outcomes during fitting or calibration.
    """

    internal_states = list(states)
    internal_labels = list(labels)
    if not internal_states or any(
        state.split is not PMV2Split.INTERNAL_TEST for state in internal_states
    ):
        raise ValueError(f"{domain} internal input contains non-internal states")
    state_map = _validate_unique_states(internal_states, domain=f"{domain} internal")
    keys = [(label.state_id, label.action_id) for label in internal_labels]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{domain} internal labels contain duplicate rows")
    expected = {
        (state.state_id, action_id)
        for state in internal_states
        for action_id in state.allowed_actions
    }
    if set(keys) != expected:
        raise ValueError(
            f"{domain} internal labels are not the exact legal matrix; "
            f"missing={sorted(expected - set(keys))[:10]}, "
            f"extra={sorted(set(keys) - expected)[:10]}"
        )
    for label in internal_labels:
        state = state_map.get(label.state_id)
        if state is None:
            raise ValueError(f"{domain} internal labels reference another domain")
        if label.user_id != state.user_id or label.card_id != state.card_id:
            raise ValueError(
                f"{domain} internal label identity mismatch for {label.state_id}"
            )
        if domain == ESCONV_AUXILIARY_DOMAIN:
            if label.action_id not in ESCONV_AUXILIARY_ACTIONS:
                raise ValueError("ESConv internal labels contain a memory action")
            if str(label.provenance.get("esconv_auxiliary_split") or "") != (
                PMV2Split.INTERNAL_TEST.value
            ):
                raise ValueError("ESConv internal label split provenance mismatch")
    return internal_states, internal_labels


def audit_domain_label_matrix(
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    *,
    domain: str,
    low_mad_threshold: float,
    minimum_reliable_rate: float,
    minimum_low_mad_coverage_per_dimension: float,
    minimum_low_mad_coverage_per_action_dimension: float,
) -> dict[str, Any]:
    """Outcome audit reported independently for each training domain."""

    state_map = {state.state_id: state for state in states}
    if not state_map or len(state_map) != len(states):
        raise ValueError(f"{domain} audit requires unique non-empty states")
    label_map = {(label.state_id, label.action_id): label for label in labels}
    expected = {
        (state.state_id, action_id)
        for state in states
        for action_id in state.allowed_actions
    }
    if set(label_map) != expected:
        raise ValueError(f"{domain} audit requires the exact action matrix")
    action_counts: dict[str, int] = {}
    for _state_id, action_id in label_map:
        action_counts[action_id] = action_counts.get(action_id, 0) + 1
    action_ids_present = sorted(action_counts)
    # Risk dimensions that are structurally inapplicable to an action (e.g.
    # selected_context_misuse when no memory source was ever selected) are
    # not a judge-reliability signal at all: they must be excluded from the
    # low-MAD-coverage gate outright (N/A), never scored as 0 (which would
    # spuriously fail the gate) or silently dropped in a way that could
    # inflate the reported minimum and help a real failure PASS.
    inapplicable_risk_dimensions = dimensions_inapplicable_to_every_action(
        action_ids_present
    )
    inapplicable_risk_dimensions_by_action = dimension_applicability_by_action(
        action_ids_present
    )
    dimension_names = sorted(next(iter(label_map.values())).dimension_mad)
    dimension_coverage: dict[str, float | None] = {}
    for name in dimension_names:
        if name in inapplicable_risk_dimensions:
            dimension_coverage[name] = None
            continue
        dimension_coverage[name] = float(
            np.mean(
                [float(label.dimension_mad[name]) <= low_mad_threshold for label in labels]
            )
        )
    action_dimension_coverage: dict[str, dict[str, float | None]] = {}
    for action_id in action_ids_present:
        inapplicable_for_action = inapplicable_risk_dimensions_by_action.get(
            action_id, frozenset()
        )
        row: dict[str, float | None] = {}
        for name in dimension_names:
            if name in inapplicable_for_action:
                row[name] = None
                continue
            row[name] = float(
                np.mean(
                    [
                        float(label.dimension_mad[name]) <= low_mad_threshold
                        for label in labels
                        if label.action_id == action_id
                    ]
                )
            )
        action_dimension_coverage[action_id] = row
    reliable_rate = float(np.mean([label.label_reliable for label in labels]))
    applicable_dimension_coverage_values = [
        value for value in dimension_coverage.values() if value is not None
    ]
    if not applicable_dimension_coverage_values:
        raise RuntimeError(
            f"{domain} audit has no applicable dimensions left to gate coverage on"
        )
    applicable_action_dimension_coverage_values = [
        value
        for action_row in action_dimension_coverage.values()
        for value in action_row.values()
        if value is not None
    ]
    if not applicable_action_dimension_coverage_values:
        raise RuntimeError(
            f"{domain} audit has no applicable action-dimension cells left to "
            "gate coverage on"
        )
    # label_reliable_rate is diagnostic-only (matches validate_judge_table's
    # joint_reliable_rate_is_diagnostic_only convention): it is reported, but
    # never itself a hard gate. Matrix completeness (checked above via the
    # exact expected-action-set comparison) and the MAD-based coverage checks
    # below remain hard gates.
    checks = {
        "low_mad_coverage_per_dimension": min(applicable_dimension_coverage_values)
        >= float(minimum_low_mad_coverage_per_dimension),
        "low_mad_coverage_per_action_dimension": min(
            applicable_action_dimension_coverage_values
        )
        >= float(minimum_low_mad_coverage_per_action_dimension),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{domain} label-quality gate failed: {checks}")
    return {
        "domain": domain,
        "status": "PASS",
        "state_count": len(states),
        "group_count": len({state.user_id for state in states}),
        "label_count": len(labels),
        "action_distribution": dict(sorted(action_counts.items())),
        "label_reliable_rate_diagnostic": reliable_rate,
        "label_reliable_rate_is_diagnostic_only": True,
        "minimum_reliable_rate_diagnostic_threshold": float(minimum_reliable_rate),
        "maximum_dimension_mad": float(max(label.max_dimension_mad for label in labels)),
        "low_mad_threshold": float(low_mad_threshold),
        "dimension_coverage": dimension_coverage,
        "action_dimension_coverage": action_dimension_coverage,
        "inapplicable_risk_dimensions": sorted(inapplicable_risk_dimensions),
        "inapplicable_risk_dimensions_by_action": {
            action_id: sorted(dims)
            for action_id, dims in sorted(inapplicable_risk_dimensions_by_action.items())
        },
        "dimension_applicability_contract_protocol": DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL,
        "minimum_dimension_low_mad_coverage": min(applicable_dimension_coverage_values),
        "minimum_action_dimension_low_mad_coverage": min(
            applicable_action_dimension_coverage_values
        ),
        "checks": checks,
        "state_universe_sha256": _state_universe_sha256(states),
        "label_universe_sha256": _label_universe_sha256(labels),
    }


def require_equal_domain_training_weight(training_report: Mapping[str, Any]) -> dict[str, float]:
    """Fail closed unless both model heads and routing objective weight domains equally."""

    head = dict(training_report.get("domain_dialogue_state_action_weighting") or {})
    routing = dict(
        ((training_report.get("routing_objective") or {}).get(
            "domain_dialogue_state_action_weighting"
        ))
        or {}
    )
    expected_domains = {LONGITUDINAL_DOMAIN, ESCONV_AUXILIARY_DOMAIN}
    for name, row in (("prediction_heads", head), ("routing_objective", routing)):
        weights = {str(key): float(value) for key, value in (row.get("effective_weight_by_domain") or row.get("domain_weight") or {}).items()}
        if set(weights) != expected_domains:
            raise RuntimeError(f"{name} lacks the exact two training domains")
        total = fsum(weights.values())
        normalized = {key: value / total for key, value in weights.items()}
        if any(abs(value - 0.5) > 1e-9 for value in normalized.values()):
            raise RuntimeError(f"{name} permits one domain to swamp the other")
    return {LONGITUDINAL_DOMAIN: 0.5, ESCONV_AUXILIARY_DOMAIN: 0.5}


def fixed_action_metrics(model, states, labels, action_id: str) -> dict[str, Any] | None:
    """Evaluate a fixed comparator without pretending it is a learned policy."""

    label_map = {(label.state_id, label.action_id): label for label in labels}
    rows: list[dict[str, Any]] = []
    for state in states:
        if action_id not in state.allowed_actions:
            return None
        label = label_map.get((state.state_id, action_id))
        if label is None:
            return None
        quality = model.selection_config.composite_spec.score(label.response)
        risk = max(
            float(getattr(label.risk, name)) / 3.0
            for name in applicable_risk_fields(action_id)
        )
        cost = estimated_action_cost_profile(model.feature_builder, state)[action_id]
        normalized_cost = float(cost["normalized_estimated_resource_cost"])
        utility = (
            quality
            - model.selection_config.risk_weight * risk
            - model.selection_config.cost_weight * normalized_cost
        )
        conservative = mad_adjusted_conservative_utility(
            label=label,
            action_id=action_id,
            composite_spec=model.selection_config.composite_spec,
            risk_weight=model.selection_config.risk_weight,
            cost_weight=model.selection_config.cost_weight,
            normalized_cost=normalized_cost,
        )
        rows.append(
            {
                "state_id": state.state_id,
                "user_id": state.user_id,
                "quality": float(quality),
                "risk": float(risk),
                "realized_utility": float(utility),
                # See mad_adjusted_conservative_utility (pm_v2_model.py):
                # fixed lambda=1.0, never tuned from calibration/internal-
                # test; a disclosed conservative adjustment, not a
                # confidence interval, gold standard, or true user utility.
                # nominal_* fields are never overwritten by these.
                "nominal_quality": float(quality),
                "nominal_risk": float(risk),
                "nominal_utility": float(utility),
                "conservative_quality": conservative["conservative_quality"],
                "conservative_risk": conservative["conservative_risk"],
                "conservative_utility": conservative["conservative_utility"],
                "observed_input_tokens": float(label.observed_input_tokens),
                "response_dimensions": {
                    name: float(getattr(label.response, name)) for name in RESPONSE_FIELDS
                },
            }
        )
    if not rows:
        return None
    return {
        "action_id": action_id,
        "n": len(rows),
        "mean_quality": float(np.mean([row["quality"] for row in rows])),
        "mean_risk": float(np.mean([row["risk"] for row in rows])),
        "mean_realized_utility": float(
            np.mean([row["realized_utility"] for row in rows])
        ),
        "mean_conservative_quality": float(
            np.mean([row["conservative_quality"] for row in rows])
        ),
        "mean_conservative_risk": float(
            np.mean([row["conservative_risk"] for row in rows])
        ),
        "mean_conservative_utility": float(
            np.mean([row["conservative_utility"] for row in rows])
        ),
        "mean_observed_input_tokens": float(
            np.mean([row["observed_input_tokens"] for row in rows])
        ),
        "mean_response_dimensions": {
            name: float(
                np.mean([row["response_dimensions"][name] for row in rows])
            )
            for name in RESPONSE_FIELDS
        },
        "rows": rows,
    }


def _paired_group_bootstrap(
    policy_rows,
    comparator_rows,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    policy = {str(row["state_id"]): row for row in policy_rows}
    comparator = {str(row["state_id"]): row for row in comparator_rows}
    if set(policy) != set(comparator) or not policy:
        raise ValueError("paired domain gate requires identical non-empty states")
    by_group: dict[str, list[dict[str, float]]] = {}
    for state_id in sorted(policy):
        left, right = policy[state_id], comparator[state_id]
        if left["user_id"] != right["user_id"]:
            raise ValueError("paired domain gate group mismatch")
        by_group.setdefault(str(left["user_id"]), []).append(
            {
                "quality": float(left["quality"] - right["quality"]),
                "emotional_support": float(
                    left["response_dimensions"]["emotional_support"]
                    - right["response_dimensions"]["emotional_support"]
                ),
                "risk": float(left["risk"] - right["risk"]),
                "utility": float(
                    left["realized_utility"] - right["realized_utility"]
                ),
                # MAD-adjusted conservative deltas (lambda=1.0 fixed): the
                # dual-domain gate's own PASS/NOT_SUPPORTED check binds to
                # these, never the nominal deltas above.
                "conservative_quality": float(
                    left["conservative_quality"] - right["conservative_quality"]
                ),
                "conservative_risk": float(
                    left["conservative_risk"] - right["conservative_risk"]
                ),
                "conservative_utility": float(
                    left["conservative_utility"] - right["conservative_utility"]
                ),
            }
        )
    if int(replicates) < 100:
        raise ValueError("paired domain bootstrap requires at least 100 replicates")
    if not 0.5 < float(confidence_level) < 1.0:
        raise ValueError("paired domain bootstrap confidence must be in (0.5, 1.0)")
    metric_names = (
        "quality",
        "emotional_support",
        "risk",
        "utility",
        "conservative_quality",
        "conservative_risk",
        "conservative_utility",
    )
    groups = sorted(by_group)
    if len(groups) < 3:
        raise ValueError("paired domain bootstrap requires at least three groups")
    # Each ESConv dialogue and each longitudinal user is one independent block,
    # irrespective of how many states it contributes.
    group_means = np.asarray(
        [
            [float(np.mean([row[name] for row in by_group[group]])) for name in metric_names]
            for group in groups
        ],
        dtype=float,
    )
    point = np.mean(group_means, axis=0)
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, len(groups), size=(int(replicates), len(groups)))
    bootstrap = np.mean(group_means[sampled], axis=1)
    alpha = 1.0 - float(confidence_level)
    lower = np.quantile(bootstrap, alpha / 2.0, axis=0)
    upper = np.quantile(bootstrap, 1.0 - alpha / 2.0, axis=0)
    return {
        "cluster_key": "user_id_or_dialogue_id",
        "n_groups": len(groups),
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "metrics": {
            name: {
                "mean_delta": float(point[index]),
                "ci_lower": float(lower[index]),
                "ci_upper": float(upper[index]),
            }
            for index, name in enumerate(metric_names)
        },
    }


def domain_internal_gate(
    *,
    model,
    rule_router,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    domain: str,
    fixed_actions: Sequence[str],
    gate_config: Mapping[str, Any],
    uncertainty_confidence_level: float,
    bootstrap_replicates: int,
    bootstrap_confidence_level: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Independent comparator gate; no domain may compensate for another."""

    policy = evaluate_policy(model, states, labels)
    rule = evaluate_policy(rule_router, states, labels)
    coverage = evaluate_prediction_coverage(
        model,
        states,
        labels,
        confidence_level=float(uncertainty_confidence_level),
    )
    comparators = {"transparent_rule": rule}
    for action_id in fixed_actions:
        row = fixed_action_metrics(model, states, labels, action_id)
        if row is None:
            raise RuntimeError(f"{domain} fixed comparator {action_id} is not legal")
        comparators[action_id] = row
    thresholds_m = dict(gate_config["gate_m_learned_vs_rule"])
    thresholds_f = dict(gate_config["gate_f_fixed_guardrail"])

    def check(
        comparator: Mapping[str, Any],
        thresholds: Mapping[str, Any],
        *,
        strict_utility: bool,
        seed_offset: int,
    ) -> dict[str, Any]:
        bootstrap = _paired_group_bootstrap(
            policy["rows"],
            comparator["rows"],
            replicates=bootstrap_replicates,
            confidence_level=bootstrap_confidence_level,
            seed=int(bootstrap_seed) + int(seed_offset),
        )
        metrics = bootstrap["metrics"]
        # The gate binds explicitly to the MAD-adjusted conservative metrics
        # (lambda=1.0 fixed), never the nominal ones -- nominal deltas remain
        # in the report (bootstrap["metrics"]) for transparency only.
        checks = {
            "quality_noninferior": metrics["conservative_quality"]["ci_lower"] >= float(thresholds["minimum_quality_delta"]),
            "emotional_support_noninferior": metrics["emotional_support"]["ci_lower"] >= float(thresholds["minimum_emotional_support_delta"]),
            "risk_nonincrease": metrics["conservative_risk"]["ci_upper"] <= float(thresholds["maximum_risk_delta"]),
            "utility": metrics["conservative_utility"]["ci_lower"] > float(thresholds["minimum_utility_delta"])
            if strict_utility
            else metrics["conservative_utility"]["ci_lower"] >= float(thresholds["minimum_utility_delta"]),
        }
        return {
            "status": "PASS" if all(checks.values()) else "NOT_SUPPORTED",
            "checks": checks,
            "paired_group_bootstrap": bootstrap,
        }

    learned_vs_rule = check(
        rule, thresholds_m, strict_utility=True, seed_offset=0
    )
    fixed = {
        action_id: check(
            comparators[action_id],
            thresholds_f,
            strict_utility=False,
            seed_offset=index + 1,
        )
        for index, action_id in enumerate(fixed_actions)
    }
    fixed_status = all(row["status"] == "PASS" for row in fixed.values())
    status = "PASS" if learned_vs_rule["status"] == "PASS" and fixed_status else "NOT_SUPPORTED"
    return {
        "protocol": DUAL_DOMAIN_GATE_PROTOCOL,
        "domain": domain,
        "status": status,
        "learned_policy": {key: value for key, value in policy.items() if key != "rows"},
        "prediction_coverage": coverage,
        "learned_vs_transparent_rule": learned_vs_rule,
        "learned_vs_fixed": fixed,
        "fixed_comparators": {
            key: {name: value for name, value in row.items() if name != "rows"}
            for key, row in comparators.items()
            if key != "transparent_rule"
        },
    }


@dataclass(frozen=True)
class DualDomainTrainingInputs:
    longitudinal_states: tuple[PMV2State, ...]
    auxiliary_states: tuple[PMV2State, ...]
    longitudinal_train_calibration_labels: tuple[ActionLabel, ...]
    auxiliary_train_calibration_labels: tuple[ActionLabel, ...]
    report: dict[str, Any]

    @property
    def combined_states(self) -> list[PMV2State]:
        return [*self.longitudinal_states, *self.auxiliary_states]

    @property
    def combined_train_calibration_labels(self) -> list[ActionLabel]:
        return [
            *self.longitudinal_train_calibration_labels,
            *self.auxiliary_train_calibration_labels,
        ]

    def states_for_split(self, split: PMV2Split) -> list[PMV2State]:
        return [state for state in self.combined_states if state.split is split]

    def labels_for_split(self, split: PMV2Split) -> list[ActionLabel]:
        state_ids = {state.state_id for state in self.states_for_split(split)}
        return [
            label
            for label in self.combined_train_calibration_labels
            if label.state_id in state_ids
        ]

    def fit_and_calibration_views(
        self,
    ) -> tuple[
        dict[PMV2Split, list[PMV2State]],
        dict[PMV2Split, list[ActionLabel]],
    ]:
        """Return the only outcome-bearing views permitted before candidate freeze."""

        allowed = (PMV2Split.TRAIN, PMV2Split.CALIBRATION)
        states = {split: self.states_for_split(split) for split in allowed}
        labels = {split: self.labels_for_split(split) for split in allowed}
        internal_ids = {
            state.state_id for state in self.states_for_split(PMV2Split.INTERNAL_TEST)
        }
        leaked = sorted(
            label.state_id
            for rows in labels.values()
            for label in rows
            if label.state_id in internal_ids
        )
        if leaked:
            raise RuntimeError(
                f"internal outcomes entered fit/calibration views: {leaked[:10]}"
            )
        return states, labels


def validate_dual_domain_training_inputs(
    *,
    longitudinal_states: Sequence[PMV2State],
    auxiliary_states: Sequence[PMV2State],
    longitudinal_train_calibration_labels: Sequence[ActionLabel],
    auxiliary_train_calibration_labels: Sequence[ActionLabel],
    pm_config: Mapping[str, Any],
) -> DualDomainTrainingInputs:
    """Validate the two training domains without opening either internal label file."""

    long_states = tuple(longitudinal_states)
    aux_states = tuple(auxiliary_states)
    long_labels = tuple(longitudinal_train_calibration_labels)
    aux_labels = tuple(auxiliary_train_calibration_labels)
    long_map = _validate_unique_states(long_states, domain=LONGITUDINAL_DOMAIN)
    aux_map = _validate_unique_states(aux_states, domain=ESCONV_AUXILIARY_DOMAIN)
    overlap = sorted(set(long_map) & set(aux_map))
    if overlap:
        raise ValueError(f"dual-domain state ids overlap: {overlap[:10]}")
    card_overlap = sorted(
        {state.card_id for state in long_states}
        & {state.card_id for state in aux_states}
    )
    if card_overlap:
        raise ValueError(f"dual-domain card ids overlap: {card_overlap[:10]}")
    user_overlap = sorted(
        {state.user_id for state in long_states}
        & {state.user_id for state in aux_states}
    )
    if user_overlap:
        raise ValueError(f"dual-domain user/dialogue ids overlap: {user_overlap[:10]}")

    long_counts = _validate_split_group_isolation(
        long_states, domain=LONGITUDINAL_DOMAIN
    )
    aux_counts = _validate_split_group_isolation(
        aux_states, domain=ESCONV_AUXILIARY_DOMAIN
    )
    data_cfg = dict(pm_config.get("data_generation") or {})
    expected_long_groups = {
        "train": int(data_cfg.get("train_users") or 0),
        "calibration": int(data_cfg.get("calibration_users") or 0),
        "internal_test": int(data_cfg.get("internal_test_users") or 0),
    }
    cases_per_user = int(data_cfg.get("cases_per_user") or 0)
    expected_long_states = {
        split: groups * cases_per_user
        for split, groups in expected_long_groups.items()
    }
    aux_cfg = dict(pm_config.get("esconv_auxiliary_training") or {})
    if aux_cfg.get("protocol") != (
        "pm-v1.5-esconv-auxiliary-bank-disjoint-seed-training-support-v1"
    ):
        raise ValueError("ESConv auxiliary training contract is missing or stale")
    expected_aux_groups = {
        str(key): int(value)
        for key, value in (aux_cfg.get("split_dialogue_counts") or {}).items()
    }
    expected_aux_states = {
        str(key): int(value)
        for key, value in (aux_cfg.get("expected_state_counts") or {}).items()
    }
    if long_counts["group_counts"] != expected_long_groups:
        raise ValueError("longitudinal group counts differ from the frozen contract")
    if long_counts["state_counts"] != expected_long_states:
        raise ValueError("longitudinal state counts differ from the frozen contract")
    if aux_counts["group_counts"] != expected_aux_groups:
        raise ValueError("ESConv auxiliary dialogue counts differ from the frozen contract")
    if aux_counts["state_counts"] != expected_aux_states:
        raise ValueError("ESConv auxiliary state counts differ from the frozen contract")
    if sum(expected_aux_states.values()) != int(aux_cfg.get("expected_total_states") or 0):
        raise ValueError("ESConv auxiliary expected total is internally inconsistent")

    for state in aux_states:
        if training_domain_for_state(state) != ESCONV_AUXILIARY_DOMAIN:
            raise ValueError(f"invalid ESConv auxiliary state {state.state_id}")
        if set(state.allowed_actions) != ESCONV_AUXILIARY_ACTIONS:
            raise ValueError(
                f"ESConv auxiliary state {state.state_id} has non-canonical actions"
            )
        for source in MemorySource:
            summary = state.inventory[source]
            if summary.available or summary.count != 0 or summary.estimated_tokens != 0:
                raise ValueError(
                    f"ESConv auxiliary state {state.state_id} exposes {source.value}"
                )
    if any(
        training_domain_for_state(state) != LONGITUDINAL_DOMAIN
        for state in long_states
    ):
        raise ValueError("longitudinal input contains ESConv auxiliary states")
    _validate_exact_labels(long_states, long_labels, domain=LONGITUDINAL_DOMAIN)
    _validate_exact_labels(aux_states, aux_labels, domain=ESCONV_AUXILIARY_DOMAIN)
    aux_state_map = {state.state_id: state for state in aux_states}
    for label in aux_labels:
        split = str(label.provenance.get("esconv_auxiliary_split") or "")
        if split != aux_state_map[label.state_id].split.value:
            raise ValueError(
                f"ESConv auxiliary label split provenance mismatch for {label.state_id}"
            )

    report = {
        "protocol": DUAL_DOMAIN_TRAINING_PROTOCOL,
        "status": "PASS",
        "internal_labels_opened": False,
        "domains": {
            LONGITUDINAL_DOMAIN: {
                **long_counts,
                "state_universe_sha256": _state_universe_sha256(long_states),
                "train_calibration_label_universe_sha256": _label_universe_sha256(
                    long_labels
                ),
            },
            ESCONV_AUXILIARY_DOMAIN: {
                **aux_counts,
                "state_universe_sha256": _state_universe_sha256(aux_states),
                "train_calibration_label_universe_sha256": _label_universe_sha256(
                    aux_labels
                ),
                "legal_actions": sorted(ESCONV_AUXILIARY_ACTIONS),
            },
        },
        "top_level_domain_weight": {
            LONGITUDINAL_DOMAIN: 0.5,
            ESCONV_AUXILIARY_DOMAIN: 0.5,
        },
        "state_id_overlap_count": 0,
        "card_id_overlap_count": 0,
        "user_or_dialogue_id_overlap_count": 0,
    }
    return DualDomainTrainingInputs(
        longitudinal_states=long_states,
        auxiliary_states=aux_states,
        longitudinal_train_calibration_labels=long_labels,
        auxiliary_train_calibration_labels=aux_labels,
        report=report,
    )
