from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import MemorySource, StrategyMode, parse_action_id
from .io import canonical_json, sha256_text
from .pm_v1_6_contracts import AlgorithmCandidate, Step0Observation
from .pm_v1_6_models import ConservativeResidualPolicy, OutcomeMode, OverrideConfig, PMV16OutcomeModel
from .pm_v1_6_router import StrongTransparentRouter, TransparentRouterConfig, router_grid
from .pm_v2_contracts import ActionLabel, CompositeSpec, PMV2State
from .pm_v2_model import applicable_risk_fields

ALGORITHM_COMPLEXITY_ORDER = (
    AlgorithmCandidate.ABSOLUTE_HGB,
    AlgorithmCandidate.STATE_CENTERED_HGB,
    AlgorithmCandidate.RULE_RELATIVE_HGB,
)


@dataclass(frozen=True)
class PolicyMetricRow:
    state_id: str
    user_id: str
    chosen_action: str
    rule_action: str
    quality: float
    emotional_support: float
    risk: float
    normalized_cost: float
    utility: float
    oracle_utility: float
    regret: float


def _label_map(labels: Sequence[ActionLabel]) -> dict[tuple[str, str], ActionLabel]:
    result: dict[tuple[str, str], ActionLabel] = {}
    for label in labels:
        key = (label.state_id, label.action_id)
        if key in result:
            raise ValueError(f"duplicate action label: {key}")
        result[key] = label
    return result


def _observed_metrics(*, label: ActionLabel, composite: CompositeSpec, risk_weight: float, cost_weight: float, estimated_cost: float, maximum_cost: float) -> tuple[float, float, float, float, float]:
    quality = composite.score(label.response)
    support = (float(label.response.emotional_support) - 1.0) / 4.0
    risk = max(float(getattr(label.risk, head)) / 3.0 for head in applicable_risk_fields(label.action_id))
    normalized_cost = estimated_cost / max(maximum_cost, 1.0)
    utility = quality - risk_weight * risk - cost_weight * normalized_cost
    return quality, support, risk, normalized_cost, utility


def _rule_actions(states: Sequence[PMV2State], step0_by_state: Mapping[str, Step0Observation], router: StrongTransparentRouter) -> dict[str, str]:
    result = {}
    for state in states:
        action = router.choose(step0_by_state[state.state_id]).action_id
        if action not in state.allowed_actions:
            raise RuntimeError(f"router produced illegal action for {state.state_id}")
        result[state.state_id] = action
    return result


def evaluate_observed_policy(*, model: PMV16OutcomeModel, router: StrongTransparentRouter, override_config: OverrideConfig, states: Sequence[PMV2State], labels: Sequence[ActionLabel], step0_by_state: Mapping[str, Step0Observation]) -> list[PolicyMetricRow]:
    label_by_key = _label_map(labels)
    policy = ConservativeResidualPolicy(outcome_model=model, rule_router=router, override_config=override_config)
    rows: list[PolicyMetricRow] = []
    for state in states:
        step0 = step0_by_state[state.state_id]
        decision = policy.choose(state, step0)
        max_cost = max(model.feature_builder.estimate_action_cost(action, step0) for action in state.allowed_actions)
        observed = {}
        for action in state.allowed_actions:
            observed[action] = _observed_metrics(
                label=label_by_key[(state.state_id, action)],
                composite=model.composite_spec,
                risk_weight=override_config.risk_weight,
                cost_weight=override_config.cost_weight,
                estimated_cost=model.feature_builder.estimate_action_cost(action, step0),
                maximum_cost=max_cost,
            )
        oracle_action = max(state.allowed_actions, key=lambda action: (observed[action][4], action))
        chosen = observed[decision.chosen_action]
        rows.append(PolicyMetricRow(
            state_id=state.state_id,
            user_id=state.user_id,
            chosen_action=decision.chosen_action,
            rule_action=decision.rule_action,
            quality=chosen[0],
            emotional_support=chosen[1],
            risk=chosen[2],
            normalized_cost=chosen[3],
            utility=chosen[4],
            oracle_utility=observed[oracle_action][4],
            regret=observed[oracle_action][4] - chosen[4],
        ))
    return rows


def summarize_policy_rows(rows: Sequence[PolicyMetricRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty policy rows")
    by_user: dict[str, list[PolicyMetricRow]] = defaultdict(list)
    for row in rows:
        by_user[row.user_id].append(row)
    user_utility = np.asarray([np.mean([row.utility for row in values]) for values in by_user.values()])
    user_regret = np.asarray([np.mean([row.regret for row in values]) for values in by_user.values()])
    actions = [row.chosen_action for row in rows]
    return {
        "n_states": len(rows),
        "n_users": len(by_user),
        "mean_quality": float(np.mean([row.quality for row in rows])),
        "mean_emotional_support": float(np.mean([row.emotional_support for row in rows])),
        "mean_risk": float(np.mean([row.risk for row in rows])),
        "mean_normalized_cost": float(np.mean([row.normalized_cost for row in rows])),
        "mean_utility": float(np.mean(user_utility)),
        "utility_user_standard_error": float(np.std(user_utility, ddof=1) / math.sqrt(len(user_utility)) if len(user_utility) > 1 else 0.0),
        "mean_regret": float(np.mean(user_regret)),
        "distinct_actions": len(set(actions)),
        "maximum_action_share": float(max(actions.count(action) for action in set(actions)) / len(actions)),
        "rule_override_rate": float(np.mean([row.chosen_action != row.rule_action for row in rows])),
    }


def select_transparent_router_train_only(*, states: Sequence[PMV2State], labels: Sequence[ActionLabel], step0_by_state: Mapping[str, Step0Observation], composite_spec: CompositeSpec | None = None, risk_weight: float = 0.25, cost_weight: float = 0.10, configs: Sequence[TransparentRouterConfig] | None = None) -> tuple[TransparentRouterConfig, dict[str, Any]]:
    spec = composite_spec or CompositeSpec()
    label_by_key = _label_map(labels)
    reports = []
    for config in list(configs or router_grid()):
        router = StrongTransparentRouter(config)
        per_user: dict[str, list[float]] = defaultdict(list)
        for state in states:
            action = router.choose(step0_by_state[state.state_id]).action_id
            if action not in state.allowed_actions:
                raise RuntimeError("transparent router produced illegal action")
            all_costs = {}
            for candidate in state.allowed_actions:
                sources, strategy = parse_action_id(candidate)
                cost = sum(step0_by_state[state.state_id].sources[source].estimated_retrievable_tokens for source in sources)
                cost += 240.0 if strategy is StrategyMode.RS else 0.0
                cost += 24.0 * (len(sources) + int(strategy is StrategyMode.RS))
                all_costs[candidate] = cost
            label = label_by_key[(state.state_id, action)]
            quality = spec.score(label.response)
            risk = max(float(getattr(label.risk, head)) / 3.0 for head in applicable_risk_fields(action))
            normalized_cost = all_costs[action] / max(max(all_costs.values()), 1.0)
            per_user[state.user_id].append(quality - risk_weight * risk - cost_weight * normalized_cost)
        values = np.asarray([np.mean(rows) for rows in per_user.values()])
        reports.append({
            "config": config.model_dump(mode="json"),
            "config_sha256": config.digest(),
            "mean_user_utility": float(np.mean(values)),
            "user_standard_error": float(np.std(values, ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0.0),
            "maximum_sources": config.maximum_sources,
        })
    best = max(reports, key=lambda row: (row["mean_user_utility"], -row["maximum_sources"], row["config_sha256"]))
    best_lower = best["mean_user_utility"] - best["user_standard_error"]
    eligible = [row for row in reports if row["mean_user_utility"] >= best_lower]
    selected_report = min(eligible, key=lambda row: (row["maximum_sources"], row["config"]["source_selection_threshold"], row["config"]["strategy_selection_threshold"], row["config_sha256"]))
    selected = TransparentRouterConfig.model_validate(selected_report["config"])
    return selected, {
        "status": "COMPLETE",
        "protocol": "pm-v1.6-train-only-transparent-router-selection-v1",
        "selection_rule": "one_standard_error_then_simpler_maximum_sources",
        "selected_config": selected.model_dump(mode="json"),
        "selected_config_sha256": selected.digest(),
        "candidates": reports,
    }


def deterministic_user_folds(states: Sequence[PMV2State], *, folds: int = 6, seed: int = 1701) -> dict[str, int]:
    users = sorted({state.user_id for state in states})
    if folds < 2 or len(users) < folds:
        raise ValueError("not enough users for requested group CV folds")
    return {user: int(sha256_text(canonical_json([seed, user]))[:8], 16) % folds for user in users}


def factorial_action_diagnostics(*, states: Sequence[PMV2State], labels: Sequence[ActionLabel], composite_spec: CompositeSpec | None = None) -> dict[str, Any]:
    spec = composite_spec or CompositeSpec()
    label_by_key = _label_map(labels)
    order_energy: dict[int, list[float]] = defaultdict(list)
    top_margins, judge_mads = [], []
    for state in states:
        if len(state.allowed_actions) != 16:
            continue
        outcomes = {action: spec.score(label_by_key[(state.state_id, action)].response) for action in state.allowed_actions}
        actions = sorted(state.allowed_actions)
        factors = {}
        for action in actions:
            sources, strategy = parse_action_id(action)
            factors[action] = np.asarray([
                1.0 if MemorySource.MP in sources else -1.0,
                1.0 if MemorySource.MS in sources else -1.0,
                1.0 if MemorySource.ME in sources else -1.0,
                1.0 if strategy is StrategyMode.RS else -1.0,
            ])
        coefficients = {}
        for mask in range(16):
            indices = tuple(index for index in range(4) if mask & (1 << index))
            coefficients[indices] = float(np.mean([
                outcomes[action] * float(np.prod(factors[action][list(indices)])) if indices else outcomes[action]
                for action in actions
            ]))
        total_action_energy = sum(value * value for indices, value in coefficients.items() if indices)
        for order in range(1, 5):
            energy = sum(value * value for indices, value in coefficients.items() if len(indices) == order)
            order_energy[order].append(energy / max(total_action_energy, 1e-12))
        sorted_values = sorted(outcomes.values(), reverse=True)
        top_margins.append(sorted_values[0] - sorted_values[1])
        judge_mads.append(float(np.mean([label_by_key[(state.state_id, action)].max_dimension_mad for action in actions])))
    return {
        "status": "COMPLETE",
        "protocol": "pm-v1.6-train-only-factorial-diagnostic-v1",
        "n_complete_states": len(top_margins),
        "mean_action_variance_share_by_order": {str(order): float(np.mean(values)) if values else None for order, values in sorted(order_energy.items())},
        "mean_top1_top2_quality_margin": float(np.mean(top_margins)) if top_margins else None,
        "median_top1_top2_quality_margin": float(np.median(top_margins)) if top_margins else None,
        "mean_action_label_max_dimension_mad": float(np.mean(judge_mads)) if judge_mads else None,
    }


def train_only_algorithm_competition(*, states: Sequence[PMV2State], labels: Sequence[ActionLabel], step0_by_state: Mapping[str, Step0Observation], router_config: TransparentRouterConfig, override_config: OverrideConfig, composite_spec: CompositeSpec | None = None, folds: int = 6, seed: int = 1701, n_models: int = 7) -> dict[str, Any]:
    spec = composite_spec or CompositeSpec()
    fold_by_user = deterministic_user_folds(states, folds=folds, seed=seed)
    router = StrongTransparentRouter(router_config)
    candidate_rows: dict[AlgorithmCandidate, list[PolicyMetricRow]] = {candidate: [] for candidate in ALGORITHM_COMPLEXITY_ORDER}
    for fold in range(folds):
        train_states = [state for state in states if fold_by_user[state.user_id] != fold]
        held_states = [state for state in states if fold_by_user[state.user_id] == fold]
        if not train_states or not held_states:
            raise RuntimeError("deterministic group fold is empty")
        train_ids, held_ids = {state.state_id for state in train_states}, {state.state_id for state in held_states}
        train_labels = [label for label in labels if label.state_id in train_ids]
        held_labels = [label for label in labels if label.state_id in held_ids]
        train_step0 = {state_id: step0_by_state[state_id] for state_id in train_ids}
        held_step0 = {state_id: step0_by_state[state_id] for state_id in held_ids}
        train_rules = _rule_actions(train_states, train_step0, router)
        for index, candidate in enumerate(ALGORITHM_COMPLEXITY_ORDER):
            mode = OutcomeMode(candidate.value)
            model = PMV16OutcomeModel.train(
                mode=mode,
                states=train_states,
                labels=train_labels,
                step0_by_state=train_step0,
                composite_spec=spec,
                rule_action_by_state=train_rules if mode is OutcomeMode.RULE_RELATIVE else None,
                n_models=n_models,
                seed=seed + fold * 1000 + index * 100,
            )
            candidate_rows[candidate].extend(evaluate_observed_policy(
                model=model, router=router, override_config=override_config,
                states=held_states, labels=held_labels, step0_by_state=held_step0,
            ))
    summaries = {candidate.value: summarize_policy_rows(rows) for candidate, rows in candidate_rows.items()}
    best_candidate = max(ALGORITHM_COMPLEXITY_ORDER, key=lambda candidate: (summaries[candidate.value]["mean_utility"], -summaries[candidate.value]["mean_regret"]))
    best = summaries[best_candidate.value]
    lower = best["mean_utility"] - best["utility_user_standard_error"]
    eligible = [candidate for candidate in ALGORITHM_COMPLEXITY_ORDER if summaries[candidate.value]["mean_utility"] >= lower]
    selected = next(candidate for candidate in ALGORITHM_COMPLEXITY_ORDER if candidate in eligible)
    return {
        "status": "COMPLETE",
        "protocol": "pm-v1.6-train-only-algorithm-competition-v1",
        "folds": folds,
        "seed": seed,
        "selection_metric": "mean_user_utility",
        "one_standard_error_rule": True,
        "complexity_order": [value.value for value in ALGORITHM_COMPLEXITY_ORDER],
        "best_raw_candidate": best_candidate.value,
        "selected_primary": selected.value,
        "summaries": summaries,
        "factorial_diagnostics": factorial_action_diagnostics(states=states, labels=labels, composite_spec=spec),
        "router_config_sha256": router_config.digest(),
        "override_config_sha256": override_config.digest(),
    }


def calibrate_override_config(*, model: PMV16OutcomeModel, router_config: TransparentRouterConfig, states: Sequence[PMV2State], labels: Sequence[ActionLabel], step0_by_state: Mapping[str, Step0Observation], candidates: Sequence[OverrideConfig]) -> tuple[OverrideConfig, dict[str, Any]]:
    if not candidates:
        raise ValueError("override calibration candidate grid is empty")
    router = StrongTransparentRouter(router_config)
    reports = []
    for config in candidates:
        rows = evaluate_observed_policy(model=model, router=router, override_config=config, states=states, labels=labels, step0_by_state=step0_by_state)
        summary = summarize_policy_rows(rows)
        feasible = summary["mean_risk"] <= config.absolute_risk_ceiling
        reports.append({"config": config.model_dump(mode="json"), "config_sha256": config.digest(), "feasible": feasible, "summary": summary})
    feasible_rows = [row for row in reports if row["feasible"]]
    if not feasible_rows:
        raise RuntimeError("no override calibration candidate satisfies risk ceiling")
    selected_row = max(feasible_rows, key=lambda row: (row["summary"]["mean_utility"], -row["summary"]["mean_risk"], -row["summary"]["mean_normalized_cost"], -row["summary"]["rule_override_rate"], row["config_sha256"]))
    selected = OverrideConfig.model_validate(selected_row["config"])
    return selected, {
        "status": "COMPLETE",
        "protocol": "pm-v1.6-calibration-only-override-selection-v1",
        "selected_config": selected.model_dump(mode="json"),
        "selected_config_sha256": selected.digest(),
        "candidates": reports,
    }
