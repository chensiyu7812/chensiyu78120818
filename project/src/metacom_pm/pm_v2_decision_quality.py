"""Decision-quality diagnostics: did the PM call resources correctly?

Answers "how good was the chosen action" without assuming a single gold
action exists. For each state, all legal actions already have judged
outcomes (from the full action sweep + judging), so we can compute:

- regret: oracle utility minus the utility of the action PM actually chose;
- oracle-hit rate: whether the chosen action falls within epsilon of the
  best feasible action;
- source-level precision/recall: whether PM activated the memory sources an
  independent evaluator-context audit says were actually needed;
- M0/RS correctness: whether PM's abstention (M0) and strategy-retrieval
  (RS) decisions match the regime label;
- an item-utilization *proxy* built from continuous risk-dimension scores,
  since PM-v2's own judging does not produce discrete per-item M2-style
  use/omission judgments (only the legacy v1/EvoEmo pipeline does). Fields
  using this proxy are suffixed `_proxy` and must be reported as such, not
  as a substitute for real item-level audits.

Reuses `pm_v2_audit.py`'s existing `_quality`/`_risk` helpers for utility
computation rather than re-deriving the scoring rule.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

from .contracts import MemorySource, StrategyMode, canonical_action_id, parse_action_id
from .pm_v2_audit import _quality, _risk
from .pm_v2_contracts import ActionLabel, CompositeSpec


def compute_utility_by_action(
    labels_by_action: Mapping[str, ActionLabel],
    *,
    composite_spec: CompositeSpec,
    risk_weight: float,
    cost_weight: float,
    cost_by_action: Mapping[str, float],
) -> dict[str, float]:
    """utility(a) = quality(a) - risk_weight*risk(a) - cost_weight*cost(a)."""

    utility: dict[str, float] = {}
    for action_id, label in labels_by_action.items():
        quality = _quality(label, composite_spec)
        risk = _risk(label)
        cost = float(cost_by_action.get(action_id, 0.0))
        utility[action_id] = quality - float(risk_weight) * risk - float(cost_weight) * cost
    return utility


def compute_action_regret(
    utility_by_action: Mapping[str, float],
    chosen_action: str,
    *,
    feasible_actions: Sequence[str] | None = None,
) -> dict[str, float | str]:
    """Oracle utility (over feasible actions) minus the chosen action's utility."""

    if chosen_action not in utility_by_action:
        raise ValueError(f"chosen action {chosen_action!r} has no judged outcome")
    candidates = feasible_actions if feasible_actions is not None else list(utility_by_action)
    candidates = [a for a in candidates if a in utility_by_action]
    if not candidates:
        raise ValueError("no feasible action has a judged outcome")
    oracle_action = max(candidates, key=lambda a: (utility_by_action[a], a))
    oracle_utility = utility_by_action[oracle_action]
    chosen_utility = utility_by_action[chosen_action]
    return {
        "oracle_action": oracle_action,
        "oracle_utility": oracle_utility,
        "chosen_utility": chosen_utility,
        "regret": oracle_utility - chosen_utility,
    }


def compute_quality_by_action(
    labels_by_action: Mapping[str, ActionLabel], *, composite_spec: CompositeSpec
) -> dict[str, float]:
    return {
        action_id: _quality(label, composite_spec)
        for action_id, label in labels_by_action.items()
    }


def compute_cost_aware_oracle_action(
    quality_by_action: Mapping[str, float],
    cost_by_action: Mapping[str, float],
    *,
    epsilon: float,
    feasible_actions: Sequence[str] | None = None,
) -> dict[str, float | str]:
    """Cost-aware oracle: cheapest action within epsilon of the best quality.

    This is a distinct, complementary definition to `compute_action_regret`
    (which trades quality/risk/cost off through a single weighted utility).
    Here the acceptable set A_eps = {a : Q(a) >= max_a' Q(a') - eps} is
    defined on quality alone, and the oracle is the lowest-cost action within
    that set: a* = argmin_{a in A_eps} Cost(a). This avoids needing to commit
    to risk/cost weights to define "did PM call the right resource for the
    right cost" -- only the quality-acceptability margin eps is chosen.
    """

    candidates = (
        feasible_actions if feasible_actions is not None else list(quality_by_action)
    )
    candidates = [a for a in candidates if a in quality_by_action]
    if not candidates:
        raise ValueError("no feasible action has a judged outcome")
    best_quality = max(quality_by_action[a] for a in candidates)
    acceptable = [
        a for a in candidates if quality_by_action[a] >= best_quality - epsilon
    ]
    oracle_action = min(
        acceptable, key=lambda a: (cost_by_action.get(a, 0.0), a)
    )
    return {
        "acceptable_set": acceptable,
        "oracle_action": oracle_action,
        "oracle_cost": float(cost_by_action.get(oracle_action, 0.0)),
        "best_quality": best_quality,
    }


def compute_excess_cost(
    quality_by_action: Mapping[str, float],
    cost_by_action: Mapping[str, float],
    chosen_action: str,
    *,
    epsilon: float,
    feasible_actions: Sequence[str] | None = None,
) -> dict[str, float | str | bool]:
    """How much more PM's chosen action costs than the cheapest
    quality-acceptable action, and whether PM's own choice was itself
    quality-acceptable in the first place."""

    oracle = compute_cost_aware_oracle_action(
        quality_by_action,
        cost_by_action,
        epsilon=epsilon,
        feasible_actions=feasible_actions,
    )
    chosen_quality_acceptable = chosen_action in oracle["acceptable_set"]
    chosen_cost = float(cost_by_action.get(chosen_action, 0.0))
    return {
        "oracle_action": oracle["oracle_action"],
        "oracle_cost": oracle["oracle_cost"],
        "chosen_quality_acceptable": chosen_quality_acceptable,
        "chosen_cost": chosen_cost,
        "excess_cost": (
            chosen_cost - oracle["oracle_cost"]
            if chosen_quality_acceptable
            else None
        ),
    }


def compute_oracle_hit_rate(
    utility_by_action: Mapping[str, float],
    chosen_action: str,
    *,
    epsilon: float,
    feasible_actions: Sequence[str] | None = None,
) -> bool:
    """Whether the chosen action is within epsilon of the best feasible action."""

    regret_row = compute_action_regret(
        utility_by_action, chosen_action, feasible_actions=feasible_actions
    )
    return float(regret_row["regret"]) <= float(epsilon)


def compute_source_prf(
    chosen_action: str, needed_memory_sources: Sequence[str]
) -> dict[str, float]:
    """Precision/recall/F1 of PM's activated memory sources vs. the sources an
    independent evaluator-context audit says were actually needed."""

    chosen_sources, _ = parse_action_id(chosen_action)
    chosen = {s.value if isinstance(s, MemorySource) else str(s) for s in chosen_sources}
    needed = {s.value if isinstance(s, MemorySource) else str(s) for s in needed_memory_sources}
    if not chosen and not needed:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    true_positive = len(chosen & needed)
    precision = true_positive / len(chosen) if chosen else 1.0 if not needed else 0.0
    recall = true_positive / len(needed) if needed else 1.0 if not chosen else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_m0_correct(chosen_action: str, regime: str) -> bool | None:
    """Whether choosing no memory (M0) matches a regime where that is correct.

    Returns None when the regime does not have an unambiguous M0 answer
    (e.g. a regime where some memory subset is needed but M0 itself is not
    strictly wrong or right) -- callers should treat None as indeterminate,
    not as a failure.
    """

    chosen_sources, _ = parse_action_id(chosen_action)
    is_m0 = len(chosen_sources) == 0
    if regime in ("context_only", "memory_harmful"):
        return is_m0
    if regime in (
        "profile_useful",
        "profile_needed",
        "summary_useful",
        "summary_needed",
        "event_useful",
        "event_needed",
        "multi_source_useful",
        "multi_source_needed",
    ):
        return not is_m0
    return None


def compute_rs_correct(
    chosen_action: str,
    regime: str,
    utility_by_action: Mapping[str, float],
) -> bool | None:
    """Whether RS on/off matches the regime label, falling back to the paired
    R0-vs-RS utility comparison (same memory subset) when the regime alone is
    ambiguous."""

    chosen_sources, chosen_mode = parse_action_id(chosen_action)
    if regime == "strategy_helpful":
        return chosen_mode == StrategyMode.RS
    if regime == "strategy_harmful":
        return chosen_mode == StrategyMode.R0
    other_mode = StrategyMode.R0 if chosen_mode == StrategyMode.RS else StrategyMode.RS
    paired_action = canonical_action_id(chosen_sources, other_mode)
    if paired_action not in utility_by_action:
        return None
    chosen_utility = utility_by_action[chosen_action]
    paired_utility = utility_by_action[paired_action]
    if chosen_mode == StrategyMode.RS:
        return chosen_utility >= paired_utility
    return chosen_utility >= paired_utility


def compute_utilization_proxy(label: ActionLabel) -> dict[str, float]:
    """Continuous risk-dimension proxy for item utilization/misuse.

    Not a substitute for discrete M2-style item judgments (PM-v2's own
    judging does not produce those); field names are suffixed accordingly.
    """

    risk = label.risk
    return {
        "selected_context_misuse_proxy": float(risk.selected_context_misuse),
        "unnecessary_exposure_proxy": float(risk.unnecessary_exposure),
        "memory_omission_proxy": float(risk.memory_omission),
        "strategy_overuse_proxy": float(risk.strategy_overuse),
    }


def build_decision_quality_report(
    *,
    states: Sequence[str],
    labels_by_state: Mapping[str, Mapping[str, ActionLabel]],
    chosen_action_by_state: Mapping[str, str],
    cost_by_state_action: Mapping[str, Mapping[str, float]],
    regime_by_state: Mapping[str, str],
    needed_memory_sources_by_state: Mapping[str, Sequence[str]],
    user_id_by_state: Mapping[str, str],
    composite_spec: CompositeSpec,
    risk_weight: float,
    cost_weight: float,
    epsilon: float,
    quality_epsilon: float | None = None,
    resource_cost_by_state_action: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    """Aggregate per-state decision-quality rows into a report.

    Reports means only; callers needing confidence intervals should cluster
    the returned `rows` by `user_id` themselves (kept in each row) rather
    than treating states as independent samples, since multiple states can
    share a user.
    """

    rows = []
    for state_id in states:
        labels = labels_by_state[state_id]
        chosen = chosen_action_by_state[state_id]
        utility = compute_utility_by_action(
            labels,
            composite_spec=composite_spec,
            risk_weight=risk_weight,
            cost_weight=cost_weight,
            cost_by_action=cost_by_state_action[state_id],
        )
        regret_row = compute_action_regret(utility, chosen)
        hit = compute_oracle_hit_rate(utility, chosen, epsilon=epsilon)
        quality = compute_quality_by_action(labels, composite_spec=composite_spec)
        excess_cost = compute_excess_cost(
            quality,
            (
                resource_cost_by_state_action[state_id]
                if resource_cost_by_state_action is not None
                else cost_by_state_action[state_id]
            ),
            chosen,
            epsilon=(epsilon if quality_epsilon is None else quality_epsilon),
        )
        source_prf = compute_source_prf(
            chosen, needed_memory_sources_by_state.get(state_id, [])
        )
        regime = regime_by_state.get(state_id, "ambiguous")
        rows.append(
            {
                "state_id": state_id,
                "user_id": user_id_by_state.get(state_id),
                "chosen_action": chosen,
                "oracle_action": regret_row["oracle_action"],
                "regret": regret_row["regret"],
                "oracle_hit": hit,
                "cost_aware_oracle_action": excess_cost["oracle_action"],
                "chosen_quality_acceptable": excess_cost[
                    "chosen_quality_acceptable"
                ],
                "excess_cost_if_quality_acceptable": excess_cost["excess_cost"],
                "source_precision": source_prf["precision"],
                "source_recall": source_prf["recall"],
                "source_f1": source_prf["f1"],
                "m0_correct": compute_m0_correct(chosen, regime),
                "rs_correct": compute_rs_correct(chosen, regime, utility),
                **compute_utilization_proxy(labels[chosen]),
            }
        )

    def _mean(key: str, *, skip_none: bool = False) -> float | None:
        values = [r[key] for r in rows if not (skip_none and r[key] is None)]
        return sum(values) / len(values) if values else None

    return {
        "n_states": len(rows),
        "mean_regret": _mean("regret"),
        "oracle_hit_rate": _mean("oracle_hit"),
        "quality_acceptable_rate": _mean("chosen_quality_acceptable"),
        "mean_excess_cost_if_quality_acceptable": _mean(
            "excess_cost_if_quality_acceptable", skip_none=True
        ),
        "mean_source_precision": _mean("source_precision"),
        "mean_source_recall": _mean("source_recall"),
        "mean_source_f1": _mean("source_f1"),
        "m0_accuracy": _mean("m0_correct", skip_none=True),
        "rs_accuracy": _mean("rs_correct", skip_none=True),
        "mean_selected_context_misuse_proxy": _mean("selected_context_misuse_proxy"),
        "mean_unnecessary_exposure_proxy": _mean("unnecessary_exposure_proxy"),
        "mean_memory_omission_proxy": _mean("memory_omission_proxy"),
        "mean_strategy_overuse_proxy": _mean("strategy_overuse_proxy"),
        "rows": rows,
    }
