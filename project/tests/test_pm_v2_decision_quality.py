from __future__ import annotations

from metacom_pm.pm_v2_contracts import (
    ActionLabel,
    CompositeSpec,
    ResponseDimensions,
    RiskDimensions,
)
from metacom_pm.pm_v2_decision_quality import (
    build_decision_quality_report,
    compute_action_regret,
    compute_cost_aware_oracle_action,
    compute_excess_cost,
    compute_m0_correct,
    compute_oracle_hit_rate,
    compute_rs_correct,
    compute_source_prf,
    compute_utility_by_action,
    compute_utilization_proxy,
)

COMPOSITE = CompositeSpec()


def _dimension_mad() -> dict[str, float]:
    return {
        **{f"response.{name}": 0.0 for name in ResponseDimensions.model_fields},
        **{f"risk.{name}": 0.0 for name in RiskDimensions.model_fields},
    }


def _make_label(
    action_id: str,
    *,
    emotional_support: float = 3.0,
    risk: dict[str, float] | None = None,
    state_id: str = "s1",
) -> ActionLabel:
    risk = risk or {}
    return ActionLabel(
        state_id=state_id,
        card_id="c1",
        user_id="u1",
        semantic_family="family_a",
        action_id=action_id,
        response=ResponseDimensions(
            emotional_support=emotional_support,
            personalization=3.0,
            memory_appropriateness=3.0,
            factual_grounding=3.0,
            temporal_consistency=3.0,
            non_intrusiveness=3.0,
        ),
        risk=RiskDimensions(
            selected_context_misuse=risk.get("selected_context_misuse", 0.0),
            unnecessary_exposure=risk.get("unnecessary_exposure", 0.0),
            stale_or_conflicting_use=risk.get("stale_or_conflicting_use", 0.0),
            unsupported_personal_claim=risk.get("unsupported_personal_claim", 0.0),
            memory_omission=risk.get("memory_omission", 0.0),
            strategy_overuse=risk.get("strategy_overuse", 0.0),
            strategy_omission=risk.get("strategy_omission", 0.0),
        ),
        observed_input_tokens=100,
        retrieval_calls=0,
        judge_families=["a", "b"],
        judge_count=2,
        max_dimension_mad=0.0,
        dimension_mad=_dimension_mad(),
        label_reliable=True,
        composite_weights_sha256="0" * 64,
    )


def test_compute_utility_by_action_penalizes_risk_and_cost():
    labels = {
        "M0+R0": _make_label("M0+R0", emotional_support=3.0),
        "ME+R0": _make_label(
            "ME+R0", emotional_support=3.0, risk={"selected_context_misuse": 3.0}
        ),
    }
    utility = compute_utility_by_action(
        labels,
        composite_spec=COMPOSITE,
        risk_weight=0.5,
        cost_weight=0.1,
        cost_by_action={"M0+R0": 0.0, "ME+R0": 1.0},
    )
    # Same quality, but ME+R0 has strictly higher risk and cost, so it must
    # score strictly lower.
    assert utility["ME+R0"] < utility["M0+R0"]


def test_regret_and_oracle_hit_rate():
    utility = {"M0+R0": 0.5, "ME+R0": 0.8, "MP+RS": 0.7}
    regret_row = compute_action_regret(utility, "M0+R0")
    assert regret_row["oracle_action"] == "ME+R0"
    assert abs(regret_row["regret"] - 0.3) < 1e-9

    assert compute_oracle_hit_rate(utility, "MP+RS", epsilon=0.15) is True
    assert compute_oracle_hit_rate(utility, "M0+R0", epsilon=0.15) is False


def test_regret_respects_feasible_action_restriction():
    utility = {"M0+R0": 0.5, "ME+R0": 0.9}
    # If ME+R0 is infeasible (e.g. inventory does not have that source),
    # the oracle must be computed only over feasible actions.
    row = compute_action_regret(utility, "M0+R0", feasible_actions=["M0+R0"])
    assert row["oracle_action"] == "M0+R0"
    assert row["regret"] == 0.0


def test_cost_aware_oracle_picks_cheapest_within_quality_margin():
    # ME+R0 and MP+R0 are both within epsilon of the best quality
    # (MPE+R0, at the 0.85 boundary inclusive), but ME+R0 is cheaper -> it
    # should be the cost-aware oracle, not the highest-quality action.
    quality = {"MPE+R0": 0.90, "ME+R0": 0.88, "MP+R0": 0.85, "M0+R0": 0.40}
    cost = {"MPE+R0": 2.0, "ME+R0": 0.5, "MP+R0": 1.0, "M0+R0": 0.0}
    oracle = compute_cost_aware_oracle_action(quality, cost, epsilon=0.05)
    assert set(oracle["acceptable_set"]) == {"MPE+R0", "ME+R0", "MP+R0"}
    assert oracle["oracle_action"] == "ME+R0"
    assert oracle["oracle_cost"] == 0.5


def test_excess_cost_when_chosen_is_quality_acceptable():
    quality = {"MPE+R0": 0.90, "ME+R0": 0.88, "M0+R0": 0.40}
    cost = {"MPE+R0": 2.0, "ME+R0": 0.5, "M0+R0": 0.0}
    result = compute_excess_cost(quality, cost, "MPE+R0", epsilon=0.05)
    assert result["chosen_quality_acceptable"] is True
    assert result["oracle_action"] == "ME+R0"
    assert abs(result["excess_cost"] - 1.5) < 1e-9


def test_excess_cost_when_chosen_is_not_quality_acceptable():
    quality = {"MPE+R0": 0.90, "M0+R0": 0.40}
    cost = {"MPE+R0": 2.0, "M0+R0": 0.0}
    result = compute_excess_cost(quality, cost, "M0+R0", epsilon=0.05)
    assert result["chosen_quality_acceptable"] is False
    assert result["excess_cost"] is None


def test_source_prf_exact_match_and_partial():
    perfect = compute_source_prf("MPE+R0", ["MP", "ME"])
    assert perfect == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    over_selecting = compute_source_prf("MPMSME+R0", ["MP"])
    assert over_selecting["recall"] == 1.0
    assert abs(over_selecting["precision"] - 1 / 3) < 1e-9

    under_selecting = compute_source_prf("MP+R0", ["MP", "ME"])
    assert under_selecting["precision"] == 1.0
    assert under_selecting["recall"] == 0.5

    both_empty = compute_source_prf("M0+R0", [])
    assert both_empty == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_m0_correct_by_regime():
    assert compute_m0_correct("M0+R0", "context_only") is True
    assert compute_m0_correct("MP+R0", "context_only") is False
    assert compute_m0_correct("M0+R0", "event_useful") is False
    assert compute_m0_correct("ME+R0", "event_useful") is True
    assert compute_m0_correct("ME+R0", "event_needed") is True
    # Regimes without an unambiguous M0 answer are indeterminate, not wrong.
    assert compute_m0_correct("M0+R0", "ambiguous") is None


def test_rs_correct_uses_regime_label_when_available():
    utility = {"MP+R0": 0.5, "MP+RS": 0.5}
    assert compute_rs_correct("MP+RS", "strategy_helpful", utility) is True
    assert compute_rs_correct("MP+R0", "strategy_helpful", utility) is False
    assert compute_rs_correct("MP+RS", "strategy_harmful", utility) is False


def test_rs_correct_falls_back_to_paired_utility_when_regime_ambiguous():
    # RS strictly worse than the same-memory R0 counterfactual -> RS choice
    # should be judged incorrect even though the regime label itself is
    # ambiguous about strategy use.
    utility = {"MP+R0": 0.8, "MP+RS": 0.5}
    assert compute_rs_correct("MP+RS", "ambiguous", utility) is False
    assert compute_rs_correct("MP+R0", "ambiguous", utility) is True


def test_rs_correct_indeterminate_when_paired_action_missing():
    utility = {"MP+RS": 0.5}
    assert compute_rs_correct("MP+RS", "ambiguous", utility) is None


def test_utilization_proxy_extracts_risk_fields():
    label = _make_label(
        "MP+R0",
        risk={"selected_context_misuse": 1.0, "strategy_overuse": 2.0},
    )
    proxy = compute_utilization_proxy(label)
    assert proxy["selected_context_misuse_proxy"] == 1.0
    assert proxy["strategy_overuse_proxy"] == 2.0


def test_build_decision_quality_report_end_to_end():
    labels_by_state = {
        "s1": {
            "M0+R0": _make_label("M0+R0", emotional_support=2.0, state_id="s1"),
            "ME+R0": _make_label("ME+R0", emotional_support=4.0, state_id="s1"),
        },
        "s2": {
            "M0+R0": _make_label("M0+R0", emotional_support=4.0, state_id="s2"),
            "MP+RS": _make_label("MP+RS", emotional_support=3.0, state_id="s2"),
        },
    }
    report = build_decision_quality_report(
        states=["s1", "s2"],
        labels_by_state=labels_by_state,
        chosen_action_by_state={"s1": "M0+R0", "s2": "M0+R0"},
        cost_by_state_action={
            "s1": {"M0+R0": 0.0, "ME+R0": 0.0},
            "s2": {"M0+R0": 0.0, "MP+RS": 0.0},
        },
        regime_by_state={"s1": "event_useful", "s2": "context_only"},
        needed_memory_sources_by_state={"s1": ["ME"], "s2": []},
        user_id_by_state={"s1": "user_a", "s2": "user_b"},
        composite_spec=COMPOSITE,
        risk_weight=0.5,
        cost_weight=0.1,
        epsilon=0.0,
    )
    assert report["n_states"] == 2
    # s1: PM chose M0+R0 but ME+R0 is strictly better -> positive regret,
    # oracle miss, m0 incorrect (regime says event memory is needed).
    # s2: PM chose M0+R0 which is also the oracle for that state -> zero
    # regret, oracle hit, m0 correct (context_only).
    s1_row = next(r for r in report["rows"] if r["state_id"] == "s1")
    s2_row = next(r for r in report["rows"] if r["state_id"] == "s2")
    assert s1_row["regret"] > 0
    assert s1_row["oracle_hit"] is False
    assert s1_row["m0_correct"] is False
    assert s2_row["regret"] == 0
    assert s2_row["oracle_hit"] is True
    assert s2_row["m0_correct"] is True
    assert report["oracle_hit_rate"] == 0.5
