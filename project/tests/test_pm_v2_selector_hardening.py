from __future__ import annotations

import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from metacom_pm.contracts import (
    DialogueTurn,
    MemorySource,
    StrategyMode,
    canonical_action_id,
)
from metacom_pm.config import load_config
from metacom_pm.pm_v2_audit import (
    audit_deployable_feature_observability,
    audit_pilot_label_value_feasibility,
    audit_training_labels,
)
from metacom_pm.pm_v2_contracts import (
    ActionLabel,
    ActionPrediction,
    CompositeSpec,
    ObservableSourceSummary,
    PMV2Split,
    PMV2State,
    PredictionInterval,
    ResponseDimensions,
    RiskDimensions,
)
from metacom_pm.pm_v2_data import EvaluatorContextIndex
from metacom_pm.pm_v2_features import PMV2FeatureBuilder
from metacom_pm.pm_v2_model import (
    NO_FEASIBLE_FALLBACK_REASON,
    PMV2Model,
    RESPONSE_FIELDS,
    RISK_FIELDS,
    SelectionConfig,
    calibrate_uncertainty_multiplier,
    composite_weights_digest,
    cost_calibration_diagnostics,
    decision_fallback_kind,
    evaluate_policy,
    evaluate_prediction_coverage,
    tune_selection_config,
)
from metacom_pm.pm_v1_5_algorithm_selection import (
    select_routing_algorithm_group_cv,
)
from metacom_pm.pm_v1_5_rule_router import FixedActionBaselineRouter
from metacom_pm.pm_v2_fixed_model import FixedActionPMV2Model


def make_state(
    state_id: str,
    *,
    user_id: str | None = None,
    regime: str = "ambiguous",
) -> PMV2State:
    mp_available = regime in {"profile_needed", "multi_source_needed"}
    ms_available = regime == "summary_needed"
    inventory = {
        MemorySource.MP: ObservableSourceSummary(
            available=mp_available,
            count=2 if mp_available else 0,
            estimated_tokens=80 if mp_available else 0,
        ),
        MemorySource.MS: ObservableSourceSummary(
            available=ms_available,
            count=2 if ms_available else 0,
            estimated_tokens=90 if ms_available else 0,
        ),
        MemorySource.ME: ObservableSourceSummary(
            available=True,
            count=2,
            min_age_sessions=1,
            median_age_sessions=2.0,
            max_age_sessions=3,
            estimated_tokens=100,
        ),
    }
    available_sources = [
        source for source in MemorySource if inventory[source].available
    ]
    allowed_actions = []
    for subset_bits in range(1 << len(available_sources)):
        subset = frozenset(
            source
            for index, source in enumerate(available_sources)
            if subset_bits & (1 << index)
        )
        for strategy in (StrategyMode.R0, StrategyMode.RS):
            allowed_actions.append(canonical_action_id(subset, strategy))
    return PMV2State(
        state_id=state_id,
        card_id=f"card_{state_id}",
        user_id=user_id or f"user_{state_id}",
        split=PMV2Split.TRAIN,
        semantic_family=f"family_{state_id}",
        surface_form_id=f"surface_{state_id}",
        current_user_text="I am having a difficult day.",
        current_session_history=[DialogueTurn(role="user", content="I feel tense.")],
        current_session_summary="The user describes current stress.",
        session_index=4,
        inventory=inventory,
        strategy_catalog_count=10,
        strategy_estimated_tokens=120,
        allowed_actions=allowed_actions,
        provenance={"evaluator_context_id": f"eval_{state_id}"},
    )


def test_fixed_action_model_does_not_read_step0_or_prediction_heads() -> None:
    state = make_state("fixed_no_step0")
    state = state.model_copy(update={"step0_observation": None})
    model = FixedActionPMV2Model(
        feature_builder=object(),
        response_heads={},
        risk_heads={},
        selection_config=SelectionConfig(),
        fixed_action="M0+R0",
    )

    decision = model.choose(state)

    assert decision.chosen_action == "M0+R0"
    assert decision.predictions == {}
    assert decision.semantic_ood_score == 0.0
    assert decision.metadata_ood_score == 0.0


def test_ood_thresholds_are_calibrated_on_holdout_and_challenge_gated():
    train = [make_state(f"ood_train_{index}") for index in range(5)]
    calibration = [make_state(f"ood_cal_{index}") for index in range(5)]
    builder = PMV2FeatureBuilder(use_precomputed_embeddings=False).fit(train)
    report = builder.calibrate_ood(
        calibration,
        semantic_false_positive_quantile=0.99,
        metadata_false_positive_quantile=0.99,
        maximum_joint_in_distribution_fallback_rate=0.05,
        minimum_semantic_challenge_detection_rate=0.80,
        minimum_metadata_challenge_detection_rate=0.95,
    )
    assert report["status"] == "PASS"
    assert report["joint_in_distribution_fallback_rate"] == 0.0
    assert builder.ood_report(calibration[0])["threshold_source"] == (
        "calibration_split"
    )


def make_evaluator_contexts(states, regimes) -> EvaluatorContextIndex:
    needed_sources_by_regime = {
        "context_only": [],
        "profile_needed": ["MP"],
        "summary_needed": ["MS"],
        "event_needed": ["ME"],
        "multi_source_needed": ["MP", "ME"],
        "memory_harmful": [],
        "strategy_helpful": [],
        "strategy_harmful": [],
        "ambiguous": [],
    }
    by_state = {
        state.state_id: {
            "state_id": state.state_id,
            "card_id": state.card_id,
            "evaluator_context_id": state.provenance["evaluator_context_id"],
            "regime": regimes[state.state_id],
            "needed_memory_sources": needed_sources_by_regime[
                regimes[state.state_id]
            ],
            "memory_annotations": [],
        }
        for state in states
    }
    return EvaluatorContextIndex(
        by_state=by_state,
        by_card={row["card_id"]: row for row in by_state.values()},
        source_sha256="a" * 64,
        map_sha256="b" * 64,
    )


def make_response(value: float = 3.0, **updates: float) -> ResponseDimensions:
    values = {name: value for name in RESPONSE_FIELDS}
    values.update(updates)
    return ResponseDimensions(**values)


def make_risk(value: float = 0.0, **updates: float) -> RiskDimensions:
    values = {name: value for name in RISK_FIELDS}
    values.update(updates)
    return RiskDimensions(**values)


def make_label(
    state: PMV2State,
    action_id: str,
    *,
    reliable: bool = True,
    response: ResponseDimensions | None = None,
    risk: RiskDimensions | None = None,
    cost: int = 100,
    composite_weights_sha256: str = (
        "854996e300bc0506a71679f12a179e1a0"
        "ae33b050b62eb4d77d7c8d222fcf0b2"
    ),
) -> ActionLabel:
    dimension_mad = {
        **{f"response.{name}": 0.0 for name in RESPONSE_FIELDS},
        **{f"risk.{name}": 0.0 for name in RISK_FIELDS},
    }
    if not reliable:
        dimension_mad["response.emotional_support"] = 1.0
    return ActionLabel(
        state_id=state.state_id,
        card_id=state.card_id,
        user_id=state.user_id,
        semantic_family=state.semantic_family,
        action_id=action_id,
        response=response or make_response(),
        risk=risk or make_risk(),
        observed_input_tokens=cost,
        retrieval_calls=0,
        judge_families=["judge_a", "judge_b"],
        judge_count=2,
        max_dimension_mad=max(dimension_mad.values()),
        dimension_mad=dimension_mad,
        label_reliable=reliable,
        composite_weights_sha256=composite_weights_sha256,
    )


def interval(
    mean: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> PredictionInterval:
    return PredictionInterval(
        mean=mean,
        std=0.0,
        lower=mean if lower is None else lower,
        upper=mean if upper is None else upper,
    )


def action_prediction(
    action_id: str,
    *,
    utility: float,
    feasible: bool = True,
    quality: float = 0.75,
    estimated_cost: float | None = None,
    normalized_cost: float | None = None,
) -> ActionPrediction:
    if estimated_cost is None:
        estimated_cost = 0.0 if action_id == "M0+R0" else 100.0
    if normalized_cost is None:
        normalized_cost = 0.0 if action_id == "M0+R0" else 1.0
    return ActionPrediction(
        action_id=action_id,
        response={name: interval(1.0 + 4.0 * quality) for name in RESPONSE_FIELDS},
        risk={name: interval(0.0) for name in RISK_FIELDS},
        quality_mean=quality,
        quality_lcb=quality,
        risk_ucb=0.0,
        estimated_cost=estimated_cost,
        normalized_cost=normalized_cost,
        utility=utility,
        feasible=feasible,
        resource_gate_passed=True,
        strategy_gate_passed=True,
    )


class RoutingBuilder:
    def ood_report(self, state: PMV2State):
        severe = bool(state.provenance.get("severe_ood"))
        return {
            "semantic_ood_score": 0.9 if severe else 0.0,
            "metadata_ood_score": 0.0,
            "severe_semantic_ood": severe,
            "severe_metadata_ood": False,
        }

    def estimate_action_cost(self, state: PMV2State, action_id: str) -> float:
        return float({"M0+R0": 0, "M0+RS": 120, "ME+R0": 100, "ME+RS": 220}[action_id])

    def transform(self, rows):
        return np.zeros((len(rows), 1), dtype=float)


class ConstantHead:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_members(self, x):
        # Identical members deliberately produce zero bootstrap std.
        return np.full((2, x.shape[0]), self.value, dtype=float)


class OrderedHead:
    def predict_members(self, x):
        values = np.arange(x.shape[0], dtype=float)
        return np.vstack([values, values])


def fake_routing_model(predictions_by_state) -> PMV2Model:
    model = PMV2Model(
        feature_builder=RoutingBuilder(),
        response_heads={},
        risk_heads={},
        selection_config=SelectionConfig(),
        response_conformal_radii={name: 0.0 for name in RESPONSE_FIELDS},
        risk_conformal_radii={name: 0.0 for name in RISK_FIELDS},
        quality_conformal_radius=0.0,
        conformal_calibration_report={"status": "COMPLETE"},
    )

    def fake_predict(self, state):
        return predictions_by_state[state.state_id]

    model.predict_actions = MethodType(fake_predict, model)
    return model


def make_bundle(quality_members, *, risk_upper=None, broad_intervals=False):
    risk_upper = risk_upper or {}
    response = {}
    risk = {}
    for action_id, members in quality_members.items():
        response[action_id] = {
            name: (
                interval(3.0, lower=1.0, upper=5.0)
                if broad_intervals
                else interval(1.0 + 4.0 * float(np.mean(members)))
            )
            for name in RESPONSE_FIELDS
        }
        risk[action_id] = {
            name: interval(
                float(risk_upper.get((action_id, name), 0.0)),
                lower=0.0,
                upper=(
                    3.0
                    if broad_intervals
                    else float(risk_upper.get((action_id, name), 0.0))
                ),
            )
            for name in RISK_FIELDS
        }
    return SimpleNamespace(
        response=response,
        risk=risk,
        quality_members={
            action_id: np.asarray(members, dtype=float)
            for action_id, members in quality_members.items()
        },
    )


def model_with_bundle(bundle, *, config: SelectionConfig) -> PMV2Model:
    model = PMV2Model(
        feature_builder=RoutingBuilder(),
        response_heads={},
        risk_heads={},
        selection_config=config,
    )

    def fake_bundle(self, state):
        return bundle

    model._prediction_bundle = MethodType(fake_bundle, model)
    return model


def test_training_requires_every_input_state_to_have_complete_aggregated_matrix():
    first = make_state("first")
    second = make_state("second")
    labels = [make_label(first, action) for action in first.allowed_actions]
    with pytest.raises(ValueError, match="complete set of aggregated labels"):
        PMV2Model.train([first, second], labels, n_models=1)


def test_dimension_mad_downweights_only_corresponding_head_without_row_deletion():
    states = [make_state(f"unreliable_{index}") for index in range(3)]
    labels = [
        make_label(state, action, reliable=action != "ME+RS")
        for state in states
        for action in state.allowed_actions
    ]
    model = PMV2Model.train(
        states,
        labels,
        n_models=1,
        word_features=8,
        char_features=8,
        use_precomputed_embeddings=False,
    )
    report = model.training_report
    assert report["n_action_labels"] == len(labels)
    assert report["n_unreliable_labels_included"] == 3
    assert not report["row_reliability_used_for_filtering"]
    weight_report = report["dimension_mad_weighting"]["response_heads"]
    assert weight_report["emotional_support"]["minimum"] < weight_report[
        "personalization"
    ]["minimum"]


def test_default_bootstrap_group_is_user_id_not_state_id():
    states = [
        make_state(f"u{user}_s{state}", user_id=f"user_{user}")
        for user in range(3)
        for state in range(2)
    ]
    labels = [
        make_label(state, action)
        for state in states
        for action in state.allowed_actions
    ]
    model = PMV2Model.train(
        states,
        labels,
        n_models=1,
        word_features=8,
        char_features=8,
        use_precomputed_embeddings=False,
    )
    assert model.training_report["bootstrap_group_key"] == "user_id"
    assert model.training_report["bootstrap_unique_groups"] == 3
    assert model.training_report["n_states"] == 6
    cost_contract = model.training_report["estimated_resource_cost_contract"]
    assert cost_contract["memory_top_k"] == {"MP": 2, "MS": 2, "ME": 3}
    assert cost_contract["retrieval_call_penalty"] == 24.0


def test_paired_delta_objective_is_fitted_over_complete_paired_states():
    states = [make_state(f"delta_{index}") for index in range(3)]
    labels = [
        make_label(
            state,
            action,
            response=make_response(
                2.0 + 0.5 * state.allowed_actions.index(action)
            ),
        )
        for state in states
        for action in state.allowed_actions
    ]
    model = PMV2Model.train(
        states,
        labels,
        n_models=1,
        word_features=8,
        char_features=8,
        use_precomputed_embeddings=False,
    )
    model.fit_routing_objective(
        states,
        labels,
        algorithm="state_centered_paired_delta_hgb",
        n_models=1,
        seed=17,
    )
    assert model.routing_objective_head is not None
    assert model.routing_objective_report["target"] == (
        "state_centered_realized_utility"
    )
    assert model.routing_objective_report["n_rows"] == sum(
        len(state.allowed_actions) for state in states
    )
    assert model.choose(states[0]).chosen_action in states[0].allowed_actions


def test_rule_residual_overrides_only_when_all_safe_delta_checks_pass():
    state = make_state("safe_residual")
    predictions = {
        action: action_prediction(action, utility=0.5)
        for action in state.allowed_actions
    }
    model = fake_routing_model({state.state_id: predictions})
    model.routing_algorithm = "rule_relative_safe_residual_hgb"
    model.routing_objective_head = OrderedHead()
    model.routing_rule_router = FixedActionBaselineRouter()
    model.routing_safe_thresholds = {
        "minimum_quality_delta_lcb": -0.02,
        "minimum_emotional_support_delta_lcb": -0.025,
        "maximum_risk_delta_ucb": 0.02,
        "minimum_utility_delta_lcb": 0.0,
    }
    assert model.choose(state).chosen_action == "ME+RS"
    model.routing_safe_thresholds["minimum_utility_delta_lcb"] = 10.0
    assert model.choose(state).chosen_action == "M0+R0"


def test_algorithm_family_selection_is_train_user_group_disjoint():
    states = [make_state(f"cv_{index}") for index in range(6)]
    labels = [
        make_label(
            state,
            action,
            response=make_response(
                2.0 + 0.4 * state.allowed_actions.index(action)
            ),
        )
        for state in states
        for action in state.allowed_actions
    ]
    selected, report = select_routing_algorithm_group_cv(
        states=states,
        labels=labels,
        selection_config=SelectionConfig(),
        candidates=[
            "state_centered_paired_delta_hgb",
            "absolute_outcome_factorized_hgb",
        ],
        folds=3,
        n_models=1,
        seed=17,
        dimension_mad_scale=0.75,
        bootstrap_group_key="user_id",
        use_precomputed_embeddings=False,
        word_features=8,
        char_features=8,
        rule_grid={},
        rule_minimum_quality=0.0,
        rule_maximum_risk=1.0,
        minimum_validation_quality=0.0,
        maximum_validation_risk=1.0,
        safe_residual_thresholds={},
        simplicity_order=[
            "absolute_outcome_factorized_hgb",
            "state_centered_paired_delta_hgb",
        ],
    )
    assert selected in {
        "state_centered_paired_delta_hgb",
        "absolute_outcome_factorized_hgb",
    }
    assert report["selection_data_role"] == "train_only"
    for fold in report["fold_assignments"]:
        assert set(fold["fit_users"]).isdisjoint(fold["validation_users"])


def test_training_rejects_composite_weight_hash_mismatch():
    state = make_state("bad_hash")
    labels = [make_label(state, action) for action in state.allowed_actions]
    labels[-1].composite_weights_sha256 = "c" * 64
    with pytest.raises(ValueError, match="composite weights hash"):
        PMV2Model.train([state], labels, n_models=1)


def test_versions_are_bumped_and_old_checkpoint_is_rejected(tmp_path):
    model = PMV2Model(
        feature_builder=RoutingBuilder(),
        response_heads={},
        risk_heads={},
        selection_config=SelectionConfig(),
        response_conformal_radii={name: 0.0 for name in RESPONSE_FIELDS},
        risk_conformal_radii={name: 0.0 for name in RISK_FIELDS},
        quality_conformal_radius=0.0,
        conformal_calibration_report={"status": "COMPLETE"},
    )
    assert model.selection_config.version == "pmv2-selection-v2"
    assert model.format_version == "pm-v2.1"
    checkpoint = tmp_path / "model.joblib"
    model.save(checkpoint)
    assert PMV2Model.load(checkpoint).format_version == "pm-v2.1"
    model.format_version = "pm-v2.0"
    model.save(checkpoint)
    with pytest.raises(RuntimeError, match="unsupported PM-v2 format"):
        PMV2Model.load(checkpoint)


def test_evaluate_policy_fails_when_selected_label_is_missing():
    state = make_state("missing")
    predictions = {
        action: action_prediction(action, utility=1.0 if action == "M0+R0" else 0.0)
        for action in state.allowed_actions
    }
    model = fake_routing_model({state.state_id: predictions})
    labels = [make_label(state, "ME+R0")]
    with pytest.raises(ValueError, match="missing labels for selected actions"):
        evaluate_policy(model, [state], labels)


def test_evaluate_policy_realized_utility_uses_one_risk_and_estimated_cost_term():
    state = make_state("utility_formula")
    predictions = {
        action: action_prediction(
            action,
            utility=1.0 if action == "ME+R0" else 0.0,
            normalized_cost=0.5 if action == "ME+R0" else 0.0,
        )
        for action in state.allowed_actions
    }
    model = fake_routing_model({state.state_id: predictions})
    model.selection_config = SelectionConfig(
        risk_weight=0.30,
        cost_weight=0.20,
    )
    label = make_label(
        state,
        "ME+R0",
        response=make_response(3.0),
        risk=make_risk(0.0, unsupported_personal_claim=1.5),
    )
    metrics = evaluate_policy(model, [state], [label])
    # quality=.5, applicable max risk=.5, normalized estimated cost=.5
    assert metrics["mean_realized_utility"] == pytest.approx(
        0.5 - 0.30 * 0.5 - 0.20 * 0.5
    )


def test_fallback_classes_do_not_count_as_learned_m0_r0():
    learned = make_state("learned")
    no_feasible = make_state("no_feasible")
    severe = make_state("severe")
    severe.provenance["severe_ood"] = True
    predictions_by_state = {
        learned.state_id: {
            action: action_prediction(
                action, utility=1.0 if action == "M0+R0" else 0.0
            )
            for action in learned.allowed_actions
        },
        no_feasible.state_id: {
            action: action_prediction(action, utility=0.0, feasible=False)
            for action in no_feasible.allowed_actions
        },
        severe.state_id: {
            action: action_prediction(action, utility=1.0)
            for action in severe.allowed_actions
        },
    }
    model = fake_routing_model(predictions_by_state)
    no_feasible_decision = model.choose(no_feasible)
    assert no_feasible_decision.decision_reason == NO_FEASIBLE_FALLBACK_REASON
    assert not no_feasible_decision.ood_fallback_used
    assert decision_fallback_kind(no_feasible_decision) == "no_feasible"
    assert decision_fallback_kind(model.choose(severe)) == "severe_ood"

    labels = [make_label(state, "M0+R0") for state in (learned, no_feasible, severe)]
    metrics = evaluate_policy(model, [learned, no_feasible, severe], labels)
    assert metrics["n"] == metrics["expected_n"] == 3
    assert metrics["m0_rate"] == 1.0
    assert metrics["learned_m0_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["nonfallback_m0_r0_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["no_feasible_fallback_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["severe_ood_fallback_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["learned_action_distribution"] == {"M0+R0": 1}
    assert set(metrics["mean_response_dimensions"]) == set(RESPONSE_FIELDS)
    assert {row["user_id"] for row in metrics["rows"]} == {
        learned.user_id,
        no_feasible.user_id,
        severe.user_id,
    }
    assert "mean_normalized_estimated_resource_cost" in metrics
    assert "estimated_vs_observed_selected_cost" in metrics


def test_memory_gate_uses_same_strategy_baseline():
    state = make_state("same_strategy")
    bundle = make_bundle(
        {
            "M0+R0": [0.20, 0.20],
            "M0+RS": [0.80, 0.80],
            "ME+R0": [0.70, 0.70],
            "ME+RS": [0.70, 0.70],
        }
    )
    config = SelectionConfig(
        max_risk_ucb=1.0,
        resource_min_gain=0.0,
        strategy_min_gain=-0.01,
        m0_omission_trigger=1.0,
        r0_strategy_omission_trigger=1.0,
    )
    predictions = model_with_bundle(bundle, config=config).predict_actions(state)
    assert predictions["ME+R0"].resource_gate_passed
    assert not predictions["ME+RS"].resource_gate_passed


def test_policy_regime_gate_reports_missing_regime_as_zero():
    script = Path(__file__).resolve().parents[1] / "scripts" / "22_train_pm_v2.py"
    spec = importlib.util.spec_from_file_location("train_pm_v2_regimes", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = make_state("only_context", regime="context_only")
    predictions = {
        action: action_prediction(
            action,
            utility=1.0 if action == "M0+R0" else 0.0,
        )
        for action in state.allowed_actions
    }
    model = fake_routing_model({state.state_id: predictions})
    contexts = make_evaluator_contexts(
        [state], {state.state_id: "context_only"}
    )
    report = module.policy_regime_alignment(model, [state], contexts)
    assert report["regime_pass_rate"]["context_only"] == 1.0
    assert report["mean_regime_pass_rate"] == 1.0
    assert report["minimum_regime_pass_rate"] == 0.0
    assert report["minimum_regime_wilson_lower_bound"] == 0.0
    assert "memory_harmful" in report["missing_regimes"]


def test_regime_alignment_uses_user_block_wilson_evidence():
    script = Path(__file__).resolve().parents[1] / "scripts" / "22_train_pm_v2.py"
    spec = importlib.util.spec_from_file_location("train_pm_v2_wilson", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    states = [
        make_state(
            f"context_{index}",
            user_id=f"internal_user_{index}",
            regime="context_only",
        )
        for index in range(16)
    ]
    predictions_by_state = {}
    for index, state in enumerate(states):
        selected = "M0+R0" if index < 12 else "ME+RS"
        predictions_by_state[state.state_id] = {
            action: action_prediction(
                action,
                utility=1.0 if action == selected else 0.0,
            )
            for action in state.allowed_actions
        }
    model = fake_routing_model(predictions_by_state)
    contexts = make_evaluator_contexts(
        states, {state.state_id: "context_only" for state in states}
    )
    report = module.policy_regime_alignment(
        model, states, contexts, confidence_level=0.95
    )
    assert report["regime_pass_rate"]["context_only"] == 0.75
    assert report["regime_user_block_counts"]["context_only"] == 16
    assert report["regime_wilson_lower_bound"]["context_only"] > 0.50


def test_paired_bootstrap_delta_preserves_within_member_covariance():
    state = make_state("paired")
    bundle = make_bundle(
        {
            "M0+R0": [0.30, 0.50],
            "M0+RS": [0.30, 0.50],
            "ME+R0": [0.40, 0.60],
            "ME+RS": [0.40, 0.60],
        }
    )
    config = SelectionConfig(
        uncertainty_z=1.0,
        max_risk_ucb=1.0,
        resource_min_gain=0.05,
        strategy_min_gain=0.0,
        m0_omission_trigger=1.0,
        r0_strategy_omission_trigger=1.0,
    )
    predictions = model_with_bundle(bundle, config=config).predict_actions(state)
    # The paired delta is exactly +0.10 for both members. Independent marginal
    # LCB(action)-UCB(baseline) would be negative and reject this action.
    assert predictions["ME+R0"].resource_gate_passed


def test_risk_thresholds_are_dimension_specific_and_action_applicable():
    state = make_state("risks")
    quality_members = {action: [0.5, 0.5] for action in state.allowed_actions}
    bundle = make_bundle(
        quality_members,
        risk_upper={
            ("M0+R0", "stale_or_conflicting_use"): 3.0,
            ("M0+RS", "strategy_overuse"): 0.9,
        },
    )
    config = SelectionConfig(
        max_risk_ucb=1.0,
        risk_ucb_thresholds={
            "stale_or_conflicting_use": 0.10,
            "strategy_overuse": 0.20,
        },
        m0_omission_trigger=1.0,
        r0_strategy_omission_trigger=1.0,
    )
    predictions = model_with_bundle(bundle, config=config).predict_actions(state)
    assert predictions["M0+R0"].feasible
    assert not predictions["M0+RS"].feasible
    assert "risk_ucb_exceeds_threshold:strategy_overuse" in predictions[
        "M0+RS"
    ].exclusion_reasons


def test_prediction_coverage_requires_exact_legal_matrix_and_reports_applicability():
    state = make_state("coverage")
    quality_members = {action: [0.0, 1.0] for action in state.allowed_actions}
    bundle = make_bundle(quality_members, broad_intervals=True)
    model = model_with_bundle(bundle, config=SelectionConfig(max_risk_ucb=1.0))
    labels = [make_label(state, action) for action in state.allowed_actions]
    report = evaluate_prediction_coverage(model, [state], labels)
    assert report["n_state_actions"] == report["expected_state_actions"] == 4
    assert report["n_users"] == 1
    assert report["coverage_unit"] == "user_block_all_legal_actions"
    assert report["quality"]["coverage"] == 1.0
    assert report["risk_dimensions_applicable_only"]["stale_or_conflicting_use"][
        "n"
    ] == 1
    assert report["risk_dimensions_applicable_only"]["strategy_overuse"]["n"] == 1
    assert report["state_action_marginal_diagnostics"][
        "risk_dimensions_applicable_only"
    ]["stale_or_conflicting_use"]["n"] == 2
    with pytest.raises(ValueError, match="exact labels for every legal action"):
        evaluate_prediction_coverage(model, [state], labels[:-1])


def test_split_conformal_radius_covers_nonzero_residual_with_zero_ensemble_std():
    states = [
        make_state(f"conformal_{index}", user_id=f"conformal_user_{index}")
        for index in range(4)
    ]
    model = PMV2Model(
        feature_builder=RoutingBuilder(),
        response_heads={name: ConstantHead(0.5) for name in RESPONSE_FIELDS},
        risk_heads={name: ConstantHead(0.0) for name in RISK_FIELDS},
        selection_config=SelectionConfig(max_risk_ucb=1.0),
    )
    labels = [
        make_label(
            state,
            action,
            response=make_response(4.0),
            risk=make_risk(1.0),
        )
        for state in states
        for action in state.allowed_actions
    ]
    report = calibrate_uncertainty_multiplier(
        model,
        states,
        labels,
        z_candidates=[1.0],
        target_coverage=0.80,
        minimum_quality_coverage_lower_bound=0.50,
        minimum_response_coverage_lower_bound=0.50,
        minimum_risk_coverage_lower_bound=0.50,
        coverage_confidence_level=0.95,
    )
    assert report["method"] == "additive_user_block_split_conformal_residual_radius"
    assert report["n_calibration_users"] == 4
    assert report["selected"]["uncertainty_z"] == 1.0
    assert model.response_conformal_radii["emotional_support"] == pytest.approx(1.0)
    assert model.risk_conformal_radii["unsupported_personal_claim"] == pytest.approx(
        1.0
    )
    assert model.quality_conformal_radius == pytest.approx(0.25)
    assert report["coverage"]["quality"]["coverage"] == 1.0
    predictions = model.predict_actions(states[0])
    assert predictions["M0+R0"].risk_ucb == pytest.approx(1.0 / 3.0)


def test_internal_wilson_coverage_gate_is_informative_not_all_hit_only():
    config = load_config("configs/pm_v2.yaml")
    uncertainty = config["uncertainty_calibration"]
    internal = config["internal_reportability_gate"]
    thresholds = {
        float(uncertainty["minimum_quality_coverage_lower_bound"]),
        float(uncertainty["minimum_response_coverage_lower_bound"]),
        float(uncertainty["minimum_risk_coverage_lower_bound"]),
        float(internal["minimum_quality_interval_coverage"]),
        float(internal["minimum_response_interval_coverage"]),
        float(internal["minimum_risk_interval_coverage"]),
    }
    assert thresholds == {0.60}

    def wilson_lower(
        successes: int, total: int, z: float = 1.959963984540054
    ) -> float:
        proportion = successes / total
        denominator = 1.0 + z**2 / total
        center = proportion + z**2 / (2.0 * total)
        margin = z * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        return float((center - margin) / denominator)

    # The frozen n=16 design requires 14 covered user blocks, not a brittle
    # 16/16 for every one of the fourteen modeled heads.
    assert wilson_lower(13, 16) < 0.60
    assert wilson_lower(14, 16) > 0.60


def test_cost_calibration_reports_linear_relation():
    report = cost_calibration_diagnostics([0.0, 1.0, 2.0], [100.0, 110.0, 120.0])
    assert report["pearson_correlation"] == pytest.approx(1.0)
    assert report["spearman_correlation"] == pytest.approx(1.0)
    assert report["observed_on_estimated_ols"]["intercept"] == pytest.approx(100.0)
    assert report["observed_on_estimated_ols"]["slope"] == pytest.approx(10.0)


def test_tuning_objective_uses_normalized_estimated_not_observed_cost():
    state = make_state("cost_tuning")
    model = fake_routing_model({})

    def predict_for_cost_weight(self, current_state):
        cost_weight = self.selection_config.cost_weight
        return {
            "M0+R0": action_prediction(
                "M0+R0",
                quality=0.60,
                utility=0.60,
                estimated_cost=0.0,
                normalized_cost=0.0,
            ),
            "M0+RS": action_prediction(
                "M0+RS", quality=0.0, utility=-1.0, feasible=False
            ),
            "ME+R0": action_prediction(
                "ME+R0",
                quality=0.71,
                utility=0.71 - cost_weight,
                estimated_cost=100.0,
                normalized_cost=1.0,
            ),
            "ME+RS": action_prediction(
                "ME+RS", quality=0.0, utility=-1.0, feasible=False
            ),
        }

    model.predict_actions = MethodType(predict_for_cost_weight, model)
    labels = [
        make_label(state, "M0+R0", response=make_response(3.4), cost=100),
        make_label(state, "ME+R0", response=make_response(3.84), cost=10000),
    ]
    report = tune_selection_config(
        model,
        [state],
        labels,
        cost_weights=[0.10, 0.20],
        risk_weights=[0.10],
        resource_gains=[0.0],
        strategy_gains=[0.0],
        max_risks=[1.0],
    )
    assert report["selected"]["config"]["cost_weight"] == pytest.approx(0.10)
    assert report["selected"]["metrics"][
        "mean_normalized_estimated_resource_cost"
    ] == pytest.approx(1.0)
    assert report["selected"]["metrics"]["mean_observed_input_tokens"] == 10000.0
    assert report["objective_spec"] == {
        "version": "pmv2-calibration-utility-v1",
        "risk_weight": 0.25,
        "cost_weight": 0.10,
        "cost_basis": "within_state_normalized_estimated_resource_cost",
        "observed_input_tokens_used": False,
    }


def test_tuning_candidates_cannot_improve_objective_by_shrinking_own_penalty():
    state = make_state("fixed_tuning_ruler")
    model = fake_routing_model({})

    def always_choose_m0(self, current_state):
        return {
            "M0+R0": action_prediction(
                "M0+R0",
                quality=0.60,
                utility=0.60,
                estimated_cost=0.0,
                normalized_cost=0.0,
            ),
            "M0+RS": action_prediction(
                "M0+RS", quality=0.0, utility=-1.0, feasible=False
            ),
            "ME+R0": action_prediction(
                "ME+R0", quality=0.0, utility=-1.0, feasible=False
            ),
            "ME+RS": action_prediction(
                "ME+RS", quality=0.0, utility=-1.0, feasible=False
            ),
        }

    model.predict_actions = MethodType(always_choose_m0, model)
    labels = [
        make_label(
            state,
            "M0+R0",
            response=make_response(3.4),
            risk=make_risk(0.6),
            cost=100,
        )
    ]
    report = tune_selection_config(
        model,
        [state],
        labels,
        cost_weights=[0.10],
        risk_weights=[0.10, 0.40],
        resource_gains=[0.0],
        strategy_gains=[0.0],
        max_risks=[1.0],
        objective_risk_weight=0.25,
        objective_cost_weight=0.10,
    )
    rows = report["pareto_candidates"]
    assert len(rows) == 2
    assert rows[0]["objective"] == pytest.approx(rows[1]["objective"])
    assert rows[0]["metrics"]["mean_realized_utility"] != pytest.approx(
        rows[1]["metrics"]["mean_realized_utility"]
    )


def test_complete_matrix_keeps_aggregated_rows_and_gates_per_dimension_mad():
    script = Path(__file__).resolve().parents[1] / "scripts" / "22_train_pm_v2.py"
    spec = importlib.util.spec_from_file_location("train_pm_v2_matrix", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kept = make_state("matrix_kept", user_id="user_kept")
    noisy = make_state("matrix_noisy", user_id="user_noisy")
    labels = [make_label(kept, action) for action in kept.allowed_actions] + [
        make_label(noisy, action, reliable=action != "ME+RS")
        for action in noisy.allowed_actions
    ]
    gate = {
        "minimum_retained_states": 1,
        "minimum_state_retention_rate": 0.50,
        "minimum_retained_users": 1,
        "minimum_user_retention_rate": 0.50,
        "minimum_retained_semantic_families": 1,
        "minimum_semantic_family_retention_rate": 0.50,
        # Joint all-13-dimension reliability is diagnostic only and cannot fail.
        "minimum_reliable_action_label_coverage_rate": 1.00,
        "minimum_low_mad_coverage_per_dimension": 0.80,
        "minimum_low_mad_coverage_per_action_dimension": 0.50,
    }
    retained_states, retained_labels, report = (
        module.complete_reliable_action_matrix_subset(
            [kept, noisy],
            labels,
            split_name="train",
            gate_cfg=gate,
            low_mad_threshold=0.75,
        )
    )
    assert [state.state_id for state in retained_states] == [
        kept.state_id,
        noisy.state_id,
    ]
    assert len(retained_labels) == len(labels)
    assert report["incomplete_state_count"] == 0
    assert report["metrics"]["state_retention_rate"] == 1.0
    assert report["metrics"]["minimum_action_dimension_low_mad_coverage"] == 0.5
    strict_gate = dict(gate, minimum_low_mad_coverage_per_action_dimension=0.75)
    with pytest.raises(RuntimeError, match="action-matrix gate failed"):
        module.complete_reliable_action_matrix_subset(
            [kept, noisy],
            labels,
            split_name="train",
            gate_cfg=strict_gate,
            low_mad_threshold=0.75,
        )


def test_user_cluster_bootstrap_can_fail_noninferiority_despite_positive_mean():
    script = Path(__file__).resolve().parents[1] / "scripts" / "22_train_pm_v2.py"
    spec = importlib.util.spec_from_file_location("train_pm_v2_bootstrap", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pm_rows = []
    fixed_rows = []
    for user_index, delta in enumerate((0.4, -0.1, -0.1)):
        for state_index in range(2):
            state_id = f"boot_{user_index}_{state_index}"
            fixed_rows.append(
                {
                    "state_id": state_id,
                    "user_id": f"user_{user_index}",
                    "quality": 0.5,
                    "risk": 0.1,
                    "realized_utility": 0.4,
                    "response_dimensions": {name: 3.0 for name in RESPONSE_FIELDS},
                }
            )
            pm_dimensions = {name: 3.0 for name in RESPONSE_FIELDS}
            pm_dimensions["emotional_support"] += delta
            pm_rows.append(
                {
                    "state_id": state_id,
                    "user_id": f"user_{user_index}",
                    "quality": 0.5 + delta,
                    "risk": 0.1,
                    "realized_utility": 0.4 + delta,
                    "response_dimensions": pm_dimensions,
                }
            )
    report = module.paired_user_cluster_bootstrap(
        pm_rows,
        fixed_rows,
        replicates=5000,
        confidence_level=0.95,
        seed=17,
    )
    quality = report["metrics"]["quality"]
    assert quality["mean_delta"] > 0.0
    assert quality["ci_lower"] < 0.0
    assert report["cluster_key"] == "user_id"


def test_audit_consumes_passed_composite_and_objective_weights():
    regimes = [
        "context_only",
        "profile_needed",
        "summary_needed",
        "event_needed",
        "multi_source_needed",
        "memory_harmful",
        "strategy_helpful",
        "strategy_harmful",
        "ambiguous",
    ]
    states = [make_state(f"audit_{index}", regime=regime) for index, regime in enumerate(regimes)]
    evaluator_contexts = make_evaluator_contexts(
        states,
        {state.state_id: regime for state, regime in zip(states, regimes)},
    )
    weights = {name: 0.0 for name in RESPONSE_FIELDS}
    weights["personalization"] = 1.0
    spec = CompositeSpec(version="personalization-only", weights=weights)
    weights_hash = composite_weights_digest(spec)
    labels = []
    for state in states:
        for action in state.allowed_actions:
            if action == "M0+R0":
                response = make_response(1.0, emotional_support=5.0)
            elif action == "M0+RS":
                response = make_response(1.0, personalization=5.0)
            else:
                response = make_response(1.0)
            labels.append(
                make_label(
                    state,
                    action,
                    reliable=not (
                        state is states[0] and action == "M0+R0"
                    ),
                    response=response,
                    composite_weights_sha256=weights_hash,
                )
            )
    report = audit_training_labels(
        states,
        labels,
        evaluator_contexts=evaluator_contexts,
        composite_spec=spec,
        risk_weight=0.0,
        cost_weight=0.0,
        maximum_single_action_share=1.0,
        minimum_m0_share=0.0,
        minimum_r0_share=0.0,
        minimum_rs_share=0.0,
        minimum_distinct_actions=1,
        minimum_regime_pass_rate=0.0,
        minimum_reliable_rate=1.0,
    )
    assert report["status"] == "PASS"
    assert report["reliable_label_rate_below_diagnostic_reference"]
    assert report["oracle_action_distribution"] == {"M0+RS": len(states)}
    assert report["thresholds"]["composite_spec"]["version"] == "personalization-only"
    assert report["thresholds"]["risk_weight"] == 0.0
    assert report["thresholds"]["cost_weight"] == 0.0


def test_pilot_label_value_feasibility_rejects_constant_no_signal_matrix():
    pilot_actions = [
        "M0+R0",
        "M0+RS",
        "MP+R0",
        "MP+RS",
        "MS+R0",
        "MS+RS",
        "ME+R0",
        "ME+RS",
        "MPMSME+R0",
        "MPMSME+RS",
    ]
    regimes = [
        "context_only",
        "profile_needed",
        "summary_needed",
        "event_needed",
        "multi_source_needed",
        "memory_harmful",
        "strategy_helpful",
        "strategy_harmful",
        "ambiguous",
    ]
    states = []
    regime_by_state = {}
    for regime in regimes:
        for replicate in range(2):
            state = make_state(
                f"pilot_signal_{regime}_{replicate}", regime=regime
            )
            state.allowed_actions = list(pilot_actions)
            states.append(state)
            regime_by_state[state.state_id] = regime
    contexts = make_evaluator_contexts(states, regime_by_state)
    labels = [
        make_label(
            state,
            action,
            response=make_response(3.0),
            risk=make_risk(0.0),
        )
        for state in states
        for action in pilot_actions
    ]
    report = audit_pilot_label_value_feasibility(
        states,
        labels,
        evaluator_contexts=contexts,
        composite_spec=CompositeSpec(),
        pilot_actions=pilot_actions,
        thresholds={
            "minimum_distinct_oracle_actions": 6,
            "maximum_oracle_action_share": 0.35,
            "minimum_m0_oracle_share": 0.10,
            "minimum_r0_oracle_share": 0.15,
            "minimum_rs_oracle_share": 0.15,
            "minimum_distinct_oracle_actions_per_regime": 2,
            "minimum_mean_within_state_quality_range": 0.05,
            "minimum_mean_within_state_quality_variance": 0.0002,
            "minimum_strategy_directional_separation": 0.01,
            "minimum_memory_directional_separation": 0.01,
            "minimum_oracle_utility_headroom": 0.005,
            "minimum_oracle_quality_headroom": 0.005,
            "minimum_oracle_emotional_support_headroom": 0.005,
        },
        risk_weight=0.25,
        cost_weight=0.10,
    )
    assert report["status"] == "FAIL"
    assert report["oracle_action_distribution"] == {"M0+R0": len(states)}
    assert not report["checks"]["within_state_quality_range"]
    assert not report["checks"]["within_state_quality_variance"]
    assert not report["checks"]["per_regime_oracle_diversity"]
    assert not report["checks"]["oracle_utility_headroom"]
    assert not report["checks"]["oracle_quality_headroom"]
    assert not report["checks"]["oracle_emotional_support_headroom"]


def test_oracle_headroom_uses_same_utility_oracle_for_support():
    actions = ["M0+R0", "M0+RS", "ME+R0", "ME+RS"]
    regimes = [
        "context_only",
        "profile_needed",
        "summary_needed",
        "event_needed",
        "multi_source_needed",
        "memory_harmful",
        "strategy_helpful",
        "strategy_harmful",
        "ambiguous",
    ]
    states = []
    regime_by_state = {}
    for regime in regimes:
        for replicate in range(2):
            state = make_state(
                f"same_oracle_{regime}_{replicate}", regime=regime
            )
            state.allowed_actions = list(actions)
            states.append(state)
            regime_by_state[state.state_id] = regime
    contexts = make_evaluator_contexts(states, regime_by_state)
    labels = []
    for state in states:
        labels.append(
            make_label(
                state,
                "M0+R0",
                response=make_response(
                    1.0,
                    emotional_support=5.0,
                ),
            )
        )
        for memory_action in ("ME+R0", "ME+RS"):
            labels.append(
                make_label(
                    state,
                    memory_action,
                    response=make_response(1.0),
                )
            )
        labels.append(
            make_label(
                state,
                "M0+RS",
                response=make_response(
                    5.0,
                    emotional_support=1.0,
                ),
            )
        )
    report = audit_pilot_label_value_feasibility(
        states,
        labels,
        evaluator_contexts=contexts,
        composite_spec=CompositeSpec(),
        pilot_actions=actions,
        thresholds={
            "minimum_distinct_oracle_actions": 1,
            "maximum_oracle_action_share": 1.0,
            "minimum_m0_oracle_share": 0.0,
            "minimum_r0_oracle_share": 0.0,
            "minimum_rs_oracle_share": 0.0,
            "minimum_distinct_oracle_actions_per_regime": 1,
            "minimum_mean_within_state_quality_range": 0.0,
            "minimum_mean_within_state_quality_variance": 0.0,
            "minimum_strategy_directional_separation": -1.0,
            "minimum_memory_directional_separation": -1.0,
            "minimum_oracle_utility_headroom": -1.0,
            "minimum_oracle_quality_headroom": -1.0,
            "minimum_oracle_emotional_support_headroom": 0.0,
        },
        risk_weight=0.0,
        cost_weight=0.0,
    )
    assert report["oracle_action_distribution"] == {"M0+RS": len(states)}
    support = report["oracle_headroom"]["emotional_support"]
    assert support["utility_oracle_policy_mean"] == 0.0
    assert support["best_fixed_action"] == "M0+R0"
    assert support["oracle_minus_best_fixed"] == -1.0
    assert not report["checks"]["oracle_emotional_support_headroom"]


def test_train_only_deployable_feature_observability_passes_and_fails_closed():
    regimes = [
        "context_only",
        "profile_needed",
        "summary_needed",
        "event_needed",
        "multi_source_needed",
        "memory_harmful",
        "strategy_helpful",
        "strategy_harmful",
        "ambiguous",
    ]
    settings = {
        "action_id": "M0+R0",
        "cv_folds": 3,
        "classifier_c": 1.0,
        "classifier_max_iter": 1000,
        "random_seed": 8171,
        "minimum_train_states": 54,
        "minimum_regime_macro_f1": 0.60,
        "minimum_needed_sources_macro_f1": 0.60,
        "minimum_memory_need_macro_f1": 0.60,
        "minimum_strategy_direction_macro_f1": 0.60,
        "minimum_memory_direction_macro_f1": 0.60,
    }
    signaled_states = []
    signaled_regimes = {}
    for user_index in range(6):
        for regime in regimes:
            state = make_state(
                f"observable_{user_index}_{regime}",
                user_id=f"observable_user_{user_index}",
                regime=regime,
            )
            state.current_user_text = f"distinct_{regime} support request"
            state.current_session_history = [
                DialogueTurn(role="user", content=f"history_{regime}")
            ]
            state.current_session_summary = f"summary_{regime}"
            signaled_states.append(state)
            signaled_regimes[state.state_id] = regime
    signaled_contexts = make_evaluator_contexts(
        signaled_states, signaled_regimes
    )
    passed = audit_deployable_feature_observability(
        signaled_states,
        evaluator_contexts=signaled_contexts,
        settings=settings,
    )
    assert passed["status"] == "PASS"
    assert passed["train_only"] is True
    assert all(passed["checks"].values())

    identical_states = []
    identical_regimes = {}
    for user_index in range(6):
        for regime in regimes:
            state = make_state(
                f"unobservable_{user_index}_{regime}",
                user_id=f"unobservable_user_{user_index}",
                regime="multi_source_needed",
            )
            state.current_user_text = "the same support request"
            state.current_session_history = [
                DialogueTurn(role="user", content="the same history")
            ]
            state.current_session_summary = "the same summary"
            state.inventory[MemorySource.MS] = ObservableSourceSummary(
                available=True,
                count=2,
                min_age_sessions=1,
                median_age_sessions=2.0,
                max_age_sessions=3,
                estimated_tokens=90,
            )
            state.allowed_actions = ["M0+R0"]
            identical_states.append(state)
            identical_regimes[state.state_id] = regime
    identical_contexts = make_evaluator_contexts(
        identical_states, identical_regimes
    )
    failed = audit_deployable_feature_observability(
        identical_states,
        evaluator_contexts=identical_contexts,
        settings=settings,
    )
    assert failed["status"] == "FAIL"
    assert not failed["checks"]["strategy_direction_macro_f1"]
    assert not failed["checks"]["memory_direction_macro_f1"]


def test_config_loader_rejects_duplicate_yaml_keys(tmp_path: Path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("uncertainty_calibration: {}\nuncertainty_calibration: {}\n")
    with pytest.raises(ValueError, match="duplicate YAML configuration key"):
        load_config(path)


def test_internal_gate_uses_max_share_entropy_support_and_coverage():
    script = Path(__file__).resolve().parents[1] / "scripts" / "22_train_pm_v2.py"
    spec = importlib.util.spec_from_file_location("train_pm_v2_hardening", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dimensions = {name: 4.0 for name in RESPONSE_FIELDS}
    internal = {
        "mean_quality": 0.75,
        "mean_response_dimensions": dimensions,
        "mean_risk": 0.10,
        "mean_observed_input_tokens": 100.0,
        "mean_realized_utility": 0.70,
        "learned_m0_rate": 0.10,
        "learned_r0_rate": 0.20,
        "nonfallback_m0_r0_rate": 0.10,
        "severe_ood_fallback_rate": 0.0,
        "no_feasible_fallback_rate": 0.0,
    }
    fixed_dimensions = dict(dimensions)
    fixed_dimensions["emotional_support"] = 4.2
    fixed = {
        "mean_quality": 0.75,
        "mean_response_dimensions": fixed_dimensions,
        "mean_risk": 0.10,
        "mean_observed_input_tokens": 100.0,
        "mean_realized_utility": 0.70,
    }
    alignment = {
        "distinct_actions": 4,
        "maximum_action_share": 0.90,
        "action_entropy_bits": 0.5,
        "mean_regime_pass_rate": 1.0,
        "minimum_regime_pass_rate": 0.0,
        "minimum_regime_wilson_lower_bound": 0.0,
        "regime_wilson_confidence_level": 0.95,
    }
    coverage = {
        "quality": {"coverage_lower_bound": 1.0},
        "minimum_response_dimension_coverage_lower_bound": 1.0,
        "minimum_risk_dimension_coverage_lower_bound": 1.0,
    }
    cost_diagnostics = {
        "all_legal_actions": {"spearman_correlation": 0.90}
    }
    paired = {
        "metrics": {
            "quality": {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "emotional_support": {
                "mean_delta": -0.20,
                "ci_lower": -0.20,
                "ci_upper": -0.20,
            },
            "risk": {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
            "utility": {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
        }
    }
    gate_cfg = {
        "minimum_quality_delta_vs_cost_matched": -0.02,
        "minimum_emotional_support_delta_vs_cost_matched": -0.10,
        "minimum_utility_delta_vs_cost_matched": -1.0,
        "minimum_advantage_quality_delta_vs_cost_matched": 0.0,
        "minimum_advantage_emotional_support_delta_vs_cost_matched": 0.0,
        "minimum_advantage_utility_delta_vs_cost_matched": 0.0,
        "require_learned_routing_advantage_before_external": True,
        "maximum_risk_delta_vs_cost_matched": 0.10,
        "maximum_cost_matched_relative_deviation": 0.10,
        "minimum_estimated_observed_cost_spearman": 0.70,
        "minimum_distinct_actions": 4,
        "maximum_learned_action_share": 0.50,
        "minimum_learned_action_entropy_bits": 1.0,
        "minimum_m0_rate": 0.05,
        "minimum_r0_rate": 0.10,
        "minimum_nonfallback_m0_r0_rate": 0.05,
        "maximum_severe_ood_fallback_rate": 0.10,
        "maximum_no_feasible_fallback_rate": 0.10,
        "minimum_quality_interval_coverage": 0.50,
        "minimum_response_interval_coverage": 0.50,
        "minimum_risk_interval_coverage": 0.50,
        "minimum_mean_regime_pass_rate": 0.50,
        "regime_wilson_confidence_level": 0.95,
        "minimum_each_regime_wilson_lower_bound": 0.50,
    }
    calibration_cost_matched = {
        "calibration_observed_token_relative_deviation": 0.0,
        "cost_frontier_eligible": True,
    }
    assessment = module.build_internal_reportability_assessment(
        internal=internal,
        internal_cost_matched=fixed,
        calibration_cost_matched=calibration_cost_matched,
        internal_alignment=alignment,
        internal_coverage=coverage,
        internal_cost_diagnostics=cost_diagnostics,
        paired_bootstrap=paired,
        gate_cfg=gate_cfg,
    )
    assert not assessment["checks"]["learned_maximum_action_share"]
    assert not assessment["checks"]["learned_action_entropy"]
    assert not assessment["checks"]["emotional_support_vs_cost_matched"]
    assert not assessment["checks"]["each_regime_user_block_wilson"]
    assert not assessment["learned_routing_advantage_verified"]
    assert not assessment["reportable"]

    # A small support deficit can satisfy the explicitly labeled deployment
    # tradeoff diagnostic, but it must not authorize the stronger claim that
    # learned routing beat the same-token fixed policy.
    passing_alignment = {
        **alignment,
        "maximum_action_share": 0.25,
        "action_entropy_bits": 2.0,
        "minimum_regime_wilson_lower_bound": 0.55,
    }
    tradeoff_fixed = {
        **fixed,
        "mean_response_dimensions": {
            **dimensions,
            "emotional_support": 4.05,
        },
    }
    tradeoff_paired = {
        "metrics": {
            **paired["metrics"],
            "emotional_support": {
                "mean_delta": -0.05,
                "ci_lower": -0.05,
                "ci_upper": -0.05,
            },
        }
    }
    tradeoff = module.build_internal_reportability_assessment(
        internal=internal,
        internal_cost_matched=tradeoff_fixed,
        calibration_cost_matched=calibration_cost_matched,
        internal_alignment=passing_alignment,
        internal_coverage=coverage,
        internal_cost_diagnostics=cost_diagnostics,
        paired_bootstrap=tradeoff_paired,
        gate_cfg=gate_cfg,
    )
    assert tradeoff["deployment_tradeoff_reportable"]
    assert not tradeoff["learned_routing_advantage_verified"]
    assert not tradeoff["reportable"]

    # Zero-width all-tie confidence intervals establish at most non-inferiority;
    # they must never be promoted to evidence of a learned-routing advantage.
    all_tie_fixed = {
        **fixed,
        "mean_response_dimensions": dimensions,
    }
    all_tie_paired = {
        "metrics": {
            name: {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
            for name in ("quality", "emotional_support", "risk", "utility")
        }
    }
    all_tie = module.build_internal_reportability_assessment(
        internal=internal,
        internal_cost_matched=all_tie_fixed,
        calibration_cost_matched=calibration_cost_matched,
        internal_alignment=passing_alignment,
        internal_coverage=coverage,
        internal_cost_diagnostics=cost_diagnostics,
        paired_bootstrap=all_tie_paired,
        gate_cfg=gate_cfg,
    )
    assert all_tie["deployment_tradeoff_reportable"]
    assert not all_tie["learned_routing_advantage_verified"]
    assert not all_tie["reportable"]

    # The old one-sided PM/fixed ratio admitted a fixed policy twice as
    # expensive as PM.  The observed-token relative-deviation gate rejects it.
    expensive_fixed = {
        **tradeoff_fixed,
        "mean_observed_input_tokens": 200.0,
    }
    expensive = module.build_internal_reportability_assessment(
        internal=internal,
        internal_cost_matched=expensive_fixed,
        calibration_cost_matched=calibration_cost_matched,
        internal_alignment=passing_alignment,
        internal_coverage=coverage,
        internal_cost_diagnostics=cost_diagnostics,
        paired_bootstrap=tradeoff_paired,
        gate_cfg=gate_cfg,
    )
    assert expensive["deltas"]["baseline_to_pm_observed_token_ratio"] == 2.0
    assert expensive["deltas"]["observed_token_relative_deviation"] == 1.0
    assert not expensive["checks"]["observed_token_cost_match"]

    calibration_mismatch = module.build_internal_reportability_assessment(
        internal=internal,
        internal_cost_matched=all_tie_fixed,
        calibration_cost_matched={
            "calibration_observed_token_relative_deviation": 0.20,
            "cost_frontier_eligible": False,
        },
        internal_alignment=passing_alignment,
        internal_coverage=coverage,
        internal_cost_diagnostics=cost_diagnostics,
        paired_bootstrap=all_tie_paired,
        gate_cfg=gate_cfg,
    )
    assert not calibration_mismatch["checks"][
        "calibration_observed_token_cost_match"
    ]
    assert not calibration_mismatch["reportable"]
