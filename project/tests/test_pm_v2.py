from __future__ import annotations

from types import MethodType

import pytest
from pydantic import ValidationError

from metacom_pm.contracts import DialogueTurn, MemorySource
from metacom_pm.pm_v2_contracts import (
    ActionLabel,
    ActionPrediction,
    ObservableSourceSummary,
    PMV2Split,
    PMV2State,
    PredictionInterval,
    ResponseDimensions,
    RiskDimensions,
)
from metacom_pm.pm_v2_data import validate_split_manifests
from metacom_pm.pm_v2_judging import (
    ResponseJudgeOutput,
    build_response_messages,
    build_risk_messages,
    validate_judge_table,
)
from metacom_pm.pm_v2_model import PMV2Model, SelectionConfig


class DummyBuilder:
    def ood_report(self, state):
        return {
            "semantic_ood_score": 0.1,
            "metadata_ood_score": 0.0,
            "severe_semantic_ood": False,
            "severe_metadata_ood": False,
        }


def make_state(*, state_id: str = "state_1", split: PMV2Split = PMV2Split.TRAIN):
    inventory = {
        MemorySource.MP: ObservableSourceSummary(
            available=False, count=0, estimated_tokens=0
        ),
        MemorySource.MS: ObservableSourceSummary(
            available=False, count=0, estimated_tokens=0
        ),
        MemorySource.ME: ObservableSourceSummary(
            available=True,
            count=3,
            min_age_sessions=1,
            median_age_sessions=2.0,
            max_age_sessions=3,
            estimated_tokens=120,
            query_similarity_mean=0.2,
            catalog_embedding=[0.0] * 64,
        ),
    }
    return PMV2State(
        state_id=state_id,
        card_id=f"card_{state_id}",
        user_id=f"user_{state_id}",
        split=split,
        semantic_family=f"family_{state_id}",
        surface_form_id=f"surface_{state_id}",
        current_user_text="I feel unsettled after what happened at work.",
        current_session_history=[
            DialogueTurn(role="user", content="It has been a difficult week.")
        ],
        current_session_summary="The user is discussing current work stress.",
        session_index=4,
        inventory=inventory,
        strategy_catalog_count=20,
        strategy_estimated_tokens=240,
        allowed_actions=["M0+R0", "M0+RS", "ME+R0", "ME+RS"],
    )


def interval(mean: float, lower: float | None = None, upper: float | None = None):
    return PredictionInterval(
        mean=mean,
        std=0.0,
        lower=mean if lower is None else lower,
        upper=mean if upper is None else upper,
    )


def prediction(
    action: str,
    *,
    quality: float,
    utility: float,
    cost: float,
    feasible: bool = True,
    resource_gate: bool = True,
    strategy_gate: bool = True,
):
    response = {
        name: interval(1.0 + 4.0 * quality)
        for name in ResponseDimensions.model_fields
    }
    risk = {
        name: interval(0.0)
        for name in (
            "selected_context_misuse",
            "unnecessary_exposure",
            "stale_or_conflicting_use",
            "unsupported_personal_claim",
            "memory_omission",
            "strategy_overuse",
            "strategy_omission",
        )
    }
    return ActionPrediction(
        action_id=action,
        response=response,
        risk=risk,
        quality_mean=quality,
        quality_lcb=quality,
        risk_ucb=0.0,
        estimated_cost=cost,
        normalized_cost=cost / 1000.0,
        utility=utility,
        feasible=feasible,
        resource_gate_passed=resource_gate,
        strategy_gate_passed=strategy_gate,
    )


def model_with_predictions(predictions):
    model = PMV2Model(
        feature_builder=DummyBuilder(),
        response_heads={},
        risk_heads={},
        selection_config=SelectionConfig(),
    )

    def fake_predict(self, state):
        return predictions

    model.predict_actions = MethodType(fake_predict, model)
    return model


def test_judge_schema_has_no_overall_field():
    assert "overall" not in ResponseJudgeOutput.model_fields
    with pytest.raises(ValidationError):
        ResponseJudgeOutput(
            emotional_support=4.0,
            personalization=4.0,
            memory_appropriateness=4.0,
            factual_grounding=4.0,
            temporal_consistency=4.0,
            non_intrusiveness=4.0,
            rationale="ok",
            overall=4.0,
        )


def test_judge_prompts_include_the_current_session_summary():
    state = make_state()
    response_prompt = build_response_messages(
        state=state,
        authorized_user_context="Authorized context.",
        candidate_response="A candidate response.",
    )
    risk_prompt = build_risk_messages(
        state=state,
        authorized_user_context="Authorized context.",
        selected_context="Selected context.",
        candidate_response="A candidate response.",
    )
    for messages in (response_prompt, risk_prompt):
        rendered = "\n".join(message["content"] for message in messages)
        assert "CURRENT SESSION SUMMARY" in rendered
        assert state.current_session_summary in rendered


def test_explicit_abstention_can_choose_m0_r0():
    state = make_state()
    predictions = {
        "M0+R0": prediction("M0+R0", quality=0.80, utility=0.75, cost=0.0),
        "M0+RS": prediction(
            "M0+RS", quality=0.81, utility=0.60, cost=240.0, strategy_gate=False
        ),
        "ME+R0": prediction(
            "ME+R0", quality=0.79, utility=0.55, cost=120.0, resource_gate=False
        ),
        "ME+RS": prediction(
            "ME+RS",
            quality=0.82,
            utility=0.40,
            cost=360.0,
            resource_gate=False,
            strategy_gate=False,
        ),
    }
    decision = model_with_predictions(predictions).choose(state)
    assert decision.chosen_action == "M0+R0"


def test_strategy_off_can_win_with_memory_on():
    state = make_state()
    predictions = {
        "M0+R0": prediction("M0+R0", quality=0.70, utility=0.60, cost=0.0),
        "M0+RS": prediction("M0+RS", quality=0.70, utility=0.50, cost=240.0),
        "ME+R0": prediction("ME+R0", quality=0.84, utility=0.78, cost=120.0),
        "ME+RS": prediction(
            "ME+RS", quality=0.85, utility=0.65, cost=360.0, strategy_gate=False
        ),
    }
    decision = model_with_predictions(predictions).choose(state)
    assert decision.chosen_action == "ME+R0"


def test_cost_is_part_of_utility_not_only_tie_break():
    state = make_state()
    predictions = {
        "M0+R0": prediction("M0+R0", quality=0.75, utility=0.70, cost=0.0),
        "M0+RS": prediction("M0+RS", quality=0.80, utility=0.60, cost=240.0),
        "ME+R0": prediction("ME+R0", quality=0.79, utility=0.74, cost=120.0),
        "ME+RS": prediction("ME+RS", quality=0.81, utility=0.55, cost=360.0),
    }
    decision = model_with_predictions(predictions).choose(state)
    assert decision.chosen_action == "ME+R0"


def test_severe_ood_fails_closed_to_m0_r0():
    state = make_state()
    predictions = {
        action: prediction(action, quality=0.9, utility=0.9, cost=100.0)
        for action in state.allowed_actions
    }
    model = model_with_predictions(predictions)

    class OODBuilder:
        def ood_report(self, state):
            return {
                "semantic_ood_score": 0.9,
                "metadata_ood_score": 0.0,
                "severe_semantic_ood": True,
                "severe_metadata_ood": False,
            }

    model.feature_builder = OODBuilder()
    decision = model.choose(state)
    assert decision.chosen_action == "M0+R0"
    assert decision.ood_fallback_used


def test_joint_split_manifest_rejects_overlap():
    train = make_state(state_id="train", split=PMV2Split.TRAIN)
    calibration = make_state(state_id="cal", split=PMV2Split.CALIBRATION)
    calibration.user_id = train.user_id
    with pytest.raises(ValidationError):
        validate_split_manifests(
            {
                PMV2Split.TRAIN: [train],
                PMV2Split.CALIBRATION: [calibration],
                PMV2Split.INTERNAL_TEST: [],
            }
        )


def test_evaluator_oracles_are_rejected_from_model_state_schema():
    state = make_state(state_id="clean")
    payload = state.model_dump()
    payload["provenance"] = {"regime": "event_needed"}
    with pytest.raises(ValidationError, match="operational references only"):
        PMV2State.model_validate(payload)

    summary = state.inventory[MemorySource.ME].model_dump()
    summary["conflict_fraction"] = 1.0
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ObservableSourceSummary.model_validate(summary)


def test_degenerate_risk_labels_fail_closed():
    labels = []
    for index in range(4):
        response = ResponseDimensions(
            emotional_support=2.0 + index * 0.5,
            personalization=2.5 + index * 0.4,
            memory_appropriateness=3.0 + index * 0.3,
            factual_grounding=3.5 + index * 0.2,
            temporal_consistency=4.0 + index * 0.1,
            non_intrusiveness=4.5 - index * 0.2,
        )
        labels.append(
            ActionLabel(
                state_id=f"state_{index}",
                card_id=f"card_{index}",
                user_id=f"user_{index}",
                semantic_family=f"family_{index}",
                action_id="M0+R0",
                response=response,
                risk=RiskDimensions(
                    selected_context_misuse=0.0,
                    unnecessary_exposure=0.0,
                    stale_or_conflicting_use=0.0,
                    unsupported_personal_claim=0.0,
                    memory_omission=0.0,
                    strategy_overuse=0.0,
                    strategy_omission=0.0,
                ),
                observed_input_tokens=100,
                retrieval_calls=0,
                judge_families=["a", "b"],
                judge_count=2,
                max_dimension_mad=0.0,
                dimension_mad={
                    **{
                        f"response.{name}": 0.0
                        for name in ResponseDimensions.model_fields
                    },
                    **{
                        f"risk.{name}": 0.0
                        for name in RiskDimensions.model_fields
                    },
                },
                label_reliable=True,
                composite_weights_sha256=(
                    "854996e300bc0506a71679f12a179e1a0"
                    "ae33b050b62eb4d77d7c8d222fcf0b2"
                ),
            )
        )
    with pytest.raises(RuntimeError, match="judge quality gate failed"):
        validate_judge_table(labels)
