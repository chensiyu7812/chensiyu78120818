from __future__ import annotations

import numpy as np

from metacom_pm.contracts import DialogueTurn, MemorySource
from metacom_pm.pm_v1_6_contracts import (
    SourceProbeObservation,
    Step0Cost,
    Step0Observation,
    StrategyFamilyObservation,
    StrategyReadiness,
    STRATEGY_FAMILIES,
)
from metacom_pm.pm_v1_6_models import (
    ConservativeResidualPolicy,
    OutcomeMode,
    OverrideConfig,
    PMV16OutcomeModel,
)
from metacom_pm.pm_v1_6_router import (
    StrongTransparentRouter,
    TransparentRouterConfig,
)
from metacom_pm.pm_v2_contracts import (
    ActionLabel,
    CompositeSpec,
    ObservableSourceSummary,
    PMV2Split,
    PMV2State,
    ResponseDimensions,
    RiskDimensions,
)
from metacom_pm.pm_v2_model import composite_weights_digest


def _state(index: int) -> PMV2State:
    return PMV2State(
        state_id=f"state_{index:024x}",
        card_id=f"card_{index:024x}",
        user_id=f"user_{index}",
        split=PMV2Split.TRAIN,
        semantic_family=f"family_{index}",
        surface_form_id=f"surface_{index}",
        current_user_text="I feel unsettled and would like a gentle next step.",
        current_session_history=[
            DialogueTurn(role="user", content="This week has been difficult."),
            DialogueTurn(role="assistant", content="I am listening."),
        ],
        current_session_summary="The user is under pressure.",
        session_index=4,
        inventory={
            source: ObservableSourceSummary(
                available=False,
                count=0,
                min_age_sessions=None,
                median_age_sessions=None,
                max_age_sessions=None,
                estimated_tokens=0,
                query_similarity_mean=0.0,
                catalog_embedding=[],
            )
            for source in MemorySource
        },
        strategy_catalog_count=10,
        strategy_estimated_tokens=100,
        text_embedding=[],
        allowed_actions=["M0+R0", "M0+RS"],
        provenance={},
    )


def _step0(state: PMV2State) -> Step0Observation:
    return Step0Observation(
        state_id=state.state_id,
        sources={
            source: SourceProbeObservation(
                source=source,
                available=False,
                bounded_count=0,
                estimated_retrievable_tokens=0,
                query_to_source_similarity=0.0,
                representation_valid=False,
            )
            for source in MemorySource
        },
        strategy_families=[
            StrategyFamilyObservation(
                family=family,
                query_to_family_similarity=(
                    0.8 if family == "suggestion_action" else 0.0
                ),
                representation_valid=(family == "suggestion_action"),
            )
            for family in STRATEGY_FAMILIES
        ],
        readiness=StrategyReadiness(
            advice_requested=True,
            advice_rejected=False,
            listening_requested=False,
            clarification_needed=False,
            action_readiness=0.8,
        ),
        step0_cost=Step0Cost(
            query_encoding_latency_ms=1.0,
            memory_source_comparisons=3,
            strategy_family_comparisons=len(STRATEGY_FAMILIES),
            catalog_build_amortized_ms=0.0,
        ),
    )


def _dimension_mad() -> dict[str, float]:
    return {
        **{
            f"response.{name}": 0.05
            for name in ResponseDimensions.model_fields
        },
        **{f"risk.{name}": 0.05 for name in RiskDimensions.model_fields},
    }


def _label(state: PMV2State, action: str) -> ActionLabel:
    use_strategy = action.endswith("+RS")
    response_score = 4.2 if use_strategy else 3.6
    strategy_overuse = 0.2 if use_strategy else 0.0
    strategy_omission = 0.0 if use_strategy else 0.5
    spec = CompositeSpec()
    return ActionLabel(
        state_id=state.state_id,
        card_id=state.card_id,
        user_id=state.user_id,
        semantic_family=state.semantic_family,
        action_id=action,
        response=ResponseDimensions(
            emotional_support=response_score,
            personalization=3.5,
            memory_appropriateness=3.5,
            factual_grounding=4.0,
            temporal_consistency=4.0,
            non_intrusiveness=4.0,
        ),
        risk=RiskDimensions(
            selected_context_misuse=0.0,
            unnecessary_exposure=0.0,
            stale_or_conflicting_use=0.0,
            unsupported_personal_claim=0.0,
            memory_omission=0.0,
            strategy_overuse=strategy_overuse,
            strategy_omission=strategy_omission,
        ),
        observed_input_tokens=180 if use_strategy else 100,
        retrieval_calls=1 if use_strategy else 0,
        judge_families=["google_gemini", "deepseek"],
        judge_count=2,
        max_dimension_mad=0.05,
        dimension_mad=_dimension_mad(),
        label_reliable=True,
        composite_spec_version=spec.version,
        composite_weights_sha256=composite_weights_digest(spec),
        provenance={"shared_quality_label_weight": 1.0},
    )


def test_all_three_candidate_models_train_and_predict() -> None:
    states = [_state(index) for index in range(1, 4)]
    step0 = {state.state_id: _step0(state) for state in states}
    labels = [
        _label(state, action)
        for state in states
        for action in state.allowed_actions
    ]
    rule_actions = {state.state_id: "M0+RS" for state in states}

    for mode in (
        OutcomeMode.ABSOLUTE,
        OutcomeMode.STATE_CENTERED,
        OutcomeMode.RULE_RELATIVE,
    ):
        model = PMV16OutcomeModel.train(
            mode=mode,
            states=states,
            labels=labels,
            step0_by_state=step0,
            rule_action_by_state=(
                rule_actions if mode is OutcomeMode.RULE_RELATIVE else None
            ),
            n_models=1,
            seed=101,
        )
        predictions = model.predict_members(
            state=states[0],
            step0=step0[states[0].state_id],
            action_id="M0+RS",
            rule_action="M0+RS",
        )
        assert set(predictions) == {
            *ResponseDimensions.model_fields,
            *RiskDimensions.model_fields,
        }
        assert all(
            values.shape == (1,) and np.isfinite(values).all()
            for values in predictions.values()
        )


def test_conservative_override_returns_legal_action() -> None:
    states = [_state(index) for index in range(1, 4)]
    step0 = {state.state_id: _step0(state) for state in states}
    labels = [
        _label(state, action)
        for state in states
        for action in state.allowed_actions
    ]
    model = PMV16OutcomeModel.train(
        mode=OutcomeMode.STATE_CENTERED,
        states=states,
        labels=labels,
        step0_by_state=step0,
        n_models=1,
        seed=202,
    )
    policy = ConservativeResidualPolicy(
        outcome_model=model,
        rule_router=StrongTransparentRouter(TransparentRouterConfig()),
        override_config=OverrideConfig(),
    )
    decision = policy.choose(states[0], step0[states[0].state_id])
    assert decision.chosen_action in states[0].allowed_actions
    assert decision.rule_action in states[0].allowed_actions
