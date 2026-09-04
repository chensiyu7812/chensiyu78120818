from __future__ import annotations

import pytest

from metacom_pm.v1_5_observation_contract import (
    FACTOR_NAMES,
    SemanticObservationRecord,
    factor_gold_from_evidence_codes,
    transparent_semantic_observation,
)


def test_evidence_codes_produce_partial_not_closed_world_factor_gold() -> None:
    gold = factor_gold_from_evidence_codes(
        ["WRONG_ENTITY_OR_GOAL", "GENERIC_OR_NO_INCREMENT"]
    )
    # The historical combined code cannot tell an entity error from a goal
    # error, so owner/time must remain unknown rather than receive false gold.
    assert gold["owner_time_entity_valid"] is None
    assert gold["goal_function_fit"] == 0
    assert gold["specific_increment"] == 0
    assert gold["boundary_burden_compatible"] is None
    assert gold["currently_nonredundant"] is None


def test_redundancy_is_separate_from_safety_and_goal_fit() -> None:
    gold = factor_gold_from_evidence_codes(
        [
            "CURRENT_GOAL_FIT",
            "OWNER_AND_TIME_VALID",
            "SAFE_AND_BOUNDARY_COMPATIBLE",
            "CURRENTLY_REDUNDANT",
        ]
    )
    assert gold["goal_function_fit"] == 1
    assert gold["boundary_burden_compatible"] == 1
    assert gold["currently_nonredundant"] == 0
    assert gold["specific_increment"] == 0


def test_observation_record_rejects_gold_or_outcome_access() -> None:
    scores = {factor: 0.5 for factor in FACTOR_NAMES}
    with pytest.raises(ValueError, match="forbidden supervision"):
        SemanticObservationRecord(
            state_id="state_001",
            component="ME",
            factor_builder_protocol="visible-input-factor-builder-v1",
            factor_scores=scores,
            final_decision_read=True,
        )


def test_valid_observation_contains_complete_bounded_factor_surface() -> None:
    record = SemanticObservationRecord(
        state_id="state_002",
        component="RS",
        factor_builder_protocol="visible-input-factor-builder-v1",
        factor_scores={factor: 0.5 for factor in FACTOR_NAMES},
    )
    assert set(record.factor_scores) == set(FACTOR_NAMES)


def test_transparent_mp_observation_separates_redundant_preference() -> None:
    observation = transparent_semantic_observation(
        state_id="state_mp",
        component="MP",
        current_user_text="A brief factual reminder would help.",
        visible_dialogue=[],
        candidate_present=True,
        candidate_text="Stable Preference Response Format: prefers reminders as one concise factual sentence",
        candidate_subtype="MP_PREFERENCE",
    )
    assert observation.factor_scores["goal_function_fit"] == 1.0
    assert observation.factor_scores["currently_nonredundant"] == 0.0
    assert observation.factor_scores["specific_increment"] == 0.0


def test_transparent_me_observation_requires_reusable_action_and_result() -> None:
    positive = transparent_semantic_observation(
        state_id="state_me_positive",
        component="ME",
        current_user_text="The budget issue is back and I welcome one small option.",
        visible_dialogue=[],
        candidate_present=True,
        candidate_text="I wrote one budget note, and it helped make the next step easier.",
        candidate_subtype="ME_REUSABLE_OUTCOME",
    )
    context = transparent_semantic_observation(
        state_id="state_me_context",
        component="ME",
        current_user_text="The budget issue is back and I welcome one small option.",
        visible_dialogue=[],
        candidate_present=True,
        candidate_text="The budget issue was present in the background that week.",
        candidate_subtype="ME_CONTEXT_EVENT",
    )
    assert positive.factor_scores["specific_increment"] == 1.0
    assert context.factor_scores["specific_increment"] == 0.0


def test_transparent_rs_observation_honors_nonrepeat_boundary() -> None:
    observation = transparent_semantic_observation(
        state_id="state_rs",
        component="RS",
        current_user_text="The last focused question already identified it, so do not repeat it.",
        visible_dialogue=[],
        candidate_present=True,
        candidate_text=(
            "support_move: Ask one focused question to clarify the feeling. "
            "when_to_use: Use when the feeling is unclear. "
            "when_not_to_use: Do not use after a no-question boundary."
        ),
        candidate_subtype="RS_ATOMIC_MOVE",
    )
    assert observation.factor_scores["boundary_burden_compatible"] == 0.0
    assert observation.factor_scores["currently_nonredundant"] == 0.0
