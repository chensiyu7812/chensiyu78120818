import json
from pathlib import Path

from metacom_pm.v1_5_observation_contract import (
    StructuredCandidateMetadata,
    structured_hard_gate_observation_v3,
    structured_policy_observation_v4,
    transparent_semantic_observation_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def _observe(current: str) -> dict[str, float]:
    record = transparent_semantic_observation_v2(
        state_id="test-state",
        component="ME",
        current_user_text=current,
        visible_dialogue=[],
        candidate_present=True,
        candidate_text=(
            "For the earlier issue, I wrote one short note and it helped "
            "identify one reversible next step."
        ),
        candidate_subtype="ME_REUSABLE_OUTCOME",
    )
    return dict(record.factor_scores)


def test_v2_scores_boundary_goal_and_increment_independently() -> None:
    scores = _observe(
        "I am ready to consider one past action-result now. "
        "For this turn, do not use one optional past approach. "
        "The exact candidate detail is absent from the visible current exchange."
    )
    assert scores["goal_function_fit"] == 1.0
    assert scores["boundary_burden_compatible"] == 0.0
    assert scores["specific_increment"] == 1.0


def test_explicit_permission_is_not_misread_as_leave_out() -> None:
    scores = _observe(
        "My actual question is which factual detail was recorded earlier, not which "
        "past action to reuse. My present boundary leaves room for one optional past "
        "approach. Nothing new is needed from the candidate because I just said it."
    )
    assert scores["goal_function_fit"] == 0.0
    assert scores["boundary_burden_compatible"] == 1.0
    assert scores["specific_increment"] == 0.0


def test_independence_repair_passed_fit_but_failed_frozen_confirmation() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_v3_observation_independence_repair_v2/qualification_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "PASS_FIT_FAIL_ONE_SHOT_CONFIRMATION"
    assert report["fit_pass"] is True
    assert all(
        row["status"] == "PASS" for row in report["fit_factor_reports"].values()
    )
    assert report["one_shot_confirmation"]["one_shot_consumed"] is True
    assert report["one_shot_confirmation"]["used_for_rule_or_threshold_selection"] is False
    assert report["one_shot_confirmation"]["composite_global"]["balanced_accuracy"] == 0.71875
    assert report["one_shot_confirmation"]["composite_global"]["specificity"] == 0.5
    assert report["step1_training_authorized"] is False
    assert report["step2_qualification_authorized"] is False
    assert report["response_or_outcome_read"] is False
    assert report["external_lockbox_read"] is False


def test_structured_hard_gate_amendment_preserves_full_pm_scope() -> None:
    contract = json.loads(
        (
            ROOT
            / "data/pm_v1_5_contracts/v3_observation_structured_hard_gate_amendment_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["evidence"]["confirmation_consumed_once"] is True
    assert contract["scope_boundary"]["eligibility_is_step1_gold"] is False
    assert contract["structured_hard_gates"]["owner_identity"]["operation"] == "exact equality"
    assert contract["bounded_semantic_factors"]["boundary_burden_compatible"]["unknown_policy"] == "OFF"
    assert contract["bounded_semantic_factors"]["specific_increment"]["unknown_policy"] == "OFF"
    assert contract["runtime_invariants"]["components_preserved"] == ["MP", "MS", "ME", "RS"]
    assert contract["runtime_invariants"]["legal_actions_preserved"] == 16
    assert contract["claim_effect"]["primary_qrc_claim_changed"] is False
    assert contract["validation_and_stop_rule"][
        "consumed_confirmation_may_not_be_reused_for_model_or_rule_selection"
    ] is True


def test_v3_uses_backend_owner_and_version_as_hard_gates() -> None:
    observation = structured_hard_gate_observation_v3(
        state_id="structured-stale",
        component="MP",
        current_user_text=(
            "One relevant stored profile detail would stay within my limits. "
            "The exact candidate detail is absent from the visible current exchange."
        ),
        visible_dialogue=[],
        candidate_text="Stable Practical Constraint: one private hour after dinner.",
        candidate_subtype="MP_PROFILE",
        metadata=StructuredCandidateMetadata(
            candidate_present=True,
            state_owner_id="user-a",
            candidate_owner_id="user-a",
            candidate_active=False,
            candidate_superseded=True,
        ),
    )
    assert observation.gate_decisions["owner_time_entity_valid"] == "deny"
    assert observation.runtime_eligible is False
    assert observation.eligibility_is_step1_gold is False


def test_v3_keeps_unknown_distinct_from_negative_and_fails_closed() -> None:
    observation = structured_hard_gate_observation_v3(
        state_id="structured-unknown",
        component="ME",
        current_user_text="I am ready to consider one past action-result now.",
        visible_dialogue=[],
        candidate_text="I wrote one short note and it helped clarify one next step.",
        candidate_subtype="ME_REUSABLE_OUTCOME",
        metadata=StructuredCandidateMetadata(
            candidate_present=True,
            state_owner_id="user-a",
            candidate_owner_id="user-a",
            candidate_active=True,
            candidate_superseded=False,
        ),
    )
    assert observation.gate_decisions["owner_time_entity_valid"] == "allow"
    assert observation.gate_decisions["boundary_burden_compatible"] == "unknown"
    assert observation.gate_decisions["specific_increment"] == "unknown"
    assert observation.runtime_eligible is False
    assert observation.unknown_forced_off is True


def test_v3_allows_candidate_only_when_all_four_gates_are_positive() -> None:
    observation = structured_hard_gate_observation_v3(
        state_id="structured-allow",
        component="ME",
        current_user_text=(
            "I am ready to consider one past action-result now. "
            "My present boundary leaves room for one optional past approach. "
            "The exact candidate detail is absent from the visible current exchange."
        ),
        visible_dialogue=[],
        candidate_text="I wrote one short note and it helped clarify one next step.",
        candidate_subtype="ME_REUSABLE_OUTCOME",
        metadata=StructuredCandidateMetadata(
            candidate_present=True,
            state_owner_id="user-a",
            candidate_owner_id="user-a",
            candidate_active=True,
            candidate_superseded=False,
        ),
    )
    assert set(observation.gate_decisions.values()) == {"allow"}
    assert observation.runtime_eligible is True
    assert observation.unknown_forced_off is False


def test_v4_routes_semantic_unknown_to_step1_instead_of_fabricated_off() -> None:
    observation = structured_policy_observation_v4(
        state_id="policy-unknown",
        component="ME",
        current_user_text="I am trying to describe what feels difficult today.",
        visible_dialogue=[],
        candidate_text="I wrote one short note and it helped clarify one next step.",
        candidate_subtype="ME_REUSABLE_OUTCOME",
        metadata=StructuredCandidateMetadata(
            candidate_present=True,
            state_owner_id="user-a",
            candidate_owner_id="user-a",
            candidate_active=True,
            candidate_superseded=False,
        ),
    )
    assert observation.step1_candidate_admissible is True
    assert observation.hard_denial_reasons == ()
    assert "unknown" in observation.semantic_feature_decisions.values()
    assert observation.eligibility_is_step1_gold is False


def test_v4_hard_denies_wrong_owner_and_explicit_boundary_conflict() -> None:
    wrong_owner = structured_policy_observation_v4(
        state_id="policy-owner",
        component="MS",
        current_user_text="Please remind me of the earlier factual observation.",
        visible_dialogue=[],
        candidate_text="The earlier session recorded one workable time.",
        candidate_subtype="MS_SESSION",
        metadata=StructuredCandidateMetadata(
            candidate_present=True,
            state_owner_id="user-a",
            candidate_owner_id="user-b",
            candidate_active=True,
            candidate_superseded=False,
        ),
    )
    assert wrong_owner.step1_candidate_admissible is False
    assert wrong_owner.hard_denial_reasons == ("owner_time_entity_valid",)

    refused = structured_policy_observation_v4(
        state_id="policy-boundary",
        component="RS",
        current_user_text="For this turn, do not give advice.",
        visible_dialogue=[],
        candidate_text="support_move: Offer one suggestion when_to_use: advice is welcome when_not_to_use: advice is refused",
        candidate_subtype="RS_ATOMIC_MOVE",
        metadata=StructuredCandidateMetadata(
            candidate_present=True,
            state_owner_id="user-a",
            candidate_owner_id="SHARED_STRATEGY_BANK",
            candidate_active=True,
            candidate_superseded=False,
        ),
    )
    assert refused.step1_candidate_admissible is False
    assert refused.hard_denial_reasons == ("explicit_boundary_compatible",)


def test_v2_policy_contract_preserves_effect_learning_target() -> None:
    contract = json.loads(
        (
            ROOT
            / "data/pm_v1_5_contracts/v3_observation_structured_hard_gate_amendment_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["trigger"]["human_labels_read"] == 0
    assert contract["runtime_rule"]["no_hard_denial"] == (
        "candidate reaches its component-effect head"
    )
    assert contract["training_rule"]["unknown_candidates_may_receive_same-state_on_off_treatments"] is True
    assert contract["training_rule"]["eligibility_or_construction_intent_is_step1_gold"] is False
    assert contract["scope"]["legal_actions"] == 16
