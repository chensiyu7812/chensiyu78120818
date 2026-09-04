from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.v1_5_v5_3_contribution_slot_features import (
    me_contribution_slots,
    mp_contribution_slots,
    ms_contribution_slots,
    rs_contribution_slots,
)


def _item(text: str, *, source: MemorySource, session: int = 10) -> MemoryItem:
    return MemoryItem(memory_id="mem_" + "a" * 24, source=source, created_session=session, text=text)


# 2026-08-06: these four ME cases are the exact real, double-machine-
# verified cases from 59_me_pm_effect_fit_pilot_v1_5.py's ON/action-
# declined/context-only/redundant families (8/8 matched when checked
# directly against this module before committing -- see
# PM_V1_5_V5_3_OPEN_PROBLEMS_AND_SEQUENCE_20260806_ZH.md item 5).


def test_me_on_case_action_invited_with_valid_candidate() -> None:
    candidate = "When dealing with a workplace conflict with a coworker, I wrote down the hardest moment before responding, and it helped me feel calmer."
    item = _item(candidate, source=MemorySource.ME)
    obs = me_contribution_slots(
        current_user_text="I could really use some advice for handling a workplace conflict with a coworker. What should I do?",
        candidate_text=candidate, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.past_action_result is True
    assert obs.current_action_invitation is True
    assert obs.current_action_readiness == "INVITES_ACTION"
    assert obs.current_redundant is False


def test_me_off_case_action_declined() -> None:
    candidate = "When dealing with a workplace conflict with a coworker, I wrote down the hardest moment before responding, and it helped me feel calmer."
    item = _item(candidate, source=MemorySource.ME)
    obs = me_contribution_slots(
        current_user_text="I just want to vent about a workplace conflict with a coworker right now. I don't want any advice, I just need you to listen.",
        candidate_text=candidate, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.past_action_result is True
    assert obs.current_action_invitation is False
    assert obs.current_action_readiness == "DECLINES_ACTION"


def test_me_off_case_context_only_candidate_has_no_action_result() -> None:
    candidate = "In a prior session, the user last spring described a private incident involving a workplace conflict with a coworker."
    item = _item(candidate, source=MemorySource.ME)
    obs = me_contribution_slots(
        current_user_text="I could really use some advice for handling a workplace conflict with a coworker. What should I do?",
        candidate_text=candidate, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.past_action_result is False


def test_me_off_case_current_turn_already_states_the_outcome() -> None:
    candidate = "When dealing with a workplace conflict with a coworker, I wrote down the hardest moment before responding, and it helped me feel calmer."
    item = _item(candidate, source=MemorySource.ME)
    obs = me_contribution_slots(
        current_user_text=(
            "I could really use some advice for handling a workplace conflict with a "
            "coworker. What should I do? Actually, I already figured it out myself -- "
            "it helped me feel calmer, so I've got that covered."
        ),
        candidate_text=candidate, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.current_redundant is True


def test_me_candidate_absent_reports_none_not_false() -> None:
    obs = me_contribution_slots(
        current_user_text="What should I do?", candidate_text=None,
        source_items=[], selected_items=[], session_index=45,
    )
    assert obs.candidate_present is False
    assert obs.past_action_result is None


def test_me_unknown_action_readiness_is_preserved_not_negative_gold() -> None:
    candidate = "When work felt crowded, I wrote one note before replying, and it helped me feel calmer."
    item = _item(candidate, source=MemorySource.ME)
    obs = me_contribution_slots(
        current_user_text="Work has felt crowded this week.",
        candidate_text=candidate,
        source_items=[item],
        selected_items=[item],
        session_index=45,
    )
    assert obs.current_action_readiness == "UNKNOWN"
    assert obs.current_action_invitation is False


def test_ms_continuity_request_detected() -> None:
    candidate = "The user mentioned last session that they had already tried talking to their manager about the workload, but nothing changed afterward."
    item = _item(candidate, source=MemorySource.MS)
    obs = ms_contribution_slots(
        current_user_text="Like I said before, the same issue with my manager is still going on.",
        candidate_text=candidate, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.has_specific_prior_observation is True
    assert obs.continuity_request is True


def test_ms_no_continuity_request_on_a_fresh_topic() -> None:
    candidate = "The user mentioned last session that they had already tried talking to their manager about the workload, but nothing changed afterward."
    item = _item(candidate, source=MemorySource.MS)
    obs = ms_contribution_slots(
        current_user_text="Something new happened today that I want to talk about.",
        candidate_text=candidate, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.continuity_request is False


def test_mp_preference_applies_unless_listen_only() -> None:
    item = _item("The user prefers acknowledgement before suggestions, one gentle question at a time.", source=MemorySource.MP)
    obs = mp_contribution_slots(
        current_user_text="What should I do about this?", candidate_text=item.text,
        candidate_is_preference=True, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.preference_applies_to_response_act is True
    assert obs.profile_goal_needs_advice_or_arrangement is None


def test_mp_profile_constraint_needs_advice_context() -> None:
    item = _item("The user works night shifts and is unavailable for suggestions before 3pm.", source=MemorySource.MP)
    obs = mp_contribution_slots(
        current_user_text="What should I do about this?", candidate_text=item.text,
        candidate_is_preference=False, source_items=[item], selected_items=[item], session_index=45,
    )
    assert obs.profile_goal_needs_advice_or_arrangement is True
    assert obs.preference_applies_to_response_act is None


def test_rs_card_precondition_reuses_real_eligibility_runtime() -> None:
    dialogue = [{"role": "user", "content": "What should I do about this? Any advice?"}]
    obs = rs_contribution_slots(
        current_user_text="What should I do about this? Any advice?", recent_dialogue=dialogue,
        move_id="AM10_offer_one_optional_micro_step", already_executed_last_turn=False,
    )
    assert obs.card_precondition_met is True


def test_rs_card_precondition_false_when_move_not_eligible() -> None:
    dialogue = [{"role": "user", "content": "I just want to vent, please don't give me advice."}]
    obs = rs_contribution_slots(
        current_user_text="I just want to vent, please don't give me advice.", recent_dialogue=dialogue,
        move_id="AM10_offer_one_optional_micro_step", already_executed_last_turn=False,
    )
    assert obs.card_precondition_met is False
