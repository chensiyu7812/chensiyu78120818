from __future__ import annotations

from pathlib import Path

from metacom_pm.io import iter_jsonl
from metacom_pm.v1_5_strategy_rag_v4 import (
    assess_v4_card_applicability,
    build_v4_candidate_cards,
    eligible_families,
    observable_card_cues,
    observable_opportunity_flags,
    rank_applicable_v4_cards,
    selected_execution_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def _core_cards():
    return [
        dict(row)
        for row in iter_jsonl(
            ROOT
            / "outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1"
            / "strategy_cards_v3_core_candidate.jsonl"
        )
    ]


def test_v4_expands_fifty_core_moves_into_one_hundred_profiles():
    cards = build_v4_candidate_cards(_core_cards())
    assert len(cards) == 100
    assert len({row["core_submove_id"] for row in cards}) == 50
    assert sum(row["execution_profile"] == "minimal" for row in cards) == 50
    assert sum(row["execution_profile"] == "dialogic" for row in cards) == 50
    assert not any(row["eligible_for_formal_rs"] for row in cards)


def test_v4_phatic_and_routine_closing_have_no_rs_opportunity():
    for text in (
        "hello",
        "Thanks.",
        "bye",
        "Have a good night!",
        "I hope you have a great Christmas and new year. Stay safe!",
    ):
        flags = observable_opportunity_flags(current_user_text=text)
        assert flags["pure_phatic"]
        assert eligible_families(flags) == ()


def test_v4_ambiguous_give_up_language_is_hard_off_for_strategy_rag():
    for text in (
        "I feel like I just need to give up.",
        "I feel like giving up.",
        "I can't go on.",
    ):
        flags = observable_opportunity_flags(current_user_text=text)
        assert flags["active_high_stakes"]
        assert eligible_families(flags) == ()


def test_v4_advice_and_low_burden_select_suggestion_minimal_profile():
    flags = observable_opportunity_flags(
        current_user_text=(
            "I feel overwhelmed. Could you give me one small suggestion?"
        )
    )
    assert "Providing Suggestions" in eligible_families(flags)
    assert selected_execution_profile(flags) == "minimal"


def test_v4_recognizes_implicit_practical_request_but_not_factual_idea_query():
    practical = observable_opportunity_flags(
        current_user_text=(
            "I don't know what else to do to get a job."
        )
    )
    factual = observable_opportunity_flags(
        current_user_text=(
            "Any idea how would it work? Would the instruments be delayed?"
        )
    )
    assert practical["advice_welcome"]
    assert not factual["advice_welcome"]


def test_v4_bare_help_me_request_does_not_imply_advice_permission():
    flags = observable_opportunity_flags(
        current_user_text=(
            "Well, can you help me? Why did you step out?"
        )
    )
    assert not flags["advice_welcome"]
    assert "Providing Suggestions" not in eligible_families(flags)


def test_v4_emotion_cue_requires_user_experience_not_legal_word_overlap():
    legal = observable_opportunity_flags(
        current_user_text=(
            "The company was found guilty and the judge awarded damages."
        )
    )
    emotional = observable_opportunity_flags(
        current_user_text=(
            "I am very worried that I will fail the class."
        )
    )
    assert not legal["emotion_visible"]
    assert emotional["emotion_visible"]


def test_v4_listen_only_excludes_question_and_suggestion():
    flags = observable_opportunity_flags(
        current_user_text=(
            "I am upset and only want you to listen. I do not want advice."
        )
    )
    families = eligible_families(flags)
    assert "Question" not in families
    assert "Providing Suggestions" not in families
    assert "Reflection of feelings" in families
    assert selected_execution_profile(flags) == "minimal"


def _card(core_submove_id: str, profile: str = "dialogic"):
    return next(
        row
        for row in build_v4_candidate_cards(_core_cards())
        if row["core_submove_id"] == core_submove_id
        and row["execution_profile"] == profile
    )


def test_v4_card_applicability_blocks_open_invitation_after_focus_is_visible():
    text = (
        "I am overwhelmed by caring for my dad while working and studying. "
        "I am not sure which responsibility to handle first."
    )
    flags = observable_opportunity_flags(current_user_text=text)
    decision = assess_v4_card_applicability(
        _card("question_open_invitation", profile="minimal"),
        current_user_text=text,
        flags=flags,
    )
    meaning = assess_v4_card_applicability(
        _card("question_clarify_meaning", profile="minimal"),
        current_user_text=text,
        flags=flags,
    )
    assert not decision["eligible"]
    assert "required_visible_cue_absent" in decision["reasons"]
    assert meaning["eligible"]
    assert meaning["compatibility_tier"] == 3


def test_v4_card_applicability_requires_explicit_reflection_construct():
    text = (
        "I have to stay strong for the people I supervise, but it leaves me "
        "feeling exhausted."
    )
    flags = observable_opportunity_flags(current_user_text=text)
    cues = observable_card_cues(current_user_text=text, flags=flags)
    mixed = assess_v4_card_applicability(
        _card("reflection_mixed_feelings", profile="minimal"),
        current_user_text=text,
        flags=flags,
    )
    value = assess_v4_card_applicability(
        _card("reflection_value_tension", profile="minimal"),
        current_user_text=text,
        flags=flags,
    )
    lingering = assess_v4_card_applicability(
        _card("reflection_lingering_feeling", profile="minimal"),
        current_user_text=text,
        flags=flags,
    )
    assert cues["value_tension_visible"]
    assert value["eligible"]
    assert not mixed["eligible"]
    assert not lingering["eligible"]


def test_v4_card_applicability_prefers_specific_suggestion_prerequisite():
    text = "Do you have any suggestions to improve sleep the night before?"
    flags = observable_opportunity_flags(current_user_text=text)
    environment = assess_v4_card_applicability(
        _card("suggestion_adjust_environment"),
        current_user_text=text,
        flags=flags,
    )
    communication = assess_v4_card_applicability(
        _card("suggestion_communication_opening"),
        current_user_text=text,
        flags=flags,
    )
    fallback = assess_v4_card_applicability(
        _card("suggestion_single_microstep"),
        current_user_text=text,
        flags=flags,
    )
    assert environment["compatibility_tier"] == 3
    assert not communication["eligible"]
    assert fallback["compatibility_tier"] == 2


def test_v4_card_applicability_fails_closed_on_domain_decisions():
    text = "Can you recommend what medication dose I should take?"
    flags = observable_opportunity_flags(current_user_text=text)
    decision = assess_v4_card_applicability(
        _card("suggestion_single_microstep"),
        current_user_text=text,
        flags=flags,
    )
    assert not decision["eligible"]
    assert "domain_decision_outside_technique_bank" in decision["reasons"]


def test_v4_applicability_tier_precedes_lexical_similarity():
    text = "Do you have any suggestions to improve sleep the night before?"
    flags = observable_opportunity_flags(current_user_text=text)
    cards = [
        row
        for row in build_v4_candidate_cards(_core_cards())
        if row["source_support"]["provisional_source_support_pass"]
    ]
    ranked = rank_applicable_v4_cards(
        query=text,
        current_user_text=text,
        recent_user_text=text,
        flags=flags,
        cards=cards,
    )
    assert ranked
    assert ranked[0]["core_submove_id"] == "suggestion_adjust_environment"
    assert ranked[0]["compatibility_tier"] == 3


def test_v4_emotionally_exhausted_is_not_a_redundancy_control():
    flags = observable_opportunity_flags(
        current_user_text="I'm so emotionally exhausted all the time lately."
    )
    assert flags["emotion_visible"]
    assert "Reflection of feelings" in eligible_families(flags)


def test_v4_suggestion_scope_blocks_police_and_relationship_decisions():
    for text in (
        "Any suggestions before she calls the cops and presses charges?",
        "I want to end this relationship. How do I get out?",
    ):
        flags = observable_opportunity_flags(current_user_text=text)
        decision = assess_v4_card_applicability(
            _card("suggestion_single_microstep"),
            current_user_text=text,
            recent_user_text=text,
            flags=flags,
        )
        assert not decision["eligible"]
        assert "domain_decision_outside_technique_bank" in decision["reasons"]


def test_v4_grounded_hope_requires_current_positive_evidence():
    worry = "I am worried because I have a family and money is tight."
    plan = "Maybe I will try a short call with my friends this weekend."
    worry_flags = observable_opportunity_flags(current_user_text=worry)
    plan_flags = observable_opportunity_flags(current_user_text=plan)
    rejected = assess_v4_card_applicability(
        _card("affirmation_grounded_hope"),
        current_user_text=worry,
        flags=worry_flags,
    )
    accepted = assess_v4_card_applicability(
        _card("affirmation_grounded_hope"),
        current_user_text=plan,
        flags=plan_flags,
    )
    assert not rejected["eligible"]
    assert accepted["eligible"]
