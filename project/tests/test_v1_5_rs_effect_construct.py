from __future__ import annotations

from metacom_pm.v1_5_strategy_rag_repair import (
    effect_study_observable_flags,
    effect_study_rank_applicable_cards,
)


def _card(family: str, core: str) -> dict[str, object]:
    return {
        "card_id": f"card_{core}",
        "core_submove_id": core,
        "strategy_family": family,
        "execution_profile": "dialogic",
        "retrieval_text": f"{family} {core}",
    }


def test_advice_request_keeps_suggestions_and_removes_questions() -> None:
    dialogue = [
        {"speaker": "seeker", "content": "I feel stuck. Any advice?"},
    ]
    cards = [
        _card("Providing Suggestions", "suggestion_single_microstep"),
        _card("Question", "question_clarify_goal"),
    ]
    ranked = effect_study_rank_applicable_cards(
        query="one suggestion advice",
        current_user_text=dialogue[-1]["content"],
        recent_user_text=dialogue[-1]["content"],
        visible_dialogue=dialogue,
        cards=cards,
    )
    assert ranked
    assert {row["strategy_family"] for row in ranked} == {
        "Providing Suggestions"
    }


def test_listen_only_is_not_a_component_wide_hard_off() -> None:
    dialogue = [
        {
            "speaker": "seeker",
            "content": "I feel very sad. I only need you to listen.",
        }
    ]
    flags = effect_study_observable_flags(
        current_user_text=dialogue[-1]["content"],
        recent_user_text=dialogue[-1]["content"],
        visible_dialogue=dialogue,
    )
    assert flags["listen_only"]
    assert not flags["ordinary_rag_hard_off"]
    cards = [
        _card("Reflection of feelings", "reflection_value_tension"),
        _card("Question", "question_clarify_goal"),
        _card("Providing Suggestions", "suggestion_single_microstep"),
    ]
    ranked = effect_study_rank_applicable_cards(
        query="reflect sad emotion",
        current_user_text=dialogue[-1]["content"],
        recent_user_text=dialogue[-1]["content"],
        visible_dialogue=dialogue,
        cards=cards,
    )
    assert all(
        row["strategy_family"]
        not in {"Question", "Providing Suggestions"}
        for row in ranked
    )


def test_explicit_legal_process_is_v1_5_scope_off() -> None:
    dialogue = [
        {
            "speaker": "seeker",
            "content": "The judge denied my restraining order in court.",
        }
    ]
    flags = effect_study_observable_flags(
        current_user_text=dialogue[-1]["content"],
        recent_user_text=dialogue[-1]["content"],
        visible_dialogue=dialogue,
    )
    assert flags["ordinary_rag_hard_off"]
    assert "legal_process_outside_v1_5_ordinary_bank" in flags[
        "ordinary_rag_hard_off_reasons"
    ]
