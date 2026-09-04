from __future__ import annotations

from metacom_pm.v1_5_strategy_rag_repair import (
    latest_visible_turn_is_seeker,
    repaired_observable_opportunity_flags,
)


def _dialogue(last_speaker: str, last_text: str):
    return [
        {"speaker": "seeker", "content": "Earlier context."},
        {"speaker": last_speaker, "content": last_text},
    ]


def test_repair_rejects_supporter_ended_state() -> None:
    dialogue = _dialogue("supporter", "What else happened?")
    assert not latest_visible_turn_is_seeker(dialogue)
    flags = repaired_observable_opportunity_flags(
        current_user_text="Earlier context.",
        recent_user_text="Earlier context.",
        visible_dialogue=dialogue,
    )
    assert flags["ordinary_rag_hard_off"]
    assert "latest_visible_turn_not_seeker" in flags[
        "ordinary_rag_hard_off_reasons"
    ]


def test_repair_routes_violence_and_substance_child_safety_out() -> None:
    for latest, recent, expected in (
        (
            "Part of me wants her and him to die.",
            "Part of me wants her and him to die.",
            "active_violence_or_self_harm_signal",
        ),
        (
            "I am not sure if she is clean.",
            "She used opioids and has three kids. I am not sure if she is clean.",
            "substance_and_dependent_safety_signal",
        ),
    ):
        flags = repaired_observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=recent,
            visible_dialogue=_dialogue("seeker", latest),
        )
        assert flags["ordinary_rag_hard_off"]
        assert expected in flags["ordinary_rag_hard_off_reasons"]


def test_repair_routes_resource_request_and_routine_closing_out() -> None:
    for text, expected in (
        (
            "Can you recommend a good site to find affordable services?",
            "factual_or_resource_request_outside_technique_bank",
        ),
        (
            "The one thing I have is time. Thanks for talking through this with me.",
            "stop_phatic_or_routine_closing",
        ),
    ):
        flags = repaired_observable_opportunity_flags(
            current_user_text=text,
            visible_dialogue=_dialogue("seeker", text),
        )
        assert flags["ordinary_rag_hard_off"]
        assert expected in flags["ordinary_rag_hard_off_reasons"]


def test_repair_preserves_ordinary_advice_opportunity() -> None:
    text = "I feel overwhelmed. Could you give me one small suggestion?"
    flags = repaired_observable_opportunity_flags(
        current_user_text=text,
        visible_dialogue=_dialogue("seeker", text),
    )
    assert not flags["ordinary_rag_hard_off"]
    assert flags["advice_welcome"]
