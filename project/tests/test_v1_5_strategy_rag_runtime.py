from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.contracts import StrategyCard
from metacom_pm.prompts import generation_messages
from metacom_pm.v1_5_strategy_rag_runtime import (
    QualifiedStrategyRAG,
    eligible_moves,
    observable_flags,
)


ROOT = Path(__file__).resolve().parents[1]
BANK = (
    ROOT
    / "outputs/pm_v1_5_strategy_g1_final_bank_v1"
    / "strategy_cards_g1_runtime.jsonl"
)


def _rag() -> QualifiedStrategyRAG:
    cards = [
        StrategyCard.model_validate(json.loads(line))
        for line in BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return QualifiedStrategyRAG(cards)


def test_strategy_rag_is_fail_closed_on_boundaries() -> None:
    rag = _rag()
    high_stakes = rag.retrieve(
        [{"speaker": "seeker", "content": "I want to hurt myself tonight."}]
    )
    assert high_stakes.status == "off_active_high_stakes"
    assert not high_stakes.selected_cards

    no_permission = rag.retrieve(
        [{"speaker": "seeker", "content": "Please just listen; no advice."}]
    )
    assert no_permission.status == "off_no_explicit_opportunity"
    assert not no_permission.selected_cards


def test_strategy_rag_selects_one_card_for_advice_but_not_routine_stop() -> None:
    rag = _rag()
    advice = rag.retrieve(
        [{"speaker": "seeker", "content": "What should I do? One idea is enough."}]
    )
    assert advice.status == "retrieved_top1"
    assert [card.strategy_label for card in advice.selected_cards] == [
        "AM10_offer_one_optional_micro_step"
    ]

    stop = rag.retrieve([{"speaker": "seeker", "content": "Bye"}])
    assert stop.status == "off_no_explicit_opportunity"
    assert not stop.selected_cards


def test_runtime_matches_saved_real_g2_lexical_decisions() -> None:
    rag = _rag()
    rows = [
        json.loads(line)
        for line in (
            ROOT
            / "outputs/pm_v1_5_strategy_g2_real_qualification_v1"
            / "real_query_results.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for row in rows:
        recent_dialogue = []
        for line in row["query_text"].splitlines():
            role, content = line.split(": ", 1)
            recent_dialogue.append(
                {
                    "speaker": "seeker" if role == "User" else "supporter",
                    "content": content,
                }
            )
        flags = observable_flags(recent_dialogue)
        assert flags == row["observable_flags"]
        expected_eligible = [
            move
            for move in row["eligible_move_ids"]
            if move != "AM14_supportive_transition"
        ]
        assert eligible_moves(flags) == expected_eligible
        ranked = rag.rank_eligible(row["query_text"], expected_eligible)
        actual = (
            ranked[0][1].strategy_label
            if ranked and ranked[0][0] >= rag.minimum_score
            else None
        )
        expected_top1 = row["lexical_top1"]
        if expected_top1 == "AM14_supportive_transition":
            expected_top1 = None
        assert actual == expected_top1


def test_prompt_compiler_omits_withdrawn_source_example_text() -> None:
    rag = _rag()
    decision = rag.retrieve(
        [{"speaker": "seeker", "content": "What should I do? One idea is enough."}]
    )
    # A tiny state-shaped test double is sufficient for the pure prompt compiler.
    class Turn:
        role = "assistant"
        content = "What brings you here?"

    class State:
        current_session_history = [Turn()]
        current_session_summary = ""
        current_user_text = "What should I do? One idea is enough."

    messages = generation_messages(State(), [], decision.selected_cards)
    prompt = messages[-1]["content"]
    assert "No source response is provided" not in prompt
    assert "Example style" not in prompt
    assert "Offer one concrete, low-risk, low-burden next step" in prompt
