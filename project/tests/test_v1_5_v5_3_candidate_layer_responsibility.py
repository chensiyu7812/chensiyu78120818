from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.contracts import StrategyCard
from metacom_pm.v1_5_v5_3_candidate_layer_responsibility import (
    rs_mechanical_candidate_pool,
    rs_shared_candidate_top1,
    rs_transparent_rule_top1,
)

ROOT = Path(__file__).resolve().parents[1]
CARDS = [
    StrategyCard(**row)
    for row in (
        json.loads(line)
        for line in (ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl").open()
        if line.strip()
    )
]


def _dialogue(text: str) -> list[dict]:
    return [{"speaker": "seeker", "content": text}]


def test_no_explicit_opportunity_still_returns_full_mechanical_pool():
    # Old eligible_moves()-gated retrieve() would return status
    # off_no_explicit_opportunity with an EMPTY candidate list here -- this
    # is exactly the case the responsibility split is meant to fix.
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("Work has been a lot lately, I don't really know."),
        cards=CARDS,
    )
    assert pool.hard_off is False
    assert len(pool.candidates) == 6
    assert all(not c.explicit_advice_welcome for c in pool.candidates)


def test_active_high_stakes_is_still_mechanical_hard_off():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("I want to kill myself, I can't go on."),
        cards=CARDS,
    )
    assert pool.hard_off is True
    assert pool.hard_off_reason == "active_high_stakes"
    assert pool.candidates == ()


def test_explicit_stop_is_mechanical_hard_off():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("Please stop this conversation now."),
        cards=CARDS,
    )
    assert pool.hard_off is True
    assert pool.hard_off_reason == "explicit_stop"


def test_already_executed_move_dropped_individually_not_whole_pool():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("I'm not sure how to explain it, it's complicated."),
        cards=CARDS,
        already_executed_move_ids=["AM04_tentative_paraphrase_check"],
    )
    assert pool.hard_off is False
    move_ids = {c.move_id for c in pool.candidates}
    assert "AM04_tentative_paraphrase_check" not in move_ids
    assert len(pool.candidates) == 5


def test_semantic_flags_attached_as_features_on_every_candidate():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("What should I do? Any advice would help."),
        cards=CARDS,
    )
    assert pool.hard_off is False
    assert len(pool.candidates) == 6
    assert all(c.explicit_advice_welcome for c in pool.candidates)


def test_transparent_rule_top1_returns_none_when_no_signal_present():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("Work has been a lot lately, I don't really know."),
        cards=CARDS,
    )
    assert rs_transparent_rule_top1(pool) is None


def test_transparent_rule_top1_picks_open_expression_when_signaled():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("I don't know where to start, something is bothering me."),
        cards=CARDS,
    )
    top = rs_transparent_rule_top1(pool)
    assert top is not None
    assert top.move_id == "AM01_invite_open_expression"


def test_shared_rank1_keeps_candidate_when_transparent_rule_is_off():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("Work has been a lot lately, I don't really know."),
        cards=CARDS,
    )
    shared = rs_shared_candidate_top1(pool)
    assert shared is not None
    assert shared.selection_mode == "lexical_fallback"
    assert shared.transparent_rule_on is False
    assert shared.observation in pool.candidates


def test_shared_rank1_uses_same_transparent_candidate_for_every_policy():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("I don't know where to start, something is bothering me."),
        cards=CARDS,
    )
    transparent = rs_transparent_rule_top1(pool)
    shared = rs_shared_candidate_top1(pool)
    assert transparent is not None and shared is not None
    assert shared.selection_mode == "transparent_priority"
    assert shared.transparent_rule_on is True
    assert shared.observation.move_id == transparent.move_id


def test_shared_rank1_is_absent_only_under_mechanical_hard_off():
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=_dialogue("Please stop this conversation now."),
        cards=CARDS,
    )
    assert rs_shared_candidate_top1(pool) is None


def test_missing_required_card_raises():
    try:
        rs_mechanical_candidate_pool(
            recent_dialogue=_dialogue("hello"), cards=CARDS[:5],
        )
    except ValueError as exc:
        assert "missing required strategy card" in str(exc)
    else:
        raise AssertionError("expected ValueError")
