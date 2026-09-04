from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_g0_contract_resolves_exactly_21_items() -> None:
    contract = json.loads(
        (
            ROOT
            / "data/pm_v1_5_contracts"
            / "strategy_g0_disagreement_adjudication_v1.json"
        ).read_text(encoding="utf-8")
    )
    rows = contract["resolutions"]
    assert len(rows) == 21
    assert len({row["blind_item_id"] for row in rows}) == 21
    for row in rows:
        assert set(row["final"]) == {
            "meaningful_support_action",
            "mainly_information",
            "mainly_self_disclosure",
            "reusable_as_general_technique",
            "clear_risk_or_boundary_problem",
        }
        assert all(isinstance(value, bool) for value in row["final"].values())


def test_atomic_move_codebook_is_nonempty_unique_and_not_a_bank() -> None:
    codebook = json.loads(
        (
            ROOT
            / "data/strategy"
            / "pm_v1_5_strategy_atomic_move_codebook_v1.json"
        ).read_text(encoding="utf-8")
    )
    moves = codebook["moves"]
    move_ids = [move["move_id"] for move in moves]
    assert len(moves) == 17
    assert len(move_ids) == len(set(move_ids))
    assert codebook["status"].endswith("NOT_A_FROZEN_STRATEGY_BANK")
    assert all(move["definition"] for move in moves)
    assert all(move["behavior_regions"] for move in moves)
