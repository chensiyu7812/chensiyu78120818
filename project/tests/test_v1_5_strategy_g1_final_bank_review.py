from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_final_bank_review_is_bounded_and_outcome_blind() -> None:
    root = ROOT / "outputs/pm_v1_5_strategy_g1_final_bank_review_candidate_v1"
    manifest = json.loads(
        (root / "review_manifest.json").read_text(encoding="utf-8")
    )
    cards = _rows(root / "strategy_cards_g1_candidate.jsonl")
    packet = _rows(root / "human_review_packet.jsonl")
    assert manifest["candidate_card_count"] == len(cards) == len(packet) == 8
    assert manifest["standard_gate_card_count"] == 6
    assert manifest["narrow_exception_card_count"] == 2
    assert manifest["fixed_source_example_count"] == 40
    assert manifest["weak_labels_are_gold"] is False
    assert (
        manifest["survey_quality_judge_or_external_outcomes_visible_to_human"]
        is False
    )
    assert all(len(row["source_examples"]) == 5 for row in packet)
    assert all(card["eligible_for_g2"] is False for card in cards)


def test_only_explicit_narrow_exceptions_miss_standard_gate() -> None:
    root = ROOT / "outputs/pm_v1_5_strategy_g1_final_bank_review_candidate_v1"
    cards = _rows(root / "strategy_cards_g1_candidate.jsonl")
    exceptions = [
        row
        for row in cards
        if row["source_gate_status"]
        == "NARROW_EXCEPTION_REQUIRES_HUMAN_APPROVAL"
    ]
    assert {row["move_id"] for row in exceptions} == {
        "AM07_acknowledge_effort_strength_or_resource",
        "AM15_explore_interpersonal_boundary",
    }
    assert all(row["clean_weak_source_dialogues"] >= 10 for row in exceptions)
    assert all(
        row["clean_weak_source_dialogues"] >= 20
        for row in cards
        if row not in exceptions
    )
