from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.contracts import StrategyCard


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_strategy_g1_final_bank_v1"


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_dual_human_freeze_keeps_raw_and_decomposed_decisions() -> None:
    report = json.loads(
        (OUT / "g1_dual_human_freeze_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == (
        "G1_COMPLETE_SIX_CARD_BANK_FROZEN_FOR_G2_DEVELOPMENT"
    )
    assert report["reviewer_1_overall_counts"] == {
        "yes": 2,
        "uncertain": 4,
        "no": 2,
    }
    assert report["reviewer_2_overall_counts"] == {"yes": 1, "no": 7}
    assert report["source_packet_grain_correction"][
        "card_example_judgments"
    ] == 40
    assert report["source_packet_grain_correction"]["unique_blind_items"] == 39
    assert report["source_packet_grain_correction"][
        "unique_source_dialogues"
    ] == 38
    assert report["g1_human_review_complete"] is True
    assert report["additional_g1_human_review_authorized"] is False


def test_final_bank_is_six_corrected_definition_cards_without_sources() -> None:
    cards = _rows(OUT / "strategy_cards_g1_final.jsonl")
    runtime = _rows(OUT / "strategy_cards_g1_runtime.jsonl")
    assert len(cards) == len(runtime) == 6
    move_ids = {row["move_id"] for row in cards}
    assert "AM07_acknowledge_effort_strength_or_resource" not in move_ids
    assert "AM15_explore_interpersonal_boundary" not in move_ids
    assert all(row["eligible_for_g2_development"] for row in cards)
    assert not any(row["eligible_for_formal_rs"] for row in cards)
    assert all(
        row["source_qualification_status"]
        == "FORMATIVE_PROVENANCE_ONLY_WEAK_SOURCE_GATE_NOT_QUALIFIED"
        for row in cards
    )
    assert all(
        StrategyCard.model_validate(row) is not None for row in runtime
    )
    assert all(
        row["source_dialogue_id"] == "g1_dual_human_definition_bank"
        for row in runtime
    )


def test_global_rules_enforce_permission_stop_crisis_and_top1() -> None:
    rules = json.loads(
        (OUT / "global_eligibility_rules.json").read_text(encoding="utf-8")
    )
    ids = {row["rule_id"] for row in rules["rules"]}
    assert {
        "active_high_stakes_override",
        "explicit_stop_boundary",
        "advice_permission",
        "question_repetition",
        "single_card_cap",
    } <= ids
    assert rules["raw_source_responses_available_to_runtime"] is False
    assert rules["source_example_labels_used_as_PM_or_permission_gold"] is False
