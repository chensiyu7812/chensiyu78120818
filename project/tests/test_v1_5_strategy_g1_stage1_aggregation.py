from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage1_aggregation_preserves_weak_vs_gold_boundary() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1"
            / "coverage_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["combined_rows"] == 339
    assert report["unique_blind_items"] == 339
    assert report["all_evidence_literal_validation_passed"] is True
    assert report["clean_weak_source_is_human_gold"] is False
    assert report["twenty_dialogue_gate_is_fully_human_verified"] is False
    assert len(report["per_move"]) == 17


def test_stage1_human_source_audit_is_bounded() -> None:
    path = (
        ROOT
        / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1"
        / "fixed_literal_source_human_audit.jsonl"
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) <= 17 * 5
    assert all(row["human_source_approved"] is None for row in rows)
    assert all(row["literal_response_excerpt"] for row in rows)
