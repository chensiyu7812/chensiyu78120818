from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_g1_pilot_structural_pass_requires_recall_redesign() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_qualification_v1"
            / "qualification_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == (
        "PILOT_LABELER_STRUCTURAL_PASS_CANDIDATE_RECALL_REDESIGN_REQUIRED"
    )
    assert report["pilot_rows"] == 122
    assert report["all_evidence_literal"] is True
    assert report["all_17_moves_activated_at_least_once"] is True
    assert report["retrieval_warning_move_count"] > 0
    assert report["current_1273_pool_decision"].startswith("do_not_run_yet")
