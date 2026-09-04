from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_coverage_completion_wave_is_fixed_bounded_and_disjoint() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_strategy_g1_coverage_completion_wave_v1"
            / "coverage_completion_preflight.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "READY_NO_NEW_API_CALLS"
    assert report["target_move_count"] == 5
    assert 0 < report["wave_rows"] <= 4 + 10 + 12 + 12 + 25
    assert report["planned_single_coder_calls_at_batch_8"] <= 8
    assert report["selected_ids_overlap_prior_accepted_labels"] is False
    assert (
        report["selection_uses_validation_test_external_or_pm_outcomes"]
        is False
    )
    assert report["outputs_are_gold"] is False
