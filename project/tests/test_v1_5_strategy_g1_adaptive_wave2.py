from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_adaptive_wave_is_bounded_and_non_gold() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_strategy_g1_adaptive_wave2_v1"
            / "adaptive_wave_preflight.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "READY_NO_NEW_API_CALLS"
    assert report["accepted_stage1_rows_before_wave"] > 0
    assert 0 < report["adaptive_wave_rows"] < report[
        "remaining_stage1_rows_before_wave"
    ]
    assert report["selection_uses_external_outcomes"] is False
    assert report["selection_uses_validation_or_test"] is False
    assert report["all_outputs_remain_non_gold"] is True
