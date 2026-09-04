from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_g1_stage1_is_bounded_and_covers_all_moves() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_strategy_g1_stage1_weak_labels_v1"
            / "stage1_preflight.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "READY_NO_NEW_API_CALLS"
    assert report["rank_cap_per_move"] == 20
    assert report["frozen_move_coverage"] == 17
    assert report["new_rows"] > 0
    assert report["new_rows"] < 931
