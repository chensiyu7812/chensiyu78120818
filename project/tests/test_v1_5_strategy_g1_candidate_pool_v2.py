from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_pool_v2_is_targeted_and_hides_native_labels() -> None:
    out = ROOT / "outputs/pm_v1_5_strategy_g1_candidate_pool_v2"
    report = json.loads(
        (out / "candidate_pool_v2_report.json").read_text(encoding="utf-8")
    )
    public = [
        json.loads(line)
        for line in (out / "new_unlabeled_public_packet.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert report["completed_pilot_rows_reused"] == 122
    assert report["native_pattern_candidates_per_move"] == 60
    assert len(public) == report["new_unlabeled_rows"]
    assert all(
        set(row)
        == {
            "blind_item_id",
            "recent_visible_dialogue",
            "supporter_response_to_label",
        }
        for row in public
    )
    assert report["native_labels_visible_to_coder"] is False
