from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_g1_candidate_pool_public_private_contract() -> None:
    out = ROOT / "outputs/pm_v1_5_strategy_g1_candidate_pool_v1"
    report = json.loads(
        (out / "candidate_pool_report.json").read_text(encoding="utf-8")
    )
    public = _read_jsonl(out / "weak_label_public_packet.jsonl")
    private = _read_jsonl(out / "private_candidate_lineage.jsonl")
    assert report["full_train_universe_rows_scored"] == 9148
    assert report["frozen_atomic_moves"] == 17
    assert report["low_score_control_memberships"] == 85
    assert len(public) == len(private) == report["candidate_union_rows"]
    assert {row["blind_item_id"] for row in public} == {
        row["blind_item_id"] for row in private
    }
    assert all(
        set(row)
        == {
            "blind_item_id",
            "recent_visible_dialogue",
            "supporter_response_to_label",
        }
        for row in public
    )
    assert all("native_strategy_label_audit_only" not in row for row in public)


def test_g1_pilot_covers_every_move_and_is_no_call() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
            / "pilot_preflight.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "READY_NO_API_CALLS_MADE"
    assert report["pilot_positive_candidate_move_coverage"] == 17
    assert report["pilot_negative_control_move_coverage"] == 17
    assert report["planned_calls_for_one_coder"] > 0
