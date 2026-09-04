#!/usr/bin/env python3
"""Prepare the first sequential G1 weak-label expansion stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    source = ROOT / "outputs/pm_v1_5_strategy_g1_candidate_pool_v2"
    out = ROOT / "outputs/pm_v1_5_strategy_g1_stage1_weak_labels_v1"
    public = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(source / "new_unlabeled_public_packet.jsonl")
    }
    private = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(source / "new_unlabeled_private_lineage.jsonl")
    }
    if set(public) != set(private):
        raise ValueError("v2 public/private sets differ")

    selected_ids = [
        item_id
        for item_id, row in sorted(private.items())
        if any(
            int(proposal["native_pattern_rank"]) <= 20
            for proposal in row["native_pattern_candidate_proposals"]
        )
    ]
    public_rows = [public[item_id] for item_id in selected_ids]
    private_rows = [private[item_id] for item_id in selected_ids]
    covered_moves = {
        proposal["move_id"]
        for row in private_rows
        for proposal in row["native_pattern_candidate_proposals"]
        if int(proposal["native_pattern_rank"]) <= 20
    }
    if len(covered_moves) != 17:
        raise ValueError("stage 1 does not cover every frozen move")
    report = {
        "protocol": "pm-v1.5-strategy-g1-sequential-stage1-preparation-v1",
        "status": "READY_NO_NEW_API_CALLS",
        "rank_cap_per_move": 20,
        "frozen_move_coverage": len(covered_moves),
        "new_rows": len(public_rows),
        "planned_single_coder_calls_at_batch_8": (len(public_rows) + 7) // 8,
        "sequential_stop_rule": (
            "After literal-source audit, expand only moves that remain below "
            "20 independent clean source dialogues; do not automatically run "
            "rank 21-60 for already-supported moves."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "public_packet.jsonl", public_rows)
    _write_jsonl(out / "private_lineage.jsonl", private_rows)
    _write_json(out / "stage1_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
