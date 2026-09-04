#!/usr/bin/env python3
"""Prepare an adaptive G1 wave from still-unlabeled Stage-1 candidates.

The wave size is based only on current clean weak-source deficits. It does not
use validation/test/EvoEmo outcomes or alter the frozen move definitions.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g1-adaptive-wave2-preparation-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _budget(clean_dialogues: int) -> int:
    if clean_dialogues < 5:
        return 8
    if clean_dialogues < 10:
        return 6
    if clean_dialogues < 15:
        return 4
    if clean_dialogues < 20:
        return 2
    return 0


def main() -> None:
    codebook = _read_json(
        ROOT / "data/strategy/pm_v1_5_strategy_atomic_move_codebook_v1.json"
    )
    move_ids = [str(move["move_id"]) for move in codebook["moves"]]
    stage_dir = ROOT / "outputs/pm_v1_5_strategy_g1_stage1_weak_labels_v1"
    run_dir = ROOT / "outputs/pm_v1_5_strategy_g1_stage1_weak_label_run_v1"
    pilot_dir = ROOT / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
    pilot_run_dir = ROOT / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_run_v1"
    out_dir = ROOT / "outputs/pm_v1_5_strategy_g1_adaptive_wave2_v1"

    stage_public = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(stage_dir / "public_packet.jsonl")
    }
    stage_private = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(stage_dir / "private_lineage.jsonl")
    }
    accepted_stage = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(run_dir / "weak_labels.jsonl")
    }
    pilot_private = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(pilot_dir / "private_lineage.jsonl")
    }
    pilot_labels = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(pilot_run_dir / "weak_labels.jsonl")
    }
    if set(stage_public) != set(stage_private) or len(stage_public) != 323:
        raise ValueError("Stage-1 public/private packet mismatch")
    if not set(accepted_stage).issubset(stage_public):
        raise ValueError("accepted Stage-1 labels are outside its packet")
    if set(pilot_private) != set(pilot_labels) or len(pilot_labels) != 122:
        raise ValueError("pilot coverage mismatch")

    clean_dialogues: dict[str, set[str]] = {
        move_id: set() for move_id in move_ids
    }
    for labels, private in (
        (pilot_labels, pilot_private),
        (accepted_stage, stage_private),
    ):
        for item_id, label in labels.items():
            if not label["source_compatible_weak_proposal"]:
                continue
            dialogue_id = str(private[item_id]["source_dialogue_id"])
            for move_id in label["recognized_move_ids"]:
                clean_dialogues[str(move_id)].add(dialogue_id)

    per_move_budget = {
        move_id: _budget(len(clean_dialogues[move_id]))
        for move_id in move_ids
    }
    proposals: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for item_id, row in stage_private.items():
        if item_id in accepted_stage:
            continue
        for proposal in row["native_pattern_candidate_proposals"]:
            move_id = str(proposal["move_id"])
            if per_move_budget.get(move_id, 0) > 0:
                proposals[move_id].append(
                    (int(proposal["native_pattern_rank"]), item_id)
                )

    selected_reasons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for move_id in move_ids:
        candidates = sorted(proposals[move_id])
        budget = per_move_budget[move_id]
        for rank, item_id in candidates[:budget]:
            selected_reasons[item_id].append(
                {
                    "move_id": move_id,
                    "native_pattern_rank": rank,
                    "current_clean_weak_source_dialogues": len(
                        clean_dialogues[move_id]
                    ),
                    "adaptive_new_row_budget": budget,
                }
            )

    selected_ids = sorted(selected_reasons)
    public_rows = [stage_public[item_id] for item_id in selected_ids]
    private_rows = [
        {
            **stage_private[item_id],
            "adaptive_selection_reasons": selected_reasons[item_id],
        }
        for item_id in selected_ids
    ]
    report = {
        "protocol": PROTOCOL,
        "status": "READY_NO_NEW_API_CALLS",
        "accepted_stage1_rows_before_wave": len(accepted_stage),
        "remaining_stage1_rows_before_wave": len(stage_public) - len(accepted_stage),
        "current_clean_weak_source_dialogues_per_move": {
            move_id: len(clean_dialogues[move_id]) for move_id in move_ids
        },
        "adaptive_new_row_budget_per_move": per_move_budget,
        "adaptive_wave_rows": len(public_rows),
        "planned_single_coder_calls_at_batch_8": (len(public_rows) + 7) // 8,
        "selection_uses_external_outcomes": False,
        "selection_uses_validation_or_test": False,
        "all_outputs_remain_non_gold": True,
        "after_wave_rule": (
            "reaggregate before any further calls; exclude or document a "
            "narrow-move exception rather than force every move to 20"
        ),
    }
    _write_jsonl(out_dir / "public_packet.jsonl", public_rows)
    _write_jsonl(out_dir / "private_lineage.jsonl", private_rows)
    _write_json(out_dir / "adaptive_wave_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
