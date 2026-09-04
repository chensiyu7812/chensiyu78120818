#!/usr/bin/env python3
"""Prepare a bounded no-call pilot for the G1 Coder-B weak labeler."""

from __future__ import annotations

import argparse
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


def _by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["blind_item_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label} repeats blind_item_id")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_g1_candidate_pool_v1",
    )
    parser.add_argument("--maximum-semantic-rank", type=int, default=6)
    parser.add_argument("--maximum-lexical-rank", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1",
    )
    args = parser.parse_args()

    public = _by_id(
        _read_jsonl(args.candidate_dir / "weak_label_public_packet.jsonl"),
        "public",
    )
    private = _by_id(
        _read_jsonl(args.candidate_dir / "private_candidate_lineage.jsonl"),
        "private",
    )
    if set(public) != set(private):
        raise ValueError("candidate public/private ID sets differ")

    selected_ids: list[str] = []
    for item_id in sorted(private):
        row = private[item_id]
        selected_by_positive_rank = any(
            (
                proposal.get("semantic_rank") is not None
                and int(proposal["semantic_rank"]) <= args.maximum_semantic_rank
            )
            or (
                proposal.get("lexical_rank") is not None
                and int(proposal["lexical_rank"]) <= args.maximum_lexical_rank
            )
            for proposal in row["candidate_move_proposals"]
        )
        selected_as_control = bool(row["low_score_control_for_move_ids"])
        if selected_by_positive_rank or selected_as_control:
            selected_ids.append(item_id)

    public_rows = [public[item_id] for item_id in selected_ids]
    private_rows = [private[item_id] for item_id in selected_ids]
    if not public_rows:
        raise ValueError("pilot selection is empty")
    move_ids = {
        proposal["move_id"]
        for row in private_rows
        for proposal in row["candidate_move_proposals"]
        if (
            (
                proposal.get("semantic_rank") is not None
                and int(proposal["semantic_rank"]) <= args.maximum_semantic_rank
            )
            or (
                proposal.get("lexical_rank") is not None
                and int(proposal["lexical_rank"]) <= args.maximum_lexical_rank
            )
        )
    }
    control_move_ids = {
        move_id
        for row in private_rows
        for move_id in row["low_score_control_for_move_ids"]
    }
    report = {
        "protocol": "pm-v1.5-strategy-g1-weak-label-pilot-preparation-v1",
        "status": "READY_NO_API_CALLS_MADE",
        "pilot_rows": len(public_rows),
        "pilot_positive_candidate_move_coverage": len(move_ids),
        "pilot_negative_control_move_coverage": len(control_move_ids),
        "maximum_semantic_rank": args.maximum_semantic_rank,
        "maximum_lexical_rank": args.maximum_lexical_rank,
        "all_low_score_control_rows_included": True,
        "batch_size": args.batch_size,
        "planned_calls_for_one_coder": (
            len(public_rows) + args.batch_size - 1
        )
        // args.batch_size,
        "qualified_coder": "training_judge_deepseek_flash",
        "pilot_role": (
            "schema, literal-evidence, multi-label behavior, and low-score "
            "control-activation review only; low BGE score is not a gold "
            "negative, and the discovery set influenced "
            "the codebook, so this is not an independent accuracy estimate"
        ),
        "promotion_rule": (
            "Do not run the full 1273-row pool unless the pilot has exact ID "
            "coverage, literal evidence, no invalid move IDs, and a human "
            "review finds no systematic over-labeling in fixed controls."
        ),
    }
    if len(move_ids) != 17 or len(control_move_ids) != 17:
        raise ValueError("pilot lacks positive or control coverage for all 17 moves")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "public_packet.jsonl", public_rows)
    _write_jsonl(args.out_dir / "private_lineage.jsonl", private_rows)
    _write_json(args.out_dir / "pilot_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
