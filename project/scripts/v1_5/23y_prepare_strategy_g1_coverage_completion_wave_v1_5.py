#!/usr/bin/env python3
"""Prepare the final bounded G1 source-coverage completion wave."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g1-coverage-completion-wave-preparation-v1"


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


def main() -> None:
    contract = _read_json(
        ROOT
        / "data/pm_v1_5_contracts/strategy_g1_coverage_completion_v1.json"
    )
    pool_dir = ROOT / "outputs/pm_v1_5_strategy_g1_candidate_pool_v2"
    public = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(pool_dir / "new_unlabeled_public_packet.jsonl")
    }
    private = {
        str(row["blind_item_id"]): row
        for row in _read_jsonl(pool_dir / "new_unlabeled_private_lineage.jsonl")
    }
    if set(public) != set(private) or len(public) != 931:
        raise ValueError("G1 v2 new-unlabeled public/private pool mismatch")

    used_ids: set[str] = set()
    for path in [
        ROOT
        / "outputs/pm_v1_5_strategy_g1_stage1_weak_label_run_v1"
        / "weak_labels.jsonl",
        ROOT
        / "outputs/pm_v1_5_strategy_g1_adaptive_wave2_run_v1"
        / "weak_labels.jsonl",
    ]:
        used_ids.update(str(row["blind_item_id"]) for row in _read_jsonl(path))

    reasons: dict[str, list[dict[str, Any]]] = {}
    targets = contract["coverage_completion_targets"]
    for move_id, target in targets.items():
        candidates: list[tuple[int, str]] = []
        for item_id, row in private.items():
            if item_id in used_ids:
                continue
            ranks = [
                int(proposal["native_pattern_rank"])
                for proposal in row["native_pattern_candidate_proposals"]
                if proposal["move_id"] == move_id
            ]
            if ranks:
                candidates.append((min(ranks), item_id))
        cap = int(target["new_candidate_row_cap"])
        for rank, item_id in sorted(candidates)[:cap]:
            reasons.setdefault(item_id, []).append(
                {
                    "coverage_target_move_id": move_id,
                    "native_pattern_rank_audit_only": rank,
                    "pre_wave_clean_dialogues": target[
                        "current_clean_dialogues"
                    ],
                    "fixed_new_candidate_row_cap": cap,
                }
            )

    selected_ids = sorted(reasons)
    public_rows = [public[item_id] for item_id in selected_ids]
    private_rows = [
        {
            **private[item_id],
            "coverage_completion_selection_reasons": reasons[item_id],
        }
        for item_id in selected_ids
    ]
    out_dir = (
        ROOT / "outputs/pm_v1_5_strategy_g1_coverage_completion_wave_v1"
    )
    report = {
        "protocol": PROTOCOL,
        "status": "READY_NO_NEW_API_CALLS",
        "target_move_count": len(targets),
        "target_move_ids": list(targets),
        "wave_rows": len(public_rows),
        "planned_single_coder_calls_at_batch_8": (len(public_rows) + 7) // 8,
        "selected_ids_overlap_prior_accepted_labels": bool(
            set(selected_ids).intersection(used_ids)
        ),
        "selection_uses_validation_test_external_or_pm_outcomes": False,
        "outputs_are_gold": False,
        "post_wave_stop_rule": contract["post_wave_rule"],
    }
    _write_jsonl(out_dir / "public_packet.jsonl", public_rows)
    _write_jsonl(out_dir / "private_lineage.jsonl", private_rows)
    _write_json(out_dir / "coverage_completion_preflight.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
