#!/usr/bin/env python3
"""Aggregate G1 pilot and Stage-1 weak source proposals.

This stage measures candidate-source coverage only.  A clean weak proposal is
not a gold move label, a human-approved source, or a finished strategy card.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g1-stage1-coverage-aggregation-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_id(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result = {str(row["blind_item_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label} repeats blind_item_id")
    return result


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


def _proposal_rank(lineage: dict[str, Any], move_id: str) -> int:
    ranks: list[int] = []
    for proposal in lineage.get("native_pattern_candidate_proposals", []):
        if proposal["move_id"] == move_id:
            ranks.append(int(proposal["native_pattern_rank"]))
    for proposal in lineage.get("candidate_move_proposals", []):
        if proposal["move_id"] != move_id:
            continue
        for field in ("semantic_rank", "lexical_rank"):
            if proposal.get(field) is not None:
                ranks.append(int(proposal[field]))
    return min(ranks, default=10_000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1",
    )
    args = parser.parse_args()

    codebook = _read_json(
        ROOT / "data/strategy/pm_v1_5_strategy_atomic_move_codebook_v1.json"
    )
    move_ids = [str(move["move_id"]) for move in codebook["moves"]]
    if len(move_ids) != len(set(move_ids)) or len(move_ids) != 17:
        raise ValueError("expected the frozen 17-move codebook")

    sources = [
        (
            "pilot",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
            / "private_lineage.jsonl",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_run_v1"
            / "weak_labels.jsonl",
            122,
            122,
        ),
        (
            "stage1_pre_adaptive",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_stage1_weak_labels_v1"
            / "private_lineage.jsonl",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_stage1_weak_label_run_v1"
            / "weak_labels.jsonl",
            323,
            64,
        ),
        (
            "adaptive_wave2",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_adaptive_wave2_v1"
            / "private_lineage.jsonl",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_adaptive_wave2_run_v1"
            / "weak_labels.jsonl",
            90,
            90,
        ),
        (
            "coverage_completion_wave",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_coverage_completion_wave_v1"
            / "private_lineage.jsonl",
            ROOT
            / "outputs/pm_v1_5_strategy_g1_coverage_completion_wave_run_v1"
            / "weak_labels.jsonl",
            63,
            63,
        ),
    ]
    joined: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    all_ids: set[str] = set()
    for (
        source_name,
        lineage_path,
        label_path,
        expected_lineage_rows,
        expected_label_rows,
    ) in sources:
        lineage = _by_id(_read_jsonl(lineage_path), label=f"{source_name}_lineage")
        labels = _by_id(_read_jsonl(label_path), label=f"{source_name}_labels")
        if (
            len(lineage) != expected_lineage_rows
            or len(labels) != expected_label_rows
            or not set(labels).issubset(lineage)
        ):
            raise ValueError(
                f"{source_name} requires exact expected lineage/label coverage"
            )
        if all_ids.intersection(labels):
            raise ValueError("pilot and Stage-1 blind IDs overlap")
        all_ids.update(labels)
        source_counts[source_name] = len(labels)
        for item_id in sorted(labels):
            label = labels[item_id]
            if not label.get("evidence_literal_validation_passed"):
                raise ValueError(f"{item_id}: literal evidence validation failed")
            joined.append(
                {
                    "blind_item_id": item_id,
                    "weak_label_source": source_name,
                    "source_dialogue_id": lineage[item_id]["source_dialogue_id"],
                    "source_turn_index": lineage[item_id]["source_turn_index"],
                    "strategy_id": lineage[item_id]["strategy_id"],
                    "problem_type_audit_only": lineage[item_id][
                        "problem_type_audit_only"
                    ],
                    "emotion_type_audit_only": lineage[item_id][
                        "emotion_type_audit_only"
                    ],
                    "recognized_move_ids": label["recognized_move_ids"],
                    "hard_exclusion_flags": label["hard_exclusion_flags"],
                    "confidence": label["confidence"],
                    "move_evidence": label["move_evidence"],
                    "exclusion_evidence": label["exclusion_evidence"],
                    "source_compatible_weak_proposal": label[
                        "source_compatible_weak_proposal"
                    ],
                    "private_candidate_lineage": lineage[item_id],
                }
            )

    confidence_order = {"high": 0, "medium": 1, "low": 2}
    per_move: dict[str, dict[str, Any]] = {}
    audit_candidates: list[dict[str, Any]] = []
    moves_below_gate: list[str] = []
    for move_id in move_ids:
        recognized = [
            row for row in joined if move_id in row["recognized_move_ids"]
        ]
        clean = [
            row
            for row in recognized
            if row["source_compatible_weak_proposal"]
        ]
        clean_dialogues = {
            str(row["source_dialogue_id"]) for row in clean
        }
        problem_counts = Counter(
            str(row["problem_type_audit_only"]) for row in clean
        )
        emotion_counts = Counter(
            str(row["emotion_type_audit_only"]) for row in clean
        )
        max_problem_share = (
            max(problem_counts.values()) / len(clean)
            if clean
            else None
        )
        if len(clean_dialogues) < 20:
            moves_below_gate.append(move_id)

        # Keep one strongest response per dialogue for a later literal-source
        # audit. Candidate rank affects audit ordering only, never the label.
        best_by_dialogue: dict[str, dict[str, Any]] = {}
        for row in clean:
            dialogue_id = str(row["source_dialogue_id"])
            rank = _proposal_rank(row["private_candidate_lineage"], move_id)
            key = (
                confidence_order.get(str(row["confidence"]), 9),
                rank,
                int(row["source_turn_index"]),
                str(row["blind_item_id"]),
            )
            current = best_by_dialogue.get(dialogue_id)
            if current is None or key < current["_sort_key"]:
                evidence = next(
                    evidence
                    for evidence in row["move_evidence"]
                    if evidence["move_id"] == move_id
                )
                best_by_dialogue[dialogue_id] = {
                    "_sort_key": key,
                    "move_id": move_id,
                    "blind_item_id": row["blind_item_id"],
                    "source_dialogue_id": dialogue_id,
                    "source_turn_index": row["source_turn_index"],
                    "literal_response_excerpt": evidence["response_excerpt"],
                    "weak_confidence": row["confidence"],
                    "candidate_recall_rank_audit_only": (
                        None if rank == 10_000 else rank
                    ),
                    "problem_type_audit_only": row[
                        "problem_type_audit_only"
                    ],
                    "emotion_type_audit_only": row[
                        "emotion_type_audit_only"
                    ],
                    "human_source_approved": None,
                    "human_source_reason": "",
                }
        ranked = sorted(best_by_dialogue.values(), key=lambda row: row["_sort_key"])
        # Five fixed exemplars per move keep the human pass bounded. The 20
        # dialogue gate remains a weak-screened source-support gate and is
        # reported as such, never mislabeled as 20 human approvals.
        for row in ranked[:5]:
            row.pop("_sort_key")
            audit_candidates.append(row)

        per_move[move_id] = {
            "recognized_rows": len(recognized),
            "recognized_independent_dialogues": len(
                {str(row["source_dialogue_id"]) for row in recognized}
            ),
            "clean_weak_source_rows_not_human_gold": len(clean),
            "clean_weak_source_independent_dialogues_not_human_gold": len(
                clean_dialogues
            ),
            "passes_20_dialogue_weak_source_gate": len(clean_dialogues) >= 20,
            "problem_type_count_audit_only": len(problem_counts),
            "emotion_type_count_audit_only": len(emotion_counts),
            "maximum_single_problem_type_share_audit_only": (
                None
                if max_problem_share is None
                else round(max_problem_share, 4)
            ),
            "fixed_literal_source_exemplars_for_human_audit": min(5, len(ranked)),
            "hard_exclusion_counts_among_recognized": dict(
                sorted(
                    Counter(
                        flag
                        for row in recognized
                        for flag in row["hard_exclusion_flags"]
                    ).items()
                )
            ),
        }

    report = {
        "protocol": PROTOCOL,
        "status": "G1_WEAK_SOURCE_SCREEN_COMPLETE_STOP_API_EXPANSION",
        "combined_rows": len(joined),
        "source_counts": source_counts,
        "unique_blind_items": len(all_ids),
        "unique_source_dialogues": len(
            {str(row["source_dialogue_id"]) for row in joined}
        ),
        "all_evidence_literal_validation_passed": True,
        "source_compatible_definition": (
            "at least one recognized frozen atomic move and zero hard "
            "source-exclusion flags"
        ),
        "clean_weak_source_is_human_gold": False,
        "twenty_dialogue_gate_is_fully_human_verified": False,
        "bounded_human_source_audit": (
            "five fixed literal exemplars per move; every candidate card "
            "definition, when-to-use, when-not-to-use, and correction is "
            "reviewed separately"
        ),
        "moves_below_20_independent_clean_weak_source_dialogues": moves_below_gate,
        "move_count_below_gate": len(moves_below_gate),
        "next_expansion_rule": (
            "do not force all 17 moves to pass. Draft only source-qualified "
            "coverage-relevant cards; delete or narrow sparse moves, or use "
            "a written narrow-move exception fixed before retrieval testing"
        ),
        "per_move": per_move,
    }
    _write_json(args.out_dir / "coverage_report.json", report)
    _write_jsonl(args.out_dir / "combined_private_weak_sources.jsonl", joined)
    _write_jsonl(
        args.out_dir / "fixed_literal_source_human_audit.jsonl",
        audit_candidates,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
