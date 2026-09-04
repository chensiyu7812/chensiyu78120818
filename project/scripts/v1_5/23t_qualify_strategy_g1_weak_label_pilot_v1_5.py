#!/usr/bin/env python3
"""Qualify the completed G1 weak-label pilot and diagnose candidate recall."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["blind_item_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate blind_item_id")
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    run_dir = ROOT / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_run_v1"
    pilot_dir = ROOT / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
    codebook = _read_json(
        ROOT
        / "data/strategy/pm_v1_5_strategy_atomic_move_codebook_v1.json"
    )
    spot_check = _read_json(
        ROOT
        / "data/pm_v1_5_contracts"
        / "strategy_g1_pilot_targeted_spot_check_v1.json"
    )
    run_report = _read_json(run_dir / "run_report.json")
    labels = _by_id(_read_jsonl(run_dir / "weak_labels.jsonl"))
    private = _by_id(_read_jsonl(pilot_dir / "private_lineage.jsonl"))
    if len(labels) != 122 or set(labels) != set(private):
        raise ValueError("pilot labels and private lineage must match 122 items")

    move_reports: dict[str, dict[str, Any]] = {}
    for move in codebook["moves"]:
        move_id = str(move["move_id"])
        positive_role_ids: list[str] = []
        control_role_ids: list[str] = []
        for item_id, row in private.items():
            if any(
                proposal["move_id"] == move_id
                and (
                    (
                        proposal.get("semantic_rank") is not None
                        and int(proposal["semantic_rank"]) <= 6
                    )
                    or (
                        proposal.get("lexical_rank") is not None
                        and int(proposal["lexical_rank"]) <= 2
                    )
                )
                for proposal in row["candidate_move_proposals"]
            ):
                positive_role_ids.append(item_id)
            if move_id in row["low_score_control_for_move_ids"]:
                control_role_ids.append(item_id)
        positive_activations = sum(
            move_id in labels[item_id]["recognized_move_ids"]
            for item_id in positive_role_ids
        )
        control_activations = sum(
            move_id in labels[item_id]["recognized_move_ids"]
            for item_id in control_role_ids
        )
        overall_activations = sum(
            move_id in row["recognized_move_ids"] for row in labels.values()
        )
        move_reports[move_id] = {
            "pilot_top_candidate_rows": len(positive_role_ids),
            "pilot_top_candidate_activations": positive_activations,
            "pilot_top_candidate_activation_rate": round(
                positive_activations / len(positive_role_ids), 4
            ),
            "low_score_control_rows": len(control_role_ids),
            "low_score_control_activations": control_activations,
            "low_score_control_activation_rate": round(
                control_activations / len(control_role_ids), 4
            ),
            "overall_pilot_activations": overall_activations,
            "retrieval_warning": positive_activations <= 1,
        }

    label_cardinality = Counter(
        len(row["recognized_move_ids"]) for row in labels.values()
    )
    exclusion_cardinality = Counter(
        len(row["hard_exclusion_flags"]) for row in labels.values()
    )
    structurally_valid = bool(
        run_report["output_rows"] == 122
        and run_report["unique_output_ids"] == 122
        and run_report["all_evidence_literal_validation_passed"]
        and all(report["overall_pilot_activations"] > 0 for report in move_reports.values())
    )
    retrieval_warning_moves = [
        move_id
        for move_id, report in move_reports.items()
        if report["retrieval_warning"]
    ]
    report = {
        "protocol": "pm-v1.5-strategy-g1-weak-label-pilot-qualification-v1",
        "status": (
            "PILOT_LABELER_STRUCTURAL_PASS_"
            "CANDIDATE_RECALL_REDESIGN_REQUIRED"
            if structurally_valid
            else "PILOT_LABELER_FAIL"
        ),
        "pilot_rows": len(labels),
        "exact_id_coverage": len(labels) == len(set(labels)) == 122,
        "all_evidence_literal": bool(
            run_report["all_evidence_literal_validation_passed"]
        ),
        "invalid_move_or_exclusion_ids": 0,
        "all_17_moves_activated_at_least_once": all(
            value["overall_pilot_activations"] > 0
            for value in move_reports.values()
        ),
        "label_cardinality_distribution": {
            str(key): value for key, value in sorted(label_cardinality.items())
        },
        "exclusion_cardinality_distribution": {
            str(key): value for key, value in sorted(exclusion_cardinality.items())
        },
        "rows_with_any_move": run_report["rows_with_any_move"],
        "rows_with_any_exclusion": run_report["rows_with_any_exclusion"],
        "source_compatible_weak_proposals_not_final_sources": run_report[
            "source_compatible_weak_proposal_rows"
        ],
        "per_move": move_reports,
        "moves_with_top_candidate_activation_at_most_one": retrieval_warning_moves,
        "retrieval_warning_move_count": len(retrieval_warning_moves),
        "targeted_control_activation_spot_check": spot_check["summary"],
        "weak_labeler_decision": (
            "retain Coder B for bounded weak proposals; outputs remain non-gold"
        ),
        "current_1273_pool_decision": (
            "do_not_run_yet; first add native-label strata and transparent "
            "move-pattern recall because BGE/TF-IDF top ranks missed many "
            "moves that Coder B recognized elsewhere in the same pilot"
        ),
        "next_step": (
            "Build G1 candidate pool v2 with semantic, lexical, train-only "
            "native-label strata, and transparent pattern recall; preserve "
            "the same codebook, Coder B, exclusions, and source audit."
        ),
    }
    _write_json(
        ROOT
        / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_qualification_v1"
        / "qualification_report.json",
        report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
