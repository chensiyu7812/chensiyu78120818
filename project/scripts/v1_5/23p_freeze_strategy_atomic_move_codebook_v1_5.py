#!/usr/bin/env python3
"""Resolve the 21 human disagreements and validate the G0 atomic-move codebook.

This is a rule-based synthesis of two non-independent human annotation sets.
It is valid for codebook/source-screening work only, never as PM outcome gold.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "meaningful_support_action",
    "mainly_information",
    "mainly_self_disclosure",
    "reusable_as_general_technique",
    "clear_risk_or_boundary_problem",
)


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


def _by_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("blind_item_id") or "")
        if not item_id or item_id in result:
            raise ValueError(f"{label}: empty or duplicate blind_item_id")
        result[item_id] = row
    return result


def _validate_final(final: dict[str, Any], *, item_id: str) -> None:
    for field in FIELDS[:-1]:
        if final.get(field) not in (True, False):
            raise ValueError(f"{item_id}: {field} must be boolean")
    if final.get(FIELDS[-1]) not in (True, False):
        raise ValueError(f"{item_id}: final source risk must be resolved to boolean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-a",
        type=Path,
        default=Path(
            "/home/tokkio/.codex/attachments/"
            "9315c46b-ceb9-4469-bc8d-50a1ee5e0cbb/pasted-text.txt"
        ),
    )
    parser.add_argument(
        "--annotation-b",
        type=Path,
        default=Path(
            "/home/tokkio/.codex/attachments/"
            "5d08ea20-ec35-4050-955e-4a1e438e1206/pasted-text.txt"
        ),
    )
    parser.add_argument(
        "--paired-model-codes",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_strategy_open_coding_aggregation_v1"
        / "paired_model_codes.jsonl",
    )
    parser.add_argument(
        "--adjudication",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts"
        / "strategy_g0_disagreement_adjudication_v1.json",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=ROOT
        / "data/strategy"
        / "pm_v1_5_strategy_atomic_move_codebook_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_g0_freeze_v1",
    )
    args = parser.parse_args()

    left = _by_id(_read_jsonl(args.annotation_a), label="annotation_a")
    right = _by_id(_read_jsonl(args.annotation_b), label="annotation_b")
    if len(left) != 72 or set(left) != set(right):
        raise ValueError("the two human inputs must contain the same 72 unique items")

    actual_disagreement_ids = {
        item_id
        for item_id in left
        if any(left[item_id].get(field) != right[item_id].get(field) for field in FIELDS)
    }
    adjudication = _read_json(args.adjudication)
    resolutions = _by_id(adjudication["resolutions"], label="resolutions")
    if len(resolutions) != 21 or set(resolutions) != actual_disagreement_ids:
        raise ValueError(
            "adjudication IDs must exactly equal the 21 observed disagreement IDs"
        )

    merged: list[dict[str, Any]] = []
    for item_id in sorted(left):
        if item_id in resolutions:
            final = dict(resolutions[item_id]["final"])
            reason_codes = list(resolutions[item_id]["reason_codes"])
            resolution_source = "frozen_rule_based_resolution"
            codebook_use = str(resolutions[item_id]["codebook_use"])
        else:
            final = {field: left[item_id][field] for field in FIELDS}
            if final[FIELDS[-1]] == "uncertain":
                final[FIELDS[-1]] = True
            reason_codes = ["exact_human_field_agreement"]
            resolution_source = "exact_human_agreement"
            codebook_use = "apply_frozen_source_rules"
        _validate_final(final, item_id=item_id)
        mechanically_compatible = bool(
            final["meaningful_support_action"]
            and final["reusable_as_general_technique"]
            and not final["mainly_information"]
            and not final["mainly_self_disclosure"]
            and not final["clear_risk_or_boundary_problem"]
        )
        merged.append(
            {
                "blind_item_id": item_id,
                **final,
                "resolution_source": resolution_source,
                "reason_codes": reason_codes,
                "codebook_use": codebook_use,
                "mechanically_compatible_source_seed": mechanically_compatible,
                "final_card_source_eligible": None,
                "final_card_source_eligible_note": (
                    "deferred_to_G1_full_source_text_and_high_stakes_review"
                ),
            }
        )

    codebook = _read_json(args.codebook)
    moves = list(codebook.get("moves") or [])
    move_ids = [str(move.get("move_id") or "") for move in moves]
    if not moves or any(not move_id for move_id in move_ids):
        raise ValueError("codebook contains an empty move_id")
    if len(move_ids) != len(set(move_ids)):
        raise ValueError("codebook move_ids must be unique")

    model_ids = set(_by_id(_read_jsonl(args.paired_model_codes), label="model_codes"))
    human_ids = set(left)
    allowed_evidence_ids = model_ids | human_ids
    for move in moves:
        if not str(move.get("definition") or "").strip():
            raise ValueError(f"{move['move_id']}: empty definition")
        if not move.get("behavior_regions"):
            raise ValueError(f"{move['move_id']}: no behavior region")
        evidence_ids = set(move.get("source_evidence_ids") or [])
        counterexample_ids = set(move.get("counterexample_or_boundary_ids") or [])
        unknown = (evidence_ids | counterexample_ids) - allowed_evidence_ids
        if unknown:
            raise ValueError(f"{move['move_id']}: unknown evidence IDs {sorted(unknown)}")

    field_distributions = {
        field: dict(
            Counter(str(row[field]).lower() for row in merged)
        )
        for field in FIELDS
    }
    behavior_region_counts = Counter(
        region for move in moves for region in move["behavior_regions"]
    )
    evidence_mode_counts = Counter(
        str(move["source_evidence_mode"]) for move in moves
    )
    report = {
        "protocol": "pm-v1.5-strategy-g0-freeze-report-v1",
        "status": "G0_COMPLETE_G1_NOT_STARTED",
        "human_item_count": len(merged),
        "observed_disagreement_item_count": len(actual_disagreement_ids),
        "resolved_disagreement_item_count": len(resolutions),
        "unresolved_disagreement_item_count": 0,
        "field_distributions_after_resolution": field_distributions,
        "mechanically_compatible_source_seed_count": sum(
            row["mechanically_compatible_source_seed"] for row in merged
        ),
        "mechanical_seed_warning": (
            "not final card-source eligibility; G1 must inspect literal source text, "
            "high-stakes context, factual content, and hidden self-disclosure"
        ),
        "atomic_move_count": len(moves),
        "behavior_region_move_memberships": dict(sorted(behavior_region_counts.items())),
        "source_evidence_mode_counts": dict(sorted(evidence_mode_counts.items())),
        "codebook_role": "frozen vocabulary for G1 weak backlabeling and source screening",
        "forbidden_codebook_roles": [
            "PM training labels",
            "final Strategy Cards",
            "ESConv population prevalence",
            "response quality gold"
        ],
        "g0_complete": True,
        "next_gate": "G1_backlabel_9148_train_turns_and_build_source_qualified_cards",
    }

    _write_jsonl(args.out_dir / "adjudicated_human_labels.jsonl", merged)
    _write_json(args.out_dir / "g0_freeze_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
