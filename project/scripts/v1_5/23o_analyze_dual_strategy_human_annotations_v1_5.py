#!/usr/bin/env python3
"""Validate and compare the two 72-item Strategy open-coding adjudications."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


BOOLEAN_FIELDS = (
    "meaningful_support_action",
    "mainly_information",
    "mainly_self_disclosure",
    "reusable_as_general_technique",
)
RISK_FIELD = "clear_risk_or_boundary_problem"
ALL_FIELDS = (*BOOLEAN_FIELDS, RISK_FIELD)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(rows) != 72:
        raise ValueError(f"{path}: expected 72 rows, found {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        item_id = str(row.get("blind_item_id") or "")
        if not item_id or item_id in by_id:
            raise ValueError(f"{path}: empty or duplicate ID at row {index}")
        for field in BOOLEAN_FIELDS:
            if row.get(field) not in (True, False):
                raise ValueError(f"{path}: {item_id} invalid {field}")
        if row.get(RISK_FIELD) not in (True, False, "uncertain"):
            raise ValueError(f"{path}: {item_id} invalid {RISK_FIELD}")
        meaningful = row["meaningful_support_action"]
        primary = str(row.get("primary_action") or "").strip()
        if meaningful and not primary:
            raise ValueError(f"{path}: {item_id} meaningful but primary action empty")
        if not meaningful and primary:
            raise ValueError(f"{path}: {item_id} nonmeaningful but primary action present")
        risk = row[RISK_FIELD]
        risk_description = str(row.get("risk_description") or "").strip()
        if risk is True and not risk_description:
            raise ValueError(f"{path}: {item_id} risk true but description empty")
        if risk is False and risk_description:
            raise ValueError(f"{path}: {item_id} risk false but description present")
        by_id[item_id] = row
    return by_id


def _cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_positive = sum(left for left, _ in pairs) / len(pairs)
    right_positive = sum(right for _, right in pairs) / len(pairs)
    expected = (
        left_positive * right_positive
        + (1.0 - left_positive) * (1.0 - right_positive)
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _field_report(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    ids = sorted(left)
    exact = [left[item_id][field] == right[item_id][field] for item_id in ids]
    boolean_pairs = [
        (left[item_id][field], right[item_id][field])
        for item_id in ids
        if left[item_id][field] in (True, False)
        and right[item_id][field] in (True, False)
    ]
    disagreements = [item_id for item_id, same in zip(ids, exact) if not same]
    return {
        "exact_agreement_count": sum(exact),
        "total_count": len(ids),
        "exact_agreement_rate": round(sum(exact) / len(ids), 8),
        "boolean_pair_count": len(boolean_pairs),
        "cohen_kappa": (
            None
            if _cohen_kappa(boolean_pairs) is None
            else round(float(_cohen_kappa(boolean_pairs)), 8)
        ),
        "left_distribution": dict(
            Counter(str(left[item_id][field]).lower() for item_id in ids)
        ),
        "right_distribution": dict(
            Counter(str(right[item_id][field]).lower() for item_id in ids)
        ),
        "disagreement_count": len(disagreements),
        "disagreement_ids": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-a", type=Path, required=True)
    parser.add_argument("--annotation-b", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--paired-model-codes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    left_rows = _load_jsonl(args.annotation_a)
    right_rows = _load_jsonl(args.annotation_b)
    left = _validate_rows(args.annotation_a, left_rows)
    right = _validate_rows(args.annotation_b, right_rows)
    if set(left) != set(right):
        raise ValueError("annotation ID sets differ")

    packet_rows = _load_jsonl(args.packet)
    packet = {str(row["blind_item_id"]): row for row in packet_rows}
    if set(packet) != set(left):
        raise ValueError("annotation and 72-item packet ID sets differ")

    paired_rows = _load_jsonl(args.paired_model_codes)
    if len(paired_rows) != 160:
        raise ValueError("expected 160 paired model-code rows")

    disagreements: list[dict[str, Any]] = []
    for item_id in sorted(left):
        field_differences = {
            field: {
                "annotation_a": left[item_id][field],
                "annotation_b": right[item_id][field],
            }
            for field in ALL_FIELDS
            if left[item_id][field] != right[item_id][field]
        }
        if field_differences:
            disagreements.append(
                {
                    "blind_item_id": item_id,
                    "supporter_response": packet[item_id]["supporter_response"],
                    "field_differences": field_differences,
                    "annotation_a": left[item_id],
                    "annotation_b": right[item_id],
                }
            )

    robust_llm_source_ids = [
        item_id
        for item_id in sorted(left)
        if left[item_id]["meaningful_support_action"] is True
        and right[item_id]["meaningful_support_action"] is True
        and left[item_id]["reusable_as_general_technique"] is True
        and right[item_id]["reusable_as_general_technique"] is True
        and left[item_id][RISK_FIELD] is False
        and right[item_id][RISK_FIELD] is False
        and left[item_id]["mainly_information"] is False
        and right[item_id]["mainly_information"] is False
        and left[item_id]["mainly_self_disclosure"] is False
        and right[item_id]["mainly_self_disclosure"] is False
    ]

    coder_a_nonliteral_72 = sum(
        "evidence_excerpt_not_literal"
        in row["coder_a"].get("validation_flags", [])
        for row in packet_rows
    )
    coder_a_nonliteral_160 = sum(
        "evidence_excerpt_not_literal"
        in row["coder_a"].get("validation_flags", [])
        for row in paired_rows
    )
    coder_b_nonliteral_160 = sum(
        "evidence_excerpt_not_literal"
        in row["coder_b"].get("validation_flags", [])
        for row in paired_rows
    )

    report = {
        "protocol": "pm-v1.5-dual-strategy-human-annotation-analysis-v1",
        "status": "VALIDATED_DISCOVERY_EVIDENCE_NOT_PM_TRAINING_GOLD",
        "inputs": {
            "annotation_a": {
                "path": str(args.annotation_a),
                "sha256": _sha256(args.annotation_a),
            },
            "annotation_b": {
                "path": str(args.annotation_b),
                "sha256": _sha256(args.annotation_b),
            },
            "packet": {"path": str(args.packet), "sha256": _sha256(args.packet)},
            "paired_model_codes": {
                "path": str(args.paired_model_codes),
                "sha256": _sha256(args.paired_model_codes),
            },
        },
        "row_count": 72,
        "same_id_set": True,
        "field_reports": {
            field: _field_report(left, right, field) for field in ALL_FIELDS
        },
        "items_with_any_field_disagreement": len(disagreements),
        "robust_llm_compatible_source_seed_count": len(robust_llm_source_ids),
        "robust_llm_compatible_source_seed_ids": robust_llm_source_ids,
        "model_coder_grounding": {
            "coder_a_nonliteral_72": coder_a_nonliteral_72,
            "coder_a_nonliteral_160": coder_a_nonliteral_160,
            "coder_b_nonliteral_160": coder_b_nonliteral_160,
            "interpretation": (
                "Coder A is invalid as a voting or gold source. Coder B remains "
                "a weak proposal generator, not an independent gold annotator."
            ),
        },
        "allowed_use": [
            "inductive atomic-move codebook discovery",
            "source-screening rule design",
            "targeted disagreement adjudication",
        ],
        "forbidden_use": [
            "direct PM on/off training labels",
            "population prevalence estimates from the selected Tier-1 sample",
            "automatic card eligibility by majority vote",
            "formal response-quality or PM-risk outcome labels",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dual_annotation_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "disagreements.jsonl").open("w", encoding="utf-8") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
