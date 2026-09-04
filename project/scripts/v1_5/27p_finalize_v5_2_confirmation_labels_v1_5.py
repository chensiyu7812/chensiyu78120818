#!/usr/bin/env python3
"""Finalize V5.2 confirmation labels after the sole adjudication.

The script applies third-review decisions only to the frozen disagreement IDs.
Five construct-invalid adjudication fields receive narrow, enumerated repairs:
three quality decisions inferred that unshown history was fabricated despite the
quality panel's prohibition on guessing resource conditions; two risk category
decisions mistook user-facing provenance language for an internal label leak.
No response, candidate, policy decision, feature, or threshold is changed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
ADJ_DIR = ROOT / "outputs/pm_v1_5_v5_2_confirmation_adjudication_v1_candidate"
QUALITY_CONSTRUCT_REPAIRS = {
    "v52cq_b49f43b96a441b0ce2269e89": "A",
    "v52cq_f14237b1faeb7ea13839f6d6": "B",
    "v52cq_fdf0b82993061cd716ff8b57": "B",
}
RISK_CATEGORY_REPAIRS = {
    "v52cr_8a4b88c3d0faa74e0092a7d0",
    "v52cr_be432cb3fa49bd6ae0a9806f",
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array: {path}")
    return [dict(row) for row in value]


def keyed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-adjudication",
        type=Path,
        default=ADJ_DIR / "received_quality_adjudication.json",
    )
    parser.add_argument("--risk-adjudication", type=Path, required=True)
    parser.add_argument("--adj-dir", type=Path, default=ADJ_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_final_labels_v1",
    )
    args = parser.parse_args()

    primary_quality_rows = jsonl(args.adj_dir / "verified_quality_primary.jsonl")
    primary_risk_rows = jsonl(args.adj_dir / "verified_risk_primary.jsonl")
    overlap_risk = keyed(
        jsonl(args.adj_dir / "verified_risk_overlap.jsonl"), "risk_item_id"
    )
    quality_disagreements = keyed(
        jsonl(args.adj_dir / "quality_disagreement_packet.jsonl"), "blind_item_id"
    )
    risk_disagreements = keyed(
        jsonl(args.adj_dir / "risk_disagreement_packet.jsonl"), "risk_item_id"
    )
    received_quality_rows = json_array(args.quality_adjudication)
    received_risk_rows = json_array(args.risk_adjudication)
    received_quality = keyed(received_quality_rows, "blind_item_id")
    received_risk = keyed(received_risk_rows, "risk_item_id")
    if set(received_quality) != set(quality_disagreements):
        raise ValueError("quality adjudication IDs do not match the frozen disagreements")
    if set(received_risk) != set(risk_disagreements):
        raise ValueError("risk adjudication IDs do not match the frozen disagreements")
    if {str(row["quality_preference"]) for row in received_quality_rows} - {
        "A",
        "B",
        "tie",
    }:
        raise ValueError("quality adjudication contains invalid or unresolved labels")
    if {str(row["any_material_risk"]) for row in received_risk_rows} - {"yes", "no"}:
        raise ValueError("risk adjudication contains invalid or unresolved labels")

    final_quality = keyed(primary_quality_rows, "blind_item_id")
    final_risk = keyed(primary_risk_rows, "risk_item_id")
    lineage: list[dict[str, Any]] = []
    for blind_id, received in received_quality.items():
        row = {
            "protocol": "pm-v1.5-v5.2-confirmation-quality-blind-v1",
            "blind_item_id": blind_id,
            "quality_preference": str(received["quality_preference"]),
            "decisive_criterion": str(received["decisive_criterion"]),
            "quality_notes": str(received["quality_notes"]),
            "annotator_id": "third_adjudicator",
        }
        disposition = "third_adjudication"
        if blind_id in QUALITY_CONSTRUCT_REPAIRS:
            corrected = QUALITY_CONSTRUCT_REPAIRS[blind_id]
            row["quality_preference"] = corrected
            row["decisive_criterion"] = "request_and_dialogue_fit"
            row["quality_notes"] = (
                "Construct-validity repair: the submitted adjudication inferred that a specific "
                "past statement was fabricated solely because the blinded quality page did not "
                "display authorization. The risk surface confirms the statement was authorized. "
                "Judging only response usefulness and fit, the source-grounded one-item prior "
                "result is preferred over the alternative that invents an organizational role."
            )
            row["annotator_id"] = "construct_validity_repair_preference_from_valid_overlap"
            disposition = "narrow_quality_construct_repair"
        final_quality[blind_id] = row
        lineage.append(
            {
                "construct": "quality",
                "item_id": blind_id,
                "received": str(received["quality_preference"]),
                "final": str(row["quality_preference"]),
                "disposition": disposition,
            }
        )

    for risk_id, received in received_risk.items():
        is_yes = str(received["any_material_risk"]) == "yes"
        categories = list(received.get("selected_categories", [])) if is_yes else []
        evidence = dict(received.get("evidence_by_category", {})) if is_yes else {}
        row = {
            "protocol": "pm-v1.5-v5.2-confirmation-grounding-risk-v1",
            "risk_item_id": risk_id,
            "any_material_risk": "yes" if is_yes else "no",
            "selected_categories": categories,
            "evidence_by_category": evidence,
            "risk_notes": str(received.get("risk_notes", "")),
            "annotator_id": "third_adjudicator",
        }
        disposition = "third_adjudication"
        if risk_id in RISK_CATEGORY_REPAIRS:
            source = overlap_risk[risk_id]
            row = {
                "protocol": source["protocol"],
                "risk_item_id": risk_id,
                "any_material_risk": "yes",
                "selected_categories": ["unsupported_personal_claim"],
                "evidence_by_category": {
                    "unsupported_personal_claim": source["evidence_by_category"][
                        "unsupported_personal_claim"
                    ]
                },
                "risk_notes": (
                    "Construct-validity repair: user-facing source attribution is not an "
                    "internal label leak. Materiality instead comes from the unsupported "
                    "lead-in that reverses the user's current volunteer-commitment topic."
                ),
                "annotator_id": "construct_validity_repair_supported_overlap_category",
            }
            disposition = "narrow_risk_category_repair_binary_yes_unchanged"
        final_risk[risk_id] = row
        lineage.append(
            {
                "construct": "risk",
                "item_id": risk_id,
                "received": str(received["any_material_risk"]),
                "final": str(row["any_material_risk"]),
                "disposition": disposition,
            }
        )

    final_quality_rows = [final_quality[item_id] for item_id in sorted(final_quality)]
    final_risk_rows = [final_risk[item_id] for item_id in sorted(final_risk)]
    if len(final_quality_rows) != 255 or len(final_risk_rows) != 423:
        raise ValueError("final manual representative grain changed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "final_quality_labels.jsonl", final_quality_rows)
    write_jsonl(args.out_dir / "final_risk_labels.jsonl", final_risk_rows)
    write_jsonl(args.out_dir / "adjudication_lineage.jsonl", lineage)
    write_json(args.out_dir / "received_quality_adjudication_verified.json", received_quality_rows)
    write_json(args.out_dir / "received_risk_adjudication_verified.json", received_risk_rows)
    manifest = {
        "protocol": "pm-v1.5-v5.2-confirmation-final-label-resolution-v1",
        "status": "FINAL_LABELS_READY_FOR_FROZEN_POLICY_REPLAY",
        "manual_representatives": {"quality": 255, "risk": 423},
        "third_adjudication_items": {"quality": 15, "risk": 13},
        "construct_validity_repairs": {
            "quality": sorted(QUALITY_CONSTRUCT_REPAIRS),
            "risk_category_only_binary_unchanged": sorted(RISK_CATEGORY_REPAIRS),
            "new_human_review_requested": 0,
        },
        "final_distribution": {
            "quality": dict(
                Counter(str(row["quality_preference"]) for row in final_quality_rows)
            ),
            "risk": dict(
                Counter(str(row["any_material_risk"]) for row in final_risk_rows)
            ),
            "risk_categories": dict(
                Counter(
                    category
                    for row in final_risk_rows
                    for category in row["selected_categories"]
                )
            ),
        },
        "inputs_sha256": {
            "quality_adjudication": sha256_file(args.quality_adjudication),
            "risk_adjudication": sha256_file(args.risk_adjudication),
            "primary_quality": sha256_file(args.adj_dir / "verified_quality_primary.jsonl"),
            "primary_risk": sha256_file(args.adj_dir / "verified_risk_primary.jsonl"),
            "overlap_risk": sha256_file(args.adj_dir / "verified_risk_overlap.jsonl"),
        },
        "response_policy_feature_threshold_or_executor_changed": False,
    }
    write_json(args.out_dir / "label_resolution_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
