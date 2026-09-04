#!/usr/bin/env python3
"""Aggregate the complete primary V1.5b functional review and sensitivity summary."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-resource-application-human-functional-check-v1-aggregate"
EXPECTED_REVIEW_PROTOCOL = "pm-v1.5b-resource-application-human-functional-check-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _component_counts(
    annotations: list[dict[str, Any]], packet: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for component in ("MP", "MS", "ME", "RS"):
        selected = [
            row
            for row in annotations
            if packet[str(row["review_item_id"])]["component"] == component
        ]
        result[component] = {
            "items": len(selected),
            "functional_execution": dict(sorted(Counter(row["functional_execution"] for row in selected).items())),
            "material_misuse": dict(sorted(Counter(row["material_misuse"] for row in selected).items())),
        }
    return result


def build(
    *,
    annotations_path: Path,
    candidate_dir: Path,
    secondary_summary_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    annotations = _rows(annotations_path)
    packet_rows = _rows(candidate_dir / "human_review_packet.jsonl")
    packet = {str(row["review_item_id"]): row for row in packet_rows}
    private = {
        str(row["review_item_id"]): row
        for row in _rows(candidate_dir / "private_review_key.jsonl")
    }
    ids = [str(row.get("review_item_id") or "") for row in annotations]
    expected_ids = set(packet)
    secondary = read_json(secondary_summary_path)
    allowed_tri = {"yes", "no", "uncertain"}
    checks = {
        "exactly_32_annotations": len(annotations) == 32,
        "unique_review_item_ids": len(ids) == len(set(ids)),
        "exact_packet_id_coverage": set(ids) == expected_ids,
        "private_key_same_coverage": set(private) == expected_ids,
        "expected_protocol": all(row.get("protocol") == EXPECTED_REVIEW_PROTOCOL for row in annotations),
        "all_adjudicable_yes": all(row.get("adjudicable") == "yes" for row in annotations),
        "valid_functional_values": all(row.get("functional_execution") in allowed_tri for row in annotations),
        "valid_misuse_values": all(row.get("material_misuse") in allowed_tri for row in annotations),
        "zero_uncertain": all(row.get("functional_execution") != "uncertain" and row.get("material_misuse") != "uncertain" for row in annotations),
        "all_notes_and_literal_evidence_present": all(str(row.get("review_notes") or "").strip() and str(row.get("literal_response_excerpt") or "").strip() for row in annotations),
        "secondary_summary_is_32_item_sensitivity_only": secondary.get("items") == 32 and secondary.get("status") == "SUMMARY_ONLY_NO_ITEM_LEVEL_ROWS_PROVIDED",
    }
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])

    functional_yes = sum(row["functional_execution"] == "yes" for row in annotations)
    misuse_yes = sum(row["material_misuse"] == "yes" for row in annotations)
    fallback_rows = [row for row in annotations if row["fallback_usable"] != "not_applicable"]
    categories = Counter(
        category.strip()
        for row in annotations
        for category in str(row.get("misuse_categories") or "").split(",")
        if category.strip()
    )
    primary_by_component = _component_counts(annotations, packet)
    promotion_checks = {
        "overall_functional_use_min_0_80": functional_yes / 32 >= 0.80,
        "each_source_functional_use_min_6_of_8": all(
            values["functional_execution"].get("yes", 0) >= 6
            for values in primary_by_component.values()
        ),
        "material_misuse_max_1_of_32": misuse_yes <= 1,
        "displayed_fallbacks_usable": len(fallback_rows) == 2 and all(row["fallback_usable"] == "yes" for row in fallback_rows),
    }
    secondary_functional = int(secondary["functional_execution_yes"])
    secondary_misuse = int(secondary["material_misuse_yes"])
    robust = {
        "functional_yes_range": [min(functional_yes, secondary_functional), max(functional_yes, secondary_functional)],
        "functional_rate_range": [min(functional_yes, secondary_functional) / 32, max(functional_yes, secondary_functional) / 32],
        "misuse_yes_range": [min(misuse_yes, secondary_misuse), max(misuse_yes, secondary_misuse)],
        "both_reviews_fail_material_misuse_gate": min(misuse_yes, secondary_misuse) > 1,
        "both_reviews_support_two_of_two_fallbacks_usable": bool(secondary.get("fallback_usable") == 2 and promotion_checks["displayed_fallbacks_usable"]),
        "item_level_iaa_computable": False,
        "reason_item_level_iaa_not_computable": "secondary review supplied aggregate summary only",
    }
    status = "FAIL_GENERATOR_FUNCTIONAL_AND_MISUSE_QUALIFICATION" if not all(promotion_checks.values()) else "PASS_GENERATOR_QUALIFICATION"
    out_dir.mkdir(parents=True, exist_ok=True)
    imported_path = out_dir / "primary_human_annotations.jsonl"
    write_jsonl(imported_path, annotations)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scientific_scope": "Step2 functional adherence and material interaction-and-grounding misuse; not PM routing accuracy or pairwise response quality.",
        "primary_complete_review": {
            "items": 32,
            "adjudicable": 32,
            "functional_execution_yes": functional_yes,
            "functional_execution_no": 32 - functional_yes,
            "functional_execution_rate": functional_yes / 32,
            "material_misuse_yes": misuse_yes,
            "material_misuse_rate": misuse_yes / 32,
            "misuse_categories": dict(sorted(categories.items())),
            "by_component": primary_by_component,
        },
        "secondary_summary_sensitivity": secondary,
        "robust_interpretation": robust,
        "promotion_checks": promotion_checks,
        "checks": checks,
        "fabricated_recall": {
            "minimum_confirmed_items": 2,
            "review_id_suffixes": ["abef0aa9", "a3b86721"],
            "definition": "response attributes a prior outcome or helpfulness claim to the user that is absent from the selected resource",
            "current_misuse_taxonomy_extension_required": True,
        },
        "inputs": {
            "annotations_sha256": sha256_file(annotations_path),
            "packet_sha256": sha256_file(candidate_dir / "human_review_packet.jsonl"),
            "private_key_sha256": sha256_file(candidate_dir / "private_review_key.jsonl"),
            "secondary_summary_sha256": sha256_file(secondary_summary_path),
        },
    }
    write_json(out_dir / "qualification_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "outputs/pm_v1_5b_resource_application_human_check_v1_candidate")
    parser.add_argument("--secondary-summary", type=Path, default=ROOT / "data/pm_v1_5_contracts/v1_5b_resource_application_secondary_sensitivity_summary_v1.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5b_resource_application_human_check_v1")
    args = parser.parse_args()
    report = build(
        annotations_path=args.annotations,
        candidate_dir=args.candidate_dir,
        secondary_summary_path=args.secondary_summary,
        out_dir=args.out_dir,
    )
    print({"protocol": report["protocol"], "status": report["status"], "primary": report["primary_complete_review"], "robust": report["robust_interpretation"], "promotion_checks": report["promotion_checks"]})


if __name__ == "__main__":
    main()
