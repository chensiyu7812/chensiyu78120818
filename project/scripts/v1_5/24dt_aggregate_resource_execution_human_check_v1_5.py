#!/usr/bin/env python3
"""Validate and aggregate the fixed 10-item resource-execution human check."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-resource-execution-minimum-human-check-v1"
STATUS_PASS = "PASS_FREEZE_STEP2_AND_PROCEED_TO_16_ACTION_RUNTIME"
STATUS_FAIL = "REPAIR_GENERATOR_ONLY_DO_NOT_CHANGE_PM_LABELS"
CHOICES = {"yes", "no", "uncertain"}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def aggregate(
    *,
    annotations: list[dict[str, Any]],
    packet: list[dict[str, Any]],
    private_key: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packet_by_id = {str(row["review_item_id"]): row for row in packet}
    private_by_id = {str(row["review_item_id"]): row for row in private_key}
    annotation_by_id = {
        str(row["review_item_id"]): row for row in annotations
    }
    expected_ids = set(packet_by_id)
    if (
        len(packet) != 10
        or len(packet_by_id) != 10
        or len(private_by_id) != 10
        or len(annotation_by_id) != 10
        or set(private_by_id) != expected_ids
        or set(annotation_by_id) != expected_ids
    ):
        raise RuntimeError("human check must contain exactly the fixed 10 review items")

    frozen: list[dict[str, Any]] = []
    annotators: set[str] = set()
    for review_id in sorted(expected_ids):
        source = annotation_by_id[review_id]
        public = packet_by_id[review_id]
        if source.get("protocol") != PROTOCOL:
            raise RuntimeError(f"{review_id}: protocol mismatch")
        for field in (
            "declaration_supported",
            "functionally_used",
            "material_misuse",
        ):
            if source.get(field) not in CHOICES:
                raise RuntimeError(f"{review_id}: invalid or missing {field}")
        annotator = str(source.get("annotator_id") or "").strip()
        notes = str(source.get("review_notes") or "").strip()
        excerpt = str(source.get("literal_response_excerpt") or "").strip()
        if not annotator or not notes or not excerpt:
            raise RuntimeError(f"{review_id}: annotator, notes, and excerpt are required")
        if excerpt != "[none]" and excerpt not in str(public["response"]):
            raise RuntimeError(f"{review_id}: response excerpt is not literal")
        if (
            public["declared_decision"] == "use"
            and source["declaration_supported"] == "yes"
            and excerpt == "[none]"
        ):
            raise RuntimeError(f"{review_id}: supported use requires a response excerpt")
        annotators.add(annotator)
        qualifies = (
            source["declaration_supported"] == "yes"
            and source["functionally_used"] == "yes"
            and source["material_misuse"] == "no"
        )
        frozen.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "state_id": private_by_id[review_id]["state_id"],
                "component": public["component"],
                "resource_subtype": public["resource_subtype"],
                "declared_decision": public["declared_decision"],
                "declaration_supported": source["declaration_supported"],
                "functionally_used": source["functionally_used"],
                "material_misuse": source["material_misuse"],
                "literal_response_excerpt": excerpt,
                "review_notes": notes,
                "annotator_id": annotator,
                "qualifies_execution": qualifies,
            }
        )

    supported_and_functional = sum(
        int(
            row["declaration_supported"] == "yes"
            and row["functionally_used"] == "yes"
        )
        for row in frozen
    )
    material_misuse = sum(
        int(row["material_misuse"] == "yes") for row in frozen
    )
    uncertain = sum(
        int(
            "uncertain"
            in {
                row["declaration_supported"],
                row["functionally_used"],
                row["material_misuse"],
            }
        )
        for row in frozen
    )
    passed = supported_and_functional >= 8 and material_misuse <= 1
    by_component: dict[str, dict[str, int]] = {}
    for component in ("RS", "MP", "MS", "ME"):
        subset = [row for row in frozen if row["component"] == component]
        by_component[component] = {
            "n": len(subset),
            "supported_and_functional": sum(
                int(
                    row["declaration_supported"] == "yes"
                    and row["functionally_used"] == "yes"
                )
                for row in subset
            ),
            "material_misuse": sum(
                int(row["material_misuse"] == "yes") for row in subset
            ),
        }
    report = {
        "protocol": PROTOCOL,
        "status": STATUS_PASS if passed else STATUS_FAIL,
        "passed": passed,
        "items": len(frozen),
        "independent_states": len({row["state_id"] for row in frozen}),
        "declared_decision_counts": dict(
            sorted(Counter(row["declared_decision"] for row in frozen).items())
        ),
        "supported_and_functional_n": supported_and_functional,
        "supported_and_functional_rate": supported_and_functional / len(frozen),
        "material_misuse_n": material_misuse,
        "material_misuse_rate": material_misuse / len(frozen),
        "uncertain_item_n": uncertain,
        "component_results": by_component,
        "annotator_count": len(annotators),
        "gate": {
            "supported_and_functional_required": "at_least_8_of_10",
            "material_misuse_allowed": "at_most_1_of_10",
            "uncertain_counts_as_pass": False,
        },
        "next_action": (
            "Freeze the minimum Step2 execution interface, preserve all 16 actions, and run the joint four-head runtime preflight."
            if passed
            else "Repair only source-matched generator execution; do not relabel PM routing, change retrieval gold, or add routing data."
        ),
    }
    return report, frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_resource_execution_human_check_v1_candidate",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_resource_execution_human_check_v1",
    )
    args = parser.parse_args()
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    report, frozen = aggregate(
        annotations=_rows(args.annotations),
        packet=_rows(args.packet_dir / "human_review_packet.jsonl"),
        private_key=_rows(args.packet_dir / "private_review_key.jsonl"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "annotations_frozen.jsonl", frozen)
    report["inputs"] = {
        "human_annotations_sha256": sha256_file(args.annotations),
        "human_review_packet_sha256": sha256_file(
            args.packet_dir / "human_review_packet.jsonl"
        ),
        "private_review_key_sha256": sha256_file(
            args.packet_dir / "private_review_key.jsonl"
        ),
    }
    write_json(args.out_dir / "qualification_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
