#!/usr/bin/env python3
"""Aggregate the frozen 16-item corrected Step2 semantic qualification gate."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-corrected-step2-minimum-human-gate-v1"
CANDIDATE_DIR = ROOT / "outputs/pm_v1_5b_corrected_step2_minimum_human_gate_v1_candidate"
OUT_DIR = ROOT / "outputs/pm_v1_5b_corrected_step2_minimum_human_gate_v1"
TRI = {"yes", "no", "uncertain"}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def aggregate(
    *,
    annotations: list[dict[str, Any]],
    packet: list[dict[str, Any]],
    private: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packet_by_id = {str(row["review_item_id"]): row for row in packet}
    private_by_id = {str(row["review_item_id"]): row for row in private}
    annotation_by_id = {str(row.get("review_item_id") or ""): row for row in annotations}
    expected = set(packet_by_id)
    checks = {
        "exactly_16_unique_annotations": len(annotations) == len(annotation_by_id) == 16,
        "exact_packet_coverage": set(annotation_by_id) == expected,
        "private_key_same_coverage": set(private_by_id) == expected,
        "expected_protocol": all(row.get("protocol") == PROTOCOL for row in annotations),
    }
    if not all(checks.values()):
        raise RuntimeError([key for key, value in checks.items() if not value])

    frozen: list[dict[str, Any]] = []
    for review_id in sorted(expected):
        source = annotation_by_id[review_id]
        public = packet_by_id[review_id]
        components = [str(value) for value in public["requested_components"]]
        component_functional = source.get("component_functional")
        if not isinstance(component_functional, dict) or set(component_functional) != set(components):
            raise RuntimeError(f"{review_id}: component judgments must exactly match requested components")
        if source.get("adjudicable") not in TRI or source.get("material_misuse") not in TRI:
            raise RuntimeError(f"{review_id}: invalid adjudicable or material_misuse")
        if any(component_functional[component] not in TRI for component in components):
            raise RuntimeError(f"{review_id}: invalid component functional judgment")
        fallback_expected = bool(public.get("fallback_response"))
        fallback_value = source.get("fallback_usable")
        if fallback_value not in {"yes", "no", "uncertain", "not_applicable"}:
            raise RuntimeError(f"{review_id}: invalid fallback_usable")
        if fallback_expected != (fallback_value != "not_applicable"):
            raise RuntimeError(f"{review_id}: fallback judgment does not match displayed fallback")
        annotator = str(source.get("annotator_id") or "").strip()
        notes = str(source.get("review_notes") or "").strip()
        excerpt = str(source.get("literal_response_excerpt") or "").strip()
        if not annotator or not notes or not excerpt:
            raise RuntimeError(f"{review_id}: annotator, notes, and excerpt are required")
        if excerpt != "[none]" and excerpt not in str(public["application_response"]):
            raise RuntimeError(f"{review_id}: evidence excerpt is not literal")
        machine_fallback = bool(private_by_id[review_id]["fallback_executed_private"])
        frozen.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "call_id": private_by_id[review_id]["call_id"],
                "state_id": private_by_id[review_id]["state_id"],
                "requested_action_id": public["requested_action_id"],
                "requested_components": components,
                "adjudicable": source["adjudicable"],
                "component_functional": {component: component_functional[component] for component in components},
                "all_requested_components_functional": all(component_functional[component] == "yes" for component in components),
                "material_misuse_in_primary": source["material_misuse"],
                "machine_fallback_executed": machine_fallback,
                "material_misuse_reached_user": source["material_misuse"] == "yes" and not machine_fallback,
                "fallback_usable": fallback_value,
                "misuse_categories": str(source.get("misuse_categories") or "").strip(),
                "literal_response_excerpt": excerpt,
                "review_notes": notes,
                "annotator_id": annotator,
            }
        )

    functional_by_component: dict[str, dict[str, int]] = {}
    minimums = {"MP": 6, "MS": 6, "ME": 8, "RS": 8}
    for component in ("MP", "MS", "ME", "RS"):
        values = [
            row["component_functional"][component]
            for row in frozen
            if component in row["component_functional"]
        ]
        functional_by_component[component] = {
            "n": len(values),
            "yes": values.count("yes"),
            "no": values.count("no"),
            "uncertain": values.count("uncertain"),
            "minimum_yes": minimums[component],
        }
    primary_misuse = sum(row["material_misuse_in_primary"] == "yes" for row in frozen)
    delivered_misuse = sum(row["material_misuse_reached_user"] for row in frozen)
    all_functional = sum(row["all_requested_components_functional"] for row in frozen)
    fallback_rows = [row for row in frozen if row["machine_fallback_executed"]]
    gates = {
        "all_16_adjudicable": all(row["adjudicable"] == "yes" for row in frozen),
        "all_components_functional_at_least_14_of_16_items": all_functional >= 14,
        "per_component_functional_minimum": all(
            functional_by_component[component]["yes"] >= minimums[component]
            for component in minimums
        ),
        "primary_material_misuse_at_most_1_of_16": primary_misuse <= 1,
        "delivered_material_misuse_zero": delivered_misuse == 0,
        "every_machine_fallback_human_usable": bool(fallback_rows) and all(
            row["fallback_usable"] == "yes" for row in fallback_rows
        ),
        "zero_uncertain": all(
            row["adjudicable"] != "uncertain"
            and row["material_misuse_in_primary"] != "uncertain"
            and all(value != "uncertain" for value in row["component_functional"].values())
            for row in frozen
        ),
    }
    passed = all(gates.values())
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_FREEZE_CORRECTED_STEP2" if passed else "FAIL_STOP_BEFORE_EXTERNAL_QUALIFICATION",
        "passed": passed,
        "items": len(frozen),
        "independent_states": len({row["state_id"] for row in frozen}),
        "requested_action_counts": dict(sorted(Counter(row["requested_action_id"] for row in frozen).items())),
        "all_requested_components_functional_items": all_functional,
        "functional_by_component": functional_by_component,
        "primary_material_misuse_items": primary_misuse,
        "delivered_material_misuse_items": delivered_misuse,
        "machine_fallback_items": len(fallback_rows),
        "gates": gates,
        "m0_r0_remains_legal": True,
        "scientific_scope": "Step2 functional execution and interaction-and-grounding misuse only; not PM routing accuracy, retrieval quality, or pairwise reply quality.",
    }
    return report, frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    report, frozen = aggregate(
        annotations=_rows(args.annotations),
        packet=_rows(args.candidate_dir / "human_review_packet.jsonl"),
        private=_rows(args.candidate_dir / "private_review_key.jsonl"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "annotations_frozen.jsonl", frozen)
    report["inputs"] = {
        "annotations_sha256": sha256_file(args.annotations),
        "packet_sha256": sha256_file(args.candidate_dir / "human_review_packet.jsonl"),
        "private_key_sha256": sha256_file(args.candidate_dir / "private_review_key.jsonl"),
    }
    write_json(args.out_dir / "qualification_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
