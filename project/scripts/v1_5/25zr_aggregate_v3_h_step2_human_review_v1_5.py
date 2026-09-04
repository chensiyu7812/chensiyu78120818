#!/usr/bin/env python3
"""Aggregate the frozen 28-item V3 H-Step2 human qualification."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v3-h-step2-functional-human-review-v2"
CANDIDATE_DIR = ROOT / "outputs/pm_v1_5_v3_h_step2_human_review_v2_candidate"
OUT_DIR = ROOT / "outputs/pm_v1_5_v3_h_step2_human_review_v2"
TRI = {"yes", "no", "uncertain"}
MISUSE = {
    "fabricated_recall",
    "stale_or_conflicting_use",
    "unsupported_personal_claim",
    "overgeneralized_pattern_or_cause",
    "boundary_violation",
    "excessive_directiveness",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _categories(value: Any) -> list[str]:
    return sorted(
        {
            token.strip()
            for token in str(value or "").replace(";", ",").split(",")
            if token.strip()
        }
    )


def aggregate(
    *, annotations: list[dict[str, Any]], packet: list[dict[str, Any]], private: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    public_by_id = {str(row["review_item_id"]): row for row in packet}
    private_by_id = {str(row["review_item_id"]): row for row in private}
    annotation_by_id = {str(row.get("review_item_id") or ""): row for row in annotations}
    expected = set(public_by_id)
    if (
        len(packet) != 28
        or len(public_by_id) != 28
        or len(annotations) != len(annotation_by_id)
        or set(annotation_by_id) != expected
        or set(private_by_id) != expected
    ):
        raise RuntimeError("annotations must cover the exact 28-item packet")
    frozen: list[dict[str, Any]] = []
    for review_id in sorted(expected):
        source = annotation_by_id[review_id]
        public = public_by_id[review_id]
        hidden = private_by_id[review_id]
        if source.get("protocol") != PROTOCOL:
            raise RuntimeError(f"{review_id}: protocol mismatch")
        components = [str(value) for value in public["requested_components"]]
        judgments = source.get("component_functional")
        if not isinstance(judgments, dict) or set(judgments) != set(components):
            raise RuntimeError(f"{review_id}: component coverage mismatch")
        if any(judgments[component] not in TRI for component in components):
            raise RuntimeError(f"{review_id}: invalid functional value")
        if source.get("adjudicable") not in TRI or source.get("material_misuse") not in TRI:
            raise RuntimeError(f"{review_id}: invalid tri-state value")
        categories = _categories(source.get("misuse_categories"))
        if set(categories) - MISUSE:
            raise RuntimeError(f"{review_id}: unknown misuse category")
        if (source["material_misuse"] == "yes") != bool(categories):
            raise RuntimeError(f"{review_id}: misuse category/value mismatch")
        fallback_expected = bool(public.get("fallback_response"))
        fallback = str(source.get("fallback_usable") or "")
        if fallback not in {"yes", "no", "uncertain", "not_applicable"}:
            raise RuntimeError(f"{review_id}: invalid fallback value")
        if fallback_expected != (fallback != "not_applicable"):
            raise RuntimeError(f"{review_id}: fallback visibility mismatch")
        annotator = str(source.get("annotator_id") or "").strip()
        excerpt = str(source.get("literal_response_excerpt") or "").strip()
        notes = str(source.get("review_notes") or "").strip()
        if not annotator or not excerpt or not notes:
            raise RuntimeError(f"{review_id}: annotator, excerpt and notes required")
        if excerpt != "[none]" and excerpt not in str(public["application_response"]):
            raise RuntimeError(f"{review_id}: response excerpt is not literal")
        frozen.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "call_id": hidden["call_id"],
                "state_id": hidden["state_id"],
                "call_kind": hidden["call_kind"],
                "requested_action_id": public["requested_action_id"],
                "requested_components": components,
                "adjudicable": source["adjudicable"],
                "component_functional": {component: judgments[component] for component in components},
                "all_requested_components_functional": all(judgments[c] == "yes" for c in components),
                "material_misuse": source["material_misuse"],
                "misuse_categories": categories,
                "fallback_usable": fallback,
                "machine_fallback_executed": bool(hidden["fallback_executed_private"]),
                "machine_schema_failure": bool(hidden["schema_failure_private"]),
                "literal_response_excerpt": excerpt,
                "review_notes": notes,
                "annotator_id": annotator,
            }
        )
    singles = [row for row in frozen if row["call_kind"] == "single"]
    multis = [row for row in frozen if row["call_kind"] == "multi"]
    functional_by_component: dict[str, dict[str, int]] = {}
    for component in ("MP", "MS", "ME", "RS"):
        values = [row["component_functional"][component] for row in singles if component in row["component_functional"]]
        functional_by_component[component] = {
            "n": len(values),
            "yes": values.count("yes"),
            "no": values.count("no"),
            "uncertain": values.count("uncertain"),
            "minimum_yes": 4,
        }
    misuse_rows = [row for row in frozen if row["material_misuse"] == "yes"]
    category_counts = Counter(category for row in frozen for category in row["misuse_categories"])
    fallback_rows = [row for row in frozen if row["machine_fallback_executed"]]
    gates = {
        "all_28_adjudicable": all(row["adjudicable"] == "yes" for row in frozen),
        "single_component_functional_minimum_4_of_5_each": all(
            functional_by_component[component]["n"] == 5
            and functional_by_component[component]["yes"] >= 4
            for component in functional_by_component
        ),
        "multi_all_requested_functional_minimum_6_of_8": sum(
            row["all_requested_components_functional"] for row in multis
        ) >= 6,
        "material_misuse_at_most_1_of_28": len(misuse_rows) <= 1,
        "fabricated_recall_zero": category_counts["fabricated_recall"] == 0,
        "explicit_boundary_violation_zero": category_counts["boundary_violation"] == 0,
        "all_displayed_fallbacks_adoptable": all(
            row["fallback_usable"] == "yes" for row in fallback_rows
        ),
        "zero_uncertain": all(
            row["adjudicable"] != "uncertain"
            and row["material_misuse"] != "uncertain"
            and all(value != "uncertain" for value in row["component_functional"].values())
            and row["fallback_usable"] != "uncertain"
            for row in frozen
        ),
    }
    passed = all(gates.values())
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_FREEZE_V3_STEP2" if passed else "FAIL_REPAIR_STEP2_BEFORE_EFFECT_GENERATION",
        "passed": passed,
        "items": len(frozen),
        "single_items": len(singles),
        "multi_items": len(multis),
        "functional_by_single_component": functional_by_component,
        "multi_all_requested_functional": sum(
            row["all_requested_components_functional"] for row in multis
        ),
        "material_misuse_items": len(misuse_rows),
        "misuse_category_counts": dict(sorted(category_counts.items())),
        "machine_fallback_items": len(fallback_rows),
        "machine_schema_failure_items": sum(row["machine_schema_failure"] for row in frozen),
        "gates": gates,
        "scientific_scope": (
            "Step2 exact evidence application and interaction-and-grounding misuse only; "
            "not Step1 routing, retrieval accuracy, or pairwise response quality."
        ),
    }
    return report, frozen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    report, rows = aggregate(
        annotations=_rows(args.annotations),
        packet=_rows(args.candidate_dir / "human_review_packet.jsonl"),
        private=_rows(args.candidate_dir / "private_review_key.jsonl"),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "annotations_frozen.jsonl", rows)
    report["inputs"] = {
        "annotations_sha256": sha256_file(args.annotations),
        "packet_sha256": sha256_file(args.candidate_dir / "human_review_packet.jsonl"),
        "private_key_sha256": sha256_file(args.candidate_dir / "private_review_key.jsonl"),
    }
    write_json(args.out_dir / "qualification_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
