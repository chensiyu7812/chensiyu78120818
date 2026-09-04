#!/usr/bin/env python3
"""Aggregate the separated-construct repaired Step2 human check."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-repaired-resource-execution-human-check-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "outputs/pm_v1_5_repaired_resource_execution_human_check_v1_candidate")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_repaired_resource_execution_human_check_v1")
    args = parser.parse_args()
    packet = _rows(args.candidate_dir / "human_review_packet.jsonl")
    public = {str(row["review_item_id"]): row for row in packet}
    annotations = _rows(args.annotations)
    if len(annotations) != len(packet) or len({str(r.get("review_item_id")) for r in annotations}) != len(packet):
        raise RuntimeError("annotations must contain exactly one row for every packet item")
    allowed3 = {"yes", "no", "uncertain"}
    allowed4 = {"yes", "no", "not_applicable", "uncertain"}
    for row in annotations:
        rid = str(row.get("review_item_id"))
        if row.get("protocol") != PROTOCOL or rid not in public:
            raise RuntimeError(f"unknown protocol or review_item_id: {rid}")
        if any(row.get(k) not in allowed3 for k in ("adjudicable", "declaration_matches_response", "resource_decision_appropriate", "material_misuse")):
            raise RuntimeError(f"invalid ternary field: {rid}")
        if row.get("functional_execution") not in allowed4:
            raise RuntimeError(f"invalid functional_execution: {rid}")
        if not str(row.get("literal_response_excerpt", "")).strip() or not str(row.get("review_notes", "")).strip() or not str(row.get("annotator_id", "")).strip():
            raise RuntimeError(f"missing evidence, note, or annotator: {rid}")
    adjudicable = [row for row in annotations if row["adjudicable"] == "yes"]
    n = len(adjudicable)
    declared_use = [row for row in adjudicable if public[str(row["review_item_id"])]["declared_decision"] == "use"]
    rates = {
        "adjudicable": f"{n}/{len(packet)}",
        "declaration_matches_response": sum(r["declaration_matches_response"] == "yes" for r in adjudicable) / n if n else 0.0,
        "resource_decision_appropriate": sum(r["resource_decision_appropriate"] == "yes" for r in adjudicable) / n if n else 0.0,
        "functional_execution_on_declared_use": sum(r["functional_execution"] == "yes" for r in declared_use) / len(declared_use) if declared_use else 0.0,
        "material_misuse_count": sum(r["material_misuse"] == "yes" for r in adjudicable),
    }
    gates = {
        "adjudicable_at_least_8": n >= 8,
        "declaration_match_at_least_0_8": rates["declaration_matches_response"] >= 0.8,
        "decision_appropriate_at_least_0_8": rates["resource_decision_appropriate"] >= 0.8,
        "functional_use_at_least_0_8": rates["functional_execution_on_declared_use"] >= 0.8,
        "material_misuse_at_most_1": rates["material_misuse_count"] <= 1,
    }
    passed = all(gates.values())
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_POSITIVE_EXECUTION_FREEZE_PROMPT" if passed else "FAIL_STEP2_REPORT_LIMITATION_DO_NOT_CHANGE_PM_LABELS",
        "scope": "positive-opportunity execution and misuse only; ignore path not independently qualified",
        "counts": {"items": len(packet), "adjudicability": dict(Counter(r["adjudicable"] for r in annotations)), "component": dict(Counter(public[str(r["review_item_id"])]["component"] for r in annotations))},
        "rates": rates,
        "gates": gates,
        "training_gold": False,
        "external_test_feedback": False,
        "input_sha256": sha256_file(args.annotations),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "annotations_frozen.jsonl", annotations)
    write_json(args.out_dir / "qualification_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
