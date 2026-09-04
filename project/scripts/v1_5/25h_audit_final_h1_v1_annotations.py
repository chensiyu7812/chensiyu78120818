#!/usr/bin/env python3
"""Audit the first final-H1 packet and primary annotations before training.

This audit is deliberately label- and split-aware because its only purpose is
to decide whether the reviewed packet is fit to become PM training gold.  It
does not train a model, choose a threshold, or read a generated reply/outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-h1-v1-data-quality-audit-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")


def _public_payload_key(item: dict[str, Any]) -> str:
    payload = {key: value for key, value in item.items() if key != "blind_state_id"}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_final_h1_candidate_gold_v1_candidate/human_review_packet.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_final_h1_candidate_gold_v1_candidate/private_binding.jsonl",
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_final_h1_v1_data_quality_audit/audit.json",
    )
    args = parser.parse_args()

    packet = read_json(args.packet)
    items = list(packet["items"])
    bindings = {row["blind_state_id"]: row for row in iter_jsonl(args.binding)}
    annotations = {
        row["blind_state_id"]: row for row in iter_jsonl(args.annotations)
    }
    item_ids = {item["blind_state_id"] for item in items}
    if len(items) != len(item_ids) or item_ids != set(bindings) or item_ids != set(annotations):
        raise RuntimeError("packet, binding, and annotation state IDs do not align")

    component_counts: dict[str, Counter[str]] = {
        component: Counter() for component in COMPONENTS
    }
    split_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {component: Counter() for component in COMPONENTS}
    )
    note_templates: dict[str, Counter[str]] = {
        component: Counter() for component in COMPONENTS
    }
    present_total = 0
    for item in items:
        blind_id = item["blind_state_id"]
        split = str(bindings[blind_id]["split"])
        decisions = annotations[blind_id]["component_decisions"]
        for surface in item["components"]:
            component = str(surface["component"])
            decision = str(decisions[component]["decision"])
            if surface["candidate_present"]:
                present_total += 1
                component_counts[component][decision] += 1
                split_counts[split][component][decision] += 1
                note_templates[component][str(decisions[component].get("notes", ""))] += 1

    payload_groups: dict[str, list[str]] = defaultdict(list)
    for item in items:
        payload_groups[_public_payload_key(item)].append(item["blind_state_id"])
    duplicate_groups = [group for group in payload_groups.values() if len(group) > 1]
    cross_split_groups = [
        group
        for group in duplicate_groups
        if len({bindings[blind_id]["split"] for blind_id in group}) > 1
    ]

    component_summary: dict[str, Any] = {}
    for component in COMPONENTS:
        counts = component_counts[component]
        n = sum(counts.values())
        majority = max(counts.get("on", 0), counts.get("off", 0))
        component_summary[component] = {
            "present_candidates": n,
            "on": counts.get("on", 0),
            "off": counts.get("off", 0),
            "abstain": counts.get("abstain", 0),
            "on_rate": counts.get("on", 0) / n if n else None,
            "majority_constant_accuracy": majority / n if n else None,
            "unique_nonempty_note_templates": len(
                {note for note in note_templates[component] if note}
            ),
            "most_common_note": (
                {
                    "count": note_templates[component].most_common(1)[0][1],
                    "text": note_templates[component].most_common(1)[0][0],
                }
                if note_templates[component]
                else None
            ),
        }

    critical_findings = [
        {
            "id": "CROSS_SPLIT_EXACT_DUPLICATES",
            "severity": "CRITICAL",
            "evidence": {
                "exact_duplicate_groups": len(duplicate_groups),
                "states_in_duplicate_groups": sum(len(group) for group in duplicate_groups),
                "cross_split_duplicate_groups": len(cross_split_groups),
            },
            "impact": "Confirmation and sealed estimates would reuse identical public states seen in another split.",
        },
        {
            "id": "MP_DEGENERATE_LABEL_DISTRIBUTION",
            "severity": "CRITICAL",
            "evidence": component_summary["MP"],
            "impact": "An always-OFF classifier already exceeds the planned accuracy gate, so MP learning cannot be demonstrated.",
        },
        {
            "id": "RS_DEGENERATE_AND_INCREMENTALITY_NOT_ADJUDICATED",
            "severity": "CRITICAL",
            "evidence": component_summary["RS"],
            "impact": "An always-ON classifier already exceeds the planned accuracy gate, while the dominant rationale establishes surface fit rather than incremental value over R0.",
        },
        {
            "id": "NO_INDEPENDENT_OVERLAP_LABELS_YET",
            "severity": "HIGH",
            "evidence": {"primary_states": len(annotations), "independent_overlap_received": False},
            "impact": "The pre-registered H1 agreement and kappa gates cannot be evaluated.",
        },
    ]

    result = {
        "protocol": PROTOCOL,
        "status": "INVALID_FOR_TRAINING_REBUILD_H1_PACKET",
        "intended_grain": "one visible state by one exact rank-1 component candidate",
        "states": len(items),
        "present_candidate_judgments": present_total,
        "deterministic_absent_judgments": 4 * len(items) - present_total,
        "component_summary": component_summary,
        "split_component_summary": {
            split: {
                component: dict(counts[component]) for component in COMPONENTS
            }
            for split, counts in sorted(split_counts.items())
        },
        "duplicate_audit": {
            "unique_public_payloads": len(payload_groups),
            "exact_duplicate_groups": len(duplicate_groups),
            "states_in_exact_duplicate_groups": sum(len(group) for group in duplicate_groups),
            "duplicate_excess_rows": sum(len(group) - 1 for group in duplicate_groups),
            "cross_split_exact_duplicate_groups": len(cross_split_groups),
            "examples": duplicate_groups[:12],
        },
        "critical_findings": critical_findings,
        "decision": {
            "annotations_usable_as_final_gold": False,
            "annotations_usable_as_development_diagnostic": True,
            "training_authorized": False,
            "required_remediation": [
                "rebuild unique public states and enforce zero exact/near-template leakage across splits",
                "construct genuinely incremental MP preference/profile positives rather than current-turn echoes",
                "construct MS positives that add a prior distinction/outcome/open goal beyond the visible turn",
                "separate RS card surface fit from incremental expected value over R0 and balance both ON and OFF candidate-present cases",
                "freeze the one-point boundary rule before review",
                "run static shortcut and class-balance gates before requesting another human label",
            ],
        },
        "provenance": {
            "packet_sha256": sha256_file(args.packet),
            "binding_sha256": sha256_file(args.binding),
            "annotations_sha256": sha256_file(args.annotations),
            "generated_response_or_outcome_read": False,
            "external_lockbox_read": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    write_jsonl(
        args.out.parent / "component_metrics.jsonl",
        [
            {
                "component": component,
                "present": component_summary[component]["present_candidates"],
                "on": component_summary[component]["on"],
                "off": component_summary[component]["off"],
                "on_rate": component_summary[component]["on_rate"],
                "majority_constant_accuracy": component_summary[component][
                    "majority_constant_accuracy"
                ],
                "decision": (
                    "INVALID / rebuild"
                    if component in {"MP", "RS"}
                    else "diagnostic only"
                ),
            }
            for component in COMPONENTS
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
