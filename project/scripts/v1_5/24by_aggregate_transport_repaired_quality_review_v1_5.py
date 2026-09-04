#!/usr/bin/env python3
"""Validate and aggregate the final four-component blind quality review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    "pm-v1.5-transport-repaired-four-component-human-quality-blind-v1"
)
AGGREGATION_PROTOCOL = (
    "pm-v1.5-transport-repaired-four-component-quality-aggregation-v1"
)
ALLOWED_PREFERENCES = {"A", "B", "tie", "uncertain"}
ALLOWED_CRITERIA = {
    "visible_context_fidelity",
    "immediate_helpfulness",
    "request_and_dialogue_fit",
    "emotional_understanding",
    "clarity_naturalness",
    "materially_equivalent",
}


def _semantic_verdict(annotation: dict[str, Any], key: dict[str, Any]) -> str:
    preference = str(annotation["quality_preference"])
    if preference == "A":
        return str(key["a_role"])
    if preference == "B":
        return str(key["b_role"])
    return preference


def _count_nested(
    rows: list[dict[str, Any]],
    outer: str,
    inner: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row[outer])][str(row[inner])] += 1
    return {
        key: dict(sorted(value.items()))
        for key, value in sorted(counts.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_blind_v1",
    )
    parser.add_argument(
        "--runtime-states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "runtime_states.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_quality_aggregation_v1",
    )
    args = parser.parse_args()

    annotations = [dict(row) for row in iter_jsonl(args.annotations)]
    private_keys = [
        dict(row)
        for row in iter_jsonl(args.packet_dir / "private_blind_key.jsonl")
    ]
    public_packet = [
        dict(row)
        for row in iter_jsonl(args.packet_dir / "human_blind_packet.jsonl")
    ]
    runtime_states = {
        str(row["state_id"]): dict(row)
        for row in iter_jsonl(args.runtime_states)
    }

    annotation_by_id = {
        str(row["blind_item_id"]): row for row in annotations
    }
    key_by_id = {
        str(row["blind_item_id"]): row for row in private_keys
    }
    packet_by_id = {
        str(row["blind_item_id"]): row for row in public_packet
    }

    checks = {
        "annotation_rows_308": len(annotations) == 308,
        "annotation_ids_unique": len(annotation_by_id) == len(annotations),
        "private_key_rows_308": len(private_keys) == 308,
        "public_packet_rows_308": len(public_packet) == 308,
        "annotation_ids_match_private_key": (
            set(annotation_by_id) == set(key_by_id)
        ),
        "annotation_ids_match_public_packet": (
            set(annotation_by_id) == set(packet_by_id)
        ),
        "protocol_exact": all(
            row.get("protocol") == PROTOCOL for row in annotations
        ),
        "preferences_valid": all(
            row.get("quality_preference") in ALLOWED_PREFERENCES
            for row in annotations
        ),
        "criteria_valid": all(
            row.get("decisive_criterion") in ALLOWED_CRITERIA
            for row in annotations
        ),
        "required_fields_complete": all(
            str(row.get(field) or "").strip()
            for row in annotations
            for field in (
                "blind_item_id",
                "quality_preference",
                "decisive_criterion",
                "annotator_id",
            )
        ),
        "one_annotator": len(
            {str(row["annotator_id"]) for row in annotations}
        )
        == 1,
        "tie_criterion_consistent": all(
            (
                row["quality_preference"] == "tie"
                and row["decisive_criterion"] == "materially_equivalent"
            )
            or (
                row["quality_preference"] != "tie"
                and row["decisive_criterion"] != "materially_equivalent"
            )
            for row in annotations
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"annotation validation failed: {failed}")

    joined: list[dict[str, Any]] = []
    for annotation in annotations:
        blind_id = str(annotation["blind_item_id"])
        key = key_by_id[blind_id]
        state = runtime_states[str(key["state_id"])]
        joined.append(
            {
                **annotation,
                **{
                    field: key[field]
                    for field in (
                        "component",
                        "contrast_slot_id",
                        "state_id",
                        "split",
                        "a_role",
                        "b_role",
                        "is_reliability_repeat",
                        "repeat_group_id",
                        "control_action",
                        "treatment_action",
                    )
                },
                "user_id": str(state["user_id"]),
                "semantic_verdict": _semantic_verdict(annotation, key),
            }
        )

    by_repeat_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_repeat_group[str(row["repeat_group_id"])].append(row)
    repeated_groups = [
        rows for rows in by_repeat_group.values() if len(rows) == 2
    ]
    invalid_repeat_group_sizes = [
        len(rows)
        for rows in by_repeat_group.values()
        if len(rows) not in {1, 2}
    ]
    repeat_direction_matches = sum(
        rows[0]["semantic_verdict"] == rows[1]["semantic_verdict"]
        for rows in repeated_groups
    )
    repeat_criterion_matches = sum(
        rows[0]["decisive_criterion"] == rows[1]["decisive_criterion"]
        for rows in repeated_groups
    )

    primary = [
        row for row in joined if not bool(row["is_reliability_repeat"])
    ]
    labels: list[dict[str, Any]] = []
    for row in primary:
        verdict = str(row["semantic_verdict"])
        if verdict == "treatment":
            quality_label = "ON_QUALITY_WIN_PENDING_MATERIAL_RISK"
        elif verdict == "control":
            quality_label = "OFF_CONTROL_QUALITY_WIN"
        elif verdict == "tie":
            quality_label = "OFF_NO_MATERIAL_QUALITY_GAIN"
        else:
            quality_label = "UNKNOWN"
        labels.append(
            {
                "protocol": AGGREGATION_PROTOCOL,
                "contrast_slot_id": row["contrast_slot_id"],
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "split": row["split"],
                "component": row["component"],
                "control_action": row["control_action"],
                "treatment_action": row["treatment_action"],
                "quality_verdict": verdict,
                "quality_label_pre_risk": quality_label,
                "decisive_criterion": row["decisive_criterion"],
                "annotator_id": row["annotator_id"],
                "final_effect_label": (
                    "UNKNOWN_UNTIL_COMPONENT_ON_WIN_RISK_REVIEW"
                    if verdict == "treatment"
                    else quality_label
                ),
            }
        )

    primary_position = Counter(
        row["quality_preference"] for row in primary
    )
    component_verdicts = _count_nested(
        labels, "component", "quality_verdict"
    )
    component_quality_labels = _count_nested(
        labels, "component", "quality_label_pre_risk"
    )
    split_verdicts = _count_nested(labels, "split", "quality_verdict")
    unique_notes = {
        str(row.get("quality_notes") or "") for row in annotations
    }
    manual_audit_sample: list[dict[str, Any]] = []
    for component in ("RS", "MP", "MS", "ME"):
        for verdict in ("treatment", "control", "tie"):
            candidates = sorted(
                (
                    row
                    for row in primary
                    if row["component"] == component
                    and row["semantic_verdict"] == verdict
                ),
                key=lambda row: str(row["blind_item_id"]),
            )[:2]
            for row in candidates:
                public = packet_by_id[str(row["blind_item_id"])]
                manual_audit_sample.append(
                    {
                        "blind_item_id": row["blind_item_id"],
                        "component": component,
                        "semantic_verdict": verdict,
                        "quality_preference": row["quality_preference"],
                        "decisive_criterion": row["decisive_criterion"],
                        "quality_notes": row["quality_notes"],
                        "current_session_summary": public[
                            "current_session_summary"
                        ],
                        "recent_dialogue": public["recent_dialogue"],
                        "current_user_text": public["current_user_text"],
                        "response_a": public["response_a"],
                        "response_b": public["response_b"],
                        "a_role": row["a_role"],
                        "b_role": row["b_role"],
                    }
                )
    report = {
        "protocol": AGGREGATION_PROTOCOL,
        "status": (
            "QUALITY_LABELS_VALID_PENDING_COMPONENT_ON_WIN_RISK_REVIEW"
            if (
                len(primary) == 256
                and len(repeated_groups) == 52
                and repeat_direction_matches == 52
            )
            else "BLOCKED_QUALITY_REVIEW_INTEGRITY_FAILURE"
        ),
        "checks": checks,
        "review_grain": {
            "presentations": len(joined),
            "independent_contrasts": len(primary),
            "reliability_repeats": len(repeated_groups),
            "components": dict(
                sorted(Counter(row["component"] for row in primary).items())
            ),
        },
        "reliability": {
            "invalid_repeat_group_sizes": invalid_repeat_group_sizes,
            "semantic_direction_matches": repeat_direction_matches,
            "semantic_direction_total": len(repeated_groups),
            "semantic_direction_agreement": (
                repeat_direction_matches / len(repeated_groups)
            ),
            "decisive_criterion_matches": repeat_criterion_matches,
            "decisive_criterion_total": len(repeated_groups),
            "decisive_criterion_agreement": (
                repeat_criterion_matches / len(repeated_groups)
            ),
            "interpretation": (
                "Reversed-position repeats test within-annotator stability, "
                "not inter-annotator validity."
            ),
        },
        "blind_position_profile": {
            "all_presentations": dict(
                sorted(
                    Counter(
                        row["quality_preference"] for row in joined
                    ).items()
                )
            ),
            "primary_only": dict(sorted(primary_position.items())),
        },
        "unblinded_primary_quality_verdicts": dict(
            sorted(Counter(row["quality_verdict"] for row in labels).items())
        ),
        "quality_verdicts_by_component": component_verdicts,
        "quality_labels_by_component": component_quality_labels,
        "quality_verdicts_by_split": split_verdicts,
        "component_on_quality_wins_pending_risk": sum(
            row["quality_verdict"] == "treatment" for row in labels
        ),
        "decisive_criteria_primary": dict(
            sorted(
                Counter(
                    row["decisive_criterion"] for row in primary
                ).items()
            )
        ),
        "notes_profile": {
            "nonempty": sum(
                bool(str(row.get("quality_notes") or "").strip())
                for row in annotations
            ),
            "unique": len(unique_notes),
            "training_feature": False,
            "interpretation": (
                "Notes are optional audit context and are excluded from "
                "model features and labels; template reuse therefore does "
                "not duplicate independent contrast groups."
            ),
        },
        "validity_boundary": {
            "supported": [
                "complete and schema-valid blind review",
                "perfect within-annotator reversed-position stability",
                "absence of raw A/B position imbalance",
                "usable quality-effect proxy after risk review",
            ],
            "not_supported_by_this_review_alone": [
                "inter-annotator agreement",
                "objective clinical correctness",
                "memory retrieval relevance or temporal consistency",
                "material safety of component-on quality winners",
            ],
        },
        "inputs": {
            "annotations": str(args.annotations),
            "annotations_sha256": sha256_file(args.annotations),
            "private_key": str(
                args.packet_dir / "private_blind_key.jsonl"
            ),
            "private_key_sha256": sha256_file(
                args.packet_dir / "private_blind_key.jsonl"
            ),
            "public_packet": str(
                args.packet_dir / "human_blind_packet.jsonl"
            ),
            "public_packet_sha256": sha256_file(
                args.packet_dir / "human_blind_packet.jsonl"
            ),
            "runtime_states": str(args.runtime_states),
            "runtime_states_sha256": sha256_file(args.runtime_states),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "quality_aggregation.json", report)
    write_jsonl(
        args.out_dir / "validated_quality_annotations.jsonl",
        annotations,
    )
    write_jsonl(
        args.out_dir / "unblinded_quality_presentations.jsonl",
        joined,
    )
    write_jsonl(
        args.out_dir / "quality_effect_labels_pre_risk.jsonl",
        labels,
    )
    write_jsonl(
        args.out_dir / "manual_accuracy_audit_sample.jsonl",
        manual_audit_sample,
    )
    print(
        {
            "protocol": AGGREGATION_PROTOCOL,
            "status": report["status"],
            "independent_contrasts": len(primary),
            "repeat_agreement": (
                f"{repeat_direction_matches}/{len(repeated_groups)}"
            ),
            "component_on_wins_pending_risk": report[
                "component_on_quality_wins_pending_risk"
            ],
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
