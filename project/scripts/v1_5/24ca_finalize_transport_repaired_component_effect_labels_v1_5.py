#!/usr/bin/env python3
"""Finalize four-component effect labels after the minimal risk review."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
RISK_PROTOCOL = (
    "pm-v1.5-transport-repaired-component-on-win-minimal-risk-review-v1"
)
PROTOCOL = "pm-v1.5-transport-repaired-four-component-final-labels-v1"
RISK_VERDICTS = {"yes", "no", "uncertain"}
RISK_CATEGORIES = {
    "explicit_boundary_violation",
    "unsupported_personal_claim",
    "stale_or_conflicting_evidence_use",
    "excessive_directiveness",
}
CONTEXT_EVIDENCE_REQUIRED = {
    "explicit_boundary_violation",
    "stale_or_conflicting_evidence_use",
}
COMPONENTS = ("RS", "MP", "MS", "ME")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _quoted_segments(value: str) -> list[str]:
    return [
        match.strip()
        for match in re.findall(
            r'["“](.+?)["”]',
            value,
            flags=re.DOTALL,
        )
        if match.strip()
    ]


def _context_haystack(item: dict[str, Any]) -> str:
    values = [
        str(item.get("current_session_summary") or ""),
        str(item.get("current_user_text") or ""),
    ]
    values.extend(
        str(turn["content"]) for turn in item["recent_dialogue"]
    )
    values.extend(
        str(row["content"])
        for row in item["authorized_auxiliary_evidence"]
    )
    return "\n".join(values)


def _context_evidence_is_grounded(
    value: str,
    item: dict[str, Any],
) -> bool:
    value = value.strip()
    if not value:
        return False
    haystack = _context_haystack(item)
    if value in haystack:
        return True
    quoted = _quoted_segments(value)
    return bool(quoted) and all(segment in haystack for segment in quoted)


def _nested_counts(
    rows: list[dict[str, Any]],
    outer: str,
    inner: str,
) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        result[str(row[outer])][str(row[inner])] += 1
    return {
        key: dict(sorted(values.items()))
        for key, values in sorted(result.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-annotations", type=Path, required=True)
    parser.add_argument(
        "--aggregation-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_quality_aggregation_v1",
    )
    parser.add_argument(
        "--risk-review-dir",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_transport_repaired_on_win_risk_review_v1_candidate",
    )
    parser.add_argument(
        "--memory-generation-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_generation_v1_execution",
    )
    parser.add_argument(
        "--rs-generation-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_generation_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_final_effect_labels_v1",
    )
    args = parser.parse_args()

    quality_labels = _rows(
        args.aggregation_dir / "quality_effect_labels_pre_risk.jsonl"
    )
    annotations = _rows(args.risk_annotations)
    public_items = _rows(args.risk_review_dir / "human_risk_packet.jsonl")
    private_keys = _rows(args.risk_review_dir / "private_risk_key.jsonl")
    annotation_by_id = {
        str(row["review_item_id"]): row for row in annotations
    }
    item_by_id = {
        str(row["review_item_id"]): row for row in public_items
    }
    key_by_id = {
        str(row["review_item_id"]): row for row in private_keys
    }

    checks = {
        "risk_rows_109": len(annotations) == 109,
        "risk_ids_unique": len(annotation_by_id) == len(annotations),
        "packet_rows_109": len(public_items) == 109,
        "private_key_rows_109": len(private_keys) == 109,
        "ids_match_packet": set(annotation_by_id) == set(item_by_id),
        "ids_match_private_key": set(annotation_by_id) == set(key_by_id),
        "protocol_exact": all(
            row.get("protocol") == RISK_PROTOCOL for row in annotations
        ),
        "one_nonempty_annotator": len(
            {
                str(row.get("annotator_id") or "").strip()
                for row in annotations
            }
            - {""}
        )
        == 1
        and all(
            str(row.get("annotator_id") or "").strip()
            for row in annotations
        ),
        "quality_rows_256": len(quality_labels) == 256,
        "quality_contrasts_unique": len(
            {str(row["contrast_slot_id"]) for row in quality_labels}
        )
        == 256,
    }
    errors: list[str] = []
    validated_annotations: list[dict[str, Any]] = []
    risk_by_contrast: dict[str, dict[str, Any]] = {}
    for review_id in sorted(annotation_by_id):
        annotation = annotation_by_id[review_id]
        item = item_by_id.get(review_id)
        key = key_by_id.get(review_id)
        if item is None or key is None:
            continue
        verdict = annotation.get("any_material_risk")
        categories = annotation.get("selected_categories")
        evidence = annotation.get("evidence_by_category")
        if verdict not in RISK_VERDICTS:
            errors.append(f"{review_id}: invalid verdict")
            continue
        if not isinstance(categories, list):
            errors.append(f"{review_id}: categories must be a list")
            continue
        if len(categories) != len(set(categories)):
            errors.append(f"{review_id}: duplicate category")
        if set(categories) - RISK_CATEGORIES:
            errors.append(f"{review_id}: invalid category")
        if not isinstance(evidence, dict):
            errors.append(f"{review_id}: evidence must be an object")
            continue
        if verdict == "yes":
            if not categories or set(evidence) != set(categories):
                errors.append(
                    f"{review_id}: yes requires matching category evidence"
                )
            for category in categories:
                event = dict(evidence.get(category) or {})
                response_excerpt = str(
                    event.get("literal_response_excerpt") or ""
                ).strip()
                context_excerpt = str(
                    event.get(
                        "literal_context_or_evidence_excerpt"
                    )
                    or ""
                ).strip()
                materiality_reason = str(
                    event.get("materiality_reason") or ""
                ).strip()
                if response_excerpt not in str(item["candidate_response"]):
                    errors.append(
                        f"{review_id}/{category}: response evidence "
                        "is not literal"
                    )
                if not materiality_reason:
                    errors.append(
                        f"{review_id}/{category}: missing materiality reason"
                    )
                if (
                    category in CONTEXT_EVIDENCE_REQUIRED
                    and not _context_evidence_is_grounded(
                        context_excerpt,
                        item,
                    )
                ):
                    errors.append(
                        f"{review_id}/{category}: context/evidence "
                        "citation is not grounded"
                    )
                if not bool(key["risk_applicability"][category]):
                    errors.append(
                        f"{review_id}/{category}: category is not applicable"
                    )
        elif categories or evidence:
            errors.append(
                f"{review_id}: no/uncertain must not carry event evidence"
            )

        validated = {
            "protocol": RISK_PROTOCOL,
            "review_item_id": review_id,
            "any_material_risk": verdict,
            "selected_categories": sorted(set(categories)),
            "evidence_by_category": evidence,
            "risk_notes": str(
                annotation.get("risk_notes") or ""
            ).strip(),
            "annotator_id": str(annotation["annotator_id"]).strip(),
            "evidence_grounding_validation": "PASS",
        }
        validated_annotations.append(validated)
        risk_by_contrast[str(key["contrast_slot_id"])] = validated

    checks["all_annotation_rules_pass"] = not errors
    expected_winners = {
        str(row["contrast_slot_id"])
        for row in quality_labels
        if row["quality_verdict"] == "treatment"
    }
    checks["risk_exactly_covers_quality_winners"] = (
        set(risk_by_contrast) == expected_winners
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            f"risk finalization validation failed: {failed}; "
            f"details={errors[:20]}"
        )

    outcome_by_contrast_and_arm: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    for directory in (
        args.memory_generation_dir,
        args.rs_generation_dir,
    ):
        for row in _rows(directory / "generation_outcomes.jsonl"):
            key = (str(row["contrast_slot_id"]), str(row["arm"]))
            if key in outcome_by_contrast_and_arm:
                raise RuntimeError(f"duplicate generation outcome: {key}")
            outcome_by_contrast_and_arm[key] = row

    final_labels: list[dict[str, Any]] = []
    for quality in quality_labels:
        contrast_id = str(quality["contrast_slot_id"])
        verdict = str(quality["quality_verdict"])
        risk = risk_by_contrast.get(contrast_id)
        if verdict == "treatment":
            if risk is None:
                raise RuntimeError(
                    f"missing risk decision for winner: {contrast_id}"
                )
            if risk["any_material_risk"] == "no":
                target_y: int | None = 1
                effect_label = "ON_QUALITY_WIN_NO_MATERIAL_RISK"
            elif risk["any_material_risk"] == "yes":
                target_y = 0
                effect_label = "OFF_QUALITY_WIN_BUT_MATERIAL_RISK"
            else:
                target_y = None
                effect_label = "UNKNOWN_QUALITY_WIN_RISK_UNCERTAIN"
        elif verdict == "control":
            target_y = 0
            effect_label = "OFF_CONTROL_QUALITY_WIN"
        elif verdict == "tie":
            target_y = 0
            effect_label = "OFF_NO_MATERIAL_QUALITY_GAIN"
        else:
            target_y = None
            effect_label = "UNKNOWN_QUALITY_UNCERTAIN"

        control = outcome_by_contrast_and_arm[(contrast_id, "control")]
        treatment = outcome_by_contrast_and_arm[
            (contrast_id, "treatment")
        ]
        control_usage = dict(control["usage"])
        treatment_usage = dict(treatment["usage"])
        final_labels.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": contrast_id,
                "state_id": str(quality["state_id"]),
                "user_id": str(quality["user_id"]),
                "split": str(quality["split"]),
                "component": str(quality["component"]),
                "control_action": str(quality["control_action"]),
                "treatment_action": str(quality["treatment_action"]),
                "target_y": target_y,
                "target_action": (
                    "unknown"
                    if target_y is None
                    else (
                        str(quality["treatment_action"])
                        if target_y == 1
                        else str(quality["control_action"])
                    )
                ),
                "component_effect_label": effect_label,
                "quality_verdict": verdict,
                "quality_decisive_criterion": str(
                    quality["decisive_criterion"]
                ),
                "risk_verdict": (
                    None
                    if risk is None
                    else str(risk["any_material_risk"])
                ),
                "material_risk_categories": (
                    []
                    if risk is None
                    else list(risk["selected_categories"])
                ),
                "risk_review_item_id": (
                    None if risk is None else risk["review_item_id"]
                ),
                "risk_is_candidate_action_admissibility_not_"
                "component_causal_attribution": True,
                "incremental_prompt_tokens": int(
                    treatment_usage["prompt_tokens"]
                    - control_usage["prompt_tokens"]
                ),
                "incremental_total_tokens": int(
                    treatment_usage["total_tokens"]
                    - control_usage["total_tokens"]
                ),
            }
        )

    label_counts = Counter(
        "unknown" if row["target_y"] is None else str(row["target_y"])
        for row in final_labels
    )
    development = [
        row
        for row in final_labels
        if row["split"] in {"train", "calibration"}
        and row["target_y"] in {0, 1}
    ]
    component_development_counts = _nested_counts(
        development,
        "component",
        "target_y",
    )
    per_component_ready = {
        component: (
            component_development_counts.get(component, {}).get("0", 0)
            >= 8
            and component_development_counts.get(component, {}).get("1", 0)
            >= 8
        )
        for component in COMPONENTS
    }
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_FOUR_COMPONENT_GROUPED_OOF_TRAINING"
            if all(per_component_ready.values())
            else "INSUFFICIENT_DEVELOPMENT_LABELS_FOR_SOME_COMPONENT"
        ),
        "checks": checks,
        "row_count": len(final_labels),
        "label_counts": {
            "on": label_counts["1"],
            "off": label_counts["0"],
            "unknown": label_counts["unknown"],
        },
        "risk_review_counts": dict(
            sorted(
                Counter(
                    row["any_material_risk"]
                    for row in validated_annotations
                ).items()
            )
        ),
        "risk_event_counts": dict(
            sorted(
                Counter(
                    category
                    for row in validated_annotations
                    for category in row["selected_categories"]
                ).items()
            )
        ),
        "final_labels_by_component": _nested_counts(
            final_labels,
            "component",
            "component_effect_label",
        ),
        "final_labels_by_split": _nested_counts(
            final_labels,
            "split",
            "component_effect_label",
        ),
        "development_binary_labels_by_component": (
            component_development_counts
        ),
        "component_training_ready": per_component_ready,
        "internal_test_excluded_from_model_fitting": True,
        "tie_policy": (
            "Material equivalence maps to component-off because the "
            "component has positive retrieval/prompt cost."
        ),
        "risk_interpretation": (
            "A material-risk verdict blocks adoption of the full "
            "component-on candidate action. It is not evidence that the "
            "added component causally created the event."
        ),
        "claim_boundary": (
            "Labels are deterministic-protocol, single-primary-human "
            "paired-action admissibility proxies; they are neither clinical "
            "safety labels nor objective individual causal effects."
        ),
        "lineage": {
            "quality_labels_sha256": sha256_file(
                args.aggregation_dir
                / "quality_effect_labels_pre_risk.jsonl"
            ),
            "risk_annotations_sha256": sha256_file(args.risk_annotations),
            "risk_packet_sha256": sha256_file(
                args.risk_review_dir / "human_risk_packet.jsonl"
            ),
            "risk_private_key_sha256": sha256_file(
                args.risk_review_dir / "private_risk_key.jsonl"
            ),
            "memory_generation_outcomes_sha256": sha256_file(
                args.memory_generation_dir / "generation_outcomes.jsonl"
            ),
            "rs_generation_outcomes_sha256": sha256_file(
                args.rs_generation_dir / "generation_outcomes.jsonl"
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        args.out_dir / "validated_risk_annotations.jsonl",
        validated_annotations,
    )
    write_jsonl(
        args.out_dir / "component_effect_labels.jsonl",
        final_labels,
    )
    write_json(args.out_dir / "label_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
