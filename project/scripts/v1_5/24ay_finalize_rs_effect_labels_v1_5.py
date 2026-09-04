#!/usr/bin/env python3
"""Finalize RS effect labels after the minimal human risk review."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-effect-final-labels-v1"
RISK_VERDICTS = {"yes", "no", "uncertain"}
RISK_CATEGORIES = {
    "explicit_boundary_or_permission_violation",
    "unsupported_or_overstated_claim",
    "excessive_directiveness_or_burden",
    "domain_or_high_stakes_overreach",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality-analysis-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_effect_human_quality_analysis_v1",
    )
    parser.add_argument(
        "--risk-review-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_win_minimal_risk_review_v1_candidate",
    )
    parser.add_argument(
        "--risk-annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_win_minimal_risk_review_v1_candidate"
        / "independent_rs_win_risk_annotations.jsonl",
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_effect_final_labels_v1",
    )
    args = parser.parse_args()

    quality = {
        str(row["pair_id"]): row
        for row in _rows(args.quality_analysis_dir / "quality_decisions.jsonl")
    }
    key = {
        str(row["review_item_id"]): row
        for row in _rows(args.risk_review_dir / "private_risk_key.jsonl")
    }
    annotation_rows = _rows(args.risk_annotations)
    annotations = {
        str(row["review_item_id"]): row for row in annotation_rows
    }
    if len(annotations) != len(annotation_rows):
        raise RuntimeError("duplicate review_item_id in risk annotations")
    if set(key) != set(annotations):
        raise RuntimeError("risk annotation/key mismatch")
    selected = {
        str(row["pair_id"]): row
        for row in _rows(args.plan_dir / "selected_states.jsonl")
    }
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _rows(args.execution_dir / "generation_outcomes.jsonl"):
        outcomes.setdefault(str(row["pair_id"]), {})[str(row["arm"])] = row

    errors: list[str] = []
    risk_by_pair: dict[str, dict[str, Any]] = {}
    for review_id in sorted(key):
        private = key[review_id]
        annotation = annotations[review_id]
        verdict = annotation.get("any_material_risk")
        categories = annotation.get("selected_categories")
        evidence = str(
            annotation.get("response_evidence_excerpt") or ""
        ).strip()
        if verdict not in RISK_VERDICTS:
            errors.append(f"{review_id}: invalid risk verdict")
            continue
        if not isinstance(categories, list) or any(
            value not in RISK_CATEGORIES for value in categories
        ):
            errors.append(f"{review_id}: invalid categories")
            continue
        if verdict == "yes" and (not categories or not evidence):
            errors.append(f"{review_id}: yes requires category and evidence")
        if verdict != "yes" and (categories or evidence):
            errors.append(
                f"{review_id}: no/uncertain must not carry event evidence"
            )
        pair_id = str(private["pair_id"])
        risk_by_pair[pair_id] = {
            "risk_verdict": verdict,
            "material_risk_categories": sorted(set(categories)),
            "response_evidence_excerpt": evidence,
            "risk_notes": str(annotation.get("risk_notes") or "").strip(),
            "risk_annotator_id": str(
                annotation.get("annotator_id") or ""
            ).strip(),
        }
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
    expected_risk_pairs = {
        pair_id
        for pair_id, row in quality.items()
        if row["quality_result"] == "RS"
    }
    if set(risk_by_pair) != expected_risk_pairs:
        raise RuntimeError("risk review does not exactly cover RS wins")

    labels: list[dict[str, Any]] = []
    for pair_id in sorted(quality):
        q = quality[pair_id]
        quality_result = str(q["quality_result"])
        risk = risk_by_pair.get(pair_id)
        if quality_result == "uncertain":
            target_y = None
            reason = "quality_uncertain"
        elif quality_result in {"R0", "tie"}:
            target_y = 0
            reason = (
                "R0_material_quality_win"
                if quality_result == "R0"
                else "material_quality_equivalence_choose_lower_cost_R0"
            )
        elif risk and risk["risk_verdict"] == "uncertain":
            target_y = None
            reason = "RS_quality_win_but_risk_uncertain"
        elif risk and risk["risk_verdict"] == "yes":
            target_y = 0
            reason = "RS_quality_win_but_material_risk"
        else:
            target_y = 1
            reason = "RS_material_quality_win_and_no_material_risk"
        r0_usage = outcomes[pair_id]["R0"]["usage"]
        rs_usage = outcomes[pair_id]["RS"]["usage"]
        labels.append(
            {
                "protocol": PROTOCOL,
                "pair_id": pair_id,
                "state_id": q["state_id"],
                "user_id": q["user_id"],
                "target_y": target_y,
                "target_action": (
                    "unknown"
                    if target_y is None
                    else ("M0+RS" if target_y else "M0+R0")
                ),
                "target_reason": reason,
                "quality_result": quality_result,
                "risk_verdict": None if risk is None else risk["risk_verdict"],
                "material_risk_categories": (
                    [] if risk is None else risk["material_risk_categories"]
                ),
                "selected_strategy_family_diagnostic_only": q[
                    "selected_strategy_family"
                ],
                "observable_pre_generation_features": selected[pair_id][
                    "observable_flags"
                ],
                "incremental_prompt_tokens": int(
                    rs_usage["prompt_tokens"] - r0_usage["prompt_tokens"]
                ),
                "incremental_total_tokens": int(
                    rs_usage["total_tokens"] - r0_usage["total_tokens"]
                ),
            }
        )

    counts = Counter(
        "unknown" if row["target_y"] is None else str(row["target_y"])
        for row in labels
    )
    positive = counts["1"]
    negative = counts["0"]
    ready = positive >= 8 and negative >= 8
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_FIRST_GROUPED_OOF_PM_RS_FIT"
            if ready
            else "NOT_ENOUGH_INDEPENDENT_EFFECT_LABELS_FOR_PM_RS"
        ),
        "pair_count": len(labels),
        "label_counts": {
            "RS_on_positive": positive,
            "RS_off_negative": negative,
            "unknown": counts["unknown"],
        },
        "minimum_independent_groups_per_class": 8,
        "training_ready": ready,
        "tie_policy": "material equivalence maps to R0 because RS costs more",
        "selected_strategy_family_is_pm_feature": False,
        "claim_boundary": (
            "Labels are protocol-specific human-adjudicated paired-effect "
            "proxies, not objective individual causal effects."
        ),
        "lineage": {
            "quality_decisions_sha256": sha256_file(
                args.quality_analysis_dir / "quality_decisions.jsonl"
            ),
            "risk_annotations_sha256": sha256_file(args.risk_annotations),
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.execution_dir / "generation_outcomes.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "pm_rs_effect_labels.jsonl", labels)
    write_json(args.out_dir / "label_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
