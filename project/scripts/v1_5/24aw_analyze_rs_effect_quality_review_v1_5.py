#!/usr/bin/env python3
"""Unblind the bounded RS effect quality review and select risk candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-effect-human-quality-analysis-v1"
PREFERENCES = {"A", "B", "tie", "uncertain"}
CRITERIA = {
    "request_and_conversation_fit",
    "emotional_attunement",
    "visible_context_fidelity",
    "immediate_helpfulness",
    "clarity_and_naturalness",
    "materially_equivalent",
    "uncertain",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_effect_human_quality_blind_v1_candidate",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_effect_human_quality_blind_v1_candidate"
        / "independent_quality_annotations.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_effect_human_quality_analysis_v1",
    )
    args = parser.parse_args()

    key = {
        str(row["blind_item_id"]): row
        for row in _rows(args.blind_dir / "private_blinding_key.jsonl")
    }
    annotations_list = _rows(args.annotations)
    annotations = {
        str(row["blind_item_id"]): row for row in annotations_list
    }
    if len(annotations) != len(annotations_list):
        raise RuntimeError("duplicate blind_item_id in annotations")
    if set(key) != set(annotations):
        missing = sorted(set(key) - set(annotations))
        extra = sorted(set(annotations) - set(key))
        raise RuntimeError(
            f"annotation/key mismatch; missing={missing[:5]} extra={extra[:5]}"
        )

    errors: list[str] = []
    decisions: list[dict[str, Any]] = []
    for blind_id in sorted(key):
        private = key[blind_id]
        annotation = annotations[blind_id]
        if annotation.get("protocol") != (
            "pm-v1.5-rs-effect-human-quality-blind-v1"
        ):
            errors.append(f"{blind_id}: protocol mismatch")
        preference = annotation.get("quality_preference")
        criterion = annotation.get("decisive_criterion")
        if preference not in PREFERENCES:
            errors.append(f"{blind_id}: invalid quality_preference")
            continue
        if criterion not in CRITERIA:
            errors.append(f"{blind_id}: invalid decisive_criterion")
            continue
        if preference == "tie" and criterion != "materially_equivalent":
            errors.append(
                f"{blind_id}: tie requires materially_equivalent criterion"
            )
        if preference == "uncertain" and criterion != "uncertain":
            errors.append(f"{blind_id}: uncertain requires uncertain criterion")
        if preference in {"A", "B"} and criterion in {
            "materially_equivalent",
            "uncertain",
        }:
            errors.append(
                f"{blind_id}: A/B requires a substantive decisive criterion"
            )
        quality_result = (
            str(preference)
            if preference in {"tie", "uncertain"}
            else str(
                private[
                    "response_a_arm"
                    if preference == "A"
                    else "response_b_arm"
                ]
            )
        )
        decisions.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "pair_id": private["pair_id"],
                "state_id": private["state_id"],
                "user_id": private["user_id"],
                "quality_result": quality_result,
                "decisive_criterion": criterion,
                "quality_notes": str(
                    annotation.get("quality_notes") or ""
                ).strip(),
                "annotator_id": str(
                    annotation.get("annotator_id") or ""
                ).strip(),
                "selected_card_id": private["selected_card_id"],
                "selected_core_submove_id": private[
                    "selected_core_submove_id"
                ],
                "selected_strategy_family": private[
                    "selected_strategy_family"
                ],
                "risk_review_required": quality_result == "RS",
                "provisional_training_target": (
                    None
                    if quality_result == "uncertain"
                    else int(quality_result == "RS")
                ),
                "provisional_only_until_rs_risk_review": (
                    quality_result == "RS"
                ),
            }
        )
    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    result_counts = Counter(row["quality_result"] for row in decisions)
    criterion_counts = Counter(
        row["decisive_criterion"] for row in decisions
    )
    family_counts: dict[str, Counter[str]] = {}
    for row in decisions:
        family_counts.setdefault(
            str(row["selected_strategy_family"]), Counter()
        )[str(row["quality_result"])] += 1
    risk_candidates = [
        row for row in decisions if row["risk_review_required"]
    ]
    known_negative_count = result_counts["R0"] + result_counts["tie"]
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_RS_WIN_ATOMIC_RISK_REVIEW"
            if len(risk_candidates) >= 8 and known_negative_count >= 8
            else "RS_EFFECT_POSITIVES_NOT_YET_TRAINING_QUALIFIED"
        ),
        "pair_count": len(decisions),
        "quality_result_counts": dict(sorted(result_counts.items())),
        "decisive_criterion_counts": dict(sorted(criterion_counts.items())),
        "quality_results_by_strategy_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(family_counts.items())
        },
        "rs_win_risk_review_candidate_count": len(risk_candidates),
        "known_negative_count_before_risk": known_negative_count,
        "minimum_independent_groups_per_class": 8,
        "training_ready": False,
        "why_not_yet_training_ready": (
            "Every RS quality win must pass the atomic risk review first; "
            "R0 wins and ties are already RS-off quality outcomes."
        ),
        "decision_rule_after_risk_review": [
            "RS quality win plus no material RS risk becomes y=1",
            "RS quality win with material RS risk becomes y=0",
            "R0 win or tie becomes y=0",
            "uncertain remains unknown and is excluded from training",
            "cost sets the opening threshold/budget gate and never rescues a "
            "quality or risk loser",
        ],
        "lineage": {
            "annotations_sha256": sha256_file(args.annotations),
            "private_blinding_key_sha256": sha256_file(
                args.blind_dir / "private_blinding_key.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "quality_decisions.jsonl", decisions)
    write_jsonl(
        args.out_dir / "rs_win_risk_review_candidates.jsonl",
        risk_candidates,
    )
    write_json(args.out_dir / "quality_analysis_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
