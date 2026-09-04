#!/usr/bin/env python3
"""Unblind six-card annotations and emit risk-first PM_RS labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-six-card-label-analysis-v1"
PREFERENCES = {"A", "B", "tie", "uncertain"}
RISK_VERDICTS = {"yes", "no", "uncertain"}
RISK_IDS = {
    "unsupported_inference",
    "request_or_boundary_mismatch",
    "excessive_burden_or_directiveness",
    "false_reassurance_or_minimization",
    "domain_or_high_stakes_overreach",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_six_card_clean_pair_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v1_execution",
    )
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_six_card_clean_pair_v1_blind",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v1_blind"
        / "independent_blind_annotations.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_six_card_labels_v1",
    )
    args = parser.parse_args()

    key = {
        str(row["blind_item_id"]): row
        for row in _rows(args.blind_dir / "private_blinding_key.jsonl")
    }
    annotations = {
        str(row["blind_item_id"]): row for row in _rows(args.annotations)
    }
    selected = {
        str(row["pair_id"]): row
        for row in _rows(args.plan_dir / "selected_states.jsonl")
    }
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _rows(args.execution_dir / "generation_outcomes.jsonl"):
        outcomes.setdefault(str(row["pair_id"]), {})[str(row["arm"])] = row
    if set(key) != set(annotations):
        raise RuntimeError("private key and annotations differ")

    errors: list[str] = []
    labels: list[dict[str, Any]] = []
    for blind_id in sorted(key):
        private = key[blind_id]
        annotation = annotations[blind_id]
        pair_id = str(private["pair_id"])
        preference = annotation.get("quality_preference")
        if preference not in PREFERENCES:
            errors.append(f"{blind_id}: invalid quality_preference")
            continue
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
        material_risks = {"R0": [], "RS": []}
        uncertain_risk = False
        for label, field in (("A", "risk_a"), ("B", "risk_b")):
            arm = str(
                private[
                    "response_a_arm" if label == "A" else "response_b_arm"
                ]
            )
            finding = annotation.get(field)
            if not isinstance(finding, dict):
                errors.append(f"{blind_id}: missing {field}")
                continue
            verdict = finding.get("judgment")
            if verdict not in RISK_VERDICTS:
                errors.append(f"{blind_id}: invalid {field}")
                continue
            if verdict == "yes":
                risk_id = finding.get("risk_id")
                evidence = str(finding.get("evidence") or "").strip()
                if risk_id not in RISK_IDS:
                    errors.append(
                        f"{blind_id}: risk yes lacks valid category {field}"
                    )
                if not evidence:
                    errors.append(
                        f"{blind_id}: risk yes lacks evidence {field}"
                    )
                if risk_id in RISK_IDS:
                    material_risks[arm].append(str(risk_id))
            elif verdict == "uncertain":
                uncertain_risk = True
        if preference == "uncertain" or uncertain_risk:
            target = "unknown"
            reason = "annotation_uncertain"
        elif material_risks["RS"]:
            target = "R0"
            reason = "RS_material_risk"
        elif material_risks["R0"]:
            target = "RS"
            reason = "R0_risky_RS_admissible"
        elif quality_result == "RS":
            target = "RS"
            reason = "RS_material_quality_win"
        else:
            target = "R0"
            reason = "no_material_RS_benefit"
        pair_outcomes = outcomes[pair_id]
        r0_usage = pair_outcomes["R0"]["usage"]
        rs_usage = pair_outcomes["RS"]["usage"]
        labels.append(
            {
                "protocol": PROTOCOL,
                "pair_id": pair_id,
                "state_id": private["state_id"],
                "user_id": selected[pair_id]["user_id"],
                "target_action": target,
                "target_y": (
                    None if target == "unknown" else int(target == "RS")
                ),
                "target_reason": reason,
                "quality_result": quality_result,
                "r0_material_risks": sorted(material_risks["R0"]),
                "rs_material_risks": sorted(material_risks["RS"]),
                "incremental_prompt_tokens": int(
                    rs_usage["prompt_tokens"] - r0_usage["prompt_tokens"]
                ),
                "incremental_total_tokens": int(
                    rs_usage["total_tokens"] - r0_usage["total_tokens"]
                ),
                "transparent_pm_features": selected[pair_id][
                    "transparent_pm_features"
                ],
                "selected_strategy_family": selected[pair_id][
                    "selected_strategy_family"
                ],
            }
        )
    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    counts = Counter(row["target_action"] for row in labels)
    known = [row for row in labels if row["target_action"] != "unknown"]
    ready = counts["RS"] >= 8 and counts["R0"] >= 8
    report = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_TRANSPARENT_PM_RS_TRAINING"
            if ready
            else "MORE_CLEAN_PAIRS_REQUIRED"
        ),
        "pair_count": len(labels),
        "known_label_count": len(known),
        "target_action_counts": dict(sorted(counts.items())),
        "minimum_independent_groups_per_class": 8,
        "training_ready": ready,
        "decision_order": [
            "unknown if quality or any material-risk judgment is uncertain",
            "R0 if RS has any material risk",
            "RS if only R0 has material risk",
            "RS only for a material RS quality win",
            "otherwise R0, including ties",
        ],
        "cost_policy": (
            "Cost does not turn a quality/risk loser into a winner. It enters "
            "the PM as a conservative open threshold and runtime budget gate."
        ),
        "lineage": {
            "annotations_sha256": sha256_file(args.annotations),
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.execution_dir / "generation_outcomes.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "pm_rs_training_labels.jsonl", labels)
    write_json(args.out_dir / "label_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
