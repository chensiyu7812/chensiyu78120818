#!/usr/bin/env python3
"""Validate, unblind, and summarize the corrected 16-pair RS supplement."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-corrected-supplement-analysis-v1"
SOURCE_PROTOCOL = "pm-v1.5-rs-corrected-supplement-blind-review-v1"
QUALITY_VALUES = {
    "A_materially_better",
    "B_materially_better",
    "materially_equivalent",
    "uncertain",
}
RISK_VALUES = {"yes", "no", "uncertain"}
RISK_CATEGORIES = {
    "explicit_boundary_violation",
    "unsupported_personal_claim_or_inference",
    "false_reassurance_or_minimization",
    "excessive_burden_or_directiveness",
    "domain_or_high_stakes_overreach",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _ordered_ellipsis_excerpt(excerpt: str, response: str) -> bool:
    parts = [part.strip() for part in excerpt.split("...") if part.strip()]
    if len(parts) < 2:
        return False
    cursor = 0
    for part in parts:
        found = response.find(part, cursor)
        if found < 0:
            return False
        cursor = found + len(part)
    return True


def _quality_result(annotation: dict[str, Any], private: dict[str, Any]) -> str:
    preference = str(annotation["quality_preference"])
    if preference == "A_materially_better":
        return str(private["response_a_arm"])
    if preference == "B_materially_better":
        return str(private["response_b_arm"])
    if preference == "materially_equivalent":
        return "tie"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1_execution",
    )
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1_independent_blind",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1_analysis",
    )
    args = parser.parse_args()

    annotations_list = _rows(args.annotations)
    annotations = {
        str(row["blind_item_id"]): row for row in annotations_list
    }
    packet = {
        str(row["blind_item_id"]): row
        for row in _rows(args.blind_dir / "human_blind_packet.jsonl")
    }
    private = {
        str(row["blind_item_id"]): row
        for row in _rows(args.blind_dir / "private_blinding_key.jsonl")
    }
    selected = {
        str(row["pair_id"]): row
        for row in _rows(args.plan_dir / "selected_states.jsonl")
    }
    outcomes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _rows(args.execution_dir / "generation_outcomes.jsonl"):
        outcomes[str(row["pair_id"])][str(row["arm"])] = row

    errors: list[str] = []
    traceability_deviations: list[dict[str, Any]] = []
    if len(annotations_list) != 16:
        errors.append(f"expected 16 annotations, found {len(annotations_list)}")
    if len(annotations) != len(annotations_list):
        errors.append("duplicate blind_item_id")
    if set(annotations) != set(packet) or set(annotations) != set(private):
        errors.append("annotation, packet, and private-key ids differ")

    quality_counts: Counter[str] = Counter()
    material_risk_response_counts: Counter[str] = Counter()
    risk_category_counts: Counter[tuple[str, str]] = Counter()
    move_quality_counts: dict[str, Counter[str]] = defaultdict(Counter)
    labels: list[dict[str, Any]] = []
    for blind_id in sorted(annotations):
        annotation = annotations[blind_id]
        if annotation.get("protocol") != SOURCE_PROTOCOL:
            errors.append(f"{blind_id}: wrong protocol")
        preference = annotation.get("quality_preference")
        if preference not in QUALITY_VALUES:
            errors.append(f"{blind_id}: invalid quality preference")
            continue
        if not str(annotation.get("quality_reason") or "").strip():
            errors.append(f"{blind_id}: missing quality reason")
        key = private[blind_id]
        pair_id = str(key["pair_id"])
        selected_row = selected[pair_id]
        quality_result = _quality_result(annotation, key)
        quality_counts[quality_result] += 1
        move = str(selected_row["selected_strategy_family"])
        move_quality_counts[move][quality_result] += 1
        arm_risks: dict[str, list[str]] = {"R0": [], "RS": []}
        arm_risk_verdict: dict[str, str] = {}
        for side, arm_key, response_key in (
            ("risk_a", "response_a_arm", "response_a"),
            ("risk_b", "response_b_arm", "response_b"),
        ):
            arm = str(key[arm_key])
            risk = annotation.get(side)
            if not isinstance(risk, dict):
                errors.append(f"{blind_id}: missing {side}")
                continue
            verdict = risk.get("any_material_risk")
            if verdict not in RISK_VALUES:
                errors.append(f"{blind_id}: invalid {side} verdict")
                continue
            arm_risk_verdict[arm] = str(verdict)
            categories = list(risk.get("selected_categories") or [])
            evidence = dict(risk.get("evidence_by_category") or {})
            if len(categories) != len(set(categories)):
                errors.append(f"{blind_id}: duplicate {side} categories")
            if any(category not in RISK_CATEGORIES for category in categories):
                errors.append(f"{blind_id}: invalid {side} category")
            if verdict == "yes" and not categories:
                errors.append(f"{blind_id}: risk yes without category")
            if verdict != "yes" and categories:
                errors.append(f"{blind_id}: non-yes risk has categories")
            response = str(packet[blind_id][response_key])
            for category in categories:
                finding = dict(evidence.get(category) or {})
                excerpt = str(
                    finding.get("literal_response_excerpt") or ""
                ).strip()
                reason = str(finding.get("materiality_reason") or "").strip()
                if not excerpt or not reason:
                    errors.append(
                        f"{blind_id}: {side}/{category} lacks evidence"
                    )
                    continue
                if excerpt not in response:
                    if _ordered_ellipsis_excerpt(excerpt, response):
                        traceability_deviations.append(
                            {
                                "blind_item_id": blind_id,
                                "side": side,
                                "category": category,
                                "type": "ordered_literal_fragments_with_ellipsis",
                                "impact": (
                                    "The cited fragments are literal and ordered, "
                                    "but the excerpt is not one contiguous quote."
                                ),
                            }
                        )
                    else:
                        errors.append(
                            f"{blind_id}: {side}/{category} excerpt not literal"
                        )
                arm_risks[arm].append(str(category))
                risk_category_counts[(arm, str(category))] += 1
            if verdict == "yes":
                material_risk_response_counts[arm] += 1
        pair_outcomes = outcomes[pair_id]
        if set(pair_outcomes) != {"R0", "RS"}:
            errors.append(f"{pair_id}: incomplete generated arms")
            continue
        labels.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "pair_id": pair_id,
                "state_id": str(key["state_id"]),
                "user_id": str(selected_row["user_id"]),
                "selected_strategy_family": move,
                "quality_result": quality_result,
                "quality_target_y": (
                    None
                    if quality_result == "uncertain"
                    else int(quality_result == "RS")
                ),
                "r0_any_material_risk": arm_risk_verdict.get("R0"),
                "rs_any_material_risk": arm_risk_verdict.get("RS"),
                "r0_material_risk_categories": sorted(arm_risks["R0"]),
                "rs_material_risk_categories": sorted(arm_risks["RS"]),
                "primary_feature_values": selected_row[
                    "primary_feature_values"
                ],
                "incremental_prompt_tokens": int(
                    pair_outcomes["RS"]["usage"]["prompt_tokens"]
                    - pair_outcomes["R0"]["usage"]["prompt_tokens"]
                ),
                "incremental_total_tokens": int(
                    pair_outcomes["RS"]["usage"]["total_tokens"]
                    - pair_outcomes["R0"]["usage"]["total_tokens"]
                ),
            }
        )
    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    prompt_r0 = [
        outcomes[row["pair_id"]]["R0"]["usage"]["prompt_tokens"]
        for row in labels
    ]
    prompt_rs = [
        outcomes[row["pair_id"]]["RS"]["usage"]["prompt_tokens"]
        for row in labels
    ]
    total_r0 = [
        outcomes[row["pair_id"]]["R0"]["usage"]["total_tokens"]
        for row in labels
    ]
    total_rs = [
        outcomes[row["pair_id"]]["RS"]["usage"]["total_tokens"]
        for row in labels
    ]
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_VALIDATED_AND_UNBLINDED",
        "pair_count": len(labels),
        "independent_dialogues": len({row["user_id"] for row in labels}),
        "quality_counts": dict(sorted(quality_counts.items())),
        "netwin_rs": (
            quality_counts["RS"] - quality_counts["R0"]
        )
        / len(labels),
        "quality_by_move": {
            move: dict(sorted(counts.items()))
            for move, counts in sorted(move_quality_counts.items())
        },
        "material_risk_response_counts": dict(
            sorted(material_risk_response_counts.items())
        ),
        "risk_category_counts": {
            f"{arm}:{category}": count
            for (arm, category), count in sorted(risk_category_counts.items())
        },
        "atomic_risk_head_qualification": {
            "minimum_events_and_nonevents": 8,
            "qualified_heads": [],
            "decision": (
                "No atomic risk head is trainable from this supplement; "
                "retain deterministic guards and descriptive event counts."
            ),
        },
        "cost": {
            "r0_mean_prompt_tokens": sum(prompt_r0) / len(prompt_r0),
            "rs_mean_prompt_tokens": sum(prompt_rs) / len(prompt_rs),
            "rs_prompt_token_increase_fraction": (
                sum(prompt_rs) / sum(prompt_r0) - 1
            ),
            "r0_mean_total_tokens": sum(total_r0) / len(total_r0),
            "rs_mean_total_tokens": sum(total_rs) / len(total_rs),
            "rs_total_token_increase_fraction": (
                sum(total_rs) / sum(total_r0) - 1
            ),
        },
        "traceability_deviations": traceability_deviations,
        "data_quality": {
            "complete": True,
            "unique_ids": True,
            "ids_match_packet_and_private_key": True,
            "all_required_quality_and_risk_fields_valid": True,
            "literal_evidence_note": (
                "One evidence field uses ordered literal fragments separated "
                "by an ellipsis; it is retained as a non-substantive "
                "traceability deviation and does not affect the quality label."
                if traceability_deviations
                else "All cited evidence is one contiguous literal excerpt."
            ),
        },
        "lineage": {
            "annotations_sha256": sha256_file(args.annotations),
            "packet_sha256": sha256_file(
                args.blind_dir / "human_blind_packet.jsonl"
            ),
            "private_key_sha256": sha256_file(
                args.blind_dir / "private_blinding_key.jsonl"
            ),
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.execution_dir / "generation_outcomes.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "human_annotations_archived.jsonl", annotations_list)
    write_jsonl(args.out_dir / "unblinded_quality_risk_labels.jsonl", labels)
    write_json(args.out_dir / "supplement_analysis_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
