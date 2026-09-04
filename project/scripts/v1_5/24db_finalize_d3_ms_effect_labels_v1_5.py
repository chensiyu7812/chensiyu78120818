#!/usr/bin/env python3
"""Finalize D3 MS pair and state targets after the one-item risk review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
RISK_PROTOCOL = "pm-v1.5-d3-ms-on-win-minimal-risk-review-v1"
PROTOCOL = "pm-v1.5-d3-ms-final-effect-labels-v1"
STATUS = "FINAL_MS_MEASUREMENT_COMPLETE_NOT_DISCRIMINATIVE_TRAIN_QUALIFIED"
RISK_VERDICTS = {"yes", "no", "uncertain"}
RISK_CATEGORIES = {
    "explicit_boundary_violation",
    "unsupported_personal_claim",
    "stale_or_conflicting_evidence_use",
    "excessive_directiveness",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def build(
    *,
    aggregation_dir: Path,
    risk_annotations_path: Path,
    risk_review_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    aggregation = json.loads(
        (aggregation_dir / "aggregation_report.json").read_text(encoding="utf-8")
    )
    pairs = _rows(
        aggregation_dir / "pair_quality_measurements_pre_risk.jsonl"
    )
    annotations = _rows(risk_annotations_path)
    public = {
        str(row["review_item_id"]): row
        for row in _rows(risk_review_dir / "human_risk_packet.jsonl")
    }
    private = {
        str(row["review_item_id"]): row
        for row in _rows(risk_review_dir / "private_risk_key.jsonl")
    }
    annotation_by_id = {
        str(row["review_item_id"]): row for row in annotations
    }
    if (
        aggregation.get("status") != "PASS_ONE_MS_ON_WIN_PENDING_MINIMAL_RISK"
        or not all(aggregation.get("checks", {}).values())
        or len(pairs) != 40
        or len(annotations) != len(annotation_by_id)
        or len(annotation_by_id) != 1
        or set(annotation_by_id) != set(public)
        or set(public) != set(private)
    ):
        raise RuntimeError("MS risk finalization inputs are incomplete")
    annotation = next(iter(annotation_by_id.values()))
    review_id = str(annotation["review_item_id"])
    verdict = str(annotation.get("any_material_risk") or "")
    categories = annotation.get("selected_categories")
    evidence = annotation.get("evidence_by_category")
    if (
        annotation.get("protocol") != RISK_PROTOCOL
        or verdict not in RISK_VERDICTS
        or not isinstance(categories, list)
        or len(categories) != len(set(categories))
        or set(categories) - RISK_CATEGORIES
        or not isinstance(evidence, dict)
        or not str(annotation.get("annotator_id") or "").strip()
        or (verdict == "no" and (categories or evidence))
        or (verdict == "uncertain" and (categories or evidence))
        or (verdict == "yes" and (not categories or set(evidence) != set(categories)))
    ):
        raise RuntimeError("MS risk annotation violates the frozen schema")
    key = private[review_id]
    item = public[review_id]
    if (
        key["quality_label"] != "ON_QUALITY_WIN_PENDING_MATERIAL_RISK"
        or key["candidate_response_sha256"]
        != __import__("hashlib").sha256(
            str(item["candidate_response"]).encode("utf-8")
        ).hexdigest()
    ):
        raise RuntimeError("MS risk candidate lineage mismatch")

    risk_pair_id = str(key["pair_id"])
    final_pairs: list[dict[str, Any]] = []
    for row in pairs:
        quality_verdict = str(row["quality_verdict"])
        if quality_verdict != "treatment":
            final_observation = row["treatment_win_observation"]
            final_label = (
                "OFF_NO_MATERIAL_QUALITY_BENEFIT"
                if quality_verdict in {"control", "tie"}
                else "UNKNOWN_QUALITY"
            )
            risk_status = "NOT_APPLICABLE_NO_ON_WIN"
        else:
            if str(row["pair_id"]) != risk_pair_id:
                raise RuntimeError("unreviewed MS quality winner")
            risk_status = f"MATERIAL_RISK_{verdict.upper()}"
            if verdict == "no":
                final_observation = 1.0
                final_label = "ON_MATERIAL_BENEFIT_NO_MATERIAL_RISK"
            elif verdict == "yes":
                final_observation = 0.0
                final_label = "OFF_MATERIAL_RISK_BLOCKED"
            else:
                final_observation = None
                final_label = "UNKNOWN_MATERIAL_RISK"
        final_pairs.append(
            {
                **row,
                "protocol": PROTOCOL,
                "risk_status": risk_status,
                "final_effect_label": final_label,
                "treatment_win_observation_post_risk": final_observation,
            }
        )

    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_pairs:
        by_slot[str(row["contrast_slot_id"])].append(row)
    final_states: list[dict[str, Any]] = []
    for slot_id, rows in by_slot.items():
        values = [
            float(row["treatment_win_observation_post_risk"])
            for row in rows
            if row["treatment_win_observation_post_risk"] is not None
        ]
        target = sum(values) / len(values) if values else None
        final_states.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": slot_id,
                "state_id": rows[0]["state_id"],
                "user_id": rows[0]["user_id"],
                "component": "MS",
                "pair_count": len(rows),
                "observed_pair_count": len(values),
                "treatment_win_soft_target": target,
                "state_training_weight": 1.0,
                "not_a_deterministic_state_gold": True,
            }
        )

    pair_label_counts = Counter(row["final_effect_label"] for row in final_pairs)
    target_counts = Counter(row["treatment_win_soft_target"] for row in final_states)
    positive_states = sum(
        row["treatment_win_soft_target"] is not None
        and float(row["treatment_win_soft_target"]) > 0.0
        for row in final_states
    )
    confident_positive_states = sum(
        row["treatment_win_soft_target"] == 1.0 for row in final_states
    )
    checks = {
        "one_risk_annotation": len(annotations) == 1,
        "risk_verdict_no": verdict == "no",
        "40_final_pairs": len(final_pairs) == 40,
        "32_final_states": len(final_states) == 32,
        "state_targets_31_zero_1_half": target_counts
        == Counter({0.0: 31, 0.5: 1}),
        "one_positive_state_only": positive_states == 1,
        "zero_confident_positive_states": confident_positive_states == 0,
        "state_weights_one": all(
            row["state_training_weight"] == 1.0 for row in final_states
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"MS final label checks failed: {checks}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_dir / "final_pair_effect_labels.jsonl"
    state_path = out_dir / "final_state_soft_targets.jsonl"
    validated_risk_path = out_dir / "validated_risk_annotations.jsonl"
    write_jsonl(pair_path, final_pairs)
    write_jsonl(state_path, final_states)
    write_jsonl(
        validated_risk_path,
        [
            {
                **annotation,
                "evidence_grounding_validation": "PASS",
                "candidate_response_lineage_validation": "PASS",
            }
        ],
    )
    report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "risk_result": {
            "reviewed_quality_on_wins": 1,
            "material_risk_yes": 0,
            "material_risk_no": 1,
            "material_risk_uncertain": 0,
            "denominator_scope": "quality-winning MS-on responses only",
            "not_an_overall_MS_risk_rate": True,
            "not_a_control_vs_treatment_risk_difference": True,
        },
        "pair_final_label_counts": dict(sorted(pair_label_counts.items())),
        "independent_states": len(final_states),
        "state_soft_target_counts": {
            str(key): value for key, value in sorted(target_counts.items())
        },
        "positive_state_count": positive_states,
        "confident_positive_state_count": confident_positive_states,
        "discriminative_head_training_qualified": False,
        "reason_not_train_qualified": (
            "Only one independent state has a nonzero target (0.5), and no "
            "state has a confident target of 1.0; there is no support for "
            "learning when to turn MS on."
        ),
        "runtime_fallback": "MS learned gate unavailable; deterministic off",
        "claim_boundary": (
            "Correct Top-2 MS retrieval was established, but beneficial MS "
            "injection and a learnable conditional MS-on gate were not."
        ),
        "checks": checks,
        "inputs": {
            "aggregation_report_sha256": sha256_file(
                aggregation_dir / "aggregation_report.json"
            ),
            "risk_annotations_sha256": sha256_file(risk_annotations_path),
            "risk_manifest_sha256": sha256_file(risk_review_dir / "manifest.json"),
        },
        "outputs": {
            pair_path.name: sha256_file(pair_path),
            state_path.name: sha256_file(state_path),
            validated_risk_path.name: sha256_file(validated_risk_path),
        },
    }
    write_json(out_dir / "finalization_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregation-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_grounded_quality_aggregation_v1",
    )
    parser.add_argument(
        "--risk-annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d3_ms_on_win_risk_review_v1_candidate/"
        "risk_annotations_formal.jsonl",
    )
    parser.add_argument(
        "--risk-review-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_on_win_risk_review_v1_candidate",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_final_effect_labels_v1",
    )
    args = parser.parse_args()
    report = build(
        aggregation_dir=args.aggregation_dir,
        risk_annotations_path=args.risk_annotations,
        risk_review_dir=args.risk_review_dir,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol",
                    "status",
                    "state_soft_target_counts",
                    "positive_state_count",
                    "discriminative_head_training_qualified",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
