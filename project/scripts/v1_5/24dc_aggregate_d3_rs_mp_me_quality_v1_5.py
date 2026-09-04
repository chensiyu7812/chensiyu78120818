#!/usr/bin/env python3
"""Validate D3 quality review and aggregate the still-valid RS/MP/ME pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROTOCOL = "pm-v1.5-d3-human-quality-blind-v1"
PROTOCOL = "pm-v1.5-d3-rs-mp-me-quality-aggregation-v1"
STATUS = "PASS_35_COMPONENT_ON_WINS_PENDING_MINIMAL_RISK"
VALID_COMPONENTS = {"RS", "MP", "ME"}
PREFERENCES = {"A", "B", "tie", "uncertain"}
CRITERIA = {
    "visible_context_fidelity",
    "immediate_helpfulness",
    "request_and_dialogue_fit",
    "emotional_understanding",
    "clarity_naturalness",
    "materially_equivalent",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _verdict(annotation: dict[str, Any], key: dict[str, Any]) -> str:
    preference = str(annotation["quality_preference"])
    if preference == "A":
        return str(key["a_role"])
    if preference == "B":
        return str(key["b_role"])
    return preference


def build(*, annotations_path: Path, blind_dir: Path, out_dir: Path) -> dict[str, Any]:
    annotations = _rows(annotations_path)
    public = {
        str(row["blind_item_id"]): row
        for row in _rows(blind_dir / "human_blind_packet.jsonl")
    }
    private = {
        str(row["blind_item_id"]): row
        for row in _rows(blind_dir / "private_blind_key.jsonl")
    }
    annotation_by_id = {
        str(row["blind_item_id"]): row for row in annotations
    }
    checks = {
        "160_annotations": len(annotations) == 160,
        "annotation_ids_unique": len(annotation_by_id) == 160,
        "ids_exactly_match_blind_packet": set(annotation_by_id) == set(public) == set(private),
        "protocol_exact": all(row.get("protocol") == SOURCE_PROTOCOL for row in annotations),
        "preference_enum_valid": all(row.get("quality_preference") in PREFERENCES for row in annotations),
        "criterion_enum_valid": all(row.get("decisive_criterion") in CRITERIA for row in annotations),
        "required_text_complete": all(
            str(row.get(field) or "").strip()
            for row in annotations
            for field in ("blind_item_id", "quality_notes", "annotator_id")
        ),
        "one_annotator": len({str(row["annotator_id"]) for row in annotations}) == 1,
    }
    if not all(checks.values()):
        raise RuntimeError(f"D3 review validation failed: {checks}")

    joined: list[dict[str, Any]] = []
    for annotation in annotations:
        blind_id = str(annotation["blind_item_id"])
        key = private[blind_id]
        joined.append(
            {
                **annotation,
                **key,
                "semantic_verdict": _verdict(annotation, key),
            }
        )
    retained = [row for row in joined if row["component"] in VALID_COMPONENTS]
    excluded_ms = [row for row in joined if row["component"] == "MS"]
    if len(retained) != 120 or len(excluded_ms) != 40:
        raise RuntimeError("D3 component partition is not 120 retained + 40 invalidated MS")

    pair_rows: list[dict[str, Any]] = []
    for row in retained:
        verdict = str(row["semantic_verdict"])
        observation = 1.0 if verdict == "treatment" else 0.0 if verdict in {"control", "tie"} else None
        pair_rows.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": row["blind_item_id"],
                "pair_id": row["pair_id"],
                "pair_role": row["pair_role"],
                "contrast_slot_id": row["contrast_slot_id"],
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "component": row["component"],
                "control_action": row.get("control_action"),
                "treatment_action": row.get("treatment_action"),
                "a_role": row["a_role"],
                "b_role": row["b_role"],
                "quality_preference": row["quality_preference"],
                "quality_verdict": verdict,
                "decisive_criterion": row["decisive_criterion"],
                "quality_notes": row["quality_notes"],
                "annotator_id": row["annotator_id"],
                "treatment_win_observation_pre_risk": observation,
                "quality_label_pre_risk": (
                    "ON_QUALITY_WIN_PENDING_MATERIAL_RISK"
                    if verdict == "treatment"
                    else "OFF_NO_MATERIAL_QUALITY_BENEFIT"
                    if verdict in {"control", "tie"}
                    else "UNKNOWN_QUALITY"
                ),
            }
        )

    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_slot[str(row["contrast_slot_id"])].append(row)
    state_rows: list[dict[str, Any]] = []
    for slot_id, rows in sorted(by_slot.items()):
        values = [
            float(row["treatment_win_observation_pre_risk"])
            for row in rows
            if row["treatment_win_observation_pre_risk"] is not None
        ]
        state_rows.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": slot_id,
                "state_id": rows[0]["state_id"],
                "user_id": rows[0]["user_id"],
                "component": rows[0]["component"],
                "pair_count": len(rows),
                "observed_pair_count": len(values),
                "treatment_win_soft_target_pre_risk": sum(values) / len(values) if values else None,
                "state_training_weight": 1.0,
                "not_a_deterministic_state_gold": True,
            }
        )

    repeat_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        repeat_groups[(str(row["component"]), str(row["contrast_slot_id"]))].append(row)
    repeat_agreement: dict[str, dict[str, int]] = {}
    for component in sorted(VALID_COMPONENTS):
        groups = [rows for (comp, _), rows in repeat_groups.items() if comp == component and len(rows) == 2]
        repeat_agreement[component] = {
            "exact_semantic_agreement": sum(rows[0]["quality_verdict"] == rows[1]["quality_verdict"] for rows in groups),
            "repeat_groups": len(groups),
        }

    verdicts = {
        component: dict(sorted(Counter(row["quality_verdict"] for row in pair_rows if row["component"] == component).items()))
        for component in sorted(VALID_COMPONENTS)
    }
    on_wins = sum(row["quality_verdict"] == "treatment" for row in pair_rows)
    fidelity_rows = [
        row for row in pair_rows
        if row["component"] in {"MP", "ME"} and row["decisive_criterion"] == "visible_context_fidelity"
    ]
    checks.update(
        {
            "120_retained_pairs": len(pair_rows) == 120,
            "96_independent_states": len(state_rows) == 96,
            "40_original_ms_rows_excluded": len(excluded_ms) == 40,
            "35_on_wins": on_wins == 35,
            "memory_fidelity_rows_exhaustively_identified": len(fidelity_rows) == 4,
            "state_weights_one": all(row["state_training_weight"] == 1.0 for row in state_rows),
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"D3 aggregation checks failed: {checks}")

    out_dir.mkdir(parents=True, exist_ok=True)
    validated_path = out_dir / "validated_all_160_quality_annotations.jsonl"
    pair_path = out_dir / "pair_quality_measurements_pre_risk.jsonl"
    state_path = out_dir / "state_soft_targets_pre_risk.jsonl"
    fidelity_path = out_dir / "mp_me_fidelity_rows_for_evidence_audit.jsonl"
    write_jsonl(validated_path, annotations)
    write_jsonl(pair_path, pair_rows)
    write_jsonl(state_path, state_rows)
    write_jsonl(fidelity_path, fidelity_rows)
    report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "retained_components": ["RS", "MP", "ME"],
        "invalidated_component": "original D3 MS",
        "retained_pairs": len(pair_rows),
        "independent_states": len(state_rows),
        "component_pair_verdict_counts": verdicts,
        "component_on_quality_wins_pending_risk": on_wins,
        "repeat_agreement": repeat_agreement,
        "mp_me_visible_fidelity_rows": len(fidelity_rows),
        "checks": checks,
        "inputs": {
            "annotations_sha256": sha256_file(annotations_path),
            "blind_manifest_sha256": sha256_file(blind_dir / "manifest.json"),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (validated_path, pair_path, state_path, fidelity_path)
        },
    }
    write_json(out_dir / "aggregation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/home/tokkio/.codex/attachments/f209b261-6f55-4c9d-a585-ce4ce1fe1c6b/pasted-text.txt"),
    )
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_blind_review_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_rs_mp_me_quality_aggregation_v1",
    )
    args = parser.parse_args()
    report = build(annotations_path=args.annotations, blind_dir=args.blind_dir, out_dir=args.out_dir)
    print(json.dumps({key: report[key] for key in ("protocol", "status", "component_pair_verdict_counts", "component_on_quality_wins_pending_risk")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
