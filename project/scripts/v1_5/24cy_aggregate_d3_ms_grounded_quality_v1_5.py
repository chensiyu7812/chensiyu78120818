#!/usr/bin/env python3
"""Replace five invalid fidelity decisions and aggregate D3 MS soft targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROTOCOL = "pm-v1.5-d3-ms-replacement-human-quality-blind-v1"
GROUNDED_PROTOCOL = "pm-v1.5-d3-ms-grounded-fidelity-adjudication-v1"
PROTOCOL = "pm-v1.5-d3-ms-grounded-quality-aggregation-v1"
STATUS = "PASS_ONE_MS_ON_WIN_PENDING_MINIMAL_RISK"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _decode(preference: str, a_role: str, b_role: str) -> str:
    if preference == "A":
        return a_role
    if preference == "B":
        return b_role
    if preference in {"tie", "uncertain"}:
        return preference
    raise ValueError(f"invalid quality preference: {preference}")


def build(
    *,
    source_annotations_path: Path,
    grounded_annotations_path: Path,
    source_blind_dir: Path,
    grounded_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    source = _rows(source_annotations_path)
    grounded = _rows(grounded_annotations_path)
    source_keys = {
        str(row["blind_item_id"]): row
        for row in _rows(source_blind_dir / "private_blind_key.jsonl")
    }
    grounded_keys = {
        str(row["blind_item_id"]): row
        for row in _rows(grounded_dir / "private_grounded_key.jsonl")
    }
    source_by_id = {str(row["blind_item_id"]): row for row in source}
    grounded_by_id = {str(row["blind_item_id"]): row for row in grounded}
    if (
        len(source) != len(source_by_id) != 0
        or len(source_by_id) != 40
        or set(source_by_id) != set(source_keys)
        or any(row.get("protocol") != SOURCE_PROTOCOL for row in source)
    ):
        raise RuntimeError("source quality annotations are invalid")
    if (
        len(grounded) != len(grounded_by_id)
        or len(grounded_by_id) != 5
        or set(grounded_by_id) != set(grounded_keys)
        or any(row.get("protocol") != GROUNDED_PROTOCOL for row in grounded)
        or any(
            row.get("quality_preference") not in {"A", "B", "tie", "uncertain"}
            or not str(row.get("decisive_criterion") or "").strip()
            or not str(row.get("annotator_id") or "").strip()
            for row in grounded
        )
    ):
        raise RuntimeError("grounded adjudications are invalid")

    replacement_by_source_id = {
        str(key["source_blind_item_id"]): grounded_by_id[grounded_id]
        for grounded_id, key in grounded_keys.items()
    }
    if set(replacement_by_source_id) != {
        blind_id
        for blind_id, row in source_by_id.items()
        if row["decisive_criterion"] == "visible_context_fidelity"
    } or len(replacement_by_source_id) != 5:
        raise RuntimeError("replacement scope is not exactly the five fidelity rows")

    pair_rows: list[dict[str, Any]] = []
    for old_id, old_annotation in source_by_id.items():
        key = source_keys[old_id]
        replacement = replacement_by_source_id.get(old_id)
        if replacement is None:
            preference = str(old_annotation["quality_preference"])
            verdict = _decode(preference, str(key["a_role"]), str(key["b_role"]))
            criterion = str(old_annotation["decisive_criterion"])
            notes = str(old_annotation["quality_notes"])
            annotation_source = "original_non_fidelity_frozen"
            annotation_id = old_id
        else:
            grounded_id = next(
                grounded_id
                for grounded_id, grounded_key in grounded_keys.items()
                if grounded_key["source_blind_item_id"] == old_id
            )
            grounded_key = grounded_keys[grounded_id]
            preference = str(replacement["quality_preference"])
            verdict = _decode(
                preference,
                str(grounded_key["new_a_role"]),
                str(grounded_key["new_b_role"]),
            )
            criterion = str(replacement["decisive_criterion"])
            notes = str(replacement["quality_notes"])
            annotation_source = "grounded_fidelity_replacement"
            annotation_id = grounded_id
        observation = (
            None if verdict == "uncertain" else float(verdict == "treatment")
        )
        pair_rows.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": old_id,
                "active_annotation_id": annotation_id,
                "annotation_source": annotation_source,
                "pair_id": key["pair_id"],
                "pair_role": key["pair_role"],
                "contrast_slot_id": key["contrast_slot_id"],
                "state_id": key["state_id"],
                "user_id": key["user_id"],
                "component": "MS",
                "quality_preference": preference,
                "quality_verdict": verdict,
                "decisive_criterion": criterion,
                "quality_notes": notes,
                "treatment_win_observation": observation,
                "pair_is_independent_state": key["pair_role"] == "primary",
                "risk_status": (
                    "PENDING_MINIMAL_RISK"
                    if verdict == "treatment"
                    else "NOT_APPLICABLE_NO_ON_WIN"
                ),
            }
        )

    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_slot[str(row["contrast_slot_id"])].append(row)
    state_rows: list[dict[str, Any]] = []
    for slot_id, rows in by_slot.items():
        observations = [
            float(row["treatment_win_observation"])
            for row in rows
            if row["treatment_win_observation"] is not None
        ]
        target = sum(observations) / len(observations) if observations else None
        state_rows.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": slot_id,
                "state_id": rows[0]["state_id"],
                "user_id": rows[0]["user_id"],
                "component": "MS",
                "pair_count": len(rows),
                "observed_pair_count": len(observations),
                "treatment_win_soft_target_pre_risk": target,
                "not_a_deterministic_state_gold": True,
                "state_training_weight": 1.0,
            }
        )

    verdict_counts = Counter(row["quality_verdict"] for row in pair_rows)
    replacement_counts = Counter(row["annotation_source"] for row in pair_rows)
    target_counts = Counter(row["treatment_win_soft_target_pre_risk"] for row in state_rows)
    repeat_slots = [rows for rows in by_slot.values() if len(rows) == 2]
    repeat_exact = sum(
        rows[0]["quality_verdict"] == rows[1]["quality_verdict"]
        for rows in repeat_slots
    )
    checks = {
        "40_pair_measurements": len(pair_rows) == 40,
        "pair_ids_unique": len({row["pair_id"] for row in pair_rows}) == 40,
        "five_for_five_replacement": replacement_counts
        == Counter(
            {
                "original_non_fidelity_frozen": 35,
                "grounded_fidelity_replacement": 5,
            }
        ),
        "verdicts_on1_off1_tie38": verdict_counts
        == Counter({"treatment": 1, "control": 1, "tie": 38}),
        "32_state_targets": len(state_rows) == 32,
        "state_targets_31_zero_1_half": target_counts
        == Counter({0.0: 31, 0.5: 1}),
        "eight_repeat_states": len(repeat_slots) == 8,
        "repeat_exact_7_of_8": repeat_exact == 7,
        "exactly_one_on_win_pending_risk": sum(
            row["risk_status"] == "PENDING_MINIMAL_RISK" for row in pair_rows
        )
        == 1,
        "repeat_does_not_increase_state_weight": all(
            row["state_training_weight"] == 1.0 for row in state_rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"MS grounded aggregation failed: {checks}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_dir / "pair_quality_measurements_pre_risk.jsonl"
    state_path = out_dir / "state_soft_targets_pre_risk.jsonl"
    write_jsonl(pair_path, pair_rows)
    write_jsonl(state_path, state_rows)
    report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "pair_verdict_counts": dict(sorted(verdict_counts.items())),
        "replacement_counts": dict(sorted(replacement_counts.items())),
        "independent_states": len(state_rows),
        "state_soft_target_counts": {
            str(key): value for key, value in sorted(target_counts.items())
        },
        "repeat_exact_agreement": {
            "numerator": repeat_exact,
            "denominator": len(repeat_slots),
            "rate": repeat_exact / len(repeat_slots),
        },
        "quality_on_wins_pending_risk": 1,
        "formal_training_authorized": False,
        "blocked_only_on": "one_on_win_minimal_risk_review",
        "checks": checks,
        "inputs": {
            "source_annotations_sha256": sha256_file(source_annotations_path),
            "grounded_annotations_sha256": sha256_file(grounded_annotations_path),
        },
        "outputs": {
            pair_path.name: sha256_file(pair_path),
            state_path.name: sha256_file(state_path),
        },
    }
    write_json(out_dir / "aggregation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d3_ms_grounded_fidelity_adjudication_v1/"
        "source_visible_only_annotations.jsonl",
    )
    parser.add_argument(
        "--grounded-annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d3_ms_grounded_fidelity_adjudication_v1/"
        "grounded_adjudications_formal.jsonl",
    )
    parser.add_argument(
        "--source-blind-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_replacement_blind_v1",
    )
    parser.add_argument(
        "--grounded-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d3_ms_grounded_fidelity_adjudication_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_grounded_quality_aggregation_v1",
    )
    args = parser.parse_args()
    report = build(
        source_annotations_path=args.source_annotations,
        grounded_annotations_path=args.grounded_annotations,
        source_blind_dir=args.source_blind_dir,
        grounded_dir=args.grounded_dir,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol",
                    "status",
                    "pair_verdict_counts",
                    "state_soft_target_counts",
                    "quality_on_wins_pending_risk",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
