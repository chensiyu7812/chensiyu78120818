#!/usr/bin/env python3
"""Validate and aggregate D2 primary and independent-overlap reviews."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d2-measurement-human-quality-blind-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


CRITERION_ALIASES = {
    "visible_context_fidelity": "visible_context_fidelity",
    "request_dialogue_fit": "request_and_dialogue_fit",
    "request_and_dialogue_fit": "request_and_dialogue_fit",
    "emotional_understanding": "emotional_understanding",
    "immediate_helpfulness": "immediate_helpfulness",
    "clarity_naturalness_not_overloaded": "clarity_naturalness",
    "clarity_naturalness": "clarity_naturalness",
    "materially_equivalent": "materially_equivalent",
}


def _annotation_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    marker = "var ANN = "
    if marker not in text:
        return _rows(path)
    start = text.index(marker) + len(marker)
    end = text.index("\n};", start) + 2
    annotations = json.loads(text[start:end])
    return [
        {
            "protocol": PROTOCOL,
            "blind_item_id": blind_item_id,
            **dict(values),
        }
        for blind_item_id, values in annotations.items()
    ]


def _validate_annotations(
    *,
    path: Path,
    expected_packet: list[dict[str, Any]],
    expected_role: str,
) -> dict[str, Any]:
    rows = _annotation_rows(path)
    expected_ids = {
        str(row["blind_item_id"]) for row in expected_packet
    }
    actual_ids = [str(row.get("blind_item_id")) for row in rows]
    protocols = Counter(str(row.get("protocol")) for row in rows)
    preferences = Counter(
        str(row.get("quality_preference")) for row in rows
    )
    criteria = Counter(
        str(row.get("decisive_criterion")) for row in rows
    )
    normalized_criteria = Counter(
        CRITERION_ALIASES.get(value, f"INVALID:{value}")
        for value, count in criteria.items()
        for _ in range(count)
    )
    annotators = Counter(str(row.get("annotator_id")) for row in rows)
    checks = {
        "row_count": len(rows) == len(expected_packet),
        "unique_blind_item_ids": len(actual_ids) == len(set(actual_ids)),
        "exact_expected_id_set": set(actual_ids) == expected_ids,
        "protocol": protocols == Counter({PROTOCOL: len(rows)}),
        "valid_preferences": set(preferences)
        <= {"A", "B", "tie", "uncertain"},
        "valid_criteria_or_frozen_aliases": set(criteria)
        <= set(CRITERION_ALIASES),
        "all_reasons_nonempty": all(
            str(row.get("quality_notes") or "").strip() for row in rows
        ),
        "all_annotator_ids_nonempty": all(
            str(row.get("annotator_id") or "").strip() for row in rows
        ),
    }
    return {
        "role": expected_role,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "row_count": len(rows),
        "expected_row_count": len(expected_packet),
        "protocol_counts": dict(protocols),
        "preference_counts": dict(preferences),
        "criterion_counts_raw": dict(criteria),
        "criterion_counts_normalized": dict(normalized_criteria),
        "annotator_counts": dict(annotators),
        "id_overlap_count": len(set(actual_ids) & expected_ids),
        "missing_expected_ids": len(expected_ids - set(actual_ids)),
        "unexpected_ids": len(set(actual_ids) - expected_ids),
        "checks": checks,
        "valid": all(checks.values()),
    }


def _semantic_direction(
    annotation: dict[str, Any],
    key: dict[str, Any],
) -> str:
    preference = str(annotation["quality_preference"])
    if preference in {"tie", "uncertain"}:
        return preference
    return str(key[f"{preference.lower()}_role"])


def _kappa(confusion: Counter[tuple[str, str]]) -> float:
    total = sum(confusion.values())
    left: Counter[str] = Counter()
    right: Counter[str] = Counter()
    observed = 0
    for (a_value, b_value), count in confusion.items():
        left[a_value] += count
        right[b_value] += count
        observed += count * int(a_value == b_value)
    observed_rate = observed / total
    expected_rate = sum(
        left[value] / total * right[value] / total
        for value in set(left) | set(right)
    )
    return (observed_rate - expected_rate) / (1.0 - expected_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d2_measurement_blind_v1",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d2_measurement_blueprint_v1/"
        "d2_measurement_blueprint.jsonl",
    )
    parser.add_argument(
        "--historical-labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_final_effect_labels_v1/"
        "component_effect_labels.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d2_measurement_review_analysis_v1",
    )
    args = parser.parse_args()

    primary_packet = _rows(
        args.review_dir / "primary_72/human_blind_packet.jsonl"
    )
    primary_key = _rows(
        args.review_dir / "primary_72/private_blind_key.jsonl"
    )
    secondary_packet = _rows(
        args.review_dir
        / "independent_overlap_32/human_blind_packet.jsonl"
    )
    secondary_key = _rows(
        args.review_dir
        / "independent_overlap_32/private_blind_key.jsonl"
    )
    primary = _validate_annotations(
        path=args.primary,
        expected_packet=primary_packet,
        expected_role="primary_72",
    )
    secondary = _validate_annotations(
        path=args.secondary,
        expected_packet=secondary_packet,
        expected_role="independent_overlap_32",
    )

    result: dict[str, Any] = {
        "protocol": "pm-v1.5-d2-measurement-review-analysis-v1",
        "status": "BLOCKED_BY_INVALID_PRIMARY",
        "primary_validation": {
            key: value for key, value in primary.items() if key != "rows"
        },
        "secondary_validation": {
            key: value for key, value in secondary.items() if key != "rows"
        },
        "effect_labels_created": False,
        "formal_D2_gate_passed": False,
    }
    if not primary["valid"]:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.out_dir / "analysis.json", result)
        print({"status": result["status"]})
        return

    primary_annotations = {
        str(row["blind_item_id"]): row for row in primary["rows"]
    }
    primary_key_by_pair = {
        str(row["pair_id"]): row for row in primary_key
    }
    primary_directions_by_state: dict[str, list[str]] = defaultdict(list)
    primary_directions_by_pair: dict[str, str] = {}
    new_pair_direction_counts: Counter[tuple[str, str]] = Counter()
    for key in primary_key:
        annotation = primary_annotations[str(key["blind_item_id"])]
        direction = _semantic_direction(annotation, key)
        primary_directions_by_state[str(key["d2_state_id"])].append(
            direction
        )
        primary_directions_by_pair[str(key["pair_id"])] = direction
        new_pair_direction_counts[(str(key["component"]), direction)] += 1

    historical = {
        str(row["contrast_slot_id"]): row
        for row in _rows(args.historical_labels)
    }
    state_rows: list[dict[str, Any]] = []
    for blueprint in _rows(args.blueprint):
        state_id = str(blueprint["d2_state_id"])
        directions = list(primary_directions_by_state[state_id])
        reference = blueprint.get("historical_pair_reference")
        if isinstance(reference, dict):
            directions.insert(
                0,
                str(
                    historical[str(reference["contrast_slot_id"])][
                        "quality_verdict"
                    ]
                ),
            )
        counts = Counter(directions)
        majority = [
            direction
            for direction, count in counts.items()
            if direction != "uncertain" and count >= 2
        ]
        binary = [
            "on"
            if direction == "treatment"
            else (
                "off"
                if direction in {"control", "tie"}
                else "unknown"
            )
            for direction in directions
        ]
        binary_counts = Counter(binary)
        binary_majority = (
            "on"
            if binary_counts["on"] >= 2
            else ("off" if binary_counts["off"] >= 2 else None)
        )
        state_rows.append(
            {
                "d2_state_id": state_id,
                "component": blueprint["component"],
                "directions": directions,
                "exact_three_way_majority": (
                    majority[0] if len(majority) == 1 else None
                ),
                "binary_on_off_majority": binary_majority,
            }
        )

    exact_pass_count = sum(
        row["exact_three_way_majority"] is not None for row in state_rows
    )
    binary_pass_count = sum(
        row["binary_on_off_majority"] is not None for row in state_rows
    )
    uncertain_fraction = (
        primary["preference_counts"].get("uncertain", 0)
        / primary["row_count"]
    )
    by_component: dict[str, Any] = {}
    for component in ("RS", "MP", "MS", "ME"):
        rows = [
            row for row in state_rows if row["component"] == component
        ]
        by_component[component] = {
            "states": len(rows),
            "exact_three_way_reproduced": sum(
                row["exact_three_way_majority"] is not None
                for row in rows
            ),
            "binary_on_off_reproduced": sum(
                row["binary_on_off_majority"] is not None
                for row in rows
            ),
            "new_pair_directions": {
                direction: new_pair_direction_counts[
                    (component, direction)
                ]
                for direction in (
                    "treatment",
                    "control",
                    "tie",
                    "uncertain",
                )
            },
        }

    primary_gate = {
        "exact_three_way_reproduction_rate": exact_pass_count
        / len(state_rows),
        "exact_three_way_gate_min": 0.70,
        "exact_three_way_gate_passed": exact_pass_count
        / len(state_rows)
        >= 0.70,
        "binary_on_off_reproduction_rate_diagnostic": binary_pass_count
        / len(state_rows),
        "uncertain_fraction": uncertain_fraction,
        "uncertain_fraction_max": 0.10,
        "uncertain_gate_passed": uncertain_fraction <= 0.10,
        "by_component": by_component,
        "nonreproduced_exact_states": [
            row
            for row in state_rows
            if row["exact_three_way_majority"] is None
        ],
    }
    result["primary_measurement_gate"] = primary_gate

    if not secondary["valid"]:
        result["status"] = (
            "PRIMARY_REPRODUCIBILITY_PASS_SECONDARY_INVALID_REVIEW_REQUIRED"
            if primary_gate["exact_three_way_gate_passed"]
            and primary_gate["uncertain_gate_passed"]
            else "PRIMARY_REPRODUCIBILITY_FAIL_SECONDARY_INVALID"
        )
        result["secondary_issue"] = {
            "severity": "critical for formal D2 completion",
            "finding": (
                "The supplied second file is not the frozen D2 overlap packet; "
                "it cannot estimate inter-rater agreement."
            ),
            "required_remediation": (
                "Complete the frozen independent_overlap_32 page with a "
                "reviewer who did not read the primary annotations."
            ),
        }
    else:
        primary_annotators = set(
            primary["annotator_counts"]
        )
        secondary_annotators = set(
            secondary["annotator_counts"]
        )
        annotators_independent = not (
            primary_annotators & secondary_annotators
        )
        secondary_annotations = {
            str(row["blind_item_id"]): row for row in secondary["rows"]
        }
        secondary_key_by_pair = {
            str(row["pair_id"]): row for row in secondary_key
        }
        secondary_direction_by_pair: dict[str, str] = {}
        for key in secondary_key:
            annotation = secondary_annotations[
                str(key["blind_item_id"])
            ]
            secondary_direction_by_pair[str(key["pair_id"])] = (
                _semantic_direction(annotation, key)
            )
        shared = sorted(secondary_direction_by_pair)
        confusion: Counter[tuple[str, str]] = Counter(
            (
                primary_directions_by_pair[pair_id],
                secondary_direction_by_pair[pair_id],
            )
            for pair_id in shared
        )
        matches = sum(
            count
            for (primary_value, secondary_value), count in confusion.items()
            if primary_value == secondary_value
        )
        agreement = matches / len(shared)
        binary_confusion: Counter[tuple[str, str]] = Counter()
        for (primary_value, secondary_value), count in confusion.items():
            primary_binary = (
                "on" if primary_value == "treatment" else "off"
            )
            secondary_binary = (
                "on" if secondary_value == "treatment" else "off"
            )
            binary_confusion[(primary_binary, secondary_binary)] += count
        binary_matches = sum(
            count
            for (primary_value, secondary_value), count in (
                binary_confusion.items()
            )
            if primary_value == secondary_value
        )
        component_agreement: dict[str, Any] = {}
        for component in ("RS", "MP", "MS", "ME"):
            component_pairs = [
                str(key["pair_id"])
                for key in secondary_key
                if key["component"] == component
            ]
            component_matches = sum(
                primary_directions_by_pair[pair_id]
                == secondary_direction_by_pair[pair_id]
                for pair_id in component_pairs
            )
            component_agreement[component] = {
                "pairs": len(component_pairs),
                "exact_matches": component_matches,
                "exact_agreement": component_matches
                / len(component_pairs),
            }
        disagreement_rows: list[dict[str, Any]] = []
        for pair_id in shared:
            primary_direction = primary_directions_by_pair[pair_id]
            secondary_direction = secondary_direction_by_pair[pair_id]
            if primary_direction == secondary_direction:
                continue
            primary_key_row = primary_key_by_pair[pair_id]
            secondary_key_row = secondary_key_by_pair[pair_id]
            primary_annotation = primary_annotations[
                str(primary_key_row["blind_item_id"])
            ]
            secondary_annotation = secondary_annotations[
                str(secondary_key_row["blind_item_id"])
            ]
            disagreement_rows.append(
                {
                    "pair_id": pair_id,
                    "component": secondary_key_row["component"],
                    "primary_blind_item_id": primary_key_row[
                        "blind_item_id"
                    ],
                    "secondary_blind_item_id": secondary_key_row[
                        "blind_item_id"
                    ],
                    "primary_preference": primary_annotation[
                        "quality_preference"
                    ],
                    "secondary_preference": secondary_annotation[
                        "quality_preference"
                    ],
                    "primary_direction": primary_direction,
                    "secondary_direction": secondary_direction,
                    "primary_binary": (
                        "on" if primary_direction == "treatment" else "off"
                    ),
                    "secondary_binary": (
                        "on"
                        if secondary_direction == "treatment"
                        else "off"
                    ),
                    "primary_criterion": CRITERION_ALIASES[
                        primary_annotation["decisive_criterion"]
                    ],
                    "secondary_criterion": CRITERION_ALIASES[
                        secondary_annotation["decisive_criterion"]
                    ],
                    "primary_notes": primary_annotation["quality_notes"],
                    "secondary_notes": secondary_annotation[
                        "quality_notes"
                    ],
                }
            )
        result["inter_rater_gate"] = {
            "shared_pairs": len(shared),
            "exact_matches": matches,
            "exact_agreement": agreement,
            "exact_cohen_kappa_diagnostic": _kappa(confusion),
            "gate_min": 0.75,
            "annotator_ids_are_disjoint": annotators_independent,
            "confusion_primary_by_secondary": {
                f"{primary_value}|{secondary_value}": count
                for (primary_value, secondary_value), count in sorted(
                    confusion.items()
                )
            },
            "component_agreement": component_agreement,
            "disagreements": disagreement_rows,
            "exact_disagreement_count": len(disagreement_rows),
            "binary_disagreement_count": sum(
                row["primary_binary"] != row["secondary_binary"]
                for row in disagreement_rows
            ),
            "gate_passed": agreement >= 0.75
            and annotators_independent,
            "binary_on_off_diagnostic": {
                "matches": binary_matches,
                "agreement": binary_matches / len(shared),
                "cohen_kappa": _kappa(binary_confusion),
                "confusion_primary_by_secondary": {
                    f"{primary_value}|{secondary_value}": count
                    for (
                        primary_value,
                        secondary_value,
                    ), count in sorted(binary_confusion.items())
                },
                "not_a_substitute_for_the_frozen_exact_three_way_gate": True,
            },
        }
        formal_pass = (
            primary_gate["exact_three_way_gate_passed"]
            and primary_gate["uncertain_gate_passed"]
            and agreement >= 0.75
            and annotators_independent
        )
        result["formal_D2_gate_passed"] = formal_pass
        result["status"] = (
            "D2_MEASUREMENT_GATE_PASS"
            if formal_pass
            else "D2_MEASUREMENT_GATE_FAIL"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if secondary["valid"]:
        score = {
            "treatment": 1.0,
            "control": 0.0,
            "tie": 0.0,
            "uncertain": None,
        }
        blueprint_by_state = {
            str(row["d2_state_id"]): row
            for row in _rows(args.blueprint)
        }
        pair_measurements: list[dict[str, Any]] = []
        for state_id, blueprint in sorted(blueprint_by_state.items()):
            reference = blueprint.get("historical_pair_reference")
            if not isinstance(reference, dict):
                continue
            slot_id = str(reference["contrast_slot_id"])
            direction = str(historical[slot_id]["quality_verdict"])
            pair_measurements.append(
                {
                    "protocol": (
                        "pm-v1.5-d2-soft-expected-benefit-measurement-v1"
                    ),
                    "d2_state_id": state_id,
                    "component": blueprint["component"],
                    "state_id": blueprint["state_id"],
                    "user_id": blueprint["user_id"],
                    "split": blueprint["split"],
                    "pair_id": f"historical:{slot_id}",
                    "pair_origin": "historical_realized_pair",
                    "replicate_index": 0,
                    "reviewer_directions": [direction],
                    "reviewer_count": 1,
                    "pair_soft_benefit": score[direction],
                    "same_pair_reviewers_disagree": False,
                }
            )
        secondary_key_by_pair = {
            str(row["pair_id"]): row for row in secondary_key
        }
        for pair_id, key in sorted(primary_key_by_pair.items()):
            primary_direction = primary_directions_by_pair[pair_id]
            reviewer_directions = [primary_direction]
            if pair_id in secondary_direction_by_pair:
                reviewer_directions.append(
                    secondary_direction_by_pair[pair_id]
                )
            valid_scores = [
                score[direction]
                for direction in reviewer_directions
                if score[direction] is not None
            ]
            pair_measurements.append(
                {
                    "protocol": (
                        "pm-v1.5-d2-soft-expected-benefit-measurement-v1"
                    ),
                    "d2_state_id": key["d2_state_id"],
                    "component": key["component"],
                    "state_id": key["state_id"],
                    "user_id": key["user_id"],
                    "split": blueprint_by_state[
                        str(key["d2_state_id"])
                    ]["split"],
                    "pair_id": pair_id,
                    "pair_origin": "new_D2_independent_generation",
                    "replicate_index": key["replicate_index"],
                    "reviewer_directions": reviewer_directions,
                    "reviewer_count": len(reviewer_directions),
                    "pair_soft_benefit": (
                        sum(valid_scores) / len(valid_scores)
                        if valid_scores
                        else None
                    ),
                    "same_pair_reviewers_disagree": len(
                        set(reviewer_directions)
                    )
                    > 1,
                    "secondary_blind_item_id": (
                        secondary_key_by_pair[pair_id]["blind_item_id"]
                        if pair_id in secondary_key_by_pair
                        else None
                    ),
                }
            )
        pair_measurements.sort(
            key=lambda row: (
                str(row["d2_state_id"]),
                int(row["replicate_index"]),
                str(row["pair_id"]),
            )
        )
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pair_measurements:
            by_state[str(row["d2_state_id"])].append(row)
        soft_targets: list[dict[str, Any]] = []
        for state_id, blueprint in sorted(blueprint_by_state.items()):
            pairs = by_state[state_id]
            if len(pairs) != 3:
                raise RuntimeError(
                    f"{state_id}: expected three independent pairs, "
                    f"found {len(pairs)}"
                )
            pair_scores = [row["pair_soft_benefit"] for row in pairs]
            if any(value is None for value in pair_scores):
                state_soft_target = None
            else:
                state_soft_target = sum(pair_scores) / len(pair_scores)
            soft_targets.append(
                {
                    "protocol": (
                        "pm-v1.5-d2-soft-expected-benefit-measurement-v1"
                    ),
                    "d2_state_id": state_id,
                    "component": blueprint["component"],
                    "state_id": blueprint["state_id"],
                    "user_id": blueprint["user_id"],
                    "split": blueprint["split"],
                    "independent_generated_pairs": 3,
                    "reviewer_decisions": sum(
                        int(row["reviewer_count"]) for row in pairs
                    ),
                    "pair_soft_benefits": pair_scores,
                    "soft_expected_benefit": state_soft_target,
                    "not_a_hard_effect_label": True,
                }
            )
        pair_path = args.out_dir / "pair_soft_measurements.jsonl"
        target_path = args.out_dir / "state_soft_targets.jsonl"
        write_jsonl(pair_path, pair_measurements)
        write_jsonl(target_path, soft_targets)
        result["soft_target_recovery_outputs"] = {
            "protocol": (
                "pm-v1.5-d2-soft-expected-benefit-measurement-v1"
            ),
            "pair_rows": len(pair_measurements),
            "state_rows": len(soft_targets),
            "pairs_per_state": 3,
            "same_pair_reviewer_disagreements": sum(
                bool(row["same_pair_reviewers_disagree"])
                for row in pair_measurements
            ),
            "pair_soft_measurements_sha256": sha256_file(pair_path),
            "state_soft_targets_sha256": sha256_file(target_path),
            "effect_labels_created": False,
            "use": "pre-D3 diagnostic only",
        }
    write_json(args.out_dir / "analysis.json", result)
    print(
        {
            "status": result["status"],
            "primary_exact_reproduction": primary_gate[
                "exact_three_way_reproduction_rate"
            ],
            "primary_binary_reproduction": primary_gate[
                "binary_on_off_reproduction_rate_diagnostic"
            ],
            "uncertain_fraction": uncertain_fraction,
            "secondary_valid": secondary["valid"],
        }
    )


if __name__ == "__main__":
    main()
