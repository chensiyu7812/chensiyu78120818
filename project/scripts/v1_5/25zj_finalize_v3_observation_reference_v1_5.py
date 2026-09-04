#!/usr/bin/env python3
"""Merge 80 unchanged V1 labels with the 16-item V2 semantic delta."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-human-reference-v2"
COMPONENTS = ("MP", "MS", "ME", "RS")
FACTORS = (
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "specific_increment",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _derived(row: dict[str, Any]) -> str:
    values = [
        row["goal_function_fit"],
        row["boundary_burden_compatible"],
        row["specific_increment"],
    ]
    if row["component"] != "RS":
        values.append(row["owner_time_entity_valid"])
    if "no" in values:
        return "ineligible"
    if all(value == "yes" for value in values):
        return "eligible"
    return "uncertain"


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-labels", type=Path, required=True)
    parser.add_argument(
        "--delta-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_semantic_repair_delta_v2_candidate",
    )
    parser.add_argument(
        "--v2-review-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_review_candidate_v2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_human_reference_v2",
    )
    args = parser.parse_args()

    carry = {
        str(row["blind_item_id"]): row
        for row in _rows(args.delta_dir / "carry_forward_reference.jsonl")
    }
    delta = {str(row["blind_item_id"]): row for row in _rows(args.delta_labels)}
    expected_delta = {
        str(row["blind_item_id"])
        for row in __import__("json").loads(
            (args.delta_dir / "human_review_packet.json").read_text(encoding="utf-8")
        )["items"]
    }
    bindings = {
        str(row["blind_item_id"]): row
        for row in _rows(args.v2_review_dir / "private_binding.jsonl")
    }
    failures: list[str] = []
    if len(carry) != 80:
        failures.append(f"carry_forward_count_{len(carry)}_not_80")
    if len(delta) != 16 or set(delta) != expected_delta:
        failures.append("delta_membership_or_count")
    if set(carry) & set(delta):
        failures.append("carry_delta_overlap")
    if set(carry) | set(delta) != set(bindings):
        failures.append("merged_reference_not_equal_v2_packet")

    merged_input = {**carry, **delta}
    reference_rows: list[dict[str, Any]] = []
    for item_id in sorted(merged_input):
        label = merged_input[item_id]
        binding = bindings[item_id]
        if label.get("component") != binding.get("component"):
            failures.append(f"component_mismatch_{item_id}")
        if label.get("derived_eligibility") != _derived(label):
            failures.append(f"derived_eligibility_inconsistent_{item_id}")
        expected_owner = (
            {"structural_yes_not_rated"}
            if label["component"] == "RS"
            else {"yes", "no"}
        )
        if label.get("owner_time_entity_valid") not in expected_owner:
            failures.append(f"invalid_owner_{item_id}")
        if any(label.get(factor) not in {"yes", "no"} for factor in FACTORS[1:]):
            failures.append(f"invalid_factor_value_{item_id}")
        reference_rows.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": item_id,
                "state_id": binding["state_id"],
                "component": label["component"],
                "track": binding["track"],
                "counterfactual_group_id": binding["counterfactual_group_id"],
                "candidate_id": binding["candidate_id"],
                "candidate_text_sha256": binding["candidate_text_sha256"],
                "adjudicated_subtype": label["adjudicated_subtype"],
                "owner_time_entity_valid": label["owner_time_entity_valid"],
                "goal_function_fit": label["goal_function_fit"],
                "boundary_burden_compatible": label["boundary_burden_compatible"],
                "specific_increment": label["specific_increment"],
                "derived_eligibility": label["derived_eligibility"],
                "label_lineage": "V2_DELTA" if item_id in delta else "V1_UNCHANGED_CARRY_FORWARD",
                "step1_worth_opening_gold": None,
            }
        )

    shape: dict[str, Any] = {}
    fit_balance_pass = True
    confirmation_balance_pass = True
    for track in ("FACTOR_FIT", "ELIGIBILITY_CONFIRMATION"):
        shape[track] = {}
        for component in COMPONENTS:
            subset = [
                row
                for row in reference_rows
                if row["track"] == track and row["component"] == component
            ]
            factor_counts = {}
            for factor in FACTORS:
                if component == "RS" and factor == "owner_time_entity_valid":
                    continue
                factor_counts[factor] = dict(Counter(row[factor] for row in subset))
                if track == "FACTOR_FIT" and factor_counts[factor] != {"no": 8, "yes": 8}:
                    fit_balance_pass = False
            eligibility_counts = dict(Counter(row["derived_eligibility"] for row in subset))
            if track == "ELIGIBILITY_CONFIRMATION" and eligibility_counts != {
                "eligible": 4,
                "ineligible": 4,
            }:
                confirmation_balance_pass = False
            shape[track][component] = {
                "n": len(subset),
                "factors": factor_counts,
                "derived_eligibility": eligibility_counts,
            }

    if not fit_balance_pass:
        failures.append("factor_fit_not_exact_8_8")
    if not confirmation_balance_pass:
        failures.append("confirmation_not_exact_4_4")
    status = "PASS_REFERENCE_FROZEN" if not failures else "FAIL_DO_NOT_TRAIN"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.out_dir / "observation_reference.jsonl"
    write_jsonl(reference_path, reference_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "rows": len(reference_rows),
        "carry_forward_rows": len(carry),
        "delta_rows": len(delta),
        "per_component": dict(Counter(row["component"] for row in reference_rows)),
        "label_shape": shape,
        "factor_fit_exact_8_8_pass": fit_balance_pass,
        "eligibility_confirmation_exact_4_4_pass": confirmation_balance_pass,
        "uncertain_rows": sum(
            any(row[factor] == "uncertain" for factor in FACTORS)
            for row in reference_rows
        ),
        "construction_intent_used_as_gold": False,
        "step1_worth_opening_labels_created": 0,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "failures": failures,
        "delta_labels_sha256": sha256_file(args.delta_labels),
        "carry_forward_sha256": sha256_file(
            args.delta_dir / "carry_forward_reference.jsonl"
        ),
        "v2_binding_sha256": sha256_file(args.v2_review_dir / "private_binding.jsonl"),
        "reference_sha256": sha256_file(reference_path),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "next_step_if_pass": "RUN_PRE_FROZEN_FACTOR_BAKEOFF_ON_FACTOR_FIT_THEN_ONE_SHOT_ELIGIBILITY_CONFIRMATION",
    }
    write_json(args.out_dir / "reference_report.json", report)
    print(
        {
            "status": status,
            "rows": len(reference_rows),
            "fit_8_8": fit_balance_pass,
            "confirmation_4_4": confirmation_balance_pass,
            "out": str(reference_path),
        }
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
