#!/usr/bin/env python3
"""Qualify the factor-independence Observation implementation repair."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import numpy as np

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_observation_contract import (
    transparent_semantic_observation_v2,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-independence-repair-execution-v2"
FACTORS = (
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "specific_increment",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_human_reference_v2/observation_reference.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_orthogonal_exact_rank1_v2/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_v3_observation_orthogonal_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/v3_observation_independence_repair_v1.json",
    )
    parser.add_argument(
        "--failed-primary",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_orthogonal_bakeoff_v2/qualification_report.json",
    )
    parser.add_argument(
        "--failed-nli",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_nli_recovery_v1/qualification_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_independence_repair_v2",
    )
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    observed_hashes = {
        "reference_sha256": sha256_file(args.reference),
        "candidate_rows_sha256": sha256_file(args.candidates),
        "blueprint_sha256": sha256_file(args.blueprint),
        "failed_primary_bakeoff_sha256": sha256_file(args.failed_primary),
        "failed_nli_recovery_sha256": sha256_file(args.failed_nli),
    }
    for key, observed in observed_hashes.items():
        if observed != contract[key]:
            raise RuntimeError(f"frozen hash mismatch for {key}")

    reference = _rows(args.reference)
    candidates = {str(row["state_id"]): row for row in _rows(args.candidates)}
    blueprint = {str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)}
    if len(reference) != 96 or len(candidates) != 96 or len(blueprint) != 96:
        raise RuntimeError("independence repair requires 96 aligned rows")
    records = []
    for label in reference:
        state_id = str(label["state_id"])
        candidate_row = candidates[state_id]
        exact = candidate_row["exact_rank1_candidate"]
        if exact["candidate_text_sha256"] != label["candidate_text_sha256"]:
            raise RuntimeError(f"candidate/reference hash mismatch: {state_id}")
        construction = blueprint[state_id]
        observation = transparent_semantic_observation_v2(
            state_id=state_id,
            component=str(label["component"]),
            current_user_text=str(candidate_row["current_user_text"]),
            visible_dialogue=candidate_row.get("visible_dialogue", []),
            candidate_present=True,
            candidate_text=str(exact["candidate_text"]),
            candidate_subtype=str(exact["compiler_subtype_hint"]),
        )
        records.append(
            {
                "state_id": state_id,
                "component": str(label["component"]),
                "track": str(label["track"]),
                "counterfactual_group_id": str(label["counterfactual_group_id"]),
                "topic_family": str(construction["topic_family"]),
                "history_scale": str(construction["history_scale"]),
                "prefix_bucket": str(construction["prefix_family"]),
                "semantic_families": construction["private_semantic_surface_family"],
                "state_text": str(candidate_row["current_user_text"]),
                "candidate_text": str(exact["candidate_text"]),
                "factor_scores": dict(observation.factor_scores),
                "human": label,
            }
        )

    helpers = runpy.run_path(
        str(ROOT / "scripts/v1_5/25zl_run_v3_observation_nli_recovery_v1_5.py")
    )
    legacy = runpy.run_path(
        str(ROOT / "scripts/v1_5/25zc_run_v3_observation_factor_bakeoff_v1_5.py")
    )
    fit = [row for row in records if row["track"] == "FACTOR_FIT"]
    confirmation = [row for row in records if row["track"] == "ELIGIBILITY_CONFIRMATION"]
    fit_reports = {}
    nuisance = {}
    fit_prediction_rows = []
    for factor in FACTORS:
        applicable = [
            row
            for row in fit
            if not (factor == "owner_time_entity_valid" and row["component"] == "RS")
        ]
        labels = np.asarray([int(row["human"][factor] == "yes") for row in applicable])
        probabilities = np.asarray(
            [float(row["factor_scores"][factor]) for row in applicable]
        )
        groups = np.asarray([row["counterfactual_group_id"] for row in applicable])
        nuisance[factor] = legacy["_nuisance_probe"](applicable, labels, groups)
        fit_reports[factor] = helpers["_factor_report"](
            applicable, factor, probabilities, contract
        )
        for row, label, probability in zip(
            applicable, labels, probabilities, strict=True
        ):
            fit_prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "state_id": row["state_id"],
                    "component": row["component"],
                    "factor": factor,
                    "gold": int(label),
                    "transparent_v2_probability": float(probability),
                }
            )
    nuisance_pass = all(
        row["balanced_accuracy"]
        < contract["gates"][
            "raw_surface_nuisance_probe_balanced_accuracy_max_exclusive"
        ]
        for row in nuisance.values()
    )
    fit_pass = nuisance_pass and all(
        row["status"] == "PASS" for row in fit_reports.values()
    )

    confirmation_report = None
    confirmation_prediction_rows = []
    confirmation_pass = False
    if fit_pass:
        factor_probabilities: dict[str, dict[str, float]] = defaultdict(dict)
        factor_reports = {}
        for factor in FACTORS:
            applicable = [
                row
                for row in confirmation
                if not (
                    factor == "owner_time_entity_valid" and row["component"] == "RS"
                )
            ]
            labels = np.asarray(
                [int(row["human"][factor] == "yes") for row in applicable]
            )
            probabilities = np.asarray(
                [float(row["factor_scores"][factor]) for row in applicable]
            )
            factor_reports[factor] = helpers["_metric"](labels, probabilities)
            for row, probability in zip(applicable, probabilities, strict=True):
                factor_probabilities[factor][row["state_id"]] = float(probability)

        global_gold = []
        global_probability = []
        by_component: defaultdict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"gold": [], "probability": []}
        )
        for row in confirmation:
            values = [
                factor_probabilities[factor][row["state_id"]]
                for factor in FACTORS
                if not (
                    factor == "owner_time_entity_valid" and row["component"] == "RS"
                )
            ]
            probability = float(min(values))
            gold = int(row["human"]["derived_eligibility"] == "eligible")
            global_gold.append(gold)
            global_probability.append(probability)
            by_component[row["component"]]["gold"].append(gold)
            by_component[row["component"]]["probability"].append(probability)
            confirmation_prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "state_id": row["state_id"],
                    "component": row["component"],
                    "gold_eligibility": gold,
                    "composed_probability_min_gate": probability,
                    "predicted_eligibility": int(probability >= 0.5),
                }
            )
        global_metric = helpers["_metric"](
            np.asarray(global_gold), np.asarray(global_probability)
        )
        gates = contract["gates"]
        confirmation_pass = bool(
            all(
                row["balanced_accuracy"]
                >= gates["confirmation_each_factor_balanced_accuracy_min"]
                for row in factor_reports.values()
            )
            and global_metric["balanced_accuracy"]
            >= gates["confirmation_composite_balanced_accuracy_min"]
            and global_metric["recall"]
            >= gates["confirmation_composite_recall_min"]
            and global_metric["specificity"]
            >= gates["confirmation_composite_specificity_min"]
        )
        confirmation_report = {
            "status": "PASS" if confirmation_pass else "FAIL",
            "one_shot_consumed": True,
            "used_for_rule_or_threshold_selection": False,
            "per_factor": factor_reports,
            "composite_global": global_metric,
            "composite_per_component": {
                component: helpers["_metric"](
                    np.asarray(values["gold"]),
                    np.asarray(values["probability"]),
                )
                for component, values in sorted(by_component.items())
            },
        }

    status = (
        "FAIL_FIT_CONFIRMATION_NOT_CONSUMED"
        if not fit_pass
        else (
            "PASS_FIT_AND_ONE_SHOT_CONFIRMATION"
            if confirmation_pass
            else "PASS_FIT_FAIL_ONE_SHOT_CONFIRMATION"
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit_path = args.out_dir / "fit_predictions.jsonl"
    confirmation_path = args.out_dir / "confirmation_predictions.jsonl"
    write_jsonl(fit_path, fit_prediction_rows)
    write_jsonl(confirmation_path, confirmation_prediction_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "candidate": contract["candidate"]["id"],
        "fit_rows": len(fit),
        "confirmation_rows": len(confirmation),
        "fit_pass": fit_pass,
        "fit_factor_reports": fit_reports,
        "raw_surface_nuisance_probe": nuisance,
        "raw_surface_nuisance_probe_pass": nuisance_pass,
        "one_shot_confirmation": confirmation_report,
        "step1_training_authorized": False,
        "step2_qualification_authorized": confirmation_pass,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "confirmation_read_before_fit_decision": False,
        "factor_scores_independently_computed": True,
        "frozen_input_hashes": observed_hashes,
        "contract_sha256": sha256_file(args.contract),
        "fit_predictions_sha256": sha256_file(fit_path),
        "confirmation_predictions_sha256": sha256_file(confirmation_path),
        "component_counts": dict(Counter(row["component"] for row in records)),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    write_json(args.out_dir / "qualification_report.json", report)
    print(
        {
            "status": status,
            "fit_pass": fit_pass,
            "fit_factor_statuses": {
                factor: value["status"] for factor, value in fit_reports.items()
            },
            "confirmation_composite": (
                confirmation_report["composite_global"]
                if confirmation_report is not None
                else None
            ),
            "out": str(args.out_dir / "qualification_report.json"),
        }
    )


if __name__ == "__main__":
    main()
