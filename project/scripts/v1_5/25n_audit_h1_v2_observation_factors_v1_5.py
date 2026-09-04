#!/usr/bin/env python3
"""Audit old H1-v2 runtime features against factor-level evidence-code gold."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, write_json
from metacom_pm.v1_5_observation_contract import (
    FACTOR_NAMES,
    factor_gold_from_evidence_codes,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h1-v2-observation-factor-audit-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")

# This table describes the old feature surface.  ``None`` is intentional: it
# exposes a missing runtime observation rather than inventing a proxy after
# seeing the labels.
OLD_FACTOR_SCORES: dict[str, dict[str, str | tuple[str, ...] | None]] = {
    "MP": {
        "owner_time_entity_valid": "candidate_profile_entity_scope_fit",
        "goal_function_fit": (
            "candidate_preference_scope_fit",
            "candidate_profile_relevance",
        ),
        "boundary_burden_compatible": None,
        "currently_nonredundant": "candidate_incremental_information",
        "specific_increment": "candidate_incremental_information",
    },
    "MS": {
        "owner_time_entity_valid": None,
        "goal_function_fit": "candidate_current_goal_fit",
        "boundary_burden_compatible": None,
        "currently_nonredundant": "candidate_incremental_information",
        "specific_increment": (
            "candidate_prior_outcome_or_distinction",
            "candidate_specific_issue_or_distinction",
        ),
    },
    "ME": {
        "owner_time_entity_valid": None,
        "goal_function_fit": "candidate_current_goal_fit",
        "boundary_burden_compatible": None,
        "currently_nonredundant": "candidate_incremental_information",
        "specific_increment": (
            "candidate_contains_action",
            "candidate_contains_result",
            "candidate_contains_mechanism",
        ),
    },
    "RS": {
        "owner_time_entity_valid": None,
        "goal_function_fit": "candidate_goal_fit",
        "boundary_burden_compatible": (
            "candidate_boundary_fit",
            "candidate_burden_fit",
        ),
        "currently_nonredundant": "candidate_nonredundancy",
        "specific_increment": "candidate_nonredundancy",
    },
}


def _score(features: dict[str, Any], spec: str | tuple[str, ...]) -> float:
    if isinstance(spec, str):
        return float(features[spec])
    # A tuple means all named requirements matter; min is transparent and
    # prevents one strong field from hiding a failed companion requirement.
    return min(float(features[name]) for name in spec)


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "n": len(labels),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_final_h1_v2_annotation_audit_v1/primary_annotations.jsonl",
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_v2_candidate/private_binding.jsonl",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_p2_h1_v2_exact_rank1_v10/model_feature_rows.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_v2_observation_factor_audit_v1/audit_report.json",
    )
    args = parser.parse_args()

    annotations = {row["blind_state_id"]: row for row in iter_jsonl(args.annotations)}
    bindings = {row["blind_state_id"]: row for row in iter_jsonl(args.bindings)}
    features = {
        (row["state_id"], row["component"]): row["model_features"]
        for row in iter_jsonl(args.features)
    }
    collected: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    coverage: defaultdict[str, dict[str, int]] = defaultdict(dict)
    missing_specs: list[dict[str, str]] = []

    for component in COMPONENTS:
        for factor in FACTOR_NAMES:
            spec = OLD_FACTOR_SCORES[component][factor]
            if spec is None:
                missing_specs.append({"component": component, "factor": factor})
                continue
            labels: list[int] = []
            predictions: list[int] = []
            for blind_id, annotation in annotations.items():
                decision = annotation["component_decisions"][component]
                if decision["adjudicated_subtype"] == "CANDIDATE_ABSENT":
                    continue
                gold = factor_gold_from_evidence_codes(decision["evidence_codes"])[factor]
                if gold is None:
                    continue
                state_id = bindings[blind_id]["state_id"]
                value = _score(features[(state_id, component)], spec)
                labels.append(gold)
                predictions.append(int(value >= 0.5))
            collected[(component, factor)] = (labels, predictions)
            coverage[component][factor] = len(labels)

    results: dict[str, Any] = {}
    failures: list[str] = []
    for component in COMPONENTS:
        results[component] = {}
        for factor in FACTOR_NAMES:
            key = (component, factor)
            if key not in collected:
                results[component][factor] = {
                    "status": "MISSING_RUNTIME_FACTOR",
                    "n": 0,
                }
                failures.append(f"{component}_{factor}_missing")
                continue
            labels, predictions = collected[key]
            metric = _metrics(labels, predictions)
            metric["status"] = (
                "PASS_DEVELOPMENT_THRESHOLD"
                if metric["balanced_accuracy"] >= 0.70
                and metric["recall"] >= 0.60
                and metric["specificity"] >= 0.60
                else "FAIL_DEVELOPMENT_THRESHOLD"
            )
            if metric["status"].startswith("FAIL"):
                failures.append(f"{component}_{factor}_below_gate")
            results[component][factor] = metric

    report = {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "P1R_REPAIR_REQUIRED",
        "role": "DEVELOPMENT_DIAGNOSTIC_NOT_MODEL_QUALIFICATION",
        "thresholds": {
            "balanced_accuracy_min": 0.70,
            "recall_min": 0.60,
            "specificity_min": 0.60,
        },
        "gold_source": "H1 evidence codes; missing factor evidence remains unknown, not negative",
        "old_feature_mapping_frozen_before_metric_computation": OLD_FACTOR_SCORES,
        "results": results,
        "missing_runtime_factor_specs": missing_specs,
        "failures": failures,
        "conclusion": (
            "Old routing features do not constitute a qualified semantic-observation layer. "
            "Repair the visible-input factor builder before building H1R or fitting router heads."
        ),
    }
    write_json(args.out, report)
    print({"status": report["status"], "failures": len(failures), "out": str(args.out)})


if __name__ == "__main__":
    main()

