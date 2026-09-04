#!/usr/bin/env python3
"""Development-only evaluation of the P1R transparent observation builder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, write_json
from metacom_pm.v1_5_observation_contract import (
    FACTOR_NAMES,
    factor_gold_from_evidence_codes,
    transparent_semantic_observation,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-transparent-observation-development-evaluation-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (
        0.5 * (recall + specificity)
        if recall is not None and specificity is not None
        else None
    )
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
        "balanced_accuracy": balanced,
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
        "--candidate-rows",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_p2_h1_v2_exact_rank1_v10/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transparent_observation_development_v1/report.json",
    )
    args = parser.parse_args()

    annotations = {row["blind_state_id"]: row for row in iter_jsonl(args.annotations)}
    bindings = {row["blind_state_id"]: row for row in iter_jsonl(args.bindings)}
    candidates = {row["state_id"]: row for row in iter_jsonl(args.candidate_rows)}
    buckets: dict[tuple[str, str], tuple[list[int], list[int]]] = {
        (component, factor): ([], [])
        for component in COMPONENTS
        for factor in FACTOR_NAMES
    }

    for blind_id, annotation in annotations.items():
        row = candidates[bindings[blind_id]["state_id"]]
        for component in COMPONENTS:
            decision = annotation["component_decisions"][component]
            if decision["adjudicated_subtype"] == "CANDIDATE_ABSENT":
                continue
            surface = row["candidate_surfaces"][component]
            observation = transparent_semantic_observation(
                state_id=row["state_id"],
                component=component,
                current_user_text=row["current_user_text"],
                visible_dialogue=row["visible_dialogue"],
                candidate_present=surface["candidate_present"],
                candidate_text=surface["candidate_text"],
                candidate_subtype=surface["compiler_subtype_hint"],
            )
            gold = factor_gold_from_evidence_codes(decision["evidence_codes"])
            for factor in FACTOR_NAMES:
                if gold[factor] is None:
                    continue
                labels, predictions = buckets[(component, factor)]
                labels.append(int(gold[factor]))
                predictions.append(int(observation.factor_scores[factor] >= 0.5))

    results: dict[str, Any] = {}
    failures: list[str] = []
    for component in COMPONENTS:
        results[component] = {}
        for factor in FACTOR_NAMES:
            labels, predictions = buckets[(component, factor)]
            metric = _metrics(labels, predictions)
            identifiable = metric["positive"] > 0 and metric["negative"] > 0
            qualified = bool(
                identifiable
                and metric["balanced_accuracy"] is not None
                and metric["balanced_accuracy"] >= 0.70
                and metric["recall"] >= 0.60
                and metric["specificity"] >= 0.60
            )
            if not identifiable:
                metric["status"] = "UNQUALIFIABLE_SINGLE_CLASS_OLD_DEVELOPMENT"
            else:
                metric["status"] = (
                    "PASS_DEVELOPMENT_THRESHOLD"
                    if qualified
                    else "FAIL_DEVELOPMENT_THRESHOLD"
                )
            if identifiable and not qualified:
                failures.append(f"{component}_{factor}")
            results[component][factor] = metric

    report = {
        "protocol": PROTOCOL,
        "status": "DEVELOPMENT_PASS" if not failures else "DEVELOPMENT_REPAIR_REQUIRED",
        "role": "DEVELOPMENT_ONLY_OLD_H1_V2_NOT_CONFIRMATION",
        "results": results,
        "failures": failures,
        "fresh_h1r_qualification_required": True,
        "bge_was_used": False,
        "human_final_decision_was_model_input": False,
        "construction_intent_was_model_input": False,
        "external_outcome_was_read": False,
    }
    write_json(args.out, report)
    print({"status": report["status"], "failures": failures, "out": str(args.out)})


if __name__ == "__main__":
    main()
