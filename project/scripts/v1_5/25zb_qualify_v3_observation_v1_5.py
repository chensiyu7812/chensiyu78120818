#!/usr/bin/env python3
"""Qualify the transparent V3 Observation against frozen Eligibility factors."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping

from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.v1_5_observation_contract import transparent_semantic_observation


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-transparent-observation-qualification-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
AUDITED_FACTORS = {
    "owner_time_entity_valid": "owner_time_valid",
    "goal_function_fit": "goal_function_fit",
    "boundary_burden_compatible": "boundary_burden_fit",
    "specific_increment": "specific_nonredundant_increment",
}
THRESHOLDS = {
    "balanced_accuracy_min": 0.70,
    "recall_min": 0.60,
    "specificity_min": 0.60,
    "minimal_counterfactual_direction_min": 0.75,
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (
        (recall + specificity) / 2
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


def _qualified(metric: Mapping[str, Any]) -> bool:
    return bool(
        metric["balanced_accuracy"] is not None
        and metric["balanced_accuracy"] >= THRESHOLDS["balanced_accuracy_min"]
        and metric["recall"] >= THRESHOLDS["recall_min"]
        and metric["specificity"] >= THRESHOLDS["specificity_min"]
    )


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 P2 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_h_eligibility_reference_v3/eligibility_reference.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v3/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_qualification_v1/report.json",
    )
    args = parser.parse_args()

    reference = _rows(args.reference)
    candidates = {str(row["state_id"]): row for row in _rows(args.candidates)}
    blueprint = {str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)}
    if len(reference) != 128 or len({row["state_id"] for row in reference}) != 128:
        raise RuntimeError("Observation qualification requires the frozen 128-row reference")

    factor_rows: list[dict[str, Any]] = []
    for label in reference:
        state_id = str(label["state_id"])
        candidate_row = candidates[state_id]
        exact = candidate_row["exact_rank1_candidate"]
        if (
            exact["candidate_id"] != label["candidate_id"]
            or exact["candidate_text_sha256"] != label["candidate_text_sha256"]
        ):
            raise RuntimeError(f"reference/candidate binding mismatch: {state_id}")
        observation = transparent_semantic_observation(
            state_id=state_id,
            component=str(label["component"]),
            current_user_text=str(candidate_row["current_user_text"]),
            visible_dialogue=candidate_row["visible_dialogue"],
            candidate_present=bool(exact["candidate_present"]),
            candidate_text=str(exact["candidate_text"]),
            candidate_subtype=str(exact["compiler_subtype_hint"]),
        )
        for factor, gold_field in AUDITED_FACTORS.items():
            factor_rows.append(
                {
                    "state_id": state_id,
                    "component": label["component"],
                    "factor": factor,
                    "gold": int(label[gold_field] == "yes"),
                    "score": float(observation.factor_scores[factor]),
                    "prediction": int(observation.factor_scores[factor] >= 0.5),
                    "counterfactual_group_id": blueprint[state_id]["counterfactual_group_id"],
                    "logic_family": blueprint[state_id]["logic_family"],
                }
            )

    global_metrics: dict[str, Any] = {}
    per_component: dict[str, Any] = {component: {} for component in COMPONENTS}
    failures: list[str] = []
    for factor in AUDITED_FACTORS:
        selected = [row for row in factor_rows if row["factor"] == factor]
        metric = _metrics(
            [row["gold"] for row in selected],
            [row["prediction"] for row in selected],
        )
        metric["status"] = "PASS" if _qualified(metric) else "FAIL"
        global_metrics[factor] = metric
        if metric["status"] == "FAIL":
            failures.append(f"GLOBAL_{factor}")
        for component in COMPONENTS:
            subset = [row for row in selected if row["component"] == component]
            diagnostic = _metrics(
                [row["gold"] for row in subset],
                [row["prediction"] for row in subset],
            )
            diagnostic["status"] = (
                "SINGLE_CLASS_DIAGNOSTIC"
                if diagnostic["balanced_accuracy"] is None
                else "PASS_DIAGNOSTIC"
                if _qualified(diagnostic)
                else "FAIL_DIAGNOSTIC"
            )
            per_component[component][factor] = diagnostic

    direction: dict[str, Any] = {}
    for factor in AUDITED_FACTORS:
        selected = [row for row in factor_rows if row["factor"] == factor]
        grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[(row["component"], row["counterfactual_group_id"])].append(row)
        comparisons: list[dict[str, Any]] = []
        for (component, group_id), rows in grouped.items():
            positives = [row for row in rows if row["gold"] == 1]
            negatives = [row for row in rows if row["gold"] == 0]
            for positive in positives:
                for negative in negatives:
                    comparisons.append(
                        {
                            "component": component,
                            "counterfactual_group_id": group_id,
                            "positive_family": positive["logic_family"],
                            "negative_family": negative["logic_family"],
                            "correct": positive["score"] > negative["score"],
                        }
                    )
        correct = sum(row["correct"] for row in comparisons)
        rate = correct / len(comparisons) if comparisons else None
        family_failures = Counter(
            f"{row['positive_family']} -> {row['negative_family']}"
            for row in comparisons
            if not row["correct"]
        )
        direction[factor] = {
            "n": len(comparisons),
            "correct": correct,
            "rate": rate,
            "status": (
                "PASS"
                if rate is not None
                and rate >= THRESHOLDS["minimal_counterfactual_direction_min"]
                else "FAIL"
            ),
            "failed_family_pairs": dict(family_failures),
        }
        if direction[factor]["status"] == "FAIL":
            failures.append(f"COUNTERFACTUAL_{factor}")

    composite_labels: list[int] = []
    composite_predictions: list[int] = []
    for label in reference:
        rows = [row for row in factor_rows if row["state_id"] == label["state_id"]]
        composite_labels.append(int(label["derived_eligibility"] == "eligible"))
        composite_predictions.append(int(all(row["prediction"] == 1 for row in rows)))

    report = {
        "protocol": PROTOCOL,
        "status": "FAIL_TRANSPARENT_OBSERVATION_REPAIR_REQUIRED" if failures else "PASS",
        "role": "OBSERVATION_QUALIFICATION_NOT_STEP1_GOLD_NOT_STEP2_EVALUATION",
        "thresholds": THRESHOLDS,
        "audited_factors": AUDITED_FACTORS,
        "auxiliary_currently_nonredundant_not_separately_supervised": True,
        "global_factor_metrics": global_metrics,
        "per_component_diagnostics": per_component,
        "minimal_counterfactual_direction": direction,
        "composite_eligibility_diagnostic": _metrics(composite_labels, composite_predictions),
        "failures": failures,
        "reference_sha256": sha256_file(args.reference),
        "candidate_rows_sha256": sha256_file(args.candidates),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "bge_used": False,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "step1_training_authorized": False,
    }
    write_json(args.out, report)
    print({"status": report["status"], "failures": failures, "out": str(args.out)})


if __name__ == "__main__":
    main()
