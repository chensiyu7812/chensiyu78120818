#!/usr/bin/env python3
"""Audit whether controlled memory-use labels are recoverable from construction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from metacom_pm.io import iter_jsonl, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-controlled-memory-routing-shortcut-audit-v1"
COMPONENTS = ("MP", "MS", "ME")
SPLITS = ("train", "calibration", "internal_test")
FEATURES = (
    "query_similarity_mean",
    "estimated_tokens",
    "min_age_sessions",
    "max_age_sessions",
    "current_user_word_count",
)
MATERIAL_SHORTCUT_BALANCED_ACCURACY = 0.80


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _feature(state: dict[str, Any], component: str, name: str) -> float:
    if name == "current_user_word_count":
        return float(len(str(state["current_user_text"]).split()))
    return float(state["inventory"][component][name])


def _best_threshold(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    values = sorted(set(x.tolist()))
    thresholds = [values[0] - 1e-9, values[-1] + 1e-9]
    thresholds.extend(
        (left + right) / 2 for left, right in zip(values, values[1:])
    )
    candidates: list[tuple[float, str, float]] = []
    for threshold in thresholds:
        candidates.append(
            (
                float(balanced_accuracy_score(y, x >= threshold)),
                "ge",
                float(threshold),
            )
        )
        candidates.append(
            (
                float(balanced_accuracy_score(y, x <= threshold)),
                "le",
                float(threshold),
            )
        )
    score, direction, threshold = max(candidates)
    return {
        "balanced_accuracy": score,
        "direction": direction,
        "threshold": threshold,
        "positive_mean": float(x[y == 1].mean()),
        "negative_mean": float(x[y == 0].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_controlled_memory_routing_heads_v1"
        / "posthoc_shortcut_audit.json",
    )
    args = parser.parse_args()

    states = _rows(args.data_dir / "pm_v2_states.jsonl")
    contexts = {
        str(row["state_id"]): row
        for row in _rows(args.data_dir / "evaluator_contexts.jsonl")
    }
    reports: dict[str, Any] = {}
    material_findings: list[dict[str, Any]] = []
    for component in COMPONENTS:
        reports[component] = {}
        for split in SPLITS:
            split_states = [
                row for row in states if str(row["split"]) == split
            ]
            target = np.asarray(
                [
                    int(
                        component
                        in contexts[str(row["state_id"])][
                            "needed_memory_sources"
                        ]
                    )
                    for row in split_states
                ],
                dtype=int,
            )
            feature_reports: dict[str, Any] = {}
            for name in FEATURES:
                values = np.asarray(
                    [_feature(row, component, name) for row in split_states],
                    dtype=float,
                )
                result = _best_threshold(values, target)
                feature_reports[name] = result
                if (
                    split == "internal_test"
                    and result["balanced_accuracy"]
                    >= MATERIAL_SHORTCUT_BALANCED_ACCURACY
                ):
                    material_findings.append(
                        {
                            "component": component,
                            "feature": name,
                            **result,
                        }
                    )
            reports[component][split] = {
                "rows": len(split_states),
                "positive": int(target.sum()),
                "features": feature_reports,
            }

    report = {
        "protocol": PROTOCOL,
        "status": (
            "MATERIAL_CONSTRUCTION_SHORTCUT_RISK"
            if material_findings
            else "NO_MATERIAL_UNIVARIATE_SHORTCUT_FOUND"
        ),
        "analysis_role": (
            "Post-hoc data-validity audit after the frozen internal evaluation; "
            "not a new model-selection or threshold-tuning result."
        ),
        "outcome_labels_read": False,
        "target": (
            "Synthetic evaluator-side needed_memory_sources, not response "
            "quality or realized treatment benefit."
        ),
        "material_shortcut_balanced_accuracy": (
            MATERIAL_SHORTCUT_BALANCED_ACCURACY
        ),
        "reports": reports,
        "material_internal_findings": material_findings,
        "interpretation": (
            "A source-level construction statistic alone recovers the synthetic "
            "component-use label out of user. This is not direct answer-field "
            "leakage, but it can arise from helpful/decoy template construction "
            "rather than a deployable semantic decision rule. The controlled "
            "routing result is therefore diagnostic evidence of pipeline "
            "learnability, not confirmatory evidence that real memory routing "
            "has been learned."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
