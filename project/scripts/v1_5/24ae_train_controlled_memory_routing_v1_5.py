#!/usr/bin/env python3
"""Train controlled MP/MS/ME use heads and evaluate frozen internal users."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-controlled-memory-routing-heads-v1"
COMPONENTS = ("MP", "MS", "ME")
FEATURE_NAMES = (
    "component_query_similarity",
    "component_estimated_tokens",
    "component_min_age_sessions",
    "component_max_age_sessions",
    "current_user_token_estimate",
    "visible_history_turn_count",
)
C_GRID = (0.01, 0.1, 1.0, 10.0)
THRESHOLD = 0.5
SEED = 20260730


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    class_weight=None,
                    random_state=SEED,
                    max_iter=2000,
                ),
            ),
        ]
    )


def _features(state: dict[str, Any], component: str) -> list[float]:
    inventory = dict(state["inventory"][component])
    return [
        float(inventory["query_similarity_mean"]),
        float(inventory["estimated_tokens"]),
        float(inventory["min_age_sessions"]),
        float(inventory["max_age_sessions"]),
        float(len(str(state["current_user_text"]).split())),
        float(len(state["current_session_history"])),
    ]


def _select_c(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, Any]]:
    fold_count = min(6, len(set(groups.tolist())))
    cv = GroupKFold(n_splits=fold_count)
    fold_losses: dict[float, list[float]] = {value: [] for value in C_GRID}
    for train_index, test_index in cv.split(x, y, groups):
        for c_value in C_GRID:
            probability = _pipeline(c_value).fit(
                x[train_index], y[train_index]
            ).predict_proba(x[test_index])[:, 1]
            fold_losses[c_value].append(
                float(log_loss(y[test_index], probability, labels=[0, 1]))
            )
    means = {
        value: float(np.mean(losses))
        for value, losses in fold_losses.items()
    }
    standard_errors = {
        value: float(np.std(losses, ddof=1) / np.sqrt(len(losses)))
        for value, losses in fold_losses.items()
    }
    minimum_c = min(C_GRID, key=lambda value: means[value])
    limit = means[minimum_c] + standard_errors[minimum_c]
    selected = min(value for value in C_GRID if means[value] <= limit)
    return selected, {
        "folds": fold_count,
        "mean_fold_log_loss_by_C": {
            str(value): means[value] for value in C_GRID
        },
        "standard_error_by_C": {
            str(value): standard_errors[value] for value in C_GRID
        },
        "minimum_mean_C": minimum_c,
        "one_standard_error_limit": limit,
        "selected_C": selected,
    }


def _select_similarity_rule(
    similarity: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    values = sorted(set(similarity.tolist()))
    candidates = [values[0] - 1e-9, values[-1] + 1e-9]
    candidates.extend(
        (left + right) / 2 for left, right in zip(values, values[1:])
    )
    scored = [
        (
            float(balanced_accuracy_score(y, similarity >= threshold)),
            float(threshold),
        )
        for threshold in candidates
    ]
    best_score = max(score for score, _ in scored)
    thresholds = [
        threshold for score, threshold in scored if score == best_score
    ]
    return max(thresholds), best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/fast_controlled_memory_routing_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_controlled_memory_routing_heads_v1",
    )
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (
        contract["status"]
        != "FROZEN_BEFORE_INTERNAL_HEAD_EVALUATION_AFTER_DEVELOPMENT_DIAGNOSTIC"
    ):
        raise RuntimeError("controlled-routing contract is not frozen")
    states = {
        str(row["state_id"]): row
        for row in _rows(args.data_dir / "pm_v2_states.jsonl")
    }
    contexts = _rows(args.data_dir / "evaluator_contexts.jsonl")
    if len(states) != 468 or len(contexts) != 468:
        raise RuntimeError("expected the repaired 468-state controlled corpus")
    if set(states) != {str(row["state_id"]) for row in contexts}:
        raise RuntimeError("state and evaluator-context identities differ")

    development_contexts = [
        row
        for row in contexts
        if states[str(row["state_id"])]["split"] in {"train", "calibration"}
    ]
    internal_contexts = [
        row
        for row in contexts
        if states[str(row["state_id"])]["split"] == "internal_test"
    ]
    development_users = {
        str(states[str(row["state_id"])]["user_id"])
        for row in development_contexts
    }
    internal_users = {
        str(states[str(row["state_id"])]["user_id"])
        for row in internal_contexts
    }
    if len(development_users) != 36 or len(internal_users) != 16:
        raise RuntimeError("unexpected development/internal user counts")
    if development_users & internal_users:
        raise RuntimeError("development and internal users overlap")

    reports: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for component in COMPONENTS:
        x_dev = np.asarray(
            [
                _features(states[str(row["state_id"])], component)
                for row in development_contexts
            ],
            dtype=float,
        )
        y_dev = np.asarray(
            [
                int(component in row["needed_memory_sources"])
                for row in development_contexts
            ],
            dtype=int,
        )
        groups = np.asarray(
            [
                str(states[str(row["state_id"])]["user_id"])
                for row in development_contexts
            ]
        )
        x_internal = np.asarray(
            [
                _features(states[str(row["state_id"])], component)
                for row in internal_contexts
            ],
            dtype=float,
        )
        y_internal = np.asarray(
            [
                int(component in row["needed_memory_sources"])
                for row in internal_contexts
            ],
            dtype=int,
        )
        selected_c, selection = _select_c(x_dev, y_dev, groups)
        model = _pipeline(selected_c).fit(x_dev, y_dev)
        probability = model.predict_proba(x_internal)[:, 1]
        action = probability >= THRESHOLD
        prior_probability = np.full(
            len(y_internal), float(y_dev.mean()), dtype=float
        )
        similarity_threshold, development_rule_ba = _select_similarity_rule(
            x_dev[:, 0], y_dev
        )
        rule_action = x_internal[:, 0] >= similarity_threshold
        metrics = {
            "model_log_loss": float(
                log_loss(y_internal, probability, labels=[0, 1])
            ),
            "development_prior_log_loss": float(
                log_loss(y_internal, prior_probability, labels=[0, 1])
            ),
            "model_brier": float(
                brier_score_loss(y_internal, probability)
            ),
            "development_prior_brier": float(
                brier_score_loss(y_internal, prior_probability)
            ),
            "accuracy_at_0_5": float(accuracy_score(y_internal, action)),
            "balanced_accuracy_at_0_5": float(
                balanced_accuracy_score(y_internal, action)
            ),
            "predicted_on": int(action.sum()),
            "predicted_off": int((~action).sum()),
            "positive_recall": float(
                ((action) & (y_internal == 1)).sum()
                / max(1, (y_internal == 1).sum())
            ),
            "similarity_rule_threshold": similarity_threshold,
            "similarity_rule_development_balanced_accuracy": (
                development_rule_ba
            ),
            "similarity_rule_internal_accuracy": float(
                accuracy_score(y_internal, rule_action)
            ),
            "similarity_rule_internal_balanced_accuracy": float(
                balanced_accuracy_score(y_internal, rule_action)
            ),
        }
        passed = (
            metrics["model_log_loss"]
            < metrics["development_prior_log_loss"]
            and metrics["model_brier"]
            < metrics["development_prior_brier"]
            and metrics["accuracy_at_0_5"] >= 0.8
            and metrics["balanced_accuracy_at_0_5"] >= 0.7
            and metrics["predicted_on"] >= 8
            and metrics["predicted_off"] >= 8
        )
        scaler = model.named_steps["scale"]
        logistic = model.named_steps["logistic"]
        reports[component] = {
            "status": (
                "HISTORICAL_PROXY_NUMERIC_GATE_PASSED_NOT_FORMAL"
                if passed
                else "HISTORICAL_PROXY_NUMERIC_GATE_FAILED_NOT_FORMAL"
            ),
            "development_rows": len(y_dev),
            "development_users": len(set(groups.tolist())),
            "internal_rows": len(y_internal),
            "internal_users": len(internal_users),
            "development_class_counts": dict(
                sorted(Counter(y_dev.tolist()).items())
            ),
            "internal_class_counts": dict(
                sorted(Counter(y_internal.tolist()).items())
            ),
            "model_selection": selection,
            "metrics": metrics,
            "beats_similarity_rule_on_balanced_accuracy": (
                metrics["balanced_accuracy_at_0_5"]
                > metrics["similarity_rule_internal_balanced_accuracy"]
            ),
            "proxy_numeric_gate_passed": passed,
            "fitted_parameters": {
                "feature_names": list(FEATURE_NAMES),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coefficient_for_on": logistic.coef_[0].tolist(),
                "intercept_for_on": float(logistic.intercept_[0]),
            },
        }
        for context, target, prob, predicted, rule_predicted in zip(
            internal_contexts,
            y_internal.tolist(),
            probability.tolist(),
            action.tolist(),
            rule_action.tolist(),
            strict=True,
        ):
            state = states[str(context["state_id"])]
            prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "component": component,
                    "state_id": str(context["state_id"]),
                    "user_id": str(state["user_id"]),
                    "target_y": int(target),
                    "predicted_probability": float(prob),
                    "predicted_on": bool(predicted),
                    "similarity_rule_on": bool(rule_predicted),
                }
            )

    passed_components = [
        component
        for component in COMPONENTS
        if reports[component]["proxy_numeric_gate_passed"]
    ]
    report = {
        "protocol": PROTOCOL,
        "status": "HISTORICAL_CONSTRUCTION_PROXY_DIAGNOSTIC_ONLY",
        "formal_component_status_authorized": False,
        "clean_component_treatment_executed": False,
        "estimand": (
            "Constructed component-use/applicability proxy, not response-level "
            "individual treatment effect."
        ),
        "development_rows": len(development_contexts),
        "development_users": len(development_users),
        "internal_rows": len(internal_contexts),
        "internal_users": len(internal_users),
        "features": list(FEATURE_NAMES),
        "forbidden_fields_used_as_features": [],
        "embedding_features_used": False,
        "component_reports": reports,
        "proxy_passed_components_before_shortcut_audit": passed_components,
        "formal_passed_components": [],
        "rs_status": "NOT_ASSESSED_BY_THIS_MEMORY_PROXY_DIAGNOSTIC",
        "external_claim_authorized": False,
        "next": (
            "Do not assign MP/MS/ME a formal learned/not-learned status from "
            "this artifact. It used old synthetic applicability proxies and "
            "did not run the current clean component treatments. Use it only "
            "to design anti-shortcut controls for future generated data."
            if passed_components
            else "No formal component conclusion is authorized."
        ),
        "lineage": {
            "contract_sha256": sha256_file(args.contract),
            "states_sha256": sha256_file(args.data_dir / "pm_v2_states.jsonl"),
            "evaluator_contexts_sha256": sha256_file(
                args.data_dir / "evaluator_contexts.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "controlled_memory_routing_report.json", report)
    write_jsonl(
        args.out_dir / "internal_predictions.jsonl", prediction_rows
    )
    print(report)


if __name__ == "__main__":
    main()
