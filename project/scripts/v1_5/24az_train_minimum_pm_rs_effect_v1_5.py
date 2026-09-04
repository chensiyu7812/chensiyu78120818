#!/usr/bin/env python3
"""Train the first low-capacity grouped-OOF PM_RS effect head."""

from __future__ import annotations

import argparse
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
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.text import estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-minimum-grouped-oof-pm-rs-effect-v1"
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
OUTER_FOLDS = 5
INNER_FOLDS = 4
C_VALUE = 0.3
THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MIN_DECISION_FRACTION = 0.20
FEATURE_NAMES = (
    "advice_welcome",
    "emotion_visible",
    "uncertainty_or_multi_concern",
    "current_user_token_estimate",
    "visible_dialogue_turn_count",
    "current_user_has_question_mark",
)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "quality",
    "risk",
    "card",
    "family",
    "retrieval",
    "score",
    "response",
    "judge",
    "cost",
    "token_delta",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=C_VALUE,
                    penalty="l2",
                    solver="liblinear",
                    class_weight=None,
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )


def _feature_values(
    label: dict[str, Any], state: dict[str, Any]
) -> dict[str, float]:
    flags = label["observable_pre_generation_features"]
    current = str(state["current_user_text"])
    return {
        "advice_welcome": float(bool(flags["advice_welcome"])),
        "emotion_visible": float(bool(flags["emotion_visible"])),
        "uncertainty_or_multi_concern": float(
            bool(flags["uncertainty_or_multi_concern"])
        ),
        "current_user_token_estimate": float(estimate_tokens(current)),
        "visible_dialogue_turn_count": float(
            len(state["visible_dialogue"])
        ),
        "current_user_has_question_mark": float("?" in current),
    }


def _select_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates: list[tuple[float, float]] = []
    for threshold in THRESHOLD_GRID:
        action = probability >= threshold
        fraction = float(action.mean())
        if (
            fraction < MIN_DECISION_FRACTION
            or fraction > 1.0 - MIN_DECISION_FRACTION
        ):
            continue
        candidates.append(
            (
                float(balanced_accuracy_score(y, action)),
                float(threshold),
            )
        )
    if not candidates:
        return 0.50
    # Higher threshold breaks an equal-BA tie conservatively toward RS-off.
    return max(candidates)[1]


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    action: np.ndarray,
    prior_probability: np.ndarray,
) -> dict[str, Any]:
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "prior_log_loss": float(
            log_loss(y, prior_probability, labels=[0, 1])
        ),
        "brier": float(brier_score_loss(y, probability)),
        "prior_brier": float(brier_score_loss(y, prior_probability)),
        "brier_improvement_vs_train_fold_prevalence_prior": float(
            brier_score_loss(y, prior_probability)
            - brier_score_loss(y, probability)
        ),
        "accuracy": float(accuracy_score(y, action)),
        "balanced_accuracy": float(balanced_accuracy_score(y, action)),
        "positive_recall": float(recall_score(y, action, pos_label=1)),
        "predicted_on": int(action.sum()),
        "predicted_off": int((~action).sum()),
        "predicted_on_fraction": float(action.mean()),
        "predicted_off_fraction": float((~action).mean()),
    }


def _nested_oof(
    x: np.ndarray, y: np.ndarray, seed: int
) -> dict[str, Any]:
    outer = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=seed
    )
    probability = np.zeros(len(y), dtype=float)
    prior = np.zeros(len(y), dtype=float)
    action = np.zeros(len(y), dtype=bool)
    fold_ids = np.zeros(len(y), dtype=int)
    thresholds: list[float] = []
    for fold, (train_index, test_index) in enumerate(
        outer.split(x, y), start=1
    ):
        train_y = y[train_index]
        inner_minority = int(
            min(Counter(train_y.tolist()).values())
        )
        inner_folds = min(INNER_FOLDS, inner_minority)
        inner = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + fold,
        )
        inner_probability = cross_val_predict(
            _model(seed + fold),
            x[train_index],
            train_y,
            cv=inner,
            method="predict_proba",
        )[:, 1]
        threshold = _select_threshold(train_y, inner_probability)
        thresholds.append(threshold)
        fitted = _model(seed + fold).fit(x[train_index], train_y)
        probability[test_index] = fitted.predict_proba(x[test_index])[:, 1]
        prior[test_index] = float(train_y.mean())
        action[test_index] = probability[test_index] >= threshold
        fold_ids[test_index] = fold
    return {
        "seed": seed,
        "probability": probability,
        "prior_probability": prior,
        "action": action,
        "fold_ids": fold_ids,
        "outer_fold_thresholds": thresholds,
        "metrics": _metrics(y, probability, action, prior),
    }


def _bootstrap_brier_gain(
    y: np.ndarray,
    probability: np.ndarray,
    prior_probability: np.ndarray,
    draws: int = 5000,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEEDS[0])
    index = np.arange(len(y))
    gains: list[float] = []
    for _ in range(draws):
        sample = rng.choice(index, size=len(index), replace=True)
        gains.append(
            float(
                brier_score_loss(y[sample], prior_probability[sample])
                - brier_score_loss(y[sample], probability[sample])
            )
        )
    return {
        "draws": draws,
        "lower_95": float(np.quantile(gains, 0.025)),
        "median": float(np.quantile(gains, 0.5)),
        "upper_95": float(np.quantile(gains, 0.975)),
        "ci_is_descriptive_not_a_hard_gate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_effect_final_labels_v1"
        / "pm_rs_effect_labels.jsonl",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_paired_effect_v1"
        / "selected_states.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_pm_rs_effect_v1",
    )
    args = parser.parse_args()

    labels = _rows(args.labels)
    states = {
        str(row["pair_id"]): row for row in _rows(args.states)
    }
    if len(labels) != 32 or len(states) != 32:
        raise RuntimeError("expected exactly 32 labels and selected states")
    if len({str(row["pair_id"]) for row in labels}) != len(labels):
        raise RuntimeError("duplicate pair_id")
    if len({str(row["user_id"]) for row in labels}) != len(labels):
        raise RuntimeError("one independent dialogue/user group is required")
    if set(states) != {str(row["pair_id"]) for row in labels}:
        raise RuntimeError("label/state lineage mismatch")
    if any(row["target_y"] not in {0, 1} for row in labels):
        raise RuntimeError("all labels must be known binary targets")
    counts = Counter(int(row["target_y"]) for row in labels)
    if counts[0] < 8 or counts[1] < 8:
        raise RuntimeError("at least eight groups per class are required")
    if any(
        token in feature.casefold()
        for feature in FEATURE_NAMES
        for token in FORBIDDEN_FEATURE_TOKENS
    ):
        raise RuntimeError("outcome/resource-item leakage in feature contract")

    values = [
        _feature_values(row, states[str(row["pair_id"])])
        for row in labels
    ]
    x = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in values],
        dtype=float,
    )
    y = np.asarray([int(row["target_y"]) for row in labels], dtype=int)
    if not np.isfinite(x).all():
        raise RuntimeError("features contain non-finite values")
    variable_features = [
        name
        for index, name in enumerate(FEATURE_NAMES)
        if len(np.unique(x[:, index])) > 1
    ]
    if len(variable_features) != len(FEATURE_NAMES):
        raise RuntimeError("feature contract contains constant columns")

    runs = [_nested_oof(x, y, seed) for seed in SEEDS]
    probability_matrix = np.stack(
        [run["probability"] for run in runs], axis=0
    )
    prior_matrix = np.stack(
        [run["prior_probability"] for run in runs], axis=0
    )
    action_matrix = np.stack([run["action"] for run in runs], axis=0)
    aggregate_probability = probability_matrix.mean(axis=0)
    aggregate_prior = prior_matrix.mean(axis=0)
    aggregate_action = action_matrix.sum(axis=0) >= 3
    aggregate_metrics = _metrics(
        y,
        aggregate_probability,
        aggregate_action,
        aggregate_prior,
    )
    seed_balanced_accuracy = [
        float(run["metrics"]["balanced_accuracy"]) for run in runs
    ]
    seed_ba_std = float(np.std(seed_balanced_accuracy, ddof=1))

    transparent_action = np.asarray(
        [
            bool(
                value["advice_welcome"]
                or value["emotion_visible"]
                or value["uncertainty_or_multi_concern"]
            )
            for value in values
        ],
        dtype=bool,
    )
    transparent_metrics = {
        "definition": (
            "open when advice_welcome OR emotion_visible OR "
            "uncertainty_or_multi_concern; otherwise off"
        ),
        "outcome_free_and_uses_only_three_registered_binary_cues": True,
        "accuracy": float(accuracy_score(y, transparent_action)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, transparent_action)
        ),
        "positive_recall": float(
            recall_score(y, transparent_action, pos_label=1)
        ),
        "predicted_on": int(transparent_action.sum()),
        "predicted_off": int((~transparent_action).sum()),
    }
    always_off = {
        "accuracy": float(accuracy_score(y, np.zeros(len(y), dtype=int))),
        "balanced_accuracy": 0.5,
    }
    always_on = {
        "accuracy": float(accuracy_score(y, np.ones(len(y), dtype=int))),
        "balanced_accuracy": 0.5,
    }

    minimum_signal = (
        aggregate_metrics[
            "brier_improvement_vs_train_fold_prevalence_prior"
        ]
        > 0
        and aggregate_metrics["balanced_accuracy"] > 0.50
        and aggregate_metrics["predicted_on_fraction"]
        >= MIN_DECISION_FRACTION
        and aggregate_metrics["predicted_off_fraction"]
        >= MIN_DECISION_FRACTION
    )
    formal_gate_checks = {
        "brier_beats_train_fold_prevalence_prior": (
            aggregate_metrics[
                "brier_improvement_vs_train_fold_prevalence_prior"
            ]
            > 0
        ),
        "balanced_accuracy_at_least_0_70": (
            aggregate_metrics["balanced_accuracy"] >= 0.70
        ),
        "positive_recall_at_least_0_60": (
            aggregate_metrics["positive_recall"] >= 0.60
        ),
        "on_fraction_at_least_0_20": (
            aggregate_metrics["predicted_on_fraction"]
            >= MIN_DECISION_FRACTION
        ),
        "off_fraction_at_least_0_20": (
            aggregate_metrics["predicted_off_fraction"]
            >= MIN_DECISION_FRACTION
        ),
        "five_seed_balanced_accuracy_std_at_most_0_03": (
            seed_ba_std <= 0.03
        ),
        "beats_outcome_free_transparent_rule_balanced_accuracy": (
            aggregate_metrics["balanced_accuracy"]
            > transparent_metrics["balanced_accuracy"]
        ),
    }
    formal_pass = all(formal_gate_checks.values())

    final_threshold = _select_threshold(y, aggregate_probability)
    final_fit = _model(SEEDS[0]).fit(x, y)
    scaler = final_fit.named_steps["scale"]
    logistic = final_fit.named_steps["logistic"]
    report = {
        "protocol": PROTOCOL,
        "status": (
            "FORMAL_TRAIN_ONLY_PM_RS_LEARNING_GATE_PASSED"
            if formal_pass
            else (
                "EXPLORATORY_MINIMUM_SIGNAL_ONLY"
                if minimum_signal
                else "PM_RS_MINIMUM_SIGNAL_NOT_LEARNED"
            )
        ),
        "dataset": {
            "rows": len(labels),
            "independent_user_dialogue_groups": len(labels),
            "class_counts": {"RS_off": counts[0], "RS_on": counts[1]},
            "unknown": 0,
            "one_row_per_group": True,
        },
        "feature_contract": {
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "all_features_pre_generation_observable": True,
            "selected_card_family_score_or_response_features": False,
            "embedding_features": False,
        },
        "algorithm": {
            "model": "standardized L2 logistic regression",
            "loss": "binary cross-entropy plus L2",
            "C": C_VALUE,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seeds": list(SEEDS),
            "threshold_grid": list(THRESHOLD_GRID),
            "threshold_selection": (
                "inner train-fold balanced accuracy, requiring both "
                "decisions; higher threshold breaks ties"
            ),
            "aggregate_action": "majority of five nested-OOF seed decisions",
        },
        "oof": {
            "aggregate_metrics": aggregate_metrics,
            "per_seed": [
                {
                    "seed": run["seed"],
                    "outer_fold_thresholds": run[
                        "outer_fold_thresholds"
                    ],
                    "metrics": run["metrics"],
                }
                for run in runs
            ],
            "five_seed_balanced_accuracy": seed_balanced_accuracy,
            "five_seed_balanced_accuracy_std": seed_ba_std,
            "brier_gain_bootstrap": _bootstrap_brier_gain(
                y, aggregate_probability, aggregate_prior
            ),
        },
        "baselines": {
            "always_off": always_off,
            "always_on": always_on,
            "transparent_rule": transparent_metrics,
        },
        "gates": {
            "minimum_exploratory_signal_passed": minimum_signal,
            "formal_checks": formal_gate_checks,
            "formal_train_only_learning_gate_passed": formal_pass,
        },
        "final_fit_for_future_frozen_evaluation": {
            "promoted": formal_pass,
            "threshold": final_threshold,
            "standard_scaler_mean": scaler.mean_.tolist(),
            "standard_scaler_scale": scaler.scale_.tolist(),
            "coefficient_for_RS_on": logistic.coef_[0].tolist(),
            "intercept_for_RS_on": float(logistic.intercept_[0]),
        },
        "claim_boundary": (
            "This is grouped-OOF train-only learnability evidence on "
            "human-adjudicated fixed-protocol effect proxies. It is not "
            "external performance, objective human preference, or an "
            "individual causal-effect estimate."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "states_sha256": sha256_file(args.states),
        },
    }
    predictions = [
        {
            "protocol": PROTOCOL,
            "pair_id": row["pair_id"],
            "user_id": row["user_id"],
            "target_y": int(y[index]),
            "feature_values": values[index],
            "mean_oof_rs_probability": float(
                aggregate_probability[index]
            ),
            "majority_oof_action": (
                "M0+RS" if aggregate_action[index] else "M0+R0"
            ),
            "seed_oof_actions": [
                "M0+RS" if run["action"][index] else "M0+R0"
                for run in runs
            ],
        }
        for index, row in enumerate(labels)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "pm_rs_training_report.json", report)
    write_jsonl(args.out_dir / "oof_predictions.jsonl", predictions)
    print(report)


if __name__ == "__main__":
    main()
