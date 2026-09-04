#!/usr/bin/env python3
"""Train a coarse-family-aware PM_RS after state-only models fail."""

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

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-candidate-family-aware-pm-rs-development-v1"
FAMILIES = (
    "Question",
    "Providing Suggestions",
    "Affirmation and Reassurance",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
)
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
OUTER_FOLDS = 5
INNER_FOLDS = 4
C_VALUE = 0.3
THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MIN_DECISION_FRACTION = 0.20


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _model(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="liblinear",
        class_weight=None,
        random_state=seed,
        max_iter=2000,
    )


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
    return max(candidates)[1] if candidates else 0.50


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    action: np.ndarray,
    prior: np.ndarray,
) -> dict[str, Any]:
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "prior_log_loss": float(log_loss(y, prior, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "prior_brier": float(brier_score_loss(y, prior)),
        "brier_improvement_vs_train_fold_prevalence_prior": float(
            brier_score_loss(y, prior) - brier_score_loss(y, probability)
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
        minority = min(Counter(train_y.tolist()).values())
        inner = StratifiedKFold(
            n_splits=min(INNER_FOLDS, minority),
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
        "prior": prior,
        "action": action,
        "fold_ids": fold_ids,
        "outer_fold_thresholds": thresholds,
        "metrics": _metrics(y, probability, action, prior),
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
        "--transparent-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_pm_rs_effect_v1"
        / "pm_rs_training_report.json",
    )
    parser.add_argument(
        "--bge-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_bge_pm_rs_challenger_v1"
        / "bge_pm_rs_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_candidate_aware_pm_rs_development_v1",
    )
    args = parser.parse_args()

    labels = _rows(args.labels)
    if len(labels) != 32 or any(
        row["target_y"] not in {0, 1} for row in labels
    ):
        raise RuntimeError("expected 32 known effect labels")
    if len({str(row["user_id"]) for row in labels}) != len(labels):
        raise RuntimeError("one row per independent dialogue/user is required")
    transparent = __import__("json").loads(
        args.transparent_report.read_text(encoding="utf-8")
    )
    bge = __import__("json").loads(
        args.bge_report.read_text(encoding="utf-8")
    )
    if (
        transparent["status"] != "PM_RS_MINIMUM_SIGNAL_NOT_LEARNED"
        or bge["status"] != "BGE_PM_RS_SIGNAL_NOT_LEARNED"
    ):
        raise RuntimeError(
            "candidate-aware head is authorized only after both state-only "
            "models fail"
        )

    families = [
        str(row["selected_strategy_family_diagnostic_only"])
        for row in labels
    ]
    if any(family not in FAMILIES for family in families):
        raise RuntimeError("unexpected candidate family")
    x = np.asarray(
        [
            [float(family == candidate) for candidate in FAMILIES]
            for family in families
        ],
        dtype=float,
    )
    y = np.asarray([int(row["target_y"]) for row in labels], dtype=int)
    runs = [_nested_oof(x, y, seed) for seed in SEEDS]
    probability = np.stack(
        [run["probability"] for run in runs], axis=0
    ).mean(axis=0)
    prior = np.stack([run["prior"] for run in runs], axis=0).mean(axis=0)
    action_matrix = np.stack([run["action"] for run in runs], axis=0)
    action = action_matrix.sum(axis=0) >= 3
    aggregate = _metrics(y, probability, action, prior)
    seed_ba = [
        float(run["metrics"]["balanced_accuracy"]) for run in runs
    ]
    seed_ba_std = float(np.std(seed_ba, ddof=1))
    formal_checks = {
        "brier_beats_train_fold_prevalence_prior": (
            aggregate[
                "brier_improvement_vs_train_fold_prevalence_prior"
            ]
            > 0
        ),
        "balanced_accuracy_at_least_0_70": (
            aggregate["balanced_accuracy"] >= 0.70
        ),
        "positive_recall_at_least_0_60": (
            aggregate["positive_recall"] >= 0.60
        ),
        "on_fraction_at_least_0_20": (
            aggregate["predicted_on_fraction"]
            >= MIN_DECISION_FRACTION
        ),
        "off_fraction_at_least_0_20": (
            aggregate["predicted_off_fraction"]
            >= MIN_DECISION_FRACTION
        ),
        "five_seed_balanced_accuracy_std_at_most_0_03": (
            seed_ba_std <= 0.03
        ),
        "beats_state_only_transparent_and_BGE_balanced_accuracy": (
            aggregate["balanced_accuracy"]
            > max(
                float(
                    transparent["oof"]["aggregate_metrics"][
                        "balanced_accuracy"
                    ]
                ),
                float(
                    bge["oof"]["aggregate_metrics"]["balanced_accuracy"]
                ),
            )
        ),
    }
    train_only_gate = all(formal_checks.values())
    minimum_development_signal = (
        aggregate[
            "brier_improvement_vs_train_fold_prevalence_prior"
        ]
        > 0
        and aggregate["balanced_accuracy"] > 0.50
        and aggregate["positive_recall"] >= 0.60
        and aggregate["predicted_on_fraction"]
        >= MIN_DECISION_FRACTION
        and aggregate["predicted_off_fraction"]
        >= MIN_DECISION_FRACTION
    )
    final_threshold = _select_threshold(y, probability)
    final_model = _model(SEEDS[0]).fit(x, y)
    family_counts: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        mask = np.asarray([value == family for value in families])
        family_counts[family] = {
            "groups": int(mask.sum()),
            "RS_on": int(y[mask].sum()),
            "RS_off": int((1 - y[mask]).sum()),
        }

    report = {
        "protocol": PROTOCOL,
        "status": (
            "DEVELOPMENT_CANDIDATE_AWARE_PM_RS_SIGNAL_LEARNED"
            if train_only_gate
            else (
                "MINIMUM_DEVELOPMENT_SIGNAL_LEARNED_FORMAL_GATE_NOT_PASSED"
                if minimum_development_signal
                else "CANDIDATE_AWARE_PM_RS_SIGNAL_NOT_LEARNED"
            )
        ),
        "selection_role": (
            "Development candidate admitted after state-only transparent "
            "and BGE failures; requires frozen unseen evaluation."
        ),
        "decision_timing_correction": {
            "before": "pre-retrieval",
            "now": "post-candidate-discovery but pre-injection/pre-generation",
            "why": (
                "The PM must know the coarse type of resource opportunity "
                "whose marginal effect it is predicting."
            ),
            "retrieval_lookup_cost_is_measured_separately": True,
        },
        "dataset": {
            "rows": len(labels),
            "class_counts": dict(
                sorted(Counter(y.tolist()).items())
            ),
            "family_by_label": family_counts,
            "one_row_per_independent_dialogue_user": True,
        },
        "feature_contract": {
            "features": [f"family__{family}" for family in FAMILIES],
            "feature_count": len(FAMILIES),
            "candidate_card_id_visible": False,
            "candidate_text_visible": False,
            "retrieval_score_visible": False,
            "generated_response_or_outcome_visible": False,
            "coarse_candidate_family_visible": True,
        },
        "algorithm": {
            "model": "L2 logistic regression on five family one-hot values",
            "loss": "binary cross-entropy plus L2",
            "C": C_VALUE,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "seeds": list(SEEDS),
            "threshold_selection": (
                "inner train-fold only; both decisions required"
            ),
        },
        "oof": {
            "aggregate_metrics": aggregate,
            "five_seed_balanced_accuracy": seed_ba,
            "five_seed_balanced_accuracy_std": seed_ba_std,
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
        },
        "gates": {
            "minimum_development_signal_definition": [
                "Brier beats train-fold prevalence prior",
                "balanced accuracy is above 0.50",
                "positive recall is at least 0.60",
                "on and off each comprise at least 20%",
            ],
            "minimum_development_signal_passed": (
                minimum_development_signal
            ),
            "formal_metric_checks": formal_checks,
            "development_train_only_gate_passed": train_only_gate,
            "external_or_fresh_holdout_gate_passed": False,
        },
        "final_fit_for_frozen_unseen_evaluation": {
            "promoted_as_development_candidate": (
                minimum_development_signal
            ),
            "formal_train_only_gate_passed": train_only_gate,
            "threshold": final_threshold,
            "coefficient_for_RS_on_by_family": {
                family: float(value)
                for family, value in zip(
                    FAMILIES, final_model.coef_[0].tolist(), strict=True
                )
            },
            "intercept_for_RS_on": float(final_model.intercept_[0]),
        },
        "claim_boundary": (
            "Passing this gate would show only that a fixed-Bank coarse "
            "resource descriptor carries grouped-OOF development signal. "
            "Because the descriptor was admitted after earlier failures, "
            "unseen evaluation is required before a paper-level PM claim."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "transparent_report_sha256": sha256_file(
                args.transparent_report
            ),
            "bge_report_sha256": sha256_file(args.bge_report),
        },
    }
    predictions = [
        {
            "protocol": PROTOCOL,
            "pair_id": row["pair_id"],
            "user_id": row["user_id"],
            "candidate_family": families[index],
            "target_y": int(y[index]),
            "mean_oof_rs_probability": float(probability[index]),
            "majority_oof_action": (
                "M0+RS" if action[index] else "M0+R0"
            ),
            "seed_oof_actions": [
                "M0+RS" if run["action"][index] else "M0+R0"
                for run in runs
            ],
        }
        for index, row in enumerate(labels)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "candidate_aware_pm_rs_report.json", report)
    write_jsonl(args.out_dir / "oof_predictions.jsonl", predictions)
    print(report)


if __name__ == "__main__":
    main()
