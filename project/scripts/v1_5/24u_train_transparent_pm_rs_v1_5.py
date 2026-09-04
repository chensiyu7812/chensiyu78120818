#!/usr/bin/env python3
"""Train and audit the minimum transparent PM_RS logistic head."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-transparent-pm-rs-logistic-v1"
C = 0.3
OPEN_THRESHOLD = 0.60
RANDOM_SEED = 20260730


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_labels_v1"
        / "pm_rs_training_labels.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_transparent_pm_rs_v1",
    )
    args = parser.parse_args()

    all_rows = _rows(args.labels)
    rows = [row for row in all_rows if row.get("target_y") in {0, 1}]
    counts = Counter(int(row["target_y"]) for row in rows)
    if counts[0] < 8 or counts[1] < 8:
        raise RuntimeError(
            "training requires at least eight independent R0 and eight RS "
            f"groups; found R0={counts[0]}, RS={counts[1]}"
        )
    if len({str(row["user_id"]) for row in rows}) != len(rows):
        raise RuntimeError("PM_RS labels are not dialogue-group independent")
    feature_names = sorted(
        {
            str(key)
            for row in rows
            for key in row["transparent_pm_features"]
        }
    )
    x = np.asarray(
        [
            [
                float(row["transparent_pm_features"].get(name, 0.0))
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )
    y = np.asarray([int(row["target_y"]) for row in rows], dtype=int)
    minimum_class = min(counts.values())
    folds = min(4, minimum_class)
    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_SEED
    )

    def model() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=C,
                        penalty="l2",
                        solver="liblinear",
                        class_weight=None,
                        random_state=RANDOM_SEED,
                        max_iter=2000,
                    ),
                ),
            ]
        )

    oof_probability = cross_val_predict(
        model(), x, y, cv=cv, method="predict_proba"
    )[:, 1]
    oof_action = (oof_probability >= OPEN_THRESHOLD).astype(int)
    prevalence = float(y.mean())
    baseline_probability = np.full_like(
        oof_probability, prevalence, dtype=float
    )
    metrics = {
        "oof_log_loss": float(log_loss(y, oof_probability, labels=[0, 1])),
        "prevalence_only_log_loss": float(
            log_loss(y, baseline_probability, labels=[0, 1])
        ),
        "oof_brier": float(brier_score_loss(y, oof_probability)),
        "prevalence_only_brier": float(
            brier_score_loss(y, baseline_probability)
        ),
        "oof_balanced_accuracy_at_0_60": float(
            balanced_accuracy_score(y, oof_action)
        ),
        "oof_open_rate_at_0_60": float(oof_action.mean()),
        "oof_predicted_R0": int((oof_action == 0).sum()),
        "oof_predicted_RS": int((oof_action == 1).sum()),
    }
    learned_signal = (
        metrics["oof_log_loss"] < metrics["prevalence_only_log_loss"]
        and metrics["oof_balanced_accuracy_at_0_60"] > 0.50
        and metrics["oof_predicted_R0"] > 0
        and metrics["oof_predicted_RS"] > 0
    )
    fitted = model().fit(x, y)
    scaler = fitted.named_steps["scale"]
    logistic = fitted.named_steps["logistic"]
    artifact = {
        "protocol": PROTOCOL,
        "status": (
            "MINIMUM_LEARNED_SIGNAL_PASSED"
            if learned_signal
            else "TRAINED_BUT_MINIMUM_LEARNED_SIGNAL_NOT_DEMONSTRATED"
        ),
        "scope": (
            "PM_RS opportunity-conditional component-effect head for the "
            "fixed six-card Bank"
        ),
        "algorithm": {
            "model": "standardized L2 logistic regression",
            "loss": "binary cross-entropy with L2 regularization",
            "C": C,
            "class_weight": None,
            "open_threshold": OPEN_THRESHOLD,
            "threshold_reason": (
                "RS is default-off; 0.60 requires evidence above an even "
                "chance before paying retrieval/prompt cost."
            ),
            "baai_or_other_embedding_features": False,
        },
        "training_rows": len(rows),
        "class_counts": {"R0": counts[0], "RS": counts[1]},
        "feature_names": feature_names,
        "standard_scaler_mean": scaler.mean_.tolist(),
        "standard_scaler_scale": scaler.scale_.tolist(),
        "coefficient_for_RS": logistic.coef_[0].tolist(),
        "intercept_for_RS": float(logistic.intercept_[0]),
        "oof": {
            "protocol": (
                f"{folds}-fold stratified OOF; one row per independent "
                "dialogue"
            ),
            "metrics": metrics,
            "minimum_learned_signal_passed": learned_signal,
        },
        "runtime_order": [
            "hard eligibility and safety gates",
            "retrieve frozen Top-1 card",
            "PM_RS probability >= 0.60",
            "cost within runtime budget",
            "otherwise keep RS off",
        ],
        "claim_limit": (
            "OOF is a training qualification diagnostic, not the paper's "
            "external ESConv/EvoEmo effect estimate."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
        },
    }
    predictions = [
        {
            "pair_id": row["pair_id"],
            "user_id": row["user_id"],
            "target_y": int(target),
            "oof_rs_probability": float(probability),
            "oof_action": (
                "RS" if probability >= OPEN_THRESHOLD else "R0"
            ),
        }
        for row, target, probability in zip(
            rows, y.tolist(), oof_probability.tolist(), strict=True
        )
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "pm_rs_model.json", artifact)
    write_jsonl(args.out_dir / "oof_predictions.jsonl", predictions)
    print(artifact)


if __name__ == "__main__":
    main()
