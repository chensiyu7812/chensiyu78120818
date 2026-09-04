#!/usr/bin/env python3
"""Run the frozen BGE-small visible-state PM_RS challenger."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
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
from transformers import AutoModel, AutoTokenizer

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-bge-visible-state-pm-rs-challenger-v1"
MODEL_PATH = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
OUTER_FOLDS = 5
INNER_FOLDS = 4
PCA_COMPONENTS = 4
C_VALUE = 0.03
THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MIN_DECISION_FRACTION = 0.20


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _visible_state_text(state: dict[str, Any]) -> str:
    dialogue = "\n".join(
        f"{turn['speaker']}: {turn['content']}"
        for turn in state["visible_dialogue"]
    )
    return (
        "Emotional-support dialogue state before the next response:\n"
        + dialogue
    )


def _encode(texts: list[str], model_path: Path) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            encoded = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device) for key, value in encoded.items()
            }
            cls = model(**encoded).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            chunks.append(cls.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("pca", PCA(n_components=PCA_COMPONENTS, random_state=seed)),
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
        "--states",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_paired_effect_v1"
        / "selected_states.jsonl",
    )
    parser.add_argument(
        "--transparent-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_pm_rs_effect_v1"
        / "pm_rs_training_report.json",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_bge_pm_rs_challenger_v1",
    )
    args = parser.parse_args()

    labels = _rows(args.labels)
    states = {
        str(row["pair_id"]): row for row in _rows(args.states)
    }
    if (
        len(labels) != 32
        or len(states) != 32
        or set(states) != {str(row["pair_id"]) for row in labels}
    ):
        raise RuntimeError("expected 32 aligned labels and visible states")
    if len({str(row["user_id"]) for row in labels}) != len(labels):
        raise RuntimeError("one independent user/dialogue group is required")
    if any(row["target_y"] not in {0, 1} for row in labels):
        raise RuntimeError("all targets must be known")
    if not args.model.is_dir():
        raise RuntimeError(f"local BGE model is unavailable: {args.model}")
    transparent = json.loads(
        args.transparent_report.read_text(encoding="utf-8")
    )
    if transparent["status"] != "PM_RS_MINIMUM_SIGNAL_NOT_LEARNED":
        raise RuntimeError(
            "BGE challenger is authorized only after transparent failure"
        )

    texts = [
        _visible_state_text(states[str(row["pair_id"])])
        for row in labels
    ]
    if any(
        token in text
        for text in texts
        for token in ("Candidate support guidance", "Response A", "Response B")
    ):
        raise RuntimeError("post-retrieval or outcome text entered BGE input")
    x = _encode(texts, args.model)
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
    transparent_ba = float(
        transparent["oof"]["aggregate_metrics"]["balanced_accuracy"]
    )
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
        "beats_transparent_features_balanced_accuracy": (
            aggregate["balanced_accuracy"] > transparent_ba
        ),
    }
    formal_pass = all(formal_checks.values())
    minimum_signal = (
        aggregate["brier_improvement_vs_train_fold_prevalence_prior"] > 0
        and aggregate["balanced_accuracy"] > 0.50
        and aggregate["predicted_on_fraction"] >= MIN_DECISION_FRACTION
        and aggregate["predicted_off_fraction"] >= MIN_DECISION_FRACTION
    )

    final_threshold = _select_threshold(y, probability)
    final_fit = _model(SEEDS[0]).fit(x, y)
    pca = final_fit.named_steps["pca"]
    scaler = final_fit.named_steps["scale"]
    logistic = final_fit.named_steps["logistic"]
    report = {
        "protocol": PROTOCOL,
        "status": (
            "FORMAL_TRAIN_ONLY_BGE_PM_RS_GATE_PASSED"
            if formal_pass
            else (
                "BGE_EXPLORATORY_MINIMUM_SIGNAL_ONLY"
                if minimum_signal
                else "BGE_PM_RS_SIGNAL_NOT_LEARNED"
            )
        ),
        "dataset": {
            "rows": len(labels),
            "class_counts": dict(
                sorted(Counter(y.tolist()).items())
            ),
            "one_row_per_independent_user_dialogue": True,
        },
        "representation": {
            "model": "BAAI/bge-small-en-v1.5",
            "model_path": str(args.model),
            "embedding_dimension": int(x.shape[1]),
            "input": "visible dialogue state only",
            "candidate_card_or_family_visible": False,
            "retrieval_score_visible": False,
            "generated_response_or_outcome_visible": False,
            "PCA_components": PCA_COMPONENTS,
            "PCA_fit_inside_each_training_fold": True,
        },
        "algorithm": {
            "model": "PCA4 + standardized L2 logistic regression",
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
        "comparison": {
            "transparent_feature_balanced_accuracy": transparent_ba,
            "bge_minus_transparent_balanced_accuracy": (
                aggregate["balanced_accuracy"] - transparent_ba
            ),
        },
        "gates": {
            "minimum_exploratory_signal_passed": minimum_signal,
            "formal_checks": formal_checks,
            "formal_train_only_learning_gate_passed": formal_pass,
        },
        "final_fit_for_future_frozen_evaluation": {
            "promoted": formal_pass,
            "threshold": final_threshold,
            "pca_mean": pca.mean_.tolist(),
            "pca_components": pca.components_.tolist(),
            "standard_scaler_mean": scaler.mean_.tolist(),
            "standard_scaler_scale": scaler.scale_.tolist(),
            "coefficient_for_RS_on": logistic.coef_[0].tolist(),
            "intercept_for_RS_on": float(logistic.intercept_[0]),
        },
        "claim_boundary": (
            "Post-transparent-failure train-only challenger on the same "
            "fixed labels. It can qualify a representation for later frozen "
            "evaluation but is not external evidence."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "states_sha256": sha256_file(args.states),
            "transparent_report_sha256": sha256_file(
                args.transparent_report
            ),
        },
    }
    predictions = [
        {
            "protocol": PROTOCOL,
            "pair_id": row["pair_id"],
            "user_id": row["user_id"],
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
    write_json(args.out_dir / "bge_pm_rs_report.json", report)
    write_jsonl(args.out_dir / "oof_predictions.jsonl", predictions)
    print(report)


if __name__ == "__main__":
    main()
