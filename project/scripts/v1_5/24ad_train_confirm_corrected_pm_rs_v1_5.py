#!/usr/bin/env python3
"""Fit the frozen six-feature PM_RS and run pooled and prospective checks."""

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
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-corrected-six-feature-pm-rs-v1"
SEED = 20260730
OPEN_THRESHOLD = 0.60
C_GRID = (0.01, 0.1, 1.0, 10.0)
FEATURE_NAMES = (
    "current_user_token_estimate",
    "history_turn_count",
    "move__AM01_invite_open_expression",
    "move__AM04_tentative_paraphrase_check",
    "move__AM05_grounded_validation",
    "move__AM10_offer_one_optional_micro_step",
)


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


def _matrix(
    rows: list[dict[str, Any]], feature_field: str
) -> np.ndarray:
    return np.asarray(
        [
            [
                float(row[feature_field].get(name, 0.0))
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=float,
    )


def _cv_c_scores(
    x: np.ndarray, y: np.ndarray, folds: int, seed: int
) -> tuple[float, dict[str, Any]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_losses: dict[float, list[float]] = {value: [] for value in C_GRID}
    for train_index, test_index in cv.split(x, y):
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
    one_se_limit = means[minimum_c] + standard_errors[minimum_c]
    eligible = [
        value for value in C_GRID if means[value] <= one_se_limit
    ]
    selected = min(eligible)
    return selected, {
        "folds": folds,
        "mean_fold_log_loss_by_C": {
            str(value): means[value] for value in C_GRID
        },
        "standard_error_by_C": {
            str(value): standard_errors[value] for value in C_GRID
        },
        "minimum_mean_C": minimum_c,
        "one_standard_error_limit": one_se_limit,
        "selected_stronger_regularization_C": selected,
    }


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    prior_probability: np.ndarray,
) -> dict[str, Any]:
    action = probability >= OPEN_THRESHOLD
    prior_action = prior_probability >= OPEN_THRESHOLD
    return {
        "model_log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "prior_log_loss": float(
            log_loss(y, prior_probability, labels=[0, 1])
        ),
        "model_brier": float(brier_score_loss(y, probability)),
        "prior_brier": float(brier_score_loss(y, prior_probability)),
        "brier_improvement_vs_prior": float(
            brier_score_loss(y, prior_probability)
            - brier_score_loss(y, probability)
        ),
        "accuracy_at_0_60": float(accuracy_score(y, action)),
        "balanced_accuracy_at_0_60": float(
            balanced_accuracy_score(y, action)
        ),
        "predicted_on": int(action.sum()),
        "predicted_off": int((~action).sum()),
        "positive_recall_at_0_60": float(
            ((action) & (y == 1)).sum() / max(1, (y == 1).sum())
        ),
        "always_off_accuracy": float(accuracy_score(y, np.zeros_like(y))),
        "prior_threshold_accuracy": float(accuracy_score(y, prior_action)),
    }


def _bootstrap_improvement(
    y: np.ndarray,
    probability: np.ndarray,
    prior_probability: np.ndarray,
    draws: int = 10000,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    indices = np.arange(len(y))
    for _ in range(draws):
        sample = rng.choice(indices, size=len(indices), replace=True)
        values.append(
            float(
                brier_score_loss(y[sample], prior_probability[sample])
                - brier_score_loss(y[sample], probability[sample])
            )
        )
    return {
        "draws": draws,
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wave1-labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_labels_v1"
        / "pm_rs_training_labels.jsonl",
    )
    parser.add_argument(
        "--supplement-labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1_analysis"
        / "unblinded_quality_risk_labels.jsonl",
    )
    parser.add_argument(
        "--wave1-execution",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v1_execution"
        / "generation_outcomes.jsonl",
    )
    parser.add_argument(
        "--supplement-execution",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1_execution"
        / "generation_outcomes.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_corrected_pm_rs_v1",
    )
    args = parser.parse_args()

    wave1 = _rows(args.wave1_labels)
    supplement = _rows(args.supplement_labels)
    if len(wave1) != 32 or len(supplement) != 16:
        raise RuntimeError("expected 32 wave1 and 16 supplement groups")
    if {str(row["user_id"]) for row in wave1} & {
        str(row["user_id"]) for row in supplement
    }:
        raise RuntimeError("wave1 and supplement dialogue groups overlap")
    if len({str(row["user_id"]) for row in wave1 + supplement}) != 48:
        raise RuntimeError("all 48 rows must be dialogue-group independent")

    x_wave1 = _matrix(wave1, "transparent_pm_features")
    y_wave1 = np.asarray(
        [int(row["quality_result"] == "RS") for row in wave1], dtype=int
    )
    x_supplement = _matrix(supplement, "primary_feature_values")
    y_supplement = np.asarray(
        [int(row["quality_result"] == "RS") for row in supplement], dtype=int
    )

    selected_c, old_selection = _cv_c_scores(
        x_wave1, y_wave1, folds=4, seed=SEED
    )
    prospective_model = _pipeline(selected_c).fit(x_wave1, y_wave1)
    prospective_probability = prospective_model.predict_proba(
        x_supplement
    )[:, 1]
    old_prior = float(y_wave1.mean())
    prospective_prior = np.full(len(y_supplement), old_prior, dtype=float)
    prospective_metrics = _metrics(
        y_supplement, prospective_probability, prospective_prior
    )
    prospective_pass = (
        prospective_metrics["model_log_loss"]
        < prospective_metrics["prior_log_loss"]
        and prospective_metrics["model_brier"]
        < prospective_metrics["prior_brier"]
        and prospective_metrics["balanced_accuracy_at_0_60"] > 0.5
        and prospective_metrics["predicted_on"] >= 4
        and prospective_metrics["predicted_off"] >= 4
    )

    x_pooled = np.concatenate([x_wave1, x_supplement], axis=0)
    y_pooled = np.concatenate([y_wave1, y_supplement], axis=0)
    outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    pooled_probability = np.zeros(len(y_pooled), dtype=float)
    pooled_prior = np.zeros(len(y_pooled), dtype=float)
    outer_c: list[float] = []
    fold_id = np.zeros(len(y_pooled), dtype=int)
    for fold, (train_index, test_index) in enumerate(
        outer.split(x_pooled, y_pooled), start=1
    ):
        inner_c, _ = _cv_c_scores(
            x_pooled[train_index],
            y_pooled[train_index],
            folds=3,
            seed=SEED + fold,
        )
        outer_c.append(inner_c)
        pooled_probability[test_index] = _pipeline(inner_c).fit(
            x_pooled[train_index], y_pooled[train_index]
        ).predict_proba(x_pooled[test_index])[:, 1]
        pooled_prior[test_index] = float(y_pooled[train_index].mean())
        fold_id[test_index] = fold
    pooled_metrics = _metrics(
        y_pooled, pooled_probability, pooled_prior
    )
    pooled_bootstrap = _bootstrap_improvement(
        y_pooled, pooled_probability, pooled_prior
    )
    pooled_pass = (
        pooled_bootstrap["lower_95"] > 0
        and pooled_metrics["predicted_on"] >= 4
        and pooled_metrics["predicted_off"] >= 4
        and pooled_metrics["predicted_on"] / len(y_pooled) >= 0.1
        and pooled_metrics["predicted_off"] / len(y_pooled) >= 0.1
    )

    final_c, pooled_selection = _cv_c_scores(
        x_pooled, y_pooled, folds=4, seed=SEED
    )
    fitted = _pipeline(final_c).fit(x_pooled, y_pooled)
    scaler = fitted.named_steps["scale"]
    logistic = fitted.named_steps["logistic"]

    wave1_quality = Counter(str(row["quality_result"]) for row in wave1)
    supplement_quality = Counter(
        str(row["quality_result"]) for row in supplement
    )
    pooled_quality = wave1_quality + supplement_quality
    risk_counts = {
        "wave1": {
            "R0": sum(bool(row["r0_material_risks"]) for row in wave1),
            "RS": sum(bool(row["rs_material_risks"]) for row in wave1),
        },
        "supplement": {
            "R0": sum(
                row["r0_any_material_risk"] == "yes" for row in supplement
            ),
            "RS": sum(
                row["rs_any_material_risk"] == "yes" for row in supplement
            ),
        },
    }
    risk_counts["pooled_descriptive"] = {
        arm: risk_counts["wave1"][arm] + risk_counts["supplement"][arm]
        for arm in ("R0", "RS")
    }

    report = {
        "protocol": PROTOCOL,
        "status": (
            "CORRECTED_PM_RS_LEARNED_SIGNAL_PASSED"
            if prospective_pass and pooled_pass
            else "CURRENT_RS_RESPONSE_WINNER_HEAD_NOT_LEARNED"
        ),
        "scope": (
            "Fixed six-definition-card/five-active-move RS runtime; "
            "material immediate quality-benefit head only."
        ),
        "feature_contract": {
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "frozen_before_supplement_outcomes": True,
            "baai_features": False,
        },
        "target": {
            "positive": "RS materially better",
            "negative": "R0 materially better or materially equivalent",
            "risk_is_not_collapsed_into_target": True,
            "cost_is_not_collapsed_into_target": True,
        },
        "quality_effect": {
            "wave1": dict(sorted(wave1_quality.items())),
            "supplement": dict(sorted(supplement_quality.items())),
            "pooled": dict(sorted(pooled_quality.items())),
            "pooled_netwin_rs": (
                pooled_quality["RS"] - pooled_quality["R0"]
            )
            / len(y_pooled),
        },
        "prospective_32_train_16_confirmation": {
            "role_frozen_before_supplement_annotations": True,
            "training_class_counts": {
                "nonbenefit": int((y_wave1 == 0).sum()),
                "benefit": int((y_wave1 == 1).sum()),
            },
            "confirmation_class_counts": {
                "nonbenefit": int((y_supplement == 0).sum()),
                "benefit": int((y_supplement == 1).sum()),
            },
            "model_selection": old_selection,
            "metrics": prospective_metrics,
            "passed": prospective_pass,
            "interpretation": (
                "At threshold 0.60, an always-off prediction can achieve high "
                "raw accuracy because only 2/16 confirmation pairs are "
                "positive; balanced accuracy, positive recall, and proper "
                "scores show that this is not learned routing."
            ),
        },
        "pooled_48_group_nested_oof": {
            "outer_folds": 4,
            "inner_selected_C_by_outer_fold": outer_c,
            "metrics": pooled_metrics,
            "brier_improvement_bootstrap": pooled_bootstrap,
            "passed": pooled_pass,
        },
        "risk": {
            "material_response_counts": risk_counts,
            "atomic_heads_qualified": [],
            "decision": (
                "Retain deterministic guards. Wave1 and supplement risk UIs "
                "differ, so pooled counts are descriptive and no risk model "
                "is trained."
            ),
        },
        "final_diagnostic_fit_not_promoted": {
            "model_selection": pooled_selection,
            "selected_C": final_c,
            "standard_scaler_mean": scaler.mean_.tolist(),
            "standard_scaler_scale": scaler.scale_.tolist(),
            "coefficient_for_RS": logistic.coef_[0].tolist(),
            "intercept_for_RS": float(logistic.intercept_[0]),
        },
        "decision": {
            "promote_pm_rs": False,
            "scope_of_nonpromotion": (
                "Only the six-feature classifier whose target is the winner "
                "of one stochastic R0/RS generation pair."
            ),
            "run_more_same_packet_human_annotation": False,
            "run_more_same_stochastic_winner_expansion": False,
            "rs_resource_rejected": False,
            "four_component_research_question_rejected": False,
            "runtime_fallback": (
                "Do not deploy this learned candidate. Continue the research "
                "with constructed pre-action routing labels and transparent "
                "opportunity gates."
            ),
            "next_component": (
                "Return to the registered RS positive-expansion design: "
                "construct balanced, human-audited pre-action on/off states; "
                "use response pairs to qualify treatment and end-to-end "
                "quality, not as a one-sample routing gold label."
            ),
        },
        "lineage": {
            "wave1_labels_sha256": sha256_file(args.wave1_labels),
            "supplement_labels_sha256": sha256_file(args.supplement_labels),
            "wave1_execution_sha256": sha256_file(args.wave1_execution),
            "supplement_execution_sha256": sha256_file(
                args.supplement_execution
            ),
        },
    }
    ids = [str(row["pair_id"]) for row in wave1] + [
        str(row["pair_id"]) for row in supplement
    ]
    pooled_predictions = [
        {
            "protocol": PROTOCOL,
            "pair_id": pair_id,
            "data_role": "wave1" if index < len(wave1) else "supplement",
            "outer_fold": int(fold_id[index]),
            "target_y": int(y_pooled[index]),
            "oof_rs_probability": float(pooled_probability[index]),
            "oof_action": (
                "RS"
                if pooled_probability[index] >= OPEN_THRESHOLD
                else "R0"
            ),
        }
        for index, pair_id in enumerate(ids)
    ]
    prospective_predictions = [
        {
            "protocol": PROTOCOL,
            "pair_id": str(row["pair_id"]),
            "target_y": int(target),
            "prospective_rs_probability": float(probability),
            "prospective_action": (
                "RS" if probability >= OPEN_THRESHOLD else "R0"
            ),
        }
        for row, target, probability in zip(
            supplement,
            y_supplement.tolist(),
            prospective_probability.tolist(),
            strict=True,
        )
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "pm_rs_corrected_report.json", report)
    write_jsonl(
        args.out_dir / "pooled_oof_predictions.jsonl", pooled_predictions
    )
    write_jsonl(
        args.out_dir / "prospective_predictions.jsonl",
        prospective_predictions,
    )
    print(report)


if __name__ == "__main__":
    main()
