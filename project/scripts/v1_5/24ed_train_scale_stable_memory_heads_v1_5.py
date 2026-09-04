#!/usr/bin/env python3
"""Train the final MP/MS/ME heads after the one representation repair."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-scale-stable-memory-opportunity-head-fit-v1"
COMPONENTS = ("MP", "MS", "ME")
SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
DECISION_THRESHOLD = 0.5
LOGISTIC_C = 0.3
CONTENT_MATCH_THRESHOLD = 0.5


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=LOGISTIC_C,
                    penalty="l2",
                    solver="liblinear",
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )


def _metrics(y: np.ndarray, probability: np.ndarray, prior: np.ndarray) -> dict[str, Any]:
    prediction = probability >= DECISION_THRESHOLD
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "positive_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "train_fold_prevalence_prior_brier": float(brier_score_loss(y, prior)),
        "brier_gain_vs_train_fold_prior": float(
            brier_score_loss(y, prior) - brier_score_loss(y, probability)
        ),
        "predicted_on": int(prediction.sum()),
        "predicted_off": int((~prediction).sum()),
        "minority_decision_fraction": float(min(prediction.mean(), (~prediction).mean())),
    }


def _oof(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, *, logo: bool) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros(len(y), dtype=float)
    prior = np.zeros(len(y), dtype=float)
    splitter = LeaveOneGroupOut() if logo else StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=seed
    )
    for train, test in splitter.split(x, y, groups):
        if len(set(y[train].tolist())) != 2:
            raise RuntimeError("an OOF training fold contains only one class")
        model = _pipeline(seed)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
        prior[test] = float(y[train].mean())
    return probability, prior


def _agreement(predictions: list[np.ndarray]) -> float:
    matrix = np.stack(predictions, axis=0)
    return float(np.mean(np.logical_or(np.all(matrix, axis=0), np.all(~matrix, axis=0))))


def _fit_component(component: str, fit: list[dict[str, Any]], confirmation: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    feature_names = tuple(sorted(fit[0]["model_features"]))
    if any(tuple(sorted(row["model_features"])) != feature_names for row in [*fit, *confirmation]):
        raise RuntimeError(f"{component} feature schema changed across fit/V3")
    x = np.asarray([[float(row["model_features"][name]) for name in feature_names] for row in fit])
    y = np.asarray([int(row["opportunity_target"]) for row in fit])
    users = np.asarray([row["user_id_private_not_model_input"] for row in fit], dtype=object)
    conditions = np.asarray([row["condition_family_private_not_model_input"] for row in fit], dtype=object)
    x_confirmation = np.asarray([[float(row["model_features"][name]) for name in feature_names] for row in confirmation])
    y_confirmation = np.asarray([int(row["opportunity_target"]) for row in confirmation])

    reports: list[dict[str, Any]] = []
    oof_decisions: list[np.ndarray] = []
    condition_decisions: list[np.ndarray] = []
    confirmation_decisions: list[np.ndarray] = []
    final_models: list[Pipeline] = []
    for seed in SEEDS:
        oof_probability, oof_prior = _oof(x, y, users, seed, logo=False)
        condition_probability, condition_prior = _oof(x, y, conditions, seed, logo=True)
        model = _pipeline(seed)
        model.fit(x, y)
        confirmation_probability = model.predict_proba(x_confirmation)[:, 1]
        confirmation_prior = np.full(len(y_confirmation), float(y.mean()))
        reports.append(
            {
                "seed": seed,
                "user_grouped_oof": _metrics(y, oof_probability, oof_prior),
                "leave_condition_family_out": _metrics(y, condition_probability, condition_prior),
                "untouched_v3_confirmation": _metrics(y_confirmation, confirmation_probability, confirmation_prior),
            }
        )
        oof_decisions.append(oof_probability >= DECISION_THRESHOLD)
        condition_decisions.append(condition_probability >= DECISION_THRESHOLD)
        confirmation_decisions.append(confirmation_probability >= DECISION_THRESHOLD)
        final_models.append(model)

    oof = [row["user_grouped_oof"] for row in reports]
    condition = [row["leave_condition_family_out"] for row in reports]
    confirm = [row["untouched_v3_confirmation"] for row in reports]
    checks = {
        "48_fit_rows_48_users_24_on_24_off": len(fit) == 48 and len(set(users.tolist())) == 48 and Counter(y.tolist()) == Counter({0: 24, 1: 24}),
        "16_v3_rows_16_users_8_on_8_off": len(confirmation) == 16 and len({row["user_id_private_not_model_input"] for row in confirmation}) == 16 and Counter(y_confirmation.tolist()) == Counter({0: 8, 1: 8}),
        "fit_v3_users_disjoint": not (set(users.tolist()) & {row["user_id_private_not_model_input"] for row in confirmation}),
        "worst_five_seed_user_grouped_ba_min_0_70": min(row["balanced_accuracy"] for row in oof) >= 0.70,
        "worst_five_seed_positive_recall_min_0_65": min(row["positive_recall"] for row in oof) >= 0.65,
        "worst_five_seed_specificity_min_0_65": min(row["specificity"] for row in oof) >= 0.65,
        "every_seed_oof_brier_beats_train_fold_prior": min(row["brier_gain_vs_train_fold_prior"] for row in oof) > 0.0,
        "five_seed_oof_decision_agreement_min_0_90": _agreement(oof_decisions) >= 0.90,
        "worst_five_seed_leave_condition_ba_min_0_65": min(row["balanced_accuracy"] for row in condition) >= 0.65,
        "untouched_v3_ba_min_0_70": min(row["balanced_accuracy"] for row in confirm) >= 0.70,
        "untouched_v3_recall_specificity_min_0_65": min(row["positive_recall"] for row in confirm) >= 0.65 and min(row["specificity"] for row in confirm) >= 0.65,
        "v3_minority_decision_fraction_min_0_20": min(row["minority_decision_fraction"] for row in confirm) >= 0.20,
        "five_seed_v3_decision_agreement_min_0_90": _agreement(confirmation_decisions) >= 0.90,
        "no_outcome_fields_read": all(not row["outcome_read"] for row in fit),
    }
    model_path = out_dir / f"{component.lower()}_opportunity_router.joblib"
    joblib.dump(final_models[0], model_path)
    rule_prediction = np.asarray([
        float(row["model_features"]["candidate_content_match_level"]) >= CONTENT_MATCH_THRESHOLD
        and float(row["model_features"]["candidate_grounding_or_nonredundancy_score"]) >= 0.999
        for row in fit
    ], dtype=float)
    return {
        "status": "PASS_PROMOTED" if all(checks.values()) else "FAIL_NOT_PROMOTED",
        "feature_names": list(feature_names),
        "fixed_hyperparameters": {
            "model": "standardized_l2_logistic_regression",
            "C": LOGISTIC_C,
            "decision_threshold": DECISION_THRESHOLD,
            "seeds": list(SEEDS),
            "content_match_levels": [0.0, 0.5, 1.0],
            "no_post_confirmation_floor": True,
        },
        "counts": {
            "fit_rows": len(fit),
            "confirmation_rows": len(confirmation),
            "fit_condition_families": len(set(conditions.tolist())),
        },
        "transparent_rule_same_fit_rows": _metrics(y, rule_prediction, np.full(len(y), float(y.mean()))),
        "five_seed_user_grouped_decision_agreement": _agreement(oof_decisions),
        "five_seed_condition_holdout_decision_agreement": _agreement(condition_decisions),
        "five_seed_v3_decision_agreement": _agreement(confirmation_decisions),
        "seed_reports": reports,
        "checks": checks,
        "model_path": str(model_path.relative_to(ROOT)),
    }


def build(*, root: Path = ROOT) -> dict[str, Any]:
    fit_path = root / "outputs/pm_v1_5_scale_stable_memory_fit_recompile_v1/fit_rows.jsonl"
    confirmation_path = root / "outputs/pm_v1_5_memory_opportunity_scale_stable_confirmation_v3/opportunity_rows.jsonl"
    fit_rows = _rows(fit_path)
    confirmation_rows = [row for row in _rows(confirmation_path) if row["evidence_role"] == "untouched_confirmation"]
    out_dir = root / "outputs/pm_v1_5_scale_stable_memory_opportunity_heads_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        component: _fit_component(
            component,
            [row for row in fit_rows if row["component"] == component],
            [row for row in confirmation_rows if row["component"] == component],
            out_dir,
        )
        for component in COMPONENTS
    }
    status = "PASS_THREE_SCALE_STABLE_MEMORY_HEADS_PROMOTED" if all(row["status"] == "PASS_PROMOTED" for row in reports.values()) else "FAIL_SCALE_STABLE_MEMORY_HEADS"
    write_jsonl(out_dir / "fit_rows_audit.jsonl", fit_rows)
    write_jsonl(out_dir / "confirmation_rows_audit.jsonl", confirmation_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scientific_scope": "Outcome-blind post-candidate-discovery, pre-injection opportunity routing; not item correctness, generator uptake, or external utility.",
        "components": reports,
        "repair_budget": "ONE_SCALE_STABLE_REPRESENTATION_REPAIR_CONSUMED",
        "v1_v2_confirmation_reused": False,
        "external_quality_risk_or_response_outcome_read": False,
        "inputs": {str(fit_path.relative_to(root)): sha256_file(fit_path), str(confirmation_path.relative_to(root)): sha256_file(confirmation_path)},
    }
    write_json(out_dir / "fit_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "fit_report_sha256": sha256_file(out_dir / "fit_report.json"),
            "fit_rows_sha256": sha256_file(out_dir / "fit_rows_audit.jsonl"),
            "confirmation_rows_sha256": sha256_file(out_dir / "confirmation_rows_audit.jsonl"),
            "models": {component: sha256_file(out_dir / f"{component.lower()}_opportunity_router.joblib") for component in COMPONENTS},
            "v3_used_for_feature_or_hyperparameter_selection": False,
            "external_outcome_used": False,
        },
    )
    return report


def main() -> None:
    report = build()
    print({"protocol": report["protocol"], "status": report["status"], "components": {key: value["status"] for key, value in report["components"].items()}})


if __name__ == "__main__":
    main()
