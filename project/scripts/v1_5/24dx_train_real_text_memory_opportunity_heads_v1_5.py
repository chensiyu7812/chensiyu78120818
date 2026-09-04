#!/usr/bin/env python3
"""Fit the production-feature MP/MS/ME opportunity heads.

The fit combines the 32 frozen D3 real-text users with 16 newly frozen users.
The separate 16-user confirmation split is read only after feature names,
regularization, seeds, and the 0.5 decision threshold are fixed in this file.
No response, human quality preference, risk decision, ESConv outcome, or
EvoEmo outcome is read.
"""

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
PROTOCOL = "pm-v1.5-real-text-memory-opportunity-head-fit-v1"
COMPONENTS = ("MP", "MS", "ME")
SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
DECISION_THRESHOLD = 0.5
LOGISTIC_C = 0.3
TRANSPARENT_MATCH_THRESHOLD = 0.10
POSITIVE_MATCH_FLOOR_QUANTILE = 0.20
FLOOR_COMPONENTS = frozenset({"MP", "MS"})


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


def _metrics(
    y: np.ndarray, probability: np.ndarray, prior_probability: np.ndarray
) -> dict[str, Any]:
    prediction = probability >= DECISION_THRESHOLD
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "positive_recall": float(
            recall_score(y, prediction, pos_label=1, zero_division=0)
        ),
        "specificity": float(
            recall_score(y, prediction, pos_label=0, zero_division=0)
        ),
        "brier": float(brier_score_loss(y, probability)),
        "train_fold_prevalence_prior_brier": float(
            brier_score_loss(y, prior_probability)
        ),
        "brier_gain_vs_train_fold_prior": float(
            brier_score_loss(y, prior_probability)
            - brier_score_loss(y, probability)
        ),
        "predicted_on": int(prediction.sum()),
        "predicted_off": int((~prediction).sum()),
        "minority_decision_fraction": float(
            min(prediction.mean(), (~prediction).mean())
        ),
    }


def _oof(
    *,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    leave_one_group_out: bool,
    rows: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros(len(y), dtype=float)
    prior = np.zeros(len(y), dtype=float)
    splitter = (
        LeaveOneGroupOut()
        if leave_one_group_out
        else StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    )
    for train, test in splitter.split(x, y, groups):
        if len(set(y[train].tolist())) != 2:
            raise RuntimeError("an OOF training fold contains only one class")
        model = _pipeline(seed)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
        if rows is not None:
            train_positive_scores = np.asarray(
                [
                    float(rows[index]["model_features"]["candidate_state_match_score"])
                    for index in train
                    if y[index] == 1
                ],
                dtype=float,
            )
            floor = float(
                np.quantile(
                    train_positive_scores,
                    POSITIVE_MATCH_FLOOR_QUANTILE,
                    method="lower",
                )
            )
            for index in test:
                if (
                    float(
                        rows[index]["model_features"][
                            "candidate_state_match_score"
                        ]
                    )
                    < floor
                ):
                    probability[index] = 0.0
        prior[test] = float(y[train].mean())
    return probability, prior


def _transparent_rule(
    *, rows: list[dict[str, Any]], y: np.ndarray
) -> dict[str, Any]:
    prediction = np.asarray(
        [
            float(row["model_features"]["candidate_state_match_score"])
            >= TRANSPARENT_MATCH_THRESHOLD
            and float(
                row["model_features"][
                    "candidate_grounding_or_nonredundancy_score"
                ]
            )
            >= 0.999
            for row in rows
        ],
        dtype=bool,
    )
    probability = prediction.astype(float)
    prior = np.full(len(y), float(y.mean()), dtype=float)
    return _metrics(y, probability, prior)


def _apply_trained_lexical_floor(
    probability: np.ndarray,
    rows: list[dict[str, Any]],
    floor: float,
) -> np.ndarray:
    """Fail closed below a floor estimated from fit positives only.

    The first confirmation motivated this repair and is therefore consumed.
    The floor itself is the lower empirical quintile of positive fit scores;
    grouped OOF computes it inside each train fold and final confirmation uses
    the full-fit value.  No confirmation or external value selects the floor.
    """

    result = probability.copy()
    for index, row in enumerate(rows):
        if (
            float(row["model_features"]["candidate_state_match_score"])
            < floor
        ):
            result[index] = 0.0
    return result


def _component_rows(
    component: str,
    d3_rows: list[dict[str, Any]],
    fit_supplement_rows: list[dict[str, Any]],
    confirmation_supplement_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = [row for row in d3_rows if row["component"] == component]
    fit_supplement = [
        row
        for row in fit_supplement_rows
        if row["component"] == component
        and row["evidence_role"] == "supplemental_fit"
    ]
    confirmation = [
        row
        for row in confirmation_supplement_rows
        if row["component"] == component
        and row["evidence_role"] == "untouched_confirmation"
    ]
    fit: list[dict[str, Any]] = []
    for row in base:
        fit.append(
            {
                "component": component,
                "evidence_source": "d3_step0_real_text",
                "user_id_private_not_model_input": row["user_id"],
                "condition_family_private_not_model_input": (
                    "d3_" + row["condition_family_private_not_model_input"]
                ),
                "opportunity_target": int(row["opportunity_target"]),
                "model_features": dict(row["model_features"]),
                "outcome_read": False,
            }
        )
    for row in fit_supplement:
        fit.append(
            {
                "component": component,
                "evidence_source": "real_text_supplement",
                "user_id_private_not_model_input": row[
                    "user_id_private_not_model_input"
                ],
                "condition_family_private_not_model_input": row[
                    "condition_family_private_not_model_input"
                ],
                "opportunity_target": int(row["opportunity_target"]),
                "model_features": dict(row["model_features"]),
                "outcome_read": False,
            }
        )
    return fit, confirmation


def _fit_component(
    *, component: str, fit: list[dict[str, Any]], confirmation: list[dict[str, Any]], out_dir: Path
) -> dict[str, Any]:
    feature_names = tuple(sorted(fit[0]["model_features"]))
    if any(tuple(sorted(row["model_features"])) != feature_names for row in [*fit, *confirmation]):
        raise RuntimeError(f"{component} feature schema changed across splits")
    x = np.asarray(
        [[float(row["model_features"][name]) for name in feature_names] for row in fit],
        dtype=float,
    )
    y = np.asarray([row["opportunity_target"] for row in fit], dtype=int)
    users = np.asarray(
        [row["user_id_private_not_model_input"] for row in fit], dtype=object
    )
    conditions = np.asarray(
        [row["condition_family_private_not_model_input"] for row in fit],
        dtype=object,
    )
    x_confirmation = np.asarray(
        [
            [float(row["model_features"][name]) for name in feature_names]
            for row in confirmation
        ],
        dtype=float,
    )
    y_confirmation = np.asarray(
        [row["opportunity_target"] for row in confirmation], dtype=int
    )

    seed_reports: list[dict[str, Any]] = []
    oof_predictions: list[np.ndarray] = []
    condition_predictions: list[np.ndarray] = []
    confirmation_predictions: list[np.ndarray] = []
    final_models: list[Pipeline] = []
    for seed in SEEDS:
        oof_probability, oof_prior = _oof(
            x=x,
            y=y,
            groups=users,
            seed=seed,
            leave_one_group_out=False,
            rows=fit if component in FLOOR_COMPONENTS else None,
        )
        condition_probability, condition_prior = _oof(
            x=x,
            y=y,
            groups=conditions,
            seed=seed,
            leave_one_group_out=True,
            rows=fit if component in FLOOR_COMPONENTS else None,
        )
        model = _pipeline(seed)
        model.fit(x, y)
        confirmation_probability = model.predict_proba(x_confirmation)[:, 1]
        final_match_floor = float(
            np.quantile(
                [
                    float(row["model_features"]["candidate_state_match_score"])
                    for row in fit
                    if row["opportunity_target"] == 1
                ],
                POSITIVE_MATCH_FLOOR_QUANTILE,
                method="lower",
            )
        )
        if component in FLOOR_COMPONENTS:
            confirmation_probability = _apply_trained_lexical_floor(
                confirmation_probability, confirmation, final_match_floor
            )
        confirmation_prior = np.full(
            len(y_confirmation), float(y.mean()), dtype=float
        )
        final_models.append(model)
        oof_predictions.append(oof_probability >= DECISION_THRESHOLD)
        condition_predictions.append(
            condition_probability >= DECISION_THRESHOLD
        )
        confirmation_predictions.append(
            confirmation_probability >= DECISION_THRESHOLD
        )
        seed_reports.append(
            {
                "seed": seed,
                "user_grouped_oof": _metrics(y, oof_probability, oof_prior),
                "leave_condition_family_out": _metrics(
                    y, condition_probability, condition_prior
                ),
                "untouched_confirmation": _metrics(
                    y_confirmation,
                    confirmation_probability,
                    confirmation_prior,
                ),
            }
        )

    def agreement(values: list[np.ndarray]) -> float:
        matrix = np.stack(values, axis=0)
        return float(
            np.mean(
                np.logical_or(
                    np.all(matrix, axis=0), np.all(~matrix, axis=0)
                )
            )
        )

    oof = [row["user_grouped_oof"] for row in seed_reports]
    condition = [row["leave_condition_family_out"] for row in seed_reports]
    confirm = [row["untouched_confirmation"] for row in seed_reports]
    checks = {
        "48_fit_rows_48_users_24_on_24_off": (
            len(fit) == 48
            and len(set(users.tolist())) == 48
            and Counter(y.tolist()) == Counter({0: 24, 1: 24})
        ),
        "16_confirmation_rows_16_users_8_on_8_off": (
            len(confirmation) == 16
            and len(
                {
                    row["user_id_private_not_model_input"]
                    for row in confirmation
                }
            )
            == 16
            and Counter(y_confirmation.tolist()) == Counter({0: 8, 1: 8})
        ),
        "fit_confirmation_users_disjoint": not (
            set(users.tolist())
            & {
                row["user_id_private_not_model_input"]
                for row in confirmation
            }
        ),
        "worst_five_seed_user_grouped_ba_min_0_70": min(
            row["balanced_accuracy"] for row in oof
        )
        >= 0.70,
        "worst_five_seed_positive_recall_min_0_65": min(
            row["positive_recall"] for row in oof
        )
        >= 0.65,
        "worst_five_seed_specificity_min_0_65": min(
            row["specificity"] for row in oof
        )
        >= 0.65,
        "every_seed_oof_brier_beats_train_fold_prior": min(
            row["brier_gain_vs_train_fold_prior"] for row in oof
        )
        > 0.0,
        "five_seed_oof_decision_agreement_min_0_90": agreement(oof_predictions)
        >= 0.90,
        "worst_five_seed_leave_condition_ba_min_0_65": min(
            row["balanced_accuracy"] for row in condition
        )
        >= 0.65,
        "untouched_confirmation_ba_min_0_70": min(
            row["balanced_accuracy"] for row in confirm
        )
        >= 0.70,
        "untouched_confirmation_recall_specificity_min_0_65": (
            min(row["positive_recall"] for row in confirm) >= 0.65
            and min(row["specificity"] for row in confirm) >= 0.65
        ),
        "confirmation_minority_decision_fraction_min_0_20": min(
            row["minority_decision_fraction"] for row in confirm
        )
        >= 0.20,
        "five_seed_confirmation_decision_agreement_min_0_90": agreement(
            confirmation_predictions
        )
        >= 0.90,
        "no_outcome_fields_read": all(
            not row["outcome_read"] for row in fit
        ),
    }
    promoted = all(checks.values())
    model_path = out_dir / f"{component.lower()}_opportunity_router.joblib"
    joblib.dump(final_models[0], model_path)
    return {
        "status": "PASS_PROMOTED" if promoted else "FAIL_NOT_PROMOTED",
        "feature_names": list(feature_names),
        "fixed_hyperparameters": {
            "model": "standardized_l2_logistic_regression",
            "C": LOGISTIC_C,
            "decision_threshold": DECISION_THRESHOLD,
            "candidate_match_floor_rule": (
                "MP/MS only: lower empirical 20th percentile of positive fit rows, "
                "estimated inside each OOF train fold; ME has no lexical floor"
            ),
            "final_fit_candidate_match_floor": (
                final_match_floor if component in FLOOR_COMPONENTS else None
            ),
            "seeds": list(SEEDS),
        },
        "counts": {
            "fit_rows": len(fit),
            "confirmation_rows": len(confirmation),
            "fit_class_counts": dict(sorted(Counter(y.tolist()).items())),
            "confirmation_class_counts": dict(
                sorted(Counter(y_confirmation.tolist()).items())
            ),
            "fit_condition_families": len(set(conditions.tolist())),
        },
        "transparent_rule_same_fit_rows": _transparent_rule(rows=fit, y=y),
        "five_seed_user_grouped_decision_agreement": agreement(oof_predictions),
        "five_seed_condition_holdout_decision_agreement": agreement(
            condition_predictions
        ),
        "five_seed_confirmation_decision_agreement": agreement(
            confirmation_predictions
        ),
        "seed_reports": seed_reports,
        "checks": checks,
        "model_path": str(model_path.relative_to(ROOT)),
    }


def build(*, root: Path = ROOT) -> dict[str, Any]:
    d3_path = (
        root
        / "outputs/pm_v1_5_d3_memory_opportunity_recompile_v1/production_compiled_opportunity_rows.jsonl"
    )
    fit_supplement_path = (
        root
        / "outputs/pm_v1_5_memory_opportunity_supplement_v1/opportunity_rows.jsonl"
    )
    confirmation_supplement_path = (
        root
        / "outputs/pm_v1_5_memory_opportunity_fresh_confirmation_v2/opportunity_rows.jsonl"
    )
    d3_rows = _rows(d3_path)
    fit_supplement_rows = _rows(fit_supplement_path)
    confirmation_supplement_rows = _rows(confirmation_supplement_path)
    out_dir = root / "outputs/pm_v1_5_real_text_memory_opportunity_heads_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, Any]] = {}
    fit_audit: list[dict[str, Any]] = []
    confirmation_audit: list[dict[str, Any]] = []
    for component in COMPONENTS:
        fit, confirmation = _component_rows(
            component,
            d3_rows,
            fit_supplement_rows,
            confirmation_supplement_rows,
        )
        reports[component] = _fit_component(
            component=component,
            fit=fit,
            confirmation=confirmation,
            out_dir=out_dir,
        )
        fit_audit.extend(fit)
        confirmation_audit.extend(confirmation)
    write_jsonl(out_dir / "fit_rows_audit.jsonl", fit_audit)
    write_jsonl(out_dir / "confirmation_rows_audit.jsonl", confirmation_audit)
    all_promoted = all(
        report["status"] == "PASS_PROMOTED" for report in reports.values()
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "PASS_THREE_MEMORY_HEADS_PROMOTED"
            if all_promoted
            else "FAIL_ONE_OR_MORE_MEMORY_HEADS_NOT_PROMOTED"
        ),
        "scientific_scope": (
            "Outcome-blind opportunity routing after production candidate discovery. "
            "This does not establish item retrieval quality, generator uptake, or system benefit."
        ),
        "components": reports,
        "external_outcome_retuning": False,
        "human_response_label_read": False,
        "confirmation_status": "FRESH_V2_READ_ONCE_AFTER_REPAIR_FREEZE",
        "repair_lineage": (
            "V1 confirmation was consumed by the lexical-only diagnosis and is not "
            "reused. BGE was rejected on fit-only qualification. MP/MS then froze a "
            "train-derived positive-quintile lexical floor; ME retained logistic-only."
        ),
        "claim_boundary": {
            "allowed_if_pass": (
                "The memory heads learned the prespecified bounded opportunity definition "
                "on user- and semantic-family-disjoint controlled text."
            ),
            "requires_later_system_test": (
                "Whether the frozen heads improve the quality-risk-cost tradeoff on ESConv/EvoEmo."
            ),
        },
        "inputs": {
            str(d3_path.relative_to(root)): sha256_file(d3_path),
            str(fit_supplement_path.relative_to(root)): sha256_file(
                fit_supplement_path
            ),
            str(confirmation_supplement_path.relative_to(root)): sha256_file(
                confirmation_supplement_path
            ),
        },
    }
    write_json(out_dir / "fit_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "fit_report_sha256": sha256_file(out_dir / "fit_report.json"),
            "fit_rows_sha256": sha256_file(out_dir / "fit_rows_audit.jsonl"),
            "confirmation_rows_sha256": sha256_file(
                out_dir / "confirmation_rows_audit.jsonl"
            ),
            "models": {
                component: sha256_file(
                    out_dir / f"{component.lower()}_opportunity_router.joblib"
                )
                for component in COMPONENTS
            },
            "fresh_v2_confirmation_used_for_feature_threshold_or_hyperparameter_selection": False,
            "fresh_v2_confirmation_reusable_after_this_run": False,
            "external_outcome_used": False,
        },
    )
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "components": {
                component: {
                    "status": value["status"],
                    "checks": value["checks"],
                }
                for component, value in report["components"].items()
            },
        }
    )


if __name__ == "__main__":
    main()
