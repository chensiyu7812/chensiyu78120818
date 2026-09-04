#!/usr/bin/env python3
"""Train and evaluate four leakage-safe component-effect PM heads."""

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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.contracts import MemorySource, StrategyMode, parse_action_id
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-transport-repaired-four-component-pm-fit-v1"
COMPONENTS = ("RS", "MP", "MS", "ME")
MEMORY_COMPONENTS = ("MP", "MS", "ME")
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
N_SPLITS = 5
C_VALUE = 0.3
THRESHOLD = 0.5
MIN_DECISION_FRACTION = 0.20
MEMORY_FEATURES = (
    "top1_relevance_bucket",
    "relevance_margin_bucket",
    "token_cost_bucket",
    "retrieved_fraction",
    "minimum_relative_age_bucket",
    "maximum_relative_age_bucket",
    "background_other_memory_count",
    "background_strategy_on",
)
RS_FEATURES = (
    "retrieval_score",
    "execution_profile_minimal",
    "family_is_question",
    "family_is_suggestion",
    "advice_welcome",
    "low_burden_or_listen_only",
    "background_MP_on",
    "background_MS_on",
    "background_ME_on",
)
FORBIDDEN_FEATURE_TOKENS = (
    "target",
    "quality",
    "risk",
    "response",
    "judge",
    "split",
    "user_id",
    "state_id",
    "semantic_family",
    "needed",
    "outcome",
    "external",
    "memory_id",
    "card_id",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _pipeline(seed: int) -> Pipeline:
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


def _group_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    weights = np.asarray(
        [1.0 / counts[value] for value in groups],
        dtype=float,
    )
    return weights * (len(weights) / weights.sum())


def _weighted_prevalence(
    y: np.ndarray,
    groups: np.ndarray,
) -> float:
    weights = _group_weights(groups)
    return float(np.average(y, weights=weights))


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    prior_probability: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    weights = _group_weights(groups)
    action = probability >= THRESHOLD
    brier = float(
        brier_score_loss(y, probability, sample_weight=weights)
    )
    prior_brier = float(
        brier_score_loss(y, prior_probability, sample_weight=weights)
    )
    return {
        "threshold": THRESHOLD,
        "group_weighted_log_loss": float(
            log_loss(
                y,
                probability,
                labels=[0, 1],
                sample_weight=weights,
            )
        ),
        "group_weighted_prior_log_loss": float(
            log_loss(
                y,
                prior_probability,
                labels=[0, 1],
                sample_weight=weights,
            )
        ),
        "group_weighted_brier": brier,
        "group_weighted_prior_brier": prior_brier,
        "group_weighted_brier_gain_vs_prevalence_prior": (
            prior_brier - brier
        ),
        "group_weighted_accuracy": float(
            accuracy_score(y, action, sample_weight=weights)
        ),
        "group_weighted_balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                action,
                sample_weight=weights,
            )
        ),
        "group_weighted_positive_recall": float(
            recall_score(
                y,
                action,
                pos_label=1,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "row_accuracy": float(accuracy_score(y, action)),
        "row_balanced_accuracy": float(
            balanced_accuracy_score(y, action)
        ),
        "predicted_on": int(action.sum()),
        "predicted_off": int((~action).sum()),
        "predicted_on_fraction": float(action.mean()),
        "predicted_off_fraction": float((~action).mean()),
    }


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Pipeline:
    model = _pipeline(seed)
    model.fit(
        x,
        y,
        logistic__sample_weight=_group_weights(groups),
    )
    return model


def _grouped_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    probability = np.zeros(len(y), dtype=float)
    prior = np.zeros(len(y), dtype=float)
    folds = np.zeros(len(y), dtype=int)
    train_test_group_overlap: list[int] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups),
        start=1,
    ):
        train_groups = groups[train_index]
        test_groups = groups[test_index]
        overlap = set(train_groups.tolist()) & set(test_groups.tolist())
        train_test_group_overlap.append(len(overlap))
        if len(set(y[train_index].tolist())) != 2:
            raise RuntimeError(
                f"OOF training fold has one class only: seed={seed}, "
                f"fold={fold}"
            )
        model = _fit(
            x[train_index],
            y[train_index],
            train_groups,
            seed + fold,
        )
        probability[test_index] = model.predict_proba(
            x[test_index]
        )[:, 1]
        prior[test_index] = _weighted_prevalence(
            y[train_index],
            train_groups,
        )
        folds[test_index] = fold
    if any(train_test_group_overlap):
        raise RuntimeError("user group leaked across OOF fold")
    return {
        "seed": seed,
        "probability": probability,
        "prior_probability": prior,
        "fold": folds,
        "metrics": _metrics(y, probability, prior, groups),
    }


def _memory_values(
    blueprint: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, float]:
    model_features = dict(descriptor["model_features"])
    background_sources, background_strategy = parse_action_id(
        str(blueprint["control_action"])
    )
    return {
        "top1_relevance_bucket": float(
            model_features["top1_relevance_bucket"]
        ),
        "relevance_margin_bucket": float(
            model_features["relevance_margin_bucket"]
        ),
        "token_cost_bucket": float(
            model_features["token_cost_bucket"]
        ),
        "retrieved_fraction": float(
            model_features["retrieved_fraction"]
        ),
        "minimum_relative_age_bucket": float(
            model_features["minimum_relative_age_bucket"]
        ),
        "maximum_relative_age_bucket": float(
            model_features["maximum_relative_age_bucket"]
        ),
        "background_other_memory_count": float(len(background_sources)),
        "background_strategy_on": float(
            background_strategy is StrategyMode.RS
        ),
    }


def _rs_values(blueprint: dict[str, Any]) -> dict[str, float]:
    candidate = dict(blueprint["current_strategy_candidate"])
    flags = dict(candidate["observable_flags"])
    background_sources, _ = parse_action_id(
        str(blueprint["control_action"])
    )
    family = str(candidate["strategy_family"])
    return {
        "retrieval_score": float(candidate["retrieval_score"]),
        "execution_profile_minimal": float(
            candidate["execution_profile"] == "minimal"
        ),
        "family_is_question": float(family == "Question"),
        "family_is_suggestion": float(family == "Providing Suggestions"),
        "advice_welcome": float(bool(flags["advice_welcome"])),
        "low_burden_or_listen_only": float(
            bool(flags["low_burden"] or flags["listen_only"])
        ),
        "background_MP_on": float(MemorySource.MP in background_sources),
        "background_MS_on": float(MemorySource.MS in background_sources),
        "background_ME_on": float(MemorySource.ME in background_sources),
    }


def _transparent_action(
    component: str,
    values: list[dict[str, float]],
) -> np.ndarray:
    if component == "RS":
        return np.asarray(
            [
                bool(
                    row["advice_welcome"]
                    or row["low_burden_or_listen_only"]
                )
                for row in values
            ],
            dtype=bool,
        )
    return np.asarray(
        [
            bool(
                row["top1_relevance_bucket"] >= 3
                and row["relevance_margin_bucket"] >= 2
                and row["token_cost_bucket"] <= 4
            )
            for row in values
        ],
        dtype=bool,
    )


def _binary_baseline_metrics(
    y: np.ndarray,
    action: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    weights = _group_weights(groups)
    return {
        "group_weighted_accuracy": float(
            accuracy_score(y, action, sample_weight=weights)
        ),
        "group_weighted_balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                action,
                sample_weight=weights,
            )
        ),
        "group_weighted_positive_recall": float(
            recall_score(
                y,
                action,
                pos_label=1,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "predicted_on": int(action.sum()),
        "predicted_off": int((~action).sum()),
    }


def _gate(metrics: dict[str, Any], seed_ba_std: float) -> dict[str, bool]:
    return {
        "balanced_accuracy_at_least_0_70": (
            metrics["group_weighted_balanced_accuracy"] >= 0.70
        ),
        "positive_recall_at_least_0_60": (
            metrics["group_weighted_positive_recall"] >= 0.60
        ),
        "brier_beats_prevalence_prior": (
            metrics["group_weighted_brier"]
            < metrics["group_weighted_prior_brier"]
        ),
        "predicted_on_fraction_at_least_0_20": (
            metrics["predicted_on_fraction"]
            >= MIN_DECISION_FRACTION
        ),
        "predicted_off_fraction_at_least_0_20": (
            metrics["predicted_off_fraction"]
            >= MIN_DECISION_FRACTION
        ),
        "five_seed_ba_std_at_most_0_03": seed_ba_std <= 0.03,
    }


def _support(
    x_development: np.ndarray,
    x_evaluation: np.ndarray,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    minimum = x_development.min(axis=0)
    maximum = x_development.max(axis=0)
    row_in_support = np.all(
        (x_evaluation >= minimum) & (x_evaluation <= maximum),
        axis=1,
    )
    return {
        "feature_minimum": {
            name: float(minimum[index])
            for index, name in enumerate(feature_names)
        },
        "feature_maximum": {
            name: float(maximum[index])
            for index, name in enumerate(feature_names)
        },
        "evaluation_rows": len(x_evaluation),
        "joint_absolute_range_in_support_rows": int(
            row_in_support.sum()
        ),
        "joint_absolute_range_in_support_fraction": float(
            row_in_support.mean()
        ),
        "row_in_support": row_in_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_final_effect_labels_v1/"
        "component_effect_labels.jsonl",
    )
    parser.add_argument(
        "--memory-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1/"
        "memory_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--rs-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1/"
        "rs_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--candidate-descriptors",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "candidate_descriptors.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1",
    )
    args = parser.parse_args()

    labels = _rows(args.labels)
    blueprints = {
        str(row["contrast_slot_id"]): row
        for row in (
            _rows(args.memory_blueprint) + _rows(args.rs_blueprint)
        )
    }
    descriptors = {
        (
            str(row["state_id"]),
            str(row["candidate_descriptor"]["source"]),
        ): dict(row["candidate_descriptor"])
        for row in _rows(args.candidate_descriptors)
    }
    if len(labels) != 256 or len(blueprints) != 256:
        raise RuntimeError("expected 256 labels and 256 blueprints")
    if any(row["target_y"] not in {0, 1} for row in labels):
        raise RuntimeError("unknown labels cannot enter the primary fit")
    if set(blueprints) != {
        str(row["contrast_slot_id"]) for row in labels
    }:
        raise RuntimeError("label/blueprint contrast identities differ")

    feature_names_by_component = {
        "RS": RS_FEATURES,
        "MP": MEMORY_FEATURES,
        "MS": MEMORY_FEATURES,
        "ME": MEMORY_FEATURES,
    }
    if any(
        token in feature.casefold()
        for features in feature_names_by_component.values()
        for feature in features
        for token in FORBIDDEN_FEATURE_TOKENS
    ):
        raise RuntimeError("forbidden leakage token in feature contract")

    reports: dict[str, Any] = {}
    model_artifact: dict[str, Any] = {
        "protocol": PROTOCOL,
        "model_family": "four independent L2 logistic regressions",
        "C": C_VALUE,
        "threshold": THRESHOLD,
        "seeds": list(SEEDS),
        "heads": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for component in COMPONENTS:
        component_rows = [
            row for row in labels if row["component"] == component
        ]
        if len(component_rows) != 64:
            raise RuntimeError(f"{component}: expected 64 labels")
        values: list[dict[str, float]] = []
        for row in component_rows:
            blueprint = blueprints[str(row["contrast_slot_id"])]
            if component == "RS":
                values.append(_rs_values(blueprint))
            else:
                descriptor = descriptors[
                    (str(row["state_id"]), component)
                ]
                values.append(_memory_values(blueprint, descriptor))
        feature_names = feature_names_by_component[component]
        x = np.asarray(
            [
                [row[name] for name in feature_names]
                for row in values
            ],
            dtype=float,
        )
        y = np.asarray(
            [int(row["target_y"]) for row in component_rows],
            dtype=int,
        )
        groups = np.asarray(
            [str(row["user_id"]) for row in component_rows],
            dtype=object,
        )
        split = np.asarray(
            [str(row["split"]) for row in component_rows],
            dtype=object,
        )
        development_index = np.flatnonzero(
            np.isin(split, ["train", "calibration"])
        )
        internal_index = np.flatnonzero(split == "internal_test")
        if len(development_index) != 48 or len(internal_index) != 16:
            raise RuntimeError(f"{component}: unexpected split sizes")
        if set(groups[development_index]) & set(groups[internal_index]):
            raise RuntimeError(
                f"{component}: development/internal user overlap"
            )

        x_dev = x[development_index]
        y_dev = y[development_index]
        groups_dev = groups[development_index]
        x_internal = x[internal_index]
        y_internal = y[internal_index]
        groups_internal = groups[internal_index]
        if len(set(y_dev.tolist())) != 2 or len(
            set(y_internal.tolist())
        ) != 2:
            raise RuntimeError(
                f"{component}: both classes required in both stages"
            )

        oof_runs = [
            _grouped_oof(x_dev, y_dev, groups_dev, seed)
            for seed in SEEDS
        ]
        oof_probability = np.mean(
            np.stack(
                [run["probability"] for run in oof_runs],
                axis=0,
            ),
            axis=0,
        )
        oof_prior = np.mean(
            np.stack(
                [run["prior_probability"] for run in oof_runs],
                axis=0,
            ),
            axis=0,
        )
        oof_metrics = _metrics(
            y_dev,
            oof_probability,
            oof_prior,
            groups_dev,
        )
        seed_ba = [
            float(
                run["metrics"][
                    "group_weighted_balanced_accuracy"
                ]
            )
            for run in oof_runs
        ]
        seed_ba_std = float(np.std(seed_ba, ddof=1))
        gate_checks = _gate(oof_metrics, seed_ba_std)

        fitted_models = [
            _fit(x_dev, y_dev, groups_dev, seed) for seed in SEEDS
        ]
        internal_probability = np.mean(
            np.stack(
                [
                    model.predict_proba(x_internal)[:, 1]
                    for model in fitted_models
                ],
                axis=0,
            ),
            axis=0,
        )
        internal_prior = np.full(
            len(y_internal),
            _weighted_prevalence(y_dev, groups_dev),
            dtype=float,
        )
        internal_metrics = _metrics(
            y_internal,
            internal_probability,
            internal_prior,
            groups_internal,
        )
        support = _support(x_dev, x_internal, feature_names)

        background_feature_names = (
            (
                "background_MP_on",
                "background_MS_on",
                "background_ME_on",
            )
            if component == "RS"
            else (
                "background_other_memory_count",
                "background_strategy_on",
            )
        )
        background_indices = [
            feature_names.index(name)
            for name in background_feature_names
        ]
        background_oof_runs = [
            _grouped_oof(
                x_dev[:, background_indices],
                y_dev,
                groups_dev,
                seed,
            )
            for seed in SEEDS
        ]
        background_probability = np.mean(
            np.stack(
                [run["probability"] for run in background_oof_runs],
                axis=0,
            ),
            axis=0,
        )
        background_prior = np.mean(
            np.stack(
                [run["prior_probability"] for run in background_oof_runs],
                axis=0,
            ),
            axis=0,
        )
        background_metrics = _metrics(
            y_dev,
            background_probability,
            background_prior,
            groups_dev,
        )
        transparent = _transparent_action(
            component,
            [values[index] for index in development_index],
        )
        baselines = {
            "always_off": _binary_baseline_metrics(
                y_dev,
                np.zeros(len(y_dev), dtype=bool),
                groups_dev,
            ),
            "always_on": _binary_baseline_metrics(
                y_dev,
                np.ones(len(y_dev), dtype=bool),
                groups_dev,
            ),
            "frozen_transparent_rule": _binary_baseline_metrics(
                y_dev,
                transparent,
                groups_dev,
            ),
            "background_only_grouped_oof": background_metrics,
        }
        coefficients = []
        for seed, model in zip(SEEDS, fitted_models, strict=True):
            scaler = model.named_steps["scale"]
            logistic = model.named_steps["logistic"]
            coefficients.append(
                {
                    "seed": seed,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "coefficient": logistic.coef_[0].tolist(),
                    "intercept": float(logistic.intercept_[0]),
                }
            )

        reports[component] = {
            "feature_names": list(feature_names),
            "development_rows": len(y_dev),
            "development_users": len(set(groups_dev.tolist())),
            "development_class_counts": dict(
                sorted(Counter(y_dev.tolist()).items())
            ),
            "internal_test_rows": len(y_internal),
            "internal_test_users": len(set(groups_internal.tolist())),
            "internal_test_class_counts": dict(
                sorted(Counter(y_internal.tolist()).items())
            ),
            "grouped_oof": {
                "metrics": oof_metrics,
                "balanced_accuracy_by_seed": seed_ba,
                "balanced_accuracy_seed_std": seed_ba_std,
                "gate_checks": gate_checks,
                "gate_passed": all(gate_checks.values()),
            },
            "internal_test_frozen_threshold": {
                "metrics": internal_metrics,
                "development_range_support": {
                    key: value
                    for key, value in support.items()
                    if key != "row_in_support"
                },
                "used_for_model_or_threshold_selection": False,
            },
            "baselines": baselines,
            "shortcut_diagnostic": {
                "background_only_ba": background_metrics[
                    "group_weighted_balanced_accuracy"
                ],
                "full_minus_background_only_ba": (
                    oof_metrics["group_weighted_balanced_accuracy"]
                    - background_metrics[
                        "group_weighted_balanced_accuracy"
                    ]
                ),
                "background_only_is_not_a_forbidden_feature_set": True,
                "interpretation": (
                    "If background-only matches or exceeds the full model, "
                    "the data do not show candidate-aware learning."
                ),
            },
        }
        model_artifact["heads"][component] = {
            "feature_names": list(feature_names),
            "C": C_VALUE,
            "threshold": THRESHOLD,
            "development_support_minimum": support["feature_minimum"],
            "development_support_maximum": support["feature_maximum"],
            "ensemble_members": coefficients,
        }

        oof_action = oof_probability >= THRESHOLD
        for local_index, global_index in enumerate(development_index):
            row = component_rows[int(global_index)]
            prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "component": component,
                    "stage": "development_grouped_oof",
                    "contrast_slot_id": row["contrast_slot_id"],
                    "state_id": row["state_id"],
                    "user_id": row["user_id"],
                    "split": row["split"],
                    "target_y": int(y_dev[local_index]),
                    "probability_on": float(
                        oof_probability[local_index]
                    ),
                    "predicted_on": bool(oof_action[local_index]),
                    "in_development_feature_support": True,
                }
            )
        internal_action = internal_probability >= THRESHOLD
        for local_index, global_index in enumerate(internal_index):
            row = component_rows[int(global_index)]
            prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "component": component,
                    "stage": "internal_test_frozen_threshold",
                    "contrast_slot_id": row["contrast_slot_id"],
                    "state_id": row["state_id"],
                    "user_id": row["user_id"],
                    "split": row["split"],
                    "target_y": int(y_internal[local_index]),
                    "probability_on": float(
                        internal_probability[local_index]
                    ),
                    "predicted_on": bool(
                        internal_action[local_index]
                    ),
                    "in_development_feature_support": bool(
                        support["row_in_support"][local_index]
                    ),
                }
            )

    all_primary_gates_pass = all(
        reports[component]["grouped_oof"]["gate_passed"]
        for component in COMPONENTS
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "FOUR_COMPONENT_PRIMARY_LEARNING_GATE_PASSED"
            if all_primary_gates_pass
            else "FOUR_COMPONENT_PRIMARY_LEARNING_GATE_NOT_PASSED"
        ),
        "primary_model": {
            "heads": 4,
            "family": "L2 logistic regression",
            "loss": "user-group-weighted binary cross entropy plus L2",
            "C": C_VALUE,
            "threshold": THRESHOLD,
            "grouped_oof_folds": N_SPLITS,
            "seeds": list(SEEDS),
            "feature_count_by_component": {
                component: len(feature_names_by_component[component])
                for component in COMPONENTS
            },
        },
        "outcome_fields_excluded": True,
        "opaque_item_ids_excluded": True,
        "semantic_family_excluded": True,
        "split_identity_excluded": True,
        "external_results_used": False,
        "internal_test_used_for_selection": False,
        "head_reports": reports,
        "all_primary_gates_pass": all_primary_gates_pass,
        "formal_external_promotion_allowed": all_primary_gates_pass,
        "descriptive_external_diagnostic_allowed": True,
        "claim_boundary": (
            "This evaluates protocol-specific candidate-action benefit "
            "labels. It does not prove correct item retrieval, clinical "
            "safety, or transport to arbitrary memory stores."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "memory_blueprint_sha256": sha256_file(
                args.memory_blueprint
            ),
            "rs_blueprint_sha256": sha256_file(args.rs_blueprint),
            "candidate_descriptors_sha256": sha256_file(
                args.candidate_descriptors
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "training_report.json", report)
    write_json(args.out_dir / "model.json", model_artifact)
    write_jsonl(args.out_dir / "predictions.jsonl", prediction_rows)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "all_primary_gates_pass": all_primary_gates_pass,
            "heads": {
                component: {
                    "oof_ba": reports[component]["grouped_oof"][
                        "metrics"
                    ]["group_weighted_balanced_accuracy"],
                    "oof_brier_gain": reports[component][
                        "grouped_oof"
                    ]["metrics"][
                        "group_weighted_brier_gain_vs_prevalence_prior"
                    ],
                    "internal_ba": reports[component][
                        "internal_test_frozen_threshold"
                    ]["metrics"]["group_weighted_balanced_accuracy"],
                    "gate": reports[component]["grouped_oof"][
                        "gate_passed"
                    ],
                }
                for component in COMPONENTS
            },
        }
    )


if __name__ == "__main__":
    main()
