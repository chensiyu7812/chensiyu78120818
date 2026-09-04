from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from .pm_v1_5_rule_router import (
    transparent_rule_candidates,
    tune_transparent_rule_router,
)
from .pm_v2_contracts import ActionLabel, PMV2State
from .pm_v2_model import (
    PMV2Model,
    ROUTING_ALGORITHMS,
    SelectionConfig,
    evaluate_policy,
)


ALGORITHM_SELECTION_PROTOCOL = (
    "pm-v1.5-train-user-group-cv-one-standard-error-selection-v2"
)


def _action_distribution_instability(rows: Sequence[Mapping[str, Any]]) -> float:
    distributions = []
    action_ids = sorted(
        {
            str(action)
            for row in rows
            for action in (row.get("action_distribution") or {})
        }
    )
    for row in rows:
        raw = row.get("action_distribution") or {}
        values = np.asarray([float(raw.get(action, 0.0)) for action in action_ids])
        total = float(values.sum())
        distributions.append(values / total if total > 0.0 else values)
    pairwise = [
        0.5 * float(np.abs(left - right).sum())
        for index, left in enumerate(distributions)
        for right in distributions[index + 1 :]
    ]
    return float(np.mean(pairwise)) if pairwise else 0.0


def _select_one_standard_error_candidate(
    eligible: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any], float, list[Mapping[str, Any]]]:
    """Apply the frozen one-standard-error rule to completed CV summaries."""

    if not eligible:
        raise RuntimeError("one-standard-error selection requires eligible candidates")
    best = max(
        eligible,
        key=lambda row: (
            float(row["mean_realized_utility"]),
            float(row["mean_quality"]),
            -float(row["mean_risk"]),
            -float(row["mean_observed_input_tokens"]),
            -int(row["priority"]),
        ),
    )
    floor = float(best["mean_realized_utility"]) - float(
        best["mean_realized_utility_standard_error"]
    )
    candidates = [
        row
        for row in eligible
        if float(row["mean_realized_utility"]) >= floor
    ]
    selected = min(
        candidates,
        key=lambda row: (
            int(row["simplicity_rank"]),
            float(row["action_distribution_instability"]),
            -float(row["mean_realized_utility"]),
            -float(row["mean_quality"]),
        ),
    )
    return best, selected, floor, candidates


def _subset_labels(
    labels: Sequence[ActionLabel], state_ids: set[str]
) -> list[ActionLabel]:
    return [label for label in labels if label.state_id in state_ids]


def select_routing_algorithm_group_cv(
    *,
    states: Sequence[PMV2State],
    labels: Sequence[ActionLabel],
    selection_config: SelectionConfig,
    candidates: Sequence[str],
    folds: int,
    n_models: int,
    seed: int,
    dimension_mad_scale: float,
    bootstrap_group_key: str,
    use_precomputed_embeddings: bool,
    require_precomputed_embeddings: bool = False,
    semantic_projection_dimensions: int = 48,
    word_features: int,
    char_features: int,
    rule_grid: Mapping[str, Sequence[float | int]],
    rule_minimum_quality: float,
    rule_maximum_risk: float,
    minimum_validation_quality: float,
    maximum_validation_risk: float,
    safe_residual_thresholds: Mapping[str, float],
    simplicity_order: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Choose the routing target using only train-user group-aware CV."""

    candidate_names = [str(value) for value in candidates]
    if not candidate_names or len(candidate_names) != len(set(candidate_names)):
        raise ValueError("algorithm candidates must be unique and non-empty")
    unknown = sorted(set(candidate_names) - set(ROUTING_ALGORITHMS))
    if unknown:
        raise ValueError(f"unknown routing algorithms: {unknown}")
    simplicity = [str(value) for value in simplicity_order]
    if set(simplicity) != set(candidate_names) or len(simplicity) != len(candidate_names):
        raise ValueError("algorithm simplicity_order must exactly cover candidates")
    simplicity_rank = {name: index for index, name in enumerate(simplicity)}
    state_map = {state.state_id: state for state in states}
    if len(state_map) != len(states) or not state_map:
        raise ValueError("algorithm selection requires unique non-empty states")
    label_keys = {(label.state_id, label.action_id) for label in labels}
    expected_keys = {
        (state.state_id, action_id)
        for state in states
        for action_id in state.allowed_actions
    }
    if label_keys != expected_keys or len(labels) != len(expected_keys):
        raise ValueError("algorithm selection requires an exact complete label matrix")
    users = np.asarray([state.user_id for state in states], dtype=object)
    unique_users = sorted(set(str(value) for value in users))
    fold_count = int(folds)
    if fold_count < 2 or fold_count > len(unique_users):
        raise ValueError("algorithm CV folds must be in [2, unique train users]")
    if int(n_models) < 1:
        raise ValueError("algorithm CV bootstrap model count must be positive")

    splitter = GroupKFold(n_splits=fold_count)
    candidate_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in candidate_names
    }
    split_rows: list[dict[str, Any]] = []
    state_array = np.asarray(list(states), dtype=object)
    for fold_index, (fit_indices, validation_indices) in enumerate(
        splitter.split(state_array, groups=users)
    ):
        fit_states = [states[int(index)] for index in fit_indices]
        validation_states = [states[int(index)] for index in validation_indices]
        fit_ids = {state.state_id for state in fit_states}
        validation_ids = {state.state_id for state in validation_states}
        fit_users = {state.user_id for state in fit_states}
        validation_users = {state.user_id for state in validation_states}
        if fit_users & validation_users:
            raise RuntimeError("algorithm CV leaked users across a fold")
        fit_labels = _subset_labels(labels, fit_ids)
        validation_labels = _subset_labels(labels, validation_ids)
        base = PMV2Model.train(
            fit_states,
            fit_labels,
            selection_config=selection_config,
            n_models=int(n_models),
            seed=int(seed) + fold_index * 1000,
            dimension_mad_scale=float(dimension_mad_scale),
            bootstrap_group_key=bootstrap_group_key,
            use_precomputed_embeddings=use_precomputed_embeddings,
            require_precomputed_embeddings=require_precomputed_embeddings,
            semantic_projection_dimensions=int(semantic_projection_dimensions),
            word_features=int(word_features),
            char_features=int(char_features),
        )
        fold_rule = None
        fold_rule_report = None
        if "rule_relative_safe_residual_hgb" in candidate_names:
            fold_rule, fold_rule_report = tune_transparent_rule_router(
                states=fit_states,
                labels=fit_labels,
                selection_config=selection_config,
                candidates=transparent_rule_candidates(rule_grid),
                minimum_quality=float(rule_minimum_quality),
                maximum_risk=float(rule_maximum_risk),
                selection_data_role="train_fold",
            )
        split_rows.append(
            {
                "fold": fold_index,
                "fit_users": sorted(fit_users),
                "validation_users": sorted(validation_users),
                "fit_states": len(fit_states),
                "validation_states": len(validation_states),
                "fit_user_count": len(fit_users),
                "validation_user_count": len(validation_users),
                "rule_config_sha256": (
                    fold_rule_report["selected_config_sha256"]
                    if fold_rule_report is not None
                    else None
                ),
            }
        )
        for candidate_index, algorithm in enumerate(candidate_names):
            model = deepcopy(base)
            model.fit_routing_objective(
                fit_states,
                fit_labels,
                algorithm=algorithm,
                n_models=int(n_models),
                seed=int(seed) + fold_index * 1000 + candidate_index * 100,
                bootstrap_group_key=bootstrap_group_key,
                rule_router=(
                    fold_rule
                    if algorithm == "rule_relative_safe_residual_hgb"
                    else None
                ),
                safe_thresholds=(
                    dict(safe_residual_thresholds)
                    if algorithm == "rule_relative_safe_residual_hgb"
                    else None
                ),
            )
            metrics = evaluate_policy(model, validation_states, validation_labels)
            candidate_rows[algorithm].append(
                {
                    "fold": fold_index,
                    "validation_states": len(validation_states),
                    "validation_users": len(validation_users),
                    "mean_quality": float(metrics["mean_quality"]),
                    "mean_emotional_support": float(
                        metrics["mean_response_dimensions"]["emotional_support"]
                    ),
                    "mean_risk": float(metrics["mean_risk"]),
                    "mean_realized_utility": float(
                        metrics["mean_realized_utility"]
                    ),
                    "mean_observed_input_tokens": float(
                        metrics["mean_observed_input_tokens"]
                    ),
                    "action_distribution": dict(metrics["action_distribution"]),
                }
            )

    summaries: list[dict[str, Any]] = []
    for priority, algorithm in enumerate(candidate_names):
        rows = candidate_rows[algorithm]
        if len(rows) != fold_count:
            raise RuntimeError("algorithm CV candidate has an incomplete fold matrix")
        total_states = sum(int(row["validation_states"]) for row in rows)

        def weighted(field: str) -> float:
            return float(
                sum(
                    float(row[field]) * int(row["validation_states"])
                    for row in rows
                )
                / total_states
            )

        action_counts = Counter()
        for row in rows:
            action_counts.update(row["action_distribution"])
        summary = {
            "algorithm": algorithm,
            "priority": priority,
            "folds": rows,
            "validation_states": total_states,
            "mean_quality": weighted("mean_quality"),
            "mean_emotional_support": weighted("mean_emotional_support"),
            "mean_risk": weighted("mean_risk"),
            "mean_realized_utility": weighted("mean_realized_utility"),
            "mean_observed_input_tokens": weighted(
                "mean_observed_input_tokens"
            ),
            "action_distribution": dict(sorted(action_counts.items())),
            "mean_realized_utility_standard_error": float(
                np.std(
                    [float(row["mean_realized_utility"]) for row in rows], ddof=1
                )
                / np.sqrt(len(rows))
            ),
            "action_distribution_instability": _action_distribution_instability(rows),
            "simplicity_rank": simplicity_rank[algorithm],
        }
        summary["eligible"] = bool(
            summary["mean_quality"] >= float(minimum_validation_quality)
            and summary["mean_risk"] <= float(maximum_validation_risk)
        )
        summaries.append(summary)
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        raise RuntimeError("no train-CV routing algorithm satisfies frozen guards")
    (
        best,
        selected,
        one_standard_error_floor,
        one_standard_error_candidates,
    ) = _select_one_standard_error_candidate(
        eligible
    )
    report = {
        "protocol": ALGORITHM_SELECTION_PROTOCOL,
        "selection_data_role": "train_only",
        "group_key": "user_id",
        "fold_count": fold_count,
        "unique_train_users": len(unique_users),
        "cv_bootstrap_models": int(n_models),
        "minimum_validation_quality": float(minimum_validation_quality),
        "maximum_validation_risk": float(maximum_validation_risk),
        "safe_residual_thresholds": {
            str(key): float(value)
            for key, value in safe_residual_thresholds.items()
        },
        "selected_algorithm": selected["algorithm"],
        "raw_best_mean_utility_algorithm": best["algorithm"],
        "one_standard_error_floor": one_standard_error_floor,
        "one_standard_error_candidates": [
            row["algorithm"] for row in one_standard_error_candidates
        ],
        "simplicity_order": simplicity,
        "selection_rule": (
            "simplest eligible candidate within one standard error of the "
            "best train-user CV utility; action stability breaks equal-complexity ties"
        ),
        "fold_assignments": split_rows,
        "candidates": summaries,
    }
    return str(selected["algorithm"]), report
