from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .artifacts import require_artifact_attestation
from .config import load_config
from .contracts import MemorySource, StrategyMode, parse_action_id
from .io import canonical_json, read_json, sha256_file, sha256_text
from .pm_v2_contracts import ADVICE_READINESS_IDS, PMV2State
from .pm_v2_data import EvaluatorContextIndex


SHORTCUT_AUDIT_PROTOCOL = (
    "pm-v1.5-step0-shortcut-audit-v3-user-group-multivariate-train-oracle-only"
)
SHORTCUT_AUDIT_STAGE = "pm_v1_5_step0_shortcut_audit"


def _balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    values = sorted(set(bool(value) for value in target))
    if values != [False, True]:
        raise ValueError("shortcut target must contain both classes")
    recalls = []
    for value in values:
        mask = target == value
        recalls.append(float(np.mean(prediction[mask] == target[mask])))
    return float(np.mean(recalls))


def _best_threshold(feature: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    values = np.unique(feature.astype(float))
    if not len(values):
        raise ValueError("shortcut feature is empty")
    thresholds = np.concatenate(
        (
            [values[0] - 1e-9],
            (values[:-1] + values[1:]) / 2.0,
            [values[-1] + 1e-9],
        )
    )
    candidates = []
    for threshold in thresholds:
        for direction in ("ge", "le"):
            prediction = (
                feature >= threshold if direction == "ge" else feature <= threshold
            )
            candidates.append(
                {
                    "balanced_accuracy": _balanced_accuracy(target, prediction),
                    "threshold": float(threshold),
                    "direction": direction,
                }
            )
    return max(
        candidates,
        key=lambda row: (
            row["balanced_accuracy"],
            -abs(row["threshold"]),
            row["direction"],
        ),
    )


def _nearest_neighbor_identifiability(
    matrix: np.ndarray, labels: Sequence[str]
) -> dict[str, Any]:
    labels = [str(value) for value in labels]
    if len(matrix) != len(labels) or len(matrix) < 2:
        raise ValueError("identifiability matrix/labels are incomplete")
    scale = np.std(matrix, axis=0)
    normalized = (matrix - np.mean(matrix, axis=0)) / np.where(
        scale > 1e-12, scale, 1.0
    )
    distances = np.sum(
        (normalized[:, None, :] - normalized[None, :, :]) ** 2,
        axis=2,
    )
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    accuracy = float(
        np.mean([labels[index] == labels[int(nearest[index])] for index in range(len(labels))])
    )
    majority = max(Counter(labels).values()) / len(labels)
    return {
        "protocol": "leave-one-state-out-nearest-neighbor-diagnostic-v1",
        "accuracy": accuracy,
        "majority_baseline": float(majority),
        "class_count": len(set(labels)),
    }


def _group_cv_probe_scores(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str],
    *,
    folds: int,
    logistic_c: float,
    logistic_max_iter: int,
    tree_max_depth: int,
    tree_min_samples_leaf: int,
    random_seed: int,
) -> dict[str, Any]:
    """Out-of-user low-capacity probes for combined Step-0 shortcuts."""

    groups = np.asarray([str(value) for value in groups])
    if len(matrix) != len(target) or len(matrix) != len(groups):
        raise ValueError("multivariate shortcut probe inputs are incomplete")
    if len(np.unique(groups)) < int(folds) or int(folds) < 2:
        raise ValueError("multivariate shortcut probe has too few user groups")
    if sorted(set(bool(value) for value in target)) != [False, True]:
        raise ValueError("multivariate shortcut target must contain both classes")
    splitter = GroupKFold(n_splits=int(folds))
    splits = list(splitter.split(matrix, target, groups))
    estimators = {
        "regularized_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(logistic_c),
                class_weight="balanced",
                max_iter=int(logistic_max_iter),
                random_state=int(random_seed),
                solver="liblinear",
            ),
        ),
        "shallow_tree": DecisionTreeClassifier(
            max_depth=int(tree_max_depth),
            min_samples_leaf=int(tree_min_samples_leaf),
            class_weight="balanced",
            random_state=int(random_seed),
        ),
    }
    scores: dict[str, float] = {}
    for name, estimator in estimators.items():
        prediction = np.empty(len(target), dtype=bool)
        covered = np.zeros(len(target), dtype=bool)
        for train_index, test_index in splits:
            train_target = target[train_index]
            if sorted(set(bool(value) for value in train_target)) != [False, True]:
                raise RuntimeError(
                    "user-group shortcut fold lost a target class; corpus is not auditable"
                )
            fitted = clone(estimator).fit(matrix[train_index], train_target)
            prediction[test_index] = fitted.predict(matrix[test_index]).astype(bool)
            covered[test_index] = True
        if not bool(np.all(covered)):
            raise RuntimeError("user-group shortcut folds did not cover every train state")
        scores[name] = _balanced_accuracy(target, prediction)
    return {
        "scores": scores,
        "maximum_balanced_accuracy": max(scores.values()),
        "folds": int(folds),
        "group_count": len(np.unique(groups)),
        "fold_test_group_counts": [
            len(np.unique(groups[test_index])) for _, test_index in splits
        ],
    }


def audit_step0_shortcuts(
    *,
    predictive_states: Sequence[PMV2State],
    structural_states: Sequence[PMV2State],
    evaluator_contexts: EvaluatorContextIndex,
    expected_predictive_states: int,
    expected_structural_states: int,
    maximum_single_threshold_balanced_accuracy: float,
    maximum_multivariate_probe_balanced_accuracy: float,
    centroid_noise_std: float,
    shuffle_seed: int,
    group_cv_folds: int,
    probe_logistic_c: float,
    probe_logistic_max_iter: int,
    probe_tree_max_depth: int,
    probe_tree_min_samples_leaf: int,
) -> dict[str, Any]:
    """Separate oracle-bearing train diagnostics from all-split structure checks.

    ``predictive_states`` must be train-only.  Evaluator resource/regime labels
    are read only for those states.  The all-split branch sees Step-0 values and
    operational user/split IDs, never ``needed_memory_sources`` or ``regime``.
    """

    if (
        not predictive_states
        or not structural_states
        or len({state.state_id for state in structural_states})
        != len(structural_states)
        or len({state.state_id for state in predictive_states})
        != len(predictive_states)
    ):
        raise ValueError("shortcut audit requires unique non-empty state sets")
    if any(state.split.value != "train" for state in predictive_states):
        raise RuntimeError("shortcut predictive audit may read oracle labels only on train")
    if not {state.state_id for state in predictive_states} <= {
        state.state_id for state in structural_states
    }:
        raise RuntimeError("shortcut predictive states are outside structural universe")
    if not 0.5 <= float(maximum_single_threshold_balanced_accuracy) <= 1.0:
        raise ValueError("shortcut threshold must be in [0.5, 1.0]")
    if not 0.5 <= float(maximum_multivariate_probe_balanced_accuracy) <= 1.0:
        raise ValueError("multivariate shortcut threshold must be in [0.5, 1.0]")
    if float(centroid_noise_std) <= 0.0:
        raise ValueError("centroid noise std must be positive")
    contexts = evaluator_contexts.require_states(predictive_states, exact=False)
    predictive_ordered = sorted(predictive_states, key=lambda value: value.state_id)
    structural_ordered = sorted(structural_states, key=lambda value: value.state_id)
    rng = np.random.default_rng(int(shuffle_seed))
    source_order = list(MemorySource)
    feature_names = [f"{source.value}_similarity" for source in source_order] + [
        "strategy_max_family_similarity",
        *[
            f"strategy_readiness_{readiness}_similarity"
            for readiness in ADVICE_READINESS_IDS
        ],
        "strategy_question_present",
    ]
    def matrix_for(ordered: Sequence[PMV2State]) -> np.ndarray:
        rows = []
        for state in ordered:
            if state.step0_observation is None:
                raise RuntimeError("shortcut audit requires formal Step-0 on every state")
            strategy = state.step0_observation.strategy
            rows.append(
                [
                    *[
                        float(
                            state.step0_observation.memory_sources[
                                source
                            ].query_to_source_similarity
                        )
                        for source in source_order
                    ],
                    max(strategy.family_similarities.values(), default=0.0),
                    *[
                        float(strategy.advice_readiness_similarities[readiness])
                        for readiness in ADVICE_READINESS_IDS
                    ],
                    float(strategy.question_present),
                ]
            )
        return np.asarray(rows, dtype=float)

    predictive_matrix = matrix_for(predictive_ordered)
    structural_matrix = matrix_for(structural_ordered)
    targets: dict[str, list[bool]] = {
        **{source.value: [] for source in source_order},
        "RS": [],
    }
    regime_labels: list[str] = []
    rs_target_mask: list[bool] = []
    for state in predictive_ordered:
        context = contexts[state.state_id]
        needed = set(str(value) for value in context.get("needed_memory_sources") or [])
        regime = str(context.get("regime") or "")
        for source in source_order:
            targets[source.value].append(source.value in needed)
        strategy_target = str(context.get("strategy_resource_target") or "ambiguous")
        targets["RS"].append(strategy_target == "use")
        rs_target_mask.append(strategy_target in {"use", "skip"})
        regime_labels.append(regime)
    target_arrays = {
        key: np.asarray(values, dtype=bool) for key, values in targets.items()
    }
    target_masks = {
        **{
            source.value: np.ones(len(predictive_ordered), dtype=bool)
            for source in source_order
        },
        "RS": np.asarray(rs_target_mask, dtype=bool),
    }
    feature_columns = {
        name: predictive_matrix[:, index] for index, name in enumerate(feature_names)
    }
    target_feature_sets = {
        "MP": ["MP_similarity"],
        "MS": ["MS_similarity"],
        "ME": ["ME_similarity"],
        "RS": feature_names[3:],
    }
    threshold_rows: dict[str, dict[str, Any]] = {}
    for target_name, names in target_feature_sets.items():
        mask = target_masks[target_name]
        target = target_arrays[target_name][mask]
        rows = {
            name: _best_threshold(feature_columns[name][mask], target)
            for name in names
        }
        selected_name, selected = max(
            rows.items(), key=lambda item: item[1]["balanced_accuracy"]
        )
        shuffled = target.copy()
        rng.shuffle(shuffled)
        shuffled_score = max(
            _best_threshold(feature_columns[name][mask], shuffled)["balanced_accuracy"]
            for name in names
        )
        noisy_score = max(
            _best_threshold(
                np.clip(
                    feature_columns[name][mask]
                    + rng.normal(
                        0.0, float(centroid_noise_std), int(np.sum(mask))
                    ),
                    -1.0,
                    1.0,
                ),
                target,
            )["balanced_accuracy"]
            for name in names
        )
        threshold_rows[target_name] = {
            "selected_feature": selected_name,
            "selected": selected,
            "all_features": rows,
            "shuffled_label_best_balanced_accuracy": float(shuffled_score),
            "centroid_noise_best_balanced_accuracy": float(noisy_score),
            "no_step0_balanced_accuracy": 0.5,
            "positive_rate": float(np.mean(target)),
            "state_count": int(np.sum(mask)),
        }

    permuted_source = {}
    for index, source in enumerate(source_order):
        permuted = source_order[(index + 1) % len(source_order)]
        permuted_source[source.value] = {
            "replacement_feature": f"{permuted.value}_similarity",
            "balanced_accuracy": _best_threshold(
                feature_columns[f"{permuted.value}_similarity"],
                target_arrays[source.value],
            )["balanced_accuracy"],
        }
    maximum_observed = max(
        row["selected"]["balanced_accuracy"] for row in threshold_rows.values()
    )
    predictive_groups = [state.user_id for state in predictive_ordered]
    multivariate_targets: dict[str, dict[str, Any]] = {}
    for target_name, target in target_arrays.items():
        mask = target_masks[target_name]
        selected_matrix = predictive_matrix[mask]
        selected_target = target[mask]
        selected_groups = np.asarray(predictive_groups)[mask]
        observed = _group_cv_probe_scores(
            selected_matrix,
            selected_target,
            selected_groups,
            folds=group_cv_folds,
            logistic_c=probe_logistic_c,
            logistic_max_iter=probe_logistic_max_iter,
            tree_max_depth=probe_tree_max_depth,
            tree_min_samples_leaf=probe_tree_min_samples_leaf,
            random_seed=shuffle_seed,
        )
        shuffled_target = selected_target.copy()
        rng.shuffle(shuffled_target)
        shuffled = _group_cv_probe_scores(
            selected_matrix,
            shuffled_target,
            selected_groups,
            folds=group_cv_folds,
            logistic_c=probe_logistic_c,
            logistic_max_iter=probe_logistic_max_iter,
            tree_max_depth=probe_tree_max_depth,
            tree_min_samples_leaf=probe_tree_min_samples_leaf,
            random_seed=shuffle_seed,
        )
        multivariate_targets[target_name] = {
            **observed,
            "shuffled_label_scores": shuffled["scores"],
            "positive_rate": float(np.mean(selected_target)),
            "state_count": int(np.sum(mask)),
        }
    maximum_multivariate_observed = max(
        row["maximum_balanced_accuracy"]
        for row in multivariate_targets.values()
    )
    checks = {
        "train_predictive_state_count": len(predictive_ordered)
        == int(expected_predictive_states),
        "all_split_structural_state_count": len(structural_ordered)
        == int(expected_structural_states),
        "predictive_oracle_scope_train_only": all(
            state.split.value == "train" for state in predictive_ordered
        ),
        "formal_step0_complete": all(
            state.step0_observation is not None for state in structural_ordered
        ),
        "single_threshold_not_near_oracle": maximum_observed
        < float(maximum_single_threshold_balanced_accuracy),
        "multivariate_probe_not_near_oracle": maximum_multivariate_observed
        < float(maximum_multivariate_probe_balanced_accuracy),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": SHORTCUT_AUDIT_PROTOCOL,
        "outcome_labels_read": False,
        "resource_oracle_scope": "train_only",
        "internal_resource_oracle_read": False,
        "n_predictive_states": len(predictive_ordered),
        "n_structural_states": len(structural_ordered),
        "feature_names": feature_names,
        "maximum_single_threshold_balanced_accuracy": float(
            maximum_single_threshold_balanced_accuracy
        ),
        "maximum_observed_single_threshold_balanced_accuracy": float(
            maximum_observed
        ),
        "maximum_multivariate_probe_balanced_accuracy": float(
            maximum_multivariate_probe_balanced_accuracy
        ),
        "maximum_observed_multivariate_probe_balanced_accuracy": float(
            maximum_multivariate_observed
        ),
        "predictive_audit": {
            "data_role": "train_only",
            "evaluator_fields_read": [
                "needed_memory_sources",
                "regime",
                "strategy_resource_target",
            ],
            "targets": threshold_rows,
            "permuted_source": permuted_source,
            "regime_identifiability": _nearest_neighbor_identifiability(
                predictive_matrix, regime_labels
            ),
            "multivariate_user_group_probes": {
                "protocol": "pm-v1.5-low-capacity-user-group-probes-v1",
                "feature_names": feature_names,
                "models": {
                    "regularized_logistic": {
                        "C": float(probe_logistic_c),
                        "max_iter": int(probe_logistic_max_iter),
                        "class_weight": "balanced",
                    },
                    "shallow_tree": {
                        "max_depth": int(probe_tree_max_depth),
                        "min_samples_leaf": int(probe_tree_min_samples_leaf),
                        "class_weight": "balanced",
                    },
                },
                "targets": multivariate_targets,
            },
        },
        "structural_audit": {
            "data_role": "all_splits_without_evaluator_oracle",
            "evaluator_fields_read": [],
            "feature_min": {
                name: float(np.min(structural_matrix[:, index]))
                for index, name in enumerate(feature_names)
            },
            "feature_max": {
                name: float(np.max(structural_matrix[:, index]))
                for index, name in enumerate(feature_names)
            },
            "missing_or_nonfinite_values": int(
                np.size(structural_matrix) - np.isfinite(structural_matrix).sum()
            ),
            "identifiability_diagnostics": {
                "user": _nearest_neighbor_identifiability(
                    structural_matrix,
                    [state.user_id for state in structural_ordered],
                ),
                "split_environment": _nearest_neighbor_identifiability(
                    structural_matrix,
                    [state.split.value for state in structural_ordered],
                ),
            },
        },
        "action_factor_coverage": {
            "factors": ["MP", "MS", "ME", "RS"],
            "main_effect_levels": {name: [0, 1] for name in ("MP", "MS", "ME", "RS")},
            "pairwise_interactions": [
                "MP*MS",
                "MP*ME",
                "MP*RS",
                "MS*ME",
                "MS*RS",
                "ME*RS",
            ],
            "observed_action_factor_combinations": sorted(
                {
                    tuple(
                        [
                            int(MemorySource.MP in parse_action_id(action)[0]),
                            int(MemorySource.MS in parse_action_id(action)[0]),
                            int(MemorySource.ME in parse_action_id(action)[0]),
                            int(parse_action_id(action)[1] is StrategyMode.RS),
                        ]
                    )
                    for state in structural_ordered
                    for action in state.allowed_actions
                }
            ),
        },
        "checks": checks,
    }


def require_step0_shortcut_audit_pass(
    report_path: str | Path,
    attestation_path: str | Path,
    *,
    expected_states_path: str | Path,
    expected_evaluator_contexts_path: str | Path,
    expected_pm_config_path: str | Path,
    expected_generation_attestation_path: str | Path,
) -> dict[str, Any]:
    """Verify the no-API shortcut gate before any full action sweep."""

    verification = require_artifact_attestation(
        attestation_path,
        required_stage=SHORTCUT_AUDIT_STAGE,
        required_output_paths={"shortcut_audit_report": report_path},
    )
    require_artifact_attestation(
        expected_generation_attestation_path,
        required_stage="pm_v1_5_development_data",
        required_output_paths={
            "states": expected_states_path,
            "evaluator_contexts": expected_evaluator_contexts_path,
        },
    )
    report = read_json(report_path)
    attestation = read_json(attestation_path)
    expected_inputs = {
        "states": expected_states_path,
        "evaluator_contexts": expected_evaluator_contexts_path,
        "pm_v1_5_config": expected_pm_config_path,
        "development_data_attestation": expected_generation_attestation_path,
    }
    for logical_name, expected_path in expected_inputs.items():
        record = (attestation.get("inputs") or {}).get(logical_name)
        if not isinstance(record, Mapping) or record.get("sha256") != sha256_file(
            expected_path
        ):
            raise RuntimeError(
                "Step-0 shortcut attestation does not bind current " + logical_name
            )
    config = load_config(expected_pm_config_path)
    shortcut_cfg = dict(config.get("shortcut_audit") or {})
    expected_hashes = {
        "states": sha256_file(expected_states_path),
        "evaluator_contexts": sha256_file(expected_evaluator_contexts_path),
        "pm_v1_5_config": sha256_file(expected_pm_config_path),
        "development_data_attestation": sha256_file(
            expected_generation_attestation_path
        ),
    }
    checks = report.get("checks") or {}
    if (
        report.get("protocol") != SHORTCUT_AUDIT_PROTOCOL
        or report.get("status") != "PASS"
        or not checks
        or not all(value is True for value in checks.values())
        or report.get("input_hashes") != expected_hashes
        or int(report.get("n_predictive_states") or 0)
        != int(shortcut_cfg.get("expected_predictive_train_states") or 0)
        or int(report.get("n_structural_states") or 0)
        != int(shortcut_cfg.get("expected_structural_all_split_states") or 0)
        or float(report.get("maximum_single_threshold_balanced_accuracy"))
        != float(shortcut_cfg.get("maximum_single_threshold_balanced_accuracy"))
        or float(report.get("maximum_multivariate_probe_balanced_accuracy"))
        != float(shortcut_cfg.get("maximum_multivariate_probe_balanced_accuracy"))
        or shortcut_cfg.get("fail_on_near_oracle_threshold") is not True
        or shortcut_cfg.get("fail_on_near_oracle_multivariate_probe") is not True
    ):
        raise RuntimeError("Step-0 shortcut pre-sweep gate did not PASS")
    return {
        "status": "PASS",
        "report": report,
        "report_sha256": sha256_file(report_path),
        "attestation_sha256": verification["attestation_sha256"],
        "binding_sha256": sha256_text(
            canonical_json(
                {
                    "report_sha256": sha256_file(report_path),
                    "attestation_sha256": verification["attestation_sha256"],
                    "input_hashes": expected_hashes,
                }
            )
        ),
    }
