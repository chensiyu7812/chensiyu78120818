from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .contracts import MemorySource
from .io import canonical_json, sha256_text
from .pm_v1_6_contracts import STRATEGY_FAMILIES, Step0Observation
from .pm_v2_contracts import PMV2Split, PMV2State
from .pm_v2_data import EvaluatorContextIndex
from .pm_v2_features import state_text

PROTOCOL = "pm-v1.6-train-only-shortcut-audit-v1"
SOURCE_ORDER = (MemorySource.MP, MemorySource.MS, MemorySource.ME)


@dataclass(frozen=True)
class ShortcutAuditConfig:
    folds: int = 6
    seed: int = 6113
    high_similarity_threshold: float = 0.45
    low_similarity_threshold: float = 0.15
    maximum_probe_only_exact_match: float = 0.85
    minimum_combined_gain_over_text: float = 0.01
    minimum_shuffled_probe_drop: float = 0.01
    minimum_challenge_count: int = 1

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("shortcut audit requires at least two folds")
        if not -1.0 <= self.low_similarity_threshold < self.high_similarity_threshold <= 1.0:
            raise ValueError("shortcut similarity thresholds are invalid")
        if not 0.0 <= self.maximum_probe_only_exact_match <= 1.0:
            raise ValueError("probe-only ceiling must be in [0,1]")
        if self.minimum_challenge_count < 1:
            raise ValueError("minimum challenge count must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "folds": self.folds,
            "seed": self.seed,
            "high_similarity_threshold": self.high_similarity_threshold,
            "low_similarity_threshold": self.low_similarity_threshold,
            "maximum_probe_only_exact_match": self.maximum_probe_only_exact_match,
            "minimum_combined_gain_over_text": self.minimum_combined_gain_over_text,
            "minimum_shuffled_probe_drop": self.minimum_shuffled_probe_drop,
            "minimum_challenge_count": self.minimum_challenge_count,
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))


def balanced_user_folds(
    states: Sequence[PMV2State], *, folds: int, seed: int
) -> dict[str, int]:
    users = sorted({state.user_id for state in states})
    if len(users) < folds:
        raise ValueError("not enough users for balanced group folds")
    ordered = sorted(
        users,
        key=lambda user: sha256_text(canonical_json([int(seed), user])),
    )
    return {user: index % folds for index, user in enumerate(ordered)}


def _probe_vector(observation: Step0Observation) -> np.ndarray:
    values: list[float] = []
    for source in SOURCE_ORDER:
        row = observation.sources[source]
        values.extend(
            [
                float(row.available),
                float(row.bounded_count) / 3.0,
                float(row.min_age_sessions or 0) / 64.0,
                float(row.median_age_sessions or 0) / 64.0,
                float(row.max_age_sessions or 0) / 64.0,
                float(row.estimated_retrievable_tokens) / 768.0,
                float(row.query_to_source_similarity),
                float(row.representation_valid),
            ]
        )
    family = {row.family: row for row in observation.strategy_families}
    for name in STRATEGY_FAMILIES:
        row = family[name]
        values.extend(
            [
                float(row.query_to_family_similarity),
                float(row.representation_valid),
            ]
        )
    readiness = observation.readiness
    values.extend(
        [
            float(readiness.advice_requested),
            float(readiness.advice_rejected),
            float(readiness.listening_requested),
            float(readiness.clarification_needed),
            float(readiness.action_readiness),
        ]
    )
    return np.asarray(values, dtype=np.float64)


def _text_matrix(states: Sequence[PMV2State]) -> np.ndarray:
    texts = [state_text(state) for state in states]
    word = HashingVectorizer(
        n_features=512,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
    ).transform(texts).toarray()
    char = HashingVectorizer(
        n_features=512,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
    ).transform(texts).toarray()
    return np.concatenate([word, char], axis=1).astype(np.float64)


def _predict_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    unique = np.unique(y_train)
    if unique.size == 1:
        return np.full(x_test.shape[0], int(unique[0]), dtype=int)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=int(seed),
        solver="liblinear",
    )
    model.fit(x_train, y_train)
    return model.predict(x_test).astype(int)


def _target_rows(
    states: Sequence[PMV2State],
    evaluator_contexts: EvaluatorContextIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_targets = []
    strategy_targets = []
    strategy_mask = []
    for state in states:
        context = evaluator_contexts.by_state[state.state_id]
        needed = {MemorySource(value) for value in context["needed_memory_sources"]}
        source_targets.append([int(source in needed) for source in SOURCE_ORDER])
        regime = str(context["regime"])
        if regime == "strategy_helpful":
            strategy_targets.append(1)
            strategy_mask.append(True)
        elif regime == "strategy_harmful":
            strategy_targets.append(0)
            strategy_mask.append(True)
        else:
            strategy_targets.append(0)
            strategy_mask.append(False)
    return (
        np.asarray(source_targets, dtype=int),
        np.asarray(strategy_targets, dtype=int),
        np.asarray(strategy_mask, dtype=bool),
    )


def _metric(
    source_true: np.ndarray,
    source_pred: np.ndarray,
    strategy_true: np.ndarray,
    strategy_pred: np.ndarray,
    strategy_mask: np.ndarray,
) -> dict[str, Any]:
    per_source = {
        source.value: float(
            f1_score(
                source_true[:, index],
                source_pred[:, index],
                zero_division=0,
            )
        )
        for index, source in enumerate(SOURCE_ORDER)
    }
    exact = float(np.mean(np.all(source_true == source_pred, axis=1)))
    strategy_f1 = (
        float(
            f1_score(
                strategy_true[strategy_mask],
                strategy_pred[strategy_mask],
                zero_division=0,
            )
        )
        if np.any(strategy_mask)
        else None
    )
    return {
        "source_f1": per_source,
        "source_macro_f1": float(np.mean(list(per_source.values()))),
        "source_exact_match": exact,
        "strategy_f1_on_helpful_harmful": strategy_f1,
        "strategy_rows": int(np.sum(strategy_mask)),
    }


def _challenge_coverage(
    states: Sequence[PMV2State],
    step0_by_state: Mapping[str, Step0Observation],
    evaluator_contexts: EvaluatorContextIndex,
    config: ShortcutAuditConfig,
) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for state in states:
        observation = step0_by_state[state.state_id]
        context = evaluator_contexts.by_state[state.state_id]
        regime = str(context["regime"])
        needed = {MemorySource(value) for value in context["needed_memory_sources"]}
        similarities = {
            source: observation.sources[source].query_to_source_similarity
            for source in SOURCE_ORDER
            if observation.sources[source].representation_valid
        }
        high_sources = {
            source
            for source, value in similarities.items()
            if value >= config.high_similarity_threshold
        }
        if regime == "memory_harmful" and high_sources:
            counts["high_similarity_stale_or_conflicting"] += 1
        if not needed and regime not in {"memory_harmful"} and high_sources:
            counts["high_similarity_irrelevant"] += 1
        if any(
            source in similarities
            and similarities[source] <= config.low_similarity_threshold
            for source in needed
        ):
            counts["low_similarity_helpful"] += 1
        if needed and sum(
            observation.sources[source].available for source in SOURCE_ORDER
        ) >= 2:
            counts["swapped_source_counterfactual_available"] += 1
        if regime in {"context_only", "ambiguous"} and len(high_sources) >= 2:
            counts["multi_source_redundant"] += 1
        max_strategy = max(
            (
                row.query_to_family_similarity
                for row in observation.strategy_families
                if row.representation_valid
            ),
            default=0.0,
        )
        if (
            regime == "strategy_harmful"
            and observation.readiness.advice_rejected
            and max_strategy >= config.high_similarity_threshold
        ):
            counts["strategy_similarity_but_advice_rejected"] += 1
    required = (
        "high_similarity_stale_or_conflicting",
        "high_similarity_irrelevant",
        "low_similarity_helpful",
        "swapped_source_counterfactual_available",
        "multi_source_redundant",
        "strategy_similarity_but_advice_rejected",
    )
    return {name: int(counts[name]) for name in required}


def run_shortcut_audit(
    *,
    states: Sequence[PMV2State],
    step0_by_state: Mapping[str, Step0Observation],
    evaluator_contexts: EvaluatorContextIndex,
    config: ShortcutAuditConfig | None = None,
) -> dict[str, Any]:
    frozen = config or ShortcutAuditConfig()
    train_states = [state for state in states if state.split is PMV2Split.TRAIN]
    if len(train_states) != len(states):
        raise RuntimeError(
            "shortcut model diagnostics accept train users only; internal/calibration "
            "must not influence the feature contract"
        )
    if set(step0_by_state) != {state.state_id for state in train_states}:
        raise RuntimeError("Step-0/state universe mismatch in shortcut audit")
    evaluator_contexts.require_states(train_states, exact=True)
    fold_by_user = balanced_user_folds(
        train_states, folds=frozen.folds, seed=frozen.seed
    )
    probe = np.vstack(
        [step0_by_state[state.state_id] and _probe_vector(step0_by_state[state.state_id]) for state in train_states]
    )
    text = _text_matrix(train_states)
    combined = np.concatenate([text, probe], axis=1)
    source_y, strategy_y, strategy_mask = _target_rows(
        train_states, evaluator_contexts
    )

    predictions = {
        "probe_only": np.zeros_like(source_y),
        "text_only": np.zeros_like(source_y),
        "combined": np.zeros_like(source_y),
        "shuffled_probe": np.zeros_like(source_y),
    }
    strategy_predictions = {
        name: np.zeros_like(strategy_y) for name in predictions
    }
    fold_rows = []
    for fold in range(frozen.folds):
        train_index = np.asarray(
            [fold_by_user[state.user_id] != fold for state in train_states]
        )
        test_index = ~train_index
        if not np.any(train_index) or not np.any(test_index):
            raise RuntimeError("balanced shortcut-audit fold is empty")
        test_positions = np.flatnonzero(test_index)
        rng = np.random.default_rng(frozen.seed + fold * 101)
        shuffled_positions = rng.permutation(test_positions)
        shuffled_combined_test = np.concatenate(
            [text[test_index], probe[shuffled_positions]], axis=1
        )
        matrices = {
            "probe_only": (probe[train_index], probe[test_index]),
            "text_only": (text[train_index], text[test_index]),
            "combined": (combined[train_index], combined[test_index]),
            "shuffled_probe": (combined[train_index], shuffled_combined_test),
        }
        for name, (x_train, x_test) in matrices.items():
            for target_index in range(len(SOURCE_ORDER)):
                predictions[name][test_index, target_index] = _predict_binary(
                    x_train,
                    source_y[train_index, target_index],
                    x_test,
                    seed=frozen.seed + fold * 1000 + target_index,
                )
            train_strategy = train_index & strategy_mask
            test_strategy = test_index & strategy_mask
            if np.any(test_strategy):
                if np.any(train_strategy):
                    x_strategy_train = (
                        probe[train_strategy]
                        if name == "probe_only"
                        else text[train_strategy]
                        if name == "text_only"
                        else combined[train_strategy]
                    )
                    x_strategy_test = (
                        probe[test_strategy]
                        if name == "probe_only"
                        else text[test_strategy]
                        if name == "text_only"
                        else (
                            np.concatenate(
                                [
                                    text[test_strategy],
                                    probe[
                                        rng.permutation(
                                            np.flatnonzero(test_strategy)
                                        )
                                    ],
                                ],
                                axis=1,
                            )
                            if name == "shuffled_probe"
                            else combined[test_strategy]
                        )
                    )
                    strategy_predictions[name][test_strategy] = _predict_binary(
                        x_strategy_train,
                        strategy_y[train_strategy],
                        x_strategy_test,
                        seed=frozen.seed + fold * 1000 + 99,
                    )
            fold_rows.append(
                {
                    "fold": fold,
                    "train_users": sorted(
                        {
                            state.user_id
                            for state, keep in zip(
                                train_states, train_index, strict=True
                            )
                            if keep
                        }
                    ),
                    "test_users": sorted(
                        {
                            state.user_id
                            for state, keep in zip(
                                train_states, test_index, strict=True
                            )
                            if keep
                        }
                    ),
                }
            )
    metrics = {
        name: _metric(
            source_y,
            predictions[name],
            strategy_y,
            strategy_predictions[name],
            strategy_mask,
        )
        for name in predictions
    }
    challenges = _challenge_coverage(
        train_states,
        step0_by_state,
        evaluator_contexts,
        frozen,
    )
    checks = {
        "probe_only_not_oracle_key": metrics["probe_only"]["source_exact_match"]
        <= frozen.maximum_probe_only_exact_match,
        "combined_improves_over_text": metrics["combined"]["source_exact_match"]
        >= metrics["text_only"]["source_exact_match"]
        + frozen.minimum_combined_gain_over_text,
        "shuffled_probe_degrades": metrics["shuffled_probe"]["source_exact_match"]
        <= metrics["combined"]["source_exact_match"]
        - frozen.minimum_shuffled_probe_drop,
        **{
            f"challenge_{name}": count >= frozen.minimum_challenge_count
            for name, count in challenges.items()
        },
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": PROTOCOL,
        "config": frozen.payload(),
        "config_sha256": frozen.digest(),
        "scope": "train_users_only_for_predictive_diagnostics",
        "n_states": len(train_states),
        "n_users": len({state.user_id for state in train_states}),
        "metrics": metrics,
        "challenge_coverage": challenges,
        "checks": checks,
        "folds": fold_rows,
        "claim_boundary": (
            "diagnostic shortcut gate; not evidence of real-world source correctness"
        ),
    }
