"""Train-only support-need observations and small-sample learning utilities.

This module deliberately does not replace the frozen PM-v1.5 ``Step0Observation``.
It implements the separate V3 research path described in
``PM_V1_5_UNIFIED_NEED_SEMANTICS_TRAINING_PLAN_V3_ZH.md``:

* BAAI is a frozen multi-view representation, not a need classifier;
* visible boundary cues remain observable inputs, not action labels;
* user/dialogue groups, rather than action/judge rows, are the independent units;
* categorical heads consume soft partial labels and emit calibrated distributions;
* every formal consumer must use out-of-fold observations during development.

No function in this module reads memory items, retrieved cards, evaluator outcomes,
internal-test labels, or external-test outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import Field, field_validator, model_validator
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .contracts import MemorySource
from .io import canonical_json, sha256_text
from .pm_v1_5_semantic import SemanticTextEncoder
from .pm_v2_contracts import PMV2Split, PMV2State, STRATEGY_FAMILY_IDS, StrictModel


SUPPORT_NEED_PROTOCOL = "pm-v1.5-support-need-observation-v3"
SUPPORT_NEED_LEARNING_PROTOCOL = (
    "pm-v1.5-support-need-small-sample-crossfit-v1"
)
SUPPORT_NEED_CANARY_PROTOCOL = "pm-v1.5-support-need-learning-canary-v1"

SUPPORT_MODE_IDS: tuple[str, ...] = (
    "listen",
    "explore",
    "comfort_reassure",
    "light_guidance",
    "structured_planning",
)
GOAL_IDS: tuple[str, ...] = (
    "be_heard",
    "make_sense",
    "stabilize",
    "decide",
    "act",
)
DIALOGUE_PHASE_IDS: tuple[str, ...] = (
    "exploration",
    "comforting",
    "action",
)
NONCLINICAL_URGENCY_IDS: tuple[str, ...] = (
    "routine",
    "elevated",
    "acute",
)
VIEW_NAMES: tuple[str, ...] = (
    "current_user",
    "last_assistant",
    "recent_history",
    "session_summary",
    "current_vs_context_delta",
)


def _normalize_text(value: Any, *, empty: str = "[none]") -> str:
    normalized = " ".join(str(value or "").split())
    return normalized or empty


def _turn_parts(turn: Any) -> tuple[str, str]:
    if isinstance(turn, Mapping):
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "")
    else:
        role = str(getattr(turn, "role", ""))
        content = str(getattr(turn, "content", ""))
    if role not in {"user", "assistant"} or not content.strip():
        raise ValueError("support-need views contain an invalid dialogue turn")
    return role, _normalize_text(content)


def _exact_probability_distribution(
    value: Mapping[str, float],
    *,
    expected: Sequence[str],
    label: str,
    require_sum_one: bool = True,
) -> dict[str, float]:
    parsed = {str(key): float(score) for key, score in value.items()}
    if set(parsed) != set(expected):
        raise ValueError(f"{label} keys do not match the frozen contract")
    if any(not math.isfinite(score) or score < 0.0 or score > 1.0 for score in parsed.values()):
        raise ValueError(f"{label} contains a probability outside [0, 1]")
    if require_sum_one and not math.isclose(
        sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(f"{label} probabilities must sum to one")
    return parsed


class ExplicitSupportBoundaries(StrictModel):
    """Conservative observable cues; none is itself an RS/action gold label."""

    advice_rejected: bool
    advice_requested: bool
    one_small_step_requested: bool
    listen_first_requested: bool
    question_or_task_burden_limit: bool


class MemorySourceOpportunityObservation(StrictModel):
    available: bool
    eligible_count: int = Field(ge=0)
    semantic_fit: float = Field(ge=-1.0, le=1.0)
    recency_compatibility: float = Field(ge=0.0, le=1.0)
    expected_cost: float = Field(ge=0.0)
    opportunity_probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def unavailable_is_zero(self):
        if not self.available and (
            self.eligible_count != 0
            or self.semantic_fit != 0.0
            or self.recency_compatibility != 0.0
            or self.expected_cost != 0.0
            or self.opportunity_probability != 0.0
        ):
            raise ValueError("unavailable memory source must expose zero opportunity")
        return self


class StrategyFamilyOpportunityObservation(StrictModel):
    eligible_card_count: int = Field(ge=0)
    semantic_fit: float = Field(ge=-1.0, le=1.0)
    mode_phase_goal_compatibility: float = Field(ge=0.0, le=1.0)
    directive_burden_fit: float = Field(ge=0.0, le=1.0)
    opportunity_probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def empty_family_is_zero(self):
        if self.eligible_card_count == 0 and (
            self.semantic_fit != 0.0
            or self.mode_phase_goal_compatibility != 0.0
            or self.directive_burden_fit != 0.0
            or self.opportunity_probability != 0.0
        ):
            raise ValueError("empty strategy family must expose zero opportunity")
        return self


class SupportNeedUncertainty(StrictModel):
    predictive_entropy: float = Field(ge=0.0)
    group_bootstrap_dispersion: float = Field(ge=0.0)
    semantic_ood: bool
    metadata_ood: bool
    abstain: bool
    abstain_reasons: list[str]

    @model_validator(mode="after")
    def coherent_abstention(self):
        if self.abstain != bool(self.abstain_reasons):
            raise ValueError("abstain must agree with abstain_reasons")
        if len(self.abstain_reasons) != len(set(self.abstain_reasons)):
            raise ValueError("abstain_reasons must be unique")
        return self


class SupportNeedObservation(StrictModel):
    protocol: Literal["pm-v1.5-support-need-observation-v3"] = (
        SUPPORT_NEED_PROTOCOL
    )
    observation_stage: Literal["pre_item_retrieval"] = "pre_item_retrieval"
    state_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    explicit_boundaries: ExplicitSupportBoundaries
    support_mode_distribution: dict[str, float]
    goal_probabilities: dict[str, float]
    dialogue_phase_distribution: dict[str, float]
    nonclinical_urgency_distribution: dict[str, float]
    memory_source_opportunity: dict[MemorySource, MemorySourceOpportunityObservation]
    strategy_family_opportunity: dict[str, StrategyFamilyOpportunityObservation]
    uncertainty: SupportNeedUncertainty
    visible_view_sha256: dict[str, str]
    feature_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: dict[str, Any]

    @field_validator("support_mode_distribution")
    @classmethod
    def exact_modes(cls, value):
        return _exact_probability_distribution(
            value, expected=SUPPORT_MODE_IDS, label="support mode"
        )

    @field_validator("goal_probabilities")
    @classmethod
    def exact_goals(cls, value):
        return _exact_probability_distribution(
            value,
            expected=GOAL_IDS,
            label="goal",
            require_sum_one=False,
        )

    @field_validator("dialogue_phase_distribution")
    @classmethod
    def exact_phases(cls, value):
        return _exact_probability_distribution(
            value, expected=DIALOGUE_PHASE_IDS, label="dialogue phase"
        )

    @field_validator("nonclinical_urgency_distribution")
    @classmethod
    def exact_urgency(cls, value):
        return _exact_probability_distribution(
            value,
            expected=NONCLINICAL_URGENCY_IDS,
            label="nonclinical urgency",
        )

    @field_validator("memory_source_opportunity", mode="before")
    @classmethod
    def parse_memory_keys(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError("memory_source_opportunity must be an object")
        return {
            key if isinstance(key, MemorySource) else MemorySource(str(key)): row
            for key, row in value.items()
        }

    @model_validator(mode="after")
    def complete_contract(self):
        if set(self.memory_source_opportunity) != set(MemorySource):
            raise ValueError("support-need observation must cover MP, MS, and ME")
        if set(self.strategy_family_opportunity) != set(STRATEGY_FAMILY_IDS):
            raise ValueError("strategy opportunity must cover every frozen family")
        if set(self.visible_view_sha256) != set(VIEW_NAMES):
            raise ValueError("visible-view hashes do not match the frozen contract")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in self.visible_view_sha256.values()
        ):
            raise ValueError("visible-view hashes must be SHA-256 values")
        if self.provenance.get("internal_test_outcomes_opened") is not False:
            raise ValueError("support-need provenance must seal internal outcomes")
        if self.provenance.get("external_outcomes_opened") is not False:
            raise ValueError("support-need provenance must seal external outcomes")
        return self


class SupportNeedPartialLabel(StrictModel):
    """One source-specific train-only partial label; never an action gold."""

    protocol: Literal["pm-v1.5-support-need-partial-label-v1"] = (
        "pm-v1.5-support-need-partial-label-v1"
    )
    state_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    label_source_id: str = Field(min_length=1)
    label_family: Literal[
        "explicit_boundary",
        "minimal_counterfactual",
        "human_anchor",
        "llm_weak",
        "bank_eligibility",
    ]
    label_role: Literal[
        "support_need",
        "resource_opportunity",
        "mixed_partial",
    ]
    source_reliability: float = Field(gt=0.0, le=1.0)
    order_stability: float = Field(gt=0.0, le=1.0)
    abstain: bool
    support_mode_distribution: dict[str, float] | None = None
    goal_probabilities: dict[str, float] | None = None
    dialogue_phase_distribution: dict[str, float] | None = None
    nonclinical_urgency_distribution: dict[str, float] | None = None
    memory_source_opportunity_probabilities: dict[MemorySource, float] | None = None
    strategy_family_opportunity_probabilities: dict[str, float] | None = None
    provenance: dict[str, Any]

    @field_validator("support_mode_distribution")
    @classmethod
    def valid_modes(cls, value):
        if value is None:
            return None
        return _exact_probability_distribution(
            value, expected=SUPPORT_MODE_IDS, label="support-mode partial label"
        )

    @field_validator("goal_probabilities")
    @classmethod
    def valid_goals(cls, value):
        if value is None:
            return None
        return _exact_probability_distribution(
            value,
            expected=GOAL_IDS,
            label="goal partial label",
            require_sum_one=False,
        )

    @field_validator("dialogue_phase_distribution")
    @classmethod
    def valid_phases(cls, value):
        if value is None:
            return None
        return _exact_probability_distribution(
            value,
            expected=DIALOGUE_PHASE_IDS,
            label="phase partial label",
        )

    @field_validator("nonclinical_urgency_distribution")
    @classmethod
    def valid_urgency(cls, value):
        if value is None:
            return None
        return _exact_probability_distribution(
            value,
            expected=NONCLINICAL_URGENCY_IDS,
            label="urgency partial label",
        )

    @field_validator("memory_source_opportunity_probabilities", mode="before")
    @classmethod
    def parse_partial_memory_keys(cls, value):
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("memory opportunity partial label must be an object")
        return {
            key if isinstance(key, MemorySource) else MemorySource(str(key)): score
            for key, score in value.items()
        }

    @model_validator(mode="after")
    def coherent_partial_label(self):
        payloads = (
            self.support_mode_distribution,
            self.goal_probabilities,
            self.dialogue_phase_distribution,
            self.nonclinical_urgency_distribution,
            self.memory_source_opportunity_probabilities,
            self.strategy_family_opportunity_probabilities,
        )
        if self.abstain:
            if any(value is not None for value in payloads):
                raise ValueError("abstaining partial label cannot carry targets")
        elif not any(value is not None for value in payloads):
            raise ValueError("non-abstaining partial label must carry a target")
        if self.memory_source_opportunity_probabilities is not None:
            if set(self.memory_source_opportunity_probabilities) != set(MemorySource):
                raise ValueError("memory opportunity partial label must cover MP/MS/ME")
            if any(
                not 0.0 <= float(value) <= 1.0
                for value in self.memory_source_opportunity_probabilities.values()
            ):
                raise ValueError("memory opportunity probability is outside [0, 1]")
        if self.strategy_family_opportunity_probabilities is not None:
            if set(self.strategy_family_opportunity_probabilities) != set(
                STRATEGY_FAMILY_IDS
            ):
                raise ValueError(
                    "strategy opportunity partial label must cover frozen families"
                )
            if any(
                not 0.0 <= float(value) <= 1.0
                for value in self.strategy_family_opportunity_probabilities.values()
            ):
                raise ValueError("strategy opportunity probability is outside [0, 1]")
        if self.provenance.get("automatic_gold_label") is not False:
            raise ValueError("partial labels must reject an automatic-gold claim")
        if self.provenance.get("internal_test_outcomes_opened") is not False:
            raise ValueError("partial labels must seal internal outcomes")
        if self.provenance.get("external_outcomes_opened") is not False:
            raise ValueError("partial labels must seal external outcomes")
        return self


@dataclass(frozen=True)
class PreparedSupportNeedViews:
    """In-memory numerical features; text and vectors are never label sources."""

    state_id: str
    user_id: str
    view_names: tuple[str, ...]
    matrix: np.ndarray
    explicit_boundaries: ExplicitSupportBoundaries
    audit: Mapping[str, Any]

    @property
    def flattened_features(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float).reshape(-1)


def explicit_support_boundaries(text: str) -> ExplicitSupportBoundaries:
    """Extract narrow English boundary cues without inferring an action."""

    normalized = " ".join(text.lower().split())

    def has(*patterns: str) -> bool:
        return any(re.search(pattern, normalized) for pattern in patterns)

    advice_rejected = has(
        r"\b(?:do not|don't|dont|not looking for|no)\s+(?:want\s+)?(?:any\s+)?advice\b",
        r"\bjust (?:listen|hear me out)\b",
        r"\bwithout (?:giving|offering) (?:me )?advice\b",
    )
    advice_requested = has(
        r"\bwhat (?:should|could) i do\b",
        r"\b(?:can|could|would) you (?:give|offer) (?:me )?(?:some )?advice\b",
        r"\bhelp me (?:decide|plan|figure out what to do)\b",
    )
    one_small_step_requested = has(
        r"\b(?:one|a) (?:small|tiny|manageable) (?:step|thing)\b",
        r"\bwhere (?:do|should|could) i start\b",
    )
    listen_first_requested = has(
        r"\b(?:just|only) (?:want|need) (?:you|someone) to listen\b",
        r"\bcan i (?:just )?(?:talk|vent)\b",
        r"\bhear me out\b",
        r"\bplease (?:just )?listen\b",
    )
    question_or_task_burden_limit = has(
        r"\bone question at a time\b",
        r"\b(?:don't|do not) ask (?:me )?(?:too many|multiple) questions\b",
        r"\bkeep it (?:simple|brief|short)\b",
    )
    return ExplicitSupportBoundaries(
        advice_rejected=advice_rejected,
        advice_requested=advice_requested,
        one_small_step_requested=one_small_step_requested,
        listen_first_requested=listen_first_requested,
        question_or_task_burden_limit=question_or_task_burden_limit,
    )


def prepare_support_need_views(
    encoder: SemanticTextEncoder,
    state: PMV2State,
) -> PreparedSupportNeedViews:
    """Encode four visible views and one deterministic current-context delta."""

    turns = [_turn_parts(turn) for turn in state.current_session_history]
    last_assistant = next(
        (content for role, content in reversed(turns) if role == "assistant"),
        "[none]",
    )
    recent_history = "\n".join(f"{role}: {content}" for role, content in turns)
    texts = (
        _normalize_text(state.current_user_text),
        _normalize_text(last_assistant),
        _normalize_text(recent_history),
        _normalize_text(state.current_session_summary),
    )
    matrix = np.asarray(encoder.encode(texts), dtype=float)
    expected = (4, int(encoder.spec.output_dimension))
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            f"support-need view matrix has shape {matrix.shape}, expected {expected}"
        )
    context = np.mean(matrix[1:], axis=0)
    delta = matrix[0] - context
    norm = float(np.linalg.norm(delta))
    delta = delta / norm if norm > 0.0 else np.zeros_like(delta)
    full = np.vstack([matrix, delta])
    view_hashes = {
        name: sha256_text(text)
        for name, text in zip(VIEW_NAMES[:4], texts, strict=True)
    }
    view_hashes[VIEW_NAMES[4]] = sha256_text(canonical_json(delta.tolist()))
    vector_hashes = {
        name: sha256_text(canonical_json(full[index].tolist()))
        for index, name in enumerate(VIEW_NAMES)
    }
    feature_contract = {
        "protocol": SUPPORT_NEED_PROTOCOL,
        "view_names": list(VIEW_NAMES),
        "per_view_dimension": int(encoder.spec.output_dimension),
        "delta": "l2-normalized-current-minus-mean-context",
        "encoder_spec_sha256": encoder.binding.spec_sha256,
        "encoder_snapshot_tree_sha256": encoder.binding.snapshot_tree_sha256,
        "item_level_retrieval_performed": False,
    }
    feature_contract_sha256 = sha256_text(canonical_json(feature_contract))
    audit = {
        **feature_contract,
        "feature_contract_sha256": feature_contract_sha256,
        "visible_view_sha256": view_hashes,
        "view_vector_sha256": vector_hashes,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    audit["observation_sha256"] = sha256_text(canonical_json(audit))
    return PreparedSupportNeedViews(
        state_id=state.state_id,
        user_id=state.user_id,
        view_names=VIEW_NAMES,
        matrix=full,
        explicit_boundaries=explicit_support_boundaries(state.current_user_text),
        audit=audit,
    )


def support_need_feature_vector(
    prepared: PreparedSupportNeedViews,
    state: PMV2State,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Combine frozen view vectors with transparent pre-retrieval scalars."""

    if prepared.state_id != state.state_id or prepared.user_id != state.user_id:
        raise ValueError("prepared support-need views do not match the state")
    boundary_names = (
        "advice_rejected",
        "advice_requested",
        "one_small_step_requested",
        "listen_first_requested",
        "question_or_task_burden_limit",
    )
    boundary_values = [
        float(getattr(prepared.explicit_boundaries, name))
        for name in boundary_names
    ]
    metadata_names: list[str] = []
    metadata_values: list[float] = []
    for source in MemorySource:
        row = state.inventory[source]
        metadata_names.extend(
            [
                f"{source.value}.available",
                f"{source.value}.count_log1p",
                f"{source.value}.median_age_log1p",
                f"{source.value}.estimated_tokens_log1p",
            ]
        )
        metadata_values.extend(
            [
                float(row.available),
                float(math.log1p(row.count)),
                float(math.log1p(row.median_age_sessions or 0.0)),
                float(math.log1p(row.estimated_tokens)),
            ]
        )
    metadata_names.extend(
        ["strategy_catalog_count_log1p", "strategy_estimated_tokens_log1p"]
    )
    metadata_values.extend(
        [
            float(math.log1p(state.strategy_catalog_count)),
            float(math.log1p(state.strategy_estimated_tokens)),
        ]
    )
    semantic_names = tuple(
        f"{view}.semantic_{dimension:04d}"
        for view in prepared.view_names
        for dimension in range(prepared.matrix.shape[1])
    )
    names = (
        semantic_names
        + tuple(f"boundary.{name}" for name in boundary_names)
        + tuple(metadata_names)
    )
    values = np.concatenate(
        [
            prepared.flattened_features,
            np.asarray(boundary_values, dtype=float),
            np.asarray(metadata_values, dtype=float),
        ]
    )
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise RuntimeError("support-need feature vector is malformed")
    return values, names


def group_normalized_weights(
    groups: Sequence[str],
    base_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Cap every independent group at total weight one without amplifying it."""

    if not groups:
        raise ValueError("groups must be non-empty")
    raw = np.ones(len(groups), dtype=float)
    if base_weights is not None:
        raw = np.asarray(base_weights, dtype=float)
        if raw.shape != (len(groups),):
            raise ValueError("base_weights do not align with groups")
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("base weights must be finite and positive")
    totals: dict[str, float] = {}
    for group, weight in zip(groups, raw, strict=True):
        totals[str(group)] = totals.get(str(group), 0.0) + float(weight)
    return np.asarray(
        [
            float(weight) / max(1.0, totals[str(group)])
            for group, weight in zip(groups, raw, strict=True)
        ],
        dtype=float,
    )


def effective_sample_size(weights: Sequence[float]) -> float:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ESS requires a non-empty one-dimensional weight vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("ESS weights must be finite and non-negative")
    denominator = float(values @ values)
    if denominator <= 0.0:
        raise ValueError("ESS weights must contain positive mass")
    return float(values.sum() ** 2 / denominator)


def group_effective_sample_size(
    groups: Sequence[str], weights: Sequence[float]
) -> float:
    """Compute ESS after collapsing repeated rows to independent groups."""

    values = np.asarray(weights, dtype=float)
    if values.shape != (len(groups),):
        raise ValueError("group ESS weights do not align with groups")
    totals: dict[str, float] = {}
    for group, weight in zip(groups, values, strict=True):
        totals[str(group)] = totals.get(str(group), 0.0) + float(weight)
    return effective_sample_size(list(totals.values()))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _soft_log_loss(
    truth: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> float:
    clipped = np.clip(predicted, 1e-12, 1.0)
    per_row = -np.sum(truth * np.log(clipped), axis=1)
    return float(np.average(per_row, weights=weights))


def _projection_dimension(
    requested: int,
    *,
    feature_count: int,
    training_rows: int,
    training_group_count: int,
) -> int:
    capacity = max(1, training_group_count // 5)
    return max(
        1,
        min(
            int(requested),
            int(feature_count),
            max(1, int(training_rows) - 1),
            capacity,
        ),
    )


def _fit_predict_ridge(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    test_x: np.ndarray,
    dimension: int,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_test = scaler.transform(test_x)
    projector = PCA(n_components=int(dimension), svd_solver="full")
    projected_train = projector.fit_transform(scaled_train)
    projected_test = projector.transform(scaled_test)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(projected_train, train_y, sample_weight=train_weights)
    return _softmax(np.asarray(model.predict(projected_test), dtype=float))


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("calibration temperature must be positive")
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
    return _softmax(logits)


def cross_fit_categorical_soft_head(
    *,
    features: Sequence[Sequence[float]],
    targets: Sequence[Mapping[str, float]],
    groups: Sequence[str],
    class_ids: Sequence[str],
    base_weights: Sequence[float] | None = None,
    projection_dimensions: Sequence[int] = (4, 8, 12),
    alphas: Sequence[float] = (1.0, 10.0, 100.0),
    folds: int = 5,
) -> dict[str, Any]:
    """Return strictly group-held-out predictions for one soft categorical head."""

    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(groups) or x.shape[0] == 0:
        raise ValueError("features must be a non-empty row-aligned matrix")
    if np.any(~np.isfinite(x)):
        raise ValueError("features contain non-finite values")
    classes = tuple(str(value) for value in class_ids)
    if len(classes) < 2 or len(classes) != len(set(classes)):
        raise ValueError("class_ids must contain at least two unique values")
    y = np.asarray(
        [
            [
                _exact_probability_distribution(
                    row, expected=classes, label="cross-fit target"
                )[class_id]
                for class_id in classes
            ]
            for row in targets
        ],
        dtype=float,
    )
    if y.shape != (x.shape[0], len(classes)):
        raise ValueError("targets do not align with features")
    group_values = np.asarray([str(group) for group in groups], dtype=object)
    unique_groups = sorted(set(group_values.tolist()))
    if len(unique_groups) < 3:
        raise ValueError("cross-fitting requires at least three independent groups")
    sample_weights = group_normalized_weights(groups, base_weights)
    split_count = min(int(folds), len(unique_groups))
    if split_count < 2:
        raise ValueError("cross-fitting requires at least two folds")
    outer = GroupKFold(n_splits=split_count)
    predictions = np.zeros_like(y)
    fold_rows: list[dict[str, Any]] = []

    for fold_index, (train_index, test_index) in enumerate(
        outer.split(x, np.argmax(y, axis=1), group_values)
    ):
        train_groups = group_values[train_index]
        if set(train_groups.tolist()) & set(group_values[test_index].tolist()):
            raise RuntimeError("group leakage detected in outer cross-fit")
        inner_group_count = len(set(train_groups.tolist()))
        inner_folds = min(3, inner_group_count)
        if inner_folds < 2:
            raise RuntimeError("outer training fold lacks groups for model selection")
        candidate_rows: list[dict[str, Any]] = []
        for requested_dimension in sorted(set(int(value) for value in projection_dimensions)):
            dimension = _projection_dimension(
                requested_dimension,
                feature_count=x.shape[1],
                training_rows=len(train_index),
                training_group_count=inner_group_count,
            )
            for alpha in sorted(set(float(value) for value in alphas), reverse=True):
                losses: list[float] = []
                inner = GroupKFold(n_splits=inner_folds)
                for inner_train_local, inner_valid_local in inner.split(
                    x[train_index],
                    np.argmax(y[train_index], axis=1),
                    train_groups,
                ):
                    inner_train = train_index[inner_train_local]
                    inner_valid = train_index[inner_valid_local]
                    inner_dimension = _projection_dimension(
                        dimension,
                        feature_count=x.shape[1],
                        training_rows=len(inner_train),
                        training_group_count=len(
                            set(group_values[inner_train].tolist())
                        ),
                    )
                    predicted = _fit_predict_ridge(
                        train_x=x[inner_train],
                        train_y=y[inner_train],
                        train_weights=sample_weights[inner_train],
                        test_x=x[inner_valid],
                        dimension=inner_dimension,
                        alpha=alpha,
                    )
                    losses.append(
                        _soft_log_loss(
                            y[inner_valid],
                            predicted,
                            sample_weights[inner_valid],
                        )
                    )
                candidate_rows.append(
                    {
                        "requested_dimension": requested_dimension,
                        "effective_dimension": dimension,
                        "alpha": alpha,
                        "fold_losses": losses,
                        "mean_loss": float(np.mean(losses)),
                        "standard_error": (
                            float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
                            if len(losses) > 1
                            else 0.0
                        ),
                    }
                )
        best = min(
            candidate_rows,
            key=lambda row: (
                row["mean_loss"],
                row["effective_dimension"],
                -row["alpha"],
            ),
        )
        eligible = [
            row
            for row in candidate_rows
            if row["mean_loss"]
            <= best["mean_loss"] + best["standard_error"] + 1e-12
        ]
        chosen = min(
            eligible,
            key=lambda row: (
                row["effective_dimension"],
                -row["alpha"],
                row["mean_loss"],
            ),
        )
        # Fit calibration only on inner held-out predictions.  The outer fold
        # remains unseen until both the model and temperature are frozen.
        calibration_predictions = np.zeros_like(y[train_index])
        inner = GroupKFold(n_splits=inner_folds)
        for inner_train_local, inner_valid_local in inner.split(
            x[train_index],
            np.argmax(y[train_index], axis=1),
            train_groups,
        ):
            inner_train = train_index[inner_train_local]
            inner_valid = train_index[inner_valid_local]
            inner_dimension = _projection_dimension(
                int(chosen["effective_dimension"]),
                feature_count=x.shape[1],
                training_rows=len(inner_train),
                training_group_count=len(
                    set(group_values[inner_train].tolist())
                ),
            )
            calibration_predictions[inner_valid_local] = _fit_predict_ridge(
                train_x=x[inner_train],
                train_y=y[inner_train],
                train_weights=sample_weights[inner_train],
                test_x=x[inner_valid],
                dimension=inner_dimension,
                alpha=float(chosen["alpha"]),
            )
        temperature_candidates = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
        temperature = min(
            temperature_candidates,
            key=lambda candidate: (
                _soft_log_loss(
                    y[train_index],
                    _temperature_scale(calibration_predictions, candidate),
                    sample_weights[train_index],
                ),
                abs(math.log(candidate)),
            ),
        )
        effective_dimension = _projection_dimension(
            int(chosen["effective_dimension"]),
            feature_count=x.shape[1],
            training_rows=len(train_index),
            training_group_count=inner_group_count,
        )
        outer_predictions = _fit_predict_ridge(
            train_x=x[train_index],
            train_y=y[train_index],
            train_weights=sample_weights[train_index],
            test_x=x[test_index],
            dimension=effective_dimension,
            alpha=float(chosen["alpha"]),
        )
        predictions[test_index] = _temperature_scale(
            outer_predictions, temperature
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "training_groups": inner_group_count,
                "validation_groups": len(set(group_values[test_index].tolist())),
                "effective_dimension": effective_dimension,
                "alpha": float(chosen["alpha"]),
                "calibration_temperature": float(temperature),
                "calibration_source": "inner-group-oof-only",
                "selection_rule": "one-standard-error-simplest",
            }
        )

    truth_class = np.argmax(y, axis=1)
    predicted_class = np.argmax(predictions, axis=1)
    recalls = {}
    for index, class_id in enumerate(classes):
        mask = truth_class == index
        recalls[class_id] = (
            float(np.mean(predicted_class[mask] == index)) if np.any(mask) else None
        )
    report = {
        "protocol": SUPPORT_NEED_LEARNING_PROTOCOL,
        "status": "COMPLETE_TRAIN_ONLY_OUT_OF_FOLD",
        "rows": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "independent_groups": len(unique_groups),
        "total_group_weight": float(sample_weights.sum()),
        "group_normalized_ess": group_effective_sample_size(
            groups, sample_weights
        ),
        "class_ids": list(classes),
        "folds": fold_rows,
        "soft_log_loss": _soft_log_loss(y, predictions, sample_weights),
        "brier_score": float(
            np.average(
                np.sum((predictions - y) ** 2, axis=1),
                weights=sample_weights,
            )
        ),
        "accuracy": float(
            np.average(predicted_class == truth_class, weights=sample_weights)
        ),
        "per_class_recall": recalls,
        "oof_predictions": [
            {class_id: float(predictions[row, column]) for column, class_id in enumerate(classes)}
            for row in range(len(groups))
        ],
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    report["report_sha256"] = sha256_text(canonical_json(report))
    return report


def _aggregate_partial_head(
    *,
    labels: Sequence[SupportNeedPartialLabel],
    value,
    class_ids: Sequence[str],
) -> tuple[list[str], list[dict[str, float]], list[float], dict[str, Any]]:
    """Reliability-weight source rows into one soft target per state."""

    by_state: dict[str, list[tuple[Mapping[str, float], float]]] = {}
    source_rows = 0
    abstentions = 0
    for label in labels:
        if label.abstain:
            abstentions += 1
            continue
        target = value(label)
        if target is None:
            continue
        source_rows += 1
        weight = float(label.source_reliability * label.order_stability)
        by_state.setdefault(label.state_id, []).append((target, weight))
    state_ids: list[str] = []
    targets: list[dict[str, float]] = []
    weights: list[float] = []
    for state_id, rows in sorted(by_state.items()):
        total = sum(weight for _, weight in rows)
        if total <= 0.0:
            continue
        target = {
            class_id: sum(float(row[class_id]) * weight for row, weight in rows)
            / total
            for class_id in class_ids
        }
        state_ids.append(state_id)
        targets.append(target)
        # Multiple correlated labeling functions improve confidence but may not
        # create more than one state-level unit of mass.
        weights.append(min(1.0, total))
    return state_ids, targets, weights, {
        "source_rows": source_rows,
        "abstentions": abstentions,
        "states_with_target": len(state_ids),
    }


def _binary_target(probability: float) -> dict[str, float]:
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("binary partial-label probability is outside [0, 1]")
    return {"negative": 1.0 - value, "positive": value}


def run_train_only_support_need_pilot(
    *,
    states: Sequence[PMV2State],
    prepared_views: Sequence[PreparedSupportNeedViews],
    labels: Sequence[SupportNeedPartialLabel],
) -> dict[str, Any]:
    """Fit every sufficiently supported head with group-held-out predictions."""

    if not states:
        raise ValueError("support-need pilot requires train states")
    if any(state.split is not PMV2Split.TRAIN for state in states):
        raise RuntimeError("support-need pilot is strictly train-only")
    state_by_id = {state.state_id: state for state in states}
    if len(state_by_id) != len(states):
        raise ValueError("support-need pilot states must be unique")
    views_by_id = {row.state_id: row for row in prepared_views}
    if set(views_by_id) != set(state_by_id):
        raise RuntimeError("prepared views do not exactly cover train states")
    if any(
        row.user_id != state_by_id[row.state_id].user_id
        for row in prepared_views
    ):
        raise RuntimeError("support-need views/state users differ")
    for label in labels:
        state = state_by_id.get(label.state_id)
        if (
            state is None
            or label.user_id != state.user_id
            or label.group_id != state.user_id
        ):
            raise RuntimeError(
                "partial label is outside the train state/user group universe"
            )
    feature_rows = {}
    feature_names: tuple[str, ...] | None = None
    for state_id, state in sorted(state_by_id.items()):
        values, names = support_need_feature_vector(views_by_id[state_id], state)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("support-need feature names drifted between states")
        feature_rows[state_id] = values

    head_specs: list[tuple[str, tuple[str, ...], Any]] = [
        (
            "support_mode",
            SUPPORT_MODE_IDS,
            lambda row: row.support_mode_distribution,
        ),
        (
            "dialogue_phase",
            DIALOGUE_PHASE_IDS,
            lambda row: row.dialogue_phase_distribution,
        ),
        (
            "nonclinical_urgency",
            NONCLINICAL_URGENCY_IDS,
            lambda row: row.nonclinical_urgency_distribution,
        ),
    ]
    for goal in GOAL_IDS:
        head_specs.append(
            (
                f"goal.{goal}",
                ("negative", "positive"),
                lambda row, goal=goal: (
                    _binary_target(row.goal_probabilities[goal])
                    if row.goal_probabilities is not None
                    else None
                ),
            )
        )
    for source in MemorySource:
        head_specs.append(
            (
                f"memory_opportunity.{source.value}",
                ("negative", "positive"),
                lambda row, source=source: (
                    _binary_target(
                        row.memory_source_opportunity_probabilities[source]
                    )
                    if row.memory_source_opportunity_probabilities is not None
                    else None
                ),
            )
        )
    for family in STRATEGY_FAMILY_IDS:
        head_specs.append(
            (
                f"strategy_opportunity.{family}",
                ("negative", "positive"),
                lambda row, family=family: (
                    _binary_target(
                        row.strategy_family_opportunity_probabilities[family]
                    )
                    if row.strategy_family_opportunity_probabilities is not None
                    else None
                ),
            )
        )

    reports: dict[str, Any] = {}
    for head_name, class_ids, getter in head_specs:
        state_ids, targets, weights, coverage = _aggregate_partial_head(
            labels=labels,
            value=getter,
            class_ids=class_ids,
        )
        groups = [state_by_id[state_id].user_id for state_id in state_ids]
        hard_counts = {
            class_id: sum(
                max(target, key=target.get) == class_id for target in targets
            )
            for class_id in class_ids
        }
        if (
            len(set(groups)) < 3
            or any(count == 0 for count in hard_counts.values())
        ):
            reports[head_name] = {
                "status": "UNSUPPORTED_INSUFFICIENT_PARTIAL_LABELS",
                "class_ids": list(class_ids),
                "hard_target_group_counts": hard_counts,
                **coverage,
            }
            continue
        report = cross_fit_categorical_soft_head(
            features=[feature_rows[state_id] for state_id in state_ids],
            targets=targets,
            groups=groups,
            class_ids=class_ids,
            base_weights=weights,
        )
        reports[head_name] = {
            **coverage,
            "hard_target_group_counts": hard_counts,
            "cross_fit": report,
        }

    complete_heads = sorted(
        name for name, row in reports.items() if "cross_fit" in row
    )
    unsupported_heads = sorted(set(reports) - set(complete_heads))
    report = {
        "protocol": "pm-v1.5-support-need-train-only-pilot-v1",
        "status": "COMPLETE_TRAIN_ONLY_DIAGNOSTIC",
        "states": len(states),
        "independent_user_groups": len({state.user_id for state in states}),
        "feature_count": len(feature_names or ()),
        "partial_label_rows": len(labels),
        "complete_heads": complete_heads,
        "unsupported_heads": unsupported_heads,
        "heads": reports,
        "formal_fit_authorized": False,
        "automatic_gold_labels_created": False,
        "api_calls_made": 0,
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
    }
    report["report_sha256"] = sha256_text(canonical_json(report))
    return report


def run_support_need_learning_canary(seed: int = 4311) -> dict[str, Any]:
    """Prove the implementation can recover known signal before real fitting."""

    rng = np.random.default_rng(int(seed))
    rows_per_class = 12
    class_count = len(SUPPORT_MODE_IDS)
    feature_count = 20
    features: list[list[float]] = []
    targets: list[dict[str, float]] = []
    groups: list[str] = []
    for class_index, class_id in enumerate(SUPPORT_MODE_IDS):
        center = np.zeros(feature_count, dtype=float)
        center[class_index] = 4.0
        center[5 + class_index] = 2.0
        for row_index in range(rows_per_class):
            features.append((center + rng.normal(0.0, 0.35, feature_count)).tolist())
            targets.append(
                {
                    candidate: 1.0 if candidate == class_id else 0.0
                    for candidate in SUPPORT_MODE_IDS
                }
            )
            groups.append(f"canary_{class_id}_{row_index:02d}")
    recovered = cross_fit_categorical_soft_head(
        features=features,
        targets=targets,
        groups=groups,
        class_ids=SUPPORT_MODE_IDS,
    )
    permutation = rng.permutation(len(targets))
    permuted = cross_fit_categorical_soft_head(
        features=features,
        targets=[targets[int(index)] for index in permutation],
        groups=groups,
        class_ids=SUPPORT_MODE_IDS,
    )
    original_weights = group_normalized_weights(groups)
    duplicated_groups = [group for group in groups for _ in range(2)]
    duplicated_weights = group_normalized_weights(duplicated_groups)
    checks = {
        "known_signal_accuracy_at_least_0_90": recovered["accuracy"] >= 0.90,
        "every_class_recall_at_least_0_80": all(
            value is not None and value >= 0.80
            for value in recovered["per_class_recall"].values()
        ),
        "permuted_accuracy_at_most_0_40": permuted["accuracy"] <= 0.40,
        "duplicating_rows_does_not_inflate_group_ess": math.isclose(
            group_effective_sample_size(groups, original_weights),
            group_effective_sample_size(duplicated_groups, duplicated_weights),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "feature_capacity_respected_in_every_fold": all(
            int(row["effective_dimension"])
            <= int(row["training_groups"]) // 5
            for row in recovered["folds"]
        ),
    }
    report = {
        "protocol": SUPPORT_NEED_CANARY_PROTOCOL,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seed": int(seed),
        "checks": checks,
        "known_signal": {
            key: recovered[key]
            for key in (
                "rows",
                "independent_groups",
                "group_normalized_ess",
                "accuracy",
                "soft_log_loss",
                "brier_score",
                "per_class_recall",
                "folds",
                "report_sha256",
            )
        },
        "label_permutation": {
            "accuracy": permuted["accuracy"],
            "soft_log_loss": permuted["soft_log_loss"],
            "report_sha256": permuted["report_sha256"],
        },
        "internal_test_outcomes_opened": False,
        "external_outcomes_opened": False,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    report["report_sha256"] = sha256_text(canonical_json(report))
    return report
