from __future__ import annotations

from enum import Enum
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import (
    ALL_ACTION_IDS,
    DialogueTurn,
    MemorySource,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=False)


class PMV2Split(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    INTERNAL_TEST = "internal_test"
    EXTERNAL_TEST = "external_test"


class ResourceNeedRegime(str, Enum):
    """Development-only regime label.

    It is used to enforce balanced data generation and auditing. It is never exposed
    to the policy at inference time.
    """

    CONTEXT_ONLY = "context_only"
    PROFILE_NEEDED = "profile_needed"
    SUMMARY_NEEDED = "summary_needed"
    EVENT_NEEDED = "event_needed"
    MULTI_SOURCE_NEEDED = "multi_source_needed"
    MEMORY_HARMFUL = "memory_harmful"
    STRATEGY_HELPFUL = "strategy_helpful"
    STRATEGY_HARMFUL = "strategy_harmful"
    AMBIGUOUS = "ambiguous"


class ObservableSourceSummary(StrictModel):
    """Pre-retrieval source summary.

    The schema intentionally contains no memory text, IDs, retrieved items, or
    per-item similarity values. Query-to-catalog statistics must be computed from a
    cached catalog representation before item retrieval.
    """

    available: bool
    count: int = Field(ge=0)
    min_age_sessions: int | None = Field(default=None, ge=0)
    median_age_sessions: float | None = Field(default=None, ge=0)
    max_age_sessions: int | None = Field(default=None, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    query_similarity_mean: float = Field(default=0.0, ge=-1.0, le=1.0)
    catalog_embedding: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        if self.available != (self.count > 0):
            raise ValueError("available must equal count > 0")
        if not self.available:
            if self.estimated_tokens != 0:
                raise ValueError("unavailable source must have zero estimated_tokens")
            if any(
                value is not None
                for value in (
                    self.min_age_sessions,
                    self.median_age_sessions,
                    self.max_age_sessions,
                )
            ):
                raise ValueError("unavailable source cannot expose age metadata")
        if (
            self.min_age_sessions is not None
            and self.max_age_sessions is not None
            and self.min_age_sessions > self.max_age_sessions
        ):
            raise ValueError("min_age_sessions exceeds max_age_sessions")
        return self


class PMV2State(StrictModel):
    state_id: str
    card_id: str
    user_id: str
    split: PMV2Split
    semantic_family: str = Field(min_length=1)
    surface_form_id: str = Field(min_length=1)
    current_user_text: str = Field(min_length=1)
    current_session_history: list[DialogueTurn]
    current_session_summary: str
    session_index: int = Field(ge=1)
    inventory: dict[MemorySource, ObservableSourceSummary]
    strategy_catalog_count: int = Field(default=0, ge=0)
    strategy_estimated_tokens: int = Field(default=0, ge=0)
    text_embedding: list[float] = Field(default_factory=list)
    allowed_actions: list[str]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("inventory", mode="before")
    @classmethod
    def parse_inventory_keys(cls, value):
        if not isinstance(value, dict):
            raise TypeError("inventory must be an object")
        return {
            key if isinstance(key, MemorySource) else MemorySource(str(key)): item
            for key, item in value.items()
        }

    @field_validator("allowed_actions")
    @classmethod
    def valid_actions(cls, values: list[str]) -> list[str]:
        if not values or len(values) != len(set(values)):
            raise ValueError("allowed_actions must be non-empty and unique")
        for action_id in values:
            parse_action_id(action_id)
        return values

    @model_validator(mode="after")
    def action_mask_matches_inventory(self):
        missing_sources = set(MemorySource) - set(self.inventory)
        if missing_sources:
            raise ValueError(f"inventory missing sources: {sorted(x.value for x in missing_sources)}")
        available = {src for src, cat in self.inventory.items() if cat.available}
        expected = {
            canonical_action_id(sources, strategy)
            for memory_code in (action.split("+")[0] for action in ALL_ACTION_IDS)
            for sources, strategy in [parse_action_id(f"{memory_code}+R0")]
            if sources <= available
            for strategy in StrategyMode
        }
        if set(self.allowed_actions) != expected:
            raise ValueError(
                "action mask mismatch: "
                f"missing={sorted(expected - set(self.allowed_actions))}, "
                f"extra={sorted(set(self.allowed_actions) - expected)}"
            )
        allowed_provenance = {
            "backend_record_id",
            "evaluator_context_id",
            "data_generation_sha256",
            "adapted_from_runtime_state",
            "runtime_provenance_sha256",
        }
        unexpected_provenance = sorted(set(self.provenance) - allowed_provenance)
        if unexpected_provenance:
            raise ValueError(
                "PMV2State provenance may contain operational references only; "
                f"unexpected keys={unexpected_provenance}"
            )
        for key in ("backend_record_id", "evaluator_context_id"):
            if key in self.provenance and (
                not isinstance(self.provenance[key], str)
                or not self.provenance[key]
            ):
                raise ValueError(f"PMV2State provenance {key} must be a non-empty string")
        if (
            "backend_record_id" in self.provenance
            and self.provenance["backend_record_id"] != self.card_id
        ):
            raise ValueError("backend_record_id must equal card_id")
        for key in ("data_generation_sha256", "runtime_provenance_sha256"):
            if key in self.provenance and (
                not isinstance(self.provenance[key], str)
                or len(self.provenance[key]) != 64
            ):
                raise ValueError(f"PMV2State provenance {key} must be a SHA-256 digest")
        if "adapted_from_runtime_state" in self.provenance and not isinstance(
            self.provenance["adapted_from_runtime_state"], bool
        ):
            raise ValueError(
                "PMV2State provenance adapted_from_runtime_state must be boolean"
            )
        return self


class ResponseDimensions(StrictModel):
    """Judge outputs without an LLM-produced overall field.

    PM-v2 never asks a judge for `overall`. A deterministic, versioned composite is
    computed from these independent fields after validation.
    """

    emotional_support: float = Field(ge=1.0, le=5.0)
    personalization: float = Field(ge=1.0, le=5.0)
    memory_appropriateness: float = Field(ge=1.0, le=5.0)
    factual_grounding: float = Field(ge=1.0, le=5.0)
    temporal_consistency: float = Field(ge=1.0, le=5.0)
    non_intrusiveness: float = Field(ge=1.0, le=5.0)


class RiskDimensions(StrictModel):
    selected_context_misuse: float = Field(ge=0.0, le=3.0)
    unnecessary_exposure: float = Field(ge=0.0, le=3.0)
    stale_or_conflicting_use: float = Field(ge=0.0, le=3.0)
    unsupported_personal_claim: float = Field(ge=0.0, le=3.0)
    memory_omission: float = Field(ge=0.0, le=3.0)
    strategy_overuse: float = Field(ge=0.0, le=3.0)
    strategy_omission: float = Field(ge=0.0, le=3.0)


class CompositeSpec(StrictModel):
    version: str = "pmv2-quality-v2"
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "emotional_support": 0.30,
            "personalization": 0.20,
            "memory_appropriateness": 0.15,
            "factual_grounding": 0.15,
            "temporal_consistency": 0.10,
            "non_intrusiveness": 0.10,
        }
    )

    @model_validator(mode="after")
    def validate_weights(self):
        expected = set(ResponseDimensions.model_fields)
        if set(self.weights) != expected:
            raise ValueError(
                f"weights must exactly match response dimensions: {sorted(expected)}"
            )
        if any(value < 0 for value in self.weights.values()):
            raise ValueError("composite weights must be non-negative")
        if abs(sum(self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("composite weights must sum to 1")
        return self

    def score(self, dimensions: ResponseDimensions) -> float:
        data = dimensions.model_dump()
        normalized = {key: (float(value) - 1.0) / 4.0 for key, value in data.items()}
        return float(sum(self.weights[key] * normalized[key] for key in self.weights))


class ActionLabel(StrictModel):
    state_id: str
    card_id: str
    user_id: str
    semantic_family: str
    action_id: str
    response: ResponseDimensions
    risk: RiskDimensions
    observed_input_tokens: int = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    judge_families: list[str] = Field(min_length=1)
    judge_count: int = Field(ge=1)
    max_dimension_mad: float = Field(ge=0.0)
    dimension_mad: dict[str, float]
    label_reliable: bool
    composite_spec_version: str = "pmv2-quality-v2"
    composite_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_id")
    @classmethod
    def valid_action(cls, value: str) -> str:
        parse_action_id(value)
        return value

    @field_validator("dimension_mad")
    @classmethod
    def valid_dimension_mad(cls, value: dict[str, float]) -> dict[str, float]:
        expected = {
            *(f"response.{name}" for name in ResponseDimensions.model_fields),
            *(f"risk.{name}" for name in RiskDimensions.model_fields),
        }
        if set(value) != expected:
            raise ValueError(
                "dimension_mad must exactly cover response and risk dimensions: "
                f"{sorted(expected)}"
            )
        parsed = {str(name): float(score) for name, score in value.items()}
        if any(not math.isfinite(score) or score < 0.0 for score in parsed.values()):
            raise ValueError("dimension_mad values must be finite and non-negative")
        return parsed


class PredictionInterval(StrictModel):
    mean: float
    std: float = Field(ge=0.0)
    lower: float
    upper: float


class ActionPrediction(StrictModel):
    action_id: str
    response: dict[str, PredictionInterval]
    risk: dict[str, PredictionInterval]
    quality_mean: float
    quality_lcb: float
    risk_ucb: float
    estimated_cost: float
    normalized_cost: float
    utility: float
    feasible: bool
    resource_gate_passed: bool
    strategy_gate_passed: bool
    exclusion_reasons: list[str] = Field(default_factory=list)


class PolicyDecision(StrictModel):
    state_id: str
    chosen_action: str
    predictions: dict[str, ActionPrediction]
    semantic_ood_score: float = Field(ge=0.0)
    metadata_ood_score: float = Field(ge=0.0)
    ood_fallback_used: bool
    decision_reason: str
    config_hash: str


class SplitManifest(StrictModel):
    train_users: list[str]
    calibration_users: list[str]
    internal_test_users: list[str]
    train_semantic_families: list[str]
    calibration_semantic_families: list[str]
    internal_test_semantic_families: list[str]
    normalized_text_overlap: int = 0
    user_overlap: int = 0
    semantic_family_overlap: int = 0
    total_states: int = Field(ge=1)
    unique_normalized_current_user_texts: int = Field(ge=1)
    normalized_current_user_text_unique_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def no_overlap(self):
        if self.normalized_text_overlap or self.user_overlap or self.semantic_family_overlap:
            raise ValueError("PM-v2 split manifest contains leakage")
        if self.unique_normalized_current_user_texts != self.total_states:
            raise ValueError(
                "PM-v2 current_user_text must be globally unique across all states"
            )
        expected_rate = self.unique_normalized_current_user_texts / self.total_states
        if abs(self.normalized_current_user_text_unique_rate - expected_rate) > 1e-12:
            raise ValueError("PM-v2 normalized text unique-rate accounting mismatch")
        if self.normalized_current_user_text_unique_rate != 1.0:
            raise ValueError("PM-v2 normalized current-user-text unique rate must be 1.0")
        return self
