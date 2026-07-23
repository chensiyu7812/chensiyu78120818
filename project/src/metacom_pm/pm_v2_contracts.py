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
from .io import canonical_json, sha256_text


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
    representation_valid: bool = False
    # Transient construction input for the runtime catalog only. It is excluded
    # from the serialized PM state and must never be consumed by model features.
    catalog_embedding: list[float] = Field(default_factory=list, exclude=True)

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
            if self.representation_valid:
                raise ValueError("unavailable source cannot have a valid representation")
        if (
            self.min_age_sessions is not None
            and self.max_age_sessions is not None
            and self.min_age_sessions > self.max_age_sessions
        ):
            raise ValueError("min_age_sessions exceeds max_age_sessions")
        return self


STRATEGY_FAMILY_IDS: tuple[str, ...] = (
    "question",
    "other",
    "suggestion",
    "affirmation_reassurance",
    "self_disclosure",
    "reflection",
    "information",
    "restatement",
)

ADVICE_READINESS_IDS: tuple[str, ...] = (
    "listen_only",
    "explore_first",
    "light_suggestion",
    "structured_plan",
    "ambiguous",
)


class Step0MemoryObservation(StrictModel):
    available: bool
    count: int = Field(ge=0)
    min_age_sessions: int | None = Field(default=None, ge=0)
    median_age_sessions: float | None = Field(default=None, ge=0)
    max_age_sessions: int | None = Field(default=None, ge=0)
    expected_retrieval_tokens: int = Field(ge=0)
    representation_valid: bool
    query_to_source_similarity: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def coherent(self):
        if self.available != (self.count > 0):
            raise ValueError("Step-0 source availability must equal count > 0")
        if not self.available and (
            self.expected_retrieval_tokens != 0
            or self.representation_valid
            or self.query_to_source_similarity != 0.0
        ):
            raise ValueError("unavailable Step-0 source must have zero/invalid signals")
        return self


class Step0StrategyObservation(StrictModel):
    available: bool
    count: int = Field(ge=0)
    expected_retrieval_tokens: int = Field(ge=0)
    representation_valid: bool
    family_similarities: dict[str, float]
    advice_readiness_similarities: dict[str, float]
    question_present: bool

    @field_validator("family_similarities")
    @classmethod
    def exact_families(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != set(STRATEGY_FAMILY_IDS):
            raise ValueError("Step-0 strategy family keys do not match the contract")
        if any(not -1.0 <= float(score) <= 1.0 for score in value.values()):
            raise ValueError("Step-0 strategy family similarity is outside [-1, 1]")
        return value

    @field_validator("advice_readiness_similarities")
    @classmethod
    def exact_readiness(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != set(ADVICE_READINESS_IDS):
            raise ValueError("Step-0 advice-readiness keys do not match the contract")
        if any(not -1.0 <= float(score) <= 1.0 for score in value.values()):
            raise ValueError("Step-0 advice-readiness similarity is outside [-1, 1]")
        return value

    @model_validator(mode="after")
    def coherent(self):
        if self.available != (self.count > 0):
            raise ValueError("Step-0 strategy availability must equal count > 0")
        if not self.available and (
            self.expected_retrieval_tokens != 0 or self.representation_valid
        ):
            raise ValueError("unavailable strategy catalog must have zero/invalid signals")
        if not self.representation_valid and any(
            float(value) != 0.0 for value in self.family_similarities.values()
        ):
            raise ValueError("invalid strategy representation must expose zero similarities")
        return self


class Step0Observation(StrictModel):
    protocol: Literal[
        "pm-v1.5-step0-semantic-source-observation-v3-unified-state-query"
    ] = (
        "pm-v1.5-step0-semantic-source-observation-v3-unified-state-query"
    )
    observation_stage: Literal["pre_item_retrieval"] = "pre_item_retrieval"
    memory_sources: dict[MemorySource, Step0MemoryObservation]
    strategy: Step0StrategyObservation

    @field_validator("memory_sources", mode="before")
    @classmethod
    def parse_memory_keys(cls, value):
        if not isinstance(value, dict):
            raise TypeError("Step-0 memory_sources must be an object")
        return {
            key if isinstance(key, MemorySource) else MemorySource(str(key)): item
            for key, item in value.items()
        }

    @model_validator(mode="after")
    def complete(self):
        if set(self.memory_sources) != set(MemorySource):
            raise ValueError("Step-0 observation must cover MP, MS, and ME")
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
    step0_observation: Step0Observation | None = None
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
        if self.step0_observation is not None:
            for source, catalog in self.inventory.items():
                observed = self.step0_observation.memory_sources[source]
                if (
                    observed.available != catalog.available
                    or observed.count != catalog.count
                    or observed.min_age_sessions != catalog.min_age_sessions
                    or observed.median_age_sessions != catalog.median_age_sessions
                    or observed.max_age_sessions != catalog.max_age_sessions
                    or observed.representation_valid != catalog.representation_valid
                    or observed.query_to_source_similarity
                    != catalog.query_similarity_mean
                ):
                    raise ValueError(
                        f"Step-0 memory observation drifts from {source.value} inventory"
                    )
            if (
                self.step0_observation.strategy.count
                != self.strategy_catalog_count
                or self.step0_observation.strategy.expected_retrieval_tokens
                != self.strategy_estimated_tokens
            ):
                raise ValueError("Step-0 strategy observation drifts from state metadata")
        allowed_provenance = {
            "backend_record_id",
            "evaluator_context_id",
            "data_generation_sha256",
            "adapted_from_runtime_state",
            "runtime_provenance_sha256",
            "semantic_observation",
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
        if "semantic_observation" in self.provenance:
            semantic = self.provenance["semantic_observation"]
            expected_semantic_keys = {
                "protocol",
                "semantic_query_protocol",
                "encoder_spec_sha256",
                "encoder_snapshot_tree_sha256",
                "views",
                "per_view_dimension",
                "combined_dimension",
                "tokenization",
                "step0_semantic_query_sha256",
                "state_embedding_query_sha256",
                "step0_semantic_query_vector_sha256",
                "state_embedding_query_vector_sha256",
                "visible_input_sha256",
                "observation_sha256",
            }
            if not isinstance(semantic, dict) or set(semantic) != expected_semantic_keys:
                raise ValueError("semantic observation audit does not match the contract")
            if semantic.get("combined_dimension") != len(self.text_embedding):
                raise ValueError("semantic observation dimension drifts from text_embedding")
            for key in (
                "encoder_spec_sha256",
                "encoder_snapshot_tree_sha256",
                "step0_semantic_query_sha256",
                "state_embedding_query_sha256",
                "step0_semantic_query_vector_sha256",
                "state_embedding_query_vector_sha256",
                "visible_input_sha256",
                "observation_sha256",
            ):
                value = semantic.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"semantic observation {key} must be SHA-256")
            if semantic.get("semantic_query_protocol") != (
                "pm-v1.5-unified-step0-state-semantic-query-v1"
            ):
                raise ValueError("semantic observation query protocol is stale")
            if (
                semantic["step0_semantic_query_sha256"]
                != semantic["state_embedding_query_sha256"]
                or semantic["step0_semantic_query_vector_sha256"]
                != semantic["state_embedding_query_vector_sha256"]
            ):
                raise ValueError("Step-0 and state embedding semantic queries drift")
            observation_payload = {
                key: value
                for key, value in semantic.items()
                if key != "observation_sha256"
            }
            if sha256_text(canonical_json(observation_payload)) != semantic[
                "observation_sha256"
            ]:
                raise ValueError("semantic observation content hash is stale")
            dimension = int(semantic.get("per_view_dimension") or 0)
            if dimension <= 0 or len(self.text_embedding) != 2 * dimension:
                raise ValueError("semantic observation per-view dimensions are invalid")
            state_vector_sha256 = sha256_text(
                canonical_json(self.text_embedding[dimension:])
            )
            if state_vector_sha256 != semantic[
                "state_embedding_query_vector_sha256"
            ]:
                raise ValueError("state embedding vector hash drifts from its audit")
            tokenization = semantic.get("tokenization")
            base_tokenization_keys = {
                "protocol",
                "max_length",
                "truncation_side",
                "views",
            }
            if not isinstance(tokenization, dict) or frozenset(tokenization) not in {
                frozenset(base_tokenization_keys),
                frozenset(base_tokenization_keys | {"section_allocation"}),
            }:
                raise ValueError("semantic tokenization telemetry is malformed")
            telemetry_views = tokenization.get("views")
            if not isinstance(telemetry_views, dict):
                raise ValueError("semantic tokenization views must be a mapping")
            for view_name, row in telemetry_views.items():
                if view_name not in {"current_user_text", "visible_dialogue_state"}:
                    raise ValueError("semantic tokenization view is not authorized")
                if not isinstance(row, dict) or set(row) != {
                    "original_token_count",
                    "encoded_token_count",
                    "truncated",
                    "truncated_token_count",
                    "input_sha256",
                }:
                    raise ValueError("semantic tokenization row is malformed")
                if (
                    not isinstance(row["input_sha256"], str)
                    or len(row["input_sha256"]) != 64
                    or int(row["original_token_count"]) < int(row["encoded_token_count"])
                    or int(row["truncated_token_count"])
                    != int(row["original_token_count"])
                    - int(row["encoded_token_count"])
                    or bool(row["truncated"])
                    != (int(row["truncated_token_count"]) > 0)
                ):
                    raise ValueError("semantic tokenization row is inconsistent")
            visible_view = telemetry_views.get("visible_dialogue_state") or {}
            if visible_view and visible_view.get("input_sha256") != semantic[
                "state_embedding_query_sha256"
            ]:
                raise ValueError("semantic query hash drifts from tokenization audit")
            section_allocation = tokenization.get("section_allocation")
            if section_allocation is not None:
                expected_allocation_keys = {
                    "protocol",
                    "max_length",
                    "current_user_state_token_budget",
                    "session_summary_token_budget",
                    "history_retention_policy",
                    "implicit_full_state_truncation",
                    "final_visible_state_token_count",
                    "sections",
                }
                if (
                    not isinstance(section_allocation, dict)
                    or set(section_allocation) != expected_allocation_keys
                    or section_allocation.get("protocol")
                    != "pm-v1.5-visible-dialogue-state-v2-section-aware"
                    or section_allocation.get("history_retention_policy")
                    != "most-recent-token-suffix-preserve-chronology"
                    or section_allocation.get("implicit_full_state_truncation")
                    != "forbidden"
                    or int(section_allocation.get("max_length", -1))
                    != int(tokenization.get("max_length", -2))
                    or int(section_allocation.get("final_visible_state_token_count", -1))
                    != int(visible_view.get("original_token_count", -2))
                    or bool(visible_view.get("truncated"))
                ):
                    raise ValueError("semantic section allocation is malformed")
                sections = section_allocation.get("sections")
                if not isinstance(sections, dict) or set(sections) != {
                    "current_user",
                    "session_summary",
                    "recent_dialogue",
                }:
                    raise ValueError("semantic section allocation is incomplete")
                for section_name, row in sections.items():
                    if not isinstance(row, dict) or set(row) != {
                        "original_token_count",
                        "retained_token_count",
                        "dropped_token_count",
                        "input_sha256",
                    }:
                        raise ValueError("semantic section allocation row is malformed")
                    original = int(row["original_token_count"])
                    retained = int(row["retained_token_count"])
                    dropped = int(row["dropped_token_count"])
                    if (
                        min(original, retained, dropped) < 0
                        or original != retained + dropped
                        or not isinstance(row["input_sha256"], str)
                        or len(row["input_sha256"]) != 64
                    ):
                        raise ValueError("semantic section allocation row is inconsistent")
                    budget_key = {
                        "current_user": "current_user_state_token_budget",
                        "session_summary": "session_summary_token_budget",
                    }.get(section_name)
                    if budget_key and retained > int(section_allocation[budget_key]):
                        raise ValueError("semantic section allocation exceeds its budget")
        elif self.text_embedding:
            raise ValueError("text_embedding requires semantic observation provenance")
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
        expected_rate = self.unique_normalized_current_user_texts / self.total_states
        if abs(self.normalized_current_user_text_unique_rate - expected_rate) > 1e-12:
            raise ValueError("PM-v2 normalized text unique-rate accounting mismatch")
        return self
