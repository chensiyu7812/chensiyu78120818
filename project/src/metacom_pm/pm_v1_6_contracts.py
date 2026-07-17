from __future__ import annotations

from enum import Enum
import math
import re
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import MemorySource, StrategyMode, canonical_action_id, parse_action_id
from .io import canonical_json, sha256_text

PROTOCOL_VERSION = "pm-v1.6"
STEP0_PROTOCOL = "pm-v1.6-step0-source-observation-v1"
ACTION_LINEAGE_PROTOCOL = "pm-v1.6-requested-attempted-realized-v1"
COST_VECTOR_PROTOCOL = "pm-v1.6-complete-cost-vector-v1"
ALGORITHM_SELECTION_PROTOCOL = "pm-v1.6-train-only-algorithm-selection-v1"
INTERNAL_LEDGER_PROTOCOL = "pm-v1.6-internal-test-consumption-ledger-v1"
CLAIM_PROTOCOL = "pm-v1.6-hierarchical-claim-assessment-v1"

STRATEGY_FAMILIES: tuple[str, ...] = (
    "exploration_question",
    "restatement_paraphrase",
    "reflection",
    "self_disclosure",
    "affirmation_reassurance",
    "suggestion_action",
    "information",
    "non_directive_listening",
    "other",
)

PM_VISIBLE_STEP0_FIELDS = frozenset(
    {
        "protocol",
        "state_id",
        "sources",
        "strategy_families",
        "readiness",
        "step0_cost",
    }
)
FORBIDDEN_PM_VISIBLE_FIELDS = frozenset(
    {
        "raw_text",
        "text",
        "snippet",
        "memory_id",
        "strategy_id",
        "item_id",
        "retrieval_score",
        "top_k_score",
        "max_score",
        "p90_score",
        "catalog_embedding",
        "embedding",
        "embedding_norm",
        "embedding_mean",
        "embedding_std",
        "regime",
        "oracle",
        "helpful",
        "harmful",
        "stale",
        "conflict",
        "representation_sha256",
        "catalog_build_sha256",
        "algorithm_id",
        "encoder_id",
        "representation_version",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceProbeObservation(StrictModel):
    source: MemorySource
    available: bool
    bounded_count: int = Field(ge=0)
    min_age_sessions: int | None = Field(default=None, ge=0)
    median_age_sessions: float | None = Field(default=None, ge=0)
    max_age_sessions: int | None = Field(default=None, ge=0)
    estimated_retrievable_tokens: int = Field(ge=0)
    query_to_source_similarity: float = Field(ge=-1.0, le=1.0)
    representation_valid: bool

    @model_validator(mode="after")
    def coherent(self) -> "SourceProbeObservation":
        if self.available != (self.bounded_count > 0):
            raise ValueError("available must equal bounded_count > 0")
        if not self.available:
            if self.estimated_retrievable_tokens != 0:
                raise ValueError("unavailable source must have zero retrievable tokens")
            if self.representation_valid:
                raise ValueError("unavailable source cannot have a valid representation")
            if abs(self.query_to_source_similarity) > 1e-12:
                raise ValueError("unavailable source similarity must be zero")
            if any(
                value is not None
                for value in (
                    self.min_age_sessions,
                    self.median_age_sessions,
                    self.max_age_sessions,
                )
            ):
                raise ValueError("unavailable source cannot expose ages")
        if (
            self.min_age_sessions is not None
            and self.max_age_sessions is not None
            and self.min_age_sessions > self.max_age_sessions
        ):
            raise ValueError("min age exceeds max age")
        return self


class StrategyFamilyObservation(StrictModel):
    family: str
    query_to_family_similarity: float = Field(ge=-1.0, le=1.0)
    representation_valid: bool

    @field_validator("family")
    @classmethod
    def known_family(cls, value: str) -> str:
        if value not in STRATEGY_FAMILIES:
            raise ValueError(f"unknown strategy family: {value}")
        return value


class StrategyReadiness(StrictModel):
    advice_requested: bool
    advice_rejected: bool
    listening_requested: bool
    clarification_needed: bool
    action_readiness: float = Field(ge=0.0, le=1.0)


class Step0Cost(StrictModel):
    query_encoding_latency_ms: float = Field(ge=0.0)
    memory_source_comparisons: int = Field(ge=0)
    strategy_family_comparisons: int = Field(ge=0)
    catalog_build_amortized_ms: float = Field(ge=0.0)


class Step0Observation(StrictModel):
    protocol: Literal["pm-v1.6-step0-source-observation-v1"] = STEP0_PROTOCOL
    state_id: str = Field(min_length=1)
    sources: dict[MemorySource, SourceProbeObservation]
    strategy_families: list[StrategyFamilyObservation]
    readiness: StrategyReadiness
    step0_cost: Step0Cost

    @field_validator("sources", mode="before")
    @classmethod
    def parse_sources(cls, value: Any) -> dict[MemorySource, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("sources must be an object")
        return {
            key if isinstance(key, MemorySource) else MemorySource(str(key)): item
            for key, item in value.items()
        }

    @model_validator(mode="after")
    def complete(self) -> "Step0Observation":
        if set(self.sources) != set(MemorySource):
            raise ValueError("Step-0 must cover MP, MS, and ME exactly")
        if [row.family for row in self.strategy_families] != list(STRATEGY_FAMILIES):
            raise ValueError("strategy-family observations must use frozen order")
        assert_pm_visible_step0_payload(self.model_dump(mode="json"))
        return self


class Step0AuditBinding(StrictModel):
    protocol: Literal["pm-v1.6-step0-audit-binding-v1"] = (
        "pm-v1.6-step0-audit-binding-v1"
    )
    state_id: str
    algorithm_id: str
    encoder_id: str
    representation_dimension: int = Field(gt=0)
    source_representation_sha256: dict[MemorySource, str]
    strategy_family_representation_sha256: dict[str, str]
    catalog_build_sha256: str
    refresh_policy: str

    @field_validator("source_representation_sha256", mode="before")
    @classmethod
    def parse_source_hashes(cls, value: Any) -> dict[MemorySource, str]:
        if not isinstance(value, Mapping):
            raise TypeError("source representation hashes must be an object")
        return {
            key if isinstance(key, MemorySource) else MemorySource(str(key)): str(item)
            for key, item in value.items()
        }

    @model_validator(mode="after")
    def hashes(self) -> "Step0AuditBinding":
        values = [
            *self.source_representation_sha256.values(),
            *self.strategy_family_representation_sha256.values(),
            self.catalog_build_sha256,
        ]
        if any(not _HEX64.fullmatch(value) for value in values):
            raise ValueError("all Step-0 audit bindings must be SHA-256 digests")
        return self


class RetrievalAttempt(StrictModel):
    source: Literal["MP", "MS", "ME", "RS"]
    call_count: int = Field(ge=0)
    hit_count: int = Field(ge=0)
    retrieved_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def coherent(self) -> "RetrievalAttempt":
        if self.hit_count > 0 and self.call_count == 0:
            raise ValueError("hits require at least one call")
        if self.hit_count == 0 and self.retrieved_tokens != 0:
            raise ValueError("zero-hit retrieval must have zero retrieved tokens")
        return self


class ActionLineage(StrictModel):
    protocol: Literal["pm-v1.6-requested-attempted-realized-v1"] = (
        ACTION_LINEAGE_PROTOCOL
    )
    requested_action_id: str
    retrieval_attempts: list[RetrievalAttempt]
    realized_action_id: str
    prompt_equivalence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_equivalence_class_size: int = Field(ge=1)

    @field_validator("requested_action_id", "realized_action_id")
    @classmethod
    def valid_action(cls, value: str) -> str:
        parse_action_id(value)
        return value

    @model_validator(mode="after")
    def complete_attempts(self) -> "ActionLineage":
        sources, strategy = parse_action_id(self.requested_action_id)
        required = {source.value for source in sources}
        if strategy is StrategyMode.RS:
            required.add("RS")
        observed = {attempt.source for attempt in self.retrieval_attempts}
        if observed != required:
            raise ValueError(
                f"retrieval attempts differ from requested action: "
                f"missing={sorted(required-observed)}, extra={sorted(observed-required)}"
            )
        return self


class CompleteCostVector(StrictModel):
    protocol: Literal["pm-v1.6-complete-cost-vector-v1"] = COST_VECTOR_PROTOCOL
    step0_latency_ms: float = Field(ge=0.0)
    step0_comparisons: int = Field(ge=0)
    catalog_build_amortized_ms: float = Field(ge=0.0)
    router_inference_latency_ms: float = Field(ge=0.0)
    item_retrieval_calls: int = Field(ge=0)
    item_retrieval_hits: int = Field(ge=0)
    item_retrieval_tokens: int = Field(ge=0)
    item_retrieval_latency_ms: float = Field(ge=0.0)
    generator_input_tokens: int = Field(ge=0)
    generator_output_tokens: int = Field(ge=0)
    generator_api_cost_usd: float = Field(ge=0.0)
    generator_latency_ms: float = Field(ge=0.0)
    total_variable_cost_usd: float = Field(ge=0.0)
    end_to_end_latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def lower_bounds(self) -> "CompleteCostVector":
        component_latency = (
            self.step0_latency_ms
            + self.catalog_build_amortized_ms
            + self.router_inference_latency_ms
            + self.item_retrieval_latency_ms
            + self.generator_latency_ms
        )
        if self.end_to_end_latency_ms + 1e-9 < component_latency:
            raise ValueError("end-to-end latency is below the sum of recorded components")
        if self.item_retrieval_hits > 0 and self.item_retrieval_calls == 0:
            raise ValueError("retrieval hits require retrieval calls")
        if self.total_variable_cost_usd + 1e-12 < self.generator_api_cost_usd:
            raise ValueError("total variable cost cannot be below generator API cost")
        return self


class AlgorithmCandidate(str, Enum):
    ABSOLUTE_HGB = "absolute_outcome_hgb"
    STATE_CENTERED_HGB = "state_centered_factorized_hgb"
    RULE_RELATIVE_HGB = "rule_relative_residual_hgb"


class CandidateFamilyManifest(StrictModel):
    protocol: Literal["pm-v1.6-train-only-algorithm-selection-v1"] = (
        ALGORITHM_SELECTION_PROTOCOL
    )
    candidate_algorithms: list[AlgorithmCandidate]
    selected_primary: AlgorithmCandidate
    selection_metric: str
    one_standard_error_rule: bool
    complexity_order: list[AlgorithmCandidate]
    train_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    internal_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step0_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strong_router_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def selected_from_frozen_family(self) -> "CandidateFamilyManifest":
        if not self.candidate_algorithms:
            raise ValueError("candidate family cannot be empty")
        if self.selected_primary not in self.candidate_algorithms:
            raise ValueError("selected primary is outside candidate family")
        if set(self.complexity_order) != set(self.candidate_algorithms):
            raise ValueError("complexity order must cover candidate family exactly")
        return self


def assert_pm_visible_step0_payload(payload: Mapping[str, Any]) -> None:
    """Fail if audit-only or item-level information appears in the PM payload."""

    if set(payload) != PM_VISIBLE_STEP0_FIELDS:
        raise ValueError(
            "PM-visible Step-0 keys differ from the frozen contract: "
            f"missing={sorted(PM_VISIBLE_STEP0_FIELDS-set(payload))}, "
            f"extra={sorted(set(payload)-PM_VISIBLE_STEP0_FIELDS)}"
        )

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                normalized = key_text.lower()
                if normalized in FORBIDDEN_PM_VISIBLE_FIELDS:
                    raise ValueError(f"forbidden PM-visible Step-0 field at {path}.{key_text}")
                walk(child, f"{path}.{key_text}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite PM-visible value at {path}")

    walk(payload, "step0")


def realized_action_from_evidence(
    memory_sources: Sequence[MemorySource | str],
    *,
    strategy_card_count: int,
) -> str:
    parsed = {
        source if isinstance(source, MemorySource) else MemorySource(str(source))
        for source in memory_sources
    }
    strategy = StrategyMode.RS if int(strategy_card_count) > 0 else StrategyMode.R0
    return canonical_action_id(parsed, strategy)


def prompt_equivalence_id(messages: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json(list(messages)))
