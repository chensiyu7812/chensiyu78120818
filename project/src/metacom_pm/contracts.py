from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import re

OPAQUE_ID_RE = re.compile(r"^(?:mem|strat|card|state|pair)_[0-9a-f]{12,64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=False)


class MemorySource(str, Enum):
    MP = "MP"
    MS = "MS"
    ME = "ME"


class StrategyMode(str, Enum):
    R0 = "R0"
    RS = "RS"


ACTION_MEMORY_MAP: dict[str, frozenset[MemorySource]] = {
    "M0": frozenset(),
    "MP": frozenset({MemorySource.MP}),
    "MS": frozenset({MemorySource.MS}),
    "ME": frozenset({MemorySource.ME}),
    "MPMS": frozenset({MemorySource.MP, MemorySource.MS}),
    "MPE": frozenset({MemorySource.MP, MemorySource.ME}),
    "MSE": frozenset({MemorySource.MS, MemorySource.ME}),
    "MPMSME": frozenset({MemorySource.MP, MemorySource.MS, MemorySource.ME}),
}
MEMORY_ACTION_BY_SET = {v: k for k, v in ACTION_MEMORY_MAP.items()}
ALL_ACTION_IDS = tuple(
    f"{mem}+{strategy.value}"
    for mem in ACTION_MEMORY_MAP
    for strategy in StrategyMode
)


def parse_action_id(action_id: str) -> tuple[frozenset[MemorySource], StrategyMode]:
    parts = action_id.split("+")
    if len(parts) != 2 or parts[0] not in ACTION_MEMORY_MAP:
        raise ValueError(f"Invalid action_id: {action_id!r}")
    try:
        strategy = StrategyMode(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid strategy in action_id: {action_id!r}") from exc
    return ACTION_MEMORY_MAP[parts[0]], strategy


def canonical_action_id(
    sources: frozenset[MemorySource] | set[MemorySource],
    strategy: StrategyMode,
) -> str:
    frozen = frozenset(sources)
    try:
        mem = MEMORY_ACTION_BY_SET[frozen]
    except KeyError as exc:
        raise ValueError(f"Unsupported memory source set: {sorted(s.value for s in frozen)}") from exc
    return f"{mem}+{strategy.value}"


class DialogueTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class SourceCatalog(StrictModel):
    available: bool
    count: int = Field(ge=0)
    min_age_sessions: int | None = Field(default=None, ge=0)
    max_age_sessions: int | None = Field(default=None, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    catalog_fingerprint: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        if self.available != (self.count > 0):
            raise ValueError("available must equal count > 0")
        if not self.available and any(
            x is not None for x in (self.min_age_sessions, self.max_age_sessions)
        ):
            raise ValueError("unavailable source cannot have ages")
        if (
            self.min_age_sessions is not None
            and self.max_age_sessions is not None
            and self.min_age_sessions > self.max_age_sessions
        ):
            raise ValueError("min_age_sessions exceeds max_age_sessions")
        return self


class RuntimeState(StrictModel):
    @field_validator("inventory", mode="before")
    @classmethod
    def parse_inventory_keys(cls, value):
        if not isinstance(value, dict):
            raise TypeError("inventory must be an object")
        out = {}
        for key, item in value.items():
            source = key if isinstance(key, MemorySource) else MemorySource(str(key))
            out[source] = item
        return out

    state_id: str
    card_id: str
    user_id: str
    split: Literal["train", "validation", "development", "esconv_test", "evoemo_test"]
    semantic_family: str
    current_user_text: str = Field(min_length=1)
    current_session_history: list[DialogueTurn]
    current_session_summary: str
    session_index: int = Field(ge=1)
    inventory: dict[MemorySource, SourceCatalog]
    allowed_actions: list[str]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("allowed_actions")
    @classmethod
    def valid_actions(cls, values: list[str]) -> list[str]:
        if not values or len(values) != len(set(values)):
            raise ValueError("allowed_actions must be non-empty and unique")
        for value in values:
            parse_action_id(value)
        return values

    @model_validator(mode="after")
    def action_mask_matches_inventory(self):
        available = {src for src, cat in self.inventory.items() if cat.available}
        expected = {
            canonical_action_id(sources, strategy)
            for sources in ACTION_MEMORY_MAP.values()
            if sources <= available
            for strategy in StrategyMode
        }
        if set(self.allowed_actions) != expected:
            missing = sorted(expected - set(self.allowed_actions))
            extra = sorted(set(self.allowed_actions) - expected)
            raise ValueError(f"action mask mismatch: missing={missing}, extra={extra}")
        return self


class MemoryItem(StrictModel):
    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, value):
        return value if isinstance(value, MemorySource) else MemorySource(str(value))

    memory_id: str
    source: MemorySource
    created_session: int = Field(ge=0)
    timestamp: str | None = None
    text: str = Field(min_length=1)

    @field_validator("memory_id")
    @classmethod
    def opaque_id(cls, value: str) -> str:
        if not OPAQUE_ID_RE.match(value):
            raise ValueError("memory_id must be opaque: mem_<hex>")
        return value


class MemoryBackendRecord(StrictModel):
    card_id: str
    items: list[MemoryItem]

    @model_validator(mode="after")
    def unique_and_causal(self):
        ids = [x.memory_id for x in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate memory IDs")
        return self


class StrategyCard(StrictModel):
    strategy_id: str
    strategy_label: str
    retrieval_text: str = Field(min_length=1)
    guidance_text: str = Field(min_length=1)
    example_response: str = Field(min_length=1)
    source_dialogue_id: str
    source_turn_index: int = Field(ge=0)

    @field_validator("strategy_id")
    @classmethod
    def opaque_id(cls, value: str) -> str:
        if not OPAQUE_ID_RE.match(value):
            raise ValueError("strategy_id must be opaque: strat_<hex>")
        return value


class SelectedEvidence(StrictModel):
    memory_items: list[MemoryItem]
    strategy_cards: list[StrategyCard]


class EvidenceItemDecision(StrictModel):
    evidence_type: Literal["memory", "strategy"]
    item_id: str
    source: str
    current_turn_score: float = Field(ge=0.0, le=1.0)
    context_score: float = Field(ge=0.0, le=1.0)
    selector_score: float = Field(ge=0.0, le=1.0)
    helpfulness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    estimated_tokens: int = Field(ge=0)
    keep: bool
    reason: str = Field(min_length=1)


class EvidenceFilterDecision(StrictModel):
    protocol: str = Field(min_length=1)
    enabled: bool
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    memory_filter_mode: Literal["lexical_bootstrap", "supervised_helpfulness"]
    strategy_filter_mode: Literal["lexical_contextual_fail_safe"]
    memory_filter_checkpoint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    requested_action_id: str
    effective_action_id: str
    candidate_memory_ids: list[str]
    kept_memory_ids: list[str]
    dropped_memory_ids: list[str]
    candidate_strategy_ids: list[str]
    kept_strategy_ids: list[str]
    dropped_strategy_ids: list[str]
    candidate_memory_tokens: int = Field(ge=0)
    kept_memory_tokens: int = Field(ge=0)
    candidate_strategy_tokens: int = Field(ge=0)
    kept_strategy_tokens: int = Field(ge=0)
    item_decisions: list[EvidenceItemDecision]

    @model_validator(mode="after")
    def coherent(self):
        parse_action_id(self.requested_action_id)
        parse_action_id(self.effective_action_id)
        if (
            self.memory_filter_mode == "supervised_helpfulness"
            and self.memory_filter_checkpoint_sha256 is None
        ):
            raise ValueError("supervised memory filter requires checkpoint SHA-256")
        for candidate, kept, dropped, label in (
            (
                self.candidate_memory_ids,
                self.kept_memory_ids,
                self.dropped_memory_ids,
                "memory",
            ),
            (
                self.candidate_strategy_ids,
                self.kept_strategy_ids,
                self.dropped_strategy_ids,
                "strategy",
            ),
        ):
            if len(candidate) != len(set(candidate)):
                raise ValueError(f"duplicate candidate {label} IDs")
            if len(kept) != len(set(kept)) or len(dropped) != len(set(dropped)):
                raise ValueError(f"duplicate kept/dropped {label} IDs")
            if set(kept) & set(dropped) or set(kept) | set(dropped) != set(candidate):
                raise ValueError(f"kept/dropped {label} IDs must partition candidates")
        if self.kept_memory_tokens > self.candidate_memory_tokens:
            raise ValueError("kept memory tokens exceed candidate tokens")
        if self.kept_strategy_tokens > self.candidate_strategy_tokens:
            raise ValueError("kept strategy tokens exceed candidate tokens")
        expected_item_ids = self.candidate_memory_ids + self.candidate_strategy_ids
        actual_item_ids = [row.item_id for row in self.item_decisions]
        if actual_item_ids != expected_item_ids:
            raise ValueError("item decisions must follow the candidate evidence order")
        return self


class CostRecord(StrictModel):
    pm_input_tokens_est: int = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    reranker_calls: int = Field(ge=0)
    memory_tokens: int = Field(ge=0)
    strategy_tokens: int = Field(ge=0)
    base_prompt_tokens: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    api_cost_usd: float | None = Field(default=None, ge=0)
    # Latency breakdowns (optional; populated in EvoEmo runs)
    catalog_reads: int = Field(default=0, ge=0)
    pre_evidence_compute_ms: float = Field(default=0.0, ge=0)
    pm_inference_ms: float = Field(default=0.0, ge=0)
    retrieval_latency_ms: float = Field(default=0.0, ge=0)
    generation_latency_ms: float = Field(default=0.0, ge=0)
    evidence_filter_calls: int = Field(default=0, ge=0)
    evidence_filter_latency_ms: float = Field(default=0.0, ge=0)
    candidate_memory_count: int = Field(default=0, ge=0)
    kept_memory_count: int = Field(default=0, ge=0)
    candidate_strategy_count: int = Field(default=0, ge=0)
    kept_strategy_count: int = Field(default=0, ge=0)
    candidate_memory_tokens: int = Field(default=0, ge=0)
    candidate_strategy_tokens: int = Field(default=0, ge=0)
    dropped_memory_tokens: int = Field(default=0, ge=0)
    dropped_strategy_tokens: int = Field(default=0, ge=0)


class ActionOutcome(StrictModel):
    card_id: str
    state_id: str
    user_id: str
    action_id: str
    response: str = Field(min_length=1)
    selected_memory_ids: list[str]
    selected_strategy_ids: list[str]
    memory_view: list[MemoryItem]
    strategy_view: list[StrategyCard]
    effective_action_id: str | None = None
    candidate_memory_view: list[MemoryItem] | None = None
    candidate_strategy_view: list[StrategyCard] | None = None
    evidence_filter_decision: EvidenceFilterDecision | None = None
    cost: CostRecord
    model_name: str
    prompt_hash: str
    request_hash: str
    provenance: dict[str, Any]

    @model_validator(mode="after")
    def selection_matches_action(self):
        sources, strategy = parse_action_id(self.action_id)
        actual_sources = {item.source for item in self.memory_view}
        if not actual_sources <= sources:
            raise ValueError("memory_view includes a source outside the action")
        if strategy is StrategyMode.R0 and self.strategy_view:
            raise ValueError("R0 outcome cannot include strategy cards")
        if (
            strategy is StrategyMode.RS
            and not self.strategy_view
            and self.evidence_filter_decision is None
        ):
            raise ValueError("RS outcome must include strategy cards")
        if self.selected_memory_ids != [x.memory_id for x in self.memory_view]:
            raise ValueError("selected_memory_ids mismatch")
        if self.selected_strategy_ids != [x.strategy_id for x in self.strategy_view]:
            raise ValueError("selected_strategy_ids mismatch")
        if self.evidence_filter_decision is not None:
            if self.candidate_memory_view is None or self.candidate_strategy_view is None:
                raise ValueError("filtered outcomes require candidate evidence views")
            decision = self.evidence_filter_decision
            if decision.requested_action_id != self.action_id:
                raise ValueError("filter requested action does not match action_id")
            if decision.candidate_memory_ids != [
                x.memory_id for x in self.candidate_memory_view
            ]:
                raise ValueError("candidate memory IDs mismatch")
            if decision.candidate_strategy_ids != [
                x.strategy_id for x in self.candidate_strategy_view
            ]:
                raise ValueError("candidate strategy IDs mismatch")
            if decision.kept_memory_ids != self.selected_memory_ids:
                raise ValueError("filter kept memory IDs mismatch")
            if decision.kept_strategy_ids != self.selected_strategy_ids:
                raise ValueError("filter kept strategy IDs mismatch")
            candidate_sources = {item.source for item in self.candidate_memory_view}
            if not candidate_sources <= sources:
                raise ValueError("candidate memory includes a source outside the action")
            if strategy is StrategyMode.R0 and self.candidate_strategy_view:
                raise ValueError("R0 outcome cannot retrieve strategy candidates")
            computed_effective = canonical_action_id(
                actual_sources,
                StrategyMode.RS if self.strategy_view else StrategyMode.R0,
            )
            if decision.effective_action_id != computed_effective:
                raise ValueError("filter effective action does not match kept evidence")
            if self.effective_action_id != computed_effective:
                raise ValueError("effective_action_id does not match kept evidence")
        elif self.effective_action_id is not None and self.effective_action_id != self.action_id:
            raise ValueError("unfiltered effective_action_id must equal action_id")
        return self


class PairRecord(StrictModel):
    pair_id: str
    canonical_pair_id: str
    card_id: str
    action_a: str
    action_b: str
    pair_type: Literal["strategy", "memory", "combination", "chord"]
    is_reversal: bool = False
    is_same_order_repeat: bool = False
    training_eligible: bool = True

    @model_validator(mode="after")
    def distinct_actions(self):
        if self.action_a == self.action_b:
            raise ValueError("pair actions must differ")
        parse_action_id(self.action_a)
        parse_action_id(self.action_b)
        if (self.is_reversal or self.is_same_order_repeat) and self.training_eligible:
            raise ValueError("audit repeats must not be training eligible")
        return self


class ResponsePairJudgment(StrictModel):
    preference: Literal["A", "B", "tie"]
    empathy: Literal["A", "B", "tie"]
    contextual_fit: Literal["A", "B", "tie"]
    guidance_fit: Literal["A", "B", "tie"]
    non_intrusiveness: Literal["A", "B", "tie"]
    coherence: Literal["A", "B", "tie"]
    reason: str = Field(min_length=1, max_length=500)


class MemoryOpportunityItem(StrictModel):
    memory_id: str
    current_relevance: int = Field(ge=0, le=2)
    potential_helpfulness: int = Field(ge=0, le=2)
    stale: bool
    conflicts_with_newer_information: bool
    intrusive_if_mentioned: bool


class MemoryOpportunityJudgment(StrictModel):
    items: list[MemoryOpportunityItem]
    reason: str = Field(min_length=1, max_length=500)


class SourceUseAssessment(StrictModel):
    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, value):
        return value if isinstance(value, MemorySource) else MemorySource(str(value))

    source: MemorySource
    utilization: int = Field(ge=0, le=2)
    unused_retrieval: int = Field(ge=0, le=2)
    unnecessary_exposure: int = Field(ge=0, le=2)
    stale_or_conflicting_use: int = Field(ge=0, le=2)
    unsupported_personal_claim: int = Field(ge=0, le=2)


class MemoryUseJudgment(StrictModel):
    source_assessments: list[SourceUseAssessment]
    overall_source_set_appropriateness: int = Field(ge=0, le=2)
    reason: str = Field(min_length=1, max_length=500)


class MemorySelectedSetOmissionJudgment(StrictModel):
    """Audit whether a non-M0 selected memory source set omitted needed sources."""

    selected_set_sufficiency: int = Field(ge=0, le=2)
    selected_set_omission_severity: int = Field(ge=0, le=2)
    missed_useful_sources: list[MemorySource] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("missed_useful_sources", mode="before")
    @classmethod
    def parse_missed_sources(cls, value):
        if value is None:
            return []
        return [x if isinstance(x, MemorySource) else MemorySource(str(x)) for x in value]


class MemoryOmissionJudgment(StrictModel):
    omission_appropriateness: int = Field(ge=0, le=2)
    missed_memory_opportunity_severity: int = Field(ge=0, le=2)
    unsupported_personal_claim: int = Field(ge=0, le=2)
    reason: str = Field(min_length=1, max_length=500)


class StrategyUseJudgment(StrictModel):
    strategy_relevance: int = Field(ge=0, le=2)
    strategy_utilization: int = Field(ge=0, le=2)
    over_structuring: int = Field(ge=0, le=2)
    premature_advice: int = Field(ge=0, le=2)
    reason: str = Field(min_length=1, max_length=500)


class StrategyOmissionJudgment(StrictModel):
    strategy_omission_appropriateness: int = Field(ge=0, le=2)
    missed_strategy_opportunity_severity: int = Field(ge=0, le=2)
    premature_or_overstructured_without_strategy: int = Field(ge=0, le=2)
    reason: str = Field(min_length=1, max_length=500)


class OfficialDialogueScores(StrictModel):
    memory: int = Field(ge=1, le=5)
    personalization: int = Field(ge=1, le=5)
    emotional_support: int = Field(ge=1, le=5)
    factual_grounding: int = Field(ge=1, le=5)
    temporal_consistency: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=1000)


class ObservationRelevance(StrictModel):
    relevance: Literal[0.0, 0.5, 1.0]


class ObservationUsage(StrictModel):
    used: bool
    already_disclosed_in_current_session: bool
    attributable_to_long_term_memory: bool


class DialoguePairJudgment(StrictModel):
    preference: Literal["A", "B", "tie"]
    emotional_support: Literal["A", "B", "tie"]
    personalization: Literal["A", "B", "tie"]
    memory_appropriateness: Literal["A", "B", "tie"]
    non_intrusiveness: Literal["A", "B", "tie"]
    coherence: Literal["A", "B", "tie"]
    reason: str = Field(min_length=1, max_length=800)
