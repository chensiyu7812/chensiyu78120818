from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

import numpy as np
from pydantic import Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from sklearn.feature_extraction.text import HashingVectorizer

from .api import CallResult, Endpoint, make_client
from .contracts import (
    ACTION_MEMORY_MAP,
    DialogueTurn,
    MemoryBackendRecord,
    MemoryItem,
    MemorySource,
    RuntimeState,
    SourceCatalog,
    StrategyMode,
    canonical_action_id,
)
from .io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)
from .pm_v2_contracts import (
    ObservableSourceSummary,
    PMV2Split,
    PMV2State,
    ResourceNeedRegime,
    SplitManifest,
    StrictModel,
)
from .text import estimate_tokens


SPACE_RE = re.compile(r"\s+")
CATALOG_HASH_FEATURES = 64
DATA_GENERATION_CONTRACT_VERSION = (
    "pm-v2-data-generation-v11-marginal-value-grounded-randomized-evidence"
)
# The 52 bundles already vary by seed dialogue, user, semantic family, and
# attempt seed.  A lower temperature prioritizes compliance with the dense
# cross-field data contract without removing those controlled diversity axes.
GENERATION_TEMPERATURE = 0.4
GENERATION_MAX_OUTPUT_TOKENS = 8000
EVALUATOR_CONTEXT_FIELDS = frozenset(
    {
        "evaluator_context_id",
        "state_id",
        "card_id",
        "regime",
        "needed_memory_sources",
        "authorized_user_context",
        "coverage_rationale",
        "memory_annotations",
        "context_payload_sha256",
    }
)
EVALUATOR_MEMORY_ANNOTATION_FIELDS = frozenset(
    {
        "memory_id",
        "source",
        "created_session",
        "stale",
        "conflicts_with_current_state",
        "private_sensitivity",
        "item_utility",
    }
)


def validate_regime_needed_sources(
    regime: ResourceNeedRegime | str,
    sources: Sequence[MemorySource | str],
) -> list[MemorySource]:
    """Validate the legacy serialized source target.

    V8 interprets this target as *materially useful memory sources*: sources
    expected to add material, nonredundant value over Context Only.  The legacy
    function/field name remains frozen for downstream artifact compatibility; it
    no longer means that every listed source is absolutely indispensable.
    """
    parsed_regime = (
        regime if isinstance(regime, ResourceNeedRegime) else ResourceNeedRegime(str(regime))
    )
    parsed = [
        source if isinstance(source, MemorySource) else MemorySource(str(source))
        for source in sources
    ]
    if len(parsed) != len(set(parsed)):
        raise ValueError("needed_memory_sources must be unique")
    exact_needs = {
        ResourceNeedRegime.CONTEXT_ONLY: set(),
        ResourceNeedRegime.PROFILE_NEEDED: {MemorySource.MP},
        ResourceNeedRegime.SUMMARY_NEEDED: {MemorySource.MS},
        ResourceNeedRegime.EVENT_NEEDED: {MemorySource.ME},
        ResourceNeedRegime.MEMORY_HARMFUL: set(),
        ResourceNeedRegime.STRATEGY_HELPFUL: set(),
        ResourceNeedRegime.STRATEGY_HARMFUL: set(),
    }
    parsed_set = set(parsed)
    if parsed_regime in exact_needs and parsed_set != exact_needs[parsed_regime]:
        raise ValueError(
            f"{parsed_regime.value} requires needed_memory_sources="
            f"{sorted(source.value for source in exact_needs[parsed_regime])}"
        )
    if parsed_regime is ResourceNeedRegime.MULTI_SOURCE_NEEDED and not (
        2 <= len(parsed_set) <= 3
    ):
        raise ValueError(
            "multi_source_needed requires two or three complementary sources"
        )
    if parsed_regime is ResourceNeedRegime.AMBIGUOUS and len(parsed_set) > 1:
        raise ValueError("ambiguous allows at most one needed memory source")
    return parsed


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip().lower())


def evaluator_context_payload_sha256(row: dict[str, Any]) -> str:
    payload = {
        key: row[key]
        for key in sorted(EVALUATOR_CONTEXT_FIELDS - {"context_payload_sha256"})
    }
    return sha256_text(canonical_json(payload))


@dataclass(frozen=True)
class EvaluatorContextIndex:
    """Strict evaluator-only rows indexed independently from PM-visible state."""

    by_state: dict[str, dict[str, Any]]
    by_card: dict[str, dict[str, Any]]
    source_sha256: str
    map_sha256: str

    def require_states(
        self,
        states: Sequence[PMV2State],
        *,
        exact: bool = True,
    ) -> dict[str, dict[str, Any]]:
        state_map = {state.state_id: state for state in states}
        if len(state_map) != len(states):
            raise RuntimeError("duplicate PM-v2 state_id while joining evaluator contexts")
        if exact and set(self.by_state) != set(state_map):
            missing = sorted(set(state_map) - set(self.by_state))
            extra = sorted(set(self.by_state) - set(state_map))
            raise RuntimeError(
                "evaluator-context/state map is not exact: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        selected: dict[str, dict[str, Any]] = {}
        for state_id, state in state_map.items():
            try:
                context = self.by_state[state_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"missing evaluator context for state {state_id}"
                ) from exc
            if context["card_id"] != state.card_id:
                raise RuntimeError(f"evaluator context card mismatch for {state_id}")
            expected_context_id = state.provenance.get("evaluator_context_id")
            if not isinstance(expected_context_id, str) or not expected_context_id:
                raise RuntimeError(
                    f"state {state_id} lacks its opaque evaluator_context_id"
                )
            if context["evaluator_context_id"] != expected_context_id:
                raise RuntimeError(f"evaluator context ID mismatch for {state_id}")
            future = [
                annotation["memory_id"]
                for annotation in context["memory_annotations"]
                if int(annotation["created_session"]) >= state.session_index
            ]
            if future:
                raise RuntimeError(
                    f"evaluator context for {state_id} contains non-prior memories: "
                    f"{future[:10]}"
                )
            needed = {
                MemorySource(source) for source in context["needed_memory_sources"]
            }
            unavailable_needed = sorted(
                source.value
                for source in needed
                if not state.inventory[source].available
            )
            if unavailable_needed:
                raise RuntimeError(
                    f"evaluator context for {state_id} requires unavailable sources: "
                    f"{unavailable_needed}"
                )
            selected[state_id] = context
        return selected


def build_evaluator_context_index(
    rows: Sequence[dict[str, Any]],
    *,
    states: Sequence[PMV2State] | None = None,
    require_exact: bool = True,
    source_sha256: str | None = None,
) -> EvaluatorContextIndex:
    by_state: dict[str, dict[str, Any]] = {}
    by_card: dict[str, dict[str, Any]] = {}
    seen_context_ids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(rows, 1):
        if not isinstance(raw, dict):
            raise TypeError(f"evaluator context row {row_index} is not an object")
        if set(raw) != EVALUATOR_CONTEXT_FIELDS:
            raise RuntimeError(
                f"evaluator context row {row_index} has schema mismatch: "
                f"missing={sorted(EVALUATOR_CONTEXT_FIELDS - set(raw))}, "
                f"extra={sorted(set(raw) - EVALUATOR_CONTEXT_FIELDS)}"
            )
        row = dict(raw)
        for key in (
            "evaluator_context_id",
            "state_id",
            "card_id",
            "regime",
            "authorized_user_context",
            "coverage_rationale",
            "context_payload_sha256",
        ):
            if not isinstance(row[key], str) or not row[key].strip():
                raise RuntimeError(
                    f"evaluator context row {row_index} has invalid {key}"
                )
        if row["regime"] not in {regime.value for regime in ResourceNeedRegime}:
            raise RuntimeError(
                f"evaluator context row {row_index} has unknown regime {row['regime']!r}"
            )
        needed_sources = row["needed_memory_sources"]
        if (
            not isinstance(needed_sources, list)
            or any(
                not isinstance(source, str)
                or source not in {member.value for member in MemorySource}
                for source in needed_sources
            )
            or len(needed_sources) != len(set(needed_sources))
        ):
            raise RuntimeError(
                f"evaluator context row {row_index} has invalid needed_memory_sources"
            )
        try:
            validate_regime_needed_sources(row["regime"], needed_sources)
        except ValueError as exc:
            raise RuntimeError(
                f"evaluator context row {row_index} has inconsistent structured need: {exc}"
            ) from exc
        annotations = row["memory_annotations"]
        if not isinstance(annotations, list):
            raise RuntimeError(
                f"evaluator context row {row_index} memory_annotations is not a list"
            )
        memory_ids: set[str] = set()
        for annotation_index, annotation in enumerate(annotations, 1):
            if not isinstance(annotation, dict) or set(annotation) != EVALUATOR_MEMORY_ANNOTATION_FIELDS:
                raise RuntimeError(
                    f"evaluator context row {row_index} annotation {annotation_index} "
                    "has a schema mismatch"
                )
            memory_id = annotation["memory_id"]
            if not isinstance(memory_id, str) or not memory_id:
                raise RuntimeError("evaluator memory annotation has invalid memory_id")
            if memory_id in memory_ids:
                raise RuntimeError(
                    f"evaluator context row {row_index} repeats memory_id {memory_id}"
                )
            memory_ids.add(memory_id)
            if annotation["source"] not in {source.value for source in MemorySource}:
                raise RuntimeError("evaluator memory annotation has invalid source")
            created = annotation["created_session"]
            if isinstance(created, bool) or not isinstance(created, int) or created < 0:
                raise RuntimeError(
                    "evaluator memory annotation has invalid created_session"
                )
            for boolean_key in ("stale", "conflicts_with_current_state"):
                if not isinstance(annotation[boolean_key], bool):
                    raise RuntimeError(
                        f"evaluator memory annotation has invalid {boolean_key}"
                    )
            if annotation["private_sensitivity"] not in {"ordinary", "sensitive"}:
                raise RuntimeError(
                    "evaluator memory annotation has invalid private_sensitivity"
                )
            if annotation["item_utility"] not in {
                "helpful",
                "irrelevant",
                "harmful",
            }:
                raise RuntimeError(
                    "evaluator memory annotation has invalid item_utility"
                )
        expected_hash = evaluator_context_payload_sha256(row)
        if row["context_payload_sha256"] != expected_hash:
            raise RuntimeError(
                f"evaluator context payload hash mismatch for state {row['state_id']}"
            )
        if row["state_id"] in by_state:
            raise RuntimeError(
                f"duplicate evaluator context state_id {row['state_id']}"
            )
        if row["card_id"] in by_card:
            raise RuntimeError(
                f"duplicate evaluator context card_id {row['card_id']}"
            )
        if row["evaluator_context_id"] in seen_context_ids:
            raise RuntimeError(
                "duplicate evaluator_context_id " + row["evaluator_context_id"]
            )
        by_state[row["state_id"]] = row
        by_card[row["card_id"]] = row
        seen_context_ids.add(row["evaluator_context_id"])
        normalized_rows.append(row)
    if not normalized_rows:
        raise RuntimeError("evaluator context table is empty")
    map_rows = [
        {
            "state_id": row["state_id"],
            "card_id": row["card_id"],
            "evaluator_context_id": row["evaluator_context_id"],
            "context_payload_sha256": row["context_payload_sha256"],
        }
        for row in sorted(normalized_rows, key=lambda item: item["state_id"])
    ]
    index = EvaluatorContextIndex(
        by_state=by_state,
        by_card=by_card,
        source_sha256=source_sha256 or sha256_text(canonical_json(normalized_rows)),
        map_sha256=sha256_text(canonical_json(map_rows)),
    )
    if states is not None:
        index.require_states(states, exact=require_exact)
    return index


def load_evaluator_context_index(
    path: str | Path,
    *,
    states: Sequence[PMV2State] | None = None,
    require_exact: bool = True,
) -> EvaluatorContextIndex:
    path = Path(path)
    return build_evaluator_context_index(
        list(iter_jsonl(path)),
        states=states,
        require_exact=require_exact,
        source_sha256=sha256_file(path),
    )


class GeneratedMemory(StrictModel):
    memory_id: str
    source: MemorySource
    text: str = Field(min_length=1)
    created_session: int = Field(ge=0)
    # These labels must be emitted explicitly.  Defaults made the provider
    # schema optional and hid whether the generator actually considered each
    # item-level risk dimension.
    stale: bool
    conflicts_with_current_state: bool
    private_sensitivity: Literal["ordinary", "sensitive"]
    item_utility: Literal["helpful", "irrelevant", "harmful"]

    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, value):
        return value if isinstance(value, MemorySource) else MemorySource(str(value))


class GeneratedProfileMemoryDraft(StrictModel):
    """A stable user attribute fragment, never advice or an episodic event."""

    stable_user_fact: str = Field(min_length=5, max_length=160)
    relevance_explanation: str = Field(min_length=5, max_length=120)
    private_sensitivity: Literal["ordinary", "sensitive"]


class GeneratedSummaryMemoryDraft(StrictModel):
    """A cross-session recurring pattern fragment, never a recommendation."""

    cross_session_pattern: str = Field(min_length=5, max_length=160)
    relevance_explanation: str = Field(min_length=5, max_length=120)
    private_sensitivity: Literal["ordinary", "sensitive"]


class GeneratedEventMemoryDraft(StrictModel):
    """A concrete past-event fragment, never general advice."""

    concrete_past_event: str = Field(min_length=5, max_length=160)
    relevance_explanation: str = Field(min_length=5, max_length=120)
    private_sensitivity: Literal["ordinary", "sensitive"]


class GeneratedProfileNeededSourceDraft(StrictModel):
    helpful: GeneratedProfileMemoryDraft
    distractor: GeneratedProfileMemoryDraft


class GeneratedSummaryNeededSourceDraft(StrictModel):
    helpful: GeneratedSummaryMemoryDraft
    distractor: GeneratedSummaryMemoryDraft


class GeneratedEventNeededSourceDraft(StrictModel):
    helpful: GeneratedEventMemoryDraft
    distractor: GeneratedEventMemoryDraft


class GeneratedProfileDistractorSourceDraft(StrictModel):
    distractor: GeneratedProfileMemoryDraft


class GeneratedSummaryDistractorSourceDraft(StrictModel):
    distractor: GeneratedSummaryMemoryDraft


class GeneratedEventDistractorSourceDraft(StrictModel):
    distractor: GeneratedEventMemoryDraft


class GeneratedHarmfulProfileSourceDraft(StrictModel):
    outdated_user_fact: str = Field(min_length=5, max_length=160)
    explicit_current_update: str = Field(min_length=5, max_length=160)
    why_recall_is_harmful: str = Field(min_length=5, max_length=120)


class GeneratedHarmfulSummarySourceDraft(StrictModel):
    outdated_cross_session_pattern: str = Field(min_length=5, max_length=160)
    explicit_current_update: str = Field(min_length=5, max_length=160)
    why_recall_is_harmful: str = Field(min_length=5, max_length=120)


class GeneratedHarmfulEventSourceDraft(StrictModel):
    unrelated_sensitive_past_event: str = Field(min_length=5, max_length=160)
    why_recall_is_intrusive: str = Field(min_length=5, max_length=120)


class GeneratedDialogueTurnDraft(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=180)


class GeneratedCaseSurfaceDraft(StrictModel):
    """Provider-authored surface fields; semantic checks run after transport.

    Do not put cross-field validators on this provider-facing class.  Strict
    Structured Outputs can enforce field shape but not these Python validators;
    chronology/topic drift is handled by deterministic case-level fallback.
    """

    current_user_text: str = Field(min_length=1, max_length=220)
    dialogue_before_current: list[GeneratedDialogueTurnDraft] = Field(
        min_length=2, max_length=4
    )
    session_summary: str = Field(min_length=1, max_length=250)
    authorized_user_context: str = Field(min_length=1, max_length=250)
    coverage_rationale: str = Field(min_length=1, max_length=180)

class GeneratedNoMemoryNeedCaseDraft(GeneratedCaseSurfaceDraft):
    profile_source: GeneratedProfileDistractorSourceDraft
    summary_source: GeneratedSummaryDistractorSourceDraft
    event_source: GeneratedEventDistractorSourceDraft


class GeneratedProfileNeededCaseDraft(GeneratedCaseSurfaceDraft):
    profile_source: GeneratedProfileNeededSourceDraft
    summary_source: GeneratedSummaryDistractorSourceDraft
    event_source: GeneratedEventDistractorSourceDraft


class GeneratedSummaryNeededCaseDraft(GeneratedCaseSurfaceDraft):
    profile_source: GeneratedProfileDistractorSourceDraft
    summary_source: GeneratedSummaryNeededSourceDraft
    event_source: GeneratedEventDistractorSourceDraft


class GeneratedEventNeededCaseDraft(GeneratedCaseSurfaceDraft):
    profile_source: GeneratedProfileDistractorSourceDraft
    summary_source: GeneratedSummaryDistractorSourceDraft
    event_source: GeneratedEventNeededSourceDraft


class GeneratedMultiSourceNeededCaseDraft(GeneratedCaseSurfaceDraft):
    # The compatibility/full data contract uses all three sources here.  The
    # single-source regimes still provide the subset contrasts required to
    # learn MP/MS/ME abstention independently.
    profile_source: GeneratedProfileNeededSourceDraft
    summary_source: GeneratedSummaryNeededSourceDraft
    event_source: GeneratedEventNeededSourceDraft
    profile_unique_contribution: str = Field(min_length=5, max_length=120)
    summary_unique_contribution: str = Field(min_length=5, max_length=120)
    event_unique_contribution: str = Field(min_length=5, max_length=120)


class GeneratedMemoryHarmfulCaseDraft(GeneratedCaseSurfaceDraft):
    profile_source: GeneratedHarmfulProfileSourceDraft
    summary_source: GeneratedHarmfulSummarySourceDraft
    event_source: GeneratedHarmfulEventSourceDraft


class GeneratedBundleDraft(StrictModel):
    """Provider-facing schema with exact named slots instead of self-reported labels.

    Regime, source, item utility, case/session IDs, and timestamps are deliberate
    experimental-design metadata.  They are compiled locally after generation so
    the language model cannot silently omit or contradict them.  Human semantic
    audit remains authoritative for whether the generated text fits each slot.
    """

    profile_summary: str = Field(min_length=1, max_length=400)
    stable_preferences: list[
        Annotated[str, Field(min_length=1, max_length=120)]
    ] = Field(min_length=1, max_length=5)
    boundaries: list[
        Annotated[str, Field(min_length=1, max_length=120)]
    ] = Field(min_length=1, max_length=5)
    context_only: GeneratedNoMemoryNeedCaseDraft
    profile_needed: GeneratedProfileNeededCaseDraft
    summary_needed: GeneratedSummaryNeededCaseDraft
    event_needed: GeneratedEventNeededCaseDraft
    multi_source_needed: GeneratedMultiSourceNeededCaseDraft
    memory_harmful: GeneratedMemoryHarmfulCaseDraft
    strategy_helpful: GeneratedNoMemoryNeedCaseDraft
    strategy_harmful: GeneratedNoMemoryNeedCaseDraft
    ambiguous: GeneratedNoMemoryNeedCaseDraft


class GenerationDraftCompilationError(RuntimeError):
    """Preserve a paid, schema-valid draft rejected by the local compiler."""

    def __init__(
        self,
        *,
        call: CallResult,
        draft: GeneratedBundleDraft,
        compilation_error: Exception,
    ) -> None:
        self.call = call
        self.parsed_payload = draft.model_dump(mode="json")
        self.compilation_error = (
            f"{type(compilation_error).__name__}: {compilation_error}"
        )
        super().__init__(
            "GeneratedBundleDraft passed provider schema but failed deterministic "
            f"bundle compilation: {self.compilation_error}"
        )


class GeneratedStateCase(StrictModel):
    case_id: str
    semantic_family: str = Field(min_length=1)
    surface_form_id: str = Field(min_length=1)
    regime: ResourceNeedRegime
    current_user_text: str = Field(min_length=1)
    recent_dialogue: list[DialogueTurn]
    session_summary: str
    session_index: int = Field(ge=1)
    profile_memories: list[GeneratedMemory]
    summary_memories: list[GeneratedMemory]
    event_memories: list[GeneratedMemory]
    needed_memory_sources: list[MemorySource]
    authorized_user_context: str = Field(min_length=1)
    coverage_rationale: str = Field(min_length=1)

    @property
    def materially_useful_memory_sources(self) -> list[MemorySource]:
        """Canonical V8 meaning for the legacy structured-source target.

        ``needed_memory_sources`` remains serialized for frozen V1/V7 artifacts and
        downstream compatibility.  It must be interpreted as sources expected to
        add material, nonredundant value over Context Only, not as sources without
        which a minimally acceptable response is impossible.
        """

        return list(self.needed_memory_sources)

    @field_validator("regime", mode="before")
    @classmethod
    def parse_regime(cls, value):
        return (
            value
            if isinstance(value, ResourceNeedRegime)
            else ResourceNeedRegime(str(value))
        )

    @field_validator("needed_memory_sources", mode="before")
    @classmethod
    def parse_needed_sources(cls, value):
        if not isinstance(value, list):
            raise TypeError("needed_memory_sources must be a list")
        return [
            source if isinstance(source, MemorySource) else MemorySource(str(source))
            for source in value
        ]

    @model_validator(mode="after")
    def source_consistency(self):
        if not self.recent_dialogue:
            raise ValueError("recent_dialogue must contain strictly prior dialogue")
        if self.recent_dialogue[-1].role != "assistant":
            raise ValueError("recent_dialogue must end before the current user turn")
        if any(
            self.recent_dialogue[index - 1].role
            == self.recent_dialogue[index].role
            for index in range(1, len(self.recent_dialogue))
        ):
            raise ValueError("recent_dialogue roles must alternate")
        normalized_current = normalize_text(self.current_user_text)
        if any(
            normalize_text(turn.content) == normalized_current
            for turn in self.recent_dialogue
        ):
            raise ValueError("recent_dialogue repeats current_user_text")
        expected = {
            "profile_memories": MemorySource.MP,
            "summary_memories": MemorySource.MS,
            "event_memories": MemorySource.ME,
        }
        for field_name, source in expected.items():
            memories = getattr(self, field_name)
            if any(item.source is not source for item in memories):
                raise ValueError(f"{field_name} contains a wrong source")
            future = [
                item.memory_id
                for item in memories
                if item.created_session >= self.session_index
            ]
            if future:
                raise ValueError(
                    f"{field_name} contains memory not strictly prior to session "
                    f"{self.session_index}: {future}"
                )
        self.needed_memory_sources = validate_regime_needed_sources(
            self.regime,
            self.needed_memory_sources,
        )
        helpful_sources = {
            item.source
            for memories in expected
            for item in getattr(self, memories)
            if item.item_utility == "helpful"
        }
        needed_sources = set(self.needed_memory_sources)
        if helpful_sources != needed_sources:
            raise ValueError(
                "item-level helpful sources must exactly equal needed_memory_sources"
            )
        same_source_without_item_contrast = [
            source.value
            for source in sorted(needed_sources, key=lambda value: value.value)
            if not any(
                item.item_utility != "helpful"
                for memories in expected
                for item in getattr(self, memories)
                if item.source is source
            )
        ]
        if same_source_without_item_contrast:
            raise ValueError(
                "every needed source must also contain an item-level distractor: "
                f"{same_source_without_item_contrast}"
            )
        unsafe_helpful = [
            item.memory_id
            for memories in expected
            for item in getattr(self, memories)
            if item.item_utility == "helpful"
            and (item.stale or item.conflicts_with_current_state)
        ]
        if unsafe_helpful:
            raise ValueError(
                "helpful memory items cannot be stale or conflicting: "
                f"{unsafe_helpful}"
            )
        if self.regime is ResourceNeedRegime.MEMORY_HARMFUL and not any(
            item.item_utility == "harmful"
            for memories in expected
            for item in getattr(self, memories)
        ):
            raise ValueError("memory_harmful requires at least one harmful item")
        return self


class GeneratedUserBundle(StrictModel):
    user_id: str
    profile_summary: str = Field(min_length=1)
    stable_preferences: list[str]
    boundaries: list[str]
    cases: list[GeneratedStateCase] = Field(min_length=4)
    generator_seed_id: str
    # Provenance is attached locally after a successful provider response.  It
    # is deliberately absent from the provider-facing Structured Outputs
    # schema, which cannot permit arbitrary object keys in strict mode.
    provenance: SkipJsonSchema[dict[str, Any]] = Field(default_factory=dict)


GENERATION_CASE_FIELDS: tuple[tuple[str, ResourceNeedRegime], ...] = (
    ("context_only", ResourceNeedRegime.CONTEXT_ONLY),
    ("profile_needed", ResourceNeedRegime.PROFILE_NEEDED),
    ("summary_needed", ResourceNeedRegime.SUMMARY_NEEDED),
    ("event_needed", ResourceNeedRegime.EVENT_NEEDED),
    ("multi_source_needed", ResourceNeedRegime.MULTI_SOURCE_NEEDED),
    ("memory_harmful", ResourceNeedRegime.MEMORY_HARMFUL),
    ("strategy_helpful", ResourceNeedRegime.STRATEGY_HELPFUL),
    ("strategy_harmful", ResourceNeedRegime.STRATEGY_HARMFUL),
    ("ambiguous", ResourceNeedRegime.AMBIGUOUS),
)

GENERATION_FAMILY_ANCHORS: dict[str, tuple[str, ...]] = {
    "relocation_loneliness": ("move", "moved", "relocat", "new city", "new place"),
    "workload_burnout": ("work", "workload", "deadline", "overtime", "burnout"),
    "friendship_distance": ("friend", "friendship", "drifted", "distant"),
    "family_expectations": ("family", "parent", "relative", "expectation"),
    "relationship_uncertainty": ("partner", "relationship", "dating", "break up"),
    "academic_pressure": ("exam", "grade", "study", "school", "academic"),
    "career_change": ("career", "job change", "profession", "resign"),
    "caregiving_stress": ("caregiv", "caring for", "dependent", "elder care"),
    "social_anxiety": ("social", "crowd", "meeting people", "judged"),
    "sleep_disruption": ("sleep", "insomnia", "awake", "rest"),
    "identity_transition": ("identity", "who i am", "transition", "sense of self"),
    "financial_uncertainty": ("money", "financial", "rent", "debt", "budget"),
    "grief_adjustment": ("grief", "loss", "passed away", "bereav"),
    "health_routine_stress": ("health", "exercise", "routine", "appointment"),
    "conflict_repair": ("conflict", "argument", "apolog", "repair"),
    "self_confidence": ("confidence", "self-doubt", "capable", "insecure"),
    "belonging_and_isolation": ("belong", "isolat", "left out", "alone"),
    "decision_paralysis": ("decision", "choose", "choice", "stuck"),
    "parenting_pressure": ("parenting", "child", "kid", "parent"),
    "workplace_conflict": ("coworker", "manager", "workplace", "colleague"),
    "life_stage_transition": ("life stage", "milestone", "retire", "adulthood"),
    "motivation_loss": ("motivation", "unmotivated", "drive", "procrastinat"),
    "trust_rebuilding": ("trust", "betray", "rebuild", "let down"),
    "uncertain_future": ("future", "uncertain", "unknown", "what comes next"),
}

# Frozen, human-readable topics used by the deterministic evidence compiler.
# The provider may paraphrase case surfaces, but it never decides which memory
# is helpful, irrelevant, harmful, stale, or conflicting.  Every topic contains
# at least one literal anchor from GENERATION_FAMILY_ANCHORS.
GENERATION_FAMILY_TOPICS: dict[str, str] = {
    "relocation_loneliness": "moving to a new city",
    "workload_burnout": "work deadline pressure and burnout",
    "friendship_distance": "a distant friendship",
    "family_expectations": "family expectations",
    "relationship_uncertainty": "uncertainty in a partner relationship",
    "academic_pressure": "academic study and exams",
    "career_change": "a career and job change",
    "caregiving_stress": "caregiving for a dependent relative",
    "social_anxiety": "social anxiety when meeting people",
    "sleep_disruption": "sleep disruption and insomnia",
    "identity_transition": "an identity transition",
    "financial_uncertainty": "financial uncertainty about rent and budget",
    "grief_adjustment": "grief after a loss",
    "health_routine_stress": "stress around a health and exercise routine",
    "conflict_repair": "repairing a conflict after an argument",
    "self_confidence": "self-doubt and confidence",
    "belonging_and_isolation": "feeling isolated and unsure of belonging",
    "decision_paralysis": "feeling stuck over a decision",
    "parenting_pressure": "parenting pressure involving a child",
    "workplace_conflict": "a workplace conflict with a coworker",
    "life_stage_transition": "a life stage transition",
    "motivation_loss": "loss of motivation and procrastination",
    "trust_rebuilding": "rebuilding trust after feeling let down",
    "uncertain_future": "an uncertain future",
}

if set(GENERATION_FAMILY_TOPICS) != set(GENERATION_FAMILY_ANCHORS):
    raise RuntimeError("generation topic blueprints do not cover every family")

SOURCE_DRAFT_FIELDS: tuple[tuple[str, MemorySource], ...] = (
    ("profile_source", MemorySource.MP),
    ("summary_source", MemorySource.MS),
    ("event_source", MemorySource.ME),
)


def _require_complete_generation_design(
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> tuple[list[str], list[ResourceNeedRegime]]:
    families = [str(value).strip() for value in semantic_families]
    if len(families) < 3 or any(not value for value in families):
        raise ValueError("synthetic generation requires at least three semantic families")
    if len(families) != len(set(families)):
        raise ValueError("synthetic generation semantic families must be unique")
    unknown_families = sorted(set(families) - set(GENERATION_FAMILY_ANCHORS))
    if unknown_families:
        raise ValueError(
            f"synthetic generation has unknown semantic families: {unknown_families}"
        )
    parsed_regimes = [
        value
        if isinstance(value, ResourceNeedRegime)
        else ResourceNeedRegime(str(value))
        for value in regimes
    ]
    expected = [regime for _, regime in GENERATION_CASE_FIELDS]
    if len(parsed_regimes) != len(expected) or set(parsed_regimes) != set(expected):
        raise ValueError(
            "role-slot generation requires exactly the complete nine-regime design"
        )
    return families, parsed_regimes


def generation_case_family_assignments(
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> dict[str, str]:
    """Bind every named regime slot to a semantic family before API use."""

    families, _ = _require_complete_generation_design(semantic_families, regimes)
    return {
        field_name: families[index % len(families)]
        for index, (field_name, _) in enumerate(GENERATION_CASE_FIELDS)
    }


def generation_distractor_family_assignments(
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> dict[str, dict[str, str]]:
    """Assign every distractor source a frozen off-topic family."""

    families, _ = _require_complete_generation_design(semantic_families, regimes)
    targets = generation_case_family_assignments(families, regimes)
    result: dict[str, dict[str, str]] = {}
    for case_index, (case_field, _) in enumerate(GENERATION_CASE_FIELDS):
        target_index = families.index(targets[case_field])
        result[case_field] = {}
        for source_index, (source_field, _) in enumerate(SOURCE_DRAFT_FIELDS):
            offset = 1 + (source_index % (len(families) - 1))
            result[case_field][source_field] = families[
                (target_index + offset) % len(families)
            ]
    return result


def _family_anchor_hits(text: str, family: str) -> list[str]:
    normalized = normalize_text(text)
    return [
        anchor
        for anchor in GENERATION_FAMILY_ANCHORS[family]
        if anchor in normalized
    ]


def _compiled_memory_id(
    *,
    user_id: str,
    case_field: str,
    source: MemorySource,
    role: str,
    text: str,
) -> str:
    return "draft_" + sha256_text(
        f"{user_id}|{case_field}|{source.value}|{role}|{text}"
    )[:20]


MEMORY_ADVICE_RE = re.compile(
    r"\b(?:you|your|should|could|try|consider|have you|might help|need to|"
    r"important to|remember to)\b|\?",
    re.IGNORECASE,
)


def _profile_memory_text(item: GeneratedProfileMemoryDraft) -> str:
    fact = item.stable_user_fact.strip().rstrip(".")
    if fact.casefold().startswith("the user "):
        return fact + "."
    return f"The user {fact}."


def _summary_memory_text(item: GeneratedSummaryMemoryDraft) -> str:
    pattern = item.cross_session_pattern.strip().rstrip(".")
    if pattern.casefold().startswith("across "):
        return pattern + "."
    return f"Across multiple prior sessions, the user {pattern}."


def _event_memory_text(item: GeneratedEventMemoryDraft) -> str:
    event = item.concrete_past_event.strip().rstrip(".")
    if event.casefold().startswith(("in ", "during ", "at ")):
        return event + "."
    return f"In a prior session, the user {event}."


def _source_draft_text(item: Any) -> str:
    if isinstance(item, GeneratedProfileMemoryDraft):
        return _profile_memory_text(item)
    if isinstance(item, GeneratedSummaryMemoryDraft):
        return _summary_memory_text(item)
    if isinstance(item, GeneratedEventMemoryDraft):
        return _event_memory_text(item)
    raise TypeError(f"unsupported source item draft: {type(item).__name__}")


def _source_draft_raw_content(item: Any) -> str:
    if isinstance(item, GeneratedProfileMemoryDraft):
        return item.stable_user_fact
    if isinstance(item, GeneratedSummaryMemoryDraft):
        return item.cross_session_pattern
    if isinstance(item, GeneratedEventMemoryDraft):
        return item.concrete_past_event
    raise TypeError(f"unsupported source item draft: {type(item).__name__}")


def _generation_user_design_offset(user_id: str) -> int:
    """Counterbalance regime positions without using a regime/oracle field.

    Formal development IDs end in ``_uNNN``; cycling those numbers guarantees
    every split visits all nine positions as its user count grows.  The held-out
    compatibility user uses a stable hash.  Neither value enters PM text.
    """

    match = re.search(r"_u(\d+)$", user_id)
    if match:
        return (int(match.group(1)) - 1) % len(GENERATION_CASE_FIELDS)
    return int(sha256_text(user_id)[:8], 16) % len(GENERATION_CASE_FIELDS)


def _memory_age_sessions(
    *, user_id: str, case_field: str, source: MemorySource, slot_index: int
) -> int:
    """Sample age from a utility-blind, reproducible case/source stream.

    ``slot_index`` identifies a pre-label construction slot. Utility, stale,
    conflict, and needed-source labels are deliberately absent from the seed.
    Sampling both ages from one stream prevents a fixed slot/age interval from
    encoding helpfulness while retaining deterministic trace replay.
    """

    if slot_index not in (0, 1):
        raise ValueError("memory blueprint slot_index must be 0 or 1")
    digest = sha256_text(f"pmv2-v11-age|{user_id}|{case_field}|{source.value}")
    rng = random.Random(int(digest[:16], 16))
    return rng.sample(list(range(1, 13)), k=2)[slot_index]


def _shuffle_compiled_memories(
    items: list[GeneratedMemory],
    *,
    user_id: str,
    case_field: str,
    source: MemorySource,
) -> list[GeneratedMemory]:
    """Shuffle display/retrieval rows using a stream independent from age/utility."""

    digest = sha256_text(
        f"pmv2-v11-row-order|{user_id}|{case_field}|{source.value}"
    )
    shuffled = list(items)
    random.Random(int(digest[:16], 16)).shuffle(shuffled)
    return shuffled


def lint_generation_draft(
    *,
    draft: GeneratedBundleDraft,
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> dict[str, Any]:
    """Deterministic pre-compiler lint for failures visible without judgment.

    This catches chronology, family-topic, source-form, and off-topic distractor
    violations.  It deliberately does not claim to replace blinded human review.
    """

    assignments = generation_case_family_assignments(semantic_families, regimes)
    distractor_assignments = generation_distractor_family_assignments(
        semantic_families, regimes
    )
    errors: list[dict[str, Any]] = []

    def fail(case_field: str, check: str, detail: str) -> None:
        errors.append({"case_field": case_field, "check": check, "detail": detail})

    profile_blob = " ".join(
        [draft.profile_summary, *draft.stable_preferences, *draft.boundaries]
    )
    for family in semantic_families:
        if not _family_anchor_hits(profile_blob, family):
            fail("bundle", "profile_family_coverage", family)

    normalized_currents: list[str] = []
    for case_field, regime in GENERATION_CASE_FIELDS:
        surface = getattr(draft, case_field)
        target_family = assignments[case_field]
        current_blob = f"{surface.current_user_text} {surface.session_summary}"
        if not _family_anchor_hits(current_blob, target_family):
            fail(case_field, "current_family_anchor", target_family)
        normalized_currents.append(normalize_text(surface.current_user_text))

        for source_field, source in SOURCE_DRAFT_FIELDS:
            source_draft = getattr(surface, source_field)
            helpful = getattr(source_draft, "helpful", None)
            distractor = getattr(source_draft, "distractor", None)
            if helpful is not None:
                raw = _source_draft_raw_content(helpful)
                if MEMORY_ADVICE_RE.search(raw):
                    fail(case_field, f"{source.value}_helpful_source_form", raw)
                if not _family_anchor_hits(raw, target_family):
                    fail(
                        case_field,
                        f"{source.value}_helpful_family_anchor",
                        target_family,
                    )
            if distractor is not None:
                raw = _source_draft_raw_content(distractor)
                distractor_family = distractor_assignments[case_field][source_field]
                if MEMORY_ADVICE_RE.search(raw):
                    fail(case_field, f"{source.value}_distractor_source_form", raw)
                if not _family_anchor_hits(raw, distractor_family):
                    fail(
                        case_field,
                        f"{source.value}_distractor_family_anchor",
                        distractor_family,
                    )
                if _family_anchor_hits(raw, target_family):
                    fail(
                        case_field,
                        f"{source.value}_distractor_leaks_target_family",
                        target_family,
                    )

        if regime is ResourceNeedRegime.MULTI_SOURCE_NEEDED:
            contributions = {
                normalize_text(surface.profile_unique_contribution),
                normalize_text(surface.summary_unique_contribution),
                normalize_text(surface.event_unique_contribution),
            }
            if len(contributions) != 3:
                fail(case_field, "multi_source_unique_contributions", "not unique")

        if regime is ResourceNeedRegime.MEMORY_HARMFUL:
            harmful_pairs = (
                (
                    surface.profile_source.outdated_user_fact,
                    surface.profile_source.explicit_current_update,
                    MemorySource.MP,
                ),
                (
                    surface.summary_source.outdated_cross_session_pattern,
                    surface.summary_source.explicit_current_update,
                    MemorySource.MS,
                ),
            )
            for old, update, source in harmful_pairs:
                if normalize_text(old) == normalize_text(update):
                    fail(case_field, f"{source.value}_harmful_has_no_update", old)
                if MEMORY_ADVICE_RE.search(old) or MEMORY_ADVICE_RE.search(update):
                    fail(case_field, f"{source.value}_harmful_source_form", old)
                if not _family_anchor_hits(old, target_family):
                    fail(
                        case_field,
                        f"{source.value}_harmful_old_family_anchor",
                        target_family,
                    )
                if not _family_anchor_hits(update, target_family):
                    fail(
                        case_field,
                        f"{source.value}_harmful_update_family_anchor",
                        target_family,
                    )
            event_family = distractor_assignments[case_field]["event_source"]
            event_text = surface.event_source.unrelated_sensitive_past_event
            if not _family_anchor_hits(event_text, event_family):
                fail(case_field, "ME_harmful_off_topic_anchor", event_family)
            if _family_anchor_hits(event_text, target_family):
                fail(case_field, "ME_harmful_leaks_target_family", target_family)

        if regime is ResourceNeedRegime.STRATEGY_HELPFUL and not re.search(
            r"\b(?:strategy|reflection|question|reassurance|suggestion|guidance|support)\b",
            surface.coverage_rationale,
            re.IGNORECASE,
        ):
            fail(case_field, "strategy_helpful_rationale", surface.coverage_rationale)
        if regime is ResourceNeedRegime.STRATEGY_HARMFUL and not re.search(
            r"\b(?:premature|directive|intrusive|listen|non-directive|not advice)\b",
            surface.coverage_rationale,
            re.IGNORECASE,
        ):
            fail(case_field, "strategy_harmful_rationale", surface.coverage_rationale)

    if len(normalized_currents) != len(set(normalized_currents)):
        fail("bundle", "unique_current_user_text", "duplicate normalized text")
    return {
        "status": "PASS" if not errors else "FAIL",
        "protocol": DATA_GENERATION_CONTRACT_VERSION,
        "errors": errors,
        "case_family_assignments": assignments,
        "distractor_family_assignments": distractor_assignments,
    }


def _surface_payload(surface: GeneratedCaseSurfaceDraft) -> dict[str, Any]:
    return {
        "current_user_text": surface.current_user_text,
        "dialogue_before_current": [
            turn.model_dump(mode="json") for turn in surface.dialogue_before_current
        ],
        "session_summary": surface.session_summary,
        "authorized_user_context": surface.authorized_user_context,
        "coverage_rationale": surface.coverage_rationale,
    }


def _fallback_generation_surface(
    *, case_field: str, regime: ResourceNeedRegime, family: str
) -> GeneratedCaseSurfaceDraft:
    """Create a transparent, deterministic surface when a paraphrase drifts.

    Fallback is case-local and recorded in provenance.  It prevents one bad
    paraphrase from discarding a paid user bundle, while the later blinded
    human gate remains authoritative for naturalness and regime validity.
    """

    topic = GENERATION_FAMILY_TOPICS[family]
    current_by_regime = {
        ResourceNeedRegime.CONTEXT_ONLY: (
            f"The situation around {topic} has been weighing on me today. I keep "
            "coming back to how it felt."
        ),
        ResourceNeedRegime.PROFILE_NEEDED: (
            f"I am unsure how to respond to {topic}, and I feel more overwhelmed "
            "than I expected."
        ),
        ResourceNeedRegime.SUMMARY_NEEDED: (
            f"Something about {topic} is bothering me again today. I cannot tell "
            "why it feels so familiar."
        ),
        ResourceNeedRegime.EVENT_NEEDED: (
            f"What happened with {topic} today brought up a strong sense of déjà vu, "
            "and I am trying to understand why."
        ),
        ResourceNeedRegime.MULTI_SOURCE_NEEDED: (
            f"I am having a hard time with {topic} again. Several feelings seem "
            "tangled together, and I cannot tell where to begin."
        ),
        ResourceNeedRegime.MEMORY_HARMFUL: (
            f"Something difficult happened around {topic} today, and I want to put "
            "the details into words."
        ),
        ResourceNeedRegime.STRATEGY_HELPFUL: (
            f"I feel stuck with {topic}. Could you help me think of one gentle thing "
            "I might try next?"
        ),
        ResourceNeedRegime.STRATEGY_HARMFUL: (
            f"I am upset about {topic}. Please listen for now; I do not want advice "
            "in this turn."
        ),
        ResourceNeedRegime.AMBIGUOUS: (
            f"I am uncertain about {topic}; either listening or one small suggestion "
            "might help."
        ),
    }
    return GeneratedCaseSurfaceDraft(
        current_user_text=current_by_regime[regime],
        dialogue_before_current=[
            GeneratedDialogueTurnDraft(
                role="user", content=f"Earlier I mentioned {topic}."
            ),
            GeneratedDialogueTurnDraft(
                role="assistant",
                content="Which part of that feels most important right now?",
            ),
        ],
        session_summary=f"The current concern centers on {topic}.",
        authorized_user_context=(
            f"Use only facts stated in the visible dialogue about {topic}; do not "
            "infer hidden events or preferences."
        ),
        # Backward-compatible provider-schema placeholder.  The compiler
        # replaces this field with the frozen rationale below, just as it
        # ignores provider-authored memory/oracle slots.
        coverage_rationale=_deterministic_coverage_rationale(regime),
    )


def _deterministic_coverage_rationale(regime: ResourceNeedRegime) -> str:
    """Return the evaluator-only resource rationale frozen by the design.

    A provider-written rationale is itself an oracle label in prose.  Trusting
    it caused the v6 event-needed case to claim that no memory was needed.
    Keeping this mapping local makes the resource contract auditable and keeps
    provider prose from silently reversing a label.
    """

    return {
        ResourceNeedRegime.CONTEXT_ONLY: (
            "Recent dialogue fully explains the concern, and the base generator can "
            "provide natural acknowledgment and exploration; the displayed external "
            "resources add no material marginal value."
        ),
        ResourceNeedRegime.PROFILE_NEEDED: (
            "A stable preference about how support should be offered is material; "
            "a recurring pattern or isolated event does not supply that preference."
        ),
        ResourceNeedRegime.SUMMARY_NEEDED: (
            "A recurring cross-session pattern is material; a stable preference or "
            "one isolated event does not establish that pattern."
        ),
        ResourceNeedRegime.EVENT_NEEDED: (
            "One concrete prior coping incident is material; a stable preference or "
            "broad recurring summary does not supply that event detail."
        ),
        ResourceNeedRegime.MULTI_SOURCE_NEEDED: (
            "The stable support preference, recurring pattern, and concrete coping "
            "event provide distinct material marginal value; they are not claimed to "
            "be absolutely indispensable."
        ),
        ResourceNeedRegime.MEMORY_HARMFUL: (
            "An explicit current correction makes old profile and summary assumptions "
            "unsafe, while an unrelated sensitive event would be intrusive."
        ),
        ResourceNeedRegime.STRATEGY_HELPFUL: (
            "A matching ESC card may add value beyond base empathy through reflection, "
            "reassurance, a question, or a light suggestion; it need not be a plan."
        ),
        ResourceNeedRegime.STRATEGY_HARMFUL: (
            "The user asks to be heard without advice, so directive cards are harmful; "
            "listening cards may still help because RS and directiveness are separate."
        ),
        ResourceNeedRegime.AMBIGUOUS: (
            "The current evidence does not clearly separate listening from one small "
            "suggestion, so several low-cost actions remain competitive."
        ),
    }[regime]


def lint_generation_surfaces(
    *,
    surfaces: dict[str, GeneratedCaseSurfaceDraft],
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> dict[str, Any]:
    """Check only provider-authored observable surfaces, never oracle evidence."""

    assignments = generation_case_family_assignments(semantic_families, regimes)
    errors: list[dict[str, str]] = []
    normalized_currents: list[str] = []
    for case_field, regime in GENERATION_CASE_FIELDS:
        surface = surfaces[case_field]
        target = assignments[case_field]
        blob = f"{surface.current_user_text} {surface.session_summary}"
        if not _family_anchor_hits(blob, target):
            errors.append(
                {"case_field": case_field, "check": "current_family_anchor", "detail": target}
            )
        for other in semantic_families:
            if other != target and _family_anchor_hits(blob, other):
                errors.append(
                    {
                        "case_field": case_field,
                        "check": "current_leaks_other_family",
                        "detail": str(other),
                    }
                )
        normalized_currents.append(normalize_text(surface.current_user_text))
        turns = surface.dialogue_before_current
        if turns[-1].role != "assistant":
            errors.append(
                {
                    "case_field": case_field,
                    "check": "dialogue_must_end_with_assistant",
                    "detail": turns[-1].role,
                }
            )
        if any(
            turns[index - 1].role == turns[index].role
            for index in range(1, len(turns))
        ):
            errors.append(
                {
                    "case_field": case_field,
                    "check": "dialogue_roles_must_alternate",
                    "detail": "non-alternating roles",
                }
            )
        normalized_current = normalize_text(surface.current_user_text)
        if any(normalize_text(turn.content) == normalized_current for turn in turns):
            errors.append(
                {
                    "case_field": case_field,
                    "check": "dialogue_repeats_current_user_text",
                    "detail": normalized_current,
                }
            )
        if regime is ResourceNeedRegime.STRATEGY_HELPFUL and not re.search(
            r"\b(?:structured|step|plan|guidance|framework|what to do|next)\b",
            surface.current_user_text,
            re.IGNORECASE,
        ):
            errors.append(
                {
                    "case_field": case_field,
                    "check": "strategy_helpful_user_request",
                    "detail": surface.current_user_text,
                }
            )
        if regime is ResourceNeedRegime.STRATEGY_HARMFUL and not re.search(
            r"\b(?:listen|hear me|do not give|don't give|without advice|not advice|"
            r"not ready for advice|just need)\b",
            surface.current_user_text,
            re.IGNORECASE,
        ):
            errors.append(
                {
                    "case_field": case_field,
                    "check": "strategy_harmful_user_boundary",
                    "detail": surface.current_user_text,
                }
            )
    if len(normalized_currents) != len(set(normalized_currents)):
        errors.append(
            {
                "case_field": "bundle",
                "check": "unique_current_user_text",
                "detail": "duplicate normalized text",
            }
        )
    return {
        "status": "PASS" if not errors else "FAIL",
        "protocol": DATA_GENERATION_CONTRACT_VERSION,
        "errors": errors,
        "case_family_assignments": assignments,
    }


def select_generation_surfaces(
    *,
    draft: GeneratedBundleDraft,
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> tuple[dict[str, GeneratedCaseSurfaceDraft], dict[str, Any]]:
    provider = {
        case_field: GeneratedCaseSurfaceDraft.model_validate(
            _surface_payload(getattr(draft, case_field))
        )
        for case_field, _ in GENERATION_CASE_FIELDS
    }
    provider_lint = lint_generation_surfaces(
        surfaces=provider,
        semantic_families=semantic_families,
        regimes=regimes,
    )
    invalid_cases = {
        str(error["case_field"])
        for error in provider_lint["errors"]
        if error["case_field"] != "bundle"
    }
    if any(error["case_field"] == "bundle" for error in provider_lint["errors"]):
        seen: set[str] = set()
        for case_field, _ in GENERATION_CASE_FIELDS:
            normalized = normalize_text(provider[case_field].current_user_text)
            if normalized in seen:
                invalid_cases.add(case_field)
            seen.add(normalized)
    assignments = generation_case_family_assignments(semantic_families, regimes)
    selected: dict[str, GeneratedCaseSurfaceDraft] = {}
    for case_field, regime in GENERATION_CASE_FIELDS:
        selected[case_field] = (
            _fallback_generation_surface(
                case_field=case_field,
                regime=regime,
                family=assignments[case_field],
            )
            if case_field in invalid_cases
            else provider[case_field]
        )
    compiled_lint = lint_generation_surfaces(
        surfaces=selected,
        semantic_families=semantic_families,
        regimes=regimes,
    )
    if compiled_lint["status"] != "PASS":
        raise RuntimeError(
            "deterministic surface fallback failed: "
            + canonical_json(compiled_lint["errors"])
        )
    report = {
        "protocol": DATA_GENERATION_CONTRACT_VERSION,
        "provider_surface_lint": provider_lint,
        "compiled_surface_lint": compiled_lint,
        "provider_cases": [
            case_field
            for case_field, _ in GENERATION_CASE_FIELDS
            if case_field not in invalid_cases
        ],
        "fallback_cases": sorted(invalid_cases),
        "fallback_case_count": len(invalid_cases),
        "fallback_case_rate": len(invalid_cases) / len(GENERATION_CASE_FIELDS),
        "compiled_surfaces": {
            case_field: _surface_payload(selected[case_field])
            for case_field, _ in GENERATION_CASE_FIELDS
        },
    }
    return selected, report


def _blueprint_source_raw(family: str, source: MemorySource) -> str:
    topic = GENERATION_FAMILY_TOPICS[family]
    if source is MemorySource.MP:
        return (
            "prefers acknowledgement before suggestions about "
            f"{topic}, with one gentle question at a time"
        )
    if source is MemorySource.MS:
        return (
            f"has repeatedly felt more distressed about {topic} when juggling "
            "demands, and calmer after naming one immediate concern"
        )
    return (
        f"described how, during a difficult period related to {topic}, they chose one "
        "manageable next step and felt less overwhelmed"
    )


def _blueprint_source_text(family: str, source: MemorySource) -> str:
    raw = _blueprint_source_raw(family, source)
    if source is MemorySource.MP:
        return _profile_memory_text(
            GeneratedProfileMemoryDraft(
                stable_user_fact=raw,
                relevance_explanation="Frozen deterministic evidence blueprint.",
                private_sensitivity="ordinary",
            )
        )
    if source is MemorySource.MS:
        return _summary_memory_text(
            GeneratedSummaryMemoryDraft(
                cross_session_pattern=raw,
                relevance_explanation="Frozen deterministic evidence blueprint.",
                private_sensitivity="ordinary",
            )
        )
    return _event_memory_text(
        GeneratedEventMemoryDraft(
            concrete_past_event=raw,
            relevance_explanation="Frozen deterministic evidence blueprint.",
            private_sensitivity="ordinary",
        )
    )


def generation_evidence_blueprint_hash() -> str:
    return sha256_text(
        canonical_json(
            {
                "protocol": DATA_GENERATION_CONTRACT_VERSION,
                "topics": GENERATION_FAMILY_TOPICS,
                "coverage_rationales": {
                    regime.value: _deterministic_coverage_rationale(regime)
                    for regime in ResourceNeedRegime
                },
                "sources": {
                    family: {
                        source.value: _blueprint_source_raw(family, source)
                        for source in MemorySource
                    }
                    for family in GENERATION_FAMILY_TOPICS
                },
            }
        )
    )


def _compile_source_draft(
    *,
    user_id: str,
    case_field: str,
    source: MemorySource,
    session_index: int,
    case_index: int,
    draft: (
        GeneratedProfileNeededSourceDraft
        | GeneratedSummaryNeededSourceDraft
        | GeneratedEventNeededSourceDraft
        | GeneratedProfileDistractorSourceDraft
        | GeneratedSummaryDistractorSourceDraft
        | GeneratedEventDistractorSourceDraft
        | GeneratedHarmfulProfileSourceDraft
        | GeneratedHarmfulSummarySourceDraft
        | GeneratedHarmfulEventSourceDraft
    ),
) -> list[GeneratedMemory]:
    def memory(
        *,
        slot_index: int,
        role: str,
        text: str,
        private_sensitivity: Literal["ordinary", "sensitive"],
        item_utility: Literal["helpful", "irrelevant", "harmful"],
        stale: bool = False,
        conflicts: bool = False,
    ) -> GeneratedMemory:
        age = _memory_age_sessions(
            user_id=user_id,
            case_field=case_field,
            source=source,
            slot_index=slot_index,
        )
        return GeneratedMemory(
            memory_id=_compiled_memory_id(
                user_id=user_id,
                case_field=case_field,
                source=source,
                role=role,
                text=text,
            ),
            source=source,
            text=text,
            created_session=session_index - age,
            stale=stale,
            conflicts_with_current_state=conflicts,
            private_sensitivity=private_sensitivity,
            item_utility=item_utility,
        )

    if isinstance(
        draft,
        (
            GeneratedProfileNeededSourceDraft,
            GeneratedSummaryNeededSourceDraft,
            GeneratedEventNeededSourceDraft,
        ),
    ):
        return [
            memory(
                slot_index=0,
                role="helpful",
                text=_source_draft_text(draft.helpful),
                private_sensitivity=draft.helpful.private_sensitivity,
                item_utility="helpful",
            ),
            memory(
                slot_index=1,
                role="distractor",
                text=_source_draft_text(draft.distractor),
                private_sensitivity=draft.distractor.private_sensitivity,
                item_utility="irrelevant",
            ),
        ]
    if isinstance(
        draft,
        (
            GeneratedProfileDistractorSourceDraft,
            GeneratedSummaryDistractorSourceDraft,
            GeneratedEventDistractorSourceDraft,
        ),
    ):
        return [
            memory(
                slot_index=0,
                role="distractor",
                text=_source_draft_text(draft.distractor),
                private_sensitivity=draft.distractor.private_sensitivity,
                item_utility="irrelevant",
            )
        ]
    if isinstance(draft, GeneratedHarmfulProfileSourceDraft):
        text = _profile_memory_text(
            GeneratedProfileMemoryDraft(
                stable_user_fact=draft.outdated_user_fact,
                relevance_explanation=draft.why_recall_is_harmful,
                private_sensitivity="sensitive",
            )
        )
        return [
            memory(
                slot_index=0,
                role="harmful",
                text=text,
                private_sensitivity="sensitive",
                item_utility="harmful",
                stale=True,
                conflicts=True,
            )
        ]
    if isinstance(draft, GeneratedHarmfulSummarySourceDraft):
        text = _summary_memory_text(
            GeneratedSummaryMemoryDraft(
                cross_session_pattern=draft.outdated_cross_session_pattern,
                relevance_explanation=draft.why_recall_is_harmful,
                private_sensitivity="sensitive",
            )
        )
        return [
            memory(
                slot_index=0,
                role="harmful",
                text=text,
                private_sensitivity="sensitive",
                item_utility="harmful",
                stale=True,
                conflicts=True,
            )
        ]
    if isinstance(draft, GeneratedHarmfulEventSourceDraft):
        text = _event_memory_text(
            GeneratedEventMemoryDraft(
                concrete_past_event=draft.unrelated_sensitive_past_event,
                relevance_explanation=draft.why_recall_is_intrusive,
                private_sensitivity="sensitive",
            )
        )
        return [
            memory(
                slot_index=0,
                role="harmful",
                text=text,
                private_sensitivity="sensitive",
                item_utility="harmful",
            )
        ]
    raise TypeError(f"unsupported generated source draft: {type(draft).__name__}")


def compile_generation_draft(
    *,
    draft: GeneratedBundleDraft,
    seed_dialogue: str,
    user_id: str,
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> GeneratedUserBundle:
    """Compile natural surfaces plus deterministic evidence into one bundle.

    Provider-authored memory slots remain in the raw trace for audit, but never
    determine oracle evidence or labels.  Invalid natural-language surfaces are
    replaced case-by-case by a recorded deterministic fallback and still face
    the independent human semantic gate.
    """

    assignments = generation_case_family_assignments(semantic_families, regimes)
    distractor_assignments = generation_distractor_family_assignments(
        semantic_families, regimes
    )
    selected_surfaces, surface_selection = select_generation_surfaces(
        draft=draft,
        semantic_families=semantic_families,
        regimes=regimes,
    )
    needed_by_regime = {
        ResourceNeedRegime.CONTEXT_ONLY: [],
        ResourceNeedRegime.PROFILE_NEEDED: [MemorySource.MP],
        ResourceNeedRegime.SUMMARY_NEEDED: [MemorySource.MS],
        ResourceNeedRegime.EVENT_NEEDED: [MemorySource.ME],
        ResourceNeedRegime.MULTI_SOURCE_NEEDED: [
            MemorySource.MP,
            MemorySource.MS,
            MemorySource.ME,
        ],
        ResourceNeedRegime.MEMORY_HARMFUL: [],
        ResourceNeedRegime.STRATEGY_HELPFUL: [],
        ResourceNeedRegime.STRATEGY_HARMFUL: [],
        ResourceNeedRegime.AMBIGUOUS: [],
    }
    output_field_by_source = {
        MemorySource.MP: "profile_memories",
        MemorySource.MS: "summary_memories",
        MemorySource.ME: "event_memories",
    }
    design_offset = _generation_user_design_offset(user_id)
    design_positions = {
        case_field: (index + design_offset) % len(GENERATION_CASE_FIELDS)
        for index, (case_field, _) in enumerate(GENERATION_CASE_FIELDS)
    }
    cases: list[GeneratedStateCase] = []
    for index, (case_field, regime) in enumerate(GENERATION_CASE_FIELDS):
        surface = selected_surfaces[case_field]
        target_family = assignments[case_field]
        # Regime order must not be recoverable from session_index.  Formal users
        # cyclically rotate all nine positions within each split; ages below use
        # the same neutral position and slot, never utility/helpfulness.
        design_position = design_positions[case_field]
        session_index = 20 + design_position * 3
        memory_pools: dict[str, list[GeneratedMemory]] = {}
        for source_field, source in SOURCE_DRAFT_FIELDS:
            def blueprint_memory(
                *,
                slot_index: int,
                role: str,
                family: str,
                utility: Literal["helpful", "irrelevant", "harmful"],
                stale: bool = False,
                conflicts: bool = False,
                sensitive: bool = False,
                raw_override: str | None = None,
            ) -> GeneratedMemory:
                if raw_override is None:
                    text = _blueprint_source_text(family, source)
                elif source is MemorySource.MP:
                    text = _profile_memory_text(
                        GeneratedProfileMemoryDraft(
                            stable_user_fact=raw_override,
                            relevance_explanation="Frozen harmful-memory blueprint.",
                            private_sensitivity="sensitive",
                        )
                    )
                elif source is MemorySource.MS:
                    text = _summary_memory_text(
                        GeneratedSummaryMemoryDraft(
                            cross_session_pattern=raw_override,
                            relevance_explanation="Frozen harmful-memory blueprint.",
                            private_sensitivity="sensitive",
                        )
                    )
                else:
                    text = _event_memory_text(
                        GeneratedEventMemoryDraft(
                            concrete_past_event=raw_override,
                            relevance_explanation="Frozen harmful-memory blueprint.",
                            private_sensitivity="sensitive",
                        )
                    )
                age = _memory_age_sessions(
                    user_id=user_id,
                    case_field=case_field,
                    source=source,
                    slot_index=slot_index,
                )
                return GeneratedMemory(
                    memory_id=_compiled_memory_id(
                        user_id=user_id,
                        case_field=case_field,
                        source=source,
                        role=role,
                        text=text,
                    ),
                    source=source,
                    text=text,
                    created_session=session_index - age,
                    stale=stale,
                    conflicts_with_current_state=conflicts,
                    private_sensitivity=("sensitive" if sensitive else "ordinary"),
                    item_utility=utility,
                )

            items: list[GeneratedMemory] = []
            if regime is ResourceNeedRegime.MEMORY_HARMFUL:
                topic = GENERATION_FAMILY_TOPICS[target_family]
                if source is MemorySource.MP:
                    raw = f"previously preferred not to discuss {topic}"
                elif source is MemorySource.MS:
                    raw = f"previously withdrew whenever {topic} arose"
                else:
                    off_topic = distractor_assignments[case_field][source_field]
                    raw = (
                        "last spring described a private incident involving "
                        f"{GENERATION_FAMILY_TOPICS[off_topic]}"
                    )
                items.append(
                    blueprint_memory(
                        slot_index=0,
                        role="harmful",
                        family=(
                            target_family
                            if source is not MemorySource.ME
                            else distractor_assignments[case_field][source_field]
                        ),
                        utility="harmful",
                        stale=source is not MemorySource.ME,
                        conflicts=source is not MemorySource.ME,
                        sensitive=True,
                        raw_override=raw,
                    )
                )
                first_off_topic = distractor_assignments[case_field][source_field]
                second_off_topic = next(
                    family
                    for family in semantic_families
                    if family not in {target_family, first_off_topic}
                )
                items.append(
                    blueprint_memory(
                        slot_index=1,
                        role="distractor",
                        family=second_off_topic,
                        utility="irrelevant",
                        sensitive=source is MemorySource.ME,
                    )
                )
            else:
                if source in needed_by_regime[regime]:
                    items.append(
                        blueprint_memory(
                            slot_index=0,
                            role="helpful",
                            family=target_family,
                            utility="helpful",
                        )
                    )
                first_off_topic = distractor_assignments[case_field][source_field]
                items.append(
                    blueprint_memory(
                        slot_index=(1 if source in needed_by_regime[regime] else 0),
                        role="distractor",
                        family=first_off_topic,
                        utility="irrelevant",
                        sensitive=source is MemorySource.ME,
                    )
                )
                if source not in needed_by_regime[regime]:
                    second_off_topic = next(
                        family
                        for family in semantic_families
                        if family not in {target_family, first_off_topic}
                    )
                    items.append(
                        blueprint_memory(
                            slot_index=1,
                            role="distractor_secondary",
                            family=second_off_topic,
                            utility="irrelevant",
                            sensitive=source is MemorySource.ME,
                        )
                    )
            memory_pools[output_field_by_source[source]] = _shuffle_compiled_memories(
                items,
                user_id=user_id,
                case_field=case_field,
                source=source,
            )
        session_summary = surface.session_summary
        authorized_user_context = surface.authorized_user_context
        current_user_text = surface.current_user_text
        if regime is ResourceNeedRegime.MEMORY_HARMFUL:
            topic = GENERATION_FAMILY_TOPICS[target_family]
            visible_correction = (
                f"I used to avoid talking about {topic}, but this time I want "
                "to describe what happened instead of pushing it away."
            )
            current_user_text = (
                f"{current_user_text.rstrip('.')} {visible_correction}"
            )
            current_updates = (
                "The user explicitly says they previously avoided this topic but "
                "now want to describe the current event rather than push it away."
            )
            session_summary = f"{session_summary.rstrip('.')}. {current_updates}"
            authorized_user_context = (
                f"{authorized_user_context.rstrip('.')}. {current_updates}"
            )
        case_digest = sha256_text(
            f"{user_id}|{case_field}|{current_user_text}"
        )
        cases.append(
            GeneratedStateCase(
                case_id=f"case_{case_digest[:20]}",
                semantic_family=assignments[case_field],
                surface_form_id=f"sf_{case_digest[12:24]}",
                regime=regime,
                current_user_text=current_user_text,
                recent_dialogue=[
                    DialogueTurn.model_validate(turn.model_dump(mode="json"))
                    for turn in surface.dialogue_before_current
                ],
                session_summary=session_summary,
                session_index=session_index,
                profile_memories=memory_pools["profile_memories"],
                summary_memories=memory_pools["summary_memories"],
                event_memories=memory_pools["event_memories"],
                needed_memory_sources=needed_by_regime[regime],
                authorized_user_context=authorized_user_context,
                coverage_rationale=_deterministic_coverage_rationale(regime),
            )
        )
    bundle = GeneratedUserBundle(
        user_id=user_id,
        profile_summary=(
            "A privacy-safe synthetic user with longitudinal context about "
            + ", ".join(
                GENERATION_FAMILY_TOPICS[family] for family in semantic_families
            )
            + "."
        ),
        stable_preferences=[
            _blueprint_source_raw(family, MemorySource.MP)
            for family in semantic_families
        ],
        boundaries=[
            "does not want old assumptions treated as current facts",
            "prefers sensitive memories to be used only when clearly relevant",
        ],
        cases=cases,
        generator_seed_id=sha256_text(seed_dialogue)[:16],
        provenance={
            "generation_structure": DATA_GENERATION_CONTRACT_VERSION,
            "generation_draft_sha256": sha256_text(
                canonical_json(draft.model_dump(mode="json"))
            ),
            "case_family_assignments": assignments,
            "distractor_family_assignments": distractor_assignments,
            "deterministic_draft_lint": surface_selection[
                "compiled_surface_lint"
            ],
            "surface_selection": surface_selection,
            "session_design_offset": design_offset,
            "case_design_positions": design_positions,
            "provider_memory_slots_ignored": True,
            "provider_coverage_rationale_ignored": True,
            "provider_memory_slots_sha256": sha256_text(
                canonical_json(
                    {
                        case_field: {
                            source_field: getattr(
                                getattr(draft, case_field), source_field
                            ).model_dump(mode="json")
                            for source_field, _ in SOURCE_DRAFT_FIELDS
                        }
                        for case_field, _ in GENERATION_CASE_FIELDS
                    }
                )
            ),
            "evidence_blueprint_sha256": generation_evidence_blueprint_hash(),
            "memory_age_protocol": {
                "protocol": "v11-utility-blind-sha256-separated-rng-streams",
                "age_range_sessions": [1, 12],
                "age_seed_fields": ["user_id", "case_field", "source"],
                "row_order_seed_fields": ["user_id", "case_field", "source"],
                "excluded_seed_fields": [
                    "utility",
                    "needed_memory_sources",
                    "stale",
                    "conflict",
                ],
                "row_order_randomized_after_item_construction": True,
            },
            "multi_source_unique_contributions": {
                "MP": "A durable personal response preference.",
                "MS": "A recurring cross-session emotional pattern.",
                "ME": "A concrete prior incident at a distinct time.",
            },
        },
    )
    validate_bundle(bundle)
    return bundle


REGIME_INSTRUCTIONS: dict[ResourceNeedRegime, str] = {
    ResourceNeedRegime.CONTEXT_ONLY: (
        "The current turn must be fully understandable from recent dialogue. Long-term "
        "memory and the actual candidate strategy cards should add no clear marginal "
        "value beyond a capable base generator."
    ),
    ResourceNeedRegime.PROFILE_NEEDED: (
        "A stable preference, boundary, or personal constraint from profile memory is "
        "material to a good response. Other memories should be distractors."
    ),
    ResourceNeedRegime.SUMMARY_NEEDED: (
        "A cross-session pattern summarized in MS is material, while isolated events are "
        "insufficient or misleading."
    ),
    ResourceNeedRegime.EVENT_NEEDED: (
        "A concrete past event in ME is material to understanding the current turn. "
        "Generic summaries and strategy cards should not be necessary."
    ),
    ResourceNeedRegime.MULTI_SOURCE_NEEDED: (
        "At least two memory sources provide material, nonredundant marginal value. "
        "Do not claim that an acceptable reply is impossible without every source."
    ),
    ResourceNeedRegime.MEMORY_HARMFUL: (
        "Available memories must include stale, weakly related, or conflicting material. "
        "The safest response should avoid long-term memory."
    ),
    ResourceNeedRegime.STRATEGY_HELPFUL: (
        "The current state benefits from a matching ESC support-strategy card beyond "
        "the base generator. This may be exploration, reflection, reassurance, a "
        "question, or a suggestion; it is not synonymous with a plan."
    ),
    ResourceNeedRegime.STRATEGY_HARMFUL: (
        "Treat this legacy slot as advice_harmful: directive suggestions would be "
        "premature, while listening/reflection cards may still help. The surface must "
        "separate advice readiness from Strategy RAG marginal value."
    ),
    ResourceNeedRegime.AMBIGUOUS: (
        "Several low-cost actions should be genuinely competitive; do not create an "
        "obvious high-resource winner."
    ),
}


def generation_messages(
    *,
    seed_dialogue: str,
    user_id: str,
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
) -> list[dict[str, str]]:
    families, _ = _require_complete_generation_design(semantic_families, regimes)
    family_assignments = generation_case_family_assignments(families, regimes)
    family_anchor_text = "\n".join(
        f"- {family}: include at least one literal topic anchor such as "
        f"{', '.join(GENERATION_FAMILY_ANCHORS[family][:3])}"
        for family in families
    )
    regime_text = "\n".join(
        f"- {field_name}: only topic={family_assignments[field_name]}. "
        f"Surface design: {REGIME_INSTRUCTIONS[regime]}"
        for field_name, regime in GENERATION_CASE_FIELDS
    )
    system = """You create privacy-safe synthetic longitudinal emotional-support users for
training a pre-retrieval resource policy. Generate natural, diverse English. Do not copy
names or events from known benchmarks. Avoid clinical diagnosis, self-harm, acute crisis,
or treatment instructions. The output is development data, not a conversation answer."""
    user = f"""SEED DIALOGUE (style inspiration only; do not copy phrases or topics)
{seed_dialogue}

LOCAL USER BINDING (do not output this ID): {user_id}
SEMANTIC FAMILIES: {', '.join(families)}

Fill every named field in GeneratedBundleDraft. The compiler adds regime/source/utility
labels, varied prior-session ages, IDs, and the user binding. Your prose must make each
frozen slot semantically true. Never state an action code or the best policy.

MANDATORY FAMILY ANCHORS
{family_anchor_text}
For each case, current_user_text plus session_summary must mention its one target-family
anchor and must not mention either of the other two families. Keep the three families
independent: never explain one family through another family and never combine two
listed families in the same current turn, prior dialogue, or session summary.

NAMED CASES AND FROZEN FAMILY ASSIGNMENTS
{regime_text}

SURFACE AND TIME RULES
1. dialogue_before_current contains only turns that occurred before current_user_text.
   It alternates roles, ends with assistant, and never repeats current_user_text. Do not
   continue the conversation beyond the current turn.
2. Focus semantic effort on current_user_text, dialogue_before_current, session_summary,
   and authorized_user_context. coverage_rationale and memory source objects remain
   required only for backward-compatible strict JSON transport; the local compiler
   ignores provider-written rationale and every provider memory fragment, then
   deterministically constructs the evaluator rationale, MP/MS/ME evidence, and oracle
   labels. Fill those required placeholders concisely and never let them change a case's
   one-topic surface.
3. Fill multi_source unique-contribution strings concisely; the compiler replaces them
   with frozen source-specific contributions.
4. In strategy_helpful, create a turn where an appropriate ESC support card can add
   value beyond ordinary base empathy; an explicit request for steps is sufficient but
   not required. In the legacy strategy_harmful slot, current_user_text must ask to be
   heard without directive advice, while remaining compatible with a listening or
   reflection card. Advice readiness and Strategy RAG are separate.
5. In memory_harmful, the user's current correction must be explicit in
   current_user_text or prior visible dialogue. session_summary and authorized context
   may summarize visible facts but must never introduce a hidden update.
6. Use nine distinct, natural current-user turns. Never use meta-label phrases such as
   'one detail alone does not explain', 'helpful memory', 'strategy is needed',
   'context only is enough', or 'regime'. The three occurrences of each family
   should be different situations, not paraphrases.
7. Use privacy-safe ordinary life events. Output no checklist and no extra JSON keys.
Return strict JSON matching GeneratedBundleDraft."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generation_contract_hash() -> str:
    """Hash the exact data-generation prompt and response contract.

    Dynamic seed text, user IDs, and family assignments are bound separately by the
    run binding.  This digest changes when the shared instructions, regime wording,
    or strict response schema changes.
    """

    messages = generation_messages(
        seed_dialogue="<SEED_DIALOGUE>",
        user_id="<USER_ID>",
        semantic_families=[
            "relocation_loneliness",
            "workload_burnout",
            "friendship_distance",
        ],
        regimes=list(ResourceNeedRegime),
    )
    return sha256_text(
        canonical_json(
            {
                "version": DATA_GENERATION_CONTRACT_VERSION,
                "messages": messages,
                "response_schema": GeneratedBundleDraft.model_json_schema(),
                "temperature": GENERATION_TEMPERATURE,
                "max_output_tokens": GENERATION_MAX_OUTPUT_TOKENS,
            }
        )
    )


def bind_bundle_to_generation_run(
    bundle: GeneratedUserBundle, binding: dict[str, Any]
) -> None:
    """Attach a non-secret immutable run binding to a newly generated bundle."""

    digest = sha256_text(canonical_json(binding))
    bundle.provenance["generation_run_binding"] = binding
    bundle.provenance["generation_run_binding_sha256"] = digest


def require_bundle_generation_binding(
    bundle: GeneratedUserBundle, expected: dict[str, Any]
) -> None:
    """Reject legacy or mixed-run rows during resumable generation."""

    expected_digest = sha256_text(canonical_json(expected))
    actual = bundle.provenance.get("generation_run_binding")
    actual_digest = bundle.provenance.get("generation_run_binding_sha256")
    if not isinstance(actual, dict) or not isinstance(actual_digest, str):
        raise RuntimeError(
            f"bundle {bundle.user_id} predates immutable generation-run binding; "
            "use a new output directory or --overwrite"
        )
    if canonical_json(actual) != canonical_json(expected) or actual_digest != expected_digest:
        raise RuntimeError(
            f"bundle {bundle.user_id} belongs to a different generator/seed/config/"
            "prompt/code run; use a new output directory or --overwrite"
        )


def validate_successful_generation_trace(
    bundle: GeneratedUserBundle,
) -> dict[str, Any]:
    """Require reconstructable provider output and deterministic compilation."""

    provenance = bundle.provenance
    draft = provenance.get("provider_draft")
    response = provenance.get("provider_response")
    if not isinstance(draft, dict):
        raise RuntimeError(f"bundle {bundle.user_id} lacks its provider draft")
    if not isinstance(response, dict):
        raise RuntimeError(f"bundle {bundle.user_id} lacks its raw provider response")
    try:
        parsed_draft = GeneratedBundleDraft.model_validate(draft)
    except Exception as exc:
        raise RuntimeError(
            f"bundle {bundle.user_id} provider draft no longer matches schema: {exc}"
        ) from exc
    draft_sha256 = sha256_text(canonical_json(draft))
    if provenance.get("generation_draft_sha256") != draft_sha256:
        raise RuntimeError(f"bundle {bundle.user_id} provider draft hash mismatch")
    response_sha256 = sha256_text(canonical_json(response))
    if provenance.get("provider_response_sha256") != response_sha256:
        raise RuntimeError(f"bundle {bundle.user_id} provider response hash mismatch")
    lint = provenance.get("deterministic_draft_lint")
    selection = provenance.get("surface_selection")
    if not isinstance(lint, dict) or lint.get("status") != "PASS":
        raise RuntimeError(f"bundle {bundle.user_id} lacks a passing compiled lint")
    if not isinstance(selection, dict):
        raise RuntimeError(f"bundle {bundle.user_id} lacks surface-selection provenance")
    if provenance.get("provider_memory_slots_ignored") is not True:
        raise RuntimeError(f"bundle {bundle.user_id} trusts provider oracle evidence")
    if provenance.get("provider_coverage_rationale_ignored") is not True:
        raise RuntimeError(f"bundle {bundle.user_id} trusts provider oracle rationale")
    if provenance.get("evidence_blueprint_sha256") != generation_evidence_blueprint_hash():
        raise RuntimeError(f"bundle {bundle.user_id} evidence blueprint is stale")
    assignments = provenance.get("case_family_assignments")
    if not isinstance(assignments, dict) or set(assignments) != {
        field for field, _ in GENERATION_CASE_FIELDS
    }:
        raise RuntimeError(f"bundle {bundle.user_id} lacks frozen family assignments")
    ordered_families: list[str] = []
    for case_field, _ in GENERATION_CASE_FIELDS:
        family = str(assignments[case_field])
        if family not in ordered_families:
            ordered_families.append(family)
    selected_surfaces, rerun_selection = select_generation_surfaces(
        draft=parsed_draft,
        semantic_families=ordered_families,
        regimes=list(ResourceNeedRegime),
    )
    if canonical_json(rerun_selection) != canonical_json(selection):
        raise RuntimeError(f"bundle {bundle.user_id} stored surface selection is stale")
    if canonical_json(rerun_selection["compiled_surface_lint"]) != canonical_json(lint):
        raise RuntimeError(f"bundle {bundle.user_id} stored compiled lint is stale")
    recompiled = compile_generation_draft(
        draft=parsed_draft,
        seed_dialogue="<trace-replay-seed>",
        user_id=bundle.user_id,
        semantic_families=ordered_families,
        regimes=list(ResourceNeedRegime),
    )
    deterministic_fields = (
        "profile_summary",
        "stable_preferences",
        "boundaries",
        "cases",
    )
    recompiled_payload = recompiled.model_dump(mode="json")
    observed_payload = bundle.model_dump(mode="json")
    for field in deterministic_fields:
        if canonical_json(recompiled_payload[field]) != canonical_json(
            observed_payload[field]
        ):
            raise RuntimeError(
                f"bundle {bundle.user_id} deterministic compiler output drifted: {field}"
            )
    for (case_field, _), case in zip(GENERATION_CASE_FIELDS, bundle.cases, strict=True):
        expected_surface = selected_surfaces[case_field]
        observed_surface = {
            "current_user_text": case.current_user_text,
            "dialogue_before_current": [
                turn.model_dump(mode="json") for turn in case.recent_dialogue
            ],
            "session_summary": case.session_summary,
            "authorized_user_context": case.authorized_user_context,
            "coverage_rationale": case.coverage_rationale,
        }
        expected_payload = _surface_payload(expected_surface)
        expected_payload["coverage_rationale"] = _deterministic_coverage_rationale(
            case.regime
        )
        if case.regime is ResourceNeedRegime.MEMORY_HARMFUL:
            topic = GENERATION_FAMILY_TOPICS[case.semantic_family]
            expected_payload["current_user_text"] = (
                f"{expected_surface.current_user_text.rstrip('.')} I used to avoid "
                f"talking about {topic}, but this time I want to describe what "
                "happened instead of pushing it away."
            )
            summary_matches = case.session_summary.startswith(
                expected_surface.session_summary.rstrip(".")
                + ". The user explicitly says they previously avoided this topic"
            )
            context_matches = case.authorized_user_context.startswith(
                expected_surface.authorized_user_context.rstrip(".")
                + ". The user explicitly says they previously avoided this topic"
            )
            observed_surface.pop("session_summary")
            observed_surface.pop("authorized_user_context")
            expected_payload.pop("session_summary")
            expected_payload.pop("authorized_user_context")
        else:
            summary_matches = True
            context_matches = True
        if (
            canonical_json(observed_surface) != canonical_json(expected_payload)
            or not summary_matches
            or not context_matches
        ):
            raise RuntimeError(
                f"bundle {bundle.user_id} compiled surface drifted: {case_field}"
            )
    response_payload = None
    try:
        response_content = response["choices"][0]["message"]["content"]
        if isinstance(response_content, str):
            response_payload = json.loads(response_content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        response_payload = None
    if response_payload != draft:
        raise RuntimeError(
            f"bundle {bundle.user_id} raw provider response does not reconstruct draft"
        )
    if provenance.get("generation_structure") != DATA_GENERATION_CONTRACT_VERSION:
        raise RuntimeError(
            f"bundle {bundle.user_id} was not generated under the current contract"
        )
    return {
        "status": "PASS",
        "generation_structure": DATA_GENERATION_CONTRACT_VERSION,
        "provider_draft_sha256": draft_sha256,
        "provider_response_sha256": response_sha256,
        "provider_response_reconstructs_draft": True,
        "deterministic_bundle_recompiled": True,
        "deterministic_draft_lint_status": lint["status"],
        "provider_surface_fallback_case_count": int(
            selection["fallback_case_count"]
        ),
        "provider_oracle_evidence_used": False,
        "provider_oracle_rationale_used": False,
        "evidence_blueprint_sha256": generation_evidence_blueprint_hash(),
    }


def _hash_text(text: str, *, n_features: int = CATALOG_HASH_FEATURES) -> np.ndarray:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    return vectorizer.transform([text]).toarray()[0].astype(np.float64)


def build_deployable_catalog_statistics(
    *,
    texts: Sequence[str],
    created_sessions: Sequence[int],
    session_index: int,
    n_features: int = CATALOG_HASH_FEATURES,
) -> dict[str, Any]:
    """Construct the one deployable source-catalog representation used everywhere.

    Only source-level text fingerprint, count, age bounds, and estimated size are
    emitted.  Item-level relevance, generated stale flags, and current-state conflict
    labels are deliberately absent.  ``median_age_sessions`` is the midpoint of the
    deployable min/max bounds because the legacy RuntimeState catalog does not expose
    an item-age distribution.
    """

    if len(texts) != len(created_sessions):
        raise ValueError("catalog texts and created_sessions must have equal lengths")
    if session_index < 1:
        raise ValueError("session_index must be positive")
    future = [
        int(created)
        for created in created_sessions
        if int(created) >= int(session_index)
    ]
    if future:
        raise ValueError(
            "catalog contains memory from the current or a future session: "
            f"session_index={session_index}, created_sessions={future}"
        )
    if not texts:
        return {
            "available": False,
            "count": 0,
            "min_age_sessions": None,
            "median_age_sessions": None,
            "max_age_sessions": None,
            "estimated_tokens": 0,
            "catalog_fingerprint": [0.0] * n_features,
        }

    ages = [int(session_index) - int(created) for created in created_sessions]
    fingerprint = _hash_text("\n".join(texts), n_features=n_features)
    rounded = [round(float(value), 8) for value in fingerprint]
    min_age = min(ages)
    max_age = max(ages)
    return {
        "available": True,
        "count": len(texts),
        "min_age_sessions": min_age,
        "median_age_sessions": float(min_age + max_age) / 2.0,
        "max_age_sessions": max_age,
        "estimated_tokens": sum(estimate_tokens(text) for text in texts),
        "catalog_fingerprint": rounded,
    }


def source_catalog_similarity(
    query_text: str,
    catalog_fingerprint: Sequence[float],
    *,
    n_features: int = CATALOG_HASH_FEATURES,
) -> float:
    """Cosine similarity to the cached source-level catalog fingerprint."""

    fingerprint = np.asarray(catalog_fingerprint, dtype=np.float64)
    if fingerprint.size != n_features:
        raise ValueError(
            f"catalog fingerprint has {fingerprint.size} dimensions; expected {n_features}"
        )
    query = _hash_text(query_text, n_features=n_features)
    qnorm = float(np.linalg.norm(query))
    cnorm = float(np.linalg.norm(fingerprint))
    if qnorm == 0.0 or cnorm == 0.0:
        return 0.0
    return float(query @ fingerprint / (qnorm * cnorm))


def generate_user_bundle(
    *,
    endpoint: Endpoint,
    seed_dialogue: str,
    user_id: str,
    semantic_families: Sequence[str],
    regimes: Sequence[ResourceNeedRegime],
    seed: int,
    request_retries: int = 1,
) -> GeneratedUserBundle:
    if request_retries < 1:
        raise ValueError("request_retries must be positive")
    client = make_client(endpoint)
    try:
        messages = generation_messages(
            seed_dialogue=seed_dialogue,
            user_id=user_id,
            semantic_families=semantic_families,
            regimes=regimes,
        )
        call, draft = client.chat(
            messages,
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_OUTPUT_TOKENS,
            seed=seed,
            response_schema=GeneratedBundleDraft,
            retries=request_retries,
        )
        assert draft is not None
        try:
            bundle = compile_generation_draft(
                draft=draft,
                seed_dialogue=seed_dialogue,
                user_id=user_id,
                semantic_families=semantic_families,
                regimes=regimes,
            )
        except Exception as exc:
            raise GenerationDraftCompilationError(
                call=call,
                draft=draft,
                compilation_error=exc,
            ) from exc
        bundle.provenance.update(
            {
                "generator_model": endpoint.model,
                "generator_family": endpoint.family,
                "request_hash": call.request_hash,
                "messages_hash": sha256_text(canonical_json(messages)),
                "reported_usage": dict(call.usage),
                # Preserve the complete successful provider trace.  Earlier
                # pilots retained only a draft hash, which made semantic audit
                # and provider-output reconstruction impossible.
                "provider_draft": draft.model_dump(mode="json"),
                "provider_response": call.raw_response,
                "provider_response_sha256": sha256_text(
                    canonical_json(call.raw_response)
                ),
            }
        )
        return bundle
    finally:
        client.close()


def _catalog_summary(
    *,
    source: MemorySource,
    memories: Sequence[GeneratedMemory],
    query_text: str,
    session_index: int,
    n_features: int = CATALOG_HASH_FEATURES,
) -> ObservableSourceSummary:
    statistics = build_deployable_catalog_statistics(
        texts=[memory.text for memory in memories],
        created_sessions=[memory.created_session for memory in memories],
        session_index=session_index,
        n_features=n_features,
    )
    similarity = source_catalog_similarity(
        query_text,
        statistics["catalog_fingerprint"],
        n_features=n_features,
    )
    return ObservableSourceSummary(
        available=bool(statistics["available"]),
        count=int(statistics["count"]),
        min_age_sessions=statistics["min_age_sessions"],
        median_age_sessions=statistics["median_age_sessions"],
        max_age_sessions=statistics["max_age_sessions"],
        estimated_tokens=int(statistics["estimated_tokens"]),
        query_similarity_mean=similarity,
        catalog_embedding=list(statistics["catalog_fingerprint"]),
    )


def case_to_state(
    *,
    user_id: str,
    case: GeneratedStateCase,
    split: PMV2Split,
    strategy_catalog_count: int,
    strategy_estimated_tokens: int,
    bundle_provenance: dict[str, Any] | None = None,
) -> PMV2State:
    query = "\n".join(
        [case.current_user_text, case.session_summary]
        + [turn.content for turn in case.recent_dialogue]
    )
    memories = {
        MemorySource.MP: case.profile_memories,
        MemorySource.MS: case.summary_memories,
        MemorySource.ME: case.event_memories,
    }
    inventory = {
        source: _catalog_summary(
            source=source,
            memories=items,
            query_text=query,
            session_index=case.session_index,
        )
        for source, items in memories.items()
    }
    available = {source for source, summary in inventory.items() if summary.available}
    allowed_actions = sorted(
        {
            canonical_action_id(sources, strategy)
            for sources in ACTION_MEMORY_MAP.values()
            if sources <= available
            for strategy in StrategyMode
        }
    )
    state_id = "state_" + sha256_text(f"{user_id}|{case.case_id}|{case.surface_form_id}")[:24]
    card_id = "card_" + sha256_text(f"{state_id}|pm-v2-backend")[:24]
    evaluator_context_id = "eval_" + sha256_text(state_id)[:24]
    generation_lineage = None
    if bundle_provenance:
        generation_lineage = bundle_provenance.get("generation_run_binding_sha256")
        if not generation_lineage:
            safe_hash_inputs = {
                key: bundle_provenance[key]
                for key in ("generator_model", "generator_family", "request_hash", "messages_hash")
                if key in bundle_provenance
            }
            if safe_hash_inputs:
                generation_lineage = sha256_text(canonical_json(safe_hash_inputs))
    provenance = {
        "backend_record_id": card_id,
        "evaluator_context_id": evaluator_context_id,
    }
    if generation_lineage:
        provenance["data_generation_sha256"] = str(generation_lineage)
    return PMV2State(
        state_id=state_id,
        card_id=card_id,
        user_id=user_id,
        split=split,
        semantic_family=case.semantic_family,
        surface_form_id=case.surface_form_id,
        current_user_text=case.current_user_text,
        current_session_history=case.recent_dialogue,
        current_session_summary=case.session_summary,
        session_index=case.session_index,
        inventory=inventory,
        strategy_catalog_count=strategy_catalog_count,
        strategy_estimated_tokens=strategy_estimated_tokens,
        allowed_actions=allowed_actions,
        provenance=provenance,
    )


def validate_bundle(bundle: GeneratedUserBundle) -> dict[str, Any]:
    regimes = Counter(case.regime.value for case in bundle.cases)
    current_texts = [normalize_text(case.current_user_text) for case in bundle.cases]
    memory_ages: list[int] = []
    if len(current_texts) != len(set(current_texts)):
        raise ValueError(f"bundle {bundle.user_id} repeats normalized current-user text")
    required = {
        ResourceNeedRegime.CONTEXT_ONLY.value,
        ResourceNeedRegime.EVENT_NEEDED.value,
        ResourceNeedRegime.STRATEGY_HELPFUL.value,
        ResourceNeedRegime.STRATEGY_HARMFUL.value,
    }
    missing = required - set(regimes)
    if missing:
        raise ValueError(f"bundle {bundle.user_id} missing required regimes: {sorted(missing)}")
    for case in bundle.cases:
        if not case.profile_memories and not case.summary_memories and not case.event_memories:
            raise ValueError(f"case {case.case_id} has no inventory variation")
        # Re-run the structured need contract at bundle validation time so any
        # future construction path that bypasses normal field validation still
        # fails before data are written.
        validated_case = GeneratedStateCase.model_validate(case.model_dump())
        if validated_case.needed_memory_sources != case.needed_memory_sources:
            raise ValueError(
                f"case {case.case_id} changed needed_memory_sources on validation"
            )
        memory_ages.extend(
            case.session_index - item.created_session
            for pool in (
                case.profile_memories,
                case.summary_memories,
                case.event_memories,
            )
            for item in pool
        )
    return {
        "user_id": bundle.user_id,
        "case_count": len(bundle.cases),
        "unique_normalized_current_user_text_count": len(set(current_texts)),
        "normalized_current_user_text_unique_rate": (
            len(set(current_texts)) / len(current_texts)
        ),
        "regime_distribution": dict(regimes),
        "semantic_families": sorted({case.semantic_family for case in bundle.cases}),
        "needed_memory_sources": {
            case.case_id: [source.value for source in case.needed_memory_sources]
            for case in bundle.cases
        },
        "memory_age_sessions": {
            "minimum": min(memory_ages),
            "maximum": max(memory_ages),
            "unique": sorted(set(memory_ages)),
            "unique_count": len(set(memory_ages)),
        },
    }


def validate_split_manifests(split_states: dict[PMV2Split, Sequence[PMV2State]]) -> SplitManifest:
    train = split_states.get(PMV2Split.TRAIN, [])
    calibration = split_states.get(PMV2Split.CALIBRATION, [])
    test = split_states.get(PMV2Split.INTERNAL_TEST, [])
    user_sets = [{state.user_id for state in rows} for rows in (train, calibration, test)]
    family_sets = [{state.semantic_family for state in rows} for rows in (train, calibration, test)]
    text_sets = [{normalize_text(state.current_user_text) for state in rows} for rows in (train, calibration, test)]
    user_overlap = sum(len(user_sets[i] & user_sets[j]) for i in range(3) for j in range(i + 1, 3))
    family_overlap = sum(
        len(family_sets[i] & family_sets[j]) for i in range(3) for j in range(i + 1, 3)
    )
    text_overlap = sum(len(text_sets[i] & text_sets[j]) for i in range(3) for j in range(i + 1, 3))
    all_states = [*train, *calibration, *test]
    normalized_texts = [
        normalize_text(state.current_user_text) for state in all_states
    ]
    unique_normalized_texts = len(set(normalized_texts))
    return SplitManifest(
        train_users=sorted(user_sets[0]),
        calibration_users=sorted(user_sets[1]),
        internal_test_users=sorted(user_sets[2]),
        train_semantic_families=sorted(family_sets[0]),
        calibration_semantic_families=sorted(family_sets[1]),
        internal_test_semantic_families=sorted(family_sets[2]),
        normalized_text_overlap=text_overlap,
        user_overlap=user_overlap,
        semantic_family_overlap=family_overlap,
        total_states=len(all_states),
        unique_normalized_current_user_texts=unique_normalized_texts,
        normalized_current_user_text_unique_rate=(
            unique_normalized_texts / len(all_states) if all_states else 0.0
        ),
    )


def audit_cross_split_near_duplicates(
    split_states: dict[PMV2Split, Sequence[PMV2State]],
    *,
    maximum_word_hash_cosine: float,
    maximum_char_hash_cosine: float,
    report_top_pairs: int = 50,
) -> dict[str, Any]:
    """Fail closed on highly similar current-user texts across frozen splits.

    Exact normalized duplicates are already prohibited by ``SplitManifest``.
    This second, deterministic no-API audit uses independent word-(1,2)-gram
    and character-(3,5)-gram hashing views.  It reports nearest cross-split
    pairs and rejects a pair when either preregistered cosine ceiling is
    exceeded.  Hashing is fixed and fit-free, so the audit itself cannot leak
    held-out vocabulary into PM features.
    """

    if not 0.0 < maximum_word_hash_cosine < 1.0:
        raise ValueError("maximum_word_hash_cosine must be in (0, 1)")
    if not 0.0 < maximum_char_hash_cosine < 1.0:
        raise ValueError("maximum_char_hash_cosine must be in (0, 1)")
    if report_top_pairs < 1:
        raise ValueError("report_top_pairs must be positive")

    rows: list[tuple[PMV2Split, PMV2State]] = []
    for split in (
        PMV2Split.TRAIN,
        PMV2Split.CALIBRATION,
        PMV2Split.INTERNAL_TEST,
    ):
        rows.extend((split, state) for state in split_states.get(split, []))
    if not rows:
        raise ValueError("near-duplicate audit requires non-empty PM-v2 states")
    texts = [normalize_text(state.current_user_text) for _, state in rows]
    word = HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
    ).transform(texts)
    char = HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
    ).transform(texts)

    comparisons: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for left in range(len(rows)):
        left_split, left_state = rows[left]
        for right in range(left + 1, len(rows)):
            right_split, right_state = rows[right]
            if left_split == right_split:
                continue
            word_cosine = float(word[left].multiply(word[right]).sum())
            char_cosine = float(char[left].multiply(char[right]).sum())
            row = {
                "left_state_id": left_state.state_id,
                "left_split": left_split.value,
                "right_state_id": right_state.state_id,
                "right_split": right_split.value,
                "word_hash_cosine": word_cosine,
                "char_hash_cosine": char_cosine,
                "similarity_ratio_to_limit": max(
                    word_cosine / maximum_word_hash_cosine,
                    char_cosine / maximum_char_hash_cosine,
                ),
            }
            comparisons.append(row)
            if (
                word_cosine > maximum_word_hash_cosine
                or char_cosine > maximum_char_hash_cosine
            ):
                flagged.append(row)
    nearest = sorted(
        comparisons,
        key=lambda row: (
            -float(row["similarity_ratio_to_limit"]),
            str(row["left_state_id"]),
            str(row["right_state_id"]),
        ),
    )[:report_top_pairs]
    report = {
        "status": "PASS" if not flagged else "FAIL",
        "method": "fixed_hash_word_1_2_and_char_3_5_cosine",
        "thresholds": {
            "maximum_word_hash_cosine": float(maximum_word_hash_cosine),
            "maximum_char_hash_cosine": float(maximum_char_hash_cosine),
        },
        "state_count": len(rows),
        "cross_split_pair_count": len(comparisons),
        "flagged_pair_count": len(flagged),
        "nearest_cross_split_pairs": nearest,
        "flagged_pairs": sorted(
            flagged,
            key=lambda row: -float(row["similarity_ratio_to_limit"]),
        )[:report_top_pairs],
    }
    if flagged:
        raise ValueError(
            "PM-v2 cross-split near-duplicate audit failed: "
            f"{report['flagged_pairs']}"
        )
    return report


def state_to_v1_runtime(state: PMV2State) -> RuntimeState:
    inventory = {}
    for source, summary in state.inventory.items():
        fp = list(summary.catalog_embedding)
        if len(fp) < 64:
            fp = fp + [0.0] * (64 - len(fp))
        elif len(fp) > 64:
            fp = fp[:64]
        inventory[source] = SourceCatalog(
            available=summary.available,
            count=summary.count,
            min_age_sessions=summary.min_age_sessions,
            max_age_sessions=summary.max_age_sessions,
            estimated_tokens=summary.estimated_tokens,
            catalog_fingerprint=fp,
        )
    split_map = {
        PMV2Split.TRAIN: "train",
        PMV2Split.CALIBRATION: "validation",
        PMV2Split.INTERNAL_TEST: "development",
        PMV2Split.EXTERNAL_TEST: "evoemo_test",
    }
    return RuntimeState(
        state_id=state.state_id,
        card_id=state.card_id,
        user_id=state.user_id,
        split=split_map[state.split],
        semantic_family=state.semantic_family,
        current_user_text=state.current_user_text,
        current_session_history=state.current_session_history,
        current_session_summary=state.current_session_summary,
        session_index=state.session_index,
        inventory=inventory,
        allowed_actions=state.allowed_actions,
        provenance={
            "pm_v2_state_id": state.state_id,
            "surface_form_id": state.surface_form_id,
            "pm_v2_provenance_sha256": sha256_text(canonical_json(state.provenance)),
        },
    )


def _case_memories(case: GeneratedStateCase) -> dict[MemorySource, list[GeneratedMemory]]:
    return {
        MemorySource.MP: case.profile_memories,
        MemorySource.MS: case.summary_memories,
        MemorySource.ME: case.event_memories,
    }


def case_to_memory_backend(
    state: PMV2State, case: GeneratedStateCase
) -> MemoryBackendRecord:
    """Build the retrieval backend without placing memory text in the PM state."""

    items: list[MemoryItem] = []
    for source, memories in _case_memories(case).items():
        for index, memory in enumerate(memories):
            if memory.created_session >= case.session_index:
                raise ValueError(
                    f"memory {memory.memory_id} is not prior to case {case.case_id}"
                )
            memory_id = "mem_" + sha256_text(
                f"{state.card_id}|{source.value}|{index}|{memory.text}"
            )[:24]
            items.append(
                MemoryItem(
                    memory_id=memory_id,
                    source=source,
                    created_session=memory.created_session,
                    text=memory.text,
                )
            )
    return MemoryBackendRecord(card_id=state.card_id, items=items)


def case_to_evaluator_context(
    state: PMV2State, case: GeneratedStateCase
) -> dict[str, Any]:
    """Return evaluator-only context and annotations in a physically separate row."""

    annotations = []
    for source, memories in _case_memories(case).items():
        for index, memory in enumerate(memories):
            memory_id = "mem_" + sha256_text(
                f"{state.card_id}|{source.value}|{index}|{memory.text}"
            )[:24]
            annotations.append(
                {
                    "memory_id": memory_id,
                    "source": source.value,
                    "created_session": memory.created_session,
                    "stale": memory.stale,
                    "conflicts_with_current_state": memory.conflicts_with_current_state,
                    "private_sensitivity": memory.private_sensitivity,
                    "item_utility": memory.item_utility,
                }
            )
    row = {
        "evaluator_context_id": state.provenance["evaluator_context_id"],
        "state_id": state.state_id,
        "card_id": state.card_id,
        "regime": case.regime.value,
        "needed_memory_sources": [
            source.value for source in case.needed_memory_sources
        ],
        "authorized_user_context": case.authorized_user_context,
        "coverage_rationale": case.coverage_rationale,
        "memory_annotations": annotations,
    }
    row["context_payload_sha256"] = evaluator_context_payload_sha256(row)
    return row


def validate_generation_shortcut_controls(
    *,
    bundles: Sequence[GeneratedUserBundle],
    split_by_user: dict[str, PMV2Split],
) -> dict[str, Any]:
    """Fail if observable metadata or IDs encode a fixed regime shortcut."""

    positions = len(GENERATION_CASE_FIELDS)
    expected_session_indices = {20 + 3 * position for position in range(positions)}
    regime_case_index = {
        regime: index for index, (_, regime) in enumerate(GENERATION_CASE_FIELDS)
    }
    case_field_by_regime = {
        regime: case_field for case_field, regime in GENERATION_CASE_FIELDS
    }
    position_counts: dict[str, dict[str, Counter[int]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    user_counts: Counter[str] = Counter()
    for bundle in bundles:
        split = split_by_user[bundle.user_id]
        split_name = split.value
        user_counts[split_name] += 1
        offset = _generation_user_design_offset(bundle.user_id)
        observed_indices = {case.session_index for case in bundle.cases}
        if observed_indices != expected_session_indices:
            raise ValueError(
                f"bundle {bundle.user_id} does not expose one neutral session position "
                f"per case: {sorted(observed_indices)}"
            )
        for case in bundle.cases:
            if any(regime.value in case.case_id for regime in ResourceNeedRegime):
                raise ValueError(
                    f"case_id leaks its evaluator-only regime: {case.case_id}"
                )
            expected_position = (regime_case_index[case.regime] + offset) % positions
            if case.session_index != 20 + 3 * expected_position:
                raise ValueError(
                    f"case {case.case_id} session_index is not counterbalanced"
                )
            position_counts[split_name][case.regime.value][expected_position] += 1
            for source, memories in _case_memories(case).items():
                observed_ages = sorted(
                    case.session_index - memory.created_session for memory in memories
                )
                expected_ages = sorted(
                    _memory_age_sessions(
                        user_id=bundle.user_id,
                        case_field=case_field_by_regime[case.regime],
                        source=source,
                        slot_index=slot_index,
                    )
                    for slot_index in (0, 1)
                )
                if observed_ages != expected_ages:
                    raise ValueError(
                        f"case {case.case_id} {source.value} ages reveal item utility: "
                        f"observed={observed_ages}, expected={expected_ages}"
                    )

    report_splits: dict[str, Any] = {}
    for split_name, regime_rows in position_counts.items():
        split_users = int(user_counts[split_name])
        report_splits[split_name] = {}
        for regime in ResourceNeedRegime:
            counts = regime_rows[regime.value]
            dense = [int(counts[position]) for position in range(positions)]
            if split_users >= positions and (
                any(count == 0 for count in dense) or max(dense) - min(dense) > 1
            ):
                raise ValueError(
                    f"split {split_name} regime {regime.value} has an unbalanced "
                    f"session-position shortcut: {dense}"
                )
            report_splits[split_name][regime.value] = dense
    return {
        "status": "PASS",
        "protocol": (
            "pm-v2-generation-shortcut-controls-v1-neutral-id-session-age-"
            "counterbalance"
        ),
        "session_indices": sorted(expected_session_indices),
        "splits": report_splits,
        "case_ids_hide_regime": True,
        "memory_age_multiset_ignores_utility": True,
        "memory_row_order_uses_independent_seed": True,
    }


def write_development_dataset(
    *,
    bundles: Sequence[GeneratedUserBundle],
    split_by_user: dict[str, PMV2Split],
    out_dir: str | Path,
    strategy_catalog_count: int,
    strategy_estimated_tokens: int,
    strategy_top_k: int,
    strategy_bank_sha256: str,
    expected_semantic_families_by_split: dict[
        PMV2Split, Sequence[str]
    ] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if strategy_catalog_count < 1 or strategy_top_k < 1:
        raise ValueError("development data requires a non-empty strategy catalog")
    if strategy_estimated_tokens < 0:
        raise ValueError("strategy_estimated_tokens must be non-negative")
    if len(strategy_bank_sha256) != 64:
        raise ValueError("strategy_bank_sha256 must be a SHA-256 hex digest")
    out_dir.mkdir(parents=True, exist_ok=True)
    states_by_split: dict[PMV2Split, list[PMV2State]] = {
        split: [] for split in (PMV2Split.TRAIN, PMV2Split.CALIBRATION, PMV2Split.INTERNAL_TEST)
    }
    private_case_by_state: dict[str, GeneratedStateCase] = {}
    bundle_reports = []
    for bundle in bundles:
        bundle_reports.append(validate_bundle(bundle))
        try:
            split = split_by_user[bundle.user_id]
        except KeyError as exc:
            raise ValueError(f"no split assigned for user {bundle.user_id}") from exc
        for case in bundle.cases:
            state = case_to_state(
                user_id=bundle.user_id,
                case=case,
                split=split,
                strategy_catalog_count=strategy_catalog_count,
                strategy_estimated_tokens=strategy_estimated_tokens,
                bundle_provenance=bundle.provenance,
            )
            if state.state_id in private_case_by_state:
                raise ValueError(f"duplicate generated PM-v2 state_id: {state.state_id}")
            states_by_split[split].append(state)
            private_case_by_state[state.state_id] = case
    manifest = validate_split_manifests(states_by_split)
    semantic_family_coverage: dict[str, Any] = {
        "status": "NOT_PREREGISTERED",
        "splits": {},
    }
    if expected_semantic_families_by_split is not None:
        coverage_failures: dict[str, Any] = {}
        coverage_rows: dict[str, Any] = {}
        for split in (
            PMV2Split.TRAIN,
            PMV2Split.CALIBRATION,
            PMV2Split.INTERNAL_TEST,
        ):
            expected = set(expected_semantic_families_by_split.get(split, []))
            observed = {state.semantic_family for state in states_by_split[split]}
            coverage_rows[split.value] = {
                "expected_count": len(expected),
                "observed_count": len(observed),
                "expected": sorted(expected),
                "observed": sorted(observed),
            }
            if observed != expected:
                coverage_failures[split.value] = {
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                }
        semantic_family_coverage = {
            "status": "PASS" if not coverage_failures else "FAIL",
            "splits": coverage_rows,
            "failures": coverage_failures,
        }
        if coverage_failures:
            raise ValueError(
                "generated PM-v2 semantic-family union does not match the frozen "
                f"split assignment: {coverage_failures}"
            )
    all_states = [state for rows in states_by_split.values() for state in rows]
    state_path = out_dir / "pm_v2_states.jsonl"
    runtime_path = out_dir / "runtime_states.jsonl"
    backend_path = out_dir / "memory_backend.jsonl"
    evaluator_path = out_dir / "evaluator_contexts.jsonl"
    for path in (state_path, runtime_path, backend_path, evaluator_path):
        path.write_text("", encoding="utf-8")
    for state in all_states:
        case = private_case_by_state[state.state_id]
        append_jsonl(state_path, state.model_dump(mode="json"))
        append_jsonl(runtime_path, state_to_v1_runtime(state).model_dump(mode="json"))
        append_jsonl(
            backend_path,
            case_to_memory_backend(state, case).model_dump(mode="json"),
        )
        append_jsonl(evaluator_path, case_to_evaluator_context(state, case))
    evaluator_index = load_evaluator_context_index(
        evaluator_path,
        states=all_states,
        require_exact=True,
    )
    bundle_path = out_dir / "pm_v2_bundles.jsonl"
    bundle_path.write_text("", encoding="utf-8")
    for bundle in bundles:
        append_jsonl(bundle_path, bundle.model_dump(mode="json"))
    shortcut_controls = validate_generation_shortcut_controls(
        bundles=bundles,
        split_by_user=split_by_user,
    )
    report = {
        "status": "COMPLETE",
        "n_users": len(bundles),
        "n_states": len(all_states),
        "split_counts": {
            split.value: len(rows) for split, rows in states_by_split.items()
        },
        "split_manifest": manifest.model_dump(mode="json"),
        "semantic_family_coverage": semantic_family_coverage,
        "generation_shortcut_controls": shortcut_controls,
        "bundle_reports": bundle_reports,
        "states_path": str(state_path),
        "runtime_path": str(runtime_path),
        "memory_backend_path": str(backend_path),
        "evaluator_contexts_path": str(evaluator_path),
        "evaluator_contexts_sha256": evaluator_index.source_sha256,
        "evaluator_contexts_map_sha256": evaluator_index.map_sha256,
        "bundles_path": str(bundle_path),
        "model_visible_state_contains_private_context": False,
        "strategy_catalog": {
            "count": strategy_catalog_count,
            "estimated_action_tokens": strategy_estimated_tokens,
            "top_k": strategy_top_k,
            "strategy_bank_sha256": strategy_bank_sha256,
        },
    }
    write_json(out_dir / "pm_v2_data_report.json", report)
    return report


def load_states(path: str | Path) -> list[PMV2State]:
    states: list[PMV2State] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                # Strict models should still accept their canonical JSON wire
                # representation (for example ``split: \"train\"``). Parsing the
                # decoded Python dict would incorrectly require an Enum instance.
                states.append(PMV2State.model_validate_json(line))
    return states


def load_bundles(path: str | Path) -> list[GeneratedUserBundle]:
    bundles: list[GeneratedUserBundle] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                bundles.append(GeneratedUserBundle.model_validate_json(line))
    return bundles


def runtime_to_pmv2_state(
    state: RuntimeState,
    *,
    split: PMV2Split = PMV2Split.EXTERNAL_TEST,
    strategy_catalog_count: int = 0,
    strategy_estimated_tokens: int = 240,
) -> PMV2State:
    query = "\n".join(
        [state.current_user_text, state.current_session_summary]
        + [turn.content for turn in state.current_session_history]
    )
    inventory: dict[MemorySource, ObservableSourceSummary] = {}
    for source, cat in state.inventory.items():
        fingerprint = list(cat.catalog_fingerprint)
        if not fingerprint and not cat.available:
            fingerprint = [0.0] * CATALOG_HASH_FEATURES
        similarity = source_catalog_similarity(query, fingerprint)
        inventory[source] = ObservableSourceSummary(
            available=cat.available,
            count=cat.count,
            min_age_sessions=cat.min_age_sessions,
            median_age_sessions=(
                float(cat.min_age_sessions + cat.max_age_sessions) / 2.0
                if cat.min_age_sessions is not None and cat.max_age_sessions is not None
                else None
            ),
            max_age_sessions=cat.max_age_sessions,
            estimated_tokens=cat.estimated_tokens,
            query_similarity_mean=similarity,
            catalog_embedding=[float(value) for value in fingerprint],
        )
    return PMV2State(
        state_id=state.state_id,
        card_id=state.card_id,
        user_id=state.user_id,
        split=split,
        semantic_family=state.semantic_family,
        surface_form_id=str(state.provenance.get("surface_form_id", state.state_id)),
        current_user_text=state.current_user_text,
        current_session_history=state.current_session_history,
        current_session_summary=state.current_session_summary,
        session_index=state.session_index,
        inventory=inventory,
        strategy_catalog_count=strategy_catalog_count,
        strategy_estimated_tokens=strategy_estimated_tokens,
        allowed_actions=state.allowed_actions,
        provenance={
            "adapted_from_runtime_state": True,
            "runtime_provenance_sha256": sha256_text(
                canonical_json(state.provenance)
            ),
        },
    )
