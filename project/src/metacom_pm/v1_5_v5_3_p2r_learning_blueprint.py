"""P2R learning-blueprint schema and outcome-blind materialization helpers.

This module is the method-level replacement for the W7/W7R flat templates.
It deliberately separates four objects that must not be conflated:

1. a same-user longitudinal catalog (strictly past, source typed);
2. a current-state surface written without copying the hidden answer;
3. one frozen Rank-1 candidate snapshot shared by every treatment arm;
4. candidate-aware Step1 features computed only from runtime-observable data.

No worth-opening label, generated reply, quality preference, grounding-risk
outcome, or policy action is accepted by these schemas.  Those labels are
created later from paired ON/OFF effects under the frozen Step2 stack.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field, field_validator, model_validator

from .contracts import (
    DialogueTurn,
    MemoryItem,
    MemorySource,
    StrategyCard,
    StrictModel,
)
from .io import canonical_json, sha256_text, stable_hex
from .text import lexical_score, normalize_space
from .v1_5_candidate_discovery import (
    discover_final_typed_memory_candidates,
    final_typed_content_match_level,
    final_typed_content_words,
)
from .v1_5_v5_2_atomic_memory import compile_atomic_reusable_outcome
from .v1_5_v5_3_contribution_slot_features import (
    me_contribution_slots,
    mp_contribution_slots,
    ms_contribution_slots,
    rs_contribution_slots,
)
from .v1_5_v5_3_candidate_layer_responsibility import (
    rs_mechanical_candidate_pool,
    rs_shared_candidate_top1,
)


P2R_PROTOCOL = "pm-v1.5-v5.3-p2r-learning-blueprint-v1"
CATALOG_PROTOCOL = "pm-v1.5-v5.3-p2r-longitudinal-catalog-v1"

Component = Literal["MP", "MS", "ME", "RS"]
MemoryComponent = Literal["MP", "MS", "ME"]
StateCondition = Literal[
    "positive_opportunity",
    "current_redundant",
    "goal_mismatch",
    "explicit_decline",
    "unknown_readiness",
    "resolved_or_stale",
    "wrong_entity",
    "already_executed",
    "no_candidate",
]


class P2RLongitudinalCatalogItem(StrictModel):
    """One immutable item in a user's strictly longitudinal catalog.

    ``catalog_user_id`` is the user whose backend owns the item;
    ``subject_entity_id`` is the person/entity described by the text.  Keeping
    both prevents the old owner/entity conflation (a user's memory about a
    friend is not automatically a fact about the user).
    """

    protocol: Literal["pm-v1.5-v5.3-p2r-longitudinal-catalog-v1"] = (
        CATALOG_PROTOCOL
    )
    catalog_user_id: str = Field(min_length=1)
    memory_id: str
    component: MemoryComponent
    subtype: str = Field(min_length=1)
    created_session: int = Field(ge=0)
    valid_from_session: int = Field(ge=0)
    valid_until_session: int | None = Field(default=None, ge=0)
    candidate_version: int = Field(default=1, ge=1)
    subject_entity_id: str = Field(min_length=1)
    semantic_family: str = Field(min_length=1)
    topic_key: str = Field(min_length=1)
    literal_text: str = Field(min_length=1)
    source_session_ref: str = Field(min_length=1)
    field_type: str | None = None
    field_value: str | None = None
    action_span: str | None = None
    outcome_span: str | None = None
    catalog_role: Literal[
        "target_capable",
        "same_topic_competitor",
        "cross_topic_distractor",
        "resolved_or_stale",
        "wrong_entity",
    ]

    @field_validator("memory_id")
    @classmethod
    def opaque_memory_id(cls, value: str) -> str:
        if not value.startswith("mem_") or len(value) < 16:
            raise ValueError("memory_id must be an opaque mem_ identifier")
        return value

    @model_validator(mode="after")
    def coherent_time_and_component_payload(self):
        if self.valid_from_session < self.created_session:
            raise ValueError("valid_from_session cannot precede created_session")
        if (
            self.valid_until_session is not None
            and self.valid_until_session < self.valid_from_session
        ):
            raise ValueError("valid_until_session precedes valid_from_session")
        if self.component == "MP" and (self.field_type is None or self.field_value is None):
            raise ValueError("MP catalog items require field_type and field_value")
        if self.component == "ME" and self.subtype == "ME_REUSABLE_OUTCOME":
            if self.action_span is None or self.outcome_span is None:
                raise ValueError(
                    "ME_REUSABLE_OUTCOME requires explicit action_span and outcome_span"
                )
            compiled = compile_atomic_reusable_outcome(self.literal_text)
            if compiled is None:
                raise ValueError("ME_REUSABLE_OUTCOME is rejected by the frozen compiler")
            if self.action_span not in self.literal_text or self.outcome_span not in self.literal_text:
                raise ValueError("ME action/outcome spans must be literal evidence substrings")
        if (
            self.component == "MP"
            and self.subtype == "MP_PROFILE"
            and self.field_value not in self.literal_text
        ):
            raise ValueError("MP field_value must be a literal part of the stored item")
        return self

    def as_memory_item(self) -> MemoryItem:
        return MemoryItem(
            memory_id=self.memory_id,
            source=MemorySource(self.component),
            created_session=self.created_session,
            text=self.literal_text,
        )

    def is_active_strict_past(self, current_session_index: int) -> bool:
        return (
            self.created_session < current_session_index
            and self.valid_from_session <= current_session_index
            and (
                self.valid_until_session is None
                or current_session_index <= self.valid_until_session
            )
        )


class P2RCurrentStateSpec(StrictModel):
    """A current state written independently from the hidden catalog answer."""

    protocol: Literal["pm-v1.5-v5.3-p2r-learning-blueprint-v1"] = P2R_PROTOCOL
    state_id: str = Field(pattern=r"^state_[0-9a-f]{12,64}$")
    catalog_user_id: str = Field(min_length=1)
    current_subject_entity_id: str = Field(min_length=1)
    component: Component
    semantic_family: str = Field(min_length=1)
    counterfactual_group_id: str = Field(pattern=r"^group_[0-9a-f]{12,64}$")
    state_condition: StateCondition
    current_session_index: int = Field(gt=0)
    visible_dialogue: list[DialogueTurn] = Field(min_length=1)
    already_executed_move_ids: list[str] = Field(default_factory=list)
    intended_response_act: str = Field(min_length=1)
    construction_note: str = Field(min_length=1)
    interaction_key: str | None = None
    interaction_variant: str | None = None

    @model_validator(mode="after")
    def last_turn_is_current_user(self):
        if self.visible_dialogue[-1].role != "user":
            raise ValueError("visible_dialogue must end with the current user turn")
        return self

    @property
    def current_user_text(self) -> str:
        return self.visible_dialogue[-1].content


class P2RCandidateLineage(StrictModel):
    candidate_id: str | None
    component: Component
    candidate_version: int | None
    created_session: int | None
    candidate_surface_sha256: str | None
    retrieval_method: str
    rank1_compiler_valid: bool | None = None


class P2RModelInput(StrictModel):
    """The only state object a fitted Step1 head is allowed to consume."""

    component: Component
    candidate_present: bool
    contribution_slots: dict[str, Any]

    @model_validator(mode="after")
    def no_gold_or_identity_fields(self):
        forbidden = {
            "label",
            "gold",
            "worth_opening",
            "quality",
            "risk",
            "outcome",
            "candidate_id",
            "intended_target_id",
            "state_condition",
            "construction_condition",
        }
        present = forbidden & set(self.contribution_slots)
        if present:
            raise ValueError(f"Step1 model input contains forbidden fields: {sorted(present)}")
        return self


class P2RBlueprintRow(StrictModel):
    protocol: Literal["pm-v1.5-v5.3-p2r-learning-blueprint-v1"] = P2R_PROTOCOL
    state: P2RCurrentStateSpec
    split_group_key: str = Field(min_length=1)
    proposed_split: Literal["fit", "validation", "confirmation"] | None = None
    model_input: P2RModelInput
    candidate_lineage: P2RCandidateLineage
    execution_candidate_text: str | None = None
    topk_candidate_ids: list[str]
    catalog_session_count: int = Field(ge=0)
    audit_only: dict[str, Any]

    @model_validator(mode="after")
    def group_and_identity_isolation(self):
        if self.split_group_key != make_split_group_key(self.state.catalog_user_id):
            raise ValueError("split must be keyed by the full user cluster")
        encoded_features = canonical_json(self.model_input.model_dump(mode="json"))
        identity = self.candidate_lineage.candidate_id
        if identity and identity in encoded_features:
            raise ValueError("candidate identity leaked into Step1 model input")
        encoded_audit = canonical_json(self.audit_only)
        if self.state.state_condition not in encoded_audit:
            raise ValueError("state condition must remain in audit_only provenance")
        return self


class P2RInteractionBlueprintRow(StrictModel):
    """One shared current state with two or more independently discovered components."""

    protocol: Literal["pm-v1.5-v5.3-p2r-learning-blueprint-v1"] = P2R_PROTOCOL
    interaction_id: str = Field(pattern=r"^state_[0-9a-f]{12,64}$")
    catalog_user_id: str = Field(min_length=1)
    semantic_family: str = Field(min_length=1)
    counterfactual_group_id: str = Field(pattern=r"^group_[0-9a-f]{12,64}$")
    split_group_key: str = Field(min_length=1)
    current_session_index: int = Field(gt=0)
    visible_dialogue: list[DialogueTurn] = Field(min_length=1)
    components: dict[Component, P2RModelInput]
    candidate_lineage: dict[Component, P2RCandidateLineage]
    execution_candidate_texts: dict[Component, str | None]
    audit_only: dict[str, Any]

    @model_validator(mode="after")
    def coherent_interaction(self):
        component_keys = set(self.components)
        if len(component_keys) < 2:
            raise ValueError("interaction rows require at least two components")
        if component_keys != set(self.candidate_lineage):
            raise ValueError("interaction component and lineage keys differ")
        if component_keys != set(self.execution_candidate_texts):
            raise ValueError("interaction component and execution keys differ")
        if self.split_group_key != make_split_group_key(self.catalog_user_id):
            raise ValueError("interaction split must remain at user-cluster grain")
        if self.visible_dialogue[-1].role != "user":
            raise ValueError("interaction dialogue must end with current user turn")
        return self


def normalize_worker_catalog_row(raw: Mapping[str, Any]) -> P2RLongitudinalCatalogItem:
    """Normalize a narrow set of harmless worker field aliases.

    The adapter does not infer missing semantic content.  A worker artifact
    with no owner, time, subtype, family, or structured ME/MP payload fails
    loudly instead of being silently repaired by the Leader.
    """

    row = dict(raw)
    aliases = {
        "user_id": "catalog_user_id",
        "owner_id": "catalog_user_id",
        "item_id": "memory_id",
        "candidate_id": "memory_id",
        "source": "component",
        "family": "semantic_family",
        "text": "literal_text",
        "version": "candidate_version",
        "created_session_index": "created_session",
    }
    for source, target in aliases.items():
        if target not in row and source in row:
            row[target] = row.pop(source)
    row.setdefault("protocol", CATALOG_PROTOCOL)
    row.setdefault("valid_from_session", row.get("created_session"))
    row.setdefault("subject_entity_id", row.get("catalog_user_id"))
    row.setdefault("source_session_ref", f"session_{row.get('created_session', 'unknown')}")
    row.setdefault(
        "semantic_family",
        row.get("topic_thread")
        or ("response_preference" if row.get("subtype") == "MP_PREFERENCE" else row.get("field_type")),
    )
    row.setdefault(
        "topic_key",
        str(
            row.get("topic_thread")
            or (
                row.get("field_value")
                if row.get("subtype") == "MP_PREFERENCE"
                else row.get("field_type")
            )
            or ""
        ).replace("_", " "),
    )
    if row.get("subtype") == "ME_REUSABLE_OUTCOME":
        compiled = compile_atomic_reusable_outcome(str(row.get("literal_text") or ""))
        if compiled is not None:
            row.setdefault("action_span", compiled.past_action_span)
            row.setdefault("outcome_span", compiled.observed_outcome_span)
    if "catalog_role" not in row:
        if row.get("subtype") == "ME_REUSABLE_OUTCOME":
            row["catalog_role"] = "target_capable"
        elif row.get("subtype") in {"ME_UNRESOLVED_EVENT", "ME_CONTEXT_EVENT"}:
            row["catalog_role"] = "same_topic_competitor"
        elif row.get("subtype") == "MS_SESSION":
            row["catalog_role"] = (
                "resolved_or_stale"
                if row.get("resolution_status") == "resolved"
                else "target_capable"
                if row.get("resolution_status") == "unresolved"
                else "same_topic_competitor"
            )
        elif row.get("component") == "MP":
            row["catalog_role"] = (
                "resolved_or_stale" if row.get("superseded") else "target_capable"
            )
    # Worker-only audit fields have already served their construction checks;
    # none is silently promoted into a Step1 feature.
    for key in (
        "source_span",
        "surface_sha256",
        "topic_thread",
        "involves_person",
        "compiler_valid_intended",
        "me_compiler_valid_actual",
        "ms_compiler_valid_actual",
        "resolution_status",
        "active",
        "superseded",
        "supersedes_item_id",
    ):
        row.pop(key, None)
    return P2RLongitudinalCatalogItem.model_validate(row)


def normalize_worker_catalog(rows: Sequence[Mapping[str, Any]]) -> list[P2RLongitudinalCatalogItem]:
    prepared = [dict(row) for row in rows]
    superseded_at: dict[str, int] = {}
    for row in prepared:
        prior = row.get("supersedes_item_id")
        if prior:
            superseded_at[str(prior)] = int(row["created_session"])
    for row in prepared:
        item_id = str(row.get("item_id") or row.get("memory_id") or row.get("candidate_id") or "")
        if item_id in superseded_at:
            row["valid_until_session"] = superseded_at[item_id] - 1
    normalized = [normalize_worker_catalog_row(row) for row in prepared]
    ids = [row.memory_id for row in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("longitudinal catalog contains duplicate memory_id values")
    return normalized


def _eligible_catalog_for_state(
    state: P2RCurrentStateSpec,
    catalog: Sequence[P2RLongitudinalCatalogItem],
) -> list[P2RLongitudinalCatalogItem]:
    result: list[P2RLongitudinalCatalogItem] = []
    for item in catalog:
        if item.catalog_user_id != state.catalog_user_id:
            continue
        if not item.is_active_strict_past(state.current_session_index):
            continue
        # Response preferences belong to the user across topics; other facts,
        # session observations and outcomes require entity alignment.
        is_response_preference = (
            item.component == "MP" and item.subtype == "MP_PREFERENCE"
        )
        if not is_response_preference and item.subject_entity_id != state.current_subject_entity_id:
            continue
        result.append(item)
    return result


def _catalog_metadata(
    catalog: Sequence[P2RLongitudinalCatalogItem],
) -> dict[str, dict[str, Any]]:
    return {
        item.memory_id: {
            "mp_subtype": item.subtype if item.component == "MP" else None,
            "me_subtype_hint": item.subtype if item.component == "ME" else None,
            "field_type": item.field_type,
            "field_value": item.field_value,
            "subject_entity_id": item.subject_entity_id,
            "candidate_version": item.candidate_version,
        }
        for item in catalog
    }


def _slot_payload(observation: Any) -> dict[str, Any]:
    payload = asdict(observation)
    payload.pop("component", None)
    return payload


def _structured_mp_rank(
    *,
    current_user_text: str,
    catalog: Sequence[P2RLongitudinalCatalogItem],
) -> tuple[list[MemoryItem], dict[str, float]]:
    """Rank MP by pre-stored applicability scope, never by field labels.

    ``topic_key`` describes where a profile fact/preference can change a
    response (for example, appointment logistics or advice presentation).
    The hidden ``field_value`` remains the increment to inject and is not
    required to appear in the current turn.  This removes the W7R circularity
    where a positive state had to repeat the stored preference to retrieve it.
    """

    scored: list[
        tuple[int, float, float, int, str, P2RLongitudinalCatalogItem]
    ] = []
    for item in catalog:
        if item.component != "MP":
            continue
        scope_document = " ".join(
            part for part in (item.field_type, item.topic_key) if part
        )
        explicit_scope_score = final_typed_content_match_level(
            current_user_text, scope_document
        )
        # A stable response preference is a structurally available fallback;
        # unlike a profile fact, retrieving it must not require the user to
        # restate the preference in the current message.  A profile fact must
        # still have an explicit applicability-scope match and outranks the
        # preference fallback when present.
        scope_score = (
            max(0.25, explicit_scope_score)
            if item.subtype == "MP_PREFERENCE"
            else explicit_scope_score
        )
        if scope_score <= 0.0:
            continue
        surface_score = lexical_score(current_user_text, item.literal_text)
        scored.append(
            (
                int(item.subtype == "MP_PROFILE" and explicit_scope_score > 0.0),
                scope_score,
                surface_score,
                item.created_session,
                item.memory_id,
                item,
            )
        )
    scored.sort(key=lambda row: row[:5], reverse=True)
    selected_catalog = [row[5] for row in scored[:2]]
    selected = [item.as_memory_item() for item in selected_catalog]
    top1 = scored[0][1] if scored else 0.0
    top2 = scored[1][1] if len(scored) > 1 else 0.0
    return selected, {
        "top1_structured_scope_relevance": top1,
        "top1_top2_structured_scope_margin": top1 - top2,
    }


def materialize_memory_state(
    *,
    state: P2RCurrentStateSpec,
    catalog: Sequence[P2RLongitudinalCatalogItem],
    ms_semantic_encoder: Any | None = None,
) -> P2RBlueprintRow:
    """Materialize MP/MS/ME with one shared exact Rank-1 candidate.

    Candidate discovery reads only the strictly-past eligible catalog and the
    visible current text.  It never sees state_condition or intended labels.
    ME keeps the frozen rule: exact Rank-1 must compile or the component is
    unavailable; Rank-2 is never promoted.
    """

    if state.component == "RS":
        raise ValueError("RS requires the dedicated strategy materializer")
    source = MemorySource(state.component)
    eligible_catalog = _eligible_catalog_for_state(state, catalog)
    memory_items = [item.as_memory_item() for item in eligible_catalog]
    queries = {candidate_source: state.current_user_text for candidate_source in MemorySource}
    if source is MemorySource.MP:
        topk, structured_mp_descriptor = _structured_mp_rank(
            current_user_text=state.current_user_text,
            catalog=eligible_catalog,
        )
        descriptor: dict[str, Any] = {
            "ms_semantic_reranked": False,
            **structured_mp_descriptor,
        }
    else:
        discoveries = discover_final_typed_memory_candidates(
            queries=queries,
            items=memory_items,
            source_metadata=_catalog_metadata(eligible_catalog),
            session_index=state.current_session_index,
            ms_semantic_encoder=ms_semantic_encoder,
        )
        discovery = discoveries[source]
        topk = list(discovery.selected_items)
        descriptor = discovery.descriptor
    rank1 = topk[0] if topk else None
    rank1_catalog = next(
        (item for item in eligible_catalog if rank1 and item.memory_id == rank1.memory_id),
        None,
    )
    rank1_compiler_valid: bool | None = None
    execution_rank1 = rank1
    if source is MemorySource.ME and rank1 is not None:
        rank1_compiler_valid = compile_atomic_reusable_outcome(rank1.text) is not None
        if not rank1_compiler_valid:
            execution_rank1 = None

    selected_for_slots = topk if execution_rank1 is not None else []
    candidate_text = execution_rank1.text if execution_rank1 is not None else None
    source_items = [item for item in memory_items if item.source is source]
    if source is MemorySource.MP:
        is_preference = bool(
            rank1_catalog is not None and rank1_catalog.subtype == "MP_PREFERENCE"
        )
        observation = mp_contribution_slots(
            current_user_text=state.current_user_text,
            candidate_text=candidate_text,
            candidate_is_preference=is_preference,
            source_items=source_items,
            selected_items=selected_for_slots,
            session_index=state.current_session_index,
        )
    elif source is MemorySource.MS:
        observation = ms_contribution_slots(
            current_user_text=state.current_user_text,
            candidate_text=candidate_text,
            source_items=source_items,
            selected_items=selected_for_slots,
            session_index=state.current_session_index,
        )
    else:
        observation = me_contribution_slots(
            current_user_text=state.current_user_text,
            candidate_text=candidate_text,
            source_items=source_items,
            selected_items=selected_for_slots,
            session_index=state.current_session_index,
        )

    slots = _slot_payload(observation)
    if source is MemorySource.MP and rank1_catalog is not None:
        increment_words = final_typed_content_words(
            str(rank1_catalog.field_value or "").replace("_", " ")
        )
        current_words = final_typed_content_words(state.current_user_text)
        slots["current_redundant"] = bool(increment_words) and increment_words <= current_words
    elif source is MemorySource.MS and candidate_text:
        slots["current_redundant"] = (
            normalize_space(candidate_text).casefold()
            in normalize_space(state.current_user_text).casefold()
        )
    for role in (
        "target_capable",
        "same_topic_competitor",
        "cross_topic_distractor",
        "resolved_or_stale",
        "wrong_entity",
    ):
        slots[f"candidate_role_{role}"] = bool(
            rank1_catalog is not None and rank1_catalog.catalog_role == role
        )
    if source is MemorySource.MP:
        slots.update(structured_mp_descriptor)
    if source is MemorySource.MS and descriptor.get("ms_semantic_reranked"):
        slots["topk_top1_semantic_relevance"] = descriptor.get(
            "top1_semantic_relevance", 0.0
        )
        slots["topk_top1_top2_semantic_margin"] = descriptor.get(
            "top1_top2_semantic_margin", 0.0
        )

    session_count = len({item.created_session for item in eligible_catalog})
    candidate_version = (
        rank1_catalog.candidate_version
        if execution_rank1 is not None and rank1_catalog is not None
        else None
    )
    return P2RBlueprintRow(
        state=state,
        split_group_key=make_split_group_key(state.catalog_user_id),
        model_input=P2RModelInput(
            component=state.component,
            candidate_present=execution_rank1 is not None,
            contribution_slots=slots,
        ),
        candidate_lineage=P2RCandidateLineage(
            candidate_id=execution_rank1.memory_id if execution_rank1 else None,
            component=state.component,
            candidate_version=candidate_version,
            created_session=execution_rank1.created_session if execution_rank1 else None,
            candidate_surface_sha256=(
                sha256_text(execution_rank1.text) if execution_rank1 else None
            ),
            retrieval_method=(
                "BGE_M3_COSINE_FULL_CAUSAL_POOL"
                if descriptor.get("ms_semantic_reranked")
                else "STRUCTURED_MP_APPLICABILITY_SCOPE_RANKER"
                if source is MemorySource.MP
                else "PRODUCTION_CONTENT_MATCH_TYPED_RANKER"
            ),
            rank1_compiler_valid=rank1_compiler_valid,
        ),
        execution_candidate_text=candidate_text,
        topk_candidate_ids=[item.memory_id for item in topk],
        catalog_session_count=session_count,
        audit_only={
            "state_condition": state.state_condition,
            "intended_response_act": state.intended_response_act,
            "construction_note": state.construction_note,
            "eligible_catalog_item_count": len(eligible_catalog),
            "rank1_catalog_role": rank1_catalog.catalog_role if rank1_catalog else None,
            "generated_response_or_outcome_read": False,
        },
    )


def materialize_rs_state(
    *,
    state: P2RCurrentStateSpec,
    cards: Sequence[StrategyCard],
) -> P2RBlueprintRow:
    """Materialize the frozen six-card RS candidate and runtime slots."""

    if state.component != "RS":
        raise ValueError("materialize_rs_state only accepts RS states")
    dialogue = [turn.model_dump(mode="json") for turn in state.visible_dialogue]
    pool = rs_mechanical_candidate_pool(
        recent_dialogue=dialogue,
        cards=cards,
        already_executed_move_ids=state.already_executed_move_ids,
    )
    shared = rs_shared_candidate_top1(pool)
    if shared is None:
        slots = {
            "candidate_present": False,
            "card_precondition_met": False,
            "card_already_executed_last_turn": bool(state.already_executed_move_ids),
            **{f"observable_{key}": value for key, value in pool.observable_flags.items()},
        }
        lineage = P2RCandidateLineage(
            candidate_id=None,
            component="RS",
            candidate_version=None,
            created_session=None,
            candidate_surface_sha256=None,
            retrieval_method="SIX_CARD_MECHANICAL_POOL_HARD_OFF",
        )
        candidate_text = None
        topk_ids: list[str] = []
    else:
        observation = rs_contribution_slots(
            current_user_text=state.current_user_text,
            recent_dialogue=dialogue,
            move_id=shared.observation.move_id,
            already_executed_last_turn=False,
        )
        slots = _slot_payload(observation)
        slots.update(
            {
                "selection_mode_transparent_priority": (
                    shared.selection_mode == "transparent_priority"
                ),
                "transparent_rule_on": shared.transparent_rule_on,
                **{
                    f"observable_{key}": value
                    for key, value in pool.observable_flags.items()
                },
            }
        )
        card = shared.observation.strategy_card
        candidate_text = card.guidance_text
        lineage = P2RCandidateLineage(
            candidate_id=card.strategy_id,
            component="RS",
            candidate_version=1,
            created_session=None,
            candidate_surface_sha256=sha256_text(candidate_text),
            retrieval_method=f"SIX_CARD_{shared.selection_mode.upper()}",
        )
        topk_ids = [candidate.move_id for candidate in pool.candidates]
    return P2RBlueprintRow(
        state=state,
        split_group_key=make_split_group_key(state.catalog_user_id),
        model_input=P2RModelInput(
            component="RS",
            candidate_present=shared is not None,
            contribution_slots=slots,
        ),
        candidate_lineage=lineage,
        execution_candidate_text=candidate_text,
        topk_candidate_ids=topk_ids,
        catalog_session_count=0,
        audit_only={
            "state_condition": state.state_condition,
            "intended_response_act": state.intended_response_act,
            "construction_note": state.construction_note,
            "hard_off_reason": pool.hard_off_reason,
            "generated_response_or_outcome_read": False,
        },
    )


def assemble_interaction_row(
    *,
    component_rows: Mapping[Component, P2RBlueprintRow],
    variant: str,
) -> P2RInteractionBlueprintRow:
    """Combine independently materialized components without changing candidates.

    Every input row must describe the exact same current dialogue, user,
    session and family.  The function copies each component's already frozen
    candidate identity; it never reranks for an interaction arm.
    """

    if len(component_rows) < 2:
        raise ValueError("need at least two component rows")
    rows = list(component_rows.values())
    first = rows[0]
    surface = canonical_json(
        [turn.model_dump(mode="json") for turn in first.state.visible_dialogue]
    )
    for component, row in component_rows.items():
        if row.state.component != component:
            raise ValueError("component map key differs from row component")
        if row.state.catalog_user_id != first.state.catalog_user_id:
            raise ValueError("interaction rows have different users")
        if row.state.current_session_index != first.state.current_session_index:
            raise ValueError("interaction rows have different session indices")
        if row.state.semantic_family != first.state.semantic_family:
            raise ValueError("interaction rows have different families")
        if canonical_json(
            [turn.model_dump(mode="json") for turn in row.state.visible_dialogue]
        ) != surface:
            raise ValueError("interaction rows do not share the same visible dialogue")
    group = "group_" + stable_hex(
        "INTERACTION",
        first.state.catalog_user_id,
        first.state.semantic_family,
        ",".join(sorted(component_rows)),
        n=24,
    )
    interaction_id = "state_" + stable_hex(group, variant, n=24)
    return P2RInteractionBlueprintRow(
        interaction_id=interaction_id,
        catalog_user_id=first.state.catalog_user_id,
        semantic_family=first.state.semantic_family,
        counterfactual_group_id=group,
        split_group_key=make_split_group_key(first.state.catalog_user_id),
        current_session_index=first.state.current_session_index,
        visible_dialogue=first.state.visible_dialogue,
        components={key: value.model_input for key, value in component_rows.items()},
        candidate_lineage={
            key: value.candidate_lineage for key, value in component_rows.items()
        },
        execution_candidate_texts={
            key: value.execution_candidate_text for key, value in component_rows.items()
        },
        audit_only={
            "variant": variant,
            "source_state_ids": {
                key: value.state.state_id for key, value in component_rows.items()
            },
            "generated_response_or_outcome_read": False,
        },
    )


def make_state_identity(
    *, component: Component, catalog_user_id: str, semantic_family: str, variant: str
) -> tuple[str, str]:
    """Return state and counterfactual-group IDs at the correct grain.

    All variants for one component/user/family share a group.  This prevents
    state-level clones from being split across fit and confirmation.
    """

    group = "group_" + stable_hex(component, catalog_user_id, semantic_family, n=24)
    state = "state_" + stable_hex(group, variant, n=24)
    return state, group


def make_split_group_key(catalog_user_id: str) -> str:
    """One indivisible split cluster per synthetic/real user."""

    return "cluster_" + stable_hex("USER_SPLIT", catalog_user_id, n=24)


def audit_blueprint_rows(
    rows: Sequence[P2RBlueprintRow],
    *,
    require_final_condition_crossing: bool = False,
    require_assigned_splits: bool = False,
) -> dict[str, Any]:
    """Run stable, outcome-blind data-quality checks over materialized rows."""

    state_ids = [row.state.state_id for row in rows]
    duplicate_state_ids = len(state_ids) - len(set(state_ids))
    future_or_current_candidates = sum(
        1
        for row in rows
        if row.candidate_lineage.created_session is not None
        and row.candidate_lineage.created_session >= row.state.current_session_index
    )
    group_splits: dict[str, set[str]] = {}
    conditions_by_group: dict[str, set[str]] = {}
    conditions_by_family: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row.proposed_split is not None:
            group_splits.setdefault(row.split_group_key, set()).add(row.proposed_split)
        conditions_by_group.setdefault(row.state.counterfactual_group_id, set()).add(
            row.state.state_condition
        )
        if row.state.interaction_key is None:
            conditions_by_family.setdefault(
                (row.state.component, row.state.semantic_family), set()
            ).add(row.state.state_condition)
    split_leak_groups = sorted(
        group for group, splits in group_splits.items() if len(splits) > 1
    )
    groups_without_condition_crossing = sorted(
        group for group, conditions in conditions_by_group.items() if len(conditions) < 2
    )
    families_without_condition_crossing = sorted(
        f"{component}:{family}"
        for (component, family), conditions in conditions_by_family.items()
        if len(conditions) < 2
    )
    gold_tokens = ("gold", "worth_opening", "quality_preference", "material_risk")
    model_input_gold_leaks = sum(
        any(token in canonical_json(row.model_input.model_dump(mode="json")).lower() for token in gold_tokens)
        for row in rows
    )
    missing_runtime_feature_rows = sum(
        not row.model_input.contribution_slots for row in rows
    )
    candidate_counts = [len(row.topk_candidate_ids) for row in rows if row.state.component != "RS"]
    eligible_item_counts = [
        int(row.audit_only.get("eligible_catalog_item_count", 0))
        for row in rows
        if row.state.component != "RS"
    ]
    catalog_session_counts = [row.catalog_session_count for row in rows if row.state.component != "RS"]
    critical_failures = {
        "duplicate_state_ids": duplicate_state_ids,
        "future_or_current_candidates": future_or_current_candidates,
        "split_leak_groups": len(split_leak_groups),
        "model_input_gold_leaks": model_input_gold_leaks,
        "missing_runtime_feature_rows": missing_runtime_feature_rows,
    }
    if require_final_condition_crossing:
        critical_failures["families_without_condition_crossing"] = len(
            families_without_condition_crossing
        )
    if require_assigned_splits:
        critical_failures["unassigned_split_rows"] = sum(
            row.proposed_split is None for row in rows
        )
    return {
        "protocol": P2R_PROTOCOL,
        "row_count": len(rows),
        "component_counts": {
            component: sum(row.state.component == component for row in rows)
            for component in ("MP", "MS", "ME", "RS")
        },
        "unique_user_clusters": len({row.state.catalog_user_id for row in rows}),
        "unique_counterfactual_groups": len(conditions_by_group),
        "critical_failures": critical_failures,
        "groups_without_condition_crossing": groups_without_condition_crossing,
        "families_without_condition_crossing": families_without_condition_crossing,
        "split_leak_groups": split_leak_groups,
        "selected_topk_size_observed": {
            "minimum": min(candidate_counts) if candidate_counts else 0,
            "maximum": max(candidate_counts) if candidate_counts else 0,
        },
        "eligible_memory_catalog_item_count_observed": {
            "minimum": min(eligible_item_counts) if eligible_item_counts else 0,
            "maximum": max(eligible_item_counts) if eligible_item_counts else 0,
        },
        "item_bearing_strict_past_session_count_observed": {
            "minimum": min(catalog_session_counts) if catalog_session_counts else 0,
            "maximum": max(catalog_session_counts) if catalog_session_counts else 0,
        },
        "static_structure_pass": not any(critical_failures.values()),
        "generated_response_or_outcome_read": False,
        "api_calls": 0,
    }
