"""Final candidate-first contracts for PM V1.5 Step 1.

The contract closes three failure modes from the earlier experiments:

* a gold label must refer to the exact rank-1 item that would be injected;
* the feature compiler and the human/adjudication labeler are separate;
* identifiers, split/domain names, target actions, responses, and outcomes can
  never enter the low-capacity routing heads.

The module does not retrieve, generate, judge, or train.  It is a static
boundary shared by internal construction and external evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Any, Literal, Mapping, Sequence

from .contracts import MemorySource
from .text import normalize_space
from .text import content_word_match_level, normalize_for_hash
from .v1_5_candidate_discovery import MemoryCandidate
from .v1_5_generator_alignment_audit import (
    materialize_strategy_card_for_execution,
)


FINAL_CANDIDATE_CONTRACT_PROTOCOL = (
    "pm-v1.5-final-exact-rank1-candidate-contract-v1"
)
FINAL_FEATURE_CONTRACT_PROTOCOL = (
    "pm-v1.5-final-independent-low-capacity-feature-contract-v1"
)

Component = Literal["MP", "MS", "ME", "RS"]
Split = Literal["FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"]
OpportunityDecision = Literal["on", "off", "abstain"]


class CandidateSubtype(str, Enum):
    MP_PREFERENCE = "MP_PREFERENCE"
    MP_PROFILE = "MP_PROFILE"
    MS_SESSION = "MS_SESSION"
    ME_REUSABLE_OUTCOME = "ME_REUSABLE_OUTCOME"
    ME_CONTEXT_EVENT = "ME_CONTEXT_EVENT"
    ME_UNRESOLVED_EVENT = "ME_UNRESOLVED_EVENT"
    RS_ATOMIC_MOVE = "RS_ATOMIC_MOVE"
    CANDIDATE_ABSENT = "CANDIDATE_ABSENT"


_ALLOWED_SUBTYPES: dict[str, frozenset[CandidateSubtype]] = {
    "MP": frozenset(
        {CandidateSubtype.MP_PREFERENCE, CandidateSubtype.MP_PROFILE}
    ),
    "MS": frozenset({CandidateSubtype.MS_SESSION}),
    "ME": frozenset(
        {
            CandidateSubtype.ME_REUSABLE_OUTCOME,
            CandidateSubtype.ME_CONTEXT_EVENT,
            CandidateSubtype.ME_UNRESOLVED_EVENT,
        }
    ),
    "RS": frozenset({CandidateSubtype.RS_ATOMIC_MOVE}),
}

# These are the complete model-facing namespaces.  Audit-only descriptors may
# contain more fields, but the four logistic heads may not consume them.
FINAL_FEATURE_NAMES: dict[str, tuple[str, ...]] = {
    "MP": (
        "candidate_content_match_level",
        "candidate_incremental_information",
        "candidate_preference_scope_fit",
        "candidate_profile_relevance",
        "candidate_profile_entity_scope_fit",
        "candidate_current_scope_conflict",
    ),
    "MS": (
        "candidate_content_match_level",
        "candidate_incremental_information",
        "candidate_current_goal_fit",
        "candidate_prior_outcome_or_distinction",
        "candidate_specific_issue_or_distinction",
        "current_continuity_invitation",
        "candidate_prior_issue_marked_resolved",
    ),
    "ME": (
        "candidate_content_match_level",
        "candidate_incremental_information",
        "candidate_current_goal_fit",
        "candidate_contains_action",
        "candidate_contains_result",
        "candidate_contains_mechanism",
        "current_continuity_invitation",
    ),
    "RS": (
        "candidate_content_match_level",
        "candidate_mode_fit",
        "candidate_goal_fit",
        "candidate_burden_fit",
        "candidate_boundary_fit",
        "candidate_nonredundancy",
    ),
}

_FORBIDDEN_NAME_PARTS = (
    "label",
    "gold",
    "target",
    "winner",
    "quality",
    "risk_outcome",
    "response",
    "judge",
    "dataset",
    "domain",
    "split",
    "state_id",
    "user_id",
    "candidate_id",
    "memory_id",
    "strategy_id",
    "action_id",
    "condition",
)


def candidate_text_sha256(text: str) -> str:
    normalized = normalize_space(text)
    if not normalized:
        raise ValueError("candidate text must be non-empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExactRank1CandidateSurface:
    protocol: str
    state_id: str
    component: Component
    candidate_present: bool
    candidate_id: str | None
    candidate_text: str | None
    candidate_text_sha256: str | None
    candidate_age_sessions: int | None
    compiler_subtype_hint: CandidateSubtype
    selected_rank: int | None
    compiler_protocol: str

    def __post_init__(self) -> None:
        if self.protocol != FINAL_CANDIDATE_CONTRACT_PROTOCOL:
            raise ValueError("unexpected exact-candidate protocol")
        if self.component not in FINAL_FEATURE_NAMES:
            raise ValueError(f"unknown component: {self.component}")
        if not normalize_space(self.state_id):
            raise ValueError("state_id must be non-empty")
        if not normalize_space(self.compiler_protocol):
            raise ValueError("compiler_protocol must be non-empty")
        if not self.candidate_present:
            if any(
                value is not None
                for value in (
                    self.candidate_id,
                    self.candidate_text,
                    self.candidate_text_sha256,
                    self.candidate_age_sessions,
                    self.selected_rank,
                )
            ):
                raise ValueError("absent candidate cannot carry an item surface")
            if self.compiler_subtype_hint is not CandidateSubtype.CANDIDATE_ABSENT:
                raise ValueError("absent candidate must use CANDIDATE_ABSENT")
            return
        if self.selected_rank != 1:
            raise ValueError("the Step-1 label surface must be exact rank-1")
        if self.component == "RS":
            if self.candidate_age_sessions is not None:
                raise ValueError("RS cards do not carry a user-history age")
        elif not isinstance(self.candidate_age_sessions, int) or self.candidate_age_sessions <= 0:
            raise ValueError("memory rank-1 requires a strictly positive age")
        if not normalize_space(self.candidate_id or ""):
            raise ValueError("present candidate requires candidate_id")
        if not normalize_space(self.candidate_text or ""):
            raise ValueError("present candidate requires candidate_text")
        expected_hash = candidate_text_sha256(self.candidate_text or "")
        if self.candidate_text_sha256 != expected_hash:
            raise ValueError("candidate text SHA-256 mismatch")
        if self.compiler_subtype_hint not in _ALLOWED_SUBTYPES[self.component]:
            raise ValueError("candidate subtype is incompatible with component")


@dataclass(frozen=True)
class CandidateOpportunityGold:
    state_id: str
    component: Component
    candidate_id: str | None
    candidate_text_sha256: str | None
    adjudicated_subtype: CandidateSubtype
    decision: OpportunityDecision
    labeler_protocol: str
    evidence_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.component not in FINAL_FEATURE_NAMES:
            raise ValueError(f"unknown component: {self.component}")
        if self.decision not in {"on", "off", "abstain"}:
            raise ValueError(f"unknown opportunity decision: {self.decision}")
        if not normalize_space(self.labeler_protocol):
            raise ValueError("labeler_protocol must be non-empty")
        if not self.evidence_codes:
            raise ValueError("gold requires at least one auditable evidence code")
        if self.adjudicated_subtype is CandidateSubtype.CANDIDATE_ABSENT:
            if self.candidate_id is not None or self.candidate_text_sha256 is not None:
                raise ValueError("absent gold cannot bind a candidate")
            if self.decision != "off":
                raise ValueError("absent candidate is deterministically off")
        elif self.adjudicated_subtype not in _ALLOWED_SUBTYPES[self.component]:
            raise ValueError("gold subtype is incompatible with component")
        if (
            self.component == "ME"
            and self.decision == "on"
            and self.adjudicated_subtype is not CandidateSubtype.ME_REUSABLE_OUTCOME
        ):
            raise ValueError("ME-on requires an adjudicated reusable outcome")


@dataclass(frozen=True)
class IndependentFeatureRecord:
    protocol: str
    state_id: str
    component: Component
    feature_builder_protocol: str
    model_features: Mapping[str, float]
    forbidden_inputs_read: tuple[str, ...] = ()
    outcome_read: bool = False

    def __post_init__(self) -> None:
        if self.protocol != FINAL_FEATURE_CONTRACT_PROTOCOL:
            raise ValueError("unexpected final feature protocol")
        if self.component not in FINAL_FEATURE_NAMES:
            raise ValueError(f"unknown component: {self.component}")
        if not normalize_space(self.feature_builder_protocol):
            raise ValueError("feature_builder_protocol must be non-empty")
        if self.outcome_read or self.forbidden_inputs_read:
            raise ValueError("feature construction read a forbidden/outcome input")
        expected = set(FINAL_FEATURE_NAMES[self.component])
        actual = set(self.model_features)
        if actual != expected:
            raise ValueError(
                f"{self.component} feature schema mismatch: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        if not 5 <= len(actual) <= 7:
            raise ValueError("each V1.5 head must use 5-7 features")
        for name, value in self.model_features.items():
            lowered = name.lower()
            if any(part in lowered for part in _FORBIDDEN_NAME_PARTS):
                raise ValueError(f"forbidden feature name: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"feature {name} must be a real numeric scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"feature {name} must be finite")


@dataclass(frozen=True)
class ExactCandidateTrainingRow:
    surface: ExactRank1CandidateSurface
    gold: CandidateOpportunityGold
    features: IndependentFeatureRecord
    split: Split
    group_id: str
    semantic_family: str

    def __post_init__(self) -> None:
        if not normalize_space(self.group_id):
            raise ValueError("group_id must be non-empty")
        if not normalize_space(self.semantic_family):
            raise ValueError("semantic_family must be non-empty")
        if self.surface.state_id != self.gold.state_id or self.surface.state_id != self.features.state_id:
            raise ValueError("surface, gold, and features must share state_id")
        if self.surface.component != self.gold.component or self.surface.component != self.features.component:
            raise ValueError("surface, gold, and features must share component")
        if self.surface.candidate_id != self.gold.candidate_id:
            raise ValueError("gold is not bound to the exact rank-1 candidate ID")
        if self.surface.candidate_text_sha256 != self.gold.candidate_text_sha256:
            raise ValueError("gold is not bound to the exact rank-1 candidate text")
        if self.features.feature_builder_protocol == self.gold.labeler_protocol:
            raise ValueError("feature builder and labeler protocols must be independent")


def subtype_hint_from_metadata(
    *, component: Component, metadata: Mapping[str, Any]
) -> CandidateSubtype:
    """Read only compiler-authored subtype hints; never infer from a gold label."""

    if component == "MP":
        raw = str(metadata.get("mp_subtype") or "")
    elif component == "MS":
        raw = "MS_SESSION"
    elif component == "ME":
        raw = str(metadata.get("me_subtype_hint") or "")
    else:
        raw = "RS_ATOMIC_MOVE"
    try:
        subtype = CandidateSubtype(raw)
    except ValueError as exc:
        raise ValueError(
            f"missing or invalid compiler subtype hint for {component}: {raw!r}"
        ) from exc
    if subtype not in _ALLOWED_SUBTYPES[component]:
        raise ValueError("compiler subtype hint is incompatible with component")
    return subtype


def exact_rank1_memory_surface(
    *,
    state_id: str,
    candidate: MemoryCandidate,
    source_metadata: Mapping[str, Mapping[str, Any]],
    current_session_index: int,
    compiler_protocol: str,
) -> ExactRank1CandidateSurface:
    """Freeze the actual retriever rank-1 item before any label is created."""

    component: Component = candidate.source.value  # type: ignore[assignment]
    if not candidate.selected_items:
        return ExactRank1CandidateSurface(
            protocol=FINAL_CANDIDATE_CONTRACT_PROTOCOL,
            state_id=state_id,
            component=component,
            candidate_present=False,
            candidate_id=None,
            candidate_text=None,
            candidate_text_sha256=None,
            candidate_age_sessions=None,
            compiler_subtype_hint=CandidateSubtype.CANDIDATE_ABSENT,
            selected_rank=None,
            compiler_protocol=compiler_protocol,
        )
    item = candidate.selected_items[0]
    if item.source is not candidate.source:
        raise ValueError("rank-1 item source does not match candidate source")
    metadata = source_metadata.get(item.memory_id)
    if metadata is None:
        raise ValueError("rank-1 item has no compiler metadata")
    text = normalize_space(item.text)
    age = int(current_session_index) - int(item.created_session)
    if age <= 0:
        raise ValueError("rank-1 memory candidate is not strictly historical")
    return ExactRank1CandidateSurface(
        protocol=FINAL_CANDIDATE_CONTRACT_PROTOCOL,
        state_id=state_id,
        component=component,
        candidate_present=True,
        candidate_id=item.memory_id,
        candidate_text=text,
        candidate_text_sha256=candidate_text_sha256(text),
        candidate_age_sessions=age,
        compiler_subtype_hint=subtype_hint_from_metadata(
            component=component, metadata=metadata
        ),
        selected_rank=1,
        compiler_protocol=compiler_protocol,
    )


def exact_rank1_strategy_surface(
    *,
    state_id: str,
    ranked_cards: Sequence[Mapping[str, Any]],
    cards_by_id: Mapping[str, Mapping[str, Any]],
    compiler_protocol: str,
) -> ExactRank1CandidateSurface:
    """Freeze the actual ranked RS card and its complete execution surface."""

    if not ranked_cards:
        return ExactRank1CandidateSurface(
            protocol=FINAL_CANDIDATE_CONTRACT_PROTOCOL,
            state_id=state_id,
            component="RS",
            candidate_present=False,
            candidate_id=None,
            candidate_text=None,
            candidate_text_sha256=None,
            candidate_age_sessions=None,
            compiler_subtype_hint=CandidateSubtype.CANDIDATE_ABSENT,
            selected_rank=None,
            compiler_protocol=compiler_protocol,
        )
    ranked_id = str(ranked_cards[0].get("card_id") or "")
    if ranked_id not in cards_by_id:
        raise ValueError("rank-1 strategy card is absent from the frozen bank")
    card = cards_by_id[ranked_id]
    if str(card.get("card_id") or "") != ranked_id:
        raise ValueError("rank-1 strategy ID does not match bank card")
    text = materialize_strategy_card_for_execution(card)
    return ExactRank1CandidateSurface(
        protocol=FINAL_CANDIDATE_CONTRACT_PROTOCOL,
        state_id=state_id,
        component="RS",
        candidate_present=True,
        candidate_id=ranked_id,
        candidate_text=text,
        candidate_text_sha256=candidate_text_sha256(text),
        candidate_age_sessions=None,
        compiler_subtype_hint=CandidateSubtype.RS_ATOMIC_MOVE,
        selected_rank=1,
        compiler_protocol=compiler_protocol,
    )


def final_model_feature_projection(
    *, component: Component, source_features: Mapping[str, Any]
) -> dict[str, float]:
    """Project an audit-rich observation onto the frozen 5-7 feature surface."""

    projected: dict[str, float] = {}
    for name in FINAL_FEATURE_NAMES[component]:
        if name not in source_features:
            raise ValueError(f"missing final {component} feature: {name}")
        value = source_features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"feature {name} must be numeric")
        projected[name] = float(value)
    # Construction itself invokes the full static validator.
    IndependentFeatureRecord(
        protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
        state_id="STATIC_PROJECTION_CHECK",
        component=component,
        feature_builder_protocol="static-final-feature-projection-v1",
        model_features=projected,
    )
    return projected


def final_rs_model_features(
    *,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    observable_flags: Mapping[str, Any],
    ranked_card: Mapping[str, Any] | None,
    bank_card: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Compile the frozen transparent RS feature surface before generation."""

    if (ranked_card is None) != (bank_card is None):
        raise ValueError("ranked RS audit row and bank card must be both present or absent")
    if ranked_card is None or bank_card is None:
        return {name: 0.0 for name in FINAL_FEATURE_NAMES["RS"]}
    if str(ranked_card.get("card_id") or "") != str(bank_card.get("card_id") or ""):
        raise ValueError("ranked RS card and bank execution card do not match")
    visible = normalize_space(
        " ".join(
            [
                *(str(row.get("content") or "") for row in visible_dialogue),
                current_user_text,
            ]
        )
    )
    family = str(bank_card.get("strategy_family") or "")
    profile = str(bank_card.get("execution_profile") or "")
    listen_only = bool(observable_flags.get("listen_only"))
    advice_welcome = bool(observable_flags.get("advice_welcome"))
    low_burden = bool(observable_flags.get("low_burden"))
    hard_off = bool(observable_flags.get("ordinary_rag_hard_off"))
    mode_fit = float(
        not hard_off
        and not (
            listen_only
            and family in {"Question", "Providing Suggestions"}
        )
        and not (advice_welcome and family != "Providing Suggestions")
    )
    burden_fit = 1.0 if profile == "minimal" else 0.0 if low_burden else 0.5
    boundary_fit = float(
        not hard_off
        and not (listen_only and family in {"Question", "Providing Suggestions"})
    )
    card_text = normalize_space(
        " ".join(
            str(bank_card.get(name) or "")
            for name in ("retrieval_text", "support_move", "when_to_use")
        )
    )
    move = normalize_for_hash(str(bank_card.get("support_move") or ""))
    visible_hash = normalize_for_hash(visible)
    features = {
        "candidate_content_match_level": float(
            content_word_match_level(visible, card_text)
        ),
        "candidate_mode_fit": mode_fit,
        "candidate_goal_fit": float(
            content_word_match_level(current_user_text, card_text)
        ),
        "candidate_burden_fit": burden_fit,
        "candidate_boundary_fit": boundary_fit,
        "candidate_nonredundancy": float(bool(move) and move not in visible_hash),
    }
    return final_model_feature_projection(component="RS", source_features=features)


def audit_exact_candidate_rows(
    rows: Sequence[ExactCandidateTrainingRow],
) -> dict[str, Any]:
    """Run P1 row-level anti-cheating and uniqueness checks."""

    keys = [(row.surface.state_id, row.surface.component) for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    split_by_state: dict[str, set[str]] = {}
    split_by_group: dict[str, set[str]] = {}
    for row in rows:
        split_by_state.setdefault(row.surface.state_id, set()).add(row.split)
        split_by_group.setdefault(row.group_id, set()).add(row.split)
    cross_split_states = sorted(
        state_id for state_id, splits in split_by_state.items() if len(splits) > 1
    )
    cross_split_groups = sorted(
        group_id for group_id, splits in split_by_group.items() if len(splits) > 1
    )
    if duplicates or cross_split_states or cross_split_groups:
        raise ValueError(
            "exact-candidate row audit failed: "
            f"duplicates={duplicates}, cross_split_states={cross_split_states}, "
            f"cross_split_groups={cross_split_groups}"
        )
    return {
        "protocol": "pm-v1.5-final-candidate-row-static-audit-v1",
        "status": "PASS",
        "row_count": len(rows),
        "unique_state_component_keys": len(keys),
        "duplicate_keys": [],
        "cross_split_states": [],
        "cross_split_groups": [],
        "outcome_features_present": False,
        "exact_rank1_binding_enforced": True,
    }
