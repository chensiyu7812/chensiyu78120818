"""Candidate-level Step1 observation: the contribution_slot schema.

PM_V1_5_V5_3_INTEGRATED_EVIDENCE_EXECUTION_PLAN_20260805_ZH.md section 2.3
specifies what Step1 must observe per component (a per-component
"contribution slot": is the current turn's need actually addressable by
THIS candidate, not just "does a candidate exist"). Until this module, that
table was a plan-doc description with no real feature-extraction code
behind it -- the only fitted feature builder in this project
(PMV2FeatureBuilder, see its own docstring) is explicitly "strictly
pre-retrieval" and never reads candidate-level content at all, which is a
different, already-deprecated question (see PM_V1_5_V5_3_MASTER_STATUS_
20260806_ZH.md section 9).

Deliberately reuses already-validated primitives rather than inventing new
ones: describe_memory_candidate() (v1_5_candidate_discovery.py) already
computes the candidate-level shared numeric features the plan calls for
(top1 relevance, top1/top2 margin, relative age, token cost, capacity
fraction) -- this module only adds the semantic slots that function does
not cover.  ``observable_flags()`` remains the shared RS/MP structural
observer.  ME uses a separate three-valued action-readiness observer because
the old binary explicit-advice surface missed 128/128 consumed development
states; UNKNOWN is retained rather than collapsed to a negative.
compile_atomic_reusable_outcome() and
compile_atomic_session_observation() (v1_5_v5_2_atomic_memory.py) are the
same real, tested compilers used all session for ME/MS candidate validity.

Every slot here is a cheap, transparent, regex/structural signal -- matching
this project's own stated Step1 design ("四个低容量value head...narrow
语义因素可由冻结BGE表示加小型线性分类器提供"): this module provides the
narrow, observable factors; it does not itself decide whether a component
is worth opening.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from .contracts import MemoryItem, MemorySource
from .text import content_word_match_level, content_words, estimate_tokens
from .v1_5_candidate_discovery import describe_memory_candidate
from .v1_5_strategy_rag_runtime import observable_flags
from .v1_5_v5_2_atomic_memory import (
    compile_atomic_reusable_outcome,
    compile_atomic_session_observation,
)
from .v1_5_v5_3_action_readiness import (
    ActionReadiness,
    observe_action_readiness,
)


def _dialogue_from_current_text(current_user_text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": current_user_text}]


def _is_redundant(current_user_text: str, candidate_text: str) -> bool:
    """Cheap, deterministic proxy for "current turn already states the
    SPECIFIC INCREMENT the candidate would add" -- same content-word-
    overlap primitive used elsewhere in this project
    (evidence_usage_plausibility_errors), not a new mechanism. A pass here
    is "no obvious redundancy", not proof of genuine novelty -- same
    honesty bound as that function documents.

    2026-08-06: checked directly against this module's own real ME FIT
    pilot cases -- overlapping the FULL candidate text (action span
    included) against current_user_text flags every same-topic case as
    redundant, since ON/OFF cases deliberately share a topic to isolate the
    action-invitation variable. That conflates "same topic" with "already
    said the specific benefit", which is a different question (the plan's
    own interaction term is specific_increment x current_redundancy, about
    the increment, not the topic). Prefer the outcome span alone when the
    candidate is a compiler-valid ME_REUSABLE_OUTCOME -- that is the actual
    "increment" a reusable-outcome candidate would add.
    """

    outcome = compile_atomic_reusable_outcome(candidate_text)
    compare_text = outcome.observed_outcome_span if outcome is not None else candidate_text
    shared = content_words(current_user_text) & content_words(compare_text)
    return len(shared) >= 3


@dataclass(frozen=True)
class ContributionSlotObservation:
    """2026-08-06: an independent review found this dataclass originally
    conflated two different things under names that read as "the candidate's
    own" values -- see PM_V1_5_V5_3_CONTRIBUTION_SLOT_METHOD_CORRECTION_
    20260806_ZH.md. Split explicitly now:

    rank1_* fields describe ONLY the single exact-Rank-1 candidate that
    v1_5_v5_3_typed_response_program.py actually injects into a real reply
    (this project's "top-k discovery, top-1 execution" principle, applied
    consistently here too) -- computed directly from that one MemoryItem,
    not from describe_memory_candidate()'s selected_items aggregate.

    topk_* fields are legitimate Top-k diagnostics (this project's plan
    explicitly keeps Top-k descriptors for margin/coverage/OOD -- see
    PM_V1_5_V5_3_INTEGRATED_EVIDENCE_EXECUTION_PLAN_20260805_ZH.md section
    2.1) and may aggregate over more than one item. They must not be read
    as "the injected candidate's own cost/age."
    """

    component: str
    candidate_present: bool
    rank1_relative_age: float | None = None
    rank1_injected_tokens: int = 0
    topk_top1_lexical_relevance: float = 0.0
    topk_top1_top2_lexical_margin: float = 0.0
    topk_minimum_relative_age: float | None = None
    topk_median_relative_age: float | None = None
    topk_maximum_relative_age: float | None = None
    topk_incremental_injected_tokens: int = 0
    topk_capacity_fraction: float = 0.0
    current_redundant: bool = False

    # ME
    past_action_result: bool | None = None
    current_action_invitation: bool | None = None
    current_action_readiness: str | None = None

    # MS
    has_specific_prior_observation: bool | None = None
    continuity_request: bool | None = None

    # MP
    candidate_is_preference: bool | None = None
    preference_applies_to_response_act: bool | None = None
    profile_goal_needs_advice_or_arrangement: bool | None = None

    # RS
    card_precondition_met: bool | None = None
    card_already_executed_last_turn: bool | None = None


def _shared_descriptor_fields(
    *, source: MemorySource, query: str, source_items: Sequence[MemoryItem],
    selected_items: Sequence[MemoryItem], session_index: int,
) -> dict[str, Any]:
    """rank1_* is computed from ONLY selected_items[0] -- the one item that
    would actually be injected -- never from the full (possibly
    multi-item, up to top_k) selected_items list describe_memory_candidate()
    aggregates over. topk_* keeps using that aggregate, correctly labeled."""

    descriptor = describe_memory_candidate(
        source=source, query=query, source_items=source_items,
        selected_items=selected_items, session_index=session_index,
    )
    rank1 = selected_items[0] if selected_items else None
    rank1_relative_age = (
        (session_index - rank1.created_session) / float(session_index)
        if rank1 is not None else None
    )
    rank1_injected_tokens = estimate_tokens(rank1.text) if rank1 is not None else 0
    return {
        "candidate_present": descriptor["candidate_present"],
        "rank1_relative_age": rank1_relative_age,
        "rank1_injected_tokens": rank1_injected_tokens,
        "topk_top1_lexical_relevance": descriptor["top1_lexical_relevance"],
        "topk_top1_top2_lexical_margin": descriptor["top1_top2_lexical_margin"],
        "topk_minimum_relative_age": descriptor["minimum_relative_age"],
        "topk_median_relative_age": descriptor["median_relative_age"],
        "topk_maximum_relative_age": descriptor["maximum_relative_age"],
        "topk_incremental_injected_tokens": descriptor["incremental_injected_tokens"],
        "topk_capacity_fraction": descriptor["top_k_capacity_fraction"],
    }


def me_contribution_slots(
    *, current_user_text: str, candidate_text: str | None,
    source_items: Sequence[MemoryItem], selected_items: Sequence[MemoryItem],
    session_index: int,
) -> ContributionSlotObservation:
    """past_action_result x current_action_invitation, per the plan's ME row:
    ON = "当前欢迎一个行动选项，且候选含过去动作及结果/机制"."""

    shared = _shared_descriptor_fields(
        source=MemorySource.ME, query=current_user_text, source_items=source_items,
        selected_items=selected_items, session_index=session_index,
    )
    readiness = observe_action_readiness(current_user_text)
    past_action_result = (
        compile_atomic_reusable_outcome(candidate_text) is not None
        if candidate_text else None
    )
    redundant = bool(candidate_text) and _is_redundant(current_user_text, candidate_text)
    return ContributionSlotObservation(
        component="ME", **shared, current_redundant=redundant,
        past_action_result=past_action_result,
        current_action_invitation=(readiness is ActionReadiness.INVITES_ACTION),
        current_action_readiness=readiness.value,
    )


# MS turns asking for continuity with the past ("last time", "like I said
# before", "still the same issue", "we talked about this") -- a new,
# not-yet-independently-validated heuristic. Deliberately narrow and
# conservative rather than broad, matching this project's established
# fail-closed convention for new regex signals.
_CONTINUITY_REQUEST_RE = re.compile(
    r"\b(?:last time|like I (?:said|mentioned|told you) before|"
    r"(?:we|you and I) (?:talked|spoke) about this (?:before|already)|"
    r"still (?:the )?same (?:issue|problem|thing)|"
    r"remember (?:when|what) I|as I (?:said|mentioned) (?:before|earlier)|"
    r"back to what I (?:said|mentioned))\b",
    re.IGNORECASE,
)


def ms_contribution_slots(
    *, current_user_text: str, candidate_text: str | None,
    source_items: Sequence[MemoryItem], selected_items: Sequence[MemoryItem],
    session_index: int,
) -> ContributionSlotObservation:
    """continuity_request x specific_prior_observation, per the plan's MS
    row: ON = "用户明确要连续性、旧观察、未完成线程或一个具体区分能回答
    当前问题"."""

    shared = _shared_descriptor_fields(
        source=MemorySource.MS, query=current_user_text, source_items=source_items,
        selected_items=selected_items, session_index=session_index,
    )
    has_observation = (
        compile_atomic_session_observation(candidate_text) is not None
        if candidate_text else None
    )
    redundant = bool(candidate_text) and _is_redundant(current_user_text, candidate_text)
    return ContributionSlotObservation(
        component="MS", **shared, current_redundant=redundant,
        has_specific_prior_observation=has_observation,
        continuity_request=bool(_CONTINUITY_REQUEST_RE.search(current_user_text)),
    )


# MP profile/constraint eligibility: does the current turn's shape actually
# need advice/arrangement (where a constraint could change feasible
# content), as opposed to only wanting to be heard. Reuses the same
# advice-welcome signal ME reuses from RS's runtime, for the same reason
# (one validated judgment, not two possibly-inconsistent ones).


def mp_contribution_slots(
    *, current_user_text: str, candidate_text: str | None, candidate_is_preference: bool,
    source_items: Sequence[MemoryItem], selected_items: Sequence[MemoryItem],
    session_index: int,
) -> ContributionSlotObservation:
    """Per the plan's two MP rows: preference ON = "当前回复动作确实可被该
    格式/负担偏好改变"; profile/constraint ON = "当前目标需要建议或安排，
    约束会改变可行内容"."""

    shared = _shared_descriptor_fields(
        source=MemorySource.MP, query=current_user_text, source_items=source_items,
        selected_items=selected_items, session_index=session_index,
    )
    flags = observable_flags(_dialogue_from_current_text(current_user_text))
    redundant = bool(candidate_text) and _is_redundant(current_user_text, candidate_text)
    return ContributionSlotObservation(
        component="MP", **shared, current_redundant=redundant,
        candidate_is_preference=candidate_is_preference,
        preference_applies_to_response_act=(
            not flags["listen_only"] if candidate_is_preference else None
        ),
        profile_goal_needs_advice_or_arrangement=(
            flags["explicit_advice_welcome"] if not candidate_is_preference else None
        ),
    )


def rs_contribution_slots(
    *, current_user_text: str, recent_dialogue: Sequence[dict[str, Any]],
    move_id: str, already_executed_last_turn: bool,
) -> ContributionSlotObservation:
    """Per the plan's RS row: ON = "卡片的atomic move与当前请求/边界匹配
    且未在上一轮执行". RS has no memory catalog, so the shared descriptor
    fields (age/margin/tokens) do not apply -- reuses the SAME eligibility
    runtime already validated for real RS retrieval this session
    (v1_5_strategy_rag_runtime.observable_flags/eligible_moves), not a new
    judgment."""

    from .v1_5_strategy_rag_runtime import eligible_moves  # noqa: PLC0415

    flags = observable_flags(recent_dialogue)
    eligible = set(eligible_moves(flags))
    return ContributionSlotObservation(
        component="RS", candidate_present=True,
        card_precondition_met=move_id in eligible,
        card_already_executed_last_turn=already_executed_last_turn,
    )
