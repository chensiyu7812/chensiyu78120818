"""Unified candidate-layer responsibility split (2026-08-06).

Per the plan doc section 2.2, hard eligibility should only block machine-
provable errors: candidate absent, wrong owner, version/time invalid,
explicit current refusal, explicit repeat/already-executed. Semantic
opportunity judgments (does this turn welcome advice, is this a good moment
to invite one optional step) belong in Step1 as an input feature, not as a
prerequisite gate that empties the candidate pool before Step1 ever sees it.

Audited where this project currently stands across all four components
(see PM_V1_5_V5_3_CANDIDATE_LAYER_RESPONSIBILITY_SPLIT_20260806_ZH.md for
the full audit): MP/MS/ME's discover_final_typed_memory_candidates()
(v1_5_candidate_discovery.py) already only hard-gates on
match_level(item) > 0.0 (a mechanical content-overlap floor) plus exact-text
dedup -- explicit_advice_welcome / current_action_invitation /
continuity_request are computed only as contribution_slot OUTPUT features
downstream, never as a discovery-time filter. That part needed no change.

RS is the real offender: QualifiedStrategyRAG.retrieve()
(v1_5_strategy_rag_runtime.py) calls eligible_moves(), and when it returns
empty for ANY reason other than active_high_stakes, the status is
"off_no_explicit_opportunity" and the candidate list is empty -- Step1 never
gets a chance to see or weigh anything. effect_study_rank_applicable_cards()
(v1_5_strategy_rag_repair.py) has the same shape one layer up: it filters
which strategy FAMILIES are even ranked based on advice_welcome/listen_only/
no_probing, before any Step1 judgment.

This module adds an ADDITIVE alternative for the 6-card system only
(matching this project's established pattern: new module, do not modify
existing frozen call sites in place). It separates:
  - mechanical hard-off: active_high_stakes (kept hard for safety, see the
    module docstring note below) and explicit_stop (an explicit current
    refusal, allowed by the plan's own list) as the only pool-emptying
    conditions, plus per-move already-executed-last-turn exclusion;
  - the semantic opportunity flags (explicit_advice_welcome, listen_only,
    one_low_risk_step_available, etc.) as ATTACHED FEATURES on every
    remaining candidate move, not a pre-filter -- Step1 (or, until a
    learned head exists, a transparent-rule baseline) decides.

Safety-critical flags (active_high_stakes / violence / self-harm) are kept
as mechanical hard-offs here, not demoted to a feature, even though the
plan's five allowed hard-off categories do not explicitly name "safety
redirect." This project consistently treats active_high_stakes as an
early-return safety mechanism elsewhere (v1_5_strategy_rag_runtime.py's own
retrieve(), effect_study_observable_flags()'s ordinary_rag_hard_off). This
module keeps that convention rather than silently overriding it, but flags
it explicitly as a leader decision, not something settled unilaterally here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import StrategyCard
from .v1_5_strategy_rag_runtime import (
    MOVE_IDS,
    observable_flags,
    query_text,
)
from .text import lexical_score


@dataclass(frozen=True)
class RSCandidateObservation:
    move_id: str
    strategy_card: StrategyCard
    lexical_relevance: float
    already_executed_last_turn: bool
    # semantic opportunity features -- inputs to Step1, not a discovery gate
    explicit_advice_welcome: bool
    listen_only: bool
    one_low_risk_step_available: bool
    open_expression_opportunity: bool
    focused_clarification_opportunity: bool
    paraphrase_check_opportunity: bool
    grounded_validation_opportunity: bool
    question_repetition_block: bool


@dataclass(frozen=True)
class RSCandidatePool:
    hard_off: bool
    hard_off_reason: str | None
    observable_flags: dict[str, bool]
    candidates: tuple[RSCandidateObservation, ...]


@dataclass(frozen=True)
class RSSharedRank1:
    """The single RS execution candidate shared by every policy arm.

    ``transparent_priority`` means the narrow observable rule identified a
    specific move.  ``lexical_fallback`` means no such signal fired, but the
    mechanical pool remains available so the learned PM can still decide
    whether the best common candidate is worth opening.  Policy arms may
    change only the ON/OFF bit; they may not substitute another card.
    """

    observation: RSCandidateObservation
    selection_mode: str
    transparent_rule_on: bool


def rs_mechanical_candidate_pool(
    *,
    recent_dialogue: Sequence[Mapping[str, Any] | object],
    cards: Sequence[StrategyCard],
    already_executed_move_ids: Sequence[str] = (),
) -> RSCandidatePool:
    """Mechanical-only RS candidate discovery for the 6-card system.

    Unlike QualifiedStrategyRAG.retrieve(), this NEVER empties the pool
    because of explicit_advice_welcome/listen_only/no_probing -- those
    become per-candidate features. The pool is empty only for
    active_high_stakes (safety) or explicit_stop (an explicit current
    refusal, one of the plan's allowed hard-off categories); a move is
    dropped individually if it was already executed last turn (also an
    allowed category), not the whole component.
    """

    flags = observable_flags(recent_dialogue)
    if flags["active_high_stakes"]:
        return RSCandidatePool(True, "active_high_stakes", flags, ())
    if flags["explicit_stop"]:
        return RSCandidatePool(True, "explicit_stop", flags, ())

    by_move = {card.strategy_label: card for card in cards}
    missing = MOVE_IDS - set(by_move)
    if missing:
        raise ValueError(f"missing required strategy card(s): {sorted(missing)}")

    query = query_text(recent_dialogue)
    already_executed = set(already_executed_move_ids)
    observations = []
    for move_id in sorted(MOVE_IDS):
        if move_id in already_executed:
            continue
        card = by_move[move_id]
        observations.append(
            RSCandidateObservation(
                move_id=move_id,
                strategy_card=card,
                lexical_relevance=lexical_score(query, card.retrieval_text),
                already_executed_last_turn=False,
                explicit_advice_welcome=flags["explicit_advice_welcome"],
                listen_only=flags["listen_only"],
                one_low_risk_step_available=flags["one_low_risk_step_available"],
                open_expression_opportunity=flags["open_expression_opportunity"],
                focused_clarification_opportunity=flags["focused_clarification_opportunity"],
                paraphrase_check_opportunity=flags["paraphrase_check_opportunity"],
                grounded_validation_opportunity=flags["grounded_validation_opportunity"],
                question_repetition_block=flags["question_repetition_block"],
            )
        )
    return RSCandidatePool(False, None, flags, tuple(observations))


def rs_transparent_rule_top1(pool: RSCandidatePool) -> RSCandidateObservation | None:
    """A transparent-rule baseline over the mechanical pool -- NOT a
    replacement for a learned Step1 head. Reuses the same semantic priority
    order eligible_moves() encoded (repetition-safe open expression, then
    focused clarification, then paraphrase, then grounded validation, then
    one low-risk step), but as a ranking preference over an always-nonempty
    pool rather than a pool-emptying filter. Ties broken by lexical
    relevance then move_id for determinism."""

    if pool.hard_off or not pool.candidates:
        return None

    # Each rank tier names ONE specific move -- the flags are turn-level
    # (identical across all 6 candidates), so the rank must be keyed by
    # move_id too, or every candidate that happens to share a True flag
    # would tie at the same rank and the outcome would be decided by the
    # lexical/move_id tie-break instead of the actual matched opportunity.
    def priority(obs: RSCandidateObservation) -> tuple[int, float, str]:
        rank = 0
        if (
            obs.move_id == "AM01_invite_open_expression"
            and obs.open_expression_opportunity
            and not obs.question_repetition_block
        ):
            rank = 5
        elif (
            obs.move_id == "AM02_ask_one_focused_clarification"
            and obs.focused_clarification_opportunity
            and not obs.question_repetition_block
            and not obs.listen_only
        ):
            rank = 4
        elif obs.move_id == "AM04_tentative_paraphrase_check" and obs.paraphrase_check_opportunity:
            rank = 3
        elif obs.move_id == "AM05_grounded_validation" and obs.grounded_validation_opportunity:
            rank = 2
        elif (
            obs.move_id == "AM10_offer_one_optional_micro_step"
            and obs.explicit_advice_welcome
            and obs.one_low_risk_step_available
            and not obs.listen_only
        ):
            rank = 1
        return (rank, obs.lexical_relevance, obs.move_id)

    ranked = sorted(pool.candidates, key=priority, reverse=True)
    top = ranked[0]
    if priority(top)[0] == 0:
        return None
    return top


def rs_shared_candidate_top1(pool: RSCandidatePool) -> RSSharedRank1 | None:
    """Select one common Rank-1 without turning low-recall semantics into a gate.

    When the transparent opportunity rules identify a move, that move is the
    shared candidate for *all* policy arms.  Otherwise a deterministic lexical
    fallback keeps one candidate visible to fixed-high and learned-PM while
    the transparent-rule baseline stays OFF.  This separates candidate
    identity from the policy decision and prevents a baseline from winning by
    receiving a different card.
    """

    if pool.hard_off or not pool.candidates:
        return None
    transparent = rs_transparent_rule_top1(pool)
    if transparent is not None:
        return RSSharedRank1(
            observation=transparent,
            selection_mode="transparent_priority",
            transparent_rule_on=True,
        )
    lexical = max(
        pool.candidates,
        key=lambda obs: (obs.lexical_relevance, obs.move_id),
    )
    return RSSharedRank1(
        observation=lexical,
        selection_mode="lexical_fallback",
        transparent_rule_on=False,
    )
