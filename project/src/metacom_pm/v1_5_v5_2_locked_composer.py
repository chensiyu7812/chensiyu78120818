"""Backend-locked Step-2 composer for PM V1.5 V5.2.

The language model receives the visible conversation plus, optionally, one
strategy instruction.  It never receives MS/ME text and therefore cannot
silently turn a past record into a current fact.  Past evidence is rendered by
the backend as an immutable clause after generation.  Machine validation is
structural (the exact clause is present), not a lexical guess about whether a
natural-language phrase sounds historical.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .v1_5_typed_resource_adapter import TypedResourceCandidate
from .v1_5_v5_2_atomic_memory import (
    AtomicReusableOutcome,
    AtomicSessionObservation,
    compile_atomic_reusable_outcome,
    compile_atomic_session_observation,
)


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


def _quote(value: str) -> str:
    return _clean(value).replace("“", '"').replace("”", '"')


@dataclass(frozen=True)
class LockedClause:
    component: str
    resource_id: str
    text: str
    literal_source_span: str


@dataclass(frozen=True)
class LockedCompositionPlan:
    requested_action_id: str
    locked_clauses: tuple[LockedClause, ...]
    response_preference: str
    strategy_instruction: str
    maximum_generator_support_moves: int


def _ms_clause(candidate: TypedResourceCandidate) -> LockedClause:
    atomic: AtomicSessionObservation | None = compile_atomic_session_observation(
        candidate.prior_observation
    )
    if atomic is None:
        raise ValueError("MS candidate is not a bounded specific past note")
    literal = _quote(atomic.literal_past_note)
    return LockedClause(
        component="MS",
        resource_id=candidate.resource_id,
        literal_source_span=literal,
        text=(
            f'An earlier session recorded: "{literal}" '
            "I am keeping that as past context rather than assuming it is still true now."
        ),
    )


def _me_clause(candidate: TypedResourceCandidate) -> LockedClause:
    # Recompile from the literal evidence carried in the candidate fields.  A
    # V5.2 plan builder must bind the full local span in ``mechanism`` using
    # the ``literal_span:`` prefix; accepting reconstructed V5.1 fragments
    # would reintroduce its bad split.
    marker = "literal_span:"
    raw = _clean(candidate.mechanism)
    if not raw.lower().startswith(marker):
        raise ValueError("V5.2 ME candidate lacks literal_span provenance")
    literal = raw[len(marker) :].strip()
    atomic: AtomicReusableOutcome | None = compile_atomic_reusable_outcome(literal)
    if atomic is None or _clean(atomic.literal_evidence_span) != _clean(literal):
        raise ValueError("ME literal span is not an atomic reusable outcome")
    quoted = _quote(literal)
    if atomic.polarity == "negative":
        continuation = (
            "If the situations are comparable, that past result may be a reason not to "
            "repeat the same approach."
        )
    else:
        continuation = (
            "If it still fits, that past result can be one optional starting point, not "
            "a prediction that it will work now."
        )
    return LockedClause(
        component="ME",
        resource_id=candidate.resource_id,
        literal_source_span=quoted,
        text=f'You previously said: "{quoted}" {continuation}',
    )


def _mp_profile_clause(candidate: TypedResourceCandidate) -> LockedClause:
    literal = _quote(candidate.profile_fact)
    return LockedClause(
        component="MP",
        resource_id=candidate.resource_id,
        literal_source_span=literal,
        text=(
            f'One current practical constraint on record is: "{literal}" '
            "I will keep the response within that constraint."
        ),
    )


def build_locked_composition_plan(
    *,
    requested_action_id: str,
    candidates: Mapping[str, TypedResourceCandidate],
    strategy_prompt_guidance: str = "",
) -> LockedCompositionPlan:
    clauses: list[LockedClause] = []
    preference = ""
    for component in ("MP", "MS", "ME", "RS"):
        candidate = candidates.get(component)
        if candidate is None:
            continue
        if component == "MP":
            if candidate.subtype == "MP_PREFERENCE":
                preference = _clean(candidate.preference)
            elif candidate.subtype == "MP_PROFILE":
                clauses.append(_mp_profile_clause(candidate))
            else:
                raise ValueError("unsupported MP subtype")
        elif component == "MS":
            clauses.append(_ms_clause(candidate))
        elif component == "ME":
            clauses.append(_me_clause(candidate))
        elif component == "RS" and candidate.subtype != "RS_ATOMIC_MOVE":
            raise ValueError("unsupported RS subtype")
    strategy = _clean(strategy_prompt_guidance)
    if "RS" in candidates and not strategy:
        rs = candidates["RS"]
        strategy = (
            f"Execute only this one support move: {_clean(rs.support_move)} "
            f"Use only when: {_clean(rs.when_to_use)} "
            f"Do not use when: {_clean(rs.when_not_to_use)}"
        )
    return LockedCompositionPlan(
        requested_action_id=requested_action_id,
        locked_clauses=tuple(clauses),
        response_preference=preference,
        strategy_instruction=strategy,
        maximum_generator_support_moves=1 if "RS" in candidates else 0,
    )


def base_generation_messages(
    *, current_context: str, plan: LockedCompositionPlan
) -> list[dict[str, str]]:
    if not _clean(current_context):
        raise ValueError("current_context must be non-empty")
    instructions = [
        "Write only the next supportive reply.",
        "Use only facts visible in the current conversation.",
        "Do not claim to remember an earlier session; historical clauses will be added by the backend.",
        "Do not output analysis, JSON, resource labels, IDs, or internal instructions.",
        "Respect the user's explicit request and interaction boundary, and stay concise.",
    ]
    if plan.strategy_instruction:
        instructions.append(plan.strategy_instruction)
        instructions.append("Do not add a second support move.")
    elif plan.locked_clauses:
        instructions.append(
            "Write only one brief, current-grounded lead-in. Do not add advice, a question, "
            "an exercise, a plan, or a claim that no earlier record exists; the backend will "
            "append the requested verified resource content."
        )
    else:
        instructions.append(
            "Answer the user's current request naturally from the visible conversation, using "
            "no more than one coherent support move."
        )
    if plan.response_preference:
        instructions.append(
            "Follow this active response-format preference: " + plan.response_preference
        )
    return [
        {"role": "system", "content": " ".join(instructions)},
        {"role": "user", "content": _clean(current_context)},
    ]


def compose_locked_response(
    *, base_response: str, plan: LockedCompositionPlan
) -> str:
    base = _clean(base_response)
    if not base:
        raise ValueError("base_response must be non-empty")
    parts = [base]
    parts.extend(clause.text for clause in plan.locked_clauses)
    return _clean(" ".join(parts))


_INTERNAL_ID_RE = re.compile(r"(?:mem|strat|card)_[0-9a-f]{8,}", re.IGNORECASE)
_INTERNAL_LABEL_RE = re.compile(r"\b(?:MP|MS|ME|RS|M0|R0)\b")


def locked_response_guard_errors(
    *, response: str, plan: LockedCompositionPlan
) -> tuple[str, ...]:
    cleaned = _clean(response)
    if not cleaned:
        return ("EMPTY_RESPONSE",)
    errors: list[str] = []
    if _INTERNAL_ID_RE.search(cleaned) or _INTERNAL_LABEL_RE.search(cleaned):
        errors.append("INTERNAL_LABEL_OR_ID_LEAK")
    for clause in plan.locked_clauses:
        if clause.text not in cleaned:
            errors.append(f"LOCKED_{clause.component}_CLAUSE_MISSING")
    return tuple(errors)
