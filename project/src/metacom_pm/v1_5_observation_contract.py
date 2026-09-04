"""Auditable semantic-observation contract for PM V1.5.

Observation is deliberately upstream of the learned router.  Human H1 evidence
codes provide factor-level audit targets; they are never model inputs.  Runtime
observations must be computed from the visible state and exact rank-1 candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping, Sequence

from .text import content_word_match_level, normalize_for_hash, normalize_space


OBSERVATION_CONTRACT_PROTOCOL = "pm-v1.5-semantic-observation-contract-v1"

FactorName = Literal[
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "currently_nonredundant",
    "specific_increment",
]
FactorLabel = Literal[0, 1]
GateDecision = Literal["allow", "deny", "unknown"]

FACTOR_NAMES: tuple[FactorName, ...] = (
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "currently_nonredundant",
    "specific_increment",
)

_POSITIVE_CODES: dict[FactorName, frozenset[str]] = {
    "owner_time_entity_valid": frozenset({"OWNER_AND_TIME_VALID"}),
    "goal_function_fit": frozenset({"CURRENT_GOAL_FIT"}),
    "boundary_burden_compatible": frozenset({"SAFE_AND_BOUNDARY_COMPATIBLE"}),
    "currently_nonredundant": frozenset({"SPECIFIC_FUNCTIONAL_INCREMENT"}),
    "specific_increment": frozenset({"SPECIFIC_FUNCTIONAL_INCREMENT"}),
}

_NEGATIVE_CODES: dict[FactorName, frozenset[str]] = {
    "owner_time_entity_valid": frozenset(
        {
            "WRONG_OWNER_OR_ENTITY",
            "TIME_STALE_OR_CONFLICTING",
            "STALE_OR_CONFLICTING",
        }
    ),
    "goal_function_fit": frozenset(
        {
            "WRONG_ENTITY_OR_GOAL",
            "WRONG_GOAL_OR_FUNCTION",
            "WRONG_SUBTYPE_OR_FUNCTION",
        }
    ),
    "boundary_burden_compatible": frozenset(
        {"BOUNDARY_OR_BURDEN_CONFLICT"}
    ),
    "currently_nonredundant": frozenset({"CURRENTLY_REDUNDANT"}),
    "specific_increment": frozenset(
        {"CURRENTLY_REDUNDANT", "GENERIC_OR_NO_INCREMENT"}
    ),
}


def factor_gold_from_evidence_codes(
    evidence_codes: Sequence[str],
) -> dict[FactorName, FactorLabel | None]:
    """Map H1 reason codes to separately auditable factor targets.

    Missing support is unknown, not a negative.  Contradictory code sets are
    rejected so that an annotation cannot silently supervise both directions.
    """

    codes = frozenset(str(code) for code in evidence_codes)
    targets: dict[FactorName, FactorLabel | None] = {}
    for factor in FACTOR_NAMES:
        positive = bool(codes & _POSITIVE_CODES[factor])
        negative = bool(codes & _NEGATIVE_CODES[factor])
        if positive and negative:
            raise ValueError(f"contradictory evidence codes for {factor}")
        targets[factor] = 1 if positive else 0 if negative else None
    return targets


@dataclass(frozen=True)
class SemanticObservationRecord:
    state_id: str
    component: str
    factor_builder_protocol: str
    factor_scores: Mapping[FactorName, float]
    evidence_codes_read: bool = False
    final_decision_read: bool = False
    construction_intent_read: bool = False
    outcome_read: bool = False

    def __post_init__(self) -> None:
        if not self.state_id or not self.component or not self.factor_builder_protocol:
            raise ValueError("observation identity fields must be non-empty")
        if (
            self.evidence_codes_read
            or self.final_decision_read
            or self.construction_intent_read
            or self.outcome_read
        ):
            raise ValueError("semantic observation read a forbidden supervision field")
        if set(self.factor_scores) != set(FACTOR_NAMES):
            raise ValueError("semantic observation factor schema mismatch")
        for name, value in self.factor_scores.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"factor {name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"factor {name} must be in [0, 1]")


@dataclass(frozen=True)
class StructuredCandidateMetadata:
    """Backend facts that must not be guessed from conversational prose."""

    candidate_present: bool
    state_owner_id: str | None
    candidate_owner_id: str | None
    candidate_active: bool | None
    candidate_superseded: bool | None
    candidate_content_sha256: str | None = None
    visible_fact_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuredEligibilityObservation:
    """Three-way pre-Step1 eligibility with unknown mapped to runtime OFF."""

    state_id: str
    component: str
    gate_decisions: Mapping[str, GateDecision]
    runtime_eligible: bool
    unknown_forced_off: bool
    eligibility_is_step1_gold: bool = False

    def __post_init__(self) -> None:
        expected = {
            "owner_time_entity_valid",
            "goal_function_fit",
            "boundary_burden_compatible",
            "specific_increment",
        }
        if set(self.gate_decisions) != expected:
            raise ValueError("structured eligibility gate schema mismatch")
        if self.eligibility_is_step1_gold:
            raise ValueError("eligibility must not be promoted to Step1 gold")
        expected_runtime = all(
            decision == "allow" for decision in self.gate_decisions.values()
        )
        if self.runtime_eligible != expected_runtime:
            raise ValueError("runtime eligibility must be the conjunction of four allows")
        expected_unknown_off = (
            not expected_runtime
            and any(decision == "unknown" for decision in self.gate_decisions.values())
        )
        if self.unknown_forced_off != expected_unknown_off:
            raise ValueError("unknown_forced_off is inconsistent with gate decisions")


@dataclass(frozen=True)
class StructuredPolicyObservation:
    """Objective denials plus non-label semantic uncertainty for Step1."""

    state_id: str
    component: str
    hard_gate_decisions: Mapping[str, GateDecision]
    semantic_feature_decisions: Mapping[str, GateDecision]
    step1_candidate_admissible: bool
    hard_denial_reasons: tuple[str, ...]
    eligibility_is_step1_gold: bool = False

    def __post_init__(self) -> None:
        if set(self.hard_gate_decisions) != {
            "candidate_present",
            "owner_time_entity_valid",
            "explicit_boundary_compatible",
        }:
            raise ValueError("structured policy hard-gate schema mismatch")
        if set(self.semantic_feature_decisions) != {
            "goal_function_fit",
            "specific_increment",
            "boundary_evidence",
        }:
            raise ValueError("structured policy semantic-feature schema mismatch")
        expected_reasons = tuple(
            name
            for name, decision in self.hard_gate_decisions.items()
            if decision != "allow"
        )
        if self.hard_denial_reasons != expected_reasons:
            raise ValueError("hard denial reasons do not match hard gates")
        if self.step1_candidate_admissible != (not expected_reasons):
            raise ValueError("Step1 admissibility must depend only on hard gates")
        if self.eligibility_is_step1_gold:
            raise ValueError("candidate admissibility must not become Step1 gold")


_FRIEND_FOCUS_RE = re.compile(
    r"\bmy (?:friend|coworker|colleague|sibling|brother|sister|partner|"
    r"spouse|child|parent)\b",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"\b(?:please end|end this conversation|stop (?:here|now)|goodbye|bye)\b",
    re.IGNORECASE,
)
_ROUTINE_CLOSING_RE = re.compile(
    r"^(?:thanks|thank you)(?: for listening)?[.! ]*$", re.IGNORECASE
)
_BRIEF_RE = re.compile(
    r"\b(?:brief|concise|one point|one small|not a list|avoid a list|"
    r"no follow-up|avoid .*follow-up|keep .*short)\b",
    re.IGNORECASE,
)
_NO_QUESTION_RE = re.compile(
    r"\b(?:do not repeat|don't repeat|no (?:more )?questions?|"
    r"avoid .*questions?|without .*questions?)\b",
    re.IGNORECASE,
)
_LISTEN_ONLY_RE = re.compile(
    r"\b(?:please listen|just listen|only listen|need space to say|"
    r"before deciding what to do)\b",
    re.IGNORECASE,
)
_ADVICE_WELCOME_RE = re.compile(
    r"\b(?:open to advice|one .*idea|one .*option|suggestion .*welcome|"
    r"what (?:should|could|can) i do|place to begin)\b",
    re.IGNORECASE,
)
_FACT_RECALL_RE = re.compile(
    r"\b(?:what (?:was|did we) (?:record|note)|remind me (?:what|which)|"
    r"brief factual reminder)\b",
    re.IGNORECASE,
)
_UNCLEAR_RE = re.compile(
    r"\b(?:cannot tell which|can't tell which|tangled|unclear|uncertainty|"
    r"what part|which part)\b",
    re.IGNORECASE,
)
_EMOTION_RE = re.compile(
    r"\b(?:feel|feeling|emotion|hurt|sad|afraid|anxious|pressure|lonely|"
    r"grief|frustrat|overwhelm|tension|reaction|hard to name|accurate words)\w*\b",
    re.IGNORECASE,
)
_RESOLVED_RE = re.compile(
    r"\b(?:resolved|completed|fully settled|no longer a problem|closed)\b",
    re.IGNORECASE,
)
_GENERIC_MEMORY_RE = re.compile(
    r"\b(?:felt difficult and uncertain|general .* concern|"
    r"without a new present-day conclusion|present in the background|"
    r"pursued a logistical comparison rather than)\b",
    re.IGNORECASE,
)
_RESULT_RE = re.compile(
    r"\b(?:helped|worked|eased|reduced|made .* easier|less rushed|"
    r"more manageable|clarified)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(?:wrote|tried|took|used|made|asked|paused|walked|"
    r"brief pause|one .* note)\b",
    re.IGNORECASE,
)

_EXPLICIT_STALE_STORED_RE = re.compile(
    r"\b(?:stored|old|earlier|previous).{0,45}(?:preference|record|detail).{0,50}"
    r"(?:out of date|withdrawn|replaced|superseded|no longer current|stale|fully settled|resolved)\b|"
    r"\b(?:preference|record|detail).{0,45}(?:explicitly )?"
    r"(?:withdrawn|replaced|superseded|no longer current|stale)\b",
    re.IGNORECASE,
)
_EXPLICIT_OTHER_OWNER_RE = re.compile(
    r"\b(?:question concerns|present situation belongs to|person i am asking about is|"
    r"subject now).{0,55}(?:friend|coworker|colleague|sibling|brother|sister|partner|"
    r"someone close).{0,80}(?:not me|rather than (?:to )?me|stored record describes me|"
    r"my own history has the wrong owner)\b|"
    r"\bmy (?:friend|coworker|colleague|sibling|brother|sister|partner)'s (?:history|record)\b",
    re.IGNORECASE,
)
_EXPLICIT_RESOURCE_REFUSAL_RE = re.compile(
    r"\b(?:do not|don't|avoid|leave\b).{0,35}(?:use|give|reuse|include|add)?\s*"
    r"(?:one )?(?:relevant stored profile detail|tentative reference to an earlier session|"
    r"optional past approach|retrieved atomic support move|advice|suggestion|question|"
    r"reflection|paraphrase)\b|"
    r"\bnot consenting to (?:one |the )?(?:relevant stored profile detail|"
    r"tentative reference to an earlier session|optional past approach|"
    r"retrieved atomic support move)\b",
    re.IGNORECASE,
)
_EXPLICIT_RESOURCE_PERMISSION_RE = re.compile(
    r"\b(?:comfortable with|comfortable considering|allowed if|"
    r"would stay within my limits to use|boundary leaves room for|"
    r"present boundary leaves room for|using).{0,45}"
    r"(?:relevant stored profile detail|tentative reference to an earlier session|"
    r"optional past approach|retrieved atomic support move)\b",
    re.IGNORECASE,
)
_EXPLICIT_REDUNDANT_RE = re.compile(
    r"\b(?:current exchange already contains this same information|"
    r"current dialogue already contains|already stated the candidate's exact point|"
    r"relevant detail is already visible here|nothing new is needed from the candidate)\b",
    re.IGNORECASE,
)
_EXPLICIT_INCREMENT_RE = re.compile(
    r"\b(?:exact candidate detail is absent|current dialogue does not yet contain|"
    r"specific fact in the candidate would be new|concrete stored detail may add something|"
    r"candidate's concrete information is absent)\b",
    re.IGNORECASE,
)


def _rs_sections(candidate_text: str) -> tuple[str, str, str]:
    match = re.search(
        r"support_move:\s*(.*?)\s*when_to_use:\s*(.*?)\s*when_not_to_use:\s*(.*)",
        candidate_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return candidate_text, "", ""
    return tuple(normalize_space(value) for value in match.groups())  # type: ignore[return-value]


def _same_subject_and_time(current: str, candidate: str, component: str) -> float:
    if component == "RS":
        return 1.0
    if _FRIEND_FOCUS_RE.search(current) and re.search(
        r"\b(?:the user|user personally|\bi\b|\bmy\b)", candidate, re.IGNORECASE
    ):
        return 0.0
    if not _FRIEND_FOCUS_RE.search(current) and re.search(
        r"\b(?:belonged to the user's friend|my friend (?:tried|used|wrote|asked))\b",
        candidate,
        re.IGNORECASE,
    ):
        return 0.0
    if _RESOLVED_RE.search(candidate) and re.search(
        r"\b(?:again|returned|still|current|now|this week|today)\b",
        current,
        re.IGNORECASE,
    ):
        return 0.0
    return 1.0


def _memory_boundary(current: str, candidate: str, component: str) -> float:
    if _STOP_RE.search(current) or _ROUTINE_CLOSING_RE.match(current.strip()):
        return 0.0
    if component == "MP" and re.search(
        r"several detailed suggestions|follow-up questions", candidate, re.IGNORECASE
    ) and (_BRIEF_RE.search(current) or _LISTEN_ONLY_RE.search(current)):
        return 0.0
    return 1.0


def _mp_factors(current: str, candidate: str, subtype: str) -> dict[FactorName, float]:
    owner = _same_subject_and_time(current, candidate, "MP")
    boundary = _memory_boundary(current, candidate, "MP")
    overlap = content_word_match_level(current, candidate)
    body = normalize_space(candidate.split(":", 1)[-1])
    if subtype == "MP_PREFERENCE":
        detailed = bool(
            re.search(r"several detailed suggestions|follow-up questions", candidate, re.I)
        )
        goal = float(boundary == 1.0 and overlap > 0.0)
        preference_claim = re.sub(r"^prefers\s+", "", body, flags=re.I)
        preference_terms = {
            term
            for term in (
                "advice",
                "option",
                "question",
                "brief",
                "concise",
                "reflection",
                "factual",
                "reminder",
                "directive",
                "response",
            )
            if re.search(rf"\b{re.escape(term)}s?\b", candidate, re.I)
            and re.search(rf"\b{re.escape(term)}s?\b", current, re.I)
        }
        explicitly_repeated = bool(
            preference_claim
            and normalize_for_hash(preference_claim) in normalize_for_hash(current)
        ) or len(preference_terms) >= 2
        nonredundant = float(not explicitly_repeated)
        specific = min(owner, boundary, goal, nonredundant)
    else:
        generic_current_context = bool(re.search(r"current known context", candidate, re.I))
        practical_constraint = bool(
            re.search(r"^stable .{1,80} constraint:", candidate, re.I)
            or re.search(r"stable practical constraint|education:|job:", candidate, re.I)
        )
        goal = float(owner == 1.0 and overlap > 0.0)
        tail = normalize_space(body.split(",", 1)[-1])
        repeated_tail = bool(
            tail and normalize_for_hash(tail) in normalize_for_hash(current)
        )
        nonredundant = float(
            not generic_current_context and practical_constraint and not repeated_tail
        )
        specific = min(owner, boundary, goal, nonredundant)
    return {
        "owner_time_entity_valid": owner,
        "goal_function_fit": goal,
        "boundary_burden_compatible": boundary,
        "currently_nonredundant": nonredundant,
        "specific_increment": specific,
    }


def _ms_factors(current: str, candidate: str) -> dict[FactorName, float]:
    owner = _same_subject_and_time(current, candidate, "MS")
    boundary = _memory_boundary(current, candidate, "MS")
    generic = bool(_GENERIC_MEMORY_RE.search(candidate))
    recalled_fact = bool(_FACT_RECALL_RE.search(current))
    topic_fit = content_word_match_level(current, candidate) > 0.0
    wrong_function = bool(
        re.search(r"pursued a logistical comparison rather than", candidate, re.I)
        and not recalled_fact
    )
    goal = float(owner == 1.0 and topic_fit and not wrong_function)
    repeated_distinction = bool(
        re.search(r"emotional uncertainty.{0,40}practical workload", candidate, re.I)
        and re.search(r"emotional uncertainty.{0,40}practical workload", current, re.I)
    )
    nonredundant = float(not generic and not repeated_distinction)
    has_distinction_or_result = bool(
        _RESULT_RE.search(candidate)
        or re.search(r"\b(?:rather than|main .* was|most .* part)\b", candidate, re.I)
    )
    specific = float(
        owner == 1.0
        and boundary == 1.0
        and goal == 1.0
        and not generic
        and has_distinction_or_result
    )
    return {
        "owner_time_entity_valid": owner,
        "goal_function_fit": goal,
        "boundary_burden_compatible": boundary,
        "currently_nonredundant": nonredundant,
        "specific_increment": specific,
    }


def _me_factors(current: str, candidate: str, subtype: str) -> dict[FactorName, float]:
    owner = _same_subject_and_time(current, candidate, "ME")
    boundary = _memory_boundary(current, candidate, "ME")
    topic_fit = content_word_match_level(current, candidate) > 0.0
    reusable = subtype == "ME_REUSABLE_OUTCOME"
    goal = float(owner == 1.0 and topic_fit and reusable)
    nonredundant = float(reusable and not bool(_GENERIC_MEMORY_RE.search(candidate)))
    action_result = bool(_ACTION_RE.search(candidate) and _RESULT_RE.search(candidate))
    specific = float(
        owner == 1.0
        and boundary == 1.0
        and goal == 1.0
        and nonredundant == 1.0
        and action_result
    )
    return {
        "owner_time_entity_valid": owner,
        "goal_function_fit": goal,
        "boundary_burden_compatible": boundary,
        "currently_nonredundant": nonredundant,
        "specific_increment": specific,
    }


def _rs_factors(
    current: str,
    candidate: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
) -> dict[FactorName, float]:
    move, when_to_use, when_not_to_use = _rs_sections(candidate)
    question = bool(re.search(r"\bask\b|\bquestion\b", move, re.I))
    advice = bool(re.search(r"\boffer\b|\bsuggest|experiment|adjustment", move, re.I))
    reflection = bool(re.search(r"reflect|emotion|feeling|paraphrase", move, re.I))
    restatement = bool(re.search(r"restate|condense|summar", move, re.I))
    communication = bool(re.search(r"conversation|communication|repair", when_to_use, re.I))
    current_interpersonal = bool(
        re.search(
            r"\b(?:friend|partner|spouse|family|coworker|colleague|"
            r"relationship|trust|conversation|message)\b",
            current,
            re.I,
        )
    )
    explicit_conversation_goal = bool(
        re.search(
            r"\b(?:begin|have|start) (?:a |the )?(?:conversation|talk)|"
            r"\b(?:talk|speak|message|write) (?:to|with) (?:them|him|her)|"
            r"\b(?:communication|repair the relationship)\b",
            current,
            re.I,
        )
    )
    hard_boundary = bool(
        _STOP_RE.search(current)
        or _ROUTINE_CLOSING_RE.match(current.strip())
        or (question and (_NO_QUESTION_RE.search(current) or _LISTEN_ONLY_RE.search(current)))
        or (advice and not _ADVICE_WELCOME_RE.search(current))
        or (_BRIEF_RE.search(current) and "do not use under an explicit low-burden" in when_not_to_use.lower())
    )
    lower_move = move.lower()
    if "low-conflict way to begin a conversation" in lower_move:
        goal = bool(_ADVICE_WELCOME_RE.search(current) and explicit_conversation_goal)
    elif "reversible adjustment to the immediate environment" in lower_move:
        goal = bool(
            _ADVICE_WELCOME_RE.search(current)
            and not re.search(r"emotional pressure rather than the practical", current, re.I)
        )
    elif advice:
        goal = bool(_ADVICE_WELCOME_RE.search(current)) and (
            not communication or current_interpersonal
        )
    elif question:
        goal = bool(
            (_UNCLEAR_RE.search(current) or re.search(r"open to a question", current, re.I))
            and not re.search(r"last focused question already", current, re.I)
        )
    elif "paraphrase the concern" in lower_move:
        goal = True
    elif reflection:
        goal = bool(_EMOTION_RE.search(current))
    elif restatement:
        if re.search(r"condense .*facts", move, re.I):
            goal = True
        elif re.search(r"change .*earlier and current|paraphrase .*concern", move, re.I):
            goal = True
        elif re.search(r"cause-and-then", move, re.I):
            goal = bool(re.search(r"\b(?:because|after|then|led to)\b", current, re.I))
        else:
            goal = bool(re.search(r"\b(?:because|after|then|led to|rather than)\b", current, re.I))
    elif re.search(r"acknowledge .*preference", move, re.I):
        goal = bool(_BRIEF_RE.search(current) or _LISTEN_ONLY_RE.search(current))
    else:
        goal = False
    prior_supporter = " ".join(
        str(turn.get("content") or "")
        for turn in visible_dialogue
        if str(turn.get("role") or turn.get("speaker") or "").lower()
        in {"assistant", "supporter"}
    )
    repeated = bool(
        (
            question
            and (
                _NO_QUESTION_RE.search(current)
                or re.search(r"last (?:focused )?question already", current, re.I)
                or (
                    "?" in prior_supporter
                    and re.search(r"already|do not ask|don't ask", current, re.I)
                )
            )
        )
        or (
            restatement
            and re.search(r"summary already|already captures|beyond repeating", current, re.I)
        )
        or (
            reflection
            and re.search(r"already named|another (?:emotional )?reflection", current, re.I)
        )
        or (
            advice
            and re.search(r"already (?:have|offered)|before adding another", current, re.I)
        )
    )
    boundary = float(not hard_boundary)
    nonredundant = float(not repeated)
    specific = float(goal and boundary == 1.0 and nonredundant == 1.0)
    return {
        "owner_time_entity_valid": 1.0,
        "goal_function_fit": float(goal),
        "boundary_burden_compatible": boundary,
        "currently_nonredundant": nonredundant,
        "specific_increment": specific,
    }


def transparent_semantic_observation(
    *,
    state_id: str,
    component: str,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    candidate_present: bool,
    candidate_text: str | None,
    candidate_subtype: str,
) -> SemanticObservationRecord:
    """Build the P1R transparent factor surface without labels or outcomes."""

    current = normalize_space(current_user_text)
    candidate = normalize_space(candidate_text or "")
    if not candidate_present:
        scores = {factor: 0.0 for factor in FACTOR_NAMES}
    elif component == "MP":
        scores = _mp_factors(current, candidate, candidate_subtype)
    elif component == "MS":
        scores = _ms_factors(current, candidate)
    elif component == "ME":
        scores = _me_factors(current, candidate, candidate_subtype)
    elif component == "RS":
        scores = _rs_factors(current, candidate, visible_dialogue)
    else:
        raise ValueError(f"unknown component: {component}")
    return SemanticObservationRecord(
        state_id=state_id,
        component=component,
        factor_builder_protocol="pm-v1.5-transparent-semantic-observation-v1",
        factor_scores=scores,
    )


def _independent_owner_time(
    current: str, candidate: str, component: str, subtype: str
) -> float:
    """Score ownership/time without consulting goal, boundary, or increment."""

    if component == "RS":
        return 1.0
    if _EXPLICIT_STALE_STORED_RE.search(current):
        return 0.0
    # A response-format preference remains the user's preference even when the
    # current story concerns another person. Other memories are person-bound.
    if subtype != "MP_PREFERENCE" and _EXPLICIT_OTHER_OWNER_RE.search(current):
        return 0.0
    return _same_subject_and_time(current, candidate, component)


def _independent_boundary(current: str, candidate: str, component: str) -> float:
    """Score present-turn permission without consulting relevance or novelty."""

    if _STOP_RE.search(current) or _ROUTINE_CLOSING_RE.match(current.strip()):
        return 0.0
    if _EXPLICIT_RESOURCE_REFUSAL_RE.search(current):
        return 0.0
    if _EXPLICIT_RESOURCE_PERMISSION_RE.search(current):
        return 1.0
    if component == "RS":
        move, _, when_not_to_use = _rs_sections(candidate)
        question = bool(re.search(r"\bask\b|\bquestion\b", move, re.I))
        advice = bool(re.search(r"\boffer\b|\bsuggest|experiment|adjustment", move, re.I))
        if question and (
            _NO_QUESTION_RE.search(current) or _LISTEN_ONLY_RE.search(current)
        ):
            return 0.0
        if advice and re.search(
            r"\b(?:no|without|do not|don't)\s+(?:advice|suggestions?)\b",
            current,
            re.I,
        ):
            return 0.0
        if _BRIEF_RE.search(current) and (
            "do not use under an explicit low-burden" in when_not_to_use.lower()
        ):
            return 0.0
    return 1.0


def _independent_goal(
    current: str,
    candidate: str,
    component: str,
    subtype: str,
) -> float:
    """Score current requested function without consulting other gates."""

    if component == "MP":
        if re.search(
            r"profile personalization is not the task|same-topic profile detail would not|"
            r"what i need now is help naming|only want the feeling named",
            current,
            re.I,
        ):
            return 0.0
        if subtype == "MP_PREFERENCE":
            return float(
                bool(
                    re.search(
                        r"shape the response around my (?:stable )?response preference|"
                        r"response-format preference|format preference",
                        current,
                        re.I,
                    )
                )
            )
        return float(
            bool(
                re.search(
                    r"actual circumstances|practical constraint|stored profile detail|"
                    r"one relevant profile detail",
                    current,
                    re.I,
                )
            )
        )
    if component == "MS":
        if re.search(
            r"same-topic earlier session would not perform that function|"
            r"help name my present reaction",
            current,
            re.I,
        ):
            return 0.0
        return float(
            bool(
                re.search(
                    r"use the one specific distinction recorded|"
                    r"which factual detail was recorded earlier|"
                    r"remind me (?:what|which)|specific earlier observation",
                    current,
                    re.I,
                )
            )
        )
    if component == "ME":
        if re.search(
            r"not which past action to reuse|do not (?:offer or )?reuse an action|"
            r"only want the feeling named",
            current,
            re.I,
        ):
            return 0.0
        return float(
            bool(
                re.search(
                    r"ready to consider one past action-result|past approach may matter|"
                    r"what helped before|reuse (?:one )?(?:past|earlier) (?:action|approach)",
                    current,
                    re.I,
                )
            )
        )

    move, _, _ = _rs_sections(candidate)
    lower_move = move.lower()
    asks_question = bool(re.search(r"\bask\b|\bquestion\b", lower_move))
    trusted_person = "trusted person" in lower_move
    named_emotion = bool(
        re.search(r"mirror an emotion|emotion the user explicitly named", lower_move)
    )
    paraphrase = bool(
        re.search(r"\b(?:paraphrase|condense|summarize|restate)\b", lower_move)
    )
    tension_reflection = "reflect an emotional tension" in lower_move
    advice = bool(re.search(r"\boffer\b|\bsuggest|experiment|adjustment", lower_move))
    if trusted_person:
        return float(
            bool(
                re.search(
                    r"trusted person|ordinary social support|support person",
                    current,
                    re.I,
                )
                and _ADVICE_WELCOME_RE.search(current)
            )
        )
    if asks_question:
        return float(
            bool(
                re.search(
                    r"ask exactly one focused question|one focused question",
                    current,
                    re.I,
                )
                or (
                    _UNCLEAR_RE.search(current)
                    and not re.search(
                        r"question could fit later|what i need now is one brief paraphrase",
                        current,
                        re.I,
                    )
                )
            )
        )
    if named_emotion:
        return float(
            bool(
                re.search(
                    r"explicitly name(?:d)? the feeling|name the feeling as|\banxiety\b",
                    current,
                    re.I,
                )
                and re.search(r"mirror|reflect", current, re.I)
            )
        )
    if paraphrase:
        return float(
            bool(
                re.search(
                    r"briefly restate .*tentative paraphrase|"
                    r"tentative paraphrase that i can correct|"
                    r"paraphrase the (?:most )?(?:relevant|pressing)|"
                    r"brief account would help",
                    current,
                    re.I,
                )
            )
        )
    if tension_reflection:
        return float(
            bool(
                re.search(r"reflection", current, re.I)
                and re.search(r"matters|valued|important|tension", current, re.I)
            )
        )
    if advice:
        return float(bool(_ADVICE_WELCOME_RE.search(current)))
    return 0.0


def _independent_increment(
    current: str, candidate: str, component: str, subtype: str
) -> float:
    """Score specificity/novelty without consulting owner, goal, or boundary."""

    if _EXPLICIT_REDUNDANT_RE.search(current):
        return 0.0
    if _EXPLICIT_INCREMENT_RE.search(current):
        return 1.0
    normalized_candidate = normalize_for_hash(candidate)
    if normalized_candidate and normalized_candidate in normalize_for_hash(current):
        return 0.0
    if component == "MP":
        return float(
            subtype == "MP_PREFERENCE"
            or bool(
                re.search(
                    r"stable practical constraint|education:|job:|"
                    r"stable .{1,80} constraint:",
                    candidate,
                    re.I,
                )
            )
        )
    if component == "MS":
        return float(not bool(_GENERIC_MEMORY_RE.search(candidate)))
    if component == "ME":
        return float(
            subtype == "ME_REUSABLE_OUTCOME"
            and bool(_ACTION_RE.search(candidate) and _RESULT_RE.search(candidate))
        )
    if component == "RS":
        move, _, _ = _rs_sections(candidate)
        return float(bool(normalize_space(move)))
    return 0.0


def transparent_semantic_observation_v2(
    *,
    state_id: str,
    component: str,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    candidate_present: bool,
    candidate_text: str | None,
    candidate_subtype: str,
) -> SemanticObservationRecord:
    """Build contract-correct, independently scored transparent factors."""

    current = normalize_space(current_user_text)
    candidate = normalize_space(candidate_text or "")
    if not candidate_present:
        scores = {factor: 0.0 for factor in FACTOR_NAMES}
    else:
        owner = _independent_owner_time(
            current, candidate, component, candidate_subtype
        )
        goal = _independent_goal(current, candidate, component, candidate_subtype)
        boundary = _independent_boundary(current, candidate, component)
        increment = _independent_increment(
            current, candidate, component, candidate_subtype
        )
        scores = {
            "owner_time_entity_valid": owner,
            "goal_function_fit": goal,
            "boundary_burden_compatible": boundary,
            "currently_nonredundant": increment,
            "specific_increment": increment,
        }
    return SemanticObservationRecord(
        state_id=state_id,
        component=component,
        factor_builder_protocol="pm-v1.5-transparent-semantic-observation-v2",
        factor_scores=scores,
    )


def _structured_owner_time_gate(
    component: str, metadata: StructuredCandidateMetadata
) -> GateDecision:
    if not metadata.candidate_present:
        return "deny"
    if component == "RS":
        return "allow"
    if metadata.candidate_active is False or metadata.candidate_superseded is True:
        return "deny"
    if not metadata.state_owner_id or not metadata.candidate_owner_id:
        return "unknown"
    if metadata.state_owner_id != metadata.candidate_owner_id:
        return "deny"
    if metadata.candidate_active is None or metadata.candidate_superseded is None:
        return "unknown"
    return "allow"


def _three_way_boundary_gate(
    current: str, candidate: str, component: str
) -> GateDecision:
    if _EXPLICIT_RESOURCE_REFUSAL_RE.search(current):
        return "deny"
    if _STOP_RE.search(current) or _ROUTINE_CLOSING_RE.match(current.strip()):
        return "deny"
    if _EXPLICIT_RESOURCE_PERMISSION_RE.search(current):
        return "allow"
    # A direct request for the exact RS move is itself positive boundary
    # evidence.  It overrides a card's generic low-burden exclusion, but never
    # an explicit refusal or stop caught above.
    if component == "RS" and _independent_goal(
        current, candidate, component, "RS_ATOMIC_MOVE"
    ) == 1.0:
        return "allow"
    if _independent_boundary(current, candidate, component) == 0.0:
        return "deny"
    return "unknown"


def _three_way_increment_gate(
    current: str,
    candidate: str,
    metadata: StructuredCandidateMetadata,
) -> GateDecision:
    if _EXPLICIT_REDUNDANT_RE.search(current):
        return "deny"
    if (
        metadata.candidate_content_sha256
        and metadata.candidate_content_sha256 in metadata.visible_fact_sha256s
    ):
        return "deny"
    if _EXPLICIT_INCREMENT_RE.search(current):
        return "allow"
    normalized_candidate = normalize_for_hash(candidate)
    if normalized_candidate and normalized_candidate in normalize_for_hash(current):
        return "deny"
    return "unknown"


def structured_hard_gate_observation_v3(
    *,
    state_id: str,
    component: str,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    candidate_text: str | None,
    candidate_subtype: str,
    metadata: StructuredCandidateMetadata,
) -> StructuredEligibilityObservation:
    """Apply V3 structured eligibility without inferring backend facts from text.

    This function deliberately does not produce a Step1 label.  It only decides
    whether an actual candidate may enter marginal-utility routing.  Unresolved
    boundary or increment evidence is represented as ``unknown`` and therefore
    remains OFF at runtime while still being distinct from a supervised negative.
    """

    del visible_dialogue  # Reserved for a later bounded goal representation.
    current = normalize_space(current_user_text)
    candidate = normalize_space(candidate_text or "")
    if not metadata.candidate_present:
        decisions: dict[str, GateDecision] = {
            "owner_time_entity_valid": "deny",
            "goal_function_fit": "deny",
            "boundary_burden_compatible": "deny",
            "specific_increment": "deny",
        }
    else:
        goal = _independent_goal(current, candidate, component, candidate_subtype)
        decisions = {
            "owner_time_entity_valid": _structured_owner_time_gate(
                component, metadata
            ),
            "goal_function_fit": "allow" if goal == 1.0 else "deny",
            "boundary_burden_compatible": _three_way_boundary_gate(
                current, candidate, component
            ),
            "specific_increment": _three_way_increment_gate(
                current, candidate, metadata
            ),
        }
    runtime_eligible = all(value == "allow" for value in decisions.values())
    return StructuredEligibilityObservation(
        state_id=state_id,
        component=component,
        gate_decisions=decisions,
        runtime_eligible=runtime_eligible,
        unknown_forced_off=(
            not runtime_eligible and any(value == "unknown" for value in decisions.values())
        ),
    )


def structured_policy_observation_v4(
    *,
    state_id: str,
    component: str,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    candidate_text: str | None,
    candidate_subtype: str,
    metadata: StructuredCandidateMetadata,
) -> StructuredPolicyObservation:
    """Route only objective denials around Step1; retain semantic uncertainty.

    V3 incorrectly converted every unresolved semantic eligibility factor into
    a runtime OFF decision, which produced zero candidate coverage on all four
    components.  V4 keeps the safety-critical objective facts as hard gates,
    but exposes goal, increment, and non-conflicting boundary uncertainty to
    the component-effect head.  No uncertainty value is a training label.
    """

    del visible_dialogue
    current = normalize_space(current_user_text)
    candidate = normalize_space(candidate_text or "")
    present = "allow" if metadata.candidate_present else "deny"
    owner_time = _structured_owner_time_gate(component, metadata)
    boundary = _three_way_boundary_gate(current, candidate, component)
    hard_gates: dict[str, GateDecision] = {
        "candidate_present": present,
        "owner_time_entity_valid": owner_time,
        "explicit_boundary_compatible": (
            "deny" if boundary == "deny" else "allow"
        ),
    }
    if not metadata.candidate_present:
        goal: GateDecision = "deny"
        increment: GateDecision = "deny"
    else:
        goal = (
            "allow"
            if _independent_goal(current, candidate, component, candidate_subtype)
            == 1.0
            else "unknown"
        )
        increment = _three_way_increment_gate(current, candidate, metadata)
    semantic = {
        "goal_function_fit": goal,
        "specific_increment": increment,
        "boundary_evidence": boundary,
    }
    reasons = tuple(
        name for name, decision in hard_gates.items() if decision != "allow"
    )
    return StructuredPolicyObservation(
        state_id=state_id,
        component=component,
        hard_gate_decisions=hard_gates,
        semantic_feature_decisions=semantic,
        step1_candidate_admissible=not reasons,
        hard_denial_reasons=reasons,
    )
