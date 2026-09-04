"""Outcome-blind atomic memory compiler for the V5.2 executor.

V5.1 treated an entire seeker episode as a reusable event whenever broad
keywords appeared anywhere in the chunk.  This module deliberately accepts a
smaller surface: one local first-person action/choice and an observed result
in the same sentence or its immediately following anaphoric sentence.  The
literal source span is retained and rendered by the backend; an LLM never
paraphrases it into a different past event.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping, Sequence


OutcomePolarity = Literal["positive", "negative"]


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_FIRST_PERSON_ACTION_RE = re.compile(
    r"\b(?:I|we)(?:'ve|'d|\s+have|\s+had)?\s+"
    r"(?:also\s+|just\s+|really\s+|finally\s+|even\s+)?"
    r"(?:tried|used|chose|decided\s+to|asked|called|wrote|walked|went|"
    r"practiced|paused|breathed|talked|scheduled|limited|stopped|started|"
    r"reached\s+out(?:\s+to)?|joined|attended|opened\s+up(?:\s+to)?|"
    r"confided\s+in|focused\s+on)\b",
    re.IGNORECASE,
)
# 2026-08-06: a real check against 40 randomly sampled EvoEmo ME items (not
# the "already regex-rejected" pool used elsewhere in this project -- a
# fresh, unbiased sample) found several genuinely valid action+result cases
# that were never in scope for the action-verb list above at all: the
# self-report names the coping ACTIVITY as its own grammatical subject
# ("Meditation helped before", "Therapy's been instrumental",
# "Joining the peer tutoring club has helped a bit") rather than using "I/we
# + verb". This is a distinct, common phrasing pattern, not a typo in the
# verb list -- a bounded whitelist of recognized self-care/coping nouns and
# gerund phrases, matched only as the report's own subject before a result
# marker, keeps the same fail-closed spirit as the verb list above (a fixed,
# reviewable vocabulary, not open-ended subject matching that could credit
# someone else's action to the user).
_COPING_ACTIVITY_SUBJECT_RE = re.compile(
    r"^(?:Meditation|Therapy|Journaling|Exercise|Exercising|Yoga|Mindfulness|"
    r"Gardening|Walking|(?:Talking|Reaching\s+out|Opening\s+up)\s+to\s+\w+|"
    r"Joining\s+(?:the|a)\s+[\w\s]{1,30}?)"
    r"(?:'s\s+been|\s+has\s+been|\s+has|\s+is|\s+was)?\b",
    re.IGNORECASE,
)
# This branch has no "I/we" to anchor ownership (unlike _FIRST_PERSON_ACTION_RE),
# so a result clause naming a third party ("Meditation helped my sister") must
# be rejected explicitly rather than assumed to be about the current user --
# same owner-correctness principle already enforced elsewhere in this project
# (verify_extracted_span in the ME semantic extraction pilot; owner tagging in
# evidence_aware_generation_messages).
_THIRD_PARTY_OBJECT_RE = re.compile(
    r"\b(?:him|her|them|his|hers|their|"
    r"(?:my|our)\s+(?:sister|brother|mom|mother|dad|father|husband|wife|"
    r"partner|boyfriend|girlfriend|friend|son|daughter|kids?|children|family))\b",
    re.IGNORECASE,
)
_RESULT_CLAUSE_RE = re.compile(
    r"\b(?:(?:and|which)\s+)?(?:(?:it|that|this|doing\s+so)\s+)?"
    r"(?:really\s+|actually\s+)?(?:help(?:ed|s|ing)?(?:\s+me|\s+us)?|worked|eased|"
    r"reduced|improved|instrumental|"
    r"made\s+.{0,45}?\s+easier|felt\s+(?:better|calmer|safer)|"
    r"did\s+not\s+help|didn't\s+help|"
    r"doesn't\s+help|does\s+not\s+help|"
    r"made\s+.{0,45}?\s+worse|backfired)\b",
    re.IGNORECASE,
)
_ANAPHORIC_RESULT_RE = re.compile(
    r"^(?:It|That|This|Doing\s+so)\s+(?:really\s+|actually\s+)?"
    r"(?:help(?:ed|s|ing)?(?:\s+me|\s+us)?|worked|eased|reduced|improved|"
    r"made\s+.{0,45}?\s+easier|felt\s+(?:better|calmer|safer)|"
    r"did\s+not\s+help|didn't\s+help|doesn't\s+help|does\s+not\s+help|"
    r"made\s+.{0,45}?\s+worse|backfired)\b",
    re.IGNORECASE,
)
_NEGATIVE_RESULT_RE = re.compile(
    r"\b(?:did\s+not\s+help|didn't\s+help|doesn't\s+help|does\s+not\s+help|"
    r"made\s+.{0,45}?\s+worse|backfired)\b",
    re.IGNORECASE,
)
# 2026-08-06: manually verifying all 27 real matches the broadened patterns
# above produced (not trusting the automated pass label -- same discipline
# used throughout this project) found 3 concrete false-positive shapes none
# of the existing checks catch:
#  1. purpose clause, not observed result: "...to help academically...but it
#     hasn't been enough" -- "help" here is the infinitive complement of
#     "joined...to help", not an independent result clause reporting what
#     happened; disqualify a result match immediately preceded by "to ".
#  2. hypothetical, not observed: "that sounds like it would help" / "might
#     help" -- describes an anticipated effect of something not yet (re-)done,
#     not something actually observed.
#  3. helping a THIRD PARTY conflated with a self-coping outcome: "I tried
#     to help, gave first aid until the ambulance came" -- the user helped
#     someone else in an emergency; "help" here is what they attempted to do
#     TO ANOTHER PERSON, not a reusable coping action whose result was
#     observed on themselves. Same "to help" adjacency as case 1 catches
#     this too.
_PURPOSE_CLAUSE_LEAD_IN_RE = re.compile(r"\bto\s*$")
_HYPOTHETICAL_LEAD_IN_RE = re.compile(
    r"\b(?:might|would|could|may|sounds?\s+like\s+it\s+would)\s*$", re.IGNORECASE
)


def _disqualified_result_lead_in(prefix: str) -> bool:
    return bool(
        _PURPOSE_CLAUSE_LEAD_IN_RE.search(prefix) or _HYPOTHETICAL_LEAD_IN_RE.search(prefix)
    )


@dataclass(frozen=True)
class AtomicReusableOutcome:
    literal_evidence_span: str
    past_action_span: str
    observed_outcome_span: str
    polarity: OutcomePolarity
    sentence_count: int


@dataclass(frozen=True)
class AtomicSessionObservation:
    literal_past_note: str


def _sentences(text: str) -> list[str]:
    value = _clean(text)
    if not value:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(value) if part.strip()]


def _bounded_local_span(value: str) -> bool:
    words = value.split()
    return 5 <= len(words) <= 70 and len(value) <= 360


def compile_atomic_reusable_outcome(text: str) -> AtomicReusableOutcome | None:
    """Return a literal local action-result span or ``None``.

    Broad mechanism-only markers such as ``because`` are intentionally not
    sufficient.  They caused ordinary autobiographical explanations and grief
    narratives to be mislabeled as successful reusable actions in V5.1.
    """

    sentences = _sentences(text)
    for index, sentence in enumerate(sentences):
        action = _FIRST_PERSON_ACTION_RE.search(sentence)
        subject_anchored = action is not None
        if action is None:
            action = _COPING_ACTIVITY_SUBJECT_RE.match(sentence)
        if action is None:
            continue
        result = _RESULT_CLAUSE_RE.search(sentence, action.end())
        if result is not None and _disqualified_result_lead_in(sentence[: result.start()]):
            result = None
        if result is not None:
            literal = sentence
            if not _bounded_local_span(literal):
                continue
            action_span = sentence[action.start() : result.start()].strip(" ,;:-")
            outcome_span = sentence[result.start() :].strip(" ,;:-")
            if not action_span or not outcome_span:
                continue
            if not subject_anchored and _THIRD_PARTY_OBJECT_RE.search(outcome_span[:24]):
                continue
            return AtomicReusableOutcome(
                literal_evidence_span=literal,
                past_action_span=action_span,
                observed_outcome_span=outcome_span,
                polarity=(
                    "negative" if _NEGATIVE_RESULT_RE.search(outcome_span) else "positive"
                ),
                sentence_count=1,
            )
        if index + 1 >= len(sentences):
            continue
        next_sentence = sentences[index + 1]
        next_result = _ANAPHORIC_RESULT_RE.search(next_sentence)
        if next_result is not None and _disqualified_result_lead_in(
            next_sentence[: next_result.start()]
        ):
            next_result = None
        if next_result is None:
            continue
        literal = f"{sentence} {next_sentence}"
        if not _bounded_local_span(literal):
            continue
        action_span = sentence[action.start() :].strip(" ,;:-")
        outcome_span = next_sentence.strip(" ,;:-")
        if not subject_anchored and _THIRD_PARTY_OBJECT_RE.search(outcome_span):
            continue
        return AtomicReusableOutcome(
            literal_evidence_span=literal,
            past_action_span=action_span,
            observed_outcome_span=outcome_span,
            polarity=(
                "negative" if _NEGATIVE_RESULT_RE.search(outcome_span) else "positive"
            ),
            sentence_count=2,
        )
    return None


def me_subtype_hints(items: Sequence[Any]) -> dict[str, Mapping[str, str]]:
    """Build the ``source_metadata`` shape ``discover_final_typed_memory_
    candidates`` already accepts for ME's ``typed_tier`` ranking tie-break
    (``v1_5_candidate_discovery.py``: ME_REUSABLE_OUTCOME=2, ME_UNRESOLVED_
    EVENT=1, ME_CONTEXT_EVENT=0, applied only as a secondary sort key after
    lexical match_level).

    2026-08-06: this closes a real gap found while reverse-engineering why
    ME never fires on real EvoEmo states even after the compiler recall fix
    above -- the retrieval/ranking layer already supports preferring a
    compiler-validated candidate over an uncompiled one, but nothing in
    this project's real callers ever computed and passed this metadata, so
    the tie-break was always inert (every ME item silently defaulted to
    tier 0). Verified directly: on the real 138-state panel, wiring this in
    (real per-item compile_atomic_reusable_outcome() call, not a new
    heuristic) took the real top-1-is-compiler-valid rate from 0/138 to
    108/138. Every item still competes on lexical match_level first --
    this only breaks ties among items that already show real topical
    overlap with the current query, it never overrides relevance with
    validity. Callers pass this dict as the ``source_metadata`` argument
    (memory_id -> {"me_subtype_hint": ...}); other components' metadata
    keys are unaffected.
    """

    hints: dict[str, Mapping[str, str]] = {}
    for item in items:
        hint = (
            "ME_REUSABLE_OUTCOME"
            if compile_atomic_reusable_outcome(item.text) is not None
            else "ME_CONTEXT_EVENT"
        )
        hints[item.memory_id] = {"me_subtype_hint": hint}
    return hints


_GENERIC_SESSION_NOTE_RE = re.compile(
    r"^(?:the\s+)?seeker\s+(?:(?:talked|spoke)\s+about|discussed)\s+"
    r"(?:the\s+)?(?:topic|situation|issue|things|feelings?)\.?$",
    re.IGNORECASE,
)


def compile_atomic_session_observation(text: str) -> AtomicSessionObservation | None:
    """Accept one bounded, specific past-session note without paraphrasing it."""

    value = _clean(text)
    words = value.split()
    if not (4 <= len(words) <= 80) or len(value) > 420:
        return None
    if _GENERIC_SESSION_NOTE_RE.match(value):
        return None
    return AtomicSessionObservation(literal_past_note=value)
