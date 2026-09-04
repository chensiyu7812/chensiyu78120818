"""V5.3 Step2: typed response program, replacing V5.2's clause-append composer.

V5.2's ``base_generation_messages`` (v1_5_v5_2_locked_composer.py) never shows
MP/MS/ME content to the generator; ``compose_locked_response`` string-appends
``locked_clauses`` after generation. That guarantees the resource text cannot
be rewritten, but also guarantees the generator cannot absorb, select,
bridge, or organize a reply around the resource -- diagnosed as the primary
V5.2 architecture failure (global failure ledger V15-ARCH-31), and confirmed
directly in real E7 grounding-risk data: 11 of 37 independent-overlap risk
items have V5.2's exact backend clause boilerplate ("An earlier session
recorded...", "I am keeping that as past context...", "You previously
said...", "...on record is...") pasted verbatim into the user-visible reply,
and this happens in 0 of the 22 items with no MS/ME/MP evidence block.

V5.3 instead compiles a machine-auditable *typed response program* per
execution candidate (current_goal, allowed reply actions, owner/time/source/
evidence_id/literal_evidence, required_contribution, epistemic_mode,
forbidden_inferences, atomic move budget, visible style constraints, and a
``cannot_integrate`` reason when no natural use exists). The generator reads
this program plus the literal evidence and must return one natural reply
*and* which evidence IDs it actually used, as a single structured response
(not free text with an appended trace line -- fragile to parse and an extra
leak surface). The final user-visible reply is never
``primary_response + locked_clauses``.

This module only builds and structurally validates the program and the
generator request/response contract. It does not call any generator; that is
wired in a separate execution script, mirroring how v1_5_v5_2_locked_composer
is consumed by the (separate) V5.2 execution scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Literal, Mapping

from .text import content_words
from .v1_5_typed_resource_adapter import TypedResourceCandidate
from .v1_5b_policy_runtime import COMPONENTS, Component, action_component_bits


def _clean(value: str) -> str:
    return " ".join(str(value or "").split())


EpistemicMode = Literal[
    "current_fact", "past_fact", "unverified_continuity", "defeasible_analogy"
]

# Per plan section 2.4's fixed component execution semantics.
_REQUIRED_CONTRIBUTION_BY_COMPONENT: Mapping[str, str] = {
    "MP_PREFERENCE": (
        "change the response's phrasing/format/burden to match the preference; "
        "never read the preference back to the user as a rule"
    ),
    "MP_PROFILE": (
        "actually change the scope, timing, or burden of what is suggested; "
        "omit if the constraint has no object in this reply"
    ),
    "MS": (
        "carry forward one specific prior goal or observation as a naturally "
        "sourced past reference, with a falsifiable opening for whether it "
        "still applies now"
    ),
    "ME": (
        "offer a specific past action-and-result as one declinable option; "
        "never escalate a single past outcome into a rule or guarantee"
    ),
    "RS": "complete exactly one atomic support move, with no appended second task",
}

_EPISTEMIC_MODE_BY_COMPONENT: Mapping[str, EpistemicMode] = {
    "MP_PREFERENCE": "current_fact",
    "MP_PROFILE": "current_fact",
    "MS": "unverified_continuity",
    "ME": "defeasible_analogy",
    "RS": "current_fact",
}

_FORBIDDEN_INFERENCES: tuple[str, ...] = (
    "current_cause",
    "stable_personality_trait",
    "unmentioned_third_party",
    "diagnosis",
    "outcome_guarantee",
)


@dataclass(frozen=True)
class ExecutionEvidence:
    component: str
    evidence_id: str
    owner_id: str | None
    source_kind: str
    strictly_prior: bool
    age_sessions: int | None
    literal_evidence: str
    epistemic_mode: EpistemicMode
    required_contribution: str
    usage_boundary: str = ""


@dataclass(frozen=True)
class TypedResponseProgram:
    requested_action_id: str
    current_goal: str
    allowed_reply_actions: tuple[str, ...]
    evidence: tuple[ExecutionEvidence, ...]
    forbidden_inferences: tuple[str, ...]
    atomic_move_budget: int
    maximum_burden: int
    visible_style_constraints: tuple[str, ...]
    cannot_integrate_reason: str | None = field(default=None)
    # 2026-08-06: some evidence sources (e.g. EvoEmo's own MS session
    # summaries) refer to the current user by a third-person pseudonym
    # ("Anna", "Emily") rather than "the seeker" or a pronoun. A real live
    # test found the generator sometimes fails to resolve that name back to
    # "you" and instead addresses it as a separate third party ("You
    # mentioned Anna... I'm worried she might be..."). If the caller knows
    # the current user's display name(s), passing them here lets the prompt
    # resolve this explicitly instead of leaving it to the model to infer.
    # Empty by default -- not every caller has this information, and not
    # every evidence source uses named pseudonyms.
    current_user_known_aliases: tuple[str, ...] = field(default=())

    @property
    def is_m0(self) -> bool:
        return not self.evidence and self.cannot_integrate_reason is None and self.requested_action_id == "M0+R0"


def _literal_evidence_for(candidate: TypedResourceCandidate) -> str:
    if candidate.component == "MP":
        return _clean(candidate.preference or candidate.profile_fact)
    if candidate.component == "MS":
        return _clean(candidate.prior_observation)
    if candidate.component == "ME":
        # past_action/observed_outcome are the fields TypedResourceCandidate
        # actually requires and validates for ME_REUSABLE_OUTCOME; the
        # optional ``mechanism`` field is a V5.2-specific "literal_span:"
        # provenance convention for that architecture's own clause builder
        # and must not be reused as if it were the required evidence text.
        return _clean(f"{candidate.past_action} {candidate.observed_outcome}")
    if candidate.component == "RS":
        return _clean(candidate.support_move)
    raise ValueError(f"unsupported component: {candidate.component}")


def _usage_boundary_for(candidate: TypedResourceCandidate) -> str:
    if candidate.component == "RS":
        return _clean(f"use when: {candidate.when_to_use} | do not use when: {candidate.when_not_to_use}")
    return ""


def _evidence_key(candidate: TypedResourceCandidate) -> str:
    return candidate.subtype if candidate.component == "MP" else candidate.component


def build_typed_response_program(
    *,
    requested_action_id: str,
    current_goal: str,
    current_user_id: str,
    candidates: Mapping[str, TypedResourceCandidate],
    cannot_integrate_reason: str | None = None,
    expected_execution_candidate_ids: Mapping[str, str] | None = None,
    current_user_known_aliases: tuple[str, ...] = (),
) -> TypedResponseProgram:
    if not _clean(current_goal):
        raise ValueError("current_goal must be non-empty")
    if not _clean(current_user_id):
        raise ValueError("current_user_id must be non-empty")

    requested_bits: Mapping[Component, bool] = action_component_bits(requested_action_id)
    requested_components = {c for c in COMPONENTS if requested_bits[c]}
    if requested_components != set(candidates):
        raise ValueError(
            f"requested_action_id {requested_action_id!r} implies components "
            f"{sorted(requested_components)} but candidates provided {sorted(candidates)}"
        )

    evidence: list[ExecutionEvidence] = []
    for component in COMPONENTS:
        candidate = candidates.get(component)
        if candidate is None:
            continue
        if candidate.component != component:
            raise ValueError(f"candidate under key {component!r} has component {candidate.component!r}")
        if component == "RS":
            if candidate.owner_id is not None:
                raise ValueError("RS candidates must not carry a user owner_id")
        elif candidate.owner_id != current_user_id:
            raise ValueError(
                f"{component} candidate owner_id {candidate.owner_id!r} does not match "
                f"current_user_id {current_user_id!r}"
            )
        if expected_execution_candidate_ids is not None:
            expected_id = expected_execution_candidate_ids.get(component)
            if expected_id is not None and expected_id != candidate.resource_id:
                raise ValueError(
                    f"{component} candidate resource_id {candidate.resource_id!r} does not match "
                    f"the exact Rank-1 execution candidate id {expected_id!r}"
                )
        key = _evidence_key(candidate)
        if key not in _REQUIRED_CONTRIBUTION_BY_COMPONENT:
            raise ValueError(f"unsupported candidate key: {key}")
        evidence.append(
            ExecutionEvidence(
                component=component,
                evidence_id=candidate.resource_id,
                owner_id=candidate.owner_id,
                source_kind=candidate.source_kind,
                strictly_prior=candidate.strictly_prior,
                age_sessions=candidate.age_sessions,
                literal_evidence=_literal_evidence_for(candidate),
                epistemic_mode=_EPISTEMIC_MODE_BY_COMPONENT[key],
                required_contribution=_REQUIRED_CONTRIBUTION_BY_COMPONENT[key],
                usage_boundary=_usage_boundary_for(candidate),
            )
        )
    if any(not item.literal_evidence for item in evidence):
        raise ValueError("compiled empty literal_evidence for at least one candidate")

    atomic_budget = 1 if "RS" in candidates else 0
    return TypedResponseProgram(
        requested_action_id=requested_action_id,
        current_goal=_clean(current_goal),
        allowed_reply_actions=("respond",) if not cannot_integrate_reason else ("m0_fallback",),
        evidence=tuple(evidence),
        forbidden_inferences=_FORBIDDEN_INFERENCES,
        atomic_move_budget=atomic_budget,
        maximum_burden=1,
        visible_style_constraints=(
            "concise",
            "no internal labels, resource IDs, or field names",
            "no verbatim record-log phrasing",
        ),
        cannot_integrate_reason=cannot_integrate_reason,
        current_user_known_aliases=tuple(
            alias for alias in current_user_known_aliases if _clean(alias)
        ),
    )


@dataclass(frozen=True)
class GeneratorResponse:
    """The generator's single structured output (e.g. via a JSON-schema
    constrained provider response_schema, not free text with an appended
    trace line -- see the module docstring for why the free-text form was
    replaced)."""

    reply: str
    used_evidence_ids: tuple[str, ...]
    realized_response_act: str
    # Preserve the model's raw trace for telemetry when the runtime can
    # deterministically normalize it.  A legal M0+R0 program has no
    # authorized evidence, so an invented id must not invalidate an
    # otherwise usable visible reply.
    reported_used_evidence_ids: tuple[str, ...] = ()


class RewritePolicy(str, Enum):
    """Pre-outcome Step2 recovery alternatives.

    The project has not yet selected which alternative belongs in the formal
    V5.3 release.  Making the choice explicit lets consumed development cases
    compare one bounded correction with immediate deterministic fallback
    without silently changing the generator stack or hiding the second call.
    """

    SINGLE_BOUNDED_REWRITE = "single_bounded_rewrite"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


@dataclass(frozen=True)
class TypedResponseExecutionResult:
    response: GeneratorResponse | None
    status: Literal[
        "clean", "fixed_by_rewrite", "fell_back_to_m0", "no_structured_output"
    ]
    rewrite_policy: RewritePolicy
    calls_made: int
    rewrite_attempted: bool
    first_pass_errors: tuple[str, ...]
    final_guard_errors: tuple[str, ...]
    requested_action_id: str
    realized_action_id: str | None

    @property
    def used_evidence_ids(self) -> tuple[str, ...]:
        return self.response.used_evidence_ids if self.response is not None else ()

    @property
    def reported_used_evidence_ids(self) -> tuple[str, ...]:
        if self.response is None:
            return ()
        return self.response.reported_used_evidence_ids


def parse_generator_response_dict(raw: Mapping[str, object]) -> GeneratorResponse:
    reply = raw.get("reply")
    used_ids = raw.get("used_evidence_ids")
    act = raw.get("realized_response_act")
    if not isinstance(reply, str) or not isinstance(act, str):
        raise ValueError("generator response missing reply/realized_response_act strings")
    if not isinstance(used_ids, (list, tuple)) or not all(isinstance(i, str) for i in used_ids):
        raise ValueError("generator response used_evidence_ids must be a list of strings")
    return GeneratorResponse(
        reply=_clean(reply),
        used_evidence_ids=tuple(used_ids),
        realized_response_act=_clean(act),
        reported_used_evidence_ids=tuple(used_ids),
    )


def normalize_generator_response_for_program(
    *, response: GeneratorResponse, program: TypedResponseProgram
) -> GeneratorResponse:
    """Apply only program-provable trace normalization.

    ``used_evidence_ids`` is generator telemetry, not gold.  For a genuine
    M0+R0 program the authorized evidence set is structurally empty, so the
    realized set is deterministically empty regardless of what identifier
    the model self-reports.  The raw report remains available for audit.
    Evidence-bearing programs remain unchanged and are still checked
    strictly by :func:`typed_response_guard_errors`.
    """

    reported = response.reported_used_evidence_ids or response.used_evidence_ids
    if program.is_m0 and response.used_evidence_ids:
        return GeneratorResponse(
            reply=response.reply,
            used_evidence_ids=(),
            realized_response_act=response.realized_response_act,
            reported_used_evidence_ids=tuple(reported),
        )
    if response.reported_used_evidence_ids:
        return response
    return GeneratorResponse(
        reply=response.reply,
        used_evidence_ids=response.used_evidence_ids,
        realized_response_act=response.realized_response_act,
        reported_used_evidence_ids=tuple(reported),
    )


def evidence_aware_generation_messages(
    *, current_context: str, program: TypedResponseProgram
) -> list[dict[str, str]]:
    """Build the generator request.

    2026-08-06: a real live test (10 real states, real NVIDIA-hosted
    Llama-3.1-8B-Instruct calls) found 5/10 replies presented the user's own
    evidence in first person as the assistant's own experience -- one
    fabricated a spouse and child that belong to the user. The worst case's
    evidence was clean third person ("Emily is feeling..."), so this is not
    just "the model continued a quoted I" -- it is a narrator-identity
    collapse the previous prompt never addressed. ExecutionEvidence already
    carried owner_id (the real user id for MP/MS/ME, None for RS), but this
    function never rendered it into the request at all, so the generator had
    no signal whatsoever about whose facts these were. The fix below is
    deliberately the minimal, already-available-data version of a claim
    ownership tag (render owner_id, do not invent a new predicate/object
    extraction layer -- see PM_V1_5_V5_3_STEP2_LIVE_TEST_FINDINGS_
    20260806_ZH.md for why a full semantic Claim IR is a separate, harder,
    not-yet-justified investment). Evidence also moves into the system
    message, separated from current_context, so it reads as background the
    assistant knows rather than more things the user just said in this turn.

    2026-08-06 (later same day): a real Step1 MS minimal pilot (with-MS vs
    without-MS paired generation, see PM_V1_5_V5_3_STEP1_MS_MINIMAL_PILOT_
    FINDINGS_20260806_ZH.md) found the "without" arm -- a genuine M0+R0
    program with zero evidence, exactly the untested path the M0+R0 note
    below flags -- fell back to a generic canned reply in 4/12 real states
    and needed a guard-triggered rewrite in 2 more (6/12 total), all on
    TRACE_REFERENCES_UNAUTHORIZED_EVIDENCE_ID. Reproduced directly outside
    the pilot with a real call on the same state: the model returned
    used_evidence_ids=["user_message_1"] -- a self-invented id for the
    user's own current turn, not a hallucinated memory fact. Root cause:
    nothing ever told the model used_evidence_ids means "which item from
    the Background facts list," so it invents a plausible-looking id
    whenever it wants to note it drew on the user's message (which is
    normal, every reply does). This degraded the M0+R0 baseline shared by
    every "without X" arm across the MP/MS/RS pilots, not just MS's own
    comparison. Fixed with an explicit scope instruction on
    used_evidence_ids below; reproduced-then-fixed on the same real state
    and seed before being trusted.
    """

    if not _clean(current_context):
        raise ValueError("current_context must be non-empty")
    if program.cannot_integrate_reason is not None:
        raise ValueError("cannot_integrate programs must not be sent to the generator")
    system_lines = [
        f"Current goal: {program.current_goal}",
        "You are the assistant responding to the user. You are not the user and you are not "
        "role-playing the user.",
    ]
    if program.evidence:
        system_lines.append(
            "Write one natural, coherent reply that genuinely incorporates every evidence "
            "item below; every item listed has already been confirmed to have a natural use "
            "in this reply, so use all of them."
        )
    else:
        # 2026-08-06: found while checking a design question, not from a real
        # failure -- with zero evidence (a genuine M0+R0 Step1 decision, not
        # a degraded fallback -- see TypedResponseProgram.is_m0), the old
        # instruction above referenced "every evidence item below" with
        # nothing to point to. No test in this project's real-call history
        # has exercised a true empty-evidence program (every real batch so
        # far required at least one MS candidate to exist), so this was
        # never actually observed misbehaving -- fixed on inspection, not on
        # evidence of a real failure.
        system_lines.append(
            "There is no memory or profile evidence for this turn. Write one natural, "
            "supportive reply grounded only in what the user has visibly said -- do not "
            "claim or imply that you recall anything from an earlier session."
        )
    if program.evidence:
        system_lines.append(
            "Every evidence item below with an owner describes THAT PERSON's own fact, "
            "statement, or past experience -- never yours, regardless of whether its literal "
            "wording is first person, third person, or a name. Always address that person "
            "directly, in the second person (for example: \"you mentioned...\", \"your "
            "husband...\"). Never claim their spouse, child, job, education, relationship, "
            "decision, emotion, or past action as your own experience or biography."
        )
    system_lines.append(
        "You may use first person only to describe your own present conversational act "
        "(e.g. \"I hear you\", \"I'm sorry\", \"I want to understand\"), never to narrate a "
        "personal life event, relationship, or biography."
    )
    system_lines.append(
        "used_evidence_ids must only contain evidence_id values copied exactly from the "
        "\"Background facts\" list below, if one is present. The user's current message is "
        "not itself an evidence_id and must never be listed (for example, never invent an id "
        "like \"user_message_1\"). If there is no Background facts list, or you did not rely "
        "on any item from it, used_evidence_ids must be an empty list."
    )
    if program.current_user_known_aliases:
        # Deliberately no slash-separated pronoun notation here (e.g. writing
        # "you/your" as shorthand): a real live test found the generator
        # copy the literal notational string into a visible reply instead of
        # treating it as an instruction (see the 2026-08-06 fix in this same
        # function). Full natural sentences only.
        alias_list = ", ".join(f"\"{a}\"" for a in program.current_user_known_aliases)
        system_lines.append(
            f"The person you are talking to is sometimes referred to by name in evidence "
            f"text below: {alias_list}. That name refers to the same person you are "
            f"talking to right now, not a different, separate person. If evidence mentions "
            f"that name, address that fact directly to the person you are talking to, the "
            f"same way you would address any other evidence about them. Never talk about "
            f"that name as if it were someone else (for example, never say something like "
            f"\"I'm worried she might be feeling overwhelmed\")."
        )
    system_lines += [
        "Forbidden: inventing a current cause, a stable personality trait, an unmentioned "
        "third party, a diagnosis, or an outcome guarantee.",
        "Forbidden in the visible reply: internal labels (MP/MS/ME/RS), resource IDs, field "
        "names, or verbatim record-log phrasing that sounds like a system note.",
        f"Atomic support-move budget: {program.atomic_move_budget}. Do not exceed it.",
        "Return the required structured response: reply, used_evidence_ids, realized_response_act.",
    ]
    evidence_lines: list[str] = []
    for item in program.evidence:
        tentativeness = (
            "state this tentatively and leave room for it to no longer apply"
            if item.epistemic_mode == "unverified_continuity"
            else "offer this as one declinable option, not a rule"
            if item.epistemic_mode == "defeasible_analogy"
            else "this is presently true"
        )
        boundary = f" -- boundary: {item.usage_boundary}" if item.usage_boundary else ""
        owner = (
            "owner=the person you are talking to (address them in the second person)"
            if item.owner_id
            else "owner=none (a permitted support move, not a personal fact)"
        )
        evidence_lines.append(
            f"- evidence_id={item.evidence_id} component={item.component} {owner} "
            f"epistemic_mode={item.epistemic_mode} ({tentativeness}): "
            f'"{item.literal_evidence}" -- required contribution: {item.required_contribution}{boundary}'
        )
    if evidence_lines:
        system_lines.append(
            "Background facts the assistant already knows (not things the user is currently "
            "saying in this turn):\n" + "\n".join(evidence_lines)
        )
    return [
        {"role": "system", "content": " ".join(system_lines)},
        {"role": "user", "content": _clean(current_context)},
    ]


_INTERNAL_ID_RE = re.compile(r"(?:mem|strat|card)_[0-9a-f]{8,}", re.IGNORECASE)
_INTERNAL_LABEL_RE = re.compile(r"\b(?:MP|MS|ME|RS|M0|R0)\b")

# Derived directly from v1_5_v5_2_locked_composer.py's own fixed clause
# templates (the exact source of the leak, confirmed against real E7
# grounding-risk data: 11/37 independent-overlap items reproduce these
# fragments verbatim, 0/37 non-evidence items ever do), not from a handful
# of examples noticed by inspection.
_RECORD_LOG_PHRASE_RE = re.compile(
    r"an earlier session recorded"
    r"|i am keeping (?:that|this) as past context"
    r"|you previously said"
    r"|that past result (?:may be a reason not to repeat|can be one optional starting point)"
    r"|(?:on record| on record is)\b"
    r"|i will keep the response within that constraint"
    r"|\bthe seeker\b",
    re.IGNORECASE,
)


def typed_response_guard_errors(
    *, response: GeneratorResponse, program: TypedResponseProgram
) -> tuple[str, ...]:
    """Machine-checkable structural/binding errors only.

    Whether a used evidence item is stale, conflicting, overgeneralized, or
    imposes unwarranted meaning is a semantic question for the independent
    grounding-risk review, not this guard (plan section 2.5).
    """

    cleaned = _clean(response.reply)
    errors: list[str] = []
    if not cleaned:
        return ("EMPTY_RESPONSE",)
    if _INTERNAL_ID_RE.search(cleaned) or _INTERNAL_LABEL_RE.search(cleaned):
        errors.append("INTERNAL_LABEL_OR_ID_LEAK")
    if _RECORD_LOG_PHRASE_RE.search(cleaned):
        errors.append("RECORD_LOG_PHRASING_LEAK")
    authorized_ids = {item.evidence_id for item in program.evidence}
    used_ids = set(response.used_evidence_ids)
    unauthorized = used_ids - authorized_ids
    if unauthorized:
        errors.append("TRACE_REFERENCES_UNAUTHORIZED_EVIDENCE_ID")
    # Every evidence item in a program was already confirmed (pre-generation,
    # by whatever built cannot_integrate/the program itself) to have a
    # natural use; the generator is not given discretion to silently skip
    # one. A gap here is a real Step2 nonuse event, not a quiet pass.
    #
    # 2026-08-06: MP is exempt from this requirement -- an independent
    # review correctly pointed out this check's "must be cited" contract is
    # shaped for MS/ME (narrative fact the reply should reference), not MP.
    # MP_PREFERENCE should change the reply's form/tone/length; MP_PROFILE
    # should change a suggestion's scope/timing/feasibility -- neither
    # necessarily leaves a citable trace even when correctly incorporated.
    # A real paired-generation test found this exact contract actively
    # harmful: 3/6 real states where a correctly-matched MP fact caused an
    # otherwise-successful MS-only reply to fail this check and fall back to
    # a generic M0 response (traced directly: the model wrote a coherent
    # MS-grounded reply and never worked in the short MP fact, and this
    # check then discarded the whole reply over it). See
    # PM_V1_5_V5_3_MS_MP_ME_EFFECT_TEST_20260806_ZH.md. MS/ME keep the
    # existing strict requirement unchanged.
    #
    # 2026-08-06 (Step1 RS minimal pilot): RS is exempt for the identical
    # reason. RS_ATOMIC_MOVE's support_move/when_to_use/when_not_to_use are
    # response-shaping instructions ("offer grounded validation"), not a
    # narrative fact to reference -- a reply that correctly follows the
    # guidance has no obligation to cite it. Real paired with/without-RS
    # generation found this exact contract firing on a real RS case (p14,
    # AM04_tentative_paraphrase_check): a coherent RS-following reply
    # discarded and replaced with a generic fallback purely because the
    # model never echoed the card's evidence_id. See
    # PM_V1_5_V5_3_STEP1_RS_MINIMAL_PILOT_FINDINGS_20260806_ZH.md.
    required_ids = {
        item.evidence_id for item in program.evidence if item.component not in {"MP", "RS"}
    }
    missing = required_ids - used_ids
    if missing:
        errors.append("REQUIRED_EVIDENCE_NOT_USED")
    if len(response.used_evidence_ids) != len(set(response.used_evidence_ids)):
        errors.append("DUPLICATE_EVIDENCE_ID_IN_TRACE")
    return tuple(errors)


# Heuristic only -- calibrated against the 10 real replies from the
# 2026-08-06 live test (5 confirmed violations, 1 mixed, 4 clean; see
# PM_V1_5_V5_3_STEP2_LIVE_TEST_FINDINGS_20260806_ZH.md), NOT a complete
# semantic check. Two signals, checked independently:
# (a) a direct first-person possessive over a biographical noun ("my
#     husband", "my college degree" -- up to two words of adjective/modifier
#     tolerated between the possessive and the noun); catches p7/p10's
#     clearest cases.
# (b) within a single sentence, a sustained-personal-reflection verb phrase
#     ("I've been thinking/weighing/considering/...") co-occurring with a
#     first-person possessive ("my"/"our") anywhere in that sentence --
#     catches p9/p13's paraphrased narrative-voice cases, which have no
#     single forbidden noun.
# Recall on the 10-reply calibration set: 3/3 direct-noun cases caught, plus
# the two narrative cases this second signal was added for. This will still
# miss cases with neither signal -- treat a pass as "no obvious violation",
# not proof of correct attribution. Independent human review on a fresh
# sample is still required before trusting an aggregate pass rate.
_FIRST_PERSON_BIOGRAPHICAL_NOUN_RE = re.compile(
    r"\b(?:my|our)\b(?:\s+\w+){0,2}\s+(?:husband|wife|spouse|partner|"
    r"ex[- ]?(?:boyfriend|girlfriend|partner|husband|wife)|son|daughter|"
    r"child|kids?|mother|father|mom|dad|family|job|career|degree|boss|"
    r"supervisor|manager|therapist|doctor|diagnosis|illness|pregnancy)\b"
    r"|\bas a (?:business owner|graduate student|freelancer|software engineer|"
    r"project manager|office worker)\b",
    re.IGNORECASE,
)
_PERSONAL_REFLECTION_VERB_RE = re.compile(
    r"\bi(?:'ve| have)?(?:\s+been)?\s+(?:trying to|struggling with|weighing|"
    r"considering|dealing with|thinking about|working towards|facing|"
    r"navigating|balancing|reflecting on|reminded of)\b",
    re.IGNORECASE,
)
_MY_OUR_POSSESSIVE_RE = re.compile(r"\b(?:my|our)\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def speaker_attribution_guard_errors(*, response: GeneratorResponse) -> tuple[str, ...]:
    """Heuristic fail-fast for the assistant claiming the user's evidence as
    its own biography (see module-level 2026-08-06 note and the regex
    docstrings above). Deliberately separate from typed_response_guard_errors:
    that function is scoped to machine-checkable structural/binding errors
    only (its own docstring), and folding an incomplete heuristic into it
    would misrepresent this as a complete check.
    """

    cleaned = _clean(response.reply)
    if _FIRST_PERSON_BIOGRAPHICAL_NOUN_RE.search(cleaned):
        return ("POSSIBLE_ASSISTANT_SELF_ATTRIBUTION_OF_USER_FACT",)
    for sentence in _SENTENCE_SPLIT_RE.split(cleaned):
        if _PERSONAL_REFLECTION_VERB_RE.search(sentence) and _MY_OUR_POSSESSIVE_RE.search(sentence):
            return ("POSSIBLE_ASSISTANT_SELF_ATTRIBUTION_OF_USER_FACT",)
    return ()


def evidence_usage_plausibility_errors(
    *, response: GeneratorResponse, program: TypedResponseProgram
) -> tuple[str, ...]:
    """Cheap, deliberately incomplete check on ``used_evidence_ids``.

    used_evidence_ids is self-reported by the generator and otherwise never
    independently checked against the actual reply text (an independent
    review flagged this correctly: a model can claim it used a piece of
    evidence without any real trace of it in the reply). Building a full
    verbatim-grounding schema (each use citing an exact substring of the
    reply) was considered and deliberately not done -- it adds real new
    failure surface for an 8B model (paraphrase/quoting/whitespace variance
    would all fail an exact-match check even for genuinely correct usage),
    and answers a different question (was evidence really grounded) than
    the specific bug this module was built to fix (was it attributed to the
    right person). This is the cheap middle ground: reuse the same
    content-word-overlap primitive already used for candidate matching
    elsewhere in this project (text.content_words) to catch only the most
    blatant case -- claimed evidence with literally zero word-level trace in
    the reply. It will not catch a paraphrased-but-genuine use, and it will
    not catch a superficial word-level echo that isn't real grounding either
    -- treat a pass as "no obvious non-use", not proof of real grounding.

    MP and RS are skipped here for the same reason they are exempt from
    REQUIRED_EVIDENCE_NOT_USED above.  This word-overlap primitive assumes
    evidence is a narrative fact a faithful reply should lexically echo
    (true for MS/ME).  MP preferences/profile constraints and RS atomic
    moves are response-shaping instructions; a reply can genuinely follow
    them without sharing content words with the instruction itself.  Real
    paired generation found this false-positive for both component types;
    the global ledger freezes the shared exemption in V15-PM-43.
    """

    reply_words = content_words(response.reply)
    used_ids = set(response.used_evidence_ids)
    for item in program.evidence:
        if item.component in {"MP", "RS"} or item.evidence_id not in used_ids:
            continue
        evidence_words = content_words(item.literal_evidence)
        if evidence_words and not (evidence_words & reply_words):
            return ("EVIDENCE_CLAIMED_USED_BUT_NO_WORD_TRACE_IN_REPLY",)
    return ()


def _all_response_guard_errors(
    *, response: GeneratorResponse, program: TypedResponseProgram
) -> tuple[str, ...]:
    return (
        typed_response_guard_errors(response=response, program=program)
        + speaker_attribution_guard_errors(response=response)
        + evidence_usage_plausibility_errors(response=response, program=program)
    )


def execute_typed_response(
    client,
    response_schema,
    messages,
    program: TypedResponseProgram,
    *,
    rewrite_policy: RewritePolicy = RewritePolicy.SINGLE_BOUNDED_REWRITE,
) -> TypedResponseExecutionResult:
    """Execute one typed response program with an explicit recovery policy.

    Moved here (2026-08-06) from a standalone runner script after an
    independent review correctly pointed out that the earlier location made
    this "a verified prototype, not production-reusable infrastructure" --
    any real V5.3 executor needs this exact call/verify/retry/fallback
    orchestration and previously would have had to reimplement it. ``client``
    is duck-typed (anything with a ``.chat(messages, response_schema=...) ->
    (result, parsed_or_None)`` method) so this module does not need a hard
    dependency on any specific HTTP client implementation.

    Never retries more than once.  ``calls_made`` and ``rewrite_attempted``
    are returned explicitly so the formal runner can account for every token,
    dollar, and latency contribution.  This function does not decide which
    policy is scientifically preferable; that decision is frozen after a
    development-only comparison and before formal paired outcomes.
    """

    result, parsed = client.chat(messages, response_schema=response_schema)
    if parsed is None:
        errors = (str(result),)
        fallback_boundary = (
            "one_focused_question"
            if program.atomic_move_budget > 0
            else "concise_reflection"
        )
        fallback = parse_generator_response_dict(
            {
                "reply": m0_fallback_response(fallback_boundary),
                "used_evidence_ids": [],
                "realized_response_act": "m0_fallback",
            }
        )
        return TypedResponseExecutionResult(
            response=fallback,
            status="fell_back_to_m0",
            rewrite_policy=rewrite_policy,
            calls_made=1,
            rewrite_attempted=False,
            first_pass_errors=errors,
            final_guard_errors=errors,
            requested_action_id=program.requested_action_id,
            realized_action_id="M0+R0",
        )
    response = normalize_generator_response_for_program(
        response=parse_generator_response_dict(parsed.model_dump()), program=program
    )
    errors = _all_response_guard_errors(response=response, program=program)
    if not errors:
        return TypedResponseExecutionResult(
            response=response,
            status="clean",
            rewrite_policy=rewrite_policy,
            calls_made=1,
            rewrite_attempted=False,
            first_pass_errors=(),
            final_guard_errors=(),
            requested_action_id=program.requested_action_id,
            realized_action_id=program.requested_action_id,
        )

    if rewrite_policy is RewritePolicy.SINGLE_BOUNDED_REWRITE:
        rewrite_messages = messages + [
            {"role": "assistant", "content": response.reply},
            {
                "role": "user",
                "content": (
                    "Your reply violated a stated rule -- for example, presenting someone "
                    "else's fact as your own experience, or claiming to use a piece of "
                    "evidence that has no real connection to what you wrote. Keep the same "
                    "content and evidence, but rewrite your reply, correctly addressing "
                    "evidence about the user directly to them, and only listing an evidence "
                    "id in used_evidence_ids if you genuinely incorporated it. Do not add "
                    "new facts."
                ),
            },
        ]
        _, parsed2 = client.chat(rewrite_messages, response_schema=response_schema)
        if parsed2 is not None:
            response2 = normalize_generator_response_for_program(
                response=parse_generator_response_dict(parsed2.model_dump()), program=program
            )
            errors2 = _all_response_guard_errors(response=response2, program=program)
            if not errors2:
                return TypedResponseExecutionResult(
                    response=response2,
                    status="fixed_by_rewrite",
                    rewrite_policy=rewrite_policy,
                    calls_made=2,
                    rewrite_attempted=True,
                    first_pass_errors=errors,
                    final_guard_errors=(),
                    requested_action_id=program.requested_action_id,
                    realized_action_id=program.requested_action_id,
                )
        else:
            errors2 = ("REWRITE_NO_STRUCTURED_OUTPUT",)
    else:
        errors2 = errors

    fallback_boundary = (
        "one_focused_question" if program.atomic_move_budget > 0 else "concise_reflection"
    )
    fallback_reply = m0_fallback_response(fallback_boundary)
    fallback = parse_generator_response_dict(
        {"reply": fallback_reply, "used_evidence_ids": [], "realized_response_act": "m0_fallback"}
    )
    return TypedResponseExecutionResult(
        response=fallback,
        status="fell_back_to_m0",
        rewrite_policy=rewrite_policy,
        calls_made=(2 if rewrite_policy is RewritePolicy.SINGLE_BOUNDED_REWRITE else 1),
        rewrite_attempted=(rewrite_policy is RewritePolicy.SINGLE_BOUNDED_REWRITE),
        first_pass_errors=errors,
        final_guard_errors=tuple(errors2),
        requested_action_id=program.requested_action_id,
        realized_action_id="M0+R0",
    )


def call_with_guard_and_rewrite(
    client,
    response_schema,
    messages,
    program,
    *,
    rewrite_policy: RewritePolicy = RewritePolicy.SINGLE_BOUNDED_REWRITE,
):
    """Backward-compatible tuple API over :func:`execute_typed_response`.

    Existing development scripts consume ``(response, status,
    first_pass_errors)``.  New formal wiring should retain the complete
    :class:`TypedResponseExecutionResult` instead.
    """

    execution = execute_typed_response(
        client,
        response_schema,
        messages,
        program,
        rewrite_policy=rewrite_policy,
    )
    return execution.response, execution.status, execution.first_pass_errors


_M0_FALLBACK_FAMILY: Mapping[str, str] = {
    "listen_only": "I hear you.",
    "acknowledge": "That sounds like a lot to carry.",
    "concise_reflection": "It sounds like that's been weighing on you.",
    "one_focused_question": "Can you tell me a bit more about what feels most important right now?",
    "no_history_available": "I don't have anything else on record for this -- can you fill me in?",
    "one_optional_suggestion": "One thing that sometimes helps is taking a short pause before responding.",
}


def m0_fallback_response(boundary: str = "one_focused_question") -> str:
    if boundary not in _M0_FALLBACK_FAMILY:
        raise ValueError(f"unknown M0 fallback boundary: {boundary!r}")
    return _M0_FALLBACK_FAMILY[boundary]
