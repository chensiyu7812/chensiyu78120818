"""Deterministic Step-2 resource adapter for PM V1.5 V5.

The policy manager chooses component bits and exact candidate IDs.  This
module performs the mechanical part that must not be delegated to a language
model: bind owner/time/source metadata, compile each typed candidate into a
bounded response directive, and provide a resource-free deterministic
fallback.  It does not decide whether a component should be requested and it
does not score response quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Mapping

from .v1_5b_policy_runtime import COMPONENTS, Component, action_component_bits


ResourceSubtype = Literal[
    "MP_PREFERENCE",
    "MP_PROFILE",
    "MS_SESSION_OBSERVATION",
    "ME_REUSABLE_OUTCOME",
    "RS_ATOMIC_MOVE",
]

_EXPECTED_COMPONENT: dict[ResourceSubtype, Component] = {
    "MP_PREFERENCE": "MP",
    "MP_PROFILE": "MP",
    "MS_SESSION_OBSERVATION": "MS",
    "ME_REUSABLE_OUTCOME": "ME",
    "RS_ATOMIC_MOVE": "RS",
}


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


@dataclass(frozen=True)
class TypedResourceCandidate:
    """One exact Rank-1 candidate with source-controlled typed fields.

    Fields irrelevant to the declared subtype must remain empty.  Memory
    candidates are bound to one user.  MS/ME additionally must be strictly
    prior.  The adapter never reconstructs these facts from a free-text span.
    """

    component: Component
    subtype: ResourceSubtype
    resource_id: str
    candidate_version: str
    source_kind: Literal["profile", "session", "event", "strategy"]
    owner_id: str | None = None
    active: bool = True
    strictly_prior: bool = False
    age_sessions: int | None = None
    preference: str = ""
    profile_fact: str = ""
    prior_observation: str = ""
    past_action: str = ""
    observed_outcome: str = ""
    mechanism: str = ""
    support_move: str = ""
    when_to_use: str = ""
    when_not_to_use: str = ""

    def __post_init__(self) -> None:
        if self.component not in COMPONENTS:
            raise ValueError(f"unknown component: {self.component}")
        if _EXPECTED_COMPONENT[self.subtype] != self.component:
            raise ValueError("component/subtype mismatch")
        if not _clean(self.resource_id) or not _clean(self.candidate_version):
            raise ValueError("resource_id and candidate_version must be non-empty")
        expected_source = {
            "MP_PREFERENCE": "profile",
            "MP_PROFILE": "profile",
            "MS_SESSION_OBSERVATION": "session",
            "ME_REUSABLE_OUTCOME": "event",
            "RS_ATOMIC_MOVE": "strategy",
        }[self.subtype]
        if self.source_kind != expected_source:
            raise ValueError("source_kind/subtype mismatch")
        if not self.active:
            raise ValueError("inactive candidate cannot be compiled")

        if self.component != "RS" and not _clean(self.owner_id):
            raise ValueError("memory candidates require owner_id")
        if self.component == "RS" and self.owner_id is not None:
            raise ValueError("strategy candidates are not user-owned memories")
        if self.component in {"MS", "ME"}:
            if not self.strictly_prior or self.age_sessions is None or self.age_sessions < 1:
                raise ValueError("MS/ME candidates must be strictly prior")

        required = {
            "MP_PREFERENCE": (self.preference,),
            "MP_PROFILE": (self.profile_fact,),
            "MS_SESSION_OBSERVATION": (self.prior_observation,),
            "ME_REUSABLE_OUTCOME": (self.past_action, self.observed_outcome),
            "RS_ATOMIC_MOVE": (
                self.support_move,
                self.when_to_use,
                self.when_not_to_use,
            ),
        }[self.subtype]
        if any(not _clean(value) for value in required):
            raise ValueError(f"missing required typed field for {self.subtype}")


@dataclass(frozen=True)
class CompiledResourceDirective:
    component: Component
    resource_id: str
    candidate_version: str
    source_kind: str
    owner_id: str | None
    evidence: str
    response_instruction: str
    forbidden_claims: tuple[str, ...]
    attribution_required: bool
    maximum_support_moves: int


@dataclass(frozen=True)
class CompiledResourceBundle:
    requested_action_id: str
    directives: tuple[CompiledResourceDirective, ...]

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(item.component for item in self.directives)


def compile_typed_resource(
    candidate: TypedResourceCandidate, *, current_user_id: str
) -> CompiledResourceDirective:
    """Compile one already-selected candidate without semantic invention."""

    if candidate.component != "RS" and candidate.owner_id != current_user_id:
        raise ValueError("memory owner does not match current user")

    common_forbidden = (
        "Do not invent a past conversation, result, preference, relationship, or cause.",
        "Do not turn a past fact into a present fact without a tentative check.",
        "Do not reveal resource IDs, profile labels, memory labels, or internal routing.",
    )
    if candidate.subtype == "MP_PREFERENCE":
        evidence = f"Current active response preference: {_clean(candidate.preference)}"
        instruction = (
            "Follow this response-format preference only when expressing the answer; "
            "do not mention that a profile exists and do not add personal facts."
        )
        attribution_required = False
    elif candidate.subtype == "MP_PROFILE":
        evidence = f"Current active practical constraint: {_clean(candidate.profile_fact)}"
        instruction = (
            "Use this stable constraint only if it changes the response. Do not infer a "
            "motive, emotion, diagnosis, or current cause from it."
        )
        attribution_required = False
    elif candidate.subtype == "MS_SESSION_OBSERVATION":
        evidence = f"An earlier session recorded: {_clean(candidate.prior_observation)}"
        instruction = (
            "Keep this explicitly in the past. If it matters now, present it as something "
            "previously noted and tentatively check whether it still applies."
        )
        attribution_required = True
    elif candidate.subtype == "ME_REUSABLE_OUTCOME":
        mechanism = (
            f" Mechanism recorded: {_clean(candidate.mechanism)}"
            if _clean(candidate.mechanism)
            else ""
        )
        evidence = (
            f"Previously tried action: {_clean(candidate.past_action)}. "
            f"Observed result: {_clean(candidate.observed_outcome)}.{mechanism}"
        )
        instruction = (
            "Attribute this to the user's prior experience. Offer it only as a tentative, "
            "rejectable option; never claim it will work now or that the current situation "
            "has the same cause."
        )
        attribution_required = True
    else:
        evidence = (
            f"Permitted support move: {_clean(candidate.support_move)}. "
            f"Use only when: {_clean(candidate.when_to_use)}. "
            f"Do not use when: {_clean(candidate.when_not_to_use)}."
        )
        instruction = (
            "Express at most this one atomic support move. Do not add a second question, "
            "exercise, plan, or unsupported reassurance."
        )
        attribution_required = False

    return CompiledResourceDirective(
        component=candidate.component,
        resource_id=candidate.resource_id,
        candidate_version=candidate.candidate_version,
        source_kind=candidate.source_kind,
        owner_id=candidate.owner_id,
        evidence=evidence,
        response_instruction=instruction,
        forbidden_claims=common_forbidden,
        attribution_required=attribution_required,
        maximum_support_moves=1,
    )


def compile_typed_bundle(
    *,
    requested_action_id: str,
    candidates: Mapping[str, TypedResourceCandidate],
    current_user_id: str,
) -> CompiledResourceBundle:
    """Bind exactly the candidates requested by one of the legal 16 actions.

    ``M0+R0`` is intentionally valid with an empty candidate mapping.  For all
    other actions there must be exactly one candidate for every requested bit
    and no unrequested candidate may be smuggled into the prompt.
    """

    bits = action_component_bits(requested_action_id)
    expected = {component for component, enabled in bits.items() if enabled}
    unknown = set(candidates) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown component candidates: {sorted(unknown)}")
    if set(candidates) != expected:
        raise ValueError(
            "candidate/action mismatch: "
            f"missing={sorted(expected - set(candidates))}, "
            f"extra={sorted(set(candidates) - expected)}"
        )
    directives = tuple(
        compile_typed_resource(candidates[component], current_user_id=current_user_id)
        for component in COMPONENTS
        if component in expected
    )
    return CompiledResourceBundle(
        requested_action_id=requested_action_id,
        directives=directives,
    )


def response_only_messages(
    *, current_context: str, bundle: CompiledResourceBundle
) -> list[dict[str, str]]:
    """Build a response-only prompt; no use/ignore/reason self-report is requested."""

    if not _clean(current_context):
        raise ValueError("current_context must be non-empty")
    if bundle.directives:
        sections = []
        for index, directive in enumerate(bundle.directives, start=1):
            sections.append(
                f"Resource {index}\nEvidence: {directive.evidence}\n"
                f"Instruction: {directive.response_instruction}"
            )
        resource_text = "\n\n".join(sections)
    else:
        resource_text = (
            "No external resource was requested. Respond only from the visible current "
            "conversation and do not claim memory of an earlier session."
        )
    system = (
        "Write only the next supportive reply. Do not output analysis, JSON, component "
        "status, resource-use claims, or internal labels. Stay grounded in the current "
        "conversation. The supplied resource evidence is permission-bounded context, not "
        "proof of a present cause.\n\n"
        + resource_text
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": current_context.strip()},
    ]


_NO_QUESTION_RE = re.compile(
    r"\b(?:do not|don't|no)\s+(?:ask|question)|\bjust\s+(?:listen|hear me)\b",
    re.IGNORECASE,
)
_STOP_RE = re.compile(r"\b(?:stop|end|goodbye|bye)\b", re.IGNORECASE)


def deterministic_context_only_fallback(current_user_text: str) -> str:
    """Return a fixed, history-free fallback without any model call."""

    text = _clean(current_user_text)
    if not text:
        raise ValueError("current_user_text must be non-empty")
    if _STOP_RE.search(text):
        return "Understood. We can stop here."
    if _NO_QUESTION_RE.search(text):
        return "I hear you. You can take your time, and I will stay with what you choose to share."
    return "I hear that this is difficult. We can stay with what you have said and take it one step at a time."


_INTERNAL_COMPONENT_LABEL_RE = re.compile(r"\b(?:MP|MS|ME|RS|M0|R0)\b")
_INTERNAL_ID_RE = re.compile(
    r"(?:mem|strat|card)_[0-9a-f]{8,}", re.IGNORECASE
)
_PAST_ATTRIBUTION_RE = re.compile(
    r"\b(?:last time|previously|previous|prior|earlier|before|past session|you had tried)\b",
    re.IGNORECASE,
)


def response_guard_errors(
    *,
    response: str,
    bundle: CompiledResourceBundle,
    current_context: str = "",
) -> tuple[str, ...]:
    """Conservative machine-checkable guard; not a semantic quality judge."""

    errors: list[str] = []
    cleaned = _clean(response)
    if not cleaned:
        return ("EMPTY_RESPONSE",)
    if _INTERNAL_COMPONENT_LABEL_RE.search(cleaned) or _INTERNAL_ID_RE.search(cleaned):
        errors.append("INTERNAL_LABEL_OR_ID_LEAK")
    historical = any(item.component in {"MS", "ME"} for item in bundle.directives)
    if historical and any(item.attribution_required for item in bundle.directives):
        if not _PAST_ATTRIBUTION_RE.search(cleaned):
            errors.append("MISSING_PAST_ATTRIBUTION")
    return tuple(errors)
