"""Transparent, diagnostic audit of generator use of an already-selected resource.

This instrument does not decide whether PM routing or retrieval was correct and
does not treat an automated judgment as training gold.  It asks only whether a
realized response functionally used, safely ignored, merely echoed, missed, or
misused the resource supplied to the generator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


UseLabel = Literal[
    "grounded_functional_use",
    "correct_nonuse",
    "grounded_surface_only",
    "missed_usable_resource",
    "misuse",
]
ResourceFunction = Literal[
    "support_technique",
    "style_or_pacing",
    "profile_context",
    "session_continuity",
    "event_or_outcome",
    "none",
]
MisuseCategory = Literal[
    "unsupported_personal_claim",
    "stale_or_conflicting_use",
    "overgeneralized_pattern_or_cause",
    "intrusive_or_explicit_boundary_violation",
    "excessive_directiveness",
    "fabricated_recall",
]
ApplicationStatus = Literal["applied", "cannot_apply"]
CannotApplyReason = Literal[
    "none",
    "new_visible_conflict",
    "new_visible_redundancy",
    "unsafe_or_malformed_resource",
    "declared_function_not_executable",
]
ApplicationMode = Literal[
    "behavioral_preference",
    "incremental_profile_context",
    "tentative_prior_session_continuity",
    "tentative_past_event_option",
    "grounded_support_technique",
]
ExecutionVariant = Literal[
    "generic",
    "preference_triggered_constraint",
    "incremental_profile_context",
    "answer_prior_fact",
    "tentative_continuity_check",
    "past_outcome_to_tentative_option",
    "single_atomic_support_move",
]


class GeneratorResourceUseJudgment(BaseModel):
    """One auditable pointwise resource-use judgment."""

    model_config = ConfigDict(extra="forbid")

    resource_use_label: UseLabel
    resource_function: ResourceFunction
    resource_used: Literal["yes", "partial", "no"]
    response_evidence_excerpt: str = Field(min_length=1, max_length=300)
    resource_evidence_excerpt: str = Field(min_length=1, max_length=300)
    misuse_categories: list[MisuseCategory]
    reason: str = Field(min_length=1, max_length=500)
    confidence: int = Field(ge=1, le=4)

    @model_validator(mode="before")
    @classmethod
    def bound_text(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for field, limit in (
            ("response_evidence_excerpt", 300),
            ("resource_evidence_excerpt", 300),
            ("reason", 500),
        ):
            text = normalized.get(field)
            if isinstance(text, str) and len(text) > limit:
                normalized[field] = text[:limit]
        return normalized


class ResourceExecutionOutput(BaseModel):
    """Generator-declared resource execution plus the user-facing response."""

    model_config = ConfigDict(extra="forbid")

    resource_decision: Literal["use", "ignore"]
    resource_function: ResourceFunction
    used_resource_labels: list[Literal["E1", "R1"]]
    resource_evidence_excerpt: str = Field(
        min_length=1,
        max_length=300,
        description="Exact literal substring copied from SELECTED RESOURCE, or [none].",
    )
    response_evidence_excerpt: str = Field(
        min_length=1,
        max_length=300,
        description="Exact literal substring copied from response, or [none].",
    )
    concise_decision_reason: str = Field(min_length=1, max_length=300)
    response: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def bound_output_text(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for field, limit in (
            ("resource_evidence_excerpt", 300),
            ("response_evidence_excerpt", 300),
            ("concise_decision_reason", 300),
            ("response", 2000),
        ):
            text = normalized.get(field)
            if isinstance(text, str) and len(text) > limit:
                normalized[field] = text[:limit]
        return normalized


class ResourceExecutionPlan(BaseModel):
    """Deterministic Step1-to-Step2 contract, produced outside the generator."""

    model_config = ConfigDict(extra="forbid")

    component: Literal["MP", "MS", "ME", "RS"]
    resource_subtype: str = Field(min_length=1, max_length=80)
    expected_function: ResourceFunction
    application_mode: ApplicationMode
    execution_variant: ExecutionVariant = "generic"
    attribution_required: bool
    maximum_support_moves: int = Field(default=1, ge=0, le=1)
    required_contribution: str = Field(min_length=1, max_length=300)
    forbidden_inferences: list[str] = Field(min_length=1, max_length=8)
    fallback_policy: Literal["realize_component_off_then_one_resource_free_retry"]


class ResourceApplicationOutput(BaseModel):
    """Minimal generator output after Step1 has already authorized injection."""

    model_config = ConfigDict(extra="forbid")

    application_status: ApplicationStatus
    applied_function: ResourceFunction
    cannot_apply_reason: CannotApplyReason
    response: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> "ResourceApplicationOutput":
        if self.application_status == "applied":
            if self.applied_function == "none" or self.cannot_apply_reason != "none":
                raise ValueError(
                    "applied output requires a real function and cannot_apply_reason=none"
                )
        elif self.applied_function != "none" or self.cannot_apply_reason == "none":
            raise ValueError(
                "cannot_apply output requires applied_function=none and a bounded reason"
            )
        return self


class ComponentApplicationStatus(BaseModel):
    """One component declaration inside a multi-resource final action."""

    model_config = ConfigDict(extra="forbid")

    component: Literal["MP", "MS", "ME", "RS"]
    application_status: ApplicationStatus
    applied_function: ResourceFunction
    cannot_apply_reason: CannotApplyReason
    resource_support_excerpt: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Exact literal substring copied from this component's own selected "
            "resource, or [none] when cannot_apply."
        ),
    )
    response_evidence_excerpt: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Exact literal substring copied from the final response that shows "
            "this component's contribution, or [none] when cannot_apply."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_evidence_text(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for field in ("resource_support_excerpt", "response_evidence_excerpt"):
            text = normalized.get(field)
            if isinstance(text, str) and len(text) > 300:
                normalized[field] = text[:300]
        return normalized

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> "ComponentApplicationStatus":
        ResourceApplicationOutput(
            application_status=self.application_status,
            applied_function=self.applied_function,
            cannot_apply_reason=self.cannot_apply_reason,
            response="placeholder",
        )
        if self.application_status == "applied":
            if (
                self.resource_support_excerpt == "[none]"
                or self.response_evidence_excerpt == "[none]"
            ):
                raise ValueError("applied component requires both exact evidence excerpts")
        elif (
            self.resource_support_excerpt != "[none]"
            or self.response_evidence_excerpt != "[none]"
        ):
            raise ValueError("cannot_apply component requires both excerpts=[none]")
        return self


class ResourceBundleApplicationOutput(BaseModel):
    """One reply plus an auditable declaration for each requested component."""

    model_config = ConfigDict(extra="forbid")

    component_statuses: list[ComponentApplicationStatus] = Field(
        min_length=1, max_length=4
    )
    response: str = Field(min_length=1, max_length=2000)


def resource_bundle_output_schema_for_components(
    components: list[str] | tuple[str, ...],
) -> type[ResourceBundleApplicationOutput]:
    """Return a call-specific schema that cannot declare unrequested components.

    The generic bundle schema is retained for reading historical artifacts and
    unit-level validation.  Formal generation must use this exact-action schema:
    one status per requested component and no status for any other component.
    """

    ordered = tuple(dict.fromkeys(str(value) for value in components))
    if not ordered or len(ordered) != len(components):
        raise ValueError("requested components must be non-empty and unique")
    if set(ordered) - {"MP", "MS", "ME", "RS"}:
        raise ValueError(f"unknown requested component: {ordered}")
    component_literal = Literal.__getitem__(ordered)
    status_model = create_model(
        "Exact" + "".join(ordered) + "ComponentApplicationStatus",
        __base__=ComponentApplicationStatus,
        component=(component_literal, ...),
    )
    return create_model(
        "Exact" + "".join(ordered) + "ResourceBundleApplicationOutput",
        __base__=ResourceBundleApplicationOutput,
        component_statuses=(
            list[status_model],
            Field(
                min_length=len(ordered),
                max_length=len(ordered),
                description=(
                    "Exactly one status for each and only these components: "
                    + ", ".join(ordered)
                ),
            ),
        ),
    )

COMPONENT_FUNCTION = {
    "RS": "support_technique",
    "MS": "session_continuity",
    "ME": "event_or_outcome",
}

_SURFACE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "but", "by",
    "can", "do", "for", "from", "has", "have", "i", "in", "is", "it",
    "me", "my", "of", "on", "one", "or", "please", "that", "the", "this",
    "to", "use", "user", "was", "when", "with", "you", "your",
}


def _surface_content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _SURFACE_STOPWORDS
    }


def qualify_step2_resource_surface(
    *,
    component: str,
    resource_subtype: str,
    selected_resource: str,
    current_user_text: str,
) -> dict[str, Any]:
    """Static pre-generation qualification for a Step2 treatment surface.

    This does not predict benefit and never reads a generated response.  It only
    blocks known unexecutable or self-redundant treatments before they can be
    mistaken for generator/PM failures.
    """

    resource = " ".join(selected_resource.split())
    current = " ".join(current_user_text.split())
    resource_tokens = _surface_content_tokens(resource)
    current_tokens = _surface_content_tokens(current)
    overlap_rate = (
        len(resource_tokens & current_tokens) / len(resource_tokens)
        if resource_tokens
        else 1.0
    )
    errors: list[str] = []
    if not resource:
        errors.append("EMPTY_RESOURCE")
    # Stable profile facts are compact, so repeating half of their content is
    # already enough to erase the treatment increment.  MS/ME/RS surfaces also
    # contain topic/role scaffolding that legitimately overlaps the query; they
    # require near-complete content overlap before being called an echo.
    echo_threshold = 0.50 if resource_subtype == "MP_PROFILE" else 0.75
    if overlap_rate >= echo_threshold:
        errors.append("CURRENT_ECHO_HIGH_CONTENT_OVERLAP")

    if component == "MP":
        if resource_subtype == "MP_PREFERENCE":
            if not re.search(
                r"\b(?:brief|concise|reflection|question|suggestion|one point|optional|pacing)\b",
                current,
                flags=re.IGNORECASE,
            ):
                errors.append("MP_PREFERENCE_TRIGGER_NOT_VISIBLE")
        elif resource_subtype != "MP_PROFILE":
            errors.append("MP_SUBTYPE_INVALID")
    elif component == "MS":
        if resource_subtype != "MS_SESSION":
            errors.append("MS_SUBTYPE_INVALID")
        if re.search(
            r"\b(?:one exact wording question|an exact wording question|"
            r"a question remained|some detail remained|discussed this topic generally)\b",
            resource,
            flags=re.IGNORECASE,
        ):
            errors.append("MS_META_SUMMARY_WITHOUT_CONCRETE_PAYLOAD")
        if len(resource_tokens) < 5:
            errors.append("MS_PAYLOAD_TOO_THIN")
    elif component == "ME":
        if resource_subtype not in {"ME_EVENT_OUTCOME", "ME_REUSABLE_OUTCOME"}:
            errors.append("ME_SUBTYPE_INVALID")
        if not _RESOURCE_OUTCOME_RE.search(resource):
            errors.append("ME_OUTCOME_OR_MECHANISM_MISSING")
        if not re.search(
            r"\b(?:wrote|tried|chose|paused|asked|used|went|made|did|took|limited)\b",
            resource,
            flags=re.IGNORECASE,
        ):
            errors.append("ME_PAST_ACTION_OR_CHOICE_MISSING")
    elif component == "RS":
        if resource_subtype != "RS_ATOMIC_MOVE":
            errors.append("RS_SUBTYPE_INVALID")
        for field in ("support_move:", "when_to_use:", "when_not_to_use:"):
            if field not in selected_resource:
                errors.append("RS_EXECUTION_FIELD_MISSING_" + field[:-1].upper())
    else:
        errors.append("UNKNOWN_COMPONENT")

    return {
        "component": component,
        "resource_subtype": resource_subtype,
        "resource_content_token_count": len(resource_tokens),
        "current_content_token_count": len(current_tokens),
        "resource_to_current_content_overlap": overlap_rate,
        "current_echo_threshold": echo_threshold,
        "errors": errors,
        "qualified": not errors,
    }


def materialize_strategy_card_for_execution(card: Mapping[str, Any]) -> str:
    """Render the frozen, technique-only Strategy Bank execution surface."""

    fields = ("support_move", "when_to_use", "when_not_to_use")
    values = {
        name: " ".join(str(card.get(name) or "").split()) for name in fields
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(
            f"Strategy Bank card {card.get('card_id')} lacks generator fields: {missing}"
        )
    return "\n".join(f"{name}: {values[name]}" for name in fields)


def _mandatory_application_instruction(plan: ResourceExecutionPlan) -> str:
    if plan.component == "MS":
        if plan.execution_variant == "answer_prior_fact":
            return (
                "If MS is applied, answer the requested past fact directly with a "
                "clause beginning exactly 'Last time, you mentioned...' and one "
                "concrete MS detail; do not turn it into a current cause."
            )
        return (
            "If MS is applied, include a clause beginning exactly "
            "'Last time, you mentioned...' with one concrete MS detail, then make "
            "its current relevance explicitly tentative."
        )
    if plan.component == "ME":
        return (
            "If ME is applied, include a separate clause beginning exactly "
            "'In the past, ...' and fill it with one concrete ME event or outcome."
        )
    if plan.component == "MP" and plan.resource_subtype == "MP_PREFERENCE":
        return (
            "If MP is applied, the response's chosen move must trigger the stored "
            "preference and visibly follow it without announcing a profile. If the "
            "chosen move does not trigger the condition, return cannot_apply rather "
            "than claiming decorative preference use."
        )
    if plan.component == "MP":
        return (
            "If MP is applied, use one incremental profile fact to clarify the "
            "present request; do not merely repeat current text."
        )
    return (
        "If RS is applied, execute exactly the supplied support_move within its "
        "when_to_use and when_not_to_use boundary, and do not append a second "
        "support move."
    )


def build_resource_execution_plan(
    *, component: str, resource_subtype: str, current_user_text: str = ""
) -> ResourceExecutionPlan:
    """Compile one source-matched plan without reading a response or outcome."""

    if resource_subtype == "MP_PREFERENCE":
        expected: ResourceFunction = "style_or_pacing"
        mode: ApplicationMode = "behavioral_preference"
        attribution = False
        contribution = (
            "The reply must visibly follow the selected pacing or interaction "
            "preference; do not announce or quote the preference."
        )
        forbidden = [
            "announcing a hidden profile or preference",
            "adding a second question or suggestion that violates the preference",
        ]
        variant: ExecutionVariant = "preference_triggered_constraint"
    elif resource_subtype == "MP_PROFILE":
        expected = "profile_context"
        mode = "incremental_profile_context"
        attribution = False
        contribution = (
            "Use only the incremental current-relevant profile fact to clarify "
            "the user's present request; do not merely repeat visible context."
        )
        forbidden = [
            "treating a profile fact as a cause",
            "repeating a fact already visible as if memory added it",
        ]
        variant = "incremental_profile_context"
    elif component == "MS":
        expected = "session_continuity"
        mode = "tentative_prior_session_continuity"
        attribution = True
        contribution = (
            "Use the prior-session distinction to ask one tentative continuity "
            "check or to answer the present uncertainty; it must change the reply."
        )
        forbidden = [
            "promoting a prior-session detail to a current fact",
            "asserting a stable cause or pattern",
            "mentioning history and then giving an unrelated generic response",
        ]
        variant = (
            "answer_prior_fact"
            if re.search(
                r"\b(?:remind me|what (?:was|did)|recorded earlier|last time)\b",
                current_user_text,
                flags=re.IGNORECASE,
            )
            else "tentative_continuity_check"
        )
    elif component == "ME":
        expected = "event_or_outcome"
        mode = "tentative_past_event_option"
        attribution = True
        contribution = (
            "Use the specific past event or outcome as one tentative option for "
            "the present turn, while preserving that it occurred in the past."
        )
        forbidden = [
            "presupposing the old task or constraint still exists",
            "generalizing one event into a stable personal rule",
            "dropping the past provenance",
        ]
        variant = "past_outcome_to_tentative_option"
    elif component == "RS":
        expected = "support_technique"
        mode = "grounded_support_technique"
        attribution = False
        contribution = (
            "Realize the selected support move in the reply using only claims "
            "grounded in the visible exchange."
        )
        forbidden = [
            "quoting card metadata",
            "using a technique outside its boundary",
            "surface paraphrase without executing the move",
        ]
        variant = "single_atomic_support_move"
    else:
        raise ValueError(
            f"unsupported component/subtype: component={component!r}, "
            f"resource_subtype={resource_subtype!r}"
        )
    return ResourceExecutionPlan(
        component=component,
        resource_subtype=resource_subtype,
        expected_function=expected,
        application_mode=mode,
        execution_variant=variant,
        attribution_required=attribution,
        maximum_support_moves=1,
        required_contribution=contribution,
        forbidden_inferences=forbidden,
        fallback_policy="realize_component_off_then_one_resource_free_retry",
    )


def resource_execution_messages(
    *,
    component: str,
    resource_subtype: str,
    visible_context: str,
    selected_resource: str,
    resource_label: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build a structured Step2 prompt with an auditable execution trace."""

    expected = (
        "style_or_pacing"
        if resource_subtype == "MP_PREFERENCE"
        else "profile_context"
        if resource_subtype == "MP_PROFILE"
        else COMPONENT_FUNCTION[component]
    )
    user = f"""{visible_context}

SELECTED RESOURCE {resource_label}:
{selected_resource}

The resource was selected upstream as a plausible opportunity, but the current
message still has priority. Decide whether to use or ignore it in this exact
reply. If used, it must perform the expected function `{expected}` rather than
being mentioned for show. MP preference is normally applied as style/pacing;
MP profile directly clarifies the request; MS adds prior-session continuity;
ME uses the specific event/outcome without generalizing; RS realizes the support
technique with claims grounded in visible text.

Return the structured object only. If use: resource_decision=`use`,
resource_function=`{expected}`, used_resource_labels=[`{resource_label}`], and
resource_evidence_excerpt is copied exactly from SELECTED RESOURCE while
response_evidence_excerpt is copied exactly from the response and shows how the
resource affected it. If ignore: resource_function=`none`, labels=[], and both
excerpts=`[none]`. concise_decision_reason must state the observable use/ignore reason.
Never mention `{resource_label}`, retrieval, memory, or strategy cards in the
user-facing response. The response must stay concise and may use at most one
question OR one suggestion, not both or a list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def resource_execution_messages_v3(
    *,
    component: str,
    resource_subtype: str,
    visible_context: str,
    selected_resource: str,
    resource_label: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build the one repaired execution prompt after the dual human check."""

    expected = (
        "style_or_pacing"
        if resource_subtype == "MP_PREFERENCE"
        else "profile_context"
        if resource_subtype == "MP_PROFILE"
        else COMPONENT_FUNCTION[component]
    )
    source_rule = {
        "MP_PREFERENCE": (
            "Apply the preference behaviorally. Do not announce it."
        ),
        "MP_PROFILE": (
            "Use the profile only when it adds current-relevant information not "
            "already stated in the visible exchange; otherwise ignore it as redundant."
        ),
        "MS_SESSION": (
            "The summary describes an earlier session, not a guaranteed current "
            "cause. Attribute it to the earlier conversation and tentatively check "
            "whether it still fits unless the current text independently confirms it."
        ),
        "ME_EVENT_OUTCOME": (
            "The event is one past example, not a current fact or rule. If used, "
            "make the past provenance natural and visible and offer it as a "
            "tentative option; never presuppose an old task, cause, or constraint."
        ),
    }.get(
        resource_subtype,
        "Realize the strategy using claims grounded in the visible exchange.",
    )
    user = f"""{visible_context}

SELECTED RESOURCE {resource_label}:
{selected_resource}

The resource was selected upstream as a plausible opportunity, but the current
message still has priority. Decide whether to use or ignore it in this exact
reply. If used, it must perform `{expected}` rather than being mentioned for
show. {source_rule}

Hard execution rules:
- Never convert past evidence into an unqualified current fact, cause, stable
  pattern, or definite old detail. Avoid 'not just X' and 'the old task/cause'
  unless current text independently confirms it.
- A memory use must visibly do work. Except for a behavioral MP preference,
  vague wording such as 'similar comfort' or generic advice is not use.
- Ignore only for an exact mismatch, conflict, or redundancy visible in the
  current context and resource. Do not invent a request, boundary, weekday, or
  other current detail in the decision reason.

Return the structured object only. If use: resource_decision=`use`,
resource_function=`{expected}`, used_resource_labels=[`{resource_label}`], and
both evidence excerpts are exact. If ignore: resource_function=`none`, labels=[],
and both excerpts=`[none]`. The reason is audit telemetry, never training gold.
Never mention `{resource_label}`, retrieval, memory, or strategy cards in the
user-facing response. Natural phrases such as 'you mentioned last time' are
allowed. The response must stay concise and may use at most one question OR one
suggestion, not both or a list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def resource_application_messages_v1_5b(
    *,
    plan: ResourceExecutionPlan,
    visible_context: str,
    selected_resource: str,
    resource_label: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build the V1.5b prompt that executes, rather than reopens, Step1.

    Step1 has already authorized this candidate.  The generator may report
    ``cannot_apply`` only when a bounded issue becomes visible at execution
    time.  It is not asked to copy IDs, resource spans, response spans, or a
    free-form rationale.  Those fields previously measured trace-writing more
    than functional resource use.
    """

    plan_json = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)
    mandatory = _mandatory_application_instruction(plan)
    user = f"""{visible_context}

EXECUTION PLAN:
{plan_json}

SELECTED RESOURCE {resource_label}:
{selected_resource}

Step1 has already authorized this resource for injection. Execute the plan in
the user-facing reply. Do not silently make a second free use/ignore decision.
Return `application_status=applied` and
`applied_function={plan.expected_function}` only when the resource performs the
required contribution, rather than being mentioned or paraphrased for show.

Return `application_status=cannot_apply`, `applied_function=none`, and exactly
one bounded cannot_apply_reason only if the visible exchange reveals an exact
new conflict, exact redundancy, unsafe/malformed resource, or inability to
perform the declared function. Do not invent a boundary or current fact to use
this escape route. A cannot_apply result will realize this component as OFF and
trigger at most one resource-free fallback whose cost is counted.

If attribution_required is true, make past provenance natural and explicit
(for example, 'last time you mentioned...' or 'something that helped before...')
and keep its present relevance tentative. Never promote prior evidence into a
current fact, cause, diagnosis, or stable pattern. Never mention
`{resource_label}`, retrieval, memory, a policy, or a strategy card. Keep the
reply concise and do not add a list of tasks.

Mandatory visible execution: {mandatory}
Do not report application_status=applied unless the user-facing response obeys
that instruction. Return the structured object only."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def resource_bundle_application_messages_v1_5b(
    *,
    plans: Mapping[str, ResourceExecutionPlan],
    visible_context: str,
    selected_resources: Mapping[str, str],
    resource_labels: Mapping[str, str],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Compile one final-action prompt for one to four authorized resources.

    The response is generated once.  Each component nevertheless receives its
    own bounded status so requested and realized actions remain distinguishable.
    """

    components = tuple(component for component in ("MP", "MS", "ME", "RS") if component in plans)
    if not components or set(plans) != set(selected_resources) or set(plans) != set(resource_labels):
        raise ValueError("plans, resources, and labels must contain the same non-empty components")
    blocks = []
    mandatory: list[str] = []
    for component in components:
        plan = plans[component]
        if plan.component != component:
            raise ValueError(f"plan/component mismatch for {component}")
        blocks.append(
            "EXECUTION PLAN "
            + component
            + ":\n"
            + json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)
            + f"\nSELECTED RESOURCE {resource_labels[component]}:\n"
            + selected_resources[component]
        )
        mandatory.append(f"- {component}: {_mandatory_application_instruction(plan)}")
    user = f"""{visible_context}

{chr(10).join(blocks)}

Step1 has already authorized exactly these components. Compose one concise,
natural user-facing reply. Each resource must perform its declared contribution
instead of appearing as a decorative history mention. The current message has
priority over every resource.

Return one component_statuses entry for every and only the requested components:
{', '.join(components)}. Use application_status=applied only when that component
really performs its expected function in the shared response. Use cannot_apply
only for an exact new conflict, exact redundancy, unsafe/malformed resource, or
declared function that truly cannot be executed. Do not invent a reason to opt out.
For every applied component, resource_support_excerpt must be copied exactly
from that component's own SELECTED RESOURCE, and response_evidence_excerpt must
be copied exactly from the final response and isolate that component's actual
contribution. Evidence from one component may never justify another component.
For cannot_apply, both excerpts must be exactly `[none]`.

For MS/ME, preserve past provenance and keep current relevance tentative. Never
fabricate that a method helped, never promote an old cause/task into a current
fact, and never generalize one event into a stable personal rule. Do not mention
resource labels, memory, retrieval, a policy, or strategy cards. Ask at most one
question OR offer at most one suggestion, never both or a list.

Mandatory visible execution for every component marked applied:
{chr(10).join(mandatory)}
MS and ME require separate provenance clauses when both are applied. Do not mark
a component applied unless its visible instruction is satisfied. The reply may
use up to three concise sentences to realize multiple components. Return only
the structured object."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def validate_resource_execution(
    *,
    output: ResourceExecutionOutput,
    resource_label: str,
    expected_function: str,
    selected_resource: str,
) -> dict[str, bool]:
    """Machine-check a generator execution declaration without a judge."""

    if output.resource_decision == "use":
        bookkeeping = (
            output.resource_function == expected_function
            and output.used_resource_labels == [resource_label]
            and output.resource_evidence_excerpt != "[none]"
            and output.response_evidence_excerpt != "[none]"
        )
    else:
        bookkeeping = (
            output.resource_function == "none"
            and output.used_resource_labels == []
            and output.resource_evidence_excerpt == "[none]"
            and output.response_evidence_excerpt == "[none]"
        )
    excerpt_literal = (
        output.response_evidence_excerpt == "[none]"
        or output.response_evidence_excerpt in output.response
    )
    resource_excerpt_literal = (
        output.resource_evidence_excerpt == "[none]"
        or output.resource_evidence_excerpt in selected_resource
    )
    label_hidden = resource_label not in output.response
    return {
        "decision_bookkeeping_valid": bookkeeping,
        "response_excerpt_literal": excerpt_literal,
        "resource_excerpt_literal": resource_excerpt_literal,
        "resource_label_hidden_from_user": label_hidden,
        "all_machine_checks_pass": (
            bookkeeping
            and excerpt_literal
            and resource_excerpt_literal
            and label_hidden
        ),
    }


_NATURAL_PAST_ATTRIBUTION_RE = re.compile(
    r"\b(?:last time|last session|last conversation|previously|"
    r"(?:previous|prior|earlier) (?:session|conversation)|"
    r"in the past|from (?:a|the|your) past experience|"
    r"you (?:said|mentioned|shared) (?:before|previously|earlier|last time)|"
    r"had helped|helped before|worked for you)\b",
    flags=re.IGNORECASE,
)
_FABRICATED_OUTCOME_ATTRIBUTION_RE = re.compile(
    r"\b(?:you found (?:it |that |[^.!?]{0,80})?(?:helpful|useful)|"
    r"(?:it|that|this) helped last time|worked for you|"
    r"something that helped before)\b",
    flags=re.IGNORECASE,
)
_RESOURCE_OUTCOME_RE = re.compile(
    r"\b(?:helped|worked|useful|made [^.!?]{0,80} easier|"
    r"reduced|eased|clarified|successful)\b",
    flags=re.IGNORECASE,
)
_UNQUALIFIED_TEMPORAL_PROMOTION_RE = re.compile(
    r"\b(?:is affecting|are affecting|is still present|are still present|"
    r"that's contributing|that is contributing|which of those|"
    r"intertwined with)\b",
    flags=re.IGNORECASE,
)
_UNSAFE_SOCIAL_AVOIDANCE_RE = re.compile(
    r"\b(?:limit|avoid|reduce|stop)\b[^.!?]{0,90}"
    r"\b(?:interactions?|people|places|friends?|social contact|social circle)\b",
    flags=re.IGNORECASE,
)
_PANIC_WRITING_CONFLICT_RE = re.compile(
    r"\b(?:write|writing|put [^.!?]{0,40}(?:paper|down))\b",
    flags=re.IGNORECASE,
)
_WRITING_DIRECTIVE_RE = re.compile(
    r"\b(?:try|start|begin|take a moment to)\b[^.!?]{0,80}"
    r"\b(?:write|writing|put [^.!?]{0,30}(?:paper|down))\b",
    flags=re.IGNORECASE,
)


def validate_resource_application(
    *,
    output: ResourceApplicationOutput,
    plan: ResourceExecutionPlan,
    resource_label: str,
    selected_resource: str = "",
    current_user_text: str = "",
) -> dict[str, bool]:
    """Validate execution-state bookkeeping and provenance without a judge."""

    if output.application_status == "applied":
        bookkeeping = (
            output.applied_function == plan.expected_function
            and output.cannot_apply_reason == "none"
        )
        attribution = (
            not plan.attribution_required
            or bool(_NATURAL_PAST_ATTRIBUTION_RE.search(output.response))
        )
    else:
        bookkeeping = (
            output.applied_function == "none"
            and output.cannot_apply_reason != "none"
        )
        attribution = True
    label_hidden = resource_label not in output.response
    infrastructure_hidden = not re.search(
        r"\b(?:retrieval|memory item|strategy card|policy manager)\b",
        output.response,
        flags=re.IGNORECASE,
    )
    fabricated_recall_absent = not (
        _FABRICATED_OUTCOME_ATTRIBUTION_RE.search(output.response)
        and selected_resource
        and not _RESOURCE_OUTCOME_RE.search(selected_resource)
    )
    unqualified_temporal_promotion_absent = not (
        plan.attribution_required
        and _UNQUALIFIED_TEMPORAL_PROMOTION_RE.search(output.response)
    )
    unsafe_social_avoidance_absent = not _UNSAFE_SOCIAL_AVOIDANCE_RE.search(
        output.response
    )
    known_current_conflict_respected = not (
        re.search(r"\bpanic\b", current_user_text, flags=re.IGNORECASE)
        and _PANIC_WRITING_CONFLICT_RE.search(current_user_text)
        and _WRITING_DIRECTIVE_RE.search(output.response)
    )
    all_checks = bool(
        bookkeeping
        and attribution
        and label_hidden
        and infrastructure_hidden
        and fabricated_recall_absent
        and unqualified_temporal_promotion_absent
        and unsafe_social_avoidance_absent
        and known_current_conflict_respected
    )
    fallback_required = not all_checks or output.application_status == "cannot_apply"
    return {
        "status_bookkeeping_valid": bool(bookkeeping),
        "required_past_attribution_present": bool(attribution),
        "resource_label_hidden_from_user": bool(label_hidden),
        "infrastructure_hidden_from_user": bool(infrastructure_hidden),
        "fabricated_recall_absent": bool(fabricated_recall_absent),
        "unqualified_temporal_promotion_absent": bool(
            unqualified_temporal_promotion_absent
        ),
        "unsafe_social_avoidance_absent": bool(unsafe_social_avoidance_absent),
        "known_current_conflict_respected": bool(known_current_conflict_respected),
        "fallback_required": fallback_required,
        "realized_component_on": bool(
            output.application_status == "applied" and all_checks
        ),
        "all_machine_checks_pass": all_checks,
    }


def validate_resource_bundle_application(
    *,
    output: ResourceBundleApplicationOutput,
    plans: Mapping[str, ResourceExecutionPlan],
    resource_labels: Mapping[str, str],
    selected_resources: Mapping[str, str],
    current_user_text: str = "",
) -> dict[str, Any]:
    """Validate a multi-component reply without collapsing routing and use.

    Every applied component is checked only against its own literal response and
    resource evidence spans.  The resource union is never allowed to authorize a
    component claim.  Any unsafe or unbound applied component invalidates the
    shared response and forces a resource-free fallback.  Bounded cannot-apply
    declarations simply realize that component OFF; a fallback is needed only
    when none of the requested components survives.
    """

    expected = set(plans)
    if not expected or expected != set(resource_labels) or expected != set(selected_resources):
        raise ValueError("bundle validation inputs must have identical non-empty component keys")
    by_component: dict[str, list[ComponentApplicationStatus]] = {}
    for row in output.component_statuses:
        by_component.setdefault(row.component, []).append(row)
    requested_statuses: dict[str, ComponentApplicationStatus] = {}
    conflicting_requested_statuses: list[str] = []
    for component in expected:
        values = by_component.get(component, [])
        signatures = {
            (
                row.application_status,
                row.applied_function,
                row.cannot_apply_reason,
                row.resource_support_excerpt,
                row.response_evidence_excerpt,
            )
            for row in values
        }
        if len(values) == 1 and len(signatures) == 1:
            requested_statuses[component] = values[0]
        elif values:
            conflicting_requested_statuses.append(component)
    extraneous_components = set(by_component) - expected
    requested_component_bookkeeping_valid = (
        set(requested_statuses) == expected
        and set(by_component) == expected
        and not conflicting_requested_statuses
        and not extraneous_components
        and all(len(by_component[component]) == 1 for component in expected)
    )
    extraneous_statuses = [
        row.model_dump(mode="json")
        for component, values in by_component.items()
        if component not in expected
        for row in values
    ]
    duplicate_identical_requested_status_count = sum(
        max(0, len(by_component.get(component, [])) - 1)
        for component in expected
    )
    component_checks: dict[str, dict[str, bool]] = {}
    for component in sorted(expected):
        status = requested_statuses.get(component)
        if status is None:
            continue
        own_resource = selected_resources[component]
        evidence_bookkeeping = bool(
            (
                status.application_status == "applied"
                and status.resource_support_excerpt != "[none]"
                and status.response_evidence_excerpt != "[none]"
            )
            or (
                status.application_status == "cannot_apply"
                and status.resource_support_excerpt == "[none]"
                and status.response_evidence_excerpt == "[none]"
            )
        )
        resource_excerpt_literal = bool(
            status.resource_support_excerpt == "[none]"
            or status.resource_support_excerpt in own_resource
        )
        response_excerpt_literal = bool(
            status.response_evidence_excerpt == "[none]"
            or status.response_evidence_excerpt in output.response
        )
        component_output = ResourceApplicationOutput(
            application_status=status.application_status,
            applied_function=status.applied_function,
            cannot_apply_reason=status.cannot_apply_reason,
            response=(
                status.response_evidence_excerpt
                if status.application_status == "applied"
                else output.response
            ),
        )
        semantic_checks = validate_resource_application(
            output=component_output,
            plan=plans[component],
            resource_label=resource_labels[component],
            selected_resource=own_resource,
            current_user_text=current_user_text,
        )
        component_checks[component] = {
            **semantic_checks,
            "component_evidence_bookkeeping_valid": evidence_bookkeeping,
            "component_resource_excerpt_literal": resource_excerpt_literal,
            "component_response_excerpt_literal": response_excerpt_literal,
            "component_specific_binding_pass": bool(
                evidence_bookkeeping
                and resource_excerpt_literal
                and response_excerpt_literal
            ),
            "all_machine_checks_pass": bool(
                semantic_checks["all_machine_checks_pass"]
                and evidence_bookkeeping
                and resource_excerpt_literal
                and response_excerpt_literal
            ),
        }
    global_resource_labels_hidden = all(
        label not in output.response for label in resource_labels.values()
    )
    global_infrastructure_hidden = not re.search(
        r"\b(?:retrieval|memory item|strategy card|policy manager)\b",
        output.response,
        flags=re.IGNORECASE,
    )
    global_unsafe_social_avoidance_absent = not _UNSAFE_SOCIAL_AVOIDANCE_RE.search(
        output.response
    )
    global_known_current_conflict_respected = not (
        re.search(r"\bpanic\b", current_user_text, flags=re.IGNORECASE)
        and _PANIC_WRITING_CONFLICT_RE.search(current_user_text)
        and _WRITING_DIRECTIVE_RE.search(output.response)
    )
    global_outcome_claim_present = bool(
        _FABRICATED_OUTCOME_ATTRIBUTION_RE.search(output.response)
    )
    global_outcome_claim_component_bound = bool(
        not global_outcome_claim_present
        or any(
            status.application_status == "applied"
            and bool(
                _FABRICATED_OUTCOME_ATTRIBUTION_RE.search(
                    status.response_evidence_excerpt
                )
            )
            and bool(_RESOURCE_OUTCOME_RE.search(selected_resources[component]))
            for component, status in requested_statuses.items()
        )
    )
    global_checks_pass = bool(
        global_resource_labels_hidden
        and global_infrastructure_hidden
        and global_unsafe_social_avoidance_absent
        and global_known_current_conflict_respected
        and global_outcome_claim_component_bound
    )
    unsafe_applied_component = any(
        requested_statuses[component].application_status == "applied"
        and not checks["all_machine_checks_pass"]
        for component, checks in component_checks.items()
    )
    realized = {
        component: bool(
            requested_component_bookkeeping_valid
            and component in requested_statuses
            and requested_statuses[component].application_status == "applied"
            and component_checks[component]["all_machine_checks_pass"]
        )
        for component in expected
    }
    realized_count = sum(realized.values())
    fallback_required = bool(
        not requested_component_bookkeeping_valid
        or unsafe_applied_component
        or not global_checks_pass
        or realized_count == 0
    )
    if fallback_required:
        realized = {component: False for component in expected}
    return {
        "exact_component_bookkeeping": requested_component_bookkeeping_valid,
        "requested_component_bookkeeping_valid": requested_component_bookkeeping_valid,
        "extraneous_component_statuses_ignored_for_routing": extraneous_statuses,
        "duplicate_identical_requested_status_count": duplicate_identical_requested_status_count,
        "conflicting_requested_status_components": conflicting_requested_statuses,
        "component_checks": component_checks,
        "global_resource_labels_hidden": global_resource_labels_hidden,
        "global_infrastructure_hidden": bool(global_infrastructure_hidden),
        "global_unsafe_social_avoidance_absent": bool(
            global_unsafe_social_avoidance_absent
        ),
        "global_known_current_conflict_respected": bool(
            global_known_current_conflict_respected
        ),
        "global_outcome_claim_component_bound": global_outcome_claim_component_bound,
        "global_checks_pass": global_checks_pass,
        "component_realized_on": realized,
        "unsafe_applied_component": unsafe_applied_component,
        "fallback_required": fallback_required,
        "all_machine_checks_pass": bool(
            requested_component_bookkeeping_valid
            and not unsafe_applied_component
            and global_checks_pass
        ),
    }


def generator_resource_use_messages(
    *,
    component: str,
    resource_subtype: str,
    visible_context: str,
    selected_resource: str,
    response: str,
) -> list[dict[str, str]]:
    """Build a pointwise prompt that never reveals the generation arm."""

    expected = (
        "style_or_pacing or profile_context, according to the selected MP subtype"
        if component == "MP"
        else COMPONENT_FUNCTION[component]
    )
    system = """You are auditing one already-generated emotional-support reply.
Upstream routing and item retrieval have already been fixed; do not score them,
do not compare general reply quality, and do not guess the model or prompt arm.

Decide only how the supplied resource affected this response:
- grounded_functional_use: a supported resource changes the reply's content,
  framing, style, pacing, continuity, or support move in a useful discernible way;
- correct_nonuse: the response does not use the resource, and ignoring it is
  reasonable because the current request is already satisfied or the resource
  would be redundant or awkward;
- grounded_surface_only: the response mentions or paraphrases the resource but
  it performs no useful function;
- missed_usable_resource: the response does not use a clearly usable supplied
  resource and therefore misses the intended contribution;
- misuse: the response uses the resource but introduces a material unsupported,
  stale/conflicting, overgeneralized, intrusive/boundary, or excessively
  directive claim. Misuse takes priority over every other label.

Important source rules:
- MP preference may be used behaviorally without being mentioned; an MP profile
  must directly clarify the current request.
- MS should add prior-session continuity not already visible.
- ME should use the specific prior event/outcome without turning it into a rule.
- RS should realize the technique while grounding claims in visible text.
- Merely mentioning history is not a benefit. Do not reward verbosity.
- The current message and explicit interaction boundary always override a resource.

For grounded_functional_use, grounded_surface_only, misuse, or
missed_usable_resource, copy one exact literal substring from the response and
one exact literal substring from SELECTED_RESOURCE. For correct_nonuse only,
both excerpts must be "[none]". A missed_usable_resource has resource_used=no
but resource_function names the function that was missed. Return only the
required JSON object."""
    payload = {
        "component": component,
        "resource_subtype": resource_subtype,
        "expected_resource_function": expected,
        "visible_context": visible_context,
        "selected_resource": selected_resource,
        "response": response,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def validate_literal_excerpts(
    *, judgment: GeneratorResourceUseJudgment, response: str, resource: str
) -> bool:
    """Validate evidence literally; do not repair a semantic judgment."""

    response_ok = (
        judgment.response_evidence_excerpt == "[none]"
        or judgment.response_evidence_excerpt in response
    )
    resource_ok = (
        judgment.resource_evidence_excerpt == "[none]"
        or judgment.resource_evidence_excerpt in resource
    )
    return response_ok and resource_ok


def validate_judgment_contract(
    *, judgment: GeneratorResourceUseJudgment, expected_function: str
) -> bool:
    """Check cross-field semantics without making provider parsing fail."""

    label = judgment.resource_use_label
    used = judgment.resource_used != "no"
    response_excerpt = judgment.response_evidence_excerpt != "[none]"
    resource_excerpt = judgment.resource_evidence_excerpt != "[none]"
    if label in {"grounded_functional_use", "grounded_surface_only", "misuse"}:
        fields_ok = (
            used
            and response_excerpt
            and resource_excerpt
            and judgment.resource_function == expected_function
        )
    elif label == "correct_nonuse":
        fields_ok = (
            not used
            and not response_excerpt
            and not resource_excerpt
            and judgment.resource_function == "none"
        )
    else:
        fields_ok = (
            not used
            and response_excerpt
            and resource_excerpt
            and judgment.resource_function == expected_function
        )
    categories_ok = (
        bool(judgment.misuse_categories)
        if label == "misuse"
        else not judgment.misuse_categories
    )
    return fields_ok and categories_ok
