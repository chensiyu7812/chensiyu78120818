"""Intention-to-treat accounting for PM V1.5 V5 Step-1 value learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .v1_5b_policy_runtime import COMPONENTS, Component, action_component_bits


@dataclass(frozen=True)
class ITTExperimentRow:
    """One assigned system outcome under a frozen executor/generator stack.

    Functional use, fallback, and misuse are outcomes of requesting a resource,
    not reasons to discard the row.  Only mechanical assignment/binding/output
    failures make the causal row invalid.
    """

    group_id: str
    state_id: str
    requested_action_id: str
    realized_action_id: str
    assigned_executor_version: str
    actual_executor_version: str
    assigned_generator_version: str
    actual_generator_version: str
    assignment_matches_plan: bool
    candidate_binding_valid: Mapping[Component, bool]
    owner_binding_valid: Mapping[Component, bool]
    resource_presented_to_executor: Mapping[Component, bool]
    final_response: str
    usage_present: bool
    fallback_used: bool = False
    functional_use: Mapping[Component, bool] = field(default_factory=dict)
    material_misuse: bool = False

    def __post_init__(self) -> None:
        action_component_bits(self.requested_action_id)
        action_component_bits(self.realized_action_id)
        for name, values in (
            ("candidate_binding_valid", self.candidate_binding_valid),
            ("owner_binding_valid", self.owner_binding_valid),
            ("resource_presented_to_executor", self.resource_presented_to_executor),
        ):
            if set(values) != set(COMPONENTS):
                raise ValueError(f"{name} must contain exactly MP/MS/ME/RS")
        if set(self.functional_use) - set(COMPONENTS):
            raise ValueError("functional_use contains an unknown component")


@dataclass(frozen=True)
class ITTRowValidity:
    valid: bool
    invalid_reasons: tuple[str, ...]


def requested_treatment_bits(row: ITTExperimentRow) -> dict[Component, bool]:
    """The treatment is assignment/request, never realized use."""

    return action_component_bits(row.requested_action_id)


def validate_itt_row(row: ITTExperimentRow) -> ITTRowValidity:
    reasons: list[str] = []
    requested = requested_treatment_bits(row)
    if not row.assignment_matches_plan:
        reasons.append("ASSIGNMENT_MISMATCH")
    for component in COMPONENTS:
        if not requested[component]:
            continue
        if not row.candidate_binding_valid[component]:
            reasons.append(f"{component}_CANDIDATE_BINDING_INVALID")
        if not row.owner_binding_valid[component]:
            reasons.append(f"{component}_OWNER_BINDING_INVALID")
        if not row.resource_presented_to_executor[component]:
            reasons.append(f"{component}_NOT_PRESENTED_TO_EXECUTOR")
    if not row.final_response.strip():
        reasons.append("MISSING_FINAL_RESPONSE")
    if not row.usage_present:
        reasons.append("MISSING_USAGE")
    if row.assigned_executor_version != row.actual_executor_version:
        reasons.append("EXECUTOR_IDENTITY_DRIFT")
    if row.assigned_generator_version != row.actual_generator_version:
        reasons.append("GENERATOR_IDENTITY_DRIFT")
    return ITTRowValidity(valid=not reasons, invalid_reasons=tuple(reasons))


_POST_ACTION_FEATURE_TOKENS = (
    "realized_action",
    "functional_use",
    "fallback",
    "response_quality",
    "judge",
    "misuse",
    "completion_tokens",
    "response_text",
)


def validate_pre_injection_feature_names(feature_names: list[str]) -> None:
    """Reject post-treatment leakage from Step-1 feature schemas."""

    bad = sorted(
        name
        for name in feature_names
        if any(token in name.lower() for token in _POST_ACTION_FEATURE_TOKENS)
    )
    if bad:
        raise ValueError(f"post-action Step-1 feature(s): {bad}")

