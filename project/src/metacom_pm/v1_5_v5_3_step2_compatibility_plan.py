"""Outcome-blind structural compatibility plan for all 16 V5.3 actions.

This module answers a narrow pre-generation question: can every legal PM
action be materialized by the *same* typed Step2 implementation with exactly
the requested candidate IDs?  It deliberately does not call a model and does
not infer response quality, risk, or functional contribution.

In particular, a PM-selected ``M0+R0`` row is an ordinary generator action
with no resource evidence.  It is not the same event as an evidence-bearing
action that later fails execution and realizes a separately-accounted
``M0+R0`` fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import ALL_ACTION_IDS, StrictModel
from .io import canonical_json, sha256_file, stable_hex
from .v1_5_typed_resource_adapter import TypedResourceCandidate
from .v1_5_v5_3_typed_response_program import (
    RewritePolicy,
    build_typed_response_program,
    evidence_aware_generation_messages,
)
from .v1_5b_policy_runtime import COMPONENTS, action_component_bits


STEP2_COMPATIBILITY_PROTOCOL = "pm-v1.5-v5.3-step2-compatibility-plan-v1"


class ActionCompatibilityRow(StrictModel):
    action_id: str = Field(min_length=1)
    requested_components: list[str]
    candidate_ids: dict[str, str | None]
    candidate_subtypes: dict[str, str | None]
    evidence_ids: list[str]
    evidence_count: int = Field(ge=0)
    atomic_move_budget: int = Field(ge=0)
    message_roles: list[str]
    messages_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pm_selected_m0_r0_is_ordinary_generation: bool
    structural_status: Literal["PASS"] = "PASS"

    @model_validator(mode="after")
    def coherent(self):
        bits = action_component_bits(self.action_id)
        expected = [component for component in COMPONENTS if bits[component]]
        if self.requested_components != expected:
            raise ValueError("requested_components differ from action bits")
        if set(self.candidate_ids) != set(COMPONENTS):
            raise ValueError("candidate_ids must contain all four component keys")
        if set(self.candidate_subtypes) != set(COMPONENTS):
            raise ValueError("candidate_subtypes must contain all four component keys")
        present = [
            component for component in COMPONENTS if self.candidate_ids[component]
        ]
        if present != expected:
            raise ValueError("candidate presence differs from requested action")
        if any(
            bool(self.candidate_ids[c]) != bool(self.candidate_subtypes[c])
            for c in COMPONENTS
        ):
            raise ValueError("candidate id/subtype presence differs")
        expected_ids = [self.candidate_ids[c] for c in expected]
        if self.evidence_ids != expected_ids:
            raise ValueError("evidence order differs from exact execution candidates")
        if self.evidence_count != len(expected):
            raise ValueError("evidence_count differs from requested action")
        if self.atomic_move_budget != int(bits["RS"]):
            raise ValueError("atomic_move_budget differs from RS bit")
        ordinary_m0 = self.action_id == "M0+R0"
        if self.pm_selected_m0_r0_is_ordinary_generation != ordinary_m0:
            raise ValueError("ordinary M0+R0 flag differs from action identity")
        if self.message_roles != ["system", "user"]:
            raise ValueError("typed Step2 must keep system evidence separate from user text")
        return self


class Step2CompatibilityPlan(StrictModel):
    protocol: Literal["pm-v1.5-v5.3-step2-compatibility-plan-v1"] = (
        STEP2_COMPATIBILITY_PROTOCOL
    )
    status: Literal[
        "STRUCTURAL_PASS_SEMANTIC_COMPATIBILITY_AND_RECOVERY_SELECTION_PENDING"
    ] = "STRUCTURAL_PASS_SEMANTIC_COMPATIBILITY_AND_RECOVERY_SELECTION_PENDING"
    typed_step2_relative_path: str = Field(min_length=1)
    typed_step2_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: list[ActionCompatibilityRow]
    recovery_policy_alternatives: list[str]
    recovery_policy_selected: None = None
    semantic_generator_compatibility_claimed: Literal[False] = False
    generated_response_or_quality_risk_outcome_read: Literal[False] = False
    api_calls: Literal[0] = 0
    plan_identity: str = Field(pattern=r"^v53step2_[0-9a-f]{24}$")

    @model_validator(mode="after")
    def coherent(self):
        actions = [row.action_id for row in self.rows]
        if actions != list(ALL_ACTION_IDS):
            raise ValueError("rows must contain every legal action exactly once in frozen order")
        expected_recovery = [policy.value for policy in RewritePolicy]
        if self.recovery_policy_alternatives != expected_recovery:
            raise ValueError("recovery alternatives differ from executor implementation")
        payload = self.model_dump(mode="json", exclude={"plan_identity"})
        expected = "v53step2_" + stable_hex(canonical_json(payload), n=24)
        if self.plan_identity != expected:
            raise ValueError("plan_identity does not match structural compatibility content")
        return self


def _fixture_candidates(*, mp_preference: bool) -> dict[str, TypedResourceCandidate]:
    mp_candidate = (
        TypedResourceCandidate(
            component="MP",
            subtype="MP_PREFERENCE",
            resource_id="fixture_mp_preference",
            candidate_version="v53_fixture",
            source_kind="profile",
            owner_id="fixture_user",
            preference="Offer one concise option rather than a directive.",
        )
        if mp_preference
        else TypedResourceCandidate(
            component="MP",
            subtype="MP_PROFILE",
            resource_id="fixture_mp_profile",
            candidate_version="v53_fixture",
            source_kind="profile",
            owner_id="fixture_user",
            profile_fact="The user has one short private window in the evening.",
        )
    )
    return {
        "MP": mp_candidate,
        "MS": TypedResourceCandidate(
            component="MS",
            subtype="MS_SESSION_OBSERVATION",
            resource_id="fixture_ms_session",
            candidate_version="v53_fixture",
            source_kind="session",
            owner_id="fixture_user",
            strictly_prior=True,
            age_sessions=2,
            prior_observation="A prior session separated practical load from emotional uncertainty.",
        ),
        "ME": TypedResourceCandidate(
            component="ME",
            subtype="ME_REUSABLE_OUTCOME",
            resource_id="fixture_me_event",
            candidate_version="v53_fixture",
            source_kind="event",
            owner_id="fixture_user",
            strictly_prior=True,
            age_sessions=3,
            past_action="The user paused before responding.",
            observed_outcome="The pause made the conversation less reactive.",
        ),
        "RS": TypedResourceCandidate(
            component="RS",
            subtype="RS_ATOMIC_MOVE",
            resource_id="fixture_rs_card",
            candidate_version="v53_fixture",
            source_kind="strategy",
            support_move="Ask exactly one focused question about the feeling already named.",
            when_to_use="The user asks to narrow an already named feeling.",
            when_not_to_use="The user asks for no questions or wants to stop.",
        ),
    }


def build_step2_compatibility_plan(root: str | Path) -> Step2CompatibilityPlan:
    project_root = Path(root).resolve()
    step2_relative = "src/metacom_pm/v1_5_v5_3_typed_response_program.py"
    rows: list[ActionCompatibilityRow] = []
    for index, action_id in enumerate(ALL_ACTION_IDS):
        bits = action_component_bits(action_id)
        fixtures = _fixture_candidates(mp_preference=bool(index % 2))
        candidates = {
            component: fixtures[component]
            for component in COMPONENTS
            if bits[component]
        }
        expected_ids = {
            component: candidate.resource_id
            for component, candidate in candidates.items()
        }
        program = build_typed_response_program(
            requested_action_id=action_id,
            current_goal="Respond to the current concern without adding unsupported meaning.",
            current_user_id="fixture_user",
            candidates=candidates,
            expected_execution_candidate_ids=expected_ids,
        )
        messages = evidence_aware_generation_messages(
            current_context="User: I want to stay with this one concern for now.",
            program=program,
        )
        candidate_ids = {
            component: candidates[component].resource_id
            if component in candidates
            else None
            for component in COMPONENTS
        }
        candidate_subtypes = {
            component: candidates[component].subtype
            if component in candidates
            else None
            for component in COMPONENTS
        }
        rows.append(
            ActionCompatibilityRow(
                action_id=action_id,
                requested_components=[c for c in COMPONENTS if bits[c]],
                candidate_ids=candidate_ids,
                candidate_subtypes=candidate_subtypes,
                evidence_ids=[item.evidence_id for item in program.evidence],
                evidence_count=len(program.evidence),
                atomic_move_budget=program.atomic_move_budget,
                message_roles=[message["role"] for message in messages],
                messages_sha256=stable_hex(canonical_json(messages), n=64),
                pm_selected_m0_r0_is_ordinary_generation=action_id == "M0+R0",
            )
        )
    payload = {
        "protocol": STEP2_COMPATIBILITY_PROTOCOL,
        "status": "STRUCTURAL_PASS_SEMANTIC_COMPATIBILITY_AND_RECOVERY_SELECTION_PENDING",
        "typed_step2_relative_path": step2_relative,
        "typed_step2_sha256": sha256_file(project_root / step2_relative),
        "rows": rows,
        "recovery_policy_alternatives": [policy.value for policy in RewritePolicy],
        "recovery_policy_selected": None,
        "semantic_generator_compatibility_claimed": False,
        "generated_response_or_quality_risk_outcome_read": False,
        "api_calls": 0,
    }
    identity_payload = {
        key: [item.model_dump(mode="json") for item in value]
        if key == "rows"
        else value
        for key, value in payload.items()
    }
    payload["plan_identity"] = "v53step2_" + stable_hex(
        canonical_json(identity_payload), n=24
    )
    return Step2CompatibilityPlan.model_validate(payload)
