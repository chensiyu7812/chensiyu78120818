"""Pre-outcome contract for the one-time V5.3 Step2 recovery comparison.

The comparison is deliberately separated from PM effect generation.  It uses
only historically consumed development cases, shares one first-pass model
call between both alternatives, and branches *only after* a structural error:
either one bounded rewrite or the deterministic M0 fallback.  Cases used here
are permanently excluded from P3/P4/P5 outcome panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import ALL_ACTION_IDS, StrictModel
from .io import canonical_json, sha256_file, stable_hex
from .v1_5_v5_3_typed_response_program import RewritePolicy


RECOVERY_PROTOCOL = "pm-v1.5-v5.3-step2-recovery-qualification-v1"


class FileBinding(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RecoveryQualificationCase(StrictModel):
    case_id: str = Field(pattern=r"^v53recoverycase_[0-9a-f]{24}$")
    state_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    semantic_family: str = Field(min_length=1)
    requested_action_id: str = Field(min_length=1)
    seed_label: str = Field(min_length=1)
    typed_program_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    first_pass_messages_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ids: list[str]
    source_is_historically_consumed_development: Literal[True] = True
    excluded_from_p3_p4_p5: Literal[True] = True

    @model_validator(mode="after")
    def coherent(self):
        if self.requested_action_id not in ALL_ACTION_IDS:
            raise ValueError("requested_action_id is not one of the legal 16 actions")
        expected = recovery_case_id(
            state_id=self.state_id,
            requested_action_id=self.requested_action_id,
            seed_label=self.seed_label,
            typed_program_sha256=self.typed_program_sha256,
        )
        if self.case_id != expected:
            raise ValueError("case_id does not match its frozen inputs")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("evidence_ids contain duplicates")
        return self


class RecoveryQualificationPlan(StrictModel):
    protocol: Literal["pm-v1.5-v5.3-step2-recovery-qualification-v1"] = (
        RECOVERY_PROTOCOL
    )
    status: Literal["METHOD_FROZEN_CASES_AND_BUDGET_PENDING"] = (
        "METHOD_FROZEN_CASES_AND_BUDGET_PENDING"
    )
    typed_step2: FileBinding
    cases: list[RecoveryQualificationCase]
    alternatives: list[str]
    shared_first_pass_call: Literal[True] = True
    direct_fallback_extra_calls: Literal[0] = 0
    bounded_rewrite_maximum_extra_calls: Literal[1] = 1
    branch_only_after_first_pass_structural_error: Literal[True] = True
    default_if_inconclusive: Literal["deterministic_fallback"] = (
        "deterministic_fallback"
    )
    decision_rule: dict[str, object]
    accounting_fields: list[str]
    generated_response_or_quality_risk_outcome_read: Literal[False] = False
    api_calls: Literal[0] = 0
    plan_identity: str = Field(pattern=r"^v53recovery_[0-9a-f]{24}$")

    @model_validator(mode="after")
    def coherent(self):
        if self.alternatives != [policy.value for policy in RewritePolicy]:
            raise ValueError("recovery alternatives differ from typed Step2")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate recovery qualification case")
        if self.decision_rule.get("outcomes_may_change_step2_prompt") is not False:
            raise ValueError("qualification outcomes cannot reopen Step2 prompt editing")
        payload = self.model_dump(mode="json", exclude={"plan_identity"})
        expected = "v53recovery_" + stable_hex(canonical_json(payload), n=24)
        if self.plan_identity != expected:
            raise ValueError("plan_identity does not match recovery plan")
        return self


def recovery_case_id(
    *,
    state_id: str,
    requested_action_id: str,
    seed_label: str,
    typed_program_sha256: str,
) -> str:
    return "v53recoverycase_" + stable_hex(
        state_id,
        requested_action_id,
        seed_label,
        typed_program_sha256,
        n=24,
    )


def build_recovery_qualification_plan(
    root: str | Path,
    *,
    cases: list[RecoveryQualificationCase] | None = None,
) -> RecoveryQualificationPlan:
    project_root = Path(root).resolve()
    relative = "src/metacom_pm/v1_5_v5_3_typed_response_program.py"
    payload = {
        "protocol": RECOVERY_PROTOCOL,
        "status": "METHOD_FROZEN_CASES_AND_BUDGET_PENDING",
        "typed_step2": FileBinding(
            relative_path=relative,
            sha256=sha256_file(project_root / relative),
        ),
        "cases": list(cases or []),
        "alternatives": [policy.value for policy in RewritePolicy],
        "shared_first_pass_call": True,
        "direct_fallback_extra_calls": 0,
        "bounded_rewrite_maximum_extra_calls": 1,
        "branch_only_after_first_pass_structural_error": True,
        "default_if_inconclusive": "deterministic_fallback",
        "decision_rule": {
            "comparison_population": "first_pass_structural_failures_only",
            "primary": "adoptable_reply_rate_after_recovery",
            "adoptable_requires": [
                "no_material_grounding_risk",
                "no_critical_event",
                "requested_evidence_functionally_contributes_when_evidence_bearing",
                "atomic_move_contract_satisfied",
            ],
            "rewrite_can_be_selected_only_if": (
                "it yields more adoptable evidence-bearing replies than direct fallback "
                "without a higher material-risk count"
            ),
            "any_fabricated_recall_wrong_owner_or_explicit_boundary_event_selects": (
                "deterministic_fallback"
            ),
            "outcomes_may_change_step2_prompt": False,
            "additional_repair_round_allowed": False,
        },
        "accounting_fields": [
            "first_pass_calls",
            "rewrite_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "api_cost",
            "latency",
            "first_pass_errors",
            "final_guard_errors",
            "requested_action",
            "realized_action",
            "fallback_reason",
        ],
        "generated_response_or_quality_risk_outcome_read": False,
        "api_calls": 0,
    }
    identity_payload = {
        key: value.model_dump(mode="json")
        if isinstance(value, StrictModel)
        else [item.model_dump(mode="json") for item in value]
        if key == "cases"
        else value
        for key, value in payload.items()
    }
    payload["plan_identity"] = "v53recovery_" + stable_hex(
        canonical_json(identity_payload), n=24
    )
    return RecoveryQualificationPlan.model_validate(payload)
