from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.v1_5_v5_3_recovery_qualification import (
    RecoveryQualificationCase,
    RecoveryQualificationPlan,
    build_recovery_qualification_plan,
    recovery_case_id,
)


ROOT = Path(__file__).resolve().parents[1]


def _case() -> RecoveryQualificationCase:
    program_sha = "1" * 64
    return RecoveryQualificationCase(
        case_id=recovery_case_id(
            state_id="consumed_state_01",
            requested_action_id="MS+RS",
            seed_label="seed_01",
            typed_program_sha256=program_sha,
        ),
        state_id="consumed_state_01",
        user_id="consumed_user_01",
        semantic_family="historical_step2_development",
        requested_action_id="MS+RS",
        seed_label="seed_01",
        typed_program_sha256=program_sha,
        first_pass_messages_sha256="2" * 64,
        evidence_ids=["memory_session_01", "strategy_card_01"],
    )


def test_recovery_method_is_zero_api_and_does_not_consume_formal_outcomes() -> None:
    plan = build_recovery_qualification_plan(ROOT)
    assert plan.cases == []
    assert plan.api_calls == 0
    assert plan.generated_response_or_quality_risk_outcome_read is False
    assert plan.default_if_inconclusive == "deterministic_fallback"
    assert plan.bounded_rewrite_maximum_extra_calls == 1
    assert plan.decision_rule["outcomes_may_change_step2_prompt"] is False


def test_real_cases_must_be_consumed_development_and_excluded_from_formal_panels() -> None:
    plan = build_recovery_qualification_plan(ROOT, cases=[_case()])
    assert len(plan.cases) == 1
    assert plan.cases[0].source_is_historically_consumed_development is True
    assert plan.cases[0].excluded_from_p3_p4_p5 is True


def test_json_roundtrip_and_identity_detect_drift() -> None:
    plan = build_recovery_qualification_plan(ROOT, cases=[_case()])
    assert RecoveryQualificationPlan.model_validate_json(plan.model_dump_json()) == plan
    changed = plan.model_dump(mode="json")
    changed["bounded_rewrite_maximum_extra_calls"] = 2
    with pytest.raises(Exception):
        RecoveryQualificationPlan.model_validate(changed)
