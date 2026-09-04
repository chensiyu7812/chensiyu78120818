from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from metacom_pm.v1_5_v5_3_accountability import (
    FailureOwner,
    LifecycleStage,
    QualityOutcome,
    RiskOutcome,
    StagewiseAccountabilityRow,
    TernaryAssessment,
    accountability_row_id,
    validate_accountability_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _all(value):
    return {component: value for component in ("MP", "MS", "ME", "RS")}


def _pre_row(**updates) -> StagewiseAccountabilityRow:
    data = {
        "row_id": accountability_row_id(
            state_id="state_1", policy_condition="learned_pm_full", seed_label="seed_1"
        ),
        "state_id": "state_1",
        "user_id": "user_1",
        "semantic_family": "family_1",
        "counterfactual_group_id": "group_1",
        "policy_condition": "learned_pm_full",
        "seed_label": "seed_1",
        "lifecycle_stage": LifecycleStage.PRE_GENERATION,
        "candidate_ids_topk": {
            "MP": [], "MS": ["mem_111111111111"], "ME": [], "RS": []
        },
        "exact_rank1_id": {
            "MP": None, "MS": "mem_111111111111", "ME": None, "RS": None
        },
        "candidate_owner_id": {
            "MP": None, "MS": "user_1", "ME": None, "RS": None
        },
        "candidate_version": {
            "MP": None, "MS": "v1", "ME": None, "RS": None
        },
        "retrieval_fit_label": _all(TernaryAssessment.UNKNOWN),
        "eligibility_owner_time": _all(TernaryAssessment.UNKNOWN),
        "eligibility_goal_function": _all(TernaryAssessment.UNKNOWN),
        "eligibility_boundary_burden": _all(TernaryAssessment.UNKNOWN),
        "eligibility_specific_increment": _all(TernaryAssessment.UNKNOWN),
        "pm_probability": _all(None),
        "pm_threshold": _all(None),
        "pm_decision": _all(None),
        "pm_correct": _all(TernaryAssessment.UNKNOWN),
        "requested_action": "MS+R0",
        "realized_action": None,
        "generator_received_evidence_ids": [],
        "generator_reported_used_evidence_ids": [],
        "normalized_used_evidence_ids": [],
        "used_evidence_ids": [],
        "required_evidence_use": _all(TernaryAssessment.UNKNOWN),
        "functional_contribution": _all(TernaryAssessment.UNKNOWN),
        "grounding_fidelity": _all(TernaryAssessment.UNKNOWN),
        "atomic_move_compliance": TernaryAssessment.UNKNOWN,
        "execution_valid": TernaryAssessment.UNKNOWN,
        "scaffold_exposure": TernaryAssessment.UNKNOWN,
        "fallback_reason": None,
        "quality_outcome": QualityOutcome.UNKNOWN,
        "risk_outcome": RiskOutcome.UNKNOWN,
        "prompt_tokens": None,
        "total_tokens": None,
        "primary_failure_owner": FailureOwner.UNKNOWN,
        "secondary_failure_owner": FailureOwner.UNKNOWN,
    }
    data.update(updates)
    return StagewiseAccountabilityRow.model_validate(data)


def test_pre_generation_row_is_fully_typed_without_outcome() -> None:
    row = _pre_row()
    assert row.lifecycle_stage is LifecycleStage.PRE_GENERATION
    assert row.exact_rank1_id["MS"] == "mem_111111111111"


def test_row_identity_includes_policy_and_seed() -> None:
    first = _pre_row()
    second = _pre_row(
        policy_condition="always_off",
        row_id=accountability_row_id(
            state_id="state_1", policy_condition="always_off", seed_label="seed_1"
        ),
        requested_action="M0+R0",
        candidate_ids_topk=_all([]),
        exact_rank1_id=_all(None),
        candidate_owner_id=_all(None),
        candidate_version=_all(None),
    )
    validate_accountability_rows([first, second])
    with pytest.raises(ValueError, match="duplicate accountability row key"):
        validate_accountability_rows([first, first])


def test_exact_rank1_must_be_first_topk_candidate() -> None:
    with pytest.raises(ValueError, match="must equal candidate_ids_topk"):
        _pre_row(exact_rank1_id={
            "MP": None, "MS": "mem_222222222222", "ME": None, "RS": None
        })


def test_m0_post_generation_keeps_raw_trace_but_normalizes_realized_use() -> None:
    row = _pre_row(
        policy_condition="always_off",
        row_id=accountability_row_id(
            state_id="state_1", policy_condition="always_off", seed_label="seed_1"
        ),
        lifecycle_stage=LifecycleStage.POST_GENERATION,
        requested_action="M0+R0",
        realized_action="M0+R0",
        candidate_ids_topk=_all([]),
        exact_rank1_id=_all(None),
        candidate_owner_id=_all(None),
        candidate_version=_all(None),
        generator_reported_used_evidence_ids=["user_message_1"],
        prompt_tokens=100,
        total_tokens=120,
    )
    assert row.generator_reported_used_evidence_ids == ["user_message_1"]
    assert row.used_evidence_ids == []


def test_normalized_and_realized_used_ids_cannot_disagree() -> None:
    with pytest.raises(ValueError, match="must equal normalized"):
        _pre_row(
            normalized_used_evidence_ids=["mem_111111111111"],
            used_evidence_ids=[],
        )


def test_machine_contract_and_typed_schema_freeze_agree() -> None:
    path = ROOT / "scripts/v1_5/62_freeze_v5_3_stagewise_accountability_schema_v1_5.py"
    spec = importlib.util.spec_from_file_location("accountability_freeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema, report = module.build_freeze()
    assert report["missing_required_fields"] == []
    assert report["quality_risk_or_function_outcomes_read"] is False
    assert report["api_calls"] == 0
    assert "policy_condition" in schema["properties"]
