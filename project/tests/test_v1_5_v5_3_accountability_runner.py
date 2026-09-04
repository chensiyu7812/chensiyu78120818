from __future__ import annotations

import pytest

from metacom_pm.v1_5_v5_3_accountability import (
    FailureOwner,
    LifecycleStage,
    QualityOutcome,
    RiskOutcome,
    StagewiseAccountabilityRow,
    TernaryAssessment,
    accountability_row_id,
)
from metacom_pm.v1_5_v5_3_accountability_runner import (
    StagewiseAccountabilityLedger,
    post_row_from_execution,
    pre_generation_row,
)
from metacom_pm.v1_5_v5_3_typed_response_program import (
    GeneratorResponse,
    RewritePolicy,
    TypedResponseExecutionResult,
)


def _all(value):
    return {component: value for component in ("MP", "MS", "ME", "RS")}


def _pre() -> StagewiseAccountabilityRow:
    return StagewiseAccountabilityRow.model_validate(
        {
            "row_id": accountability_row_id(
                state_id="s1", policy_condition="learned_pm_full", seed_label="seed1"
            ),
            "state_id": "s1",
            "user_id": "u1",
            "semantic_family": "f1",
            "counterfactual_group_id": "g1",
            "policy_condition": "learned_pm_full",
            "seed_label": "seed1",
            "lifecycle_stage": LifecycleStage.PRE_GENERATION,
            "candidate_ids_topk": {"MP": [], "MS": ["memory_1"], "ME": [], "RS": []},
            "exact_rank1_id": {"MP": None, "MS": "memory_1", "ME": None, "RS": None},
            "candidate_owner_id": {"MP": None, "MS": "u1", "ME": None, "RS": None},
            "candidate_version": {"MP": None, "MS": "v1", "ME": None, "RS": None},
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
    )


def _execution() -> TypedResponseExecutionResult:
    response = GeneratorResponse(
        reply="You mentioned the shift change; does that still fit?",
        used_evidence_ids=("memory_1",),
        reported_used_evidence_ids=("memory_1",),
        realized_response_act="continuity_check",
    )
    return TypedResponseExecutionResult(
        response=response,
        status="clean",
        rewrite_policy=RewritePolicy.DETERMINISTIC_FALLBACK,
        calls_made=1,
        rewrite_attempted=False,
        first_pass_errors=(),
        final_guard_errors=(),
        requested_action_id="MS+R0",
        realized_action_id="MS+R0",
    )


def test_pre_generation_constructor_populates_outcome_blind_defaults() -> None:
    row = pre_generation_row(
        state_id="s1",
        user_id="u1",
        semantic_family="f1",
        counterfactual_group_id="g1",
        policy_condition="always_off",
        seed_label="seed1",
        candidate_ids_topk=_all([]),
        exact_rank1_id=_all(None),
        candidate_owner_id=_all(None),
        candidate_version=_all(None),
        pm_probability=_all(None),
        pm_threshold=_all(None),
        pm_decision=_all(False),
        requested_action="M0+R0",
    )
    assert row.lifecycle_stage is LifecycleStage.PRE_GENERATION
    assert row.quality_outcome is QualityOutcome.UNKNOWN
    assert row.generator_received_evidence_ids == []


def test_stagewise_ledger_requires_order_and_preserves_execution(tmp_path) -> None:
    ledger = StagewiseAccountabilityLedger(tmp_path)
    pre = _pre()
    post = post_row_from_execution(
        pre,
        _execution(),
        generator_received_evidence_ids=["memory_1"],
        prompt_tokens=20,
        total_tokens=30,
    )
    with pytest.raises(ValueError, match="no PRE_GENERATION parent"):
        ledger.append_post(post)

    ledger.append_pre(pre)
    ledger.append_post(post)
    review_data = post.model_dump(mode="python")
    review_data.update(
        {
            "lifecycle_stage": LifecycleStage.POST_SEMANTIC_REVIEW,
            "retrieval_fit_label": _all(TernaryAssessment.YES),
            "pm_correct": _all(TernaryAssessment.YES),
            "required_evidence_use": _all(TernaryAssessment.YES),
            "functional_contribution": _all(TernaryAssessment.YES),
            "grounding_fidelity": _all(TernaryAssessment.YES),
            "atomic_move_compliance": TernaryAssessment.YES,
            "execution_valid": TernaryAssessment.YES,
            "scaffold_exposure": TernaryAssessment.NO,
            "quality_outcome": QualityOutcome.POSITIVE,
            "risk_outcome": RiskOutcome.NO_MATERIAL_RISK,
            "primary_failure_owner": FailureOwner.NONE,
            "secondary_failure_owner": FailureOwner.NONE,
        }
    )
    review = StagewiseAccountabilityRow.model_validate(review_data)
    ledger.append_review(review)
    assert ledger.validate(
        expected_keys={("s1", "learned_pm_full", "seed1")},
        require_post=True,
        require_review=True,
    ) == {"pre": 1, "post": 1, "review": 1}


def test_review_cannot_rewrite_the_requested_or_realized_action(tmp_path) -> None:
    ledger = StagewiseAccountabilityLedger(tmp_path)
    pre = _pre()
    post = post_row_from_execution(
        pre,
        _execution(),
        generator_received_evidence_ids=["memory_1"],
        prompt_tokens=20,
        total_tokens=30,
    )
    ledger.append_pre(pre)
    ledger.append_post(post)
    data = post.model_dump(mode="python")
    data.update(
        {
            "lifecycle_stage": LifecycleStage.POST_SEMANTIC_REVIEW,
            "realized_action": "M0+R0",
            "quality_outcome": QualityOutcome.TIE,
            "risk_outcome": RiskOutcome.NO_MATERIAL_RISK,
            "primary_failure_owner": FailureOwner.NONE,
            "secondary_failure_owner": FailureOwner.NONE,
        }
    )
    review = StagewiseAccountabilityRow.model_validate(data)
    with pytest.raises(ValueError, match="immutable fields"):
        ledger.append_review(review)
