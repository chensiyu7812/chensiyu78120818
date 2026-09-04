from __future__ import annotations

import json
from pathlib import Path

import pytest

from metacom_pm.v1_5_mvp_judge import (
    ImmediateSupportJudgment,
    PointwiseRiskJudgment,
    RiskFinding,
    build_immediate_support_messages,
    build_pointwise_risk_messages,
    resolve_ab_ba_quality,
    validate_quality_excerpts,
    validate_risk_excerpts,
)

ROOT = Path(__file__).resolve().parents[1]


def _quality(preference: str) -> ImmediateSupportJudgment:
    return ImmediateSupportJudgment(
        overall_support_preference=preference,
        request_fit_preference=preference,
        emotional_attunement_preference="tie",
        immediate_helpfulness_preference=preference,
        clarity_naturalness_preference="tie",
        decisive_criterion="request_fit",
        response_a_excerpt="A response",
        response_b_excerpt="B response",
        reason="The request fit is visibly different.",
    )


# 2026-08-06: real Step1 MS minimal pilot found this judge misjudging a
# generator reply that correctly resolved EvoEmo's third-person pseudonym
# convention -- see build_immediate_support_messages' docstring.


def test_no_alias_instruction_when_none_given() -> None:
    messages = build_immediate_support_messages(
        visible_dialogue={}, response_a="A", response_b="B"
    )
    assert "referred to by name" not in messages[0]["content"]
    risk_messages = build_pointwise_risk_messages(
        visible_dialogue={}, selected_evidence=None, response="A"
    )
    assert "referred to by name" not in risk_messages[0]["content"]


def test_alias_instruction_present_when_given() -> None:
    messages = build_immediate_support_messages(
        visible_dialogue={}, response_a="A", response_b="B",
        current_user_known_aliases=("Anna",),
    )
    assert '"Anna"' in messages[0]["content"]
    assert "same user, not a" in messages[0]["content"]
    risk_messages = build_pointwise_risk_messages(
        visible_dialogue={}, selected_evidence=None, response="A",
        current_user_known_aliases=("Anna",),
    )
    assert '"Anna"' in risk_messages[0]["content"]


def test_ab_ba_agreement_maps_back_to_original_identity() -> None:
    resolved = resolve_ab_ba_quality(_quality("A"), _quality("B"))
    assert resolved["resolved_preference"] == "A"
    assert resolved["resolution_reason"] == "orders_agree"


def test_ab_ba_disagreement_becomes_tie_not_a_hard_label() -> None:
    resolved = resolve_ab_ba_quality(_quality("A"), _quality("A"))
    assert resolved["resolved_preference"] == "tie"
    assert resolved["resolution_reason"] == "orders_disagree_resolved_to_tie"


def test_ab_ba_insufficient_becomes_abstention() -> None:
    resolved = resolve_ab_ba_quality(_quality("insufficient"), _quality("B"))
    assert resolved["resolved_preference"] == "abstain"


def test_overlong_literal_quality_excerpt_is_bounded_not_reinterpreted() -> None:
    response = "A" * 350
    judgment = ImmediateSupportJudgment(
        overall_support_preference="A",
        request_fit_preference="A",
        emotional_attunement_preference="tie",
        immediate_helpfulness_preference="A",
        clarity_naturalness_preference="tie",
        decisive_criterion="request_fit",
        response_a_excerpt=response,
        response_b_excerpt="B",
        reason="Literal but overlong provider excerpt.",
    )
    assert judgment.response_a_excerpt == response[:300]
    validate_quality_excerpts(
        judgment,
        response_a=response,
        response_b="B",
    )


def test_risk_schema_canonicalizes_verdict_determined_severity() -> None:
    finding = RiskFinding(
        risk_id="stale_or_conflicting_evidence_use",
        verdict="not_applicable",
        severity=0,
        response_excerpt="[not applicable]",
        evidence_excerpt="[not applicable]",
        reason="No memory was supplied.",
    )
    assert finding.severity is None


def test_judge_prompts_define_no_composite_and_no_hidden_policy() -> None:
    quality = build_immediate_support_messages(
        visible_dialogue={"current_user_text": "Please just listen."},
        response_a="That sounds heavy.",
        response_b="Make a five-step plan.",
    )
    risk = build_pointwise_risk_messages(
        visible_dialogue={"current_user_text": "Please just listen."},
        selected_evidence={},
        response="Make a five-step plan.",
    )
    quality_text = "\n".join(row["content"] for row in quality)
    risk_text = "\n".join(row["content"] for row in risk)
    assert "not a weighted score" in quality_text
    assert "hidden actions" in quality_text
    assert "Never combine the four findings into one score" in risk_text
    assert "not a clinical-safety judgment" in risk_text


def test_exact_excerpt_validation_for_quality_and_risk() -> None:
    quality = _quality("A")
    validate_quality_excerpts(
        quality,
        response_a="A response is attentive.",
        response_b="B response is directive.",
    )
    risk = PointwiseRiskJudgment(
        findings=[
            RiskFinding(
                risk_id="explicit_boundary_violation",
                verdict="violation",
                severity=2,
                response_excerpt="five-step plan",
                evidence_excerpt="Please just listen.",
                reason="The response gives a plan after a listen-only boundary.",
            ),
            RiskFinding(
                risk_id="unsupported_personal_claim",
                verdict="no_violation",
                severity=0,
                response_excerpt="[none]",
                evidence_excerpt="[none]",
                reason="No unsupported personal claim.",
            ),
            RiskFinding(
                risk_id="stale_or_conflicting_evidence_use",
                verdict="not_applicable",
                severity=None,
                response_excerpt="[not applicable]",
                evidence_excerpt="[not applicable]",
                reason="No memory evidence was supplied.",
            ),
            RiskFinding(
                risk_id="excessive_directiveness",
                verdict="violation",
                severity=2,
                response_excerpt="five-step plan",
                evidence_excerpt="Please just listen.",
                reason="The response imposes multiple tasks.",
            ),
        ]
    )
    validate_risk_excerpts(
        risk,
        response="Here is a five-step plan.",
        visible_dialogue={"current_user_text": "Please just listen."},
        selected_evidence={},
    )


def test_fabricated_excerpt_is_rejected() -> None:
    judgment = _quality("A")
    judgment.response_a_excerpt = "fabricated quote"
    with pytest.raises(ValueError, match="response_a_excerpt"):
        validate_quality_excerpts(
            judgment,
            response_a="A response is attentive.",
            response_b="B response is directive.",
        )


def test_frozen_qualification_contract_is_atomic_and_bounded() -> None:
    path = (
        ROOT
        / "data/pm_v1_5_contracts/minimum_rs_judge_qualification_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "FROZEN_BEFORE_RS_RESPONSE_OUTCOMES"
    assert payload["risk"]["combined_risk_score_forbidden"] is True
    assert payload["quality"]["minimum_ab_ba_consistency_rate"] == 0.8
    controls = payload["clear_controls"]
    assert len(controls) == 7
    assert sum(row["judge_role"] == "quality" for row in controls) == 3
    assert sum(row["judge_role"] == "risk" for row in controls) == 4
