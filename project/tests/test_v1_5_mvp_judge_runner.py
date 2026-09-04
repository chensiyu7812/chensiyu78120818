from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from metacom_pm.io import read_json
from metacom_pm.v1_5_mvp_judge import (
    ImmediateSupportJudgment,
    PointwiseRiskJudgment,
    RiskFinding,
)
from metacom_pm.v1_5_mvp_judge_runner import (
    aggregate_outcome_measurement,
    build_control_call_plan,
    build_outcome_call_plan,
    deterministic_risk_applicability,
    qualify_controls,
    validate_parsed_judgment,
    validate_rs_judge_inputs,
)
from metacom_pm.v1_5_mvp_human_audit import (
    build_human_audit_packet,
    select_human_audit_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = {
    "name": "fixture_judge",
    "base_url": "https://judge.invalid",
    "model": "fixture-model",
    "family": "fixture-family",
    "transport": "openai_chat_completions",
}


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _real_inputs() -> dict:
    plan_dir = ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1"
    generation_dir = (
        ROOT
        / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1_execution"
    )
    report = read_json(plan_dir / "plan_report.json")
    return validate_rs_judge_inputs(
        plan_report=report,
        generation_summary=read_json(
            generation_dir / "generation_summary.json"
        ),
        selected_states=_rows(plan_dir / "selected_states.jsonl"),
        generation_plan=_rows(plan_dir / "call_plan.jsonl"),
        generation_outcomes=_rows(
            generation_dir / "generation_outcomes.jsonl"
        ),
        runtime_states=_rows(Path(report["runtime_states_path"])),
    )


def _quality(preference: str, response_a: str, response_b: str) -> dict:
    return ImmediateSupportJudgment(
        overall_support_preference=preference,
        request_fit_preference=preference,
        emotional_attunement_preference="tie",
        immediate_helpfulness_preference=preference,
        clarity_naturalness_preference="tie",
        decisive_criterion=(
            "materially_equivalent"
            if preference == "tie"
            else "request_fit"
        ),
        response_a_excerpt=response_a[: min(40, len(response_a))],
        response_b_excerpt=response_b[: min(40, len(response_b))],
        reason="The visible request fit determines the preference.",
    ).model_dump(mode="json")


def _risk_no_event() -> dict:
    return PointwiseRiskJudgment(
        findings=[
            RiskFinding(
                risk_id="explicit_boundary_violation",
                verdict="not_applicable",
                severity=None,
                response_excerpt="[not applicable]",
                evidence_excerpt="[not applicable]",
                reason="No explicit boundary is applicable.",
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
                verdict="no_violation",
                severity=0,
                response_excerpt="[none]",
                evidence_excerpt="[none]",
                reason="The response does not impose tasks.",
            ),
        ]
    ).model_dump(mode="json")


def test_verdict_determined_risk_severity_is_canonicalized() -> None:
    no_event = RiskFinding.model_validate(
        {
            "risk_id": "unsupported_personal_claim",
            "verdict": "no_violation",
            "severity": None,
            "response_excerpt": "[none]",
            "evidence_excerpt": "[none]",
            "reason": "No unsupported claim.",
        }
    )
    assert no_event.severity == 0
    minor_but_no_event = no_event.model_copy(update={"severity": 1})
    recanonicalized = RiskFinding.model_validate(
        minor_but_no_event.model_dump(mode="json")
    )
    assert recanonicalized.severity == 0
    not_applicable = RiskFinding.model_validate(
        {
            "risk_id": "stale_or_conflicting_evidence_use",
            "verdict": "not_applicable",
            "severity": 0,
            "response_excerpt": "[not applicable]",
            "evidence_excerpt": "[not applicable]",
            "reason": "No memory was supplied.",
        }
    )
    assert not_applicable.severity is None


def test_only_non_event_nonliteral_excerpt_is_replaced_by_sentinel() -> None:
    call = {
        "role": "risk",
        "response": "One exact response.",
        "visible_dialogue": {"current_user_text": "Please give one idea."},
        "selected_evidence": {},
    }
    parsed = _risk_no_event()
    parsed["findings"][0]["verdict"] = "no_violation"
    parsed["findings"][0]["severity"] = 0
    parsed["findings"][0]["response_excerpt"] = "Please give one idea."
    normalized = validate_parsed_judgment(call=call, parsed=parsed)
    assert normalized["findings"][0]["response_excerpt"] == "[none]"

    parsed["findings"][0].update(
        {
            "verdict": "violation",
            "severity": 2,
            "response_excerpt": "not in the response",
            "evidence_excerpt": "Please give one idea.",
        }
    )
    with pytest.raises(ValueError, match="response_excerpt is not exact"):
        validate_parsed_judgment(call=call, parsed=parsed)


def test_real_rs_inputs_are_complete_unique_and_same_stack() -> None:
    validated = _real_inputs()
    assert validated["audit"]["status"] == "PASS"
    assert validated["audit"]["pair_count"] == 49
    assert validated["audit"]["response_count"] == 98
    assert validated["audit"]["independent_user_groups"] == 24
    assert validated["audit"]["risk_applicable_pair_counts"] == {
        "explicit_boundary_violation": 46,
        "unsupported_personal_claim": 49,
        "stale_or_conflicting_evidence_use": 0,
        "excessive_directiveness": 49,
    }
    assert all(validated["audit"]["checks"].values())


def test_duplicate_generation_outcome_is_rejected() -> None:
    plan_dir = ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1"
    generation_dir = (
        ROOT
        / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1_execution"
    )
    report = read_json(plan_dir / "plan_report.json")
    outcomes = _rows(generation_dir / "generation_outcomes.jsonl")
    with pytest.raises(ValueError, match="98 rows"):
        validate_rs_judge_inputs(
            plan_report=report,
            generation_summary=read_json(
                generation_dir / "generation_summary.json"
            ),
            selected_states=_rows(plan_dir / "selected_states.jsonl"),
            generation_plan=_rows(plan_dir / "call_plan.jsonl"),
            generation_outcomes=outcomes + [deepcopy(outcomes[0])],
            runtime_states=_rows(Path(report["runtime_states_path"])),
        )


def test_control_and_outcome_plans_have_expected_atomic_calls() -> None:
    contract = read_json(
        ROOT
        / "data/pm_v1_5_contracts/minimum_rs_judge_qualification_v1.json"
    )
    controls = build_control_call_plan(
        qualification_contract=contract,
        endpoint=ENDPOINT,
    )
    assert len(controls) == 10
    assert sum(row["role"] == "quality" for row in controls) == 6
    assert sum(row["role"] == "risk" for row in controls) == 4
    assert all("arm" not in json.dumps(row["messages"]) for row in controls)

    outcomes = build_outcome_call_plan(
        pairs=_real_inputs()["pairs"],
        endpoint=ENDPOINT,
    )
    assert len(outcomes) == 196
    assert sum(row["role"] == "quality" for row in outcomes) == 98
    assert sum(row["role"] == "risk" for row in outcomes) == 98
    assert len({row["call_id"] for row in outcomes}) == 196


def test_control_qualification_uses_frozen_material_rule() -> None:
    contract = read_json(
        ROOT
        / "data/pm_v1_5_contracts/minimum_rs_judge_qualification_v1.json"
    )
    plan = build_control_call_plan(
        qualification_contract=contract,
        endpoint=ENDPOINT,
    )
    controls = {
        row["control_id"]: row for row in contract["clear_controls"]
    }
    results: list[dict] = []
    for call in plan:
        control = controls[call["record_ids"]["control_id"]]
        if call["role"] == "quality":
            expected = control["expected_overall_support_preference"]
            displayed = expected
            if call["order_variant"] == 1 and expected in {"A", "B"}:
                displayed = "B" if expected == "A" else "A"
            parsed = _quality(
                displayed, call["response_a"], call["response_b"]
            )
        else:
            findings = []
            for risk_id, expected in control["expected"].items():
                severity = expected.get(
                    "severity", expected.get("minimum_severity")
                )
                findings.append(
                    RiskFinding(
                        risk_id=risk_id,
                        verdict=expected["verdict"],
                        severity=severity,
                        response_excerpt=(
                            call["response"][: min(30, len(call["response"]))]
                            if expected["verdict"] == "violation"
                            else (
                                "[not applicable]"
                                if expected["verdict"] == "not_applicable"
                                else "[none]"
                            )
                        ),
                        evidence_excerpt=(
                            "[none]"
                            if expected["verdict"] == "violation"
                            else (
                                "[not applicable]"
                                if expected["verdict"] == "not_applicable"
                                else "[none]"
                            )
                        ),
                        reason="Fixture matches the frozen expected event.",
                    )
                )
            parsed = PointwiseRiskJudgment(
                findings=findings
            ).model_dump(mode="json")
        results.append({"call_id": call["call_id"], "parsed": parsed})
    report = qualify_controls(
        qualification_contract=contract,
        call_plan=plan,
        results=results,
    )
    assert report["qualified"] is True
    assert report["quality_controls_correct"] == 3
    assert report["risk_controls_correct"] == 4
    assert report["risk_detected_expected_material_events"] == 5
    assert report["risk_exact_dimensions_correct"] == 16

    nonmaterial_mismatch = deepcopy(results)
    target = next(
        row
        for row in nonmaterial_mismatch
        if row["call_id"]
        == "control::risk_unsupported_personal_claim::risk"
    )
    finding = next(
        row
        for row in target["parsed"]["findings"]
        if row["risk_id"] == "explicit_boundary_violation"
    )
    finding.update(
        {
            "verdict": "no_violation",
            "severity": 0,
            "response_excerpt": "[none]",
            "evidence_excerpt": "[none]",
        }
    )
    diagnostic_only = qualify_controls(
        qualification_contract=contract,
        call_plan=plan,
        results=nonmaterial_mismatch,
    )
    assert diagnostic_only["qualified"] is True
    assert diagnostic_only["risk_exact_dimensions_correct"] == 15

    missing_material = deepcopy(results)
    target = next(
        row
        for row in missing_material
        if row["call_id"]
        == "control::risk_unsupported_personal_claim::risk"
    )
    finding = next(
        row
        for row in target["parsed"]["findings"]
        if row["risk_id"] == "unsupported_personal_claim"
    )
    finding.update(
        {
            "verdict": "no_violation",
            "severity": 0,
            "response_excerpt": "[none]",
            "evidence_excerpt": "[none]",
        }
    )
    failed = qualify_controls(
        qualification_contract=contract,
        call_plan=plan,
        results=missing_material,
    )
    assert failed["qualified"] is False
    assert failed["status"] == "FAIL_DO_NOT_JUDGE_RS_OUTCOMES"
    assert failed["risk_missing_expected_material_events"] == 1

    invented = deepcopy(results)
    target = next(
        row
        for row in invented
        if row["call_id"] == "control::risk_no_event::risk"
    )
    finding = next(
        row
        for row in target["parsed"]["findings"]
        if row["risk_id"] == "unsupported_personal_claim"
    )
    call = next(
        row for row in plan if row["call_id"] == target["call_id"]
    )
    finding.update(
        {
            "verdict": "violation",
            "severity": 2,
            "response_excerpt": call["response"][:20],
            "evidence_excerpt": "[none]",
        }
    )
    invented_report = qualify_controls(
        qualification_contract=contract,
        call_plan=plan,
        results=invented,
    )
    assert invented_report["qualified"] is False
    assert invented_report["risk_no_event_material_events"] == 1


def test_measurement_keeps_quality_risk_and_cost_separate() -> None:
    pairs = _real_inputs()["pairs"][:2]
    plan = build_outcome_call_plan(pairs=pairs, endpoint=ENDPOINT)
    results: list[dict] = []
    for call in plan:
        if call["role"] == "quality":
            base_a_arm = call["base_identity"]["candidate_a_arm"]
            desired_base = "A" if base_a_arm == "RS" else "B"
            displayed = desired_base
            if call["order_variant"] == 1:
                displayed = "B" if desired_base == "A" else "A"
            parsed = _quality(
                displayed, call["response_a"], call["response_b"]
            )
        else:
            parsed = _risk_no_event()
        results.append({"call_id": call["call_id"], "parsed": parsed})
    report = aggregate_outcome_measurement(
        qualification_report={"qualified": True},
        call_plan=plan,
        results=results,
        pairs=pairs,
    )
    assert report["quality"]["RS_wins"] == 2
    assert report["quality"]["raw_netwin_rs_vs_r0"] == 1.0
    assert report["no_quality_risk_cost_composite"] is True
    assert set(report["risk"]) == {
        "explicit_boundary_violation",
        "unsupported_personal_claim",
        "stale_or_conflicting_evidence_use",
        "excessive_directiveness",
    }
    assert (
        report["cost"]["RS"]["generator_input_tokens_mean"]
        > report["cost"]["R0"]["generator_input_tokens_mean"]
    )
    assert (
        report["cost"]["RS"]["generator_total_tokens_mean"]
        > report["cost"]["R0"]["generator_total_tokens_mean"]
    )
    assert (
        report["risk"]["stale_or_conflicting_evidence_use"]["by_arm"]["R0"][
            "eligible_responses"
        ]
        == 0
    )


def test_programmatic_risk_applicability_distinguishes_request_from_limit() -> None:
    generic = deterministic_risk_applicability(
        visible_dialogue={"current_user_text": "Any tips?"},
        selected_evidence={},
    )
    assert generic["explicit_boundary_violation"]["applicable"] is False
    assert (
        generic["stale_or_conflicting_evidence_use"]["applicable"] is False
    )
    constrained = deterministic_risk_applicability(
        visible_dialogue={
            "current_user_text": (
                "Could you offer one gentle suggestion, without turning it "
                "into a full plan?"
            )
        },
        selected_evidence={"memory": "The user previously liked running."},
    )
    assert constrained["explicit_boundary_violation"]["applicable"] is True
    assert (
        constrained["stale_or_conflicting_evidence_use"]["applicable"] is True
    )


def test_human_audit_selection_is_outcome_blind_and_balanced() -> None:
    pairs = _real_inputs()["pairs"]
    selection = select_human_audit_pairs(pairs)
    changed = deepcopy(pairs)
    for pair in changed:
        pair["arms"]["R0"]["response"] = "changed"
        pair["arms"]["RS"]["response"] = "also changed"
    assert selection == select_human_audit_pairs(changed)
    assert len(selection) == 12
    assert len({row["user_id"] for row in selection}) == 12
    assert sum(row["boundary_cue"] == "listen_only" for row in selection) == 6
    assert sum(
        row["boundary_cue"] == "advice_welcome" for row in selection
    ) == 6

    built = build_human_audit_packet(pairs)
    assert len(built["packet"]) == 12
    assert len(built["annotation_template"]) == 12
    assert sum(
        row["response_a_arm"] == "RS" for row in built["private_key"]
    ) == 6
    assert all(
        item["risk_applicability"][
            "stale_or_conflicting_evidence_use"
        ]["applicable"]
        is False
        for item in built["packet"]
    )
