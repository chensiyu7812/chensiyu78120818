from __future__ import annotations

import pytest
from pydantic import ValidationError

from metacom_pm.v1_5_generator_alignment_audit import (
    GeneratorResourceUseJudgment,
    ResourceBundleApplicationOutput,
    ResourceApplicationOutput,
    ResourceExecutionOutput,
    build_resource_execution_plan,
    generator_resource_use_messages,
    materialize_strategy_card_for_execution,
    qualify_step2_resource_surface,
    resource_application_messages_v1_5b,
    resource_bundle_application_messages_v1_5b,
    resource_bundle_output_schema_for_components,
    resource_execution_messages_v3,
    validate_resource_application,
    validate_resource_bundle_application,
    validate_resource_execution,
    validate_literal_excerpts,
)


def test_step2_surface_gate_blocks_current_echo_incomplete_ms_and_outcomeless_me() -> None:
    echo = qualify_step2_resource_surface(
        component="MP",
        resource_subtype="MP_PROFILE",
        selected_resource=(
            "Stable Shift Sleep Constraint: can address the shift sleep issue only "
            "during one short private window in the early evening"
        ),
        current_user_text=(
            "The shift sleep issue is back. I only have a short private window "
            "in the evening."
        ),
    )
    assert "CURRENT_ECHO_HIGH_CONTENT_OVERLAP" in echo["errors"]

    incomplete_ms = qualify_step2_resource_surface(
        component="MS",
        resource_subtype="MS_SESSION",
        selected_resource="The earlier session left one exact wording question unresolved.",
        current_user_text="I want to resume the earlier thread.",
    )
    assert "MS_META_SUMMARY_WITHOUT_CONCRETE_PAYLOAD" in incomplete_ms["errors"]

    outcomeless_me = qualify_step2_resource_surface(
        component="ME",
        resource_subtype="ME_REUSABLE_OUTCOME",
        selected_resource="A prior project deadline remained unresolved.",
        current_user_text="I am open to one optional idea for the deadline.",
    )
    assert "ME_OUTCOME_OR_MECHANISM_MISSING" in outcomeless_me["errors"]


def test_step2_surface_gate_accepts_concrete_ms_and_action_outcome_me() -> None:
    ms = qualify_step2_resource_surface(
        component="MS",
        resource_subtype="MS_SESSION",
        selected_resource=(
            "The earlier shift sleep session recorded that Tuesday evening was "
            "the one workable time."
        ),
        current_user_text="Please remind me of the specific observation recorded earlier.",
    )
    assert ms["qualified"], ms["errors"]

    me = qualify_step2_resource_surface(
        component="ME",
        resource_subtype="ME_REUSABLE_OUTCOME",
        selected_resource=(
            "For the project deadline issue, I wrote one short note, and it helped "
            "make the next step easier."
        ),
        current_user_text="The deadline is back and I am open to one optional idea.",
    )
    assert me["qualified"], me["errors"]


def test_strategy_execution_surface_requires_all_three_frozen_fields() -> None:
    card = {
        "card_id": "card_1",
        "support_move": "Ask one focused question.",
        "when_to_use": "Use when one missing detail matters.",
        "when_not_to_use": "Do not use after a stop boundary.",
    }
    surface = materialize_strategy_card_for_execution(card)
    assert "support_move: Ask one focused question." in surface
    with pytest.raises(ValueError, match="when_not_to_use"):
        materialize_strategy_card_for_execution({**card, "when_not_to_use": ""})


def test_functional_use_requires_auditable_excerpts() -> None:
    row = GeneratorResourceUseJudgment.model_validate(
        {
            "resource_use_label": "grounded_functional_use",
            "resource_function": "event_or_outcome",
            "resource_used": "yes",
            "response_evidence_excerpt": "earlier in the day",
            "resource_evidence_excerpt": "earlier hour",
            "misuse_categories": [],
            "reason": "The prior outcome changes the suggestion.",
            "confidence": 4,
        }
    )
    assert validate_literal_excerpts(
        judgment=row,
        response="Try the hardest reading earlier in the day.",
        resource="Moving it to an earlier hour helped before.",
    )


def test_nonuse_has_no_evidence_excerpts() -> None:
    row = GeneratorResourceUseJudgment.model_validate(
        {
            "resource_use_label": "missed_usable_resource",
            "resource_function": "event_or_outcome",
            "resource_used": "no",
            "response_evidence_excerpt": "Tell me more.",
            "resource_evidence_excerpt": "useful earlier event",
            "misuse_categories": [],
            "reason": "The reply ignores the relevant event.",
            "confidence": 3,
        }
    )
    assert validate_literal_excerpts(
        judgment=row, response="Tell me more.", resource="A useful earlier event."
    )


def test_judge_message_hides_prompt_variant() -> None:
    messages = generator_resource_use_messages(
        component="MP",
        resource_subtype="MP_PREFERENCE",
        visible_context="Current user message: Please give one idea.",
        selected_resource="[MP] Ask permission before offering suggestions.",
        response="Would you like one small idea?",
    )
    surface = str(messages)
    assert "prompt_variant" not in surface
    assert "legacy_generic" not in surface
    assert "source_matched" not in surface


def test_resource_execution_is_machine_auditable() -> None:
    output = ResourceExecutionOutput.model_validate(
        {
            "resource_decision": "use",
            "resource_function": "style_or_pacing",
            "used_resource_labels": ["E1"],
            "resource_evidence_excerpt": "ask permission",
            "response_evidence_excerpt": "Would you like one small idea?",
            "concise_decision_reason": "The preference asks for permission first.",
            "response": "Would you like one small idea?",
        }
    )
    checks = validate_resource_execution(
        output=output,
        resource_label="E1",
        expected_function="style_or_pacing",
        selected_resource="Support preference: ask permission before suggestions.",
    )
    assert checks["all_machine_checks_pass"]


def test_repaired_execution_prompt_forbids_temporal_promotion_and_fake_ignore_reason() -> None:
    prompt = resource_execution_messages_v3(
        component="ME",
        resource_subtype="ME_EVENT_OUTCOME",
        visible_context="user: Deadlines feel tangled again.",
        selected_resource="[ME] An external dependency mattered once.",
        resource_label="E1",
        system_prompt="Be supportive.",
    )[1]["content"]
    assert "one past example" in prompt
    assert "Never convert past evidence" in prompt
    assert "Do not invent a request" in prompt
    assert "vague wording" in prompt


def test_v1_5b_execution_plan_does_not_reopen_step1_or_require_span_copy() -> None:
    plan = build_resource_execution_plan(
        component="MS", resource_subtype="MS_SESSION"
    )
    prompt = resource_application_messages_v1_5b(
        plan=plan,
        visible_context="user: The deadline pressure is back.",
        selected_resource="Earlier, waiting for another team was the main bottleneck.",
        resource_label="E1",
        system_prompt="Be supportive.",
    )[1]["content"]
    assert "Step1 has already authorized" in prompt
    assert "Do not silently make a second free use/ignore decision" in prompt
    assert "exact redundancy" in prompt
    assert "resource_evidence_excerpt" not in prompt
    assert "response_evidence_excerpt" not in prompt
    assert plan.attribution_required is True
    assert "Last time, you mentioned..." in prompt


def test_v1_5b_application_requires_past_attribution_and_realizes_fallback() -> None:
    plan = build_resource_execution_plan(
        component="ME", resource_subtype="ME_EVENT_OUTCOME"
    )
    applied = ResourceApplicationOutput.model_validate(
        {
            "application_status": "applied",
            "applied_function": "event_or_outcome",
            "cannot_apply_reason": "none",
            "response": (
                "Something that helped before was choosing one small task; "
                "would that fit today?"
            ),
        }
    )
    checks = validate_resource_application(
        output=applied, plan=plan, resource_label="E1"
    )
    assert checks["all_machine_checks_pass"]
    assert checks["realized_component_on"]
    assert not checks["fallback_required"]

    cannot = ResourceApplicationOutput.model_validate(
        {
            "application_status": "cannot_apply",
            "applied_function": "none",
            "cannot_apply_reason": "new_visible_conflict",
            "response": "That earlier detail no longer fits what you said today.",
        }
    )
    fallback = validate_resource_application(
        output=cannot, plan=plan, resource_label="E1"
    )
    assert fallback["all_machine_checks_pass"]
    assert fallback["fallback_required"]
    assert not fallback["realized_component_on"]


def test_v1_5b_past_attribution_rejects_ordinary_before_and_accepts_provenance() -> None:
    plan = build_resource_execution_plan(
        component="ME", resource_subtype="ME_EVENT_OUTCOME"
    )
    ordinary_before = ResourceApplicationOutput.model_validate(
        {
            "application_status": "applied",
            "applied_function": "event_or_outcome",
            "cannot_apply_reason": "none",
            "response": "Try taking one quiet minute before the meeting.",
        }
    )
    failed = validate_resource_application(
        output=ordinary_before, plan=plan, resource_label="E1"
    )
    assert not failed["required_past_attribution_present"]
    assert failed["fallback_required"]

    explicit_past = ResourceApplicationOutput.model_validate(
        {
            "application_status": "applied",
            "applied_function": "event_or_outcome",
            "cannot_apply_reason": "none",
            "response": (
                "In the past, a short pause gave you room to choose; "
                "would that fit today?"
            ),
        }
    )
    passed = validate_resource_application(
        output=explicit_past,
        plan=plan,
        resource_label="E1",
        selected_resource="A short pause gave the user room to choose.",
    )
    assert passed["required_past_attribution_present"]
    assert not passed["fallback_required"]


def test_v1_5b_machine_guard_rejects_fabricated_recall() -> None:
    plan = build_resource_execution_plan(
        component="MS", resource_subtype="MS_SESSION"
    )
    output = ResourceApplicationOutput.model_validate(
        {
            "application_status": "applied",
            "applied_function": "session_continuity",
            "cannot_apply_reason": "none",
            "response": (
                "Last time, you found it helpful to take short breaks. "
                "Would that fit today?"
            ),
        }
    )
    checks = validate_resource_application(
        output=output,
        plan=plan,
        resource_label="E1",
        selected_resource="The prior session discussed a late work shift.",
        current_user_text="My concentration is difficult today.",
    )
    assert not checks["fabricated_recall_absent"]
    assert checks["fallback_required"]
    assert not checks["realized_component_on"]


def test_v1_5b_machine_guard_rejects_known_panic_conflict_and_social_avoidance() -> None:
    rs_plan = build_resource_execution_plan(
        component="RS", resource_subtype="Providing Suggestions"
    )
    panic = ResourceApplicationOutput.model_validate(
        {
            "application_status": "applied",
            "applied_function": "support_technique",
            "cannot_apply_reason": "none",
            "response": "Try writing down one thing that has been on your mind.",
        }
    )
    panic_checks = validate_resource_application(
        output=panic,
        plan=rs_plan,
        resource_label="R1",
        current_user_text="When I try writing it down, I send myself into a panic.",
    )
    assert not panic_checks["known_current_conflict_respected"]
    assert panic_checks["fallback_required"]

    avoidance = ResourceApplicationOutput.model_validate(
        {
            "application_status": "applied",
            "applied_function": "support_technique",
            "cannot_apply_reason": "none",
            "response": "For a few days, try to limit interactions with people who remind you of her.",
        }
    )
    avoidance_checks = validate_resource_application(
        output=avoidance,
        plan=rs_plan,
        resource_label="R1",
    )
    assert not avoidance_checks["unsafe_social_avoidance_absent"]
    assert avoidance_checks["fallback_required"]


def test_v1_5b_bundle_preserves_passing_components_and_bounded_nonuse() -> None:
    plans = {
        "MP": build_resource_execution_plan(
            component="MP", resource_subtype="MP_PREFERENCE"
        ),
        "ME": build_resource_execution_plan(
            component="ME", resource_subtype="ME_EVENT_OUTCOME"
        ),
    }
    messages = resource_bundle_application_messages_v1_5b(
        plans=plans,
        visible_context="user: I would like one small idea.",
        selected_resources={
            "MP": "The user prefers one optional idea at a time.",
            "ME": "Last month a two-minute pause reduced the immediate pressure.",
        },
        resource_labels={"MP": "MP1", "ME": "ME1"},
        system_prompt="Be supportive.",
    )
    assert "every and only the requested components" in messages[1]["content"]
    assert "In the past, ..." in messages[1]["content"]
    output = ResourceBundleApplicationOutput.model_validate(
        {
            "component_statuses": [
                {
                    "component": "MP",
                    "application_status": "applied",
                    "applied_function": "style_or_pacing",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "prefers one optional idea",
                    "response_evidence_excerpt": "Would one small optional pause feel manageable right now?",
                },
                {
                    "component": "ME",
                    "application_status": "cannot_apply",
                    "applied_function": "none",
                    "cannot_apply_reason": "new_visible_redundancy",
                    "resource_support_excerpt": "[none]",
                    "response_evidence_excerpt": "[none]",
                },
            ],
            "response": "Would one small optional pause feel manageable right now?",
        }
    )
    checks = validate_resource_bundle_application(
        output=output,
        plans=plans,
        resource_labels={"MP": "MP1", "ME": "ME1"},
        selected_resources={
            "MP": "The user prefers one optional idea at a time.",
            "ME": "Last month a two-minute pause reduced the immediate pressure.",
        },
        current_user_text="I would like one small idea.",
    )
    assert checks["component_realized_on"] == {"MP": True, "ME": False}
    assert not checks["fallback_required"]


def test_v1_5b_bundle_unsafe_shared_response_falls_back_all_components() -> None:
    plans = {
        "MS": build_resource_execution_plan(
            component="MS", resource_subtype="MS_SESSION"
        ),
        "RS": build_resource_execution_plan(
            component="RS", resource_subtype="Providing Suggestions"
        ),
    }
    output = ResourceBundleApplicationOutput.model_validate(
        {
            "component_statuses": [
                {
                    "component": "MS",
                    "application_status": "applied",
                    "applied_function": "session_continuity",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "late work shift",
                    "response_evidence_excerpt": "Last time, you found it helpful to take short breaks.",
                },
                {
                    "component": "RS",
                    "application_status": "applied",
                    "applied_function": "support_technique",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "optional low-burden step",
                    "response_evidence_excerpt": "Try that now.",
                },
            ],
            "response": "Last time, you found it helpful to take short breaks. Try that now.",
        }
    )
    checks = validate_resource_bundle_application(
        output=output,
        plans=plans,
        resource_labels={"MS": "MS1", "RS": "RS1"},
        selected_resources={
            "MS": "Earlier, the user discussed a late work shift.",
            "RS": "Offer one optional low-burden step.",
        },
        current_user_text="My concentration is difficult today.",
    )
    assert checks["unsafe_applied_component"]
    assert checks["fallback_required"]
    assert checks["component_realized_on"] == {"MS": False, "RS": False}


def test_v1_5b_bundle_extraneous_trace_status_forces_fallback() -> None:
    plans = {
        "RS": build_resource_execution_plan(
            component="RS", resource_subtype="Question"
        )
    }
    output = ResourceBundleApplicationOutput.model_validate(
        {
            "component_statuses": [
                {
                    "component": "RS",
                    "application_status": "applied",
                    "applied_function": "support_technique",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "focused question",
                    "response_evidence_excerpt": "What part feels hardest right now?",
                },
                {
                    "component": "MS",
                    "application_status": "applied",
                    "applied_function": "session_continuity",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "old detail",
                    "response_evidence_excerpt": "What part feels hardest right now?",
                },
            ],
            "response": "What part feels hardest right now?",
        }
    )
    checks = validate_resource_bundle_application(
        output=output,
        plans=plans,
        resource_labels={"RS": "RS1"},
        selected_resources={"RS": "Ask one focused question."},
        current_user_text="Everything feels tangled.",
    )
    assert not checks["requested_component_bookkeeping_valid"]
    assert len(checks["extraneous_component_statuses_ignored_for_routing"]) == 1
    assert checks["component_realized_on"] == {"RS": False}
    assert checks["fallback_required"]


def test_v1_5b_bundle_duplicate_requested_status_forces_fallback() -> None:
    plan = build_resource_execution_plan(component="RS", resource_subtype="Question")
    base = {
        "component": "RS",
        "application_status": "applied",
        "applied_function": "support_technique",
        "cannot_apply_reason": "none",
        "resource_support_excerpt": "focused question",
        "response_evidence_excerpt": "What part feels hardest right now?",
    }
    output = ResourceBundleApplicationOutput.model_validate(
        {
            "component_statuses": [base, dict(base)],
            "response": "What part feels hardest right now?",
        }
    )
    checks = validate_resource_bundle_application(
        output=output,
        plans={"RS": plan},
        resource_labels={"RS": "RS1"},
        selected_resources={"RS": "Ask one focused question."},
    )
    assert checks["duplicate_identical_requested_status_count"] == 1
    assert checks["component_realized_on"] == {"RS": False}
    assert checks["fallback_required"]


def test_v1_5b_call_specific_bundle_schema_rejects_unrequested_components() -> None:
    schema = resource_bundle_output_schema_for_components(["RS"])
    valid = schema.model_validate(
        {
            "component_statuses": [
                {
                    "component": "RS",
                    "application_status": "applied",
                    "applied_function": "support_technique",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "focused question",
                    "response_evidence_excerpt": "What part feels hardest right now?",
                }
            ],
            "response": "What part feels hardest right now?",
        }
    )
    assert valid.component_statuses[0].component == "RS"
    with pytest.raises(ValidationError):
        schema.model_validate(
            {
                "component_statuses": [
                    {
                        "component": "MS",
                        "application_status": "applied",
                        "applied_function": "session_continuity",
                        "cannot_apply_reason": "none",
                        "resource_support_excerpt": "old detail",
                        "response_evidence_excerpt": "Last time, you mentioned this.",
                    }
                ],
                "response": "Last time, you mentioned this.",
            }
        )


def test_v1_5b_bundle_never_lets_me_outcome_authorize_ms_claim() -> None:
    plans = {
        "MS": build_resource_execution_plan(
            component="MS", resource_subtype="MS_SESSION"
        ),
        "ME": build_resource_execution_plan(
            component="ME", resource_subtype="ME_EVENT_OUTCOME"
        ),
    }
    output = ResourceBundleApplicationOutput.model_validate(
        {
            "component_statuses": [
                {
                    "component": "MS",
                    "application_status": "applied",
                    "applied_function": "session_continuity",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "distinguished emotional uncertainty from practical workload",
                    "response_evidence_excerpt": "Last time, you found that distinction helpful.",
                },
                {
                    "component": "ME",
                    "application_status": "applied",
                    "applied_function": "event_or_outcome",
                    "cannot_apply_reason": "none",
                    "resource_support_excerpt": "a brief pause helped",
                    "response_evidence_excerpt": "In the past, a brief pause helped",
                },
            ],
            "response": (
                "Last time, you found that distinction helpful. "
                "In the past, a brief pause helped; would that fit today?"
            ),
        }
    )
    checks = validate_resource_bundle_application(
        output=output,
        plans=plans,
        resource_labels={"MS": "MS1", "ME": "ME1"},
        selected_resources={
            "MS": "The earlier session distinguished emotional uncertainty from practical workload.",
            "ME": "In one past event, a brief pause helped reduce the pressure.",
        },
        current_user_text="I cannot tell which kind of pressure is strongest today.",
    )
    assert not checks["component_checks"]["MS"]["fabricated_recall_absent"]
    assert checks["component_checks"]["ME"]["fabricated_recall_absent"]
    assert checks["fallback_required"]
    assert checks["component_realized_on"] == {"MS": False, "ME": False}
