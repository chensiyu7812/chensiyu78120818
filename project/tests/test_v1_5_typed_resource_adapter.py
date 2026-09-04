from itertools import product

import pytest

from metacom_pm.v1_5_typed_resource_adapter import (
    TypedResourceCandidate,
    compile_typed_bundle,
    deterministic_context_only_fallback,
    response_guard_errors,
    response_only_messages,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS, compile_component_bits


def candidate(component: str) -> TypedResourceCandidate:
    common = {
        "component": component,
        "resource_id": f"resource-{component.lower()}",
        "candidate_version": "candidate-v1",
    }
    if component == "MP":
        return TypedResourceCandidate(
            **common,
            subtype="MP_PREFERENCE",
            source_kind="profile",
            owner_id="user-1",
            preference="offer one optional choice rather than a command",
        )
    if component == "MS":
        return TypedResourceCandidate(
            **common,
            subtype="MS_SESSION_OBSERVATION",
            source_kind="session",
            owner_id="user-1",
            strictly_prior=True,
            age_sessions=2,
            prior_observation="the late-to-early shift transition was the difficult part",
        )
    if component == "ME":
        return TypedResourceCandidate(
            **common,
            subtype="ME_REUSABLE_OUTCOME",
            source_kind="event",
            owner_id="user-1",
            strictly_prior=True,
            age_sessions=3,
            past_action="wrote one sentence before the conversation",
            observed_outcome="the discussion stayed focused",
            mechanism="limiting the first step reduced overload",
        )
    return TypedResourceCandidate(
        **common,
        subtype="RS_ATOMIC_MOVE",
        source_kind="strategy",
        support_move="ask one focused clarification question",
        when_to_use="the user welcomes one question",
        when_not_to_use="the user asks for no questions",
    )


def test_all_16_actions_compile_exactly_their_requested_candidates():
    for values in product((False, True), repeat=4):
        bits = dict(zip(COMPONENTS, values, strict=True))
        action = compile_component_bits(bits)
        candidates = {
            component: candidate(component)
            for component, enabled in bits.items()
            if enabled
        }
        bundle = compile_typed_bundle(
            requested_action_id=action,
            candidates=candidates,
            current_user_id="user-1",
        )
        assert set(bundle.components) == {key for key, value in bits.items() if value}


def test_m0_r0_is_legal_and_does_not_require_a_resource():
    bundle = compile_typed_bundle(
        requested_action_id="M0+R0", candidates={}, current_user_id="user-1"
    )
    assert bundle.directives == ()


def test_wrong_owner_and_action_candidate_mismatch_fail_mechanically():
    with pytest.raises(ValueError, match="owner"):
        compile_typed_bundle(
            requested_action_id="MS+R0",
            candidates={"MS": candidate("MS")},
            current_user_id="user-2",
        )
    with pytest.raises(ValueError, match="candidate/action mismatch"):
        compile_typed_bundle(
            requested_action_id="M0+R0",
            candidates={"RS": candidate("RS")},
            current_user_id="user-1",
        )


def test_ms_me_require_real_prior_typed_evidence():
    with pytest.raises(ValueError, match="strictly prior"):
        TypedResourceCandidate(
            component="ME",
            subtype="ME_REUSABLE_OUTCOME",
            resource_id="event-1",
            candidate_version="v1",
            source_kind="event",
            owner_id="user-1",
            past_action="paused",
            observed_outcome="felt calmer",
        )


def test_response_prompt_omits_ids_and_generator_self_reporting_contract():
    bundle = compile_typed_bundle(
        requested_action_id="MPE+RS",
        candidates={key: candidate(key) for key in ("MP", "ME", "RS")},
        current_user_id="user-1",
    )
    messages = response_only_messages(current_context="I feel stuck.", bundle=bundle)
    prompt = "\n".join(item["content"] for item in messages)
    assert "resource-mp" not in prompt
    assert "candidate-v1" not in prompt
    assert "declared_reason" not in prompt
    assert "cannot_apply" not in prompt
    assert "use/ignore" not in prompt


def test_deterministic_fallback_is_history_free_and_boundary_aware():
    listen = deterministic_context_only_fallback("Please don't ask questions; just listen.")
    stop = deterministic_context_only_fallback("Please stop. Bye.")
    for response in (listen, stop):
        lowered = response.lower()
        assert "last time" not in lowered
        assert "memory" not in lowered
        assert "profile" not in lowered
    assert "stop" in stop.lower()


def test_machine_guard_is_not_a_semantic_judge_but_blocks_clear_leaks():
    empty = compile_typed_bundle(
        requested_action_id="M0+R0", candidates={}, current_user_id="user-1"
    )
    assert response_guard_errors(response="I hear you.", bundle=empty) == ()
    errors = response_guard_errors(response="Last time, MS said this.", bundle=empty)
    assert "INTERNAL_LABEL_OR_ID_LEAK" in errors


def test_common_pronoun_me_is_not_mistaken_for_internal_ME_label():
    empty = compile_typed_bundle(
        requested_action_id="M0+R0", candidates={}, current_user_id="user-1"
    )
    assert response_guard_errors(response="Can you tell me more?", bundle=empty) == ()


def test_visible_current_reference_to_the_past_is_not_unauthorized_memory_use():
    empty = compile_typed_bundle(
        requested_action_id="M0+R0", candidates={}, current_user_id="user-1"
    )
    assert response_guard_errors(
        response="What made the earlier attempt harder?",
        bundle=empty,
        current_context="I want to avoid what made it harder before.",
    ) == ()


def test_free_temporal_language_is_left_to_grounding_risk_review():
    empty = compile_typed_bundle(
        requested_action_id="M0+R0", candidates={}, current_user_id="user-1"
    )
    assert response_guard_errors(
        response="Take a pause before returning to it.", bundle=empty
    ) == ()
