from metacom_pm.v1_5_typed_resource_adapter import TypedResourceCandidate
from metacom_pm.v1_5_v5_2_locked_composer import (
    base_generation_messages,
    build_locked_composition_plan,
    compose_locked_response,
    locked_response_guard_errors,
)


def ms() -> TypedResourceCandidate:
    return TypedResourceCandidate(
        component="MS",
        subtype="MS_SESSION_OBSERVATION",
        resource_id="memory_session_001",
        candidate_version="v1",
        source_kind="session",
        owner_id="u1",
        strictly_prior=True,
        age_sessions=2,
        prior_observation="The seeker felt overloaded after a shift change.",
    )


def me() -> TypedResourceCandidate:
    literal = "I paused before replying. That helped me avoid escalating the disagreement."
    return TypedResourceCandidate(
        component="ME",
        subtype="ME_REUSABLE_OUTCOME",
        resource_id="memory_event_001",
        candidate_version="v1",
        source_kind="event",
        owner_id="u1",
        strictly_prior=True,
        age_sessions=3,
        past_action="I paused before replying.",
        observed_outcome="That helped me avoid escalating the disagreement.",
        mechanism="literal_span:" + literal,
    )


def test_history_is_not_exposed_to_generator_but_is_locked_into_final() -> None:
    plan = build_locked_composition_plan(
        requested_action_id="MSE+R0", candidates={"MS": ms(), "ME": me()}
    )
    messages = base_generation_messages(current_context="User: I feel stuck.", plan=plan)
    assert "shift change" not in messages[0]["content"]
    assert "paused before replying" not in messages[0]["content"]
    final = compose_locked_response(base_response="That sounds difficult.", plan=plan)
    assert "An earlier session recorded" in final
    assert "You previously said" in final
    assert locked_response_guard_errors(response=final, plan=plan) == ()


def test_guard_checks_exact_locked_clause_not_history_keywords() -> None:
    plan = build_locked_composition_plan(
        requested_action_id="MS+R0", candidates={"MS": ms()}
    )
    assert locked_response_guard_errors(
        response="Previously, something happened.", plan=plan
    ) == ("LOCKED_MS_CLAUSE_MISSING",)


def test_no_resource_is_valid_and_current_only() -> None:
    plan = build_locked_composition_plan(requested_action_id="M0+R0", candidates={})
    messages = base_generation_messages(current_context="User: I feel tired.", plan=plan)
    assert "Answer the user's current request naturally" in messages[0]["content"]
    assert "backend will append the requested" not in messages[0]["content"]
    final = compose_locked_response(base_response="That sounds exhausting.", plan=plan)
    assert locked_response_guard_errors(response=final, plan=plan) == ()


def test_locked_memory_gets_lead_in_while_r0_remains_a_real_baseline() -> None:
    locked = build_locked_composition_plan(
        requested_action_id="MS+R0", candidates={"MS": ms()}
    )
    on_messages = base_generation_messages(
        current_context="User: Please remind me what was recorded.", plan=locked
    )
    assert "backend will append the requested verified resource content" in on_messages[0]["content"]
    assert "Do not add advice" in on_messages[0]["content"]

    off = build_locked_composition_plan(requested_action_id="M0+R0", candidates={})
    off_messages = base_generation_messages(
        current_context="User: Please remind me what was recorded.", plan=off
    )
    assert "Answer the user's current request naturally" in off_messages[0]["content"]
