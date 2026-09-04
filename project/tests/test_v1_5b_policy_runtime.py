from itertools import product

from metacom_pm.contracts import ALL_ACTION_IDS
from metacom_pm.v1_5b_policy_runtime import (
    COMPONENTS,
    ComponentFeasibility,
    ComponentOpportunity,
    PolicyExecutionTrace,
    action_component_bits,
    compile_component_bits,
    project_joint_feasibility,
    realize_guarded_action,
    route_policy,
)


def opportunity(component: str, **overrides) -> ComponentOpportunity:
    values = {
        "component": component,
        "candidate_present": True,
        "hard_gate_pass": True,
        "transparent_rule_on": True,
        "learned_in_support": True,
        "learned_on": True,
        "learned_probability": 0.8,
        "decision_reason": "FROZEN_HEAD_ON",
    }
    values.update(overrides)
    return ComponentOpportunity(**values)


def all_opportunities(**by_component):
    return {
        component: opportunity(component, **by_component.get(component, {}))
        for component in COMPONENTS
    }


def test_all_16_bit_patterns_round_trip_to_existing_actions():
    observed = set()
    for values in product((False, True), repeat=4):
        bits = dict(zip(COMPONENTS, values, strict=True))
        action = compile_component_bits(bits)
        observed.add(action)
        assert action_component_bits(action) == bits
    assert observed == set(ALL_ACTION_IDS)


def test_full_and_conservative_are_distinct_without_hand_closing_full_claim():
    opportunities = all_opportunities()
    full = route_policy(policy="learned_pm_full", opportunities=opportunities)
    conservative = route_policy(
        policy="learned_pm_conservative", opportunities=opportunities
    )
    assert full.action_id == "MPMSME+RS"
    assert conservative.action_id == "ME+RS"


def test_no_policy_can_bypass_candidate_or_hard_gate():
    opportunities = all_opportunities(
        MP={"candidate_present": False},
        MS={"hard_gate_pass": False},
    )
    for policy in (
        "fixed_high_resource",
        "transparent_rule",
        "learned_pm_full",
        "learned_pm_conservative",
        "cost_matched_fixed",
    ):
        route = route_policy(
            policy=policy,
            opportunities=opportunities,
            cost_matched_action_id="MPMSME+RS",
        )
        assert not route.requested_bits["MP"]
        assert not route.requested_bits["MS"]


def test_learned_ood_is_fail_closed_but_only_for_that_component():
    opportunities = all_opportunities(MS={"learned_in_support": False})
    route = route_policy(policy="learned_pm_full", opportunities=opportunities)
    assert route.action_id == "MPE+RS"


def test_guard_projection_preserves_passing_components():
    realized = realize_guarded_action(
        requested_action_id="MPMSME+RS",
        component_passed={"MP": True, "MS": False, "ME": True, "RS": False},
    )
    assert realized == "MPE+R0"


def _feasibility(component: str, **overrides) -> ComponentFeasibility:
    values = {
        "component": component,
        "candidate_present": True,
        "hard_gate_pass": True,
        "opportunity_score": 0.8,
        "incremental_tokens": 40,
        "conflicts_with": frozenset(),
    }
    values.update(overrides)
    return ComponentFeasibility(**values)


def test_joint_feasibility_preserves_requested_feasible_realized_chain():
    rows = {
        component: _feasibility(component) for component in COMPONENTS
    }
    rows["MP"] = _feasibility("MP", candidate_present=False)
    rows["MS"] = _feasibility("MS", conflicts_with=frozenset({"RS"}))
    rows["RS"] = _feasibility("RS", opportunity_score=0.9)
    projected = project_joint_feasibility(
        requested_action_id="MPMSME+RS",
        feasibility=rows,
        maximum_incremental_tokens=100,
    )
    # MP is absent; MS loses its conflict with higher-scoring RS. ME+RS then
    # fits exactly under the shared budget.
    assert projected.feasible_action_id == "ME+RS"
    assert projected.dropped_components == ("MP", "MS")
    assert projected.incremental_tokens == 80
    trace = PolicyExecutionTrace(
        requested_action_id=projected.requested_action_id,
        feasible_action_id=projected.feasible_action_id,
        realized_action_id="ME+R0",
    )
    assert trace.realized_action_id == "ME+R0"


def test_joint_feasibility_drops_lower_value_until_budget_is_met():
    rows = {
        component: _feasibility(component, incremental_tokens=40)
        for component in COMPONENTS
    }
    rows["MP"] = _feasibility("MP", opportunity_score=0.55, incremental_tokens=50)
    rows["MS"] = _feasibility("MS", opportunity_score=0.60, incremental_tokens=50)
    projected = project_joint_feasibility(
        requested_action_id="MPMSME+RS",
        feasibility=rows,
        maximum_incremental_tokens=100,
    )
    assert projected.incremental_tokens <= 100
    assert not projected.feasible_bits["MP"]
    assert not projected.feasible_bits["MS"]
    assert projected.feasible_action_id == "ME+RS"


def test_execution_trace_rejects_component_added_after_feasibility():
    import pytest

    with pytest.raises(ValueError, match="infeasible"):
        PolicyExecutionTrace(
            requested_action_id="MPMSME+RS",
            feasible_action_id="ME+R0",
            realized_action_id="MPE+R0",
        )
