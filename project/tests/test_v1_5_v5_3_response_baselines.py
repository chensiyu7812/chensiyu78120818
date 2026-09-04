from collections import Counter

import pytest

from metacom_pm.v1_5_v5_3_response_baselines import (
    POLICIES,
    BaselineFreeze,
    FrozenResponseState,
    build_response_baseline_plan,
)


def state(index: int, learned: dict[str, bool], *, tokens=None, eligible=None):
    eligible = eligible or {"MP": True, "MS": True, "ME": True, "RS": True}
    candidates = {
        component: f"candidate_{component}_{index}" if on else None
        for component, on in eligible.items()
    }
    tokens = tokens or {"MP": 20, "MS": 30, "ME": 40, "RS": 25}
    return FrozenResponseState(
        state_id=f"state_{index}",
        stratum="external",
        candidate_snapshot_sha256=f"candidates-{index}",
        step2_shared_surface_sha256=f"step2-{index}",
        eligible_bits=eligible,
        candidate_ids=candidates,
        incremental_tokens={
            component: tokens[component] if eligible[component] else 0
            for component in eligible
        },
        transparent_rule_bits={
            "MP": False,
            "MS": eligible["MS"],
            "ME": False,
            "RS": eligible["RS"],
        },
        learned_pm_bits=learned,
    )


def freeze():
    return BaselineFreeze(
        protocol="test-v5.3-baselines",
        seed_labels=("a", "b"),
        cost_matched_fixed_action_by_stratum={"external": "M0+RS"},
        matched_random_seed="frozen-random-seed",
    )


def test_six_logical_arms_share_state_surface():
    states = [
        state(1, {"MP": True, "MS": False, "ME": False, "RS": True}),
        state(2, {"MP": False, "MS": True, "ME": True, "RS": False}),
    ]
    plan = build_response_baseline_plan(states, freeze=freeze())
    assert plan["report"]["policies"] == list(POLICIES)
    assert len(plan["logical_bindings"]) == 2 * 6 * 2
    assert len(plan["physical_calls"]) <= len(plan["logical_bindings"])
    for item in states:
        rows = [row for row in plan["logical_bindings"] if row["state_id"] == item.state_id]
        assert {row["candidate_snapshot_sha256"] for row in rows} == {
            item.candidate_snapshot_sha256
        }
        assert {row["step2_shared_surface_sha256"] for row in rows} == {
            item.step2_shared_surface_sha256
        }
        assert {row["policy"] for row in rows} == set(POLICIES)


def test_matched_random_exactly_preserves_on_counts_and_estimated_cost():
    states = [
        state(1, {"MP": True, "MS": False, "ME": False, "RS": True}),
        state(2, {"MP": False, "MS": True, "ME": True, "RS": False}),
        state(3, {"MP": False, "MS": False, "ME": True, "RS": True}),
        state(4, {"MP": True, "MS": True, "ME": False, "RS": False}),
    ]
    plan = build_response_baseline_plan(states, freeze=freeze())
    rows = [row for row in plan["logical_bindings"] if row["seed_label"] == "a"]
    learned = [row for row in rows if row["policy"] == "learned_pm_full"]
    random = [
        row for row in rows if row["policy"] == "cost_and_on_rate_matched_random"
    ]
    learned_counts = Counter()
    random_counts = Counter()
    for row in learned:
        learned_counts.update(component for component, on in row["requested_bits"].items() if on)
    for row in random:
        random_counts.update(component for component, on in row["requested_bits"].items() if on)
    assert learned_counts == random_counts
    assert sum(row["estimated_incremental_tokens"] for row in learned) == sum(
        row["estimated_incremental_tokens"] for row in random
    )
    assert plan["report"]["matched_random"]["has_nonalias_contrast"]


def test_singleton_exchangeability_cell_reports_no_random_contrast():
    plan = build_response_baseline_plan(
        [state(1, {"MP": True, "MS": False, "ME": False, "RS": True})],
        freeze=freeze(),
    )
    assert not plan["report"]["matched_random"]["has_nonalias_contrast"]
    logical = plan["logical_bindings"]
    learned = next(row for row in logical if row["policy"] == "learned_pm_full")
    random = next(
        row for row in logical
        if row["policy"] == "cost_and_on_rate_matched_random"
    )
    assert learned["physical_call_id"] == random["physical_call_id"]


def test_fixed_and_all_other_policies_cannot_bypass_eligibility():
    eligible = {"MP": False, "MS": True, "ME": False, "RS": True}
    learned = {"MP": False, "MS": True, "ME": False, "RS": True}
    plan = build_response_baseline_plan([state(1, learned, eligible=eligible)], freeze=freeze())
    for row in plan["logical_bindings"]:
        assert not row["requested_bits"]["MP"]
        assert not row["requested_bits"]["ME"]


def test_invalid_candidate_eligibility_identity_is_rejected():
    with pytest.raises(ValueError, match="candidate identity/eligibility mismatch"):
        FrozenResponseState(
            state_id="bad",
            stratum="external",
            candidate_snapshot_sha256="c",
            step2_shared_surface_sha256="s",
            eligible_bits={"MP": True, "MS": False, "ME": False, "RS": False},
            candidate_ids={"MP": None, "MS": None, "ME": None, "RS": None},
            incremental_tokens={"MP": 10, "MS": 0, "ME": 0, "RS": 0},
            transparent_rule_bits={"MP": False, "MS": False, "ME": False, "RS": False},
            learned_pm_bits={"MP": False, "MS": False, "ME": False, "RS": False},
        )
