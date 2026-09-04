from __future__ import annotations

from types import SimpleNamespace

import pytest

from metacom_pm.v1_5_response_mechanism_pilot import (
    RESPONSE_MECHANISM_PILOT_REGIMES,
    response_mechanism_pilot_contract,
    select_response_mechanism_pilot_states,
)


def _state(index: int, regime: str):
    return SimpleNamespace(
        state_id=f"state-{regime}-{index}",
        card_id=f"card-{regime}-{index}",
        user_id=f"user-{regime}-{index}",
        semantic_family=f"family-{regime}",
        split=SimpleNamespace(value="train"),
    )


def test_selection_is_train_only_balanced_and_user_disjoint() -> None:
    states = [
        _state(index, regime)
        for regime in RESPONSE_MECHANISM_PILOT_REGIMES
        for index in range(3)
    ]
    evaluator = {}
    for state in states:
        regime = state.semantic_family.removeprefix("family-")
        needed = {
            "profile_needed": ["MP"],
            "summary_needed": ["MS"],
            "event_needed": ["ME"],
            "multi_source_needed": ["MP", "MS"],
        }.get(regime, [])
        evaluator[state.state_id] = {
            "regime": regime,
            "needed_memory_sources": needed,
        }

    selected = select_response_mechanism_pilot_states(
        states, evaluator, states_per_regime=2
    )
    contract = response_mechanism_pilot_contract(
        selected_states=selected,
        source_lineage={"fixture": "unit"},
        states_per_regime=2,
    )

    assert len(selected) == 14
    assert len({row["user_id"] for row in selected}) == 14
    assert contract["planned_new_logical_calls"] == 14
    assert set(contract["arms"]) == {
        "frozen_current_control",
        "top1_evidence_surface_same_prompt",
    }
    assert contract["training_labels_created"] is False
    assert contract["evaluation"]["api_judges"] == "forbidden"


def test_contract_rejects_user_reuse() -> None:
    rows = [
        {
            "state_id": f"state-{index}",
            "user_id": "same-user",
            "regime": regime,
        }
        for index, regime in enumerate(RESPONSE_MECHANISM_PILOT_REGIMES * 2)
    ]
    with pytest.raises(RuntimeError, match="users are not disjoint"):
        response_mechanism_pilot_contract(
            selected_states=rows,
            source_lineage={"fixture": "unit"},
            states_per_regime=2,
        )
