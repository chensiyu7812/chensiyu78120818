from __future__ import annotations

from metacom_pm.v1_5_esconv_auxiliary_pairwise import (
    PAIRWISE_PILOT_PROTOCOL,
    aggregate_pairwise_pilot,
    build_pairwise_items,
    build_pairwise_messages,
    pairwise_contract_record,
    select_train_only_pairwise_states,
)


def _seed_rows():
    return [
        {
            "dialogue_id": f"d{index:02d}",
            "selection_ordinal": index,
            "source_split": "train",
        }
        for index in range(24)
    ]


def _runtime_rows():
    rows = []
    for dialogue_index in range(24):
        for turn_index in (2, 4, 6):
            rows.append(
                {
                    "state_id": f"s{dialogue_index:02d}_{turn_index}",
                    "card_id": f"c{dialogue_index:02d}_{turn_index}",
                    "provenance": {
                        "dialogue_id": f"d{dialogue_index:02d}",
                        "turn_index": turn_index,
                    },
                }
            )
    return rows


def test_pairwise_contract_is_train_only_and_content_addressed():
    record = pairwise_contract_record()
    assert record["protocol"] == PAIRWISE_PILOT_PROTOCOL
    assert record["scope"] == "train_only_measurement_instrument_pilot"
    assert record["decision_boundary"]["may_not_create_training_labels"] is True
    assert record["decision_boundary"]["may_not_read_calibration_or_internal_test"] is True
    assert len(record["contract_sha256"]) == 64


def test_selection_is_one_per_dialogue_and_cycles_turn_strata():
    selected = select_train_only_pairwise_states(
        seed_rows=_seed_rows(),
        runtime_rows=_runtime_rows(),
    )
    assert len(selected) == 24
    assert len({row["dialogue_id"] for row in selected}) == 24
    assert [row["stratum"] for row in selected[:6]] == [
        "early",
        "mid",
        "late",
        "early",
        "mid",
        "late",
    ]
    assert [row["turn_index"] for row in selected[:3]] == [2, 4, 6]


def test_pairwise_items_blind_and_reverse_the_same_two_responses():
    selected = select_train_only_pairwise_states(
        seed_rows=_seed_rows(),
        runtime_rows=_runtime_rows(),
    )
    states = [
        {
            "state_id": row["state_id"],
            "current_user_text": "I feel stuck.",
            "current_session_history": [],
            "current_session_summary": "",
        }
        for row in selected
    ]
    outcomes = []
    for row in selected:
        outcomes.extend(
            [
                {
                    "state_id": row["state_id"],
                    "action_id": "M0+R0",
                    "response": f"listen {row['state_id']}",
                },
                {
                    "state_id": row["state_id"],
                    "action_id": "M0+RS",
                    "response": f"strategy {row['state_id']}",
                },
            ]
        )
    items = build_pairwise_items(
        selection=selected,
        state_rows=states,
        outcome_rows=outcomes,
    )
    assert len(items) == 48
    first, reverse = items[:2]
    assert (first["action_a"], first["action_b"]) == ("M0+R0", "M0+RS")
    assert (reverse["action_a"], reverse["action_b"]) == ("M0+RS", "M0+R0")
    assert first["response_a"] == reverse["response_b"]
    assert first["response_b"] == reverse["response_a"]
    messages = build_pairwise_messages(
        state=states[0],
        response_a=first["response_a"],
        response_b=first["response_b"],
    )
    visible = str(messages)
    assert "M0+R0" not in visible
    assert "M0+RS" not in visible
    assert "quality_preference" in visible
    assert "safer_preference" in visible
    assert "qualityPreference" not in visible
    assert "nested objects" in visible


def test_aggregate_requires_order_and_family_agreement():
    rows = []
    for state_index in range(2):
        for family in ("a", "b"):
            rows.extend(
                [
                    {
                        "state_id": f"s{state_index}",
                        "judge_family": family,
                        "order_variant": 0,
                        "action_a": "M0+R0",
                        "action_b": "M0+RS",
                        "quality_preference": "B",
                        "safer_preference": "A",
                    },
                    {
                        "state_id": f"s{state_index}",
                        "judge_family": family,
                        "order_variant": 1,
                        "action_a": "M0+RS",
                        "action_b": "M0+R0",
                        "quality_preference": "A",
                        "safer_preference": "B",
                    },
                ]
            )
    report = aggregate_pairwise_pilot(rows, expected_families=["a", "b"])
    assert report["status"] == "PILOT_GO"
    assert report["training_labels_created"] is False
    assert report["dimensions"]["quality_preference"]["cross_family_agreement"] == 1.0
    assert report["dimensions"]["quality_preference"]["consensus_non_tie_rate"] == 1.0
    assert (
        report["dimensions"]["quality_preference"][
            "within_family_order_consistency_is_diagnostic_only"
        ]
        is True
    )
    assert (
        report["dimensions"]["quality_preference"]["raw_position_bias_by_family"][
            "a"
        ]["absolute_position_bias"]
        == 0.0
    )


def test_order_disagreement_resolves_to_tie_and_fails_effective_signal():
    rows = [
        {
            "state_id": "s0",
            "judge_family": family,
            "order_variant": order,
            "action_a": "M0+R0" if order == 0 else "M0+RS",
            "action_b": "M0+RS" if order == 0 else "M0+R0",
            "quality_preference": "A",
            "safer_preference": "A",
        }
        for family in ("a", "b")
        for order in (0, 1)
    ]
    report = aggregate_pairwise_pilot(rows, expected_families=["a", "b"])
    assert report["status"] == "PILOT_NO_GO"
    assert not any("order_consistency" in key for key in report["checks"])
    assert (
        report["dimensions"]["quality_preference"][
            "within_family_order_consistency"
        ]["a"]
        == 0.0
    )
