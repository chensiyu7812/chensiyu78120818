from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from metacom_pm.io import canonical_json
from metacom_pm.v1_5_oracle_memory_pilot import (
    CONTROL_ARM,
    HARMFUL_ARM,
    HELPFUL_ARM,
    ORACLE_MEMORY_PILOT_PROTOCOL,
)
from metacom_pm.v1_5_oracle_memory_quality import (
    aggregate_quality_diagnostic,
    build_quality_items,
    build_quality_messages,
    quality_contract_record,
)


def _fixture():
    selected = []
    states = []
    backends = []
    outcomes = []
    for index in range(18):
        harmful = index >= 12
        state_id = f"state-{index}"
        card_id = f"card-{index}"
        memory_id = f"memory-{index}"
        arm = HARMFUL_ARM if harmful else HELPFUL_ARM
        selected.append(
            {
                "state_id": state_id,
                "card_id": card_id,
                "user_id": f"user-{index}",
                "regime": "memory_harmful" if harmful else "profile_needed",
                "treatment_arm": arm,
                "target_action_id": "MP+R0",
                "target_item_utility": "harmful" if harmful else "helpful",
                "target_memory_ids": [memory_id],
                "target_memory_sources": ["MP"],
            }
        )
        states.append(
            {
                "state_id": state_id,
                "card_id": card_id,
                "split": "train",
                "current_user_text": f"current {index}",
                "current_session_history": [
                    {"role": "user", "content": f"history {index}"}
                ],
                "current_session_summary": f"summary {index}",
            }
        )
        backends.append(
            {
                "card_id": card_id,
                "items": [
                    {
                        "memory_id": memory_id,
                        "source": "MP",
                        "text": f"candidate record {index}",
                    }
                ],
            }
        )
        common_provenance = {
            "supporter_generation_treatment_sha256": "frozen",
            "temperature": 0.0,
            "seed": 4311,
        }
        outcomes.extend(
            [
                {
                    "state_id": state_id,
                    "card_id": card_id,
                    "action_id": "M0+R0",
                    "selected_memory_ids": [],
                    "memory_view": [],
                    "strategy_view": [],
                    "response": f"control {index}",
                    "model_name": "generator",
                    "provenance": {
                        **common_provenance,
                        "oracle_memory_pilot_arm": CONTROL_ARM,
                    },
                },
                {
                    "state_id": state_id,
                    "card_id": card_id,
                    "action_id": "MP+R0",
                    "selected_memory_ids": [memory_id],
                    "memory_view": [{"memory_id": memory_id}],
                    "strategy_view": [],
                    "response": f"treatment {index}",
                    "model_name": "generator",
                    "provenance": {
                        **common_provenance,
                        "oracle_memory_pilot_arm": arm,
                    },
                },
            ]
        )
    contract = {
        "protocol": ORACLE_MEMORY_PILOT_PROTOCOL,
        "selected_states": selected,
    }
    from metacom_pm.io import sha256_text

    contract["contract_sha256"] = sha256_text(canonical_json(contract))
    return contract, states, backends, outcomes


def test_quality_packet_is_blinded_balanced_and_directional() -> None:
    contract, states, backends, outcomes = _fixture()
    items = build_quality_items(
        pilot_contract=contract,
        state_rows=states,
        backend_rows=backends,
        outcome_rows=outcomes,
    )
    assert len(items) == 36
    assert {item["order_variant"] for item in items} == {0, 1}
    assert len({item["pair_id"] for item in items}) == 36
    for item in items:
        messages = build_quality_messages(item)
        visible = canonical_json(messages)
        assert "M0+R0" not in visible
        assert "MP+R0" not in visible
        assert CONTROL_ARM not in visible
        assert HELPFUL_ARM not in visible
        assert HARMFUL_ARM not in visible
        assert "memory-" not in visible
    by_state = {}
    for item in items:
        by_state.setdefault(item["state_id"], []).append(item)
    assert all(len(rows) == 2 for rows in by_state.values())
    for state_id, rows in by_state.items():
        expected = (
            HELPFUL_ARM if int(state_id.split("-")[1]) < 12 else CONTROL_ARM
        )
        assert {row["expected_winner"] for row in rows} == {expected}
        assert {row["arm_a"] for row in rows} == {
            CONTROL_ARM,
            HELPFUL_ARM
            if int(state_id.split("-")[1]) < 12
            else HARMFUL_ARM,
        }


def test_quality_packet_rejects_outcome_or_backend_drift() -> None:
    contract, states, backends, outcomes = _fixture()
    bad_outcomes = deepcopy(outcomes)
    bad_outcomes.pop()
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        build_quality_items(
            pilot_contract=contract,
            state_rows=states,
            backend_rows=backends,
            outcome_rows=bad_outcomes,
        )
    bad_backends = deepcopy(backends)
    bad_backends[0]["items"][0]["memory_id"] = "wrong"
    with pytest.raises(RuntimeError, match="backend does not match"):
        build_quality_items(
            pilot_contract=contract,
            state_rows=states,
            backend_rows=bad_backends,
            outcome_rows=outcomes,
        )


def test_quality_contract_forbids_majority_as_gold() -> None:
    contract = quality_contract_record(source_lineage={"fixture": "unit"})
    assert contract["planned_pair_items"] == 36
    assert (
        contract["analysis"]["third_family_majority"]
        == "descriptive diagnostic only; never a gold label"
    )
    assert contract["training_labels_created"] is False
    assert contract["contract_sha256"]


def test_tracked_quality_contract_matches_current_contract_code() -> None:
    root = Path(__file__).resolve().parents[1]
    from metacom_pm.io import read_json, sha256_file

    tracked = read_json(
        root
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_quality_diagnostic_v1.json"
    )
    assert tracked == quality_contract_record(
        source_lineage=tracked["source_lineage"]
    )
    assert tracked["source_lineage"]["preparation_code_sha256"] == sha256_file(
        root / "src/metacom_pm/v1_5_oracle_memory_quality.py"
    )
    assert tracked["source_lineage"]["preparation_runner_sha256"] == sha256_file(
        root
        / "scripts/v1_5/"
        "21l_prepare_longitudinal_oracle_memory_quality_v1_5.py"
    )


def test_quality_aggregation_keeps_families_separate() -> None:
    contract, states, backends, outcomes = _fixture()
    items = build_quality_items(
        pilot_contract=contract,
        state_rows=states,
        backend_rows=backends,
        outcome_rows=outcomes,
    )
    rows = []
    for item in items:
        expected_display = (
            "A"
            if item["arm_a"] == item["expected_winner"]
            else "B"
        )
        for family in ("f1", "f2", "f3"):
            rows.append(
                {
                    **item,
                    "judge_family": family,
                    "support_quality_preference": expected_display,
                    "evidence_handling_preference": expected_display,
                    "safety_preference": expected_display,
                }
            )
    report = aggregate_quality_diagnostic(
        rows,
        expected_families=["f1", "f2", "f3"],
        expected_state_ids=[state["state_id"] for state in states],
    )
    assert report["status"] == "COMPLETE_REPORT_ONLY"
    quality = report["dimensions"]["support_quality_preference"]
    assert quality["by_family"]["f1"]["directional_accuracy_all_states"] == 1.0
    assert (
        quality["three_family_majority_diagnostic"][
            "may_be_used_as_gold_label"
        ]
        is False
    )
    assert report["training_labels_created"] is False


def test_quality_aggregation_fails_closed_on_missing_order() -> None:
    contract, states, backends, outcomes = _fixture()
    items = build_quality_items(
        pilot_contract=contract,
        state_rows=states,
        backend_rows=backends,
        outcome_rows=outcomes,
    )
    rows = []
    for item in items:
        for family in ("f1", "f2"):
            rows.append(
                {
                    **item,
                    "judge_family": family,
                    "support_quality_preference": "tie",
                    "evidence_handling_preference": "tie",
                    "safety_preference": "tie",
                }
            )
    rows.pop()
    with pytest.raises(RuntimeError, match="matrix incomplete"):
        aggregate_quality_diagnostic(
            rows,
            expected_families=["f1", "f2"],
            expected_state_ids=[state["state_id"] for state in states],
        )


def test_quality_aggregation_rejects_mutated_hidden_binding() -> None:
    contract, states, backends, outcomes = _fixture()
    items = build_quality_items(
        pilot_contract=contract,
        state_rows=states,
        backend_rows=backends,
        outcome_rows=outcomes,
    )
    rows = []
    for item in items:
        for family in ("f1", "f2"):
            rows.append(
                {
                    **item,
                    "judge_family": family,
                    "support_quality_preference": "tie",
                    "evidence_handling_preference": "tie",
                    "safety_preference": "tie",
                }
            )
    rows[0]["expected_winner"] = HARMFUL_ARM
    with pytest.raises(RuntimeError, match="expected direction drifted"):
        aggregate_quality_diagnostic(
            rows,
            expected_families=["f1", "f2"],
            expected_state_ids=[state["state_id"] for state in states],
        )
