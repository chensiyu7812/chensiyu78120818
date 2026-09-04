from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.io import read_json, write_json
from metacom_pm.v1_5_training_data_root_cause import (
    LEARNABLE_TRANSFER_TRAINING_PROTOCOL,
    matching_target_treatment_rows,
    require_learnable_transfer_training_contract,
    summarize_memory_treatment_composition,
)


ROOT = Path(__file__).resolve().parents[1]


def _context(regime: str = "profile_needed"):
    return {
        "state_id": "state_1",
        "regime": regime,
        "memory_annotations": [
            {"memory_id": "help", "item_utility": "helpful"},
            {"memory_id": "noise", "item_utility": "irrelevant"},
            {"memory_id": "harm", "item_utility": "harmful"},
        ],
    }


def test_target_treatment_composition_exposes_helpful_noise_confound():
    contexts = {"state_1": _context()}
    rows = [
        {
            "state_id": "state_1",
            "requested_action_id": "MP+R0",
            "selected_memory_ids": ["help", "noise"],
        },
        {
            "state_id": "state_1",
            "requested_action_id": "M0+R0",
            "selected_memory_ids": [],
        },
    ]
    target = matching_target_treatment_rows(
        action_rows=rows, evaluator_contexts_by_state=contexts
    )
    assert target == rows[:1]
    report = summarize_memory_treatment_composition(
        action_rows=target, evaluator_contexts_by_state=contexts
    )
    assert report["all_non_m0_actions"]["rows"] == 1
    assert report["all_non_m0_actions"]["helpful_items"] == 1
    assert report["all_non_m0_actions"]["irrelevant_items"] == 1
    assert report["all_non_m0_actions"]["mixed_helpful_nonhelpful_row_rate"] == 1.0


def test_selected_memory_requires_evaluator_annotation():
    with pytest.raises(ValueError, match="lacks an evaluator annotation"):
        summarize_memory_treatment_composition(
            action_rows=[
                {
                    "state_id": "state_1",
                    "requested_action_id": "MP+R0",
                    "selected_memory_ids": ["unknown"],
                }
            ],
            evaluator_contexts_by_state={"state_1": _context()},
        )


def test_matching_target_treatment_excludes_wrong_source():
    rows = [
        {
            "state_id": "state_1",
            "requested_action_id": "MS+R0",
            "selected_memory_ids": ["noise"],
        }
    ]
    assert (
        matching_target_treatment_rows(
            action_rows=rows,
            evaluator_contexts_by_state={"state_1": _context()},
        )
        == []
    )


def test_learnable_transfer_contract_is_self_bound_and_external_outcome_blind():
    path = (
        ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "learnable_transfer_training_data_v2.json"
    )
    contract = require_learnable_transfer_training_contract(path)
    assert contract["protocol"] == LEARNABLE_TRANSFER_TRAINING_PROTOCOL
    assert contract["external_transfer_support"]["ESConv"][
        "test_gold_response_strategy_or_outcome_may_not_be_read"
    ]
    evoemo = contract["external_transfer_support"][
        "EvoEmo_ES_MemEval_derived"
    ]
    assert evoemo[
        "raw_external_text_topic_related_session_and_outcome_copying_forbidden"
    ]
    assert contract["pretraining_gates"][
        "full_generation_or_fit_before_all_gates_forbidden"
    ]
    assert contract["split_discipline"]["internal_test"].startswith("sealed")
    unified = contract["unified_router_definition"]
    assert unified["one_PM_checkpoint"]
    assert unified["domain_identifier_as_router_feature_forbidden"]
    assert "boundary case" in unified["ESConv_is_not_a_separate_policy"]
    assert "same state-action function" in unified[
        "EvoEmo_is_not_a_separate_policy"
    ]


def test_learnable_transfer_contract_rejects_drift(tmp_path):
    source = (
        ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "learnable_transfer_training_data_v2.json"
    )
    payload = read_json(source)
    payload["pretraining_gates"][
        "gate_6_external_input_support"
    ] = "silently relaxed"
    drifted = tmp_path / "drifted.json"
    write_json(drifted, payload)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        require_learnable_transfer_training_contract(drifted)
