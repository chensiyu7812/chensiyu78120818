from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_cause_diagnostic_is_zero_api_and_never_writes_labels() -> None:
    source = (
        ROOT
        / "scripts"
        / "v1_5"
        / "21d_diagnose_longitudinal_judge_instrument_v1_5.py"
    ).read_text(encoding="utf-8")

    assert "client.chat" not in source
    assert "endpoint_from_config" not in source
    assert "create_client" not in source
    assert '"training_labels_created": False' in source


def test_longitudinal_freeze_forbids_training_and_internal_consumption() -> None:
    path = (
        ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "longitudinal_supervision_freeze_v1.json"
    )
    freeze = json.loads(path.read_text(encoding="utf-8"))

    assert freeze["status"] == "LONGITUDINAL_SUPERVISION_NOT_SUPPORTED"
    assert freeze["matrix"]["raw_judge_rows"] == 10368
    assert freeze["matrix"]["state_action_rows"] == 5184
    assert freeze["frozen_consequences"]["training_labels_created"] is False
    assert freeze["frozen_consequences"]["pm_training_authorized"] is False
    assert (
        freeze["frozen_consequences"]["longitudinal_internal_test_authorized"]
        is False
    )


def test_response_mechanism_contract_is_train_only_and_not_a_training_gate() -> None:
    path = (
        ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "longitudinal_response_mechanism_pilot_v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    rows = contract["selected_states"]

    assert contract["scope"] == "longitudinal_train_only"
    assert contract["planned_new_logical_calls"] == 14
    assert contract["evaluation"]["api_judges"] == "forbidden"
    assert set(contract["arms"]) == {
        "frozen_current_control",
        "top1_evidence_surface_same_prompt",
    }
    assert contract["training_labels_created"] is False
    assert len(rows) == 14
    assert len({row["state_id"] for row in rows}) == 14
    assert len({row["user_id"] for row in rows}) == 14
