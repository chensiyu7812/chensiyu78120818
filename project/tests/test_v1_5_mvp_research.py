from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from metacom_pm.io import read_json
from metacom_pm.v1_5_mvp_research import (
    audit_v1_5_mvp_data,
    summarize_blind_preferences,
    summarize_resource_cost,
    summarize_risk_events,
    validate_metric_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict:
    return read_json(
        ROOT
        / "data/pm_v1_5_contracts/minimum_publishable_metric_registry_v1.json"
    )


def test_metric_registry_defines_atomic_minimum_publishable_claims() -> None:
    report = validate_metric_registry(_registry())
    assert report["status"] == "PASS"
    assert report["metric_count"] == 8
    assert report["combined_headline_score"] is False
    assert report["support_need_fit_is_primary_success_gate"] is False


def test_metric_registry_rejects_a_composite_or_missing_metric() -> None:
    registry = _registry()
    registry["quality_risk_cost_combined_score_forbidden"] = False
    with pytest.raises(ValueError, match="combined headline"):
        validate_metric_registry(registry)

    registry = _registry()
    registry["metrics"] = registry["metrics"][:-1]
    with pytest.raises(ValueError, match="exact MVP metrics"):
        validate_metric_registry(registry)


def test_blind_preference_keeps_ties_and_excludes_abstentions() -> None:
    rows = [
        {"cluster_id": f"d{index}", "preference": verdict}
        for index, verdict in enumerate(
            ["policy"] * 4
            + ["comparator"] * 2
            + ["tie"] * 2
            + ["abstain"]
        )
    ]
    report = summarize_blind_preferences(rows)
    assert report["eligible_pairs"] == 8
    assert report["netwin"] == pytest.approx(0.25)
    assert report["abstentions"] == 1
    assert report["abstain_rate"] == pytest.approx(1 / 9)


def test_risk_summary_never_turns_not_applicable_into_zero() -> None:
    rows = [
        {
            "cluster_id": "d1",
            "risk_id": "stale",
            "applicable": True,
            "severity": 2,
        },
        {
            "cluster_id": "d2",
            "risk_id": "stale",
            "applicable": True,
            "severity": 0,
        },
        {
            "cluster_id": "d3",
            "risk_id": "stale",
            "applicable": False,
            "severity": None,
        },
    ]
    report = summarize_risk_events(rows, risk_id="stale")
    assert report["eligible_responses"] == 2
    assert report["not_applicable"] == 1
    assert report["material_event_rate"] == pytest.approx(0.5)

    bad = deepcopy(rows)
    bad[-1]["severity"] = 0
    with pytest.raises(ValueError, match="severity=null"):
        summarize_risk_events(bad, risk_id="stale")


def test_cost_summary_uses_observed_nonnegative_values() -> None:
    report = summarize_resource_cost(
        [
            {"cluster_id": "d1", "input_tokens": 100},
            {"cluster_id": "d2", "input_tokens": 140},
            {"cluster_id": "d3", "input_tokens": 120},
        ],
        field="input_tokens",
    )
    assert report["mean"] == 120
    assert report["total"] == 360
    with pytest.raises(ValueError, match="non-negative"):
        summarize_resource_cost(
            [{"cluster_id": "d1", "input_tokens": -1}],
            field="input_tokens",
        )


def test_real_data_audit_detects_privileged_esconv_situation_but_not_split_leakage() -> None:
    report = audit_v1_5_mvp_data(ROOT)
    assert report["status"] == "NEEDS_ZERO_API_REBUILD_BEFORE_MVP_USE"
    assert report["blockers"] == [
        "EXISTING_ESCONV_STATES_EXPOSE_DATASET_SITUATION_TO_PM"
    ]
    assert report["privileged_situation_rows"] == 2831
    assert report["checks"][
        "dialogue_and_full_visible_state_split_isolation"
    ] is True
    assert report["checks"]["model_visible_gold_outcome_fields_absent"] is True
    assert report["checks"]["bank_v2_train_only_and_need_disjoint"] is True


def test_candidate_visible_v2_auxiliary_removes_its_privileged_rows() -> None:
    report = audit_v1_5_mvp_data(
        ROOT,
        auxiliary_dir=ROOT / "data/esconv_auxiliary_v1_5_visible_v2_candidate",
    )
    assert report["partitions"]["auxiliary_train"][
        "summary_exactly_matches_dataset_situation"
    ] == 0
    assert report["partitions"]["auxiliary_calibration"][
        "summary_exactly_matches_dataset_situation"
    ] == 0
    assert report["partitions"]["auxiliary_internal_test"][
        "summary_exactly_matches_dataset_situation"
    ] == 0
    # Until the separately rebuilt external candidate is supplied, the old
    # 2,112-row external partition remains the sole privileged-input blocker.
    assert report["privileged_situation_rows"] == 2112


def test_full_visible_v2_candidate_closes_the_privileged_input_blocker() -> None:
    report = audit_v1_5_mvp_data(
        ROOT,
        auxiliary_dir=ROOT / "data/esconv_auxiliary_v1_5_visible_v2_candidate",
        external_test_dir=ROOT / "data/esconv_test_v1_5_visible_v2_candidate",
    )
    assert report["status"] == "PASS_FOR_MVP_USE"
    assert report["privileged_situation_rows"] == 0
    assert report["blockers"] == []
    assert all(report["checks"].values())
