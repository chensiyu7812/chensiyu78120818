from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.v1_5_v5_3_measurement_freeze import (
    MeasurementFreeze,
    build_measurement_freeze,
)


ROOT = Path(__file__).resolve().parents[1]


def test_measurement_freezes_estimands_without_inventing_n() -> None:
    freeze = build_measurement_freeze(ROOT)
    assert freeze.api_calls == 0
    assert freeze.generated_response_or_quality_risk_outcome_read is False
    assert freeze.sampling["numeric_counts_frozen"] is False
    assert freeze.sampling["template_replication_to_reach_n_allowed"] is False
    assert freeze.quality_metric["primary_estimand"] == "cluster_mean_netwin"
    assert freeze.risk_metric["primary_estimand"] == (
        "cluster_mean_any_material_risk_difference"
    )


def test_learning_requires_real_on_off_and_baseline_comparisons() -> None:
    freeze = build_measurement_freeze(ROOT)
    evidence = freeze.step1_learning_evidence
    assert evidence["both_on_and_off_required_per_measurable_head"] is True
    assert evidence["constant_off_is_not_learning"] is True
    assert evidence["must_compare_with_transparent_rule"] is True
    assert evidence["must_compare_with_cost_and_on_rate_matched_random"] is True
    assert evidence["fixed_magic_accuracy_threshold"] is None


def test_json_roundtrip_and_identity_reject_drift() -> None:
    freeze = build_measurement_freeze(ROOT)
    assert MeasurementFreeze.model_validate_json(freeze.model_dump_json()) == freeze
    changed = freeze.model_dump(mode="json")
    changed["quality_metric"]["ties_contribute"] = 1
    with pytest.raises(ValueError, match="freeze_identity"):
        MeasurementFreeze.model_validate(changed)
