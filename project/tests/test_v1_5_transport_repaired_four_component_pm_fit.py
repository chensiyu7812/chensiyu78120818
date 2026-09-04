from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_four_component_fit_fails_registered_development_gates() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1/"
        "training_report.json"
    )
    assert report["status"] == (
        "FOUR_COMPONENT_PRIMARY_LEARNING_GATE_NOT_PASSED"
    )
    assert report["external_results_used"] is False
    assert report["internal_test_used_for_selection"] is False
    assert report["split_identity_excluded"] is True
    assert report["outcome_fields_excluded"] is True
    assert report["opaque_item_ids_excluded"] is True
    assert set(report["head_reports"]) == {"RS", "MP", "MS", "ME"}
    assert all(
        head["grouped_oof"]["gate_passed"] is False
        for head in report["head_reports"].values()
    )


def test_registered_bge_challenger_promotes_no_head() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_transport_repaired_bge_challenger_v1/"
        "bge_challenger_report.json"
    )
    assert report["status"] == "BGE_CHALLENGER_NOT_PROMOTED"
    assert report["promoted_heads"] == []
    assert report["external_results_used"] is False
    assert report["raw_memory_text_not_used_as_logistic_feature"] is True
    assert all(
        head["promoted"] is False
        for head in report["head_reports"].values()
    )


def test_esconv_diagnostic_does_not_promote_failed_rs_head() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_new_rs_head_esconv_frozen_diagnostic_v1/"
        "external_diagnostic_report.json"
    )
    assert report["status"] == (
        "COMPLETE_POSTHOC_DESCRIPTIVE_EXTERNAL_DIAGNOSTIC"
    )
    assert report["formal_external_promotion_allowed"] is False
    assert report["source_model_gate_passed"] is False
    assert (
        report["classification_against_quality_plus_lower_cost_tie_target"][
            "balanced_accuracy"
        ]
        < 0.5
    )


def test_evoemo_diagnostic_is_outcome_blind_and_fail_closed() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_frozen_memory_heads_evoemo_diagnostic_v1/"
        "external_support_report.json"
    )
    assert report["status"] == "COMPLETE_OUTCOME_BLIND_DIAGNOSTIC_ONLY"
    assert report["formal_external_promotion_allowed"] is False
    assert report["quality_or_risk_outcome_used"] is False
    assert report["api_calls"] == 0
    assert report["component_reports"]["MP"][
        "candidate_support_fraction"
    ] == 0.0
    assert report["component_reports"]["ME"]["predicted_on"] == 0
    assert report["component_reports"]["MS"]["predicted_on"] == 28


def test_labeled_subset_coverage_is_not_confused_with_catalog_support() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_component_label_coverage_audit_v1/"
        "coverage_audit.json"
    )
    assert report["status"] == (
        "COMPLETE_OUTCOME_BLIND_COVERAGE_DIAGNOSIS"
    )
    assert report["external_outcomes_used"] is False
    mp = report["component_reports"]["MP"]
    assert mp["external_joint_range_support"]["by_all_internal"][
        "supported_fraction"
    ] == 1.0
    assert mp["external_joint_range_support"]["by_labeled_development"][
        "supported_fraction"
    ] == 0.0
    assert (
        report["component_reports"]["ME"][
            "external_exact_pattern_coverage"
        ]["by_all_internal_fraction"]
        > report["component_reports"]["ME"][
            "external_exact_pattern_coverage"
        ]["by_labeled_development_fraction"]
    )
