from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_root_cause_audit_separates_confirmed_and_unresolved_causes() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_first_fit_root_cause_audit_v1/"
        "root_cause_audit.json"
    )
    assert report["status"] == (
        "ROOT_CAUSES_SEPARATED_REPAIR_ORDER_IDENTIFIED"
    )
    assert report["formal_model_promotion_allowed"] is False
    assert report["external_outcomes_used_for_new_model_selection"] is False
    assert [row["type"] for row in report["root_cause_assessment"]] == [
        "method implementation",
        "data design and sampling",
        "construct subtype coverage",
        "MS construct and state transport",
        "measurement",
        "model family",
    ]
    assert report["root_cause_assessment"][3]["confidence"] == "high"
    assert report["root_cause_assessment"][4]["confidence"] == "unresolved"
    assert report["root_cause_assessment"][5]["confidence"] == "low"


def test_all_primary_heads_violated_the_capacity_contract() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_first_fit_root_cause_audit_v1/"
        "root_cause_audit.json"
    )
    capacity = report["confirmed_method_failures"]["capacity_contract"][
        "by_component"
    ]
    assert set(capacity) == {"RS", "MP", "MS", "ME"}
    assert all(
        values["capacity_contract_passed"] is False
        for values in capacity.values()
    )
    assert all(
        values["minimum_48_independent_groups_passed"] is False
        for values in capacity.values()
    )
    assert report["data_and_grain"]["available_development_users"] == 36


def test_mp_construct_mismatch_is_material_not_only_scale_shift() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_first_fit_root_cause_audit_v1/"
        "root_cause_audit.json"
    )
    mp = report["confirmed_data_design_failures"]["mp_construct_mismatch"]
    assert mp["normalized_template_count"] == 3
    assert mp["acknowledgement_before_suggestions_rate"] == 1.0
    assert mp["one_gentle_question_rate"] == 1.0
    assert mp["field_name_intersection"] == []
    assert mp["historical_mp_ontology"] == (
        "stable profile, preference, or boundary memory"
    )
    assert "different valid MP subtypes" in mp["corrected_interpretation"]


def test_ms_construct_drift_invalidates_old_training_pairs() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_first_fit_root_cause_audit_v1/"
        "root_cause_audit.json"
    )
    ms = report["confirmed_data_design_failures"][
        "ms_construct_and_current_session_mismatch"
    ]
    old = ms["implementation_findings"]["old_ms_pair_lineage"]
    assert old["pairs"] == 64
    assert old["pairs_with_fallback_item"] == 59
    assert old["pairs_clean_under_narrowed_v1_5_contract"] == 4
    assert "diagnostic only" in ms["interpretation"]


def test_recovery_contract_has_bounded_measurement_gate() -> None:
    contract = _json(
        ROOT
        / "data/pm_v1_5_contracts/"
        "first_fit_root_cause_recovery_v1.json"
    )
    assert contract["status"] == "FROZEN_BEFORE_ANY_NEW_RECOVERY_OUTCOMES"
    assert contract["outcome_blind_primary_features"][
        "feature_count_by_head"
    ] == {"RS": 5, "MP": 6, "MS": 5, "ME": 5}
    measurement = contract["D2_measurement_qualification"]
    assert measurement["states"] == 32
    assert measurement["existing_RS_MP_ME_states"] == 24
    assert measurement["fresh_MS_states"] == 8
    assert measurement["new_response_api_calls"] == 144
    assert measurement["total_human_decisions"] == 104
    assert measurement["gates"] == {
        "states_with_two_of_three_matching_non_uncertain_directions_min": 0.7,
        "shared_pair_exact_inter_reviewer_agreement_min": 0.75,
        "uncertain_fraction_max": 0.1,
    }


def test_recovery_contract_preserves_step1_outcome_blinding() -> None:
    contract = _json(
        ROOT
        / "data/pm_v1_5_contracts/"
        "first_fit_root_cause_recovery_v1.json"
    )
    forbidden = set(
        contract["outcome_blind_primary_features"]["forbidden"]
    )
    assert "generated response text" in forbidden
    assert "quality or risk judgment" in forbidden
    assert "future turn or external outcome" in forbidden
    assert contract["external_integrity"]["same_stack_required"] is True
    assert (
        contract["external_integrity"][
            "external_schema_or_feature_definition_may_change"
        ]
        is False
    )
    assert contract["D3_coverage_rebuild_if_D2_passes"][
        "new_content_disjoint_development_users_min"
    ] == 32
    assert contract["MS_shared_ontology"]["MS_SESSION"].startswith(
        "A bounded summary"
    )
    assert (
        contract["MS_shared_ontology"]["old_pair_policy"]
        .startswith("All 64 old MS contrasts")
    )
