from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_mp_subtype_design_audit_v1/"
            "mp_subtype_design_audit.json"
        ).read_text(encoding="utf-8")
    )


def test_mp_audit_corrects_construct_to_subtype_coverage() -> None:
    report = _report()
    assert report["status"] == (
        "KEEP_16_ACTIONS_USE_TYPED_MP_AND_INTERNAL_COMPLEMENTARY_TESTS"
    )
    coverage = report["observed_coverage"]
    assert coverage["historical_mp_ontology"] == (
        "stable profile, preference, or boundary memory"
    )
    assert coverage["internal"]["subtype"] == "MP_PREFERENCE"
    assert coverage["evoemo"]["subtype"] == "MP_PROFILE"
    assert coverage["internal"]["normalized_preference_templates"] == 1
    assert coverage["field_intersection"] == []


def test_typed_mp_preserves_action_space_and_separates_hard_preferences() -> None:
    report = _report()
    decisions = {
        row["option"]: row["decision"]
        for row in report["design_options"]
    }
    assert decisions["typed_MP_catalog_single_MP_action_bit"] == (
        "adopt_for_v1_5"
    )
    assert decisions["add_a_fifth_MP_profile_or_preference_action_bit"] == (
        "defer_to_v2"
    )
    runtime = report["typed_mp_contract"]["runtime"]
    assert runtime["default_optional_mp_injection"] is False
    assert runtime["current_turn_overrides_stored"] is True
    assert runtime["maximum_optional_mp_items_injected"] == 1
    assert report["typed_mp_contract"]["feature_count"] == 6


def test_internal_environment_complements_external_subtype_coverage() -> None:
    report = _report()
    development = report["internal_development_requirement"]
    assert development["new_content_disjoint_users"] == 32
    assert development["MP_PREFERENCE"]["on_benefit"] == 8
    assert development["MP_PREFERENCE"]["nonpositive"] == 8
    assert development["MP_PROFILE"]["on_benefit"] == 8
    assert development["MP_PROFILE"]["nonpositive"] == 8
    roles = report["evaluation_roles"]
    assert "does not validate MP_PREFERENCE" in roles["EvoEmo"]
    assert "MP_PREFERENCE" in roles["internal_controlled_environment"]
    assert "subtype-conditioned" in roles["aggregation_rule"]
