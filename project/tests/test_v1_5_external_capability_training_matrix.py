from pathlib import Path

from metacom_pm.io import read_json


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data/pm_v1_5_contracts/external_capability_training_matrix_v1.json"


def _matrix() -> dict:
    return read_json(MATRIX)


def test_external_exam_scope_is_not_overclaimed() -> None:
    components = _matrix()["components"]
    assert components["RS"]["ESConv"]["claim"] == "PRIMARY_MEASURABLE"
    assert components["MP"]["ESConv"]["claim"] == "NOT_MEASURABLE"
    assert components["MS"]["ESConv"]["claim"] == "NOT_MEASURABLE"
    assert components["ME"]["ESConv"]["claim"] == "NOT_MEASURABLE"
    assert components["MP"]["EvoEmo"]["claim"] == "MP_PROFILE_ONLY"
    assert components["ME"]["EvoEmo"]["claim"] == "ME_REUSABLE_OUTCOME_MEASURABLE_WHEN_PRESENT"


def test_effect_families_are_all_eligible_and_benefit_enrichment_is_not_gold() -> None:
    matrix = _matrix()
    for component in matrix["components"].values():
        families = component["effect_logic_families"]
        assert len(families) == 8
        hypotheses = [row["benefit_enrichment"] for row in families]
        assert hypotheses.count("HIGH") == 4
        assert hypotheses.count("LOW_OR_NEUTRAL") == 4
    policy = matrix["construction_hypothesis_policy"]
    assert policy["is_eligibility_gold"] is False
    assert policy["is_worth_opening_gold"] is False


def test_effect_splits_are_user_disjoint_and_scale_balanced() -> None:
    splits = _matrix()["splits_per_component"]
    assert splits["ELIGIBILITY_AUDIT"]["states"] == 32
    assert splits["ELIGIBILITY_AUDIT"]["generates_responses"] is False
    assert splits["EFFECT_FIT"]["states"] == 64
    assert splits["FRESH_CONFIRMATION"]["states"] == 32
    assert splits["SEALED_INTERNAL_TEST"]["states"] == 32
    for name, split in splits.items():
        assert split["surface_and_user_overlap_with_other_splits"] == 0
        if name != "ELIGIBILITY_AUDIT":
            assert split["states_per_history_scale"] * 4 == split["states"]


def test_me_and_mp_constructs_are_typed_for_external_transfer() -> None:
    components = _matrix()["components"]
    assert set(components["MP"]["subtypes"]) == {"MP_PREFERENCE", "MP_PROFILE"}
    assert set(components["ME"]["subtypes"]) == {
        "ME_REUSABLE_OUTCOME",
        "ME_CONTEXT_EVENT",
        "ME_UNRESOLVED_EVENT",
    }
    assert "subtype_by_current_goal_four_grid" in components["ME"]["internal_only"]


def test_scale_and_surface_shortcuts_are_explicitly_forbidden() -> None:
    matrix = _matrix()
    assert matrix["history_scales"]["rule"].startswith("scale_label_is_balanced")
    targets = matrix["history_scales"]["source_specific_catalog_targets"]
    assert targets["MP"]["EVOEMO_LIKE_LARGE"] == 10
    assert targets["MS"]["LARGE"] == 22
    assert targets["ME"]["EVOEMO_LIKE_LARGE"] == 68
    assert len(set(targets["RS"].values())) == 1
    checks = set(matrix["anti_shortcut_checks"])
    assert "history_scale_cannot_predict_component_or_hypothesis" in checks
    assert "strategy_family_alone_cannot_predict_RS" in checks
    assert "explicit_invitation_alone_cannot_predict_ME" in checks
