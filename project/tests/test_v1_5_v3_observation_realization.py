from metacom_pm.v1_5_v3_observation_orthogonal import build_blueprint
from metacom_pm.v1_5_v3_observation_realization import audit_states, realize_states


def test_visible_realization_passes_pre_rank1_static_gate() -> None:
    blueprint = build_blueprint()
    states = realize_states(blueprint)
    report = audit_states(states, blueprint)
    assert report["status"] == "PASS", report["failures"]
    assert report["states"] == 96
    assert report["unique_visible_state_surfaces"] == 96
    assert report["source_catalog_size_mismatches"] == []
    assert report["me_compiler_subtype_mismatches"] == []


def test_visible_surface_never_contains_private_factor_fields() -> None:
    states = realize_states(build_blueprint())
    for state in states:
        assert "private_factor_plan" not in state
        assert "private_composed_eligibility" not in state
        assert state["construction_intent_present_in_model_input"] is False


def test_v2_uses_honest_mp_preference_owner_negative_and_card_level_rs_goal() -> None:
    blueprint = build_blueprint()
    states = {
        state["state_id"]: state
        for state in realize_states(blueprint, surface_version="v2")
    }
    mp_preference_owner_negative = [
        row
        for row in blueprint
        if row["target_component"] == "MP"
        and row["candidate_subtype_target"] == "MP_PREFERENCE"
        and not row["private_factor_plan"]["owner_time_entity_valid"]["private_target"]
    ]
    assert mp_preference_owner_negative
    for row in mp_preference_owner_negative:
        text = states[row["blueprint_row_id"]]["current_user_text"].lower()
        assert any(word in text for word in ("superseded", "withdrawn", "revoked", "stale", "former", "replaced"))
        assert "friend, not me" not in text

    rs_goal_positive = [
        row
        for row in blueprint
        if row["target_component"] == "RS"
        and row["private_factor_plan"]["goal_function_fit"]["private_target"]
    ]
    assert rs_goal_positive
    for row in rs_goal_positive:
        text = states[row["blueprint_row_id"]]["current_user_text"].lower()
        function = row["required_candidate_function"]
        if function == "one_reversible_suggestion":
            assert "trusted person" in text and "social support" in text
        elif function == "evidence_grounded_reflection":
            assert "name the feeling as anxiety" in text
        elif function == "tentative_paraphrase_check":
            assert "most relevant facts" in text and "tentative paraphrase" in text
