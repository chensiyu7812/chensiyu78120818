from collections import Counter

from metacom_pm.v1_5_v3_effect_blueprint import build_blueprint
from metacom_pm.v1_5_v3_state_realization import (
    audit_v3_states,
    realize_v3_states,
)


def test_v3_visible_state_realization_passes_without_outcomes() -> None:
    blueprint = build_blueprint()
    states = realize_v3_states(blueprint)
    report = audit_v3_states(states, blueprint)
    assert report["status"] == "PASS", report["failures"]
    assert report["states"] == 640
    assert report["unique_users"] == 640
    assert report["responses_generated"] == 0
    assert report["duplicate_visible_state_surfaces"] > 0
    assert all(not row["response_or_outcome_read"] for row in states)


def test_target_catalog_sizes_follow_component_specific_contract() -> None:
    blueprint = build_blueprint()
    states = realize_v3_states(blueprint)
    by_id = {row["blueprint_row_id"]: row for row in blueprint}
    observed = Counter()
    for state in states:
        row = by_id[state["state_id"]]
        component = row["target_component"]
        if component == "MP":
            actual = len(state["user"]["basic_info"])
        elif component in {"MS", "ME"}:
            actual = len(state["user"]["dialog_history"])
        else:
            actual = 80
        assert actual == row["source_catalog_size_target"]
        observed[(component, actual)] += 1
    assert set(size for (component, size) in observed if component == "ME") == {
        5,
        16,
        40,
        68,
    }


def test_memory_effect_counterfactuals_share_visible_state_but_not_user() -> None:
    blueprint = build_blueprint()
    states = realize_v3_states(blueprint)
    by_id = {row["state_id"]: row for row in states}
    groups: dict[str, list[dict]] = {}
    for row in blueprint:
        if row["track"] == "COMPONENT_EFFECT" and row["target_component"] != "RS":
            groups.setdefault(row["counterfactual_group_id"], []).append(row)
    for rows in groups.values():
        left, right = (by_id[row["blueprint_row_id"]] for row in rows)
        assert left["visible_dialogue"] == right["visible_dialogue"]
        assert left["current_user_text"] == right["current_user_text"]
        assert left["user_id"] != right["user_id"]
