from collections import Counter

from metacom_pm.v1_5_final_dataset_blueprint import build_h1r_blueprint
from metacom_pm.v1_5_h1r_realization import (
    audit_h1r_raw_states,
    compile_h1r_raw_state,
    realize_h1r_draft,
    validate_h1r_draft,
)


def _realized():
    rows = build_h1r_blueprint()
    states = []
    for row in rows:
        draft = realize_h1r_draft(row)
        assert validate_h1r_draft(draft=draft, row=row)["status"] == "PASS"
        states.append(compile_h1r_raw_state(draft=draft, row=row))
    return rows, states


def test_h1r_realization_is_complete_unique_and_scale_balanced() -> None:
    rows, states = _realized()
    report = audit_h1r_raw_states(rows=rows, raw_states=states)
    assert report["status"] == "PASS"
    assert report["states"] == report["unique_public_state_surfaces"] == 96


def test_h1r_realization_breaks_history_size_semantic_task_shortcut() -> None:
    rows, _ = _realized()
    counts = Counter(
        (
            row["history_shape"],
            row["private_construction_intent"]["component_plans"]["MS"]["construction_mode"],
        )
        for row in rows
        if row["private_construction_intent"]["intended_bits"]["MS"]
    )
    assert all(
        counts[(scale, mode)] > 0
        for scale in ("SMALL", "EVO_LIKE_LARGE")
        for mode in (
            "prior_distinction_answers_current_goal",
            "prior_outcome_answers_factual_recall",
        )
    )


def test_h1r_realization_never_exposes_blueprint_or_gold() -> None:
    _, states = _realized()
    for state in states:
        assert state["construction_intent_is_model_input"] is False
        assert state["h1_gold"] is None
        assert "private_construction_intent" not in state
