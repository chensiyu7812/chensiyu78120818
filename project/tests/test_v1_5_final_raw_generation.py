from metacom_pm.v1_5_final_dataset_blueprint import (
    build_h1_v2_blueprint,
    build_primary_blueprint,
)
from metacom_pm.v1_5_final_raw_generation import (
    H1_V2_RAW_GENERATION_PROTOCOL,
    FinalCriticalPriorSessionDraft,
    FinalCurrentExchangeDraft,
    FinalProfileEntryDraft,
    FinalRawStateDraft,
    compile_raw_user_state,
    deterministically_realize_raw_draft,
    deterministically_realize_h1_v2_raw_draft,
    generation_messages_for_blueprint,
    validate_raw_draft_for_blueprint,
    validate_h1_v2_raw_draft,
)


def _row(mode: str = "profile_incremental"):
    row = dict(build_primary_blueprint()[1])
    row["private_construction_intent"] = {
        **row["private_construction_intent"],
        "component_plans": {
            **row["private_construction_intent"]["component_plans"],
            "MP": {
                **row["private_construction_intent"]["component_plans"]["MP"],
                "construction_mode": mode,
            },
            "ME": {
                **row["private_construction_intent"]["component_plans"]["ME"],
                "construction_mode": "wrong_entity_or_goal",
            },
            "RS": {
                **row["private_construction_intent"]["component_plans"]["RS"],
                "construction_mode": "strategy_move_already_present",
            },
        },
    }
    return row


def _draft(distractors: int = 0) -> FinalRawStateDraft:
    sessions = [
        FinalCriticalPriorSessionDraft(
            summary=f"Session {index} concerned workload and a bounded reflection.",
            seeker_text=f"During session {index}, I described a distinct workload concern.",
            supporter_text=f"The supporter reflected concern {index} without adding advice.",
        )
        for index in range(1, 4)
    ]
    return FinalRawStateDraft(
        profile_entries=[
            FinalProfileEntryDraft(
                subtype="MP_PROFILE",
                field_key="work_schedule",
                value="Has a crowded weekly schedule.",
            )
        ],
        critical_prior_sessions=sessions,
        distractor_session_summaries=[
            f"Ordinary unrelated hobby note number {index}."
            for index in range(distractors)
        ],
        recent_dialogue=[
            FinalCurrentExchangeDraft(
                seeker_text="This week has felt unusually crowded.",
                supporter_text="Which part has been taking the most room?",
            )
        ],
        current_user_text="My schedule is making it hard to find a calm starting point.",
    )


def test_raw_generation_prompt_contains_private_constraints_but_no_gold() -> None:
    row = _row()
    messages = generation_messages_for_blueprint(row)
    rendered = repr(messages)
    assert row["logic_family"] in rendered
    assert "profile_incremental" in rendered
    assert "gold" not in messages[-1]["content"].lower()


def test_valid_raw_draft_compiles_to_shared_user_history() -> None:
    row = _row()
    row["prior_session_count_target"] = 3
    draft = _draft()
    validation = validate_raw_draft_for_blueprint(draft=draft, row=row)
    assert validation["status"] == "PASS"
    compiled = compile_raw_user_state(draft=draft, row=row)
    assert len(compiled["user"]["dialog_history"]) == 3
    assert compiled["current_session_index"] == 4
    assert compiled["actual_rank1_candidates"] is None
    assert compiled["construction_intent_is_model_input"] is False


def test_wrong_distractor_count_and_mp_absent_violation_are_rejected() -> None:
    row = _row("no_suitable_profile_candidate")
    row["prior_session_count_target"] = 10
    validation = validate_raw_draft_for_blueprint(draft=_draft(6), row=row)
    assert validation["status"] == "REJECT"
    assert "distractor_session_count" in validation["errors"]
    assert "mp_absent_mode_has_profile_entries" in validation["errors"]


def test_ordinary_lowercase_me_is_not_mistaken_for_component_acronym() -> None:
    row = _row()
    row["prior_session_count_target"] = 3
    draft = _draft()
    draft.current_user_text = "Could you help me put this experience into words?"
    assert validate_raw_draft_for_blueprint(draft=draft, row=row)["status"] == "PASS"


def test_literal_uppercase_component_acronym_is_rejected() -> None:
    row = _row()
    row["prior_session_count_target"] = 3
    draft = _draft()
    draft.current_user_text = "Please activate ME for this response."
    validation = validate_raw_draft_for_blueprint(draft=draft, row=row)
    assert "experimental_meta_language_leak" in validation["errors"]


def test_deterministic_realizer_makes_every_frozen_row_mechanically_valid() -> None:
    for row in build_primary_blueprint():
        realized = deterministically_realize_raw_draft(
            draft=_draft(max(0, int(row["prior_session_count_target"]) - 3)),
            row=row,
        )
        validation = validate_raw_draft_for_blueprint(draft=realized, row=row)
        assert validation["status"] == "PASS", (row["state_id"], validation)


def test_h1_v2_realizer_removes_literal_rs_label_shortcuts() -> None:
    public_surfaces = set()
    forbidden = {
        "only want you to listen",
        "help me put this feeling into words",
        "one focused question",
        "one small optional idea",
        "one small optional suggestion",
    }
    for row in build_h1_v2_blueprint():
        realized = deterministically_realize_h1_v2_raw_draft(
            draft=_draft(max(0, int(row["prior_session_count_target"]) - 3)),
            row=row,
        )
        validation = validate_h1_v2_raw_draft(draft=realized, row=row)
        assert validation["status"] == "PASS", (row["state_id"], validation)
        rs_mode = row["private_construction_intent"]["component_plans"]["RS"][
            "construction_mode"
        ]
        if rs_mode.endswith("_move_fit"):
            lowered = realized.current_user_text.lower()
            assert all(phrase not in lowered for phrase in forbidden)
        public_surfaces.add(
            (
                tuple(
                    (exchange.seeker_text, exchange.supporter_text)
                    for exchange in realized.recent_dialogue
                ),
                realized.current_user_text,
            )
        )
        compiled = compile_raw_user_state(
            draft=realized,
            row=row,
            validation_mode="h1_v2",
            protocol=H1_V2_RAW_GENERATION_PROTOCOL,
        )
        assert compiled["protocol"] == H1_V2_RAW_GENERATION_PROTOCOL
    assert len(public_surfaces) == 256
