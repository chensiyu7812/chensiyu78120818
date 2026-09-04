from metacom_pm.v1_5_final_construction_realization import _expectation


def test_h1r_construction_modes_bind_actual_candidate_types() -> None:
    expected = {
        ("MP", "preference_scope_conflict"): (True, "MP_PREFERENCE"),
        ("MP", "profile_wrong_entity_or_goal"): (True, "MP_PROFILE"),
        ("MS", "prior_session_wrong_owner"): (True, "MS_SESSION"),
        ("ME", "reusable_outcome_wrong_entity_or_goal"): (
            True,
            "ME_REUSABLE_OUTCOME",
        ),
        ("ME", "unresolved_event_no_reusable_result"): (
            True,
            "ME_UNRESOLVED_EVENT",
        ),
        ("RS", "same_family_currently_redundant"): (True, "RS_ATOMIC_MOVE"),
    }
    for key, value in expected.items():
        assert _expectation(*key) == value
