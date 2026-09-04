from metacom_pm.v1_5_final_construction_realization import (
    audit_construction_realization,
)


def _surface(present: bool, subtype: str):
    return {
        "candidate_present": present,
        "compiler_subtype_hint": subtype,
    }


def test_construction_realization_checks_only_mechanical_promises() -> None:
    blueprint = {
        "private_construction_intent": {
            "component_plans": {
                "MP": {"construction_mode": "preference_incremental"},
                "MS": {"construction_mode": "same_topic_wrong_goal"},
                "ME": {"construction_mode": "reusable_action_result"},
                "RS": {"construction_mode": "routine_closing_or_phatic"},
            }
        }
    }
    candidate = {
        "state_id": "s1",
        "candidate_surfaces": {
            "MP": _surface(True, "MP_PREFERENCE"),
            "MS": _surface(True, "MS_SESSION"),
            "ME": _surface(True, "ME_REUSABLE_OUTCOME"),
            "RS": _surface(False, "CANDIDATE_ABSENT"),
        },
    }
    report = audit_construction_realization(
        candidate_rows=[candidate], blueprints_by_state={"s1": blueprint}
    )
    assert report["status"] == "PASS"
    assert report["checked_component_modes"] == 3
    assert report["construction_intent_used_as_gold"] is False


def test_construction_realization_fails_wrong_me_subtype() -> None:
    blueprint = {
        "private_construction_intent": {
            "component_plans": {
                "MP": {"construction_mode": "wrong_entity"},
                "MS": {"construction_mode": "resolved_prior_issue"},
                "ME": {"construction_mode": "reusable_action_mechanism"},
                "RS": {"construction_mode": "strategy_move_already_present"},
            }
        }
    }
    candidate = {
        "state_id": "s2",
        "candidate_surfaces": {
            "MP": _surface(True, "MP_PROFILE"),
            "MS": _surface(True, "MS_SESSION"),
            "ME": _surface(True, "ME_CONTEXT_EVENT"),
            "RS": _surface(True, "RS_ATOMIC_MOVE"),
        },
    }
    report = audit_construction_realization(
        candidate_rows=[candidate], blueprints_by_state={"s2": blueprint}
    )
    assert report["status"] == "FAIL_DEVELOPMENT_CONSTRUCTION"
    assert report["mismatches"][0]["component"] == "ME"
