"""Outcome-blind checks that private P2 modes reached formal rank-1 surfaces.

The checks do not create PM labels.  They only catch impossible or malformed
synthetic questions before H1: for example, a requested reusable ME whose
actual rank-1 compiler hint is only a context event, or a routine goodbye that
the formal RS ranker nevertheless treats as having a card.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any, Mapping, Sequence


REALIZATION_PROTOCOL = "pm-v1.5-p2-construction-realization-audit-v1"


def _value(value: Any) -> str:
    return str(value.value) if isinstance(value, Enum) else str(value)


def _expectation(component: str, mode: str) -> tuple[bool, str | None] | None:
    if component == "MP":
        if mode in {"redundant_current_preference", "preference_scope_conflict"}:
            return True, "MP_PREFERENCE"
        if mode in {
            "redundant_current_profile",
            "profile_wrong_entity_or_goal",
        }:
            return True, "MP_PROFILE"
        if mode == "preference_incremental":
            return True, "MP_PREFERENCE"
        if mode == "profile_incremental":
            return True, "MP_PROFILE"
        if mode == "no_suitable_profile_candidate":
            return False, "CANDIDATE_ABSENT"
        return None
    if component == "MS":
        if mode in {
            "prior_distinction_answers_current_goal",
            "prior_outcome_answers_factual_recall",
            "same_topic_wrong_goal",
            "currently_redundant_session_summary",
            "resolved_or_stale_prior_session",
            "prior_session_wrong_owner",
        }:
            return True, "MS_SESSION"
        if mode in {"prior_distinction", "prior_outcome", "unfinished_goal"}:
            return True, "MS_SESSION"
        if mode == "no_suitable_session_candidate":
            return False, "CANDIDATE_ABSENT"
        return None
    if component == "ME":
        if mode in {
            "reusable_action_result_current_goal",
            "reusable_action_mechanism_current_goal",
            "reusable_outcome_wrong_entity_or_goal",
        }:
            return True, "ME_REUSABLE_OUTCOME"
        if mode == "context_event_same_topic_no_reusable_result":
            return True, "ME_CONTEXT_EVENT"
        if mode == "unresolved_event_no_reusable_result":
            return True, "ME_UNRESOLVED_EVENT"
        if mode in {"reusable_action_result", "reusable_action_mechanism"}:
            return True, "ME_REUSABLE_OUTCOME"
        if mode == "no_suitable_event_candidate":
            return False, "CANDIDATE_ABSENT"
        # Context, unresolved, and wrong-goal histories are ME-off whether the
        # formal retriever abstains or surfaces the correctly typed negative.
        # H1 judges that exact candidate; construction audit must not require
        # a negative item to be retrieved merely because it exists in history.
        return None
    if component == "RS":
        if mode in {
            "family_goal_fit_safe_nonredundant",
            "same_family_currently_redundant",
            "same_family_wrong_goal_or_boundary",
        }:
            return True, "RS_ATOMIC_MOVE"
        if mode.endswith("_move_fit"):
            return True, "RS_ATOMIC_MOVE"
        if mode == "strategy_move_already_present":
            # H1 may validly judge the already-executed move off whether the
            # ranker returns a redundant card or abstains before PM.
            return None
        if mode in {
            "routine_closing_or_phatic",
            "explicit_stop",
            "no_bank_scope_match",
        }:
            return False, "CANDIDATE_ABSENT"
        return None
    raise ValueError(f"unknown component: {component}")


def audit_construction_realization(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    blueprints_by_state: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare only mechanically checkable mode promises to actual rank-1."""

    checked = Counter()
    passed = Counter()
    mismatches: list[dict[str, Any]] = []
    for row in candidate_rows:
        state_id = str(row["state_id"])
        blueprint = blueprints_by_state.get(state_id)
        if blueprint is None:
            raise ValueError(f"candidate state absent from private blueprint: {state_id}")
        plans = blueprint["private_construction_intent"]["component_plans"]
        surfaces = row["candidate_surfaces"]
        for component in ("MP", "MS", "ME", "RS"):
            mode = str(plans[component]["construction_mode"])
            expectation = _expectation(component, mode)
            if expectation is None:
                continue
            expected_present, expected_subtype = expectation
            surface = surfaces[component]
            actual_present = bool(surface["candidate_present"])
            actual_subtype = _value(surface["compiler_subtype_hint"])
            key = f"{component}:{mode}"
            checked[key] += 1
            ok = (
                actual_present == expected_present
                and actual_subtype == expected_subtype
            )
            if ok:
                passed[key] += 1
            else:
                mismatches.append(
                    {
                        "state_id": state_id,
                        "component": component,
                        "construction_mode": mode,
                        "expected_candidate_present": expected_present,
                        "actual_candidate_present": actual_present,
                        "expected_compiler_subtype": expected_subtype,
                        "actual_compiler_subtype": actual_subtype,
                    }
                )
    return {
        "protocol": REALIZATION_PROTOCOL,
        "status": "PASS" if not mismatches else "FAIL_DEVELOPMENT_CONSTRUCTION",
        "checked_component_modes": sum(checked.values()),
        "passed_component_modes": sum(passed.values()),
        "mechanically_checkable_rate": (
            sum(passed.values()) / sum(checked.values()) if checked else 0.0
        ),
        "per_mode": {
            key: {"checked": checked[key], "passed": passed[key]}
            for key in sorted(checked)
        },
        "mismatches": mismatches,
        "construction_intent_used_as_gold": False,
        "response_or_outcome_read": False,
    }
