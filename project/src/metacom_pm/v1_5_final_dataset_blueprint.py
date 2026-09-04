"""Deterministic private construction matrix for the final 256-state P2 pool.

This matrix is an experimental-design instrument, never a PM input or gold
label.  Actual histories are generated later, actual rank-1 resources are
retrieved, and H1 independently adjudicates the exact candidates.  The intent
matrix only prevents post-label cherry-picking and action/family confounding.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Literal, Sequence

from .io import stable_hex
from .v1_5b_policy_runtime import COMPONENTS, compile_component_bits


BLUEPRINT_PROTOCOL = "pm-v1.5-p2-private-construction-blueprint-v8"
H1_V2_BLUEPRINT_PROTOCOL = "pm-v1.5-p2-private-construction-blueprint-v9-h1-v2"
H1R_BLUEPRINT_PROTOCOL = "pm-v1.5-p2r-private-orthogonal-blueprint-v1"
Split = Literal["FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"]

SPLIT_BLOCK_MASKS: dict[Split, tuple[int, ...]] = {
    # Every mask set flips every component equally often.  XOR with all 16
    # family indices is a bijection, so every block contains every action.
    # This balanced half-cube also contains two Hamming-distance-1 edges for
    # every component, enabling genuine within-family counterfactual checks.
    "FIT": (0, 1, 2, 5, 10, 13, 14, 15),
    "FRESH_CONFIRMATION": (0, 15, 5, 10),
    "SEALED_INTERNAL_TEST": (3, 12, 5, 10),
}

FIT_LOGIC_FAMILIES = (
    # Surface/challenge families, never component labels.  Every one is
    # deliberately compatible with every 4-bit construction.
    "direct_first_person_wording",
    "indirect_first_person_wording",
    "short_current_turn",
    "long_current_turn",
    "single_issue_focus",
    "two_related_issues",
    "emotion_named_explicitly",
    "emotion_implied_only",
    "time_marker_explicit",
    "time_marker_implicit",
    "third_party_pronoun_reference",
    "self_correction_in_current_turn",
    "cause_expressed_as_uncertain",
    "goal_expressed_as_uncertain",
    "high_lexical_overlap_with_history",
    "low_lexical_overlap_paraphrase",
)

CONFIRMATION_LOGIC_FAMILIES = (
    "ellipsis_after_visible_context",
    "contrastive_but_turn",
    "current_turn_as_question",
    "current_turn_as_statement",
    "mixed_emotion_wording",
    "metaphorical_low_overlap_wording",
    "negation_scope_challenge",
    "pronoun_resolution_challenge",
    "then_versus_now_wording",
    "quoted_third_party_words",
    "explicit_fact_correction",
    "implicit_fact_correction",
    "keyword_reuse_with_changed_meaning",
    "synonym_shift_without_keyword_reuse",
    "two_plausible_current_goals",
    "one_clear_current_goal",
)

SEALED_LOGIC_FAMILIES = (
    "dense_distractor_keyword_overlap",
    "sparse_target_keyword_overlap",
    "multiple_similar_people",
    "multiple_historical_timepoints",
    "cross_topic_same_mechanism_surface",
    "same_topic_different_goal_surface",
    "old_fact_and_current_update_surface",
    "old_request_and_current_revision_surface",
    "current_repeats_historical_wording",
    "current_adds_new_constraint_surface",
    "candidate_function_collision_surface",
    "two_candidate_functions_surface",
    "multi_component_compatible_surface",
    "multi_component_tension_surface",
    "very_low_retrieval_margin_surface",
    "mixed_out_of_support_cues_surface",
)

LOGIC_FAMILIES: dict[Split, tuple[str, ...]] = {
    "FIT": FIT_LOGIC_FAMILIES,
    "FRESH_CONFIRMATION": CONFIRMATION_LOGIC_FAMILIES,
    "SEALED_INTERNAL_TEST": SEALED_LOGIC_FAMILIES,
}

TOPIC_FAMILIES = (
    "workload_deadline",
    "relationship_transition",
    "relocation_loneliness",
    "grief_reminder",
    "shift_sleep",
    "academic_pressure",
    "caregiving_burden",
    "social_anxiety",
    "habit_recovery",
    "team_dependency",
    "family_conflict",
    "creative_block",
    "financial_uncertainty",
    "friendship_rupture",
    "decision_ambivalence",
    "routine_disruption",
)

_POSITIVE_MODES = {
    "MP": ("preference_incremental", "profile_incremental"),
    "MS": ("prior_distinction", "prior_outcome", "unfinished_goal"),
    "ME": ("reusable_action_result", "reusable_action_mechanism"),
    "RS": (
        "listen_move_fit",
        "reflection_move_fit",
        "clarification_move_fit",
        "one_option_move_fit",
    ),
}
_NEGATIVE_MODES = {
    "MP": (
        "redundant_current_fact",
        "scope_conflict",
        "wrong_entity",
        "no_suitable_profile_candidate",
    ),
    "MS": (
        "resolved_prior_issue",
        "same_topic_wrong_goal",
        "redundant_current_summary",
        "no_suitable_session_candidate",
    ),
    "ME": (
        "context_event_only",
        "unresolved_event",
        "wrong_entity_or_goal",
        "no_suitable_event_candidate",
    ),
    "RS": (
        "routine_closing_or_phatic",
        "explicit_stop",
        "strategy_move_already_present",
        "no_bank_scope_match",
    ),
}


def _bits(mask: int) -> dict[str, bool]:
    return {
        component: bool(mask & (1 << index))
        for index, component in enumerate(COMPONENTS)
    }


def _candidate_plan(
    *,
    component: str,
    intended_on: bool,
    intended_bits: dict[str, bool],
    family_index: int,
    block_index: int,
) -> dict[str, Any]:
    if intended_on:
        modes = _POSITIVE_MODES[component]
    else:
        modes = _NEGATIVE_MODES[component]
        # MS and ME are compiled from the same prior sessions.  A relevant ME
        # episode necessarily leaves an MS summary, and a relevant MS session
        # necessarily leaves an ME context episode.  The off sibling must
        # therefore be a wrong-function/redundant candidate, not a physically
        # absent catalog.  Likewise, stop/bye cannot coexist with a useful
        # memory opportunity in the same current turn.
        if component in {"MS", "ME"} and not (
            intended_bits["MS"] or intended_bits["ME"]
        ):
            # MS summaries and ME seeker episodes are compiled from the same
            # sessions.  A truly empty topical catalog for one is necessarily
            # empty for the other.  Make the fully absent construction a
            # shared joint mode; otherwise use a typed wrong-function mode.
            joint_absent = int(
                stable_hex(
                    BLUEPRINT_PROTOCOL,
                    "joint-ms-me-absence",
                    family_index,
                    block_index,
                    n=8,
                ),
                16,
            ) % 4 == 0
            if joint_absent:
                mode = (
                    "no_suitable_session_candidate"
                    if component == "MS"
                    else "no_suitable_event_candidate"
                )
                return {
                    "component": component,
                    "construction_mode": mode,
                    "intended_opportunity_for_construction_only": intended_on,
                    "must_be_retrieved_before_labeling": True,
                    "may_be_overruled_by_h1": True,
                }
            modes = tuple(mode for mode in modes if not mode.startswith("no_suitable_"))
        elif component == "MS" and intended_bits["ME"]:
            modes = tuple(
                mode for mode in modes if mode != "no_suitable_session_candidate"
            )
        elif component == "ME" and intended_bits["MS"]:
            modes = tuple(
                mode for mode in modes if mode != "no_suitable_event_candidate"
            )
        elif component == "RS" and any(
            intended_bits[name] for name in ("MP", "MS", "ME")
        ):
            modes = ("strategy_move_already_present", "no_bank_scope_match")
    # Do not derive the mode from an index parity that also determines the
    # action bit.  The first blueprint accidentally made every FIT MP-positive
    # row a profile row, leaving preference routing absent from training.
    mode_index = int(
        stable_hex(
            BLUEPRINT_PROTOCOL,
            "construction-mode",
            component,
            intended_on,
            family_index,
            block_index,
            n=8,
        ),
        16,
    ) % len(modes)
    if (
        component == "RS"
        and not intended_on
        and not any(intended_bits[name] for name in ("MP", "MS", "ME"))
    ):
        # M0+R0 occurs once per block. Cycling by block guarantees all four
        # genuine full-off mechanisms are represented rather than relying on
        # a small hash sample to happen to cover them.
        mode_index = block_index % len(modes)
    mode = modes[mode_index]
    return {
        "component": component,
        "construction_mode": mode,
        "intended_opportunity_for_construction_only": intended_on,
        "must_be_retrieved_before_labeling": True,
        "may_be_overruled_by_h1": True,
    }


def build_primary_blueprint() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_index = 0
    for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"):
        typed_split: Split = split  # type: ignore[assignment]
        families = LOGIC_FAMILIES[typed_split]
        for block_index, xor_mask in enumerate(SPLIT_BLOCK_MASKS[typed_split]):
            for family_index, logic_family in enumerate(families):
                intended_mask = family_index ^ xor_mask
                intended_bits = _bits(intended_mask)
                state_seed = stable_hex(
                    BLUEPRINT_PROTOCOL, split, block_index, family_index, n=20
                )
                # Every block contains every action exactly once.  Assigning
                # catalog scale by block gives every action the same scale
                # distribution and prevents history size from becoming a bit
                # shortcut.  It also limits expensive large-history states to
                # one quarter of the complete pool.
                history_shape = ("SMALL", "MEDIUM", "EVO_LIKE_LARGE")[
                    block_index % 3
                ]
                prior_sessions = {
                    "SMALL": 3,
                    "MEDIUM": 10,
                    "EVO_LIKE_LARGE": 34,
                }[history_shape]
                rows.append(
                    {
                        "protocol": BLUEPRINT_PROTOCOL,
                        "blueprint_index": global_index,
                        "state_id": f"state_{state_seed}",
                        "user_id": f"user_{state_seed}",
                        "group_id": f"group_{state_seed}",
                        "split": split,
                        "logic_family": logic_family,
                        "topic_family": TOPIC_FAMILIES[
                            (family_index * 5 + block_index * 3) % len(TOPIC_FAMILIES)
                        ],
                        "history_shape": history_shape,
                        "prior_session_count_target": prior_sessions,
                        "private_construction_intent": {
                            "intended_action": compile_component_bits(intended_bits),
                            "intended_bits": intended_bits,
                            "component_plans": {
                                component: _candidate_plan(
                                    component=component,
                                    intended_on=intended_bits[component],
                                    intended_bits=intended_bits,
                                    family_index=family_index,
                                    block_index=block_index,
                                )
                                for component in COMPONENTS
                            },
                        },
                        "actual_rank1_candidates": None,
                        "h1_gold": None,
                        "private_counterfactual_peers": [],
                        "construction_intent_is_gold": False,
                        "construction_intent_is_model_input": False,
                    }
                )
                global_index += 1

    fit_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "FIT":
            fit_by_family[str(row["logic_family"])].append(row)
    for family_rows in fit_by_family.values():
        for row in family_rows:
            own_bits = row["private_construction_intent"]["intended_bits"]
            peers = []
            for other in family_rows:
                if other is row:
                    continue
                other_bits = other["private_construction_intent"]["intended_bits"]
                differences = [
                    component
                    for component in COMPONENTS
                    if own_bits[component] != other_bits[component]
                ]
                if len(differences) == 1:
                    peers.append(
                        {
                            "state_id": other["state_id"],
                            "flipped_component": differences[0],
                        }
                    )
            row["private_counterfactual_peers"] = sorted(
                peers, key=lambda value: (value["flipped_component"], value["state_id"])
            )
    audit_primary_blueprint(rows)
    return rows


def _h1_v2_candidate_plan(
    *,
    component: str,
    intended_on: bool,
    intended_bits: dict[str, bool],
    family_index: int,
    block_index: int,
) -> dict[str, Any]:
    """Cover candidate-present negatives instead of letting presence be gold.

    The private bit remains a construction target, never a label.  Compared
    with v8, RS-off states with an active support problem usually surface a
    redundant card for H1 to reject; only a bounded subset is a true no-card
    or state-level hard-off case.
    """

    if intended_on:
        modes = _POSITIVE_MODES[component]
    else:
        modes = _NEGATIVE_MODES[component]
        if component in {"MS", "ME"} and not (
            intended_bits["MS"] or intended_bits["ME"]
        ):
            joint_absent = int(
                stable_hex(
                    H1_V2_BLUEPRINT_PROTOCOL,
                    "joint-ms-me-absence",
                    family_index,
                    block_index,
                    n=8,
                ),
                16,
            ) % 4 == 0
            if joint_absent:
                mode = (
                    "no_suitable_session_candidate"
                    if component == "MS"
                    else "no_suitable_event_candidate"
                )
                return {
                    "component": component,
                    "construction_mode": mode,
                    "intended_opportunity_for_construction_only": intended_on,
                    "must_be_retrieved_before_labeling": True,
                    "may_be_overruled_by_h1": True,
                }
            modes = tuple(
                mode for mode in modes if not mode.startswith("no_suitable_")
            )
        elif component == "MS" and intended_bits["ME"]:
            modes = tuple(
                mode for mode in modes if mode != "no_suitable_session_candidate"
            )
        elif component == "ME" and intended_bits["MS"]:
            modes = tuple(
                mode for mode in modes if mode != "no_suitable_event_candidate"
            )
        elif component == "RS" and any(
            intended_bits[name] for name in ("MP", "MS", "ME")
        ):
            # Three quarters are candidate-present redundant/mismatched
            # controls; one quarter is a genuine no-bank factual request.
            modes = (
                "strategy_move_already_present",
                "strategy_move_already_present",
                "strategy_move_already_present",
                "no_bank_scope_match",
            )

    if (
        component == "RS"
        and not intended_on
        and not any(intended_bits[name] for name in ("MP", "MS", "ME"))
    ):
        mode = _NEGATIVE_MODES["RS"][block_index % 4]
    else:
        mode = modes[
            int(
                stable_hex(
                    H1_V2_BLUEPRINT_PROTOCOL,
                    "construction-mode",
                    component,
                    intended_on,
                    family_index,
                    block_index,
                    n=8,
                ),
                16,
            )
            % len(modes)
        ]
    return {
        "component": component,
        "construction_mode": mode,
        "intended_opportunity_for_construction_only": intended_on,
        "must_be_retrieved_before_labeling": True,
        "may_be_overruled_by_h1": True,
    }


def build_h1_v2_blueprint() -> list[dict[str, Any]]:
    """Build a new 256-state matrix after invalidating the first H1 packet."""

    rows: list[dict[str, Any]] = []
    global_index = 0
    for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"):
        typed_split: Split = split  # type: ignore[assignment]
        families = LOGIC_FAMILIES[typed_split]
        for block_index, xor_mask in enumerate(SPLIT_BLOCK_MASKS[typed_split]):
            for family_index, logic_family in enumerate(families):
                intended_mask = family_index ^ xor_mask
                intended_bits = _bits(intended_mask)
                state_seed = stable_hex(
                    H1_V2_BLUEPRINT_PROTOCOL,
                    split,
                    block_index,
                    family_index,
                    n=20,
                )
                history_shape = ("SMALL", "MEDIUM", "EVO_LIKE_LARGE")[
                    block_index % 3
                ]
                prior_sessions = {
                    "SMALL": 3,
                    "MEDIUM": 10,
                    "EVO_LIKE_LARGE": 34,
                }[history_shape]
                rows.append(
                    {
                        "protocol": H1_V2_BLUEPRINT_PROTOCOL,
                        "blueprint_index": global_index,
                        "state_id": f"state_{state_seed}",
                        "user_id": f"user_{state_seed}",
                        "group_id": f"group_{state_seed}",
                        "split": split,
                        "logic_family": logic_family,
                        "topic_family": TOPIC_FAMILIES[
                            (family_index * 7 + block_index * 5)
                            % len(TOPIC_FAMILIES)
                        ],
                        "history_shape": history_shape,
                        "prior_session_count_target": prior_sessions,
                        "private_construction_intent": {
                            "intended_action": compile_component_bits(intended_bits),
                            "intended_bits": intended_bits,
                            "component_plans": {
                                component: _h1_v2_candidate_plan(
                                    component=component,
                                    intended_on=intended_bits[component],
                                    intended_bits=intended_bits,
                                    family_index=family_index,
                                    block_index=block_index,
                                )
                                for component in COMPONENTS
                            },
                        },
                        "actual_rank1_candidates": None,
                        "h1_gold": None,
                        "private_counterfactual_peers": [],
                        "construction_intent_is_gold": False,
                        "construction_intent_is_model_input": False,
                    }
                )
                global_index += 1

    fit_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "FIT":
            fit_by_family[str(row["logic_family"])].append(row)
    for family_rows in fit_by_family.values():
        for row in family_rows:
            own_bits = row["private_construction_intent"]["intended_bits"]
            peers = []
            for other in family_rows:
                if other is row:
                    continue
                other_bits = other["private_construction_intent"]["intended_bits"]
                differences = [
                    component
                    for component in COMPONENTS
                    if own_bits[component] != other_bits[component]
                ]
                if len(differences) == 1:
                    peers.append(
                        {
                            "state_id": other["state_id"],
                            "flipped_component": differences[0],
                        }
                    )
            row["private_counterfactual_peers"] = sorted(
                peers,
                key=lambda value: (
                    value["flipped_component"],
                    value["state_id"],
                ),
            )
    audit_primary_blueprint(rows)
    return rows


H1R_LOGIC_FAMILIES: dict[Split, tuple[str, ...]] = {
    "FIT": (
        "fit_direct_explicit",
        "fit_indirect_paraphrase",
        "fit_low_overlap_synonym",
        "fit_negation_boundary",
        "fit_current_redundancy",
        "fit_wrong_entity",
        "fit_same_topic_wrong_goal",
        "fit_valid_increment",
    ),
    "FRESH_CONFIRMATION": (
        "confirm_ellipsis_resolution",
        "confirm_metaphorical_paraphrase",
        "confirm_temporal_update",
        "confirm_pronoun_scope",
        "confirm_function_collision",
        "confirm_boundary_revision",
        "confirm_low_overlap_valid",
        "confirm_high_overlap_invalid",
    ),
    "SEALED_INTERNAL_TEST": (
        "sealed_multiple_people",
        "sealed_multiple_timepoints",
        "sealed_current_correction",
        "sealed_prior_request_stale",
        "sealed_cross_topic_mechanism",
        "sealed_same_topic_wrong_function",
        "sealed_multi_resource_tension",
        "sealed_multi_resource_compatible",
    ),
}

H1R_RS_FAMILIES = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Providing Suggestions",
)


def _h1r_component_plan(
    *, component: str, intended_on: bool, mask: int, repeat: int
) -> dict[str, Any]:
    """Return an orthogonal construction challenge, never a gold label."""

    if component == "MP":
        subtype = "MP_PREFERENCE" if ((mask // 2 + repeat) % 2 == 0) else "MP_PROFILE"
        if intended_on:
            mode = (
                "preference_incremental"
                if subtype == "MP_PREFERENCE"
                else "profile_incremental"
            )
        elif subtype == "MP_PREFERENCE":
            mode = (
                "redundant_current_preference"
                if (mask // 4 + repeat) % 2 == 0
                else "preference_scope_conflict"
            )
        else:
            mode = (
                "redundant_current_profile"
                if (mask // 4 + repeat) % 2 == 0
                else "profile_wrong_entity_or_goal"
            )
        extra = {"candidate_subtype_target": subtype}
    elif component == "MS":
        if intended_on:
            # Both semantic tasks occur at both history sizes.  Binding one
            # task to SMALL and the other to EVO_LIKE_LARGE would make catalog
            # size an unintended shortcut.
            mode = (
                "prior_distinction_answers_current_goal"
                if ((mask + repeat) % 2 == 0)
                else "prior_outcome_answers_factual_recall"
            )
        else:
            mode = (
                "same_topic_wrong_goal",
                "currently_redundant_session_summary",
                "resolved_or_stale_prior_session",
                "prior_session_wrong_owner",
            )[(mask // 2 + repeat) % 4]
        extra = {"candidate_subtype_target": "MS_SESSION"}
    elif component == "ME":
        invitation_visible = bool(mask & (1 << 1))
        if intended_on:
            mode = (
                "reusable_action_result_current_goal"
                if ((mask + repeat) % 2 == 0)
                else "reusable_action_mechanism_current_goal"
            )
            subtype = "ME_REUSABLE_OUTCOME"
        else:
            negative_index = (mask + repeat) % 3
            if negative_index == 0:
                mode = "reusable_outcome_wrong_entity_or_goal"
                subtype = "ME_REUSABLE_OUTCOME"
            elif negative_index == 1:
                mode = "context_event_same_topic_no_reusable_result"
                subtype = "ME_CONTEXT_EVENT"
            else:
                mode = "unresolved_event_no_reusable_result"
                subtype = "ME_UNRESOLVED_EVENT"
        extra = {
            "candidate_subtype_target": subtype,
            "past_help_invitation_visible": invitation_visible,
        }
    elif component == "RS":
        family = H1R_RS_FAMILIES[(mask + repeat) % len(H1R_RS_FAMILIES)]
        if intended_on:
            mode = "family_goal_fit_safe_nonredundant"
        else:
            mode = (
                "same_family_currently_redundant"
                if (mask // 2 + repeat) % 2 == 0
                else "same_family_wrong_goal_or_boundary"
            )
        extra = {"strategy_family_target": family}
    else:
        raise ValueError(f"unknown component: {component}")
    return {
        "component": component,
        "construction_mode": mode,
        "intended_opportunity_for_construction_only": intended_on,
        "must_be_retrieved_before_labeling": True,
        "may_be_overruled_by_h1": True,
        **extra,
    }


def build_h1r_blueprint() -> list[dict[str, Any]]:
    """Build the single 96-state P2R matrix before any H1R labels exist."""

    rows: list[dict[str, Any]] = []
    index = 0
    for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"):
        typed_split: Split = split  # type: ignore[assignment]
        families = H1R_LOGIC_FAMILIES[typed_split]
        split_index = ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST").index(split)
        for repeat in range(2):
            for mask in range(16):
                bits = _bits(mask)
                pair_index = min(mask, mask ^ 15)
                logic_family = families[pair_index]
                seed = stable_hex(H1R_BLUEPRINT_PROTOCOL, split, repeat, mask, n=20)
                history_shape = "SMALL" if repeat == 0 else "EVO_LIKE_LARGE"
                rows.append(
                    {
                        "protocol": H1R_BLUEPRINT_PROTOCOL,
                        "blueprint_index": index,
                        "state_id": f"h1r_state_{seed}",
                        "user_id": f"h1r_user_{seed}",
                        "group_id": f"h1r_group_{seed}",
                        "split": split,
                        "logic_family": logic_family,
                        # Each topic receives both members of one complement
                        # pair in three split/scale combinations.  Therefore
                        # every topic is exactly 3 ON / 3 OFF for every bit;
                        # topic identity cannot solve any routing head.
                        "topic_family": TOPIC_FAMILIES[
                            pair_index * 2 + ((repeat + split_index) % 2)
                        ],
                        "history_shape": history_shape,
                        "prior_session_count_target": 4 if repeat == 0 else 34,
                        "private_construction_intent": {
                            "intended_action": compile_component_bits(bits),
                            "intended_bits": bits,
                            "component_plans": {
                                component: _h1r_component_plan(
                                    component=component,
                                    intended_on=bits[component],
                                    mask=mask,
                                    repeat=repeat,
                                )
                                for component in COMPONENTS
                            },
                        },
                        "actual_rank1_candidates": None,
                        "h1_gold": None,
                        "construction_intent_is_gold": False,
                        "construction_intent_is_model_input": False,
                        "external_content_or_outcome_read": False,
                    }
                )
                index += 1
    audit_h1r_blueprint(rows)
    return rows


def audit_h1r_blueprint(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 96:
        raise ValueError("H1R must contain exactly 96 states")
    if len({row["state_id"] for row in rows}) != 96:
        raise ValueError("H1R state IDs must be unique")
    if len({row["user_id"] for row in rows}) != 96:
        raise ValueError("H1R users must be unique")
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
        if row["construction_intent_is_gold"] or row["construction_intent_is_model_input"]:
            raise ValueError("H1R construction intent cannot be gold/model input")
        if row["external_content_or_outcome_read"]:
            raise ValueError("H1R cannot read external content or outcomes")
    report: dict[str, Any] = {}
    for split, split_rows in by_split.items():
        if len(split_rows) != 32:
            raise ValueError(f"{split} must contain 32 states")
        actions = Counter(
            row["private_construction_intent"]["intended_action"]
            for row in split_rows
        )
        if len(actions) != 16 or set(actions.values()) != {2}:
            raise ValueError(f"{split} must cover each construction action twice")
        if len({row["logic_family"] for row in split_rows}) != 8:
            raise ValueError(f"{split} must contain eight logic families")
        for family in {row["logic_family"] for row in split_rows}:
            family_rows = [row for row in split_rows if row["logic_family"] == family]
            if len(family_rows) != 4:
                raise ValueError(f"{split}/{family} must contain four states")
            for component in COMPONENTS:
                balance = Counter(
                    bool(row["private_construction_intent"]["intended_bits"][component])
                    for row in family_rows
                )
                if balance != Counter({False: 2, True: 2}):
                    raise ValueError(f"{split}/{family}/{component} is confounded")
        for component in COMPONENTS:
            balance = Counter(
                bool(row["private_construction_intent"]["intended_bits"][component])
                for row in split_rows
            )
            if balance != Counter({False: 16, True: 16}):
                raise ValueError(f"{split}/{component} bit balance failed")
        for topic in TOPIC_FAMILIES:
            topic_rows = [row for row in rows if row["topic_family"] == topic]
            if topic_rows and len(topic_rows) != 6:
                raise ValueError(f"H1R topic {topic} must contain six states")
            for component in COMPONENTS:
                if topic_rows:
                    topic_balance = Counter(
                        bool(row["private_construction_intent"]["intended_bits"][component])
                        for row in topic_rows
                    )
                    if topic_balance != Counter({False: 3, True: 3}):
                        raise ValueError(f"H1R topic shortcut for {topic}/{component}")
        mp_grid = Counter(
            (
                row["private_construction_intent"]["component_plans"]["MP"]["candidate_subtype_target"],
                bool(row["private_construction_intent"]["intended_bits"]["MP"]),
            )
            for row in split_rows
        )
        if any(mp_grid[(subtype, bit)] < 4 for subtype in ("MP_PREFERENCE", "MP_PROFILE") for bit in (False, True)):
            raise ValueError(f"{split} MP subtype/bit grid is sparse")
        me_grid = Counter(
            (
                bool(row["private_construction_intent"]["component_plans"]["ME"]["past_help_invitation_visible"]),
                bool(row["private_construction_intent"]["intended_bits"]["ME"]),
            )
            for row in split_rows
        )
        if set(me_grid.values()) != {8} or len(me_grid) != 4:
            raise ValueError(f"{split} ME invitation/validity grid failed")
        rs_grid = Counter(
            (
                row["private_construction_intent"]["component_plans"]["RS"]["strategy_family_target"],
                bool(row["private_construction_intent"]["intended_bits"]["RS"]),
            )
            for row in split_rows
        )
        if any(rs_grid[(family, bit)] < 4 for family in H1R_RS_FAMILIES for bit in (False, True)):
            raise ValueError(f"{split} RS family/bit grid is sparse")
        report[split] = {
            "states": len(split_rows),
            "actions": dict(sorted(actions.items())),
            "logic_families": 8,
            "per_component_bit_counts": {
                component: {"off": 16, "on": 16} for component in COMPONENTS
            },
            "mp_subtype_bit_grid": {str(key): value for key, value in sorted(mp_grid.items())},
            "me_invitation_validity_grid": {str(key): value for key, value in sorted(me_grid.items())},
            "rs_family_bit_grid": {str(key): value for key, value in sorted(rs_grid.items())},
        }
    split_family_sets = [set(H1R_LOGIC_FAMILIES[split]) for split in H1R_LOGIC_FAMILIES]
    if any(split_family_sets[i] & split_family_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("H1R logic families must be disjoint across splits")
    return {
        "protocol": "pm-v1.5-p2r-private-orthogonal-blueprint-audit-v1",
        "status": "PASS",
        "states": 96,
        "unique_users": 96,
        "construction_intent_is_gold": False,
        "construction_intent_is_model_input": False,
        "external_content_or_outcome_read": False,
        "by_split": report,
    }


def audit_primary_blueprint(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected_sizes = {
        "FIT": 128,
        "FRESH_CONFIRMATION": 64,
        "SEALED_INTERNAL_TEST": 64,
    }
    if len(rows) != 256:
        raise ValueError(f"expected 256 primary rows, got {len(rows)}")
    ids = [str(row["state_id"]) for row in rows]
    users = [str(row["user_id"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(users)) != len(users):
        raise ValueError("every target state must have a unique state and user")

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
        if row["construction_intent_is_gold"]:
            raise ValueError("construction intent cannot be gold")
        if row["construction_intent_is_model_input"]:
            raise ValueError("construction intent cannot be a model input")
    for split, expected in expected_sizes.items():
        split_rows = by_split[split]
        if len(split_rows) != expected:
            raise ValueError(f"{split} size mismatch")
        per_action = Counter(
            row["private_construction_intent"]["intended_action"]
            for row in split_rows
        )
        expected_per_action = expected // 16
        if set(per_action.values()) != {expected_per_action} or len(per_action) != 16:
            raise ValueError(f"{split} action construction is not balanced")
        scale_signature_by_action = {
            action: Counter(
                str(row["history_shape"])
                for row in split_rows
                if row["private_construction_intent"]["intended_action"] == action
            )
            for action in per_action
        }
        scale_signatures = {
            tuple(sorted(counts.items()))
            for counts in scale_signature_by_action.values()
        }
        if len(scale_signatures) != 1:
            raise ValueError(f"{split} history scale is confounded with action")
        per_head = {
            component: Counter(
                bool(row["private_construction_intent"]["intended_bits"][component])
                for row in split_rows
            )
            for component in COMPONENTS
        }
        if any(counts[True] != counts[False] for counts in per_head.values()):
            raise ValueError(f"{split} per-head construction is not balanced")

    # Every FIT component subtype/negative mechanism must actually be learned
    # in FIT. Confirmation and sealed splits may stress different prevalence,
    # but cannot introduce a component mode that training never contained.
    fit_modes = {
        component: Counter(
            str(
                row["private_construction_intent"]["component_plans"][component][
                    "construction_mode"
                ]
            )
            for row in by_split["FIT"]
        )
        for component in COMPONENTS
    }
    for component in COMPONENTS:
        expected_modes = set(_POSITIVE_MODES[component]) | set(
            _NEGATIVE_MODES[component]
        )
        if set(fit_modes[component]) != expected_modes:
            raise ValueError(f"FIT does not cover every {component} mode")
        for mode, count in fit_modes[component].items():
            minimum = (
                1
                if (
                    component == "RS"
                    and mode in {"routine_closing_or_phatic", "explicit_stop"}
                )
                or (
                    component in {"MS", "ME"}
                    and mode
                    in {
                        "no_suitable_session_candidate",
                        "no_suitable_event_candidate",
                    }
                )
                else 6
            )
            if count < minimum:
                raise ValueError(
                    f"FIT {component}/{mode} support is too sparse: {count} < {minimum}"
                )

    if set(FIT_LOGIC_FAMILIES) & set(CONFIRMATION_LOGIC_FAMILIES):
        raise ValueError("confirmation logic families must be unseen in FIT")
    if set(FIT_LOGIC_FAMILIES) & set(SEALED_LOGIC_FAMILIES):
        raise ValueError("sealed logic families must be unseen in FIT")

    # FIT family-level counterfactual balance: every logic family sees each
    # component both on and off four times, rather than acting as a label cue.
    fit_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_split["FIT"]:
        fit_by_family[str(row["logic_family"])].append(row)
    for family, family_rows in fit_by_family.items():
        if len(family_rows) != 8:
            raise ValueError(f"FIT family size mismatch: {family}")
        for component in COMPONENTS:
            values = Counter(
                bool(row["private_construction_intent"]["intended_bits"][component])
                for row in family_rows
            )
            if values != Counter({False: 4, True: 4}):
                raise ValueError(f"FIT family/component confounding: {family}/{component}")
        peer_components = Counter(
            peer["flipped_component"]
            for row in family_rows
            for peer in row["private_counterfactual_peers"]
        )
        # Each undirected edge appears on both endpoint rows.  Two edges per
        # component therefore yield four directed peer references.
        if peer_components != Counter({component: 4 for component in COMPONENTS}):
            raise ValueError(f"FIT counterfactual coverage mismatch: {family}")

    return {
        "protocol": "pm-v1.5-p2-private-blueprint-static-audit-v1",
        "status": "PASS",
        "primary_rows": len(rows),
        "unique_users": len(set(users)),
        "split_sizes": expected_sizes,
        "actions_per_split": 16,
        "fit_logic_families": len(FIT_LOGIC_FAMILIES),
        "fit_single_bit_counterfactual_edges_per_component_per_family": 2,
        "confirmation_unseen_logic_families": len(CONFIRMATION_LOGIC_FAMILIES),
        "sealed_unseen_logic_families": len(SEALED_LOGIC_FAMILIES),
        "fit_construction_mode_counts": {
            component: dict(sorted(counts.items()))
            for component, counts in fit_modes.items()
        },
        "construction_intent_is_gold": False,
        "construction_intent_is_model_input": False,
    }
