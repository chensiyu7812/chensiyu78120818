"""Outcome-blind V3 blueprint for finite-PM eligibility and effect data.

The blueprint is private construction metadata.  It is neither human gold nor
a model input.  Exact candidates, visible text, Step2 execution, and paired
response outcomes are produced and qualified in later phases.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .io import stable_hex


PROTOCOL = "pm-v1.5-v3-finite-pm-effect-blueprint-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
EFFECT_SPLITS = ("EFFECT_FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST")
HISTORY_SCALES = ("SMALL", "MEDIUM", "LARGE", "EVOEMO_LIKE_LARGE")
SOURCE_CATALOG_TARGETS = {
    "MP": {"SMALL": 2, "MEDIUM": 4, "LARGE": 7, "EVOEMO_LIKE_LARGE": 10},
    "MS": {"SMALL": 4, "MEDIUM": 10, "LARGE": 22, "EVOEMO_LIKE_LARGE": 34},
    "ME": {"SMALL": 5, "MEDIUM": 16, "LARGE": 40, "EVOEMO_LIKE_LARGE": 68},
    "RS": {"SMALL": 80, "MEDIUM": 80, "LARGE": 80, "EVOEMO_LIKE_LARGE": 80},
}
SPLIT_STATES_PER_LOGIC = {
    "EFFECT_FIT": 8,
    "FRESH_CONFIRMATION": 4,
    "SEALED_INTERNAL_TEST": 4,
}
TOPICS = {
    "ELIGIBILITY_AUDIT": (
        "workload_deadline",
        "relationship_change",
        "family_care",
        "sleep_schedule",
        "social_disconnection",
        "study_pressure",
        "creative_block",
        "relocation_adjustment",
    ),
    "EFFECT_FIT": (
        "workload_deadline",
        "relationship_change",
        "family_care",
        "sleep_schedule",
        "social_disconnection",
        "study_pressure",
        "creative_block",
        "relocation_adjustment",
    ),
    "FRESH_CONFIRMATION": (
        "financial_uncertainty",
        "health_routine_nonclinical",
        "friendship_tension",
        "public_speaking",
        "job_transition",
        "caregiving_balance",
        "housing_change_nonurgent",
        "grief_reminder_nonacute",
    ),
    "SEALED_INTERNAL_TEST": (
        "team_conflict",
        "identity_transition",
        "long_distance_relationship",
        "exam_setback",
        "loneliness_weekend",
        "boundary_setting",
        "habit_disruption",
        "decision_uncertainty",
    ),
}


EFFECT_PAIRS: dict[str, tuple[dict[str, str], ...]] = {
    "MP": (
        {
            "pair": "MP_FORMAT",
            "high": "MP_PREFERENCE_FORMAT_DECISIVE",
            "low": "MP_PREFERENCE_MINOR_REFINEMENT",
            "subtype": "MP_PREFERENCE",
            "goal": "response_format",
            "function": "change_response_form",
            "external_scope": "INTERNAL_ONLY_MP_PREFERENCE",
        },
        {
            "pair": "MP_BURDEN",
            "high": "MP_PREFERENCE_BURDEN_DECISIVE",
            "low": "MP_HIGH_TOKEN_COST_FOR_MINOR_PERSONALIZATION",
            "subtype": "MP_PREFERENCE",
            "goal": "low_burden_support",
            "function": "change_burden_and_optionality",
            "external_scope": "INTERNAL_ONLY_MP_PREFERENCE",
        },
        {
            "pair": "MP_CONSTRAINT",
            "high": "MP_PROFILE_CONSTRAINT_CHANGES_RESPONSE",
            "low": "MP_PROFILE_WEAKLY_RELEVANT",
            "subtype": "MP_PROFILE",
            "goal": "practical_feasibility",
            "function": "apply_practical_constraint",
            "external_scope": "EVOEMO_MP_PROFILE",
        },
        {
            "pair": "MP_CONTEXT",
            "high": "MP_PROFILE_CONTEXT_PREVENTS_GENERIC_REPLY",
            "low": "MP_CONTEXT_ALREADY_SUFFICIENT_WITH_SMALL_INCREMENT",
            "subtype": "MP_PROFILE",
            "goal": "contextual_understanding",
            "function": "add_relevant_profile_context",
            "external_scope": "EVOEMO_MP_PROFILE",
        },
    ),
    "MS": (
        {
            "pair": "MS_GOAL",
            "high": "MS_PRIOR_GOAL_DIRECTLY_CONTINUES",
            "low": "MS_BROAD_PRIOR_GOAL_MINOR_INCREMENT",
            "subtype": "MS_SESSION",
            "goal": "continue_prior_goal",
            "function": "carry_prior_goal",
            "external_scope": "EVOEMO_MS_SESSION",
        },
        {
            "pair": "MS_DISTINCTION",
            "high": "MS_PRIOR_DISTINCTION_ANSWERS_CURRENT",
            "low": "MS_RELEVANT_DETAIL_WITH_WEAK_DECISION_VALUE",
            "subtype": "MS_SESSION",
            "goal": "disambiguate_current_problem",
            "function": "carry_prior_distinction",
            "external_scope": "EVOEMO_MS_SESSION",
        },
        {
            "pair": "MS_UNFINISHED",
            "high": "MS_UNFINISHED_THREAD_DIRECTLY_REQUESTED",
            "low": "MS_SAFE_CONTINUITY_BUT_SELF_CONTAINED_CURRENT",
            "subtype": "MS_SESSION",
            "goal": "resume_unfinished_thread",
            "function": "resume_prior_thread",
            "external_scope": "EVOEMO_MS_SESSION",
        },
        {
            "pair": "MS_RECALL",
            "high": "MS_EXPLICIT_FACT_RECALL",
            "low": "MS_HIGH_TOKEN_COST_FOR_SMALL_CONTINUITY_GAIN",
            "subtype": "MS_SESSION",
            "goal": "recall_prior_fact",
            "function": "answer_prior_fact_recall",
            "external_scope": "EVOEMO_MS_SESSION",
        },
    ),
    "ME": (
        {
            "pair": "ME_SUCCESS",
            "high": "ME_SUCCESS_RESULT_ACTION_READY",
            "low": "ME_VALID_RESULT_BUT_OBVIOUS_BASE_STEP",
            "subtype": "ME_REUSABLE_OUTCOME",
            "goal": "one_optional_step",
            "function": "reuse_successful_action_result",
            "external_scope": "EVOEMO_ME_REUSABLE_OUTCOME",
        },
        {
            "pair": "ME_MECHANISM",
            "high": "ME_MECHANISM_PREVENTS_REPEAT",
            "low": "ME_VALID_RESULT_WITH_CHANGED_NONCONFLICTING_CONTEXT",
            "subtype": "ME_REUSABLE_OUTCOME",
            "goal": "avoid_known_failure",
            "function": "apply_past_mechanism_tentatively",
            "external_scope": "EVOEMO_ME_REUSABLE_OUTCOME",
        },
        {
            "pair": "ME_FAILURE",
            "high": "ME_FAILURE_RESULT_AVOIDS_KNOWN_FAILURE",
            "low": "ME_VALID_RESULT_AMONG_MANY_EQUIVALENT_OPTIONS",
            "subtype": "ME_REUSABLE_OUTCOME",
            "goal": "choose_safe_alternative",
            "function": "avoid_failed_prior_action",
            "external_scope": "EVOEMO_ME_REUSABLE_OUTCOME",
        },
        {
            "pair": "ME_EXPERIMENT",
            "high": "ME_REVERSIBLE_PRIOR_EXPERIMENT_DIRECTLY_INVITED",
            "low": "ME_HIGH_TOKEN_COST_FOR_WEAK_PERSONALIZATION",
            "subtype": "ME_REUSABLE_OUTCOME",
            "goal": "try_reversible_experiment",
            "function": "offer_prior_experiment_as_limited_option",
            "external_scope": "EVOEMO_ME_REUSABLE_OUTCOME",
        },
    ),
    "RS": (
        {
            "pair": "RS_QUESTION",
            "high": "RS_FOCUSED_QUESTION_ADDS_NEEDED_AXIS",
            "low": "RS_SAFE_MINOR_STRUCTURE_BASE_LLM_LIKELY_KNOWS",
            "subtype": "RS_ATOMIC_MOVE",
            "goal": "one_focused_question",
            "function": "focused_question",
            "external_scope": "ESCONV_PRIMARY_EVOEMO_SECONDARY",
        },
        {
            "pair": "RS_PARAPHRASE",
            "high": "RS_TENTATIVE_PARAPHRASE_REPAIRS_AMBIGUITY",
            "low": "RS_BROAD_SUPPORT_CARD_WITH_SMALL_INCREMENT",
            "subtype": "RS_ATOMIC_MOVE",
            "goal": "brief_paraphrase",
            "function": "tentative_paraphrase_check",
            "external_scope": "ESCONV_PRIMARY_EVOEMO_SECONDARY",
        },
        {
            "pair": "RS_REFLECTION",
            "high": "RS_REFLECTION_ADDS_EVIDENCE_BOUND_EMOTION",
            "low": "RS_LOW_STAKES_OPTIONAL_TECHNIQUE",
            "subtype": "RS_ATOMIC_MOVE",
            "goal": "emotion_reflection",
            "function": "evidence_grounded_reflection",
            "external_scope": "ESCONV_PRIMARY_EVOEMO_SECONDARY",
        },
        {
            "pair": "RS_SUGGESTION",
            "high": "RS_REVERSIBLE_SUGGESTION_DIRECTLY_REQUESTED",
            "low": "RS_HIGH_TOKEN_COST_FOR_SIMPLE_MOVE",
            "subtype": "RS_ATOMIC_MOVE",
            "goal": "one_optional_step",
            "function": "one_reversible_suggestion",
            "external_scope": "ESCONV_PRIMARY_EVOEMO_SECONDARY",
        },
    ),
}


ELIGIBILITY_INVALID: dict[str, tuple[tuple[str, str], ...]] = {
    "MP": (
        ("MP_CURRENT_ECHO", "CURRENTLY_REDUNDANT"),
        ("MP_CURRENT_CONFLICT", "BOUNDARY_OR_CURRENT_CONFLICT"),
        ("MP_IRRELEVANT_PROFILE", "WRONG_GOAL_OR_FUNCTION"),
        ("MP_WRONG_OWNER_OR_STALE", "WRONG_OWNER_OR_STALE"),
    ),
    "MS": (
        ("MS_CURRENT_SESSION_ECHO", "NOT_STRICTLY_PRIOR_OR_REDUNDANT"),
        ("MS_TOPIC_ONLY_GENERIC", "NO_SPECIFIC_INCREMENT"),
        ("MS_RESOLVED_OR_STALE", "RESOLVED_OR_STALE"),
        ("MS_WRONG_OWNER_OR_GOAL", "WRONG_OWNER_OR_GOAL"),
    ),
    "ME": (
        ("ME_CONTEXT_EVENT_NO_OUTCOME", "WRONG_SUBTYPE_NO_OUTCOME"),
        ("ME_UNRESOLVED_EVENT", "WRONG_SUBTYPE_UNRESOLVED"),
        ("ME_VALID_OUTCOME_WRONG_CURRENT_GOAL", "WRONG_GOAL_OR_FUNCTION"),
        ("ME_WRONG_OWNER_REDUNDANT_OR_STALE", "WRONG_OWNER_REDUNDANT_OR_STALE"),
    ),
    "RS": (
        ("RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE", "CURRENTLY_REDUNDANT"),
        ("RS_SAME_MOVE_ALREADY_EXECUTED", "ALREADY_EXECUTED"),
        ("RS_GREETING_STOP_OR_LISTEN_ONLY", "BOUNDARY_OR_NO_RESOURCE_NEED"),
        ("RS_WRONG_FAMILY_BURDEN_OR_HIGH_STAKES", "WRONG_FUNCTION_OR_BOUNDARY"),
    ),
}


def _row_id(*parts: object) -> str:
    return f"v3bp_{stable_hex(PROTOCOL, *parts, n=24)}"


def _base_row(*, row_id: str, component: str, split: str, track: str) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "blueprint_row_id": row_id,
        "user_id": f"user_{row_id}",
        "group_id": f"group_{row_id}",
        "track": track,
        "split": split,
        "target_component": component,
        "content_origin": "SYNTHETIC_INTERNAL_CONTENT_DISJOINT_FROM_EXTERNAL",
        "external_text_or_outcome_read": False,
        "construction_intent_is_gold": False,
        "construction_intent_is_model_input": False,
        "exact_rank1_candidate": None,
        "human_eligibility_gold": None,
        "step2_qualification": None,
        "paired_response_outcomes": None,
        "worth_opening_gold": None,
    }


def _candidate_requirements(component: str, subtype: str) -> dict[str, Any]:
    return {
        "subtype": subtype,
        "same_owner": True,
        "time_valid": True,
        "goal_function_compatible": True,
        "boundary_burden_compatible": True,
        "currently_redundant": False,
        "specific_increment_present": True,
        "must_be_rediscovered_by_formal_retriever": True,
        "component_specific": {
            "MP": "stable_preference_or_relevant_incremental_profile",
            "MS": "strictly_prior_session_goal_distinction_or_thread",
            "ME": "past_action_or_choice_plus_result_or_mechanism",
            "RS": "bank_atomic_move_with_no_card_external_advice",
        }[component],
    }


def build_effect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_offsets = {"EFFECT_FIT": 0, "FRESH_CONFIRMATION": 1, "SEALED_INTERNAL_TEST": 3}
    for component in COMPONENTS:
        for split in EFFECT_SPLITS:
            per_logic = SPLIT_STATES_PER_LOGIC[split]
            topics = TOPICS[split]
            for pair_index, pair in enumerate(EFFECT_PAIRS[component]):
                for ordinal in range(per_logic):
                    topic = topics[(ordinal + 2 * pair_index + split_offsets[split]) % len(topics)]
                    history_scale = HISTORY_SCALES[ordinal % len(HISTORY_SCALES)]
                    counterfactual = f"cf_{stable_hex(PROTOCOL, component, split, pair['pair'], ordinal, n=20)}"
                    surface_family = f"surface_{stable_hex(PROTOCOL, split, pair['pair'], ordinal, n=20)}"
                    for enrichment, logic_family in (
                        ("HIGH", pair["high"]),
                        ("LOW_OR_NEUTRAL", pair["low"]),
                    ):
                        row_id = _row_id("effect", component, split, logic_family, ordinal)
                        row = _base_row(
                            row_id=row_id,
                            component=component,
                            split=split,
                            track="COMPONENT_EFFECT",
                        )
                        row.update(
                            {
                                "logic_family": logic_family,
                                "logic_pair": pair["pair"],
                                "surface_family": surface_family,
                                "counterfactual_group_id": counterfactual,
                                "topic_family": topic,
                                "history_scale": history_scale,
                                "source_catalog_size_target": SOURCE_CATALOG_TARGETS[component][history_scale],
                                "explicit_current_goal": pair["goal"],
                                "candidate_subtype_target": pair["subtype"],
                                "required_candidate_function": pair["function"],
                                "external_construct_scope": pair["external_scope"],
                                "private_benefit_enrichment": enrichment,
                                "intended_eligibility_for_construction_only": "ELIGIBLE",
                                "candidate_requirements": _candidate_requirements(
                                    component, pair["subtype"]
                                ),
                                "effect_treatments": {
                                    "control": "M0+R0",
                                    "treatment": {
                                        "MP": "MP+R0",
                                        "MS": "MS+R0",
                                        "ME": "ME+R0",
                                        "RS": "M0+RS",
                                    }[component],
                                    "generation_seeds": [
                                        stable_hex(PROTOCOL, row_id, "seed-a", n=16),
                                        stable_hex(PROTOCOL, row_id, "seed-b", n=16),
                                    ],
                                    "single_component_change_only": True,
                                },
                            }
                        )
                        rows.append(row)
    return rows


def _apply_invalid_requirement(requirements: dict[str, Any], reason: str) -> None:
    if "REDUNDANT" in reason:
        requirements["currently_redundant"] = True
        requirements["specific_increment_present"] = False


def _eligibility_subtype(
    component: str,
    positive_subtype: str,
    invalid_family: str,
) -> str:
    """Return the subtype the formal compiler must actually rediscover.

    Eligibility negatives are not allowed to lie about the candidate surface.
    In particular, an ME context-only or unresolved episode cannot be declared
    as ``ME_REUSABLE_OUTCOME`` merely because its paired positive uses that
    subtype.  The subtype is an observable compiler output, not a desired gold
    label.
    """

    if component != "ME":
        return positive_subtype
    if invalid_family == "ME_CONTEXT_EVENT_NO_OUTCOME":
        return "ME_CONTEXT_EVENT"
    if invalid_family == "ME_UNRESOLVED_EVENT":
        return "ME_UNRESOLVED_EVENT"
    return positive_subtype
    if "WRONG_OWNER" in reason:
        requirements["same_owner"] = False
    if "STALE" in reason or "RESOLVED" in reason:
        requirements["time_valid"] = False
    if "GOAL" in reason or "FUNCTION" in reason or "SUBTYPE" in reason:
        requirements["goal_function_compatible"] = False
    if "BOUNDARY" in reason or "CONFLICT" in reason:
        requirements["boundary_burden_compatible"] = False
    if "NO_SPECIFIC_INCREMENT" in reason or "ALREADY_EXECUTED" in reason:
        requirements["specific_increment_present"] = False


def build_eligibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topics = TOPICS["ELIGIBILITY_AUDIT"]
    for component in COMPONENTS:
        pairs = EFFECT_PAIRS[component]
        invalid = ELIGIBILITY_INVALID[component]
        for pair_index in range(16):
            topic = topics[pair_index % len(topics)]
            history_scale = HISTORY_SCALES[pair_index % len(HISTORY_SCALES)]
            positive = pairs[pair_index % len(pairs)]
            invalid_family, invalid_reason = invalid[pair_index % len(invalid)]
            counterfactual = f"eligcf_{stable_hex(PROTOCOL, component, pair_index, n=20)}"
            surface_family = f"eligsurface_{stable_hex(PROTOCOL, component, pair_index, n=20)}"
            for intent, family, reason in (
                ("ELIGIBLE", f"{component}_ELIGIBLE_{positive['pair']}", "NONE"),
                ("INELIGIBLE", invalid_family, invalid_reason),
            ):
                row_id = _row_id("eligibility", component, pair_index, intent)
                row = _base_row(
                    row_id=row_id,
                    component=component,
                    split="ELIGIBILITY_AUDIT",
                    track="ELIGIBILITY_AUDIT",
                )
                subtype = _eligibility_subtype(
                    component,
                    positive["subtype"],
                    family if intent == "INELIGIBLE" else "",
                )
                requirements = _candidate_requirements(component, subtype)
                if intent == "INELIGIBLE":
                    _apply_invalid_requirement(requirements, reason)
                row.update(
                    {
                        "logic_family": family,
                        "logic_pair": positive["pair"],
                        "surface_family": surface_family,
                        "counterfactual_group_id": counterfactual,
                        "topic_family": topic,
                        "history_scale": history_scale,
                        "source_catalog_size_target": SOURCE_CATALOG_TARGETS[component][history_scale],
                        "explicit_current_goal": positive["goal"],
                        "candidate_subtype_target": subtype,
                        "required_candidate_function": positive["function"],
                        "external_construct_scope": positive["external_scope"],
                        "private_eligibility_intent": intent,
                        "private_ineligibility_reason": reason,
                        "candidate_requirements": requirements,
                        "effect_treatments": None,
                    }
                )
                rows.append(row)
    return rows


def build_blueprint() -> list[dict[str, Any]]:
    return build_eligibility_rows() + build_effect_rows()


def audit_blueprint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    ids = [str(row["blueprint_row_id"]) for row in rows]
    users = [str(row["user_id"]) for row in rows]
    if len(rows) != 640:
        failures.append("row_count_not_640")
    if len(set(ids)) != len(ids):
        failures.append("duplicate_blueprint_row_id")
    if len(set(users)) != len(users):
        failures.append("duplicate_user_id")
    if any(row.get("external_text_or_outcome_read") for row in rows):
        failures.append("external_text_or_outcome_read")
    if any(row.get("construction_intent_is_gold") for row in rows):
        failures.append("construction_intent_marked_gold")
    if any(row.get("construction_intent_is_model_input") for row in rows):
        failures.append("construction_intent_marked_model_input")

    per_component_split: dict[str, dict[str, int]] = defaultdict(dict)
    for component in COMPONENTS:
        for split in ("ELIGIBILITY_AUDIT",) + EFFECT_SPLITS:
            count = sum(
                row["target_component"] == component and row["split"] == split
                for row in rows
            )
            per_component_split[component][split] = count
            expected = 32 if split != "EFFECT_FIT" else 64
            if count != expected:
                failures.append(f"wrong_count_{component}_{split}_{count}")

    effect_rows = [row for row in rows if row["track"] == "COMPONENT_EFFECT"]
    eligibility_rows = [row for row in rows if row["track"] == "ELIGIBILITY_AUDIT"]
    if any(
        row["intended_eligibility_for_construction_only"] != "ELIGIBLE"
        for row in effect_rows
    ):
        failures.append("ineligible_intent_in_effect_dataset")
    if any(row["effect_treatments"] is not None for row in eligibility_rows):
        failures.append("eligibility_audit_has_generation_treatment")

    eligibility_balance: dict[str, dict[str, int]] = {}
    for component in COMPONENTS:
        counts = Counter(
            row["private_eligibility_intent"]
            for row in eligibility_rows
            if row["target_component"] == component
        )
        eligibility_balance[component] = dict(counts)
        if counts != Counter({"ELIGIBLE": 16, "INELIGIBLE": 16}):
            failures.append(f"eligibility_not_16_16_{component}")

    effect_balance: dict[str, Any] = {}
    for component in COMPONENTS:
        effect_balance[component] = {}
        for split in EFFECT_SPLITS:
            subset = [
                row
                for row in effect_rows
                if row["target_component"] == component and row["split"] == split
            ]
            enrichment = Counter(row["private_benefit_enrichment"] for row in subset)
            logic = Counter(row["logic_family"] for row in subset)
            scales = Counter(row["history_scale"] for row in subset)
            catalog_targets = Counter(row["source_catalog_size_target"] for row in subset)
            topics_by_enrichment = {
                label: Counter(
                    row["topic_family"]
                    for row in subset
                    if row["private_benefit_enrichment"] == label
                )
                for label in ("HIGH", "LOW_OR_NEUTRAL")
            }
            half = len(subset) // 2
            if enrichment != Counter({"HIGH": half, "LOW_OR_NEUTRAL": half}):
                failures.append(f"effect_enrichment_unbalanced_{component}_{split}")
            expected_per_logic = SPLIT_STATES_PER_LOGIC[split]
            if len(logic) != 8 or any(value != expected_per_logic for value in logic.values()):
                failures.append(f"logic_family_unbalanced_{component}_{split}")
            expected_per_scale = len(subset) // len(HISTORY_SCALES)
            if any(scales[scale] != expected_per_scale for scale in HISTORY_SCALES):
                failures.append(f"history_scale_unbalanced_{component}_{split}")
            expected_topic = half // len(TOPICS[split])
            if any(
                topics_by_enrichment[label][topic] != expected_topic
                for label in topics_by_enrichment
                for topic in TOPICS[split]
            ):
                failures.append(f"topic_predicts_enrichment_{component}_{split}")
            effect_balance[component][split] = {
                "enrichment": dict(enrichment),
                "logic_families": dict(logic),
                "history_scales": dict(scales),
                "source_catalog_size_targets": dict(catalog_targets),
                "topics_by_enrichment": {
                    key: dict(value) for key, value in topics_by_enrichment.items()
                },
            }

    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["counterfactual_group_id"])].append(row)
    malformed_groups = {
        group: [row["blueprint_row_id"] for row in members]
        for group, members in groups.items()
        if len(members) != 2
        or len({row["split"] for row in members}) != 1
        or len({row["target_component"] for row in members}) != 1
        or len({row["topic_family"] for row in members}) != 1
        or len({row["history_scale"] for row in members}) != 1
        or len({row["explicit_current_goal"] for row in members}) != 1
    }
    if malformed_groups:
        failures.append("malformed_counterfactual_groups")

    surface_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        surface_splits[str(row["surface_family"])].add(str(row["split"]))
    cross_split_surfaces = {
        key: sorted(value) for key, value in surface_splits.items() if len(value) > 1
    }
    if cross_split_surfaces:
        failures.append("surface_family_cross_split_leakage")

    external_scope_counts = Counter(row["external_construct_scope"] for row in effect_rows)
    return {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "grain": "one_blueprint_row_per_independent_user_and_target_component",
        "rows": len(rows),
        "unique_row_ids": len(set(ids)),
        "unique_users": len(set(users)),
        "eligibility_rows": len(eligibility_rows),
        "effect_rows": len(effect_rows),
        "counterfactual_groups": len(groups),
        "per_component_split": dict(per_component_split),
        "eligibility_balance": eligibility_balance,
        "effect_balance": effect_balance,
        "external_scope_counts": dict(external_scope_counts),
        "cross_split_surface_family_count": len(cross_split_surfaces),
        "malformed_counterfactual_group_count": len(malformed_groups),
        "external_text_or_outcome_reads": sum(
            bool(row.get("external_text_or_outcome_read")) for row in rows
        ),
        "construction_intent_is_gold_count": sum(
            bool(row.get("construction_intent_is_gold")) for row in rows
        ),
        "construction_intent_is_model_input_count": sum(
            bool(row.get("construction_intent_is_model_input")) for row in rows
        ),
        "failures": failures,
        "scientific_interpretation": (
            "Blueprint structure is eligible for visible-state realization; it is not human gold, "
            "not a trained PM, and not evidence of resource benefit."
        ),
    }
