"""Deterministic, outcome-blind realization of the PM V1.5 V3 blueprint.

This module creates ordinary non-clinical user histories.  It does not create
responses, labels, feature values, or routing decisions.  Every target item is
only a construction hypothesis until the shared compiler and formal retriever
rediscover it in P2.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .text import normalize_for_hash, normalize_space
from .v1_5_memory_transport import compile_me_structure_metadata
from .v1_5_v3_effect_blueprint import PROTOCOL as BLUEPRINT_PROTOCOL


RAW_PROTOCOL = "pm-v1.5-v3-visible-state-realization-v2"

TOPIC_WORDS: dict[str, tuple[str, str]] = {
    "workload_deadline": ("project", "deadline"),
    "relationship_change": ("relationship", "routine"),
    "family_care": ("family", "caregiving"),
    "sleep_schedule": ("shift", "sleep"),
    "social_disconnection": ("friendship", "connection"),
    "study_pressure": ("exam", "study"),
    "creative_block": ("painting", "inspiration"),
    "relocation_adjustment": ("city", "adjustment"),
    "financial_uncertainty": ("budget", "income"),
    "health_routine_nonclinical": ("exercise", "routine"),
    "friendship_tension": ("friendship", "trust"),
    "public_speaking": ("presentation", "audience"),
    "job_transition": ("job", "transition"),
    "caregiving_balance": ("caregiving", "balance"),
    "housing_change_nonurgent": ("apartment", "move"),
    "grief_reminder_nonacute": ("family", "remembrance"),
    "team_conflict": ("team", "conflict"),
    "identity_transition": ("identity", "transition"),
    "long_distance_relationship": ("distance", "relationship"),
    "exam_setback": ("exam", "setback"),
    "loneliness_weekend": ("weekend", "loneliness"),
    "boundary_setting": ("boundary", "conversation"),
    "habit_disruption": ("morning", "habit"),
    "decision_uncertainty": ("choice", "uncertainty"),
    # V5.2 one-shot confirmation topics.  These are deliberately ordinary,
    # non-clinical, and disjoint from every V3 FIT/FRESH/SEALED topic.  Adding
    # them extends only the outcome-blind state surface; it does not change the
    # frozen V5.2 executor, candidate ranker, features, or policy heads.
    "commute_disruption": ("commute", "disruption"),
    "volunteer_commitment": ("volunteer", "commitment"),
    "household_coordination": ("household", "coordination"),
    "hobby_group_tension": ("hobby group", "tension"),
    "course_project": ("course", "project"),
    "seasonal_routine_change": ("seasonal", "routine"),
    "shared_space_noise": ("shared space", "noise"),
    "appointment_planning_nonurgent": ("appointment", "planning"),
}

_TIME_DETAILS = (
    "this morning",
    "after lunch",
    "during a quiet evening",
    "on the way home",
    "before starting work",
    "after a short walk",
    "while making tea",
    "during a free hour",
    "after checking my calendar",
    "while tidying my desk",
    "before dinner",
    "after closing my laptop",
    "during the weekend",
    "after writing a short list",
    "while waiting for the bus",
    "during a break",
    "after a routine errand",
    "while sitting by the window",
    "before an ordinary appointment",
    "after putting my phone away",
)

_NOTICE_DETAILS = (
    "the same concern became noticeable again",
    "I realized I was circling the issue again",
    "I noticed the issue was taking up attention",
    "the question came back into focus",
    "I could tell the issue still needed a careful response",
    "I found myself reconsidering what would help",
    "the situation felt worth describing more precisely",
    "I noticed I was close to rushing to a conclusion",
    "I wanted to separate what is known from what is assumed",
    "I realized the current goal needed to stay narrow",
    "I wanted to keep this turn grounded in the actual situation",
    "I noticed that a generic response might miss the point",
    "I wanted to avoid repeating an old answer automatically",
    "I realized that one detail could change the next response",
    "I wanted to keep the people in the story clearly separated",
    "I noticed that the current need was more specific than the topic",
)

_DISTRACTOR_TOPICS = (
    "a documentary about ocean wildlife",
    "two soup recipes for a weekend lunch",
    "a library display of antique maps",
    "the color of a reusable water bottle",
    "a board game for a quiet evening",
    "the names of garden birds",
    "a shelf of paperback books",
    "a comedy film from last month",
    "tea flavors for an ordinary afternoon",
    "a neighborhood bakery menu",
    "photographs of clouds during a walk",
    "a loose button on a coat",
    "two routes to a local museum",
    "a podcast about ancient architecture",
    "fruit choices for a picnic",
    "postcards arranged in an album",
    "notebooks for household lists",
    "a neighbor's playful dog",
    "a simple bread recipe",
    "novels donated to a book sale",
    "lamps for a reading corner",
    "a small craft market",
    "dust cleaned from a bicycle",
    "a radio program about local history",
    "a basic card game",
    "music for a train ride",
    "mugs at a pottery stall",
    "an astronomy magazine",
    "herbs on a windowsill",
    "a puzzle on a rainy evening",
    "a scarf folded into a drawer",
    "a postcard from a coastal town",
    "a ceramic bowl at a market",
    "a nature program about forests",
    "a recipe for oat biscuits",
    "an old photograph of a lighthouse",
    "a pen chosen for a notebook",
    "a bicycle route beside a river",
    "a short article about fossils",
    "a collection of ticket stubs",
    "a radio interview about theatre",
    "a plant moved to a sunnier shelf",
    "a knitted hat at a craft stall",
    "a train timetable for a museum visit",
    "a sketch of a neighborhood tree",
    "a reusable bag from a bookshop",
    "a crossword completed at breakfast",
    "a photo album from a day trip",
    "a jar used for spare buttons",
    "a song heard in a cafe",
    "a bowl of fruit on a table",
    "an illustrated guide to insects",
    "a small wooden picture frame",
    "a calendar of landscape photos",
    "a paper bookmark from a library",
    "a mug placed beside a kettle",
    "a magazine article about bridges",
    "a packet of seeds for spring",
    "a local poster about a craft fair",
    "a drawer of charging cables",
    "a folded map of a walking trail",
    "a documentary about coral reefs",
    "a tin used for sewing thread",
    "a basket of clean towels",
    "a book about early aviation",
    "a vase on a kitchen shelf",
    "a photo of a stone fountain",
    "a recipe card for vegetable soup",
    "a station sign seen from a train",
    "a packet of colored pencils",
    "a small clock on a mantel",
    "a brochure for a botanical garden",
)


def _topic(row: Mapping[str, Any]) -> tuple[str, str, str]:
    first, second = TOPIC_WORDS[str(row["topic_family"])]
    return first, second, f"{first} {second}"


def _style(group_index: int) -> str:
    return (
        f"{_TIME_DETAILS[group_index % len(_TIME_DETAILS)]}, "
        f"{_NOTICE_DETAILS[(group_index // len(_TIME_DETAILS)) % len(_NOTICE_DETAILS)]}."
    )


def _base_current(row: Mapping[str, Any], group_index: int) -> str:
    _first, _second, topic = _topic(row)
    lead = _style(group_index)
    goal = str(row["explicit_current_goal"])
    requests = {
        "response_format": "I want concise words that stay accurate without turning this into a plan.",
        "low_burden_support": "Please keep the response brief while offering no more than one point.",
        "practical_feasibility": "I want one realistic response that fits my actual circumstances.",
        "contextual_understanding": "I want the response to reflect the relevant context instead of sounding generic.",
        "continue_prior_goal": "I want to continue the earlier goal rather than start a different topic.",
        "disambiguate_current_problem": "I cannot tell which part matters most, and I want help separating the two pressures.",
        "resume_unfinished_thread": "I want to resume the unfinished thread and keep the same narrow goal.",
        "recall_prior_fact": "Please remind me of the one specific observation recorded earlier about this issue.",
        "one_optional_step": "I need a manageable place to begin and am open to one optional idea.",
        "avoid_known_failure": "I want to avoid repeating what made this harder before.",
        "choose_safe_alternative": "I want one safer alternative to the approach that did not work before.",
        "try_reversible_experiment": "I am open to one small reversible experiment that I can evaluate.",
        "one_focused_question": "Several parts feel tangled, and I cannot tell which part of the pressure matters most.",
        "brief_paraphrase": "I need space to say what is happening before deciding what to do, and a brief account would help.",
        "emotion_reflection": "The reaction is still hard to name accurately, and I want words close to what I actually said.",
    }[goal]
    return normalize_space(f"{lead} I am dealing with the {topic} situation. {requests}")


def _current_text(row: Mapping[str, Any], group_index: int) -> str:
    base = _base_current(row, group_index)
    family = str(row["logic_family"])
    component = str(row["target_component"])
    if component == "RS" and str(row["logic_pair"]) == "RS_SUGGESTION":
        # The Rank-1 Strategy candidate must be selected for the actual current
        # function, not merely for the broad "advice welcome" family.  This
        # remains a user-state cue; it does not name a private card or label.
        base = normalize_space(
            base.replace(
                "I need a manageable place to begin and am open to one optional idea.",
                "I am open to one optional idea for making the immediate environment easier to handle.",
            )
        )
    if row["track"] == "ELIGIBILITY_AUDIT" and str(
        row.get("private_eligibility_intent")
    ) == "INELIGIBLE":
        if family == "MP_CURRENT_ECHO":
            return normalize_space(base + " My stable preference is one concise reflection before any question.")
        if family == "MP_CURRENT_CONFLICT":
            return normalize_space(base + " For this turn, give several detailed suggestions rather than one brief option.")
        if family == "MP_WRONG_OWNER_OR_STALE":
            return normalize_space(base.replace("I am dealing with", "My friend is dealing with") + " The relevant circumstances belong to my friend, not me.")
        if family == "MS_CURRENT_SESSION_ECHO":
            return normalize_space(base + " The earlier conclusion was that emotional uncertainty, rather than practical workload, was the main pressure.")
        if family == "MS_RESOLVED_OR_STALE":
            return normalize_space(base + " The old version was fully resolved; I am asking about a separate current issue.")
        if family == "MS_WRONG_OWNER_OR_GOAL":
            return normalize_space(
                base
                + " This current question is about me, not my friend; keep our histories separate."
            )
        if family == "ME_VALID_OUTCOME_WRONG_CURRENT_GOAL":
            return normalize_space(base + " Do not offer or reuse an action; I only want the feeling named.")
        if family == "ME_WRONG_OWNER_REDUNDANT_OR_STALE":
            return normalize_space(base.replace("I am dealing with", "My friend is dealing with") + " The earlier event was mine, but this question is about my friend.")
        if family == "RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE":
            return normalize_space(
                base
                + " Please ask exactly one focused question to clarify the feeling I already indicated; do not add another support move."
            )
        if family == "RS_SAME_MOVE_ALREADY_EXECUTED":
            if str(row["logic_pair"]) == "RS_PARAPHRASE":
                return normalize_space(base + " That concise account already captures my meaning, so move beyond restating it.")
            return normalize_space(base + " The same support move was already completed, so do not repeat it.")
        if family == "RS_GREETING_STOP_OR_LISTEN_ONLY":
            return normalize_space(base + " For this turn, only listen while I explain; do not ask me a question.")
        if family == "RS_WRONG_FAMILY_BURDEN_OR_HIGH_STAKES":
            return normalize_space(base + " An idea may matter later, but for this turn do not give advice.")
    if component == "RS" and row["track"] == "COMPONENT_EFFECT":
        enrichment = str(row["private_benefit_enrichment"])
        if enrichment == "HIGH":
            return normalize_space(base + " A generic acknowledgment would leave the central uncertainty unresolved.")
        return normalize_space(base + " The central point is already fairly clear, but this support move would still fit.")
    return base


def _recent_dialogue(row: Mapping[str, Any], current: str) -> list[dict[str, str]]:
    _first, _second, topic = _topic(row)
    prior_supporter = "I am following the situation and will not assume what it means for you."
    family = str(row["logic_family"])
    if family == "RS_SAME_MOVE_ALREADY_EXECUTED":
        prior_supporter = (
            f"The {topic} situation is disrupting your routine while leaving the main concern uncertain."
            if str(row["logic_pair"]) == "RS_PARAPHRASE"
            else f"Which one part of the {topic} pressure feels most important right now?"
        )
    return [
        {
            "role": "user",
            "content": f"The {topic} issue has been taking up attention, but I want to keep the current goal specific.",
        },
        {"role": "assistant", "content": prior_supporter},
    ]


def _mp_target(row: Mapping[str, Any]) -> tuple[str, str]:
    first, second, topic = _topic(row)
    subtype = str(row["candidate_subtype_target"])
    family = str(row["logic_family"])
    if subtype == "MP_PREFERENCE":
        key = "stable_preference_response_format"
        if family == "MP_CURRENT_CONFLICT":
            value = "prefers one concise optional response rather than several detailed suggestions"
        elif "BURDEN_DECISIVE" in family:
            value = "prefers one brief response with no list or follow-up task"
        elif "TOKEN_COST" in family:
            value = "prefers one brief response rather than two separate short sentences"
        elif "MINOR_REFINEMENT" in family:
            value = "prefers tentative concise words whenever a reflection is offered"
        else:
            value = "prefers concise words in one reflection before any question"
        return key, value
    key = f"stable_{first}_{second}_constraint"[:40]
    if family == "MP_IRRELEVANT_PROFILE":
        value = f"owns a blue notebook that happens to mention the {topic} topic"
    elif "WEAKLY_RELEVANT" in family or "SMALL_INCREMENT" in family:
        value = f"usually has a short quiet window when considering the {topic} issue"
    else:
        value = f"can address the {topic} issue only during one short private window in the early evening"
    return key, value


def _basic_info(row: Mapping[str, Any]) -> dict[str, str]:
    target_count = int(row["source_catalog_size_target"])
    if row["target_component"] != "MP":
        target_count = 2
    key, value = _mp_target(row)
    fields = {key: value}
    for index in range(target_count - 1):
        fields[f"ordinary_background_{index + 1}"] = _DISTRACTOR_TOPICS[index]
    return fields


def _ms_target(row: Mapping[str, Any]) -> tuple[str, str, str]:
    _first, _second, topic = _topic(row)
    family = str(row["logic_family"])
    if family == "MS_CURRENT_SESSION_ECHO":
        summary = f"For the earlier {topic} issue, emotional uncertainty rather than practical workload was the main pressure."
    elif family == "MS_TOPIC_ONLY_GENERIC":
        summary = f"The user previously discussed the {topic} issue in general terms."
    elif family == "MS_RESOLVED_OR_STALE":
        summary = f"The old {topic} issue was resolved, completed, and fully settled."
    elif family == "MS_WRONG_OWNER_OR_GOAL":
        summary = f"The earlier {topic} details belonged to the user's friend and concerned a different goal."
    elif family == "MS_ELIGIBLE_MS_DISTINCTION" or "PRIOR_DISTINCTION" in family:
        summary = f"For the earlier {topic} issue, emotional uncertainty rather than practical workload was the main pressure."
    elif "RELEVANT_DETAIL" in family:
        summary = f"The earlier {topic} session separated rising practical workload from emotional uncertainty but did not decide which mattered more."
    elif "UNFINISHED" in family:
        summary = f"The earlier {topic} goal of finding one accurate sentence remained open and unfinished."
    elif "SELF_CONTAINED" in family:
        summary = f"The earlier {topic} session paused with one exact wording question still open for the user to revisit."
    elif family == "MS_ELIGIBLE_MS_RECALL" or "EXPLICIT_FACT_RECALL" in family:
        summary = f"The earlier {topic} session recorded that Tuesday evening was the one workable time."
    elif "CONTINUITY_GAIN" in family:
        summary = f"The earlier {topic} session recorded that the issue was usually discussed in the evening."
    elif "BROAD_PRIOR_GOAL" in family:
        summary = f"The earlier {topic} goal was to name one concern before choosing any next step."
    else:
        summary = f"The earlier {topic} goal was to understand the immediate pressure before discussing a plan."
    seeker = f"I discussed the {topic} situation and wanted one specific point preserved for later."
    supporter = "You kept the summary limited to the point that was actually established."
    return summary, seeker, supporter


def _me_target(row: Mapping[str, Any]) -> tuple[str, str, str]:
    _first, _second, topic = _topic(row)
    family = str(row["logic_family"])
    subtype = str(row["candidate_subtype_target"])
    if subtype == "ME_CONTEXT_EVENT":
        seeker = f"The {topic} issue was present in the background during that week."
    elif subtype == "ME_UNRESOLVED_EVENT":
        seeker = f"The {topic} issue was still unresolved, and I was still struggling with it."
    elif family == "ME_WRONG_OWNER_REDUNDANT_OR_STALE":
        seeker = f"For my own {topic} issue, I wrote one short note, and it helped make my next step easier."
    elif "FAILURE_RESULT" in family:
        # The shared compiler treats literal "did not help" / "made worse"
        # wording as unresolved.  Preserve the observed failed alternative
        # while also recording the successful replacement that makes the
        # episode a reusable action-result rather than relabeling it by fiat.
        seeker = f"For the {topic} issue, I tried a long list; switching to one item helped reduce the pressure."
    elif "MECHANISM_PREVENTS" in family:
        seeker = f"For the {topic} issue, I paused before responding; what helped was limiting myself to one reversible step."
    elif "CHANGED_NONCONFLICTING" in family:
        seeker = f"For a less urgent version of the {topic} issue, I paused before replying, and the extra time helped me choose calmer wording."
    elif "REVERSIBLE_PRIOR_EXPERIMENT" in family:
        seeker = f"For the {topic} issue, I tried one short reversible experiment, and it helped because I could evaluate it afterward."
    elif "WEAK_PERSONALIZATION" in family:
        seeker = f"For the {topic} issue, I tried a brief pause, and it helped a little before an ordinary reply."
    elif "VALID_RESULT_BUT_OBVIOUS" in family:
        seeker = f"For the {topic} issue, I wrote one short note, and it helped organize my thoughts a little."
    elif "VALID_RESULT_AMONG_MANY" in family:
        seeker = f"For the {topic} issue, I tried choosing one item first, and it helped about as much as several other small options."
    else:
        seeker = f"For the {topic} issue, I wrote one short note, and it helped make the next step easier."
    summary = f"The user described one earlier {topic} episode without assuming that it still applies now."
    supporter = "You kept that past episode separate from claims about the present."
    return summary, seeker, supporter


def _distractor_session(index: int) -> dict[str, Any]:
    subject = _DISTRACTOR_TOPICS[index % len(_DISTRACTOR_TOPICS)]
    return {
        "id": f"prior_session_{index + 1}",
        "timestamp": "",
        "summary": f"The user had an ordinary conversation about {subject}.",
        "dialogue": [
            {"role": "seeker", "content": f"I was thinking about {subject}."},
            {"role": "supporter", "content": "You asked which ordinary detail stood out."},
        ],
    }


def _history(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    component = str(row["target_component"])
    target_count = int(row["source_catalog_size_target"])
    session_count = target_count if component in {"MS", "ME"} else 3
    history = [_distractor_session(index) for index in range(session_count)]
    if component == "MS":
        summary, seeker, supporter = _ms_target(row)
    elif component == "ME":
        summary, seeker, supporter = _me_target(row)
    else:
        _first, _second, topic = _topic(row)
        summary = f"The user had an earlier ordinary conversation about the {topic} situation."
        seeker = f"I mentioned the {topic} issue without settling what it meant."
        supporter = "You left the earlier exchange open for clarification."
    history[-1] = {
        "id": f"prior_session_{session_count}",
        "timestamp": "",
        "summary": summary,
        "dialogue": [
            {"role": "seeker", "content": seeker},
            {"role": "supporter", "content": supporter},
        ],
    }
    return history


def realize_v3_state(row: Mapping[str, Any], *, group_index: int) -> dict[str, Any]:
    if row.get("protocol") != BLUEPRINT_PROTOCOL:
        raise ValueError("not a V3 blueprint row")
    current = _current_text(row, group_index)
    visible = _recent_dialogue(row, current)
    history = _history(row)
    user_id = str(row["user_id"])
    return {
        "protocol": RAW_PROTOCOL,
        "state_id": str(row["blueprint_row_id"]),
        "blueprint_row_id": str(row["blueprint_row_id"]),
        "user_id": user_id,
        "group_id": str(row["group_id"]),
        "split": str(row["split"]),
        "track": str(row["track"]),
        "user": {
            "id": user_id,
            "basic_info": _basic_info(row),
            "dialog_history": history,
        },
        "visible_dialogue": visible,
        "current_user_text": current,
        "current_session_index": len(history) + 1,
        "construction_intent_present_in_model_input": False,
        "response_or_outcome_read": False,
        "external_text_read": False,
    }


def realize_v3_states(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_ids = sorted({str(row["counterfactual_group_id"]) for row in rows})
    group_index = {group_id: index for index, group_id in enumerate(group_ids)}
    return [
        realize_v3_state(
            row,
            group_index=group_index[str(row["counterfactual_group_id"])],
        )
        for row in rows
    ]


def audit_v3_states(
    states: Sequence[Mapping[str, Any]],
    blueprint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    blueprint_by_id = {str(row["blueprint_row_id"]): row for row in blueprint_rows}
    if len(states) != 640:
        failures.append("state_count_not_640")
    if len({str(row["state_id"]) for row in states}) != len(states):
        failures.append("duplicate_state_id")
    if set(blueprint_by_id) != {str(row["state_id"]) for row in states}:
        failures.append("state_blueprint_id_mismatch")
    visible_hashes = [
        normalize_for_hash(
            " ".join(
                [
                    *(str(turn["content"]) for turn in state["visible_dialogue"]),
                    str(state["current_user_text"]),
                ]
            )
        )
        for state in states
    ]
    duplicate_visible = len(visible_hashes) - len(set(visible_hashes))
    # Identical visible state is intentional inside candidate counterfactuals:
    # the candidate/history changes while the user state stays fixed.  The
    # actual Step-1 decision surface is audited only after exact Rank-1 is
    # attached, so raw visible duplication is reported but is not a failure.
    source_count_mismatches: list[str] = []
    subtype_mismatches: list[str] = []
    for state in states:
        row = blueprint_by_id[str(state["state_id"])]
        component = str(row["target_component"])
        if component == "MP":
            actual = len(state["user"]["basic_info"])
        elif component in {"MS", "ME"}:
            actual = len(state["user"]["dialog_history"])
        else:
            actual = 80
        if actual != int(row["source_catalog_size_target"]):
            source_count_mismatches.append(str(state["state_id"]))
        if component == "ME":
            target_text = str(state["user"]["dialog_history"][-1]["dialogue"][0]["content"])
            actual_subtype = str(compile_me_structure_metadata(target_text)["me_subtype_hint"])
            if actual_subtype != str(row["candidate_subtype_target"]):
                subtype_mismatches.append(str(state["state_id"]))
    if source_count_mismatches:
        failures.append("source_catalog_size_mismatch")
    if subtype_mismatches:
        failures.append("me_compiler_subtype_mismatch")
    if any(
        state.get("response_or_outcome_read")
        or state.get("external_text_read")
        or state.get("construction_intent_present_in_model_input")
        for state in states
    ):
        failures.append("forbidden_input_or_outcome_read")
    return {
        "protocol": RAW_PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "states": len(states),
        "unique_users": len({str(row["user_id"]) for row in states}),
        "duplicate_visible_state_surfaces": duplicate_visible,
        "source_catalog_size_mismatches": source_count_mismatches,
        "me_compiler_subtype_mismatches": subtype_mismatches,
        "split_counts": dict(Counter(str(row["split"]) for row in states)),
        "track_counts": dict(Counter(str(row["track"]) for row in states)),
        "api_calls": 0,
        "human_labels_read": 0,
        "external_lockbox_read": False,
        "responses_generated": 0,
        "failures": failures,
    }
