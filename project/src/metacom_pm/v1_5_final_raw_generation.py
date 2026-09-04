"""Structured raw-user generation contract for the final P2 dataset."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Mapping

from pydantic import Field

from .contracts import StrictModel
from .text import normalize_for_hash, normalize_space
from .v1_5_candidate_discovery import (
    final_typed_content_match_level,
    preference_scope_match_level,
)
from .v1_5_memory_transport import compile_me_structure_metadata
from .v1_5_strategy_rag_v4 import observable_opportunity_flags


RAW_GENERATION_PROTOCOL = "pm-v1.5-p2-raw-user-generation-v9"
RAW_REALIZATION_PROTOCOL = "pm-v1.5-p2-deterministic-construction-realizer-v2"
H1_V2_RAW_GENERATION_PROTOCOL = "pm-v1.5-p2-h1-v2-raw-user-generation-v10"
H1_V2_REALIZATION_PROTOCOL = "pm-v1.5-p2-h1-v2-controlled-realizer-v3"

TOPIC_BRIDGE_WORDS: dict[str, tuple[str, str]] = {
    "workload_deadline": ("project", "deadline"),
    "relationship_transition": ("relationship", "routine"),
    "relocation_loneliness": ("city", "weekend"),
    "grief_reminder": ("family", "remembrance"),
    "shift_sleep": ("shift", "sleep"),
    "academic_pressure": ("exam", "study"),
    "caregiving_burden": ("caregiving", "schedule"),
    "social_anxiety": ("social", "gathering"),
    "habit_recovery": ("daily", "habit"),
    "team_dependency": ("team", "input"),
    "family_conflict": ("family", "decision"),
    "creative_block": ("painting", "inspiration"),
    "financial_uncertainty": ("budget", "income"),
    "friendship_rupture": ("friendship", "trust"),
    "decision_ambivalence": ("job", "choice"),
    "routine_disruption": ("morning", "routine"),
}

_DISTRACTOR_SURFACES = (
    "The user compared two harmless soup recipes for a weekend lunch.",
    "The user described a documentary about ocean wildlife.",
    "The user discussed choosing a color for a reusable water bottle.",
    "The user mentioned a local library display about old maps.",
    "The user compared two board games for a quiet evening.",
    "The user described learning the names of garden birds.",
    "The user mentioned reorganizing a shelf of paperback books.",
    "The user discussed a comedy film they recently enjoyed.",
    "The user compared tea flavors for an ordinary afternoon.",
    "The user described a neighborhood bakery's seasonal menu.",
    "The user mentioned photographing clouds during a walk.",
    "The user discussed repairing a loose button on a coat.",
    "The user compared two routes to a nearby museum.",
    "The user described a podcast about ancient architecture.",
    "The user mentioned sorting digital photos into folders.",
    "The user discussed a puzzle completed on a rainy evening.",
    "The user compared notebooks for keeping household lists.",
    "The user described watching a neighbor's playful dog.",
    "The user mentioned learning a simple bread recipe.",
    "The user discussed donating old novels to a book sale.",
    "The user compared two lamps for a reading corner.",
    "The user described visiting a small craft market.",
    "The user mentioned cleaning dust from a bicycle.",
    "The user discussed a radio program about local history.",
    "The user compared fruit choices for a picnic snack.",
    "The user described arranging postcards in an album.",
    "The user mentioned learning a basic card game.",
    "The user discussed choosing music for a train ride.",
    "The user compared two mugs at a pottery stall.",
    "The user described reading an article about astronomy.",
    "The user mentioned watering herbs on a windowsill.",
)

_H1_V2_PROFILE_VALUES: dict[str, str] = {
    "workload_deadline": "the user can plan focused work only during a short train ride",
    "relationship_transition": "the user has a quiet ten-minute window only in the early evening",
    "relocation_loneliness": "the user has one regular local activity on Sunday afternoons",
    "grief_reminder": "the user keeps family-remembrance objects in a shared room",
    "shift_sleep": "the user works a rotating late-to-early shift pattern",
    "academic_pressure": "the user studies mainly during a short daily commute",
    "caregiving_burden": "the user has a fixed caregiving handoff in the early evening",
    "social_anxiety": "the user usually attends gatherings with one familiar companion",
    "habit_recovery": "the user has only a ten-minute window for the morning routine",
    "team_dependency": "the user cannot begin one task until another team supplies an input",
    "family_conflict": "the user can discuss family decisions only in a private one-to-one setting",
    "creative_block": "the user has one uninterrupted creative window before work",
    "financial_uncertainty": "the user's income arrives weekly rather than monthly",
    "friendship_rupture": "the user and the friend usually communicate by text rather than calls",
    "decision_ambivalence": "the user's job choice must fit an existing caregiving schedule",
    "routine_disruption": "the user has no phone access during the first hour of the morning",
}

_H1_V2_RECENT_USER_STARTS = (
    "A few days ago I first raised",
    "Earlier this week I briefly mentioned",
    "In our prior exchange I touched on",
    "Before today I had started describing",
    "Last time I only partly explained",
    "In an earlier conversation I brought up",
    "A little while ago I began talking about",
    "Previously I mentioned",
    "When we spoke before I referred to",
    "In a past exchange I started unpacking",
    "Before this message I had noted",
    "During our earlier conversation I described",
    "Some time ago I briefly raised",
    "In the previous session I mentioned",
    "Earlier I had only begun to explain",
    "When this first came up I described",
)

_H1_V2_RECENT_SUPPORTER = (
    "You invited me to continue at my own pace.",
    "You left room for me to clarify what mattered.",
    "You acknowledged the topic without adding a conclusion.",
    "You kept the response brief and did not assume a cause.",
    "You reflected that the situation deserved a closer look.",
    "You invited me to say more when I was ready.",
    "You stayed with the issue without turning it into a plan.",
    "You noted that more context might change the picture.",
    "You made space for a more specific description.",
    "You responded without deciding what the problem meant for me.",
    "You acknowledged that I had only begun describing it.",
    "You kept the focus on what I had actually said.",
    "You invited a careful next turn rather than a quick conclusion.",
    "You recognized that the issue was still being explained.",
    "You did not add advice before understanding the situation.",
    "You left the earlier exchange open for continuation.",
)

_H1_V2_CURRENT_STYLE = (
    "It has returned to mind today.",
    "I am noticing it again now.",
    "It feels relevant again this week.",
    "The same issue has come back into focus.",
    "I keep returning to it today.",
    "It is affecting how I think about the next step.",
    "I am seeing the issue from a slightly different angle now.",
    "It is still difficult to sort through clearly.",
    "The concern has become noticeable again.",
    "I want to respond to it more deliberately this time.",
    "I am trying not to rush to a conclusion about it.",
    "It is taking up more attention again today.",
    "I want to understand what would actually help now.",
    "I am revisiting it without assuming the old answer still fits.",
    "I can tell that the issue needs a more careful response.",
    "I want this turn to move the conversation forward usefully.",
)


def _topic_statement(
    *, bridge_words: tuple[str, str], logic_family: str, wrong_entity: bool
) -> str:
    first, second = bridge_words
    subject = "My friend" if wrong_entity else "I"
    object_pronoun = "my friend" if wrong_entity else "me"
    phrase = f"{first} {second}"
    if "emotion_named" in logic_family or "mixed_emotion" in logic_family:
        verb = "feels" if wrong_entity else "feel"
        return f"{subject} {verb} frustrated and uncertain about the {phrase} situation."
    if "time_marker_explicit" in logic_family or "then_versus_now" in logic_family:
        verb = "has" if wrong_entity else "have"
        return f"This week, {subject} {verb} been dealing with the {phrase} situation."
    if "time_marker_implicit" in logic_family:
        return f"Lately, the {phrase} situation has been weighing on {object_pronoun}."
    if "self_correction" in logic_family or "explicit_fact_correction" in logic_family:
        return f"At first I called it minor, but actually the {phrase} situation is bothering {object_pronoun}."
    if "cause_expressed_as_uncertain" in logic_family:
        return f"I'm not sure why the {phrase} situation is affecting {object_pronoun} this much."
    if "goal_expressed_as_uncertain" in logic_family or "two_plausible" in logic_family:
        if wrong_entity:
            return f"I'm unsure what my friend needs from the {phrase} situation right now."
        return f"I'm unsure what I need from the {phrase} situation right now."
    if "metaphorical" in logic_family:
        return f"The {phrase} situation feels like a knot that {subject} cannot quite loosen."
    if "long_current_turn" in logic_family or "dense_distractor" in logic_family:
        return f"{subject} keep circling around the {phrase} situation, and the more details I consider, the harder it is to find a clear next response."
    if "indirect" in logic_family or "emotion_implied" in logic_family:
        return f"The {phrase} situation keeps sitting heavily in the background for {object_pronoun}."
    verb = "is" if wrong_entity else "am"
    return f"{subject} {verb} having a hard time with the {phrase} situation."


def _preference_value(rs_mode: str) -> str:
    if rs_mode == "listen_move_fit":
        return "prefers the supporter to listen without advice"
    if rs_mode == "reflection_move_fit":
        return "prefers a concise reflection using their own words"
    if rs_mode == "clarification_move_fit":
        return "prefers one brief question at a time"
    if rs_mode == "one_option_move_fit":
        return "prefers one optional idea rather than a list"
    return "prefers a brief factual response and reminder"


def _realized_current_text(row: Mapping[str, Any]) -> str:
    intent = row["private_construction_intent"]
    plans = intent["component_plans"]
    rs_mode = str(plans["RS"]["construction_mode"])
    mp_mode = str(plans["MP"]["construction_mode"])
    bridge_words = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    first, second = bridge_words
    if rs_mode == "routine_closing_or_phatic":
        return "Thanks for listening."
    if rs_mode == "explicit_stop":
        return "Please end this conversation now."
    wrong_entity = mp_mode == "wrong_entity"
    topic = _topic_statement(
        bridge_words=bridge_words,
        logic_family=str(row["logic_family"]),
        wrong_entity=wrong_entity,
    )
    if rs_mode == "listen_move_fit":
        return f"{topic} I only want you to listen; I do not want advice."
    if rs_mode == "reflection_move_fit":
        return f"{topic} I feel frustrated about it. Help me put this feeling into words."
    if rs_mode == "clarification_move_fit":
        return f"{topic} I'm not sure which part matters most. Can you ask me one focused question?"
    if rs_mode == "one_option_move_fit":
        return f"{topic} Can you give me one small optional idea?"
    if rs_mode == "no_bank_scope_match":
        return (
            f"Can you remind me what was previously recorded about the {first} "
            f"{second} situation? Please answer only that factual question."
        )
    if rs_mode == "strategy_move_already_present":
        if any(bool(intent["intended_bits"][name]) for name in ("MP", "MS", "ME")):
            return (
                f"Did I mention the {first} {second} situation in an earlier "
                "conversation? Please answer only that factual question."
            )
        return "Yes, that captures it exactly."
    raise ValueError(f"unsupported RS construction mode: {rs_mode}")


def _realized_profile_entries(row: Mapping[str, Any]) -> list[dict[str, str]]:
    plans = row["private_construction_intent"]["component_plans"]
    mode = str(plans["MP"]["construction_mode"])
    rs_mode = str(plans["RS"]["construction_mode"])
    first, second = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    if mode == "no_suitable_profile_candidate":
        return []
    if mode == "preference_incremental":
        return [{
            "subtype": "MP_PREFERENCE",
            "field_key": "stable_preference_response_format",
            "value": _preference_value(rs_mode),
        }]
    if mode == "scope_conflict":
        return [{
            "subtype": "MP_PREFERENCE",
            "field_key": "stable_preference_response_format",
            "value": "prefers several detailed suggestions and follow-up questions",
        }]
    if mode == "wrong_entity":
        return [{
            "subtype": "MP_PROFILE",
            "field_key": "personal_responsibility",
            "value": f"the user personally manages a recurring {first} {second} responsibility",
        }]
    if mode == "redundant_current_fact":
        return [{
            "subtype": "MP_PROFILE",
            "field_key": "current_known_context",
            "value": f"the user's current concern is the {first} {second} situation",
        }]
    if mode == "profile_incremental":
        return [{
            "subtype": "MP_PROFILE",
            "field_key": "stable_personal_context",
            "value": f"the user has a recurring personal constraint involving {first} {second}",
        }]
    raise ValueError(f"unsupported MP construction mode: {mode}")


def _ms_session(mode: str, first: str, second: str) -> dict[str, str]:
    topic = f"{first} {second}"
    if mode == "prior_distinction":
        return {
            "seeker_text": f"The {topic} situation has two parts that are easy to mix together.",
            "supporter_text": "You clarified which part was most important in that moment.",
            "summary": f"For the user's {topic}, the main issue was emotional pressure rather than the practical workload.",
        }
    if mode == "prior_outcome":
        return {
            "seeker_text": f"A brief pause around the {topic} issue helped me feel less rushed.",
            "supporter_text": "You noticed that a short pause changed how manageable it felt.",
            "summary": f"For the user's {topic}, a brief pause previously helped reduce the sense of pressure.",
        }
    if mode == "unfinished_goal":
        return {
            "seeker_text": f"I wanted clearer language for the {topic} issue, but I did not reach it yet.",
            "supporter_text": "That goal can remain open until a later conversation.",
            "summary": f"The user's concrete {topic} goal remained open and unfinished.",
        }
    if mode == "resolved_prior_issue":
        return {
            "seeker_text": f"The earlier {topic} issue is no longer bothering me.",
            "supporter_text": "You confirmed that the earlier concern had ended.",
            "summary": f"The old {topic} issue was resolved, completed, and fully settled.",
        }
    if mode == "same_topic_wrong_goal":
        return {
            "seeker_text": f"For the earlier {topic} discussion, I only wanted to compare logistics.",
            "supporter_text": "You kept that exchange focused on factual logistics.",
            "summary": f"The earlier {topic} session pursued a logistical comparison rather than the user's current support goal.",
        }
    if mode == "redundant_current_summary":
        return {
            "seeker_text": f"The {topic} situation felt difficult and uncertain.",
            "supporter_text": "You summarized the difficulty without adding a new distinction.",
            "summary": f"The user said the current {topic} situation felt difficult and uncertain.",
        }
    if mode == "no_suitable_session_candidate":
        return {
            "seeker_text": "I watched a documentary about volcanic islands and ocean currents.",
            "supporter_text": "You described which landscape in the documentary stood out.",
            "summary": "A documentary compared volcanic islands and deep-ocean currents.",
        }
    raise ValueError(f"unsupported MS construction mode: {mode}")


def _me_session(
    mode: str, first: str, second: str, *, ms_intended_on: bool
) -> dict[str, str]:
    topic = f"{first} {second}"
    if mode == "reusable_action_result":
        seeker = f"I wrote one {topic} note, and it helped make the next step easier."
    elif mode == "reusable_action_mechanism":
        seeker = f"I wrote one {topic} note; what helped was limiting it to one reversible step."
    elif mode == "unresolved_event":
        seeker = f"The {topic} situation is still unresolved, and I am still struggling with it."
    elif mode == "wrong_entity_or_goal":
        seeker = f"My neighbor's {topic} decision concerned a different person and a different goal."
    elif mode == "context_event_only":
        seeker = f"The {topic} situation was present in the background of that week."
    elif mode == "no_suitable_event_candidate":
        seeker = "I photographed a bronze statue beside a botanical greenhouse."
    else:
        raise ValueError(f"unsupported ME construction mode: {mode}")
    if mode.startswith("reusable_action_") and not ms_intended_on:
        summary = f"The user mentioned a general {first} concern from an earlier week."
    elif mode == "no_suitable_event_candidate":
        summary = "The user photographed a bronze statue near a botanical greenhouse."
    else:
        summary = f"The user described an earlier {topic} situation without a new present-day conclusion."
    return {
        "seeker_text": seeker,
        "supporter_text": "You kept the earlier episode separate from the user's current situation.",
        "summary": summary,
    }


def deterministically_realize_raw_draft(
    *, draft: FinalRawStateDraft, row: Mapping[str, Any]
) -> FinalRawStateDraft:
    """Separate natural surface generation from exact construction control.

    This compiler reads only the pre-frozen private construction row and the
    provider draft.  It never reads H1 labels, responses, outcomes, or an
    external split.  H1 remains free to reject every mechanically realized
    candidate.
    """

    plans = row["private_construction_intent"]["component_plans"]
    first, second = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    current = _realized_current_text(row)
    ms_mode = str(plans["MS"]["construction_mode"])
    me_mode = str(plans["ME"]["construction_mode"])
    ms = _ms_session(ms_mode, first, second)
    me = _me_session(
        me_mode,
        first,
        second,
        ms_intended_on=bool(
            row["private_construction_intent"]["intended_bits"]["MS"]
        ),
    )
    background = {
        "summary": "The user described an ordinary museum visit and a favorite exhibit.",
        "seeker_text": "A museum exhibit featuring antique maps caught my attention.",
        "supporter_text": "You described which part of the exhibit stood out.",
    }
    rs_mode = str(plans["RS"]["construction_mode"])
    fallback_supporter = (
        f"It sounds like the {first} {second} situation has been weighing on you."
        if rs_mode == "strategy_move_already_present"
        else "You were beginning to describe what makes that situation difficult."
    )
    provider_supporter = normalize_space(
        draft.recent_dialogue[0].supporter_text
        if draft.recent_dialogue
        else ""
    )
    recent_supporter = (
        provider_supporter
        if provider_supporter
        and not _FORBIDDEN_META_RE.search(provider_supporter)
        and not _HIGH_STAKES_RE.search(provider_supporter)
        and normalize_for_hash(provider_supporter) != normalize_for_hash(current)
        else fallback_supporter
    )
    target_sessions = int(row["prior_session_count_target"])
    distractor_target = target_sessions - 3
    safe_distractors: list[str] = []
    seen = {normalize_for_hash(current)}
    for candidate in ([] if distractor_target == 0 else [
        *draft.distractor_session_summaries,
        *_DISTRACTOR_SURFACES,
    ]):
        text = normalize_space(candidate)
        key = normalize_for_hash(text)
        if (
            not text
            or key in seen
            or _FORBIDDEN_META_RE.search(text)
            or _HIGH_STAKES_RE.search(text)
            or final_typed_content_match_level(current, text) > 0.0
        ):
            continue
        safe_distractors.append(text)
        seen.add(key)
        if len(safe_distractors) == distractor_target:
            break
    if len(safe_distractors) != distractor_target:
        raise RuntimeError("deterministic distractor pool is too small")
    payload = {
        "profile_entries": _realized_profile_entries(row),
        "critical_prior_sessions": [ms, me, background],
        "distractor_session_summaries": safe_distractors,
        "recent_dialogue": [{
            "seeker_text": f"Earlier, the {first} {second} situation was on my mind.",
            "supporter_text": recent_supporter,
        }],
        "current_user_text": current,
    }
    return FinalRawStateDraft.model_validate(payload)


def _h1_v2_preference_value(rs_mode: str) -> str:
    if rs_mode == "listen_move_fit":
        return "prefers a brief acknowledgment before any question or advice"
    if rs_mode == "reflection_move_fit":
        return "prefers a concise reflection grounded in their own words"
    if rs_mode == "clarification_move_fit":
        return "prefers no more than one question at a time"
    if rs_mode == "one_option_move_fit":
        return "prefers advice as one optional choice rather than a command"
    return "prefers reminders as one concise factual sentence"


def _h1_v2_profile_entries(row: Mapping[str, Any]) -> list[dict[str, str]]:
    plans = row["private_construction_intent"]["component_plans"]
    mode = str(plans["MP"]["construction_mode"])
    rs_mode = str(plans["RS"]["construction_mode"])
    first, second = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    if mode == "no_suitable_profile_candidate":
        return []
    if mode == "preference_incremental":
        return [
            {
                "subtype": "MP_PREFERENCE",
                "field_key": "stable_preference_response_format",
                "value": _h1_v2_preference_value(rs_mode),
            }
        ]
    if mode == "profile_incremental":
        return [
            {
                "subtype": "MP_PROFILE",
                "field_key": "stable_practical_constraint",
                "value": (
                    f"for the {first} {second} situation, "
                    + _H1_V2_PROFILE_VALUES[str(row["topic_family"])]
                ),
            }
        ]
    if mode == "scope_conflict":
        return [
            {
                "subtype": "MP_PREFERENCE",
                "field_key": "stable_preference_response_format",
                "value": "prefers several detailed suggestions and follow-up questions",
            }
        ]
    if mode == "wrong_entity":
        return [
            {
                "subtype": "MP_PROFILE",
                "field_key": "personal_responsibility",
                "value": f"the user personally manages the recurring {first} {second} responsibility",
            }
        ]
    if mode == "redundant_current_fact":
        return [
            {
                "subtype": "MP_PROFILE",
                "field_key": "current_known_context",
                "value": f"the user's current concern is the {first} {second} situation",
            }
        ]
    raise ValueError(f"unsupported H1-v2 MP mode: {mode}")


def _h1_v2_current_text(row: Mapping[str, Any]) -> str:
    intent = row["private_construction_intent"]
    plans = intent["component_plans"]
    bits = intent["intended_bits"]
    rs_mode = str(plans["RS"]["construction_mode"])
    mp_mode = str(plans["MP"]["construction_mode"])
    ms_mode = str(plans["MS"]["construction_mode"])
    first, second = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    topic = f"{first} {second}"
    if rs_mode == "routine_closing_or_phatic":
        return "Thanks for listening."
    if rs_mode == "explicit_stop":
        return "Please end this conversation now."
    if rs_mode == "no_bank_scope_match":
        return (
            f"Can you remind me what was recorded about the {topic} situation "
            "in a prior session? Please answer only that factual question."
        )

    subject = "My friend is" if mp_mode == "wrong_entity" else "I am"
    parts = [
        f"{subject} dealing with the {topic} situation.",
        _H1_V2_CURRENT_STYLE[int(row["blueprint_index"]) % 16],
    ]
    if rs_mode == "listen_move_fit":
        parts.append(
            "I need space to say what this is like before deciding what to do."
        )
    elif rs_mode == "reflection_move_fit":
        parts.append(
            "The reaction is hard to name, and I want help finding accurate words for it."
        )
    elif rs_mode == "clarification_move_fit":
        parts.append(
            "Several parts feel tangled together, and I cannot tell which one is driving the pressure."
        )
    elif rs_mode == "one_option_move_fit":
        parts.append(
            "I want to do something manageable but cannot see a sensible place to begin; I am open to advice."
        )
    elif rs_mode == "strategy_move_already_present":
        parts.append(
            "The last focused question already identified the uncertainty as the hardest part, so do not repeat it."
        )
    else:
        raise ValueError(f"unsupported H1-v2 RS mode: {rs_mode}")

    if bool(bits["MS"]):
        parts.append(
            "Build on an earlier distinction instead of restarting the story."
        )
    if bool(bits["ME"]):
        parts.append(
            "A tentative option based on what helped before is welcome."
        )
    if mp_mode == "preference_incremental":
        preference_scope_cue = {
            "listen_move_fit": "Please listen and keep the response brief while I explain it.",
            "reflection_move_fit": "Help me find the words for a response.",
            "clarification_move_fit": "I am open to a question that helps clarify it.",
            "one_option_move_fit": "I am open to advice.",
        }.get(rs_mode, "A brief factual reminder would help.")
        parts.append(preference_scope_cue)
    elif mp_mode == "profile_incremental":
        parts.append("Keep the response realistic for my circumstances.")
    elif mp_mode == "scope_conflict":
        parts.append("Keep it brief; avoid a list or follow-up questions.")
    elif mp_mode == "redundant_current_fact":
        parts.append(f"My current concern is exactly the {topic} situation.")

    if ms_mode == "redundant_current_summary":
        parts.append(
            "The emotional pressure rather than the practical workload is the main issue."
        )
    current = normalize_space(" ".join(parts))
    if len(current) > 420:
        raise RuntimeError(f"H1-v2 current text exceeds schema limit: {len(current)}")
    return current


def deterministically_realize_h1_v2_raw_draft(
    *, draft: FinalRawStateDraft, row: Mapping[str, Any]
) -> FinalRawStateDraft:
    """Realize a unique, candidate-first H1-v2 state without using labels."""

    plans = row["private_construction_intent"]["component_plans"]
    bits = row["private_construction_intent"]["intended_bits"]
    first, second = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    topic = f"{first} {second}"
    ms = _ms_session(str(plans["MS"]["construction_mode"]), first, second)
    me = _me_session(
        str(plans["ME"]["construction_mode"]),
        first,
        second,
        ms_intended_on=bool(bits["MS"]),
    )
    background = {
        "summary": "A magazine article compared distant constellations and telescope lenses.",
        "seeker_text": "I read about distant constellations in an astronomy magazine.",
        "supporter_text": "You described which illustration in the article stood out.",
    }
    index = int(row["blueprint_index"]) % 16
    recent_supporter = _H1_V2_RECENT_SUPPORTER[index]
    if str(plans["RS"]["construction_mode"]) == "strategy_move_already_present":
        recent_supporter = (
            f"Which one part of the {topic} situation feels most important right now?"
        )
    target_sessions = int(row["prior_session_count_target"])
    distractor_target = target_sessions - 3
    current = _h1_v2_current_text(row)
    safe_distractors: list[str] = []
    seen = {normalize_for_hash(current)}
    distractor_pool = (
        []
        if distractor_target == 0
        else [*draft.distractor_session_summaries, *_DISTRACTOR_SURFACES]
    )
    for candidate in distractor_pool:
        text = normalize_space(candidate)
        key = normalize_for_hash(text)
        if (
            not text
            or key in seen
            or _FORBIDDEN_META_RE.search(text)
            or _HIGH_STAKES_RE.search(text)
            or final_typed_content_match_level(current, text) > 0.0
        ):
            continue
        safe_distractors.append(text)
        seen.add(key)
        if len(safe_distractors) == distractor_target:
            break
    if len(safe_distractors) != distractor_target:
        raise RuntimeError("H1-v2 deterministic distractor pool is too small")
    split_lead = {
        "FIT": "During the earlier part of this conversation set,",
        "FRESH_CONFIRMATION": "When this topic first appeared in a later exchange,",
        "SEALED_INTERNAL_TEST": "In a separate earlier exchange about the same topic,",
    }[str(row["split"])]
    payload = {
        "profile_entries": _h1_v2_profile_entries(row),
        "critical_prior_sessions": [ms, me, background],
        "distractor_session_summaries": safe_distractors,
        "recent_dialogue": [
            {
                "seeker_text": (
                    f"{split_lead} {_H1_V2_RECENT_USER_STARTS[index].lower()} "
                    f"the {topic} situation."
                ),
                "supporter_text": recent_supporter,
            }
        ],
        "current_user_text": current,
    }
    return FinalRawStateDraft.model_validate(payload)


class FinalProfileEntryDraft(StrictModel):
    subtype: Literal["MP_PREFERENCE", "MP_PROFILE"]
    field_key: str = Field(min_length=3, max_length=40)
    value: str = Field(min_length=4, max_length=140)


class FinalCriticalPriorSessionDraft(StrictModel):
    summary: str = Field(min_length=8, max_length=220)
    seeker_text: str = Field(min_length=8, max_length=240)
    supporter_text: str = Field(min_length=8, max_length=240)


class FinalCurrentExchangeDraft(StrictModel):
    seeker_text: str = Field(min_length=4, max_length=220)
    supporter_text: str = Field(min_length=4, max_length=220)


class FinalRawStateDraft(StrictModel):
    """Provider output contains prose only, never IDs, labels, or actions."""

    profile_entries: list[FinalProfileEntryDraft] = Field(max_length=3)
    critical_prior_sessions: list[FinalCriticalPriorSessionDraft] = Field(
        min_length=3, max_length=3
    )
    distractor_session_summaries: list[
        Annotated[str, Field(min_length=8, max_length=140)]
    ] = Field(max_length=31)
    recent_dialogue: list[FinalCurrentExchangeDraft] = Field(
        min_length=1, max_length=2
    )
    current_user_text: str = Field(min_length=8, max_length=420)


_FORBIDDEN_ACRONYM_RE = re.compile(r"\b(?:MP|MS|ME|RS)\b")
_FORBIDDEN_META_RE = re.compile(
    r"\b(?:candidate|component|gold label|construction mode|"
    r"retriever|rank[- ]?1|dataset split)\b",
    flags=re.IGNORECASE,
)
_HIGH_STAKES_RE = re.compile(
    r"\b(?:suicid|kill|dying|self[- ]?harm|medical diagnosis|psychosis|"
    r"domestic violence|abuse|overdose|eviction|homeless|pregnan|legal advice)\w*\b",
    flags=re.IGNORECASE,
)


def _concrete_mode_requirement(
    *, component: str, mode: str, intended_bits: Mapping[str, Any]
) -> str:
    """Repeat only the selected row's executable constraint in its user prompt."""

    if component == "MP":
        return {
            "preference_incremental": (
                "profile_entries must contain 1-3 MP_PREFERENCE entries only; "
                "one stable support-format preference must matter now without "
                "being repeated verbatim in current dialogue; its value and "
                "current_user_text must share an explicit format word such as "
                "advice, option, listen, question, brief, concise, reflection, "
                "words, factual, reminder, response, or time"
            ),
            "profile_incremental": (
                "profile_entries must contain 1-3 MP_PROFILE entries only; one "
                "non-sensitive stable fact must constrain the current issue and "
                "must not be repeated in current dialogue"
            ),
            "redundant_current_fact": (
                "include a profile or preference entry, but repeat its relevant "
                "value explicitly in current dialogue so it adds no information"
            ),
            "scope_conflict": (
                "include at least one MP_PREFERENCE and make the current explicit "
                "request ask for the opposite interaction format"
            ),
            "wrong_entity": (
                "include at least one MP_PROFILE about the user, while the current "
                "question's relevant fact concerns another named person"
            ),
            "no_suitable_profile_candidate": "profile_entries must be exactly []",
        }[mode]
    if component == "MS":
        return {
            "prior_distinction": (
                "critical session 1 summary must state a specific earlier "
                "distinction using 'rather than' and share at least two topic "
                "content words with current_user_text"
            ),
            "prior_outcome": (
                "critical session 1 summary must state what previously helped or "
                "did not help and share at least two topic content words with current_user_text"
            ),
            "unfinished_goal": (
                "critical session 1 summary must explicitly say a concrete goal "
                "remained open or unfinished and share at least two topic content words with current_user_text"
            ),
            "resolved_prior_issue": (
                "a topically similar summary must explicitly say the old issue was "
                "resolved, completed, or fully settled"
            ),
            "same_topic_wrong_goal": (
                "a topically similar summary must pursue a materially different "
                "goal from current_user_text"
            ),
            "redundant_current_summary": (
                "repeat the useful content of a topically related prior summary "
                "explicitly in current dialogue"
            ),
            "no_suitable_session_candidate": (
                "all three critical summaries and all distractor summaries must "
                "use topics and content words unrelated to current_user_text"
            ),
        }[mode]
    if component == "ME":
        return {
            "reusable_action_result": (
                "critical session 2 seeker_text must literally contain 'I tried' "
                "or 'I wrote' plus an observed result containing 'helped', 'worked', "
                "'eased', or 'made it easier'; it must share at least two topic "
                "content words with current_user_text"
            ),
            "reusable_action_mechanism": (
                "critical session 2 seeker_text must literally contain the user's "
                "action plus 'what helped was' or 'worked because'; it must share "
                "at least two topic content words with current_user_text"
            ),
            "context_event_only": (
                "all critical seeker_text fields must contain background only: no "
                "user action, observed result, mechanism, or unresolved marker"
            ),
            "unresolved_event": (
                "critical session 2 seeker_text must literally contain 'still "
                "unresolved' or 'still struggling' and no successful result"
            ),
            "wrong_entity_or_goal": (
                "make a lexically similar prior seeker episode concern a different "
                "person or materially different goal from current_user_text"
            ),
            "no_suitable_event_candidate": (
                "all critical seeker_text fields and distractors must use topics "
                "and content words unrelated to current_user_text"
            ),
        }[mode]
    if component == "RS":
        memory_on = any(bool(intended_bits[name]) for name in ("MP", "MS", "ME"))
        return {
            "listen_move_fit": (
                "current_user_text must end with the literal sentence 'I only want "
                "you to listen; I do not want advice.'"
            ),
            "reflection_move_fit": (
                "current_user_text must include the literal phrase 'help me put this "
                "feeling into words'"
            ),
            "clarification_move_fit": (
                "current_user_text must include the literal phrase 'ask me one "
                "focused question'"
            ),
            "one_option_move_fit": (
                "current_user_text must include the literal phrase 'give me one "
                "small optional idea'"
            ),
            "routine_closing_or_phatic": (
                "current_user_text must be exactly 'Thanks for listening.' and must "
                "contain no other problem or request"
            ),
            "explicit_stop": (
                "current_user_text must be exactly 'Please end this conversation "
                "now.'"
            ),
            "strategy_move_already_present": (
                "the visible supporter must already perform the fitting emotional "
                "support move; current_user_text must add a direct factual continuity "
                "question about prior history, not merely say thanks"
                if memory_on
                else "the visible supporter must already perform the fitting move and "
                "current_user_text must only acknowledge it without a new support need"
            ),
            "no_bank_scope_match": (
                "current_user_text must be a direct factual/logistical recall question "
                "about the relevant profile or prior history and must end with the "
                "literal sentence 'Please answer only that factual question.'"
                if memory_on
                else "current_user_text must be a harmless factual/logistical question "
                "and end with 'Please answer only that factual question.'"
            ),
        }[mode]
    raise ValueError(f"unknown component: {component}")


def generation_messages_for_blueprint(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Render private construction intent only into the synthetic-data prompt."""

    intent = dict(row["private_construction_intent"])
    plans = dict(intent["component_plans"])
    target_sessions = int(row["prior_session_count_target"])
    distractor_count = target_sessions - 3
    if distractor_count < 0 or distractor_count > 31:
        raise ValueError("unsupported prior-session target")
    bridge_words = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    rs_mode = str(plans["RS"]["construction_mode"])
    topical_current = rs_mode not in {
        "routine_closing_or_phatic",
        "explicit_stop",
    }
    system = """You create one synthetic, non-clinical emotional-support user history for a controlled routing experiment. Return only the strict JSON schema.

The text must be natural English. Keep the same fictional user across history and current dialogue. Avoid crisis, medical, legal, abuse, housing emergency, or other high-stakes situations. Do not mention experimental terms, resource acronyms, candidates, labels, retrieval, or datasets.

Create exactly three substantive prior sessions. Each summary must faithfully summarize its seeker/supporter exchange. Then create the requested number of concise, mutually distinct, ordinary distractor session summaries on other harmless life topics. Distractors are converted locally into short sessions and must not repeat the current problem.

Chronology is strict: recent_dialogue happens before current_user_text. Never copy current_user_text into recent_dialogue, and never let a recent seeker turn contain the current turn's literal request or boundary phrase. Every text field must be distinct after lowercasing and whitespace normalization.

The component construction modes below are mandatory content constraints. They override the surface challenge family whenever the two appear to pull in different directions. The surface family changes wording or ambiguity only; it must never change whether a component has an opportunity. These modes are private writing constraints, not labels. Make them observable through ordinary wording without naming them:
- MP preference_incremental: include at least one MP_PREFERENCE and no MP_PROFILE; a stable support-format preference matters now but is not restated verbatim now.
- MP profile_incremental: include at least one MP_PROFILE and no MP_PREFERENCE; a stable personal fact constrains interpretation now but is not restated verbatim now.
- MP redundant_current_fact: include a profile/preference entry whose value is already explicit in current dialogue.
- MP scope_conflict: include at least one MP_PREFERENCE whose old preference conflicts with the current explicit request or boundary.
- MP wrong_entity: include at least one MP_PROFILE; the fact is about the user but the current issue concerns another person.
- MP no_suitable_profile_candidate: omit all profile/preference entries.
- MS prior_distinction/prior_outcome/unfinished_goal: one faithful prior-session summary supplies that exact kind of continuity.
- MS resolved_prior_issue/same_topic_wrong_goal/redundant_current_summary/no_suitable_session_candidate: summaries should instantiate the named reason not to rely on MS; in the no-suitable mode, all summaries are lexically unrelated to the current goal.
- ME reusable_action_result: one prior seeker turn must explicitly use an action verb such as "tried", "wrote", "walked", or "asked", and an observed result word such as "helped", "eased", "worked", or "made it easier". The same action is a safe, tentative option for the current goal.
- ME reusable_action_mechanism: one prior seeker turn must explicitly contain the user's action and why it worked, using wording such as "because", "so that", or "what helped was". The mechanism safely transfers to the current goal.
- ME context_event_only: a topically related prior seeker episode contains background only, with no user action and no observed result.
- ME unresolved_event: a topically related prior seeker episode explicitly says "still unresolved" or "still struggling" and contains no successful result.
- ME wrong_entity_or_goal: a lexically attractive prior episode concerns a different person or a materially different current goal.
- ME no_suitable_event_candidate: all prior seeker episodes use unrelated topics and vocabulary; none supplies a plausible event candidate.
- RS listen_move_fit: current_user_text must literally say that the user only wants the supporter to listen and does not want advice.
- RS reflection_move_fit: current_user_text must literally ask "help me put this feeling into words" or a direct close paraphrase.
- RS clarification_move_fit: current_user_text must literally invite "one focused question".
- RS one_option_move_fit: current_user_text must literally invite "one small optional idea".
- RS routine_closing_or_phatic: current_user_text is only "Thanks for listening." or a routine goodbye, with no substantive support need.
- RS explicit_stop: current_user_text literally asks to "end this conversation now" or "stop this conversation now".
- RS strategy_move_already_present: the visible supporter turn has already performed the best-fitting atomic move, and the current turn only acknowledges it without adding a new need.
- RS no_bank_scope_match: the current turn is a purely factual or logistical request with no emotional-support move to add. Keep it harmless and non-high-stakes.

MS and ME are deliberately different: a supporter-derived distinction may appear in a faithful summary even when the seeker's raw episode contains no reusable action-result; conversely, a seeker may report a useful action-result while the session summary remains broad. For any no_suitable_* mode, keep the corresponding source text lexically unrelated to the current turn so the formal retriever can truly abstain. Never fabricate facts in the summary that the exchange cannot support."""
    user = f"""Write one state with these constraints:
- surface challenge family (wording only; never overrides component modes): {row['logic_family']}
- harmless topic family: {row['topic_family']}
- total strictly prior sessions after local compilation: {target_sessions}
- exact distractor_session_summaries length: {distractor_count}
- MP construction mode: {plans['MP']['construction_mode']}
- MS construction mode: {plans['MS']['construction_mode']}
- ME construction mode: {plans['ME']['construction_mode']}
- RS construction mode: {plans['RS']['construction_mode']}
- exact topic bridge words: {bridge_words[0]} ; {bridge_words[1]}

Topic bridge contract:
- {'current_user_text must contain both bridge words exactly' if topical_current else 'current_user_text must stay the exact closing/stop sentence and need not contain bridge words'}.
- if MP mode is profile_incremental, one MP_PROFILE value must contain both bridge words exactly.
- if MS mode is prior_distinction, prior_outcome, or unfinished_goal, critical session 1 summary must contain both bridge words exactly.
- if ME mode is reusable_action_result or reusable_action_mechanism, critical session 2 seeker_text must contain both bridge words exactly.
- a no_suitable_session_candidate/no_suitable_event_candidate source must contain neither bridge word.
- use the exact word forms above, not synonyms, plurals, or paraphrases, in every required bridge field.

Mandatory row-specific checklist (all four must be satisfied):
- MP: {_concrete_mode_requirement(component='MP', mode=str(plans['MP']['construction_mode']), intended_bits=intent['intended_bits'])}
- MS: {_concrete_mode_requirement(component='MS', mode=str(plans['MS']['construction_mode']), intended_bits=intent['intended_bits'])}
- ME: {_concrete_mode_requirement(component='ME', mode=str(plans['ME']['construction_mode']), intended_bits=intent['intended_bits'])}
- RS: {_concrete_mode_requirement(component='RS', mode=str(plans['RS']['construction_mode']), intended_bits=intent['intended_bits'])}

Use 0-3 profile_entries. Use field_key beginning stable_preference_ for MP_PREFERENCE and a short snake_case key for MP_PROFILE. Produce exactly 3 critical_prior_sessions, exactly {distractor_count} distractor summaries, 1-2 recent_dialogue exchanges, and one current_user_text. Do not copy wording across fields."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def validate_raw_draft_for_blueprint(
    *, draft: FinalRawStateDraft, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply deterministic, outcome-blind post-schema construction checks."""

    target_sessions = int(row["prior_session_count_target"])
    errors: list[str] = []
    if len(draft.critical_prior_sessions) != 3:
        errors.append("critical_session_count")
    if len(draft.distractor_session_summaries) != target_sessions - 3:
        errors.append("distractor_session_count")

    mp_mode = str(
        row["private_construction_intent"]["component_plans"]["MP"][
            "construction_mode"
        ]
    )
    if mp_mode == "no_suitable_profile_candidate" and draft.profile_entries:
        errors.append("mp_absent_mode_has_profile_entries")
    if mp_mode != "no_suitable_profile_candidate" and not draft.profile_entries:
        errors.append("mp_present_mode_has_no_profile_entries")
    for entry in draft.profile_entries:
        if entry.subtype == "MP_PREFERENCE" and not entry.field_key.startswith(
            "stable_preference_"
        ):
            errors.append("preference_field_key")
        if entry.subtype == "MP_PROFILE" and entry.field_key.startswith(
            "stable_preference_"
        ):
            errors.append("profile_field_key")
    subtypes = {entry.subtype for entry in draft.profile_entries}
    if mp_mode == "preference_incremental" and subtypes != {"MP_PREFERENCE"}:
        errors.append("mp_preference_mode_subtype_mismatch")
    if mp_mode == "profile_incremental" and subtypes != {"MP_PROFILE"}:
        errors.append("mp_profile_mode_subtype_mismatch")
    if mp_mode == "scope_conflict" and "MP_PREFERENCE" not in subtypes:
        errors.append("mp_scope_conflict_has_no_preference")
    if mp_mode == "wrong_entity" and "MP_PROFILE" not in subtypes:
        errors.append("mp_wrong_entity_has_no_profile")

    visible_query = normalize_space(
        " ".join(
            [
                *(exchange.seeker_text for exchange in draft.recent_dialogue),
                *(exchange.supporter_text for exchange in draft.recent_dialogue),
                draft.current_user_text,
            ]
        )
    )
    if mp_mode == "preference_incremental" and not any(
        preference_scope_match_level(visible_query, entry.value) > 0.0
        for entry in draft.profile_entries
        if entry.subtype == "MP_PREFERENCE"
    ):
        errors.append("mp_preference_mode_has_no_observable_scope_match")
    if mp_mode == "profile_incremental" and not any(
        final_typed_content_match_level(visible_query, entry.value) > 0.0
        for entry in draft.profile_entries
        if entry.subtype == "MP_PROFILE"
    ):
        errors.append("mp_profile_mode_has_no_content_bridge")

    me_mode = str(
        row["private_construction_intent"]["component_plans"]["ME"][
            "construction_mode"
        ]
    )
    me_hints = [
        compile_me_structure_metadata(session.seeker_text)["me_subtype_hint"]
        for session in draft.critical_prior_sessions
    ]
    if me_mode in {"reusable_action_result", "reusable_action_mechanism"}:
        if "ME_REUSABLE_OUTCOME" not in me_hints:
            errors.append("me_reusable_mode_has_no_structured_reusable_episode")
        if not any(
            hint == "ME_REUSABLE_OUTCOME"
            and final_typed_content_match_level(visible_query, session.seeker_text)
            == 1.0
            for hint, session in zip(
                me_hints, draft.critical_prior_sessions, strict=True
            )
        ):
            errors.append("me_reusable_mode_has_no_two_word_content_bridge")
    elif me_mode == "unresolved_event":
        if "ME_UNRESOLVED_EVENT" not in me_hints:
            errors.append("me_unresolved_mode_has_no_unresolved_episode")
    elif me_mode == "context_event_only":
        if any(value != "ME_CONTEXT_EVENT" for value in me_hints):
            errors.append("me_context_mode_contains_noncontext_episode")

    ms_mode = str(
        row["private_construction_intent"]["component_plans"]["MS"][
            "construction_mode"
        ]
    )
    summaries = [
        *(session.summary for session in draft.critical_prior_sessions),
        *draft.distractor_session_summaries,
    ]
    seeker_episodes = [
        *(session.seeker_text for session in draft.critical_prior_sessions),
        *draft.distractor_session_summaries,
    ]
    if ms_mode in {"prior_distinction", "prior_outcome", "unfinished_goal"}:
        if not any(
            final_typed_content_match_level(visible_query, summary) == 1.0
            for summary in summaries
        ):
            errors.append("ms_positive_mode_has_no_two_word_content_bridge")
    elif ms_mode == "no_suitable_session_candidate":
        if any(
            final_typed_content_match_level(visible_query, summary) > 0.0
            for summary in summaries
        ):
            errors.append("ms_absent_mode_has_content_overlap")
    if me_mode == "no_suitable_event_candidate" and any(
        final_typed_content_match_level(visible_query, episode) > 0.0
        for episode in seeker_episodes
    ):
        errors.append("me_absent_mode_has_content_overlap")

    rs_mode = str(
        row["private_construction_intent"]["component_plans"]["RS"][
            "construction_mode"
        ]
    )
    current = normalize_space(draft.current_user_text)
    rs_flags = observable_opportunity_flags(current_user_text=current)
    lowered_current = current.lower()
    if rs_mode == "listen_move_fit" and not rs_flags["listen_only"]:
        errors.append("rs_listen_mode_not_observable")
    elif rs_mode == "reflection_move_fit" and not re.search(
        r"\bhelp me put (?:this|the|my) (?:feeling|experience|reaction|tension) into words\b",
        lowered_current,
    ):
        errors.append("rs_reflection_mode_not_observable")
    elif rs_mode == "clarification_move_fit" and not re.search(
        r"\bone focused question\b", lowered_current
    ):
        errors.append("rs_clarification_mode_not_observable")
    elif rs_mode == "one_option_move_fit" and not re.search(
        r"\bone small optional (?:idea|suggestion)\b", lowered_current
    ):
        errors.append("rs_one_option_mode_not_observable")
    elif rs_mode == "routine_closing_or_phatic" and not rs_flags["pure_phatic"]:
        errors.append("rs_routine_closing_mode_not_observable")
    elif rs_mode == "explicit_stop" and not rs_flags["explicit_stop"]:
        errors.append("rs_explicit_stop_mode_not_observable")
    elif rs_mode == "no_bank_scope_match" and not lowered_current.endswith(
        "please answer only that factual question."
    ):
        errors.append("rs_no_bank_scope_mode_not_observable")

    texts = [
        *(entry.value for entry in draft.profile_entries),
        *(
            value
            for session in draft.critical_prior_sessions
            for value in (session.summary, session.seeker_text, session.supporter_text)
        ),
        *draft.distractor_session_summaries,
        *(
            value
            for exchange in draft.recent_dialogue
            for value in (exchange.seeker_text, exchange.supporter_text)
        ),
        draft.current_user_text,
    ]
    normalized = [normalize_for_hash(text) for text in texts]
    if len(normalized) != len(set(normalized)):
        errors.append("exact_duplicate_text")
    joined = "\n".join(texts)
    if _FORBIDDEN_ACRONYM_RE.search(joined) or _FORBIDDEN_META_RE.search(joined):
        errors.append("experimental_meta_language_leak")
    if _HIGH_STAKES_RE.search(joined):
        errors.append("out_of_scope_high_stakes_content")

    report = {
        "protocol": "pm-v1.5-p2-raw-draft-static-validation-v1",
        "status": "PASS" if not errors else "REJECT",
        "errors": sorted(set(errors)),
        "prior_sessions": target_sessions,
        "profile_entries": len(draft.profile_entries),
        "critical_sessions": len(draft.critical_prior_sessions),
        "distractor_sessions": len(draft.distractor_session_summaries),
        "outcome_read": False,
        "h1_gold_read": False,
    }
    return report


_H1_V2_DEFERRED_RS_OBSERVABILITY_ERRORS = frozenset(
    {
        "rs_listen_mode_not_observable",
        "rs_reflection_mode_not_observable",
        "rs_clarification_mode_not_observable",
        "rs_one_option_mode_not_observable",
    }
)


def validate_h1_v2_raw_draft(
    *, draft: FinalRawStateDraft, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate H1-v2 states without turning explicit RS wording into the label.

    V1 required the positive RS construction modes to contain near-literal
    requests such as ``one focused question``.  Those strings made the intended
    answer directly readable from the input.  H1-v2 keeps all of the common
    structural checks but deliberately permits implicit support needs.  Whether
    the retrieved card is applicable and non-redundant is left to Step 1 H1.
    """

    base = validate_raw_draft_for_blueprint(draft=draft, row=row)
    errors = [
        error
        for error in base["errors"]
        if error not in _H1_V2_DEFERRED_RS_OBSERVABILITY_ERRORS
    ]
    plans = row["private_construction_intent"]["component_plans"]
    rs_mode = str(plans["RS"]["construction_mode"])
    current = normalize_space(draft.current_user_text)
    lowered_current = current.lower()

    # These phrases were the v1 label shortcut.  Positive examples must now
    # express the need naturally and indirectly.
    if rs_mode.endswith("_move_fit") and any(
        phrase in lowered_current
        for phrase in (
            "only want you to listen",
            "help me put this feeling into words",
            "one focused question",
            "one small optional idea",
            "one small optional suggestion",
        )
    ):
        errors.append("rs_positive_contains_v1_literal_label_cue")
    if rs_mode == "strategy_move_already_present":
        supporter_text = " ".join(
            exchange.supporter_text for exchange in draft.recent_dialogue
        ).lower()
        if "which one part" not in supporter_text:
            errors.append("rs_redundant_mode_has_no_visible_prior_move")
        if "do not repeat" not in lowered_current:
            errors.append("rs_redundant_mode_has_no_current_nonrepeat_boundary")

    mp_mode = str(plans["MP"]["construction_mode"])
    if mp_mode in {"preference_incremental", "profile_incremental"}:
        for entry in draft.profile_entries:
            if normalize_for_hash(entry.value) in normalize_for_hash(current):
                errors.append("mp_positive_value_repeated_verbatim_in_current")

    report = {
        **base,
        "protocol": "pm-v1.5-p2-h1-v2-raw-draft-static-validation-v1",
        "status": "PASS" if not errors else "REJECT",
        "errors": sorted(set(errors)),
        "v1_literal_rs_label_cues_forbidden": True,
        "rs_applicability_decision_deferred_to_h1": True,
    }
    return report


def compile_raw_user_state(
    *,
    draft: FinalRawStateDraft,
    row: Mapping[str, Any],
    validation_mode: Literal["v1", "h1_v2"] = "v1",
    protocol: str = RAW_GENERATION_PROTOCOL,
) -> dict[str, Any]:
    """Compile provider prose into the shared Evo-style user/session input."""

    validation = (
        validate_h1_v2_raw_draft(draft=draft, row=row)
        if validation_mode == "h1_v2"
        else validate_raw_draft_for_blueprint(draft=draft, row=row)
    )
    if validation["status"] != "PASS":
        raise ValueError(f"raw draft failed static validation: {validation['errors']}")

    sessions: list[dict[str, Any]] = []
    critical = list(draft.critical_prior_sessions)
    distractors = list(draft.distractor_session_summaries)
    # Spread the three critical sessions through the history so relevant
    # evidence is not deterministically newest or oldest at every catalog size.
    critical_positions = {
        0,
        max(1, int(row["prior_session_count_target"]) // 2),
        int(row["prior_session_count_target"]) - 1,
    }
    critical_index = 0
    distractor_index = 0
    for position in range(int(row["prior_session_count_target"])):
        if position in critical_positions and critical_index < len(critical):
            source = critical[critical_index]
            summary = normalize_space(source.summary)
            dialogue = [
                {"role": "seeker", "content": normalize_space(source.seeker_text)},
                {"role": "supporter", "content": normalize_space(source.supporter_text)},
            ]
            critical_index += 1
        else:
            summary = normalize_space(distractors[distractor_index])
            dialogue = [
                {"role": "seeker", "content": summary},
                {
                    "role": "supporter",
                    "content": "That sounds worth noticing. What stood out to you?",
                },
            ]
            distractor_index += 1
        sessions.append(
            {
                "id": f"prior_session_{position + 1:02d}",
                "timestamp": "",
                "summary": summary,
                "dialogue": dialogue,
            }
        )
    if critical_index != 3 or distractor_index != len(distractors):
        raise RuntimeError("raw session interleaving failed")

    basic_info = {
        entry.field_key: normalize_space(entry.value)
        for entry in draft.profile_entries
    }
    if len(basic_info) != len(draft.profile_entries):
        raise ValueError("duplicate profile field_key")
    visible_dialogue = [
        turn
        for exchange in draft.recent_dialogue
        for turn in (
            {"role": "user", "content": normalize_space(exchange.seeker_text)},
            {"role": "assistant", "content": normalize_space(exchange.supporter_text)},
        )
    ]
    return {
        "protocol": protocol,
        "state_id": str(row["state_id"]),
        "user_id": str(row["user_id"]),
        "group_id": str(row["group_id"]),
        "split": str(row["split"]),
        "logic_family": str(row["logic_family"]),
        "topic_family": str(row["topic_family"]),
        "user": {
            "id": str(row["user_id"]),
            "basic_info": basic_info,
            "dialog_history": sessions,
        },
        "visible_dialogue": visible_dialogue,
        "current_user_text": normalize_space(draft.current_user_text),
        "current_session_index": len(sessions) + 1,
        "private_blueprint_index": int(row["blueprint_index"]),
        "actual_rank1_candidates": None,
        "h1_gold": None,
        "construction_intent_is_model_input": False,
    }
