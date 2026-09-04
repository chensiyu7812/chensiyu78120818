"""Zero-API realization of the final H1R semantic-routing states.

The private blueprint balances construction challenges.  This module turns
those challenges into ordinary text without reading H1 decisions, generated
responses, external data, or outcome scores.  Construction intent remains a
test fixture, never a PM label.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .text import normalize_for_hash, normalize_space
from .v1_5_final_dataset_blueprint import H1R_BLUEPRINT_PROTOCOL
from .v1_5_final_raw_generation import (
    TOPIC_BRIDGE_WORDS,
    FinalRawStateDraft,
    _DISTRACTOR_SURFACES,
    _HIGH_STAKES_RE,
    _H1_V2_PROFILE_VALUES,
)
from .v1_5_memory_transport import compile_me_structure_metadata


H1R_RAW_PROTOCOL = "pm-v1.5-p2r-zero-api-raw-state-v1"
H1R_REALIZATION_PROTOCOL = "pm-v1.5-p2r-semantic-state-realizer-v1"

_STYLE_LINES = (
    "It has become noticeable again today.",
    "I keep circling back to it, although the reason is not obvious.",
    "It feels like a knot I have not yet found words for.",
    "I may be looking at it too narrowly, so I want to check my understanding.",
    "Some of the facts are familiar, but I do not want to repeat an old answer.",
    "There are two people in the story, and I want to keep their situations separate.",
    "The topic sounds similar to before, yet what I need from this conversation is different.",
    "There is one detail from before that may genuinely change what would help now.",
)

_SPLIT_STYLE_LEADS = {
    "FIT": "At the moment,",
    "FRESH_CONFIRMATION": "As it comes up again,",
    "SEALED_INTERNAL_TEST": "After revisiting it,",
}

_COMPACT_SPLIT_STYLES = {
    "FIT": "Right now it matters again.",
    "FRESH_CONFIRMATION": "It has surfaced again lately.",
    "SEALED_INTERNAL_TEST": "Revisiting it has changed the emphasis.",
}


def _topic(row: Mapping[str, Any]) -> tuple[str, str, str]:
    first, second = TOPIC_BRIDGE_WORDS[str(row["topic_family"])]
    return first, second, f"{first} {second}"


def _rs_request(*, family: str, mode: str) -> tuple[str, str]:
    """Return current request and the immediately preceding supporter move."""

    positive = mode == "family_goal_fit_safe_nonredundant"
    redundant = mode == "same_family_currently_redundant"
    if family == "Question":
        prior = (
            "Which part of this situation is creating the most pressure for you?"
            if redundant
            else "I am following; you can continue with the part that matters most."
        )
        if positive:
            return (
                "Several parts feel tangled, and a carefully chosen question could help narrow the uncertainty.",
                prior,
            )
        if redundant:
            return (
                "Your last question already helped me identify the hardest part, so please do not ask it again.",
                prior,
            )
        return (
            "Several parts feel tangled, but for this turn I need room to explain without being questioned.",
            prior,
        )
    if family == "Restatement or Paraphrasing":
        prior = (
            "So the situation is disrupting your routine while leaving you unsure what matters most."
            if redundant
            else "I am listening and will stay close to what you actually say."
        )
        if positive:
            return (
                "I need space to say what is happening before deciding what to do, and a concise account would help. Keep it brief.",
                prior,
            )
        if redundant:
            return (
                "I needed space to say it, and that summary already captures my meaning; I want to move beyond repeating it. Keep it brief.",
                prior,
            )
        return (
                "Although I usually need space to say more, this time answer the earlier factual detail only. Keep it brief.",
            prior,
        )
    if family == "Reflection of feelings":
        prior = (
            "It sounds as though the uncertainty feels heavy and frustrating."
            if redundant
            else "I can hear that this has been taking up a lot of attention."
        )
        if positive:
            return (
                "The emotional weight is clear, but the feeling is still hard to name accurately.",
                prior,
            )
        if redundant:
            return (
                "The feeling was hard to name, but you have already named it accurately; another reflection would add nothing.",
                prior,
            )
        return (
            "Although the feeling was hard to name before, I am now only checking what was recorded earlier.",
            prior,
        )
    if family == "Providing Suggestions":
        prior = (
            "One option is to try a very small, reversible adjustment and see what changes."
            if redundant
            else "We can stay with the situation until you decide whether ideas would be useful."
        )
        if positive:
            return (
                "I am ready to consider one manageable step, as long as it remains optional.",
                prior,
            )
        if redundant:
            return (
                "I already have a manageable step from you; I want to examine it before adding another.",
                prior,
            )
        return (
            "A manageable step may matter later, but I am not ready for suggestions yet.",
            prior,
        )
    raise ValueError(f"unsupported RS family: {family}")


def _profile_entries(row: Mapping[str, Any]) -> list[dict[str, str]]:
    plan = row["private_construction_intent"]["component_plans"]["MP"]
    mode = str(plan["construction_mode"])
    family = str(
        row["private_construction_intent"]["component_plans"]["RS"][
            "strategy_family_target"
        ]
    )
    first, second, topic = _topic(row)
    preference = {
        "Question": "prefers the response to use one exploratory question at a time",
        "Restatement or Paraphrasing": "prefers the response to begin with a concise summary before suggestions",
        "Reflection of feelings": "prefers the response to use tentative emotion words close to their own language",
        "Providing Suggestions": "prefers the response to offer one reversible option rather than a directive",
    }[family]
    if mode == "preference_incremental":
        value = preference
        subtype = "MP_PREFERENCE"
        key = "stable_preference_support_format"
    elif mode == "redundant_current_preference":
        value = preference
        subtype = "MP_PREFERENCE"
        key = "stable_preference_support_format"
    elif mode == "preference_scope_conflict":
        value = "prefers the response to contain several detailed suggestions and follow-up questions"
        subtype = "MP_PREFERENCE"
        key = "stable_preference_support_format"
    elif mode in {
        "profile_incremental",
        "redundant_current_profile",
        "profile_wrong_entity_or_goal",
    }:
        value = f"for the {topic} situation, {_H1_V2_PROFILE_VALUES[str(row['topic_family'])]}"
        subtype = "MP_PROFILE"
        key = f"stable_{first}_{second}_constraint"[:40]
    else:
        raise ValueError(f"unsupported H1R MP mode: {mode}")
    return [{"subtype": subtype, "field_key": key, "value": value}]


def _ms_session(row: Mapping[str, Any]) -> dict[str, str]:
    mode = str(
        row["private_construction_intent"]["component_plans"]["MS"][
            "construction_mode"
        ]
    )
    _, _, topic = _topic(row)
    if mode == "prior_distinction_answers_current_goal":
        return {
            "seeker_text": f"Earlier I was trying to understand the {topic} pressure without rushing to a plan.",
            "supporter_text": "You helped separate the emotional uncertainty from the amount of practical work.",
            "summary": f"For the user's {topic}, emotional uncertainty rather than practical workload was the main pressure.",
        }
    if mode == "prior_outcome_answers_factual_recall":
        return {
            "seeker_text": f"I wanted the earlier {topic} conversation to record one clear observation.",
            "supporter_text": "You noted that a brief pause reduced the feeling of being rushed before a decision.",
            "summary": f"In the user's earlier {topic} session, a brief pause reduced pressure before deciding.",
        }
    if mode == "same_topic_wrong_goal":
        return {
            "seeker_text": f"That earlier {topic} exchange was only about comparing dates and logistics.",
            "supporter_text": "You kept the discussion limited to the practical comparison requested then.",
            "summary": f"The earlier {topic} session compared logistics rather than the user's present support goal.",
        }
    if mode == "currently_redundant_session_summary":
        return {
            "seeker_text": f"The {topic} pressure was mainly emotional uncertainty rather than practical workload.",
            "supporter_text": "You recorded that distinction without adding another interpretation.",
            "summary": f"For the user's {topic}, emotional uncertainty rather than practical workload was the main pressure.",
        }
    if mode == "resolved_or_stale_prior_session":
        return {
            "seeker_text": f"The old {topic} issue was resolved and is no longer part of my current concern.",
            "supporter_text": "You confirmed that the earlier issue had fully ended.",
            "summary": f"The old {topic} issue was resolved, completed, and fully settled.",
        }
    if mode == "prior_session_wrong_owner":
        return {
            "seeker_text": f"My friend described their own {topic} concern while I was asking how to support them.",
            "supporter_text": "You kept my friend's situation separate from my own circumstances.",
            "summary": f"The earlier {topic} details belonged to the user's friend, not to the user.",
        }
    raise ValueError(f"unsupported H1R MS mode: {mode}")


def _me_session(row: Mapping[str, Any]) -> dict[str, str]:
    mode = str(
        row["private_construction_intent"]["component_plans"]["ME"][
            "construction_mode"
        ]
    )
    _, _, topic = _topic(row)
    if mode == "reusable_action_result_current_goal":
        seeker = f"For the {topic} issue, I wrote one short note, and it helped make the next step easier."
    elif mode == "reusable_action_mechanism_current_goal":
        seeker = f"For the {topic} issue, I paused before responding; what helped was limiting myself to one reversible step."
    elif mode == "reusable_outcome_wrong_entity_or_goal":
        seeker = f"My friend tried writing one {topic} note, and it helped them make a different decision."
    elif mode == "context_event_same_topic_no_reusable_result":
        seeker = f"The {topic} issue was present in the background during that week."
    elif mode == "unresolved_event_no_reusable_result":
        seeker = f"The {topic} issue was still unresolved, and I was still struggling with it."
    else:
        raise ValueError(f"unsupported H1R ME mode: {mode}")
    return {
        "seeker_text": seeker,
        "supporter_text": "You kept that earlier episode distinct from claims about the present.",
        "summary": f"The user described an earlier {topic} episode without assuming it still applies now.",
    }


def realize_h1r_draft(row: Mapping[str, Any]) -> FinalRawStateDraft:
    if row.get("protocol") != H1R_BLUEPRINT_PROTOCOL:
        raise ValueError("not an H1R blueprint row")
    plans = row["private_construction_intent"]["component_plans"]
    first, second, topic = _topic(row)
    rs_family = str(plans["RS"]["strategy_family_target"])
    rs_request, recent_supporter = _rs_request(
        family=rs_family,
        mode=str(plans["RS"]["construction_mode"]),
    )
    friend_focus = str(plans["MP"]["construction_mode"]) == "profile_wrong_entity_or_goal"
    subject = "My friend is" if friend_focus else "I am"
    style = (
        f"{_SPLIT_STYLE_LEADS[str(row['split'])]} "
        f"{_STYLE_LINES[int(row['blueprint_index']) % len(_STYLE_LINES)].lower()}"
    )
    current_parts = [
        f"{subject} dealing with the {topic} situation.",
        style,
        rs_request,
    ]
    mp_mode = str(plans["MP"]["construction_mode"])
    if mp_mode == "redundant_current_preference":
        current_parts.append(_profile_entries(row)[0]["value"].replace("prefers", "I prefer"))
    elif mp_mode == "preference_scope_conflict":
        current_parts.append("Keep this response brief and do not give me a list or several questions.")
    elif mp_mode == "preference_incremental":
        current_parts.append("Please let the response fit the request I am making here.")
    elif mp_mode == "redundant_current_profile":
        current_parts.append(_profile_entries(row)[0]["value"].replace("for the", "In the"))
    elif mp_mode == "profile_incremental":
        current_parts.append("Please keep any response realistic for my actual circumstances.")

    ms_mode = str(plans["MS"]["construction_mode"])
    if ms_mode == "prior_distinction_answers_current_goal":
        current_parts.append("I want to know which kind of pressure matters most now.")
    elif ms_mode == "prior_outcome_answers_factual_recall":
        current_parts.append("Use the specific earlier observation rather than guessing from the topic.")
    elif ms_mode == "currently_redundant_session_summary":
        current_parts.append("The emotional uncertainty, not the practical workload, is already clearly the main pressure.")

    if bool(plans["ME"]["past_help_invitation_visible"]):
        current_parts.append("A past approach may matter if it truly matches.")
    else:
        current_parts.append("Do not assume an earlier episode answers today.")
    current = normalize_space(" ".join(current_parts))
    # Some multi-component counterfactuals legitimately need several clauses.
    # Compress only the optional style clause before touching any semantic cue.
    if len(current) > 420:
        current = normalize_space(
            " ".join(
                [
                    current_parts[0],
                    _COMPACT_SPLIT_STYLES[str(row["split"])],
                    *current_parts[2:],
                ]
            )
        )
    if len(current) > 420:
        raise ValueError("H1R current text exceeds schema limit")

    target_sessions = int(row["prior_session_count_target"])
    distractor_count = target_sessions - 3
    distractors = [
        normalize_space(text)
        for text in _DISTRACTOR_SURFACES[:distractor_count]
    ]
    background = {
        "summary": "An earlier conversation compared ceramic mugs at a local craft market.",
        "seeker_text": "I compared two ceramic mugs during a visit to a local craft market.",
        "supporter_text": "You asked which design detail stood out to me.",
    }
    return FinalRawStateDraft.model_validate(
        {
            "profile_entries": _profile_entries(row),
            "critical_prior_sessions": [_ms_session(row), _me_session(row), background],
            "distractor_session_summaries": distractors,
            "recent_dialogue": [
                {
                    "seeker_text": f"Earlier today I began describing the {first} {second} situation.",
                    "supporter_text": recent_supporter,
                }
            ],
            "current_user_text": current,
        }
    )


def validate_h1r_draft(
    *, draft: FinalRawStateDraft, row: Mapping[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    target = int(row["prior_session_count_target"])
    if len(draft.critical_prior_sessions) != 3:
        errors.append("critical_session_count")
    if len(draft.distractor_session_summaries) != target - 3:
        errors.append("distractor_session_count")
    target_mp = str(
        row["private_construction_intent"]["component_plans"]["MP"][
            "candidate_subtype_target"
        ]
    )
    if {entry.subtype for entry in draft.profile_entries} != {target_mp}:
        errors.append("mp_subtype_target_not_realized")
    target_me = str(
        row["private_construction_intent"]["component_plans"]["ME"][
            "candidate_subtype_target"
        ]
    )
    actual_me = compile_me_structure_metadata(
        draft.critical_prior_sessions[1].seeker_text
    )["me_subtype_hint"]
    if actual_me != target_me:
        errors.append("me_subtype_target_not_realized")
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
        errors.append("exact_duplicate_text_within_state")
    if _HIGH_STAKES_RE.search("\n".join(texts)):
        errors.append("out_of_scope_high_stakes_content")
    return {
        "protocol": H1R_REALIZATION_PROTOCOL,
        "status": "PASS" if not errors else "REJECT",
        "errors": sorted(set(errors)),
        "human_labels_read": 0,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
    }


def compile_h1r_raw_state(
    *, draft: FinalRawStateDraft, row: Mapping[str, Any]
) -> dict[str, Any]:
    validation = validate_h1r_draft(draft=draft, row=row)
    if validation["status"] != "PASS":
        raise ValueError(validation)
    critical = list(draft.critical_prior_sessions)
    distractors = list(draft.distractor_session_summaries)
    target = int(row["prior_session_count_target"])
    positions = {0, max(1, target // 2), target - 1}
    sessions: list[dict[str, Any]] = []
    ci = di = 0
    for position in range(target):
        if position in positions and ci < 3:
            source = critical[ci]
            summary = normalize_space(source.summary)
            dialogue = [
                {"role": "seeker", "content": normalize_space(source.seeker_text)},
                {"role": "supporter", "content": normalize_space(source.supporter_text)},
            ]
            ci += 1
        else:
            summary = normalize_space(distractors[di])
            dialogue = [
                {"role": "seeker", "content": summary},
                {"role": "supporter", "content": "You asked what stood out in that ordinary experience."},
            ]
            di += 1
        sessions.append(
            {
                "id": f"prior_session_{position + 1:02d}",
                "timestamp": "",
                "summary": summary,
                "dialogue": dialogue,
            }
        )
    if ci != 3 or di != len(distractors):
        raise RuntimeError("H1R session interleaving failed")
    return {
        "protocol": H1R_RAW_PROTOCOL,
        "state_id": str(row["state_id"]),
        "user_id": str(row["user_id"]),
        "group_id": str(row["group_id"]),
        "split": str(row["split"]),
        "logic_family": str(row["logic_family"]),
        "topic_family": str(row["topic_family"]),
        "user": {
            "id": str(row["user_id"]),
            "basic_info": {
                entry.field_key: normalize_space(entry.value)
                for entry in draft.profile_entries
            },
            "dialog_history": sessions,
        },
        "visible_dialogue": [
            turn
            for exchange in draft.recent_dialogue
            for turn in (
                {"role": "user", "content": normalize_space(exchange.seeker_text)},
                {"role": "assistant", "content": normalize_space(exchange.supporter_text)},
            )
        ],
        "current_user_text": normalize_space(draft.current_user_text),
        "current_session_index": target + 1,
        "private_blueprint_index": int(row["blueprint_index"]),
        "actual_rank1_candidates": None,
        "h1_gold": None,
        "construction_intent_is_model_input": False,
    }


def audit_h1r_raw_states(
    *, rows: Sequence[Mapping[str, Any]], raw_states: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    failures: list[str] = []
    if len(raw_states) != 96 or len({row["state_id"] for row in raw_states}) != 96:
        failures.append("raw_states_not_96_unique")
    if set(row["state_id"] for row in rows) != set(row["state_id"] for row in raw_states):
        failures.append("blueprint_raw_state_id_mismatch")
    public = Counter(
        (
            tuple((turn["role"], turn["content"]) for turn in raw["visible_dialogue"]),
            raw["current_user_text"],
        )
        for raw in raw_states
    )
    if any(value > 1 for value in public.values()):
        failures.append("duplicate_public_state_surface")
    per_split_scale = Counter(
        (raw["split"], len(raw["user"]["dialog_history"])) for raw in raw_states
    )
    for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST"):
        if per_split_scale[(split, 4)] != 16 or per_split_scale[(split, 34)] != 16:
            failures.append(f"{split}_history_scale_not_balanced")
    return {
        "protocol": "pm-v1.5-p2r-zero-api-raw-state-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "states": len(raw_states),
        "unique_public_state_surfaces": len(public),
        "history_scale_by_split": {
            split: {
                "small_4": per_split_scale[(split, 4)],
                "evo_like_large_34": per_split_scale[(split, 34)],
            }
            for split in ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST")
        },
        "human_labels_read": 0,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
    }
