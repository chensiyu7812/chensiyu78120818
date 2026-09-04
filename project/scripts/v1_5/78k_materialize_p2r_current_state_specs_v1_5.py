#!/usr/bin/env python3
"""Create P2R current states independently from hidden catalog answers.

The positive surfaces use only a topic/applicability scope.  Only the
explicit ``current_redundant`` negative deliberately repeats a stored MS
observation, because redundancy is the variable under test there.  The file
contains construction provenance but no worth-opening or response outcome.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import read_json, stable_hex, write_json, write_jsonl, read_jsonl
from metacom_pm.v1_5_v5_3_p2r_learning_blueprint import (
    P2RCurrentStateSpec,
    make_state_identity,
    normalize_worker_catalog,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_longitudinal_catalog_v1"
OUTPUT_DIR = ROOT / "outputs/pm_v1_5_v5_3_p2r_current_state_specs_v1"


PROFILE_SCOPE = {
    "name": "my name and the way I am addressed",
    "age": "my age and stage of life",
    "gender": "my gender and how I describe myself",
    "job": "my job and work situation",
    "education": "my education and study situation",
    "nationality": "my nationality and cultural context",
    "location": "my location and where I live",
}


def _pick(values: list[str], *parts: str) -> str:
    return values[int(stable_hex(*parts, n=8), 16) % len(values)]


_SCENE_OPENERS = [
    "After finishing a routine errand",
    "During a quiet part of the afternoon",
    "While taking a short break",
    "Before I return to the rest of my day",
    "After putting my phone aside for a moment",
    "With a little space to think",
    "At the end of an otherwise ordinary morning",
    "While sitting somewhere private",
    "After dealing with a small unrelated task",
    "During a free moment between commitments",
]
_SCENE_QUALIFIERS = [
    "I noticed this was still on my mind",
    "I want to focus on one part of this",
    "the same concern came back into view",
    "I realized I needed a bounded response",
    "I decided to say this plainly",
    "I have enough attention for one careful exchange",
    "I want to separate this from the surrounding noise",
    "I am trying not to rush past the actual issue",
    "I can give this a little attention now",
    "I want to keep the immediate question clear",
]
_SCENE_BOUNDARIES = [
    "I can handle one focused response",
    "I want to avoid making the exchange larger than it needs to be",
    "I am keeping the question separate from everything else today",
    "I have not repeated any stored detail in this message",
    "I want the next reply to stay grounded in the actual request",
    "I am trying to leave room to correct an assumption",
    "I would rather not rush from context to a conclusion",
    "I can revisit the broader background another time",
    "I want one response before deciding whether to continue",
    "I am focusing on what would matter in this turn",
]
_SCENE_STANCES = [
    "I am not asking you to infer more than I have said",
    "I want any uncertainty to remain visible",
    "I can correct the response if it misses the point",
    "I would like the reply to distinguish context from conclusion",
    "I am leaving the wider story outside this turn",
    "I want the response to stay provisional where needed",
    "I am keeping this separate from any unrelated concern",
    "I want a response that can be revised if the fit is wrong",
    "I am not treating an earlier pattern as automatically current",
    "I want the immediate purpose of the reply to remain clear",
]


def _scene_prefix(user_id: str, family: str) -> str:
    opener = _pick(_SCENE_OPENERS, "scene_open", user_id, family)
    qualifier = _pick(_SCENE_QUALIFIERS, "scene_qual", user_id, family)
    boundary = _pick(_SCENE_BOUNDARIES, "scene_bound", user_id, family)
    stance = _pick(_SCENE_STANCES, "scene_stance", user_id, family)
    return f"{opener}, {qualifier}. {boundary}. {stance}."


def _topic_surface(kind: str, topic: str, user_id: str) -> str:
    templates = {
        "ms_positive": [
            "Last time we talked about {topic}, we reached a specific observation that I cannot fully recall. Could you bring that thread back without assuming it still applies now?",
            "Could we return to {topic}? I remember that an earlier session reached one concrete observation, but not what it was; please keep it clearly in the past.",
            "As we pick up {topic} again, remind me of the specific earlier distinction we recorded and let me check whether it remains relevant.",
            "I want continuity on {topic}: what was the one concrete observation from before? Please do not turn it into a claim about today without checking.",
        ],
        "ms_present": [
            "{topic} is on my mind today, but please stay with what I am saying now instead of bringing in a previous session.",
            "I need to describe today's version of {topic} first; do not import an older account into this response.",
            "For {topic}, listen to the present update only. We can compare it with older notes later.",
            "Something changed around {topic}. Please respond to this turn without trying to restore continuity yet.",
        ],
        "me_positive": [
            "{topic} has come up again. I am open to one small optional step if something I tried before genuinely fits.",
            "I could consider one low-pressure option for {topic}, but only if a past result really transfers rather than merely sharing the topic.",
            "With {topic} back in view, one reversible idea based on something that actually helped before would be welcome.",
            "Can we find one modest starting point for {topic} from a prior action and result, while treating it as optional now?",
        ],
        "me_decline": [
            "{topic} has come up again. Please do not offer or reuse a step; I only want the feeling reflected this turn.",
            "For this turn on {topic}, no advice or past action. I need room to say what it feels like now.",
            "Do not reuse an earlier approach for {topic} yet; please stay with the emotion I am describing.",
            "I am not ready to consider a step for {topic}. A concise reflection is enough for now.",
        ],
        "me_unknown": [
            "{topic} has come up again. I want to explain what changed before deciding whether to do anything.",
            "I need to make sense of today's {topic} first; I have not decided whether I want an idea.",
            "Let me describe the current shape of {topic} before we move toward or away from action.",
            "I am still working out what {topic} means today, so do not assume either that I want a step or that I reject one.",
        ],
    }
    return _pick(templates[kind], kind, topic, user_id).format(topic=topic)


def _state(
    *,
    component: str,
    user_id: str,
    family: str,
    variant: str,
    condition: str,
    session_index: int,
    dialogue: list[dict[str, str]],
    response_act: str,
    note: str,
    already_executed: list[str] | None = None,
    interaction_key: str | None = None,
) -> P2RCurrentStateSpec:
    state_id, group_id = make_state_identity(
        component=component,
        catalog_user_id=user_id,
        semantic_family=family,
        variant=variant,
    )
    rendered_dialogue = [dict(turn) for turn in dialogue]
    if user_id.startswith("p2r_cat_"):
        rendered_dialogue[-1]["content"] = (
            f"{_scene_prefix(user_id, family)} {rendered_dialogue[-1]['content']}"
        )
    return P2RCurrentStateSpec(
        state_id=state_id,
        catalog_user_id=user_id,
        current_subject_entity_id=user_id,
        component=component,
        semantic_family=family,
        counterfactual_group_id=group_id,
        state_condition=condition,
        current_session_index=session_index,
        visible_dialogue=rendered_dialogue,
        already_executed_move_ids=already_executed or [],
        intended_response_act=response_act,
        construction_note=note,
        interaction_key=interaction_key,
        interaction_variant=("full_four_component" if interaction_key else None),
    )


def _memory_states(catalog: list[Any], users: list[dict[str, Any]]) -> list[P2RCurrentStateSpec]:
    rows: list[P2RCurrentStateSpec] = []
    by_user: dict[str, list[Any]] = defaultdict(list)
    for item in catalog:
        by_user[item.catalog_user_id].append(item)
    session_count = {row["user_id"]: int(row["n_sessions"]) for row in users}
    for user in users:
        user_id = str(user["user_id"])
        current_session = session_count[user_id] + 1
        items = by_user[user_id]
        topics = list(user["recurring_topics"])
        ms_by_topic = {
            topic: [item for item in items if item.component == "MS" and item.topic_key == topic]
            for topic in topics
        }
        me_by_topic = {
            topic: [item for item in items if item.component == "ME" and item.topic_key == topic]
            for topic in topics
        }
        for topic in topics:
            if ms_by_topic[topic]:
                family = f"ms::{topic}"
                rows.extend(
                    [
                        _state(
                            component="MS", user_id=user_id, family=family,
                            variant="continuity", condition="positive_opportunity",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": _topic_surface("ms_positive", topic, user_id)}],
                            response_act="restore one specific earlier observation",
                            note="topic-only continuity request; hidden observation absent",
                        ),
                        _state(
                            component="MS", user_id=user_id, family=family,
                            variant="present_only", condition="goal_mismatch",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": _topic_surface("ms_present", topic, user_id)}],
                            response_act="reflect only the present disclosure",
                            note="same topic, explicit present-only boundary",
                        ),
                    ]
                )
                redundant_item = max(ms_by_topic[topic], key=lambda item: item.created_session)
                rows.append(
                    _state(
                        component="MS", user_id=user_id, family=family,
                        variant="redundant", condition="current_redundant",
                        session_index=current_session,
                        dialogue=[{"role": "user", "content": f"I already have the old note here: {redundant_item.literal_text} Please respond only to what has changed today."}],
                        response_act="address only the new change",
                        note="intentional redundancy negative; exact old note is visible",
                    )
                )
            if me_by_topic[topic]:
                family = f"me::{topic}"
                rows.extend(
                    [
                        _state(
                            component="ME", user_id=user_id, family=family,
                            variant="invites", condition="positive_opportunity",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": _topic_surface("me_positive", topic, user_id)}],
                            response_act="offer at most one evidence-grounded option",
                            note="topic-only action invitation; past action/result absent",
                        ),
                        _state(
                            component="ME", user_id=user_id, family=family,
                            variant="declines", condition="explicit_decline",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": _topic_surface("me_decline", topic, user_id)}],
                            response_act="reflect without an action",
                            note="explicit action refusal on the same topic",
                        ),
                        _state(
                            component="ME", user_id=user_id, family=family,
                            variant="unknown", condition="unknown_readiness",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": _topic_surface("me_unknown", topic, user_id)}],
                            response_act="invite clarification without presuming action readiness",
                            note="no invitation or refusal; UNKNOWN must remain distinct",
                        ),
                    ]
                )

        active_mp = [
            item
            for item in items
            if item.component == "MP" and item.is_active_strict_past(current_session)
        ]
        for item in active_mp:
            context_topic = _pick(topics, "mp_context", user_id, item.memory_id)
            if item.subtype == "MP_PROFILE":
                scope = PROFILE_SCOPE[item.field_type]
                family = f"mp_profile::{item.field_type}"
                rows.extend(
                    [
                        _state(
                            component="MP", user_id=user_id, family=family,
                            variant="applicable", condition="positive_opportunity",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": f"While I am dealing with {context_topic}, could you suggest one practical option that is realistic for {scope}? I have not repeated those details here."}],
                            response_act="offer one feasible profile-sensitive option",
                            note="applicability scope is visible; stored field value is hidden",
                        ),
                        _state(
                            component="MP", user_id=user_id, family=family,
                            variant="no_personalization", condition="goal_mismatch",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": f"While I explain {context_topic}, I am thinking about {scope}; for this turn just reflect the feeling in my own words and do not tailor advice or make a plan."}],
                            response_act="reflect without profile-based advice",
                            note="same applicability scope but no profile-dependent act",
                        ),
                    ]
                )
            elif item.subtype == "MP_PREFERENCE":
                family = f"mp_preference::{item.field_value}"
                phrase = item.literal_text.split("prefers ", 1)[-1]
                rows.extend(
                    [
                        _state(
                            component="MP", user_id=user_id, family=family,
                            variant="implicit", condition="positive_opportunity",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": f"I would like your response while I make sense of {context_topic}. Please do not assume a diagnosis or cause."}],
                            response_act="respond using the applicable stable presentation preference",
                            note="preference is not restated; structured fallback discovery is required",
                        ),
                        _state(
                            component="MP", user_id=user_id, family=family,
                            variant="redundant", condition="current_redundant",
                            session_index=current_session,
                            dialogue=[{"role": "user", "content": f"While we discuss {context_topic}, I explicitly want {phrase}. Please follow that current instruction."}],
                            response_act="follow the explicit current presentation request",
                            note="current turn already states the full stored preference increment",
                        ),
                    ]
                )
    return rows


RS_SURFACES = {
    "open_expression": (
        "AM01_invite_open_expression",
        [
            "I don't know where to start; something has been bothering me all day.",
            "There is a lot in my head and I am not sure how to begin saying it.",
            "Can I talk through something? I cannot yet tell which part matters most.",
            "I need somewhere to start with this, but the concern is still hard to name.",
        ],
        [
            "Please just note that I am here; do not invite me to begin again because I have already started explaining.",
            "I have already begun the account, so stay with it rather than opening the floor again.",
            "The concern is clear now; another broad invitation would only repeat the previous move.",
            "Let me continue from the detail I just gave instead of asking me to start anywhere.",
        ],
    ),
    "focused_clarification": (
        "AM02_ask_one_focused_clarification",
        [
            "This is complicated and hard to explain; one careful question might help.",
            "I am not sure how to describe the part that is bothering me most.",
            "The situation depends on one detail I have not managed to say clearly.",
            "I can tell something is off, but I need help narrowing what I mean.",
        ],
        [
            "You already asked the clarifying question and I answered it; please do not ask it again.",
            "That focused question was the previous move, and I have just responded to it.",
            "Please use the answer I gave rather than narrowing the issue with another question.",
            "I want a response to the clarification already on the page, not a second probe.",
        ],
    ),
    "paraphrase": (
        "AM04_tentative_paraphrase_check",
        [
            "Part of me wants to step back, but part of me worries that doing so will make things worse.",
            "I want to be honest, yet I also want to protect the relationship.",
            "I know I need rest, though I feel guilty whenever I actually take it.",
            "I want a change, but I am afraid of losing what is familiar.",
        ],
        [
            "Please just listen to the next part; do not paraphrase or check an interpretation yet.",
            "I need room to add detail before you try to put the concern into your own words.",
            "Do not summarize the tension yet; I am still correcting how I want to describe it.",
            "Hold off on an interpretation while I finish the thought.",
        ],
    ),
    "validation": (
        "AM05_grounded_validation",
        [
            "I feel anxious and exhausted by how long this has been going on.",
            "This has been painful, and I am frustrated with myself for still reacting.",
            "I feel lonely and confused even though other people are around.",
            "It hurts, and I am tired of pretending that it does not.",
        ],
        [
            "I have not named a feeling yet; please do not assign one to me before I explain.",
            "The facts are all I can give right now, so avoid adding an emotion I did not state.",
            "Please do not validate a feeling on my behalf before I say what the feeling is.",
            "I need to describe the event first rather than receive an emotional interpretation.",
        ],
    ),
    "micro_step": (
        "AM10_offer_one_optional_micro_step",
        [
            "What can I try that stays small and reversible?",
            "Could you suggest one low-pressure step rather than a plan?",
            "I am open to one small optional idea I can evaluate afterward.",
            "How could I begin without committing to a whole sequence?",
        ],
        [
            "I only need you to listen right now; no advice or suggestions this turn.",
            "An idea may matter later, but please do not give me a step in this response.",
            "Stay with what I am saying; I am not ready to consider an option.",
            "No experiment or action yet. I want the difficulty acknowledged first.",
        ],
    ),
    "transition": (
        "AM14_supportive_transition",
        [
            "I think I have said enough for now and want to continue another time.",
            "Can we pause here and change the subject without reopening everything?",
            "I am done talking about this for today, but I do not want an abrupt ending.",
            "I need to wrap this up gently and come back later.",
        ],
        [
            "I am still explaining and do not want to close or change focus yet.",
            "Please stay with this topic; I have not signaled that I am done.",
            "Do not wrap up yet, because the most important part comes next.",
            "I want to continue rather than pause or move to another subject.",
        ],
    ),
}


def _rs_states() -> list[P2RCurrentStateSpec]:
    rows: list[P2RCurrentStateSpec] = []
    for family, (move_id, positives, negatives) in RS_SURFACES.items():
        for index, positive in enumerate(positives):
            user_id = f"p2r_rs_{family}_{index}"
            rows.extend(
                [
                    _state(
                        component="RS", user_id=user_id, family=f"rs::{family}",
                        variant=f"positive_{index}", condition="positive_opportunity",
                        session_index=2,
                        dialogue=[{"role": "user", "content": positive}],
                        response_act=f"execute {move_id} once",
                        note="natural card-level opportunity",
                    ),
                    _state(
                        component="RS", user_id=user_id, family=f"rs::{family}",
                        variant=f"negative_{index}", condition="goal_mismatch",
                        session_index=2,
                        dialogue=[{"role": "user", "content": negatives[index]}],
                        response_act="respect the current non-use boundary",
                        note="same family counterfactual without current card benefit",
                        already_executed=([move_id] if family == "focused_clarification" else []),
                    ),
                ]
            )
    return rows


def _interaction_states(catalog: list[Any], users: list[dict[str, Any]]) -> list[P2RCurrentStateSpec]:
    rows: list[P2RCurrentStateSpec] = []
    by_user: dict[str, list[Any]] = defaultdict(list)
    for item in catalog:
        by_user[item.catalog_user_id].append(item)
    field_for_topic = {
        "job transition": "job", "career change": "job", "workplace conflict": "job",
        "relocation abroad": "location", "exam anxiety": "education",
        "starting therapy": "location", "chronic pain": "age",
    }
    count = 0
    for user in users:
        if count >= 12:
            break
        user_id = str(user["user_id"])
        items = by_user[user_id]
        active_fields = {
            item.field_type
            for item in items
            if item.component == "MP" and item.subtype == "MP_PROFILE"
            and item.is_active_strict_past(int(user["n_sessions"]) + 1)
        }
        for topic in user["recurring_topics"]:
            field = field_for_topic.get(topic)
            if field not in active_fields:
                continue
            if not any(item.component == "MS" and item.topic_key == topic for item in items):
                continue
            if not any(item.component == "ME" and item.topic_key == topic for item in items):
                continue
            key = f"ix::{user_id}::{topic}"
            text = (
                f"Last time we talked about {topic}. Could you suggest one small, "
                f"reversible option that is realistic for my {field} situation, while "
                "treating anything from the past as something to verify rather than a current fact?"
            )
            dialogue = [
                {"role": "assistant", "content": "We can keep the next response bounded and check any assumption."},
                {"role": "user", "content": text},
            ]
            for component in ("MP", "MS", "ME", "RS"):
                rows.append(
                    _state(
                        component=component,
                        user_id=user_id,
                        family=f"interaction::{topic}",
                        variant=f"full_{component.lower()}",
                        condition="positive_opportunity",
                        session_index=int(user["n_sessions"]) + 1,
                        dialogue=dialogue,
                        response_act="coordinate applicable profile, continuity, outcome and one atomic move",
                        note="shared four-component surface; no stored value or past answer copied",
                        interaction_key=key,
                    )
                )
            count += 1
            break
    return rows


def main() -> None:
    raw_catalog = read_jsonl(CATALOG_DIR / "catalog.jsonl")
    catalog = normalize_worker_catalog(raw_catalog)
    users = read_json(CATALOG_DIR / "users.json")
    rows = _memory_states(catalog, users) + _rs_states() + _interaction_states(catalog, users)
    state_ids = [row.state_id for row in rows]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("duplicate state IDs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        OUTPUT_DIR / "state_specs.jsonl",
        [row.model_dump(mode="json") for row in rows],
    )
    group_conditions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_conditions[row.counterfactual_group_id].add(row.state_condition)
    report = {
        "protocol": "pm-v1.5-v5.3-p2r-current-state-specs-v1",
        "status": "DRAFT_READY_FOR_REAL_CANDIDATE_MATERIALIZATION",
        "n_states": len(rows),
        "n_groups": len(group_conditions),
        "component_counts": {
            component: sum(row.component == component for row in rows)
            for component in ("MP", "MS", "ME", "RS")
        },
        "n_interaction_keys": len({row.interaction_key for row in rows if row.interaction_key}),
        "groups_with_multiple_conditions": sum(len(value) >= 2 for value in group_conditions.values()),
        "groups_with_single_condition": sum(len(value) < 2 for value in group_conditions.values()),
        "external_source_text_copied": False,
        "worth_opening_or_response_outcome_present": False,
        "api_calls": 0,
    }
    write_json(OUTPUT_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
