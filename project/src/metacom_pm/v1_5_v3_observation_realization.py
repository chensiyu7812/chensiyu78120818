"""Deterministic visible-state realization for the orthogonal Observation plan."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .text import normalize_for_hash, normalize_space
from .v1_5_memory_transport import compile_me_structure_metadata
from .v1_5_v3_observation_orthogonal import (
    FACTORS,
    PROTOCOL as BLUEPRINT_PROTOCOL,
)
from .v1_5_v3_state_realization import TOPIC_WORDS


PROTOCOL = "pm-v1.5-v3-observation-orthogonal-visible-state-v1"
REPAIRED_PROTOCOL = "pm-v1.5-v3-observation-orthogonal-visible-state-v2"

PREFIX_TEXT = {
    "prefix_0": "After an ordinary errand, I noticed the issue again.",
    "prefix_1": "While making tea, I realized this still needed a careful response.",
    "prefix_2": "During a quiet break, the question came back into focus.",
    "prefix_3": "After checking my calendar, I wanted to describe this precisely.",
    "prefix_4": "Before dinner, I noticed I was close to rushing the issue.",
    "prefix_5": "While tidying my desk, I tried to separate facts from assumptions.",
    "prefix_6": "On the way home, I decided to keep the current goal narrow.",
    "prefix_7": "After closing my laptop, I wanted a grounded response to this.",
}

NEUTRAL_FILLER_TEN = "These ordinary details add no claim about the central request."
NEUTRAL_FILLER_ELEVEN = "These ordinary details add no claim about the central request today."
NEUTRAL_FILLER_REMAINDERS = {
    1: "Separately.",
    2: "Nothing else.",
    3: "Nothing else changes.",
    4: "No other detail matters.",
    5: "No additional conclusion is intended.",
    6: "No further conclusion should be inferred.",
    7: "No further conclusion should be inferred here.",
    8: "No further conclusion should be inferred from this.",
    9: "No further conclusion should be inferred from this detail.",
}

DISTRACTORS = (
    "a library display of maps",
    "a recipe for vegetable soup",
    "a documentary about forests",
    "a postcard from a coastal town",
    "a quiet neighborhood bakery",
    "a magazine article about bridges",
    "a packet of seeds for spring",
    "a radio program about local history",
)


def _pad_words(text: str, target: int) -> str:
    words = normalize_space(text).split()
    if len(words) > target:
        raise ValueError(f"surface exceeds frozen word target {target}: {len(words)}")
    remaining = target - len(words)
    sentences: list[str] = []
    while remaining >= 10:
        if remaining == 11:
            sentences.append(NEUTRAL_FILLER_ELEVEN)
            remaining = 0
        else:
            sentences.append(NEUTRAL_FILLER_TEN)
            remaining -= 10
    if remaining:
        sentences.append(NEUTRAL_FILLER_REMAINDERS[remaining])
    return normalize_space(" ".join([text, *sentences]))


def _bits(row: Mapping[str, Any]) -> dict[str, bool]:
    return {
        factor: bool(row["private_factor_plan"][factor]["private_target"])
        for factor in FACTORS
    }


def _variant(row: Mapping[str, Any], factor: str) -> int:
    family = str(row["private_semantic_surface_family"][factor])
    return int(family.rsplit("_", 1)[-1])


def _surface_phrase(proposition: str, variant: int) -> str:
    templates = (
        "{p}",
        "To state the relevant condition plainly: {p}",
        "The important point for this reply is this: {p}",
        "For this exchange, the applicable fact is: {p}",
        "In different words: {p}",
        "The present-turn condition can be stated this way: {p}",
        "This is the condition the response must take into account: {p}",
        "The same constraint, phrased another way, is: {p}",
    )
    return templates[variant].format(p=proposition)


def _topic(row: Mapping[str, Any]) -> str:
    first, second = TOPIC_WORDS[str(row["topic_family"])]
    return f"{first} {second}"


def _target_detail(row: Mapping[str, Any]) -> str:
    component = str(row["target_component"])
    topic = _topic(row)
    subtype = str(row["candidate_subtype_target"])
    if component == "MP" and subtype == "MP_PREFERENCE":
        return "prefers one tentative option in a single sentence and no list of tasks"
    if component == "MP":
        return f"can address the {topic} issue only during one private twenty-minute window after dinner"
    if component == "MS":
        return f"the earlier {topic} session separated emotional uncertainty from practical workload"
    if component == "ME":
        return f"for the earlier {topic} issue, I wrote one two-sentence note and it helped identify one reversible next step"
    return str(row["required_candidate_function"]).replace("_", " ")


def _goal_clause(
    row: Mapping[str, Any], fit: bool, variant: int, *, surface_version: str = "v1"
) -> str:
    component = str(row["target_component"])
    function = str(row["required_candidate_function"])
    if component == "MP":
        if fit and str(row["candidate_subtype_target"]) == "MP_PREFERENCE":
            proposition = "Please shape the response around my stable response preference."
            return _surface_phrase(proposition, variant)
        if fit:
            proposition = "Please make the response feasible under my actual practical constraint."
            return _surface_phrase(proposition, variant)
        proposition = "What I need now is help naming the feeling accurately; profile personalization is not the task."
        return _surface_phrase(proposition, variant)
    if component == "MS":
        if fit:
            proposition = "Please use the one specific distinction recorded in the earlier session to answer this question."
            return _surface_phrase(proposition, variant)
        proposition = "Please help name my present reaction; a same-topic earlier session would not perform that function."
        return _surface_phrase(proposition, variant)
    if component == "ME":
        if fit:
            proposition = "I am ready to consider one past action-result as a tentative option for what to do now."
            return _surface_phrase(proposition, variant)
        proposition = "My actual question is which factual detail was recorded earlier, not which past action to reuse."
        return _surface_phrase(proposition, variant)
    if surface_version == "v2":
        # These statements are deliberately matched to the actual card-level
        # prerequisites, not merely to a coarse strategy family.  V1 human QA
        # showed that a generic request for "one suggestion" did not establish
        # a trusted-support candidate, and that an unnamed feeling did not
        # establish the precondition of an explicit-emotion reflection card.
        positive = {
            "focused_question": "The parts feel tangled and I cannot tell which pressure matters. Please ask exactly one focused question.",
            "tentative_paraphrase_check": "I need space to say this before deciding what to do. Please briefly restate the most relevant facts as a tentative paraphrase that I can correct.",
            "evidence_grounded_reflection": "I explicitly name the feeling as anxiety. Please mirror that emotion in proportionate language.",
            "one_reversible_suggestion": "Ordinary social support could fit here, and I have one trusted person I would be comfortable considering. I am open to one optional small step.",
        }[function]
    else:
        positive = {
            "focused_question": "The parts feel tangled and I cannot tell which pressure matters. Please ask exactly one focused question.",
            "tentative_paraphrase_check": "I need space to say this before deciding what to do. Please give one tentative paraphrase that I can correct.",
            "evidence_grounded_reflection": "The reaction is hard to name accurately. Please reflect the feeling using only evidence in my words.",
            "one_reversible_suggestion": "I am open to one optional suggestion that is small and reversible.",
        }[function]
    if fit:
        return _surface_phrase(positive, variant)
    alternatives = {
        "focused_question": "The parts feel tangled and I cannot tell which pressure matters, so a question could fit later; what I need now is one brief paraphrase.",
        "tentative_paraphrase_check": "I need space to say this before deciding what to do, so a paraphrase could fit later; what I need now is one focused question.",
        "evidence_grounded_reflection": "The reaction is hard to name accurately, so a reflection could fit later; what I need now is one small reversible suggestion.",
        "one_reversible_suggestion": "I am open to one optional suggestion later; what I need now is one evidence-grounded reflection.",
    }
    return _surface_phrase(alternatives[function], variant)


def _boundary_clause(row: Mapping[str, Any], compatible: bool, variant: int) -> str:
    component = str(row["target_component"])
    action = {
        "MP": "one relevant stored profile detail",
        "MS": "one tentative reference to an earlier session",
        "ME": "one optional past approach",
        "RS": "the retrieved atomic support move",
    }[component]
    if compatible:
        propositions = (
            f"Using {action} is allowed if it directly serves the request.",
            f"I am comfortable with {action} when it serves this request.",
            f"My present boundary leaves room for {action}.",
            f"It would stay within my limits to use {action} once.",
            f"You may use {action} if it remains relevant and brief.",
            f"This turn permits {action} without adding another task.",
            f"There is no present restriction against {action}.",
            f"One careful use of {action} would respect my stated burden.",
        )
    else:
        propositions = (
            f"For this turn, do not use {action}.",
            f"Please leave {action} out of the present response.",
            f"I am not consenting to {action} in this exchange.",
            f"Avoid {action} even if it could matter later.",
            f"My boundary for this turn excludes {action}.",
            f"The response must not include {action} right now.",
            f"Using {action} would go beyond the limit I set for this reply.",
            f"Keep {action} for another turn rather than using it now.",
        )
    return propositions[variant]


def _owner_clause(
    row: Mapping[str, Any], valid: bool, variant: int, *, surface_version: str = "v1"
) -> str:
    component = str(row["target_component"])
    if component == "RS":
        return "The shared technique card contains no private fact about any person."
    if valid:
        propositions = (
            "This question concerns me and my own prior information.",
            "I am the person described in the current situation and the stored record.",
            "Please keep this tied to my own history because I am asking about myself.",
            "The subject of both the question and the prior information is me.",
            "My current request refers to my own circumstances and no one else's.",
            "The relevant history and the present question belong to the same person: me.",
            "I am asking about an issue in my life, so my own record has the correct owner.",
            "There is no change of person here; both the current issue and history are mine.",
        )
    elif (
        surface_version == "v2"
        and component == "MP"
        and str(row["candidate_subtype_target"]) == "MP_PREFERENCE"
    ):
        # A response-format preference remains owned by the user even when the
        # user is discussing a friend.  The honest negative for the combined
        # owner/time gate is therefore an explicitly superseded preference.
        propositions = (
            "That stored response-format preference was mine, but I explicitly replaced it later and it is no longer current.",
            "I previously held that response preference, but a later session superseded it.",
            "The old response-format preference belongs to me, yet it has been explicitly withdrawn.",
            "My stored preference is out of date because I replaced it with a newer one.",
            "Do not treat the earlier response preference as current; I later revoked it.",
            "The preference record has the right person but the wrong time because I superseded it.",
            "I changed that response preference after it was stored, so the old record is stale.",
            "The candidate describes my former preference, not the response format I currently prefer.",
        )
    else:
        propositions = (
            "This question concerns my friend, not me; our histories must remain separate.",
            "The person I am asking about is a colleague, while the stored record describes me.",
            "My sibling is the subject now, so information from my own history has the wrong owner.",
            "The present situation belongs to someone close to me rather than to me.",
            "Please do not transfer my personal record to the friend I am describing now.",
            "The earlier information is mine, but the current question concerns another person.",
            "I am describing a relative's situation; my own stored details should not be assigned to them.",
            "The owner has changed between the record and this request, which is about my friend.",
        )
    return propositions[variant]


def _increment_clause(row: Mapping[str, Any], present: bool, variant: int) -> str:
    detail = _target_detail(row)
    if present:
        propositions = (
            "One concrete stored detail may add something that is not already stated here.",
            "The exact candidate detail is absent from the visible current exchange.",
            "A specific fact in the candidate would be new rather than an echo of my words.",
            "The current dialogue does not yet contain the candidate's concrete information.",
            "There is a candidate detail that has not been disclosed in this turn.",
            "The retrieved content could supply one specific point missing from the visible context.",
            "Nothing in the present messages already states the candidate's exact detail.",
            "The candidate has a concrete addition beyond what can be read in this exchange.",
        )
    else:
        propositions = (
            f"The relevant detail is already visible here: {detail}.",
            f"I have already stated the candidate's exact point in this message: {detail}.",
            f"Nothing new is needed from the candidate because I just said: {detail}.",
            f"The current exchange already contains this same information: {detail}.",
            f"Treat the candidate as an echo; the visible detail is: {detail}.",
            f"The specific content is not missing from this turn; it is: {detail}.",
            f"I am making the stored point explicit now, so it adds no new detail: {detail}.",
            f"This candidate information has already been disclosed in the present context: {detail}.",
        )
    return propositions[variant]


def _current_text(row: Mapping[str, Any], *, surface_version: str = "v1") -> str:
    bits = _bits(row)
    topic = _topic(row)
    clauses = [
        PREFIX_TEXT[str(row["prefix_family"])],
        f"The current situation involves {topic}.",
        _owner_clause(
            row,
            bits["owner_time_entity_valid"],
            0 if row["target_component"] == "RS" else _variant(row, "owner_time_entity_valid"),
            surface_version=surface_version,
        ),
        _goal_clause(
            row,
            bits["goal_function_fit"],
            _variant(row, "goal_function_fit"),
            surface_version=surface_version,
        ),
        _boundary_clause(
            row,
            bits["boundary_burden_compatible"],
            _variant(row, "boundary_burden_compatible"),
        ),
    ]
    if row["track"] == "ELIGIBILITY_CONFIRMATION":
        clauses.insert(1, "Earlier today, I chose a fresh wording for this present request.")
    clauses.append(
        _increment_clause(
            row,
            bits["specific_increment"],
            _variant(row, "specific_increment"),
        )
    )
    target_words = 112 if row["length_direction"] == "NEGATIVE_LONGER" else 136
    return _pad_words(" ".join(clauses), target_words)


def _visible_dialogue(row: Mapping[str, Any]) -> list[dict[str, str]]:
    topic = _topic(row)
    user = _pad_words(
        f"The {topic} issue has been taking up attention, but I want to keep this exchange precise.",
        28,
    )
    assistant = _pad_words(
        "I am following what you have said and will avoid deciding the meaning or next action for you.",
        28,
    )
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _basic_info(row: Mapping[str, Any]) -> dict[str, str]:
    count = int(row["source_catalog_size_target"]) if row["target_component"] == "MP" else 2
    subtype = str(row["candidate_subtype_target"])
    key = "stable_preference_response_format" if subtype == "MP_PREFERENCE" else "stable_practical_constraint"
    fields = {key: _pad_words(_target_detail(row), 28)}
    for index in range(count - 1):
        fields[f"ordinary_background_{index + 1}"] = DISTRACTORS[index % len(DISTRACTORS)]
    return fields


def _target_session(row: Mapping[str, Any]) -> dict[str, Any]:
    component = str(row["target_component"])
    detail = _target_detail(row)
    if component == "MS":
        summary = _pad_words(detail + ". It was preserved as one specific earlier distinction.", 36)
        seeker = _pad_words(f"I wanted this recorded: {detail}.", 40)
    elif component == "ME":
        summary = _pad_words(
            f"The user described an earlier action and observable result related to {_topic(row)}.",
            36,
        )
        seeker = _pad_words(detail + ".", 40)
    else:
        summary = _pad_words(f"An earlier ordinary discussion concerned {_topic(row)}.", 36)
        seeker = _pad_words(f"I mentioned the {_topic(row)} situation without settling it.", 40)
    supporter = _pad_words(
        "You kept the earlier record limited to what was actually said and did not assume it still applied.",
        40,
    )
    return {
        "id": "target_prior_session",
        "timestamp": "",
        "summary": summary,
        "dialogue": [
            {"role": "seeker", "content": seeker},
            {"role": "supporter", "content": supporter},
        ],
    }


def _distractor_session(index: int) -> dict[str, Any]:
    subject = DISTRACTORS[index % len(DISTRACTORS)]
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
    count = int(row["source_catalog_size_target"]) if component in {"MS", "ME"} else 3
    sessions = [_distractor_session(index) for index in range(count)]
    sessions[-1] = _target_session(row)
    return sessions


def realize_state(
    row: Mapping[str, Any], *, surface_version: str = "v1"
) -> dict[str, Any]:
    if row.get("protocol") != BLUEPRINT_PROTOCOL:
        raise ValueError("not an orthogonal Observation blueprint row")
    history = _history(row)
    return {
        "protocol": REPAIRED_PROTOCOL if surface_version == "v2" else PROTOCOL,
        "state_id": str(row["blueprint_row_id"]),
        "blueprint_row_id": str(row["blueprint_row_id"]),
        "user_id": str(row["user_id"]),
        "group_id": str(row["group_id"]),
        "split": str(row["track"]),
        "track": str(row["track"]),
        "user": {
            "id": str(row["user_id"]),
            "basic_info": _basic_info(row),
            "dialog_history": history,
        },
        "visible_dialogue": _visible_dialogue(row),
        "current_user_text": _current_text(row, surface_version=surface_version),
        "current_session_index": len(history) + 1,
        "construction_intent_present_in_model_input": False,
        "response_or_outcome_read": False,
        "external_text_read": False,
    }


def realize_states(
    rows: Sequence[Mapping[str, Any]], *, surface_version: str = "v1"
) -> list[dict[str, Any]]:
    if surface_version not in {"v1", "v2"}:
        raise ValueError(f"unsupported Observation surface version: {surface_version}")
    return [realize_state(row, surface_version=surface_version) for row in rows]


def audit_states(
    states: Sequence[Mapping[str, Any]],
    blueprint_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    blueprint = {str(row["blueprint_row_id"]): row for row in blueprint_rows}
    if len(states) != 96:
        failures.append(f"state_count_{len(states)}_not_96")
    if set(blueprint) != {str(state["state_id"]) for state in states}:
        failures.append("state_blueprint_id_mismatch")
    public_hashes = [
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
    if len(set(public_hashes)) != len(public_hashes):
        failures.append("duplicate_visible_state_surface")

    source_mismatches: list[str] = []
    me_subtype_mismatches: list[str] = []
    for state in states:
        row = blueprint[str(state["state_id"])]
        component = str(row["target_component"])
        actual = (
            len(state["user"]["basic_info"])
            if component == "MP"
            else len(state["user"]["dialog_history"])
            if component in {"MS", "ME"}
            else 80
        )
        if actual != int(row["source_catalog_size_target"]):
            source_mismatches.append(str(state["state_id"]))
        if component == "ME":
            target = state["user"]["dialog_history"][-1]["dialogue"][0]["content"]
            actual_subtype = str(compile_me_structure_metadata(target)["me_subtype_hint"])
            if actual_subtype != str(row["candidate_subtype_target"]):
                me_subtype_mismatches.append(str(state["state_id"]))
    if source_mismatches:
        failures.append("source_catalog_size_mismatch")
    if me_subtype_mismatches:
        failures.append("me_compiler_subtype_mismatch")

    fit_states = [state for state in states if state["track"] == "FACTOR_FIT"]
    length_balance: dict[str, Any] = {}
    for component in ("MP", "MS", "ME", "RS"):
        subset = [state for state in fit_states if blueprint[str(state["state_id"])]["target_component"] == component]
        length_balance[component] = {}
        for factor in FACTORS:
            if not blueprint[str(subset[0]["state_id"])]["private_factor_plan"][factor]["is_learning_target"]:
                continue
            cross = defaultdict(Counter)
            for state in subset:
                row = blueprint[str(state["state_id"])]
                label = bool(row["private_factor_plan"][factor]["private_target"])
                length = len(str(state["current_user_text"]).split())
                cross[length][label] += 1
            length_balance[component][factor] = {
                str(length): {str(label): count for label, count in counts.items()}
                for length, counts in cross.items()
            }
            if any(counts[True] != counts[False] for counts in cross.values()):
                failures.append(f"current_length_predicts_{component}_{factor}")

    if any(
        state["construction_intent_present_in_model_input"]
        or state["response_or_outcome_read"]
        or state["external_text_read"]
        for state in states
    ):
        failures.append("forbidden_input_or_outcome_read")
    return {
        "protocol": str(states[0]["protocol"]) if states else PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "scope": "VISIBLE_STATE_REALIZATION_BEFORE_EXACT_RANK1",
        "states": len(states),
        "unique_users": len({str(state["user_id"]) for state in states}),
        "unique_visible_state_surfaces": len(set(public_hashes)),
        "source_catalog_size_mismatches": source_mismatches,
        "me_compiler_subtype_mismatches": me_subtype_mismatches,
        "current_text_length_balance": length_balance,
        "api_calls": 0,
        "human_labels_read": 0,
        "external_lockbox_read": False,
        "responses_generated": 0,
        "failures": failures,
        "next_step_if_pass": "FORMAL_SAME_STACK_EXACT_RANK1_MATERIALIZATION_AND_FULL_TEXT_NUISANCE_PROBE",
    }
