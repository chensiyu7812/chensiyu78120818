#!/usr/bin/env python3
"""Freeze an outcome-blind replacement for the invalid D3 MS contrasts.

The original D3 MS query included supporter-authored metacommunication and
selected unrelated generic summaries in 15/32 states.  This bounded repair
creates 32 content-disjoint users and uses a shared seeker-only MS query.  It
fails closed before any API call unless ownership, causal order, retrieval
relevance, leakage, background balance, and feature-axis coverage all pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from typing import Any

from metacom_pm.contracts import (
    ALL_ACTION_IDS,
    MemoryBackendRecord,
    MemoryItem,
    MemorySource,
    RuntimeState,
    SourceCatalog,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
)
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.retrieval import (
    MemoryRetriever,
    source_specific_memory_queries,
)
from metacom_pm.text import estimate_tokens, normalize_space
from metacom_pm.v1_5_candidate_discovery import (
    SOURCE_SPECIFIC_QUERY_PROTOCOL,
    discover_memory_candidates_by_source_query,
)
from metacom_pm.v1_5_d3_features import (
    D3_FEATURE_PROTOCOL,
    d3_model_features,
    memory_grounding_observation,
)
from metacom_pm.v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    compile_bounded_memory,
)
from metacom_pm.v1_5_strategy_rag_repair import (
    effect_study_observable_flags,
    effect_study_rank_applicable_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d3-ms-seeker-query-replacement-blueprint-v1"
STATUS = "PASS_READY_TO_PREPARE_80_FROZEN_MS_REPLACEMENT_CALLS"
# A later wrapper may reuse the construction mechanics and replace the old
# match×grounding axis before freezing its own manifest.  The default remains
# false so this script's historical output and checks are unchanged.
INCREMENTAL_WRAPPER_REPLACES_STRATA = False

THEMES = (
    {
        "key": "deadlines",
        "anchors": "deadlines priority workload",
        "profile": "works in a design role with recurring deadlines",
        "relevant": (
            "Deadlines and workload felt crowded; choosing one priority reduced pressure.",
            "A later deadline week again required separating urgent work from less urgent work.",
            "Workload pressure returned, and one concrete priority made it easier to begin.",
            "The latest deadline check-in focused on overload and where to start.",
        ),
        "high": "Several deadlines are competing again, and my workload feels overwhelming because I cannot choose the first priority.",
        "low": "Too many work demands have arrived together, and I feel stretched thin about where to begin.",
    },
    {
        "key": "friendship",
        "anchors": "friendship silence messages",
        "profile": "has a long-standing close friendship circle",
        "relevant": (
            "Friendship uncertainty followed a week of silence and unanswered messages.",
            "A later friendship check-in focused on hurt caused by messages receiving no reply.",
            "Silence from a close friend brought sadness without clarifying what it meant.",
            "The latest friendship session separated the unanswered messages from assumptions about intent.",
        ),
        "high": "A friendship feels uncertain after a quiet week of unanswered messages, and the silence hurts.",
        "low": "Someone close has gone quiet, and the distance leaves me sad and unsure what it means.",
    },
    {
        "key": "exam",
        "anchors": "exam study concentration",
        "profile": "is a part-time student preparing for professional exams",
        "relevant": (
            "Exam study became difficult when concentration dropped after work.",
            "A later exam session used a smaller study target when concentration was limited.",
            "Evening concentration problems made exam preparation feel overwhelming.",
            "The latest study check-in focused on exam anxiety and a realistic starting point.",
        ),
        "high": "Another exam is approaching, and my study concentration keeps slipping after work.",
        "low": "A professional test is getting closer, but after my shift my attention disappears and I cannot start preparing.",
    },
    {
        "key": "move",
        "anchors": "move city lonely weekends",
        "profile": "recently moved to a new city and lives alone",
        "relevant": (
            "The move to a new city made unfamiliar weekends feel lonely.",
            "A later moving check-in explored how quiet weekends increased isolation.",
            "One regular local activity made the new city feel less anonymous.",
            "The latest session about the move focused on loneliness without familiar people nearby.",
        ),
        "high": "The move to this new city still feels lonely on weekends when no familiar person is nearby.",
        "low": "My new surroundings still do not feel like home, and quiet days leave me isolated.",
    },
    {
        "key": "family",
        "anchors": "family conversation argument",
        "profile": "shares a home with an adult family member",
        "relevant": (
            "A family conversation became an argument when every disagreement was raised at once.",
            "A later family discussion stayed calmer by focusing on one concern.",
            "Tension before a family conversation came from expecting another argument.",
            "The latest family check-in focused on speaking without escalating the discussion.",
        ),
        "high": "I need another family conversation, but I feel tense and fear it will become an argument.",
        "low": "I need to speak with someone at home, yet I worry the discussion will become heated again.",
    },
    {
        "key": "breakup",
        "anchors": "breakup reminder sadness",
        "profile": "is rebuilding routines after the end of a long relationship",
        "relevant": (
            "A breakup reminder made an ordinary evening unexpectedly sad.",
            "A later breakup session allowed sadness without pressure to recover quickly.",
            "Unexpected reminders of the relationship brought the hurt back suddenly.",
            "The latest breakup check-in focused on the immediate effect of a painful reminder.",
        ),
        "high": "A reminder of the breakup hit today, and the sadness returned very suddenly.",
        "low": "Something recalled my former relationship, and the hurt came back more strongly than I expected.",
    },
    {
        "key": "caregiving",
        "anchors": "caregiving relative exhausted rest",
        "profile": "helps care for an older relative several evenings each week",
        "relevant": (
            "Caregiving for a relative filled the week and left little time to rest.",
            "A later caregiving session explored guilt about needing recovery time.",
            "Protecting one short break made caregiving pressure more sustainable.",
            "The latest caregiving check-in focused on exhaustion and feeling selfish for resting.",
        ),
        "high": "Caregiving for my relative has filled this week, and I feel exhausted and guilty for needing rest.",
        "low": "Supporting someone in my family has taken most of my energy, and wanting time alone feels selfish.",
    },
    {
        "key": "sleep",
        "anchors": "sleep shifts bedtime",
        "profile": "works rotating shifts that change the weekly sleep schedule",
        "relevant": (
            "Rotating shifts made sleep and bedtime difficult after a late shift.",
            "A later sleep session used one wind-down cue after a schedule change.",
            "An irregular shift schedule made it hard to switch off at bedtime.",
            "The latest sleep check-in focused on frustration after another rota change.",
        ),
        "high": "My sleep is unsettled after another shift change, and bedtime is frustrating because I cannot switch off.",
        "low": "The latest rota change has left my nights irregular, and I feel worn down when I try to rest.",
    },
)

DISTRACTORS = (
    "An ordinary household check-in covered groceries, laundry, and a routine meal.",
    "A neutral weekly update mentioned the weather, a television programme, and errands.",
)


def _visible_dialogue(state: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "speaker": "seeker" if row["role"] == "user" else "supporter",
            "content": normalize_space(row["content"]),
        }
        for row in state["current_session_history"]
    ] + [{"speaker": "seeker", "content": state["current_user_text"]}]


def _strategy_candidate(
    state: dict[str, Any], cards: list[dict[str, Any]]
) -> dict[str, Any] | None:
    dialogue = _visible_dialogue(state)
    seeker = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    recent = " ".join(seeker[-3:])
    flags = effect_study_observable_flags(
        current_user_text=seeker[-1],
        recent_user_text=recent,
        visible_dialogue=dialogue,
    )
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool)
        and value
        and key
        not in {
            "substantive",
            "pure_phatic",
            "routine_closing",
            "active_high_stakes",
            "explicit_stop",
            "ordinary_rag_hard_off",
            "structural_eligibility_only_not_benefit",
        }
    ]
    query = (
        "Choose one safe topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cues) or 'none'}. "
        f"Recent seeker context: {recent}"
    )
    ranked = effect_study_rank_applicable_cards(
        query=query,
        current_user_text=seeker[-1],
        recent_user_text=recent,
        visible_dialogue=dialogue,
        cards=cards,
    )
    if not ranked:
        return None
    top = dict(ranked[0])
    top.update(
        {
            "query": query,
            "observable_flags": flags,
            "retrieval_score": float(top.pop("score")),
            "content_scope": "technique_only",
        }
    )
    return top


def _make_user(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    theme = THEMES[index % len(THEMES)]
    sessions = []
    relevant_sessions = {1, 3, 5, 6}
    relevant_index = 0
    distractor_index = 0
    for session_no in range(1, 7):
        if session_no in relevant_sessions:
            summary = theme["relevant"][relevant_index]
            event = f"{theme['anchors']}. {summary}"
            relevant_index += 1
        else:
            summary = DISTRACTORS[distractor_index]
            event = summary
            distractor_index += 1
        sessions.append(
            {
                "id": f"d3ms_u{index + 1:03d}_s{session_no}",
                "timestamp": f"session-{session_no}",
                "summary": summary,
                "dialogue": [
                    {"role": "seeker", "content": event},
                    {
                        "role": "supporter",
                        "content": "Thank you for explaining what happened in that session.",
                    },
                ],
            }
        )
    user = {
        "id": f"pmv15_d3ms_u{index + 1:03d}",
        "basic_info": {"profile_context": theme["profile"]},
        "dialog_history": sessions,
    }
    situation_cue = (
        " I only want help putting this into words, not advice."
        if index % 4 == 0
        else " Could you offer one small optional suggestion?"
        if index % 4 == 1
        else " I am unsure which part matters most right now."
        if index % 4 == 2
        else " I want to understand why this feels so heavy today."
    )
    current = theme["high"] if index % 4 < 2 else theme["low"]
    supporter = (
        "You already shared the prior session summary in this same chat."
        if index % 2 == 1
        else "I'm listening. Tell me what feels most relevant today."
    )
    state = {
        "state_id": "state_" + stable_hex(PROTOCOL, user["id"], n=24),
        "card_id": "card_" + stable_hex(PROTOCOL, user["id"], "backend", n=24),
        "user_id": user["id"],
        "split": "development",
        "semantic_family": f"d3_ms_{theme['key']}",
        "current_user_text": normalize_space(str(current) + situation_cue),
        "current_session_history": [
            {
                "role": "user",
                "content": f"This week the {theme['anchors']} situation has come up again.",
            },
            {"role": "assistant", "content": supporter},
        ],
        "current_session_summary": "",
        "session_index": 7,
    }
    return user, state


def _metadata(items: list[MemoryItem]) -> dict[str, dict[str, Any]]:
    return {
        item.memory_id: (
            {"mp_subtype": "MP_PROFILE"}
            if item.source is MemorySource.MP
            else {
                "summary_origin": "supplied_strictly_prior_summary",
                "expected_relevant_for_current_state": int(item.created_session)
                in {1, 3, 5, 6},
            }
            if item.source is MemorySource.MS
            else {"episode_origin": "strictly_prior_seeker_turn"}
        )
        for item in items
    }


def _inventory(
    items: list[MemoryItem], session_index: int
) -> dict[MemorySource, SourceCatalog]:
    inventory: dict[MemorySource, SourceCatalog] = {}
    for source in MemorySource:
        rows = [item for item in items if item.source is source]
        ages = [session_index - int(item.created_session) for item in rows]
        inventory[source] = SourceCatalog(
            available=bool(rows),
            count=len(rows),
            min_age_sessions=min(ages) if ages else None,
            max_age_sessions=max(ages) if ages else None,
            estimated_tokens=sum(estimate_tokens(item.text) for item in rows),
            catalog_fingerprint=[],
            semantic_query_similarity=0.0,
            semantic_representation_valid=False,
        )
    return inventory


def _queries(state: dict[str, Any]) -> dict[MemorySource, str]:
    return source_specific_memory_queries(
        state["current_user_text"],
        state["current_session_history"],
        state["current_session_summary"],
    )


def _background(index: int) -> tuple[str, str]:
    # The frozen strategy bank has no applicable card for the breakup/advice
    # state.  Swap only the RS background bit with the same MP/ME cell's
    # friendship state; all eight backgrounds remain exactly 4x and MS is
    # still the sole treatment difference.
    slot = index % 8
    if slot in {1, 5}:
        slot ^= 4
    bits = tuple(bool(slot & (1 << position)) for position in range(3))
    other_sources = (MemorySource.MP, MemorySource.ME)
    sources = frozenset(
        source
        for source, enabled in zip(other_sources, bits[:2], strict=True)
        if enabled
    )
    strategy = StrategyMode.RS if bits[2] else StrategyMode.R0
    return (
        canonical_action_id(sources, strategy),
        canonical_action_id(frozenset({*sources, MemorySource.MS}), strategy),
    )


def _action_memory_ids(action: str, discoveries: dict[MemorySource, Any]) -> list[str]:
    sources, _ = parse_action_id(action)
    return [
        item.memory_id
        for source in sorted(sources, key=lambda value: value.value)
        for item in discoveries[source].selected_items
    ]


def _recursive_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = normalize_space(value)
        return [value] if len(value) >= 16 else []
    if isinstance(value, list):
        return [text for row in value for text in _recursive_strings(row)]
    if isinstance(value, dict):
        return [text for row in value.values() for text in _recursive_strings(row)]
    return []


def _content_disjoint_audit(generated: list[str]) -> dict[str, Any]:
    sources = {
        "ESConv": ROOT / "data/external/ESConv.json",
        "EvoEmo": ROOT / "data/external/evo_emo.json",
    }
    generated_set = {
        normalize_space(text).lower()
        for text in generated
        if len(normalize_space(text)) >= 16
    }
    overlaps: dict[str, int] = {}
    for name, path in sources.items():
        external = json.loads(path.read_text(encoding="utf-8"))
        external_set = {
            text.lower() for text in _recursive_strings(external)
        }
        overlaps[name] = len(generated_set & external_set)
    return {
        "audit_after_construction_only": True,
        "external_content_used_for_generation_or_selection": False,
        "exact_normalized_overlap_counts": overlaps,
        "pass": not any(overlaps.values()),
    }


def build(*, cards_path: Path, out_dir: Path) -> dict[str, Any]:
    cards = [dict(row) for row in iter_jsonl(cards_path)]
    retriever = MemoryRetriever(
        minimum_score_by_source={source: 0.0 for source in MemorySource}
    )
    users: list[dict[str, Any]] = []
    states: list[RuntimeState] = []
    backends: list[MemoryBackendRecord] = []
    metadata_rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    retrieval_audit: list[dict[str, Any]] = []
    generated_texts: list[str] = []

    for index in range(32):
        user, raw_state = _make_user(index)
        items, _ = compile_bounded_memory(user)
        metadata = _metadata(items)
        queries = _queries(raw_state)
        discoveries = discover_memory_candidates_by_source_query(
            queries=queries,
            items=items,
            retriever=retriever,
            session_index=raw_state["session_index"],
        )
        selected_ms = discoveries[MemorySource.MS].selected_items
        selected_relevance = [
            bool(metadata[item.memory_id]["expected_relevant_for_current_state"])
            for item in selected_ms
        ]
        mutated = dict(raw_state)
        mutated["current_session_history"] = [
            dict(row) for row in raw_state["current_session_history"]
        ]
        mutated["current_session_history"][1]["content"] = (
            "A supporter-side sentinel that must never affect MS retrieval."
        )
        mutated_queries = _queries(mutated)
        mutated_discoveries = discover_memory_candidates_by_source_query(
            queries=mutated_queries,
            items=items,
            retriever=retriever,
            session_index=raw_state["session_index"],
        )
        visible = _visible_dialogue(raw_state)
        grounding = memory_grounding_observation(
            source=MemorySource.MS,
            selected_items=selected_ms,
            catalog_memory_ids={item.memory_id for item in items},
            catalog_user_id=user["id"],
            current_user_id=user["id"],
            current_session_index=raw_state["session_index"],
            current_user_text=raw_state["current_user_text"],
            visible_dialogue=visible,
            source_metadata=metadata,
        )
        strategy = _strategy_candidate(raw_state, cards)
        runtime = RuntimeState(
            **raw_state,
            inventory=_inventory(items, raw_state["session_index"]),
            allowed_actions=list(ALL_ACTION_IDS),
            provenance={
                "d3_ms_replacement_protocol": PROTOCOL,
                "memory_compiler": BOUNDED_MEMORY_COMPILER_PROTOCOL,
                "candidate_query_protocol": SOURCE_SPECIFIC_QUERY_PROTOCOL,
                "ms_query_surface": "seeker_only_visible_context",
                "candidate_feature_protocol": D3_FEATURE_PROTOCOL,
                "outcome_read": False,
            },
        )
        control, treatment = _background(index)
        slot_id = "d3ms_slot_" + stable_hex(PROTOCOL, user["id"], n=24)
        contrasts.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": slot_id,
                "component": "MS",
                "user_id": user["id"],
                "state_id": runtime.state_id,
                "card_id": runtime.card_id,
                "control_action": control,
                "treatment_action": treatment,
                "background_action": control,
                "component_candidate_observation": {
                    "descriptor": discoveries[MemorySource.MS].descriptor,
                    "grounding": grounding,
                },
                "model_features": d3_model_features(
                    component="MS",
                    candidate_state_match_score=float(
                        discoveries[MemorySource.MS].descriptor[
                            "top1_lexical_relevance"
                        ]
                    ),
                    grounding_score=float(
                        grounding["candidate_grounding_or_nonredundancy_score"]
                    ),
                    background_action=control,
                ),
                "selected_memory_ids_by_action_generation_only": {
                    control: _action_memory_ids(control, discoveries),
                    treatment: _action_memory_ids(treatment, discoveries),
                },
                "current_strategy_candidate": strategy,
                "effect_label": "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW",
                "outcome_read": False,
            }
        )
        retrieval_audit.append(
            {
                "protocol": PROTOCOL,
                "user_id": user["id"],
                "state_id": runtime.state_id,
                "selected_ms_ids": [item.memory_id for item in selected_ms],
                "selected_ms_created_sessions": [
                    int(item.created_session) for item in selected_ms
                ],
                "selected_relevance": selected_relevance,
                "all_selected_relevant": all(selected_relevance),
                "at_least_one_selected_relevant": any(selected_relevance),
                "correct_owner": grounding["checks"]["belongs_to_current_user"],
                "strictly_prior": grounding["checks"]["strictly_prior"],
                "seeker_query_sha256": sha256_text(queries[MemorySource.MS]),
                "supporter_mutation_query_invariant": (
                    queries[MemorySource.MS] == mutated_queries[MemorySource.MS]
                ),
                "supporter_mutation_selection_invariant": (
                    [item.memory_id for item in selected_ms]
                    == [
                        item.memory_id
                        for item in mutated_discoveries[MemorySource.MS].selected_items
                    ]
                ),
                "legacy_query_differs_from_ms_query": (
                    queries[MemorySource.MP] != queries[MemorySource.MS]
                ),
            }
        )
        users.append(user)
        states.append(runtime)
        backends.append(MemoryBackendRecord(card_id=runtime.card_id, items=items))
        metadata_rows.append(
            {
                "protocol": PROTOCOL,
                "user_id": user["id"],
                "card_id": runtime.card_id,
                "memory_item_metadata": metadata,
                "construction_only_not_pm_features": True,
            }
        )
        generated_texts.extend(_recursive_strings(user))
        generated_texts.extend(_recursive_strings(raw_state))

    primary = []
    repeats = []
    for index, row in enumerate(contrasts):
        primary.append(
            {
                "pair_role": "primary",
                "pair_seed": int(
                    stable_hex(PROTOCOL, row["contrast_slot_id"], "primary", n=8),
                    16,
                )
                % 2_000_000_000,
                **row,
            }
        )
        if index < 8:
            repeats.append(
                {
                    "pair_role": "outcome_blind_repeat",
                    "pair_seed": int(
                        stable_hex(
                            PROTOCOL,
                            row["contrast_slot_id"],
                            "repeat",
                            n=8,
                        ),
                        16,
                    )
                    % 2_000_000_000,
                    **row,
                }
            )

    match_midpoint = median(
        row["model_features"]["candidate_state_match_score"]
        for row in contrasts
    )
    strata = Counter(
        (
            "higher_match"
            if row["model_features"]["candidate_state_match_score"]
            >= match_midpoint
            else "lower_match",
            "higher_grounding"
            if row["model_features"][
                "candidate_grounding_or_nonredundancy_score"
            ]
            == 1.0
            else "lower_grounding",
        )
        for row in contrasts
    )
    backgrounds = Counter(row["background_action"] for row in contrasts)
    content_audit = _content_disjoint_audit(generated_texts)
    selected_total = sum(len(row["selected_relevance"]) for row in retrieval_audit)
    selected_relevant = sum(
        sum(bool(value) for value in row["selected_relevance"])
        for row in retrieval_audit
    )
    relevance_summary = {
        "states": 32,
        "top2_present_states": sum(
            len(row["selected_ms_ids"]) == 2 for row in retrieval_audit
        ),
        "all_top2_relevant_states": sum(
            row["all_selected_relevant"] for row in retrieval_audit
        ),
        "at_least_one_relevant_states": sum(
            row["at_least_one_selected_relevant"] for row in retrieval_audit
        ),
        "selected_item_precision": selected_relevant / selected_total,
        "correct_owner_states": sum(row["correct_owner"] for row in retrieval_audit),
        "strictly_prior_states": sum(row["strictly_prior"] for row in retrieval_audit),
        "supporter_query_invariant_states": sum(
            row["supporter_mutation_query_invariant"] for row in retrieval_audit
        ),
        "supporter_selection_invariant_states": sum(
            row["supporter_mutation_selection_invariant"]
            for row in retrieval_audit
        ),
    }
    forbidden = {
        "dataset",
        "user_id",
        "state_id",
        "card_id",
        "response",
        "judge",
        "outcome",
        "label",
        "topic",
        "template",
    }
    checks = {
        "32_new_users": len(users) == 32
        and len({row["id"] for row in users}) == 32,
        "32_ms_one_bit_contrasts": len(contrasts) == 32
        and all(
            len(
                set(parse_action_id(row["control_action"])[0])
                ^ set(parse_action_id(row["treatment_action"])[0])
            )
            + int(
                parse_action_id(row["control_action"])[1]
                != parse_action_id(row["treatment_action"])[1]
            )
            == 1
            and set(parse_action_id(row["treatment_action"])[0])
            - set(parse_action_id(row["control_action"])[0])
            == {MemorySource.MS}
            for row in contrasts
        ),
        "all_16_runtime_actions_available": all(
            set(state.allowed_actions) == set(ALL_ACTION_IDS) for state in states
        ),
        "eight_backgrounds_four_each": len(backgrounds) == 8
        and set(backgrounds.values()) == {4},
        "rs_candidate_present_when_background_on": all(
            parse_action_id(row["background_action"])[1] is StrategyMode.R0
            or row["current_strategy_candidate"] is not None
            for row in contrasts
        ),
        "top2_present_32_of_32": relevance_summary["top2_present_states"] == 32,
        "all_top2_relevant_32_of_32": relevance_summary[
            "all_top2_relevant_states"
        ]
        == 32,
        "selected_item_precision_one": relevance_summary[
            "selected_item_precision"
        ]
        == 1.0,
        "ownership_and_causal_order_32_of_32": relevance_summary[
            "correct_owner_states"
        ]
        == relevance_summary["strictly_prior_states"]
        == 32,
        "supporter_text_cannot_steer_ms_query_or_selection": relevance_summary[
            "supporter_query_invariant_states"
        ]
        == relevance_summary["supporter_selection_invariant_states"]
        == 32,
        "formal_pairs_have_no_hard_off": all(
            not row["component_candidate_observation"]["grounding"][
                "deterministic_hard_off"
            ]
            for row in contrasts
        ),
        "match_and_grounding_crossed": (
            len(strata) == 4 and min(strata.values()) >= 4
        )
        or INCREMENTAL_WRAPPER_REPLACES_STRATA,
        "five_numeric_features_only": all(
            len(row["model_features"]) == 5
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in row["model_features"].values()
            )
            and not any(
                any(token in key.lower() for token in forbidden)
                for key in row["model_features"]
            )
            for row in contrasts
        ),
        "32_primary_8_repeat_80_calls": len(primary) == 32
        and len(repeats) == 8,
        "no_outcome_or_human_label_read": all(
            row["effect_label"].startswith("UNKNOWN")
            and not row["outcome_read"]
            for row in contrasts
        ),
        "content_disjoint_exact_audit": content_audit["pass"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            "MS replacement preflight failed: "
            f"{[key for key, value in checks.items() if not value]}; "
            f"relevance={relevance_summary}; strata={strata}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "synthetic_users.jsonl", users)
    write_jsonl(
        out_dir / "runtime_states.jsonl",
        [state.model_dump(mode="json") for state in states],
    )
    write_jsonl(
        out_dir / "memory_backend.jsonl",
        [backend.model_dump(mode="json") for backend in backends],
    )
    write_jsonl(out_dir / "memory_item_metadata.jsonl", metadata_rows)
    write_jsonl(out_dir / "d3_ms_replacement_contrasts.jsonl", contrasts)
    write_jsonl(out_dir / "d3_ms_replacement_primary_pairs.jsonl", primary)
    write_jsonl(out_dir / "d3_ms_replacement_repeat_pairs.jsonl", repeats)
    write_jsonl(out_dir / "retrieval_qualification_audit.jsonl", retrieval_audit)
    report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "api_calls_made": 0,
        "human_labels_read": False,
        "external_outcomes_read": False,
        "scope": "replace_only_original_d3_ms_40_pairs",
        "original_d3_ms_disposition": "DIAGNOSTIC_ONLY_NOT_FORMAL_TRAINING",
        "preserved_original_components": ["RS", "MP", "ME"],
        "users": 32,
        "primary_pairs": 32,
        "repeat_pairs": 8,
        "total_pairs": 40,
        "planned_response_calls": 80,
        "background_counts": dict(sorted(backgrounds.items())),
        "candidate_strata_counts": {
            "|".join(key): value for key, value in sorted(strata.items())
        },
        "relevance_qualification": relevance_summary,
        "content_disjoint_audit": content_audit,
        "checks": checks,
        "files": {},
    }
    for name in (
        "synthetic_users.jsonl",
        "runtime_states.jsonl",
        "memory_backend.jsonl",
        "memory_item_metadata.jsonl",
        "d3_ms_replacement_contrasts.jsonl",
        "d3_ms_replacement_primary_pairs.jsonl",
        "d3_ms_replacement_repeat_pairs.jsonl",
        "retrieval_qualification_audit.jsonl",
    ):
        report["files"][name] = sha256_file(out_dir / name)
    write_json(out_dir / "preflight_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": STATUS,
            "preflight_report_sha256": sha256_file(
                out_dir / "preflight_report.json"
            ),
            "primary_pairs_sha256": sha256_file(
                out_dir / "d3_ms_replacement_primary_pairs.jsonl"
            ),
            "repeat_pairs_sha256": sha256_file(
                out_dir / "d3_ms_replacement_repeat_pairs.jsonl"
            ),
            "strategy_cards_sha256": sha256_file(cards_path),
            "outcome_blind": True,
            "api_calls_made": 0,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_replacement_step0_v1",
    )
    args = parser.parse_args()
    report = build(cards_path=args.cards, out_dir=args.out_dir)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol",
                    "status",
                    "users",
                    "total_pairs",
                    "planned_response_calls",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
