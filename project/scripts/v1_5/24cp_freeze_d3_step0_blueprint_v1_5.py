#!/usr/bin/env python3
"""Build and preflight the outcome-blind D3 Step-0 blueprint.

This script is intentionally zero-API.  It constructs 32 new longitudinal
users, compiles their private histories through the shared Evo-style memory
compiler, discovers candidates with the runtime lexical retrievers, freezes
128 one-bit component contrasts and 32 repeat pairs, and fails closed on any
leakage, shortcut, capacity, subtype, background, or coverage violation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any

from metacom_pm.contracts import (
    ACTION_MEMORY_MAP,
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
from metacom_pm.retrieval import MemoryRetriever, context_query
from metacom_pm.text import estimate_tokens, normalize_space
from metacom_pm.v1_5_candidate_discovery import discover_memory_candidates
from metacom_pm.v1_5_d3_features import (
    D3_FEATURE_PROTOCOL,
    d3_model_features,
    memory_grounding_observation,
    strategy_grounding_observation,
)
from metacom_pm.v1_5_memory_transport import compile_bounded_memory
from metacom_pm.v1_5_strategy_rag_repair import (
    effect_study_observable_flags,
    effect_study_rank_applicable_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d3-step0-blueprint-v1"
STATUS = "PASS_READY_TO_PREPARE_320_FROZEN_RESPONSE_CALLS"
COMPONENTS = ("RS", "MP", "MS", "ME")

THEMES = (
    {
        "key": "deadlines",
        "profile": "works in a design role with frequent deadlines",
        "prior": "A cluster of work deadlines made it hard to decide what to do first.",
        "summary": "Work deadlines felt crowded, and choosing one priority reduced the pressure.",
        "current_high": "Several deadlines are competing again, and I feel overwhelmed about which task deserves attention first.",
        "current_low": "My workload is piling up again, and I feel stretched thin about where to begin.",
    },
    {
        "key": "friendship",
        "profile": "has a long-standing close friendship circle",
        "prior": "A close friendship became distant after several messages went unanswered.",
        "summary": "Uncertainty in a friendship brought sadness, and naming the unanswered messages helped clarify it.",
        "current_high": "A friendship feels uncertain again after a quiet week, and I am hurt but unsure what the silence means.",
        "current_low": "Someone close has gone quiet, and the distance is leaving me sad and confused.",
    },
    {
        "key": "exam",
        "profile": "is a part-time student preparing for professional exams",
        "prior": "Studying for an exam became difficult when concentration dropped after work.",
        "summary": "Exam preparation felt overwhelming, and a smaller study target was easier to approach.",
        "current_high": "Another exam is approaching, and I feel anxious because my concentration keeps slipping in the evening.",
        "current_low": "I need to prepare for a test, but after my shift my attention disappears and I do not know where to start.",
    },
    {
        "key": "move",
        "profile": "recently moved to a new city and lives alone",
        "prior": "After a move to a new city, the unfamiliar weekends felt especially lonely.",
        "summary": "The move disrupted familiar routines, and one regular local activity made the city feel less anonymous.",
        "current_high": "The move still feels difficult on weekends, and I feel lonely without a familiar person nearby.",
        "current_low": "My new surroundings still do not feel like home, and quiet days leave me isolated.",
    },
    {
        "key": "family",
        "profile": "shares a home with an adult family member",
        "prior": "A family conversation escalated when both people tried to settle every disagreement at once.",
        "summary": "A tense family conversation became more manageable after focusing on one concern at a time.",
        "current_high": "I want to revisit a family conversation, but I feel tense and do not want it to become another argument.",
        "current_low": "I need to speak with someone at home, yet I am worried the discussion will become heated again.",
    },
    {
        "key": "breakup",
        "profile": "is rebuilding routines after the end of a long relationship",
        "prior": "After a breakup, reminders of the relationship made ordinary evenings unexpectedly painful.",
        "summary": "The breakup brought waves of sadness, and allowing a slower pace reduced pressure to recover quickly.",
        "current_high": "A reminder of the breakup hit me today, and I feel sad that the hurt can return so suddenly.",
        "current_low": "Something reminded me of my former relationship, and the emotion came back more strongly than I expected.",
    },
    {
        "key": "caregiving",
        "profile": "helps care for an older relative several evenings each week",
        "prior": "Caregiving tasks filled the week and left almost no quiet time to recover.",
        "summary": "Caregiving pressure felt exhausting, and protecting one short break made the week more sustainable.",
        "current_high": "Caregiving has filled this week again, and I feel exhausted and guilty for wanting a little time alone.",
        "current_low": "Supporting a relative has taken most of my energy lately, and needing rest makes me feel selfish.",
    },
    {
        "key": "sleep",
        "profile": "works rotating shifts that change the weekly sleep schedule",
        "prior": "A changing sleep schedule made it hard to settle down after a late shift.",
        "summary": "Sleep became less predictable during rotating shifts, and a simple wind-down cue helped mark the transition.",
        "current_high": "My sleep schedule is unsettled after another shift change, and I am frustrated that I cannot switch off.",
        "current_low": "The latest rota change has left my nights irregular, and I feel worn down when bedtime arrives.",
    },
)

PREFERENCE_FAMILIES = (
    "use one concise reflection before asking anything",
    "ask no more than one focused question at a time",
    "avoid advice unless I explicitly invite it",
    "offer only one small optional idea rather than a plan",
    "name the concrete facts before interpreting feelings",
    "leave room for me to correct any paraphrase",
    "keep the response gentle and low burden when I am overwhelmed",
    "help me choose one priority instead of listing many tasks",
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


def _redundant_supporter_turn(family: str) -> str:
    return {
        "Question": "What part of this feels most important to you right now?",
        "Providing Suggestions": "You could choose one very small step and see how it feels.",
        "Reflection of feelings": "It sounds like this has been weighing heavily on you.",
        "Affirmation and Reassurance": "It makes sense that this situation is affecting you.",
        "Restatement or Paraphrasing": "So, this situation has become the main pressure for you right now.",
    }[family]


def _make_user(index: int) -> tuple[dict[str, Any], dict[str, Any], str]:
    # Each of the four match/grounding cells contains every theme once, so
    # topic words cannot identify a designed stratum.
    theme = THEMES[((index // 4) + 2 * (index % 4)) % len(THEMES)]
    subtype = "MP_PREFERENCE" if index < 16 else "MP_PROFILE"
    if subtype == "MP_PREFERENCE":
        basic = {
            "support_preference": (
                f"When discussing {theme['key']}, {PREFERENCE_FAMILIES[index % 8]}."
            )
        }
    else:
        basic = {"profile_context": str(theme["profile"])}
    sessions = []
    for session_no in range(1, 7):
        if session_no == 2:
            event = str(theme["prior"])
            summary = str(theme["summary"])
        elif session_no == 4:
            event = (
                f"A later check-in about {theme['key']} showed that naming one concrete "
                "concern made the situation easier to describe."
            )
            summary = (
                f"A later {theme['key']} check-in focused on one concrete concern and "
                "reduced conversational overload."
            )
        elif session_no == 6:
            event = (
                f"The most recent {theme['key']} episode returned unexpectedly, and the "
                "user wanted space to describe its immediate impact."
            )
            summary = (
                f"The most recent {theme['key']} session centered on its immediate impact "
                "and a preference for a measured response."
            )
        else:
            event = (
                f"Session {session_no} concerned an ordinary weekly routine, including "
                "meals, errands, and making time for a short break."
            )
            summary = (
                f"An ordinary routine check-in covered errands and a short break in week {session_no}."
            )
        sessions.append(
            {
                "id": f"d3_u{index + 1:03d}_s{session_no}",
                "timestamp": f"session-{session_no}",
                "summary": summary,
                "dialogue": [
                    {"role": "seeker", "content": event},
                    {"role": "supporter", "content": "Thank you for explaining that context."},
                ],
            }
        )
    user = {
        "id": f"pmv15_d3_u{index + 1:03d}",
        "basic_info": basic,
        "dialog_history": sessions,
    }
    current = str(theme["current_high"] if index % 4 < 2 else theme["current_low"])
    cue = (
        " I mainly want help putting this into words, not advice."
        if index % 4 == 0
        else " Could you give me one small optional suggestion?"
        if index % 4 == 1
        else " I am not sure which part matters most right now."
        if index % 4 == 2
        else " I feel worried, and I want to understand why this feels so heavy today."
    )
    state = {
        "state_id": "state_" + stable_hex(PROTOCOL, user["id"], n=24),
        "card_id": "card_" + stable_hex(PROTOCOL, user["id"], "backend", n=24),
        "user_id": user["id"],
        "split": "development",
        "semantic_family": f"d3_{theme['key']}",
        "current_user_text": normalize_space(current + cue),
        "current_session_history": [
            {
                "role": "user",
                "content": f"This week I have been thinking about {theme['key']} again.",
            },
            {
                "role": "assistant",
                "content": "I'm listening. Tell me what feels most relevant today.",
            },
        ],
        "current_session_summary": "",
        "session_index": 7,
    }
    return user, state, subtype


def _metadata_for_items(
    items: list[MemoryItem], subtype: str
) -> dict[str, dict[str, Any]]:
    return {
        item.memory_id: (
            {"mp_subtype": subtype}
            if item.source is MemorySource.MP
            else {"summary_origin": "supplied_strictly_prior_summary"}
            if item.source is MemorySource.MS
            else {"episode_origin": "strictly_prior_seeker_turn"}
        )
        for item in items
    }


def _inventory(items: list[MemoryItem], session_index: int) -> dict[MemorySource, SourceCatalog]:
    result: dict[MemorySource, SourceCatalog] = {}
    for source in MemorySource:
        rows = [item for item in items if item.source is source]
        ages = [session_index - int(item.created_session) for item in rows]
        result[source] = SourceCatalog(
            available=bool(rows),
            count=len(rows),
            min_age_sessions=min(ages) if ages else None,
            max_age_sessions=max(ages) if ages else None,
            estimated_tokens=sum(estimate_tokens(item.text) for item in rows),
            catalog_fingerprint=[],
            semantic_query_similarity=0.0,
            semantic_representation_valid=False,
        )
    return result


def _background(component: str, user_index: int) -> tuple[str, str]:
    bits = tuple(bool((user_index % 8) & (1 << position)) for position in range(3))
    if component == "RS":
        sources = frozenset(
            source for source, enabled in zip(MemorySource, bits, strict=True) if enabled
        )
        return (
            canonical_action_id(sources, StrategyMode.R0),
            canonical_action_id(sources, StrategyMode.RS),
        )
    target = MemorySource(component)
    other_sources = tuple(source for source in MemorySource if source is not target)
    sources = frozenset(
        source for source, enabled in zip(other_sources, bits[:2], strict=True) if enabled
    )
    strategy = StrategyMode.RS if bits[2] else StrategyMode.R0
    return (
        canonical_action_id(sources, strategy),
        canonical_action_id(frozenset({*sources, target}), strategy),
    )


def _action_memory_ids(
    action: str, discoveries: dict[MemorySource, Any]
) -> list[str]:
    sources, _ = parse_action_id(action)
    return [
        item.memory_id
        for source in sorted(sources, key=lambda value: value.value)
        for item in discoveries[source].selected_items
    ]


def _recursive_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        text = normalize_space(value)
        return [text] if len(text) >= 16 else []
    if isinstance(value, list):
        return [text for row in value for text in _recursive_strings(row)]
    if isinstance(value, dict):
        return [text for row in value.values() for text in _recursive_strings(row)]
    return []


def _content_disjoint_audit(generated: list[str]) -> dict[str, Any]:
    # External content is read only after the deterministic blueprint exists;
    # it is used for equality auditing, never for construction or selection.
    sources = {
        "ESConv": ROOT / "data/external/ESConv.json",
        "EvoEmo": ROOT / "data/external/evo_emo.json",
    }
    generated_set = {normalize_space(text).lower() for text in generated if len(normalize_space(text)) >= 16}
    overlaps: dict[str, int] = {}
    for name, path in sources.items():
        external = json.loads(path.read_text(encoding="utf-8"))
        external_set = {text.lower() for text in _recursive_strings(external)}
        overlaps[name] = len(generated_set & external_set)
    return {
        "audit_after_construction_only": True,
        "external_content_used_for_generation_or_selection": False,
        "generated_distinct_strings": len(generated_set),
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
    contrast_rows: list[dict[str, Any]] = []
    generated_texts: list[str] = []
    per_user: list[dict[str, Any]] = []

    drafts = [_make_user(index) for index in range(32)]
    initial_scores: dict[str, list[tuple[float, int]]] = {
        component: [] for component in COMPONENTS
    }
    for index, (user, raw_state, _) in enumerate(drafts):
        candidate = _strategy_candidate(raw_state, cards)
        if candidate is None:
            raise RuntimeError(f"initial RS candidate absent for draft {index + 1}")
        initial_scores["RS"].append((float(candidate["retrieval_score"]), index))
        draft_items, _ = compile_bounded_memory(user)
        draft_query = context_query(
            raw_state["current_user_text"], raw_state["current_session_history"], ""
        )
        draft_discoveries = discover_memory_candidates(
            query=draft_query,
            items=draft_items,
            retriever=retriever,
            session_index=raw_state["session_index"],
        )
        for source in MemorySource:
            if not draft_discoveries[source].selected_items:
                raise RuntimeError(
                    f"initial {source.value} candidate absent for draft {index + 1}"
                )
            initial_scores[source.value].append(
                (
                    float(
                        draft_discoveries[source].descriptor[
                            "top1_lexical_relevance"
                        ]
                    ),
                    index,
                )
            )
    # Alternate grounding assignment inside the outcome-blind natural-match
    # order.  This makes match and nonredundancy separately identifiable
    # instead of letting a wording template define both axes.
    low_grounding_indices = {
        component: {
            index
            for rank, (_, index) in enumerate(sorted(scores))
            if rank % 2 == 1
        }
        for component, scores in initial_scores.items()
    }

    for index in range(32):
        user, raw_state, subtype = drafts[index]
        items, _ = compile_bounded_memory(user)
        metadata = _metadata_for_items(items, subtype)
        # Discover once, then make every second state an explicit redundancy
        # stratum by repeating only the actually selected items in the visible
        # dialogue.  This affects neither the private catalog nor outcomes.
        query0 = context_query(
            raw_state["current_user_text"], raw_state["current_session_history"], ""
        )
        first = discover_memory_candidates(
            query=query0,
            items=items,
            retriever=retriever,
            session_index=raw_state["session_index"],
        )
        initial_strategy = _strategy_candidate(raw_state, cards)
        if initial_strategy is None or any(
            not first[source].selected_items for source in MemorySource
        ):
            raise RuntimeError(f"candidate discovery failed for {user['id']}")
        low_components = {
            component
            for component in COMPONENTS
            if index in low_grounding_indices[component]
        }
        redundancy_sentences = []
        if "MP" in low_components:
            redundancy_sentences.append(
                "You already stated your support preference and profile context in this same chat."
            )
        if "MS" in low_components:
            redundancy_sentences.append(
                "You already shared the prior session summary in this same chat."
            )
        if "ME" in low_components:
            redundancy_sentences.append(
                "You already described the relevant past event in this same chat."
            )
        if "RS" in low_components:
            redundancy_sentences.extend(
                [
                    "I have already used this same support technique in this chat.",
                    _redundant_supporter_turn(str(initial_strategy["strategy_family"])),
                ]
            )
        if redundancy_sentences:
            raw_state["current_session_history"][1]["content"] = normalize_space(
                " ".join(redundancy_sentences)
            )

        query = context_query(
            raw_state["current_user_text"], raw_state["current_session_history"], ""
        )
        discoveries = discover_memory_candidates(
            query=query,
            items=items,
            retriever=retriever,
            session_index=raw_state["session_index"],
        )
        strategy = _strategy_candidate(raw_state, cards)
        if strategy is None:
            raise RuntimeError(f"RS candidate vanished for {user['id']}")
        visible = _visible_dialogue(raw_state)
        memory_grounding = {
            source: memory_grounding_observation(
                source=source,
                selected_items=discoveries[source].selected_items,
                catalog_memory_ids={item.memory_id for item in items},
                catalog_user_id=user["id"],
                current_user_id=user["id"],
                current_session_index=raw_state["session_index"],
                current_user_text=raw_state["current_user_text"],
                visible_dialogue=visible,
                source_metadata=metadata,
            )
            for source in MemorySource
        }
        rs_grounding = strategy_grounding_observation(
            candidate=strategy,
            current_user_text=raw_state["current_user_text"],
            visible_dialogue=visible,
        )
        all_grounding = {"RS": rs_grounding, **{source.value: row for source, row in memory_grounding.items()}}
        hard = {
            component: row["hard_off_reasons"]
            for component, row in all_grounding.items()
            if row["deterministic_hard_off"]
        }
        if hard:
            raise RuntimeError(f"formal D3 pair candidate hard-off: {user['id']}: {hard}")

        runtime = RuntimeState(
            **raw_state,
            inventory=_inventory(items, raw_state["session_index"]),
            allowed_actions=list(ALL_ACTION_IDS),
            provenance={
                "d3_step0_protocol": PROTOCOL,
                "memory_compiler": "pm-v1.5-shared-all-causal-prior-history-memory-compiler-v2",
                "candidate_feature_protocol": D3_FEATURE_PROTOCOL,
                "outcome_read": False,
            },
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
        per_user.append(
            {
                "user_id": user["id"],
                "state_id": runtime.state_id,
                "mp_subtype": subtype,
                "designed_lower_grounding_components": sorted(low_components),
                "catalog_counts": {source.value: sum(item.source is source for item in items) for source in MemorySource},
            }
        )
        generated_texts.extend(_recursive_strings(user))
        generated_texts.extend(_recursive_strings(raw_state))

        for component in COMPONENTS:
            control, treatment = _background(component, index)
            if component == "RS":
                match = float(strategy["retrieval_score"])
                grounding = rs_grounding
                is_preference = False
            else:
                source = MemorySource(component)
                match = float(discoveries[source].descriptor["top1_lexical_relevance"])
                grounding = memory_grounding[source]
                is_preference = component == "MP" and subtype == "MP_PREFERENCE"
            slot_id = "d3_slot_" + stable_hex(PROTOCOL, user["id"], component, n=24)
            contrast_rows.append(
                {
                    "protocol": PROTOCOL,
                    "contrast_slot_id": slot_id,
                    "component": component,
                    "user_id": user["id"],
                    "state_id": runtime.state_id,
                    "card_id": runtime.card_id,
                    "control_action": control,
                    "treatment_action": treatment,
                    "background_action": control,
                    "mp_subtype": subtype if component == "MP" else None,
                    "component_candidate_observation": (
                        {"candidate": strategy, "grounding": grounding}
                        if component == "RS"
                        else {
                            "descriptor": discoveries[MemorySource(component)].descriptor,
                            "grounding": grounding,
                        }
                    ),
                    "model_features": d3_model_features(
                        component=component,
                        candidate_state_match_score=match,
                        grounding_score=float(grounding["candidate_grounding_or_nonredundancy_score"]),
                        background_action=control,
                        candidate_is_preference=is_preference,
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

    # Hard exclusions are tested separately and consume no generation calls.
    hard_controls: list[dict[str, Any]] = []
    for index in range(8):
        current = users[index]
        other = users[(index + 1) % 8]
        current_items = backends[index].items
        other_items = backends[(index + 1) % 8].items
        current_meta = metadata_rows[index]["memory_item_metadata"]
        visible = _visible_dialogue(states[index].model_dump(mode="json"))
        cases = []
        other_me = next(item for item in other_items if item.source is MemorySource.ME)
        cases.append(("wrong_user", MemorySource.ME, [other_me], other, current_meta, states[index].current_user_text, visible))
        current_ms = next(item for item in current_items if item.source is MemorySource.MS)
        future = current_ms.model_copy(update={"created_session": states[index].session_index})
        cases.append(("future_memory", MemorySource.MS, [future], current, current_meta, states[index].current_user_text, visible))
        stale = MemoryItem(
            memory_id="mem_" + stable_hex(PROTOCOL, current["id"], "stale", n=20),
            source=MemorySource.ME,
            created_session=1,
            text="Please give me three suggestions about a decision from that earlier session.",
        )
        cases.append(("stale_request", MemorySource.ME, [stale], current, current_meta, "I only want to describe how today feels.", visible))
        current_mp = next(item for item in current_items if item.source is MemorySource.MP)
        conflict_visible = [*visible, {"speaker": "seeker", "content": f"That changed; this is no longer true: {current_mp.text}"}]
        cases.append(("explicit_conflict", MemorySource.MP, [current_mp], current, current_meta, "That changed; this is no longer true now.", conflict_visible))
        for kind, source, selected, owner, meta, current_text, case_visible in cases:
            observation = memory_grounding_observation(
                source=source,
                selected_items=selected,
                catalog_memory_ids={item.memory_id for item in (other_items if kind == "wrong_user" else current_items)} | {item.memory_id for item in selected},
                catalog_user_id=owner["id"],
                current_user_id=current["id"],
                current_session_index=states[index].session_index,
                current_user_text=current_text,
                visible_dialogue=case_visible,
                source_metadata=meta,
            )
            hard_controls.append(
                {
                    "protocol": PROTOCOL,
                    "control_type": kind,
                    "component": source.value,
                    "observation": observation,
                    "generation_allowed": False,
                }
            )

    primary_pairs = []
    repeat_pairs = []
    repeat_ranges = {"RS": range(0, 8), "MP": range(8, 16), "MS": range(16, 24), "ME": range(24, 32)}
    contrast_by_key = {(row["component"], row["user_id"]): row for row in contrast_rows}
    for row in contrast_rows:
        seed = int(stable_hex(PROTOCOL, row["contrast_slot_id"], "primary", n=8), 16) % 2_000_000_000
        primary_pairs.append({"pair_role": "primary", "pair_seed": seed, **row})
    for component, indices in repeat_ranges.items():
        for index in indices:
            row = contrast_by_key[(component, users[index]["id"])]
            seed = int(stable_hex(PROTOCOL, row["contrast_slot_id"], "repeat", n=8), 16) % 2_000_000_000
            repeat_pairs.append({"pair_role": "outcome_blind_repeat", "pair_seed": seed, **row})

    background_counts = {
        component: Counter(row["background_action"] for row in contrast_rows if row["component"] == component)
        for component in COMPONENTS
    }
    feature_counts = {
        component: len(next(row for row in contrast_rows if row["component"] == component)["model_features"])
        for component in COMPONENTS
    }
    grounding_values = {
        component: sorted({row["model_features"]["candidate_grounding_or_nonredundancy_score"] for row in contrast_rows if row["component"] == component})
        for component in COMPONENTS
    }
    match_values = {
        component: [row["model_features"]["candidate_state_match_score"] for row in contrast_rows if row["component"] == component]
        for component in COMPONENTS
    }
    candidate_strata_counts = {}
    for component in COMPONENTS:
        component_rows = [row for row in contrast_rows if row["component"] == component]
        midpoint = median(
            row["model_features"]["candidate_state_match_score"]
            for row in component_rows
        )
        candidate_strata_counts[component] = Counter(
            (
                "higher_match"
                if row["model_features"]["candidate_state_match_score"] >= midpoint
                else "lower_match",
                "higher_grounding"
                if row["model_features"]["candidate_grounding_or_nonredundancy_score"] == 1.0
                else "lower_grounding",
            )
            for row in component_rows
        )
    content_audit = _content_disjoint_audit(generated_texts)
    forbidden = {"dataset", "user_id", "state_id", "card_id", "response", "judge", "outcome", "label", "topic", "template"}
    checks = {
        "32_new_users": len(users) == 32 and len({row["id"] for row in users}) == 32,
        "128_one_bit_contrasts": len(contrast_rows) == 128 and all(
            len(set(parse_action_id(row["control_action"])[0]) ^ set(parse_action_id(row["treatment_action"])[0]))
            + int(parse_action_id(row["control_action"])[1] != parse_action_id(row["treatment_action"])[1]) == 1
            for row in contrast_rows
        ),
        "all_16_runtime_actions_available": all(set(state.allowed_actions) == set(ALL_ACTION_IDS) for state in states),
        "backgrounds_exactly_four_each": all(set(counts.values()) == {4} and len(counts) == 8 for counts in background_counts.values()),
        "mp_subtypes_16_16": Counter(row["mp_subtype"] for row in contrast_rows if row["component"] == "MP") == Counter({"MP_PREFERENCE": 16, "MP_PROFILE": 16}),
        "mp_preference_families_at_least_8": len({next(iter(users[index]["basic_info"].values())) for index in range(16)}) >= 8,
        "ms_all_supplied_prior": all(
            all(value.get("summary_origin") == "supplied_strictly_prior_summary" for key, value in metadata_rows[index]["memory_item_metadata"].items() if next(item for item in backends[index].items if item.memory_id == key).source is MemorySource.MS)
            for index in range(32)
        ),
        "current_summary_empty": all(not state.current_session_summary for state in states),
        "formal_pairs_have_no_hard_off": all(not row["component_candidate_observation"]["grounding"]["deterministic_hard_off"] for row in contrast_rows),
        "32_hard_controls_fail_closed": len(hard_controls) == 32 and all(row["observation"]["deterministic_hard_off"] and not row["generation_allowed"] for row in hard_controls),
        "feature_capacity_at_most_6": feature_counts == {"RS": 5, "MP": 6, "MS": 5, "ME": 5},
        "no_forbidden_feature_names": all(not any(any(token in key.lower() for token in forbidden) for key in row["model_features"]) for row in contrast_rows),
        "features_numeric_only": all(all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in row["model_features"].values()) for row in contrast_rows),
        "match_varies_each_head": all(max(values) > min(values) for values in match_values.values()),
        "grounding_varies_each_head": all(len(values) >= 2 for values in grounding_values.values()),
        "match_grounding_axes_crossed": all(
            len(counts) == 4 and min(counts.values()) >= 4
            for counts in candidate_strata_counts.values()
        ),
        "32_outcome_blind_repeats": len(repeat_pairs) == 32 and Counter(row["component"] for row in repeat_pairs) == Counter({component: 8 for component in COMPONENTS}),
        "160_pairs_320_calls": len(primary_pairs) + len(repeat_pairs) == 160,
        "no_outcome_fields_populated": all(row["effect_label"].startswith("UNKNOWN") and not row["outcome_read"] for row in contrast_rows),
        "content_disjoint_exact_audit": content_audit["pass"],
    }
    if not all(checks.values()):
        raise RuntimeError(
            "D3 Step-0 preflight failed: "
            f"{[key for key, value in checks.items() if not value]}; "
            f"candidate_strata={candidate_strata_counts}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "synthetic_users.jsonl", users)
    write_jsonl(out_dir / "runtime_states.jsonl", [row.model_dump(mode="json") for row in states])
    write_jsonl(out_dir / "memory_backend.jsonl", [row.model_dump(mode="json") for row in backends])
    write_jsonl(out_dir / "memory_item_metadata.jsonl", metadata_rows)
    write_jsonl(out_dir / "d3_contrast_blueprint.jsonl", contrast_rows)
    write_jsonl(out_dir / "d3_primary_pairs.jsonl", primary_pairs)
    write_jsonl(out_dir / "d3_repeat_pairs.jsonl", repeat_pairs)
    write_jsonl(out_dir / "hard_gate_controls.jsonl", hard_controls)
    write_jsonl(out_dir / "user_shape_audit.jsonl", per_user)
    report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "api_calls_made": 0,
        "human_labels_read": False,
        "external_outcomes_read": False,
        "users": len(users),
        "state_contrasts": len(contrast_rows),
        "primary_pairs": len(primary_pairs),
        "repeat_pairs": len(repeat_pairs),
        "total_pairs": len(primary_pairs) + len(repeat_pairs),
        "planned_response_calls": 2 * (len(primary_pairs) + len(repeat_pairs)),
        "background_counts": {component: dict(sorted(counts.items())) for component, counts in background_counts.items()},
        "feature_counts": feature_counts,
        "grounding_values": grounding_values,
        "candidate_strata_counts": {
            component: {"|".join(key): value for key, value in sorted(counts.items())}
            for component, counts in candidate_strata_counts.items()
        },
        "match_ranges": {component: {"min": min(values), "max": max(values)} for component, values in match_values.items()},
        "content_disjoint_audit": content_audit,
        "checks": checks,
        "files": {},
    }
    for name in (
        "synthetic_users.jsonl",
        "runtime_states.jsonl",
        "memory_backend.jsonl",
        "memory_item_metadata.jsonl",
        "d3_contrast_blueprint.jsonl",
        "d3_primary_pairs.jsonl",
        "d3_repeat_pairs.jsonl",
        "hard_gate_controls.jsonl",
        "user_shape_audit.jsonl",
    ):
        report["files"][name] = sha256_file(out_dir / name)
    report["blueprint_semantic_sha256"] = sha256_text(canonical_json(contrast_rows))
    write_json(out_dir / "preflight_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": STATUS,
            "preflight_report_sha256": sha256_file(out_dir / "preflight_report.json"),
            "contrast_blueprint_sha256": sha256_file(out_dir / "d3_contrast_blueprint.jsonl"),
            "primary_pairs_sha256": sha256_file(out_dir / "d3_primary_pairs.jsonl"),
            "repeat_pairs_sha256": sha256_file(out_dir / "d3_repeat_pairs.jsonl"),
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
        default=ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_step0_blueprint_v1",
    )
    args = parser.parse_args()
    report = build(cards_path=args.cards, out_dir=args.out_dir)
    print(json.dumps({key: report[key] for key in ("protocol", "status", "users", "state_contrasts", "total_pairs", "planned_response_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
