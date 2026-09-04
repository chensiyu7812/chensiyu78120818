#!/usr/bin/env python3
"""Freeze the bounded real-text supplement for MP/MS/ME routing.

The supplement is outcome blind and zero API.  Sixteen users extend the D3
development fit for MP, MS, and ME; sixteen entirely separate users are sealed
as confirmation for all three heads.  Every candidate is produced by the shared
Evo-style compiler, source-specific query contract, production retriever, and
production opportunity feature bridge.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

from metacom_pm.contracts import (
    ALL_ACTION_IDS,
    MemoryBackendRecord,
    MemorySource,
    RuntimeState,
    SourceCatalog,
    StrategyMode,
    canonical_action_id,
)
from metacom_pm.io import (
    canonical_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.text import estimate_tokens, normalize_space
from metacom_pm.v1_5_candidate_discovery import (
    discover_memory_candidates_by_source_query,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    compile_bounded_memory_with_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-memory-opportunity-real-text-supplement-v1"
STATUS = "PASS_FROZEN_16_FIT_16_CONFIRMATION_MEMORY_SUPPLEMENT"

TRAIN_THEMES = (
    ("community choir", "Rehearsals became tense after two missed cues.", "A slower warm-up made the next rehearsal easier to enter.", "Tonight's choir rehearsal feels tense again, and I want to work out what matters now."),
    ("garden project", "A shared garden task became overwhelming when every job felt urgent.", "Choosing one small garden task reduced the pressure.", "The garden project feels crowded again, and I need to decide where to place my attention."),
    ("language class", "Speaking practice felt embarrassing after losing track of a sentence.", "Preparing one opening phrase made the next class less intimidating.", "My language class is coming up, and the thought of speaking aloud feels heavy today."),
    ("volunteer shift", "A volunteer shift became draining when several requests arrived together.", "Taking one request at a time made the shift more manageable.", "Another volunteer shift is approaching, and I feel stretched before it has even started."),
    ("creative workshop", "A workshop draft stalled after too many revisions were attempted at once.", "Naming one revision goal restored some momentum.", "My workshop draft feels stuck again, and I am unsure which concern deserves attention first."),
    ("commute change", "A changed commute made mornings unpredictable and tiring.", "Preparing one transition cue made the new route feel more familiar.", "The commute has changed again, and my mornings feel unsettled in a way I want to understand."),
    ("neighborhood group", "A neighborhood meeting felt isolating when familiar people were absent.", "Talking with one known person made the next meeting less anonymous.", "There is another neighborhood meeting soon, and I already feel alone thinking about it."),
    ("music practice", "Practice became frustrating when one difficult section dominated the session.", "Working on a short passage made practice feel possible again.", "Music practice is frustrating me again, and I want help naming what is getting in the way."),
)

CONFIRM_THEMES = (
    ("book club", "A book-club discussion felt awkward after an idea was dismissed quickly.", "Writing down one question helped the next discussion feel safer.", "The book club meets tomorrow, and I am uneasy about speaking after what happened before."),
    ("cycling route", "A new cycling route felt discouraging after getting lost twice.", "Marking one familiar landmark made the route easier to approach.", "I need to use the cycling route again, and the uncertainty is weighing on me today."),
    ("cooking course", "A cooking lesson became stressful when several techniques arrived at once.", "Focusing on one technique made the following lesson less overwhelming.", "The cooking course resumes this week, and I feel tense about keeping up."),
    ("library project", "A library project became confusing when responsibilities were left vague.", "Clarifying one responsibility made the next work session smoother.", "The library project is active again, and I am unsure which responsibility is really mine."),
    ("dance rehearsal", "A dance rehearsal felt exposing after forgetting part of the sequence.", "Reviewing one transition helped rebuild confidence in the next rehearsal.", "Dance rehearsal is tonight, and I feel self-conscious before walking into the room."),
    ("repair class", "A repair class became discouraging when a small mistake stopped progress.", "Asking about one step made the next attempt easier to continue.", "The repair class starts again soon, and I am worried I will freeze at the same point."),
    ("walking group", "A walking group felt distant when conversation moved too quickly to join.", "Starting beside one approachable member made the next walk less lonely.", "The walking group meets this weekend, and I feel on the outside before it begins."),
    ("photography task", "A photography task became frustrating when every image seemed wrong.", "Choosing one detail to observe made the next session more workable.", "The photography task is back, and I feel discouraged by how quickly I judge each attempt."),
)

SUPPORT_PREFERENCES = (
    "use one concise reflection before asking anything",
    "ask no more than one focused question at a time",
    "avoid advice unless I explicitly invite it",
    "offer only one small optional idea rather than a plan",
    "name concrete facts before interpreting feelings",
    "leave room for me to correct a paraphrase",
    "keep the reply low burden when I am overwhelmed",
    "help me choose one priority instead of listing tasks",
)

DISTRACTOR_EVENT = (
    "An ordinary household routine involved sorting a cupboard and planning errands."
)
DISTRACTOR_SUMMARY = (
    "A routine household check-in covered one cupboard task and ordinary errands."
)
DISTRACTOR_EVENT_ALT = (
    "A routine transit check involved checking a timetable and packing an umbrella."
)
DISTRACTOR_SUMMARY_ALT = (
    "A routine transit check-in covered one timetable and ordinary weather planning."
)


def _inventory(items: list[Any], session_index: int) -> dict[MemorySource, SourceCatalog]:
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


def _background(component: str, index: int) -> str:
    bits = tuple(bool((index % 8) & (1 << position)) for position in range(3))
    target = MemorySource(component)
    other = tuple(source for source in MemorySource if source is not target)
    sources = frozenset(
        source for source, enabled in zip(other, bits[:2], strict=True) if enabled
    )
    strategy = StrategyMode.RS if bits[2] else StrategyMode.R0
    return canonical_action_id(sources, strategy)


def _condition(component: str, local_index: int) -> tuple[str, int]:
    offset = {"MP": 0, "MS": 1, "ME": 2}[component]
    cell = (local_index + offset) % 4
    if cell == 0:
        return "relevant_nonredundant_direct", 1
    if cell == 1:
        return "relevant_nonredundant_with_background", 1
    if cell == 2:
        return "relevant_but_already_visible", 0
    return (
        "low_relevance_household_distractor"
        if (local_index // 4) % 2 == 0
        else "low_relevance_transit_distractor",
        0,
    )


def _make_user(
    *,
    role: str,
    local_index: int,
    protocol: str = PROTOCOL,
    confirmation_themes: tuple[tuple[str, str, str, str], ...] = CONFIRM_THEMES,
    user_prefix: str = "pmv15_mem",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[str, int]]]:
    themes = TRAIN_THEMES if role == "supplemental_fit" else confirmation_themes
    key, prior_event, prior_summary, current = themes[local_index % len(themes)]
    subtype = "MP_PREFERENCE" if local_index < 8 else "MP_PROFILE"
    mp_condition, _ = _condition("MP", local_index)
    ms_condition, _ = _condition("MS", local_index)
    me_condition, _ = _condition("ME", local_index)

    if mp_condition.startswith("low_relevance_"):
        alternate = mp_condition == "low_relevance_transit_distractor"
        mp_value = (
            (
                "When discussing an ordinary transit routine, keep the reply concise."
                if alternate
                else "When discussing an ordinary household routine, keep the reply concise."
            )
            if subtype == "MP_PREFERENCE"
            else (
                "regularly checks an ordinary transit routine"
                if alternate
                else "regularly organizes an ordinary household routine"
            )
        )
    elif subtype == "MP_PREFERENCE":
        mp_value = f"When discussing {key}, {SUPPORT_PREFERENCES[local_index % 8]}."
    else:
        mp_value = f"regularly participates in a {key}"
    basic_info = {
        (
            "stable_preference_support_style"
            if subtype == "MP_PREFERENCE"
            else "profile_context"
        ): mp_value
    }

    sessions: list[dict[str, Any]] = []
    for session_no in range(1, 9):
        if me_condition.startswith("low_relevance_"):
            event = (
                DISTRACTOR_EVENT_ALT
                if me_condition == "low_relevance_transit_distractor"
                else DISTRACTOR_EVENT
            )
        elif session_no in {2, 6}:
            event = prior_event
        else:
            event = (
                f"A routine week included meals, errands, and a short break before the {key}."
            )
        if ms_condition.startswith("low_relevance_"):
            summary = (
                DISTRACTOR_SUMMARY_ALT
                if ms_condition == "low_relevance_transit_distractor"
                else DISTRACTOR_SUMMARY
            )
        elif session_no in {2, 6}:
            summary = prior_summary
        else:
            summary = (
                f"An ordinary routine check-in preceded the {key} and contained no major change."
            )
        sessions.append(
            {
                "id": f"{role}_{local_index + 1:02d}_s{session_no}",
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

    user_id = f"{user_prefix}_{'fit' if role == 'supplemental_fit' else 'confirm'}_u{local_index + 1:02d}"
    user = {"id": user_id, "basic_info": basic_info, "dialog_history": sessions}
    markers = []
    for component, condition in (
        ("MP", mp_condition),
        ("MS", ms_condition),
        ("ME", me_condition),
    ):
        if condition != "relevant_but_already_visible":
            continue
        markers.append(
            {
                "MP": "You already stated your support preference and profile context in this same chat.",
                "MS": "You already shared the prior session summary in this same chat.",
                "ME": "You already described the relevant past event in this same chat.",
            }[component]
        )
    history = [
        {
            "role": "user",
            "content": (
                f"I have been thinking about the {key} again as part of my routine."
                if role == "supplemental_fit"
                else f"Today brought the {key} back to mind during a different routine."
            ),
        },
        {
            "role": "assistant",
            "content": normalize_space(
                "I'm listening and will stay with what is relevant now. " + " ".join(markers)
            ),
        },
    ]
    state = {
        "state_id": "state_" + stable_hex(protocol, user_id, n=24),
        "card_id": "card_" + stable_hex(protocol, user_id, "catalog", n=24),
        "user_id": user_id,
        "split": "train" if role == "supplemental_fit" else "validation",
        "semantic_family": key.replace(" ", "_"),
        "current_user_text": normalize_space(
            current
            + (
                " I want one measured response that uses only what is relevant now."
                if role == "supplemental_fit"
                else " Please stay focused on what this situation calls for today."
            )
        ),
        "current_session_history": history,
        "current_session_summary": "",
        "session_index": 9,
    }
    return user, state, {
        component: _condition(component, local_index)
        for component in ("MP", "MS", "ME")
    }


def _recursive_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        text = normalize_space(value)
        return [text] if len(text) >= 16 else []
    if isinstance(value, list):
        return [text for row in value for text in _recursive_strings(row)]
    if isinstance(value, dict):
        return [text for row in value.values() for text in _recursive_strings(row)]
    return []


def _content_disjoint_audit(generated: list[str], root: Path) -> dict[str, Any]:
    generated_set = {normalize_space(text).lower() for text in generated}
    overlaps: dict[str, int] = {}
    for name, path in {
        "ESConv": root / "data/external/ESConv.json",
        "EvoEmo": root / "data/external/evo_emo.json",
    }.items():
        external = json.loads(path.read_text(encoding="utf-8"))
        external_set = {text.lower() for text in _recursive_strings(external)}
        overlaps[name] = len(generated_set & external_set)
    return {
        "audit_after_construction_only": True,
        "external_content_used_for_construction_or_selection": False,
        "exact_normalized_overlap_counts": overlaps,
        "pass": not any(overlaps.values()),
    }


def build(
    *,
    root: Path = ROOT,
    protocol: str = PROTOCOL,
    status: str = STATUS,
    out_dir_name: str = "pm_v1_5_memory_opportunity_supplement_v1",
    confirmation_themes: tuple[tuple[str, str, str, str], ...] = CONFIRM_THEMES,
    user_prefix: str = "pmv15_mem",
    roles: tuple[str, ...] = ("supplemental_fit", "untouched_confirmation"),
    observation_builder: Callable[..., dict[str, Any]] = (
        build_memory_opportunity_observation
    ),
    feature_protocol: str = MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    match_feature_name: str = "candidate_state_match_score",
) -> dict[str, Any]:
    if not roles or not set(roles) <= {
        "supplemental_fit",
        "untouched_confirmation",
    }:
        raise ValueError("roles must be a non-empty subset of the two frozen roles")
    out_dir = root / "outputs" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    retriever = MemoryRetriever(
        minimum_score_by_source={source: 0.0 for source in MemorySource}
    )
    users: list[dict[str, Any]] = []
    states: list[RuntimeState] = []
    backends: list[MemoryBackendRecord] = []
    metadata_rows: list[dict[str, Any]] = []
    candidate_surfaces: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    generated_text: list[str] = []

    for role in roles:
        for local_index in range(16):
            user, raw_state, conditions = _make_user(
                role=role,
                local_index=local_index,
                protocol=protocol,
                confirmation_themes=confirmation_themes,
                user_prefix=user_prefix,
            )
            items, _, metadata = compile_bounded_memory_with_metadata(user)
            runtime = RuntimeState(
                **raw_state,
                inventory=_inventory(items, int(raw_state["session_index"])),
                allowed_actions=list(ALL_ACTION_IDS),
                provenance={
                    "protocol": protocol,
                    "evidence_role": role,
                    "outcome_read": False,
                },
            )
            backend = MemoryBackendRecord(card_id=runtime.card_id, items=items)
            queries = source_specific_memory_queries(
                runtime.current_user_text,
                [turn.model_dump(mode="json") for turn in runtime.current_session_history],
                runtime.current_session_summary,
            )
            discoveries = discover_memory_candidates_by_source_query(
                queries=queries,
                items=items,
                retriever=retriever,
                session_index=runtime.session_index,
            )
            visible = [
                turn.model_dump(mode="json")
                for turn in runtime.current_session_history
            ]
            formal_components = ("MP", "MS", "ME")
            for component in formal_components:
                source = MemorySource(component)
                condition, target = conditions[component]
                observation = observation_builder(
                    candidate=discoveries[source],
                    catalog_items=items,
                    catalog_user_id=user["id"],
                    current_user_id=user["id"],
                    current_session_index=runtime.session_index,
                    current_user_text=runtime.current_user_text,
                    visible_dialogue=visible,
                    source_metadata=metadata,
                    background_action=_background(component, local_index),
                )
                if observation["deterministic_hard_off"]:
                    raise RuntimeError(
                        f"formal supplement unexpectedly hard-off: {user['id']} {component}"
                    )
                if condition == "relevant_but_already_visible" and observation[
                    "candidate_audit"
                ]["grounding_checks"]["not_already_visible"]:
                    raise RuntimeError("redundancy condition did not compile")
                if condition != "relevant_but_already_visible" and not observation[
                    "candidate_audit"
                ]["grounding_checks"]["not_already_visible"]:
                    raise RuntimeError("nonredundant condition compiled as redundant")
                rows.append(
                    {
                        "protocol": protocol,
                        "evidence_role": role,
                        "component": component,
                        "user_id_private_not_model_input": user["id"],
                        "semantic_family_private_not_model_input": runtime.semantic_family,
                        "condition_family_private_not_model_input": condition,
                        "opportunity_target": target,
                        "model_features": observation["model_features"],
                        "deterministic_hard_off": False,
                        "feature_protocol": observation["protocol"],
                        "compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
                        "response_quality_risk_or_external_outcome_read": False,
                    }
                )
                candidate_surfaces.append(
                    {
                        "protocol": protocol,
                        "evidence_role": role,
                        "component": component,
                        "user_id_private_not_model_input": user["id"],
                        "query_private_semantic_encoder_input": queries[source],
                        "selected_candidate_texts_private_semantic_encoder_input": [
                            item.text for item in discoveries[source].selected_items
                        ],
                        "raw_text_not_a_direct_logistic_feature": True,
                        "outcome_read": False,
                    }
                )
            users.append(user)
            states.append(runtime)
            backends.append(backend)
            metadata_rows.append(
                {"card_id": runtime.card_id, "memory_item_metadata": metadata}
            )
            generated_text.extend(_recursive_strings(user))
            generated_text.extend(_recursive_strings(raw_state))

    count_by_role_component: dict[str, dict[str, Any]] = {}
    for role in roles:
        for component in ("MP", "MS", "ME"):
            values = [
                row
                for row in rows
                if row["evidence_role"] == role and row["component"] == component
            ]
            count_by_role_component[f"{role}:{component}"] = {
                "rows": len(values),
                "users": len(
                    {row["user_id_private_not_model_input"] for row in values}
                ),
                "class_counts": dict(
                    sorted(Counter(row["opportunity_target"] for row in values).items())
                ),
                "condition_counts": dict(
                    sorted(
                        Counter(
                            row["condition_family_private_not_model_input"]
                            for row in values
                        ).items()
                    )
                ),
            }
    content_audit = _content_disjoint_audit(generated_text, root)
    match_separation: dict[str, dict[str, float]] = {}
    for key, value in count_by_role_component.items():
        if not value["rows"]:
            continue
        role, component = key.split(":", 1)
        values = [
            row
            for row in rows
            if row["evidence_role"] == role and row["component"] == component
        ]
        low = [
            float(row["model_features"][match_feature_name])
            for row in values
            if row["condition_family_private_not_model_input"].startswith(
                "low_relevance_"
            )
        ]
        relevant = [
            float(row["model_features"][match_feature_name])
            for row in values
            if not row["condition_family_private_not_model_input"].startswith(
                "low_relevance_"
            )
        ]
        match_separation[key] = {
            "low_relevance_mean": sum(low) / len(low),
            "relevant_mean": sum(relevant) / len(relevant),
        }
    forbidden = {
        "dataset",
        "user_id",
        "state_id",
        "response",
        "judge",
        "outcome",
        "label",
        "topic",
        "template",
    }
    checks = {
        (
            "32_new_users_16_fit_16_confirmation"
            if set(roles) == {"supplemental_fit", "untouched_confirmation"}
            else "expected_new_users_and_roles"
        ): (
            len(users) == 16 * len(roles)
            and len({user["id"] for user in users}) == 16 * len(roles)
            and sum(state.split == "train" for state in states)
            == (16 if "supplemental_fit" in roles else 0)
            and sum(state.split == "validation" for state in states)
            == (16 if "untouched_confirmation" in roles else 0)
        ),
        "fit_confirmation_semantic_families_disjoint": not (
            {
                state.semantic_family
                for state in states
                if state.split == "train"
            }
            & {
                state.semantic_family
                for state in states
                if state.split == "validation"
            }
        ),
        (
            "96_formal_rows"
            if set(roles) == {"supplemental_fit", "untouched_confirmation"}
            else "expected_formal_rows"
        ): len(rows) == 48 * len(roles),
        (
            "each_memory_head_each_role_16_rows_8_on_8_off"
            if set(roles) == {"supplemental_fit", "untouched_confirmation"}
            else "each_memory_head_each_requested_role_16_rows_8_on_8_off"
        ): all(
            count_by_role_component[f"{role}:{component}"]["rows"] == 16
            and count_by_role_component[f"{role}:{component}"]["class_counts"]
            == {0: 8, 1: 8}
            for role in roles
            for component in ("MP", "MS", "ME")
        ),
        "five_condition_families_each_formal_split": all(
            len(value["condition_counts"]) == 5
            and sorted(value["condition_counts"].values()) == [2, 2, 4, 4, 4]
            for value in count_by_role_component.values()
            if value["rows"]
        ),
        "same_production_feature_protocol": all(
            row["feature_protocol"] == feature_protocol
            for row in rows
        ),
        "low_relevance_compiles_below_relevant_each_formal_split": all(
            value["low_relevance_mean"] < value["relevant_mean"]
            for value in match_separation.values()
        ),
        "no_hard_off_in_formal_rows": all(
            not row["deterministic_hard_off"] for row in rows
        ),
        "no_forbidden_model_features": all(
            not any(
                any(token in name.lower() for token in forbidden)
                for name in row["model_features"]
            )
            for row in rows
        ),
        "model_features_numeric": all(
            all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in row["model_features"].values()
            )
            for row in rows
        ),
        "external_exact_content_disjoint": content_audit["pass"],
        "zero_api_zero_outcome": all(
            not row["response_quality_risk_or_external_outcome_read"] for row in rows
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"memory supplement preflight failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )

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
    write_jsonl(out_dir / "candidate_semantic_surfaces.jsonl", candidate_surfaces)
    rows_path = out_dir / "opportunity_rows.jsonl"
    write_jsonl(rows_path, rows)
    report = {
        "protocol": protocol,
        "status": status,
        "scientific_role": (
            "Real-text, production-compiled outcome-blind routing supplement. "
            "The confirmation half is sealed from model, threshold, and feature selection."
        ),
        "api_calls_made": 0,
        "human_labels_read": False,
        "external_outcomes_read": False,
        "counts": count_by_role_component,
        "match_separation": match_separation,
        "content_disjoint_audit": content_audit,
        "checks": checks,
        "feature_protocol": feature_protocol,
        "compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
    }
    write_json(out_dir / "preflight_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": protocol,
            "status": status,
            "outcome_blind": True,
            "confirmation_sealed_before_fit": True,
            "files": {
                name: sha256_file(out_dir / name)
                for name in (
                    "synthetic_users.jsonl",
                    "runtime_states.jsonl",
                    "memory_backend.jsonl",
                    "memory_item_metadata.jsonl",
                    "candidate_semantic_surfaces.jsonl",
                    "opportunity_rows.jsonl",
                    "preflight_report.json",
                )
            },
            "semantic_rows_sha256": sha256_text(canonical_json(rows)),
        },
    )
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "counts": report["counts"],
            "checks": report["checks"],
        }
    )


if __name__ == "__main__":
    main()
