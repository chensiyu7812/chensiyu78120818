#!/usr/bin/env python3
"""Validate one or more V5.3 formal longitudinal catalog users.

This is a content-ingestion gate, not a Step1 labeler.  It deliberately keeps
four conclusions separate:

1. hard schema/identity/time/span violations;
2. contract quota violations;
3. legacy ME-regex compiler coverage;
4. checks that only become meaningful across a batch or under semantic review.

In particular, a natural reusable outcome with exact, owner-correct action and
result spans is not silently rewritten to satisfy the legacy regex compiler.
Such a row is preserved and the compiler-coverage blocker is reported.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import re
from typing import Any, Iterable

from metacom_pm.io import write_json
from metacom_pm.v1_5_v5_2_atomic_memory import compile_atomic_reusable_outcome


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    ROOT / "data/pm_v1_5_contracts/v5_3_complete_training_data_generation_v1.json"
)
DEFAULT_OUT_DIR = (
    ROOT / "outputs/pm_v1_5_v5_3_formal_longitudinal_user_validation_v1"
)

PRIMARY_SUPERDOMAINS = (
    "work_education",
    "relationships",
    "family_caregiving",
    "relocation_culture",
    "sleep_health_energy",
    "grief_life_transition",
    "finance_housing",
    "social_identity_creative",
)
PROFILE_FIELDS = {"name", "age", "gender", "job", "education", "nationality", "location"}
PREFERENCE_TYPES = {
    "concise_factual_answer",
    "reflection_before_question",
    "one_optional_suggestion",
    "listen_only_no_advice",
    "direct_answer_before_explanation",
    "choices_rather_than_commands",
}
PRIMARY_SUPERDOMAIN_ALIASES = {
    "relationships_trust": "relationships",
}
ME_SUBTYPES = {
    "ME_REUSABLE_OUTCOME",
    "ME_UNRESOLVED_EVENT",
    "ME_CONTEXT_EVENT",
}
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
NAME_GROUNDING_IGNORED_TOKENS = {
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "professor",
    "aunt",
    "auntie",
    "uncle",
}


def _grounding_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in WORD_RE.findall(str(text).replace("’", "'")):
        token = raw.casefold()
        if token.endswith("'s"):
            token = token[:-2]
        if token and token not in NAME_GROUNDING_IGNORED_TOKENS:
            tokens.add(token)
    return tokens


def _read_json_values(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty input: {path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        values = value if isinstance(value, list) else [value]
    if not all(isinstance(row, dict) for row in values):
        raise ValueError(f"input must contain JSON object(s): {path}")
    return list(values)


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl", ".txt"}
    )
    if not files:
        raise ValueError(f"no JSON/JSONL/TXT inputs in {path}")
    return files


def _expected_assignment(user_id: str, content_author: str, contract: dict[str, Any]) -> dict[str, Any] | None:
    match = re.fullmatch(r"p2r_formal_(gpt|claude)_u(\d{3})", user_id)
    if not match:
        return None
    author_key, ordinal_text = match.groups()
    expected_author = "chatgpt_pro" if author_key == "gpt" else "claude"
    if content_author != expected_author:
        return None
    ordinal = int(ordinal_text)
    if not 0 <= ordinal < 40:
        return None
    domain_index, within_author_domain = divmod(ordinal, 5)
    positions_key = f"{expected_author.split('_')[0]}_schedule_positions_per_superdomain"
    if expected_author == "chatgpt_pro":
        positions_key = "chatgpt_schedule_positions_per_superdomain"
    positions = contract["catalog_authors"][positions_key]
    schedule_position = int(positions[within_author_domain])
    return {
        "expected_author": expected_author,
        "domain_index": domain_index,
        "expected_primary_superdomain": PRIMARY_SUPERDOMAINS[domain_index],
        "schedule_position": schedule_position,
        "sessions": int(contract["session_counts_per_superdomain"][schedule_position]),
        "relationships": int(
            contract["catalog_targets"]["relationships_per_user_schedule"][schedule_position]
        ),
        "events": int(contract["catalog_targets"]["events_per_user_schedule"][schedule_position]),
        "profile_updates": 2 if schedule_position <= 1 else 1 if schedule_position <= 3 else 0,
    }


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    severity: str = "hard",
    item_id: str | None = None,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "item_id": item_id,
            "message": message,
        }
    )


def _candidate_rows(user: dict[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    for session in user.get("sessions", []):
        for candidate in session.get("typed_candidates", []):
            yield session, candidate


def _normalize_user_schema(user: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Losslessly normalize known pilot schema aliases into the canonical form.

    This adapter may rename fields, derive cross-reference arrays, and copy an
    already-verbatim span from a typed candidate into its top-level history
    record.  It must never author or paraphrase user content.
    """

    value = copy.deepcopy(user)
    actions: list[str] = []
    primary_superdomain = value.get("primary_superdomain")
    if primary_superdomain in PRIMARY_SUPERDOMAIN_ALIASES:
        value["primary_superdomain"] = PRIMARY_SUPERDOMAIN_ALIASES[primary_superdomain]
        actions.append(f"primary_superdomain:{primary_superdomain}->{value['primary_superdomain']}")
    sessions = list(value.get("sessions") or [])
    typed = [candidate for session in sessions for candidate in session.get("typed_candidates", [])]
    profile_evidence = {
        candidate.get("profile_item_id"): candidate
        for candidate in typed
        if candidate.get("subtype") == "MP_PROFILE" and candidate.get("profile_item_id")
    }
    preference_evidence = {
        candidate.get("preference_item_id"): candidate
        for candidate in typed
        if candidate.get("subtype") == "MP_PREFERENCE" and candidate.get("preference_item_id")
    }

    if "topic_threads" not in value and "recurring_topic_threads" in value:
        value["topic_threads"] = [
            {"thread_id": row.get("thread_id"), "label": row.get("label")}
            for row in value.get("recurring_topic_threads", [])
        ]
        actions.append("recurring_topic_threads->topic_threads")

    normalized_profiles: list[dict[str, Any]] = []
    for row in value.get("profile_history", []):
        item = dict(row)
        if "item_id" not in item and item.get("profile_item_id"):
            item["item_id"] = item["profile_item_id"]
            actions.append("profile_item_id->item_id")
        item.setdefault("subtype", "MP_PROFILE")
        evidence = profile_evidence.get(item.get("item_id"), {})
        item.setdefault("item_role", "update" if item.get("supersedes_item_id") else "base")
        item.setdefault("source_turn_ids", evidence.get("source_turn_ids"))
        item.setdefault("literal_source_span", evidence.get("literal_source_span"))
        scope = item.get("applicability_scope")
        if isinstance(scope, str):
            item["applicability_scope"] = [scope]
            actions.append("profile_applicability_scope_string->list")
        normalized_profiles.append(item)
    value["profile_history"] = normalized_profiles

    normalized_preferences: list[dict[str, Any]] = []
    for row in value.get("response_preference_history", []):
        item = dict(row)
        if "item_id" not in item and item.get("preference_item_id"):
            item["item_id"] = item["preference_item_id"]
            actions.append("preference_item_id->item_id")
        item.setdefault("subtype", "MP_PREFERENCE")
        evidence = preference_evidence.get(item.get("item_id"), {})
        item.setdefault("source_turn_ids", evidence.get("source_turn_ids"))
        item.setdefault("literal_source_span", evidence.get("literal_source_span"))
        item.setdefault("preference_text", evidence.get("literal_source_span"))
        normalized_preferences.append(item)
    value["response_preference_history"] = normalized_preferences

    normalized_relationships: list[dict[str, Any]] = []
    for row in value.get("relationships", []):
        item = dict(row)
        if "relationship" not in item and "relation" in item:
            item["relationship"] = item["relation"]
            actions.append("relationship.relation->relationship")
        if "valid_from_session" not in item and "first_mentioned_session" in item:
            item["valid_from_session"] = item["first_mentioned_session"]
            actions.append("relationship.first_mentioned_session->valid_from_session")
        item.setdefault("valid_until_session", None)
        normalized_relationships.append(item)
    value["relationships"] = normalized_relationships

    normalized_events: list[dict[str, Any]] = []
    for row in value.get("events", []):
        item = dict(row)
        if "topic_thread_ids" not in item and item.get("thread_id"):
            item["topic_thread_ids"] = [item["thread_id"]]
            actions.append("event.thread_id->topic_thread_ids")
        normalized_events.append(item)
    value["events"] = normalized_events
    events_by_session: dict[int, list[str]] = defaultdict(list)
    for event in normalized_events:
        if isinstance(event.get("session_index"), int) and event.get("event_id"):
            events_by_session[int(event["session_index"])].append(str(event["event_id"]))

    for session in sessions:
        if "event_ids" not in session:
            session["event_ids"] = events_by_session.get(int(session.get("session_index") or 0), [])
            actions.append("derive_session_event_ids")
        for candidate in session.get("typed_candidates", []):
            subtype = candidate.get("subtype")
            if subtype in ME_SUBTYPES:
                candidate.setdefault("candidate_text", candidate.get("literal_source_span"))
            if subtype == "ME_REUSABLE_OUTCOME" and "me_tier" not in candidate and "tier" in candidate:
                candidate["me_tier"] = candidate["tier"]
                actions.append("ME.tier->me_tier")
    value["sessions"] = sessions
    return value, sorted(set(actions))


def _validate_user(user: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    user_id = str(user.get("user_id") or "")
    author = str(user.get("content_author") or "")
    assignment = _expected_assignment(user_id, author, contract)
    if user.get("protocol") != "pm-v1.5-v5.3-formal-longitudinal-user-v1":
        _add_issue(issues, code="PROTOCOL", message="unexpected or missing protocol")
    if assignment is None:
        _add_issue(issues, code="USER_ASSIGNMENT", message="user_id/content_author assignment is invalid")

    sessions = list(user.get("sessions") or [])
    relationships = list(user.get("relationships") or [])
    events = list(user.get("events") or [])
    topics = list(user.get("topic_threads") or [])
    profiles = list(user.get("profile_history") or [])
    preferences = list(user.get("response_preference_history") or [])

    if assignment is not None:
        if user.get("primary_superdomain") != assignment["expected_primary_superdomain"]:
            _add_issue(
                issues,
                code="PRIMARY_SUPERDOMAIN",
                message=(
                    f"expected {assignment['expected_primary_superdomain']!r}, got "
                    f"{user.get('primary_superdomain')!r}"
                ),
            )
        for key, rows in (("sessions", sessions), ("relationships", relationships), ("events", events)):
            if len(rows) != assignment[key]:
                _add_issue(
                    issues,
                    code=f"COUNT_{key.upper()}",
                    message=f"expected {assignment[key]} {key}, got {len(rows)}",
                )

    if not 4 <= len(topics) <= 6:
        _add_issue(issues, code="TOPIC_THREAD_COUNT", message=f"expected 4-6 topic threads, got {len(topics)}")
    topic_ids = [str(row.get("thread_id") or "") for row in topics]
    if len(topic_ids) != len(set(topic_ids)) or "" in topic_ids:
        _add_issue(issues, code="TOPIC_THREAD_IDS", message="topic thread IDs must be non-empty and unique")
    topic_set = set(topic_ids)

    session_indices = [row.get("session_index") for row in sessions]
    if session_indices != list(range(1, len(sessions) + 1)):
        _add_issue(issues, code="SESSION_INDEX", message="session indices must be contiguous and ordered from 1")

    turn_lookup: dict[str, dict[str, Any]] = {}
    turn_session: dict[str, int] = {}
    session_user_text: dict[int, str] = {}
    entity_session_indices: dict[str, set[int]] = defaultdict(set)
    event_session_indices: dict[str, set[int]] = defaultdict(set)
    event_ids = {str(row.get("event_id") or "") for row in events}
    entity_ids = {"self"} | {str(row.get("entity_id") or "") for row in relationships}
    relationship_valid_from = {
        str(row.get("entity_id") or ""): row.get("valid_from_session")
        for row in relationships
    }
    if "" in event_ids or len(event_ids) != len(events):
        _add_issue(issues, code="EVENT_IDS", message="event IDs must be non-empty and unique")
    if "" in entity_ids or len(entity_ids) != len(relationships) + 1:
        _add_issue(issues, code="ENTITY_IDS", message="relationship entity IDs must be non-empty and unique")

    for session in sessions:
        index = session.get("session_index")
        dialogue = list(session.get("dialogue") or [])
        if isinstance(index, int):
            session_user_text[index] = "\n".join(
                str(turn.get("text") or "")
                for turn in dialogue
                if turn.get("role") == "user"
            )
        if not 2 <= len(dialogue) <= 6:
            _add_issue(
                issues,
                code="DIALOGUE_TURN_COUNT",
                message=f"session {index}: expected 2-6 turns, got {len(dialogue)}",
            )
        if not any(turn.get("role") == "user" for turn in dialogue):
            _add_issue(issues, code="SESSION_NO_USER_TURN", message=f"session {index} has no user turn")
        for turn in dialogue:
            turn_id = str(turn.get("turn_id") or "")
            if not turn_id or turn_id in turn_lookup:
                _add_issue(issues, code="TURN_ID", message=f"missing or duplicate turn ID {turn_id!r}")
            else:
                turn_lookup[turn_id] = turn
                turn_session[turn_id] = int(index)
            if turn.get("role") not in {"user", "assistant"}:
                _add_issue(issues, code="TURN_ROLE", message=f"invalid turn role for {turn_id!r}")
            if not str(turn.get("text") or "").strip():
                _add_issue(issues, code="TURN_TEXT", message=f"empty turn text for {turn_id!r}")
        summary_words = WORD_RE.findall(str(session.get("summary") or ""))
        if not 12 <= len(summary_words) <= 40:
            _add_issue(
                issues,
                code="SUMMARY_LENGTH",
                message=f"session {index}: summary has {len(summary_words)} words, expected 12-40",
            )
        for thread_id in session.get("topic_thread_ids", []):
            if thread_id not in topic_set:
                _add_issue(issues, code="SESSION_TOPIC_REF", message=f"session {index}: unknown topic {thread_id!r}")
        for entity_id in session.get("entity_ids", []):
            if entity_id not in entity_ids:
                _add_issue(issues, code="SESSION_ENTITY_REF", message=f"session {index}: unknown entity {entity_id!r}")
            elif entity_id != "self" and isinstance(index, int):
                entity_session_indices[entity_id].add(index)
                valid_from = relationship_valid_from.get(entity_id)
                if isinstance(valid_from, int) and index < valid_from:
                    _add_issue(
                        issues,
                        code="SESSION_ENTITY_BEFORE_VALID_FROM",
                        item_id=entity_id,
                        message=f"session {index} references entity before valid_from_session={valid_from}",
                    )
        for event_id in session.get("event_ids", []):
            if event_id not in event_ids:
                _add_issue(issues, code="SESSION_EVENT_REF", message=f"session {index}: unknown event {event_id!r}")
            elif isinstance(index, int):
                event_session_indices[event_id].add(index)

    for event in events:
        event_id = str(event.get("event_id") or "")
        event_session = event.get("session_index")
        if not isinstance(event_session, int) or not 1 <= event_session <= len(sessions):
            _add_issue(issues, code="EVENT_SESSION", item_id=event_id, message="event session_index is invalid")
        elif event_session_indices.get(event_id, set()) != {event_session}:
            _add_issue(
                issues,
                code="EVENT_SESSION_REF_DRIFT",
                item_id=event_id,
                message=(
                    f"event declares session {event_session}, but session event_ids reference it at "
                    f"{sorted(event_session_indices.get(event_id, set()))}"
                ),
            )
        for thread_id in event.get("topic_thread_ids", []):
            if thread_id not in topic_set:
                _add_issue(issues, code="EVENT_TOPIC_REF", item_id=event_id, message=f"unknown topic {thread_id!r}")
        for entity_id in event.get("entity_ids", []):
            if entity_id not in entity_ids:
                _add_issue(issues, code="EVENT_ENTITY_REF", item_id=event_id, message=f"unknown entity {entity_id!r}")
            else:
                valid_from = relationship_valid_from.get(entity_id)
                if (
                    entity_id != "self"
                    and isinstance(event_session, int)
                    and isinstance(valid_from, int)
                    and event_session < valid_from
                ):
                    _add_issue(
                        issues,
                        code="EVENT_ENTITY_BEFORE_VALID_FROM",
                        item_id=event_id,
                        message=f"entity {entity_id!r} is not valid until session {valid_from}",
                    )

    self_name_tokens: set[str] = set()
    for profile in profiles:
        if profile.get("field_type") == "name" and profile.get("item_role", "base") == "base":
            self_name_tokens |= _grounding_tokens(str(profile.get("field_value") or ""))

    for relationship in relationships:
        entity_id = str(relationship.get("entity_id") or "")
        valid_from = relationship.get("valid_from_session")
        valid_until = relationship.get("valid_until_session")
        if not isinstance(valid_from, int) or not 1 <= valid_from <= len(sessions):
            _add_issue(issues, code="RELATIONSHIP_VALID_FROM", item_id=entity_id, message="invalid valid_from_session")
        if valid_until is not None and (
            not isinstance(valid_until, int)
            or not isinstance(valid_from, int)
            or not valid_from <= valid_until <= len(sessions)
        ):
            _add_issue(issues, code="RELATIONSHIP_VALID_UNTIL", item_id=entity_id, message="invalid valid_until_session")
        observed_sessions = sorted(entity_session_indices.get(entity_id, set()))
        if not observed_sessions:
            _add_issue(
                issues,
                code="RELATIONSHIP_NEVER_REFERENCED",
                item_id=entity_id,
                message="relationship entity never appears in any session entity_ids",
            )
        elif isinstance(valid_from, int) and observed_sessions[0] != valid_from:
            _add_issue(
                issues,
                code="RELATIONSHIP_FIRST_SESSION_DRIFT",
                item_id=entity_id,
                message=(
                    f"valid_from_session={valid_from}, but the first structured session reference "
                    f"is session {observed_sessions[0]}"
                ),
            )
        if isinstance(valid_from, int) and valid_from in session_user_text:
            declared_name_tokens = (
                _grounding_tokens(str(relationship.get("name") or "")) - self_name_tokens
            )
            visible_tokens = _grounding_tokens(session_user_text[valid_from])
            if not declared_name_tokens or declared_name_tokens.isdisjoint(visible_tokens):
                _add_issue(
                    issues,
                    code="RELATIONSHIP_NAME_NOT_GROUNDED",
                    item_id=entity_id,
                    message=(
                        "relationship name (or stable role label used as name) is not present in "
                        "the user dialogue at valid_from_session"
                    ),
                )
            visible_name_sessions = sorted(
                index
                for index, text in session_user_text.items()
                if declared_name_tokens & _grounding_tokens(text)
            )
            if visible_name_sessions and visible_name_sessions[0] != valid_from:
                _add_issue(
                    issues,
                    code="RELATIONSHIP_FIRST_VISIBLE_MENTION_DRIFT",
                    item_id=entity_id,
                    message=(
                        f"valid_from_session={valid_from}, but the first visible use of the declared "
                        f"name/role is session {visible_name_sessions[0]}"
                    ),
                )

    all_items: list[dict[str, Any]] = [*profiles, *preferences]
    candidate_pairs = list(_candidate_rows(user))
    all_items.extend(candidate for _, candidate in candidate_pairs)
    item_ids = [str(row.get("item_id") or row.get("candidate_id") or "") for row in all_items]
    if "" in item_ids or len(item_ids) != len(set(item_ids)):
        _add_issue(issues, code="ITEM_IDS", message="catalog item/candidate IDs must be non-empty and unique")

    for item in all_items:
        item_id = str(item.get("item_id") or item.get("candidate_id") or "")
        if item.get("owner_id") != user_id:
            _add_issue(issues, code="OWNER", item_id=item_id, message="owner_id does not match user_id")
        source_ids = list(item.get("source_turn_ids") or [])
        literal = str(item.get("literal_source_span") or "")
        if not source_ids or not literal:
            _add_issue(issues, code="SOURCE_SPAN_REQUIRED", item_id=item_id, message="source turn and literal span are required")
            continue
        source_text = "\n".join(str(turn_lookup.get(source_id, {}).get("text") or "") for source_id in source_ids)
        if any(source_id not in turn_lookup for source_id in source_ids):
            _add_issue(issues, code="SOURCE_TURN_MISSING", item_id=item_id, message="source turn ID is missing")
        if any(turn_lookup.get(source_id, {}).get("role") != "user" for source_id in source_ids):
            _add_issue(issues, code="SOURCE_NOT_USER", item_id=item_id, message="memory source must be a user turn")
        if literal not in source_text:
            _add_issue(issues, code="LITERAL_NOT_IN_SOURCE", item_id=item_id, message="literal_source_span is not verbatim in source user turn(s)")
        source_sessions = {turn_session[source_id] for source_id in source_ids if source_id in turn_session}
        valid_from = item.get("valid_from_session")
        if valid_from is not None and source_sessions and int(valid_from) != min(source_sessions):
            _add_issue(issues, code="VALID_FROM_SOURCE_DRIFT", item_id=item_id, message="valid_from_session must equal the source session")
        for entity_id in item.get("entity_ids", []):
            if entity_id not in entity_ids:
                _add_issue(issues, code="ITEM_ENTITY_REF", item_id=item_id, message=f"unknown entity {entity_id!r}")
            else:
                valid_from = relationship_valid_from.get(entity_id)
                if (
                    entity_id != "self"
                    and source_sessions
                    and isinstance(valid_from, int)
                    and min(source_sessions) < valid_from
                ):
                    _add_issue(
                        issues,
                        code="ITEM_ENTITY_BEFORE_VALID_FROM",
                        item_id=item_id,
                        message=f"entity {entity_id!r} is not valid until session {valid_from}",
                    )
        topic = item.get("topic_thread")
        if topic is not None and topic not in topic_set:
            _add_issue(issues, code="ITEM_TOPIC_REF", item_id=item_id, message=f"unknown topic {topic!r}")

    for session, candidate in candidate_pairs:
        candidate_id = str(candidate.get("candidate_id") or "")
        source_sessions = {
            turn_session[source_id]
            for source_id in candidate.get("source_turn_ids", [])
            if source_id in turn_session
        }
        if source_sessions != {int(session.get("session_index"))}:
            _add_issue(
                issues,
                code="CANDIDATE_SESSION_SOURCE_DRIFT",
                item_id=candidate_id,
                message="candidate must be stored under the same session as its source user turn",
            )

    base_profiles = [row for row in profiles if row.get("item_role") == "base"]
    profile_updates = [row for row in profiles if row.get("item_role") != "base"]
    if len(base_profiles) != 7 or {row.get("field_type") for row in base_profiles} != PROFILE_FIELDS:
        _add_issue(issues, code="PROFILE_BASE", message="profile must contain exactly the seven required base fields")
    if assignment is not None and len(profile_updates) != assignment["profile_updates"]:
        _add_issue(
            issues,
            code="PROFILE_UPDATE_COUNT",
            message=f"expected {assignment['profile_updates']} profile updates, got {len(profile_updates)}",
        )
    by_profile_id = {row.get("item_id"): row for row in profiles}
    for row in profile_updates:
        previous = by_profile_id.get(row.get("supersedes_item_id"))
        if previous is None or previous.get("field_type") != row.get("field_type"):
            _add_issue(issues, code="PROFILE_SUPERSESSION", item_id=row.get("item_id"), message="invalid profile supersedes link")
        elif previous.get("active") is not False or previous.get("valid_until_session") is None:
            _add_issue(issues, code="PROFILE_OLD_VERSION_ACTIVE", item_id=row.get("item_id"), message="superseded profile version must be inactive and bounded")
    for field_type in PROFILE_FIELDS:
        active = [row for row in profiles if row.get("field_type") == field_type and row.get("active") is True]
        if len(active) != 1:
            _add_issue(issues, code="PROFILE_ACTIVE_VERSION", message=f"field {field_type!r} must have exactly one active version")

    if len(preferences) != 3:
        _add_issue(issues, code="PREFERENCE_COUNT", message=f"expected 3 preference-history items, got {len(preferences)}")
    for row in preferences:
        if row.get("preference_type") not in PREFERENCE_TYPES:
            _add_issue(issues, code="PREFERENCE_TYPE", item_id=row.get("item_id"), message="unsupported preference type")
    by_pref_id = {row.get("item_id"): row for row in preferences}
    for row in preferences:
        supersedes = row.get("supersedes_item_id")
        if supersedes:
            previous = by_pref_id.get(supersedes)
            if previous is None or previous.get("active") is not False:
                _add_issue(issues, code="PREFERENCE_SUPERSESSION", item_id=row.get("item_id"), message="invalid preference supersession")
    actual_preference_types = {str(row.get("preference_type") or "") for row in preferences}
    if len(actual_preference_types) != len(preferences):
        _add_issue(
            issues,
            code="PREFERENCE_TYPES_NOT_DISTINCT",
            message="the three preference-history items must use three distinct preference types",
        )

    subtype_counts = Counter(candidate.get("subtype") for _, candidate in candidate_pairs)
    tier_counts = Counter(
        candidate.get("me_tier")
        for _, candidate in candidate_pairs
        if candidate.get("subtype") == "ME_REUSABLE_OUTCOME"
    )
    expected_items = contract["catalog_targets"]["per_user_typed_item_targets"]
    expected_subtypes = {
        "MS_SESSION": expected_items["ms_session"],
        "ME_REUSABLE_OUTCOME": (
            expected_items["me_reusable_executable_core"]
            + expected_items["me_reusable_natural_coverage_challenge"]
        ),
        "ME_UNRESOLVED_EVENT": expected_items["me_unresolved_event"],
        "ME_CONTEXT_EVENT": expected_items["me_context_event"],
    }
    for subtype, expected in expected_subtypes.items():
        if subtype_counts[subtype] != expected:
            _add_issue(issues, code="CANDIDATE_SUBTYPE_COUNT", message=f"expected {expected} {subtype}, got {subtype_counts[subtype]}")
    if tier_counts["executable_core"] != expected_items["me_reusable_executable_core"]:
        _add_issue(issues, code="ME_CORE_COUNT", message="wrong executable-core ME count")
    if tier_counts["natural_coverage_challenge"] != expected_items["me_reusable_natural_coverage_challenge"]:
        _add_issue(issues, code="ME_CHALLENGE_COUNT", message="wrong natural-coverage ME count")

    me_roles_by_thread: dict[str, set[str]] = defaultdict(set)
    for _, candidate in candidate_pairs:
        subtype = candidate.get("subtype")
        if subtype in ME_SUBTYPES:
            me_roles_by_thread[str(candidate.get("topic_thread") or "")].add(str(subtype))
    reusable_and_nonreusable = sum(
        "ME_REUSABLE_OUTCOME" in roles
        and bool(roles & {"ME_UNRESOLVED_EVENT", "ME_CONTEXT_EVENT"})
        for roles in me_roles_by_thread.values()
    )
    all_three_roles = sum(ME_SUBTYPES <= roles for roles in me_roles_by_thread.values())
    minimum_mixed = int(contract["catalog_targets"]["minimum_threads_with_reusable_and_nonreusable_me"])
    minimum_all_three = int(contract["catalog_targets"]["minimum_threads_with_all_three_me_roles"])
    if reusable_and_nonreusable < minimum_mixed:
        _add_issue(
            issues,
            code="ME_THREAD_HARD_NEGATIVE_COVERAGE",
            message=f"expected at least {minimum_mixed} mixed-role ME threads, got {reusable_and_nonreusable}",
        )
    if all_three_roles < minimum_all_three:
        _add_issue(
            issues,
            code="ME_THREAD_ALL_ROLES_COVERAGE",
            message=f"expected at least {minimum_all_three} thread with all three ME roles, got {all_three_roles}",
        )

    typed_span_core_valid = 0
    legacy_core_valid = 0
    legacy_invalid_pass = 0
    for _, candidate in candidate_pairs:
        if candidate.get("subtype") not in ME_SUBTYPES:
            continue
        item_id = str(candidate.get("candidate_id") or "")
        literal = str(candidate.get("literal_source_span") or "")
        candidate_text = str(candidate.get("candidate_text") or "")
        if candidate_text != literal:
            _add_issue(issues, code="ME_CANDIDATE_LITERAL_DRIFT", item_id=item_id, message="ME candidate_text must equal literal_source_span")
        action = candidate.get("action_span")
        result = candidate.get("result_span")
        if action is not None and str(action) not in literal:
            _add_issue(issues, code="ME_ACTION_NOT_LITERAL", item_id=item_id, message="action_span is not verbatim inside literal_source_span")
        if result is not None and str(result) not in literal:
            _add_issue(issues, code="ME_RESULT_NOT_LITERAL", item_id=item_id, message="result_span is not verbatim inside literal_source_span")
        legacy_valid = compile_atomic_reusable_outcome(candidate_text) is not None
        if candidate.get("subtype") == "ME_REUSABLE_OUTCOME" and candidate.get("me_tier") == "executable_core":
            span_valid = bool(action) and bool(result) and str(action) in literal and str(result) in literal
            typed_span_core_valid += int(span_valid)
            legacy_core_valid += int(legacy_valid)
        elif candidate.get("subtype") in {"ME_UNRESOLVED_EVENT", "ME_CONTEXT_EVENT"}:
            legacy_invalid_pass += int(legacy_valid)

    hard_issues = [row for row in issues if row["severity"] == "hard"]
    expected_core = expected_items["me_reusable_executable_core"]
    if hard_issues:
        status = "HARD_CONTENT_BLOCKED"
    elif typed_span_core_valid != expected_core:
        status = "TYPED_CORE_COMPILER_BLOCKED"
    elif legacy_core_valid != expected_core:
        status = "SINGLE_USER_CONTENT_MACHINE_PASS_LEGACY_COMPILER_DIAGNOSTIC_AND_BATCH_REVIEW_PENDING"
    else:
        status = "SINGLE_USER_MACHINE_PASS_BATCH_AND_SEMANTIC_REVIEW_PENDING"

    return {
        "user_id": user_id,
        "status": status,
        "assignment": assignment,
        "counts": {
            "sessions": len(sessions),
            "topic_threads": len(topics),
            "relationships": len(relationships),
            "events": len(events),
            "profile_items": len(profiles),
            "preference_items": len(preferences),
            "candidate_subtypes": dict(sorted(subtype_counts.items())),
            "me_tiers": {str(key): value for key, value in sorted(tier_counts.items(), key=lambda item: str(item[0]))},
        },
        "me_compiler": {
            "typed_exact_span_core_valid": typed_span_core_valid,
            "typed_exact_span_core_expected": expected_core,
            "legacy_regex_core_valid": legacy_core_valid,
            "legacy_regex_core_expected_for_coverage_report_only": expected_core,
            "legacy_regex_intended_invalid_pass": legacy_invalid_pass,
            "formal_core_compiler": "TYPED_EXACT_ACTION_RESULT_SPAN_V1",
            "interpretation": (
                "Exact typed action/result spans are the formal content compiler. Legacy regex "
                "coverage is diagnostic and must not be repaired by silently rewriting natural "
                "source text to a verb whitelist."
            ),
        },
        "hard_issue_count": len(hard_issues),
        "issues": issues,
        "pending_checks": [
            "cross_user_exact_and_near_duplicate_audit",
            "global_subtype_and_preference_distribution",
            "external_exact_and_source_significant_8gram_overlap",
            "one_concentrated_cross_model_semantic_review",
            "actual_rank1_retrieval_and_current_state_checks_after_state_generation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="one file or directory; repeat --input to validate several uploads as one batch",
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    users: list[dict[str, Any]] = []
    source_files: list[str] = []
    for input_path in args.input:
        for path in _input_files(input_path):
            rows = _read_json_values(path)
            users.extend(rows)
            source_files.extend([str(path)] * len(rows))
    normalized_rows = [_normalize_user_schema(user) for user in users]
    results = []
    for (normalized, actions), source_file in zip(normalized_rows, source_files, strict=True):
        result = _validate_user(normalized, contract)
        result["normalization_actions"] = actions
        result["source_file"] = source_file
        results.append(result)
    duplicate_user_ids = [
        user_id for user_id, count in Counter(row["user_id"] for row in results).items() if count > 1
    ]
    preference_contract = contract["catalog_targets"]["preference_acceptance"]
    users_per_author = int(preference_contract["users_per_author"])
    quota_per_type = int(preference_contract["items_per_type_per_author"])
    preference_quota_by_author: dict[str, Any] = {}
    batch_issues: list[dict[str, Any]] = []
    for author in ("chatgpt_pro", "claude"):
        author_users = [user for user in (row[0] for row in normalized_rows) if user.get("content_author") == author]
        counts = Counter(
            preference.get("preference_type")
            for user in author_users
            for preference in user.get("response_preference_history", [])
        )
        remaining_users = users_per_author - len(author_users)
        impossible_types = sorted(
            preference_type
            for preference_type in PREFERENCE_TYPES
            if counts[preference_type] > quota_per_type
            or counts[preference_type] + max(0, remaining_users) < quota_per_type
        )
        if len(author_users) > users_per_author:
            impossible_types = sorted(PREFERENCE_TYPES)
        preference_quota_by_author[author] = {
            "users_in_batch": len(author_users),
            "remaining_user_capacity": remaining_users,
            "counts": {key: counts[key] for key in sorted(PREFERENCE_TYPES)},
            "final_target_per_type": quota_per_type,
            "partial_batch_feasible": not impossible_types,
            "impossible_types": impossible_types,
        }
        if impossible_types:
            batch_issues.append(
                {
                    "severity": "hard",
                    "code": "PREFERENCE_GLOBAL_QUOTA_UNREACHABLE",
                    "author": author,
                    "message": f"partial batch cannot reach final exact quota for {impossible_types}",
                }
            )
    status_counts = Counter(row["status"] for row in results)
    report = {
        "protocol": "pm-v1.5-v5.3-formal-longitudinal-user-validation-v1",
        "status": (
            "HARD_BLOCKED"
            if status_counts["HARD_CONTENT_BLOCKED"] or duplicate_user_ids or batch_issues
            else "TYPED_CORE_COMPILER_BLOCKED"
            if status_counts["TYPED_CORE_COMPILER_BLOCKED"]
            else "MACHINE_PASS_BATCH_AND_SEMANTIC_REVIEW_PENDING"
        ),
        "contract_status": contract.get("status"),
        "input_files": sorted(set(source_files)),
        "user_count": len(results),
        "duplicate_user_ids": duplicate_user_ids,
        "batch_issues": batch_issues,
        "preference_quota_by_author": preference_quota_by_author,
        "status_counts": dict(sorted(status_counts.items())),
        "users": results,
        "batch_level_note": (
            "Per-user machine pass is necessary but not sufficient. Duplicate, diversity, "
            "global quota, external-overlap and semantic-review checks require a batch."
        ),
        "api_calls": 0,
        "training_label_or_outcome_read": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = args.out_dir / "canonical_users"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_output_paths: list[str] = []
    for (normalized, _), result in zip(normalized_rows, results, strict=True):
        output_path = canonical_dir / f"{result['user_id']}.json"
        write_json(output_path, normalized)
        canonical_output_paths.append(str(output_path))
    report["canonical_output_paths"] = canonical_output_paths
    write_json(args.out_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
