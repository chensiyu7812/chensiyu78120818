"""Content-disjoint memory transport helpers for PM V1.5.

The internal synthetic corpus and EvoEmo must not share user content.  They do
share this construction boundary: a user profile and strictly prior sessions
are compiled by :func:`metacom_pm.evoemo.build_evo_memory`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from .contracts import MemoryItem, MemorySource
from .evoemo import build_evo_memory
from .io import canonical_json, sha256_text
from .pm_v2_data import GeneratedStateCase, GeneratedUserBundle
from .text import normalize_space


TRANSPORT_ADAPTER_PROTOCOL = (
    "pm-v1.5-content-disjoint-evo-style-synthetic-memory-adapter-v1"
)
BOUNDED_MEMORY_COMPILER_PROTOCOL = (
    "pm-v1.5-shared-all-causal-prior-history-memory-compiler-v2"
)
MAX_SUMMARY_TOKENS = 64


# These are compiler hints, not PM gold.  They are extracted before retrieval,
# generation, or human labels and exist only to expose auditable ME structure.
_ME_ACTION_RE = re.compile(
    r"\b(?:tried|used|chose|decided|asked|called|wrote|walked|went|"
    r"practiced|paused|breathed|talked|scheduled|limited|stopped|started)\b",
    flags=re.IGNORECASE,
)
_ME_RESULT_RE = re.compile(
    r"\b(?:helped|worked|eased|reduced|improved|made .{0,60} easier|"
    r"felt (?:better|calmer|safer)|did not help|didn't help|made .{0,60} worse)\b",
    flags=re.IGNORECASE,
)
_ME_MECHANISM_RE = re.compile(
    r"\b(?:because|by |so that|which (?:helped|made|allowed)|"
    r"the part that|what helped was|worked because)\b",
    flags=re.IGNORECASE,
)
_ME_UNRESOLVED_RE = re.compile(
    r"\b(?:still unresolved|remained unresolved|still struggling|"
    r"kept happening|did not work|didn't work|made .{0,60} worse)\b",
    flags=re.IGNORECASE,
)


def compile_me_structure_metadata(text: str) -> dict[str, Any]:
    """Compile outcome-blind ME form hints from one historical episode.

    The hints are deliberately weaker than the H1 adjudicated subtype.  They
    may be used as transparent feature inputs, but never as the gold decision.
    """

    normalized = normalize_space(text)
    contains_action = bool(_ME_ACTION_RE.search(normalized))
    contains_result = bool(_ME_RESULT_RE.search(normalized))
    contains_mechanism = bool(_ME_MECHANISM_RE.search(normalized))
    unresolved = bool(_ME_UNRESOLVED_RE.search(normalized))
    if contains_action and (contains_result or contains_mechanism) and not unresolved:
        hint = "ME_REUSABLE_OUTCOME"
    elif unresolved:
        hint = "ME_UNRESOLVED_EVENT"
    else:
        hint = "ME_CONTEXT_EVENT"
    return {
        "me_subtype_hint": hint,
        "me_contains_action": contains_action,
        "me_contains_result": contains_result,
        "me_contains_mechanism": contains_mechanism,
        "me_unresolved_marker": unresolved,
        "subtype_hint_is_gold": False,
    }


def _truncate_estimated_tokens(text: str, maximum: int) -> str:
    """Deterministically bound text using the project's token estimator."""

    normalized = normalize_space(text)
    if not normalized:
        return ""
    # estimate_tokens is ceil(characters / 4), so this is an exact upper
    # bound under the frozen estimator without depending on a tokenizer.
    return normalize_space(normalized[: maximum * 4])


def compile_session_summary(session: dict[str, Any]) -> str:
    """One shared, outcome-blind MS compiler for both domains."""

    supplied = normalize_space(session.get("summary") or "")
    if supplied:
        return _truncate_estimated_tokens(supplied, MAX_SUMMARY_TOKENS)
    seeker_turns = [
        normalize_space(turn.get("content") or "")
        for turn in (session.get("dialogue") or [])
        if turn.get("role") == "seeker"
        and normalize_space(turn.get("content") or "")
    ]
    return _truncate_estimated_tokens(
        seeker_turns[-1] if seeker_turns else "",
        MAX_SUMMARY_TOKENS,
    )


def bounded_prior_history_user(
    user: dict[str, Any],
    *,
    maximum_prior_sessions: int | None = None,
) -> dict[str, Any]:
    """Normalize one user's complete causal prior history.

    Timestamps are removed from the generator-facing memory surface.  Order is
    retained through ``created_session`` and rendered later as relative age.
    ``maximum_prior_sessions`` exists only for explicit diagnostic ablations;
    the formal internal and external runtime both use the default complete
    prior history.
    """

    if maximum_prior_sessions is not None and maximum_prior_sessions < 1:
        raise ValueError("maximum_prior_sessions must be positive")
    history = list(user.get("dialog_history") or [])
    selected = (
        history
        if maximum_prior_sessions is None
        else history[-maximum_prior_sessions:]
    )
    sessions: list[dict[str, Any]] = []
    for index, session in enumerate(selected, 1):
        normalized = dict(session)
        normalized["id"] = str(
            session.get("id") or f"bounded_session_{index}"
        )
        normalized["timestamp"] = ""
        normalized["summary"] = compile_session_summary(normalized)
        normalized["dialogue"] = [
            {
                "role": str(turn.get("role") or ""),
                "content": normalize_space(turn.get("content") or ""),
            }
            for turn in (session.get("dialogue") or [])
            if normalize_space(turn.get("content") or "")
        ]
        sessions.append(normalized)
    return {
        "id": str(user["id"]),
        "basic_info": dict(user.get("basic_info") or {}),
        "dialog_history": sessions,
    }


def compile_bounded_memory(
    user: dict[str, Any],
    *,
    maximum_prior_sessions: int | None = None,
) -> tuple[list[MemoryItem], list[dict[str, Any]]]:
    """Compile all causal prior memory through the shared runtime wrapper."""

    return build_evo_memory(
        bounded_prior_history_user(
            user, maximum_prior_sessions=maximum_prior_sessions
        )
    )


def compile_bounded_memory_with_metadata(
    user: dict[str, Any],
    *,
    maximum_prior_sessions: int | None = None,
) -> tuple[list[MemoryItem], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Compile memory plus source metadata required by the opportunity gate.

    ``MemoryItem`` intentionally contains no evaluator labels.  The routing
    hard gates nevertheless need two provenance facts that are determined by
    the shared compiler rather than by response outcomes: whether MP is a
    profile or an explicit support preference, and whether MS came from a
    supplied prior-session summary or from the bounded fallback compiler.

    This wrapper derives those facts from the same input user before any
    retrieval or generation.  It is valid for both synthetic internal users
    and EvoEmo users and never reads dataset identity, topic annotations, or
    outcomes.
    """

    bounded = bounded_prior_history_user(
        user, maximum_prior_sessions=maximum_prior_sessions
    )
    items, session_docs = build_evo_memory(bounded)
    original_history = list(user.get("dialog_history") or [])
    if maximum_prior_sessions is not None:
        original_history = original_history[-maximum_prior_sessions:]

    metadata: dict[str, dict[str, Any]] = {}
    owner_id = str(bounded.get("id") or "")
    if not owner_id:
        raise RuntimeError("bounded memory owner id is missing")
    profile_status = dict(user.get("basic_info_record_status") or {})

    def _record_status(raw: Any) -> tuple[bool, bool]:
        status = dict(raw or {})
        active = status.get("active", True)
        superseded = status.get("superseded", False)
        if not isinstance(active, bool) or not isinstance(superseded, bool):
            raise TypeError("memory record active/superseded status must be boolean")
        if superseded and active:
            active = False
        return active, superseded

    mp_items = [item for item in items if item.source is MemorySource.MP]
    basic_fields = [
        str(key)
        for key, value in (bounded.get("basic_info") or {}).items()
        if value not in (None, "")
    ]
    if len(mp_items) != len(basic_fields):
        raise RuntimeError("MP compiler field/item count mismatch")
    for item, field in zip(mp_items, basic_fields, strict=True):
        active, superseded = _record_status(profile_status.get(field))
        metadata[item.memory_id] = {
            "mp_subtype": (
                "MP_PREFERENCE"
                if field.startswith("stable_preference_")
                else "MP_PROFILE"
            ),
            "profile_field": field,
            "owner_id": owner_id,
            "record_active": active,
            "record_superseded": superseded,
            "created_session": int(item.created_session),
            "outcome_read": False,
        }

    original_summary_by_session = {
        index: normalize_space(session.get("summary") or "")
        for index, session in enumerate(original_history, 1)
    }
    for item in items:
        session_index = int(item.created_session)
        original_session = (
            original_history[session_index - 1]
            if 0 < session_index <= len(original_history)
            else {}
        )
        active, superseded = _record_status(
            original_session.get("memory_record_status")
        )
        if item.source is MemorySource.MS:
            metadata[item.memory_id] = {
                "summary_origin": (
                    "supplied_strictly_prior_summary"
                    if original_summary_by_session.get(
                        int(item.created_session), ""
                    )
                    else "compiled_fallback_last_seeker_text"
                ),
                "owner_id": owner_id,
                "record_active": active,
                "record_superseded": superseded,
                "created_session": session_index,
                "outcome_read": False,
            }
        elif item.source is MemorySource.ME:
            metadata[item.memory_id] = {
                "event_origin": "strictly_prior_seeker_episode",
                "owner_id": owner_id,
                "record_active": active,
                "record_superseded": superseded,
                "created_session": session_index,
                **compile_me_structure_metadata(item.text),
                "outcome_read": False,
            }
    if set(metadata) != {item.memory_id for item in items}:
        raise RuntimeError("transport metadata does not cover every memory item")
    return items, session_docs, metadata


def bounded_memory_global_catalog_digest(
    users: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Bind the catalogs actually used by the shared bounded runtime."""

    per_user = []
    for user in users:
        items, _ = compile_bounded_memory(user)
        rows = [item.model_dump(mode="json") for item in items]
        per_user.append(
            {
                "user_id": str(user["id"]),
                "item_count": len(rows),
                "catalog_sha256": sha256_text(canonical_json(rows)),
            }
        )
    per_user.sort(key=lambda row: row["user_id"])
    return {
        "protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
        "user_count": len(per_user),
        "per_user": per_user,
        "global_catalog_sha256": sha256_text(canonical_json(per_user)),
    }


def synthetic_basic_info(bundle: GeneratedUserBundle) -> dict[str, str]:
    """Return only user-specific profile values already present in the bundle.

    The legacy compiler emitted two identical caution statements for all 52
    users.  Those are system-level memory-use rules, not personal profile
    facts, and therefore do not belong in MP.
    """

    return {
        f"stable_preference_{index}": normalize_space(value)
        for index, value in enumerate(bundle.stable_preferences, 1)
        if normalize_space(value)
    }


def synthetic_prior_session(case: GeneratedStateCase) -> dict[str, Any]:
    """Represent one completed synthetic case as a prior user session."""

    dialogue = [
        {
            "role": "seeker" if turn.role == "user" else "supporter",
            "content": turn.content,
        }
        for turn in case.recent_dialogue
    ]
    dialogue.append({"role": "seeker", "content": case.current_user_text})
    return {
        "id": case.case_id,
        "timestamp": f"session-{case.session_index}",
        "summary": case.session_summary or case.current_user_text,
        "dialogue": dialogue,
    }


@dataclass(frozen=True)
class SyntheticLongitudinalCatalog:
    user_id: str
    case: GeneratedStateCase
    chronological_session_index: int
    prior_session_count: int
    items: tuple[MemoryItem, ...]
    prior_case_ids: tuple[str, ...]


def compile_synthetic_longitudinal_catalogs(
    bundles: Iterable[GeneratedUserBundle],
) -> list[SyntheticLongitudinalCatalog]:
    """Compile one causal catalog for every noninitial synthetic state.

    No external file, external content, evaluator annotation, outcome, or
    per-case generated-memory label is accepted by this function.
    """

    catalogs: list[SyntheticLongitudinalCatalog] = []
    for bundle in bundles:
        basic_info = synthetic_basic_info(bundle)
        prior_sessions: list[dict[str, Any]] = []
        prior_case_ids: list[str] = []
        for ordinal, case in enumerate(
            sorted(bundle.cases, key=lambda value: value.session_index),
            1,
        ):
            if prior_sessions:
                items, _ = compile_bounded_memory(
                    {
                        "id": bundle.user_id,
                        "basic_info": basic_info,
                        "dialog_history": list(prior_sessions),
                    }
                )
                current_session = len(prior_sessions) + 1
                if any(
                    int(item.created_session) >= current_session
                    for item in items
                ):
                    raise RuntimeError(
                        "transport adapter produced current/future memory"
                    )
                catalogs.append(
                    SyntheticLongitudinalCatalog(
                        user_id=bundle.user_id,
                        case=case,
                        chronological_session_index=current_session,
                        prior_session_count=len(prior_sessions),
                        items=tuple(items),
                        prior_case_ids=tuple(prior_case_ids),
                    )
                )
            prior_sessions.append(synthetic_prior_session(case))
            prior_case_ids.append(case.case_id)
    return catalogs
