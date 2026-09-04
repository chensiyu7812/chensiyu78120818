"""Outcome-blind leakage checks for V5.3 external evidence surfaces.

The helpers in this module deliberately stop before model inference.  They
make the two external input projections explicit:

* response generation may serialize only ``basic_info`` and a prefix of the
  same user's timestamp-normalized ``dialog_history``;
* ES-MemEval-style QA generation may serialize the public question and
  same-user session documents, never evaluator-only answers or evidence.

The module also provides a content-overlap interface for a future formal
V5.3 internal superdomain.  No formal superdomain is assumed to exist.  A
caller that has no frozen internal text surface must report the overlap stage
as pending rather than converting the absence of input into a PASS.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from .io import canonical_json, stable_hex
from .v1_5_external_memory_adapter import (
    qa_messages,
    render_evoemo_session_document,
)


AUDIT_PROTOCOL = "pm-v1.5-v5.3-external-source-leakage-audit-v1"
RESPONSE_PROJECTION_KEYS = ("basic_info", "dialog_history")
QA_GOLD_ONLY_KEYS = frozenset(
    {
        "answer",
        "answers",
        "evidence",
        "capability",
        "theme",
        "group",
        "question_group",
    }
)
DEFAULT_INTERNAL_TEXT_FIELDS = frozenset(
    {
        "current_user_text",
        "current_context",
        "candidate_text",
        "literal_evidence",
        "memory_text",
        "prior_observation",
        "past_action",
        "observed_outcome",
        "support_move",
        "when_to_use",
        "when_not_to_use",
        "text",
        "question",
    }
)

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _session_id(session: Mapping[str, Any]) -> str:
    value = str(session.get("id") or "")
    if not value:
        raise ValueError("dialogue session is missing id")
    return value


def project_evoemo_response_source(
    user: Mapping[str, Any], *, strictly_prior_session_count: int
) -> dict[str, Any]:
    """Return the only user fields eligible for response-path serialization.

    ``strictly_prior_session_count`` is a causal cut in an already normalized
    chronological history.  The current session is the item at that index;
    therefore it and every later session are excluded.  A count equal to the
    history length is valid for a response state that occurs after the final
    released session.
    """

    history = list(user.get("dialog_history") or [])
    if isinstance(strictly_prior_session_count, bool) or not isinstance(
        strictly_prior_session_count, int
    ):
        raise TypeError("strictly_prior_session_count must be an integer")
    if not 0 <= strictly_prior_session_count <= len(history):
        raise ValueError("strictly_prior_session_count is outside dialog_history")
    return {
        "basic_info": deepcopy(dict(user.get("basic_info") or {})),
        "dialog_history": deepcopy(history[:strictly_prior_session_count]),
    }


def serialize_response_projection(projection: Mapping[str, Any]) -> str:
    if tuple(projection) != RESPONSE_PROJECTION_KEYS:
        raise ValueError(
            "response projection must contain exactly basic_info then dialog_history"
        )
    if not isinstance(projection["basic_info"], Mapping):
        raise TypeError("response projection basic_info must be a mapping")
    if not isinstance(projection["dialog_history"], list):
        raise TypeError("response projection dialog_history must be a list")
    return canonical_json(dict(projection))


def mutate_gold_only_fields(value: Any, *, canary: str) -> Any:
    """Deep-copy ``value`` while changing every evaluator-only gold field."""

    if isinstance(value, Mapping):
        return {
            key: (
                f"{canary}:{key}"
                if str(key) in QA_GOLD_ONLY_KEYS
                else mutate_gold_only_fields(item, canary=canary)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [mutate_gold_only_fields(item, canary=canary) for item in value]
    return deepcopy(value)


def response_gold_canary_is_invariant(
    user: Mapping[str, Any], *, strictly_prior_session_count: int
) -> bool:
    before = serialize_response_projection(
        project_evoemo_response_source(
            user, strictly_prior_session_count=strictly_prior_session_count
        )
    )
    mutated = mutate_gold_only_fields(user, canary="forbidden-response-gold")
    after = serialize_response_projection(
        project_evoemo_response_source(
            mutated, strictly_prior_session_count=strictly_prior_session_count
        )
    )
    return before == after


def qa_generation_serialization(
    *, question: str, session_documents: Sequence[Mapping[str, Any]]
) -> str:
    fragments = [
        f"{index}. [{str(document.get('date') or '')}]\n{str(document.get('text') or '')}"
        for index, document in enumerate(session_documents, start=1)
    ]
    return canonical_json(qa_messages(question=question, memory_fragments=fragments))


def qa_query_serialization(*, question: str) -> str:
    """Frozen retrieval-query surface; evaluator gold is not an argument."""

    return canonical_json({"question": " ".join(str(question).split())})


def qa_gold_canary_is_invariant(
    *, qa_row: Mapping[str, Any], session_documents: Sequence[Mapping[str, Any]]
) -> bool:
    question = str(qa_row.get("question") or "")
    if not question.strip():
        raise ValueError("QA row has no public question")
    before = (
        qa_query_serialization(question=question),
        qa_generation_serialization(
            question=question, session_documents=session_documents
        ),
    )
    mutated = mutate_gold_only_fields(qa_row, canary="forbidden-qa-gold")
    mutated_question = str(mutated.get("question") or "")
    after = (
        qa_query_serialization(question=mutated_question),
        qa_generation_serialization(
            question=mutated_question, session_documents=session_documents
        ),
    )
    return before == after


def iter_released_qa_rows(user: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield evaluator rows from both released QA containers.

    ``questions`` is grouped; ``summaries`` stores rows directly.  The latter
    is included in leakage scanning even if a particular experiment elects
    not to score it, because its answers/evidence are still protected external
    evaluator surfaces.
    """

    for group in user.get("questions") or []:
        group_id = str(group.get("id") or "")
        for row in group.get("questions") or []:
            yield {"container": "questions", "group_id": group_id, **dict(row)}
    for row in user.get("summaries") or []:
        yield {"container": "summaries", **dict(row)}


def audit_evoemo_projection_and_qa(users: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit all causal response cuts and all released QA rows, zero outcome."""

    owner_by_composite_session: dict[tuple[str, str], str] = {}
    duplicate_raw_session_ids: dict[str, set[str]] = {}
    raw_owners: dict[str, set[str]] = {}
    for user in users:
        user_id = str(user.get("id") or "")
        if not user_id:
            raise ValueError("EvoEmo user is missing id")
        for session in user.get("dialog_history") or []:
            session_id = _session_id(session)
            composite = (user_id, session_id)
            if composite in owner_by_composite_session:
                raise ValueError(f"duplicate same-user session identity: {composite}")
            owner_by_composite_session[composite] = user_id
            raw_owners.setdefault(session_id, set()).add(user_id)
    duplicate_raw_session_ids = {
        session_id: owners for session_id, owners in raw_owners.items() if len(owners) > 1
    }

    projection_count = 0
    current_or_future_exposures = 0
    cross_user_exposures = 0
    projection_key_violations = 0
    response_canary_failures = 0
    qa_rows = 0
    qa_canary_failures = 0
    qa_cross_user_exposures = 0

    for user in users:
        user_id = str(user["id"])
        history = list(user.get("dialog_history") or [])
        own_composites = {(user_id, _session_id(session)) for session in history}
        other_composites = set(owner_by_composite_session) - own_composites
        for cutoff in range(len(history) + 1):
            projection_count += 1
            projection = project_evoemo_response_source(
                user, strictly_prior_session_count=cutoff
            )
            if tuple(projection) != RESPONSE_PROJECTION_KEYS:
                projection_key_violations += 1
            observed = {
                (user_id, _session_id(session))
                for session in projection["dialog_history"]
            }
            future = {
                (user_id, _session_id(session)) for session in history[cutoff:]
            }
            current_or_future_exposures += len(observed & future)
            cross_user_exposures += len(observed & other_composites)
            if not response_gold_canary_is_invariant(
                user, strictly_prior_session_count=cutoff
            ):
                response_canary_failures += 1

        human_name = str((user.get("basic_info") or {}).get("name") or "User")
        documents = [
            render_evoemo_session_document(session, human_name=human_name)
            for session in history
        ]
        document_composites = {
            (user_id, str(document["session_id"])) for document in documents
        }
        qa_cross_user_exposures += len(document_composites & other_composites)
        for qa_row in iter_released_qa_rows(user):
            qa_rows += 1
            if not qa_gold_canary_is_invariant(
                qa_row=qa_row, session_documents=documents
            ):
                qa_canary_failures += 1

    return {
        "users": len(users),
        "causal_response_projections_checked": projection_count,
        "response_projection_allowed_top_level_fields": list(RESPONSE_PROJECTION_KEYS),
        "response_projection_key_violations": projection_key_violations,
        "response_disallowed_top_level_fields_serialized": projection_key_violations,
        "response_current_or_future_session_exposures": current_or_future_exposures,
        "response_cross_user_session_exposures": cross_user_exposures,
        "response_forbidden_gold_canary_failures": response_canary_failures,
        "qa_evaluator_rows_checked": qa_rows,
        "qa_answer_evidence_canary_failures": qa_canary_failures,
        "qa_gold_fields_visible_to_generation_or_query": qa_canary_failures,
        "qa_cross_user_session_exposures": qa_cross_user_exposures,
        "raw_session_ids_shared_across_users": len(duplicate_raw_session_ids),
        "session_identity_rule": "(user_id, session_id), never bare session_id",
        "generation_or_retrieval_outcomes_read": False,
    }


def normalized_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def normalized_ngrams(text: str, *, n: int) -> set[tuple[str, ...]]:
    if n < 2:
        raise ValueError("n must be at least 2")
    tokens = normalized_tokens(text)
    if len(tokens) < n:
        return set()
    return {tokens[index : index + n] for index in range(len(tokens) - n + 1)}


def _surface_digest(text: str) -> str:
    return sha256(str(text).encode("utf-8")).hexdigest()


def compare_text_surface_overlap(
    *,
    internal_surfaces: Sequence[Mapping[str, str]],
    external_surfaces: Sequence[Mapping[str, str]],
    ngram_size: int = 8,
) -> dict[str, Any]:
    """Compare exact and normalized n-gram overlap without returning raw text."""

    external_exact: dict[str, list[tuple[str, str]]] = {}
    external_ngrams: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for surface in external_surfaces:
        surface_id = str(surface["surface_id"])
        category = str(surface["category"])
        text = str(surface["text"])
        external_exact.setdefault(" ".join(text.split()), []).append((surface_id, category))
        for ngram in normalized_ngrams(text, n=ngram_size):
            external_ngrams.setdefault(ngram, []).append((surface_id, category))

    exact_collisions: list[dict[str, Any]] = []
    ngram_collisions: list[dict[str, Any]] = []
    internal_with_exact: set[str] = set()
    internal_with_ngram: set[str] = set()
    category_counts: dict[str, dict[str, int]] = {}
    for surface in internal_surfaces:
        internal_id = str(surface["surface_id"])
        text = str(surface["text"])
        for external_id, category in external_exact.get(" ".join(text.split()), []):
            internal_with_exact.add(internal_id)
            category_counts.setdefault(category, {"exact": 0, "ngram": 0})["exact"] += 1
            exact_collisions.append(
                {
                    "internal_surface_id": internal_id,
                    "external_surface_id": external_id,
                    "external_category": category,
                    "internal_text_sha256": _surface_digest(text),
                }
            )
        shared = normalized_ngrams(text, n=ngram_size) & set(external_ngrams)
        for ngram in sorted(shared):
            internal_with_ngram.add(internal_id)
            ngram_hash = _surface_digest(" ".join(ngram))
            for external_id, category in external_ngrams[ngram]:
                category_counts.setdefault(category, {"exact": 0, "ngram": 0})["ngram"] += 1
                ngram_collisions.append(
                    {
                        "internal_surface_id": internal_id,
                        "external_surface_id": external_id,
                        "external_category": category,
                        "ngram_sha256": ngram_hash,
                    }
                )
    return {
        "ngram_size": ngram_size,
        "internal_surface_count": len(internal_surfaces),
        "external_surface_count": len(external_surfaces),
        "internal_surfaces_with_exact_overlap": len(internal_with_exact),
        "internal_surfaces_with_normalized_ngram_overlap": len(internal_with_ngram),
        "exact_collision_count": len(exact_collisions),
        "normalized_ngram_collision_count": len(ngram_collisions),
        "collisions_by_external_category": category_counts,
        "exact_collisions": exact_collisions,
        "normalized_ngram_collisions": ngram_collisions,
        "raw_external_text_in_report": False,
    }


def external_overlap_surfaces(users: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    surfaces: list[dict[str, str]] = []
    for user in users:
        user_id = str(user["id"])
        human_name = str((user.get("basic_info") or {}).get("name") or "User")
        for session in user.get("dialog_history") or []:
            document = render_evoemo_session_document(session, human_name=human_name)
            surfaces.append(
                {
                    "surface_id": f"session:{user_id}:{document['session_id']}",
                    "category": "external_session",
                    "text": str(document["text"]),
                }
            )
        for ordinal, row in enumerate(iter_released_qa_rows(user)):
            base = f"qa:{user_id}:{row.get('container')}:{row.get('group_id', '')}:{row.get('idx', ordinal)}"
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if question:
                surfaces.append(
                    {"surface_id": base + ":question", "category": "external_question", "text": question}
                )
            if answer:
                surfaces.append(
                    {"surface_id": base + ":answer", "category": "external_answer", "text": answer}
                )
    return surfaces


def extract_internal_text_surfaces(
    value: Any,
    *,
    text_fields: frozenset[str] = DEFAULT_INTERNAL_TEXT_FIELDS,
    path: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Extract explicitly named model-visible fields from JSON/JSONL data."""

    surfaces: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        identity = next(
            (
                str(value[key])
                for key in ("state_id", "case_id", "group_id", "id")
                if value.get(key) not in (None, "")
            ),
            "",
        )
        for key, item in value.items():
            item_path = (*path, str(key))
            if str(key) in text_fields and isinstance(item, str) and item.strip():
                logical = identity or stable_hex(AUDIT_PROTOCOL, *item_path, n=16)
                surfaces.append(
                    {
                        "surface_id": f"internal:{logical}:{'.'.join(item_path)}",
                        "category": "internal_superdomain",
                        "text": item,
                    }
                )
            else:
                surfaces.extend(
                    extract_internal_text_surfaces(
                        item, text_fields=text_fields, path=item_path
                    )
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            surfaces.extend(
                extract_internal_text_surfaces(
                    item, text_fields=text_fields, path=(*path, str(index))
                )
            )
    return surfaces


def load_internal_text_surfaces(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
        return extract_internal_text_surfaces(rows)
    return extract_internal_text_surfaces(json.loads(source.read_text(encoding="utf-8")))
