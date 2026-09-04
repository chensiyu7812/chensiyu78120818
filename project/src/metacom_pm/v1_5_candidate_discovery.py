"""Shared candidate discovery for the factorized PM V1.5 resource gates.

Candidate discovery is deliberately local and outcome blind.  The retriever
selects concrete items; the PM sees only the bounded descriptor emitted here
and decides whether the selected items may be injected into the generator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import MemoryItem, MemorySource
from .retrieval import DEFAULT_MEMORY_TOP_K, MemoryRetriever
from .v1_5_v5_3_semantic_ms_retrieval import TextEncoder, rank_ms_candidates
from .text import (
    content_words,
    content_word_match_level,
    estimate_tokens,
    lexical_score,
    normalize_for_hash,
)


CANDIDATE_DISCOVERY_PROTOCOL = (
    "pm-v1.5-shared-memory-candidate-discovery-v1"
)
SOURCE_SPECIFIC_QUERY_PROTOCOL = (
    "pm-v1.5-source-specific-memory-query-contract-v1"
)
V1_5B_CANDIDATE_DISCOVERY_PROTOCOL = (
    "pm-v1.5b-source-local-deduplicated-candidate-discovery-v1"
)
V1_5B_RANK1_EXECUTION_PROTOCOL = "pm-v1.5b-rank1-memory-execution-surface-v1"
FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL = (
    "pm-v1.5-final-content-filtered-typed-rank1-discovery-v1"
)

_FINAL_RETRIEVAL_GENERIC_WORDS = frozenset(
    {
        "about",
        "feel",
        "feeling",
        "feelings",
        "felt",
        "talk",
        "talking",
        "discuss",
        "discussed",
        "discussing",
        "help",
        "support",
    }
)
_PREFERENCE_SCOPE_WORDS = frozenset(
    {
        "advice",
        "idea",
        "option",
        "suggestion",
        "listen",
        "question",
        "brief",
        "concise",
        "reflect",
        "reflection",
        "words",
        "factual",
        "reminder",
        "direct",
        "response",
        "time",
    }
)


_MP_PARENTHETICAL_STATUS_RE = re.compile(r"\([^)]*\)")


def _mp_match_document(text: str) -> str:
    """Strip structural, non-content parts of an MP item's rendered text
    before content matching: the "Label: " prefix evoemo.build_evo_memory()
    renders MP items with (e.g. "Job: office worker"), and any parenthetical
    status annotation in the value (e.g. "high school (in progress)").

    The label names a basic_info field, not conversational content: matched
    verbatim it made e.g. "Job: office worker" fire on any query that merely
    mentions the word "job" (including a query about someone else's job),
    regardless of whether "office worker" itself was relevant. Confirmed on
    the real 138-state panel: 3 users' MP selections were 100% driven by this
    (PM_V1_5_V5_3_MP_VERIFICATION_FINDINGS_20260806_ZH.md).

    Parenthetical status annotations are the same class of problem in the
    value rather than the label: "(in progress)" contributed the generic
    word "progress", which matched unrelated therapy-speak sentences
    ("healing is not a linear process") on a completely different topic
    (education status vs. an unrelated encounter/party) for 6/9 of one real
    user's MP selections. Only "high school", the actual content, should
    participate in matching -- the same rationale as stripping the label,
    not a general stopword-list expansion (which would risk suppressing
    real signal in MS/ME, per that same finding doc's discussion).
    """

    _, separator, value = text.partition(": ")
    value = value if separator else text
    return _MP_PARENTHETICAL_STATUS_RE.sub(" ", value)


def final_typed_content_words(text: str) -> set[str]:
    """Content words for final retrieval, excluding support-domain boilerplate."""

    return content_words(text) - _FINAL_RETRIEVAL_GENERIC_WORDS


def final_typed_content_match_level(query: str, document: str) -> float:
    overlap = len(final_typed_content_words(query) & final_typed_content_words(document))
    if overlap == 0:
        return 0.0
    if overlap == 1:
        return 0.5
    return 1.0


def preference_scope_match_level(query: str, document: str) -> float:
    overlap = (
        final_typed_content_words(query)
        & final_typed_content_words(document)
        & _PREFERENCE_SCOPE_WORDS
    )
    return 1.0 if len(overlap) >= 2 else 0.5 if overlap else 0.0


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float(ordered[middle - 1] + ordered[middle]) / 2.0


def _bucket(value: float, boundaries: Sequence[float]) -> int:
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return index
    return len(boundaries)


def candidate_model_features(
    descriptor: Mapping[str, Any],
) -> dict[str, float | int | bool | str]:
    """Return the bounded, cross-domain feature view available to the PM.

    Raw cost, age, and relevance remain in the descriptor for audit and exact
    cost accounting.  The learned V1.5 heads consume stable buckets so a
    harmlessly cheaper candidate or a longer external history does not become
    an accidental dataset-identity feature.
    """

    present = bool(descriptor["candidate_present"])
    if not present:
        return {
            "source": str(descriptor["source"]),
            "candidate_present": False,
            "retrieved_fraction": 0.0,
            "token_cost_bucket": 0,
            "minimum_relative_age_bucket": 0,
            "median_relative_age_bucket": 0,
            "maximum_relative_age_bucket": 0,
            "top1_relevance_bucket": 0,
            "relevance_margin_bucket": 0,
            "catalog_capacity_bucket": 0,
        }
    return {
        "source": str(descriptor["source"]),
        "candidate_present": True,
        "retrieved_fraction": round(
            float(descriptor["retrieved_count"])
            / float(descriptor["top_k"]),
            6,
        ),
        # Bucket 0 is reserved for candidate-absent states.
        "token_cost_bucket": 1
        + _bucket(
            float(descriptor["incremental_injected_tokens"]),
            (64.0, 128.0, 256.0, 512.0),
        ),
        "minimum_relative_age_bucket": 1
        + _bucket(
            float(descriptor["minimum_relative_age"]),
            (0.125, 0.25, 0.50, 0.75),
        ),
        "median_relative_age_bucket": 1
        + _bucket(
            float(descriptor["median_relative_age"]),
            (0.125, 0.25, 0.50, 0.75),
        ),
        "maximum_relative_age_bucket": 1
        + _bucket(
            float(descriptor["maximum_relative_age"]),
            (0.125, 0.25, 0.50, 0.75),
        ),
        "top1_relevance_bucket": 1
        + _bucket(
            float(descriptor["top1_lexical_relevance"]),
            (0.05, 0.15, 0.30, 0.50),
        ),
        "relevance_margin_bucket": 1
        + _bucket(
            float(descriptor["top1_top2_lexical_margin"]),
            (0.01, 0.05, 0.15, 0.30),
        ),
        "catalog_capacity_bucket": 1
        + _bucket(
            float(descriptor["top_k_capacity_fraction"]),
            (0.34, 0.67, 0.99),
        ),
    }


@dataclass(frozen=True)
class MemoryCandidate:
    source: MemorySource
    selected_items: tuple[MemoryItem, ...]
    descriptor: dict[str, Any]


def materialize_rank1_memory_for_execution(
    *,
    source: MemorySource,
    selected_items: Sequence[MemoryItem],
    session_index: int,
) -> tuple[str, MemoryItem]:
    """Materialize exactly the retriever's rank-1 item for Step2.

    Candidate discovery and PM descriptors intentionally retain the complete
    source-local Top-k. The generator gets one bounded item so the execution
    contract is identical in internal qualification and external evaluation.
    This helper never reranks, summarizes, or reads an outcome.
    """

    if not selected_items:
        raise ValueError(f"cannot materialize absent {source.value} candidate")
    item = selected_items[0]
    if item.source is not source:
        raise ValueError(
            f"rank-1 execution item source mismatch: {item.source.value} != {source.value}"
        )
    age = int(session_index) - int(item.created_session)
    if age <= 0:
        raise ValueError("rank-1 execution item must be strictly historical")
    text = " ".join(str(item.text or "").split())
    if not text:
        raise ValueError("rank-1 execution item text is empty")
    prefix = {
        MemorySource.MP: "ONE USER PROFILE OR PREFERENCE CANDIDATE",
        MemorySource.MS: (
            "ONE PRIOR-SESSION SUMMARY (past evidence; verify current relevance)"
        ),
        MemorySource.ME: (
            "ONE PAST SEEKER EPISODE (not a current fact or stable rule)"
        ),
    }[source]
    return f"{prefix} · {age} sessions ago:\n{text}", item


def deduplicate_memory_items_by_text(
    items: Sequence[MemoryItem],
) -> tuple[MemoryItem, ...]:
    """Keep the first ranked item for each normalized source-local surface.

    The retriever has already established rank order when this helper is used.
    Deduplication therefore does not introduce a second relevance scorer and
    cannot mix users or memory sources.  It only prevents identical summaries
    from occupying multiple Top-k slots and being counted as independent
    evidence by the opportunity head or generator.
    """

    seen: set[tuple[MemorySource, str]] = set()
    unique: list[MemoryItem] = []
    for item in items:
        key = (item.source, normalize_for_hash(item.text))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def describe_memory_candidate(
    *,
    source: MemorySource,
    query: str,
    source_items: Sequence[MemoryItem],
    selected_items: Sequence[MemoryItem],
    session_index: int,
) -> dict[str, Any]:
    """Describe one realized candidate without exposing its text or IDs."""

    if session_index < 1:
        raise ValueError("session_index must be positive")
    top_k = int(DEFAULT_MEMORY_TOP_K[source])
    scores = sorted(
        (lexical_score(query, item.text) for item in source_items),
        reverse=True,
    )
    ages = sorted(
        int(session_index) - int(item.created_session)
        for item in selected_items
    )
    if any(age <= 0 for age in ages):
        raise ValueError("candidate contains current or future memory")
    descriptor: dict[str, Any] = {
        "protocol": CANDIDATE_DISCOVERY_PROTOCOL,
        "source": source.value,
        "candidate_present": bool(selected_items),
        "retrieved_count": len(selected_items),
        "top_k": top_k,
        "incremental_injected_tokens": sum(
            estimate_tokens(item.text) for item in selected_items
        ),
        "minimum_age_sessions": min(ages) if ages else None,
        "median_age_sessions": _median(ages),
        "maximum_age_sessions": max(ages) if ages else None,
        # Session-relative age is the formal PM feature.  Absolute ages remain
        # for audit only; relative age lets the same gate operate on naturally
        # short internal histories and longer external histories without
        # treating dataset scale as identity.
        "minimum_relative_age": (
            min(ages) / float(session_index) if ages else None
        ),
        "median_relative_age": (
            _median(ages) / float(session_index) if ages else None
        ),
        "maximum_relative_age": (
            max(ages) / float(session_index) if ages else None
        ),
        "top1_lexical_relevance": scores[0] if scores else 0.0,
        # The retriever remains the frozen lexical retriever.  This separate
        # bounded descriptor is for the PM opportunity head and does not alter
        # ranking or the realized candidate.
        "selected_content_word_match_level": max(
            (
                content_word_match_level(query, item.text)
                for item in selected_items
            ),
            default=0.0,
        ),
        "top1_top2_lexical_margin": (
            scores[0] - scores[1]
            if len(scores) >= 2
            else scores[0]
            if scores
            else 0.0
        ),
        "top_k_capacity_fraction": min(len(source_items), top_k)
        / float(top_k),
    }
    descriptor["model_features"] = candidate_model_features(descriptor)
    return descriptor


def discover_memory_candidates(
    *,
    query: str,
    items: Sequence[MemoryItem],
    retriever: MemoryRetriever,
    session_index: int,
) -> dict[MemorySource, MemoryCandidate]:
    """Discover all three source candidates before any injection decision."""

    discoveries: dict[MemorySource, MemoryCandidate] = {}
    for source in MemorySource:
        source_items = tuple(item for item in items if item.source is source)
        selected = tuple(
            retriever.retrieve(query, items, frozenset({source}))
        )
        discoveries[source] = MemoryCandidate(
            source=source,
            selected_items=selected,
            descriptor=describe_memory_candidate(
                source=source,
                query=query,
                source_items=source_items,
                selected_items=selected,
                session_index=session_index,
            ),
        )
    return discoveries


def discover_memory_candidates_by_source_query(
    *,
    queries: Mapping[MemorySource, str],
    items: Sequence[MemoryItem],
    retriever: MemoryRetriever,
    session_index: int,
) -> dict[MemorySource, MemoryCandidate]:
    """Discover each source using its frozen, source-appropriate query.

    This is a strict extension of :func:`discover_memory_candidates`; callers
    must supply all three queries explicitly.  It prevents an MS-specific
    repair from silently changing MP or ME retrieval and is reusable for both
    internal and external users.
    """

    missing = set(MemorySource) - set(queries)
    extra = set(queries) - set(MemorySource)
    if missing or extra:
        raise ValueError(
            f"source query map must contain exactly MemorySource: "
            f"missing={sorted(value.value for value in missing)}, "
            f"extra={sorted(str(value) for value in extra)}"
        )
    discoveries: dict[MemorySource, MemoryCandidate] = {}
    for source in MemorySource:
        query = str(queries[source])
        source_items = tuple(item for item in items if item.source is source)
        selected = tuple(
            retriever.retrieve(query, items, frozenset({source}))
        )
        discoveries[source] = MemoryCandidate(
            source=source,
            selected_items=selected,
            descriptor=describe_memory_candidate(
                source=source,
                query=query,
                source_items=source_items,
                selected_items=selected,
                session_index=session_index,
            ),
        )
    return discoveries


def discover_memory_candidates_by_source_query_v1_5b(
    *,
    queries: Mapping[MemorySource, str],
    items: Sequence[MemoryItem],
    retriever: MemoryRetriever,
    session_index: int,
) -> dict[MemorySource, MemoryCandidate]:
    """V1.5b candidate discovery with exact normalized Top-k deduplication.

    This is versioned rather than silently changing the frozen V1.5 stack.
    Ranking remains the production retriever's ranking.  The returned audit
    descriptor makes the before/after count explicit so cost and abstention
    analyses do not mistake duplicates for additional evidence.
    """

    missing = set(MemorySource) - set(queries)
    extra = set(queries) - set(MemorySource)
    if missing or extra:
        raise ValueError(
            f"source query map must contain exactly MemorySource: "
            f"missing={sorted(value.value for value in missing)}, "
            f"extra={sorted(str(value) for value in extra)}"
        )
    discoveries: dict[MemorySource, MemoryCandidate] = {}
    for source in MemorySource:
        query = str(queries[source])
        source_items = tuple(item for item in items if item.source is source)
        ranked = tuple(retriever.retrieve(query, items, frozenset({source})))
        selected = deduplicate_memory_items_by_text(ranked)
        descriptor = describe_memory_candidate(
            source=source,
            query=query,
            source_items=source_items,
            selected_items=selected,
            session_index=session_index,
        )
        descriptor.update(
            {
                "protocol": V1_5B_CANDIDATE_DISCOVERY_PROTOCOL,
                "retrieved_count_before_exact_text_dedup": len(ranked),
                "retrieved_count_after_exact_text_dedup": len(selected),
                "exact_text_duplicates_removed": len(ranked) - len(selected),
            }
        )
        descriptor["model_features"] = candidate_model_features(descriptor)
        discoveries[source] = MemoryCandidate(
            source=source,
            selected_items=selected,
            descriptor=descriptor,
        )
    return discoveries


def discover_final_typed_memory_candidates(
    *,
    queries: Mapping[MemorySource, str],
    items: Sequence[MemoryItem],
    source_metadata: Mapping[str, Mapping[str, Any]],
    session_index: int,
    ms_semantic_encoder: TextEncoder | None = None,
) -> dict[MemorySource, MemoryCandidate]:
    """Final transparent P2/runtime memory selection.

    The legacy lexical retriever admitted candidates on function-word overlap
    because its configured threshold was zero.  This selector implements the
    frozen plan's missing transparent reranker: require at least one shared
    content word, prefer the coarse content-overlap tier, then use a
    source-typed structural tier and lexical score only as deterministic
    tie-breakers.  It reads compiler metadata, never gold or response outcome.

    ``ms_semantic_encoder`` is opt-in and MS-only: when given, MS ranking
    uses BGE-M3 cosine similarity over the full MS pool
    (``v1_5_v5_3_semantic_ms_retrieval.rank_ms_candidates``) instead of the
    content-match-tier + lexical tie-break above, exactly reproducing the
    same-stack qualifying trial (``scripts/v1_5/45_ms_qualifying_trial_
    same_stack_v1_5.py``; 138/138 states, bge win rate 70.7%, sign test
    p=9.3e-5 -- see ``docs/PM_V1_5_V5_3_MS_QUALIFYING_TRIAL_FINDINGS_
    20260806_ZH.md``). Left ``None`` by default: the trial is still
    single-annotator with no independent second reviewer, so this does not
    flip production behavior on its own. MP and ME are unaffected regardless
    of this argument -- BGE has only been validated for MS (see that
    module's docstring for why it must not be extended to MP/ME without
    repeating the measurement).
    """

    missing = set(MemorySource) - set(queries)
    extra = set(queries) - set(MemorySource)
    if missing or extra:
        raise ValueError(
            f"source query map must contain exactly MemorySource: "
            f"missing={sorted(value.value for value in missing)}, "
            f"extra={sorted(str(value) for value in extra)}"
        )
    discoveries: dict[MemorySource, MemoryCandidate] = {}
    for source in MemorySource:
        query = str(queries[source])
        source_items = tuple(item for item in items if item.source is source)

        def typed_tier(item: MemoryItem) -> int:
            metadata = source_metadata.get(item.memory_id, {})
            if source is MemorySource.ME:
                hint = str(metadata.get("me_subtype_hint") or "")
                return {
                    "ME_REUSABLE_OUTCOME": 2,
                    "ME_UNRESOLVED_EVENT": 1,
                    "ME_CONTEXT_EVENT": 0,
                }.get(hint, 0)
            # MP subtype is not intrinsically better; current wording and
            # content overlap must choose between preference and profile.
            return 0

        def match_level(item: MemoryItem) -> float:
            document = (
                _mp_match_document(item.text)
                if source is MemorySource.MP
                else item.text
            )
            content_level = final_typed_content_match_level(query, document)
            metadata = source_metadata.get(item.memory_id, {})
            if (
                source is MemorySource.MP
                and metadata.get("mp_subtype") == "MP_PREFERENCE"
            ):
                return max(
                    content_level,
                    preference_scope_match_level(query, document),
                )
            return content_level

        eligible = [item for item in source_items if match_level(item) > 0.0]
        ms_semantic_reranked = (
            source is MemorySource.MS and ms_semantic_encoder is not None
        )
        semantic_ranked: list[Any] = []
        if ms_semantic_reranked:
            # Full MS pool, no content-word floor -- matches what the
            # qualifying trial actually measured (script 45), not a new,
            # unvalidated combination of the two mechanisms.
            semantic_ranked = rank_ms_candidates(
                query, list(source_items), encoder=ms_semantic_encoder
            )
            ranked = [candidate.item for candidate in semantic_ranked]
        else:
            ranked = sorted(
                eligible,
                key=lambda item: (
                    match_level(item),
                    typed_tier(item),
                    lexical_score(query, item.text),
                    item.created_session,
                    item.memory_id,
                ),
                reverse=True,
            )
        top_k = int(DEFAULT_MEMORY_TOP_K[source])
        selected = deduplicate_memory_items_by_text(ranked[:top_k])
        descriptor = describe_memory_candidate(
            source=source,
            query=query,
            source_items=source_items,
            selected_items=selected,
            session_index=session_index,
        )
        descriptor.update(
            {
                "protocol": FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
                "content_zero_items_filtered": len(source_items) - len(eligible),
                "retrieved_count_before_exact_text_dedup": min(
                    len(ranked), top_k
                ),
                "retrieved_count_after_exact_text_dedup": len(selected),
                "exact_text_duplicates_removed": min(len(ranked), top_k)
                - len(selected),
                "typed_rerank_outcome_read": False,
                "ms_semantic_reranked": ms_semantic_reranked,
            }
        )
        if ms_semantic_reranked:
            # describe_memory_candidate()'s top1_lexical_relevance/
            # top1_top2_lexical_margin are computed from lexical_score over
            # source_items regardless of which mechanism actually produced
            # `selected` -- correct when lexical_score was itself part of
            # ranking, but stale/misleading once BGE selects a different
            # top-1. These two fields carry the score BGE actually ranked
            # on, so a caller can tell the two apart. NOT wired into
            # model_features/bucketing: those buckets were calibrated
            # against lexical_score's distribution, and BGE cosine
            # similarity is not the same scale -- recalibrating that is a
            # separate decision, not made here.
            semantic_scores = sorted(
                (candidate.score for candidate in semantic_ranked), reverse=True
            )
            descriptor["top1_semantic_relevance"] = (
                semantic_scores[0] if semantic_scores else 0.0
            )
            descriptor["top1_top2_semantic_margin"] = (
                semantic_scores[0] - semantic_scores[1]
                if len(semantic_scores) >= 2
                else semantic_scores[0]
                if semantic_scores
                else 0.0
            )
        descriptor["model_features"] = candidate_model_features(descriptor)
        discoveries[source] = MemoryCandidate(
            source=source,
            selected_items=selected,
            descriptor=descriptor,
        )
    return discoveries


def public_candidate_descriptors(
    discoveries: Mapping[MemorySource, MemoryCandidate],
) -> dict[str, dict[str, Any]]:
    """Serialize only PM-safe descriptors, never selected text or IDs."""

    return {
        source.value: dict(discoveries[source].descriptor)
        for source in MemorySource
    }


def choose_after_candidate_discovery(
    *,
    model: Any,
    pm_state: Any,
    discoveries: Mapping[MemorySource, MemoryCandidate],
) -> tuple[Any, Any]:
    """Attach PM-safe descriptors, then call the policy exactly once."""

    candidate_aware_state = pm_state.model_copy(
        update={
            "provenance": {
                **dict(pm_state.provenance),
                "candidate_discovery_stage": (
                    "post-retrieval_pre-injection_pre-generation"
                ),
                "memory_candidate_descriptors": (
                    public_candidate_descriptors(discoveries)
                ),
            }
        }
    )
    return candidate_aware_state, model.choose(candidate_aware_state)
