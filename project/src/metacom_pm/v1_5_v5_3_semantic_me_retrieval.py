"""V5.3 W6: BGE-M3 / hybrid ME reranker candidates (opt-in, NOT wired into
production).

Mirrors ``v1_5_v5_3_semantic_ms_retrieval.py``'s scope discipline exactly:
that module explicitly warns not to extend BGE-M3 to ME without repeating
the measurement for ME's own content type (short first-person "action +
result" sentences, not MS's longer narrative summaries). This module is
that repeated measurement's implementation, ME-scoped only, and remains
opt-in -- see ``scripts/v1_5/70w_qualify_me_reranker_v1_5.py`` for the
qualification comparison. Adoption into any production default is a leader
decision, not made here.

Two candidate rerankers, both deterministic given a fixed encoder:

- ``rank_me_candidates_bge``: pure BGE-M3 cosine similarity over the full
  candidate set, no pre-filter -- directly comparable to
  ``rank_ms_candidates``'s shape.
- ``rank_me_candidates_hybrid``: a fixed two-stage pipeline -- first keep
  only items that are strictly prior (``created_session < session_index``)
  AND pass ``compile_atomic_reusable_outcome`` (the real ME compiler; "same
  owner" is enforced by construction upstream of this function, since
  ``MemoryItem`` carries no owner field itself -- caller must already pass
  only same-user items, exactly as ``discover_final_typed_memory_
  candidates`` requires today), THEN rank the survivors by BGE-M3 cosine.
  Ties break by (created_session desc, memory_id desc), matching this
  project's established tie-break convention everywhere else. This order
  (filter first, then rank) is fixed before any qualification result is
  read -- see the runbook's W6 instruction not to add a fourth method or
  reweight after seeing results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import MemoryItem, MemorySource
from .v1_5_v5_2_atomic_memory import compile_atomic_reusable_outcome
from .v1_5_v5_3_semantic_ms_retrieval import TextEncoder

SEMANTIC_ME_RETRIEVAL_PROTOCOL = "pm-v1.5-v5.3-semantic-me-retrieval-v1"


@dataclass(frozen=True)
class RankedMeCandidate:
    item: MemoryItem
    score: float
    rank: int


def _require_me_only(items: Sequence[MemoryItem]) -> list[MemoryItem]:
    me_items = [item for item in items if item.source is MemorySource.ME]
    if len(me_items) != len(items):
        raise ValueError(
            "this module only accepts MemorySource.ME items; it has not been "
            "validated for MP or MS (see module docstring)"
        )
    return me_items


def _bge_rank(query: str, items: Sequence[MemoryItem], encoder: TextEncoder) -> list[RankedMeCandidate]:
    if not items:
        return []
    vectors = encoder.encode([query, *[item.text for item in items]])
    query_vector, item_vectors = vectors[0], vectors[1:]
    scored = [
        (float(query_vector @ item_vector), item.created_session, item.memory_id, item)
        for item_vector, item in zip(item_vectors, items)
    ]
    scored.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    return [
        RankedMeCandidate(item=item, score=score, rank=rank)
        for rank, (score, _created_session, _memory_id, item) in enumerate(scored, start=1)
    ]


def rank_me_candidates_bge(
    query: str, items: Sequence[MemoryItem], *, encoder: TextEncoder,
) -> list[RankedMeCandidate]:
    """Pure BGE-M3 cosine over the full ME candidate set, no pre-filter."""

    me_items = _require_me_only(items)
    return _bge_rank(query, me_items, encoder)


def rank_me_candidates_hybrid(
    query: str, items: Sequence[MemoryItem], *, encoder: TextEncoder, session_index: int,
) -> list[RankedMeCandidate]:
    """Stage 1 (mechanical): keep strictly-prior, compiler-valid items only.
    Stage 2: BGE-M3 cosine ranks the survivors. If stage 1 empties the pool,
    returns []  -- callers must fall back to M0/no-ME themselves, this
    function does not silently widen its own filter."""

    me_items = _require_me_only(items)
    survivors = [
        item for item in me_items
        if item.created_session < session_index
        and compile_atomic_reusable_outcome(item.text) is not None
    ]
    return _bge_rank(query, survivors, encoder)
