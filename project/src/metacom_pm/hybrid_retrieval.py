"""Diagnostic-only hybrid (lexical + semantic) retrieval scorer.

This module is strictly isolated: nothing under real training, sweep, ESConv,
EvoEmo generation, fixed-baseline, freeze, or paid-runner code may import from
here (enforced by ``tests/test_hybrid_retrieval_isolation.py``). It exists
solely to produce report-only diagnostics (see
``scripts/v1_5/40_diagnose_hybrid_retrieval.py``) that may inform a future,
separately-approved adoption decision. Importing this module must never be a
side effect of any real pipeline stage.

Fusion/normalization contract (frozen, not tuned against any result):

- Query-local min-max normalization is rejected: it is undefined for a single
  candidate or when every candidate ties, and it is unstable under catalog
  perturbation (the same item's normalized score depends on what else
  happens to be in the pool for that call).
- Instead: **reciprocal rank fusion**. Each candidate is ranked independently
  by raw lexical score and by raw semantic cosine similarity, each with a
  fully deterministic secondary/tertiary tie-break
  (``score`` desc, then ``created_session`` desc, then id desc) so ranks are
  always a strict total order -- a single candidate trivially gets rank 1 on
  both rankers, and an all-tied raw-score pool is still ranked unambiguously
  by the tie-break. Fusion score is
  ``sum(1 / (RECIPROCAL_RANK_FUSION_K + rank))`` across the two rankers.
- Rank fusion is honestly **not** invariant to catalog composition: adding or
  removing an unrelated candidate can shift another candidate's assigned
  rank by shifting how many candidates sit above it, which changes that
  candidate's fused score even though its own raw scores did not change.
  This is a known, disclosed property of rank-based fusion, not a bug -- it
  is preferred here specifically because it degrades gracefully (bounded
  score movement) rather than the undefined/infinite-sensitivity failure
  modes of query-local min-max normalization.
- **Independent per-source score floors** are applied to raw (unranked)
  scores before ranking, as a fail-closed exclusion of candidates neither
  scorer considers even weakly relevant: a candidate is dropped only if its
  raw lexical score is *strictly below* ``lexical_min_score`` *and* its raw
  semantic score is *strictly below* ``semantic_min_score`` -- either scorer
  alone can rescue a candidate the other misses, which is the entire reason
  to hybridize. The comparison is strict (``<``), not ``<=``, specifically
  so that a floor computed as "the minimum score among known-good examples"
  (see ``hybrid_retrieval_diagnostics.calibrate_source_floors``) can never
  exclude the very example that defines it: that example's score sits AT
  the floor, not below it, on whichever scorer it was the minimum for.
  Floors are calibrated only against the synthetic train/calibration
  split's evaluator-only memory-utility labels (see the diagnostic script)
  and are recorded, not hand-guessed.
- Final tie-break after fusion is the same deterministic secondary key
  pattern already used by the real, frozen retrievers in ``retrieval.py``:
  ``(fusion_score, created_session, memory_id_or_strategy_id)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .contracts import MemoryItem, MemorySource, StrategyCard
from .io import canonical_json, sha256_text
from .pm_v1_5_semantic import SemanticTextEncoder
from .retrieval import context_query
from .text import estimate_tokens, lexical_score


HYBRID_RETRIEVAL_PROTOCOL = "pm-v1.5-hybrid-retrieval-diagnostic-v1"

# Cormack et al. (2009)'s reciprocal-rank-fusion constant. Frozen at the
# standard literature default -- never tuned against this project's own
# retrieval results, which would make "not tuned" a false claim.
RECIPROCAL_RANK_FUSION_K = 60

# SemanticTextEncoder.encode(texts) puts every text into a single forward
# pass with no internal batching -- fine for the handful of texts in one
# retrieval call, but encoding a large text collection (e.g. the real
# ~11.6k-card Strategy Bank) in ONE unbatched call is a genuine
# memory/latency risk, not just a theoretical one (observed multi-hour,
# tens-of-GB blowup encoding the full bank in one shot). Always go through
# batched_encode for any collection whose size isn't bounded by a single
# retrieval call's candidate pool.
DEFAULT_ENCODE_BATCH_SIZE = 128


def batched_encode(
    encoder: SemanticTextEncoder,
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Encode ``texts`` in fixed-size batches and concatenate the result.

    ``progress(batches_done, total_batches)``, if given, is called after
    each batch -- useful for a large collection (e.g. the ~11.6k-card
    Strategy Bank, ~91 batches at the default size) where the whole call
    can take long enough that visibility into per-batch progress matters.
    """

    if not texts:
        raise ValueError("batched_encode requires at least one text")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    starts = list(range(0, len(texts), batch_size))
    chunks = []
    for batch_index, start in enumerate(starts, start=1):
        chunks.append(encoder.encode(texts[start : start + batch_size]))
        if progress is not None:
            progress(batch_index, len(starts))
    return np.concatenate(chunks, axis=0)


@dataclass(frozen=True)
class SourceScoreFloors:
    """Independent, per-source raw-score floors, calibrated externally.

    Both fields are required (no defaults) so a caller must explicitly
    supply calibrated values rather than accidentally rely on an
    unexamined default.
    """

    lexical_min_score: float
    semantic_min_score: float


def hybrid_retrieval_contract_hash(
    *,
    memory_floors_by_source: dict[MemorySource, SourceScoreFloors],
    strategy_floors: SourceScoreFloors,
    rrf_k: int,
    semantic_encoder_spec_sha256: str,
) -> str:
    payload = {
        "protocol": HYBRID_RETRIEVAL_PROTOCOL,
        "fusion_method": "reciprocal_rank_fusion",
        "rrf_k": int(rrf_k),
        "memory_floors_by_source": {
            source.value: {
                "lexical_min_score": floors.lexical_min_score,
                "semantic_min_score": floors.semantic_min_score,
            }
            for source, floors in sorted(
                memory_floors_by_source.items(), key=lambda pair: pair[0].value
            )
        },
        "strategy_floors": {
            "lexical_min_score": strategy_floors.lexical_min_score,
            "semantic_min_score": strategy_floors.semantic_min_score,
        },
        "semantic_encoder_spec_sha256": semantic_encoder_spec_sha256,
    }
    return sha256_text(canonical_json(payload))


def observable_query_with_hash(
    current_user_text: str,
    history: Sequence[dict] | Sequence[object],
    summary: str,
) -> tuple[str, str]:
    """The only sanctioned way to build a query for real retrieval.

    Built exclusively from deployment-observable fields (matching
    ``retrieval.context_query``): current user text, visible session
    history, session summary. Never evaluator-only fields (``topic``,
    ``related_sessions``, or anything else confirmatory-only). Returns
    ``(query_text, query_hash)`` so every diagnostic row can record
    ``query_hash`` for after-the-fact auditability without needing to
    persist the raw text.
    """

    query_text = context_query(current_user_text, history, summary)
    return query_text, sha256_text(query_text)


def _ranked_positions(
    scored: Sequence[tuple[float, int, str]],
) -> dict[str, int]:
    """1-indexed rank per id, using a fully deterministic total order.

    ``scored`` rows are ``(raw_score, created_session, item_id)``. Ties on
    ``raw_score`` (including an all-tied pool) are broken by
    ``created_session`` then ``item_id``, so every candidate gets a unique,
    unambiguous rank -- a pool of exactly one candidate trivially ranks it
    1st.
    """

    ordered = sorted(
        scored,
        key=lambda row: (row[0], row[1], row[2]),
        reverse=True,
    )
    return {row[2]: rank for rank, row in enumerate(ordered, start=1)}


def _reciprocal_rank_fusion_score(
    *, lexical_rank: int, semantic_rank: int, k: int
) -> float:
    return 1.0 / (k + lexical_rank) + 1.0 / (k + semantic_rank)


class HybridMemoryRetriever:
    """Diagnostic-only hybrid scorer with the same interface as MemoryRetriever.

    Never imported by any real consumer -- see the module docstring.
    """

    def __init__(
        self,
        *,
        semantic_encoder: SemanticTextEncoder,
        top_k_by_source: dict[MemorySource, int],
        floors_by_source: dict[MemorySource, SourceScoreFloors],
        rrf_k: int = RECIPROCAL_RANK_FUSION_K,
    ):
        self.semantic_encoder = semantic_encoder
        self.top_k_by_source = dict(top_k_by_source)
        self.floors_by_source = dict(floors_by_source)
        self.rrf_k = int(rrf_k)

    def _fuse_pool(
        self,
        query_text: str,
        query_vector,
        candidates: Sequence[MemoryItem],
        floors_of: Callable[[MemoryItem], SourceScoreFloors],
    ) -> list[tuple[float, int, str, MemoryItem]]:
        """Fail-closed-filter, rank, and fuse one pool of candidates.

        ``floors_of`` is called per-candidate so callers can either apply one
        source's floors to a single-source pool (``retrieve``) or each
        candidate's own source-specific floors within one combined,
        cross-source pool (``rank_all``) -- ranking itself always happens
        over exactly the candidates passed in, in one pool. Returns rows
        already sorted best-first by
        ``(fusion_score, created_session, memory_id)``.
        """

        if not candidates:
            return []
        item_vectors = self.semantic_encoder.encode([item.text for item in candidates])
        lexical_rows: list[tuple[float, int, str]] = []
        semantic_rows: list[tuple[float, int, str]] = []
        raw_by_id: dict[str, MemoryItem] = {}
        for item, item_vector in zip(candidates, item_vectors):
            floors = floors_of(item)
            lex = lexical_score(query_text, item.text)
            sem = float(query_vector @ item_vector)
            if lex < floors.lexical_min_score and sem < floors.semantic_min_score:
                continue
            lexical_rows.append((lex, item.created_session, item.memory_id))
            semantic_rows.append((sem, item.created_session, item.memory_id))
            raw_by_id[item.memory_id] = item
        if not raw_by_id:
            return []
        lexical_rank = _ranked_positions(lexical_rows)
        semantic_rank = _ranked_positions(semantic_rows)
        fused = [
            (
                _reciprocal_rank_fusion_score(
                    lexical_rank=lexical_rank[memory_id],
                    semantic_rank=semantic_rank[memory_id],
                    k=self.rrf_k,
                ),
                item.created_session,
                memory_id,
                item,
            )
            for memory_id, item in raw_by_id.items()
        ]
        return sorted(fused, key=lambda row: (row[0], row[1], row[2]), reverse=True)

    def retrieve(
        self,
        query: str,
        items: Sequence[MemoryItem],
        selected_sources: frozenset[MemorySource],
    ) -> list[MemoryItem]:
        """Per-source top-k, matching MemoryRetriever's public interface.

        Each source is ranked (and fused) within its own pool, then the
        source's own top-k slice is taken -- identical partitioning to the
        real, frozen ``MemoryRetriever.retrieve``.
        """

        out: list[MemoryItem] = []
        query_vector = self.semantic_encoder.encode([query])[0]
        for source in MemorySource:
            if source not in selected_sources:
                continue
            candidates = [item for item in items if item.source is source]
            if not candidates:
                continue
            floors = self.floors_by_source[source]
            ranked = self._fuse_pool(query, query_vector, candidates, lambda _item: floors)
            out.extend(row[3] for row in ranked[: self.top_k_by_source[source]])
        return out

    def rank_all(
        self,
        query: str,
        items: Sequence[MemoryItem],
        selected_sources: frozenset[MemorySource],
    ) -> list[MemoryItem]:
        """Full ranking across all selected sources as one combined pool.

        Each candidate is still subject to its own source's floors, but
        lexical/semantic ranks (and hence rank fusion) are computed over the
        union of every selected source's candidates as a single pool -- this
        is what the fixed-token-budget mode needs (a global evidence budget
        is not partitioned per source), unlike ``retrieve``'s per-source
        top-k. Returns the complete ranked list, never sliced.
        """

        query_vector = self.semantic_encoder.encode([query])[0]
        candidates = [item for item in items if item.source in selected_sources]
        ranked = self._fuse_pool(
            query, query_vector, candidates, lambda item: self.floors_by_source[item.source]
        )
        return [row[3] for row in ranked]


class HybridStrategyRetriever:
    """Diagnostic-only hybrid scorer with the same interface as StrategyRetriever.

    Never imported by any real consumer -- see the module docstring.
    """

    def __init__(
        self,
        cards: Sequence[StrategyCard],
        *,
        semantic_encoder: SemanticTextEncoder,
        top_k: int = 3,
        floors: SourceScoreFloors,
        rrf_k: int = RECIPROCAL_RANK_FUSION_K,
        embedding_progress: Callable[[int, int], None] | None = None,
    ):
        self.cards = list(cards)
        self.semantic_encoder = semantic_encoder
        self.top_k = top_k
        self.floors = floors
        self.rrf_k = int(rrf_k)
        # The Strategy Bank is large (11k+ cards in the real V1.5 bank) and
        # fixed once this retriever is constructed -- embed every card's
        # retrieval_text exactly once here rather than re-encoding the
        # entire bank on every rank_all()/retrieve() call. Batched (not one
        # giant encode() call): see batched_encode's docstring.
        self._card_vectors = (
            batched_encode(
                self.semantic_encoder,
                [card.retrieval_text for card in self.cards],
                progress=embedding_progress,
            )
            if self.cards
            else None
        )

    def rank_all(self, query: str) -> list[StrategyCard]:
        """Full ranking of every card passing the floor filter, never sliced."""

        if not self.cards:
            return []
        query_vector = self.semantic_encoder.encode([query])[0]
        card_vectors = self._card_vectors
        lexical_rows: list[tuple[float, int, str]] = []
        semantic_rows: list[tuple[float, int, str]] = []
        raw_by_id: dict[str, StrategyCard] = {}
        for card, card_vector in zip(self.cards, card_vectors):
            lex = lexical_score(query, card.retrieval_text)
            sem = float(query_vector @ card_vector)
            if lex < self.floors.lexical_min_score and sem < self.floors.semantic_min_score:
                continue
            # StrategyCard has no created_session; use a constant so the
            # deterministic tie-break reduces to (score, strategy_id).
            lexical_rows.append((lex, 0, card.strategy_id))
            semantic_rows.append((sem, 0, card.strategy_id))
            raw_by_id[card.strategy_id] = card
        if not raw_by_id:
            return []
        lexical_rank = _ranked_positions(lexical_rows)
        semantic_rank = _ranked_positions(semantic_rows)
        fused = [
            (
                _reciprocal_rank_fusion_score(
                    lexical_rank=lexical_rank[strategy_id],
                    semantic_rank=semantic_rank[strategy_id],
                    k=self.rrf_k,
                ),
                strategy_id,
                card,
            )
            for strategy_id, card in raw_by_id.items()
        ]
        ranked = sorted(fused, key=lambda row: (row[0], row[1]), reverse=True)
        return [row[2] for row in ranked]

    def retrieve(self, query: str) -> list[StrategyCard]:
        """Top-k, matching StrategyRetriever's public interface."""

        return self.rank_all(query)[: self.top_k]

    def confidence(self, query: str) -> float:
        """Matches StrategyRetriever.confidence's exact lexical-only semantics.

        Provided for interface parity only (real callers such as
        ``policies.py`` gate on ``confidence(query) >= threshold``). Deliberately
        NOT redefined as a hybrid/fused quantity: no real consumer uses this
        method yet, and inventing a new fused-confidence formula without any
        calibration or validation evidence would repeat the same mistake this
        project exists to avoid -- fabricating a number instead of measuring
        one. If Hybrid is ever adopted, this method should be revisited
        together with that decision, not before.
        """

        if not self.cards:
            return 0.0
        return max(lexical_score(query, card.retrieval_text) for card in self.cards)


def retrieve_fixed_token_budget(
    ranked_items: Sequence[object],
    *,
    token_budget: int,
    text_of: Callable[[object], str],
) -> list[object]:
    """Greedily fill a fixed evidence-token budget from an already-ranked list.

    Adds whole items in rank order, never truncating an item's text; an
    item that would push the running total over budget is *skipped* (not
    a stopping point), so lower-ranked items still get a chance and the
    budget is used as fully as possible without ever including a partial
    item. ``ranked_items`` must already be in final rank order (deterministic
    tie-break already applied by the caller).
    """

    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    out: list[object] = []
    used = 0
    for item in ranked_items:
        cost = estimate_tokens(text_of(item))
        if used + cost > token_budget:
            continue
        out.append(item)
        used += cost
    return out
