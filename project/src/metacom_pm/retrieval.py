from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence
from .contracts import MemoryItem, MemorySource, StrategyCard
from .text import lexical_score, normalize_space


DEFAULT_MEMORY_TOP_K = {
    MemorySource.MP: 2,
    MemorySource.MS: 2,
    MemorySource.ME: 3,
}


def context_query(
    current_user_text: str,
    history: Sequence[dict] | Sequence[object],
    summary: str,
) -> str:
    parts = [summary]
    for turn in history:
        if isinstance(turn, dict):
            parts.append(str(turn.get("content") or ""))
        else:
            parts.append(str(getattr(turn, "content", "")))
    parts.append(current_user_text)
    return normalize_space("\n".join(parts))


class MemoryRetriever:
    def __init__(
        self,
        top_k_by_source: dict[MemorySource, int] | None = None,
        minimum_score_by_source: dict[MemorySource, float] | None = None,
    ):
        self.top_k_by_source = top_k_by_source or dict(DEFAULT_MEMORY_TOP_K)
        self.minimum_score_by_source = minimum_score_by_source

    def retrieve(
        self,
        query: str,
        items: Sequence[MemoryItem],
        selected_sources: frozenset[MemorySource],
    ) -> list[MemoryItem]:
        out: list[MemoryItem] = []
        for source in MemorySource:
            if source not in selected_sources:
                continue
            candidates = [x for x in items if x.source is source]
            scored = [(lexical_score(query, item.text), item) for item in candidates]
            if self.minimum_score_by_source is not None:
                threshold = float(self.minimum_score_by_source.get(source, 0.0))
                # A configured threshold is fail-closed: exact-zero lexical matches
                # are not injected merely because a source was selected.
                scored = [(score, item) for score, item in scored if score > threshold]
            ranked = sorted(
                scored,
                key=lambda pair: (
                    pair[0],
                    pair[1].created_session,
                    pair[1].memory_id,
                ),
                reverse=True,
            )
            out.extend(item for _, item in ranked[: self.top_k_by_source[source]])
        return out


class StrategyRetriever:
    def __init__(
        self,
        cards: Sequence[StrategyCard],
        top_k: int = 3,
        minimum_score: float | None = None,
    ):
        self.cards = list(cards)
        self.top_k = top_k
        self.minimum_score = minimum_score

    def retrieve(self, query: str) -> list[StrategyCard]:
        scored = [(lexical_score(query, card.retrieval_text), card) for card in self.cards]
        if self.minimum_score is not None:
            scored = [
                (score, card)
                for score, card in scored
                if score > float(self.minimum_score)
            ]
        ranked = sorted(
            scored,
            key=lambda pair: (pair[0], pair[1].strategy_id),
            reverse=True,
        )
        return [card for _, card in ranked[: self.top_k]]

    def confidence(self, query: str) -> float:
        if not self.cards:
            return 0.0
        return max(lexical_score(query, card.retrieval_text) for card in self.cards)
