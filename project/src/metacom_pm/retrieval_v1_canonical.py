"""Byte-semantic compatibility wrapper for the frozen V1 Strategy RAG.

The implementation intentionally reproduces ``src/metacom_pm/retrieval.py``
at frozen SHA256 5202ca6e... for strategy retrieval only.  It does not import
the dirty V2 retriever and deliberately exposes no threshold, abstention,
embedding, index, filter, or reranker option.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Sequence

from .contracts import StrategyCard


CANONICAL_V1_RETRIEVER_SHA256 = (
    "5202ca6e511d06254e0092629cb197f4e4536bb7bc90024cce78a3546c698d92"
)
CANONICAL_V1_TOP_K = 3
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


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


def lexical_score(query: str, document: str) -> float:
    q = Counter(TOKEN_RE.findall(str(query or "").lower()))
    d = Counter(TOKEN_RE.findall(str(document or "").lower()))
    if not q or not d:
        return 0.0
    dot = sum(q[token] * d.get(token, 0) for token in q)
    q_norm = math.sqrt(sum(value * value for value in q.values()))
    d_norm = math.sqrt(sum(value * value for value in d.values()))
    return dot / (q_norm * d_norm) if q_norm and d_norm else 0.0


class StrategyRetrieverV1Canonical:
    """Frozen V1 full-scan lexical retriever; no optional behavior."""

    def __init__(
        self,
        cards: Sequence[StrategyCard],
        top_k: int = CANONICAL_V1_TOP_K,
    ) -> None:
        if int(top_k) != CANONICAL_V1_TOP_K:
            raise ValueError("canonical V1 Strategy RAG requires top_k=3")
        self.cards = list(cards)
        self.top_k = CANONICAL_V1_TOP_K

    def retrieve_with_scores(self, query: str) -> list[tuple[float, StrategyCard]]:
        ranked = sorted(
            (
                (lexical_score(query, card.retrieval_text), card)
                for card in self.cards
            ),
            key=lambda pair: (pair[0], pair[1].strategy_id),
            reverse=True,
        )
        return ranked[: self.top_k]

    def retrieve(self, query: str) -> list[StrategyCard]:
        return [card for _, card in self.retrieve_with_scores(query)]

    def confidence(self, query: str) -> float:
        if not self.cards:
            return 0.0
        return max(lexical_score(query, card.retrieval_text) for card in self.cards)
