from __future__ import annotations

from collections.abc import Sequence
import time

from .contracts import (
    MemoryItem,
    MemorySource,
    RetrievalAttempt,
    StrategyCard,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
)
from .retrieval import MemoryRetriever, StrategyRetriever
from .text import estimate_tokens


RETRIEVAL_CHANNELS = ("MP", "MS", "ME", "STRATEGY")


def realized_action_id_from_evidence(
    memory_view: Sequence[MemoryItem],
    strategy_view: Sequence[StrategyCard],
) -> str:
    """Derive the action that actually reached the generator prompt."""

    return canonical_action_id(
        {item.source for item in memory_view},
        StrategyMode.RS if strategy_view else StrategyMode.R0,
    )


def execute_requested_retrievals(
    *,
    requested_action_id: str,
    query: str,
    memory_items: Sequence[MemoryItem],
    memory_retriever: MemoryRetriever,
    strategy_retriever: StrategyRetriever,
) -> tuple[list[MemoryItem], list[StrategyCard], list[RetrievalAttempt]]:
    """Execute and audit each requested retrieval channel independently.

    Memory retrieval remains local and deterministic, but timing each source
    separately prevents a combined call from obscuring which requested source
    produced zero candidates.  Non-requested channels are retained explicitly
    as ``not_requested`` rows so the attempt vector is unambiguous.
    """

    requested_sources, strategy_mode = parse_action_id(requested_action_id)
    memory_view: list[MemoryItem] = []
    strategy_view: list[StrategyCard] = []
    attempts: list[RetrievalAttempt] = []

    for source in MemorySource:
        requested = source in requested_sources
        if not requested:
            attempts.append(
                RetrievalAttempt(
                    channel=source.value,
                    requested=False,
                    called=False,
                    status="not_requested",
                    hit_count=0,
                    retrieved_tokens=0,
                    latency_ms=0.0,
                    api_cost_usd=0.0,
                )
            )
            continue
        started = time.perf_counter()
        rows = memory_retriever.retrieve(query, memory_items, frozenset({source}))
        latency_ms = (time.perf_counter() - started) * 1000.0
        memory_view.extend(rows)
        attempts.append(
            RetrievalAttempt(
                channel=source.value,
                requested=True,
                called=True,
                status="succeeded",
                hit_count=len(rows),
                retrieved_tokens=sum(estimate_tokens(row.text) for row in rows),
                latency_ms=latency_ms,
                api_cost_usd=0.0,
            )
        )

    strategy_requested = strategy_mode is StrategyMode.RS
    if strategy_requested:
        started = time.perf_counter()
        strategy_view = strategy_retriever.retrieve(query)
        latency_ms = (time.perf_counter() - started) * 1000.0
        attempts.append(
            RetrievalAttempt(
                channel="STRATEGY",
                requested=True,
                called=True,
                status="succeeded",
                hit_count=len(strategy_view),
                retrieved_tokens=sum(
                    estimate_tokens(row.guidance_text + row.example_response)
                    for row in strategy_view
                ),
                latency_ms=latency_ms,
                api_cost_usd=0.0,
            )
        )
    else:
        attempts.append(
            RetrievalAttempt(
                channel="STRATEGY",
                requested=False,
                called=False,
                status="not_requested",
                hit_count=0,
                retrieved_tokens=0,
                latency_ms=0.0,
                api_cost_usd=0.0,
            )
        )

    return memory_view, strategy_view, attempts
