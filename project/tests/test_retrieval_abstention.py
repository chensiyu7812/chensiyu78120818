from __future__ import annotations

from metacom_pm.contracts import MemorySource
from metacom_pm.retrieval import MemoryRetriever, StrategyRetriever


def test_memory_retriever_can_abstain_from_zero_overlap_items(tiny_memories):
    selected = frozenset({MemorySource.MP, MemorySource.ME})
    legacy = MemoryRetriever().retrieve("completely unrelated vocabulary", tiny_memories, selected)
    guarded = MemoryRetriever(
        minimum_score_by_source={source: 0.0 for source in MemorySource}
    ).retrieve("completely unrelated vocabulary", tiny_memories, selected)

    assert legacy
    assert guarded == []


def test_strategy_retriever_can_abstain_from_zero_overlap(tiny_strategy):
    legacy = StrategyRetriever([tiny_strategy]).retrieve("unrelated vocabulary")
    guarded = StrategyRetriever(
        [tiny_strategy], minimum_score=0.0
    ).retrieve("unrelated vocabulary")

    assert legacy == [tiny_strategy]
    assert guarded == []
