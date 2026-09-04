import numpy as np
import pytest

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.v1_5_v5_3_semantic_ms_retrieval import (
    CachedTextEncoder,
    RankedMsCandidate,
    rank_ms_candidates,
)


class FakeEncoder:
    """Deterministic stand-in for BgeM3Encoder: exact keyword-match vectors.

    Keeps the test fast and dependency-free (no torch/transformers/model
    weights) while still exercising the real ranking/tie-break logic in
    rank_ms_candidates, which is what this test actually needs to cover.
    """

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary

    def encode(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([1.0 if word in lowered else 0.0 for word in self.vocabulary])
        matrix = np.array(rows, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def ms_item(memory_id: str, text: str, created_session: int = 0) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        source=MemorySource.MS,
        created_session=created_session,
        text=text,
    )


def test_ranks_topically_matching_candidate_first():
    # item2 and item3 share zero vocabulary words with the query and with
    # each other, so their relative order is a legitimate tie-break (not
    # part of what this test is checking) -- only item1's rank-1 position
    # and strictly-higher score are asserted.
    encoder = FakeEncoder(["wedding", "job", "thesis"])
    items = [
        ms_item("mem_000000000001", "Struggling with wedding planning stress with David."),
        ms_item("mem_000000000002", "Anxious about a new job offer in another city."),
        ms_item("mem_000000000003", "Thesis advisor gave discouraging feedback."),
    ]
    ranked = rank_ms_candidates(
        "I've been arguing with my partner about wedding plans again.",
        items,
        encoder=encoder,
    )
    assert ranked[0].item.memory_id == "mem_000000000001"
    assert ranked[0].rank == 1
    assert ranked[0].score > ranked[1].score
    assert {r.item.memory_id for r in ranked[1:]} == {"mem_000000000002", "mem_000000000003"}


def test_top_k_slices_ranked_output():
    encoder = FakeEncoder(["wedding", "job"])
    items = [
        ms_item("mem_000000000001", "wedding stress"),
        ms_item("mem_000000000002", "job offer anxiety"),
    ]
    ranked = rank_ms_candidates("wedding", items, encoder=encoder, top_k=1)
    assert len(ranked) == 1
    assert ranked[0].item.memory_id == "mem_000000000001"


def test_empty_candidates_returns_empty_list():
    encoder = FakeEncoder(["wedding"])
    assert rank_ms_candidates("wedding", [], encoder=encoder) == []


def test_rejects_non_ms_sources():
    encoder = FakeEncoder(["wedding"])
    items = [
        MemoryItem(
            memory_id="mem_000000000001",
            source=MemorySource.MP,
            created_session=0,
            text="prefers short replies",
        )
    ]
    with pytest.raises(ValueError, match="only accepts MemorySource.MS"):
        rank_ms_candidates("wedding", items, encoder=encoder)


def test_deterministic_tie_break_on_equal_score():
    encoder = FakeEncoder(["wedding"])
    items = [
        ms_item("mem_000000000001", "wedding wedding", created_session=1),
        ms_item("mem_000000000002", "wedding wedding", created_session=2),
    ]
    ranked = rank_ms_candidates("wedding", items, encoder=encoder)
    # Equal cosine score -> tie-break by created_session desc, matching
    # retrieval.MemoryRetriever's convention.
    assert [r.item.memory_id for r in ranked] == ["mem_000000000002", "mem_000000000001"]


def test_cached_encoder_reuses_exact_candidate_vectors_without_changing_values():
    class CountingEncoder(FakeEncoder):
        def __init__(self):
            super().__init__(["wedding", "job"])
            self.calls: list[list[str]] = []

        def encode(self, texts):
            self.calls.append(list(texts))
            return super().encode(texts)

    base = CountingEncoder()
    cached = CachedTextEncoder(base)
    first = cached.encode(["new wedding question", "same candidate"])
    second = cached.encode(["new job question", "same candidate"])
    assert len(base.calls) == 2
    assert base.calls[0] == ["new wedding question", "same candidate"]
    assert base.calls[1] == ["new job question"]
    np.testing.assert_array_equal(first[1], second[1])
