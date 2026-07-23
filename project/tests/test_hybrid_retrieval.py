from __future__ import annotations

import numpy as np
import pytest

from metacom_pm.contracts import MemoryItem, MemorySource, StrategyCard
from metacom_pm.hybrid_retrieval import (
    HYBRID_RETRIEVAL_PROTOCOL,
    RECIPROCAL_RANK_FUSION_K,
    HybridMemoryRetriever,
    HybridStrategyRetriever,
    SourceScoreFloors,
    batched_encode,
    hybrid_retrieval_contract_hash,
    observable_query_with_hash,
    retrieve_fixed_token_budget,
)
from metacom_pm.io import sha256_text
from metacom_pm.retrieval import context_query
from metacom_pm.text import lexical_score


class _FakeEncoder:
    """Deterministic stand-in: cosine similarity == normalized token overlap.

    Maps each text to a one-hot-ish vector over its token vocabulary so the
    dot product between two encoded rows equals a simple, hand-computable
    similarity -- lets tests assert exact ranks without needing the real
    BAAI/bge-small-en-v1.5 model.
    """

    def __init__(self, vocab: dict[str, np.ndarray]):
        self.vocab = vocab

    def encode(self, texts):
        return np.stack([self.vocab[text] for text in texts])


def _one_hot(dimension: int, index: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float64)
    vector[index] = 1.0
    return vector


def _floors(lexical: float = -1.0, semantic: float = -1.0) -> SourceScoreFloors:
    return SourceScoreFloors(lexical_min_score=lexical, semantic_min_score=semantic)


def _item(memory_id: str, source: MemorySource, session: int, text: str) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id, source=source, created_session=session, text=text
    )


def test_single_candidate_ranks_first_trivially_on_both_scorers():
    query = "work deadline"
    item = _item("mem_aaaaaaaaaaaa", MemorySource.ME, 1, "work deadline stress")
    encoder = _FakeEncoder(
        {query: _one_hot(2, 0), item.text: _one_hot(2, 0)}
    )
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 3},
        floors_by_source={MemorySource.ME: _floors()},
    )
    out = retriever.retrieve(query, [item], frozenset({MemorySource.ME}))
    assert [x.memory_id for x in out] == ["mem_aaaaaaaaaaaa"]


def test_all_tied_raw_scores_break_ties_deterministically():
    query = "q"
    # Every candidate has identical text (so identical lexical AND semantic
    # raw scores); only created_session/memory_id differ. The documented
    # tie-break is (score, created_session, memory_id) descending.
    vector = _one_hot(1, 0)
    items = [
        _item("mem_bbbbbbbbbbbb", MemorySource.ME, 2, "same text"),
        _item("mem_aaaaaaaaaaaa", MemorySource.ME, 2, "same text"),
        _item("mem_cccccccccccc", MemorySource.ME, 5, "same text"),
    ]
    encoder = _FakeEncoder({query: vector, "same text": vector})
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 3},
        floors_by_source={MemorySource.ME: _floors()},
    )
    out = retriever.retrieve(query, items, frozenset({MemorySource.ME}))
    # created_session=5 wins outright; the session=2 pair is broken by
    # memory_id descending ("mem_bbbbbbbbbbbb" > "mem_aaaaaaaaaaaa").
    assert [x.memory_id for x in out] == [
        "mem_cccccccccccc",
        "mem_bbbbbbbbbbbb",
        "mem_aaaaaaaaaaaa",
    ]


def test_floor_excludes_only_when_both_scorers_reject_a_candidate():
    query = "alpha beta gamma"
    # lexical_hit: shares 3/4 tokens with the query (lexical ~0.87) but is
    # mapped to a semantic vector orthogonal to the query's (semantic 0).
    # semantic_hit: shares zero tokens with the query (lexical 0) but is
    # mapped to the *same* semantic vector as the query (semantic 1.0).
    # neither: zero token overlap and an orthogonal semantic vector --
    # must be excluded since both raw scores are at or below their floor.
    encoder = _FakeEncoder(
        {
            "alpha beta gamma": _one_hot(3, 0),
            "alpha beta gamma delta": _one_hot(3, 1),
            "totally different words here": _one_hot(3, 0),
            "nothing shared whatsoever": _one_hot(3, 2),
        }
    )
    lexical_hit = _item(
        "mem_aaaaaaaaaaaa", MemorySource.ME, 1, "alpha beta gamma delta"
    )
    semantic_hit = _item(
        "mem_bbbbbbbbbbbb", MemorySource.ME, 1, "totally different words here"
    )
    neither = _item(
        "mem_cccccccccccc", MemorySource.ME, 1, "nothing shared whatsoever"
    )
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 10},
        floors_by_source={
            MemorySource.ME: SourceScoreFloors(
                lexical_min_score=0.5, semantic_min_score=0.5
            )
        },
    )
    out = retriever.retrieve(
        query, [lexical_hit, semantic_hit, neither], frozenset({MemorySource.ME})
    )
    ids = {x.memory_id for x in out}
    assert ids == {"mem_aaaaaaaaaaaa", "mem_bbbbbbbbbbbb"}
    assert "mem_cccccccccccc" not in ids


def test_floor_never_excludes_a_candidate_scoring_exactly_at_the_floor():
    # A floor calibrated as "the minimum score among known-good examples"
    # (hybrid_retrieval_diagnostics.calibrate_source_floors) must never
    # exclude the example that defines it -- that example's own score sits
    # AT the floor, not below it. This candidate is constructed to score
    # exactly 0.5 on BOTH scorers against a floor of (0.5, 0.5): under the
    # old `<=` exclusion rule this would have been wrongly dropped; the
    # comparison must be strict (`<`) so it survives.
    query = "a b"
    at_floor_text = "a c"
    # lexical_score("a b", "a c") == 1 shared token / (sqrt(2)*sqrt(2)) == 0.5.
    assert lexical_score(query, at_floor_text) == pytest.approx(0.5)
    query_vector = np.array([1.0, 0.0])
    candidate_vector = np.array([0.5, (3 ** 0.5) / 2])  # unit vector, dot == 0.5
    assert float(query_vector @ candidate_vector) == pytest.approx(0.5)
    encoder = _FakeEncoder({query: query_vector, at_floor_text: candidate_vector})
    item = _item("mem_aaaaaaaaaaaa", MemorySource.ME, 1, at_floor_text)
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 10},
        floors_by_source={
            MemorySource.ME: SourceScoreFloors(
                lexical_min_score=0.5, semantic_min_score=0.5
            )
        },
    )
    out = retriever.retrieve(query, [item], frozenset({MemorySource.ME}))
    assert [x.memory_id for x in out] == ["mem_aaaaaaaaaaaa"]


def test_top_k_by_source_is_respected_per_source():
    query = "q"
    vector = _one_hot(1, 0)
    me_items = [
        _item(f"mem_{i:012x}", MemorySource.ME, i, "q") for i in range(5)
    ]
    mp_items = [_item("mem_ffffffffffff", MemorySource.MP, 1, "q")]
    encoder = _FakeEncoder({"q": vector})
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 2, MemorySource.MP: 1},
        floors_by_source={
            MemorySource.ME: _floors(),
            MemorySource.MP: _floors(),
        },
    )
    out = retriever.retrieve(
        query, me_items + mp_items, frozenset({MemorySource.ME, MemorySource.MP})
    )
    assert sum(1 for x in out if x.source is MemorySource.ME) == 2
    assert sum(1 for x in out if x.source is MemorySource.MP) == 1


def test_unselected_source_never_appears_in_output():
    query = "q"
    vector = _one_hot(1, 0)
    items = [_item("mem_aaaaaaaaaaaa", MemorySource.MS, 1, "q")]
    encoder = _FakeEncoder({"q": vector})
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.MS: 5},
        floors_by_source={MemorySource.MS: _floors()},
    )
    out = retriever.retrieve(query, items, frozenset({MemorySource.ME}))
    assert out == []


def test_rank_all_combines_sources_into_one_pool_not_per_source_top_k():
    query = "q"
    vector = _one_hot(1, 0)
    # top_k_by_source is deliberately small (1 each) -- rank_all must
    # ignore it entirely and return every surviving candidate across both
    # sources in one combined order, unlike retrieve()'s per-source slice.
    me_items = [_item(f"mem_{i:012x}", MemorySource.ME, i, "q") for i in range(3)]
    mp_items = [_item("mem_ffffffffffff", MemorySource.MP, 9, "q")]
    encoder = _FakeEncoder({"q": vector})
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 1, MemorySource.MP: 1},
        floors_by_source={
            MemorySource.ME: _floors(),
            MemorySource.MP: _floors(),
        },
    )
    out = retriever.rank_all(
        query, me_items + mp_items, frozenset({MemorySource.ME, MemorySource.MP})
    )
    assert len(out) == 4
    # All 4 candidates tie on raw score; combined ranking still uses one
    # global (score, created_session, memory_id) order across sources --
    # the MP item (session=9) outranks every ME item (session 0-2).
    assert out[0].memory_id == "mem_ffffffffffff"


def test_rank_all_still_applies_each_items_own_source_floor():
    query = "alpha beta gamma"
    encoder = _FakeEncoder(
        {
            "alpha beta gamma": _one_hot(2, 0),
            "alpha beta gamma delta": _one_hot(2, 1),
            "nothing shared whatsoever": _one_hot(2, 1),
        }
    )
    survives = _item(
        "mem_aaaaaaaaaaaa", MemorySource.ME, 1, "alpha beta gamma delta"
    )
    excluded = _item(
        "mem_bbbbbbbbbbbb", MemorySource.MP, 1, "nothing shared whatsoever"
    )
    retriever = HybridMemoryRetriever(
        semantic_encoder=encoder,
        top_k_by_source={MemorySource.ME: 10, MemorySource.MP: 10},
        floors_by_source={
            MemorySource.ME: _floors(0.5, 0.5),
            MemorySource.MP: _floors(0.5, 0.5),
        },
    )
    out = retriever.rank_all(
        query, [survives, excluded], frozenset({MemorySource.ME, MemorySource.MP})
    )
    assert [x.memory_id for x in out] == ["mem_aaaaaaaaaaaa"]


def test_hybrid_strategy_retriever_retrieve_is_rank_all_sliced_to_top_k():
    query = "q"
    vector = _one_hot(1, 0)
    cards = [
        StrategyCard(
            strategy_id=f"strat_{i:012x}",
            strategy_label="reflect",
            retrieval_text="q",
            guidance_text="g",
            example_response="e",
            source_dialogue_id="d1",
            source_turn_index=1,
        )
        for i in range(5)
    ]
    encoder = _FakeEncoder({"q": vector})
    retriever = HybridStrategyRetriever(
        cards, semantic_encoder=encoder, top_k=2, floors=_floors()
    )
    full = retriever.rank_all(query)
    assert len(full) == 5
    assert retriever.retrieve(query) == full[:2]


def test_hybrid_strategy_retriever_ranks_and_respects_floor():
    query = "q"
    encoder = _FakeEncoder(
        {
            "q": _one_hot(2, 0),
            "matches": _one_hot(2, 0),
            "unrelated": _one_hot(2, 1),
        }
    )
    good = StrategyCard(
        strategy_id="strat_aaaaaaaaaaaa",
        strategy_label="reflect",
        retrieval_text="matches",
        guidance_text="g",
        example_response="e",
        source_dialogue_id="d1",
        source_turn_index=1,
    )
    bad = StrategyCard(
        strategy_id="strat_bbbbbbbbbbbb",
        strategy_label="reflect",
        retrieval_text="unrelated",
        guidance_text="g",
        example_response="e",
        source_dialogue_id="d1",
        source_turn_index=2,
    )
    retriever = HybridStrategyRetriever(
        [good, bad],
        semantic_encoder=encoder,
        top_k=5,
        floors=SourceScoreFloors(lexical_min_score=0.5, semantic_min_score=0.5),
    )
    out = retriever.retrieve(query)
    assert [card.strategy_id for card in out] == ["strat_aaaaaaaaaaaa"]


def test_hybrid_strategy_retriever_empty_cards_returns_empty():
    encoder = _FakeEncoder({})
    retriever = HybridStrategyRetriever(
        [],
        semantic_encoder=encoder,
        floors=_floors(),
    )
    assert retriever.retrieve("anything") == []


class _CountingEncoder:
    """Wraps _FakeEncoder and counts every encode() call, to prove card
    embeddings are computed once at construction, not re-embedded on every
    retrieve()/rank_all() call -- the real Strategy Bank has 11k+ cards, so
    re-encoding it per call would be an unacceptable production cost."""

    def __init__(self, vocab: dict[str, np.ndarray]):
        self._inner = _FakeEncoder(vocab)
        self.call_count = 0
        self.texts_seen: list[tuple[str, ...]] = []

    def encode(self, texts):
        self.call_count += 1
        self.texts_seen.append(tuple(texts))
        return self._inner.encode(texts)


def test_hybrid_strategy_retriever_encodes_the_bank_once_not_per_call():
    vector = _one_hot(1, 0)
    cards = [
        StrategyCard(
            strategy_id=f"strat_{i:012x}",
            strategy_label="reflect",
            retrieval_text="card text",
            guidance_text="g",
            example_response="e",
            source_dialogue_id="d1",
            source_turn_index=1,
        )
        for i in range(3)
    ]
    encoder = _CountingEncoder({"q": vector, "card text": vector})
    retriever = HybridStrategyRetriever(
        cards, semantic_encoder=encoder, top_k=2, floors=_floors()
    )
    # Construction alone should have already embedded every card's text.
    assert encoder.call_count == 1
    assert encoder.texts_seen[0] == ("card text", "card text", "card text")

    retriever.retrieve("q")
    retriever.retrieve("q")
    retriever.rank_all("q")
    # Only the (single-text) query is encoded per call thereafter -- the
    # card bank itself is never re-encoded.
    assert encoder.call_count == 4
    assert all(len(texts) == 1 for texts in encoder.texts_seen[1:])


def test_hybrid_strategy_retriever_confidence_matches_max_lexical_score():
    encoder = _FakeEncoder({"q": _one_hot(1, 0), "matches q well": _one_hot(1, 0)})
    card = StrategyCard(
        strategy_id="strat_aaaaaaaaaaaa",
        strategy_label="reflect",
        retrieval_text="matches q well",
        guidance_text="g",
        example_response="e",
        source_dialogue_id="d1",
        source_turn_index=1,
    )
    retriever = HybridStrategyRetriever(
        [card], semantic_encoder=encoder, floors=_floors()
    )
    assert retriever.confidence("q") == pytest.approx(lexical_score("q", "matches q well"))


def test_hybrid_strategy_retriever_confidence_is_zero_for_empty_bank():
    encoder = _FakeEncoder({})
    retriever = HybridStrategyRetriever([], semantic_encoder=encoder, floors=_floors())
    assert retriever.confidence("anything") == 0.0


def test_retrieve_fixed_token_budget_skips_overflow_and_continues():
    # "medium" alone would fit (~2 tokens), "big" would overflow the
    # budget on its own (~10 tokens) and must be skipped, not stop the
    # loop -- "small" after it still gets included.
    ranked = ["medium", "big", "small"]
    texts = {"medium": "ab cd", "big": "x" * 40, "small": "e"}
    out = retrieve_fixed_token_budget(
        ranked, token_budget=3, text_of=lambda item: texts[item]
    )
    assert out == ["medium", "small"]


def test_retrieve_fixed_token_budget_never_truncates_never_stops_early():
    ranked = ["a", "b", "c"]
    texts = {"a": "x" * 400, "b": "y", "c": "z"}
    out = retrieve_fixed_token_budget(
        ranked, token_budget=2, text_of=lambda item: texts[item]
    )
    # "a" alone (100 tokens) overflows a budget of 2 and is skipped, not
    # truncated; "b" and "c" (1 token each) both still fit afterward.
    assert out == ["b", "c"]


def test_observable_query_matches_context_query_and_hash_is_reproducible():
    current_user_text = "I feel stressed."
    history = [{"role": "seeker", "content": "hi"}]
    summary = "prior context"
    query_text, query_hash = observable_query_with_hash(
        current_user_text, history, summary
    )
    assert query_text == context_query(current_user_text, history, summary)
    assert query_hash == sha256_text(query_text)


def test_contract_hash_changes_with_floors_rrf_k_and_encoder_binding():
    base = dict(
        memory_floors_by_source={MemorySource.ME: _floors(0.1, 0.2)},
        strategy_floors=_floors(0.3, 0.4),
        rrf_k=RECIPROCAL_RANK_FUSION_K,
        semantic_encoder_spec_sha256="a" * 64,
    )
    baseline = hybrid_retrieval_contract_hash(**base)

    changed_floor = dict(base)
    changed_floor["memory_floors_by_source"] = {MemorySource.ME: _floors(0.9, 0.2)}
    assert hybrid_retrieval_contract_hash(**changed_floor) != baseline

    changed_k = dict(base)
    changed_k["rrf_k"] = RECIPROCAL_RANK_FUSION_K + 1
    assert hybrid_retrieval_contract_hash(**changed_k) != baseline

    changed_encoder = dict(base)
    changed_encoder["semantic_encoder_spec_sha256"] = "b" * 64
    assert hybrid_retrieval_contract_hash(**changed_encoder) != baseline

    assert hybrid_retrieval_contract_hash(**base) == baseline


def test_protocol_tag_is_stable_string():
    assert HYBRID_RETRIEVAL_PROTOCOL == "pm-v1.5-hybrid-retrieval-diagnostic-v1"


def test_batched_encode_chunks_large_collections_and_preserves_order():
    # A large text collection (e.g. the real ~11.6k-card Strategy Bank) must
    # never be handed to encoder.encode() in one unbatched call -- that
    # already caused a real multi-hour, tens-of-GB blowup in practice.
    texts = [f"text {i}" for i in range(10)]
    vocab = {text: _one_hot(10, i) for i, text in enumerate(texts)}
    encoder = _CountingEncoder(vocab)
    result = batched_encode(encoder, texts, batch_size=3)
    assert encoder.call_count == 4  # ceil(10 / 3)
    assert all(len(batch) <= 3 for batch in encoder.texts_seen)
    for i, text in enumerate(texts):
        assert np.array_equal(result[i], vocab[text])


def test_batched_encode_rejects_empty_input():
    encoder = _FakeEncoder({})
    with pytest.raises(ValueError):
        batched_encode(encoder, [])
