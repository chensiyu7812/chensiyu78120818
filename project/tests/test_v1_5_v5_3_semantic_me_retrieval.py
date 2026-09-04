import numpy as np
import pytest

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.v1_5_v5_3_semantic_me_retrieval import (
    rank_me_candidates_bge,
    rank_me_candidates_hybrid,
)


class FakeEncoder:
    """Deterministic keyword-match encoder -- same pattern as the MS test's
    FakeEncoder, dependency-free, exercises real ranking/tie-break logic."""

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


def me_item(memory_id: str, text: str, created_session: int = 0) -> MemoryItem:
    return MemoryItem(memory_id=memory_id, source=MemorySource.ME, created_session=created_session, text=text)


def ms_item(memory_id: str) -> MemoryItem:
    return MemoryItem(memory_id=memory_id, source=MemorySource.MS, created_session=0, text="an MS item")


VALID_TARGET_TEXT = "When dealing with the job transition, I wrote down my top three concerns, and it helped me feel steadier."
NO_RESULT_TEXT = "When dealing with the job transition, I tried journaling about it, but I'm not sure yet whether it changed anything."
CONTEXT_ONLY_TEXT = "The job transition has been going on for a few months now."


def test_bge_ranks_topically_matching_candidate_first_no_prefilter():
    encoder = FakeEncoder(["job", "garden"])
    items = [
        me_item("mem_000000000001", VALID_TARGET_TEXT),
        me_item("mem_000000000002", "When dealing with the garden project, I started composting, and it helped."),
    ]
    ranked = rank_me_candidates_bge("I have a meeting about the job transition.", items, encoder=encoder)
    assert ranked[0].item.memory_id == "mem_000000000001"


def test_bge_does_not_filter_non_compiler_valid_items():
    # Pure BGE has no compiler/prior-session filter -- a context-only item
    # can still rank first if it's the lexically closest match. This is the
    # expected, documented behavior difference from the hybrid method.
    encoder = FakeEncoder(["job"])
    items = [
        me_item("mem_000000000001", CONTEXT_ONLY_TEXT),
        me_item("mem_000000000002", "Totally unrelated garden composting text here."),
    ]
    ranked = rank_me_candidates_bge("job transition again", items, encoder=encoder)
    assert ranked[0].item.memory_id == "mem_000000000001"


def test_bge_rejects_non_me_items():
    encoder = FakeEncoder(["job"])
    with pytest.raises(ValueError, match="only accepts MemorySource.ME"):
        rank_me_candidates_bge("query", [ms_item("mem_000000000001")], encoder=encoder)


def test_hybrid_filters_out_non_compiler_valid_before_ranking():
    encoder = FakeEncoder(["job"])
    items = [
        me_item("mem_000000000001", CONTEXT_ONLY_TEXT, created_session=1),  # not compiler-valid
        me_item("mem_000000000002", VALID_TARGET_TEXT, created_session=1),  # compiler-valid
    ]
    ranked = rank_me_candidates_hybrid("job transition", items, encoder=encoder, session_index=5)
    assert [r.item.memory_id for r in ranked] == ["mem_000000000002"]


def test_hybrid_filters_out_future_or_current_session_items():
    encoder = FakeEncoder(["job"])
    items = [
        me_item("mem_000000000001", VALID_TARGET_TEXT, created_session=5),  # session_index=5, not strictly prior
        me_item("mem_000000000002", NO_RESULT_TEXT.replace("job transition", "job transition also"), created_session=1),
    ]
    ranked = rank_me_candidates_hybrid("job transition", items, encoder=encoder, session_index=5)
    # mem_000000000001 excluded (not strictly prior); mem_000000000002 excluded (not compiler-valid, no result)
    assert ranked == []


def test_hybrid_empty_pool_when_nothing_survives_stage_one():
    encoder = FakeEncoder(["job"])
    items = [me_item("mem_000000000001", CONTEXT_ONLY_TEXT, created_session=1)]
    ranked = rank_me_candidates_hybrid("job transition", items, encoder=encoder, session_index=5)
    assert ranked == []


def test_hybrid_rejects_non_me_items():
    encoder = FakeEncoder(["job"])
    with pytest.raises(ValueError, match="only accepts MemorySource.ME"):
        rank_me_candidates_hybrid("query", [ms_item("mem_000000000001")], encoder=encoder, session_index=5)


def test_hybrid_tie_break_is_created_session_desc_then_memory_id_desc():
    # Both survive stage 1 and tie on BGE score (identical vocabulary hits)
    encoder = FakeEncoder(["job"])
    items = [
        me_item("mem_000000000001", VALID_TARGET_TEXT, created_session=1),
        me_item("mem_000000000002", VALID_TARGET_TEXT, created_session=3),
    ]
    ranked = rank_me_candidates_hybrid("job transition", items, encoder=encoder, session_index=5)
    assert ranked[0].item.memory_id == "mem_000000000002"  # higher created_session wins tie
