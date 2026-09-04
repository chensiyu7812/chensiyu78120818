from __future__ import annotations

import pytest
from pydantic import ValidationError

from metacom_pm.action_execution import execute_requested_retrievals
from metacom_pm.contracts import (
    ActionOutcome,
    CostRecord,
    MemoryItem,
    MemorySource,
    RetrievalAttempt,
    StrategyCard,
)
from metacom_pm.retrieval import MemoryRetriever, StrategyRetriever


def _attempt(channel: str, requested: bool) -> RetrievalAttempt:
    return RetrievalAttempt(
        channel=channel,
        requested=requested,
        called=requested,
        status="succeeded" if requested else "not_requested",
        hit_count=0,
        retrieved_tokens=0,
        latency_ms=0.0,
        api_cost_usd=0.0,
    )


def _zero_hit_outcome(**overrides) -> ActionOutcome:
    payload = {
        "card_id": "card_aaaaaaaaaaaa",
        "state_id": "state_aaaaaaaaaaaa",
        "user_id": "u1",
        "action_id": "MP+RS",
        "requested_action_id": "MP+RS",
        "response": "I hear how difficult this feels.",
        "selected_memory_ids": [],
        "selected_strategy_ids": [],
        "memory_view": [],
        "strategy_view": [],
        "retrieval_attempts": [
            _attempt("MP", True),
            _attempt("MS", False),
            _attempt("ME", False),
            _attempt("STRATEGY", True),
        ],
        "cost": CostRecord(
            pm_input_tokens_est=1,
            retrieval_calls=2,
            reranker_calls=0,
            memory_tokens=0,
            strategy_tokens=0,
            base_prompt_tokens=1,
            total_input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
        ),
        "model_name": "fake",
        "prompt_hash": "prompt-content-sha256",
        "request_hash": "request",
        "provenance": {},
    }
    payload.update(overrides)
    return ActionOutcome(**payload)


def test_zero_hit_request_is_legal_and_realized_from_prompt_evidence() -> None:
    outcome = _zero_hit_outcome()

    assert outcome.requested_action_id == "MP+RS"
    assert outcome.realized_action_id == "M0+R0"
    assert outcome.effective_action_id == "M0+R0"
    assert outcome.prompt_equivalence_id == outcome.prompt_hash
    assert outcome.label_lineage_id == outcome.prompt_equivalence_id


def test_requested_action_cannot_be_reported_as_realized_after_zero_hits() -> None:
    with pytest.raises(ValidationError, match="realized_action_id"):
        _zero_hit_outcome(realized_action_id="MP+RS")


def test_execution_records_all_channels_and_zero_hits() -> None:
    memory = MemoryItem(
        memory_id="mem_aaaaaaaaaaaa",
        source=MemorySource.MP,
        created_session=1,
        text="unrelated profile",
    )
    strategy = StrategyCard(
        strategy_id="strat_aaaaaaaaaaaa",
        strategy_label="Reflection of feelings",
        retrieval_text="unrelated reflection",
        guidance_text="Reflect feelings.",
        example_response="That sounds hard.",
        source_dialogue_id="d1",
        source_turn_index=1,
    )
    memory_view, strategy_view, attempts = execute_requested_retrievals(
        requested_action_id="MP+RS",
        query="deadline anxiety",
        memory_items=[memory],
        memory_retriever=MemoryRetriever(
            minimum_score_by_source={source: 0.0 for source in MemorySource}
        ),
        strategy_retriever=StrategyRetriever(
            [strategy], top_k=1, minimum_score=0.0
        ),
    )

    assert memory_view == []
    assert strategy_view == []
    assert [row.channel for row in attempts] == ["MP", "MS", "ME", "STRATEGY"]
    assert {row.channel for row in attempts if row.called} == {"MP", "STRATEGY"}
    assert all(row.hit_count == 0 for row in attempts)
