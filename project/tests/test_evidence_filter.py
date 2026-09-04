from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.config import load_config
from metacom_pm.contracts import (
    ActionOutcome,
    CostRecord,
    MemoryItem,
    MemorySource,
    StrategyCard,
)
from metacom_pm.evidence_filter import EvidenceFilterConfig, filter_evidence


ROOT = Path(__file__).resolve().parents[1]


def _config() -> EvidenceFilterConfig:
    raw = load_config(ROOT / "configs" / "pm_v2.yaml")
    return EvidenceFilterConfig.from_mapping(raw["evidence_filter"])


def _memory(hex_id: str, source: MemorySource, text: str) -> MemoryItem:
    return MemoryItem(
        memory_id=f"mem_{hex_id * 12}",
        source=source,
        created_session=1,
        timestamp=None,
        text=text,
    )


def _strategy(hex_id: str, text: str) -> StrategyCard:
    return StrategyCard(
        strategy_id=f"strat_{hex_id * 12}",
        strategy_label="reflect",
        retrieval_text=text,
        guidance_text="Reflect the user's exact feeling before advice.",
        example_response="It sounds like this has been exhausting.",
        source_dialogue_id="dialogue",
        source_turn_index=1,
    )


def test_filter_can_drop_every_requested_item_without_relabeling_pm_abstention() -> None:
    result = filter_evidence(
        requested_action_id="MPMSME+RS",
        current_user_text="I am nervous about tomorrow's interview.",
        context_query_text="We have been discussing tomorrow's job interview.",
        memory_candidates=[
            _memory("a", MemorySource.MP, "The user likes jazz records."),
            _memory("b", MemorySource.MS, "They recently bought a bicycle."),
            _memory("c", MemorySource.ME, "They visited a beach five years ago."),
        ],
        strategy_candidates=[_strategy("d", "celebrate a completed achievement")],
        config=_config(),
    )

    assert result.memory_view == []
    assert result.strategy_view == []
    assert result.decision.requested_action_id == "MPMSME+RS"
    assert result.decision.effective_action_id == "M0+R0"
    assert len(result.decision.dropped_memory_ids) == 3
    assert len(result.decision.dropped_strategy_ids) == 1


def test_filter_keeps_relevant_items_and_enforces_caps_and_exact_deduplication() -> None:
    duplicate = "interview interview nervous tomorrow"
    result = filter_evidence(
        requested_action_id="MPMSME+RS",
        current_user_text=duplicate,
        context_query_text=duplicate,
        memory_candidates=[
            _memory("1", MemorySource.MP, duplicate),
            _memory("2", MemorySource.MS, duplicate),
            _memory("3", MemorySource.ME, "interview nervous tomorrow"),
            _memory("4", MemorySource.ME, "interview nervous tomorrow preparation"),
        ],
        strategy_candidates=[
            _strategy("5", duplicate),
            _strategy("6", duplicate),
            _strategy("7", "interview nervous tomorrow reflect emotion"),
        ],
        config=_config(),
    )

    assert len(result.memory_view) <= 3
    assert sum(item.source is MemorySource.ME for item in result.memory_view) <= 1
    assert len(result.strategy_view) <= 2
    memory_reasons = {
        row.reason
        for row in result.decision.item_decisions
        if row.evidence_type == "memory" and not row.keep
    }
    strategy_reasons = {
        row.reason
        for row in result.decision.item_decisions
        if row.evidence_type == "strategy" and not row.keep
    }
    assert "exact_duplicate_evidence" in memory_reasons
    assert "exact_duplicate_evidence" in strategy_reasons


def test_filtered_rs_outcome_contract_records_requested_and_effective_actions() -> None:
    memory = _memory("e", MemorySource.MP, "unrelated profile")
    strategy = _strategy("f", "unrelated strategy")
    result = filter_evidence(
        requested_action_id="MP+RS",
        current_user_text="deadline anxiety",
        context_query_text="deadline anxiety",
        memory_candidates=[memory],
        strategy_candidates=[strategy],
        config=_config(),
    )
    outcome = ActionOutcome(
        card_id="card_aaaaaaaaaaaa",
        state_id="state_aaaaaaaaaaaa",
        user_id="u1",
        action_id="MP+RS",
        response="That deadline sounds stressful.",
        selected_memory_ids=[],
        selected_strategy_ids=[],
        memory_view=[],
        strategy_view=[],
        effective_action_id="M0+R0",
        candidate_memory_view=[memory],
        candidate_strategy_view=[strategy],
        evidence_filter_decision=result.decision,
        cost=CostRecord(
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
        model_name="fake",
        prompt_hash="p",
        request_hash="r",
        provenance={},
    )
    assert outcome.action_id == "MP+RS"
    assert outcome.effective_action_id == "M0+R0"


def test_filter_config_is_fail_closed_on_unknown_keys() -> None:
    raw = load_config(ROOT / "configs" / "pm_v2.yaml")["evidence_filter"]
    with pytest.raises(ValueError, match="keys do not match"):
        EvidenceFilterConfig.from_mapping({**raw, "unknown": True})


def test_supervised_filter_uses_item_helpfulness_not_lexical_overlap() -> None:
    class FakeHelpfulnessModel:
        threshold = 0.5
        checkpoint_sha256 = "a" * 64

        @staticmethod
        def predict_helpfulness(*, item: MemoryItem, **_) -> float:
            return 0.9 if "actually useful" in item.text else 0.1

    useful = _memory(
        "8",
        MemorySource.ME,
        "The actually useful detail is that the interview was rescheduled.",
    )
    lexical_distractor = _memory(
        "9",
        MemorySource.ME,
        "interview interview nervous tomorrow interview nervous tomorrow",
    )
    result = filter_evidence(
        requested_action_id="ME+R0",
        current_user_text="I am nervous about tomorrow's interview.",
        context_query_text="We are discussing tomorrow's interview.",
        memory_candidates=[lexical_distractor, useful],
        strategy_candidates=[],
        config=_config(),
        session_index=4,
        memory_helpfulness_model=FakeHelpfulnessModel(),
    )

    assert result.memory_view == [useful]
    assert result.decision.memory_filter_mode == "supervised_helpfulness"
    assert result.decision.memory_filter_checkpoint_sha256 == "a" * 64
    scores = {
        row.item_id: row.helpfulness_score
        for row in result.decision.item_decisions
    }
    assert scores[useful.memory_id] == pytest.approx(0.9)
    assert scores[lexical_distractor.memory_id] == pytest.approx(0.1)
