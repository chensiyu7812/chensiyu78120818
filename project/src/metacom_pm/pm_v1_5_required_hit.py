from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import MemorySource, StrategyCard
from .pm_v2_contracts import PMV2State, ResourceNeedRegime
from .retrieval import MemoryRetriever, StrategyRetriever, context_query


REQUIRED_HIT_PROTOCOL = "pm-v1.5-required-hit-pre-outcome-validity-v1"
STRATEGY_CHALLENGE_REGIMES = {
    ResourceNeedRegime.STRATEGY_HELPFUL,
    ResourceNeedRegime.STRATEGY_HARMFUL,
}


def validate_required_hit_preflight(
    *,
    states: Sequence[PMV2State],
    cases_by_state: Mapping[str, Any],
    backends_by_state: Mapping[str, Any],
    strategy_cards: Sequence[StrategyCard],
    strategy_top_k: int,
    memory_min_score: float,
    strategy_min_score: float,
) -> dict[str, Any]:
    """Validate declared positive challenge cells before response outcomes exist."""

    if not strategy_cards:
        raise ValueError("required-hit preflight requires the frozen Strategy Bank")
    memory_retriever = MemoryRetriever(
        minimum_score_by_source={
            source: float(memory_min_score) for source in MemorySource
        }
    )
    strategy_retriever = StrategyRetriever(
        strategy_cards,
        top_k=int(strategy_top_k),
        minimum_score=float(strategy_min_score),
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for state in states:
        case = cases_by_state[state.state_id]
        backend = backends_by_state[state.state_id]
        query = context_query(
            state.current_user_text,
            [turn.model_dump(mode="json") for turn in state.current_session_history],
            state.current_session_summary,
        )
        helpful_texts_by_source = {
            source: {
                generated.text
                for generated in (
                    case.profile_memories
                    + case.summary_memories
                    + case.event_memories
                )
                if generated.source is source
                and generated.item_utility == "helpful"
            }
            for source in MemorySource
        }
        source_hits: dict[str, int] = {}
        source_helpful_hits: dict[str, int] = {}
        for source in case.needed_memory_sources:
            retrieved = memory_retriever.retrieve(
                query, backend.items, frozenset({source})
            )
            retrieved_texts = {item.text for item in retrieved}
            source_hits[source.value] = len(retrieved)
            source_helpful_hits[source.value] = len(
                retrieved_texts & helpful_texts_by_source[source]
            )
            if source_helpful_hits[source.value] < 1:
                failures.append(
                    {
                        "state_id": state.state_id,
                        "regime": case.regime.value,
                        "required_channel": source.value,
                        "reason": "declared_helpful_item_not_retrieved",
                    }
                )
        strategy_required = case.regime in STRATEGY_CHALLENGE_REGIMES
        strategy_hits = (
            len(strategy_retriever.retrieve(query)) if strategy_required else None
        )
        if strategy_required and not strategy_hits:
            failures.append(
                {
                    "state_id": state.state_id,
                    "regime": case.regime.value,
                    "required_channel": "STRATEGY",
                    "reason": "strategy_challenge_has_zero_candidates",
                }
            )
        rows.append(
            {
                "state_id": state.state_id,
                "split": state.split.value,
                "regime": case.regime.value,
                "required_memory_sources": [
                    source.value for source in case.needed_memory_sources
                ],
                "memory_candidate_hits": source_hits,
                "helpful_memory_hits": source_helpful_hits,
                "strategy_challenge": strategy_required,
                "strategy_candidate_hits": strategy_hits,
            }
        )
    report = {
        "protocol": REQUIRED_HIT_PROTOCOL,
        "stage": "after_dataset_construction_before_action_response_generation",
        "outcome_fields_accessed": False,
        "status": "PASS" if not failures else "FAIL",
        "states_checked": len(rows),
        "challenge_states": sum(
            bool(row["required_memory_sources"]) or row["strategy_challenge"]
            for row in rows
        ),
        "memory_min_score": float(memory_min_score),
        "strategy_min_score": float(strategy_min_score),
        "strategy_top_k": int(strategy_top_k),
        "failures": failures,
        "rows": rows,
    }
    if failures:
        raise RuntimeError(
            "required-hit pre-outcome validity gate failed: " + str(failures[:10])
        )
    return report
