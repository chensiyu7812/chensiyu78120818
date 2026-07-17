from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import MemoryItem, MemorySource, StrategyCard, parse_action_id
from .io import canonical_json, sha256_text
from .pm_v1_6_contracts import (
    ActionLineage,
    RetrievalAttempt,
    prompt_equivalence_id,
    realized_action_from_evidence,
)


@dataclass(frozen=True)
class RetrievalTelemetry:
    source: str
    call_count: int
    hit_count: int
    retrieved_tokens: int
    latency_ms: float

    def to_contract(self) -> RetrievalAttempt:
        return RetrievalAttempt(
            source=self.source,
            call_count=int(self.call_count),
            hit_count=int(self.hit_count),
            retrieved_tokens=int(self.retrieved_tokens),
            latency_ms=float(self.latency_ms),
        )


def build_action_lineage(
    *,
    requested_action_id: str,
    selected_memory: Sequence[MemoryItem],
    selected_strategy: Sequence[StrategyCard],
    telemetry: Sequence[RetrievalTelemetry],
    messages: Sequence[Mapping[str, Any]],
    prompt_equivalence_class_size: int = 1,
) -> ActionLineage:
    expected_sources, expected_strategy = parse_action_id(requested_action_id)
    expected = {source.value for source in expected_sources}
    if expected_strategy.value == "RS":
        expected.add("RS")
    observed = {row.source for row in telemetry}
    if observed != expected:
        raise RuntimeError(
            "retrieval telemetry differs from requested action: "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    realized = realized_action_from_evidence(
        [item.source for item in selected_memory],
        strategy_card_count=len(selected_strategy),
    )
    return ActionLineage(
        requested_action_id=requested_action_id,
        retrieval_attempts=[row.to_contract() for row in telemetry],
        realized_action_id=realized,
        prompt_equivalence_id=prompt_equivalence_id(messages),
        prompt_equivalence_class_size=int(prompt_equivalence_class_size),
    )


def assign_prompt_equivalence_classes(
    rows: Sequence[Mapping[str, Any]],
    *,
    messages_field: str = "messages",
) -> list[dict[str, Any]]:
    """Attach deterministic alias classes without merging requested-action costs."""

    grouped: dict[str, list[int]] = defaultdict(list)
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        if messages_field not in row:
            raise KeyError(f"row lacks {messages_field}")
        eq_id = prompt_equivalence_id(row[messages_field])
        row["prompt_equivalence_id"] = eq_id
        grouped[eq_id].append(index)
        normalized.append(row)
    for eq_id, indices in grouped.items():
        actions = sorted(str(normalized[index]["requested_action_id"]) for index in indices)
        class_binding = sha256_text(
            canonical_json({"prompt_equivalence_id": eq_id, "requested_actions": actions})
        )
        for index in indices:
            normalized[index]["prompt_equivalence_class_size"] = len(indices)
            normalized[index]["prompt_equivalence_actions"] = actions
            normalized[index]["prompt_equivalence_class_sha256"] = class_binding
            normalized[index]["shared_quality_label_weight"] = 1.0 / len(indices)
    return normalized


def verify_realized_action(
    *,
    logged_realized_action_id: str,
    selected_memory: Sequence[Mapping[str, Any]],
    selected_strategy: Sequence[Mapping[str, Any]],
) -> None:
    sources = []
    for item in selected_memory:
        sources.append(MemorySource(str(item["source"])))
    inferred = realized_action_from_evidence(
        sources,
        strategy_card_count=len(selected_strategy),
    )
    if inferred != logged_realized_action_id:
        raise RuntimeError(
            "logged realized action differs from actual generator evidence: "
            f"logged={logged_realized_action_id}, inferred={inferred}"
        )
