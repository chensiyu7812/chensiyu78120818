from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .contracts import (
    EvidenceFilterDecision,
    EvidenceItemDecision,
    MemoryItem,
    MemorySource,
    StrategyCard,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
)
from .io import canonical_json, sha256_text
from .text import estimate_tokens, lexical_score, normalize_for_hash


EVIDENCE_FILTER_PROTOCOL = "pm-v2-evidence-filter-v1"


@dataclass(frozen=True)
class EvidenceRule:
    minimum_current_score: float
    minimum_context_score: float
    maximum_items: int
    long_item_token_threshold: int | None = None
    long_item_minimum_current_score: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_current_score", self.minimum_current_score),
            ("minimum_context_score", self.minimum_context_score),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.maximum_items < 0:
            raise ValueError("maximum_items must be nonnegative")
        if (self.long_item_token_threshold is None) != (
            self.long_item_minimum_current_score is None
        ):
            raise ValueError("long-item threshold and score must be set together")
        if self.long_item_token_threshold is not None:
            if self.long_item_token_threshold < 1:
                raise ValueError("long_item_token_threshold must be positive")
            score = float(self.long_item_minimum_current_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("long-item minimum score must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceRule":
        expected = {
            "minimum_current_score",
            "minimum_context_score",
            "maximum_items",
            "long_item_token_threshold",
            "long_item_minimum_current_score",
        }
        if set(value) != expected:
            raise ValueError(
                "evidence rule keys do not match the frozen contract: "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        return cls(
            minimum_current_score=float(value["minimum_current_score"]),
            minimum_context_score=float(value["minimum_context_score"]),
            maximum_items=int(value["maximum_items"]),
            long_item_token_threshold=(
                None
                if value["long_item_token_threshold"] is None
                else int(value["long_item_token_threshold"])
            ),
            long_item_minimum_current_score=(
                None
                if value["long_item_minimum_current_score"] is None
                else float(value["long_item_minimum_current_score"])
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "minimum_current_score": self.minimum_current_score,
            "minimum_context_score": self.minimum_context_score,
            "maximum_items": self.maximum_items,
            "long_item_token_threshold": self.long_item_token_threshold,
            "long_item_minimum_current_score": self.long_item_minimum_current_score,
        }


@dataclass(frozen=True)
class EvidenceFilterConfig:
    protocol: str
    enabled: bool
    candidate_scope: str
    memory_rules: Mapping[MemorySource, EvidenceRule]
    strategy_rule: EvidenceRule
    maximum_total_memory_items: int
    drop_exact_duplicate_text: bool
    allow_empty_memory: bool
    allow_empty_strategy: bool

    def __post_init__(self) -> None:
        if self.protocol != EVIDENCE_FILTER_PROTOCOL:
            raise ValueError("unsupported evidence-filter protocol")
        if self.candidate_scope != "post_retrieval_pre_generation":
            raise ValueError("unsupported evidence-filter candidate scope")
        if set(self.memory_rules) != set(MemorySource):
            raise ValueError("memory_rules must cover MP, MS, and ME exactly")
        if self.maximum_total_memory_items < 0:
            raise ValueError("maximum_total_memory_items must be nonnegative")
        if not self.allow_empty_memory or not self.allow_empty_strategy:
            raise ValueError(
                "V2 filter must allow empty evidence so abstention is not defeated"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceFilterConfig":
        expected = {
            "protocol",
            "enabled",
            "candidate_scope",
            "memory_rules",
            "strategy_rule",
            "maximum_total_memory_items",
            "drop_exact_duplicate_text",
            "allow_empty_memory",
            "allow_empty_strategy",
        }
        if set(value) != expected:
            raise ValueError(
                "evidence_filter keys do not match the frozen contract: "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        raw_rules = value["memory_rules"]
        if not isinstance(raw_rules, Mapping) or set(raw_rules) != {
            source.value for source in MemorySource
        }:
            raise ValueError("memory_rules must contain MP, MS, and ME exactly")
        return cls(
            protocol=str(value["protocol"]),
            enabled=bool(value["enabled"]),
            candidate_scope=str(value["candidate_scope"]),
            memory_rules={
                source: EvidenceRule.from_mapping(raw_rules[source.value])
                for source in MemorySource
            },
            strategy_rule=EvidenceRule.from_mapping(value["strategy_rule"]),
            maximum_total_memory_items=int(value["maximum_total_memory_items"]),
            drop_exact_duplicate_text=bool(value["drop_exact_duplicate_text"]),
            allow_empty_memory=bool(value["allow_empty_memory"]),
            allow_empty_strategy=bool(value["allow_empty_strategy"]),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "enabled": self.enabled,
            "candidate_scope": self.candidate_scope,
            "memory_rules": {
                source.value: self.memory_rules[source].payload()
                for source in MemorySource
            },
            "strategy_rule": self.strategy_rule.payload(),
            "maximum_total_memory_items": self.maximum_total_memory_items,
            "drop_exact_duplicate_text": self.drop_exact_duplicate_text,
            "allow_empty_memory": self.allow_empty_memory,
            "allow_empty_strategy": self.allow_empty_strategy,
        }

    def digest(self) -> str:
        return sha256_text(canonical_json(self.payload()))


@dataclass(frozen=True)
class FilteredEvidence:
    memory_view: list[MemoryItem]
    strategy_view: list[StrategyCard]
    decision: EvidenceFilterDecision


def _score(
    current_user_text: str,
    context_query_text: str,
    document: str,
) -> tuple[float, float, float]:
    current = lexical_score(current_user_text, document)
    context = lexical_score(context_query_text, document)
    return current, context, max(current, context)


def _eligible(
    *,
    current: float,
    context: float,
    estimated_tokens: int,
    rule: EvidenceRule,
) -> tuple[bool, str]:
    if not (
        current >= rule.minimum_current_score
        or context >= rule.minimum_context_score
    ):
        return False, "below_relevance_threshold"
    if (
        rule.long_item_token_threshold is not None
        and estimated_tokens > rule.long_item_token_threshold
        and current < float(rule.long_item_minimum_current_score)
    ):
        return False, "long_item_requires_high_current_relevance"
    return True, "relevance_gate_passed"


def filter_evidence(
    *,
    requested_action_id: str,
    current_user_text: str,
    context_query_text: str,
    memory_candidates: Sequence[MemoryItem],
    strategy_candidates: Sequence[StrategyCard],
    config: EvidenceFilterConfig,
    session_index: int = 1,
    memory_helpfulness_model: Any | None = None,
) -> FilteredEvidence:
    requested_sources, requested_strategy = parse_action_id(requested_action_id)
    if any(item.source not in requested_sources for item in memory_candidates):
        raise ValueError("memory candidate source is outside the requested action")
    if requested_strategy is StrategyMode.R0 and strategy_candidates:
        raise ValueError("R0 action cannot have strategy candidates")

    memory_rows: list[dict[str, Any]] = []
    for index, item in enumerate(memory_candidates):
        tokens = estimate_tokens(item.text)
        current, context, selector = _score(
            current_user_text, context_query_text, item.text
        )
        eligible, reason = _eligible(
            current=current,
            context=context,
            estimated_tokens=tokens,
            rule=config.memory_rules[item.source],
        )
        helpfulness_score = None
        if memory_helpfulness_model is not None:
            helpfulness_score = float(
                memory_helpfulness_model.predict_helpfulness(
                    current_user_text=current_user_text,
                    context_query_text=context_query_text,
                    item=item,
                    session_index=int(session_index),
                )
            )
            eligible = helpfulness_score >= float(memory_helpfulness_model.threshold)
            reason = (
                "supervised_helpfulness_gate_passed"
                if eligible
                else "below_supervised_helpfulness_threshold"
            )
        memory_rows.append(
            {
                "index": index,
                "item": item,
                "current": current,
                "context": context,
                "selector": (
                    helpfulness_score
                    if helpfulness_score is not None
                    else selector
                ),
                "tokens": tokens,
                "helpfulness_score": helpfulness_score,
                "eligible": eligible,
                "reason": reason,
                "text_key": normalize_for_hash(item.text),
                "keep": False,
            }
        )

    strategy_rows: list[dict[str, Any]] = []
    for index, card in enumerate(strategy_candidates):
        document = card.retrieval_text
        tokens = estimate_tokens(card.guidance_text + "\n" + card.example_response)
        current, context, selector = _score(
            current_user_text, context_query_text, document
        )
        eligible, reason = _eligible(
            current=current,
            context=context,
            estimated_tokens=tokens,
            rule=config.strategy_rule,
        )
        strategy_rows.append(
            {
                "index": index,
                "item": card,
                "current": current,
                "context": context,
                "selector": selector,
                "tokens": tokens,
                "eligible": eligible,
                "reason": reason,
                "text_key": normalize_for_hash(
                    card.retrieval_text + "\n" + card.guidance_text
                ),
                "keep": False,
            }
        )

    if not config.enabled:
        for row in memory_rows + strategy_rows:
            row["keep"] = True
            row["reason"] = "filter_disabled_keep_candidate"
    else:
        ranked_memory = sorted(
            (row for row in memory_rows if row["eligible"]),
            key=lambda row: (
                -float(row["selector"]),
                int(row["tokens"]),
                row["item"].memory_id,
            ),
        )
        kept_memory_rows: list[dict[str, Any]] = []
        source_counts = {source: 0 for source in MemorySource}
        seen_memory_text: set[str] = set()
        for row in ranked_memory:
            source = row["item"].source
            if (
                config.drop_exact_duplicate_text
                and row["text_key"]
                and row["text_key"] in seen_memory_text
            ):
                row["reason"] = "exact_duplicate_evidence"
                continue
            if source_counts[source] >= config.memory_rules[source].maximum_items:
                row["reason"] = "source_item_cap"
                continue
            if len(kept_memory_rows) >= config.maximum_total_memory_items:
                row["reason"] = "total_memory_item_cap"
                continue
            row["keep"] = True
            row["reason"] = (
                "kept_supervised_helpful_candidate"
                if memory_helpfulness_model is not None
                else "kept_relevant_candidate"
            )
            kept_memory_rows.append(row)
            source_counts[source] += 1
            if row["text_key"]:
                seen_memory_text.add(row["text_key"])

        ranked_strategy = sorted(
            (row for row in strategy_rows if row["eligible"]),
            key=lambda row: (
                -float(row["selector"]),
                int(row["tokens"]),
                row["item"].strategy_id,
            ),
        )
        seen_strategy_text: set[str] = set()
        kept_strategy_count = 0
        for row in ranked_strategy:
            if (
                config.drop_exact_duplicate_text
                and row["text_key"]
                and row["text_key"] in seen_strategy_text
            ):
                row["reason"] = "exact_duplicate_evidence"
                continue
            if kept_strategy_count >= config.strategy_rule.maximum_items:
                row["reason"] = "strategy_item_cap"
                continue
            row["keep"] = True
            row["reason"] = "kept_relevant_candidate"
            kept_strategy_count += 1
            if row["text_key"]:
                seen_strategy_text.add(row["text_key"])

    memory_view = [row["item"] for row in memory_rows if row["keep"]]
    strategy_view = [row["item"] for row in strategy_rows if row["keep"]]
    effective_action_id = canonical_action_id(
        {item.source for item in memory_view},
        StrategyMode.RS if strategy_view else StrategyMode.R0,
    )

    item_decisions = [
        EvidenceItemDecision(
            evidence_type="memory",
            item_id=row["item"].memory_id,
            source=row["item"].source.value,
            current_turn_score=round(float(row["current"]), 8),
            context_score=round(float(row["context"]), 8),
            selector_score=round(float(row["selector"]), 8),
            helpfulness_score=(
                round(float(row["helpfulness_score"]), 8)
                if row["helpfulness_score"] is not None
                else None
            ),
            estimated_tokens=int(row["tokens"]),
            keep=bool(row["keep"]),
            reason=str(row["reason"]),
        )
        for row in memory_rows
    ] + [
        EvidenceItemDecision(
            evidence_type="strategy",
            item_id=row["item"].strategy_id,
            source="RS",
            current_turn_score=round(float(row["current"]), 8),
            context_score=round(float(row["context"]), 8),
            selector_score=round(float(row["selector"]), 8),
            helpfulness_score=None,
            estimated_tokens=int(row["tokens"]),
            keep=bool(row["keep"]),
            reason=str(row["reason"]),
        )
        for row in strategy_rows
    ]
    decision = EvidenceFilterDecision(
        protocol=config.protocol,
        enabled=config.enabled,
        config_sha256=config.digest(),
        memory_filter_mode=(
            "supervised_helpfulness"
            if memory_helpfulness_model is not None
            else "lexical_bootstrap"
        ),
        strategy_filter_mode="lexical_contextual_fail_safe",
        memory_filter_checkpoint_sha256=(
            str(memory_helpfulness_model.checkpoint_sha256)
            if memory_helpfulness_model is not None
            else None
        ),
        requested_action_id=requested_action_id,
        effective_action_id=effective_action_id,
        candidate_memory_ids=[item.memory_id for item in memory_candidates],
        kept_memory_ids=[item.memory_id for item in memory_view],
        dropped_memory_ids=[
            item.memory_id for item in memory_candidates if item not in memory_view
        ],
        candidate_strategy_ids=[item.strategy_id for item in strategy_candidates],
        kept_strategy_ids=[item.strategy_id for item in strategy_view],
        dropped_strategy_ids=[
            item.strategy_id for item in strategy_candidates if item not in strategy_view
        ],
        candidate_memory_tokens=sum(int(row["tokens"]) for row in memory_rows),
        kept_memory_tokens=sum(
            int(row["tokens"]) for row in memory_rows if row["keep"]
        ),
        candidate_strategy_tokens=sum(int(row["tokens"]) for row in strategy_rows),
        kept_strategy_tokens=sum(
            int(row["tokens"]) for row in strategy_rows if row["keep"]
        ),
        item_decisions=item_decisions,
    )
    return FilteredEvidence(memory_view, strategy_view, decision)
