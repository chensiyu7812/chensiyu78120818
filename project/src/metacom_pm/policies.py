from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import time
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .contracts import (
    ACTION_MEMORY_MAP,
    MemorySource,
    RuntimeState,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
)
from .retrieval import DEFAULT_MEMORY_TOP_K, StrategyRetriever, context_query
from .training import PMModel


def estimated_action_cost(
    state: RuntimeState,
    action_id: str,
    *,
    memory_top_k: dict[MemorySource, int] | None = None,
    strategy_tokens: int = 260,
    retrieval_call_penalty: int = 24,
) -> float:
    memory_top_k = memory_top_k or dict(DEFAULT_MEMORY_TOP_K)
    sources, strategy = parse_action_id(action_id)
    tokens = 0.0
    calls = 0
    for source in sources:
        cat = state.inventory[source]
        avg = cat.estimated_tokens / max(cat.count, 1)
        tokens += min(cat.estimated_tokens, avg * memory_top_k[source])
        calls += 1
    if strategy is StrategyMode.RS:
        tokens += strategy_tokens
        calls += 1
    return tokens + calls * retrieval_call_penalty


class Policy(Protocol):
    name: str
    def choose(self, state: RuntimeState) -> str: ...


@dataclass
class LearnedPMPolicy:
    model: PMModel
    epsilon: float
    tau_misuse: float = 0.35
    tau_omission: float = 0.35
    tau_strategy: float = 0.35
    # Fail closed by default. Development-only grid searches must opt in
    # explicitly and selection later rejects every setting that actually used
    # fallback on validation data.
    allow_constraint_fallback: bool = False
    name: str = "pm"
    last_decision_report: dict | None = None

    @staticmethod
    def _violation(value: float, threshold: float) -> float:
        return max(0.0, value - threshold) / max(threshold, 1e-6)

    def choose(self, state: RuntimeState) -> str:
        inference_start = time.perf_counter()
        scores = self.model.score_actions(state, state.allowed_actions)
        pm_inference_ms = (time.perf_counter() - inference_start) * 1000.0
        safe = [
            action_id for action_id, value in scores.items()
            if value["misuse_risk"] <= self.tau_misuse
            and value["memory_omission_risk"] <= self.tau_omission
            and value["strategy_decision_risk"] <= self.tau_strategy
        ]
        fallback_used = False
        if safe:
            qmax_safe = max(scores[x]["response_score"] for x in safe)
            candidates = [
                action_id for action_id in safe
                if scores[action_id]["response_score"] >= qmax_safe - self.epsilon
            ]
        else:
            if not self.allow_constraint_fallback:
                raise RuntimeError(
                    "No action satisfies the preregistered misuse, memory-omission "
                    "and strategy-decision constraints. State=" + state.card_id
                )
            fallback_used = True
            # Fail-soft only for development.  Choose the least normalized
            # violation rather than silently privileging the cheapest M0 action.
            def violation_tuple(action_id: str):
                value = scores[action_id]
                violations = (
                    self._violation(value["misuse_risk"], self.tau_misuse),
                    self._violation(
                        value["memory_omission_risk"], self.tau_omission
                    ),
                    self._violation(
                        value["strategy_decision_risk"], self.tau_strategy
                    ),
                )
                return (max(violations), sum(violations))

            best_violation = min(violation_tuple(x) for x in scores)
            candidates = [
                action_id for action_id in scores
                if violation_tuple(action_id) == best_violation
            ]
            qmax = max(scores[x]["response_score"] for x in candidates)
            candidates = [
                x for x in candidates
                if scores[x]["response_score"] >= qmax - self.epsilon
            ]

        chosen = min(
            candidates,
            key=lambda action_id: (
                estimated_action_cost(state, action_id),
                -scores[action_id]["memory_decision_quality"],
                -scores[action_id]["strategy_decision_quality"],
                -scores[action_id]["response_score"],
                action_id,
            ),
        )
        self.last_decision_report = {
            "card_id": state.card_id,
            "chosen_action": chosen,
            "safe_actions": sorted(safe),
            "candidate_actions": sorted(candidates),
            "constraint_fallback_used": fallback_used,
            "thresholds": {
                "epsilon": self.epsilon,
                "tau_misuse": self.tau_misuse,
                "tau_omission": self.tau_omission,
                "tau_strategy": self.tau_strategy,
            },
            "scores": scores,
            "ood_report": self.model.last_ood_report,
            "catalog_reads": sum(
                int(catalog.available) for catalog in state.inventory.values()
            ),
            "pm_inference_ms": pm_inference_ms,
            "pre_evidence_access": (
                "catalog_fingerprint_summary"
                if self.model.feature_mode in {"full", "catalog_only"}
                else "current_text_plus_inventory_metadata"
                if self.model.feature_mode == "text_metadata"
                else "metadata_or_text_only"
            ),
        }
        return chosen


@dataclass
class FixedPolicy:
    fixed_action_id: str
    name: str = "fixed"

    def choose(self, state: RuntimeState) -> str:
        sources, strategy = parse_action_id(self.fixed_action_id)
        available = {src for src, cat in state.inventory.items() if cat.available}
        effective = sources & available
        action = canonical_action_id(effective, strategy)
        if action not in state.allowed_actions:
            raise RuntimeError(f"fallback action not legal: {action}")
        return action


@dataclass
class FullAvailablePolicy:
    strategy: StrategyMode = StrategyMode.RS
    name: str = "all_available"

    def choose(self, state: RuntimeState) -> str:
        sources = frozenset(
            src for src, cat in state.inventory.items() if cat.available
        )
        return canonical_action_id(sources, self.strategy)


@dataclass
class RuleConfig:
    mp_threshold: float = 0.08
    ms_threshold: float = 0.08
    me_threshold: float = 0.08
    strategy_threshold: float = 0.06
    max_sources: int = 2


@dataclass
class StrongRulePolicy:
    config: RuleConfig
    strategy_retriever: StrategyRetriever
    name: str = "strong_rule"

    def __post_init__(self):
        self._hash = HashingVectorizer(
            n_features=64,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )

    def _source_similarity(self, state: RuntimeState, source: MemorySource) -> float:
        query = context_query(
            state.current_user_text,
            [x.model_dump(mode="json") for x in state.current_session_history],
            state.current_session_summary,
        )
        q = np.asarray(self._hash.transform([query]).toarray()[0], dtype=float)
        c = np.asarray(state.inventory[source].catalog_fingerprint, dtype=float)
        if not q.size or not c.size or np.linalg.norm(q) == 0 or np.linalg.norm(c) == 0:
            return 0.0
        return float(q @ c / (np.linalg.norm(q) * np.linalg.norm(c)))

    def choose(self, state: RuntimeState) -> str:
        threshold = {
            MemorySource.MP: self.config.mp_threshold,
            MemorySource.MS: self.config.ms_threshold,
            MemorySource.ME: self.config.me_threshold,
        }
        scored = [
            (self._source_similarity(state, source), source)
            for source in MemorySource
            if state.inventory[source].available
        ]
        selected = [
            source for score, source in sorted(scored, reverse=True)
            if score >= threshold[source]
        ][: self.config.max_sources]
        query = context_query(
            state.current_user_text,
            [x.model_dump(mode="json") for x in state.current_session_history],
            state.current_session_summary,
        )
        strategy = (
            StrategyMode.RS
            if self.strategy_retriever.confidence(query) >= self.config.strategy_threshold
            else StrategyMode.R0
        )
        return canonical_action_id(frozenset(selected), strategy)
