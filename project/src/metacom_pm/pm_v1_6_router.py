from __future__ import annotations

from dataclasses import dataclass
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import MemorySource, StrategyMode, canonical_action_id
from .io import canonical_json, sha256_text
from .pm_v1_6_contracts import Step0Observation


class TransparentRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: str = "pm-v1.6-transparent-router-v1"
    source_similarity_weight: float = 1.0
    source_age_penalty: float = 0.20
    source_cost_penalty: float = 0.10
    source_availability_bonus: float = 0.05
    source_selection_threshold: float = 0.15
    maximum_sources: int = Field(default=2, ge=0, le=3)
    strategy_family_similarity_weight: float = 0.75
    advice_request_bonus: float = 0.35
    action_readiness_weight: float = 0.20
    advice_reject_penalty: float = 0.60
    listening_request_penalty: float = 0.35
    strategy_selection_threshold: float = 0.25

    @model_validator(mode="after")
    def finite(self) -> "TransparentRouterConfig":
        values = [
            float(value)
            for key, value in self.model_dump().items()
            if key != "protocol"
        ]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("router parameters must be finite")
        return self

    def digest(self) -> str:
        return sha256_text(canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True)
class TransparentRouterDecision:
    action_id: str
    source_scores: dict[str, float]
    strategy_score: float
    config_sha256: str


class StrongTransparentRouter:
    """A non-neural, train-frozen baseline using exactly the PM Step-0 view."""

    def __init__(self, config: TransparentRouterConfig):
        self.config = config

    def choose(self, observation: Step0Observation) -> TransparentRouterDecision:
        scores: dict[MemorySource, float] = {}
        maximum_tokens = max(
            (
                row.estimated_retrievable_tokens
                for row in observation.sources.values()
            ),
            default=1,
        )
        maximum_age = max(
            (
                float(row.max_age_sessions or 0)
                for row in observation.sources.values()
            ),
            default=1.0,
        )
        for source in MemorySource:
            row = observation.sources[source]
            if not row.available or not row.representation_valid:
                # Keep serialized audit records finite. This sentinel can never
                # cross a valid frozen selection threshold.
                scores[source] = -1_000_000.0
                continue
            age = float(row.max_age_sessions or 0) / max(maximum_age, 1.0)
            cost = row.estimated_retrievable_tokens / max(maximum_tokens, 1)
            scores[source] = (
                self.config.source_similarity_weight
                * row.query_to_source_similarity
                - self.config.source_age_penalty * age
                - self.config.source_cost_penalty * cost
                + self.config.source_availability_bonus
            )
        selected = [
            source
            for source, score in sorted(
                scores.items(), key=lambda item: (-item[1], item[0].value)
            )
            if score >= self.config.source_selection_threshold
        ][: self.config.maximum_sources]

        valid_family_scores = [
            row.query_to_family_similarity
            for row in observation.strategy_families
            if row.representation_valid
        ]
        family_score = max(valid_family_scores, default=0.0)
        readiness = observation.readiness
        strategy_score = (
            self.config.strategy_family_similarity_weight * family_score
            + self.config.advice_request_bonus * float(readiness.advice_requested)
            + self.config.action_readiness_weight * readiness.action_readiness
            - self.config.advice_reject_penalty * float(readiness.advice_rejected)
            - self.config.listening_request_penalty
            * float(readiness.listening_requested)
        )
        strategy = (
            StrategyMode.RS
            if strategy_score >= self.config.strategy_selection_threshold
            else StrategyMode.R0
        )
        return TransparentRouterDecision(
            action_id=canonical_action_id(set(selected), strategy),
            source_scores={
                source.value: float(value) for source, value in scores.items()
            },
            strategy_score=float(strategy_score),
            config_sha256=self.config.digest(),
        )


def router_grid() -> list[TransparentRouterConfig]:
    """Small preregisterable grid; select using train users only."""

    rows: list[TransparentRouterConfig] = []
    for source_threshold in (0.05, 0.15, 0.25):
        for maximum_sources in (1, 2, 3):
            for strategy_threshold in (0.15, 0.25, 0.35):
                rows.append(
                    TransparentRouterConfig(
                        source_selection_threshold=source_threshold,
                        maximum_sources=maximum_sources,
                        strategy_selection_threshold=strategy_threshold,
                    )
                )
    return rows
