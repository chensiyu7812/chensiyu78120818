from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .pm_v1_6_contracts import CompleteCostVector, Step0Cost, RetrievalAttempt


@dataclass(frozen=True)
class CostPricing:
    generator_input_usd_per_mtok: float
    generator_output_usd_per_mtok: float
    step0_compute_usd_per_ms: float = 0.0
    router_compute_usd_per_ms: float = 0.0
    retrieval_compute_usd_per_ms: float = 0.0

    def __post_init__(self) -> None:
        if any(
            value < 0.0
            for value in (
                self.generator_input_usd_per_mtok,
                self.generator_output_usd_per_mtok,
                self.step0_compute_usd_per_ms,
                self.router_compute_usd_per_ms,
                self.retrieval_compute_usd_per_ms,
            )
        ):
            raise ValueError("all cost prices must be non-negative")


def complete_cost_vector(
    *,
    step0: Step0Cost | None,
    router_latency_ms: float,
    attempts: Sequence[RetrievalAttempt],
    generator_input_tokens: int,
    generator_output_tokens: int,
    generator_latency_ms: float,
    pricing: CostPricing,
) -> CompleteCostVector:
    """Create the frozen cost vector.

    Fixed policies pass ``step0=None`` and therefore do not pay a fictitious
    routing probe. Learned PM and transparent-rule conditions pass their actual
    Step-0 observation cost.
    """

    step0_latency = float(step0.query_encoding_latency_ms) if step0 else 0.0
    catalog_ms = float(step0.catalog_build_amortized_ms) if step0 else 0.0
    comparisons = (
        int(step0.memory_source_comparisons + step0.strategy_family_comparisons)
        if step0
        else 0
    )
    retrieval_latency = sum(float(row.latency_ms) for row in attempts)
    calls = sum(int(row.call_count) for row in attempts)
    hits = sum(int(row.hit_count) for row in attempts)
    retrieved_tokens = sum(int(row.retrieved_tokens) for row in attempts)
    generator_cost = (
        int(generator_input_tokens) / 1_000_000
        * pricing.generator_input_usd_per_mtok
        + int(generator_output_tokens) / 1_000_000
        * pricing.generator_output_usd_per_mtok
    )
    compute_cost = (
        (step0_latency + catalog_ms) * pricing.step0_compute_usd_per_ms
        + float(router_latency_ms) * pricing.router_compute_usd_per_ms
        + retrieval_latency * pricing.retrieval_compute_usd_per_ms
    )
    component_latency = (
        step0_latency
        + catalog_ms
        + float(router_latency_ms)
        + retrieval_latency
        + float(generator_latency_ms)
    )
    return CompleteCostVector(
        step0_latency_ms=step0_latency,
        step0_comparisons=comparisons,
        catalog_build_amortized_ms=catalog_ms,
        router_inference_latency_ms=float(router_latency_ms),
        item_retrieval_calls=calls,
        item_retrieval_hits=hits,
        item_retrieval_tokens=retrieved_tokens,
        item_retrieval_latency_ms=retrieval_latency,
        generator_input_tokens=int(generator_input_tokens),
        generator_output_tokens=int(generator_output_tokens),
        generator_api_cost_usd=float(generator_cost),
        generator_latency_ms=float(generator_latency_ms),
        total_variable_cost_usd=float(generator_cost + compute_cost),
        end_to_end_latency_ms=float(component_latency),
    )
