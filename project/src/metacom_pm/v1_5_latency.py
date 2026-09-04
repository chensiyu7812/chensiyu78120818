"""Descriptive-only latency diagnostics for frozen PM-v1.5 generations."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL = "pm-v1.5-noninterleaved-latency-diagnostic-v1"
RESOURCE_ACCOUNTING_PROTOCOL = "pm-v1.5-separated-resource-accounting-v1"


def _unit(row: Mapping[str, Any]) -> tuple[str, int, int, str, int]:
    return (
        str(row.get("user_id") or ""),
        int(row.get("topic_index") or 0),
        int(row.get("seed") or 0),
        str(row.get("simulator_id") or ""),
        int(row.get("turn_index") or 0),
    )


def _latencies(row: Mapping[str, Any]) -> dict[str, float]:
    cost = row.get("cost") or {}
    values = {
        "total_latency_ms": row.get("latency_ms", cost.get("latency_ms")),
        "generation_latency_ms": cost.get("generation_latency_ms"),
        "retrieval_latency_ms": cost.get("retrieval_latency_ms", 0.0),
        "pm_inference_ms": cost.get("pm_inference_ms", 0.0),
        "pre_evidence_compute_ms": cost.get("pre_evidence_compute_ms", 0.0),
    }
    normalized: dict[str, float] = {}
    for name, raw in values.items():
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"generation turn lacks numeric {name}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"generation turn has invalid {name}")
        normalized[name] = value
    return normalized


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        raise RuntimeError("latency summary cannot be empty")
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def build_descriptive_latency_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    treatment: str,
    expected_units: Sequence[tuple[str, int, int, str, int]],
) -> dict[str, Any]:
    """Summarize already-observed latency without an inferential claim."""

    expected = set(expected_units)
    by_condition: dict[str, dict[tuple[str, int, int, str, int], dict[str, float]]] = (
        defaultdict(dict)
    )
    allowed = set(conditions)
    for row in rows:
        condition = str(row.get("condition") or "")
        unit = _unit(row)
        if condition not in allowed or unit not in expected:
            continue
        if unit in by_condition[condition]:
            raise RuntimeError(f"duplicate latency unit for condition {condition}")
        by_condition[condition][unit] = _latencies(row)
    for condition in conditions:
        if set(by_condition.get(condition, {})) != expected:
            raise RuntimeError(
                f"latency rows for {condition} do not exactly cover scored units"
            )
    if treatment not in allowed:
        raise RuntimeError("latency treatment is absent from conditions")

    metric_names = tuple(next(iter(by_condition[treatment].values())))
    condition_summaries = {
        condition: {
            metric: _summary(
                [by_condition[condition][unit][metric] for unit in sorted(expected)]
            )
            for metric in metric_names
        }
        for condition in conditions
    }
    paired_deltas = {}
    for baseline in conditions:
        if baseline == treatment:
            continue
        paired_deltas[baseline] = {
            metric: {
                **_summary(
                    [
                        by_condition[treatment][unit][metric]
                        - by_condition[baseline][unit][metric]
                        for unit in sorted(expected)
                    ]
                ),
                "direction": "treatment_minus_baseline_ms",
            }
            for metric in metric_names
        }
    return {
        "status": "COMPLETE",
        "protocol": PROTOCOL,
        "role": "diagnostic_only",
        "confirmatory_latency_claim_allowed": False,
        "reason": (
            "condition generation was not randomized/interleaved, so provider "
            "load and run order may confound wall-clock latency"
        ),
        "treatment": treatment,
        "conditions": list(conditions),
        "scored_units_per_condition": len(expected),
        "condition_summaries": condition_summaries,
        "paired_point_deltas": paired_deltas,
    }


def build_separated_resource_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    expected_units: Sequence[tuple[str, int, int, str, int]],
    generator_pricing_usd_per_mtok: Mapping[str, float],
) -> dict[str, Any]:
    """Report compute, retrieval, tokens, latency and USD without collapsing them."""

    pricing = {
        "input": float(generator_pricing_usd_per_mtok["input"]),
        "output": float(generator_pricing_usd_per_mtok["output"]),
    }
    if any(value <= 0.0 for value in pricing.values()):
        raise ValueError("generator resource report requires positive frozen prices")
    expected = set(expected_units)
    allowed = set(str(value) for value in conditions)
    by_condition: dict[str, dict[tuple[str, int, int, str, int], dict[str, float]]] = (
        defaultdict(dict)
    )
    for row in rows:
        condition = str(row.get("condition") or "")
        unit = _unit(row)
        if condition not in allowed or unit not in expected:
            continue
        if unit in by_condition[condition]:
            raise RuntimeError(f"duplicate resource-accounting unit for {condition}")
        cost = dict(row.get("cost") or {})
        input_tokens = float(
            row.get("input_tokens") or cost.get("total_input_tokens") or 0
        )
        output_tokens = float(
            row.get("output_tokens") or cost.get("output_tokens") or 0
        )
        if input_tokens <= 0.0 or output_tokens < 0.0:
            raise RuntimeError("generation turn has invalid observed token usage")
        attempts = list(row.get("retrieval_attempts") or [])
        attempted = sum(bool(value.get("requested")) for value in attempts)
        hits = (
            sum(int(value.get("hit_count") or 0) for value in attempts)
            if attempts
            else len(row.get("selected_memory") or [])
            + len(row.get("selected_strategy") or [])
        )
        retrieval_calls = float(cost.get("retrieval_calls", attempted))
        api_cost = cost.get("api_cost_usd")
        if api_cost is None:
            api_cost = (
                input_tokens / 1_000_000 * pricing["input"]
                + output_tokens / 1_000_000 * pricing["output"]
            )
        metrics = {
            "step0_memory_comparisons": float(
                cost.get("step0_memory_comparisons", 0)
            ),
            "step0_strategy_family_comparisons": float(
                cost.get("step0_strategy_family_comparisons", 0)
            ),
            "step0_latency_ms": float(cost.get("step0_latency_ms", 0.0)),
            "item_retrieval_attempts": retrieval_calls,
            "item_retrieval_hits": float(hits),
            "retrieval_latency_ms": float(cost.get("retrieval_latency_ms", 0.0)),
            "generator_input_tokens": input_tokens,
            "generator_output_tokens": output_tokens,
            "wall_clock_latency_ms": float(
                row.get("latency_ms", cost.get("latency_ms", 0.0))
            ),
            "generator_api_cost_usd": float(api_cost),
        }
        if any(not math.isfinite(value) or value < 0.0 for value in metrics.values()):
            raise RuntimeError("generation resource metrics must be finite/non-negative")
        by_condition[condition][unit] = metrics
    for condition in conditions:
        if set(by_condition.get(str(condition), {})) != expected:
            raise RuntimeError(
                f"resource rows for {condition} do not exactly cover scored units"
            )
    metric_names = tuple(next(iter(by_condition[str(conditions[0])].values())))
    summaries = {}
    for condition in conditions:
        values = by_condition[str(condition)]
        summaries[str(condition)] = {
            metric: {
                "total": float(
                    sum(values[unit][metric] for unit in sorted(expected))
                ),
                "mean_per_scored_turn": float(
                    np.mean([values[unit][metric] for unit in sorted(expected)])
                ),
            }
            for metric in metric_names
        }
    return {
        "status": "COMPLETE",
        "protocol": RESOURCE_ACCOUNTING_PROTOCOL,
        "aggregation": "separate_metrics_no_composite_cost",
        "conditions": list(conditions),
        "scored_units_per_condition": len(expected),
        "generator_pricing_usd_per_mtok": pricing,
        "condition_summaries": summaries,
        "judge_tokens_and_cost_included": False,
    }
