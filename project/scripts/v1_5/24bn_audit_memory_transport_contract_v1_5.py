#!/usr/bin/env python3
"""Outcome-blind audit of internal versus EvoEmo memory representations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.contracts import MemoryItem, MemorySource, RuntimeState
from metacom_pm.evoemo import (
    EVO_MEMORY_PROTOCOL,
    build_evo_memory,
    evo_memory_builder_contract_hash,
    load_evoemo,
)
from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.retrieval import DEFAULT_MEMORY_TOP_K
from metacom_pm.text import estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-memory-transport-contract-audit-v1"


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _profile(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(values),
        "min": min(values),
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _outside_rate(
    external: list[float], internal: list[float], *, robust: bool
) -> float | None:
    if not external or not internal:
        return None
    lower = _quantile(internal, 0.05) if robust else min(internal)
    upper = _quantile(internal, 0.95) if robust else max(internal)
    assert lower is not None and upper is not None
    return sum(value < lower or value > upper for value in external) / len(
        external
    )


def _catalog_rows(
    *,
    domain: str,
    record_id: str,
    session_index: int,
    items: list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in MemorySource:
        selected = [item for item in items if item.source is source]
        item_tokens = [estimate_tokens(item.text) for item in selected]
        ages = [
            max(0, session_index - int(item.created_session))
            for item in selected
        ]
        top_k = int(DEFAULT_MEMORY_TOP_K[source])
        ranked_capacity = min(len(selected), top_k)
        mean_tokens = (
            sum(item_tokens) / len(item_tokens) if item_tokens else 0.0
        )
        rows.append(
            {
                "domain": domain,
                "record_id": record_id,
                "source": source.value,
                "catalog_count": len(selected),
                "catalog_tokens": sum(item_tokens),
                "mean_item_tokens": mean_tokens,
                "estimated_top_k_tokens": min(
                    sum(item_tokens), mean_tokens * ranked_capacity
                ),
                "minimum_age_sessions": min(ages) if ages else None,
                "maximum_age_sessions": max(ages) if ages else None,
                "item_tokens": item_tokens,
                "item_ages": ages,
            }
        )
    return rows


def _domain_profile(
    rows: list[dict[str, Any]], domain: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in MemorySource:
        selected = [
            row
            for row in rows
            if row["domain"] == domain and row["source"] == source.value
        ]
        result[source.value] = {
            "catalog_count": _profile(
                [float(row["catalog_count"]) for row in selected]
            ),
            "catalog_tokens": _profile(
                [float(row["catalog_tokens"]) for row in selected]
            ),
            "mean_item_tokens_per_catalog": _profile(
                [float(row["mean_item_tokens"]) for row in selected]
            ),
            "estimated_top_k_tokens": _profile(
                [float(row["estimated_top_k_tokens"]) for row in selected]
            ),
            "item_tokens": _profile(
                [
                    float(value)
                    for row in selected
                    for value in row["item_tokens"]
                ]
            ),
            "item_age_sessions": _profile(
                [
                    float(value)
                    for row in selected
                    for value in row["item_ages"]
                ]
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-backend",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "memory_backend.jsonl",
    )
    parser.add_argument(
        "--internal-runtime",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "runtime_states.jsonl",
    )
    parser.add_argument(
        "--evoemo",
        type=Path,
        default=ROOT / "data/external/evo_emo.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_memory_transport_contract_audit_v1",
    )
    args = parser.parse_args()

    runtime_by_card = {
        state.card_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(args.internal_runtime)
        )
    }
    rows: list[dict[str, Any]] = []
    internal_records = 0
    for raw in iter_jsonl(args.internal_backend):
        card_id = str(raw["card_id"])
        state = runtime_by_card[card_id]
        items = [
            MemoryItem.model_validate(item) for item in raw["items"]
        ]
        rows.extend(
            _catalog_rows(
                domain="internal_synthetic",
                record_id=card_id,
                session_index=state.session_index,
                items=items,
            )
        )
        internal_records += 1

    evoemo_users = load_evoemo(args.evoemo)
    for user in evoemo_users:
        items, _ = build_evo_memory(user)
        session_index = max(
            (item.created_session for item in items), default=0
        ) + 1
        rows.extend(
            _catalog_rows(
                domain="evoemo_external",
                record_id=str(user["id"]),
                session_index=session_index,
                items=list(items),
            )
        )

    internal_profile = _domain_profile(rows, "internal_synthetic")
    external_profile = _domain_profile(rows, "evoemo_external")
    comparisons: dict[str, Any] = {}
    for source in MemorySource:
        internal_source_rows = [
            row
            for row in rows
            if row["domain"] == "internal_synthetic"
            and row["source"] == source.value
        ]
        external_source_rows = [
            row
            for row in rows
            if row["domain"] == "evoemo_external"
            and row["source"] == source.value
        ]
        comparisons[source.value] = {}
        for field in (
            "catalog_count",
            "catalog_tokens",
            "mean_item_tokens",
            "estimated_top_k_tokens",
        ):
            internal_values = [
                float(row[field]) for row in internal_source_rows
            ]
            external_values = [
                float(row[field]) for row in external_source_rows
            ]
            comparisons[source.value][field] = {
                "external_outside_internal_min_max_rate": _outside_rate(
                    external_values, internal_values, robust=False
                ),
                "external_outside_internal_p05_p95_rate": _outside_rate(
                    external_values, internal_values, robust=True
                ),
                "external_to_internal_median_ratio": (
                    None
                    if not internal_values
                    or not external_values
                    or not _quantile(internal_values, 0.5)
                    else _quantile(external_values, 0.5)
                    / float(_quantile(internal_values, 0.5))
                ),
            }
    checks = {
        "same_memory_item_wire_schema": True,
        "same_source_ontology_MP_MS_ME": True,
        "same_retriever_implementation_available": True,
        "same_default_top_k": {
            source.value: int(DEFAULT_MEMORY_TOP_K[source])
            for source in MemorySource
        },
        "same_memory_construction_algorithm": False,
        "internal_builder": (
            "generated case supplies two preconstructed items per source"
        ),
        "external_builder": EVO_MEMORY_PROTOCOL,
        "external_builder_contract_sha256": (
            evo_memory_builder_contract_hash()
        ),
        "external_outcomes_read": False,
        "external_reference_answers_read": False,
    }
    report = {
        "protocol": PROTOCOL,
        "status": "TRANSPORT_MISMATCH_REQUIRES_CLAIM_AND_INPUT_REPAIR",
        "severity": "HIGH_FOR_STRONG_EXTERNAL_GENERALIZATION_CLAIM",
        "internal_catalog_records": internal_records,
        "external_users": len(evoemo_users),
        "profiles": {
            "internal_synthetic": internal_profile,
            "evoemo_external": external_profile,
        },
        "comparisons": comparisons,
        "contract_checks": checks,
        "scientific_interpretation": {
            "not_a_bug": (
                "Personalized memory text must differ across users and domains."
            ),
            "actual_bug": (
                "The PM is trained under a synthetic memory-construction and "
                "candidate-feature distribution that is not guaranteed to "
                "match the external memory compiler and catalog shape."
            ),
            "invalid_strong_claim": (
                "The current design alone cannot prove that PM learned when "
                "to use arbitrary external memories."
            ),
            "salvageable_claim": (
                "It can test transport of a source-routing policy under a "
                "declared catalog shift, with OOD abstention and fixed "
                "external treatment."
            ),
        },
        "minimum_repair": [
            "Freeze one canonical memory-item and candidate-descriptor contract.",
            "Run the same query builder, retriever, filter, top-k, and descriptor extractor in internal and external data.",
            "Train component heads on state plus realized candidate descriptors, not source name/count alone.",
            "Measure external support overlap before outcomes; OOD candidates abstain/off.",
            "Keep EvoEmo development-informed and use a future untouched corpus for a pristine generalization claim.",
        ],
        "inputs": {
            "internal_backend": str(args.internal_backend.relative_to(ROOT)),
            "internal_backend_sha256": sha256_file(args.internal_backend),
            "internal_runtime": str(args.internal_runtime.relative_to(ROOT)),
            "internal_runtime_sha256": sha256_file(args.internal_runtime),
            "evoemo": str(args.evoemo.relative_to(ROOT)),
            "evoemo_sha256": sha256_file(args.evoemo),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "memory_transport_audit.json", report)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": report["status"],
                "severity": report["severity"],
                "internal_catalog_records": internal_records,
                "external_users": len(evoemo_users),
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
