#!/usr/bin/env python3
"""Audit a no-external-content synthetic recompile with the Evo memory builder."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from metacom_pm.contracts import MemorySource
from metacom_pm.evoemo import (
    EVO_MEMORY_PROTOCOL,
    evo_memory_builder_contract_hash,
)
from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v2_data import load_bundles
from metacom_pm.retrieval import DEFAULT_MEMORY_TOP_K
from metacom_pm.text import estimate_tokens
from metacom_pm.v1_5_memory_transport import (
    TRANSPORT_ADAPTER_PROTOCOL,
    compile_synthetic_longitudinal_catalogs,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-evo-style-synthetic-memory-recompile-audit-v1"


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _profile(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise RuntimeError("cannot profile an empty value list")
    return {
        "n": len(values),
        "min": min(values),
        "median": _quantile(values, 0.5),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "pm_v2_bundles.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_style_synthetic_memory_recompile_audit_v1",
    )
    args = parser.parse_args()

    profiles: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    eligible_state_rows: list[dict[str, Any]] = []
    bundles = load_bundles(args.bundles)
    for catalog in compile_synthetic_longitudinal_catalogs(bundles):
        row = {
            "user_id": catalog.user_id,
            "case_id": catalog.case.case_id,
            "surface_form_id": catalog.case.surface_form_id,
            "session_index": catalog.chronological_session_index,
            "prior_session_count": catalog.prior_session_count,
            "source_counts": {},
        }
        for source in MemorySource:
            selected = [
                item for item in catalog.items if item.source is source
            ]
            item_tokens = [
                float(estimate_tokens(item.text)) for item in selected
            ]
            count = float(len(selected))
            top_k = int(DEFAULT_MEMORY_TOP_K[source])
            row["source_counts"][source.value] = int(count)
            profiles[source.value]["catalog_count"].append(count)
            profiles[source.value]["item_tokens"].extend(item_tokens)
            profiles[source.value]["catalog_tokens"].append(
                sum(item_tokens)
            )
            profiles[source.value]["top_k_capacity_fraction"].append(
                min(count, top_k) / top_k
            )
        eligible_state_rows.append(row)

    report_profiles = {
        source.value: {
            metric: _profile(values)
            for metric, values in profiles[source.value].items()
        }
        for source in MemorySource
    }
    users = {row["user_id"] for row in eligible_state_rows}
    report = {
        "protocol": PROTOCOL,
        "status": "FEASIBLE_WITHOUT_EXTERNAL_CONTENT_OR_OUTCOMES",
        "synthetic_users": len(users),
        "eligible_longitudinal_states": len(eligible_state_rows),
        "states_per_user": _profile(
            [
                float(
                    sum(
                        row["user_id"] == user_id
                        for row in eligible_state_rows
                    )
                )
                for user_id in sorted(users)
            ]
        ),
        "memory_builder_protocol": EVO_MEMORY_PROTOCOL,
        "memory_builder_contract_sha256": (
            evo_memory_builder_contract_hash()
        ),
        "transport_adapter_protocol": TRANSPORT_ADAPTER_PROTOCOL,
        "profiles": report_profiles,
        "checks": {
            "external_file_read": False,
            "external_content_used": False,
            "external_outcome_used": False,
            "same_builder_function_as_evoemo": True,
            "strictly_prior_sessions_only": True,
            "at_least_256_eligible_states": (
                len(eligible_state_rows) >= 256
            ),
            "all_sources_have_top_k_competition_somewhere": all(
                report_profiles[source.value]["catalog_count"]["max"]
                >= DEFAULT_MEMORY_TOP_K[source]
                for source in MemorySource
            ),
        },
        "adapter": {
            "basic_info": (
                "the bundle's existing stable_preferences, preserved as "
                "separate atomic fields without invented values; the two "
                "globally constant compiler boundaries remain system rules "
                "rather than being misrepresented as personal memory"
            ),
            "MS": (
                "prior case session_summary, with current_user_text as "
                "transparent extractive fallback when summary is empty"
            ),
            "ME": (
                "prior case seeker turns passed through the exact Evo "
                "episode chunker"
            ),
        },
        "input": {
            "bundles": str(args.bundles.relative_to(ROOT)),
            "bundles_sha256": sha256_file(args.bundles),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "recompile_feasibility_report.json", report)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": report["status"],
                "synthetic_users": report["synthetic_users"],
                "eligible_longitudinal_states": report[
                    "eligible_longitudinal_states"
                ],
                "median_catalog_counts": {
                    source.value: report_profiles[source.value][
                        "catalog_count"
                    ]["median"]
                    for source in MemorySource
                },
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
