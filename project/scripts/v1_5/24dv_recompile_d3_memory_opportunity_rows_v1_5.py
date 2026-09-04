#!/usr/bin/env python3
"""Recompile frozen D3 memory states through the production feature bridge.

The D3 Step-0 blueprint predates every D3 response and human quality outcome.
This script reads only that frozen blueprint, runtime states, memory catalogs,
and construction provenance.  It demonstrates that 32 real-text states per
memory component can be reused as outcome-blind routing data without touching
the later response generation or annotations.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.contracts import MemoryBackendRecord, MemorySource, RuntimeState
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_candidate_discovery import MemoryCandidate
from metacom_pm.v1_5_memory_opportunity_features import (
    build_memory_opportunity_observation,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d3-memory-opportunity-production-recompile-v1"
COMPONENTS = ("MP", "MS", "ME")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def build(*, root: Path = ROOT) -> dict[str, Any]:
    source_dir = root / "outputs/pm_v1_5_d3_step0_blueprint_v1"
    blueprint_path = source_dir / "d3_contrast_blueprint.jsonl"
    state_path = source_dir / "runtime_states.jsonl"
    backend_path = source_dir / "memory_backend.jsonl"
    metadata_path = source_dir / "memory_item_metadata.jsonl"
    preflight_path = source_dir / "preflight_report.json"

    blueprints = [
        row for row in _rows(blueprint_path) if row["component"] in COMPONENTS
    ]
    states = {
        row.state_id: row
        for row in (
            RuntimeState.model_validate(value) for value in _rows(state_path)
        )
    }
    backends = {
        row.card_id: row
        for row in (
            MemoryBackendRecord.model_validate(value) for value in _rows(backend_path)
        )
    }
    metadata = {
        str(row["card_id"]): dict(row["memory_item_metadata"])
        for row in _rows(metadata_path)
    }

    compiled: list[dict[str, Any]] = []
    for row in blueprints:
        component = str(row["component"])
        source = MemorySource(component)
        state = states[str(row["state_id"])]
        backend = backends[state.card_id]
        item_by_id = {item.memory_id: item for item in backend.items}
        selected_ids = row["selected_memory_ids_by_action_generation_only"][
            row["treatment_action"]
        ]
        selected_items = tuple(
            item_by_id[str(memory_id)]
            for memory_id in selected_ids
            if item_by_id[str(memory_id)].source is source
        )
        old_descriptor = dict(
            row["component_candidate_observation"]["descriptor"]
        )
        candidate = MemoryCandidate(
            source=source,
            selected_items=selected_items,
            descriptor=old_descriptor,
        )
        observation = build_memory_opportunity_observation(
            candidate=candidate,
            catalog_items=backend.items,
            catalog_user_id=state.user_id,
            current_user_id=state.user_id,
            current_session_index=state.session_index,
            current_user_text=state.current_user_text,
            visible_dialogue=[
                turn.model_dump(mode="json") for turn in state.current_session_history
            ],
            source_metadata=metadata[state.card_id],
            background_action=str(row["background_action"]),
        )
        old_features = {
            str(key): float(value)
            for key, value in dict(row["model_features"]).items()
        }
        new_features = {
            str(key): float(value)
            for key, value in dict(observation["model_features"]).items()
        }
        features_match = old_features == new_features
        opportunity = int(
            not observation["deterministic_hard_off"]
            and observation["candidate_audit"]["grounding_checks"][
                "not_already_visible"
            ]
        )
        compiled.append(
            {
                "protocol": PROTOCOL,
                "component": component,
                "user_id": state.user_id,
                "state_id": state.state_id,
                "condition_family_private_not_model_input": (
                    "nonredundant_candidate"
                    if opportunity
                    else "already_visible_or_cross_resource_redundant"
                ),
                "opportunity_target": opportunity,
                "model_features": new_features,
                "deterministic_hard_off": observation[
                    "deterministic_hard_off"
                ],
                "hard_off_reasons": observation["hard_off_reasons"],
                "feature_recompile_matches_frozen_step0": features_match,
                "response_quality_risk_or_external_outcome_read": False,
            }
        )

    by_component: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        values = [row for row in compiled if row["component"] == component]
        by_component[component] = {
            "rows": len(values),
            "independent_users": len({row["user_id"] for row in values}),
            "class_counts": {
                str(key): value
                for key, value in sorted(
                    Counter(row["opportunity_target"] for row in values).items()
                )
            },
            "all_features_match_frozen_step0": all(
                row["feature_recompile_matches_frozen_step0"] for row in values
            ),
            "hard_off_rows": sum(row["deterministic_hard_off"] for row in values),
        }
    checks = {
        "96_rows_32_per_component": (
            len(compiled) == 96
            and all(by_component[value]["rows"] == 32 for value in COMPONENTS)
        ),
        "32_independent_users_per_component": all(
            by_component[value]["independent_users"] == 32
            for value in COMPONENTS
        ),
        "16_on_16_off_per_component": all(
            by_component[value]["class_counts"] == {"0": 16, "1": 16}
            for value in COMPONENTS
        ),
        "production_recompile_matches_all_frozen_features": all(
            by_component[value]["all_features_match_frozen_step0"]
            for value in COMPONENTS
        ),
        "formal_rows_have_no_hard_off": all(
            by_component[value]["hard_off_rows"] == 0 for value in COMPONENTS
        ),
        "no_response_quality_risk_or_external_outcome_read": True,
    }
    status = (
        "PASS_REUSE_32_REAL_TEXT_ROWS_PER_MEMORY_HEAD"
        if all(checks.values())
        else "FAIL_D3_PRODUCTION_RECOMPILE"
    )
    out_dir = root / "outputs/pm_v1_5_d3_memory_opportunity_recompile_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "production_compiled_opportunity_rows.jsonl"
    write_jsonl(rows_path, compiled)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "evidence_role": (
            "Outcome-blind real-text routing development data. These rows may "
            "replace numeric micro-world rows but are not untouched confirmation "
            "and do not establish generator or system benefit."
        ),
        "component_reports": by_component,
        "checks": checks,
        "remaining_data": {
            "MP": "add 16 new train plus 16 untouched confirmation users",
            "MS": "retain separate 32-row incremental-value fit; add 16 untouched condition-family confirmation users",
            "ME": "add 16 new train plus 16 untouched confirmation users",
        },
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                blueprint_path,
                state_path,
                backend_path,
                metadata_path,
                preflight_path,
            )
        },
    }
    write_json(out_dir / "recompile_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "artifacts": {
                rows_path.name: sha256_file(rows_path),
                "recompile_report.json": sha256_file(
                    out_dir / "recompile_report.json"
                ),
            },
        },
    )
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "component_reports": report["component_reports"],
        }
    )


if __name__ == "__main__":
    main()
