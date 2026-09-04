#!/usr/bin/env python3
"""Recompile the frozen 48-user fit rows with the one repaired feature.

No response, judge, risk, external outcome, V2 confirmation, or V3
confirmation is read.  Existing D3 and supplemental users, catalogs, labels,
queries, candidate discovery, and grounding are reused; only the registered
feature representation changes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.contracts import MemoryBackendRecord, MemorySource, RuntimeState
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.v1_5_candidate_discovery import (
    MemoryCandidate,
    describe_memory_candidate,
    discover_memory_candidates_by_source_query,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_scale_stable_memory_opportunity_observation,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-scale-stable-memory-fit-recompile-v1"
COMPONENTS = ("MP", "MS", "ME")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _d3_rows(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    source = root / "outputs/pm_v1_5_d3_step0_blueprint_v1"
    blueprint_path = source / "d3_contrast_blueprint.jsonl"
    state_path = source / "runtime_states.jsonl"
    backend_path = source / "memory_backend.jsonl"
    metadata_path = source / "memory_item_metadata.jsonl"
    label_path = (
        root
        / "outputs/pm_v1_5_d3_memory_opportunity_recompile_v1/production_compiled_opportunity_rows.jsonl"
    )
    blueprints = [
        row for row in _rows(blueprint_path) if row["component"] in COMPONENTS
    ]
    states = {
        state.state_id: state
        for state in map(RuntimeState.model_validate, _rows(state_path))
    }
    backends = {
        backend.card_id: backend
        for backend in map(MemoryBackendRecord.model_validate, _rows(backend_path))
    }
    metadata = {
        str(row["card_id"]): dict(row["memory_item_metadata"])
        for row in _rows(metadata_path)
    }
    labels = {
        (str(row["component"]), str(row["state_id"])): row
        for row in _rows(label_path)
    }
    result: list[dict[str, Any]] = []
    for blueprint in blueprints:
        component = str(blueprint["component"])
        source_type = MemorySource(component)
        state = states[str(blueprint["state_id"])]
        backend = backends[state.card_id]
        items_by_id = {item.memory_id: item for item in backend.items}
        selected_ids = blueprint["selected_memory_ids_by_action_generation_only"][
            blueprint["treatment_action"]
        ]
        selected = tuple(
            items_by_id[str(memory_id)]
            for memory_id in selected_ids
            if items_by_id[str(memory_id)].source is source_type
        )
        visible = [
            turn.model_dump(mode="json") for turn in state.current_session_history
        ]
        query = source_specific_memory_queries(
            state.current_user_text,
            visible,
            state.current_session_summary,
        )[source_type]
        source_items = tuple(
            item for item in backend.items if item.source is source_type
        )
        candidate = MemoryCandidate(
            source=source_type,
            selected_items=selected,
            descriptor=describe_memory_candidate(
                source=source_type,
                query=query,
                source_items=source_items,
                selected_items=selected,
                session_index=state.session_index,
            ),
        )
        observation = build_scale_stable_memory_opportunity_observation(
            candidate=candidate,
            catalog_items=backend.items,
            catalog_user_id=state.user_id,
            current_user_id=state.user_id,
            current_session_index=state.session_index,
            current_user_text=state.current_user_text,
            visible_dialogue=visible,
            source_metadata=metadata[state.card_id],
            background_action=str(blueprint["background_action"]),
        )
        label = labels[(component, state.state_id)]
        result.append(
            {
                "protocol": PROTOCOL,
                "component": component,
                "evidence_source": "d3_step0_real_text",
                "user_id_private_not_model_input": state.user_id,
                "condition_family_private_not_model_input": (
                    "d3_" + str(label["condition_family_private_not_model_input"])
                ),
                "opportunity_target": int(label["opportunity_target"]),
                "model_features": observation["model_features"],
                "deterministic_hard_off": observation["deterministic_hard_off"],
                "outcome_read": False,
            }
        )
    return result, [blueprint_path, state_path, backend_path, metadata_path, label_path]


def _supplement_rows(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    source = root / "outputs/pm_v1_5_memory_opportunity_supplement_v1"
    state_path = source / "runtime_states.jsonl"
    backend_path = source / "memory_backend.jsonl"
    metadata_path = source / "memory_item_metadata.jsonl"
    label_path = source / "opportunity_rows.jsonl"
    states = {
        state.user_id: state
        for state in map(RuntimeState.model_validate, _rows(state_path))
        if state.split == "train"
    }
    backends = {
        backend.card_id: backend
        for backend in map(MemoryBackendRecord.model_validate, _rows(backend_path))
    }
    metadata = {
        str(row["card_id"]): dict(row["memory_item_metadata"])
        for row in _rows(metadata_path)
    }
    labels = [
        row
        for row in _rows(label_path)
        if row["evidence_role"] == "supplemental_fit"
    ]
    retriever = MemoryRetriever(
        minimum_score_by_source={source: 0.0 for source in MemorySource}
    )
    result: list[dict[str, Any]] = []
    for label in labels:
        component = str(label["component"])
        source_type = MemorySource(component)
        user_id = str(label["user_id_private_not_model_input"])
        state = states[user_id]
        backend = backends[state.card_id]
        visible = [
            turn.model_dump(mode="json") for turn in state.current_session_history
        ]
        queries = source_specific_memory_queries(
            state.current_user_text,
            visible,
            state.current_session_summary,
        )
        discoveries = discover_memory_candidates_by_source_query(
            queries=queries,
            items=backend.items,
            retriever=retriever,
            session_index=state.session_index,
        )
        observation = build_scale_stable_memory_opportunity_observation(
            candidate=discoveries[source_type],
            catalog_items=backend.items,
            catalog_user_id=state.user_id,
            current_user_id=state.user_id,
            current_session_index=state.session_index,
            current_user_text=state.current_user_text,
            visible_dialogue=visible,
            source_metadata=metadata[state.card_id],
            background_action=(
                "M0+R0"  # background bits are copied below from the frozen row
            ),
        )
        # The original background action is not retained as text in this
        # supplement, but its three already-frozen, outcome-blind bits are.
        features = dict(observation["model_features"])
        for name, value in dict(label["model_features"]).items():
            if name.startswith("background_"):
                features[name] = float(value)
        result.append(
            {
                "protocol": PROTOCOL,
                "component": component,
                "evidence_source": "real_text_supplement",
                "user_id_private_not_model_input": user_id,
                "condition_family_private_not_model_input": str(
                    label["condition_family_private_not_model_input"]
                ),
                "opportunity_target": int(label["opportunity_target"]),
                "model_features": features,
                "deterministic_hard_off": observation["deterministic_hard_off"],
                "outcome_read": False,
            }
        )
    return result, [state_path, backend_path, metadata_path, label_path]


def build(*, root: Path = ROOT) -> dict[str, Any]:
    d3, d3_inputs = _d3_rows(root)
    supplement, supplement_inputs = _supplement_rows(root)
    rows = [*d3, *supplement]
    components: dict[str, Any] = {}
    for component in COMPONENTS:
        selected = [row for row in rows if row["component"] == component]
        components[component] = {
            "rows": len(selected),
            "users": len(
                {row["user_id_private_not_model_input"] for row in selected}
            ),
            "class_counts": dict(
                sorted(Counter(row["opportunity_target"] for row in selected).items())
            ),
            "content_match_levels": sorted(
                {
                    float(row["model_features"]["candidate_content_match_level"])
                    for row in selected
                }
            ),
        }
    feature_names = {
        tuple(sorted(row["model_features"])) for row in rows
    }
    checks = {
        "144_rows_48_per_component": len(rows) == 144
        and all(components[value]["rows"] == 48 for value in COMPONENTS),
        "48_independent_users_per_component": all(
            components[value]["users"] == 48 for value in COMPONENTS
        ),
        "24_on_24_off_per_component": all(
            components[value]["class_counts"] == {0: 24, 1: 24}
            for value in COMPONENTS
        ),
        "one_feature_schema": len(feature_names) == 3,
        "scale_stable_feature_replaces_raw_cosine": all(
            "candidate_content_match_level" in row["model_features"]
            and "candidate_state_match_score" not in row["model_features"]
            for row in rows
        ),
        "no_hard_off_fit_rows": all(
            not row["deterministic_hard_off"] for row in rows
        ),
        "no_outcome_read": all(not row["outcome_read"] for row in rows),
    }
    # Schemas legitimately differ by component because each excludes its own
    # background bit and MP alone exposes the preference subtype bit.
    checks["one_feature_schema"] = len(feature_names) == len(COMPONENTS)
    status = (
        "PASS_SCALE_STABLE_FIT_RECOMPILED"
        if all(checks.values())
        else "FAIL_SCALE_STABLE_FIT_RECOMPILE"
    )
    out_dir = root / "outputs/pm_v1_5_scale_stable_memory_fit_recompile_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "fit_rows.jsonl"
    write_jsonl(rows_path, rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "feature_protocol": SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
        "components": components,
        "checks": checks,
        "api_calls": 0,
        "human_or_external_outcomes_read": False,
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [*d3_inputs, *supplement_inputs]
        },
    }
    write_json(out_dir / "recompile_report.json", report)
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "components": report["components"],
            "checks": report["checks"],
        }
    )


if __name__ == "__main__":
    main()
