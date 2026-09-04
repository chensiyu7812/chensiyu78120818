#!/usr/bin/env python3
"""Recompile EvoEmo memory candidates through the frozen Step-1 heads.

This is an outcome-blind transport audit.  Every external user keeps their
own causal memory catalog.  The exact internal production compiler,
source-specific query builder, retriever, opportunity feature bridge, and
frozen MP/MS/ME heads are reused.  External answers, response judgments,
related-session annotations, topic outcomes, and reference answers are never
read.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from metacom_pm.config import load_config
from metacom_pm.contracts import MemorySource
from metacom_pm.evoemo import (
    _fixed_context_before_turn,
    load_evoemo,
    make_evo_runtime_state,
)
from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.v1_5_candidate_discovery import (
    discover_memory_candidates_by_source_query,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    compile_bounded_memory_with_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-evoemo-memory-opportunity-production-recompile-v1"
COMPONENTS = ("MP", "MS", "ME")
BACKGROUND_ACTION = "M0+R0"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _fit_support(
    rows: list[dict[str, Any]], component: str, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["component"] == component]
    if not selected:
        raise RuntimeError(f"no fit rows for {component}")
    values = np.asarray(
        [
            [float(row["model_features"][name]) for name in feature_names]
            for row in selected
        ],
        dtype=float,
    )
    return values.min(axis=0), values.max(axis=0)


def _load_fixed_context_helper():
    """Bind the same fixed-track context helper used by the prior audit."""

    # The leading underscore reflects an existing experiment helper, not a
    # second context implementation.  Keep this wrapper so a future public
    # helper can replace it in one place.
    return _fixed_context_before_turn


def build(*, root: Path = ROOT) -> dict[str, Any]:
    evoemo_path = root / "data/external/evo_emo.json"
    tracks_path = (
        root
        / "outputs/evoemo_fixed_tracks_v1_5_v3_formal_fresh_candidate/fixed_seeker_tracks.jsonl"
    )
    config_path = root / "configs/pm_v1_5.yaml"
    model_dir = root / "outputs/pm_v1_5_real_text_memory_opportunity_heads_v1"
    fit_report_path = model_dir / "fit_report.json"
    fit_rows_path = model_dir / "fit_rows_audit.jsonl"

    fit_report = read_json(fit_report_path)
    if fit_report["status"] != "PASS_THREE_MEMORY_HEADS_PROMOTED":
        raise RuntimeError("frozen memory heads are not promoted")
    fit_rows = _rows(fit_rows_path)
    config = load_config(config_path)
    turn_indices = [
        int(value) for value in config["external_evaluation"]["turn_indices"]
    ]
    minimum_score = float(config["retrieval"]["memory_min_score"])
    retriever = MemoryRetriever(
        minimum_score_by_source={
            source: minimum_score for source in MemorySource
        }
    )

    users = load_evoemo(evoemo_path)
    users_by_id = {str(user["id"]): user for user in users}
    topics = {
        (str(user["id"]), int(topic["idx"])): topic
        for user in users
        for topic in (user.get("subsequent_topics") or [])
    }
    compiled = {
        user_id: compile_bounded_memory_with_metadata(user)
        for user_id, user in users_by_id.items()
    }
    models = {
        component: joblib.load(
            model_dir / f"{component.lower()}_opportunity_router.joblib"
        )
        for component in COMPONENTS
    }
    feature_names = {
        component: list(fit_report["components"][component]["feature_names"])
        for component in COMPONENTS
    }
    support = {
        component: _fit_support(
            fit_rows, component, feature_names[component]
        )
        for component in COMPONENTS
    }
    match_floors = {
        component: fit_report["components"][component]["fixed_hyperparameters"][
            "final_fit_candidate_match_floor"
        ]
        for component in COMPONENTS
    }

    output_rows: list[dict[str, Any]] = []
    fixed_context_before_turn = _load_fixed_context_helper()
    for track in iter_jsonl(tracks_path):
        user_id = str(track["user_id"])
        topic_index = int(track["topic_index"])
        user = users_by_id[user_id]
        topic = topics[(user_id, topic_index)]
        items, _session_docs, metadata = compiled[user_id]
        for turn_index in turn_indices:
            current = str(track["seeker_turns"][turn_index - 1])
            context = fixed_context_before_turn(dict(track), turn_index)
            state = make_evo_runtime_state(
                user,
                topic,
                context,
                current,
                items,
                turn_index,
                "outcome_blind_memory_opportunity_transport",
                track_id=str(track["track_id"]),
                fixed_open_loop=True,
            )
            visible_dialogue = [
                turn.model_dump(mode="json")
                for turn in state.current_session_history
            ]
            queries = source_specific_memory_queries(
                state.current_user_text,
                visible_dialogue,
                state.current_session_summary,
            )
            discoveries = discover_memory_candidates_by_source_query(
                queries=queries,
                items=items,
                retriever=retriever,
                session_index=state.session_index,
            )
            for source in MemorySource:
                component = source.value
                observation = build_memory_opportunity_observation(
                    candidate=discoveries[source],
                    catalog_items=items,
                    catalog_user_id=user_id,
                    current_user_id=state.user_id,
                    current_session_index=state.session_index,
                    current_user_text=state.current_user_text,
                    visible_dialogue=visible_dialogue,
                    source_metadata=metadata,
                    background_action=BACKGROUND_ACTION,
                )
                names = feature_names[component]
                values = np.asarray(
                    [float(observation["model_features"][name]) for name in names],
                    dtype=float,
                )
                minimum, maximum = support[component]
                violations = [
                    name
                    for name, value, low, high in zip(
                        names, values, minimum, maximum, strict=True
                    )
                    if value < low or value > high
                ]
                in_support = not violations
                probability = float(
                    models[component].predict_proba(values.reshape(1, -1))[0, 1]
                )
                floor = match_floors[component]
                below_floor = bool(
                    floor is not None
                    and float(
                        observation["model_features"][
                            "candidate_state_match_score"
                        ]
                    )
                    < float(floor)
                )
                if observation["deterministic_hard_off"]:
                    action_on = False
                    decision_reason = "DETERMINISTIC_HARD_OFF"
                elif not in_support:
                    action_on = False
                    decision_reason = "OUT_OF_DEVELOPMENT_SUPPORT_OFF"
                elif below_floor:
                    action_on = False
                    decision_reason = "TRAIN_DERIVED_MATCH_FLOOR_OFF"
                else:
                    action_on = probability >= 0.5
                    decision_reason = (
                        "FROZEN_HEAD_ON" if action_on else "FROZEN_HEAD_OFF"
                    )
                output_rows.append(
                    {
                        "protocol": PROTOCOL,
                        "state_id": state.state_id,
                        "user_id_private_audit_only": user_id,
                        "topic_index_private_audit_only": topic_index,
                        "turn_index": turn_index,
                        "component": component,
                        "model_features": observation["model_features"],
                        "candidate_audit": observation["candidate_audit"],
                        "deterministic_hard_off": observation[
                            "deterministic_hard_off"
                        ],
                        "hard_off_reasons": observation["hard_off_reasons"],
                        "development_range_in_support": in_support,
                        "out_of_support_features": violations,
                        "probability_on": probability,
                        "train_derived_match_floor": floor,
                        "below_train_derived_match_floor": below_floor,
                        "action_on": bool(action_on),
                        "decision_reason": decision_reason,
                        "compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
                        "feature_protocol": MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
                        "external_outcome_or_response_judgment_read": False,
                    }
                )

    by_component: dict[str, Any] = {}
    for component in COMPONENTS:
        rows = [row for row in output_rows if row["component"] == component]
        reasons = Counter(str(row["decision_reason"]) for row in rows)
        violations: Counter[str] = Counter()
        for row in rows:
            violations.update(row["out_of_support_features"])
        by_component[component] = {
            "states": len(rows),
            "users": len({row["user_id_private_audit_only"] for row in rows}),
            "candidate_present": sum(
                bool(row["candidate_audit"]["candidate_present"]) for row in rows
            ),
            "deterministic_hard_off": sum(
                bool(row["deterministic_hard_off"]) for row in rows
            ),
            "development_range_in_support": sum(
                bool(row["development_range_in_support"]) for row in rows
            ),
            "development_range_in_support_rate": round(
                sum(bool(row["development_range_in_support"]) for row in rows)
                / len(rows),
                8,
            ),
            "predicted_on": sum(bool(row["action_on"]) for row in rows),
            "predicted_on_rate": round(
                sum(bool(row["action_on"]) for row in rows) / len(rows), 8
            ),
            "decision_reasons": dict(sorted(reasons.items())),
            "out_of_support_feature_counts": dict(sorted(violations.items())),
        }

    checks = {
        "204_states_per_component": all(
            by_component[component]["states"] == 204
            for component in COMPONENTS
        ),
        "18_external_users_per_component": all(
            by_component[component]["users"] == 18
            for component in COMPONENTS
        ),
        "same_compiler_protocol": all(
            row["compiler_protocol"] == BOUNDED_MEMORY_COMPILER_PROTOCOL
            for row in output_rows
        ),
        "same_feature_protocol": all(
            row["feature_protocol"] == MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
            for row in output_rows
        ),
        "feature_schema_matches_frozen_heads": all(
            set(row["model_features"]) == set(feature_names[row["component"]])
            for row in output_rows
        ),
        "out_of_support_and_hard_off_fail_closed": all(
            not row["action_on"]
            for row in output_rows
            if (
                not row["development_range_in_support"]
                or row["deterministic_hard_off"]
            )
        ),
        "no_external_outcome_or_response_judgment_read": all(
            not row["external_outcome_or_response_judgment_read"]
            for row in output_rows
        ),
    }
    status = (
        "PASS_OUTCOME_BLIND_EXTERNAL_FEATURE_TRANSPORT"
        if all(checks.values())
        else "FAIL_EXTERNAL_FEATURE_TRANSPORT"
    )
    out_dir = root / "outputs/pm_v1_5_evoemo_memory_opportunity_recompile_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "external_opportunity_predictions.jsonl"
    write_jsonl(rows_path, output_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scientific_scope": (
            "Mechanism and feature transport only. Predicted-on rates are not "
            "external accuracy or utility because no external routing gold or "
            "response outcome was read."
        ),
        "components": by_component,
        "checks": checks,
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                evoemo_path,
                tracks_path,
                config_path,
                fit_report_path,
                fit_rows_path,
                *(
                    model_dir / f"{component.lower()}_opportunity_router.joblib"
                    for component in COMPONENTS
                ),
            )
        },
    }
    write_json(out_dir / "transport_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "transport_report_sha256": sha256_file(
                out_dir / "transport_report.json"
            ),
            "prediction_rows_sha256": sha256_file(rows_path),
            "external_outcome_used": False,
        },
    )
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "components": report["components"],
        }
    )


if __name__ == "__main__":
    main()
