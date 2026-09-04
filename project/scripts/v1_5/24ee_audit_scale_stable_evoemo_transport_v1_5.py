#!/usr/bin/env python3
"""Run the one-shot outcome-blind EvoEmo scale-stable transport audit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from metacom_pm.config import load_config
from metacom_pm.contracts import MemorySource
from metacom_pm.evoemo import _fixed_context_before_turn, load_evoemo, make_evo_runtime_state
from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.v1_5_candidate_discovery import discover_memory_candidates_by_source_query
from metacom_pm.v1_5_memory_opportunity_features import (
    SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_scale_stable_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import BOUNDED_MEMORY_COMPILER_PROTOCOL, compile_bounded_memory_with_metadata


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-evoemo-scale-stable-memory-transport-v1"
COMPONENTS = ("MP", "MS", "ME")
MINIMUM_TRANSPORT_COVERAGE = 0.80


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _support_spec(rows: list[dict[str, Any]], component: str, feature_names: list[str]) -> dict[str, Any]:
    selected = [row for row in rows if row["component"] == component]
    return {
        name: (
            {"kind": "registered_levels", "values": sorted({float(row["model_features"][name]) for row in selected})}
            if name == "candidate_content_match_level"
            else {
                "kind": "numeric_range",
                "minimum": min(float(row["model_features"][name]) for row in selected),
                "maximum": max(float(row["model_features"][name]) for row in selected),
            }
        )
        for name in feature_names
    }


def _violations(values: dict[str, float], support: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    for name, specification in support.items():
        value = float(values[name])
        if specification["kind"] == "registered_levels":
            if value not in specification["values"]:
                violations.append(name)
        elif value < specification["minimum"] or value > specification["maximum"]:
            violations.append(name)
    return violations


def build(*, root: Path = ROOT) -> dict[str, Any]:
    evoemo_path = root / "data/external/evo_emo.json"
    tracks_path = root / "outputs/evoemo_fixed_tracks_v1_5_v3_formal_fresh_candidate/fixed_seeker_tracks.jsonl"
    config_path = root / "configs/pm_v1_5.yaml"
    model_dir = root / "outputs/pm_v1_5_scale_stable_memory_opportunity_heads_v1"
    fit_report_path = model_dir / "fit_report.json"
    fit_rows_path = model_dir / "fit_rows_audit.jsonl"
    fit_report = read_json(fit_report_path)
    promoted = {
        component: fit_report["components"][component]["status"]
        == "PASS_PROMOTED"
        for component in COMPONENTS
    }
    fit_rows = _rows(fit_rows_path)
    config = load_config(config_path)
    turn_indices = [int(value) for value in config["external_evaluation"]["turn_indices"]]
    retriever = MemoryRetriever(minimum_score_by_source={source: float(config["retrieval"]["memory_min_score"]) for source in MemorySource})
    users = load_evoemo(evoemo_path)
    users_by_id = {str(user["id"]): user for user in users}
    topics = {(str(user["id"]), int(topic["idx"])): topic for user in users for topic in (user.get("subsequent_topics") or [])}
    compiled = {user_id: compile_bounded_memory_with_metadata(user) for user_id, user in users_by_id.items()}
    models = {component: joblib.load(model_dir / f"{component.lower()}_opportunity_router.joblib") for component in COMPONENTS}
    feature_names = {component: list(fit_report["components"][component]["feature_names"]) for component in COMPONENTS}
    support = {component: _support_spec(fit_rows, component, feature_names[component]) for component in COMPONENTS}

    output_rows: list[dict[str, Any]] = []
    for track in iter_jsonl(tracks_path):
        user_id = str(track["user_id"])
        topic_index = int(track["topic_index"])
        user = users_by_id[user_id]
        topic = topics[(user_id, topic_index)]
        items, _session_docs, metadata = compiled[user_id]
        for turn_index in turn_indices:
            current = str(track["seeker_turns"][turn_index - 1])
            context = _fixed_context_before_turn(dict(track), turn_index)
            state = make_evo_runtime_state(user, topic, context, current, items, turn_index, "outcome_blind_scale_stable_transport", track_id=str(track["track_id"]), fixed_open_loop=True)
            visible = [turn.model_dump(mode="json") for turn in state.current_session_history]
            queries = source_specific_memory_queries(state.current_user_text, visible, state.current_session_summary)
            discoveries = discover_memory_candidates_by_source_query(queries=queries, items=items, retriever=retriever, session_index=state.session_index)
            for source in MemorySource:
                component = source.value
                observation = build_scale_stable_memory_opportunity_observation(
                    candidate=discoveries[source],
                    catalog_items=items,
                    catalog_user_id=user_id,
                    current_user_id=state.user_id,
                    current_session_index=state.session_index,
                    current_user_text=state.current_user_text,
                    visible_dialogue=visible,
                    source_metadata=metadata,
                    background_action="M0+R0",
                )
                features = {name: float(observation["model_features"][name]) for name in feature_names[component]}
                violations = _violations(features, support[component])
                in_support = not violations
                probability = float(models[component].predict_proba(np.asarray([[features[name] for name in feature_names[component]]]))[0, 1])
                if observation["deterministic_hard_off"]:
                    action_on, reason = False, "DETERMINISTIC_HARD_OFF"
                elif not promoted[component]:
                    action_on, reason = False, "HEAD_NOT_PROMOTED_OFF"
                elif not in_support:
                    action_on, reason = False, "OUT_OF_DEVELOPMENT_SUPPORT_OFF"
                else:
                    action_on = probability >= 0.5
                    reason = "FROZEN_HEAD_ON" if action_on else "FROZEN_HEAD_OFF"
                output_rows.append(
                    {
                        "protocol": PROTOCOL,
                        "state_id": state.state_id,
                        "user_id_private_audit_only": user_id,
                        "topic_index_private_audit_only": topic_index,
                        "turn_index": turn_index,
                        "component": component,
                        "model_features": features,
                        "candidate_audit": observation["candidate_audit"],
                        "deterministic_hard_off": observation["deterministic_hard_off"],
                        "hard_off_reasons": observation["hard_off_reasons"],
                        "development_support": support[component],
                        "development_range_in_support": in_support,
                        "out_of_support_features": violations,
                        "probability_on": probability,
                        "action_on": bool(action_on),
                        "decision_reason": reason,
                        "compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
                        "feature_protocol": SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
                        "external_outcome_or_response_judgment_read": False,
                    }
                )

    by_component: dict[str, Any] = {}
    for component in COMPONENTS:
        selected = [row for row in output_rows if row["component"] == component]
        reasons = Counter(row["decision_reason"] for row in selected)
        violations: Counter[str] = Counter()
        for row in selected:
            violations.update(row["out_of_support_features"])
        in_support = sum(bool(row["development_range_in_support"]) for row in selected)
        eligible = [row for row in selected if not row["deterministic_hard_off"]]
        eligible_in_support = sum(
            bool(row["development_range_in_support"]) for row in eligible
        )
        by_component[component] = {
            "head_promoted": promoted[component],
            "states": len(selected),
            "users": len({row["user_id_private_audit_only"] for row in selected}),
            "candidate_present": sum(bool(row["candidate_audit"]["candidate_present"]) for row in selected),
            "deterministic_hard_off": sum(bool(row["deterministic_hard_off"]) for row in selected),
            "development_range_in_support": in_support,
            "development_range_in_support_rate": round(in_support / len(selected), 8),
            "head_eligible_states_after_hard_gates": len(eligible),
            "head_eligible_in_support": eligible_in_support,
            "head_eligible_in_support_rate": round(
                eligible_in_support / len(eligible), 8
            )
            if eligible
            else 1.0,
            "predicted_on": sum(bool(row["action_on"]) for row in selected),
            "predicted_on_rate": round(sum(bool(row["action_on"]) for row in selected) / len(selected), 8),
            "decision_reasons": dict(sorted(reasons.items())),
            "out_of_support_feature_counts": dict(sorted(violations.items())),
        }
    checks = {
        "204_states_per_component": all(by_component[value]["states"] == 204 for value in COMPONENTS),
        "18_external_users_per_component": all(by_component[value]["users"] == 18 for value in COMPONENTS),
        "same_compiler_and_scale_stable_feature_protocol": all(row["compiler_protocol"] == BOUNDED_MEMORY_COMPILER_PROTOCOL and row["feature_protocol"] == SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL for row in output_rows),
        "feature_schema_matches_frozen_heads": all(set(row["model_features"]) == set(feature_names[row["component"]]) for row in output_rows),
        "out_of_support_and_hard_off_fail_closed": all(not row["action_on"] for row in output_rows if not row["development_range_in_support"] or row["deterministic_hard_off"]),
        # Deterministic hard-off states never reach a learned head and are not
        # part of its support denominator.  They remain fully reported above.
        "minimum_0_80_outcome_blind_transport_coverage_each_source": all(by_component[value]["head_eligible_in_support_rate"] >= MINIMUM_TRANSPORT_COVERAGE for value in COMPONENTS),
        "no_external_outcome_or_response_judgment_read": all(not row["external_outcome_or_response_judgment_read"] for row in output_rows),
    }
    if all(checks.values()) and all(promoted.values()):
        status = "PASS_SCALE_STABLE_EXTERNAL_TRANSPORT_FREEZE"
    elif all(checks.values()) and any(promoted.values()):
        status = "PASS_REPRESENTATION_TRANSPORT_PARTIAL_HEADS_ONLY"
    else:
        status = "FAIL_SCALE_STABLE_EXTERNAL_TRANSPORT_REPORT_LIMITATION"
    out_dir = root / "outputs/pm_v1_5_evoemo_scale_stable_memory_transport_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "external_opportunity_predictions.jsonl"
    write_jsonl(rows_path, output_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scientific_scope": "Outcome-blind mechanism and representation transport only; coverage is not external routing accuracy or utility.",
        "repair_budget": "ONE_REPRESENTATION_REPAIR_CONSUMED_NO_MORE_RETUNING",
        "promoted_heads": [
            component for component in COMPONENTS if promoted[component]
        ],
        "unpromoted_heads_fail_closed": [
            component for component in COMPONENTS if not promoted[component]
        ],
        "components": by_component,
        "checks": checks,
        "inputs": {str(path.relative_to(root)): sha256_file(path) for path in (evoemo_path, tracks_path, config_path, fit_report_path, fit_rows_path, *(model_dir / f"{component.lower()}_opportunity_router.joblib" for component in COMPONENTS))},
    }
    write_json(out_dir / "transport_report.json", report)
    write_json(out_dir / "freeze_manifest.json", {"protocol": PROTOCOL, "status": status, "transport_report_sha256": sha256_file(out_dir / "transport_report.json"), "prediction_rows_sha256": sha256_file(rows_path), "external_outcome_used": False, "further_representation_retuning_allowed": False})
    return report


def main() -> None:
    report = build()
    print({"protocol": report["protocol"], "status": report["status"], "components": report["components"], "checks": report["checks"]})


if __name__ == "__main__":
    main()
