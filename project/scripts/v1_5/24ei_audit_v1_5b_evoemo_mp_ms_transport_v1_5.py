#!/usr/bin/env python3
"""Audit outcome-blind EvoEmo transport of the V1.5b MP/MS opportunity heads."""

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
from metacom_pm.v1_5_candidate_discovery import (
    V1_5B_CANDIDATE_DISCOVERY_PROTOCOL,
    discover_memory_candidates_by_source_query_v1_5b,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_source_specific_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import BOUNDED_MEMORY_COMPILER_PROTOCOL, compile_bounded_memory_with_metadata


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-evoemo-mp-ms-opportunity-transport-v1"
COMPONENTS = ("MP", "MS")
MINIMUM_COVERAGE = 0.80


def _support_spec(rows: list[dict[str, Any]], component: str, names: list[str]) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["component"] == component and row["evidence_role"] == "development_fit"]
    return {
        name: {
            "minimum": min(float(row["model_features"][name]) for row in selected),
            "maximum": max(float(row["model_features"][name]) for row in selected),
            "observed_levels": sorted({float(row["model_features"][name]) for row in selected}),
        }
        for name in names
    }


def _violations(features: dict[str, float], support: dict[str, dict[str, Any]]) -> list[str]:
    return [
        name
        for name, spec in support.items()
        if float(features[name]) < float(spec["minimum"])
        or float(features[name]) > float(spec["maximum"])
    ]


def build(*, root: Path = ROOT) -> dict[str, Any]:
    evoemo_path = root / "data/external/evo_emo.json"
    tracks_path = root / "outputs/evoemo_fixed_tracks_v1_5_v3_formal_fresh_candidate/fixed_seeker_tracks.jsonl"
    config_path = root / "configs/pm_v1_5.yaml"
    model_dir = root / "outputs/pm_v1_5b_mp_ms_opportunity_heads_v1"
    fit_report_path = model_dir / "fit_report.json"
    rows_path = model_dir / "fit_and_confirmation_rows_audit.jsonl"
    fit_report = read_json(fit_report_path)
    if fit_report["status"] != "PASS_INTERNAL_MP_MS_AWAITING_OUTCOME_BLIND_EXTERNAL_TRANSPORT":
        raise RuntimeError("V1.5b internal MP/MS qualification did not pass")
    training_rows = [dict(row) for row in iter_jsonl(rows_path)]
    feature_names = {
        component: list(fit_report["components"][component]["feature_names"])
        for component in COMPONENTS
    }
    support = {
        component: _support_spec(training_rows, component, feature_names[component])
        for component in COMPONENTS
    }
    models = {
        component: joblib.load(model_dir / f"{component.lower()}_opportunity_router_v1_5b.joblib")
        for component in COMPONENTS
    }

    config = load_config(config_path)
    turn_indices = [int(value) for value in config["external_evaluation"]["turn_indices"]]
    minimum_score = float(config["retrieval"]["memory_min_score"])
    retriever = MemoryRetriever(
        minimum_score_by_source={source: minimum_score for source in MemorySource}
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
            state = make_evo_runtime_state(
                user,
                topic,
                context,
                current,
                items,
                turn_index,
                "outcome_blind_v1_5b_mp_ms_transport",
                track_id=str(track["track_id"]),
                fixed_open_loop=True,
            )
            visible = [turn.model_dump(mode="json") for turn in state.current_session_history]
            queries = source_specific_memory_queries(
                state.current_user_text, visible, state.current_session_summary
            )
            discoveries = discover_memory_candidates_by_source_query_v1_5b(
                queries=queries,
                items=items,
                retriever=retriever,
                session_index=state.session_index,
            )
            for component in COMPONENTS:
                source = MemorySource(component)
                observation = build_source_specific_memory_opportunity_observation(
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
                features = {
                    name: float(observation["model_features"][name])
                    for name in feature_names[component]
                }
                violations = _violations(features, support[component])
                in_support = not violations
                probability = float(
                    models[component].predict_proba(
                        np.asarray([[features[name] for name in feature_names[component]]])
                    )[0, 1]
                )
                if observation["deterministic_hard_off"]:
                    action_on = False
                    reason = "DETERMINISTIC_HARD_OFF"
                elif not in_support:
                    action_on = False
                    reason = "OUT_OF_DEVELOPMENT_SUPPORT_OFF"
                else:
                    action_on = probability >= 0.5
                    reason = "V1_5B_HEAD_ON" if action_on else "V1_5B_HEAD_OFF"
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
                        "candidate_discovery_protocol": discoveries[source].descriptor["protocol"],
                        "deterministic_hard_off": observation["deterministic_hard_off"],
                        "hard_off_reasons": observation["hard_off_reasons"],
                        "development_range_in_support": in_support,
                        "out_of_support_features": violations,
                        "probability_on": probability,
                        "action_on": bool(action_on),
                        "decision_reason": reason,
                        "compiler_protocol": BOUNDED_MEMORY_COMPILER_PROTOCOL,
                        "feature_protocol": observation["protocol"],
                        "external_outcome_or_response_judgment_read": False,
                    }
                )

    by_component: dict[str, Any] = {}
    for component in COMPONENTS:
        selected = [row for row in output_rows if row["component"] == component]
        eligible = [row for row in selected if not row["deterministic_hard_off"]]
        in_support = [row for row in eligible if row["development_range_in_support"]]
        violations: Counter[str] = Counter()
        for row in selected:
            violations.update(row["out_of_support_features"])
        by_component[component] = {
            "states": len(selected),
            "users": len({row["user_id_private_audit_only"] for row in selected}),
            "candidate_present": sum(bool(row["candidate_audit"]["candidate_present"]) for row in selected),
            "exact_text_duplicates_removed": sum(int(row["candidate_audit"].get("exact_text_duplicates_removed", 0)) for row in selected),
            "deterministic_hard_off": len(selected) - len(eligible),
            "head_eligible_states_after_hard_gates": len(eligible),
            "head_eligible_in_support": len(in_support),
            "head_eligible_in_support_rate": round(len(in_support) / len(eligible), 8) if eligible else 1.0,
            "predicted_on": sum(bool(row["action_on"]) for row in selected),
            "predicted_on_rate": round(sum(bool(row["action_on"]) for row in selected) / len(selected), 8),
            "decision_reasons": dict(sorted(Counter(row["decision_reason"] for row in selected).items())),
            "out_of_support_feature_counts": dict(sorted(violations.items())),
        }

    checks = {
        "204_states_per_component": all(by_component[component]["states"] == 204 for component in COMPONENTS),
        "18_external_users_per_component": all(by_component[component]["users"] == 18 for component in COMPONENTS),
        "shared_compiler_v1_5b_discovery_and_feature_protocol": all(
            row["compiler_protocol"] == BOUNDED_MEMORY_COMPILER_PROTOCOL
            and row["candidate_discovery_protocol"] == V1_5B_CANDIDATE_DISCOVERY_PROTOCOL
            and row["feature_protocol"] == SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
            for row in output_rows
        ),
        "feature_schema_matches_frozen_heads": all(
            set(row["model_features"]) == set(feature_names[row["component"]])
            for row in output_rows
        ),
        "out_of_support_and_hard_off_fail_closed": all(
            not row["action_on"]
            for row in output_rows
            if not row["development_range_in_support"] or row["deterministic_hard_off"]
        ),
        "minimum_0_80_outcome_blind_transport_coverage_each_source": all(
            by_component[component]["head_eligible_in_support_rate"] >= MINIMUM_COVERAGE
            for component in COMPONENTS
        ),
        "no_external_outcome_or_response_judgment_read": all(
            not row["external_outcome_or_response_judgment_read"] for row in output_rows
        ),
    }
    status = (
        "PASS_V1_5B_MP_MS_REPRESENTATION_TRANSPORT_AWAITING_SYSTEM_EFFECT"
        if all(checks.values())
        else "FAIL_V1_5B_MP_MS_EXTERNAL_TRANSPORT_LIMITATION"
    )
    out_dir = root / "outputs/pm_v1_5b_evoemo_mp_ms_transport_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "external_opportunity_predictions.jsonl"
    write_jsonl(predictions_path, output_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scientific_scope": "Outcome-blind representation/support coverage and frozen decisions; not external routing accuracy or response utility.",
        "decision_distribution_is_diagnostic_not_a_promotion_target": True,
        "routing_accuracy_on_evoemo_claimed": False,
        "components": by_component,
        "checks": checks,
        "external_outcome_used": False,
        "further_retuning_from_this_report_allowed": False,
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                evoemo_path,
                tracks_path,
                config_path,
                fit_report_path,
                rows_path,
                *(model_dir / f"{component.lower()}_opportunity_router_v1_5b.joblib" for component in COMPONENTS),
            )
        },
    }
    write_json(out_dir / "transport_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "transport_report_sha256": sha256_file(out_dir / "transport_report.json"),
            "prediction_rows_sha256": sha256_file(predictions_path),
            "external_outcome_used": False,
            "further_retuning_allowed": False,
        },
    )
    return report


def main() -> None:
    report = build()
    print({"protocol": report["protocol"], "status": report["status"], "components": report["components"], "checks": report["checks"]})


if __name__ == "__main__":
    main()
