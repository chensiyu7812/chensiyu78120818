#!/usr/bin/env python3
"""Freeze six-policy routes on the final panels without generating replies."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from metacom_pm.contracts import RuntimeState
from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_strategy_rag_repair import repaired_observable_opportunity_flags, repaired_rank_applicable_v4_cards
from metacom_pm.v1_5b_policy_runtime import COMPONENTS, ComponentOpportunity, route_policy


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-final-outcome-blind-routing-freeze-v1"
POLICIES = (
    "always_off",
    "fixed_high_resource",
    "transparent_rule",
    "learned_pm_full",
    "learned_pm_conservative",
    "cost_matched_fixed",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _dialogue(state: RuntimeState) -> list[dict[str, str]]:
    rows = [
        {
            "speaker": "seeker"
            if str(getattr(turn.role, "value", turn.role)) == "user"
            else "supporter",
            "content": turn.content,
        }
        for turn in state.current_session_history
    ]
    rows.append({"speaker": "seeker", "content": state.current_user_text})
    return rows


def _query(dialogue: list[dict[str, str]], flags: dict[str, Any]) -> str:
    seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool)
        and value
        and key not in {"substantive", "pure_phatic", "routine_closing", "active_high_stakes", "explicit_stop", "ordinary_rag_hard_off"}
    ]
    return (
        "Represent this sentence for searching relevant passages: Choose one safe, "
        "topic-agnostic emotional-support technique. Observable cues: "
        + (", ".join(cues) or "none")
        + ". Recent seeker context: "
        + " ".join(seekers[-3:])
    )


def _rs_observation(
    *, state: RuntimeState, cards: list[dict[str, Any]], model: Any, names: list[str]
) -> tuple[ComponentOpportunity, dict[str, Any]]:
    dialogue = _dialogue(state)
    seekers = [row["content"] for row in dialogue if row["speaker"] == "seeker"]
    recent = " ".join(seekers[-3:])
    flags = repaired_observable_opportunity_flags(
        current_user_text=state.current_user_text,
        recent_user_text=recent,
        visible_dialogue=dialogue,
    )
    ranked = repaired_rank_applicable_v4_cards(
        query=_query(dialogue, flags),
        current_user_text=state.current_user_text,
        recent_user_text=recent,
        visible_dialogue=dialogue,
        cards=cards,
    )
    vector = np.asarray([[float(bool(flags.get(name, False))) for name in names]])
    probability = float(model.predict_proba(vector)[0, 1])
    opportunity = ComponentOpportunity(
        component="RS",
        candidate_present=bool(ranked),
        hard_gate_pass=bool(ranked),
        transparent_rule_on=bool(ranked),
        learned_in_support=True,
        learned_on=probability >= 0.5,
        learned_probability=probability,
        decision_reason="FROZEN_HEAD_ON" if probability >= 0.5 else "FROZEN_HEAD_OFF",
    )
    return opportunity, {
        "candidate_count_private_not_model_input": len(ranked),
        "selected_card_id_private_not_model_input": ranked[0]["card_id"] if ranked else None,
        "probability_on": probability,
        "hard_off_reasons": list(flags.get("ordinary_rag_hard_off_reasons") or []),
    }


def _memory_opportunity(row: dict[str, Any]) -> ComponentOpportunity:
    features = dict(row["model_features"])
    transparent = bool(
        not row["deterministic_hard_off"]
        and float(features["candidate_content_match_level"]) >= 0.5
        and float(features["candidate_grounding_or_nonredundancy_score"]) >= 0.999
    )
    return ComponentOpportunity(
        component=str(row["component"]),
        candidate_present=bool(row["candidate_audit"]["candidate_present"]),
        hard_gate_pass=not bool(row["deterministic_hard_off"]),
        transparent_rule_on=transparent,
        learned_in_support=bool(row["development_range_in_support"]),
        learned_on=bool(row["action_on"]),
        learned_probability=float(row["probability_on"]),
        decision_reason=str(row["decision_reason"]),
    )


def build(*, root: Path = ROOT) -> dict[str, Any]:
    panel_dir = root / "outputs/pm_v1_5b_final_external_panel_v1"
    esconv_path = panel_dir / "esconv_panel_private.jsonl"
    evoemo_path = panel_dir / "evoemo_panel_private.jsonl"
    panel_report_path = panel_dir / "panel_freeze_report.json"
    mpms_path = root / "outputs/pm_v1_5b_evoemo_mp_ms_transport_v1/external_opportunity_predictions.jsonl"
    me_path = root / "outputs/pm_v1_5_evoemo_scale_stable_memory_transport_v1/external_opportunity_predictions.jsonl"
    bank_path = root / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    rs_dir = root / "outputs/pm_v1_5_same_bank_rs_opportunity_router_fit_v1"
    rs_report_path = rs_dir / "fit_report.json"
    cost_report_path = root / "outputs/pm_v1_5b_internal_cost_matched_fixed_v1/cost_match_report.json"
    compiler_report_path = root / "outputs/pm_v1_5b_final_policy_compiler_audit_v1/compiler_audit_report.json"
    if read_json(panel_report_path)["status"] != "PASS_FINAL_EXTERNAL_PANELS_FROZEN":
        raise RuntimeError("final external panels are not frozen")
    cost_report = read_json(cost_report_path)
    cost_action = str(cost_report["selected_action_id"])
    compiler_report = read_json(compiler_report_path)
    if compiler_report["cost_matched_action_id"] != cost_action:
        raise RuntimeError("compiler and internal cost-match action disagree")

    cards = _rows(bank_path)
    rs_report = read_json(rs_report_path)
    rs_names = list(rs_report["feature_names"])
    rs_model = joblib.load(rs_dir / "rs_opportunity_router.joblib")
    mpms = {
        (str(row["state_id"]), str(row["component"])): row
        for row in _rows(mpms_path)
    }
    me = {
        (str(row["state_id"]), str(row["component"])): row
        for row in _rows(me_path)
        if row["component"] == "ME"
    }

    panels = [*_rows(esconv_path), *_rows(evoemo_path)]
    route_rows: list[dict[str, Any]] = []
    for panel in panels:
        state = RuntimeState.model_validate(panel["runtime_state"])
        domain = str(panel["domain"])
        observations: dict[str, ComponentOpportunity] = {}
        memory_audit: dict[str, Any] = {}
        if domain == "EvoEmo":
            for component in ("MP", "MS"):
                source = mpms[(state.state_id, component)]
                observations[component] = _memory_opportunity(source)
                memory_audit[component] = {
                    "probability_on": source["probability_on"],
                    "decision_reason": source["decision_reason"],
                    "candidate_audit": source["candidate_audit"],
                }
            source = me[(state.state_id, "ME")]
            observations["ME"] = _memory_opportunity(source)
            memory_audit["ME"] = {
                "probability_on": source["probability_on"],
                "decision_reason": source["decision_reason"],
                "candidate_audit": source["candidate_audit"],
            }
        else:
            for component in ("MP", "MS", "ME"):
                observations[component] = ComponentOpportunity(
                    component=component,
                    candidate_present=False,
                    hard_gate_pass=False,
                    transparent_rule_on=False,
                    learned_in_support=False,
                    learned_on=False,
                    learned_probability=None,
                    decision_reason="STRUCTURALLY_UNAVAILABLE_ESCONV_SINGLE_SESSION",
                )
                memory_audit[component] = {"structurally_unavailable": True}
        observations["RS"], rs_audit = _rs_observation(
            state=state, cards=cards, model=rs_model, names=rs_names
        )
        routes = {
            policy: route_policy(
                policy=policy,
                opportunities=observations,
                cost_matched_action_id=cost_action,
            )
            for policy in POLICIES
        }
        route_rows.append(
            {
                "protocol": PROTOCOL,
                "domain": domain,
                "panel_id": panel["panel_id"],
                "state_id": state.state_id,
                "group_id_private_analysis_only": panel.get("dialogue_id_private_analysis_only") or panel.get("user_id_private_analysis_only"),
                "policy_routes": {
                    name: {
                        "requested_action_id": route.action_id,
                        "requested_bits": dict(route.requested_bits),
                        "decision_reasons": dict(route.reasons),
                    }
                    for name, route in routes.items()
                },
                "unique_requested_action_ids": sorted({route.action_id for route in routes.values()}),
                "memory_audit_private": memory_audit,
                "rs_audit_private": rs_audit,
                "external_response_quality_risk_or_judge_outcome_read": False,
            }
        )

    out_dir = root / "outputs/pm_v1_5b_final_outcome_blind_routes_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    routes_path = out_dir / "policy_routes_private.jsonl"
    write_jsonl(routes_path, route_rows)
    assignments = len(route_rows) * len(POLICIES)
    unique_calls = sum(len(row["unique_requested_action_ids"]) for row in route_rows)
    domain_summary: dict[str, Any] = {}
    for domain in ("ESConv", "EvoEmo"):
        selected = [row for row in route_rows if row["domain"] == domain]
        domain_summary[domain] = {
            "states": len(selected),
            "policy_assignments": len(selected) * len(POLICIES),
            "unique_state_action_calls_after_exact_dedup": sum(len(row["unique_requested_action_ids"]) for row in selected),
            "unique_actions_per_state_distribution": dict(sorted(Counter(len(row["unique_requested_action_ids"]) for row in selected).items())),
            "policy_action_distributions": {
                policy: dict(sorted(Counter(row["policy_routes"][policy]["requested_action_id"] for row in selected).items()))
                for policy in POLICIES
            },
        }
    checks = {
        "373_frozen_states": len(route_rows) == 373,
        "six_policy_routes_per_state": all(set(row["policy_routes"]) == set(POLICIES) for row in route_rows),
        "same_state_same_action_deduplicated_before_generation": unique_calls <= assignments,
        "esconv_memory_always_structurally_off": all(
            not any(row["policy_routes"][policy]["requested_bits"][component] for policy in POLICIES for component in ("MP", "MS", "ME"))
            for row in route_rows if row["domain"] == "ESConv"
        ),
        "no_external_response_quality_risk_or_judge_outcome_read": all(not row["external_response_quality_risk_or_judge_outcome_read"] for row in route_rows),
    }
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_FINAL_OUTCOME_BLIND_ROUTES_FROZEN" if all(checks.values()) else "FAIL_FINAL_ROUTING_FREEZE",
        "policies": list(POLICIES),
        "cost_matched_action_id": cost_action,
        "states": len(route_rows),
        "policy_assignments": assignments,
        "unique_state_action_calls_after_exact_dedup": unique_calls,
        "generation_calls_saved_by_exact_action_dedup": assignments - unique_calls,
        "maximum_resource_free_fallback_calls": sum(
            action != "M0+R0"
            for row in route_rows
            for action in row["unique_requested_action_ids"]
        ),
        "domain_summary": domain_summary,
        "checks": checks,
        "response_generation_calls": 0,
        "external_outcome_used": False,
        "further_route_tuning_from_external_results_allowed": False,
        "inputs": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (esconv_path, evoemo_path, panel_report_path, mpms_path, me_path, bank_path, rs_report_path, rs_dir / "rs_opportunity_router.joblib", cost_report_path, compiler_report_path)
        },
        "routes_sha256": sha256_file(routes_path),
    }
    write_json(out_dir / "routing_freeze_report.json", report)
    return report


def main() -> None:
    report = build()
    print({
        "protocol": report["protocol"],
        "status": report["status"],
        "states": report["states"],
        "policy_assignments": report["policy_assignments"],
        "unique_state_action_calls_after_exact_dedup": report["unique_state_action_calls_after_exact_dedup"],
        "generation_calls_saved_by_exact_action_dedup": report["generation_calls_saved_by_exact_action_dedup"],
        "domain_summary": report["domain_summary"],
    })


if __name__ == "__main__":
    main()
