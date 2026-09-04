#!/usr/bin/env python3
"""Promote untouched states after the external execution-format pilot.

The first run exposed only structured trace/schema incompatibility.  Every
ESConv panel touched by a physical call is excluded wholesale.  No response
preference, quality, risk, or policy outcome is used to select replacements;
all remaining ESConv panels and every untouched EvoEmo panel are retained.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-post-execution-format-pilot-final-freeze-v2"
INPUT_USD_PER_MTOK = 0.15
OUTPUT_USD_PER_MTOK = 0.60


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _usd(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * INPUT_USD_PER_MTOK / 1_000_000 + output_tokens * OUTPUT_USD_PER_MTOK / 1_000_000


def build(*, root: Path = ROOT) -> dict[str, Any]:
    panel_v1 = root / "outputs/pm_v1_5b_final_external_panel_v1"
    routes_v1 = root / "outputs/pm_v1_5b_final_outcome_blind_routes_v1"
    plan_v1 = root / "outputs/pm_v1_5b_final_system_generation_v1_candidate"
    pilot_dir = root / "outputs/pm_v1_5b_final_system_generation_v1_execution"
    outcomes_path = pilot_dir / "generation_outcomes_private.jsonl"
    old_plan_path = plan_v1 / "call_plan_private.jsonl"
    outcomes = _rows(outcomes_path)
    old_calls = _rows(old_plan_path)
    if not outcomes:
        raise RuntimeError("no execution-format pilot outcomes found")
    completed_call_ids = {str(row["call_id"]) for row in outcomes}
    touched_panels = {str(row["panel_id"]) for row in outcomes}
    if {row["domain"] for row in outcomes} != {"ESConv"}:
        raise RuntimeError("compatibility pilot unexpectedly touched EvoEmo")

    esconv_v1_path = panel_v1 / "esconv_panel_private.jsonl"
    evoemo_v1_path = panel_v1 / "evoemo_panel_private.jsonl"
    esconv_v2 = [row for row in _rows(esconv_v1_path) if row["panel_id"] not in touched_panels]
    evoemo_v2 = _rows(evoemo_v1_path)
    panel_v2 = root / "outputs/pm_v1_5b_final_external_panel_v2"
    panel_v2.mkdir(parents=True, exist_ok=True)
    esconv_v2_path = panel_v2 / "esconv_panel_private.jsonl"
    evoemo_v2_path = panel_v2 / "evoemo_panel_private.jsonl"
    write_jsonl(esconv_v2_path, esconv_v2)
    write_jsonl(evoemo_v2_path, evoemo_v2)

    route_v1_path = routes_v1 / "policy_routes_private.jsonl"
    route_v2_rows = [row for row in _rows(route_v1_path) if row["panel_id"] not in touched_panels]
    routes_v2 = root / "outputs/pm_v1_5b_final_outcome_blind_routes_v2"
    routes_v2.mkdir(parents=True, exist_ok=True)
    route_v2_path = routes_v2 / "policy_routes_private.jsonl"
    write_jsonl(route_v2_path, route_v2_rows)

    call_v2_rows = [row for row in old_calls if row["panel_id"] not in touched_panels]
    if completed_call_ids & {str(row["call_id"]) for row in call_v2_rows}:
        raise RuntimeError("a physically touched logical call leaked into final V2")
    plan_v2 = root / "outputs/pm_v1_5b_final_system_generation_v2_candidate"
    plan_v2.mkdir(parents=True, exist_ok=True)
    call_v2_path = plan_v2 / "call_plan_private.jsonl"
    write_jsonl(call_v2_path, call_v2_rows)

    policies = tuple(next(iter(route_v2_rows))["policy_routes"])
    policy_assignments = len(route_v2_rows) * len(policies)
    routing_report = {
        "protocol": PROTOCOL,
        "status": "PASS_FINAL_V2_OUTCOME_BLIND_ROUTES_FROZEN",
        "states": len(route_v2_rows),
        "policy_assignments": policy_assignments,
        "unique_state_action_calls_after_exact_dedup": len(call_v2_rows),
        "generation_calls_saved_by_exact_action_dedup": policy_assignments - len(call_v2_rows),
        "domain_states": dict(sorted(Counter(row["domain"] for row in route_v2_rows).items())),
        "external_response_quality_risk_or_judge_used_for_route": False,
        "routes_sha256": sha256_file(route_v2_path),
        "source_routes_v1_sha256": sha256_file(route_v1_path),
    }
    write_json(routes_v2 / "routing_freeze_report.json", routing_report)

    primary_input = sum(int(row["input_token_upper_bound"]) for row in call_v2_rows)
    primary_output = sum(int(row["primary_output_token_cap"]) for row in call_v2_rows)
    fallback_input = sum(int(row["maximum_fallback_input_token_upper_bound"]) for row in call_v2_rows)
    fallback_output = sum(int(row["maximum_fallback_output_token_cap"]) for row in call_v2_rows)
    generation_preflight = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_FINAL_V2_PAID_GENERATION_REVIEW",
        "generator": read_json(plan_v1 / "generation_preflight.json")["generator"],
        "api_key_env": "NVIDIA_API_KEY",
        "planned_primary_calls": len(call_v2_rows),
        "maximum_fallback_calls": sum(bool(row["requested_components"]) for row in call_v2_rows),
        "domain_primary_calls": dict(sorted(Counter(row["domain"] for row in call_v2_rows).items())),
        "requested_action_distribution": dict(sorted(Counter(row["requested_action_id"] for row in call_v2_rows).items())),
        "primary_input_token_upper_bound": primary_input,
        "primary_output_token_cap_total": primary_output,
        "maximum_fallback_input_token_upper_bound": fallback_input,
        "maximum_fallback_output_token_cap_total": fallback_output,
        "estimated_primary_generation_usd": round(_usd(primary_input, primary_output), 6),
        "estimated_generation_usd_with_all_fallbacks": round(_usd(primary_input + fallback_input, primary_output + fallback_output), 6),
        "pricing_assumption_usd_per_mtok": {"input": INPUT_USD_PER_MTOK, "output": OUTPUT_USD_PER_MTOK},
        "external_response_quality_risk_or_judge_used_for_selection_or_prompt": False,
        "call_plan_sha256": sha256_file(call_v2_path),
        "source_call_plan_v1_sha256": sha256_file(old_plan_path),
    }
    write_json(plan_v2 / "generation_preflight.json", generation_preflight)

    pilot_report = {
        "protocol": PROTOCOL,
        "status": "EXECUTION_FORMAT_PILOT_CLOSED_NOT_FINAL_EVIDENCE",
        "physical_primary_outcomes_saved": len(outcomes),
        "touched_esconv_panels_excluded": len(touched_panels),
        "touched_evoemo_panels": 0,
        "resource_actions_saved": sum(row["requested_action_id"] != "M0+R0" for row in outcomes),
        "observed_format_failure": "generator emitted unrequested and sometimes duplicate component_statuses",
        "allowed_repair": "accept one consistent requested status; record but ignore extraneous statuses; accept identical requested duplicates; conflicting requested duplicates remain fail-closed",
        "pm_candidate_retrieval_prompt_or_semantic_guard_changed": False,
        "responses_quality_or_risk_used_for_repair_or_selection": False,
        "pilot_outcomes_sha256": sha256_file(outcomes_path),
    }
    write_json(pilot_dir / "execution_format_pilot_closeout.json", pilot_report)

    checks = {
        "all_47_touched_esconv_panels_excluded": len(touched_panels) == 47 and not touched_panels.intersection({row["panel_id"] for row in esconv_v2}),
        "122_untouched_esconv_dialogues_remain": len(esconv_v2) == 122 and len({row["dialogue_id_private_analysis_only"] for row in esconv_v2}) == 122,
        "all_204_evoemo_states_untouched_and_retained": len(evoemo_v2) == 204,
        "326_final_states": len(route_v2_rows) == 326,
        "no_completed_call_in_final_v2": not bool(completed_call_ids & {str(row["call_id"]) for row in call_v2_rows}),
        "repair_scope_is_execution_format_only": True,
        "no_response_quality_or_risk_used": True,
    }
    panel_report = {
        "protocol": PROTOCOL,
        "status": "PASS_POST_COMPATIBILITY_FINAL_V2_FROZEN" if all(checks.values()) else "FAIL_POST_COMPATIBILITY_FREEZE",
        "panels": {"ESConv": {"states": len(esconv_v2), "independent_dialogues": len(esconv_v2)}, "EvoEmo": {"states": len(evoemo_v2), "independent_users": len({row["user_id_private_analysis_only"] for row in evoemo_v2})}},
        "checks": checks,
        "compatibility_pilot_closeout_sha256": sha256_file(pilot_dir / "execution_format_pilot_closeout.json"),
        "outputs": {str(esconv_v2_path.relative_to(root)): sha256_file(esconv_v2_path), str(evoemo_v2_path.relative_to(root)): sha256_file(evoemo_v2_path), str(route_v2_path.relative_to(root)): sha256_file(route_v2_path), str(call_v2_path.relative_to(root)): sha256_file(call_v2_path)},
    }
    write_json(panel_v2 / "panel_freeze_report.json", panel_report)
    return {"panel": panel_report, "routing": routing_report, "generation": generation_preflight, "pilot": pilot_report}


def main() -> None:
    result = build()
    print({"panel": result["panel"], "routing": result["routing"], "generation": result["generation"], "pilot": result["pilot"]})


if __name__ == "__main__":
    main()
