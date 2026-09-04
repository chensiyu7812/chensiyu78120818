#!/usr/bin/env python3
"""Zero-API E5b: cost/token/coverage/validity/lineage accounting for the
EvoEmo Raw Session Top-4 / All Raw Sessions secondary tables.

28f_score_v5_2_external_e5_v1_5.py (E5) only ever reads
qa_outcomes_private.jsonl and response_core_outcomes_private.jsonl; it never
reads response_raw_outcomes_private.jsonl. This script fills exactly that gap
under a separate, versioned, zero-API contract. It does not touch, rescore,
or re-freeze v5_2_external_e5_scoring_v1 in any way.

This is bookkeeping only: token/cost/coverage/generation-validity/lineage.
It must never infer response quality or interaction-and-grounding risk.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import read_json, read_jsonl, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"

CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_2_external_e5b_raw_accounting_v1.json"
E3 = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
E4_OUT = ROOT / "outputs/pm_v1_5_v5_2_external_e4_degenerate_exclusion_continuation_execution_v1"
PLAN_FILE = E3 / "response_raw_call_plan_private.jsonl"
AUDIT_FILE = E3 / "response_raw_retrieval_audit_private.jsonl"
OUTCOMES_FILE = E4_OUT / "response_raw_outcomes_private.jsonl"
INVALID_FILE = E4_OUT / "registered_invalid_generation.json"

CONDITIONS = ("raw_session_top4_plus_frozen_strategy", "all_raw_sessions_plus_frozen_strategy")
PRICE_INPUT_PER_MTOK = 0.15
PRICE_OUTPUT_PER_MTOK = 0.6


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1_000_000 * PRICE_INPUT_PER_MTOK + completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_MTOK


def run(out_dir: Path) -> dict[str, Any]:
    contract = read_json(CONTRACT)
    if contract.get("status") != "FROZEN_ZERO_API":
        raise RuntimeError("E5b contract is not frozen")
    exclusion = contract["whole_state_exclusion"]
    excluded_state_id = exclusion["excluded_state_id"]
    excluded_call_ids = set(exclusion["excluded_call_ids"])
    expected = contract["expected_denominators"]

    plan = read_jsonl(PLAN_FILE)
    audit = read_jsonl(AUDIT_FILE)
    outcomes = read_jsonl(OUTCOMES_FILE)
    invalid = read_json(INVALID_FILE)

    if len(plan) != expected["original_planned_calls"]:
        raise RuntimeError(f"raw plan count drifted: {len(plan)} != {expected['original_planned_calls']}")
    if len(outcomes) != expected["valid_outcome_calls_before_state_exclusion"]:
        raise RuntimeError(f"raw outcome count drifted: {len(outcomes)}")
    if invalid.get("invalid_call_id") is None or invalid.get("invalid_call_must_not_be_retried_or_imputed") is not True:
        raise RuntimeError("registered_invalid_generation.json does not match expected shape")

    plan_ids = {row["call_id"] for row in plan}
    outcome_ids = {row["call_id"] for row in outcomes}
    missing = plan_ids - outcome_ids
    if missing != {invalid["invalid_call_id"]}:
        raise RuntimeError(
            f"plan-vs-outcome gap is not exactly the registered invalid call: {sorted(missing)}"
        )
    if len(missing) != expected["registered_invalid_calls"]:
        raise RuntimeError("registered invalid call count drifted")

    plan_by_id = {row["call_id"]: row for row in plan}
    plan_states = {row["state_id"] for row in plan}
    if len(plan_states) != expected["original_planned_states"]:
        raise RuntimeError("original planned state count drifted")

    # Per-condition physical generation validity against the original plan,
    # before any whole-state exclusion is applied.
    plan_calls_per_condition: dict[str, int] = defaultdict(int)
    valid_calls_per_condition: dict[str, int] = defaultdict(int)
    for row in plan:
        plan_calls_per_condition[row["condition"]] += 1
    for row in outcomes:
        valid_calls_per_condition[plan_by_id[row["call_id"]]["condition"]] += 1

    # Apply the frozen whole-state exclusion for the final analysis set.
    final_rows = [row for row in outcomes if row["call_id"] not in excluded_call_ids]
    if len(final_rows) != expected["final_analysis_calls"]:
        raise RuntimeError(f"final analysis call count drifted: {len(final_rows)}")
    final_states = {plan_by_id[row["call_id"]]["state_id"] for row in final_rows}
    if len(final_states) != expected["final_analysis_states"]:
        raise RuntimeError(f"final analysis state count drifted: {len(final_states)}")
    if excluded_state_id in final_states:
        raise RuntimeError("excluded state leaked into the final analysis set")

    audit_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (row["state_id"], row["condition"]): row for row in audit
    }

    per_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = [row for row in final_rows if plan_by_id[row["call_id"]]["condition"] == condition]
        states = sorted({plan_by_id[row["call_id"]]["state_id"] for row in rows})
        if len(rows) != expected["final_analysis_calls_per_condition"]:
            raise RuntimeError(f"{condition}: final call count drifted: {len(rows)}")

        prompt_tokens = [row["usage"]["prompt_tokens"] for row in rows]
        completion_tokens = [row["usage"]["completion_tokens"] for row in rows]
        total_tokens = [row["usage"]["total_tokens"] for row in rows]
        valid_flags = [
            row.get("normalized_finish_reason") == "complete"
            and not row.get("fallback_used")
            and not row.get("guard_errors")
            for row in rows
        ]

        coverage_rows = [audit_by_key[(sid, condition)] for sid in states if (sid, condition) in audit_by_key]
        if len(coverage_rows) != len(states):
            raise RuntimeError(f"{condition}: retrieval audit coverage missing for some states")
        candidate_counts = [row["candidate_session_count"] for row in coverage_rows]
        selected_before = [row["selected_before_context_fit"] for row in coverage_rows]
        selected_after = [row["selected_after_context_fit"] for row in coverage_rows]

        per_condition[condition] = {
            "final_analysis_state_count": len(states),
            "final_analysis_call_count": len(rows),
            "original_planned_call_count": plan_calls_per_condition[condition],
            "physically_valid_call_count_before_state_exclusion": valid_calls_per_condition[condition],
            "physical_generation_validity_rate_vs_original_plan": round(
                valid_calls_per_condition[condition] / plan_calls_per_condition[condition], 6
            ),
            "final_analysis_generation_validity_rate": round(sum(valid_flags) / len(valid_flags), 6),
            "tokens": {
                "sum_prompt_tokens": sum(prompt_tokens),
                "sum_completion_tokens": sum(completion_tokens),
                "sum_total_tokens": sum(total_tokens),
                "mean_prompt_tokens": round(sum(prompt_tokens) / len(rows), 3),
                "mean_completion_tokens": round(sum(completion_tokens) / len(rows), 3),
                "mean_total_tokens": round(sum(total_tokens) / len(rows), 3),
            },
            "proxy_cost_usd": round(_cost(sum(prompt_tokens), sum(completion_tokens)), 8),
            "retrieval_session_coverage": {
                "states_with_audit_row": len(coverage_rows),
                "mean_candidate_session_count": round(sum(candidate_counts) / len(coverage_rows), 3),
                "mean_selected_before_context_fit": round(sum(selected_before) / len(coverage_rows), 3),
                "mean_selected_after_context_fit": round(sum(selected_after) / len(coverage_rows), 3),
            },
        }

    combined_prompt = sum(r["usage"]["prompt_tokens"] for r in final_rows)
    combined_completion = sum(r["usage"]["completion_tokens"] for r in final_rows)

    result = {
        "protocol": "pm-v1.5-v5.2-external-e5b-raw-accounting-v1",
        "status": "COMPLETE_ZERO_API_RAW_ACCOUNTING",
        "automatic_quality_claim": False,
        "automatic_grounding_risk_claim": False,
        "api_calls": 0,
        "endpoint_model": contract["endpoint_model"],
        "pricing_usd_per_mtok": contract["pricing_usd_per_mtok"],
        "denominators": {
            "original_planned_calls": len(plan),
            "original_planned_states": len(plan_states),
            "valid_outcome_calls_before_state_exclusion": len(outcomes),
            "registered_invalid_calls": len(missing),
            "final_analysis_states": len(final_states),
            "final_analysis_calls": len(final_rows),
        },
        "whole_state_exclusion": {
            "excluded_state_id": excluded_state_id,
            "excluded_call_ids": sorted(excluded_call_ids),
            "reason": exclusion["reason"],
        },
        "per_condition": per_condition,
        "combined_final_analysis": {
            "calls": len(final_rows),
            "states": len(final_states),
            "sum_prompt_tokens": combined_prompt,
            "sum_completion_tokens": combined_completion,
            "proxy_cost_usd": round(_cost(combined_prompt, combined_completion), 8),
        },
        "input_sha256": {
            "response_raw_call_plan_private.jsonl": sha256_file(PLAN_FILE),
            "response_raw_retrieval_audit_private.jsonl": sha256_file(AUDIT_FILE),
            "response_raw_outcomes_private.jsonl": sha256_file(OUTCOMES_FILE),
            "registered_invalid_generation.json": sha256_file(INVALID_FILE),
        },
        "contract_sha256": sha256_file(CONTRACT),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "raw_accounting_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v5_2_external_e5b_raw_accounting_v1"
    )
    args = parser.parse_args()
    import sys

    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E5b requires {FORMAL_PYTHON}; got {sys.executable}")
    result = run(args.out_dir)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
