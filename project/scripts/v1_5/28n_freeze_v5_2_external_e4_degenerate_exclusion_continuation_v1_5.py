#!/usr/bin/env python3
"""Freeze E4 completion after registering one deterministic degenerate call invalid."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from metacom_pm.attempt_ledger import physical_call_key
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e4-degenerate-exclusion-continuation-freeze-v1"
STAGE = "v5_2_external_e4_generation_v2_qa_adapter_continuation_v3_exclude_degenerate_call"
BASE_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
BASE_EXECUTION = ROOT / "outputs/pm_v1_5_v5_2_external_e4_execution_v2_qa_adapter"
CONT1_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_plan_v1"
CONT1_EXECUTION = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_execution_v1"
CONT2_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_capacity_repair_continuation_plan_v1"
CONT2_EXECUTION = ROOT / "outputs/pm_v1_5_v5_2_external_e4_capacity_repair_continuation_execution_v1"
RUNNER = ROOT / "scripts/v1_5/28o_run_v5_2_external_e4_degenerate_exclusion_continuation_v1_5.py"
DEFAULT_OUT = ROOT / "outputs/pm_v1_5_v5_2_external_e4_degenerate_exclusion_continuation_plan_v1"
MAX_ATTEMPTS = 3
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60
MINIMUM_INTER_CALL_SECONDS = 2.0
RATE_LIMIT_BACKOFF_SECONDS = (60.0, 60.0)
INVALID_CALL_ID = "e3raw_1f526d93a4d7a51e2161c504"
INVALID_STATE_ID = "state_79702dbcf1f5ea0e6318"
RAW_STATE_CALL_IDS = (
    "e3raw_0381a329c552a503fec6358c",
    INVALID_CALL_ID,
    "e3raw_2afd08eb3f50ed117a4c3385",
    "e3raw_7a592b64e6848bbc2f987a28",
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def successful_outcomes(path: Path) -> list[dict[str, Any]]:
    return [
        dict((event.get("result") or {}).get("outcome") or {})
        for event in rows(path)
        if event.get("event") == "SUCCEEDED"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal continuation freeze requires {FORMAL_PYTHON}; got {sys.executable}")

    abort1 = read_json(CONT1_EXECUTION / "ABORTED_E4_CONTINUATION_V1.json")
    abort2 = read_json(CONT2_EXECUTION / "ABORTED_E4_CAPACITY_REPAIR_CONTINUATION_V1.json")
    if abort1.get("status") != "CONSUMED_INCOMPLETE_SINGLE_CONTENT_LENGTH":
        raise RuntimeError("first continuation closeout drifted")
    if abort2.get("status") != "CONSUMED_INCOMPLETE_DETERMINISTIC_REPETITION_DEGENERATION":
        raise RuntimeError("capacity-repair closeout drifted")
    if (abort2.get("failed_call") or {}).get("call_id") != INVALID_CALL_ID:
        raise RuntimeError("registered invalid call drifted")
    exclusion = dict(abort2.get("analysis_exclusion_contract") or {})
    if exclusion.get("raw_secondary_table_excluded_state_id") != INVALID_STATE_ID:
        raise RuntimeError("registered invalid state drifted")
    if tuple(exclusion.get("raw_secondary_table_excluded_call_ids") or ()) != RAW_STATE_CALL_IDS:
        raise RuntimeError("raw-state exclusion unit drifted")

    base = rows(BASE_PLAN / "physical_call_plan_private.jsonl")
    base_by = {str(item["call_id"]): item for item in base}
    if len(base_by) != 4218:
        raise RuntimeError("base E4 plan must contain 4218 unique calls")
    carried = successful_outcomes(BASE_EXECUTION / "physical_attempt_ledger.jsonl")
    carried.extend(successful_outcomes(CONT1_EXECUTION / "physical_attempt_ledger.jsonl"))
    carried.extend(successful_outcomes(CONT2_EXECUTION / "physical_attempt_ledger.jsonl"))
    carried_by_id: dict[str, dict[str, Any]] = {}
    for outcome in carried:
        call_id = str(outcome.get("call_id") or "")
        planned = base_by.get(call_id)
        if not call_id or planned is None or call_id in carried_by_id:
            raise RuntimeError(f"invalid or duplicate carry outcome: {call_id}")
        if outcome.get("messages_sha256") != planned["messages_sha256"]:
            raise RuntimeError(f"carry prompt drift: {call_id}")
        carried_by_id[call_id] = outcome
    if len(carried_by_id) != 3738 or INVALID_CALL_ID in carried_by_id:
        raise RuntimeError("expected 3738 exact carries excluding the invalid call")
    carried = sorted(carried_by_id.values(), key=lambda item: (str(item["call_kind"]), str(item["call_id"])))

    config_path = ROOT / "configs/pm_v1_5.yaml"
    experiment_path = ROOT / "configs/experiment.yaml"
    pm_config = load_config(config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(load_config(experiment_path), generation.generator_endpoint)
    remaining: list[dict[str, Any]] = []
    for old in base:
        call_id = str(old["call_id"])
        if call_id in carried_by_id or call_id == INVALID_CALL_ID:
            continue
        parameters = dict(old["request_parameters"])
        remaining.append(
            {
                **old,
                "protocol": PROTOCOL,
                "physical_call_key": physical_call_key(
                    stage=STAGE,
                    record_ids=dict(old["record_ids"]),
                    prompt_sha256=str(old["messages_sha256"]),
                    endpoint=endpoint,
                    request_parameters=parameters,
                ),
                "maximum_physical_attempts": MAX_ATTEMPTS,
            }
        )
    if len(remaining) != 479 or {str(x["call_id"]) for x in remaining} & ({INVALID_CALL_ID} | set(carried_by_id)):
        raise RuntimeError("remaining plan must contain exactly 479 never-attempted calls")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    carried_path = args.out_dir / "carried_outcomes_private.jsonl"
    remaining_path = args.out_dir / "remaining_physical_call_plan_private.jsonl"
    exclusion_path = args.out_dir / "registered_invalid_generation.json"
    write_jsonl(carried_path, carried)
    write_jsonl(remaining_path, remaining)
    write_json(
        exclusion_path,
        {
            "protocol": PROTOCOL,
            "invalid_call_id": INVALID_CALL_ID,
            "invalid_state_id": INVALID_STATE_ID,
            "failure_class": "deterministic_single_request_repetition_degeneration",
            "valid_generation_total_after_completion": 4217,
            "raw_secondary_table_excluded_state_id": INVALID_STATE_ID,
            "raw_secondary_table_excluded_call_ids": list(RAW_STATE_CALL_IDS),
            "invalid_call_must_not_be_retried_or_imputed": True,
            "main_table_non_raw_analyses_unaffected": True,
        },
    )
    input_bound = sum(int(item["input_token_upper_bound"]) for item in remaining)
    output_bound = sum(int(item["output_token_cap"]) for item in remaining)
    one_cost = (input_bound * PRICE_INPUT + output_bound * PRICE_OUTPUT) / 1_000_000
    source_paths = [
        BASE_PLAN / "execution_seal.json",
        BASE_PLAN / "physical_call_plan_private.jsonl",
        BASE_PLAN / "qa_call_plan_v2_private.jsonl",
        BASE_EXECUTION / "physical_attempt_ledger.jsonl",
        BASE_EXECUTION / "ABORTED_E4_V2.json",
        CONT1_PLAN / "execution_seal.json",
        CONT1_PLAN / "remaining_physical_call_plan_private.jsonl",
        CONT1_EXECUTION / "run_manifest.json",
        CONT1_EXECUTION / "physical_attempt_ledger.jsonl",
        CONT1_EXECUTION / "ABORTED_E4_CONTINUATION_V1.json",
        CONT2_PLAN / "execution_seal.json",
        CONT2_PLAN / "remaining_physical_call_plan_private.jsonl",
        CONT2_EXECUTION / "run_manifest.json",
        CONT2_EXECUTION / "physical_attempt_ledger.jsonl",
        CONT2_EXECUTION / "ABORTED_E4_CAPACITY_REPAIR_CONTINUATION_V1.json",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    source_hashes["derived/carried_outcomes_private.jsonl"] = sha256_file(carried_path)
    source_hashes["derived/registered_invalid_generation.json"] = sha256_file(exclusion_path)
    implementation_paths = [
        RUNNER,
        ROOT / "src/metacom_pm/v1_5_external_qa_adapter_v2.py",
        ROOT / "src/metacom_pm/v1_5_v5_2_locked_composer.py",
        ROOT / "src/metacom_pm/generation_contract.py",
        ROOT / "src/metacom_pm/attempt_ledger.py",
        ROOT / "src/metacom_pm/bounded_retry.py",
        ROOT / "src/metacom_pm/paid_run_release.py",
    ]
    implementation_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in implementation_paths}
    payload = {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "original_total_logical_calls": 4218,
        "valid_generation_target": 4217,
        "registered_invalid_generation_calls": 1,
        "carried_successful_calls": 3738,
        "remaining_paid_logical_calls": len(remaining),
        "remaining_call_counts": dict(Counter(str(item["call_kind"]) for item in remaining)),
        "maximum_physical_attempts_per_remaining_call": MAX_ATTEMPTS,
        "maximum_physical_api_attempts": len(remaining) * MAX_ATTEMPTS,
        "minimum_inter_call_seconds": MINIMUM_INTER_CALL_SECONDS,
        "rate_limit_backoff_seconds": list(RATE_LIMIT_BACKOFF_SECONDS),
        "input_token_upper_bound_one_attempt_each": input_bound,
        "output_token_upper_bound_one_attempt_each": output_bound,
        "pricing_usd_per_mtok": {"input": PRICE_INPUT, "output": PRICE_OUTPUT},
        "estimated_usd_upper_bound_one_attempt_each": one_cost,
        "absolute_usd_upper_bound_all_physical_attempts": one_cost * MAX_ATTEMPTS,
        "endpoint_model": endpoint.model,
        "generation_contract_sha256": generation.digest(),
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "remaining_physical_plan_sha256": sha256_file(remaining_path),
        "carry_forward_is_exact_only": True,
        "invalid_generation_is_registered_not_imputed": True,
        "all_remaining_provider_visible_messages_or_parameters_changed": False,
        "pm_or_route_changed": False,
        "external_quality_or_risk_outcome_used_to_change_pm_or_metrics": False,
    }
    identity = sha256_text(canonical_json(payload))
    write_json(args.out_dir / "cost_estimate.json", {**payload, "cost_estimate_sha256": identity})
    seal = {
        "protocol": PROTOCOL,
        "status": "SEALED_READY_FOR_EXPLICIT_PAID_RELEASE",
        "stage": STAGE,
        "cost_estimate_sha256": identity,
        "cost_estimate_file_sha256": sha256_file(args.out_dir / "cost_estimate.json"),
        "remaining_physical_plan_sha256": sha256_file(remaining_path),
        "carried_outcomes_sha256": sha256_file(carried_path),
        "registered_invalid_generation_sha256": sha256_file(exclusion_path),
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_path),
        "generator": endpoint.model,
        "api_calls_made": 0,
        "pm_or_route_changed": False,
    }
    write_json(args.out_dir / "execution_seal.json", seal)
    print(json.dumps({**seal, "remaining_paid_logical_calls": len(remaining), "registered_invalid_call_id": INVALID_CALL_ID}))


if __name__ == "__main__":
    main()
