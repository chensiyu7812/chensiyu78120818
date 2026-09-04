#!/usr/bin/env python3
"""Freeze the final E4 continuation with one disclosed output-cap repair."""

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
PROTOCOL = "pm-v1.5-v5.2-external-e4-capacity-repair-continuation-freeze-v1"
STAGE = "v5_2_external_e4_generation_v2_qa_adapter_continuation_v2_capacity_repair"
BASE_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
BASE_EXECUTION = ROOT / "outputs/pm_v1_5_v5_2_external_e4_execution_v2_qa_adapter"
CONTINUATION_V1_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_plan_v1"
CONTINUATION_V1_EXECUTION = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_execution_v1"
RUNNER = ROOT / "scripts/v1_5/28m_run_v5_2_external_e4_capacity_repair_continuation_v1_5.py"
DEFAULT_OUT = ROOT / "outputs/pm_v1_5_v5_2_external_e4_capacity_repair_continuation_plan_v1"
MAX_ATTEMPTS = 3
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60
MINIMUM_INTER_CALL_SECONDS = 2.0
RATE_LIMIT_BACKOFF_SECONDS = (60.0, 60.0)
FAILED_LENGTH_CALL_ID = "e3raw_1f526d93a4d7a51e2161c504"
ORIGINAL_OUTPUT_CAP = 300
REPAIRED_OUTPUT_CAP = 512


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def successful_outcomes(path: Path) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for event in rows(path):
        if event.get("event") != "SUCCEEDED":
            continue
        outcome = dict((event.get("result") or {}).get("outcome") or {})
        if not outcome.get("call_id"):
            raise RuntimeError(f"successful ledger event lacks call_id: {path}")
        outcomes.append(outcome)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal continuation freeze requires {FORMAL_PYTHON}; got {sys.executable}")

    base_seal = read_json(BASE_PLAN / "execution_seal.json")
    first_abort = read_json(BASE_EXECUTION / "ABORTED_E4_V2.json")
    second_abort = read_json(CONTINUATION_V1_EXECUTION / "ABORTED_E4_CONTINUATION_V1.json")
    if first_abort.get("status") != "CONSUMED_INCOMPLETE_TRANSPORT_RATE_LIMIT":
        raise RuntimeError("base E4 V2 closeout is not the frozen rate-limit failure")
    if second_abort.get("status") != "CONSUMED_INCOMPLETE_SINGLE_CONTENT_LENGTH":
        raise RuntimeError("continuation V1 closeout is not the frozen content-length failure")
    failed = dict(second_abort.get("failed_call") or {})
    if (
        failed.get("call_id") != FAILED_LENGTH_CALL_ID
        or failed.get("provider_finish_reason") != "length"
        or int(failed.get("frozen_output_token_cap", -1)) != ORIGINAL_OUTPUT_CAP
    ):
        raise RuntimeError("the single-call capacity repair no longer matches the audited failure")

    base_physical = rows(BASE_PLAN / "physical_call_plan_private.jsonl")
    by_call = {str(item["call_id"]): item for item in base_physical}
    if len(by_call) != 4218:
        raise RuntimeError("base E4 plan must contain 4218 unique calls")
    carried = successful_outcomes(BASE_EXECUTION / "physical_attempt_ledger.jsonl")
    carried.extend(successful_outcomes(CONTINUATION_V1_EXECUTION / "physical_attempt_ledger.jsonl"))
    carried_by_id: dict[str, dict[str, Any]] = {}
    for outcome in carried:
        call_id = str(outcome["call_id"])
        if call_id in carried_by_id:
            raise RuntimeError(f"duplicate carry-forward outcome: {call_id}")
        planned = by_call.get(call_id)
        if planned is None:
            raise RuntimeError(f"carry outcome is outside frozen E4 plan: {call_id}")
        if outcome.get("messages_sha256") != planned["messages_sha256"]:
            raise RuntimeError(f"carry prompt drift: {call_id}")
        carried_by_id[call_id] = outcome
    if len(carried_by_id) != 3738:
        raise RuntimeError(f"expected 3738 exact carried outcomes, got {len(carried_by_id)}")
    if FAILED_LENGTH_CALL_ID in carried_by_id:
        raise RuntimeError("length-failed call cannot be carried as a success")
    carried = sorted(carried_by_id.values(), key=lambda item: (str(item["call_kind"]), str(item["call_id"])))

    config_path = ROOT / "configs/pm_v1_5.yaml"
    experiment_path = ROOT / "configs/experiment.yaml"
    pm_config = load_config(config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(load_config(experiment_path), generation.generator_endpoint)
    remaining: list[dict[str, Any]] = []
    repaired_rows = 0
    for old in base_physical:
        call_id = str(old["call_id"])
        if call_id in carried_by_id:
            continue
        record_ids = dict(old["record_ids"])
        parameters = dict(old["request_parameters"])
        output_cap = int(old["output_token_cap"])
        capacity_repair = False
        if call_id == FAILED_LENGTH_CALL_ID:
            if output_cap != ORIGINAL_OUTPUT_CAP or int(parameters.get("max_tokens", -1)) != ORIGINAL_OUTPUT_CAP:
                raise RuntimeError("failed call no longer has the audited original output cap")
            output_cap = REPAIRED_OUTPUT_CAP
            parameters["max_tokens"] = REPAIRED_OUTPUT_CAP
            capacity_repair = True
            repaired_rows += 1
        remaining.append(
            {
                **old,
                "protocol": PROTOCOL,
                "output_token_cap": output_cap,
                "request_parameters": parameters,
                "physical_call_key": physical_call_key(
                    stage=STAGE,
                    record_ids=record_ids,
                    prompt_sha256=str(old["messages_sha256"]),
                    endpoint=endpoint,
                    request_parameters=parameters,
                ),
                "maximum_physical_attempts": MAX_ATTEMPTS,
                "single_call_capacity_repair": capacity_repair,
            }
        )
    if len(remaining) != 480 or repaired_rows != 1:
        raise RuntimeError(f"expected 480 remaining calls and one repaired call; got {len(remaining)} and {repaired_rows}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    carried_path = args.out_dir / "carried_outcomes_private.jsonl"
    remaining_path = args.out_dir / "remaining_physical_call_plan_private.jsonl"
    write_jsonl(carried_path, carried)
    write_jsonl(remaining_path, remaining)
    input_bound = sum(int(item["input_token_upper_bound"]) for item in remaining)
    output_bound = sum(int(item["output_token_cap"]) for item in remaining)
    one_cost = (input_bound * PRICE_INPUT + output_bound * PRICE_OUTPUT) / 1_000_000
    source_paths = [
        BASE_PLAN / "execution_seal.json",
        BASE_PLAN / "physical_call_plan_private.jsonl",
        BASE_PLAN / "qa_call_plan_v2_private.jsonl",
        BASE_EXECUTION / "physical_attempt_ledger.jsonl",
        BASE_EXECUTION / "ABORTED_E4_V2.json",
        CONTINUATION_V1_PLAN / "execution_seal.json",
        CONTINUATION_V1_PLAN / "remaining_physical_call_plan_private.jsonl",
        CONTINUATION_V1_EXECUTION / "run_manifest.json",
        CONTINUATION_V1_EXECUTION / "physical_attempt_ledger.jsonl",
        CONTINUATION_V1_EXECUTION / "ABORTED_E4_CONTINUATION_V1.json",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    # The derived carry bundle is content-addressed under a stable logical key,
    # so independently reproduced output directories yield the same identity.
    source_hashes["derived/carried_outcomes_private.jsonl"] = sha256_file(carried_path)
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
        "total_logical_calls": 4218,
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
        "base_e4_v2_identity": base_seal["cost_estimate_sha256"],
        "prior_continuation_identity": second_abort["run_identity"],
        "carry_forward_is_exact_only": True,
        "single_call_capacity_repair": {
            "call_id": FAILED_LENGTH_CALL_ID,
            "call_kind": "response_raw",
            "original_output_token_cap": ORIGINAL_OUTPUT_CAP,
            "repaired_output_token_cap": REPAIRED_OUTPUT_CAP,
            "messages_changed": False,
            "model_seed_temperature_changed": False,
            "required_final_sensitivity_analysis": "Report primary results with the completed row and a sensitivity result excluding this call/state.",
        },
        "all_other_provider_visible_messages_or_parameters_changed": False,
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
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_path),
        "generator": endpoint.model,
        "api_calls_made": 0,
        "pm_or_route_changed": False,
    }
    write_json(args.out_dir / "execution_seal.json", seal)
    print(json.dumps({**seal, "remaining_paid_logical_calls": len(remaining), "capacity_repair_call_id": FAILED_LENGTH_CALL_ID}))


if __name__ == "__main__":
    main()
