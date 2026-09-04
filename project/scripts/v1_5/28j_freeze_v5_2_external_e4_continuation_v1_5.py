#!/usr/bin/env python3
"""Freeze the rate-limited E4 V2 continuation with exact carry-forward."""

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
PROTOCOL = "pm-v1.5-v5.2-external-e4-v2-rate-limit-continuation-freeze-v1"
STAGE = "v5_2_external_e4_generation_v2_qa_adapter_continuation_v1"
BASE_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
BASE_EXECUTION = ROOT / "outputs/pm_v1_5_v5_2_external_e4_execution_v2_qa_adapter"
RUNNER = ROOT / "scripts/v1_5/28k_run_v5_2_external_e4_continuation_v1_5.py"
DEFAULT_OUT = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_plan_v1"
MAX_ATTEMPTS = 3
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60
MINIMUM_INTER_CALL_SECONDS = 2.0
RATE_LIMIT_BACKOFF_SECONDS = (60.0, 60.0)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal continuation freeze requires {FORMAL_PYTHON}; got {sys.executable}")
    base_seal = read_json(BASE_PLAN / "execution_seal.json")
    abort = read_json(BASE_EXECUTION / "ABORTED_E4_V2.json")
    if abort.get("status") != "CONSUMED_INCOMPLETE_TRANSPORT_RATE_LIMIT":
        raise RuntimeError("continuation requires the frozen rate-limit closeout")
    base_physical = rows(BASE_PLAN / "physical_call_plan_private.jsonl")
    by_call = {str(item["call_id"]): item for item in base_physical}
    success_events = [
        item
        for item in rows(BASE_EXECUTION / "physical_attempt_ledger.jsonl")
        if item.get("event") == "SUCCEEDED"
    ]
    carried: list[dict[str, Any]] = []
    for event in success_events:
        outcome = dict((event.get("result") or {}).get("outcome") or {})
        call_id = str(outcome.get("call_id") or "")
        planned = by_call.get(call_id)
        if planned is None:
            raise RuntimeError(f"carry outcome is outside frozen plan: {call_id}")
        if outcome.get("messages_sha256") != planned["messages_sha256"]:
            raise RuntimeError(f"carry prompt drift: {call_id}")
        carried.append(outcome)
    carried.sort(key=lambda item: (str(item["call_kind"]), str(item["call_id"])))
    if len(carried) != 38 or len({str(item["call_id"]) for item in carried}) != 38:
        raise RuntimeError("expected exactly 38 unique carry-forward outcomes")
    carried_ids = {str(item["call_id"]) for item in carried}

    config_path = ROOT / "configs/pm_v1_5.yaml"
    experiment_path = ROOT / "configs/experiment.yaml"
    pm_config = load_config(config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(load_config(experiment_path), generation.generator_endpoint)
    remaining: list[dict[str, Any]] = []
    for old in base_physical:
        if str(old["call_id"]) in carried_ids:
            continue
        record_ids = dict(old["record_ids"])
        parameters = dict(old["request_parameters"])
        remaining.append(
            {
                **old,
                "protocol": PROTOCOL,
                "physical_call_key": physical_call_key(
                    stage=STAGE,
                    record_ids=record_ids,
                    prompt_sha256=str(old["messages_sha256"]),
                    endpoint=endpoint,
                    request_parameters=parameters,
                ),
                "maximum_physical_attempts": MAX_ATTEMPTS,
            }
        )
    if len(remaining) != 4180:
        raise RuntimeError("continuation must contain exactly 4180 remaining calls")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    carried_path = args.out_dir / "carried_outcomes_private.jsonl"
    remaining_path = args.out_dir / "remaining_physical_call_plan_private.jsonl"
    write_jsonl(carried_path, carried)
    write_jsonl(remaining_path, remaining)
    input_bound = sum(int(item["input_token_upper_bound"]) for item in remaining)
    output_bound = sum(int(item["output_token_cap"]) for item in remaining)
    one_cost = (input_bound * PRICE_INPUT + output_bound * PRICE_OUTPUT) / 1_000_000
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [
            BASE_PLAN / "execution_seal.json",
            BASE_PLAN / "physical_call_plan_private.jsonl",
            BASE_PLAN / "qa_call_plan_v2_private.jsonl",
            BASE_EXECUTION / "physical_attempt_ledger.jsonl",
            BASE_EXECUTION / "ABORTED_E4_V2.json",
            carried_path,
        ]
    }
    implementation_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [
            RUNNER,
            ROOT / "src/metacom_pm/v1_5_external_qa_adapter_v2.py",
            ROOT / "src/metacom_pm/v1_5_v5_2_locked_composer.py",
            ROOT / "src/metacom_pm/generation_contract.py",
            ROOT / "src/metacom_pm/attempt_ledger.py",
            ROOT / "src/metacom_pm/bounded_retry.py",
            ROOT / "src/metacom_pm/paid_run_release.py",
        ]
    }
    payload = {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "total_logical_calls": 4218,
        "carried_successful_calls": 38,
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
        "carry_forward_is_exact_only": True,
        "provider_visible_messages_or_parameters_changed": False,
        "transport_only_change": True,
        "pm_or_route_changed": False,
        "external_outcome_used_to_change_pm_or_metrics": False,
    }
    identity = sha256_text(canonical_json(payload))
    cost = {**payload, "cost_estimate_sha256": identity}
    write_json(args.out_dir / "cost_estimate.json", cost)
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
    print(json.dumps({**seal, "remaining_paid_logical_calls": len(remaining)}))


if __name__ == "__main__":
    main()
