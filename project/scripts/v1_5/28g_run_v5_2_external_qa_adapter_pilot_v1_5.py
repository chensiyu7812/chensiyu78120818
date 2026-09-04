#!/usr/bin/env python3
"""Prepare or run the five-condition V5.2 external QA endpoint adapter pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.attempt_ledger import PersistentAttemptLedger, physical_call_key
from metacom_pm.bounded_retry import execute_with_bounded_retry
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.paid_run_release import (
    require_output_directory_not_previously_consumed,
    require_paid_run_release,
)
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_external_qa_adapter_v2 import (
    EXTERNAL_QA_ADAPTER_PROTOCOL,
    endpoint_compatibility_errors,
    memory_payload_from_v1_messages,
    qa_messages_v2,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-qa-adapter-compatibility-pilot-v1"
STAGE = "v5_2_external_qa_adapter_compatibility_pilot_v1"
SOURCE = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1/qa_final_call_plan_private.jsonl"
OLD_SEAL = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v1/execution_seal.json"
DEFAULT_PLAN_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_qa_adapter_pilot_v1_plan"
DEFAULT_OUT_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_qa_adapter_pilot_v1_execution"
PILOT_QUESTION_ID = "qa_81a2c6d84682789b1d66461d"
CONDITIONS = (
    "no_memory",
    "full_history",
    "official_session_rag_top4",
    "typed_memory_fixed_high",
    "typed_memory_learned_pm",
)
MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL = 3
PRICE_INPUT_PER_MTOK = 0.15
PRICE_OUTPUT_PER_MTOK = 0.60
OUTPUT_TOKEN_CAP = 300
BACKOFF_SECONDS = (10.0, 30.0)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def frozen_upstream_hashes() -> dict[str, str]:
    seal = read_json(OLD_SEAL)
    return {str(key): str(value) for key, value in seal["source_sha256"].items()}


def build_plan(endpoint: Any, generation: SupporterGenerationContract) -> list[dict[str, Any]]:
    selected = {
        str(row["condition"]): row
        for row in rows(SOURCE)
        if str(row["question_id"]) == PILOT_QUESTION_ID
    }
    if set(selected) != set(CONDITIONS):
        raise RuntimeError("pilot question does not have all five frozen conditions")
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        source = selected[condition]
        messages = qa_messages_v2(
            question=str(source["question"]),
            memory_payload=memory_payload_from_v1_messages(source["messages"]),
        )
        prompt_sha = sha256_text(canonical_json(messages))
        record_ids = {
            "question_id": PILOT_QUESTION_ID,
            "condition": condition,
            "source_call_id": str(source["call_id"]),
        }
        request_parameters = {
            "temperature": float(generation.temperature),
            "max_tokens": OUTPUT_TOKEN_CAP,
            "seed": int(source["seed"]),
            "response_schema": None,
            "client_retries": 1,
        }
        output.append(
            {
                "protocol": PROTOCOL,
                "stage": STAGE,
                "question_id": PILOT_QUESTION_ID,
                "condition": condition,
                "source_call_id": str(source["call_id"]),
                "source_messages_sha256": str(source["messages_sha256"]),
                "messages": messages,
                "messages_sha256": prompt_sha,
                "input_token_upper_bound": conservative_token_bound(
                    canonical_json(messages), safety_factor=1.25
                ),
                "output_token_cap": OUTPUT_TOKEN_CAP,
                "seed": int(source["seed"]),
                "record_ids": record_ids,
                "request_parameters": request_parameters,
                "physical_call_key": physical_call_key(
                    stage=STAGE,
                    record_ids=record_ids,
                    prompt_sha256=prompt_sha,
                    endpoint=endpoint,
                    request_parameters=request_parameters,
                ),
                "maximum_physical_attempts": MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL,
                "gold_or_answer_read": False,
            }
        )
    return output


def cost_record(plan: list[dict[str, Any]], endpoint: Any) -> dict[str, Any]:
    input_bound = sum(int(row["input_token_upper_bound"]) for row in plan)
    output_bound = sum(int(row["output_token_cap"]) for row in plan)
    one_attempt = (
        input_bound * PRICE_INPUT_PER_MTOK + output_bound * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000
    payload = {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "status": "AWAITING_EXPLICIT_PAID_RELEASE",
        "adapter_protocol": EXTERNAL_QA_ADAPTER_PROTOCOL,
        "model": endpoint.model,
        "logical_calls": len(plan),
        "conditions": list(CONDITIONS),
        "maximum_physical_attempts_per_call": MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL,
        "maximum_physical_api_attempts": len(plan) * MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL,
        "input_token_upper_bound_one_attempt_each": input_bound,
        "output_token_upper_bound_one_attempt_each": output_bound,
        "estimated_usd_upper_bound_one_attempt_each": one_attempt,
        "absolute_usd_upper_bound_all_physical_attempts": one_attempt
        * MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL,
        "call_plan_sha256": sha256_text(canonical_json(plan)),
        "source_qa_plan_sha256": sha256_file(SOURCE),
        "frozen_upstream_source_sha256": frozen_upstream_hashes(),
        "pm_or_route_changed": False,
        "gold_or_outcome_read": False,
    }
    payload["cost_estimate_sha256"] = sha256_text(canonical_json(payload))
    return payload


def prepare(plan_dir: Path, endpoint: Any, generation: SupporterGenerationContract) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = build_plan(endpoint, generation)
    cost = cost_record(plan, endpoint)
    plan_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(plan_dir / "call_plan_private.jsonl", plan)
    write_json(plan_dir / "cost_estimate.json", cost)
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_EXPLICIT_PAID_RELEASE",
        "cost_estimate_sha256": cost["cost_estimate_sha256"],
        "call_plan_file_sha256": sha256_file(plan_dir / "call_plan_private.jsonl"),
        "cost_estimate_file_sha256": sha256_file(plan_dir / "cost_estimate.json"),
        "adapter_implementation_sha256": sha256_file(
            ROOT / "src/metacom_pm/v1_5_external_qa_adapter_v2.py"
        ),
        "runner_implementation_sha256": sha256_file(Path(__file__)),
        "scope": "endpoint I/O compatibility only; no answer correctness scoring",
        "pm_or_route_changed": False,
    }
    write_json(plan_dir / "plan_manifest.json", manifest)
    return plan, cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--accept-cost-estimate-sha256", default="")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal pilot requires {FORMAL_PYTHON}; got {sys.executable}")
    config_path = ROOT / "configs/pm_v1_5.yaml"
    pm_config = load_config(config_path)
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    require_output_directory_not_previously_consumed(
        args.out_dir, config=pm_config, config_path=config_path
    )
    plan, cost = prepare(args.plan_dir, endpoint, generation)
    if not args.run:
        print(json.dumps(cost))
        return
    identity = str(cost["cost_estimate_sha256"])
    if str(args.accept_cost_estimate_sha256) != identity:
        raise RuntimeError("pilot --run requires the exact accepted cost identity")
    release = require_paid_run_release(
        pm_config,
        config_path=config_path,
        stage=STAGE,
        run=True,
        run_identity=identity,
    )
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "run_manifest.json",
        {
            "protocol": PROTOCOL,
            "stage": STAGE,
            "identity": identity,
            "release": release,
            "call_plan_sha256": cost["call_plan_sha256"],
        },
    )
    ledger = PersistentAttemptLedger(
        args.out_dir / "physical_attempt_ledger.jsonl",
        stage=STAGE,
        expected_calls={
            str(row["physical_call_key"]): int(row["maximum_physical_attempts"])
            for row in plan
        },
        maximum_total_attempts=int(cost["maximum_physical_api_attempts"]),
    )
    client = make_client(endpoint)
    try:
        for row in plan:
            key = str(row["physical_call_key"])
            if ledger.succeeded(key):
                continue
            reservation, result, _attempts = execute_with_bounded_retry(
                ledger,
                key,
                record_ids=dict(row["record_ids"]),
                prompt_sha256=str(row["messages_sha256"]),
                call_fn=lambda row=row: client.chat(
                    list(row["messages"]),
                    temperature=float(generation.temperature),
                    max_tokens=int(row["output_token_cap"]),
                    seed=int(row["seed"]),
                    retries=1,
                ),
                max_provider_output_attempts=2,
                backoff_seconds=BACKOFF_SECONDS,
                sleep=time.sleep,
                capture_provider_output_text=True,
            )
            usage = require_reported_usage(result.usage, stage=STAGE)
            errors = endpoint_compatibility_errors(
                result.text, finish_reason=result.normalized_finish_reason
            )
            ledger.finish(
                reservation,
                succeeded=not errors,
                request_hash=result.request_hash,
                usage=usage,
                error=None if not errors else f"compatibility errors: {errors}",
                result={
                    "condition": row["condition"],
                    "output": result.text,
                    "compatibility_errors": errors,
                },
                metadata={
                    "provider_finish_reason": result.provider_finish_reason,
                    "normalized_finish_reason": result.normalized_finish_reason,
                    "correctness_scored": False,
                },
            )
            if errors:
                raise RuntimeError(f"QA adapter compatibility failed: {errors}")
    finally:
        client.close()
    terminal = [ledger.terminal_row(str(row["physical_call_key"])) for row in plan]
    passed = all(row and row.get("event") == "SUCCEEDED" for row in terminal)
    summary = {
        "protocol": PROTOCOL,
        "status": "PASS_READY_TO_VERSION_E4_QA_ADAPTER" if passed else "FAIL",
        "logical_calls": len(plan),
        "condition_counts": dict(Counter(str(row["condition"]) for row in plan)),
        "physical_attempts": ledger.started_attempts,
        "correctness_scored": False,
        "pm_or_route_changed": False,
        "cost_estimate_sha256": identity,
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
