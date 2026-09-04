#!/usr/bin/env python3
"""Freeze E4 V2 by changing only the qualified QA endpoint adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from metacom_pm.attempt_ledger import physical_call_key
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
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_external_qa_adapter_v2 import (
    EXTERNAL_QA_ADAPTER_PROTOCOL,
    memory_payload_from_v1_messages,
    qa_messages_v2,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e4-execution-freeze-v2-qa-adapter"
STAGE = "v5_2_external_e4_generation_v2_qa_adapter"
E2 = ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1"
E3 = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
OLD_E4 = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v1"
PILOT = ROOT / "outputs/pm_v1_5_v5_2_external_qa_adapter_pilot_v1_execution"
RUNNER = ROOT / "scripts/v1_5/28i_run_v5_2_external_e4_qa_adapter_v2_v1_5.py"
DEFAULT_OUT = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
MAX_ATTEMPTS = 3
PRICE_INPUT = 0.15
PRICE_OUTPUT = 0.60
FORBIDDEN_QA_KEYS = frozenset(
    {"answer", "answers", "evidence", "capability", "summaries", "question_group"}
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def has_forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key) in FORBIDDEN_QA_KEYS or has_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(has_forbidden(item) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E4 V2 freeze requires {FORMAL_PYTHON}; got {sys.executable}")
    pilot = read_json(PILOT / "summary.json")
    if pilot.get("status") != "PASS_READY_TO_VERSION_E4_QA_ADAPTER":
        raise RuntimeError("E4 V2 requires the consumed five-condition adapter pilot PASS")
    old_seal = read_json(OLD_E4 / "execution_seal.json")
    for relative, expected in old_seal["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"frozen upstream drifted: {relative}")
    config_path = ROOT / "configs/pm_v1_5.yaml"
    experiment_path = ROOT / "configs/experiment.yaml"
    pm_config = load_config(config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(load_config(experiment_path), generation.generator_endpoint)
    if endpoint.model != old_seal["generator"]:
        raise RuntimeError("generator differs from frozen E4 V1")

    qa_v1_path = E3 / "qa_final_call_plan_private.jsonl"
    qa_v2: list[dict[str, Any]] = []
    for source in rows(qa_v1_path):
        messages = qa_messages_v2(
            question=str(source["question"]),
            memory_payload=memory_payload_from_v1_messages(source["messages"]),
        )
        if has_forbidden(messages):
            raise RuntimeError(f"QA adapter leaked gold key: {source['call_id']}")
        updated = dict(source)
        updated.update(
            {
                "protocol": PROTOCOL,
                "qa_prompt_protocol": EXTERNAL_QA_ADAPTER_PROTOCOL,
                "qa_system_prompt_sha256": sha256_text(messages[0]["content"]),
                "messages": messages,
                "messages_sha256": sha256_text(canonical_json(messages)),
                "input_token_upper_bound": conservative_token_bound(
                    canonical_json(messages), safety_factor=1.25
                ),
                "adapter_only_change_from_v1": True,
                "pm_or_route_changed": False,
                "gold_or_outcome_read": False,
            }
        )
        qa_v2.append(updated)
    if len(qa_v2) != 2090:
        raise RuntimeError("QA V2 call count drifted")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qa_v2_path = args.out_dir / "qa_call_plan_v2_private.jsonl"
    write_jsonl(qa_v2_path, qa_v2)
    sources = {
        "response_core": E2 / "response_core_call_plan_private.jsonl",
        "response_raw": E3 / "response_raw_call_plan_private.jsonl",
        "qa": qa_v2_path,
    }
    physical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, path in sources.items():
        for item in rows(path):
            call_id = str(item["call_id"])
            if call_id in seen:
                raise RuntimeError(f"duplicate call id: {call_id}")
            seen.add(call_id)
            message_hash = sha256_text(canonical_json(item["messages"]))
            if message_hash != item["messages_sha256"]:
                raise RuntimeError(f"message hash drift: {call_id}")
            record_ids = {
                "call_kind": kind,
                "call_id": call_id,
                "state_or_question_id": str(item.get("state_id") or item.get("question_id") or ""),
            }
            parameters = {
                "temperature": float(generation.temperature),
                "max_tokens": int(item["output_token_cap"]),
                "seed": int(item["seed"]),
                "response_schema": None,
                "client_retries": 1,
            }
            physical.append(
                {
                    "protocol": PROTOCOL,
                    "call_kind": kind,
                    "call_id": call_id,
                    "source_plan_relative_path": str(path.relative_to(ROOT)),
                    "messages_sha256": message_hash,
                    "input_token_upper_bound": int(item["input_token_upper_bound"]),
                    "output_token_cap": int(item["output_token_cap"]),
                    "seed": int(item["seed"]),
                    "record_ids": record_ids,
                    "request_parameters": parameters,
                    "physical_call_key": physical_call_key(
                        stage=STAGE,
                        record_ids=record_ids,
                        prompt_sha256=message_hash,
                        endpoint=endpoint,
                        request_parameters=parameters,
                    ),
                    "maximum_physical_attempts": MAX_ATTEMPTS,
                }
            )
    physical.sort(key=lambda item: (str(item["call_kind"]), str(item["call_id"])))
    counts = dict(Counter(str(item["call_kind"]) for item in physical))
    if counts != {"response_core": 1576, "response_raw": 552, "qa": 2090}:
        raise RuntimeError(f"call distribution drifted: {counts}")
    physical_path = args.out_dir / "physical_call_plan_private.jsonl"
    write_jsonl(physical_path, physical)
    input_bound = sum(int(item["input_token_upper_bound"]) for item in physical)
    output_bound = sum(int(item["output_token_cap"]) for item in physical)
    one_cost = (input_bound * PRICE_INPUT + output_bound * PRICE_OUTPUT) / 1_000_000
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [
            E2 / "plan_manifest.json",
            E3 / "retrieval_report.json",
            sources["response_core"],
            sources["response_raw"],
            sources["qa"],
            PILOT / "summary.json",
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
        "logical_calls": len(physical),
        "call_counts": counts,
        "maximum_physical_attempts_per_logical_call": MAX_ATTEMPTS,
        "maximum_physical_api_attempts": len(physical) * MAX_ATTEMPTS,
        "input_token_upper_bound_one_attempt_each": input_bound,
        "output_token_upper_bound_one_attempt_each": output_bound,
        "pricing_usd_per_mtok": {"input": PRICE_INPUT, "output": PRICE_OUTPUT},
        "estimated_usd_upper_bound_one_attempt_each": one_cost,
        "absolute_usd_upper_bound_all_physical_attempts": one_cost * MAX_ATTEMPTS,
        "endpoint_model": endpoint.model,
        "generation_contract_sha256": generation.digest(),
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "physical_call_plan_sha256": sha256_file(physical_path),
        "old_frozen_upstream_source_sha256": old_seal["source_sha256"],
        "response_plans_byte_identical_to_v1": True,
        "qa_change_scope": "qualified endpoint message adapter only",
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
        "physical_call_plan_sha256": sha256_file(physical_path),
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_path),
        "generator": endpoint.model,
        "api_calls_made": 0,
        "pm_or_route_changed": False,
        "human_or_llm_outcomes_read_to_change_pm": False,
    }
    write_json(args.out_dir / "execution_seal.json", seal)
    print(json.dumps({**seal, "logical_calls": len(physical)}))


if __name__ == "__main__":
    main()
