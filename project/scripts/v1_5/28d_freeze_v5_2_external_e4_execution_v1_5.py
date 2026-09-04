#!/usr/bin/env python3
"""Freeze the exact E4 paid-generation identity without making API calls."""

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


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e4-execution-freeze-v1"
STAGE = "v5_2_external_e4_generation_v1"
E2_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e2_plan_v1"
E3_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e3_retrieval_v1"
E2_MANIFEST = E2_DIR / "plan_manifest.json"
E3_REPORT = E3_DIR / "retrieval_report.json"
SOURCE_PLANS = {
    "response_core": E2_DIR / "response_core_call_plan_private.jsonl",
    "response_raw": E3_DIR / "response_raw_call_plan_private.jsonl",
    "qa": E3_DIR / "qa_final_call_plan_private.jsonl",
}
RUNNER = ROOT / "scripts/v1_5/28e_run_v5_2_external_e4_generation_v1_5.py"
MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL = 3
PRICE_INPUT_PER_MTOK = 0.15
PRICE_OUTPUT_PER_MTOK = 0.60
FORBIDDEN_QA_KEYS = frozenset(
    {"answer", "answers", "evidence", "capability", "summaries", "question_group"}
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def endpoint_record(endpoint: Any) -> dict[str, Any]:
    return {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
        "supports_strict_json_schema": endpoint.supports_strict_json_schema,
        "thinking_mode": endpoint.thinking_mode,
        "enable_thinking": endpoint.enable_thinking,
        "temperature_mode": endpoint.temperature_mode,
        "max_output_tokens_parameter": endpoint.max_output_tokens_parameter,
    }


def has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in FORBIDDEN_QA_KEYS or has_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(has_forbidden_key(item) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E4 freeze requires {FORMAL_PYTHON}; got {sys.executable}")
    if not RUNNER.is_file():
        raise RuntimeError("E4 paid runner must exist before its implementation is sealed")

    e2 = read_json(E2_MANIFEST)
    e3 = read_json(E3_REPORT)
    if e2.get("status") != "PASS_READY_FOR_E3_RETRIEVAL":
        raise RuntimeError("E4 requires the passed E2 logical plan")
    if e3.get("status") != "PASS_READY_FOR_E4_PAID_GENERATION":
        raise RuntimeError("E4 requires the passed E3 exact-prompt seal")

    config_path = ROOT / "configs/pm_v1_5.yaml"
    pm_config = load_config(config_path)
    experiment_path = ROOT / "configs/experiment.yaml"
    experiment = load_config(experiment_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    if endpoint.model != e2["generator"]:
        raise RuntimeError("E4 endpoint model differs from E2")
    if generation.temperature != float(e2["temperature"]):
        raise RuntimeError("E4 temperature differs from E2")

    all_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call_kind, path in SOURCE_PLANS.items():
        for item in rows(path):
            call_id = str(item["call_id"])
            if call_id in seen:
                raise RuntimeError(f"duplicate cross-plan call_id: {call_id}")
            seen.add(call_id)
            messages = list(item["messages"])
            messages_sha256 = sha256_text(canonical_json(messages))
            if messages_sha256 != item["messages_sha256"]:
                raise RuntimeError(f"message hash drift for {call_id}")
            if call_kind == "qa":
                if has_forbidden_key(messages):
                    raise RuntimeError(f"QA generation message leaks evaluator-only gold: {call_id}")
                if item.get("answer_evidence_capability_visible_to_generation") is not False:
                    raise RuntimeError(f"QA gold-separation flag is not false: {call_id}")
            max_output = int(item["output_token_cap"])
            request_parameters = {
                "temperature": float(generation.temperature),
                "max_tokens": max_output,
                "seed": int(item["seed"]),
                "response_schema": None,
                "client_retries": 1,
            }
            record_ids = {
                "call_kind": call_kind,
                "call_id": call_id,
                "state_or_question_id": str(
                    item.get("state_id") or item.get("question_id") or ""
                ),
            }
            key = physical_call_key(
                stage=STAGE,
                record_ids=record_ids,
                prompt_sha256=messages_sha256,
                endpoint=endpoint,
                request_parameters=request_parameters,
            )
            all_rows.append(
                {
                    "protocol": PROTOCOL,
                    "call_kind": call_kind,
                    "call_id": call_id,
                    "source_plan_relative_path": str(path.relative_to(ROOT)),
                    "messages_sha256": messages_sha256,
                    "input_token_upper_bound": int(item["input_token_upper_bound"]),
                    "output_token_cap": max_output,
                    "seed": int(item["seed"]),
                    "record_ids": record_ids,
                    "request_parameters": request_parameters,
                    "physical_call_key": key,
                    "maximum_physical_attempts": MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL,
                }
            )
    all_rows.sort(key=lambda item: (str(item["call_kind"]), str(item["call_id"])))
    if len({row["physical_call_key"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("E4 physical call identities are not unique")

    expected_counts = {
        "response_core": int(e2["response"]["core_calls_after_state_action_policy_alias_dedup"]),
        "response_raw": int(e3["response"]["raw_calls"]),
        "qa": int(e3["qa"]["calls"]),
    }
    actual_counts = Counter(str(row["call_kind"]) for row in all_rows)
    if dict(actual_counts) != expected_counts:
        raise RuntimeError(f"E4 call counts drifted: {dict(actual_counts)} != {expected_counts}")

    input_bound = sum(int(row["input_token_upper_bound"]) for row in all_rows)
    output_bound = sum(int(row["output_token_cap"]) for row in all_rows)
    logical_cost = (
        input_bound * PRICE_INPUT_PER_MTOK + output_bound * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000.0
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [E2_MANIFEST, E3_REPORT, *SOURCE_PLANS.values()]
    }
    implementation_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in [
            RUNNER,
            ROOT / "src/metacom_pm/v1_5_v5_2_locked_composer.py",
            ROOT / "src/metacom_pm/generation_contract.py",
            ROOT / "src/metacom_pm/attempt_ledger.py",
            ROOT / "src/metacom_pm/bounded_retry.py",
            ROOT / "src/metacom_pm/paid_run_release.py",
        ]
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    physical_plan_path = args.out_dir / "physical_call_plan_private.jsonl"
    write_jsonl(physical_plan_path, all_rows)
    cost_payload = {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "logical_calls": len(all_rows),
        "call_counts": expected_counts,
        "maximum_physical_attempts_per_logical_call": MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL,
        "maximum_physical_api_attempts": len(all_rows) * MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL,
        "input_token_upper_bound_one_attempt_each": input_bound,
        "output_token_upper_bound_one_attempt_each": output_bound,
        "pricing_usd_per_mtok": {
            "input": PRICE_INPUT_PER_MTOK,
            "output": PRICE_OUTPUT_PER_MTOK,
        },
        "estimated_usd_upper_bound_one_attempt_each": logical_cost,
        "absolute_usd_upper_bound_all_physical_attempts": (
            logical_cost * MAXIMUM_PHYSICAL_ATTEMPTS_PER_CALL
        ),
        "endpoint": endpoint_record(endpoint),
        "generation_contract_sha256": generation.digest(),
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "physical_call_plan_sha256": sha256_file(physical_plan_path),
        "external_outcome_used_to_change_method": False,
    }
    cost_identity = sha256_text(canonical_json(cost_payload))
    cost = {**cost_payload, "cost_estimate_sha256": cost_identity}
    write_json(args.out_dir / "cost_estimate.json", cost)
    seal = {
        "protocol": PROTOCOL,
        "status": "SEALED_READY_FOR_E4_PAID_RELEASE",
        "stage": STAGE,
        "cost_estimate_sha256": cost_identity,
        "cost_estimate_file_sha256": sha256_file(args.out_dir / "cost_estimate.json"),
        "physical_call_plan_sha256": sha256_file(physical_plan_path),
        "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "config_sha256": sha256_file(config_path),
        "experiment_config_sha256": sha256_file(experiment_path),
        "generator": endpoint.model,
        "api_calls_made": 0,
        "human_or_llm_outcomes_read": False,
    }
    write_json(args.out_dir / "execution_seal.json", seal)
    print(json.dumps({**seal, "logical_calls": len(all_rows)}))


if __name__ == "__main__":
    main()
