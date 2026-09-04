#!/usr/bin/env python3
"""Run the single sealed, resumable V5.2 external response/QA generation."""

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
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
)
from metacom_pm.bounded_retry import (
    TERMINAL_DISPOSITION,
    execute_with_bounded_retry,
    failure_metadata,
)
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
from metacom_pm.v1_5_v5_2_locked_composer import (
    LockedClause,
    LockedCompositionPlan,
    compose_locked_response,
    locked_response_guard_errors,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e4-generation-execution-v1"
STAGE = "v5_2_external_e4_generation_v1"
DEFAULT_PLAN_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v1"
DEFAULT_OUT_DIR = ROOT / "outputs/pm_v1_5_v5_2_external_e4_execution_v1"
BACKOFF_SECONDS = (10.0, 30.0)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def composition_plan(raw: dict[str, Any]) -> LockedCompositionPlan:
    return LockedCompositionPlan(
        requested_action_id=str(raw["requested_action_id"]),
        locked_clauses=tuple(
            LockedClause(**dict(item)) for item in raw["locked_clauses"]
        ),
        response_preference=str(raw["response_preference"]),
        strategy_instruction=str(raw["strategy_instruction"]),
        maximum_generator_support_moves=int(raw["maximum_generator_support_moves"]),
    )


def verify_and_load(
    plan_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    seal_path = plan_dir / "execution_seal.json"
    cost_path = plan_dir / "cost_estimate.json"
    physical_path = plan_dir / "physical_call_plan_private.jsonl"
    seal = read_json(seal_path)
    cost = read_json(cost_path)
    if seal.get("status") != "SEALED_READY_FOR_E4_PAID_RELEASE":
        raise RuntimeError("E4 execution seal is not ready")
    if seal.get("stage") != STAGE or cost.get("stage") != STAGE:
        raise RuntimeError("E4 stage identity drifted")
    if sha256_file(cost_path) != seal["cost_estimate_file_sha256"]:
        raise RuntimeError("E4 cost estimate changed after sealing")
    if sha256_file(physical_path) != seal["physical_call_plan_sha256"]:
        raise RuntimeError("E4 physical call plan changed after sealing")
    if cost["cost_estimate_sha256"] != seal["cost_estimate_sha256"]:
        raise RuntimeError("E4 accepted identity drifted")
    for relative, expected in seal["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"sealed E4 source changed: {relative}")
    for relative, expected in seal["implementation_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"sealed E4 implementation changed: {relative}")

    physical = rows(physical_path)
    sources: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for planned in physical:
        relative = str(planned["source_plan_relative_path"])
        if relative not in source_cache:
            source_rows = rows(ROOT / relative)
            source_cache[relative] = {str(item["call_id"]): item for item in source_rows}
            if len(source_cache[relative]) != len(source_rows):
                raise RuntimeError(f"duplicate call id in E4 source: {relative}")
        call_id = str(planned["call_id"])
        item = source_cache[relative].get(call_id)
        if item is None:
            raise RuntimeError(f"E4 source call missing: {call_id}")
        if sha256_text(canonical_json(item["messages"])) != planned["messages_sha256"]:
            raise RuntimeError(f"E4 prompt hash mismatch: {call_id}")
        sources[call_id] = item
    if len(sources) != len(physical):
        raise RuntimeError("E4 source resolution is not one-to-one")
    return cost, sources, physical


def finish_postcondition_failure(
    *, ledger: PersistentAttemptLedger, reservation: Any, result: Any, error: Exception
) -> None:
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=result.request_hash,
        usage=result.usage,
        error=f"{type(error).__name__}: {error}",
        result={
            "provider_output_text": str(result.text)[:8192],
            "provider_output_text_sha256": sha256_text(str(result.text)),
        },
        metadata=failure_metadata(
            retry_class="stage_postcondition_failure",
            retry_disposition=TERMINAL_DISPOSITION,
            response_diagnostics={
                "provider_finish_reason": result.provider_finish_reason,
                "normalized_finish_reason": result.normalized_finish_reason,
            },
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--accept-cost-estimate-sha256", default="")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal E4 execution requires {FORMAL_PYTHON}; got {sys.executable}")

    cost, source_by_id, physical = verify_and_load(args.plan_dir)
    config_path = ROOT / "configs/pm_v1_5.yaml"
    pm_config = load_config(config_path)
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    if endpoint.model != cost["endpoint"]["model"]:
        raise RuntimeError("E4 runtime generator differs from its seal")
    require_output_directory_not_previously_consumed(
        args.out_dir, config=pm_config, config_path=config_path
    )
    release = require_paid_run_release(
        pm_config,
        config_path=config_path,
        stage=STAGE,
        run=bool(args.run),
        run_identity=str(args.accept_cost_estimate_sha256 or ""),
    )
    ready = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_EXPLICIT_PAID_RELEASE" if not args.run else "RUNNING",
        "stage": STAGE,
        "logical_calls": len(physical),
        "call_counts": cost["call_counts"],
        "cost_estimate_sha256": cost["cost_estimate_sha256"],
        "estimated_usd_upper_bound_one_attempt_each": cost[
            "estimated_usd_upper_bound_one_attempt_each"
        ],
        "absolute_usd_upper_bound_all_physical_attempts": cost[
            "absolute_usd_upper_bound_all_physical_attempts"
        ],
        "resumable": True,
    }
    if not args.run:
        print(json.dumps(ready))
        return
    if str(args.accept_cost_estimate_sha256) != cost["cost_estimate_sha256"]:
        raise RuntimeError("E4 --run requires the exact accepted cost identity")
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL,
        "stage": STAGE,
        "cost_estimate_sha256": cost["cost_estimate_sha256"],
        "physical_call_plan_sha256": cost["physical_call_plan_sha256"],
        "generator": endpoint.model,
        "release": release,
    }
    manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
    manifest_path = args.out_dir / "run_manifest.json"
    if manifest_path.is_file() and read_json(manifest_path) != manifest:
        raise RuntimeError("E4 run manifest changed; use a fresh output directory")
    write_json(manifest_path, manifest)

    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(ledger_path, overwrite=False, stage=STAGE)
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=STAGE,
        expected_calls={
            str(item["physical_call_key"]): int(item["maximum_physical_attempts"])
            for item in physical
        },
        maximum_total_attempts=int(cost["maximum_physical_api_attempts"]),
    )
    pending = [
        item
        for item in physical
        if not ledger.succeeded(str(item["physical_call_key"]))
    ]
    client = make_client(endpoint) if pending else None
    completed_now = 0
    try:
        for planned in pending:
            call_id = str(planned["call_id"])
            call_kind = str(planned["call_kind"])
            source = source_by_id[call_id]
            key = str(planned["physical_call_key"])
            reservation, result, _ = execute_with_bounded_retry(
                ledger,
                key,
                record_ids=dict(planned["record_ids"]),
                prompt_sha256=str(planned["messages_sha256"]),
                call_fn=lambda source=source, planned=planned: client.chat(
                    list(source["messages"]),
                    temperature=float(generation.temperature),
                    max_tokens=int(planned["output_token_cap"]),
                    seed=int(planned["seed"]),
                    retries=1,
                ),
                max_provider_output_attempts=2,
                backoff_seconds=BACKOFF_SECONDS,
                sleep=time.sleep,
                capture_provider_output_text=True,
            )
            try:
                usage = require_reported_usage(result.usage, stage=STAGE)
                if usage["prompt_tokens"] > int(planned["input_token_upper_bound"]):
                    raise RuntimeError("reported prompt usage exceeds frozen E4 bound")
                finish_error = generation.completion_gate_error(
                    normalized_finish_reason=result.normalized_finish_reason,
                    provider_finish_reason=result.provider_finish_reason,
                )
                if finish_error:
                    raise RuntimeError(finish_error)
                normalized = generation.normalize_output(result.text)
                if not normalized:
                    raise RuntimeError("empty normalized E4 output")
                if call_kind == "response_core":
                    plan = composition_plan(dict(source["composition_plan"]))
                    final = compose_locked_response(base_response=normalized, plan=plan)
                    guard_errors = locked_response_guard_errors(response=final, plan=plan)
                    if guard_errors:
                        raise RuntimeError(f"locked response guard failed: {guard_errors}")
                else:
                    final = normalized
                    guard_errors = ()
                outcome = {
                    "protocol": PROTOCOL,
                    "call_kind": call_kind,
                    "call_id": call_id,
                    "state_id": source.get("state_id"),
                    "question_id": source.get("question_id"),
                    "partition": source.get("partition"),
                    "condition": source.get("condition"),
                    "requested_action_id": source.get("requested_action_id"),
                    "policy_aliases": source.get("policy_aliases"),
                    "primary_response": normalized if call_kind == "response_core" else None,
                    "final_output": final,
                    "guard_errors": list(guard_errors),
                    "fallback_used": False,
                    "usage": usage,
                    "provider_finish_reason": result.provider_finish_reason,
                    "normalized_finish_reason": result.normalized_finish_reason,
                    "messages_sha256": planned["messages_sha256"],
                    "physical_call_key": key,
                }
            except Exception as exc:
                finish_postcondition_failure(
                    ledger=ledger, reservation=reservation, result=result, error=exc
                )
                raise
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash=result.request_hash,
                usage=usage,
                error=None,
                result={"outcome": outcome},
                metadata={
                    "protocol": PROTOCOL,
                    "provider_finish_reason": result.provider_finish_reason,
                    "normalized_finish_reason": result.normalized_finish_reason,
                },
            )
            completed_now += 1
    finally:
        if client is not None:
            client.close()

    ordered: list[dict[str, Any]] = []
    for planned in physical:
        key = str(planned["physical_call_key"])
        if not ledger.succeeded(key):
            raise RuntimeError(f"E4 execution incomplete at {planned['call_id']}")
        terminal = ledger.terminal_row(key) or {}
        outcome = dict((terminal.get("result") or {}).get("outcome") or {})
        if outcome.get("call_id") != planned["call_id"]:
            raise RuntimeError(f"E4 terminal result mismatch: {planned['call_id']}")
        ordered.append(outcome)

    output_paths: dict[str, Path] = {}
    for kind in ("response_core", "response_raw", "qa"):
        path = args.out_dir / f"{kind}_outcomes_private.jsonl"
        write_jsonl(path, [item for item in ordered if item["call_kind"] == kind])
        output_paths[kind] = path
    usage_rows = [event for event in ledger.event_rows if event.get("event") == "SUCCEEDED"]
    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_READY_FOR_E5_ZERO_API_SCORING",
        "planned_calls": len(physical),
        "completed_calls": len(ordered),
        "completed_now": completed_now,
        "physical_attempts": ledger.started_attempts,
        "failed_physical_attempts": len(ledger.failures()),
        "fallback_calls": 0,
        "call_distribution": dict(Counter(item["call_kind"] for item in ordered)),
        "reported_prompt_tokens": sum(
            int((item.get("usage") or {}).get("prompt_tokens") or 0) for item in usage_rows
        ),
        "reported_completion_tokens": sum(
            int((item.get("usage") or {}).get("completion_tokens") or 0)
            for item in usage_rows
        ),
        "cost_estimate_sha256": cost["cost_estimate_sha256"],
        "physical_call_plan_sha256": cost["physical_call_plan_sha256"],
        "outcome_sha256": {
            kind: sha256_file(path) for kind, path in output_paths.items()
        },
        "resumable": True,
        "external_outcome_used_to_change_method": False,
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
