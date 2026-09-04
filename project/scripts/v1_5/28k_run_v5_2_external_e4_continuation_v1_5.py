#!/usr/bin/env python3
"""Run the paced E4 V2 continuation and merge exact carried outcomes."""

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
from metacom_pm.attempt_ledger import PersistentAttemptLedger, forbid_overwrite_of_spent_attempts
from metacom_pm.bounded_retry import execute_with_bounded_retry
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, write_json, write_jsonl
from metacom_pm.paid_run_release import require_output_directory_not_previously_consumed, require_paid_run_release
from metacom_pm.v1_5_external_qa_adapter_v2 import endpoint_compatibility_errors
from metacom_pm.v1_5_v5_2_locked_composer import LockedClause, LockedCompositionPlan, compose_locked_response, locked_response_guard_errors


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-external-e4-v2-rate-limit-continuation-execution-v1"
STAGE = "v5_2_external_e4_generation_v2_qa_adapter_continuation_v1"
DEFAULT_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_plan_v1"
DEFAULT_OUT = ROOT / "outputs/pm_v1_5_v5_2_external_e4_continuation_execution_v1"
BASE_PLAN = ROOT / "outputs/pm_v1_5_v5_2_external_e4_plan_v2_qa_adapter"
MINIMUM_INTER_CALL_SECONDS = 2.0
BACKOFF = (60.0, 60.0)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def composition(raw: dict[str, Any]) -> LockedCompositionPlan:
    return LockedCompositionPlan(
        requested_action_id=str(raw["requested_action_id"]),
        locked_clauses=tuple(LockedClause(**dict(item)) for item in raw["locked_clauses"]),
        response_preference=str(raw["response_preference"]),
        strategy_instruction=str(raw["strategy_instruction"]),
        maximum_generator_support_moves=int(raw["maximum_generator_support_moves"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--accept-cost-estimate-sha256", default="")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal continuation requires {FORMAL_PYTHON}; got {sys.executable}")
    seal = read_json(args.plan_dir / "execution_seal.json")
    cost = read_json(args.plan_dir / "cost_estimate.json")
    remaining = rows(args.plan_dir / "remaining_physical_call_plan_private.jsonl")
    carried = rows(args.plan_dir / "carried_outcomes_private.jsonl")
    if seal.get("status") != "SEALED_READY_FOR_EXPLICIT_PAID_RELEASE" or seal.get("stage") != STAGE:
        raise RuntimeError("continuation seal is not ready")
    if cost.get("cost_estimate_sha256") != seal.get("cost_estimate_sha256"):
        raise RuntimeError("continuation identity drifted")
    if sha256_file(args.plan_dir / "remaining_physical_call_plan_private.jsonl") != seal["remaining_physical_plan_sha256"]:
        raise RuntimeError("remaining plan drifted")
    if sha256_file(args.plan_dir / "carried_outcomes_private.jsonl") != seal["carried_outcomes_sha256"]:
        raise RuntimeError("carried outcomes drifted")
    for relative, expected in seal["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"continuation source drifted: {relative}")
    for relative, expected in seal["implementation_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"continuation implementation drifted: {relative}")

    all_base = rows(BASE_PLAN / "physical_call_plan_private.jsonl")
    source_cache: dict[str, dict[str, dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for planned in remaining:
        relative = str(planned["source_plan_relative_path"])
        if relative not in source_cache:
            source_rows = rows(ROOT / relative)
            source_cache[relative] = {str(item["call_id"]): item for item in source_rows}
        source = source_cache[relative].get(str(planned["call_id"]))
        if source is None or sha256_text(canonical_json(source["messages"])) != planned["messages_sha256"]:
            raise RuntimeError(f"continuation source resolution failed: {planned['call_id']}")
        sources[str(planned["call_id"])] = source
    config_path = ROOT / "configs/pm_v1_5.yaml"
    pm_config = load_config(config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(load_config(ROOT / "configs/experiment.yaml"), generation.generator_endpoint)
    require_output_directory_not_previously_consumed(args.out_dir, config=pm_config, config_path=config_path)
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
        "carried_calls": len(carried),
        "remaining_paid_calls": len(remaining),
        "cost_estimate_sha256": cost["cost_estimate_sha256"],
        "estimated_usd_upper_bound_one_attempt_each": cost["estimated_usd_upper_bound_one_attempt_each"],
        "absolute_usd_upper_bound_all_physical_attempts": cost["absolute_usd_upper_bound_all_physical_attempts"],
        "minimum_inter_call_seconds": MINIMUM_INTER_CALL_SECONDS,
        "pm_or_route_changed": False,
    }
    if not args.run:
        print(json.dumps(ready))
        return
    identity = str(cost["cost_estimate_sha256"])
    if str(args.accept_cost_estimate_sha256) != identity:
        raise RuntimeError("continuation --run requires exact identity")
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "run_manifest.json", {**ready, "release": release})
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(ledger_path, overwrite=False, stage=STAGE)
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=STAGE,
        expected_calls={str(item["physical_call_key"]): int(item["maximum_physical_attempts"]) for item in remaining},
        maximum_total_attempts=int(cost["maximum_physical_api_attempts"]),
    )
    pending = [item for item in remaining if not ledger.succeeded(str(item["physical_call_key"]))]
    client = make_client(endpoint) if pending else None
    last_start = 0.0
    try:
        for index, planned in enumerate(pending, start=1):
            elapsed = time.monotonic() - last_start
            if last_start and elapsed < MINIMUM_INTER_CALL_SECONDS:
                time.sleep(MINIMUM_INTER_CALL_SECONDS - elapsed)
            source = sources[str(planned["call_id"])]
            key = str(planned["physical_call_key"])
            last_start = time.monotonic()
            reservation, result, _attempts = execute_with_bounded_retry(
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
                backoff_seconds=BACKOFF,
                sleep=time.sleep,
                capture_provider_output_text=True,
            )
            usage = require_reported_usage(result.usage, stage=STAGE)
            normalized = generation.normalize_output(result.text)
            errors: list[str] = []
            finish_error = generation.completion_gate_error(
                normalized_finish_reason=result.normalized_finish_reason,
                provider_finish_reason=result.provider_finish_reason,
            )
            if finish_error:
                errors.append(finish_error)
            if not normalized:
                errors.append("empty normalized output")
            kind = str(planned["call_kind"])
            if kind == "response_core" and not errors:
                plan = composition(dict(source["composition_plan"]))
                final = compose_locked_response(base_response=normalized, plan=plan)
                errors.extend(locked_response_guard_errors(response=final, plan=plan))
            else:
                final = normalized
            if kind == "qa":
                errors.extend(endpoint_compatibility_errors(normalized, finish_reason=result.normalized_finish_reason))
            outcome = {
                "protocol": PROTOCOL,
                "call_kind": kind,
                "call_id": planned["call_id"],
                "state_id": source.get("state_id"),
                "question_id": source.get("question_id"),
                "partition": source.get("partition"),
                "condition": source.get("condition"),
                "requested_action_id": source.get("requested_action_id"),
                "policy_aliases": source.get("policy_aliases"),
                "primary_response": normalized if kind == "response_core" else None,
                "final_output": final,
                "guard_errors": list(errors),
                "fallback_used": False,
                "usage": usage,
                "provider_finish_reason": result.provider_finish_reason,
                "normalized_finish_reason": result.normalized_finish_reason,
                "messages_sha256": planned["messages_sha256"],
                "physical_call_key": key,
            }
            ledger.finish(
                reservation,
                succeeded=not errors,
                request_hash=result.request_hash,
                usage=usage,
                error=None if not errors else f"postcondition errors: {errors}",
                result={"outcome": outcome, "provider_output_text": result.text},
                metadata={"provider_finish_reason": result.provider_finish_reason, "normalized_finish_reason": result.normalized_finish_reason},
            )
            if errors:
                raise RuntimeError(f"continuation postcondition failed: {planned['call_id']}: {errors}")
            if index % 100 == 0:
                print(json.dumps({"progress_new_successes": index, "remaining_in_invocation": len(pending) - index}), flush=True)
    finally:
        if client is not None:
            client.close()
    new_by_id: dict[str, dict[str, Any]] = {}
    for planned in remaining:
        terminal = ledger.terminal_row(str(planned["physical_call_key"])) or {}
        outcome = dict((terminal.get("result") or {}).get("outcome") or {})
        if terminal.get("event") != "SUCCEEDED" or outcome.get("call_id") != planned["call_id"]:
            raise RuntimeError(f"continuation incomplete: {planned['call_id']}")
        new_by_id[str(planned["call_id"])] = outcome
    all_by_id = {str(item["call_id"]): item for item in carried}
    if set(all_by_id) & set(new_by_id):
        raise RuntimeError("carried/new outcome overlap")
    all_by_id.update(new_by_id)
    ordered = [all_by_id[str(item["call_id"])] for item in all_base]
    if len(ordered) != 4218:
        raise RuntimeError("merged E4 coverage is incomplete")
    hashes: dict[str, str] = {}
    for kind in ("response_core", "response_raw", "qa"):
        path = args.out_dir / f"{kind}_outcomes_private.jsonl"
        write_jsonl(path, [item for item in ordered if item["call_kind"] == kind])
        hashes[kind] = sha256_file(path)
    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_READY_FOR_VERSIONED_E5_ZERO_API_SCORING",
        "total_completed_calls": len(ordered),
        "carried_calls": len(carried),
        "new_completed_calls": len(new_by_id),
        "new_physical_attempts": ledger.started_attempts,
        "new_failed_physical_attempts": len(ledger.failures()),
        "call_distribution": dict(Counter(str(item["call_kind"]) for item in ordered)),
        "outcome_sha256": hashes,
        "cost_estimate_sha256": identity,
        "pm_or_route_changed": False,
        "external_outcome_used_to_change_pm_or_metrics": False,
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
