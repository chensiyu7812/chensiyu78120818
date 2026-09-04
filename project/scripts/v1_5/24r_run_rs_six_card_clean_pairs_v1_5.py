#!/usr/bin/env python3
"""Preflight or run the resumable six-card R0/RS clean-pair generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_text,
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-six-card-clean-pair-execution-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_six_card_clean_pair_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v1_execution",
    )
    parser.add_argument(
        "--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument("--max-new-calls", type=int, default=64)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    report = read_json(args.plan_dir / "plan_report.json")
    calls = [dict(row) for row in iter_jsonl(args.plan_dir / "call_plan.jsonl")]
    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }
    lineage_ok = (
        report["status"] == "READY_FOR_NO_PM_DIRECT_EFFECT_GENERATION"
        and report["generator_identity"] == identity
        and report["supporter_generation_treatment_sha256"]
        == generation.digest()
        and len(calls) == int(report["planned_generation_calls"]) == 64
        and all(row["generator_identity"] == identity for row in calls)
    )
    key_present = bool(os.environ.get(endpoint.api_key_env, ""))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_EXECUTION"
            if lineage_ok and key_present
            else (
                "BLOCKED_ONLY_ON_GENERATOR_API_KEY"
                if lineage_ok
                else "BLOCKED_BY_PLAN_OR_LINEAGE"
            )
        ),
        "planned_calls": len(calls),
        "lineage_ok": lineage_ok,
        "api_key_environment_variable": endpoint.api_key_env,
        "api_key_present": key_present,
        "run_requested": args.run,
        "api_calls_made": 0,
    }
    write_json(args.out_dir / "execution_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if not lineage_ok or not key_present:
        raise RuntimeError(canonical_json(preflight))

    outcomes_path = args.out_dir / "generation_outcomes.jsonl"
    existing = {
        (str(row["pair_id"]), str(row["arm"])): dict(row)
        for row in (
            iter_jsonl(outcomes_path) if outcomes_path.is_file() else []
        )
    }
    pending = [
        row
        for row in calls
        if (str(row["pair_id"]), str(row["arm"])) not in existing
    ]
    if len(pending) > args.max_new_calls:
        raise RuntimeError(
            f"{len(pending)} calls remain but --max-new-calls="
            f"{args.max_new_calls}"
        )

    client = make_client(endpoint)
    completed_now = 0
    try:
        for row in pending:
            request = dict(row["generation"])
            result, _ = client.chat(
                list(row["messages"]),
                temperature=float(request["temperature"]),
                max_tokens=int(request["max_output_tokens"]),
                seed=int(request["seed"]),
                response_schema=None,
                retries=3,
            )
            error = generation.completion_gate_error(
                normalized_finish_reason=result.normalized_finish_reason,
                provider_finish_reason=result.provider_finish_reason,
            )
            if error is not None:
                raise RuntimeError(error)
            response = generation.normalize_output(result.text)
            if not response:
                raise RuntimeError("generator returned an empty response")
            usage = require_reported_usage(
                result.usage, stage="rs_six_card_clean_pair"
            )
            append_jsonl(
                outcomes_path,
                {
                    "protocol": PROTOCOL,
                    "pair_id": row["pair_id"],
                    "state_id": row["state_id"],
                    "user_id": row["user_id"],
                    "arm": row["arm"],
                    "action_id": row["action_id"],
                    "selected_strategy_card_id": row[
                        "selected_strategy_card_id"
                    ],
                    "selected_strategy_family": row[
                        "selected_strategy_family"
                    ],
                    "response": response,
                    "prompt_sha256": row["prompt_sha256"],
                    "messages_sha256": sha256_text(
                        canonical_json(row["messages"])
                    ),
                    "request_hash": result.request_hash,
                    "provider_finish_reason": result.provider_finish_reason,
                    "normalized_finish_reason": result.normalized_finish_reason,
                    "usage": usage,
                    "latency_ms": result.latency_ms,
                    "model": endpoint.model,
                    "model_family": endpoint.family,
                    "completed_at": utc_now(),
                },
            )
            completed_now += 1
    finally:
        client.close()

    completed = (
        [dict(row) for row in iter_jsonl(outcomes_path)]
        if outcomes_path.is_file()
        else []
    )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "COMPLETE" if len(completed) == len(calls) else "PARTIAL_RESUMABLE"
        ),
        "planned_calls": len(calls),
        "completed_calls": len(completed),
        "completed_now": completed_now,
        "remaining_calls": len(calls) - len(completed),
        "resumable": True,
    }
    write_json(args.out_dir / "generation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
