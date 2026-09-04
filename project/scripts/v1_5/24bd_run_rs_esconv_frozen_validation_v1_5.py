#!/usr/bin/env python3
"""Preflight or run the frozen ESConv R0/RS validation panel once."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PROTOCOL = "pm-v1.5-rs-esconv-frozen-validation-plan-v1"
PROTOCOL = "pm-v1.5-rs-esconv-frozen-validation-execution-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1_execution",
    )
    parser.add_argument(
        "--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument("--max-new-calls", type=int, default=80)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    report = read_json(args.plan_dir / "plan_report.json")
    manifest = read_json(args.plan_dir / "freeze_manifest.json")
    calls = _rows(args.plan_dir / "call_plan.jsonl")
    selected = _rows(args.plan_dir / "selected_states.jsonl")
    expected_hashes = dict(manifest.get("outputs") or {})
    manifest_ok = (
        manifest.get("protocol") == PLAN_PROTOCOL
        and manifest.get("status") == "FROZEN_READY_FOR_PAIRED_GENERATION"
        and all(
            (args.plan_dir / name).is_file()
            and sha256_file(args.plan_dir / name) == digest
            for name, digest in expected_hashes.items()
        )
    )

    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(
        experiment, generation.generator_endpoint
    )
    identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }
    expected_pairs = {str(row["pair_id"]) for row in selected}
    expected_arms = {
        (pair_id, arm)
        for pair_id in expected_pairs
        for arm in ("R0", "RS")
    }
    call_keys = {
        (str(row["pair_id"]), str(row["arm"])) for row in calls
    }
    lineage_ok = (
        manifest_ok
        and report.get("protocol") == PLAN_PROTOCOL
        and report.get("status") == "FROZEN_READY_FOR_PAIRED_GENERATION"
        and report.get("generator_identity") == identity
        and report.get("supporter_generation_treatment_sha256")
        == generation.digest()
        and len(selected) == 40
        and len({str(row["user_id"]) for row in selected}) == 40
        and len(calls) == 80
        and call_keys == expected_arms
        and all(row["generator_identity"] == identity for row in calls)
        and all(
            row["generation"] == {**generation.payload(), "seed": 20260730}
            for row in calls
        )
        and all(
            row["outcome_label"] == "UNKNOWN_FROZEN_BEFORE_GENERATION"
            for row in selected
        )
    )
    key_present = bool(os.environ.get(endpoint.api_key_env, ""))
    estimated_input_tokens = sum(
        int(row["estimated_input_tokens"]) for row in calls
    )
    max_output_tokens_upper_bound = sum(
        int(row["generation"]["max_output_tokens"]) for row in calls
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.out_dir / "generation_outcomes.jsonl"
    existing_rows = (
        _rows(outcomes_path) if outcomes_path.is_file() else []
    )
    existing = {
        (str(row["pair_id"]), str(row["arm"])): row
        for row in existing_rows
    }
    existing_ok = (
        len(existing) == len(existing_rows)
        and set(existing) <= expected_arms
        and all(
            str(row.get("normalized_finish_reason")) == "complete"
            for row in existing.values()
        )
    )
    pending = [
        row
        for row in calls
        if (str(row["pair_id"]), str(row["arm"])) not in existing
    ]
    preflight = {
        "protocol": PROTOCOL,
        "status": (
            "READY_FOR_EXECUTION"
            if lineage_ok and existing_ok and key_present
            else (
                "BLOCKED_ONLY_ON_GENERATOR_API_KEY"
                if lineage_ok and existing_ok
                else "BLOCKED_BY_PLAN_LINEAGE_OR_EXISTING_OUTPUT"
            )
        ),
        "planned_calls": len(calls),
        "completed_calls": len(existing),
        "remaining_calls": len(pending),
        "lineage_ok": lineage_ok,
        "existing_outputs_ok": existing_ok,
        "api_key_environment_variable": endpoint.api_key_env,
        "api_key_present": key_present,
        "estimated_input_tokens_remaining": sum(
            int(row["estimated_input_tokens"]) for row in pending
        ),
        "estimated_input_tokens_full_panel": estimated_input_tokens,
        "max_output_tokens_upper_bound_full_panel": (
            max_output_tokens_upper_bound
        ),
        "run_requested": args.run,
        "api_calls_made": 0,
    }
    write_json(args.out_dir / "execution_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if not lineage_ok or not existing_ok or not key_present:
        raise RuntimeError(canonical_json(preflight))
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
                result.usage, stage="rs_esconv_frozen_validation"
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
                    "provider_finish_reason": (
                        result.provider_finish_reason
                    ),
                    "normalized_finish_reason": (
                        result.normalized_finish_reason
                    ),
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

    completed = _rows(outcomes_path)
    completed_keys = {
        (str(row["pair_id"]), str(row["arm"])) for row in completed
    }
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "COMPLETE"
            if len(completed) == len(calls)
            and completed_keys == expected_arms
            else "PARTIAL_RESUMABLE"
        ),
        "planned_calls": len(calls),
        "completed_calls": len(completed),
        "completed_now": completed_now,
        "remaining_calls": len(calls) - len(completed),
        "resumable": True,
        "plan_freeze_manifest_sha256": sha256_file(
            args.plan_dir / "freeze_manifest.json"
        ),
    }
    write_json(args.out_dir / "generation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
