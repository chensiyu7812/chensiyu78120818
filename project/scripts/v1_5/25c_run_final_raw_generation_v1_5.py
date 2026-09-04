#!/usr/bin/env python3
"""Preflight or execute the frozen, resumable P2 raw-user generation plan."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from metacom_pm.api import make_client, require_reported_usage, request_log
from metacom_pm.config import endpoint_from_config, load_config
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
from metacom_pm.v1_5_final_raw_generation import (
    FinalRawStateDraft,
    RAW_GENERATION_PROTOCOL,
    RAW_REALIZATION_PROTOCOL,
    compile_raw_user_state,
    deterministically_realize_raw_draft,
    validate_raw_draft_for_blueprint,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_PROTOCOL = "pm-v1.5-p2-raw-user-generation-plan-v9"
PLAN_STATUS = "FROZEN_READY_FOR_256_RAW_USER_CALLS"
EXECUTION_PROTOCOL = "pm-v1.5-p2-raw-user-generation-execution-v9"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)] if path.is_file() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_raw_generation_plan_v9",
    )
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_final_candidate_first_v8/private",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_raw_generation_execution_v9",
    )
    parser.add_argument(
        "--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml"
    )
    parser.add_argument("--max-new-calls", type=int, default=None)
    parser.add_argument(
        "--history-shape",
        choices=("SMALL", "MEDIUM", "EVO_LIKE_LARGE"),
        default=None,
        help="Outcome-blind execution subset for bounded smoke tests.",
    )
    parser.add_argument(
        "--state-id",
        action="append",
        default=[],
        help=(
            "Execute only these pre-frozen state IDs. Repeat the option for a "
            "bounded, coverage-directed development smoke; it never changes a prompt."
        ),
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    plan = read_json(args.plan_dir / "generation_plan_report.json")
    manifest = read_json(args.plan_dir / "freeze_manifest.json")
    calls = _rows(args.plan_dir / "call_plan.jsonl")
    blueprint_path = args.blueprint_dir / "construction_blueprint.jsonl"
    blueprints = {str(row["state_id"]): row for row in _rows(blueprint_path)}
    config = load_config(args.experiment_config)
    endpoint = endpoint_from_config(config, "p2_data_generator")
    identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }
    lineage_ok = (
        plan.get("protocol") == PLAN_PROTOCOL
        and plan.get("status") == PLAN_STATUS
        and manifest.get("protocol") == PLAN_PROTOCOL
        and manifest.get("status") == PLAN_STATUS
        and manifest.get("call_plan_sha256")
        == sha256_file(args.plan_dir / "call_plan.jsonl")
        and manifest.get("generation_plan_report_sha256")
        == sha256_file(args.plan_dir / "generation_plan_report.json")
        and manifest.get("blueprint_sha256") == sha256_file(blueprint_path)
        and manifest.get("generator_identity") == identity
        and len(calls) == len(blueprints) == 256
        and all(
            row["messages_sha256"] == sha256_text(canonical_json(row["messages"]))
            and row["h1_gold_read"] is False
            and row["external_lockbox_read"] is False
            for row in calls
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = args.out_dir / "raw_states.jsonl"
    rejected_path = args.out_dir / "rejected_attempts.jsonl"
    accepted_rows = _rows(accepted_path)
    accepted = {str(row["state_id"]): row for row in accepted_rows}
    rejected_rows = _rows(rejected_path)
    rejected_counts: dict[str, int] = {}
    for row in rejected_rows:
        state_id = str(row["state_id"])
        rejected_counts[state_id] = rejected_counts.get(state_id, 0) + 1
    existing_ok = (
        len(accepted) == len(accepted_rows)
        and set(accepted) <= set(blueprints)
        and all(row.get("protocol") == RAW_GENERATION_PROTOCOL for row in accepted.values())
        and all(count <= 3 for count in rejected_counts.values())
    )
    all_pending = [row for row in calls if str(row["state_id"]) not in accepted]
    pending = [
        row
        for row in all_pending
        if args.history_shape is None or row["history_shape"] == args.history_shape
    ]
    requested_state_ids = set(args.state_id)
    if requested_state_ids:
        unknown = requested_state_ids - set(blueprints)
        if unknown:
            raise ValueError(f"unknown --state-id values: {sorted(unknown)}")
        pending = [row for row in pending if str(row["state_id"]) in requested_state_ids]
    key_present = bool(os.environ.get(endpoint.api_key_env, ""))
    preflight = {
        "protocol": EXECUTION_PROTOCOL,
        "status": (
            "READY_FOR_EXECUTION"
            if lineage_ok and existing_ok and key_present
            else "BLOCKED_ONLY_ON_SYNTHETIC_GENERATOR_API_KEY"
            if lineage_ok and existing_ok
            else "BLOCKED_BY_PLAN_LINEAGE_OR_EXISTING_OUTPUT"
        ),
        "planned_calls": 256,
        "accepted_states": len(accepted),
        "rejected_paid_attempts": len(rejected_rows),
        "remaining_states": len(all_pending),
        "eligible_pending_for_requested_shape": len(pending),
        "requested_history_shape": args.history_shape,
        "requested_state_ids": sorted(requested_state_ids),
        "lineage_ok": lineage_ok,
        "existing_outputs_ok": existing_ok,
        "api_key_environment_variable": endpoint.api_key_env,
        "api_key_present": key_present,
        "run_requested": args.run,
        "api_calls_made_now": 0,
    }
    write_json(args.out_dir / "execution_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if not lineage_ok or not existing_ok or not key_present:
        raise RuntimeError(canonical_json(preflight))

    maximum = len(pending) if args.max_new_calls is None else int(args.max_new_calls)
    if maximum < 0:
        raise ValueError("--max-new-calls must be nonnegative")
    client = make_client(endpoint)
    paid_now = 0
    accepted_now = 0
    try:
        for call in pending:
            if paid_now >= maximum:
                break
            state_id = str(call["state_id"])
            if rejected_counts.get(state_id, 0) >= 3:
                continue
            result, parsed = client.chat(
                list(call["messages"]),
                temperature=float(call["generation"]["temperature"]),
                max_tokens=int(call["generation"]["max_output_tokens"]),
                seed=int(call["generation"]["seed"]),
                response_schema=FinalRawStateDraft,
                retries=2,
            )
            paid_now += 1
            assert parsed is not None
            provider_draft = parsed
            parsed = deterministically_realize_raw_draft(
                draft=provider_draft, row=blueprints[state_id]
            )
            validation = validate_raw_draft_for_blueprint(
                draft=parsed, row=blueprints[state_id]
            )
            trace = request_log(
                stage="p2_raw_user_generation",
                endpoint=endpoint,
                messages=list(call["messages"]),
                result=result,
                parsed=provider_draft,
                error=None if validation["status"] == "PASS" else canonical_json(validation),
                prompt_hash=str(call["messages_sha256"]),
                record_ids={"call_id": call["call_id"], "state_id": state_id},
            )
            if validation["status"] != "PASS":
                append_jsonl(
                    rejected_path,
                    {
                        "protocol": EXECUTION_PROTOCOL,
                        "state_id": state_id,
                        "call_id": call["call_id"],
                        "validation": validation,
                        "trace": trace,
                    },
                )
                rejected_counts[state_id] = rejected_counts.get(state_id, 0) + 1
                continue
            usage = require_reported_usage(result.usage, stage="p2_raw_user_generation")
            compiled = compile_raw_user_state(draft=parsed, row=blueprints[state_id])
            append_jsonl(
                accepted_path,
                {
                    **compiled,
                    "generator_identity": identity,
                    "provider_request_hash": result.request_hash,
                    "provider_draft_sha256": sha256_text(
                        canonical_json(provider_draft.model_dump(mode="json"))
                    ),
                    "construction_realizer_protocol": RAW_REALIZATION_PROTOCOL,
                    "realized_draft_sha256": sha256_text(
                        canonical_json(parsed.model_dump(mode="json"))
                    ),
                    "usage": usage,
                    "completed_at": utc_now(),
                },
            )
            accepted[state_id] = compiled
            accepted_now += 1
    finally:
        client.close()

    summary = {
        "protocol": EXECUTION_PROTOCOL,
        "status": "COMPLETE" if len(accepted) == 256 else "PARTIAL_RESUMABLE",
        "planned_states": 256,
        "accepted_states": len(accepted),
        "accepted_now": accepted_now,
        "paid_attempts_now": paid_now,
        "rejected_paid_attempts_total": sum(rejected_counts.values()),
        "remaining_states": 256 - len(accepted),
        "resumable": True,
        "plan_manifest_sha256": sha256_file(args.plan_dir / "freeze_manifest.json"),
        "external_lockbox_read": False,
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
