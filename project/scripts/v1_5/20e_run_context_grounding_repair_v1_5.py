#!/usr/bin/env python3
"""Dry-run or execute the frozen PM-v1.5 context-grounding repair stage.

``--scope pilot`` certifies six preselected repair shapes and never emits a
production overlay. ``--scope full`` requires all 25 frozen DATA_DEFECT rows
and is the only mode that can emit ``repair_overlays.jsonl`` for canonical
whole-corpus recompilation. No mode recompiles or publishes V8.19 by itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.bounded_retry import (
    TERMINAL_DISPOSITION,
    execute_with_bounded_retry,
    failure_metadata,
    retry_ledger_summary,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.paid_run_release import (
    require_paid_run_release,
    resolve_first_unconsumed_output_directory,
)
from metacom_pm.pm_v2_data import load_bundles
from metacom_pm.v1_5_context_grounding_repair import (
    DEFAULT_CLASSIFICATION_PATH,
    load_context_grounding_defect_classification,
)
from metacom_pm.v1_5_context_grounding_repair_overlay import (
    RepairOverlayRecord,
    apply_repair_overlay_to_bundles,
)
from metacom_pm.v1_5_context_grounding_repair_run import (
    REPAIR_BACKOFF_SECONDS,
    REPAIR_MAX_PHYSICAL_ATTEMPTS_PER_CALL,
    build_repair_run_plan,
    materialize_full_repair_overlays,
    repair_stage,
    response_schema_for_mode,
    validate_repair_result,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLES = (
    ROOT
    / "data"
    / "pm_v1_5_formal_v8_18_duplicate_repair_candidate"
    / "pm_v2_bundles.jsonl"
)
DEFAULT_MAX_ESTIMATED_USD = {"pilot": 0.03, "full": 0.10}
DEFAULT_MAX_API_CALLS = {"pilot": 18, "full": 75}
DEFAULT_MAX_INPUT_TOKENS_PER_CALL = 5000


def _budget_gate(cost: dict[str, Any], *, scope: str) -> dict[str, Any]:
    limits = {
        "max_physical_api_attempts": DEFAULT_MAX_API_CALLS[scope],
        "max_estimated_usd": DEFAULT_MAX_ESTIMATED_USD[scope],
        "max_input_tokens_per_call": DEFAULT_MAX_INPUT_TOKENS_PER_CALL,
    }
    checks = {
        "physical_api_attempts": int(cost["maximum_physical_api_attempts"])
        <= limits["max_physical_api_attempts"],
        "estimated_cost_usd": float(cost["maximum_estimated_cost_usd"])
        <= limits["max_estimated_usd"],
        "max_input_tokens_per_call": int(cost["max_input_tokens"])
        <= limits["max_input_tokens_per_call"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": limits,
    }


def _bind_budget(cost: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value for key, value in cost.items() if key != "cost_estimate_sha256"
    }
    payload["budget_gate"] = gate
    payload["cost_estimate_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _persist_or_require_plan(
    *, out_dir: Path, cost: dict[str, Any], rows: list[dict[str, Any]], dry_run: bool
) -> None:
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent or not dry_run:
        if not estimate_path.is_file() or not plan_path.is_file():
            raise RuntimeError("repair API mode/spent ledger requires its saved dry-run")
        if read_json(estimate_path) != cost or list(iter_jsonl(plan_path)) != rows:
            raise RuntimeError("saved repair dry-run differs from the current exact plan")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, cost)
    write_jsonl(plan_path, rows)


def _usage_totals(ledger: PersistentAttemptLedger) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in ledger.event_rows:
        if row.get("event") not in {"SUCCEEDED", "FAILED"} or row.get("usage") is None:
            continue
        usage = require_reported_usage(row["usage"], stage="context repair attempt")
        for key in totals:
            totals[key] += usage[key]
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument("--bundles", type=Path, default=DEFAULT_BUNDLES)
    parser.add_argument(
        "--classification", type=Path, default=Path(DEFAULT_CLASSIFICATION_PATH)
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--accept-cost-estimate-sha256")
    args = parser.parse_args()

    experiment = load_config(args.config)
    pm_config = load_config(args.pm_v1_5_config)
    if pm_config.get("version") != "pm-v1.5":
        raise RuntimeError("context repair runner requires the pm-v1.5 config")
    generation = dict(pm_config["data_generation"])
    if generation.get("generator_endpoint") != "synthetic_generator":
        raise RuntimeError("repair must use the frozen synthetic_generator endpoint")
    pricing = dict(generation["pricing_usd_per_mtok"])
    endpoint = endpoint_from_config(experiment, "synthetic_generator")
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])
    if safety_factor < 1.0 or not bool(
        pm_config["api_cost_planning"]["fail_on_reported_input_overrun"]
    ):
        raise RuntimeError("repair requires conservative input planning and fail-on-overrun")

    records = load_context_grounding_defect_classification(args.classification)
    cost, plan_rows, messages_by_call = build_repair_run_plan(
        records=records,
        scope=args.scope,
        endpoint=endpoint,
        original_bundles_path=args.bundles,
        classification_path=args.classification,
        experiment_config_path=args.config,
        pm_v1_5_config_path=args.pm_v1_5_config,
        input_token_safety_factor=safety_factor,
        input_usd_per_mtok=float(pricing["input"]),
        output_usd_per_mtok=float(pricing["output"]),
    )
    gate = _budget_gate(cost, scope=args.scope)
    exact_cost = _bind_budget(cost, gate)
    if gate["status"] != "PASS":
        raise RuntimeError(f"context repair budget gate failed: {gate}")

    base_out = args.out_dir or (
        ROOT / "outputs" / f"pm_v1_5_context_grounding_repair_{args.scope}_candidate"
    )
    out_dir = resolve_first_unconsumed_output_directory(
        base_out, config=pm_config, config_path=args.pm_v1_5_config
    )
    _persist_or_require_plan(
        out_dir=out_dir, cost=exact_cost, rows=plan_rows, dry_run=bool(args.dry_run)
    )
    print({"status": "DRY_RUN_COMPLETE" if args.dry_run else "RUN_PLANNED", **exact_cost})
    if args.dry_run:
        return

    identity = str(args.accept_cost_estimate_sha256 or "")
    if identity != exact_cost["cost_estimate_sha256"]:
        raise RuntimeError("repair API mode requires the exact accepted cost hash")
    stage = repair_stage(args.scope)
    release = require_paid_run_release(
        pm_config,
        config_path=args.pm_v1_5_config,
        stage=stage,
        run=True,
        run_identity=identity,
    )
    manifest_path = out_dir / "run_manifest.json"
    ensure_run_manifest(
        manifest_path,
        {
            "stage": stage,
            "scope": args.scope,
            "cost_estimate_sha256": identity,
            "contract_sha256": exact_cost["contract_sha256"],
            "call_plan_sha256": exact_cost["call_plan_sha256"],
            "physical_call_keys": [row["physical_call_key"] for row in plan_rows],
            "release": release,
        },
    )
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={
            str(row["physical_call_key"]): int(row["maximum_physical_attempts"])
            for row in plan_rows
        },
        maximum_total_attempts=int(exact_cost["maximum_physical_api_attempts"]),
    )
    records_by_state = {record.state_id: record for record in records}
    pending = [
        row for row in plan_rows if not ledger.succeeded(str(row["physical_call_key"]))
    ]
    client = make_client(endpoint) if pending else None
    try:
        for row in pending:
            call_key = str(row["physical_call_key"])
            record = records_by_state[str(row["state_id"])]
            schema = response_schema_for_mode(str(row["repair_mode"]))
            reservation, call, parsed = execute_with_bounded_retry(
                ledger,
                call_key,
                record_ids={
                    "state_id": row["state_id"],
                    "user_id": row["user_id"],
                    "repair_mode": row["repair_mode"],
                    "scope": args.scope,
                },
                prompt_sha256=str(row["prompt_sha256"]),
                call_fn=lambda row=row, schema=schema, call_key=call_key: client.chat(
                    messages_by_call[call_key],
                    temperature=0.0,
                    max_tokens=int(row["maximum_output_tokens"]),
                    seed=int(row["seed"]),
                    response_schema=schema,
                    retries=1,
                ),
                max_provider_output_attempts=2,
                backoff_seconds=REPAIR_BACKOFF_SECONDS,
                sleep=time.sleep,
                capture_provider_output_text=True,
            )
            try:
                usage = require_reported_usage(call.usage, stage="context repair")
                if usage["prompt_tokens"] > int(row["input_token_upper_bound"]):
                    raise RuntimeError("reported prompt_tokens exceed frozen repair bound")
                validated = validate_repair_result(record, parsed)
            except Exception as exc:
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=call.request_hash,
                    usage=call.usage,
                    error=f"{type(exc).__name__}: {exc}",
                    result={
                        "provider_response": call.raw_response,
                        "parsed_repair": (
                            parsed.model_dump(mode="json")
                            if parsed is not None
                            else None
                        ),
                    },
                    metadata=failure_metadata(
                        retry_class="repair_postcondition_failure",
                        retry_disposition=TERMINAL_DISPOSITION,
                    ),
                )
                raise
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash=call.request_hash,
                usage=usage,
                error=None,
                result={
                    "validated_repair": validated,
                    "provider_response": call.raw_response,
                    "structured_output_audit": call.structured_output_audit,
                },
                metadata={"repair_run_protocol": exact_cost["protocol"]},
            )
    finally:
        if client is not None:
            client.close()

    validated_by_state: dict[str, dict[str, Any]] = {}
    for row in plan_rows:
        key = str(row["physical_call_key"])
        if not ledger.succeeded(key):
            raise RuntimeError(f"repair run incomplete: {row['state_id']}")
        terminal = ledger.terminal_row(key) or {}
        validated = dict((terminal.get("result") or {}).get("validated_repair") or {})
        if validated.get("state_id") != row["state_id"]:
            raise RuntimeError("successful repair ledger result is missing or mismatched")
        validated_by_state[str(row["state_id"])] = validated

    results_path = out_dir / "validated_repairs.jsonl"
    write_jsonl(results_path, [validated_by_state[row["state_id"]] for row in plan_rows])
    output_artifacts: dict[str, tuple[Path, bool]] = {
        "validated_repairs": (results_path, True),
        "physical_attempt_ledger": (ledger_path, True),
    }
    if args.scope == "full":
        bundles = load_bundles(str(args.bundles))
        overlays = materialize_full_repair_overlays(
            records=records,
            bundles=bundles,
            classification_sha256=sha256_file(args.classification),
            validated_results=validated_by_state,
        )
        # Execute the production exact-25 fail-closed validator before writing
        # an overlay artifact. This is validation only; V8.19 recompilation is
        # a later zero-API stage using the real frozen encoder/config.
        apply_repair_overlay_to_bundles(
            bundles=bundles,
            overlays=overlays,
            classification_records=records,
            expected_classification_sha256=sha256_file(args.classification),
        )
        overlay_path = out_dir / "repair_overlays.jsonl"
        write_jsonl(overlay_path, [
            RepairOverlayRecord.model_validate(item).model_dump(mode="json")
            for item in overlays
        ])
        output_artifacts["repair_overlays"] = (overlay_path, True)

    usage = _usage_totals(ledger)
    actual_cost = (
        usage["prompt_tokens"] * float(pricing["input"])
        + usage["completion_tokens"] * float(pricing["output"])
    ) / 1_000_000
    summary_path = out_dir / "summary.json"
    write_json(
        summary_path,
        {
            "status": "PASS",
            "scope": args.scope,
            "stage": stage,
            "cost_estimate_sha256": identity,
            "contract_sha256": exact_cost["contract_sha256"],
            "logical_calls": len(plan_rows),
            "actual_usage": usage,
            "actual_cost_usd": actual_cost,
            "retry_summary": retry_ledger_summary(
                ledger, [str(row["physical_call_key"]) for row in plan_rows]
            ),
            "production_overlay_emitted": args.scope == "full",
        },
    )
    output_artifacts["summary"] = (summary_path, False)
    attestation_path = out_dir / "artifact_attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage=stage,
        inputs={
            "experiment_config": args.config,
            "pm_v1_5_config": args.pm_v1_5_config,
            "original_bundles": args.bundles,
            "classification": args.classification,
            "cost_estimate": out_dir / "cost_estimate.json",
            "call_plan": out_dir / "call_plan.jsonl",
            "run_manifest": manifest_path,
        },
        outputs=output_artifacts,
        parameters={
            "scope": args.scope,
            "cost_estimate_sha256": identity,
            "contract_sha256": exact_cost["contract_sha256"],
        },
        expected={"logical_calls": len(plan_rows)},
    )
    print(read_json(summary_path))


if __name__ == "__main__":
    main()
