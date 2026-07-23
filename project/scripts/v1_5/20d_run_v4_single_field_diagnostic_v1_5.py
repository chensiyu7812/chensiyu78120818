#!/usr/bin/env python3
"""Run the calibration-only balanced PM-v1.5 proxy diagnostic.

This runner never changes the consumed V4 gate, never authorizes V5, and never
creates held-out evidence. A dry run creates a content-bound 8-call plan. A
paid run additionally requires the central release manifest to approve that
exact cost-estimate identity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from metacom_pm.api import (
    GEMINI_USAGE_BREAKDOWN_KEYS,
    RetryableProviderError,
    chat_request_payload,
    make_client,
    request_payload_has_schema,
    require_reported_usage,
)
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
)
from metacom_pm.bounded_retry import (
    BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES,
    RETRYABLE_UP_TO_FULL_BUDGET,
    RETRY_CONTRACT_PROTOCOL,
    TERMINAL_DISPOSITION,
    call_retry_blocker,
    execute_with_bounded_retry,
    failure_metadata,
    retry_ledger_summary,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.text import conservative_token_bound, estimate_tokens
from metacom_pm.v1_5_automated_semantic_review import (
    build_judge_endpoint_descriptors,
)
from metacom_pm.v1_5_judge_isolation import require_judge_role_isolation
from metacom_pm.v1_5_semantic_review_diagnostic import (
    ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE,
    DETERMINISTIC_DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_POLARITIES,
    DIAGNOSTIC_RUN_PROTOCOL,
    DIAGNOSTIC_RUN_STAGE,
    DIAGNOSTIC_RUN_STATUS,
    SEMANTIC_DIAGNOSTIC_FIELDS,
    SingleFieldDiagnosticOutput,
    aggregate_v4_single_field_diagnostic,
    assess_single_field_diagnostic_output,
    validate_v4_diagnostic_source_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_CODE_PATHS = {
    "diagnostic_runner": Path(__file__).resolve(),
    "diagnostic_contract": ROOT
    / "src"
    / "metacom_pm"
    / "v1_5_semantic_review_diagnostic.py",
    "api_transport": ROOT / "src" / "metacom_pm" / "api.py",
    "attempt_ledger": ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
    "bounded_retry": ROOT / "src" / "metacom_pm" / "bounded_retry.py",
    "paid_run_release": ROOT / "src" / "metacom_pm" / "paid_run_release.py",
    "judge_isolation": ROOT / "src" / "metacom_pm" / "v1_5_judge_isolation.py",
    "config_loader": ROOT / "src" / "metacom_pm" / "config.py",
    "io": ROOT / "src" / "metacom_pm" / "io.py",
    "token_planning": ROOT / "src" / "metacom_pm" / "text.py",
}


def _provider_usage_breakdown(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    return {
        key: int(usage[key])
        for key in GEMINI_USAGE_BREAKDOWN_KEYS
        if key in usage
    }


def _audited_result_payload(
    *, result: Any, parsed: Any, assessment: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Retain bounded provider/accounting evidence on success or failure."""

    payload: dict[str, Any] = {
        "parsed": parsed.model_dump(mode="json") if parsed is not None else None,
        "raw_text": result.text if result is not None else None,
        "provider_usage_breakdown": (
            _provider_usage_breakdown(result.usage) if result is not None else {}
        ),
        "provider_response_sha256": (
            sha256_text(canonical_json(result.raw_response))
            if result is not None
            else None
        ),
        "structured_output_audit": (
            result.structured_output_audit if result is not None else None
        ),
    }
    if assessment is not None:
        payload["assessment"] = assessment
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--diagnostic-packet",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_v4_root_cause_balanced_proxy_diagnostic_v4_2"
            / "diagnostic_packet.json"
        ),
    )
    parser.add_argument(
        "--observed-controls",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_automated_semantic_review_v8_7_native_gemini_candidate"
            / "controls.json"
        ),
    )
    parser.add_argument(
        "--observed-control-judgments",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_automated_semantic_review_v8_7_native_gemini_candidate"
            / "control_judgments.json"
        ),
    )
    parser.add_argument(
        "--observed-gate-report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_automated_semantic_review_v8_7_native_gemini_candidate"
            / "gate_report.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_v4_root_cause_balanced_proxy_diagnostic_v4_2_candidate"
        ),
    )
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_frozen_config(pm_config: dict[str, Any]) -> dict[str, Any]:
    raw = pm_config.get("v4_root_cause_single_field_diagnostic")
    if not isinstance(raw, dict):
        raise RuntimeError("PM-v1.5 config lacks the V4 root-cause diagnostic")
    cfg = dict(raw)
    if (
        cfg.get("protocol") != DIAGNOSTIC_RUN_PROTOCOL
        or cfg.get("stage") != DIAGNOSTIC_RUN_STAGE
        or cfg.get("status") != "CALIBRATION_DIAGNOSTIC_ONLY_NOT_FORMAL_GATE"
        or list(cfg.get("diagnostic_fields") or []) != list(DIAGNOSTIC_FIELDS)
        or list(cfg.get("semantic_fields") or [])
        != list(SEMANTIC_DIAGNOSTIC_FIELDS)
        or list(cfg.get("deterministic_fields") or [])
        != list(DETERMINISTIC_DIAGNOSTIC_FIELDS)
        or list(cfg.get("diagnostic_polarities") or [])
        != list(DIAGNOSTIC_POLARITIES)
        or cfg.get("semantic_protocol")
        != ATOMIC_DEFINITION_AND_EXHAUSTIVE_EVIDENCE
        or float(cfg.get("temperature", -1)) != 0.0
        or int(cfg.get("maximum_physical_attempts_per_logical_call") or 0) != 10
        or int(cfg.get("maximum_provider_output_failures_per_logical_call") or 0)
        != 2
        or list(cfg.get("transport_backoff_seconds") or [])
        != [10, 30, 60, 120, 300, 300, 600, 600, 900]
    ):
        raise RuntimeError("V4 root-cause diagnostic config drift")
    endpoint_names = list(cfg.get("judge_endpoints") or [])
    if len(endpoint_names) != 2 or len(set(endpoint_names)) != 2:
        raise RuntimeError("diagnostic requires exactly two frozen judge aliases")
    prices = cfg.get("pricing_usd_per_mtok")
    limits = cfg.get("budget_limits")
    if (
        not isinstance(prices, dict)
        or set(prices) != {"input", "output"}
        or min(float(prices["input"]), float(prices["output"])) <= 0
        or not isinstance(limits, dict)
        or set(limits)
        != {"max_api_calls", "max_estimated_usd", "max_input_tokens_per_call"}
        or int(limits["max_api_calls"]) != 80
    ):
        raise RuntimeError("diagnostic pricing/budget contract drift")
    return cfg


def main() -> None:
    args = parse_args()
    if args.run and args.overwrite:
        raise RuntimeError("paid diagnostic runs prohibit --overwrite")
    experiment_config = load_config(args.config)
    pm_config = load_config(args.pm_v1_5_config)
    if pm_config.get("version") != "pm-v1.5":
        raise RuntimeError("diagnostic runner requires PM-v1.5")
    cfg = _require_frozen_config(pm_config)
    endpoint_names = list(cfg["judge_endpoints"])

    # The release check runs before any credential is read or client created.
    require_paid_run_release(
        pm_config,
        config_path=args.pm_v1_5_config,
        stage=DIAGNOSTIC_RUN_STAGE,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    judge_role_isolation = require_judge_role_isolation(
        experiment_config,
        pm_config,
        development_endpoint_names=endpoint_names,
    )
    endpoints = {
        name: endpoint_from_config(experiment_config, name)
        for name in endpoint_names
    }
    families = [str(endpoint.family or "") for endpoint in endpoints.values()]
    if "" in families or len(set(families)) != len(families):
        raise RuntimeError("diagnostic judge families must be declared and distinct")
    endpoint_descriptors = build_judge_endpoint_descriptors(
        experiment_config, endpoint_names
    )

    packet = read_json(args.diagnostic_packet)
    observed_controls = read_json(args.observed_controls)
    observed_judgments = read_json(args.observed_control_judgments)
    observed_gate = read_json(args.observed_gate_report)
    if not isinstance(observed_controls, list):
        raise RuntimeError("observed controls must be a list")
    if not isinstance(observed_judgments, dict) or not isinstance(observed_gate, dict):
        raise RuntimeError("observed V4 judgments/gate must be objects")
    source_validation = validate_v4_diagnostic_source_artifacts(
        packet=packet,
        observed_controls=observed_controls,
        observed_control_judgments=observed_judgments,
        observed_gate_report=observed_gate,
        endpoint_names=endpoint_names,
    )

    planning = dict(pm_config["api_cost_planning"])
    if (
        set(planning)
        != {"input_token_safety_factor", "fail_on_reported_input_overrun"}
        or float(planning["input_token_safety_factor"]) < 1.0
        or not bool(planning["fail_on_reported_input_overrun"])
    ):
        raise RuntimeError("diagnostic requires fail-closed token planning")
    prices = {key: float(value) for key, value in cfg["pricing_usd_per_mtok"].items()}
    limits = dict(cfg["budget_limits"])
    attempts_per_call = int(cfg["maximum_physical_attempts_per_logical_call"])
    provider_output_attempts = int(
        cfg["maximum_provider_output_failures_per_logical_call"]
    )
    transport_backoff_seconds = tuple(
        float(value) for value in cfg["transport_backoff_seconds"]
    )
    max_output_tokens = int(cfg["max_output_tokens"])
    temperature = float(cfg["temperature"])
    seed = int(cfg["seed"])

    call_plan: list[dict[str, Any]] = []
    execution: dict[str, dict[str, Any]] = {}
    item_by_id = {str(item["diagnostic_id"]): item for item in packet["items"]}
    for field in SEMANTIC_DIAGNOSTIC_FIELDS:
        for polarity in DIAGNOSTIC_POLARITIES:
            item = next(
                row
                for row in packet["items"]
                if row["field"] == field and row["polarity"] == polarity
            )
            messages = item["messages"]
            for endpoint_name in endpoint_names:
                endpoint = endpoints[endpoint_name]
                payload = chat_request_payload(
                    endpoint,
                    messages,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    seed=seed,
                    response_schema=SingleFieldDiagnosticOutput,
                )
                if not request_payload_has_schema(payload):
                    raise RuntimeError("diagnostic request lacks structured schema")
                payload_text = canonical_json(payload)
                prompt_sha256 = sha256_text(canonical_json(messages))
                record_ids = {
                    "diagnostic_id": str(item["diagnostic_id"]),
                    "field": field,
                    "polarity": polarity,
                    "endpoint_name": endpoint_name,
                    "judge_family": str(endpoint.family),
                }
                call_key = physical_call_key(
                    stage=DIAGNOSTIC_RUN_STAGE,
                    record_ids=record_ids,
                    prompt_sha256=prompt_sha256,
                    endpoint=endpoint,
                    request_parameters={
                        "temperature": temperature,
                        "max_tokens": max_output_tokens,
                        "seed": seed,
                        "response_schema": SingleFieldDiagnosticOutput.__name__,
                        "provider_client_retries": 1,
                        "maximum_physical_attempts": attempts_per_call,
                        "maximum_provider_output_failures": provider_output_attempts,
                        "transport_backoff_seconds": list(transport_backoff_seconds),
                    },
                )
                bound = conservative_token_bound(
                    payload_text,
                    safety_factor=float(planning["input_token_safety_factor"]),
                )
                call_plan.append(
                    {
                        **record_ids,
                        "physical_call_key": call_key,
                        "judge_model": endpoint.model,
                        "prompt_sha256": prompt_sha256,
                        "request_payload_sha256": sha256_text(payload_text),
                        "raw_estimated_input_tokens": estimate_tokens(payload_text),
                        "input_token_upper_bound": bound,
                        "maximum_output_tokens": max_output_tokens,
                        "maximum_physical_attempts": attempts_per_call,
                        "maximum_provider_output_failures": provider_output_attempts,
                        "maximum_cost_usd": attempts_per_call
                        * (
                            bound / 1_000_000 * prices["input"]
                            + max_output_tokens / 1_000_000 * prices["output"]
                        ),
                    }
                )
                execution[call_key] = {
                    "endpoint_name": endpoint_name,
                    "endpoint": endpoint,
                    "messages": messages,
                    "record_ids": record_ids,
                    "item": item,
                }

    expected_logical_calls = (
        len(SEMANTIC_DIAGNOSTIC_FIELDS)
        * len(DIAGNOSTIC_POLARITIES)
        * len(endpoint_names)
    )
    if len(call_plan) != expected_logical_calls or len(execution) != expected_logical_calls:
        raise RuntimeError("diagnostic call plan cardinality drift")
    maximum_attempts = len(call_plan) * attempts_per_call
    estimate_payload = {
        "protocol": DIAGNOSTIC_RUN_PROTOCOL,
        "stage": DIAGNOSTIC_RUN_STAGE,
        "status": "DRY_RUN_COST_ESTIMATE_NOT_A_FORMAL_GATE",
        "formal_gate": False,
        "authorizes_v5": False,
        "source_v4_result": "CONSUMED_FAILED_CLOSED",
        "source_artifacts": {
            "diagnostic_packet_sha256": sha256_file(args.diagnostic_packet),
            "observed_controls_sha256": sha256_file(args.observed_controls),
            "observed_control_judgments_sha256": sha256_file(
                args.observed_control_judgments
            ),
            "observed_gate_report_sha256": sha256_file(args.observed_gate_report),
            "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
            "experiment_config_sha256": sha256_file(args.config),
        },
        "execution_code_sha256": {
            name: sha256_file(path)
            for name, path in sorted(EXECUTION_CODE_PATHS.items())
        },
        "diagnostic_config": cfg,
        "n_fields": len(DIAGNOSTIC_FIELDS),
        "n_semantic_fields": len(SEMANTIC_DIAGNOSTIC_FIELDS),
        "n_deterministic_fields": len(DETERMINISTIC_DIAGNOSTIC_FIELDS),
        "n_polarities": len(DIAGNOSTIC_POLARITIES),
        "n_judge_families": len(endpoint_names),
        "n_logical_calls": len(call_plan),
        "maximum_physical_attempts_per_call": attempts_per_call,
        "maximum_physical_api_attempts": maximum_attempts,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
        "source_validation_sha256": sha256_text(canonical_json(source_validation)),
        "maximum_estimated_usd": sum(row["maximum_cost_usd"] for row in call_plan),
        "maximum_input_tokens_per_call": max(
            row["input_token_upper_bound"] for row in call_plan
        ),
        "pricing_usd_per_mtok": prices,
        "api_cost_planning": planning,
        "judge_role_isolation": judge_role_isolation,
        "judge_endpoint_descriptors": endpoint_descriptors,
        "call_order_protocol": "semantic-field-then-polarity-then-frozen-endpoint-v1",
        "retry_contract": {
            "protocol": RETRY_CONTRACT_PROTOCOL,
            "retryable_up_to_full_budget": sorted(RETRYABLE_UP_TO_FULL_BUDGET),
            "bounded_provider_output_retry_classes": sorted(
                BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES
            ),
            "provider_output_maximum_failures": provider_output_attempts,
            "provider_output_failures_are_independent_of_transport_attempts": True,
            "deterministic_surface_normalization": [
                "single_markdown_json_fence",
                "single_bounded_json_object",
            ],
            "never_retried": [
                "provider_request_error_4xx",
                "structured_output_validation_error",
            ],
            "backoff_seconds": list(transport_backoff_seconds),
            "cross_process_eligibility_source": "physical_attempt_ledger",
        },
        "diagnostic_result_policy": {
            "binary_support_verdict_accuracy": "required_for_instrument_ready",
            "citation_integrity": "reported_separately_not_outcome_gate",
            "joint_validated_accuracy": "reported_separately",
            "parsed_citation_defect": (
                "completed_logical_observation_no_retry_continue_frozen_matrix"
            ),
            "formal_gate_behavior_unchanged": True,
        },
        "budget_limits": limits,
    }
    estimate = {
        **estimate_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(estimate_payload)),
    }
    budget_gate = {
        "status": (
            "PASS"
            if maximum_attempts <= int(limits["max_api_calls"])
            and estimate["maximum_estimated_usd"]
            <= float(limits["max_estimated_usd"])
            and estimate["maximum_input_tokens_per_call"]
            <= int(limits["max_input_tokens_per_call"])
            else "FAIL"
        ),
        "checks": {
            "physical_api_attempts": maximum_attempts
            <= int(limits["max_api_calls"]),
            "estimated_cost_usd": estimate["maximum_estimated_usd"]
            <= float(limits["max_estimated_usd"]),
            "max_input_tokens_per_call": estimate["maximum_input_tokens_per_call"]
            <= int(limits["max_input_tokens_per_call"]),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = args.out_dir / "cost_estimate.json"
    plan_path = args.out_dir / "call_plan.jsonl"
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=args.overwrite,
        stage="PM-v1.5 V4 root-cause diagnostic",
    )
    frozen_estimate = {**estimate, "budget_gate": budget_gate}
    if args.dry_run:
        if ledger_path.is_file() and ledger_path.stat().st_size:
            if read_json(estimate_path) != frozen_estimate or list(
                iter_jsonl(plan_path)
            ) != call_plan:
                raise RuntimeError("spent diagnostic ledger has a stale dry run")
        else:
            write_json(estimate_path, frozen_estimate)
            write_jsonl(plan_path, call_plan)
        print(
            {
                "status": "DRY_RUN_COMPLETE_NOT_A_FORMAL_GATE",
                "cost_estimate_sha256": estimate["cost_estimate_sha256"],
                "n_logical_calls": len(call_plan),
                "maximum_physical_api_attempts": maximum_attempts,
                "maximum_estimated_usd": estimate["maximum_estimated_usd"],
                "budget_gate": budget_gate,
                "out_dir": str(args.out_dir.resolve()),
            }
        )
        if budget_gate["status"] != "PASS":
            raise RuntimeError("diagnostic budget gate failed")
        return

    if budget_gate["status"] != "PASS":
        raise RuntimeError("diagnostic budget gate failed")
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError("diagnostic run requires a saved matching dry run")
    if read_json(estimate_path) != frozen_estimate or list(iter_jsonl(plan_path)) != call_plan:
        raise RuntimeError("saved diagnostic dry run is stale")
    if args.accept_cost_estimate_sha256 != estimate["cost_estimate_sha256"]:
        raise RuntimeError("diagnostic --run requires the exact accepted cost identity")

    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=DIAGNOSTIC_RUN_STAGE,
        expected_calls={
            str(row["physical_call_key"]): attempts_per_call for row in call_plan
        },
        maximum_total_attempts=maximum_attempts,
    )
    blocked = {
        str(row["physical_call_key"]): blocker
        for row in call_plan
        if not ledger.succeeded(str(row["physical_call_key"]))
        and (
            blocker := call_retry_blocker(
                ledger, str(row["physical_call_key"])
            )
        )
        is not None
    }
    unavailable_retry_classes = (
        set(RETRYABLE_UP_TO_FULL_BUDGET)
        | set(BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES)
    )
    fatal_blocked: dict[str, str] = {}
    for call_key, blocker in blocked.items():
        terminal = ledger.terminal_row(call_key) or {}
        retry_class = str((terminal.get("metadata") or {}).get("retry_class") or "")
        if retry_class not in unavailable_retry_classes:
            fatal_blocked[call_key] = blocker
    if fatal_blocked:
        first = sorted(fatal_blocked)[0]
        raise RuntimeError(
            f"diagnostic matrix cannot resume: {first}: {fatal_blocked[first]}"
        )
    pending = [
        row
        for row in call_plan
        if not ledger.succeeded(str(row["physical_call_key"]))
        and str(row["physical_call_key"]) not in blocked
    ]
    clients: dict[str, Any] = {}
    if pending:
        # Credential access is deliberately delayed until every local contract,
        # source hash, saved dry run, budget, and identity check has passed.
        for endpoint_name in endpoint_names:
            endpoints[endpoint_name].api_key
        clients = {
            endpoint_name: make_client(endpoints[endpoint_name])
            for endpoint_name in endpoint_names
        }
    try:
        for row in pending:
            call_key = str(row["physical_call_key"])
            item = execution[call_key]

            def call_fn(item=item):
                return clients[item["endpoint_name"]].chat(
                    item["messages"],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    seed=seed,
                    response_schema=SingleFieldDiagnosticOutput,
                    retries=1,
                )

            reservation = None
            result = None
            parsed = None
            try:
                reservation, result, parsed = execute_with_bounded_retry(
                    ledger,
                    call_key,
                    record_ids=item["record_ids"],
                    prompt_sha256=str(row["prompt_sha256"]),
                    call_fn=call_fn,
                    max_provider_output_attempts=provider_output_attempts,
                    backoff_seconds=transport_backoff_seconds,
                    capture_provider_output_text=True,
                )
                if parsed is None:
                    raise RuntimeError("diagnostic returned no parsed object")
                assessment = assess_single_field_diagnostic_output(
                    item=item["item"],
                    output=parsed,
                )
                usage = require_reported_usage(
                    result.usage, stage="PM-v1.5 V4 root-cause diagnostic"
                )
                if usage["prompt_tokens"] > int(row["input_token_upper_bound"]):
                    raise RuntimeError("diagnostic prompt tokens exceed frozen bound")
                ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=result.request_hash,
                    usage=usage,
                    error=None,
                    result=_audited_result_payload(
                        result=result,
                        parsed=parsed,
                        assessment=assessment,
                    ),
                )
            except RetryableProviderError as exc:
                # Provider availability and provider-surface failures are
                # observations about infrastructure, not semantic judgments.
                # The ledger has already finalized the paid attempt. Continue
                # the frozen matrix and report missingness at the end.
                if exc.last_retry_class in unavailable_retry_classes:
                    continue
                raise
            except Exception as exc:
                if (
                    reservation is not None
                    and ledger.terminal_event(
                        reservation.call_key, reservation.attempt_index
                    )
                    is None
                ):
                    ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=(result.request_hash if result is not None else None),
                        usage=(result.usage if result is not None else None),
                        error=f"{type(exc).__name__}: {exc}",
                        result=_audited_result_payload(
                            result=result,
                            parsed=parsed,
                        ),
                        metadata=failure_metadata(
                            retry_class="diagnostic_evidence_postcondition_failure",
                            retry_disposition=TERMINAL_DISPOSITION,
                        ),
                    )
                raise
    finally:
        for client in clients.values():
            client.close()

    result_rows: list[dict[str, Any]] = []
    for row in call_plan:
        call_key = str(row["physical_call_key"])
        terminal = ledger.terminal_row(call_key)
        if terminal is None:
            raise RuntimeError("diagnostic call lacks a terminal ledger row")
        if not ledger.succeeded(call_key):
            continue
        result_payload = terminal.get("result") or {}
        parsed = SingleFieldDiagnosticOutput.model_validate(result_payload.get("parsed"))
        assessment = result_payload.get("assessment")
        if not isinstance(assessment, dict):
            raise RuntimeError("diagnostic result lacks validated assessment")
        result_rows.append(
            {
                "diagnostic_id": row["diagnostic_id"],
                "source_control_id": item_by_id[str(row["diagnostic_id"])][
                    "source_control_id"
                ],
                "field": row["field"],
                "polarity": row["polarity"],
                "endpoint_name": row["endpoint_name"],
                "judge_family": row["judge_family"],
                **parsed.model_dump(mode="json"),
                **assessment,
                "usage": terminal.get("usage"),
                "request_hash": terminal.get("request_hash"),
                "structured_output_audit": result_payload.get(
                    "structured_output_audit"
                ),
            }
        )

    report = aggregate_v4_single_field_diagnostic(
        deterministic_rows=source_validation["deterministic_rows"],
        result_rows=result_rows,
        endpoint_names=endpoint_names,
        require_complete=False,
    )
    unavailable_calls: list[dict[str, Any]] = []
    for row in call_plan:
        call_key = str(row["physical_call_key"])
        if ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key) or {}
        metadata = terminal.get("metadata") or {}
        unavailable_calls.append(
            {
                "diagnostic_id": row["diagnostic_id"],
                "field": row["field"],
                "polarity": row["polarity"],
                "endpoint_name": row["endpoint_name"],
                "judge_family": row["judge_family"],
                "attempts": ledger.attempts_for(call_key),
                "retry_class": metadata.get("retry_class"),
                "retry_disposition": metadata.get("retry_disposition"),
                "error": terminal.get("error"),
                "provider_output_text_sha256": (
                    (terminal.get("result") or {}).get(
                        "provider_output_text_sha256"
                    )
                ),
            }
        )
    terminal_rows = [
        row
        for row in ledger.event_rows
        if row.get("event") in {"SUCCEEDED", "FAILED"}
    ]
    reported_input = sum(
        int((row.get("usage") or {}).get("prompt_tokens") or 0)
        for row in terminal_rows
    )
    reported_output = sum(
        int((row.get("usage") or {}).get("completion_tokens") or 0)
        for row in terminal_rows
    )
    reported_candidate = sum(
        int(
            (row.get("usage") or {}).get(
                "gemini_candidate_tokens",
                (row.get("usage") or {}).get("completion_tokens") or 0,
            )
        )
        for row in terminal_rows
    )
    reported_thoughts = sum(
        int((row.get("usage") or {}).get("gemini_thought_tokens") or 0)
        for row in terminal_rows
    )
    reported_tool_use_prompt = sum(
        int(
            (row.get("usage") or {}).get("gemini_tool_use_prompt_tokens")
            or 0
        )
        for row in terminal_rows
    )
    reported_provider_total = sum(
        int((row.get("usage") or {}).get("total_tokens") or 0)
        for row in terminal_rows
    )
    if reported_provider_total != reported_input + reported_output:
        raise RuntimeError("diagnostic aggregate token accounting is inconsistent")
    report.update(
        {
            "cost_estimate_sha256": estimate["cost_estimate_sha256"],
            "judge_endpoint_descriptors": endpoint_descriptors,
            "judge_role_isolation": judge_role_isolation,
            "transport_retry_summary": retry_ledger_summary(
                ledger, [str(row["physical_call_key"]) for row in call_plan]
            ),
            "reported_input_tokens": reported_input,
            "reported_output_tokens": reported_output,
            "reported_visible_candidate_tokens": reported_candidate,
            "reported_thought_tokens": reported_thoughts,
            "reported_tool_use_prompt_tokens": reported_tool_use_prompt,
            "reported_total_tokens": reported_provider_total,
            "estimated_actual_cost_usd": (
                reported_input / 1_000_000 * prices["input"]
                + reported_output / 1_000_000 * prices["output"]
            ),
            "unavailable_calls": unavailable_calls,
            "n_initially_valid_json_surfaces": sum(
                bool((row.get("structured_output_audit") or {}).get(
                    "initially_valid_json"
                ))
                for row in result_rows
            ),
            "n_deterministically_normalized_json_surfaces": sum(
                (row.get("structured_output_audit") or {}).get(
                    "initially_valid_json"
                )
                is False
                for row in result_rows
            ),
        }
    )
    results_path = args.out_dir / "diagnostic_results.json"
    report_path = args.out_dir / "diagnostic_report.json"
    source_validation_path = args.out_dir / "source_validation.json"
    write_json(results_path, result_rows)
    write_json(report_path, report)
    write_json(source_validation_path, source_validation)
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage=DIAGNOSTIC_RUN_STAGE,
        inputs={
            "experiment_config": args.config,
            "pm_v1_5_config": args.pm_v1_5_config,
            "diagnostic_packet": args.diagnostic_packet,
            "observed_controls": args.observed_controls,
            "observed_control_judgments": args.observed_control_judgments,
            "observed_gate_report": args.observed_gate_report,
            "cost_estimate": estimate_path,
            "call_plan": plan_path,
        },
        outputs={
            "source_validation": (source_validation_path, False),
            "diagnostic_results": (results_path, False),
            "diagnostic_report": (report_path, False),
            "physical_attempt_ledger": (ledger_path, True),
        },
        parameters={
            "protocol": DIAGNOSTIC_RUN_PROTOCOL,
            "status": DIAGNOSTIC_RUN_STATUS,
            "formal_gate": False,
            "accepted_cost_estimate_sha256": estimate["cost_estimate_sha256"],
            "retry_contract": estimate["retry_contract"],
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        },
        expected={
            "logical_calls": len(call_plan),
            "physical_attempts": ledger.started_attempts,
            "formal_gate": False,
        },
    )
    print(report)


if __name__ == "__main__":
    main()
