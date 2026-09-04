#!/usr/bin/env python3
"""Execute the paid role-decomposed judge qualification after exact dry-run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from metacom_pm.api import (
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
    make_client,
    require_reported_usage,
)
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
)
from metacom_pm.bounded_retry import (
    RetryBlockedError,
    execute_with_bounded_retry,
    retry_ledger_summary,
)
from metacom_pm.config import load_config
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
from metacom_pm.v1_5_role_decomposed_judge_qualification import (
    COMPATIBILITY_PILOT_STAGE,
    RoleDecomposedEvidenceRiskOutput,
    RoleDecomposedQualityOutput,
    STAGE,
    build_call_plan,
    build_compatibility_pilot_plan,
    build_cost_estimate,
    code_manifest,
    endpoint_from_record,
    finish_paid_postcondition_failure,
    require_human_anchor,
    validate_audit_output_dimensions,
    validate_endpoint_contract,
    validate_schema_contract,
)


ROOT = Path(__file__).resolve().parents[2]
BACKOFF_SECONDS = (10.0,)


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _rebuild(
    *,
    prepared_dir: Path,
    tracked_contract_path: Path,
    human_anchor_binding_path: Path,
    endpoint_contract_path: Path,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    compatibility_pilot: bool,
) -> tuple[dict, dict, list[dict], dict]:
    contract = read_json(tracked_contract_path)
    if contract.get("status") != "READY_FOR_ZERO_API_DRY_RUN":
        raise RuntimeError("role-decomposed qualification is not execution ready")
    validate_schema_contract(contract)
    require_human_anchor(
        root=ROOT,
        binding_path=human_anchor_binding_path,
        qualification_contract=contract,
    )
    endpoint_contract = read_json(endpoint_contract_path)
    validate_endpoint_contract(endpoint_contract)
    expected_endpoint = {
        **endpoint_contract,
        "file_sha256": sha256_file(endpoint_contract_path),
        "content_sha256": sha256_text(canonical_json(endpoint_contract)),
    }
    if dict(contract["endpoint_contract"]) != expected_endpoint:
        raise RuntimeError("role-decomposed endpoint contract drifted")
    quality_path = prepared_dir / "quality_ordered_items_internal.jsonl"
    audit_path = prepared_dir / "evidence_risk_items_internal.jsonl"
    quality_rows = _rows(quality_path)
    audit_rows = _rows(audit_path)
    source = dict(contract["source_lineage"])
    if sha256_text(canonical_json(quality_rows)) != source[
        "quality_items_sha256"
    ]:
        raise RuntimeError("role-decomposed quality items drifted")
    if sha256_text(canonical_json(audit_rows)) != source[
        "audit_items_sha256"
    ]:
        raise RuntimeError("role-decomposed audit items drifted")
    plan = build_call_plan(
        quality_rows=quality_rows,
        audit_rows=audit_rows,
        endpoint_contract=endpoint_contract,
    )
    stage = STAGE
    if compatibility_pilot:
        plan = build_compatibility_pilot_plan(
            full_plan=plan,
            endpoint_contract=endpoint_contract,
        )
        stage = COMPATIBILITY_PILOT_STAGE
    estimate = build_cost_estimate(
        plan=plan,
        contract=contract,
        code_manifest=code_manifest(ROOT),
        endpoint_contract=endpoint_contract,
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
        stage=stage,
    )
    return contract, endpoint_contract, plan, estimate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--compatibility-pilot", action="store_true")
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/role_decomposed_judge_packet_v1",
    )
    parser.add_argument("--dry-run-dir", type=Path, required=True)
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_qualification_v1.json",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "role_decomposed_judge_human_anchor_v1.json",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT
        / "configs/pm_v1_5_role_decomposed_judge_qualification_v1.json",
    )
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-api-calls", type=int, default=600)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=16000)
    parser.add_argument("--accept-cost-estimate-sha256", required=True)
    args = parser.parse_args()

    contract, endpoint_contract, plan, estimate = _rebuild(
        prepared_dir=args.prepared_dir,
        tracked_contract_path=args.tracked_contract,
        human_anchor_binding_path=args.human_anchor_binding,
        endpoint_contract_path=args.endpoint_contract,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        compatibility_pilot=args.compatibility_pilot,
    )
    runtime_stage = str(estimate["stage"])
    if _rows(args.dry_run_dir / "call_plan.jsonl") != plan:
        raise RuntimeError("role-decomposed execution plan drifted from dry-run")
    if read_json(args.dry_run_dir / "cost_estimate.json") != estimate:
        raise RuntimeError("role-decomposed execution cost drifted from dry-run")
    identity = str(estimate["cost_estimate_sha256"])
    if str(args.accept_cost_estimate_sha256) != identity:
        raise RuntimeError("accepted role-decomposed identity does not match")
    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("role-decomposed qualification budget gate is not PASS")

    missing_env = sorted(
        {
            str(
                endpoint_contract["candidates"][candidate_key]["api_key_env"]
            )
            for candidate_key in {
                str(row["candidate_key"]) for row in plan
            }
            if not os.environ.get(
                str(
                    endpoint_contract["candidates"][candidate_key][
                        "api_key_env"
                    ]
                ),
                "",
            )
        }
    )
    if missing_env:
        raise RuntimeError(
            "role-decomposed qualification API keys are absent before "
            "authorization: " + ", ".join(missing_env)
        )
    config = load_config(args.pm_v1_5_config)
    require_paid_run_release(
        config,
        config_path=args.pm_v1_5_config,
        stage=runtime_stage,
        run=True,
        run_identity=identity,
    )
    require_output_directory_not_previously_consumed(
        args.out_dir,
        config=config,
        config_path=args.pm_v1_5_config,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    call_plan_path = args.out_dir / "call_plan.jsonl"
    cost_path = args.out_dir / "cost_estimate.json"
    if call_plan_path.is_file() and _rows(call_plan_path) != plan:
        raise RuntimeError("existing role-decomposed call plan drifted")
    if cost_path.is_file() and read_json(cost_path) != estimate:
        raise RuntimeError("existing role-decomposed cost estimate drifted")
    write_jsonl(call_plan_path, plan)
    write_json(cost_path, estimate)
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path, overwrite=False, stage=runtime_stage
    )
    maximum_attempts = int(
        endpoint_contract["request_contract"][
            "maximum_physical_attempts_per_logical_call"
        ]
    )
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=runtime_stage,
        expected_calls={
            str(row["physical_call_key"]): maximum_attempts for row in plan
        },
        maximum_total_attempts=int(args.max_api_calls),
    )
    endpoints = {
        key: endpoint_from_record(raw)
        for key, raw in endpoint_contract["candidates"].items()
    }
    clients: dict[str, Any] = {}
    isolated_failures: list[dict[str, str]] = []
    try:
        for key, endpoint in endpoints.items():
            if any(
                row["candidate_key"] == key
                and not ledger.succeeded(str(row["physical_call_key"]))
                for row in plan
            ):
                clients[key] = make_client(endpoint)
        for row in plan:
            call_key = str(row["physical_call_key"])
            if ledger.succeeded(call_key):
                continue
            candidate_key = str(row["candidate_key"])
            role = str(row["role"])
            response_schema = (
                RoleDecomposedQualityOutput
                if role == "quality"
                else RoleDecomposedEvidenceRiskOutput
            )

            def call_fn(
                row=row,
                candidate_key=candidate_key,
                response_schema=response_schema,
                role=role,
            ):
                result, parsed = clients[candidate_key].chat(
                    list(row["messages"]),
                    temperature=0.0,
                    max_tokens=int(row["max_output_tokens"]),
                    seed=int(row["request_parameters"]["seed"]),
                    response_schema=response_schema,
                    retries=1,
                )
                if role == "evidence_risk":
                    assert parsed is not None
                    validate_audit_output_dimensions(
                        parsed, list(row["applicable_dimensions"])
                    )
                return result, parsed

            try:
                reservation, result, parsed = execute_with_bounded_retry(
                    ledger,
                    call_key,
                    record_ids=dict(row["record_ids"]),
                    prompt_sha256=str(row["prompt_sha256"]),
                    call_fn=call_fn,
                    max_provider_output_attempts=maximum_attempts,
                    backoff_seconds=BACKOFF_SECONDS,
                )
                assert parsed is not None
            except (
                ProviderRequestError,
                RetryableProviderError,
                StructuredOutputValidationError,
                RetryBlockedError,
                ValueError,
            ) as exc:
                isolated_failures.append(
                    {
                        "physical_call_key": call_key,
                        "candidate_key": candidate_key,
                        "role": role,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            try:
                usage = require_reported_usage(
                    result.usage, stage=runtime_stage
                )
                if int(usage["prompt_tokens"]) > int(
                    row["input_tokens_est"]
                ):
                    raise RuntimeError(
                        "reported prompt usage exceeds frozen qualification "
                        f"bound: {usage['prompt_tokens']} > "
                        f"{row['input_tokens_est']}"
                    )
            except RuntimeError as exc:
                # A provider response has already happened and may be billed.
                # Always close its STARTED ledger row before isolating the
                # local planning/postcondition failure.  The parsed result is
                # retained for audit only and is never treated as succeeded.
                finish_paid_postcondition_failure(
                    ledger=ledger,
                    reservation=reservation,
                    result=result,
                    parsed=parsed,
                    error=exc,
                )
                isolated_failures.append(
                    {
                        "physical_call_key": call_key,
                        "candidate_key": candidate_key,
                        "role": role,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash=result.request_hash,
                usage=usage,
                error=None,
                result={"parsed": parsed.model_dump(mode="json")},
                metadata={
                    "provider_finish_reason": result.provider_finish_reason,
                    "normalized_finish_reason": result.normalized_finish_reason,
                },
            )
    finally:
        for client in clients.values():
            client.close()

    results: list[dict[str, Any]] = []
    for row in plan:
        call_key = str(row["physical_call_key"])
        if not ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key) or {}
        results.append(
            {
                "candidate_key": row["candidate_key"],
                "judge_family": row["judge_family"],
                "judge_model": row["judge_model"],
                "role": row["role"],
                "record_ids": row["record_ids"],
                "applicable_dimensions": row["applicable_dimensions"],
                "request_hash": terminal.get("request_hash"),
                "usage": terminal.get("usage"),
                "parsed": dict(terminal.get("result") or {}).get("parsed"),
            }
        )
    results_path = args.out_dir / "qualification_results.jsonl"
    write_jsonl(results_path, results)
    complete = len(results) == len(plan)
    summary = {
        "status": (
            "COMPLETE_ROLE_SPECIFIC_QUALIFICATION_RESULTS_NO_TRAINING_LABELS"
            if complete
            else "INCOMPLETE_NONREPORTABLE_MATRIX"
        ),
        "cost_estimate_sha256": identity,
        "logical_calls": len(plan),
        "completed_calls": len(results),
        "failed_calls": len(plan) - len(results),
        "isolated_failures": isolated_failures,
        "transport_retry_summary": retry_ledger_summary(
            ledger, [str(row["physical_call_key"]) for row in plan]
        ),
        "human_anchor_role": (
            "single_researcher_independent_reference_not_automatic_gold"
        ),
        "training_labels_created": False,
        "requires_separate_preoutcome_frozen_aggregation": complete,
    }
    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    attestation_payload = {
        "protocol": (
            "pm-v1.5-role-decomposed-judge-qualification-attestation-v1"
        ),
        "stage": runtime_stage,
        "status": summary["status"],
        "cost_estimate_sha256": identity,
        "qualification_contract_sha256": contract["contract_sha256"],
        "human_anchor_binding_sha256": contract[
            "human_anchor_binding_sha256"
        ],
        "call_plan_sha256": sha256_file(call_plan_path),
        "physical_attempt_ledger_sha256": sha256_file(ledger_path),
        "qualification_results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "training_labels_created": False,
    }
    write_json(
        args.out_dir / "artifact_attestation.json",
        {
            **attestation_payload,
            "artifact_attestation_sha256": sha256_text(
                canonical_json(attestation_payload)
            ),
        },
    )
    print(canonical_json(summary))
    if not complete:
        raise RuntimeError(
            "role-decomposed judge qualification matrix is incomplete"
        )


if __name__ == "__main__":
    main()
