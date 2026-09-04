#!/usr/bin/env python3
"""Execute the paid strict-schema judge bake-off after an exact dry-run."""

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
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.v1_5_strict_judge_bakeoff import (
    STRICT_BAKEOFF_STAGE,
    StrictBakeoffPairwiseOutput,
    build_call_plan,
    build_cost_estimate,
    endpoint_from_record,
    strict_bakeoff_carry_forward,
    validate_endpoint_contract,
)


ROOT = Path(__file__).resolve().parents[2]
BACKOFF_SECONDS = (10.0,)


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _code_manifest() -> dict[str, Any]:
    paths = {
        "execution": Path(__file__).resolve(),
        "dry_run": ROOT
        / "scripts/v1_5/21v_dry_run_strict_judge_bakeoff_v1_5.py",
        "preparer": ROOT
        / "scripts/v1_5/21u_prepare_strict_judge_bakeoff_v1_5.py",
        "bakeoff": ROOT
        / "src/metacom_pm/v1_5_strict_judge_bakeoff.py",
        "qualification_prompt": ROOT
        / "src/metacom_pm/v1_5_judge_qualification.py",
        "api": ROOT / "src/metacom_pm/api.py",
        "attempt_ledger": ROOT / "src/metacom_pm/attempt_ledger.py",
        "bounded_retry": ROOT / "src/metacom_pm/bounded_retry.py",
    }
    return {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(paths.items())
    }


def _rebuild(
    *,
    prepared_dir: Path,
    tracked_contract_path: Path,
    endpoint_contract_path: Path,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    carry_forward_from: Path | None = None,
) -> tuple[dict, dict, list[dict], dict, list[dict]]:
    prepared = read_json(prepared_dir / "qualification_contract.json")
    if read_json(tracked_contract_path) != prepared:
        raise RuntimeError("prepared and tracked strict bake-off contracts differ")
    endpoint_contract = read_json(endpoint_contract_path)
    validate_endpoint_contract(endpoint_contract)
    expected_endpoint = {
        **endpoint_contract,
        "file_sha256": sha256_file(endpoint_contract_path),
        "content_sha256": sha256_text(canonical_json(endpoint_contract)),
    }
    if dict(prepared["endpoint_contract"]) != expected_endpoint:
        raise RuntimeError("strict bake-off endpoint contract drifted")
    packet = dict(prepared["packet_files"])
    source_dir = ROOT / str(packet["source_packet_directory"])
    ordered_path = source_dir / "ordered_pairwise_prompts.jsonl"
    if sha256_file(ordered_path) != packet["ordered_pairwise_prompts"]["sha256"]:
        raise RuntimeError("strict bake-off ordered prompt artifact drifted")
    plan = build_call_plan(
        ordered_pairwise_rows=_rows(ordered_path),
        endpoint_contract=endpoint_contract,
    )
    carry_forward = None
    carry_forward_ledger_rows: list[dict] = []
    if carry_forward_from is not None:
        carry_forward, carry_forward_ledger_rows = (
            strict_bakeoff_carry_forward(
                source_dir=carry_forward_from,
                plan=plan,
                root=ROOT,
            )
        )
    request = dict(endpoint_contract["request_contract"])
    estimate = build_cost_estimate(
        plan=plan,
        contract=prepared,
        code_manifest=_code_manifest(),
        maximum_attempts=int(
            request["maximum_physical_attempts_per_logical_call"]
        ),
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
        carry_forward=carry_forward,
    )
    return (
        prepared,
        endpoint_contract,
        plan,
        estimate,
        carry_forward_ledger_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--dry-run-dir", type=Path, required=True)
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "strict_pairwise_judge_bakeoff_v1.json",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT / "configs/pm_v1_5_strict_judge_bakeoff_v1.json",
    )
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--carry-forward-from", type=Path)
    parser.add_argument("--max-api-calls", type=int, default=200)
    parser.add_argument("--max-estimated-usd", type=float, default=2.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument(
        "--accept-cost-estimate-sha256", required=True
    )
    args = parser.parse_args()

    (
        prepared,
        endpoint_contract,
        plan,
        estimate,
        carry_forward_ledger_rows,
    ) = _rebuild(
        prepared_dir=args.prepared_dir,
        tracked_contract_path=args.tracked_contract,
        endpoint_contract_path=args.endpoint_contract,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        carry_forward_from=args.carry_forward_from,
    )
    frozen_plan = _rows(args.dry_run_dir / "call_plan.jsonl")
    frozen_estimate = read_json(args.dry_run_dir / "cost_estimate.json")
    if frozen_plan != plan or frozen_estimate != estimate:
        raise RuntimeError("strict bake-off execution drifted from dry-run")
    identity = str(estimate["cost_estimate_sha256"])
    if str(args.accept_cost_estimate_sha256) != identity:
        raise RuntimeError("accepted strict bake-off identity does not match")
    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("strict bake-off budget gate is not PASS")

    missing_env = sorted(
        {
            str(raw["api_key_env"])
            for raw in endpoint_contract["candidates"].values()
            if not os.environ.get(str(raw["api_key_env"]), "")
        }
    )
    if missing_env:
        raise RuntimeError(
            "strict bake-off API keys are absent before authorization: "
            + ", ".join(missing_env)
        )
    require_paid_run_release(
        load_config(args.pm_v1_5_config),
        config_path=args.pm_v1_5_config,
        stage=STRICT_BAKEOFF_STAGE,
        run=True,
        run_identity=identity,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    call_plan_path = args.out_dir / "call_plan.jsonl"
    cost_path = args.out_dir / "cost_estimate.json"
    if call_plan_path.is_file() and _rows(call_plan_path) != plan:
        raise RuntimeError("existing execution call plan drifted")
    if cost_path.is_file() and read_json(cost_path) != estimate:
        raise RuntimeError("existing execution cost estimate drifted")
    write_jsonl(call_plan_path, plan)
    write_json(cost_path, estimate)
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path, overwrite=False, stage=STRICT_BAKEOFF_STAGE
    )
    if carry_forward_ledger_rows:
        write_jsonl(ledger_path, carry_forward_ledger_rows)
    maximum_attempts = int(
        endpoint_contract["request_contract"][
            "maximum_physical_attempts_per_logical_call"
        ]
    )
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=STRICT_BAKEOFF_STAGE,
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
            candidate = str(row["candidate_key"])

            def call_fn(row=row, candidate=candidate):
                return clients[candidate].chat(
                    list(row["messages"]),
                    temperature=0.0,
                    max_tokens=int(row["max_output_tokens"]),
                    seed=int(row["request_parameters"]["seed"]),
                    response_schema=StrictBakeoffPairwiseOutput,
                    retries=1,
                )

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
            except (
                ProviderRequestError,
                RetryableProviderError,
                StructuredOutputValidationError,
                RetryBlockedError,
            ) as exc:
                isolated_failures.append(
                    {
                        "physical_call_key": call_key,
                        "candidate_key": candidate,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            usage = require_reported_usage(
                result.usage, stage=STRICT_BAKEOFF_STAGE
            )
            if int(usage["prompt_tokens"]) > int(row["input_tokens_est"]):
                raise RuntimeError(
                    "reported prompt usage exceeds frozen bake-off bound"
                )
            assert parsed is not None
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

    results = []
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
                "record_ids": row["record_ids"],
                "request_hash": terminal.get("request_hash"),
                "usage": terminal.get("usage"),
                "parsed": dict(terminal.get("result") or {}).get("parsed"),
            }
        )
    write_jsonl(args.out_dir / "qualification_results.jsonl", results)
    complete = len(results) == len(plan)
    summary = {
        "status": (
            "COMPLETE_QUALIFICATION_RESULTS_NO_TRAINING_LABELS"
            if complete
            else "INCOMPLETE_NONREPORTABLE_MATRIX"
        ),
        "cost_estimate_sha256": identity,
        "logical_calls": len(plan),
        "carried_forward_logical_calls": estimate[
            "carried_forward_logical_calls"
        ],
        "new_logical_calls": estimate["new_logical_calls"],
        "completed_calls": len(results),
        "failed_calls": len(plan) - len(results),
        "isolated_failures": isolated_failures,
        "transport_retry_summary": retry_ledger_summary(
            ledger, [str(row["physical_call_key"]) for row in plan]
        ),
        "training_labels_created": False,
        "requires_separate_preoutcome_frozen_aggregation": complete,
    }
    write_json(args.out_dir / "summary.json", summary)
    attestation_payload = {
        "protocol": "pm-v1.5-strict-judge-bakeoff-attestation-v1",
        "stage": STRICT_BAKEOFF_STAGE,
        "status": summary["status"],
        "cost_estimate_sha256": identity,
        "qualification_contract_sha256": prepared["contract_sha256"],
        "carry_forward": estimate["carry_forward"],
        "call_plan_sha256": sha256_file(call_plan_path),
        "physical_attempt_ledger_sha256": sha256_file(ledger_path),
        "qualification_results_sha256": sha256_file(
            args.out_dir / "qualification_results.jsonl"
        ),
        "summary_sha256": sha256_file(args.out_dir / "summary.json"),
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
        raise RuntimeError("strict judge bake-off matrix is incomplete")


if __name__ == "__main__":
    main()
