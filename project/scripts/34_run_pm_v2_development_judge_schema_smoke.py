#!/usr/bin/env python3
"""Run the four-call PM-v2 development-judge structured-schema smoke gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.api import require_reported_usage
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
)
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_judge_schema_smoke import (
    JUDGE_SCHEMA_SMOKE_PROTOCOL,
    JUDGE_SCHEMA_SMOKE_STAGE,
    build_judge_schema_smoke_contract,
    pending_judge_schema_smoke_calls,
    persist_or_validate_schema_smoke_dry_run,
    preflight_judge_schema_smoke_clients,
    require_development_judge_schema_smoke_pass,
)


ROOT = Path(__file__).resolve().parents[1]


def _budget_gate(
    estimate: dict,
    *,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict:
    checks = {
        "exact_four_physical_calls": int(estimate["expected_physical_calls"])
        == int(max_api_calls)
        == 4,
        "estimated_cost_usd": float(estimate["maximum_estimated_cost_usd"])
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": int(
            estimate["max_input_tokens_per_call"]
        )
        <= int(max_input_tokens_per_call),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }


def _successful_rows(
    ledger: PersistentAttemptLedger, call_plan: list[dict]
) -> list[dict]:
    rows = []
    for plan in call_plan:
        call_key = str(plan["physical_call_key"])
        if not ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key)
        result = (terminal or {}).get("result") or {}
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            raise RuntimeError("successful schema-smoke ledger lacks parsed result")
        schema = result.get("response_schema")
        if schema != plan["response_schema"]:
            raise RuntimeError("schema-smoke ledger schema provenance mismatch")
        usage = require_reported_usage(
            (terminal or {}).get("usage"),
            stage="persisted development judge schema smoke",
        )
        if usage["prompt_tokens"] > int(plan["input_token_upper_bound"]):
            raise RuntimeError("persisted schema-smoke prompt tokens exceed bound")
        request_hash = str((terminal or {}).get("request_hash") or "")
        if not request_hash:
            raise RuntimeError("successful schema-smoke ledger lacks request hash")
        rows.append(
            {
                "status": "SUCCESS",
                "physical_call_key": call_key,
                "state_id": plan["state_id"],
                "judge_family": plan["judge_family"],
                "judge_model": plan["judge_model"],
                "judge_type": plan["judge_type"],
                "response_schema": plan["response_schema"],
                "parsed": parsed,
                "usage": usage,
                "request_hash": request_hash,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Before the 180-call action pilot, test response/risk structured output "
            "on both development judge families (exactly four physical calls)."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml"
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_pilot" / "pilot_plan.json",
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_judge_schema_smoke",
    )
    parser.add_argument("--max-api-calls", type=int, default=4)
    parser.add_argument("--max-estimated-usd", type=float, default=2.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_api_calls != 4:
        raise ValueError("development judge schema smoke requires max-api-calls=4")
    if args.max_estimated_usd < 0 or args.max_input_tokens_per_call <= 0:
        raise ValueError("invalid development judge schema-smoke budget")
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=args.overwrite,
        stage="PM-v2 development judge schema smoke",
    )
    if args.run and args.overwrite:
        raise RuntimeError("paid schema-smoke run prohibits --overwrite")

    contract, call_plan, execution = build_judge_schema_smoke_contract(
        experiment_config_path=args.config,
        pm_v2_config_path=args.pm_v2_config,
        states_path=args.states,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
        pilot_plan_path=args.pilot_plan,
        semantic_sanity_report_path=args.semantic_sanity_report,
        semantic_sanity_attestation_path=args.semantic_sanity_attestation,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = args.out_dir / "cost_estimate.json"
    call_plan_path = args.out_dir / "call_plan.jsonl"
    raw_path = args.out_dir / "raw_results.jsonl"
    summary_path = args.out_dir / "summary.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    cost_payload = {
        "stage": JUDGE_SCHEMA_SMOKE_STAGE,
        "protocol": JUDGE_SCHEMA_SMOKE_PROTOCOL,
        "compatibility_contract_sha256": contract["contract_sha256"],
        "call_plan_sha256": contract["call_plan_sha256"],
        "expected_physical_calls": 4,
        "maximum_physical_attempts": 4,
        "total_input_token_upper_bound": sum(
            int(row["input_token_upper_bound"]) for row in call_plan
        ),
        "total_maximum_output_tokens": sum(
            int(row["maximum_output_tokens"]) for row in call_plan
        ),
        "max_input_tokens_per_call": max(
            int(row["input_token_upper_bound"]) for row in call_plan
        ),
        "maximum_estimated_cost_usd": sum(
            float(row["maximum_cost_usd"]) for row in call_plan
        ),
        "pricing_usd_per_mtok": contract["pricing_usd_per_mtok"],
        "api_cost_planning": contract["api_cost_planning"],
        "budget_limits": {
            "max_api_calls": int(args.max_api_calls),
            "max_estimated_usd": float(args.max_estimated_usd),
            "max_input_tokens_per_call": int(args.max_input_tokens_per_call),
        },
    }
    cost_estimate = {
        **cost_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(cost_payload)),
    }
    budget_gate = _budget_gate(
        cost_estimate,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    if args.dry_run or budget_gate["status"] != "PASS":
        dry_run_status = persist_or_validate_schema_smoke_dry_run(
            ledger_path=ledger_path,
            estimate_path=estimate_path,
            call_plan_path=call_plan_path,
            cost_estimate=cost_estimate,
            budget_gate=budget_gate,
            call_plan=call_plan,
        )
    else:
        dry_run_status = "NOT_REQUESTED"
    print(
        {
            **cost_estimate,
            "budget_gate": budget_gate,
            "dry_run_artifact_status": dry_run_status,
        }
    )
    if budget_gate["status"] != "PASS":
        raise RuntimeError("development judge schema-smoke budget gate failed")
    if args.dry_run:
        return

    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("schema-smoke API run requires a matching saved dry-run")
    saved_estimate = read_json(estimate_path)
    saved_plan = list(iter_jsonl(call_plan_path))
    if (
        saved_estimate.get("cost_estimate_sha256")
        != cost_estimate["cost_estimate_sha256"]
        or saved_estimate.get("budget_gate") != budget_gate
        or saved_plan != call_plan
    ):
        raise RuntimeError("saved schema-smoke dry-run is stale")
    if args.accept_cost_estimate_sha256 != cost_estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "schema-smoke run requires exact --accept-cost-estimate-sha256"
        )

    expected_calls = {
        str(row["physical_call_key"]): 1 for row in call_plan
    }
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=JUDGE_SCHEMA_SMOKE_STAGE,
        expected_calls=expected_calls,
        maximum_total_attempts=4,
    )
    pending, futility_reason = pending_judge_schema_smoke_calls(
        ledger, call_plan
    )
    clients = {}
    if pending:
        # Resolve *all* credentials and instantiate *all* clients before the
        # first durable STARTED reservation. A missing second-family key can
        # therefore never spend the first-family call.
        clients = preflight_judge_schema_smoke_clients(execution)
    try:
        for plan in pending:
            call_key = str(plan["physical_call_key"])
            item = execution[call_key]
            reservation = ledger.reserve(
                call_key,
                record_ids=item["record_ids"],
                prompt_sha256=str(plan["prompt_sha256"]),
            )
            call = None
            try:
                call, parsed = clients[str(plan["judge_family"])].chat(
                    item["messages"],
                    temperature=0.0,
                    max_tokens=int(plan["maximum_output_tokens"]),
                    seed=int(plan["seed"]),
                    response_schema=item["schema"],
                    retries=1,
                )
                if parsed is None:
                    raise RuntimeError("schema-smoke call returned no parsed object")
                usage = require_reported_usage(
                    call.usage, stage="development judge schema smoke"
                )
                if usage["prompt_tokens"] > int(plan["input_token_upper_bound"]):
                    raise RuntimeError(
                        "development judge schema-smoke prompt tokens exceed bound"
                    )
                ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=call.request_hash,
                    usage=usage,
                    error=None,
                    result={
                        "parsed": parsed.model_dump(mode="json"),
                        "response_schema": item["schema"].__name__,
                        "judge_model": plan["judge_model"],
                    },
                )
            except Exception as exc:
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=call.request_hash if call is not None else None,
                    usage=call.usage if call is not None else None,
                    error=f"{type(exc).__name__}: {exc}",
                )
                break
    finally:
        for client in clients.values():
            client.close()

    raw_rows = _successful_rows(ledger, call_plan)
    write_jsonl(raw_path, raw_rows)
    passed = len(raw_rows) == 4 and ledger.started_attempts == 4 and all(
        ledger.succeeded(str(row["physical_call_key"])) for row in call_plan
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "protocol": JUDGE_SCHEMA_SMOKE_PROTOCOL,
        "compatibility_contract_sha256": contract["contract_sha256"],
        "accepted_cost_estimate_sha256": cost_estimate["cost_estimate_sha256"],
        "expected_physical_calls": 4,
        "successful_physical_calls": len(raw_rows),
        "physical_attempts": ledger.started_attempts,
        "judge_families": sorted(
            str(row["family"]) for row in contract["judge_endpoints"]
        ),
        "selected_state_id": contract["selected_train_state"]["state_id"],
        "call_plan_sha256": contract["call_plan_sha256"],
        "raw_results_sha256": sha256_file(raw_path),
        "physical_attempt_ledger_sha256": sha256_file(ledger_path),
        **(
            {"futility_reason": futility_reason}
            if futility_reason is not None
            else {}
        ),
    }
    write_json(summary_path, summary)
    create_artifact_attestation(
        attestation_path,
        stage=JUDGE_SCHEMA_SMOKE_STAGE,
        inputs={
            "experiment_config": args.config,
            "pm_v2_config": args.pm_v2_config,
            "states": args.states,
            "backend": args.backend,
            "evaluator_contexts": args.evaluator_contexts,
            "pilot_plan": args.pilot_plan,
            "semantic_sanity_report": args.semantic_sanity_report,
            "semantic_sanity_attestation": args.semantic_sanity_attestation,
            "cost_estimate": estimate_path,
            "call_plan": call_plan_path,
        },
        outputs={
            "summary": (summary_path, False),
            "raw_results": (raw_path, True),
            "physical_attempt_ledger": (ledger_path, True),
        },
        parameters={
            "protocol": JUDGE_SCHEMA_SMOKE_PROTOCOL,
            "compatibility_contract": contract,
            "compatibility_contract_sha256": contract["contract_sha256"],
            "accepted_cost_estimate_sha256": cost_estimate[
                "cost_estimate_sha256"
            ],
            "gate_status": summary["status"],
        },
        expected={"physical_calls": 4, "successful_calls": len(raw_rows)},
    )
    print(summary)
    if not passed:
        raise RuntimeError(
            "development judge schema smoke failed; do not generate the action pilot"
        )
    verification = require_development_judge_schema_smoke_pass(
        summary_path=summary_path,
        attestation_path=attestation_path,
        experiment_config_path=args.config,
        pm_v2_config_path=args.pm_v2_config,
        states_path=args.states,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
        pilot_plan_path=args.pilot_plan,
        semantic_sanity_report_path=args.semantic_sanity_report,
        semantic_sanity_attestation_path=args.semantic_sanity_attestation,
    )
    print(verification)


if __name__ == "__main__":
    main()
