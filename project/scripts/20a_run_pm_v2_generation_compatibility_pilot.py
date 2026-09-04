#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.api import (
    CallResult,
    ProviderRequestError,
    StructuredOutputValidationError,
    make_client,
    require_reported_usage,
)
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    read_json,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_contracts import ResourceNeedRegime
from metacom_pm.pm_v2_data import (
    GENERATION_CASE_FIELDS,
    GENERATION_TEMPERATURE,
    SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
    GeneratedSurfaceOnlyCaseDraft,
    compile_surface_only_user_bundle,
    generation_case_family_assignments,
    lint_generation_surface_case,
)
from metacom_pm.pm_v2_generation_pilot import (
    GENERATION_PILOT_MAX_ATTEMPTS,
    GENERATION_PILOT_MINIMUM_CALLS,
    GENERATION_PILOT_STAGE,
    build_generation_compatibility_contract,
    build_generation_compatibility_plan,
    read_generation_seed_dialogues_from_contract,
    require_generation_compatibility_attestation,
    validate_generation_pilot_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


class PilotInputTokenOverrun(RuntimeError):
    pass


def _ledger_usage(ledger: PersistentAttemptLedger) -> dict[str, int]:
    totals = {key: 0 for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for call_key in ledger.started_call_keys:
        terminal = ledger.terminal_row(call_key)
        if terminal is None or terminal.get("usage") is None:
            continue
        usage = require_reported_usage(
            terminal.get("usage"), stage="completed generation-pilot surface call"
        )
        for key in totals:
            totals[key] += usage[key]
    return totals


def _load_accepted_surface(
    ledger: PersistentAttemptLedger, row: dict
) -> tuple[GeneratedSurfaceOnlyCaseDraft, CallResult] | None:
    call_key = str(row["physical_call_key"])
    if not ledger.succeeded(call_key):
        return None
    terminal = ledger.terminal_row(call_key)
    result = (terminal or {}).get("result") or {}
    surface_payload = result.get("surface")
    response = result.get("provider_response")
    if not isinstance(surface_payload, dict) or not isinstance(response, dict):
        raise RuntimeError("successful surface ledger row lacks its provider trace")
    usage = require_reported_usage(
        (terminal or {}).get("usage"), stage="persisted accepted pilot surface"
    )
    request_hash = str((terminal or {}).get("request_hash") or "")
    if not request_hash:
        raise RuntimeError("accepted pilot surface lacks request hash")
    surface = GeneratedSurfaceOnlyCaseDraft.model_validate(surface_payload)
    return surface, CallResult(
        text=canonical_json(surface_payload),
        raw_response=response,
        usage=usage,
        latency_ms=0.0,
        request_hash=request_hash,
    )


def _materialize_success(
    *,
    bundle,
    report: dict,
    ledger: PersistentAttemptLedger,
    contract: dict,
    accepted_rows: list[dict],
    cost_estimate: dict,
    config_path: Path,
    pm_config_path: Path,
    seed_path: Path,
    manifest_path: Path,
    estimate_path: Path,
    plan_path: Path,
    ledger_path: Path,
    bundle_path: Path,
    summary_path: Path,
    attestation_path: Path,
) -> dict:
    repair_cases = sorted(
        row["case_field"]
        for row in accepted_rows
        if row["attempt_kind"] == "repair"
    )
    summary = {
        "status": "PASS",
        "compatibility_contract_sha256": contract["contract_sha256"],
        "accepted_surface_calls": accepted_rows,
        "accepted_case_count": len(accepted_rows),
        "repair_cases": repair_cases,
        "repair_case_count": len(repair_cases),
        "physical_attempts": ledger.started_attempts,
        "actual_usage": _ledger_usage(ledger),
        "bundle_validation": report,
    }
    write_json(bundle_path, bundle.model_dump(mode="json"))
    write_json(summary_path, summary)
    create_artifact_attestation(
        attestation_path,
        stage=GENERATION_PILOT_STAGE,
        inputs={
            "seed_dialogues": seed_path,
            "run_manifest": manifest_path,
            "cost_estimate": estimate_path,
            "call_plan": plan_path,
        },
        outputs={
            "pilot_bundle": (bundle_path, False),
            "physical_attempt_ledger": (ledger_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "compatibility_contract": contract,
            "compatibility_contract_sha256": contract["contract_sha256"],
            "accepted_cost_estimate_sha256": cost_estimate[
                "cost_estimate_sha256"
            ],
            "provider_trace_mode": "surface_only_casewise",
        },
        expected={
            "minimum_physical_attempts": GENERATION_PILOT_MINIMUM_CALLS,
            "maximum_physical_attempts": GENERATION_PILOT_MAX_ATTEMPTS,
            "actual_physical_attempts": ledger.started_attempts,
            "regimes": len(ResourceNeedRegime),
            "maximum_repairs_per_case": 1,
        },
    )
    return require_generation_compatibility_attestation(
        attestation_path, expected_contract=contract
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Casewise surface-only PM-v2 generation compatibility pilot with "
            "one pre-budgeted repair per case."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs/pm_v2.yaml")
    parser.add_argument("--seed-dialogues", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v2_generation_compatibility_pilot_surface_v8_2",
    )
    parser.add_argument("--input-usd-per-mtok", type=float)
    parser.add_argument("--output-usd-per-mtok", type=float)
    parser.add_argument("--max-api-calls", type=int)
    parser.add_argument("--max-estimated-usd", type=float)
    parser.add_argument("--max-input-tokens-per-call", type=int)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiment = load_config(args.config)
    pm_config = load_config(args.pm_v2_config)
    if pm_config.get("version") not in {"pm-v2.2", "pm-v1.5"}:
        raise ValueError("generation pilot requires PM-v2.2 or PM-v1.5")
    generation = dict(pm_config["data_generation"])
    pricing = dict(generation["pricing_usd_per_mtok"])
    if set(pricing) != {"input", "output"}:
        raise ValueError("data_generation pricing must contain input/output")
    frozen_input = float(pricing["input"])
    frozen_output = float(pricing["output"])
    if (frozen_input, frozen_output) != (0.15, 0.60):
        raise ValueError("PM-v2 generation pricing must be 0.15/0.60")
    if args.input_usd_per_mtok is not None and args.input_usd_per_mtok != frozen_input:
        raise RuntimeError("input pricing override differs from PM-v2 YAML")
    if args.output_usd_per_mtok is not None and args.output_usd_per_mtok != frozen_output:
        raise RuntimeError("output pricing override differs from PM-v2 YAML")
    args.input_usd_per_mtok = frozen_input
    args.output_usd_per_mtok = frozen_output
    api_cost = dict(pm_config["api_cost_planning"])
    safety_factor = float(api_cost["input_token_safety_factor"])
    fail_on_overrun = bool(api_cost["fail_on_reported_input_overrun"])
    if safety_factor < 1.0 or not fail_on_overrun:
        raise ValueError("generation pilot requires fail-safe costing")
    frozen_budget_raw = pm_config.get("generation_compatibility_pilot_budget")
    legacy_defaults = {
        "max_api_calls": GENERATION_PILOT_MAX_ATTEMPTS,
        "max_estimated_usd": 2.0,
        "max_input_tokens_per_call": 12000,
    }
    if frozen_budget_raw is not None:
        if not isinstance(frozen_budget_raw, dict):
            raise ValueError(
                "generation_compatibility_pilot_budget must be a mapping"
            )
        if set(frozen_budget_raw) != set(legacy_defaults):
            raise ValueError(
                "generation compatibility pilot budget must contain exactly "
                "max_api_calls/max_estimated_usd/max_input_tokens_per_call"
            )
        frozen_budget = {
            "max_api_calls": int(frozen_budget_raw["max_api_calls"]),
            "max_estimated_usd": float(frozen_budget_raw["max_estimated_usd"]),
            "max_input_tokens_per_call": int(
                frozen_budget_raw["max_input_tokens_per_call"]
            ),
        }
        for argument_name, frozen_value in frozen_budget.items():
            supplied_value = getattr(args, argument_name)
            if supplied_value is not None and supplied_value != frozen_value:
                flag = argument_name.replace("_", "-")
                raise RuntimeError(
                    f"--{flag} override differs from frozen generation-pilot budget"
                )
    else:
        # Preserve the version-neutral PM-v2 runner's historical defaults.
        # PM-v1.5 always supplies the frozen mapping above; legacy PM-v2 may
        # continue to make its budget limits explicit on the CLI.
        frozen_budget = {
            argument_name: (
                getattr(args, argument_name)
                if getattr(args, argument_name) is not None
                else default_value
            )
            for argument_name, default_value in legacy_defaults.items()
        }
    for argument_name, frozen_value in frozen_budget.items():
        setattr(args, argument_name, frozen_value)
    if args.max_api_calls != GENERATION_PILOT_MAX_ATTEMPTS:
        raise ValueError(
            "generation compatibility pilot requires max-api-calls="
            f"{GENERATION_PILOT_MAX_ATTEMPTS}"
        )
    if args.max_estimated_usd < 0 or args.max_input_tokens_per_call <= 0:
        raise ValueError("invalid compatibility-pilot budget")
    endpoint = endpoint_from_config(
        experiment, str(generation["generator_endpoint"])
    )
    full_user_count = sum(
        int(generation[key])
        for key in ("train_users", "calibration_users", "internal_test_users")
    )
    contract = build_generation_compatibility_contract(
        project_root=ROOT,
        experiment_config_path=args.config,
        pm_v2_config_path=args.pm_v2_config,
        seed_dialogues_path=args.seed_dialogues,
        endpoint=endpoint,
        base_generation_seed=int(generation["base_seed"]),
        full_user_count=full_user_count,
        input_token_safety_factor=safety_factor,
        fail_on_reported_input_overrun=fail_on_overrun,
        input_usd_per_mtok=frozen_input,
        output_usd_per_mtok=frozen_output,
    )
    _, call_plan, messages_by_call = build_generation_compatibility_plan(
        contract, endpoint=endpoint
    )
    initial_rows = [row for row in call_plan if row["attempt_kind"] == "initial"]
    expected_cost = sum(
        row["input_token_upper_bound"] / 1_000_000 * frozen_input
        + row["maximum_output_tokens"] / 1_000_000 * frozen_output
        for row in initial_rows
    )
    maximum_cost = sum(
        row["input_token_upper_bound"] / 1_000_000 * frozen_input
        + row["maximum_output_tokens"] / 1_000_000 * frozen_output
        for row in call_plan
    )
    max_input_bound = max(row["input_token_upper_bound"] for row in call_plan)
    cost_payload = {
        "stage": GENERATION_PILOT_STAGE,
        "compatibility_contract_sha256": contract["contract_sha256"],
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
        "minimum_api_calls_if_successful": len(initial_rows),
        "maximum_physical_api_attempts": len(call_plan),
        "maximum_repairs_per_case": 1,
        "stop_after_each_case_success": True,
        "input_token_safety_factor": safety_factor,
        "maximum_input_token_upper_bound_per_call": max_input_bound,
        "maximum_output_tokens_per_call": SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
        "pricing": {
            "input_usd_per_mtok": frozen_input,
            "output_usd_per_mtok": frozen_output,
        },
        "expected_estimated_cost_usd": expected_cost,
        "maximum_estimated_cost_usd": maximum_cost,
        "budget_limits": {
            "max_api_calls": args.max_api_calls,
            "max_estimated_usd": args.max_estimated_usd,
            "max_input_tokens_per_call": args.max_input_tokens_per_call,
        },
    }
    cost_estimate = {
        **cost_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(cost_payload)),
    }
    checks = {
        "physical_api_attempts": len(call_plan) <= args.max_api_calls,
        "estimated_cost_usd": maximum_cost <= args.max_estimated_usd,
        "input_tokens_per_call": max_input_bound <= args.max_input_tokens_per_call,
    }
    budget_gate = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": cost_payload["budget_limits"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = args.out_dir / "cost_estimate.json"
    plan_path = args.out_dir / "call_plan.jsonl"
    manifest_path = args.out_dir / "run_manifest.json"
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    bundle_path = args.out_dir / "pilot_bundle.json"
    summary_path = args.out_dir / "summary.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        ledger_path, overwrite=args.overwrite, stage="PM-v2 generation pilot"
    )
    if args.overwrite:
        for path in (
            estimate_path,
            plan_path,
            manifest_path,
            bundle_path,
            summary_path,
            attestation_path,
        ):
            if path.exists():
                path.unlink()
    ensure_run_manifest(
        manifest_path,
        {
            "stage": GENERATION_PILOT_STAGE,
            "compatibility_contract_sha256": contract["contract_sha256"],
            "physical_call_keys": [row["physical_call_key"] for row in call_plan],
            "minimum_physical_api_attempts": len(initial_rows),
            "maximum_physical_api_attempts": len(call_plan),
            "maximum_repairs_per_case": 1,
            "stop_after_each_case_success": True,
            "input_token_safety_factor": safety_factor,
            "fail_on_reported_input_overrun": fail_on_overrun,
        },
    )
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=GENERATION_PILOT_STAGE,
        expected_calls={row["physical_call_key"]: 1 for row in call_plan},
        maximum_total_attempts=len(call_plan),
    )

    exact_estimate = {**cost_estimate, "budget_gate": budget_gate}
    if args.dry_run:
        if ledger.started_attempts:
            if (
                not estimate_path.is_file()
                or not plan_path.is_file()
                or read_json(estimate_path) != exact_estimate
                or list(iter_jsonl(plan_path)) != call_plan
            ):
                raise RuntimeError("spent pilot ledger freezes its dry-run plan")
        else:
            write_json(estimate_path, exact_estimate)
            write_jsonl(plan_path, call_plan)
        print(
            {
                "status": "DRY_RUN_COMPLETE",
                "cost_estimate": cost_estimate,
                "budget_gate": budget_gate,
                "compatibility_contract": contract,
            }
        )
        if budget_gate["status"] != "PASS":
            raise RuntimeError("generation pilot budget gate failed")
        return

    if budget_gate["status"] != "PASS":
        raise RuntimeError("generation pilot budget gate failed")
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError("pilot API mode requires a matching saved dry-run")
    if read_json(estimate_path) != exact_estimate or list(iter_jsonl(plan_path)) != call_plan:
        raise RuntimeError("saved generation pilot dry-run is stale")
    if args.accept_cost_estimate_sha256 != cost_estimate["cost_estimate_sha256"]:
        raise RuntimeError("pilot API mode requires exact accepted cost hash")
    if all(path.is_file() for path in (bundle_path, summary_path, attestation_path)):
        print(
            require_generation_compatibility_attestation(
                attestation_path, expected_contract=contract
            )
        )
        return

    _ = endpoint.api_key
    seed_dialogue = read_generation_seed_dialogues_from_contract(contract)[
        int(contract["held_out_seed_index"])
    ]
    assignments = generation_case_family_assignments(
        [str(value) for value in contract["semantic_families"]],
        list(ResourceNeedRegime),
    )
    rows_by_case = {
        case_field: [row for row in call_plan if row["case_field"] == case_field]
        for case_field, _ in GENERATION_CASE_FIELDS
    }
    accepted_surfaces: dict[str, GeneratedSurfaceOnlyCaseDraft] = {}
    accepted_calls: dict[str, CallResult] = {}
    accepted_messages: dict[str, list[dict[str, str]]] = {}
    accepted_kinds: dict[str, str] = {}
    accepted_rows: list[dict] = []
    prior_currents: list[str] = []
    client = make_client(endpoint)
    try:
        for case_field, regime in GENERATION_CASE_FIELDS:
            case_rows = rows_by_case[case_field]
            accepted = None
            for row in case_rows:
                loaded = _load_accepted_surface(ledger, row)
                if loaded is not None:
                    accepted = (row, *loaded)
                    break
            if accepted is None:
                for row in case_rows:
                    call_key = str(row["physical_call_key"])
                    if call_key in ledger.started_call_keys:
                        continue
                    reservation = ledger.reserve(
                        call_key,
                        record_ids={
                            "user_id": contract["pilot_user_id"],
                            "case_field": case_field,
                            "attempt_kind": row["attempt_kind"],
                            "generation_seed": row["generation_seed"],
                        },
                        prompt_sha256=row["prompt_sha256"],
                    )
                    try:
                        call, surface = client.chat(
                            messages_by_call[call_key],
                            temperature=GENERATION_TEMPERATURE,
                            max_tokens=SURFACE_GENERATION_MAX_OUTPUT_TOKENS,
                            seed=int(row["generation_seed"]),
                            response_schema=GeneratedSurfaceOnlyCaseDraft,
                            retries=1,
                        )
                        assert surface is not None
                        usage = require_reported_usage(
                            call.usage, stage="generation pilot surface"
                        )
                        if usage["prompt_tokens"] > int(row["input_token_upper_bound"]):
                            result = {
                                "surface": surface.model_dump(mode="json"),
                                "provider_response": call.raw_response,
                                "attempt_kind": row["attempt_kind"],
                            }
                            error = (
                                "reported prompt_tokens exceed the frozen bound"
                            )
                            ledger.finish(
                                reservation,
                                succeeded=False,
                                request_hash=call.request_hash,
                                usage=usage,
                                error=error,
                                result=result,
                            )
                            raise PilotInputTokenOverrun(error)
                        lint = lint_generation_surface_case(
                            case_field=case_field,
                            regime=regime,
                            family=assignments[case_field],
                            forbidden_families=[
                                family
                                for family in contract["semantic_families"]
                                if family != assignments[case_field]
                            ],
                            surface=surface,
                            prior_current_user_texts=prior_currents,
                        )
                        result = {
                            "surface": surface.model_dump(mode="json"),
                            "provider_response": call.raw_response,
                            "lint": lint,
                            "attempt_kind": row["attempt_kind"],
                        }
                        if lint["status"] != "PASS":
                            ledger.finish(
                                reservation,
                                succeeded=False,
                                request_hash=call.request_hash,
                                usage=usage,
                                error="surface lint failed: " + canonical_json(lint["errors"]),
                                result=result,
                            )
                            write_json(
                                args.out_dir
                                / f"failed_{case_field}_{row['attempt_kind']}.json",
                                result,
                            )
                            continue
                        ledger.finish(
                            reservation,
                            succeeded=True,
                            request_hash=call.request_hash,
                            usage=usage,
                            error=None,
                            result=result,
                        )
                        accepted = (row, surface, call)
                        break
                    except StructuredOutputValidationError as exc:
                        usage = require_reported_usage(
                            exc.call.usage, stage="invalid generation pilot surface"
                        )
                        result = {
                            "provider_response": exc.call.raw_response,
                            "parsed_payload": exc.parsed_payload,
                            "validation_errors": exc.validation_errors,
                        }
                        ledger.finish(
                            reservation,
                            succeeded=False,
                            request_hash=exc.call.request_hash,
                            usage=usage,
                            error=f"{type(exc).__name__}: {exc}",
                            result=result,
                        )
                        write_json(
                            args.out_dir
                            / f"failed_{case_field}_{row['attempt_kind']}.json",
                            result,
                        )
                        continue
                    except PilotInputTokenOverrun:
                        raise
                    except ProviderRequestError as exc:
                        ledger.finish(
                            reservation,
                            succeeded=False,
                            request_hash=exc.request_hash,
                            usage=exc.usage,
                            error=f"{type(exc).__name__}: {exc}",
                            result={
                                "response_diagnostics": exc.response_diagnostics
                            },
                        )
                        raise
                    except Exception as exc:
                        ledger.finish(
                            reservation,
                            succeeded=False,
                            request_hash=None,
                            usage=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise
            if accepted is None:
                write_json(
                    summary_path,
                    {
                        "status": "FAIL",
                        "compatibility_contract_sha256": contract["contract_sha256"],
                        "failed_case": case_field,
                        "physical_attempts": ledger.started_attempts,
                        "actual_usage": _ledger_usage(ledger),
                    },
                )
                raise RuntimeError(
                    f"surface generation exhausted initial+repair for {case_field}"
                )
            row, surface, call = accepted
            accepted_surfaces[case_field] = surface
            accepted_calls[case_field] = call
            accepted_messages[case_field] = messages_by_call[
                str(row["physical_call_key"])
            ]
            accepted_kinds[case_field] = row["attempt_kind"]
            accepted_rows.append(row)
            prior_currents.append(surface.current_user_text)
    finally:
        client.close()

    bundle = compile_surface_only_user_bundle(
        surfaces=accepted_surfaces,
        accepted_calls=accepted_calls,
        accepted_messages=accepted_messages,
        accepted_attempt_kinds=accepted_kinds,
        seed_dialogue=seed_dialogue,
        user_id=str(contract["pilot_user_id"]),
        semantic_families=[str(value) for value in contract["semantic_families"]],
        regimes=list(ResourceNeedRegime),
        generator_model=endpoint.model,
        generator_family=endpoint.family,
    )
    bundle.provenance.update(
        {
            "generation_compatibility_contract_sha256": contract["contract_sha256"],
            "accepted_surface_call_keys": {
                row["case_field"]: row["physical_call_key"] for row in accepted_rows
            },
            "all_physical_attempt_usage": _ledger_usage(ledger),
        }
    )
    report = validate_generation_pilot_bundle(bundle, contract)
    if report["status"] != "PASS":
        raise RuntimeError(f"generation compatibility bundle failed: {report}")
    result = _materialize_success(
        bundle=bundle,
        report=report,
        ledger=ledger,
        contract=contract,
        accepted_rows=accepted_rows,
        cost_estimate=cost_estimate,
        config_path=args.config,
        pm_config_path=args.pm_v2_config,
        seed_path=args.seed_dialogues,
        manifest_path=manifest_path,
        estimate_path=estimate_path,
        plan_path=plan_path,
        ledger_path=ledger_path,
        bundle_path=bundle_path,
        summary_path=summary_path,
        attestation_path=attestation_path,
    )
    print(result)


if __name__ == "__main__":
    main()
