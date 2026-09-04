#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.api import (
    ProviderRequestError,
    StructuredOutputValidationError,
    require_reported_usage,
)
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
    GenerationDraftCompilationError,
    GeneratedUserBundle,
    generate_user_bundle,
)
from metacom_pm.pm_v2_generation_pilot import (
    GENERATION_PILOT_STAGE,
    GENERATION_PILOT_MAX_ATTEMPTS,
    build_generation_compatibility_contract,
    build_generation_compatibility_plan,
    read_generation_seed_dialogues_from_contract,
    require_generation_compatibility_attestation,
    validate_generation_pilot_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _successful_bundle_from_ledger(
    *,
    ledger: PersistentAttemptLedger,
    physical_call_key: str,
    contract: dict,
) -> tuple[GeneratedUserBundle, dict]:
    """Recover the canonical paid response after a post-success crash."""

    terminal = ledger.terminal_row(physical_call_key)
    payload = ((terminal or {}).get("result") or {}).get("bundle")
    if not isinstance(payload, dict):
        raise RuntimeError(
            "successful generation pilot ledger lacks its canonical bundle"
        )
    bundle = GeneratedUserBundle.model_validate(payload)
    usage = require_reported_usage(
        (terminal or {}).get("usage"),
        stage="persisted generation compatibility pilot",
    )
    if bundle.provenance.get("reported_usage") != usage:
        raise RuntimeError(
            "generation pilot ledger usage differs from its canonical bundle"
        )
    if bundle.provenance.get("physical_call_key") != physical_call_key:
        raise RuntimeError("generation pilot recovery call key mismatch")
    request_hash = str((terminal or {}).get("request_hash") or "")
    if not request_hash or bundle.provenance.get("request_hash") != request_hash:
        raise RuntimeError("generation pilot recovery request hash mismatch")
    report = validate_generation_pilot_bundle(bundle, contract)
    if report["status"] != "PASS":
        raise RuntimeError(
            f"persisted generation compatibility bundle failed: {report}"
        )
    return bundle, report


def _materialize_success_artifacts(
    *,
    bundle: GeneratedUserBundle,
    bundle_report: dict,
    ledger: PersistentAttemptLedger,
    contract: dict,
    plan_row: dict,
    cost_estimate: dict,
    config_path: Path,
    pm_v2_config_path: Path,
    seed_dialogues_path: Path,
    manifest_path: Path,
    estimate_path: Path,
    call_plan_path: Path,
    ledger_path: Path,
    bundle_path: Path,
    summary_path: Path,
    attestation_path: Path,
) -> dict:
    """Materialize rebuildable artifacts from the immutable success ledger."""

    write_json(bundle_path, bundle.model_dump(mode="json"))
    write_json(
        summary_path,
        {
            "status": "PASS",
            "compatibility_contract_sha256": contract["contract_sha256"],
            "successful_call_plan": plan_row,
            "bundle_validation": bundle_report,
            "physical_attempts": ledger.started_attempts,
        },
    )
    create_artifact_attestation(
        attestation_path,
        stage=GENERATION_PILOT_STAGE,
        inputs={
            "experiment_config": config_path,
            "pm_v2_config": pm_v2_config_path,
            "seed_dialogues": seed_dialogues_path,
            "run_manifest": manifest_path,
            "cost_estimate": estimate_path,
            "call_plan": call_plan_path,
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
        },
        expected={
            "physical_attempts": ledger.started_attempts,
            "maximum_physical_attempts": GENERATION_PILOT_MAX_ATTEMPTS,
            "regimes": len(ResourceNeedRegime),
        },
    )
    return require_generation_compatibility_attestation(
        attestation_path, expected_contract=contract
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-attempt PM-v2 source-grounded generation compatibility pilot."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs/pm_v2.yaml"
    )
    parser.add_argument("--seed-dialogues", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v2_generation_compatibility_pilot_deterministic_evidence_v7"
        ),
    )
    parser.add_argument("--input-usd-per-mtok", type=float)
    parser.add_argument("--output-usd-per-mtok", type=float)
    parser.add_argument(
        "--max-api-calls", type=int, default=GENERATION_PILOT_MAX_ATTEMPTS
    )
    parser.add_argument("--max-estimated-usd", type=float, default=2.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        args.input_usd_per_mtok is not None
        and args.input_usd_per_mtok < 0
    ) or (
        args.output_usd_per_mtok is not None
        and args.output_usd_per_mtok < 0
    ):
        raise ValueError("pricing must be non-negative")
    if args.max_api_calls != GENERATION_PILOT_MAX_ATTEMPTS:
        raise ValueError(
            "generation compatibility pilot requires max-api-calls="
            f"{GENERATION_PILOT_MAX_ATTEMPTS}"
        )
    if args.max_estimated_usd < 0 or args.max_input_tokens_per_call <= 0:
        raise ValueError("invalid compatibility-pilot budget")

    experiment_config = load_config(args.config)
    pm_config = load_config(args.pm_v2_config)
    if pm_config.get("version") not in {"pm-v2.2", "pm-v1.5"}:
        raise ValueError(
            "generation compatibility pilot requires PM-v2.2 or PM-v1.5"
        )
    generation_cfg = dict(pm_config["data_generation"])
    pricing = dict(generation_cfg["pricing_usd_per_mtok"])
    if set(pricing) != {"input", "output"}:
        raise ValueError("data_generation pricing must contain input/output")
    frozen_input_price = float(pricing["input"])
    frozen_output_price = float(pricing["output"])
    if (frozen_input_price, frozen_output_price) != (0.15, 0.60):
        raise ValueError("PM-v2 gpt-4o-mini generation pricing must be 0.15/0.60")
    if (
        args.input_usd_per_mtok is not None
        and float(args.input_usd_per_mtok) != frozen_input_price
    ):
        raise RuntimeError("input pricing override differs from PM-v2 YAML")
    if (
        args.output_usd_per_mtok is not None
        and float(args.output_usd_per_mtok) != frozen_output_price
    ):
        raise RuntimeError("output pricing override differs from PM-v2 YAML")
    args.input_usd_per_mtok = frozen_input_price
    args.output_usd_per_mtok = frozen_output_price
    api_cost_cfg = dict(pm_config["api_cost_planning"])
    if set(api_cost_cfg) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("api_cost_planning keys do not match PM-v2")
    safety_factor = float(api_cost_cfg["input_token_safety_factor"])
    fail_on_overrun = bool(api_cost_cfg["fail_on_reported_input_overrun"])
    if safety_factor < 1.0 or not fail_on_overrun:
        raise ValueError("generation compatibility pilot requires fail-safe costing")
    endpoint = endpoint_from_config(
        experiment_config, str(generation_cfg["generator_endpoint"])
    )
    full_user_count = sum(
        int(generation_cfg[key])
        for key in ("train_users", "calibration_users", "internal_test_users")
    )
    contract = build_generation_compatibility_contract(
        project_root=ROOT,
        experiment_config_path=args.config,
        pm_v2_config_path=args.pm_v2_config,
        seed_dialogues_path=args.seed_dialogues,
        endpoint=endpoint,
        base_generation_seed=int(generation_cfg["base_seed"]),
        full_user_count=full_user_count,
        input_token_safety_factor=safety_factor,
        fail_on_reported_input_overrun=fail_on_overrun,
        input_usd_per_mtok=frozen_input_price,
        output_usd_per_mtok=frozen_output_price,
    )
    plan_row, call_plan, _ = build_generation_compatibility_plan(
        contract, endpoint=endpoint
    )
    maximum_estimated_cost = sum(
        int(row["input_token_upper_bound"])
        / 1_000_000
        * args.input_usd_per_mtok
        + int(row["maximum_output_tokens"])
        / 1_000_000
        * args.output_usd_per_mtok
        for row in call_plan
    )
    maximum_input_bound = max(
        int(row["input_token_upper_bound"]) for row in call_plan
    )
    maximum_raw_input = max(
        int(row["raw_estimated_input_tokens"]) for row in call_plan
    )
    cost_payload = {
        "stage": GENERATION_PILOT_STAGE,
        "compatibility_contract_sha256": contract["contract_sha256"],
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
        "minimum_api_calls_if_successful": 1,
        "maximum_physical_api_attempts": len(call_plan),
        "stop_after_first_success": True,
        "input_token_safety_factor": safety_factor,
        "raw_estimated_input_tokens": maximum_raw_input,
        "input_token_upper_bound": maximum_input_bound,
        "maximum_output_tokens": max(
            int(row["maximum_output_tokens"]) for row in call_plan
        ),
        "pricing": {
            "input_usd_per_mtok": args.input_usd_per_mtok,
            "output_usd_per_mtok": args.output_usd_per_mtok,
        },
        "maximum_estimated_cost_usd": maximum_estimated_cost,
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
        "estimated_cost_usd": maximum_estimated_cost <= args.max_estimated_usd,
        "input_tokens_per_call": maximum_input_bound
        <= args.max_input_tokens_per_call,
    }
    budget_gate = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": cost_payload["budget_limits"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = args.out_dir / "cost_estimate.json"
    call_plan_path = args.out_dir / "call_plan.jsonl"
    manifest_path = args.out_dir / "run_manifest.json"
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    bundle_path = args.out_dir / "pilot_bundle.json"
    summary_path = args.out_dir / "summary.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=args.overwrite,
        stage="PM-v2 generation compatibility pilot",
    )
    if args.overwrite:
        targets = (ledger_path, bundle_path, summary_path, attestation_path, manifest_path)
        if not args.run:
            targets = (*targets, estimate_path, call_plan_path)
        for path in targets:
            if path.exists():
                path.unlink()
    ensure_run_manifest(
        manifest_path,
        {
            "stage": GENERATION_PILOT_STAGE,
            "compatibility_contract_sha256": contract["contract_sha256"],
            "physical_call_keys": [
                str(row["physical_call_key"]) for row in call_plan
            ],
            "maximum_physical_api_attempts": len(call_plan),
            "stop_after_first_success": True,
            "input_token_safety_factor": safety_factor,
            "fail_on_reported_input_overrun": fail_on_overrun,
        },
    )
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=GENERATION_PILOT_STAGE,
        expected_calls={
            str(row["physical_call_key"]): 1 for row in call_plan
        },
        maximum_total_attempts=len(call_plan),
    )

    if args.dry_run:
        exact_estimate = {**cost_estimate, "budget_gate": budget_gate}
        if ledger.started_attempts:
            if (
                not estimate_path.is_file()
                or not call_plan_path.is_file()
                or read_json(estimate_path) != exact_estimate
                or list(iter_jsonl(call_plan_path)) != call_plan
            ):
                raise RuntimeError(
                    "spent generation-pilot ledger freezes the saved cost "
                    "estimate and call plan; current dry-run is stale"
                )
        else:
            write_json(estimate_path, exact_estimate)
            write_jsonl(call_plan_path, call_plan)
        result = {
            "status": "DRY_RUN_COMPLETE",
            "cost_estimate": cost_estimate,
            "budget_gate": budget_gate,
            "compatibility_contract": contract,
        }
        print(result)
        if budget_gate["status"] != "PASS":
            raise RuntimeError("generation compatibility-pilot budget gate failed")
        return

    if budget_gate["status"] != "PASS":
        raise RuntimeError("generation compatibility-pilot budget gate failed")
    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("pilot API mode requires a matching saved dry-run")
    saved_estimate = read_json(estimate_path)
    saved_plan = list(iter_jsonl(call_plan_path))
    if (
        saved_estimate.get("cost_estimate_sha256")
        != cost_estimate["cost_estimate_sha256"]
        or saved_estimate.get("budget_gate") != budget_gate
        or saved_plan != call_plan
    ):
        raise RuntimeError("saved generation compatibility-pilot dry-run is stale")
    if args.accept_cost_estimate_sha256 != cost_estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "pilot API mode requires exact --accept-cost-estimate-sha256"
        )

    successful_rows = [
        row
        for row in call_plan
        if ledger.succeeded(str(row["physical_call_key"]))
    ]
    if len(successful_rows) > 1:
        raise RuntimeError("generation pilot ledger contains multiple successes")
    if successful_rows:
        successful_row = successful_rows[0]
        if all(
            path.is_file()
            for path in (bundle_path, summary_path, attestation_path)
        ):
            result = require_generation_compatibility_attestation(
                attestation_path, expected_contract=contract
            )
        else:
            recovered_bundle, recovered_report = _successful_bundle_from_ledger(
                ledger=ledger,
                physical_call_key=str(successful_row["physical_call_key"]),
                contract=contract,
            )
            result = _materialize_success_artifacts(
                bundle=recovered_bundle,
                bundle_report=recovered_report,
                ledger=ledger,
                contract=contract,
                plan_row=successful_row,
                cost_estimate=cost_estimate,
                config_path=args.config,
                pm_v2_config_path=args.pm_v2_config,
                seed_dialogues_path=args.seed_dialogues,
                manifest_path=manifest_path,
                estimate_path=estimate_path,
                call_plan_path=call_plan_path,
                ledger_path=ledger_path,
                bundle_path=bundle_path,
                summary_path=summary_path,
                attestation_path=attestation_path,
            )
        print(result)
        return
    if ledger.started_attempts >= len(call_plan):
        raise RuntimeError(
            "generation compatibility pilot exhausted all pre-approved physical "
            "attempts without a valid bundle"
        )
    _ = endpoint.api_key
    seed_dialogue = read_generation_seed_dialogues_from_contract(contract)[
        int(contract["held_out_seed_index"])
    ]
    failures = [
        {
            "attempt_index": int(row["record_ids"]["attempt_index"]),
            "error": str(row.get("error") or "persisted failure"),
        }
        for row in ledger.failures()
    ]
    for current_plan in call_plan:
        call_key = str(current_plan["physical_call_key"])
        if call_key in ledger.started_call_keys:
            continue
        reservation = ledger.reserve(
            call_key,
            record_ids={
                "user_id": str(contract["pilot_user_id"]),
                "generation_seed": int(current_plan["generation_seed"]),
                "attempt_index": int(current_plan["attempt_index"]),
            },
            prompt_sha256=str(current_plan["prompt_sha256"]),
        )
        bundle = None
        try:
            bundle = generate_user_bundle(
                endpoint=endpoint,
                seed_dialogue=seed_dialogue,
                user_id=str(contract["pilot_user_id"]),
                semantic_families=[
                    str(value) for value in contract["semantic_families"]
                ],
                regimes=list(ResourceNeedRegime),
                seed=int(current_plan["generation_seed"]),
                request_retries=1,
            )
            bundle.provenance.update(
                {
                    "generation_compatibility_contract_sha256": contract[
                        "contract_sha256"
                    ],
                    "physical_call_key": reservation.call_key,
                    "physical_attempt_index": int(current_plan["attempt_index"]),
                    "physical_attempt_key": reservation.attempt_key,
                }
            )
            reported_usage = require_reported_usage(
                bundle.provenance.get("reported_usage"),
                stage="PM-v2 generation compatibility pilot",
            )
            bundle.provenance["reported_usage"] = reported_usage
            if reported_usage["prompt_tokens"] > int(
                current_plan["input_token_upper_bound"]
            ):
                raise RuntimeError(
                    "reported pilot prompt_tokens exceed the frozen conservative "
                    f"bound: reported={reported_usage['prompt_tokens']}, "
                    f"bound={current_plan['input_token_upper_bound']}"
                )
            bundle_report = validate_generation_pilot_bundle(bundle, contract)
            if bundle_report["status"] != "PASS":
                raise RuntimeError(
                    f"generation compatibility bundle failed: {bundle_report}"
                )
        except StructuredOutputValidationError as exc:
            error = f"{type(exc).__name__}: {exc}"
            usage = require_reported_usage(
                exc.call.usage,
                stage="failed PM-v2 generation semantic attempt",
            )
            failure_result = {
                "provider_response": exc.call.raw_response,
                "parsed_payload": exc.parsed_payload,
                "validation_errors": exc.validation_errors,
            }
            ledger.finish(
                reservation,
                succeeded=False,
                request_hash=exc.call.request_hash,
                usage=usage,
                error=error,
                result=failure_result,
            )
            write_json(
                args.out_dir
                / f"failed_attempt_{int(current_plan['attempt_index']):02d}.json",
                failure_result,
            )
            failures.append(
                {
                    "attempt_index": int(current_plan["attempt_index"]),
                    "error": error,
                    "provider_response_preserved": True,
                }
            )
            continue
        except GenerationDraftCompilationError as exc:
            error = f"{type(exc).__name__}: {exc}"
            usage = require_reported_usage(
                exc.call.usage,
                stage="failed PM-v2 generation draft compilation",
            )
            failure_result = {
                "provider_response": exc.call.raw_response,
                "parsed_payload": exc.parsed_payload,
                "compilation_error": exc.compilation_error,
            }
            ledger.finish(
                reservation,
                succeeded=False,
                request_hash=exc.call.request_hash,
                usage=usage,
                error=error,
                result=failure_result,
            )
            write_json(
                args.out_dir
                / f"failed_attempt_{int(current_plan['attempt_index']):02d}.json",
                failure_result,
            )
            failures.append(
                {
                    "attempt_index": int(current_plan["attempt_index"]),
                    "error": error,
                    "provider_response_preserved": True,
                }
            )
            continue
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            fatal = isinstance(exc, ProviderRequestError) or (
                "exceed the frozen conservative bound" in error
            )
            request_hash = None
            usage = None
            failure_result = None
            if bundle is not None:
                request_hash = (
                    str(bundle.provenance.get("request_hash") or "") or None
                )
                usage = dict(bundle.provenance.get("reported_usage") or {}) or None
                failure_result = {"bundle": bundle.model_dump(mode="json")}
            elif isinstance(exc, ProviderRequestError):
                request_hash = str(current_plan["request_payload_sha256"])
            ledger.finish(
                reservation,
                succeeded=False,
                request_hash=request_hash,
                usage=usage,
                error=error,
                result=failure_result,
            )
            failures.append(
                {
                    "attempt_index": int(current_plan["attempt_index"]),
                    "error": error,
                    "fatal": fatal,
                }
            )
            if fatal:
                write_json(
                    summary_path,
                    {
                        "status": "FAIL",
                        "compatibility_contract_sha256": contract[
                            "contract_sha256"
                        ],
                        "physical_attempts": ledger.started_attempts,
                        "failures": failures,
                        "stopped_early": True,
                    },
                )
                raise RuntimeError(error) from exc
            continue

        assert bundle is not None
        ledger.finish(
            reservation,
            succeeded=True,
            request_hash=str(bundle.provenance.get("request_hash") or "") or None,
            usage=dict(bundle.provenance.get("reported_usage") or {}),
            error=None,
            result={"bundle": bundle.model_dump(mode="json")},
        )
        canonical_bundle, canonical_report = _successful_bundle_from_ledger(
            ledger=ledger,
            physical_call_key=call_key,
            contract=contract,
        )
        result = _materialize_success_artifacts(
            bundle=canonical_bundle,
            bundle_report=canonical_report,
            ledger=ledger,
            contract=contract,
            plan_row=current_plan,
            cost_estimate=cost_estimate,
            config_path=args.config,
            pm_v2_config_path=args.pm_v2_config,
            seed_dialogues_path=args.seed_dialogues,
            manifest_path=manifest_path,
            estimate_path=estimate_path,
            call_plan_path=call_plan_path,
            ledger_path=ledger_path,
            bundle_path=bundle_path,
            summary_path=summary_path,
            attestation_path=attestation_path,
        )
        print(result)
        return

    write_json(
        summary_path,
        {
            "status": "FAIL",
            "compatibility_contract_sha256": contract["contract_sha256"],
            "physical_attempts": ledger.started_attempts,
            "failures": failures,
            "stopped_early": False,
        },
    )
    raise RuntimeError(
        "generation compatibility pilot exhausted all pre-approved physical "
        "attempts without a valid bundle"
    )


if __name__ == "__main__":
    main()
