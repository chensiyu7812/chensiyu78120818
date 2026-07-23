#!/usr/bin/env python3
"""Run the train-only, balanced-order ESConv auxiliary pairwise pilot.

The preceding 13d script selects and freezes 24 train-dialogue states and a
96-call anonymous A/B plan without making API calls.  This runner independently
reconstructs every request, binds execution/retry code into a fresh paid-run
identity, and either:

* ``--dry-run``: writes the executable cost estimate and validated call plan;
* ``--run``: requires the exact approved identity, executes bounded calls, and
  writes a diagnostic-only aggregate plus an artifact attestation.

It never reads calibration/internal-test data and never creates training
labels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

from metacom_pm.api import (
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
    make_client,
    require_reported_usage,
)
from metacom_pm.artifacts import (
    create_artifact_attestation,
    require_artifact_attestation,
)
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
)
from metacom_pm.bounded_retry import (
    BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES,
    DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS,
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
from metacom_pm.v1_5_esconv_auxiliary_pairwise import (
    PAIRWISE_MAX_OUTPUT_TOKENS,
    PairwisePreferenceOutput,
    aggregate_pairwise_pilot,
    build_pairwise_messages,
    pairwise_contract_record,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE = "esconv_auxiliary_train_pairwise_measurement_pilot"
MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL = 10
TRANSPORT_BACKOFF_SECONDS: tuple[float, ...] = (
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    300.0,
    600.0,
    600.0,
    900.0,
)
CONSECUTIVE_SAME_CLASS_CIRCUIT_BREAKER = 5
TRANSPORT_PROTOCOL = "pm-v1.5-esconv-auxiliary-pairwise-pilot-transport-v2"
ISOLATABLE_FAILURE_CLASSES = frozenset(
    set(RETRYABLE_UP_TO_FULL_BUDGET)
    | set(BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES)
    | {"structured_output_validation_error", "output_token_limit"}
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(iter_jsonl(path))


def _isolatable_failure_class(exc: Exception) -> str | None:
    if isinstance(exc, RetryableProviderError):
        value = str(exc.last_retry_class)
        return value if value in ISOLATABLE_FAILURE_CLASSES else None
    if isinstance(exc, StructuredOutputValidationError):
        return "structured_output_validation_error"
    if isinstance(exc, ProviderRequestError):
        return None
    return None


def _persisted_isolatable_failure_class(
    ledger: PersistentAttemptLedger, call_key: str
) -> str | None:
    terminal = ledger.terminal_row(call_key) or {}
    metadata = terminal.get("metadata") or {}
    value = str(metadata.get("retry_class") or "")
    return value if value in ISOLATABLE_FAILURE_CLASSES else None


def _advance_failure_streak(
    *, previous_class: str | None, previous_count: int, retry_class: str
) -> tuple[str, int]:
    count = previous_count + 1 if previous_class == retry_class else 1
    if count >= CONSECUTIVE_SAME_CLASS_CIRCUIT_BREAKER:
        raise RuntimeError(
            f"pairwise pilot circuit breaker: {retry_class} recurred "
            f"{count} times consecutively"
        )
    return retry_class, count


def _record_family_failure(
    streaks: dict[str, tuple[str | None, int]],
    *,
    family: str,
    retry_class: str,
) -> None:
    previous_class, previous_count = streaks.get(family, (None, 0))
    streaks[family] = _advance_failure_streak(
        previous_class=previous_class,
        previous_count=previous_count,
        retry_class=retry_class,
    )


def _record_family_success(
    streaks: dict[str, tuple[str | None, int]], *, family: str
) -> None:
    streaks[family] = (None, 0)


def _transport_contract(
    *,
    provider_output_attempts_by_family: Mapping[str, int],
    code_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    values = {
        str(family): int(attempts)
        for family, attempts in sorted(provider_output_attempts_by_family.items())
    }
    if not values or any(
        attempts < 1 or attempts > MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        for attempts in values.values()
    ):
        raise ValueError("invalid pairwise provider-output retry budget")
    payload = {
        "protocol": TRANSPORT_PROTOCOL,
        "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
        "maximum_physical_attempts_per_logical_call": (
            MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "provider_output_maximum_attempts_by_family": values,
        "transport_backoff_seconds": list(TRANSPORT_BACKOFF_SECONDS),
        "retryable_transport_classes": sorted(RETRYABLE_UP_TO_FULL_BUDGET),
        "isolatable_failure_classes": sorted(ISOLATABLE_FAILURE_CLASSES),
        "per_family_consecutive_same_class_circuit_breaker": (
            CONSECUTIVE_SAME_CLASS_CIRCUIT_BREAKER
        ),
        "client_internal_retries": 1,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
    }
    return {**payload, "contract_sha256": sha256_text(canonical_json(payload))}


def _persist_or_require_exact(
    *,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise RuntimeError(f"existing dry-run artifact drifted: {path}")
        return
    write_json(path, dict(payload))


def _validate_prepared_plan(
    *,
    prepared_dir: Path,
    train_dir: Path,
    experiment: Mapping[str, Any],
    pm_config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[tuple[str, int], dict[str, Any]],
    list[Any],
    list[str],
]:
    plan = _load_rows(prepared_dir / "call_plan.jsonl")
    items = _load_rows(prepared_dir / "pairwise_items.jsonl")
    states = _load_rows(train_dir / "pm_v2_states.jsonl")
    prepared_contract = read_json(prepared_dir / "pairwise_contract.json")
    if prepared_contract != pairwise_contract_record():
        raise RuntimeError("prepared pairwise contract does not match current code")
    if len(plan) != 96 or len(items) != 48:
        raise RuntimeError("pairwise pilot must contain exactly 96 calls and 48 items")
    if len({str(row["physical_call_key"]) for row in plan}) != len(plan):
        raise RuntimeError("prepared pairwise call plan contains duplicate call keys")

    state_by_id = {str(row["state_id"]): row for row in states}
    if len(state_by_id) != len(states):
        raise RuntimeError("train states contain duplicate state_id")
    item_by_key = {
        (str(row["state_id"]), int(row["order_variant"])): row for row in items
    }
    if len(item_by_key) != len(items):
        raise RuntimeError("prepared pairwise items contain duplicate state/order keys")

    judging = dict(pm_config["development_judging"])
    endpoint_names = [str(value) for value in judging["judge_endpoints"]]
    endpoints = [endpoint_from_config(experiment, name) for name in endpoint_names]
    endpoint_record_by_family = {
        str(endpoint.family): (name, endpoint)
        for name, endpoint in zip(endpoint_names, endpoints)
    }
    if len(endpoint_record_by_family) != 2:
        raise RuntimeError("pairwise pilot requires exactly two judge families")
    schema_sha256 = sha256_text(
        canonical_json(PairwisePreferenceOutput.model_json_schema())
    )
    for row in plan:
        state_id = str(row["state_id"])
        order_variant = int(row["order_variant"])
        family = str(row["judge_family"])
        item = item_by_key.get((state_id, order_variant))
        state = state_by_id.get(state_id)
        endpoint_record = endpoint_record_by_family.get(family)
        endpoint = endpoint_record[1] if endpoint_record is not None else None
        if item is None or state is None or endpoint is None:
            raise RuntimeError("prepared pairwise row lacks a bound item/state/endpoint")
        messages = build_pairwise_messages(
            state=state,
            response_a=str(item["response_a"]),
            response_b=str(item["response_b"]),
        )
        prompt_sha256 = sha256_text(canonical_json(messages))
        if prompt_sha256 != str(row["prompt_sha256"]):
            raise RuntimeError("prepared pairwise prompt hash drifted")
        if str(row["response_schema_sha256"]) != schema_sha256:
            raise RuntimeError("prepared pairwise response schema hash drifted")
        request_payload_sha256 = sha256_text(
            canonical_json(
                {
                    "messages": messages,
                    "response_schema": PairwisePreferenceOutput.model_json_schema(),
                    "temperature": 0.0,
                    "max_tokens": PAIRWISE_MAX_OUTPUT_TOKENS,
                    "seed": int(row["seed"]),
                }
            )
        )
        if request_payload_sha256 != str(row["request_payload_sha256"]):
            raise RuntimeError("prepared pairwise request payload hash drifted")
        if (
            str(row["judge_model"]) != endpoint.model
            or str(row["judge_endpoint"]) != endpoint_record[0]
            or int(row["max_output_tokens"]) != PAIRWISE_MAX_OUTPUT_TOKENS
        ):
            raise RuntimeError("prepared pairwise endpoint/model/token contract drifted")
    return plan, state_by_id, item_by_key, endpoints, endpoint_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "data" / "esconv_auxiliary_v1_5" / "train",
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "outputs" / "esconv_auxiliary_generation_v1_5_full_train",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT / "outputs" / "esconv_auxiliary_pairwise_pilot_v1_5_train",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "esconv_auxiliary_pairwise_pilot_v1_5_execution",
    )
    parser.add_argument("--max-api-calls", type=int, default=2000)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=8000)
    parser.add_argument("--accept-cost-estimate-sha256")
    args = parser.parse_args()

    experiment = load_config(args.experiment_config)
    pm_config = load_config(args.pm_v1_5_config)
    require_paid_run_release(
        pm_config,
        config_path=args.pm_v1_5_config,
        stage=STAGE,
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path, overwrite=False, stage=STAGE
    )

    plan, state_by_id, item_by_key, endpoints, endpoint_names = (
        _validate_prepared_plan(
            prepared_dir=args.prepared_dir,
            train_dir=args.train_dir,
            experiment=experiment,
            pm_config=pm_config,
        )
    )
    require_artifact_attestation(
        args.generation_dir / "artifact_attestation.json",
        required_stage="action_sweep",
        required_output_paths={
            "action_outcomes": args.generation_dir / "action_outcomes.jsonl"
        },
    )
    judging = dict(pm_config["development_judging"])
    prices = {
        str(family): {
            "input": float(value["input"]),
            "output": float(value["output"]),
        }
        for family, value in dict(judging["pricing_usd_per_mtok"]).items()
    }
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    if set(prices) != set(endpoint_by_family):
        raise RuntimeError("pairwise pricing does not match judge families")
    overrides = {
        str(family): int(value)
        for family, value in dict(
            judging.get("maximum_provider_output_attempts_by_family") or {}
        ).items()
    }
    provider_output_attempts = {
        family: overrides.get(family, DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS)
        for family in sorted(endpoint_by_family)
    }
    code_paths = {
        "runner": Path(__file__).resolve(),
        "preparation_script": (
            ROOT
            / "scripts"
            / "v1_5"
            / "13d_prepare_esconv_auxiliary_pairwise_pilot_v1_5.py"
        ),
        "pairwise_contract": (
            ROOT / "src" / "metacom_pm" / "v1_5_esconv_auxiliary_pairwise.py"
        ),
        "api": ROOT / "src" / "metacom_pm" / "api.py",
        "attempt_ledger": ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
        "bounded_retry": ROOT / "src" / "metacom_pm" / "bounded_retry.py",
        "paid_run_release": ROOT / "src" / "metacom_pm" / "paid_run_release.py",
    }
    code_manifest = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(code_paths.items())
    }
    transport_contract = _transport_contract(
        provider_output_attempts_by_family=provider_output_attempts,
        code_manifest=code_manifest,
    )
    logical_cost = math.fsum(
        int(row["input_tokens_est"]) / 1_000_000
        * prices[str(row["judge_family"])]["input"]
        + PAIRWISE_MAX_OUTPUT_TOKENS
        / 1_000_000
        * prices[str(row["judge_family"])]["output"]
        for row in plan
    )
    cost_payload = {
        "stage": STAGE,
        "pairwise_contract": pairwise_contract_record(),
        "transport_contract": transport_contract,
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
        "prepared_pairwise_contract_sha256": sha256_file(
            args.prepared_dir / "pairwise_contract.json"
        ),
        "prepared_selection_sha256": sha256_file(
            args.prepared_dir / "selection.jsonl"
        ),
        "prepared_items_sha256": sha256_file(
            args.prepared_dir / "pairwise_items.jsonl"
        ),
        "prepared_call_plan_sha256": sha256_file(
            args.prepared_dir / "call_plan.jsonl"
        ),
        "train_states_sha256": sha256_file(args.train_dir / "pm_v2_states.jsonl"),
        "generation_outcomes_sha256": sha256_file(
            args.generation_dir / "action_outcomes.jsonl"
        ),
        "generation_attestation_sha256": sha256_file(
            args.generation_dir / "artifact_attestation.json"
        ),
        "judge_endpoints": [
            {
                "name": name,
                "family": endpoint.family,
                "model": endpoint.model,
                "base_url": endpoint.base_url,
            }
            for name, endpoint in zip(endpoint_names, endpoints)
        ],
        "pricing_usd_per_mtok": prices,
        "logical_calls": len(plan),
        "maximum_physical_attempts": (
            len(plan) * MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "logical_single_attempt_cost_usd": logical_cost,
        "maximum_cost_usd": (
            logical_cost * MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "maximum_input_tokens_per_call_est": max(
            int(row["input_tokens_est"]) for row in plan
        ),
        "call_plan_sha256": sha256_text(canonical_json(plan)),
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
    checks = {
        "api_calls": cost_estimate["maximum_physical_attempts"]
        <= int(args.max_api_calls),
        "estimated_cost_usd": cost_estimate["maximum_cost_usd"]
        <= float(args.max_estimated_usd),
        "max_input_tokens_per_call": cost_estimate[
            "maximum_input_tokens_per_call_est"
        ]
        <= int(args.max_input_tokens_per_call),
    }
    budget_gate = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    saved_estimate = {**cost_estimate, "budget_gate": budget_gate}
    _persist_or_require_exact(
        path=args.out_dir / "cost_estimate.json", payload=saved_estimate
    )
    plan_path = args.out_dir / "call_plan.jsonl"
    if plan_path.is_file():
        if _load_rows(plan_path) != plan:
            raise RuntimeError("saved executable pairwise call plan drifted")
    else:
        write_jsonl(plan_path, plan)
    print(saved_estimate)
    if args.dry_run:
        if budget_gate["status"] != "PASS":
            raise RuntimeError("pairwise pilot dry-run failed budget gate")
        return
    if budget_gate["status"] != "PASS":
        raise RuntimeError("pairwise pilot run blocked by budget gate")
    expected_identity = str(cost_estimate["cost_estimate_sha256"])
    if str(args.accept_cost_estimate_sha256 or "") != expected_identity:
        raise RuntimeError("accepted pairwise pilot identity does not match dry-run")

    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=STAGE,
        expected_calls={
            str(row["physical_call_key"]): (
                MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
            )
            for row in plan
        },
        maximum_total_attempts=int(args.max_api_calls),
    )
    clients = {
        family: make_client(endpoint)
        for family, endpoint in endpoint_by_family.items()
        if any(
            str(row["judge_family"]) == family
            and not ledger.succeeded(str(row["physical_call_key"]))
            for row in plan
        )
    }
    isolated_failures: dict[str, str] = {}
    failure_streaks_by_family: dict[str, tuple[str | None, int]] = {
        family: (None, 0) for family in endpoint_by_family
    }
    try:
        for row in plan:
            call_key = str(row["physical_call_key"])
            family = str(row["judge_family"])
            if ledger.succeeded(call_key):
                _record_family_success(
                    failure_streaks_by_family, family=family
                )
                continue
            blocker = call_retry_blocker(
                ledger,
                call_key,
                max_provider_output_attempts=provider_output_attempts[
                    str(row["judge_family"])
                ],
            )
            if blocker is not None:
                retry_class = _persisted_isolatable_failure_class(ledger, call_key)
                if retry_class is None:
                    raise RuntimeError(
                        f"pairwise pilot persisted non-isolatable failure: {blocker}"
                    )
                isolated_failures[call_key] = f"{retry_class}: {blocker}"
                _record_family_failure(
                    failure_streaks_by_family,
                    family=family,
                    retry_class=retry_class,
                )
                continue

            item = item_by_key[
                (str(row["state_id"]), int(row["order_variant"]))
            ]
            messages = build_pairwise_messages(
                state=state_by_id[str(row["state_id"])],
                response_a=str(item["response_a"]),
                response_b=str(item["response_b"]),
            )
            endpoint = endpoint_by_family[str(row["judge_family"])]

            def call_fn(
                *,
                endpoint=endpoint,
                messages=messages,
                row=row,
            ):
                return clients[str(endpoint.family)].chat(
                    messages,
                    temperature=0.0,
                    max_tokens=PAIRWISE_MAX_OUTPUT_TOKENS,
                    seed=int(row["seed"]),
                    response_schema=PairwisePreferenceOutput,
                    retries=1,
                )

            try:
                reservation, result, parsed = execute_with_bounded_retry(
                    ledger,
                    call_key,
                    record_ids={
                        "state_id": row["state_id"],
                        "pair_id": row["pair_id"],
                        "order_variant": row["order_variant"],
                        "judge_family": row["judge_family"],
                    },
                    prompt_sha256=str(row["prompt_sha256"]),
                    call_fn=call_fn,
                    max_provider_output_attempts=provider_output_attempts[
                        str(row["judge_family"])
                    ],
                    backoff_seconds=TRANSPORT_BACKOFF_SECONDS,
                )
            except Exception as exc:
                retry_class = _isolatable_failure_class(exc)
                if retry_class is None:
                    raise
                isolated_failures[call_key] = (
                    f"{retry_class}: {type(exc).__name__}: {exc}"
                )
                _record_family_failure(
                    failure_streaks_by_family,
                    family=family,
                    retry_class=retry_class,
                )
                continue
            try:
                usage = require_reported_usage(result.usage, stage=STAGE)
                if int(usage["prompt_tokens"]) > int(row["input_tokens_est"]):
                    raise RuntimeError(
                        "pairwise provider-reported prompt usage exceeds frozen bound"
                    )
                assert parsed is not None
            except Exception as exc:
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=result.request_hash,
                    usage=result.usage,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=failure_metadata(
                        retry_class="local_postcondition_failure",
                        retry_disposition=TERMINAL_DISPOSITION,
                    ),
                )
                raise
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash=result.request_hash,
                usage=usage,
                error=None,
                result={"parsed": parsed.model_dump(mode="json")},
            )
            _record_family_success(failure_streaks_by_family, family=family)
    finally:
        for client in clients.values():
            client.close()

    judgments: list[dict[str, Any]] = []
    actual_input_tokens = 0
    actual_output_tokens = 0
    actual_cost = 0.0
    for row in plan:
        call_key = str(row["physical_call_key"])
        if not ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key) or {}
        parsed = PairwisePreferenceOutput.model_validate(
            dict(terminal.get("result") or {}).get("parsed")
        )
        usage = require_reported_usage(terminal.get("usage"), stage=STAGE)
        family = str(row["judge_family"])
        actual_input_tokens += int(usage["prompt_tokens"])
        actual_output_tokens += int(usage["completion_tokens"])
        actual_cost += (
            int(usage["prompt_tokens"]) / 1_000_000 * prices[family]["input"]
            + int(usage["completion_tokens"]) / 1_000_000 * prices[family]["output"]
        )
        judgments.append(
            {
                "state_id": str(row["state_id"]),
                "dialogue_id": str(row["dialogue_id"]),
                "pair_id": str(row["pair_id"]),
                "order_variant": int(row["order_variant"]),
                "action_a": str(row["action_a"]),
                "action_b": str(row["action_b"]),
                "judge_endpoint": str(row["judge_endpoint"]),
                "judge_family": family,
                "request_hash": str(terminal.get("request_hash") or ""),
                **parsed.model_dump(mode="json"),
            }
        )
    judgments_path = args.out_dir / "judgments.jsonl"
    write_jsonl(judgments_path, judgments)
    complete = len(judgments) == len(plan)
    aggregate = (
        aggregate_pairwise_pilot(
            judgments, expected_families=sorted(endpoint_by_family)
        )
        if complete
        else None
    )
    summary = {
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "reportability": "PILOT_DIAGNOSTIC_ONLY",
        "training_labels_created": False,
        "logical_calls": len(plan),
        "completed_calls": len(judgments),
        "isolated_failures": isolated_failures,
        "retry_summary": retry_ledger_summary(
            ledger, [str(row["physical_call_key"]) for row in plan]
        ),
        "actual_usage": {
            "input_tokens": actual_input_tokens,
            "output_tokens": actual_output_tokens,
            "total_tokens": actual_input_tokens + actual_output_tokens,
            "estimated_cost_usd": actual_cost,
        },
        "aggregate": aggregate,
        "cost_estimate_sha256": expected_identity,
        "pairwise_contract_sha256": pairwise_contract_record()["contract_sha256"],
        "transport_contract_sha256": transport_contract["contract_sha256"],
    }
    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    if not complete:
        raise RuntimeError(
            f"pairwise pilot incomplete: {len(plan) - len(judgments)} calls missing"
        )
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage=STAGE,
        inputs={
            "pairwise_contract": args.prepared_dir / "pairwise_contract.json",
            "selection": args.prepared_dir / "selection.jsonl",
            "pairwise_items": args.prepared_dir / "pairwise_items.jsonl",
            "prepared_call_plan": args.prepared_dir / "call_plan.jsonl",
            "generation_attestation": (
                args.generation_dir / "artifact_attestation.json"
            ),
        },
        outputs={
            "summary": (summary_path, False),
            "judgments": (judgments_path, True),
            "physical_attempt_ledger": (ledger_path, True),
        },
        parameters={
            "cost_estimate_sha256": expected_identity,
            "pairwise_contract": pairwise_contract_record(),
            "transport_contract": transport_contract,
            "aggregate_status": str((aggregate or {}).get("status") or ""),
            "training_labels_created": False,
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
