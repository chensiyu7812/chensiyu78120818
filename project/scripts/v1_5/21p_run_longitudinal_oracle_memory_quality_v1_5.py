#!/usr/bin/env python3
"""Run the train-only blinded oracle-memory response-quality diagnostic.

The zero-API packet prepared by ``21l`` contains 18 frozen train states and
both A/B presentation orders for each state.  This runner reconstructs every
prompt, binds the selected judge endpoints, retry policy, schema, code, and
budget into one content-addressed identity, and then either:

* ``--dry-run`` writes the executable 72-call plan and retry-aware ceiling;
* ``--run`` requires that exact approved identity and executes the plan.

Results are report-only.  Families remain separate, a possible third-family
majority is descriptive only, and no row is emitted as a PM training label.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from metacom_pm.api import (
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
    make_client,
    require_reported_usage,
)
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
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
from metacom_pm.text import conservative_token_bound, estimate_tokens
from metacom_pm.v1_5_oracle_memory_quality import (
    ORACLE_MEMORY_QUALITY_MAX_OUTPUT_TOKENS,
    ORACLE_MEMORY_QUALITY_STATE_COUNT,
    OracleMemoryQualityOutput,
    aggregate_quality_diagnostic,
    build_quality_messages,
    quality_contract_record,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE = "longitudinal_oracle_memory_response_quality_diagnostic"
JUDGE_SEED = 8429
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
TRANSPORT_PROTOCOL = (
    "pm-v1.5-longitudinal-oracle-memory-quality-transport-v1"
)
ISOLATABLE_FAILURE_CLASSES = frozenset(
    set(RETRYABLE_UP_TO_FULL_BUDGET)
    | set(BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES)
    | {"structured_output_validation_error", "output_token_limit"}
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _persist_or_require_exact(
    *, path: Path, payload: Mapping[str, Any]
) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise RuntimeError(f"existing dry-run artifact drifted: {path}")
        return
    write_json(path, dict(payload))


def _persist_rows_or_require_exact(
    *, path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    normalized = [dict(row) for row in rows]
    if path.is_file():
        if _load_rows(path) != normalized:
            raise RuntimeError(f"existing dry-run row artifact drifted: {path}")
        return
    write_jsonl(path, normalized)


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
            f"oracle-memory quality circuit breaker: {retry_class} recurred "
            f"{count} times consecutively"
        )
    return retry_class, count


def _transport_contract(
    *,
    provider_output_attempts_by_family: Mapping[str, int],
    code_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    attempts = {
        str(family): int(value)
        for family, value in sorted(
            provider_output_attempts_by_family.items()
        )
    }
    if not attempts or any(
        value < 1
        or value > MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        for value in attempts.values()
    ):
        raise ValueError("invalid oracle-memory quality retry budget")
    payload = {
        "protocol": TRANSPORT_PROTOCOL,
        "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
        "maximum_physical_attempts_per_logical_call": (
            MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "provider_output_maximum_attempts_by_family": attempts,
        "transport_backoff_seconds": list(TRANSPORT_BACKOFF_SECONDS),
        "retryable_transport_classes": sorted(RETRYABLE_UP_TO_FULL_BUDGET),
        "isolatable_failure_classes": sorted(ISOLATABLE_FAILURE_CLASSES),
        "per_family_consecutive_same_class_circuit_breaker": (
            CONSECUTIVE_SAME_CLASS_CIRCUIT_BREAKER
        ),
        "client_internal_retries": 1,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
    }
    return {
        **payload,
        "contract_sha256": sha256_text(canonical_json(payload)),
    }


def _validate_packet(
    *,
    prepared_dir: Path,
    tracked_contract_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    items = _load_rows(prepared_dir / "quality_items.jsonl")
    prompt_rows = _load_rows(prepared_dir / "quality_prompt_rows.jsonl")
    prepared_contract = read_json(prepared_dir / "quality_contract.json")
    tracked_contract = read_json(tracked_contract_path)
    if prepared_contract != tracked_contract:
        raise RuntimeError(
            "prepared quality contract does not match tracked contract"
        )
    source_lineage = dict(prepared_contract.get("source_lineage") or {})
    if prepared_contract != quality_contract_record(
        source_lineage=source_lineage
    ):
        raise RuntimeError("prepared quality contract does not match code")
    expected_items = ORACLE_MEMORY_QUALITY_STATE_COUNT * 2
    if len(items) != expected_items or len(prompt_rows) != expected_items:
        raise RuntimeError(
            "oracle-memory quality packet must contain exactly 36 ordered items"
        )
    if (
        sha256_text(canonical_json(items))
        != str(source_lineage.get("items_sha256") or "")
    ):
        raise RuntimeError("oracle-memory quality item hash drifted")
    if (
        sha256_text(canonical_json(prompt_rows))
        != str(source_lineage.get("prompt_rows_sha256") or "")
    ):
        raise RuntimeError("oracle-memory quality prompt hash drifted")

    item_by_pair = {str(row["pair_id"]): row for row in items}
    prompt_by_pair = {str(row["pair_id"]): row for row in prompt_rows}
    if len(item_by_pair) != len(items) or len(prompt_by_pair) != len(
        prompt_rows
    ):
        raise RuntimeError("oracle-memory quality packet repeats pair_id")
    if set(item_by_pair) != set(prompt_by_pair):
        raise RuntimeError("quality item/prompt pair coverage drifted")
    states: dict[str, set[int]] = {}
    for pair_id, item in item_by_pair.items():
        prompt = prompt_by_pair[pair_id]
        state_id = str(item["state_id"])
        order = int(item["order_variant"])
        if (
            str(prompt["state_id"]) != state_id
            or int(prompt["order_variant"]) != order
            or list(prompt["messages"]) != build_quality_messages(item)
        ):
            raise RuntimeError(
                f"oracle-memory quality prompt reconstruction drifted: {pair_id}"
            )
        states.setdefault(state_id, set()).add(order)
    if len(states) != ORACLE_MEMORY_QUALITY_STATE_COUNT or any(
        orders != {0, 1} for orders in states.values()
    ):
        raise RuntimeError("oracle-memory quality AB/BA coverage drifted")
    return items, prompt_rows, prepared_contract


def _endpoint_records(
    *,
    experiment: Mapping[str, Any],
    pm_config: Mapping[str, Any],
    judge_endpoint_names: Sequence[str] | None,
) -> tuple[list[str], list[Any], dict[str, dict[str, float]]]:
    judging = dict(pm_config["development_judging"])
    names = [
        str(value)
        for value in (
            judge_endpoint_names
            if judge_endpoint_names
            else judging["judge_endpoints"]
        )
    ]
    endpoints = [endpoint_from_config(experiment, name) for name in names]
    families = [str(endpoint.family or "") for endpoint in endpoints]
    if len(families) not in {2, 3} or len(set(families)) != len(families):
        raise RuntimeError(
            "oracle-memory quality diagnostic requires two or three "
            "distinct judge families"
        )
    prices = {
        str(family): {
            "input": float(value["input"]),
            "output": float(value["output"]),
        }
        for family, value in dict(
            judging["pricing_usd_per_mtok"]
        ).items()
    }
    missing = sorted(set(families) - set(prices))
    if missing:
        raise RuntimeError(
            "oracle-memory quality pricing is not frozen for families: "
            + ", ".join(missing)
        )
    return names, endpoints, {family: prices[family] for family in families}


def build_execution_plan(
    *,
    items: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
    endpoint_names: Sequence[str],
    endpoints: Sequence[Any],
    prices: Mapping[str, Mapping[str, float]],
    input_token_safety_factor: float,
) -> list[dict[str, Any]]:
    """Build the deterministic executable matrix without creating clients."""

    if len(endpoint_names) != len(endpoints):
        raise ValueError("endpoint name/record count mismatch")
    item_by_pair = {str(row["pair_id"]): dict(row) for row in items}
    prompt_by_pair = {
        str(row["pair_id"]): dict(row) for row in prompt_rows
    }
    state_index = {
        state_id: index
        for index, state_id in enumerate(
            sorted({str(row["state_id"]) for row in items})
        )
    }
    schema = OracleMemoryQualityOutput.model_json_schema()
    schema_sha256 = sha256_text(canonical_json(schema))
    plan: list[dict[str, Any]] = []
    for pair_id in sorted(item_by_pair):
        item = item_by_pair[pair_id]
        prompt = prompt_by_pair[pair_id]
        messages = list(prompt["messages"])
        messages_json = canonical_json(messages)
        prompt_sha256 = sha256_text(messages_json)
        base_input_tokens = estimate_tokens(messages_json)
        input_tokens = conservative_token_bound(
            messages_json, safety_factor=input_token_safety_factor
        )
        for endpoint_index, (endpoint_name, endpoint) in enumerate(
            zip(endpoint_names, endpoints)
        ):
            family = str(endpoint.family)
            # One state/family gets the same seed for both display orders.
            seed = (
                JUDGE_SEED
                + state_index[str(item["state_id"])] * 10
                + endpoint_index
            )
            request_parameters = {
                "temperature": 0.0,
                "max_tokens": ORACLE_MEMORY_QUALITY_MAX_OUTPUT_TOKENS,
                "seed": seed,
                "response_schema_sha256": schema_sha256,
            }
            request_payload_sha256 = sha256_text(
                canonical_json(
                    {
                        "messages": messages,
                        "response_schema": schema,
                        "temperature": 0.0,
                        "max_tokens": request_parameters["max_tokens"],
                        "seed": seed,
                    }
                )
            )
            price = prices[family]
            row = {
                "pair_id": pair_id,
                "state_id": str(item["state_id"]),
                "user_id": str(item["user_id"]),
                "regime": str(item["regime"]),
                "target_item_utility": str(item["target_item_utility"]),
                "order_variant": int(item["order_variant"]),
                "arm_a": str(item["arm_a"]),
                "arm_b": str(item["arm_b"]),
                "expected_winner": str(item["expected_winner"]),
                "judge_endpoint": str(endpoint_name),
                "judge_family": family,
                "judge_model": str(endpoint.model),
                "seed": seed,
                "prompt_sha256": prompt_sha256,
                "response_schema_sha256": schema_sha256,
                "request_payload_sha256": request_payload_sha256,
                "base_input_tokens_est": base_input_tokens,
                "input_tokens_est": input_tokens,
                "max_output_tokens": request_parameters["max_tokens"],
                "maximum_single_attempt_cost_usd": (
                    input_tokens
                    / 1_000_000
                    * float(price["input"])
                    + request_parameters["max_tokens"]
                    / 1_000_000
                    * float(price["output"])
                ),
            }
            row["physical_call_key"] = physical_call_key(
                stage=STAGE,
                record_ids={
                    "state_id": row["state_id"],
                    "order_variant": row["order_variant"],
                    "judge_family": family,
                },
                prompt_sha256=prompt_sha256,
                endpoint=endpoint,
                request_parameters=request_parameters,
            )
            plan.append(row)
    if len({str(row["physical_call_key"]) for row in plan}) != len(plan):
        raise RuntimeError("oracle-memory quality call plan repeats a key")
    return sorted(
        plan,
        key=lambda row: (
            str(row["state_id"]),
            int(row["order_variant"]),
            str(row["judge_family"]),
        ),
    )


def build_cost_estimate(
    *,
    plan: Sequence[Mapping[str, Any]],
    quality_contract: Mapping[str, Any],
    transport_contract: Mapping[str, Any],
    code_manifest: Mapping[str, Any],
    experiment_config_path: Path,
    pm_config_path: Path,
    tracked_contract_path: Path,
    prepared_dir: Path,
    endpoint_names: Sequence[str],
    endpoints: Sequence[Any],
    prices: Mapping[str, Mapping[str, float]],
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict[str, Any]:
    logical_cost = math.fsum(
        float(row["maximum_single_attempt_cost_usd"]) for row in plan
    )
    payload = {
        "protocol": (
            "pm-v1.5-longitudinal-oracle-memory-quality-cost-v1"
        ),
        "stage": STAGE,
        "quality_contract": dict(quality_contract),
        "quality_contract_sha256": str(
            quality_contract["contract_sha256"]
        ),
        "transport_contract": dict(transport_contract),
        "code_manifest": dict(code_manifest),
        "code_manifest_sha256": sha256_text(
            canonical_json(code_manifest)
        ),
        "experiment_config_sha256": sha256_file(
            experiment_config_path
        ),
        "pm_v1_5_config_sha256": sha256_file(pm_config_path),
        "tracked_contract_file_sha256": sha256_file(
            tracked_contract_path
        ),
        "prepared_quality_contract_sha256": sha256_file(
            prepared_dir / "quality_contract.json"
        ),
        "prepared_quality_items_sha256": sha256_file(
            prepared_dir / "quality_items.jsonl"
        ),
        "prepared_quality_prompt_rows_sha256": sha256_file(
            prepared_dir / "quality_prompt_rows.jsonl"
        ),
        "judge_endpoints": [
            {
                "name": name,
                "family": endpoint.family,
                "model": endpoint.model,
                "base_url": endpoint.base_url,
                "transport": endpoint.transport,
            }
            for name, endpoint in zip(endpoint_names, endpoints)
        ],
        "pricing_usd_per_mtok": {
            family: dict(value)
            for family, value in sorted(prices.items())
        },
        "logical_calls": len(plan),
        "maximum_physical_attempts": (
            len(plan) * MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "logical_single_attempt_cost_usd": logical_cost,
        "maximum_cost_usd": (
            logical_cost * MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "maximum_input_tokens_per_call_est": max(
            [int(row["input_tokens_est"]) for row in plan], default=0
        ),
        "call_plan_sha256": sha256_text(canonical_json(list(plan))),
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(
                max_input_tokens_per_call
            ),
        },
        "api_judges_used_during_dry_run": False,
        "training_labels_created": False,
    }
    estimate = {
        **payload,
        "cost_estimate_sha256": sha256_text(canonical_json(payload)),
    }
    checks = {
        "api_calls": estimate["maximum_physical_attempts"]
        <= int(max_api_calls),
        "estimated_cost_usd": estimate["maximum_cost_usd"]
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": estimate[
            "maximum_input_tokens_per_call_est"
        ]
        <= int(max_input_tokens_per_call),
    }
    return {
        **estimate,
        "budget_gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
    }


def _parse_args() -> argparse.Namespace:
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
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/"
        "longitudinal_oracle_memory_quality_diagnostic_v1.json",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=ROOT
        / "outputs/"
        "pm_v1_5_longitudinal_oracle_memory_quality_packet_v1",
    )
    parser.add_argument("--judge-endpoints", nargs="+")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-api-calls", type=int, default=2000)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument(
        "--max-input-tokens-per-call", type=int, default=8000
    )
    parser.add_argument("--accept-cost-estimate-sha256")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment = load_config(args.experiment_config)
    pm_config = load_config(args.pm_v1_5_config)
    items, prompt_rows, quality_contract = _validate_packet(
        prepared_dir=args.prepared_dir,
        tracked_contract_path=args.tracked_contract,
    )
    endpoint_names, endpoints, prices = _endpoint_records(
        experiment=experiment,
        pm_config=pm_config,
        judge_endpoint_names=args.judge_endpoints,
    )
    judging = dict(pm_config["development_judging"])
    overrides = {
        str(family): int(value)
        for family, value in dict(
            judging.get("maximum_provider_output_attempts_by_family")
            or {}
        ).items()
    }
    provider_output_attempts = {
        str(endpoint.family): overrides.get(
            str(endpoint.family),
            DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS,
        )
        for endpoint in endpoints
    }
    code_paths = {
        "runner": Path(__file__).resolve(),
        "preparation_script": (
            ROOT
            / "scripts/v1_5/"
            "21l_prepare_longitudinal_oracle_memory_quality_v1_5.py"
        ),
        "quality_contract_code": (
            ROOT
            / "src/metacom_pm/v1_5_oracle_memory_quality.py"
        ),
        "api": ROOT / "src/metacom_pm/api.py",
        "attempt_ledger": ROOT / "src/metacom_pm/attempt_ledger.py",
        "bounded_retry": ROOT / "src/metacom_pm/bounded_retry.py",
        "paid_run_release": (
            ROOT / "src/metacom_pm/paid_run_release.py"
        ),
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
    plan = build_execution_plan(
        items=items,
        prompt_rows=prompt_rows,
        endpoint_names=endpoint_names,
        endpoints=endpoints,
        prices=prices,
        input_token_safety_factor=float(
            pm_config["api_cost_planning"]["input_token_safety_factor"]
        ),
    )
    cost_estimate = build_cost_estimate(
        plan=plan,
        quality_contract=quality_contract,
        transport_contract=transport_contract,
        code_manifest=code_manifest,
        experiment_config_path=args.experiment_config,
        pm_config_path=args.pm_v1_5_config,
        tracked_contract_path=args.tracked_contract,
        prepared_dir=args.prepared_dir,
        endpoint_names=endpoint_names,
        endpoints=endpoints,
        prices=prices,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cost_path = args.out_dir / "cost_estimate.json"
    plan_path = args.out_dir / "call_plan.jsonl"
    _persist_or_require_exact(path=cost_path, payload=cost_estimate)
    _persist_rows_or_require_exact(path=plan_path, rows=plan)
    if cost_estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("oracle-memory quality budget gate failed")
    if args.dry_run:
        summary = {
            "protocol": quality_contract["protocol"],
            "status": "DRY_RUN_COMPLETE",
            "cost_estimate_sha256": cost_estimate[
                "cost_estimate_sha256"
            ],
            "call_plan_sha256": cost_estimate["call_plan_sha256"],
            "judge_families": sorted(prices),
            "logical_calls": len(plan),
            "maximum_physical_attempts": cost_estimate[
                "maximum_physical_attempts"
            ],
            "logical_single_attempt_cost_usd": cost_estimate[
                "logical_single_attempt_cost_usd"
            ],
            "maximum_cost_usd": cost_estimate["maximum_cost_usd"],
            "api_clients_created": 0,
            "api_calls_made": 0,
            "training_labels_created": False,
        }
        _persist_or_require_exact(
            path=args.out_dir / "summary.json", payload=summary
        )
        print(summary)
        return

    expected_identity = str(cost_estimate["cost_estimate_sha256"])
    if str(args.accept_cost_estimate_sha256 or "") != expected_identity:
        raise RuntimeError(
            "accepted oracle-memory quality identity does not match dry-run"
        )
    require_paid_run_release(
        pm_config,
        config_path=args.pm_v1_5_config,
        stage=STAGE,
        run=True,
        run_identity=expected_identity,
    )
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path, overwrite=False, stage=STAGE
    )
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
    endpoint_by_family = {
        str(endpoint.family): endpoint for endpoint in endpoints
    }
    clients = {
        family: make_client(endpoint)
        for family, endpoint in endpoint_by_family.items()
        if any(
            str(row["judge_family"]) == family
            and not ledger.succeeded(str(row["physical_call_key"]))
            for row in plan
        )
    }
    item_by_pair = {
        str(row["pair_id"]): dict(row) for row in items
    }
    isolated_failures: dict[str, str] = {}
    failure_streaks = {
        family: (None, 0) for family in endpoint_by_family
    }
    try:
        for row in plan:
            call_key = str(row["physical_call_key"])
            family = str(row["judge_family"])
            if ledger.succeeded(call_key):
                failure_streaks[family] = (None, 0)
                continue
            blocker = call_retry_blocker(
                ledger,
                call_key,
                max_provider_output_attempts=provider_output_attempts[
                    family
                ],
            )
            if blocker is not None:
                retry_class = _persisted_isolatable_failure_class(
                    ledger, call_key
                )
                if retry_class is None:
                    raise RuntimeError(
                        "oracle-memory quality persisted non-isolatable "
                        f"failure: {blocker}"
                    )
                isolated_failures[call_key] = (
                    f"{retry_class}: {blocker}"
                )
                failure_streaks[family] = _advance_failure_streak(
                    previous_class=failure_streaks[family][0],
                    previous_count=failure_streaks[family][1],
                    retry_class=retry_class,
                )
                continue
            item = item_by_pair[str(row["pair_id"])]
            messages = build_quality_messages(item)
            endpoint = endpoint_by_family[family]

            def call_fn(
                *,
                endpoint=endpoint,
                messages=messages,
                row=row,
            ):
                return clients[str(endpoint.family)].chat(
                    messages,
                    temperature=0.0,
                    max_tokens=int(row["max_output_tokens"]),
                    seed=int(row["seed"]),
                    response_schema=OracleMemoryQualityOutput,
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
                        "judge_family": family,
                    },
                    prompt_sha256=str(row["prompt_sha256"]),
                    call_fn=call_fn,
                    max_provider_output_attempts=(
                        provider_output_attempts[family]
                    ),
                    backoff_seconds=TRANSPORT_BACKOFF_SECONDS,
                )
            except Exception as exc:
                retry_class = _isolatable_failure_class(exc)
                if retry_class is None:
                    raise
                isolated_failures[call_key] = (
                    f"{retry_class}: {type(exc).__name__}: {exc}"
                )
                failure_streaks[family] = _advance_failure_streak(
                    previous_class=failure_streaks[family][0],
                    previous_count=failure_streaks[family][1],
                    retry_class=retry_class,
                )
                continue
            try:
                usage = require_reported_usage(result.usage, stage=STAGE)
                if int(usage["prompt_tokens"]) > int(
                    row["input_tokens_est"]
                ):
                    raise RuntimeError(
                        "provider-reported prompt usage exceeds frozen bound"
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
            failure_streaks[family] = (None, 0)
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
        parsed = OracleMemoryQualityOutput.model_validate(
            dict(terminal.get("result") or {}).get("parsed")
        )
        usage = require_reported_usage(
            terminal.get("usage"), stage=STAGE
        )
        family = str(row["judge_family"])
        actual_input_tokens += int(usage["prompt_tokens"])
        actual_output_tokens += int(usage["completion_tokens"])
        actual_cost += (
            int(usage["prompt_tokens"])
            / 1_000_000
            * float(prices[family]["input"])
            + int(usage["completion_tokens"])
            / 1_000_000
            * float(prices[family]["output"])
        )
        judgments.append(
            {
                "state_id": str(row["state_id"]),
                "user_id": str(row["user_id"]),
                "regime": str(row["regime"]),
                "target_item_utility": str(
                    row["target_item_utility"]
                ),
                "pair_id": str(row["pair_id"]),
                "order_variant": int(row["order_variant"]),
                "arm_a": str(row["arm_a"]),
                "arm_b": str(row["arm_b"]),
                "expected_winner": str(row["expected_winner"]),
                "judge_endpoint": str(row["judge_endpoint"]),
                "judge_family": family,
                "request_hash": str(
                    terminal.get("request_hash") or ""
                ),
                **parsed.model_dump(mode="json"),
            }
        )
    judgments_path = args.out_dir / "judgments.jsonl"
    write_jsonl(judgments_path, judgments)
    complete = len(judgments) == len(plan)
    expected_state_ids = sorted(
        {str(row["state_id"]) for row in items}
    )
    aggregate = (
        aggregate_quality_diagnostic(
            judgments,
            expected_families=sorted(endpoint_by_family),
            expected_state_ids=expected_state_ids,
        )
        if complete
        else None
    )
    summary = {
        "protocol": quality_contract["protocol"],
        "status": (
            "COMPLETE_REPORT_ONLY" if complete else "INCOMPLETE"
        ),
        "reportability": "TRAIN_ONLY_DIAGNOSTIC",
        "training_labels_created": False,
        "pm_training_authorized": False,
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
        "quality_contract_sha256": quality_contract[
            "contract_sha256"
        ],
        "transport_contract_sha256": transport_contract[
            "contract_sha256"
        ],
    }
    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    if not complete:
        raise RuntimeError(
            "oracle-memory quality diagnostic incomplete: "
            f"{len(plan) - len(judgments)} calls missing"
        )
    create_artifact_attestation(
        args.out_dir / "artifact_attestation.json",
        stage=STAGE,
        inputs={
            "tracked_contract": args.tracked_contract,
            "prepared_contract": (
                args.prepared_dir / "quality_contract.json"
            ),
            "quality_items": (
                args.prepared_dir / "quality_items.jsonl"
            ),
            "quality_prompt_rows": (
                args.prepared_dir / "quality_prompt_rows.jsonl"
            ),
            "call_plan": plan_path,
            "cost_estimate": cost_path,
        },
        outputs={
            "summary": (summary_path, False),
            "judgments": (judgments_path, True),
            "physical_attempt_ledger": (ledger_path, True),
        },
        parameters={
            "cost_estimate_sha256": expected_identity,
            "quality_contract_sha256": quality_contract[
                "contract_sha256"
            ],
            "transport_contract": transport_contract,
            "aggregate_status": str(
                (aggregate or {}).get("status") or ""
            ),
            "training_labels_created": False,
            "pm_training_authorized": False,
        },
    )
    print(summary)


if __name__ == "__main__":
    main()
