from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import (
    Endpoint,
    chat_request_payload,
    make_client,
    request_log,
    request_payload_has_schema,
)
from .artifacts import create_artifact_attestation, require_artifact_attestation
from .attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
    reported_prompt_token_error,
)
from .evoemo import evaluator_context, load_evoemo
from .io import (
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .pm_v2_external_eval import build_external_pointwise_case, load_fixed_turns
from .pm_v2_judging import (
    ResponseJudgeOutput,
    RiskJudgeOutput,
    prompt_contract_hash,
)
from .text import conservative_token_bound, estimate_tokens


POINTWISE_SCHEMA_SMOKE_PROTOCOL = "pm-v2-external-pointwise-schema-smoke-v1"
POINTWISE_SCHEMA_SMOKE_STAGE = "pm_v2_external_pointwise_schema_smoke"
POINTWISE_SCHEMA_SMOKE_PURPOSE = "schema_transport_only_not_raw_family_quality"


def _endpoint_contract(endpoint: Endpoint) -> dict[str, Any]:
    payload = {
        "model": endpoint.model,
        "family": endpoint.family,
        "base_url": endpoint.base_url,
    }
    return {**payload, "sha256": sha256_text(canonical_json(payload))}


def _unit_id(unit: tuple[str, int, int, str, int]) -> str:
    return sha256_text(canonical_json(unit))[:24]


def precreate_pointwise_schema_smoke_clients(
    endpoints: Sequence[Endpoint],
) -> dict[str, Any]:
    """Validate both client constructions before any paid attempt is reserved."""

    clients: dict[str, Any] = {}
    try:
        for endpoint in endpoints:
            family = str(endpoint.family or "")
            if not family or family in clients:
                raise RuntimeError("schema-smoke endpoint families are invalid")
            clients[family] = make_client(endpoint)
    except Exception:
        for client in clients.values():
            client.close()
        raise
    return clients


def _authorized_context(
    evoemo_path: str | Path, unit: tuple[str, int, int, str, int]
) -> str:
    user_id, topic_index, _, _, _ = unit
    matches = [
        canonical_json(evaluator_context(user, topic))
        for user in load_evoemo(evoemo_path)
        if str(user["id"]) == user_id
        for topic in user.get("subsequent_topics") or []
        if int(topic["idx"]) == topic_index
    ]
    if len(matches) != 1:
        raise RuntimeError("schema-smoke unit lacks one exact authorized context")
    return matches[0]


def build_pointwise_schema_smoke_plan(
    *,
    turn_row: Mapping[str, Any],
    authorized_user_context: str,
    unit: tuple[str, int, int, str, int],
    condition: str,
    endpoints: Sequence[Endpoint],
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
    input_token_safety_factor: float,
    judge_seed: int,
    estimated_response_output_tokens: int,
    estimated_risk_output_tokens: int,
    study_freeze_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build exactly two families x response/risk using full external prompts."""

    if len(endpoints) != 2 or len({endpoint.family for endpoint in endpoints}) != 2:
        raise ValueError("pointwise schema smoke requires exactly two judge families")
    if any(not endpoint.family for endpoint in endpoints):
        raise ValueError("pointwise schema smoke endpoints require declared families")
    families = {str(endpoint.family) for endpoint in endpoints}
    pricing = {
        str(family): {str(key): float(value) for key, value in values.items()}
        for family, values in pricing_usd_per_mtok.items()
    }
    if set(pricing) != families or any(
        set(values) != {"input", "output"}
        or any(value < 0.0 for value in values.values())
        for values in pricing.values()
    ):
        raise ValueError("schema-smoke pricing must exactly cover both families")
    if input_token_safety_factor < 1.0:
        raise ValueError("schema-smoke input token safety factor must be at least one")
    if estimated_response_output_tokens <= 0 or estimated_risk_output_tokens <= 0:
        raise ValueError("schema-smoke output token bounds must be positive")

    pointwise_case = build_external_pointwise_case(
        turn_row,
        authorized_user_context=authorized_user_context,
    )
    unit_id = _unit_id(unit)
    plan: list[dict[str, Any]] = []
    execution: dict[str, dict[str, Any]] = {}
    for family_index, endpoint in enumerate(endpoints):
        family = str(endpoint.family)
        base_seed = int(judge_seed) + family_index * 2
        for judge_type, messages, schema, max_output_tokens, seed in (
            (
                "response",
                pointwise_case["response_messages"],
                ResponseJudgeOutput,
                int(estimated_response_output_tokens),
                base_seed,
            ),
            (
                "risk",
                pointwise_case["risk_messages"],
                RiskJudgeOutput,
                int(estimated_risk_output_tokens),
                base_seed + 1,
            ),
        ):
            payload = chat_request_payload(
                endpoint,
                messages,
                temperature=0.0,
                max_tokens=max_output_tokens,
                seed=seed,
                response_schema=schema,
            )
            if not request_payload_has_schema(payload):
                raise RuntimeError("schema-smoke request payload lacks structured schema")
            payload_text = canonical_json(payload)
            prompt_hash = sha256_text(canonical_json(messages))
            request_payload_sha256 = sha256_text(payload_text)
            input_tokens_est = conservative_token_bound(
                payload_text,
                safety_factor=float(input_token_safety_factor),
            )
            record_ids = {
                "unit_id": unit_id,
                "condition": condition,
                "judge_family": family,
                "judge_type": judge_type,
            }
            call_key = physical_call_key(
                stage=POINTWISE_SCHEMA_SMOKE_STAGE,
                record_ids=record_ids,
                prompt_sha256=prompt_hash,
                endpoint=endpoint,
                request_parameters={
                    "temperature": 0.0,
                    "max_tokens": max_output_tokens,
                    "seed": seed,
                    "response_schema": schema.__name__,
                    "request_payload_sha256": request_payload_sha256,
                    "retries": 1,
                    "study_freeze_sha256": study_freeze_sha256,
                },
            )
            row = {
                **record_ids,
                "unit": list(unit),
                "judge_model": endpoint.model,
                "seed": seed,
                "input_tokens_est": input_tokens_est,
                "base_input_tokens_est": estimate_tokens(payload_text),
                "max_output_tokens": max_output_tokens,
                "max_http_attempts": 1,
                "prompt_hash": prompt_hash,
                "request_payload_sha256": request_payload_sha256,
                "request_payload_includes_schema": True,
                "condition_identity_hidden_from_prompt": (
                    condition not in canonical_json(messages)
                ),
                "pricing_usd_per_mtok": pricing[family],
                "maximum_cost_usd": (
                    input_tokens_est / 1_000_000 * pricing[family]["input"]
                    + max_output_tokens / 1_000_000 * pricing[family]["output"]
                ),
                "physical_call_key": call_key,
            }
            if not row["condition_identity_hidden_from_prompt"]:
                raise RuntimeError("actual schema-smoke condition leaked into judge prompt")
            plan.append(row)
            execution[call_key] = {
                "endpoint": endpoint,
                "messages": messages,
                "schema": schema,
                "record_ids": record_ids,
            }
    expected_matrix = {
        (family, judge_type)
        for family in sorted(families)
        for judge_type in ("response", "risk")
    }
    observed_matrix = {
        (str(row["judge_family"]), str(row["judge_type"])) for row in plan
    }
    if len(plan) != 4 or observed_matrix != expected_matrix:
        raise RuntimeError("schema-smoke plan is not the exact 2x2 matrix")
    return plan, execution


def run_external_pointwise_schema_smoke(
    *,
    evoemo_path: str | Path,
    study_freeze_path: str | Path,
    study_freeze_sha256: str,
    turn_path: str | Path,
    generation_attestation_path: str | Path,
    forced_swap_summary_path: str | Path,
    forced_swap_attestation_path: str | Path,
    full_expected_units: Sequence[tuple[str, int, int, str, int]],
    smoke_unit: tuple[str, int, int, str, int],
    condition: str,
    endpoints: Sequence[Endpoint],
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
    api_cost_planning: Mapping[str, Any],
    judge_seed: int,
    estimated_response_output_tokens: int,
    estimated_risk_output_tokens: int,
    out_dir: str | Path,
    run: bool,
    accept_cost_estimate_sha256: str | None,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    if run and overwrite:
        raise RuntimeError("paid schema-smoke runs prohibit overwrite")
    planning = dict(api_cost_planning)
    if set(planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    } or float(planning["input_token_safety_factor"]) < 1.0 or not bool(
        planning["fail_on_reported_input_overrun"]
    ):
        raise ValueError("schema-smoke requires the fail-closed API planning contract")
    full_units = sorted(full_expected_units)
    if smoke_unit not in set(full_units):
        raise RuntimeError("schema-smoke unit is outside the frozen external universe")
    matrix = load_fixed_turns(
        [turn_path],
        conditions=[condition],
        turn_indices=sorted({int(unit[-1]) for unit in full_units}),
        expected_units=[smoke_unit],
        full_expected_units=full_units,
        excluded_units=[unit for unit in full_units if unit != smoke_unit],
    )
    turn_row = matrix[(*smoke_unit, condition)]
    authorized = _authorized_context(evoemo_path, smoke_unit)
    plan, execution = build_pointwise_schema_smoke_plan(
        turn_row=turn_row,
        authorized_user_context=authorized,
        unit=smoke_unit,
        condition=condition,
        endpoints=endpoints,
        pricing_usd_per_mtok=pricing_usd_per_mtok,
        input_token_safety_factor=float(planning["input_token_safety_factor"]),
        judge_seed=judge_seed,
        estimated_response_output_tokens=estimated_response_output_tokens,
        estimated_risk_output_tokens=estimated_risk_output_tokens,
        study_freeze_sha256=study_freeze_sha256,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    estimate_path = out_dir / "cost_estimate.json"
    call_plan_path = out_dir / "call_plan.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    raw_path = out_dir / "raw_judge_calls.jsonl"
    result_path = out_dir / "schema_results.jsonl"
    summary_path = out_dir / "summary.json"
    attestation_path = out_dir / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=overwrite,
        stage="external pointwise schema smoke",
    )
    if overwrite:
        targets = (manifest_path, estimate_path, call_plan_path, raw_path, result_path, summary_path, attestation_path)
        for path in targets:
            if path.exists():
                path.unlink()

    endpoint_contracts = {
        str(endpoint.family): _endpoint_contract(endpoint) for endpoint in endpoints
    }
    ensure_run_manifest(
        manifest_path,
        {
            "stage": POINTWISE_SCHEMA_SMOKE_STAGE,
            "protocol": POINTWISE_SCHEMA_SMOKE_PROTOCOL,
            "purpose": POINTWISE_SCHEMA_SMOKE_PURPOSE,
            "study_freeze_sha256": study_freeze_sha256,
            "evoemo_sha256": sha256_file(evoemo_path),
            "turns_sha256": sha256_file(turn_path),
            "generation_attestation_sha256": sha256_file(
                generation_attestation_path
            ),
            "forced_swap_summary_sha256": sha256_file(forced_swap_summary_path),
            "forced_swap_attestation_sha256": sha256_file(
                forced_swap_attestation_path
            ),
            "condition": condition,
            "condition_identity_visibility_to_judges": "hidden",
            "unit": list(smoke_unit),
            "unit_id": _unit_id(smoke_unit),
            "endpoint_contracts": endpoint_contracts,
            "judge_prompt_contract_sha256": prompt_contract_hash(),
            "judge_seed": int(judge_seed),
            "judge_retries": 1,
            "api_cost_planning": planning,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        },
    )
    call_plan_sha256 = sha256_text(canonical_json(plan))
    estimate_payload = {
        "stage": POINTWISE_SCHEMA_SMOKE_STAGE,
        "protocol": POINTWISE_SCHEMA_SMOKE_PROTOCOL,
        "purpose": POINTWISE_SCHEMA_SMOKE_PURPOSE,
        "expected_api_calls": 4,
        "maximum_physical_http_attempts": 4,
        "total_input_tokens_est": sum(int(row["input_tokens_est"]) for row in plan),
        "max_input_tokens_per_call_est": max(
            int(row["input_tokens_est"]) for row in plan
        ),
        "total_output_tokens_est": sum(int(row["max_output_tokens"]) for row in plan),
        "estimated_cost_usd": sum(float(row["maximum_cost_usd"]) for row in plan),
        "call_plan_sha256": call_plan_sha256,
        "study_freeze_sha256": study_freeze_sha256,
        "unit_id": _unit_id(smoke_unit),
        "condition": condition,
        "endpoint_contracts": endpoint_contracts,
        "judge_prompt_contract_sha256": prompt_contract_hash(),
        "api_cost_planning": planning,
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }
    estimate = {
        **estimate_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(estimate_payload)),
    }
    checks = {
        "api_calls": 4 <= int(max_api_calls),
        "estimated_cost_usd": estimate["estimated_cost_usd"]
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": estimate["max_input_tokens_per_call_est"]
        <= int(max_input_tokens_per_call),
        "exact_call_matrix": len(plan) == 4,
        "all_payloads_include_schema": all(
            bool(row["request_payload_includes_schema"]) for row in plan
        ),
        "condition_hidden": all(
            bool(row["condition_identity_hidden_from_prompt"]) for row in plan
        ),
    }
    budget_gate = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": estimate_payload["budget_limits"],
    }
    saved_estimate = {**estimate, "budget_gate": budget_gate}
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=POINTWISE_SCHEMA_SMOKE_STAGE,
        expected_calls={str(row["physical_call_key"]): 1 for row in plan},
        maximum_total_attempts=int(max_api_calls),
    )
    if ledger.started_attempts:
        if not call_plan_path.is_file() or not estimate_path.is_file():
            raise RuntimeError("spent schema-smoke ledger lacks its immutable dry-run")
        if list(iter_jsonl(call_plan_path)) != plan or read_json(estimate_path) != saved_estimate:
            raise RuntimeError("spent schema-smoke ledger forbids dry-run plan drift")
    elif not run or budget_gate["status"] != "PASS":
        write_jsonl(call_plan_path, plan)
        write_json(estimate_path, saved_estimate)
    if budget_gate["status"] != "PASS":
        raise RuntimeError("pointwise schema-smoke budget/schema gate failed")
    dry = {
        "status": "DRY_RUN_COMPLETE",
        "protocol": POINTWISE_SCHEMA_SMOKE_PROTOCOL,
        "purpose": POINTWISE_SCHEMA_SMOKE_PURPOSE,
        "condition": condition,
        "unit": list(smoke_unit),
        "unit_id": _unit_id(smoke_unit),
        "expected_calls": 4,
        "cost_estimate": estimate,
        "budget_gate": budget_gate,
        "call_plan_path": str(call_plan_path),
        "cost_estimate_path": str(estimate_path),
    }
    if not run:
        return dry
    if not call_plan_path.is_file() or not estimate_path.is_file():
        raise RuntimeError("schema-smoke API run requires its saved dry-run")
    if list(iter_jsonl(call_plan_path)) != plan or read_json(estimate_path) != saved_estimate:
        raise RuntimeError("saved schema-smoke dry-run is stale")
    if accept_cost_estimate_sha256 != estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "schema-smoke API run requires exact --accept-cost-estimate-sha256"
        )

    # Endpoint credentials/transport constructors for *both* families must work
    # before reserving the first physical attempt.  This prevents a family-1
    # charge followed by discovery that family 2 cannot even create a client.
    clients = precreate_pointwise_schema_smoke_clients(endpoints)
    try:
        for row in plan:
            call_key = str(row["physical_call_key"])
            if ledger.succeeded(call_key) or ledger.exhausted(call_key):
                continue
            item = execution[call_key]
            endpoint = item["endpoint"]
            family = str(endpoint.family)
            reservation = ledger.reserve(
                call_key,
                record_ids=item["record_ids"],
                prompt_sha256=str(row["prompt_hash"]),
            )
            result = None
            parsed = None
            error = None
            try:
                result, parsed = clients[family].chat(
                    item["messages"],
                    temperature=0.0,
                    max_tokens=int(row["max_output_tokens"]),
                    seed=int(row["seed"]),
                    response_schema=item["schema"],
                    retries=1,
                )
                if parsed is None:
                    raise RuntimeError("schema-smoke client returned no parsed object")
                error = reported_prompt_token_error(
                    result.usage,
                    maximum_prompt_tokens=int(row["input_tokens_est"]),
                    stage="external pointwise schema smoke",
                    require_positive=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            log = request_log(
                stage=POINTWISE_SCHEMA_SMOKE_STAGE,
                endpoint=endpoint,
                messages=item["messages"],
                result=result,
                parsed=parsed,
                error=error,
                prompt_hash=str(row["prompt_hash"]),
                record_ids={
                    **item["record_ids"],
                    "physical_call_key": call_key,
                    "physical_attempt_index": reservation.attempt_index,
                    "physical_attempt_key": reservation.attempt_key,
                    "request_payload_includes_schema": True,
                },
            )
            ledger.finish(
                reservation,
                succeeded=error is None,
                request_hash=result.request_hash if result is not None else None,
                usage=result.usage if result is not None else None,
                error=error,
                result={
                    "parsed": (
                        parsed.model_dump(mode="json") if parsed is not None else None
                    ),
                    "request_log": log,
                },
            )
    finally:
        for client in clients.values():
            client.close()

    raw_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for row in plan:
        call_key = str(row["physical_call_key"])
        terminal = ledger.terminal_row(call_key)
        if terminal is None:
            continue
        stored = terminal.get("result") or {}
        if isinstance(stored.get("request_log"), Mapping):
            raw_rows.append(dict(stored["request_log"]))
        if terminal.get("event") != "SUCCEEDED":
            continue
        schema = (
            ResponseJudgeOutput
            if row["judge_type"] == "response"
            else RiskJudgeOutput
        )
        parsed = schema.model_validate(stored.get("parsed"))
        usage_error = reported_prompt_token_error(
            terminal.get("usage"),
            maximum_prompt_tokens=int(row["input_tokens_est"]),
            stage="persisted external pointwise schema smoke",
            require_positive=True,
        )
        if usage_error is not None:
            raise RuntimeError(usage_error)
        result_rows.append(
            {
                **row,
                "parsed": parsed.model_dump(mode="json"),
                "usage": dict(terminal["usage"]),
                "request_hash": terminal["request_hash"],
                "physical_attempt_index": terminal["attempt_index"],
                "physical_attempt_key": terminal["attempt_key"],
                "schema_validation": "PASS",
                "usage_validation": "PASS",
            }
        )
    write_jsonl(raw_path, raw_rows)
    write_jsonl(result_path, result_rows)
    observed_matrix = {
        (str(row["judge_family"]), str(row["judge_type"]))
        for row in result_rows
    }
    expected_matrix = {
        (str(endpoint.family), judge_type)
        for endpoint in endpoints
        for judge_type in ("response", "risk")
    }
    pass_checks = {
        "exact_four_successes": len(result_rows) == 4,
        "exact_family_schema_matrix": observed_matrix == expected_matrix,
        "all_request_payloads_include_schema": all(
            bool(row["request_payload_includes_schema"]) for row in result_rows
        ),
        "all_schema_validated": all(
            row["schema_validation"] == "PASS" for row in result_rows
        ),
        "all_usage_validated": all(
            row["usage_validation"] == "PASS" for row in result_rows
        ),
        "condition_identity_hidden": all(
            bool(row["condition_identity_hidden_from_prompt"])
            for row in result_rows
        ),
    }
    summary = {
        **dry,
        "status": "PASS" if all(pass_checks.values()) else "FAIL",
        "execution_status": (
            "COMPLETE" if all(pass_checks.values()) else "INCOMPLETE"
        ),
        "schema_successes": len(result_rows),
        "schema_attempts": ledger.started_attempts,
        "checks": pass_checks,
        "endpoint_contracts": endpoint_contracts,
        "judge_prompt_contract_sha256": prompt_contract_hash(),
        "condition_identity_visibility_to_judges": "hidden",
        "raw_family_quality_claim_permitted": False,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "final_ledger_sha256": sha256_text(canonical_json(ledger.event_rows)),
    }
    write_json(summary_path, summary)
    if summary["status"] != "PASS":
        raise RuntimeError("external pointwise schema smoke did not PASS")
    create_artifact_attestation(
        attestation_path,
        stage=POINTWISE_SCHEMA_SMOKE_STAGE,
        inputs={
            "evoemo": evoemo_path,
            "study_freeze": study_freeze_path,
            "turns": turn_path,
            "generation_attestation": generation_attestation_path,
            "forced_swap_summary": forced_swap_summary_path,
            "forced_swap_attestation": forced_swap_attestation_path,
            "run_manifest": manifest_path,
            "cost_estimate": estimate_path,
            "call_plan": call_plan_path,
        },
        outputs={
            "raw_calls": (raw_path, True),
            "physical_attempt_ledger": (ledger_path, True),
            "schema_results": (result_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "protocol": POINTWISE_SCHEMA_SMOKE_PROTOCOL,
            "purpose": POINTWISE_SCHEMA_SMOKE_PURPOSE,
            "condition": condition,
            "condition_identity_visibility_to_judges": "hidden",
            "unit": list(smoke_unit),
            "unit_id": _unit_id(smoke_unit),
            "expected_calls": 4,
            "schema_successes": 4,
            "endpoint_contracts": endpoint_contracts,
            "judge_prompt_contract_sha256": prompt_contract_hash(),
            "judge_seed": int(judge_seed),
            "judge_retries": 1,
            "pricing_usd_per_mtok": {
                str(key): dict(value)
                for key, value in pricing_usd_per_mtok.items()
            },
            "api_cost_planning": planning,
            "cost_estimate_sha256": estimate["cost_estimate_sha256"],
            "raw_family_quality_claim_permitted": False,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "final_ledger_sha256": summary["final_ledger_sha256"],
        },
        expected={"schema_results": 4, "raw_calls": 4},
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary


def require_external_pointwise_schema_smoke_pass(
    summary_path: str | Path,
    attestation_path: str | Path,
    *,
    study_freeze_sha256: str,
    contract: Mapping[str, Any],
    forced_swap_selected_units_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless the exact frozen four-call transport smoke passed."""

    verification = require_artifact_attestation(
        attestation_path,
        required_stage=POINTWISE_SCHEMA_SMOKE_STAGE,
        required_output_paths={"summary": summary_path},
        expected_freeze_sha256=study_freeze_sha256,
    )
    summary = read_json(summary_path)
    attestation = read_json(attestation_path)
    parameters = attestation.get("parameters") or {}
    expected = {
        "protocol": POINTWISE_SCHEMA_SMOKE_PROTOCOL,
        "purpose": POINTWISE_SCHEMA_SMOKE_PURPOSE,
        "condition": str(contract["condition"]),
        "condition_identity_visibility_to_judges": "hidden",
        "unit": list(contract["unit"]),
        "unit_id": str(contract["unit_id"]),
        "expected_calls": 4,
        "schema_successes": 4,
        "endpoint_contracts": {
            str(row["family"]): {
                "model": str(row["model"]),
                "family": str(row["family"]),
                "base_url": str(row["base_url"]),
                "sha256": str(row["sha256"]),
            }
            for row in contract["judge_endpoints"]
        },
        "judge_prompt_contract_sha256": prompt_contract_hash(),
        "judge_seed": int(contract["judge_seed"]),
        "judge_retries": 1,
        "pricing_usd_per_mtok": contract["judge_pricing_usd_per_mtok"],
        "api_cost_planning": contract["api_cost_planning"],
        "raw_family_quality_claim_permitted": False,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    }
    if (
        contract.get("protocol") != POINTWISE_SCHEMA_SMOKE_PROTOCOL
        or contract.get("purpose") != POINTWISE_SCHEMA_SMOKE_PURPOSE
        or contract.get("expected_calls") != 4
        or contract.get("forced_swap_selected_units_sha256")
        != forced_swap_selected_units_sha256
        or contract.get("judge_prompt_contract_sha256") != prompt_contract_hash()
        or not bool(contract.get("required_before_full_external_client_creation"))
    ):
        raise RuntimeError("frozen pointwise schema-smoke contract is stale")
    for key, value in expected.items():
        if parameters.get(key) != value:
            raise RuntimeError(f"pointwise schema-smoke attestation mismatch: {key}")
    if (
        summary.get("status") != "PASS"
        or summary.get("execution_status") != "COMPLETE"
        or int(summary.get("schema_successes", -1)) != 4
        or not all(bool(value) for value in (summary.get("checks") or {}).values())
        or summary.get("raw_family_quality_claim_permitted") is not False
    ):
        raise RuntimeError("external pointwise schema smoke did not exactly PASS")
    output = (attestation.get("outputs") or {}).get("schema_results") or {}
    result_path = output.get("path")
    rows = list(iter_jsonl(result_path)) if result_path else []
    expected_matrix = {
        (str(row["family"]), judge_type)
        for row in contract["judge_endpoints"]
        for judge_type in ("response", "risk")
    }
    if (
        len(rows) != 4
        or {
            (str(row["judge_family"]), str(row["judge_type"])) for row in rows
        }
        != expected_matrix
        or any(
            not bool(row.get("request_payload_includes_schema"))
            or row.get("schema_validation") != "PASS"
            or row.get("usage_validation") != "PASS"
            or not bool(row.get("condition_identity_hidden_from_prompt"))
            for row in rows
        )
    ):
        raise RuntimeError("pointwise schema-smoke result matrix is invalid")
    return {
        "status": "PASS",
        "attestation_sha256": verification["attestation_sha256"],
        "summary_sha256": sha256_file(summary_path),
        "unit_id": str(contract["unit_id"]),
        "result_rows_sha256": sha256_text(canonical_json(rows)),
    }
