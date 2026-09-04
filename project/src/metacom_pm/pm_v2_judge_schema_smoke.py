from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .api import (
    Endpoint,
    chat_request_payload,
    make_client,
    request_payload_has_schema,
    require_reported_usage,
)
from .artifacts import require_artifact_attestation
from .attempt_ledger import PersistentAttemptLedger, physical_call_key
from .config import endpoint_from_config, load_config
from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .pm_v2_contracts import PMV2Split, PMV2State, ResourceNeedRegime
from .pm_v2_audit import audit_deployable_feature_observability
from .pm_v2_data import load_evaluator_context_index, load_states
from .pm_v2_judging import (
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_response_messages,
    build_risk_messages,
    prompt_contract_hash,
)
from .pm_v2_semantic_audit import require_semantic_sanity_pass
from .text import conservative_token_bound, estimate_tokens


JUDGE_SCHEMA_SMOKE_STAGE = "pm_v2_development_judge_schema_smoke"
JUDGE_SCHEMA_SMOKE_PROTOCOL = "pm-v2-development-judge-schema-smoke-v1"
_SMOKE_KEYS = {
    "required_before_pilot_action_api",
    "selection_seed",
    "expected_physical_calls",
    "candidate_response",
    "selected_context",
}


def judge_schema_smoke_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    pilot = (config.get("development_judging") or {}).get(
        "compatibility_pilot"
    )
    if not isinstance(pilot, Mapping):
        raise RuntimeError("PM-v2 config lacks development compatibility pilot")
    raw = pilot.get("schema_smoke")
    if not isinstance(raw, Mapping) or set(raw) != _SMOKE_KEYS:
        observed = set(raw) if isinstance(raw, Mapping) else set()
        raise RuntimeError(
            "schema_smoke must use the frozen schema: "
            f"missing={sorted(_SMOKE_KEYS-observed)}, "
            f"extra={sorted(observed-_SMOKE_KEYS)}"
        )
    if raw["required_before_pilot_action_api"] is not True:
        raise RuntimeError(
            "schema_smoke.required_before_pilot_action_api must be true"
        )
    for name in ("selection_seed", "expected_physical_calls"):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int):
            raise TypeError(f"schema smoke setting {name} must be an integer")
    settings = {
        "required_before_pilot_action_api": True,
        "selection_seed": int(raw["selection_seed"]),
        "expected_physical_calls": int(raw["expected_physical_calls"]),
        "candidate_response": str(raw["candidate_response"]).strip(),
        "selected_context": str(raw["selected_context"]).strip(),
    }
    if settings["selection_seed"] < 0:
        raise ValueError("schema smoke selection_seed must be non-negative")
    if settings["expected_physical_calls"] != 4:
        raise ValueError("schema smoke must freeze exactly four physical calls")
    if not settings["candidate_response"] or not settings["selected_context"]:
        raise ValueError("schema smoke fixed candidate/context must be non-empty")
    return settings


def preflight_judge_schema_smoke_clients(
    execution: Mapping[str, Mapping[str, Any]], *, client_factory=make_client
) -> dict[str, Any]:
    """Resolve both credentials and clients before any ledger reservation.

    Credential resolution is a separate first pass.  A missing second-family
    key therefore cannot instantiate or spend the first-family request.
    """

    endpoints: dict[str, Endpoint] = {}
    for item in execution.values():
        endpoint = item.get("endpoint")
        if not isinstance(endpoint, Endpoint) or not endpoint.family:
            raise RuntimeError("schema-smoke execution lacks a family endpoint")
        prior = endpoints.setdefault(str(endpoint.family), endpoint)
        if prior != endpoint:
            raise RuntimeError("schema-smoke family maps to multiple endpoints")
    if len(endpoints) != 2:
        raise RuntimeError("schema smoke requires exactly two endpoint families")
    for endpoint in endpoints.values():
        endpoint.api_key
    clients: dict[str, Any] = {}
    try:
        for family, endpoint in endpoints.items():
            clients[family] = client_factory(endpoint)
    except Exception:
        for client in clients.values():
            client.close()
        raise
    return clients


def pending_judge_schema_smoke_calls(
    ledger: PersistentAttemptLedger, call_plan: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str | None]:
    """Return resumable calls, stopping globally after any spent failure.

    With one physical attempt per logical call, an exhausted unsuccessful call
    makes the exact four-call PASS matrix unreachable.  Later calls must not be
    issued on a rerun merely because they were never reached previously.
    """

    failed = [
        str(row["physical_call_key"])
        for row in call_plan
        if ledger.exhausted(str(row["physical_call_key"]))
        and not ledger.succeeded(str(row["physical_call_key"]))
    ]
    if failed:
        return [], (
            "schema smoke PASS is unreachable after spent unsuccessful calls: "
            + ",".join(sorted(failed))
        )
    return (
        [
            row
            for row in call_plan
            if not ledger.succeeded(str(row["physical_call_key"]))
            and not ledger.exhausted(str(row["physical_call_key"]))
        ],
        None,
    )


def persist_or_validate_schema_smoke_dry_run(
    *,
    ledger_path: str | Path,
    estimate_path: str | Path,
    call_plan_path: str | Path,
    cost_estimate: Mapping[str, Any],
    budget_gate: Mapping[str, Any],
    call_plan: list[dict[str, Any]],
) -> str:
    """Never rewrite an accepted plan after a physical attempt exists."""

    ledger_path = Path(ledger_path)
    estimate_path = Path(estimate_path)
    call_plan_path = Path(call_plan_path)
    expected_estimate = {**dict(cost_estimate), "budget_gate": dict(budget_gate)}
    spent = ledger_path.is_file() and ledger_path.stat().st_size > 0
    if spent:
        if not estimate_path.is_file() or not call_plan_path.is_file():
            raise RuntimeError(
                "schema-smoke ledger is non-empty but its accepted dry-run is missing"
            )
        if read_json(estimate_path) != expected_estimate or list(
            iter_jsonl(call_plan_path)
        ) != call_plan:
            raise RuntimeError(
                "schema-smoke ledger is non-empty and the current dry-run is stale; "
                "refusing to rewrite paid-attempt lineage"
            )
        return "VALIDATED_EXISTING"
    write_json(estimate_path, expected_estimate)
    write_jsonl(call_plan_path, call_plan)
    return "WRITTEN"


def _plan_is_self_consistent(plan: Mapping[str, Any]) -> bool:
    payload = {key: value for key, value in plan.items() if key != "pilot_plan_sha256"}
    return plan.get("pilot_plan_sha256") == sha256_text(canonical_json(payload))


def require_exact_pilot_deployable_feature_observability(
    *,
    plan: Mapping[str, Any],
    states: list[PMV2State],
    evaluator_contexts: Any,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    observed = audit_deployable_feature_observability(
        states,
        evaluator_contexts=evaluator_contexts,
        settings=dict(settings),
    )
    if observed.get("status") != "PASS":
        raise RuntimeError(
            "judge schema smoke deployable-feature observability did not PASS"
        )
    if plan.get("deployable_feature_observability") != observed:
        raise RuntimeError(
            "judge schema smoke pilot-plan observability is absent, stale, or forged"
        )
    return observed


def _select_smoke_state(
    plan: Mapping[str, Any], state_by_id: Mapping[str, PMV2State], *, seed: int
) -> tuple[PMV2State, dict[str, Any]]:
    selected = list(plan.get("selected_states") or [])
    train_rows = [row for row in selected if row.get("split") == PMV2Split.TRAIN.value]
    if len(train_rows) != len(selected) or not train_rows:
        raise RuntimeError("judge schema smoke requires an entirely train-only pilot")
    row = min(
        train_rows,
        key=lambda value: sha256_text(
            f"{JUDGE_SCHEMA_SMOKE_PROTOCOL}|{seed}|{value.get('regime')}|"
            f"{value.get('state_id')}"
        ),
    )
    state_id = str(row.get("state_id") or "")
    state = state_by_id.get(state_id)
    if state is None or state.split is not PMV2Split.TRAIN:
        raise RuntimeError("judge schema smoke selected an invalid train state")
    if row.get("card_id") != state.card_id:
        raise RuntimeError("judge schema smoke plan/state card mismatch")
    return state, dict(row)


def build_judge_schema_smoke_contract(
    *,
    experiment_config_path: str | Path,
    pm_v2_config_path: str | Path,
    states_path: str | Path,
    backend_path: str | Path,
    evaluator_contexts_path: str | Path,
    pilot_plan_path: str | Path,
    semantic_sanity_report_path: str | Path,
    semantic_sanity_attestation_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    experiment_config_path = Path(experiment_config_path).resolve()
    pm_v2_config_path = Path(pm_v2_config_path).resolve()
    states_path = Path(states_path).resolve()
    backend_path = Path(backend_path).resolve()
    evaluator_contexts_path = Path(evaluator_contexts_path).resolve()
    pilot_plan_path = Path(pilot_plan_path).resolve()
    semantic_sanity_report_path = Path(semantic_sanity_report_path).resolve()
    semantic_sanity_attestation_path = Path(
        semantic_sanity_attestation_path
    ).resolve()

    experiment_config = load_config(experiment_config_path)
    config = load_config(pm_v2_config_path)
    if config.get("version") != "pm-v2.2":
        raise RuntimeError("judge schema smoke requires PM-v2.2")
    settings = judge_schema_smoke_settings(config)
    states = load_states(states_path)
    state_by_id = {state.state_id: state for state in states}
    evaluator_contexts = load_evaluator_context_index(
        evaluator_contexts_path, states=states, require_exact=True
    )
    semantic_sanity = require_semantic_sanity_pass(
        report_path=semantic_sanity_report_path,
        attestation_path=semantic_sanity_attestation_path,
        config=config,
        config_path=pm_v2_config_path,
        states_path=states_path,
        backend_path=backend_path,
        evaluator_contexts_path=evaluator_contexts_path,
    )
    plan = read_json(pilot_plan_path)
    if (
        plan.get("status") != "READY"
        or plan.get("protocol") != "pm_v2_development_compatibility_pilot_v1"
        or not _plan_is_self_consistent(plan)
    ):
        raise RuntimeError("judge schema smoke requires the exact READY pilot plan")
    expected_plan = {
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "states_sha256": sha256_file(states_path),
        "backend_sha256": sha256_file(backend_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
        "semantic_sanity": semantic_sanity,
    }
    for name, expected in expected_plan.items():
        if plan.get(name) != expected:
            raise RuntimeError(f"judge schema smoke pilot-plan mismatch: {name}")
    pilot_config = config["development_judging"]["compatibility_pilot"]
    deployable_feature_observability = (
        require_exact_pilot_deployable_feature_observability(
            plan=plan,
            states=states,
            evaluator_contexts=evaluator_contexts,
            settings=pilot_config["deployable_feature_observability"],
        )
    )
    expected_actions = [str(value) for value in pilot_config["actions"]]
    selected_states = list(plan.get("selected_states") or [])
    expected_regimes = {regime.value for regime in ResourceNeedRegime}
    if Counter(str(row.get("regime")) for row in selected_states) != Counter(
        {
            regime: int(pilot_config["states_per_regime"])
            for regime in expected_regimes
        }
    ):
        raise RuntimeError("judge schema smoke pilot is not regime balanced")
    if any(
        row.get("split") != PMV2Split.TRAIN.value
        or row.get("actions") != expected_actions
        for row in selected_states
    ):
        raise RuntimeError("judge schema smoke pilot must be train-only/exact-action")

    state, selected_plan_row = _select_smoke_state(
        plan, state_by_id, seed=int(settings["selection_seed"])
    )
    evaluator = evaluator_contexts.by_state[state.state_id]
    authorized_context = str(evaluator["authorized_user_context"])
    judging_config = dict(config["development_judging"])
    endpoint_names = [str(value) for value in judging_config["judge_endpoints"]]
    endpoints = [
        endpoint_from_config(experiment_config, name) for name in endpoint_names
    ]
    families = [str(endpoint.family or "") for endpoint in endpoints]
    if len(endpoints) != 2 or "" in families or len(set(families)) != 2:
        raise RuntimeError("judge schema smoke requires exactly two distinct families")
    pricing = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in dict(judging_config["pricing_usd_per_mtok"]).items()
    }
    if set(pricing) != set(families):
        raise RuntimeError("judge schema smoke pricing/family matrix mismatch")
    cost_settings = dict(config["api_cost_planning"])
    if set(cost_settings) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise RuntimeError("judge schema smoke API cost settings changed")
    safety_factor = float(cost_settings["input_token_safety_factor"])
    if safety_factor < 1.0 or cost_settings["fail_on_reported_input_overrun"] is not True:
        raise RuntimeError("judge schema smoke requires fail-closed token accounting")

    response_messages = build_response_messages(
        state=state,
        authorized_user_context=authorized_context,
        candidate_response=settings["candidate_response"],
    )
    risk_messages = build_risk_messages(
        state=state,
        authorized_user_context=authorized_context,
        selected_context=settings["selected_context"],
        candidate_response=settings["candidate_response"],
    )
    call_plan: list[dict[str, Any]] = []
    execution: dict[str, Any] = {}
    base_seed = int(judging_config["seed"]) + 500_000
    for endpoint_index, (endpoint_name, endpoint) in enumerate(
        zip(endpoint_names, endpoints)
    ):
        for type_index, (judge_type, messages, maximum_tokens, schema) in enumerate(
            (
                (
                    "response",
                    response_messages,
                    int(judging_config["response_max_output_tokens"]),
                    ResponseJudgeOutput,
                ),
                (
                    "risk",
                    risk_messages,
                    int(judging_config["risk_max_output_tokens"]),
                    RiskJudgeOutput,
                ),
            )
        ):
            seed = base_seed + endpoint_index * 2 + type_index
            request_payload = chat_request_payload(
                endpoint,
                messages,
                temperature=0.0,
                max_tokens=maximum_tokens,
                seed=seed,
                response_schema=schema,
            )
            if not request_payload_has_schema(request_payload):
                raise RuntimeError("judge schema smoke request lacks structured schema")
            request_json = canonical_json(request_payload)
            prompt_sha256 = sha256_text(canonical_json(messages))
            record_ids = {
                "state_id": state.state_id,
                "judge_family": str(endpoint.family),
                "judge_type": judge_type,
            }
            call_key = physical_call_key(
                stage=JUDGE_SCHEMA_SMOKE_STAGE,
                record_ids=record_ids,
                prompt_sha256=prompt_sha256,
                endpoint=endpoint,
                request_parameters={
                    "temperature": 0.0,
                    "max_tokens": maximum_tokens,
                    "seed": seed,
                    "response_schema": schema.__name__,
                    "request_payload_sha256": sha256_text(request_json),
                    "retries": 1,
                },
            )
            raw_tokens = estimate_tokens(request_json)
            input_bound = conservative_token_bound(
                request_json, safety_factor=safety_factor
            )
            family_pricing = pricing[str(endpoint.family)]
            row = {
                "physical_call_key": call_key,
                **record_ids,
                "endpoint_name": endpoint_name,
                "judge_model": endpoint.model,
                "endpoint_base_url": endpoint.base_url,
                "seed": seed,
                "prompt_sha256": prompt_sha256,
                "request_payload_sha256": sha256_text(request_json),
                "request_payload_includes_schema": True,
                "response_schema": schema.__name__,
                "raw_estimated_input_tokens": raw_tokens,
                "input_token_upper_bound": input_bound,
                "maximum_output_tokens": maximum_tokens,
                "maximum_physical_attempts": 1,
                "pricing_usd_per_mtok": family_pricing,
                "maximum_cost_usd": input_bound / 1_000_000
                * family_pricing["input"]
                + maximum_tokens / 1_000_000 * family_pricing["output"],
            }
            call_plan.append(row)
            execution[call_key] = {
                "endpoint": endpoint,
                "messages": messages,
                "schema": schema,
                "record_ids": record_ids,
            }
    call_plan = sorted(
        call_plan,
        key=lambda row: (str(row["judge_family"]), str(row["judge_type"])),
    )
    if len(call_plan) != 4 or len(
        {str(row["physical_call_key"]) for row in call_plan}
    ) != 4:
        raise RuntimeError("judge schema smoke call matrix must be exactly four")
    contract = {
        "protocol": JUDGE_SCHEMA_SMOKE_PROTOCOL,
        "stage": JUDGE_SCHEMA_SMOKE_STAGE,
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "states_sha256": sha256_file(states_path),
        "backend_sha256": sha256_file(backend_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "evaluator_contexts_map_sha256": evaluator_contexts.map_sha256,
        "pilot_plan_file_sha256": sha256_file(pilot_plan_path),
        "pilot_plan_sha256": plan["pilot_plan_sha256"],
        "semantic_sanity": semantic_sanity,
        "semantic_sanity_report_sha256": sha256_file(semantic_sanity_report_path),
        "semantic_sanity_attestation_sha256": sha256_file(
            semantic_sanity_attestation_path
        ),
        "schema_smoke_config": settings,
        "deployable_feature_observability": deployable_feature_observability,
        "selected_train_state": selected_plan_row,
        "evaluator_context_id": evaluator["evaluator_context_id"],
        "authorized_context_sha256": sha256_text(authorized_context),
        "candidate_response_sha256": sha256_text(settings["candidate_response"]),
        "selected_context_sha256": sha256_text(settings["selected_context"]),
        "judge_endpoints": [
            {
                "name": name,
                "model": endpoint.model,
                "family": endpoint.family,
                "base_url": endpoint.base_url,
                "api_key_env": endpoint.api_key_env,
            }
            for name, endpoint in zip(endpoint_names, endpoints)
        ],
        "prompt_contract_sha256": prompt_contract_hash(),
        "development_judging": judging_config,
        "api_cost_planning": cost_settings,
        "pricing_usd_per_mtok": pricing,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
        "expected_physical_calls": 4,
    }
    contract["contract_sha256"] = sha256_text(canonical_json(contract))
    return contract, call_plan, execution


def _attested_path(
    attestation: Mapping[str, Any], section: str, logical_name: str
) -> Path:
    record = (attestation.get(section) or {}).get(logical_name)
    if not isinstance(record, Mapping) or not record.get("path"):
        raise RuntimeError(f"schema smoke attestation lacks {section}.{logical_name}")
    return Path(str(record["path"])).resolve()


def require_development_judge_schema_smoke_pass(
    *,
    summary_path: str | Path,
    attestation_path: str | Path,
    experiment_config_path: str | Path,
    pm_v2_config_path: str | Path,
    states_path: str | Path,
    backend_path: str | Path,
    evaluator_contexts_path: str | Path,
    pilot_plan_path: str | Path,
    semantic_sanity_report_path: str | Path,
    semantic_sanity_attestation_path: str | Path,
) -> dict[str, Any]:
    contract, call_plan, _ = build_judge_schema_smoke_contract(
        experiment_config_path=experiment_config_path,
        pm_v2_config_path=pm_v2_config_path,
        states_path=states_path,
        backend_path=backend_path,
        evaluator_contexts_path=evaluator_contexts_path,
        pilot_plan_path=pilot_plan_path,
        semantic_sanity_report_path=semantic_sanity_report_path,
        semantic_sanity_attestation_path=semantic_sanity_attestation_path,
    )
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=JUDGE_SCHEMA_SMOKE_STAGE,
        required_output_paths={"summary": summary_path},
    )
    attestation = read_json(attestation_path)
    expected_inputs = {
        "experiment_config": experiment_config_path,
        "pm_v2_config": pm_v2_config_path,
        "states": states_path,
        "backend": backend_path,
        "evaluator_contexts": evaluator_contexts_path,
        "pilot_plan": pilot_plan_path,
        "semantic_sanity_report": semantic_sanity_report_path,
        "semantic_sanity_attestation": semantic_sanity_attestation_path,
    }
    for name, expected in expected_inputs.items():
        if _attested_path(attestation, "inputs", name) != Path(expected).resolve():
            raise RuntimeError(f"schema smoke attestation input mismatch: {name}")
    estimate_path = _attested_path(attestation, "inputs", "cost_estimate")
    call_plan_path = _attested_path(attestation, "inputs", "call_plan")
    raw_path = _attested_path(attestation, "outputs", "raw_results")
    ledger_path = _attested_path(attestation, "outputs", "physical_attempt_ledger")
    saved_plan = list(iter_jsonl(call_plan_path))
    if saved_plan != call_plan:
        raise RuntimeError("schema smoke attested call plan is stale")
    estimate = read_json(estimate_path)
    estimate_payload = {
        key: value
        for key, value in estimate.items()
        if key not in {"cost_estimate_sha256", "budget_gate"}
    }
    cost_hash = sha256_text(canonical_json(estimate_payload))
    if (
        estimate.get("cost_estimate_sha256") != cost_hash
        or (estimate.get("budget_gate") or {}).get("status") != "PASS"
        or estimate.get("call_plan_sha256") != contract["call_plan_sha256"]
        or int(estimate.get("expected_physical_calls", -1)) != 4
    ):
        raise RuntimeError("schema smoke accepted cost plan is stale")
    parameters = attestation.get("parameters") or {}
    if parameters != {
        "protocol": JUDGE_SCHEMA_SMOKE_PROTOCOL,
        "compatibility_contract": contract,
        "compatibility_contract_sha256": contract["contract_sha256"],
        "accepted_cost_estimate_sha256": cost_hash,
        "gate_status": "PASS",
    }:
        raise RuntimeError("schema smoke attestation contract mismatch")

    raw_rows = list(iter_jsonl(raw_path))
    raw_by_key = {
        str(row.get("physical_call_key") or ""): row for row in raw_rows
    }
    expected_keys = {str(row["physical_call_key"]) for row in call_plan}
    if len(raw_by_key) != 4 or set(raw_by_key) != expected_keys:
        raise RuntimeError("schema smoke raw result matrix is not exact")
    plan_by_key = {str(row["physical_call_key"]): row for row in call_plan}
    for key, row in raw_by_key.items():
        plan_row = plan_by_key[key]
        if (
            row.get("status") != "SUCCESS"
            or row.get("state_id") != plan_row["state_id"]
            or row.get("judge_family") != plan_row["judge_family"]
            or row.get("judge_model") != plan_row["judge_model"]
            or row.get("judge_type") != plan_row["judge_type"]
        ):
            raise RuntimeError(f"schema smoke raw lineage mismatch: {key}")
        schema = (
            ResponseJudgeOutput
            if row["judge_type"] == "response"
            else RiskJudgeOutput
        )
        schema.model_validate(row.get("parsed"))
        usage = require_reported_usage(
            row.get("usage"), stage="persisted development judge schema smoke"
        )
        if usage["prompt_tokens"] > int(plan_row["input_token_upper_bound"]):
            raise RuntimeError("schema smoke reported prompt-token overrun")

    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=JUDGE_SCHEMA_SMOKE_STAGE,
        expected_calls={key: 1 for key in expected_keys},
        maximum_total_attempts=4,
    )
    if ledger.started_attempts != 4 or any(
        not ledger.succeeded(key) for key in expected_keys
    ):
        raise RuntimeError("schema smoke ledger is not exact four-call PASS")
    for key, raw in raw_by_key.items():
        terminal = ledger.terminal_row(key)
        if terminal is None:
            raise RuntimeError("schema smoke success lacks terminal ledger row")
        if (terminal.get("result") or {}).get("parsed") != raw["parsed"]:
            raise RuntimeError("schema smoke ledger/raw parsed output mismatch")
        if require_reported_usage(
            terminal.get("usage"), stage="persisted schema smoke ledger"
        ) != raw["usage"]:
            raise RuntimeError("schema smoke ledger/raw usage mismatch")
        if terminal.get("request_hash") != raw.get("request_hash"):
            raise RuntimeError("schema smoke ledger/raw request hash mismatch")

    summary = read_json(summary_path)
    expected_summary = {
        "status": "PASS",
        "protocol": JUDGE_SCHEMA_SMOKE_PROTOCOL,
        "compatibility_contract_sha256": contract["contract_sha256"],
        "accepted_cost_estimate_sha256": cost_hash,
        "expected_physical_calls": 4,
        "successful_physical_calls": 4,
        "physical_attempts": 4,
        "judge_families": sorted(
            str(row["family"]) for row in contract["judge_endpoints"]
        ),
        "selected_state_id": contract["selected_train_state"]["state_id"],
        "call_plan_sha256": contract["call_plan_sha256"],
        "raw_results_sha256": sha256_file(raw_path),
        "physical_attempt_ledger_sha256": sha256_file(ledger_path),
    }
    if summary != expected_summary:
        raise RuntimeError("schema smoke summary contract mismatch")
    return {
        "status": "PASS",
        "protocol": JUDGE_SCHEMA_SMOKE_PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "accepted_cost_estimate_sha256": cost_hash,
        "summary_sha256": sha256_file(summary_path),
        "attestation_sha256": verification["attestation_sha256"],
        "selected_state_id": contract["selected_train_state"]["state_id"],
        "physical_attempts": 4,
    }
