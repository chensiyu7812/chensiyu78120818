"""Anonymous batched external judging for the PM-v1.5 fast track.

This module deliberately lives outside ``pm_v2_external_eval``.  It reuses the
current PM-v2.2/V1.5 fixed-turn and evaluator-context contracts, but changes
only the final judge transport: all frozen conditions for one seeker turn are
scored in one anonymous, position-balanced request.  Quality, cost, latency,
and the stratified evidence-risk audit remain separate estimands.
"""

from __future__ import annotations

from collections import defaultdict
import inspect
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from .api import (
    Endpoint,
    StructuredOutputValidationError,
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
from .contracts import MemorySource, StrategyMode, canonical_action_id, parse_action_id
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
from .pm_v2_contracts import (
    CompositeSpec,
    ResponseDimensions,
    RiskDimensions,
    StrictModel,
)
from .pm_v2_external_eval import build_external_pointwise_case, load_fixed_turns
from .pm_v2_judging import (
    RESPONSE_SCALE_ANCHORS,
    RISK_SCALE_ANCHORS,
    composite_weights_hash,
    dimension_health,
)
from .pm_v2_model import applicable_risk_fields
from .text import conservative_token_bound, estimate_tokens


PROTOCOL = "pm-v1.5-external-anonymous-batched-v1"
STAGE = "pm_v1_5_external_anonymous_batched"
PILOT_PROTOCOL = "pm-v1.5-external-batched-schema-order-pilot-v1"
PILOT_STAGE = "pm_v1_5_external_batched_schema_order_pilot"
RISK_AUDIT_PROTOCOL = "pm-v1.5-stratified-evidence-risk-audit-v1"
EXPECTED_CANDIDATE_COUNT = 5

RESPONSE_DIMENSIONS = tuple(ResponseDimensions.model_fields)
RISK_DIMENSIONS = tuple(RiskDimensions.model_fields)
EXPECTED_CANDIDATE_IDS = tuple(
    f"C{index}" for index in range(1, EXPECTED_CANDIDATE_COUNT + 1)
)


class BatchedResponseCandidateScore(StrictModel):
    candidate_id: str = Field(pattern=r"^C[1-5]$")
    emotional_support: float = Field(ge=1.0, le=5.0)
    personalization: float = Field(ge=1.0, le=5.0)
    memory_appropriateness: float = Field(ge=1.0, le=5.0)
    factual_grounding: float = Field(ge=1.0, le=5.0)
    temporal_consistency: float = Field(ge=1.0, le=5.0)
    non_intrusiveness: float = Field(ge=1.0, le=5.0)
    rationale: str = Field(min_length=1, max_length=800)


class BatchedResponseJudgment(StrictModel):
    candidates: list[BatchedResponseCandidateScore] = Field(
        min_length=EXPECTED_CANDIDATE_COUNT,
        max_length=EXPECTED_CANDIDATE_COUNT,
    )

    @model_validator(mode="after")
    def require_exact_candidate_ids(self):
        actual = [row.candidate_id for row in self.candidates]
        if len(actual) != len(set(actual)) or set(actual) != set(EXPECTED_CANDIDATE_IDS):
            raise ValueError(
                "batched response candidate IDs must be exactly "
                f"{list(EXPECTED_CANDIDATE_IDS)}; got {actual}"
            )
        return self


class BatchedRiskCandidateScore(StrictModel):
    candidate_id: str = Field(pattern=r"^C[1-5]$")
    selected_context_misuse: float = Field(ge=0.0, le=3.0)
    unnecessary_exposure: float = Field(ge=0.0, le=3.0)
    stale_or_conflicting_use: float = Field(ge=0.0, le=3.0)
    unsupported_personal_claim: float = Field(ge=0.0, le=3.0)
    memory_omission: float = Field(ge=0.0, le=3.0)
    strategy_overuse: float = Field(ge=0.0, le=3.0)
    strategy_omission: float = Field(ge=0.0, le=3.0)
    rationale: str = Field(min_length=1, max_length=800)


class BatchedRiskJudgment(StrictModel):
    candidates: list[BatchedRiskCandidateScore] = Field(
        min_length=EXPECTED_CANDIDATE_COUNT,
        max_length=EXPECTED_CANDIDATE_COUNT,
    )

    @model_validator(mode="after")
    def require_exact_candidate_ids(self):
        actual = [row.candidate_id for row in self.candidates]
        if len(actual) != len(set(actual)) or set(actual) != set(EXPECTED_CANDIDATE_IDS):
            raise ValueError(
                "batched risk candidate IDs must be exactly "
                f"{list(EXPECTED_CANDIDATE_IDS)}; got {actual}"
            )
        return self


QUALITY_SYSTEM = f"""You are an impartial evaluator of emotional-support replies.
Five anonymous candidates answered the same fixed seeker turn. Score every
candidate independently on six dimensions. Do not rank candidates, infer the
generating policy, reward verbosity, or produce an overall score. Candidate
order is systematically position-balanced and carries no meaning. Authorized user context is only
reference material for checking personalization, grounding, and time.

{RESPONSE_SCALE_ANCHORS}"""


RISK_SYSTEM = f"""You are an evidence-risk auditor for emotional-support dialogue.
Five anonymous candidates were generated for the same fixed seeker turn. Each
candidate includes only the context actually selected for its generator and its
reply. Policy names, resource levels, and costs are hidden. Score observed risk,
not resource quantity, and do not rank candidates. Candidate order is systematically position-balanced
and carries no meaning.

{RISK_SCALE_ANCHORS}"""


QUALITY_RULES = {
    "emotional_support": "empathy, validation, attunement, and helpful support",
    "personalization": "specific fit to the current situation and known history; generic warmth alone is insufficient",
    "memory_appropriateness": "use or non-use of history fits the turn; penalize unjustified recall and harmful omission",
    "factual_grounding": "personal claims are supported by authorized context or current dialogue",
    "temporal_consistency": "events and states use the correct order and tense",
    "non_intrusiveness": "no irrelevant private detail, over-personalization, or pressure",
}


RISK_RULES = {
    "selected_context_misuse": "selected evidence is actually misapplied in the reply",
    "unnecessary_exposure": "irrelevant or overly private information is surfaced",
    "stale_or_conflicting_use": "outdated or contradicted information affects the reply",
    "unsupported_personal_claim": "the reply invents personal facts",
    "memory_omission": "clearly material available history is omitted and that omission harms the reply",
    "strategy_overuse": "guidance makes the reply formulaic, premature, overly directive, or repetitive",
    "strategy_omission": "support guidance was clearly needed and its absence harms the reply",
}


def batched_prompt_contract_hash() -> str:
    payload = {
        "protocol": PROTOCOL,
        "quality_system": QUALITY_SYSTEM,
        "risk_system": RISK_SYSTEM,
        "quality_rules": QUALITY_RULES,
        "risk_rules": RISK_RULES,
        "quality_builder": inspect.getsource(build_batched_messages),
        "response_schema": BatchedResponseJudgment.model_json_schema(),
        "risk_schema": BatchedRiskJudgment.model_json_schema(),
    }
    return sha256_text(canonical_json(payload))


def unit_id(unit: Sequence[Any]) -> str:
    return sha256_text(canonical_json(list(unit)))[:24]


def balanced_candidate_order(
    conditions: Sequence[str], *, unit_ordinal: int, order_variant: int
) -> list[str]:
    """Deterministic cyclic order; variant one reverses before rotation."""

    values = [str(value) for value in conditions]
    if len(values) != EXPECTED_CANDIDATE_COUNT or len(set(values)) != len(values):
        raise ValueError(
            f"batched judging requires {EXPECTED_CANDIDATE_COUNT} unique conditions"
        )
    if int(order_variant) % 2:
        values.reverse()
    offset = (int(unit_ordinal) + int(order_variant)) % len(values)
    return values[offset:] + values[:offset]


def _external_action_contract(turn_row: Mapping[str, Any]) -> dict[str, str]:
    raw_requested = str(
        turn_row.get("requested_action_id") or turn_row.get("action_id") or ""
    )
    try:
        parse_action_id(raw_requested)
        requested = raw_requested
    except ValueError:
        requested = ""
    selected_sources: set[MemorySource] = set()
    for item in turn_row.get("selected_memory") or []:
        try:
            selected_sources.add(MemorySource(str(item.get("source") or "")))
        except ValueError:
            continue
    inferred = canonical_action_id(
        selected_sources,
        StrategyMode.RS if bool(turn_row.get("selected_strategy") or []) else StrategyMode.R0,
    )
    raw_realized = str(
        turn_row.get("realized_action_id")
        or turn_row.get("effective_action_id")
        or ""
    )
    try:
        parse_action_id(raw_realized)
        realized = raw_realized
    except ValueError:
        realized = inferred
    if realized != inferred:
        raise RuntimeError("external batched turn realized action mismatches selected evidence")
    raw_effective = str(turn_row.get("effective_action_id") or realized)
    if raw_effective != realized:
        raise RuntimeError("legacy effective action mismatches realized action")
    return {
        "requested_action_id": requested or realized,
        "realized_action_id": realized,
        "effective_action_id": realized,
    }


def _authorized_context_map(evoemo_path: str | Path) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    for user in load_evoemo(evoemo_path):
        for topic in user.get("subsequent_topics") or []:
            key = (str(user["id"]), int(topic["idx"]))
            if key in result:
                raise RuntimeError(f"duplicate EvoEmo evaluator context: {key}")
            result[key] = canonical_json(evaluator_context(user, topic))
    return result


def _case_payload(pointwise_case: Mapping[str, Any]) -> dict[str, Any]:
    state = pointwise_case["state"]
    return {
        "current_seeker_turn": state.current_user_text,
        "recent_dialogue": [
            turn.model_dump(mode="json") for turn in state.current_session_history
        ],
        "current_session_summary": state.current_session_summary or "[none]",
    }


def build_batched_messages(
    *,
    candidate_rows: Mapping[str, Mapping[str, Any]],
    condition_order: Sequence[str],
    authorized_user_context: str,
    judge_type: str,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """Build an anonymous five-candidate quality or risk request."""

    if set(candidate_rows) != set(condition_order):
        raise ValueError("candidate rows and condition order differ")
    pointwise_cases = {
        condition: build_external_pointwise_case(
            candidate_rows[condition],
            authorized_user_context=authorized_user_context,
        )
        for condition in condition_order
    }
    case_values = {
        canonical_json(_case_payload(value)) for value in pointwise_cases.values()
    }
    if len(case_values) != 1:
        raise RuntimeError("batched candidates do not share one fixed seeker context")
    case = _case_payload(pointwise_cases[str(condition_order[0])])
    try:
        authorized = json.loads(authorized_user_context)
    except json.JSONDecodeError as exc:
        raise ValueError("authorized evaluator context is not canonical JSON") from exc

    candidates: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for position, condition in enumerate(condition_order, 1):
        candidate_id = f"C{position}"
        row = candidate_rows[str(condition)]
        action = _external_action_contract(row)
        observed_tokens = int(
            row.get("input_tokens")
            or (row.get("cost") or {}).get("total_input_tokens")
            or 0
        )
        if observed_tokens <= 0:
            raise RuntimeError(
                "batched external candidate lacks positive observed input-token usage"
            )
        mapping[candidate_id] = {
            "condition": str(condition),
            "position": position,
            "observed_input_tokens": observed_tokens,
            **action,
        }
        candidate = {
            "candidate_id": candidate_id,
            "response": str(row["supporter_message"]),
        }
        if judge_type == "risk":
            candidate["selected_context_shown_to_generator"] = (
                pointwise_cases[str(condition)]["selected_context"] or "[none]"
            )
        candidates.append(candidate)

    if judge_type == "quality":
        system = QUALITY_SYSTEM
        task = "score each anonymous response independently"
        rules = QUALITY_RULES
        output_fields = {
            **{name: "number from 1 to 5" for name in RESPONSE_DIMENSIONS},
            "rationale": "brief evidence-based explanation",
        }
    elif judge_type == "risk":
        system = RISK_SYSTEM
        task = "audit observed evidence-use risk for each anonymous response"
        rules = RISK_RULES
        output_fields = {
            **{name: "number from 0 to 3" for name in RISK_DIMENSIONS},
            "rationale": "brief evidence-based explanation",
        }
    else:
        raise ValueError(f"unknown batched judge type: {judge_type}")

    payload = {
        "task": task,
        "case": case,
        "authorized_user_context": authorized,
        "rules": rules,
        "candidates": candidates,
        "output_contract": {
            "candidates": [
                {"candidate_id": "C1", **output_fields}
            ],
            "requirements": [
                "return exactly one row for each of C1 through C7",
                "do not include an overall score",
                "do not rank or compare candidates in rationales",
            ],
        },
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": canonical_json(payload)},
    ]
    return messages, mapping


def select_stratified_units(
    units: Sequence[tuple[str, int, int, str, int]],
    *,
    units_per_user: int,
    seed: int,
) -> list[tuple[str, int, int, str, int]]:
    """Select a deterministic user-complete sample balanced across turn/seed/topic."""

    if units_per_user <= 0:
        raise ValueError("units_per_user must be positive")
    grouped: dict[str, list[tuple[str, int, int, str, int]]] = defaultdict(list)
    for raw in units:
        unit = (str(raw[0]), int(raw[1]), int(raw[2]), str(raw[3]), int(raw[4]))
        grouped[unit[0]].append(unit)
    if not grouped:
        raise ValueError("cannot stratify an empty unit universe")

    selected: list[tuple[str, int, int, str, int]] = []
    for user_id in sorted(grouped):
        candidates = sorted(set(grouped[user_id]))
        if len(candidates) < units_per_user:
            raise ValueError(
                f"user {user_id} has fewer than {units_per_user} external units"
            )
        chosen: list[tuple[str, int, int, str, int]] = []
        # Cover each available evaluation turn before filling the remainder.
        for turn_index in sorted({unit[4] for unit in candidates}):
            if len(chosen) >= units_per_user:
                break
            pool = [unit for unit in candidates if unit[4] == turn_index]
            choice = min(
                pool,
                key=lambda unit: sha256_text(
                    canonical_json([int(seed), user_id, list(unit)])
                ),
            )
            if choice not in chosen:
                chosen.append(choice)
        while len(chosen) < units_per_user:
            remaining = [unit for unit in candidates if unit not in chosen]
            topic_counts = {value: sum(row[1] == value for row in chosen) for value in {u[1] for u in candidates}}
            seed_counts = {value: sum(row[2] == value for row in chosen) for value in {u[2] for u in candidates}}
            turn_counts = {value: sum(row[4] == value for row in chosen) for value in {u[4] for u in candidates}}
            choice = min(
                remaining,
                key=lambda unit: (
                    turn_counts[unit[4]],
                    seed_counts[unit[2]],
                    topic_counts[unit[1]],
                    sha256_text(canonical_json([int(seed), user_id, list(unit)])),
                ),
            )
            chosen.append(choice)
        selected.extend(chosen)
    return sorted(selected)


def _endpoint_contract(endpoint: Endpoint) -> dict[str, Any]:
    payload = {
        "model": endpoint.model,
        "family": endpoint.family,
        "base_url": endpoint.base_url,
    }
    return {**payload, "sha256": sha256_text(canonical_json(payload))}


def _schema_for(judge_type: str):
    if judge_type == "quality":
        return BatchedResponseJudgment
    if judge_type == "risk":
        return BatchedRiskJudgment
    raise ValueError(f"unknown batched judge type: {judge_type}")


def _build_call(
    *,
    stage: str,
    unit: tuple[str, int, int, str, int],
    unit_ordinal: int,
    conditions: Sequence[str],
    matrix: Mapping[tuple[str, int, int, str, int, str], Mapping[str, Any]],
    authorized_user_context: str,
    endpoint: Endpoint,
    role: str,
    judge_type: str,
    order_variant: int,
    call_seed: int,
    max_output_tokens: int,
    pricing: Mapping[str, float],
    input_token_safety_factor: float,
    study_freeze_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family = str(endpoint.family or "")
    if not family:
        raise ValueError("batched endpoint requires a declared family")
    order = balanced_candidate_order(
        conditions, unit_ordinal=unit_ordinal, order_variant=order_variant
    )
    candidate_rows = {condition: matrix[(*unit, condition)] for condition in conditions}
    messages, mapping = build_batched_messages(
        candidate_rows=candidate_rows,
        condition_order=order,
        authorized_user_context=authorized_user_context,
        judge_type=judge_type,
    )
    schema = _schema_for(judge_type)
    request_payload = chat_request_payload(
        endpoint,
        messages,
        temperature=0.0,
        max_tokens=int(max_output_tokens),
        seed=int(call_seed),
        response_schema=schema,
    )
    if not request_payload_has_schema(request_payload):
        raise RuntimeError("batched judge request lacks a structured-output schema")
    payload_text = canonical_json(request_payload)
    base_input_tokens_est = estimate_tokens(payload_text)
    input_tokens_est = conservative_token_bound(
        payload_text, safety_factor=float(input_token_safety_factor)
    )
    prompt_hash = sha256_text(canonical_json(messages))
    record_ids = {
        "unit_id": unit_id(unit),
        "judge_family": family,
        "role": str(role),
        "judge_type": str(judge_type),
        "order_variant": int(order_variant),
    }
    request_payload_sha256 = sha256_text(payload_text)
    call_key = physical_call_key(
        stage=stage,
        record_ids=record_ids,
        prompt_sha256=prompt_hash,
        endpoint=endpoint,
        request_parameters={
            "temperature": 0.0,
            "max_tokens": int(max_output_tokens),
            "seed": int(call_seed),
            "response_schema": schema.__name__,
            "request_payload_sha256": request_payload_sha256,
            "retries": 1,
            "study_freeze_sha256": study_freeze_sha256,
        },
    )
    logical_key = canonical_json(record_ids)
    plan = {
        **record_ids,
        "logical_call_key": logical_key,
        "physical_call_key": call_key,
        "unit": list(unit),
        "unit_ordinal": int(unit_ordinal),
        "judge_model": endpoint.model,
        "candidate_order": order,
        "candidate_mapping": mapping,
        "seed": int(call_seed),
        "input_tokens_est": input_tokens_est,
        "base_input_tokens_est": base_input_tokens_est,
        "max_output_tokens": int(max_output_tokens),
        "max_http_attempts": 1,
        "prompt_hash": prompt_hash,
        "request_payload_sha256": request_payload_sha256,
        "request_payload_includes_schema": True,
        "pricing_usd_per_mtok": {
            "input": float(pricing["input"]),
            "output": float(pricing["output"]),
        },
        "maximum_cost_usd": (
            input_tokens_est / 1_000_000 * float(pricing["input"])
            + int(max_output_tokens) / 1_000_000 * float(pricing["output"])
        ),
    }
    execution = {
        "messages": messages,
        "schema": schema,
        "endpoint_family": family,
    }
    return plan, execution


def _execute_plan(
    *,
    plan: Sequence[Mapping[str, Any]],
    executions: Mapping[str, Mapping[str, Any]],
    endpoints: Sequence[Endpoint],
    out_dir: str | Path,
    stage: str,
    run: bool,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    accept_cost_estimate_sha256: str | None,
    overwrite: bool,
    study_freeze_sha256: str,
    input_token_safety_factor: float,
    fail_on_reported_input_overrun: bool,
    manifest_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute or dry-run an immutable one-attempt batched call plan."""

    if run and overwrite:
        raise RuntimeError("paid batched judge runs prohibit --overwrite")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "judge_call_ledger.jsonl"
    manifest_path = out / "run_manifest.json"
    estimate_path = out / "cost_estimate.json"
    call_plan_path = out / "call_plan.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path, overwrite=overwrite, stage=stage
    )
    if overwrite:
        for path in (manifest_path, estimate_path, call_plan_path):
            if path.exists():
                path.unlink()

    plan_rows = [dict(row) for row in plan]
    logical = {str(row["logical_call_key"]): row for row in plan_rows}
    physical = {str(row["physical_call_key"]): row for row in plan_rows}
    if len(logical) != len(plan_rows) or len(physical) != len(plan_rows):
        raise RuntimeError("batched call plan contains duplicate logical or physical keys")
    if set(logical) != set(executions):
        raise RuntimeError("batched execution payloads differ from the frozen plan")
    endpoint_by_family = {str(endpoint.family or ""): endpoint for endpoint in endpoints}
    if "" in endpoint_by_family or len(endpoint_by_family) != len(endpoints):
        raise ValueError("batched judge endpoints require distinct declared families")
    if {str(row["judge_family"]) for row in plan_rows} - set(endpoint_by_family):
        raise ValueError("batched plan references an unavailable endpoint family")

    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={key: 1 for key in physical},
        maximum_total_attempts=int(max_api_calls),
    )
    successful: dict[str, dict[str, Any]] = {}
    for ledger_row in ledger.event_rows:
        physical_key = str(ledger_row["call_key"])
        row = physical[physical_key]
        logical_key = str(row["logical_call_key"])
        expected_record_ids = {
            "unit_id": row["unit_id"],
            "judge_family": row["judge_family"],
            "role": row["role"],
            "judge_type": row["judge_type"],
            "order_variant": row["order_variant"],
        }
        if (
            ledger_row.get("record_ids") != expected_record_ids
            or ledger_row.get("prompt_sha256") != row["prompt_hash"]
        ):
            raise RuntimeError(f"stale batched ledger provenance: {logical_key}")
        if ledger_row.get("event") != "SUCCEEDED":
            continue
        if logical_key in successful:
            raise RuntimeError(f"duplicate successful batched call: {logical_key}")
        payload = ledger_row.get("result") or {}
        parsed = _schema_for(str(row["judge_type"])).model_validate(
            payload.get("parsed")
        )
        usage_error = reported_prompt_token_error(
            ledger_row.get("usage"),
            maximum_prompt_tokens=int(row["input_tokens_est"]),
            stage="persisted V1.5 batched judge",
            require_positive=bool(fail_on_reported_input_overrun),
        )
        if usage_error is not None:
            raise RuntimeError(
                f"successful batched usage is invalid for {logical_key}: {usage_error}"
            )
        saved_log = payload.get("request_log")
        if not isinstance(saved_log, Mapping) or not ledger_row.get("request_hash"):
            raise RuntimeError(f"successful batched ledger is incomplete: {logical_key}")
        if saved_log.get("normalized_finish_reason") not in {"complete", "tool_call"}:
            raise RuntimeError(
                f"successful batched ledger has an invalid finish reason: {logical_key}"
            )
        successful[logical_key] = {
            **row,
            "parsed": parsed.model_dump(mode="json"),
            "request_hash": str(ledger_row["request_hash"]),
            "request_log": dict(saved_log),
            "usage": dict(ledger_row.get("usage") or {}),
        }

    pending = [
        row
        for row in plan_rows
        if not ledger.succeeded(str(row["physical_call_key"]))
        and not ledger.exhausted(str(row["physical_call_key"]))
    ]
    exhausted = [
        row["logical_call_key"]
        for row in plan_rows
        if str(row["logical_call_key"]) not in successful
        and ledger.exhausted(str(row["physical_call_key"]))
    ]
    if exhausted:
        raise RuntimeError(
            "batched judging has spent unsuccessful one-attempt calls: "
            + str(exhausted[:10])
        )
    historical = ledger.started_attempts
    # The accepted hash authorizes the immutable *full* one-attempt plan.  It
    # must not change merely because a crash-safe resume has persisted some
    # successful calls in the ledger.
    authorization_payload = {
        "stage": stage,
        "full_logical_api_calls": len(plan_rows),
        "maximum_physical_http_attempts": len(plan_rows),
        "total_input_tokens_est": sum(int(row["input_tokens_est"]) for row in plan_rows),
        "max_input_tokens_per_call_est": max(
            [int(row["input_tokens_est"]) for row in plan_rows], default=0
        ),
        "total_output_tokens_est": sum(int(row["max_output_tokens"]) for row in plan_rows),
        "estimated_cost_usd": sum(float(row["maximum_cost_usd"]) for row in plan_rows),
        "input_token_safety_factor": float(input_token_safety_factor),
        "fail_on_reported_input_overrun": bool(fail_on_reported_input_overrun),
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "call_plan_sha256": sha256_text(canonical_json(plan_rows)),
        "study_freeze_sha256": study_freeze_sha256,
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }
    estimate = {
        **authorization_payload,
        "historical_physical_http_attempts": historical,
        "planned_new_api_calls": len(pending),
        "ledger_sha256": sha256_text(canonical_json(ledger.event_rows)),
        "cost_estimate_sha256": sha256_text(canonical_json(authorization_payload)),
    }
    checks = {
        "api_calls": int(estimate["maximum_physical_http_attempts"])
        <= int(max_api_calls),
        "estimated_cost_usd": float(estimate["estimated_cost_usd"])
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": int(estimate["max_input_tokens_per_call_est"])
        <= int(max_input_tokens_per_call),
    }
    budget_gate = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": authorization_payload["budget_limits"],
    }
    manifest = ensure_run_manifest(
        manifest_path,
        {
            **dict(manifest_metadata),
            "stage": stage,
            "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "study_freeze_sha256": study_freeze_sha256,
        },
        overwrite=overwrite,
    )
    dry = {
        "status": "DRY_RUN_COMPLETE" if not run else "STARTING",
        "full_logical_api_calls": len(plan_rows),
        "planned_new_api_calls": len(pending),
        "historical_physical_http_attempts": historical,
        "run_manifest_sha256": manifest["manifest_sha256"],
        "cost_estimate": estimate,
        "budget_gate": budget_gate,
    }
    if budget_gate["status"] != "PASS":
        write_json(estimate_path, {**estimate, "budget_gate": budget_gate})
        write_jsonl(call_plan_path, plan_rows)
        raise RuntimeError("V1.5 batched judge budget gate failed")
    if not run:
        write_json(estimate_path, {**estimate, "budget_gate": budget_gate})
        write_jsonl(call_plan_path, plan_rows)
        return {
            **dry,
            "successful": successful,
            "ledger_path": str(ledger_path),
            "manifest_path": str(manifest_path),
            "estimate_path": str(estimate_path),
            "call_plan_path": str(call_plan_path),
        }
    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("batched API run requires a matching saved dry-run")
    saved_estimate = read_json(estimate_path)
    if saved_estimate.get("cost_estimate_sha256") != estimate["cost_estimate_sha256"]:
        raise RuntimeError("saved batched dry-run is stale")
    if list(iter_jsonl(call_plan_path)) != plan_rows:
        raise RuntimeError("saved batched call plan is stale")
    if accept_cost_estimate_sha256 != estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "batched API run requires the exact accepted dry-run cost hash"
        )

    clients: dict[str, Any] = {}
    try:
        # Construct every required client before reserving the first paid call.
        for family in sorted({str(row["judge_family"]) for row in pending}):
            clients[family] = make_client(endpoint_by_family[family])
        for row in pending:
            logical_key = str(row["logical_call_key"])
            endpoint = endpoint_by_family[str(row["judge_family"])]
            execution = executions[logical_key]
            record_ids = {
                "unit_id": row["unit_id"],
                "judge_family": row["judge_family"],
                "role": row["role"],
                "judge_type": row["judge_type"],
                "order_variant": row["order_variant"],
            }
            reservation = ledger.reserve(
                str(row["physical_call_key"]),
                record_ids=record_ids,
                prompt_sha256=str(row["prompt_hash"]),
            )
            try:
                result, parsed = clients[str(row["judge_family"])].chat(
                    list(execution["messages"]),
                    temperature=0.0,
                    max_tokens=int(row["max_output_tokens"]),
                    seed=int(row["seed"]),
                    response_schema=execution["schema"],
                    retries=1,
                )
                assert parsed is not None
            except StructuredOutputValidationError as exc:
                error = f"{type(exc).__name__}: {str(exc)[:2000]}"
                failed_log = request_log(
                    stage=stage,
                    endpoint=endpoint,
                    messages=list(execution["messages"]),
                    result=exc.call,
                    parsed=None,
                    error=error,
                    prompt_hash=str(row["prompt_hash"]),
                    record_ids=record_ids,
                )
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=exc.call.request_hash,
                    usage=exc.call.usage,
                    error=error,
                    result={
                        "invalid_parsed": exc.parsed_payload,
                        "request_log": failed_log,
                    },
                    metadata={
                        "plan_sha256": sha256_text(canonical_json(row)),
                        "endpoint_sha256": _endpoint_contract(endpoint)["sha256"],
                        "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
                        "study_freeze_sha256": study_freeze_sha256,
                    },
                )
                raise RuntimeError(
                    "batched judge returned a paid but schema-invalid response; "
                    f"one-attempt call is terminal: {logical_key}"
                ) from exc
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:2000]}"
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=None,
                    usage=None,
                    error=error,
                    metadata={
                        "plan_sha256": sha256_text(canonical_json(row)),
                        "endpoint_sha256": _endpoint_contract(endpoint)["sha256"],
                        "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
                        "study_freeze_sha256": study_freeze_sha256,
                    },
                )
                raise RuntimeError(
                    f"batched judge call failed terminally: {logical_key}"
                ) from exc
            parsed_payload = parsed.model_dump(mode="json")
            usage_error = reported_prompt_token_error(
                result.usage,
                maximum_prompt_tokens=int(row["input_tokens_est"]),
                stage="V1.5 batched judge",
                require_positive=bool(fail_on_reported_input_overrun),
            )
            finish_error = (
                None
                if result.normalized_finish_reason in {"complete", "tool_call"}
                else "V1.5 batched judge returned a non-terminal or truncated finish reason: "
                + str(result.normalized_finish_reason)
            )
            terminal_error = usage_error or finish_error
            log_payload = request_log(
                stage=stage,
                endpoint=endpoint,
                messages=list(execution["messages"]),
                result=result,
                parsed=parsed,
                error=terminal_error,
                prompt_hash=str(row["prompt_hash"]),
                record_ids=record_ids,
            )
            ledger.finish(
                reservation,
                succeeded=terminal_error is None,
                request_hash=result.request_hash,
                usage=result.usage,
                error=terminal_error,
                result={"parsed": parsed_payload, "request_log": log_payload},
                metadata={
                    "plan_sha256": sha256_text(canonical_json(row)),
                    "endpoint_sha256": _endpoint_contract(endpoint)["sha256"],
                    "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
                    "study_freeze_sha256": study_freeze_sha256,
                },
            )
            if terminal_error is not None:
                raise RuntimeError(
                    f"batched judge terminal validation failed: {logical_key}: {terminal_error}"
                )
            successful[logical_key] = {
                **row,
                "parsed": parsed_payload,
                "request_hash": result.request_hash,
                "request_log": log_payload,
                "usage": dict(result.usage),
            }
    finally:
        for client in clients.values():
            client.close()
    if len(successful) != len(plan_rows):
        raise RuntimeError("batched judge matrix is incomplete after API execution")
    return {
        **dry,
        "status": "COMPLETE",
        "successful": successful,
        "ledger_rows": ledger.event_rows,
        "ledger_path": str(ledger_path),
        "manifest_path": str(manifest_path),
        "estimate_path": str(estimate_path),
        "call_plan_path": str(call_plan_path),
        "physical_http_attempts": ledger.started_attempts,
    }


def _normalized_unit(raw: Sequence[Any]) -> tuple[str, int, int, str, int]:
    if len(raw) != 5:
        raise ValueError(f"malformed external unit: {raw}")
    return (str(raw[0]), int(raw[1]), int(raw[2]), str(raw[3]), int(raw[4]))


def _pricing_by_family(
    endpoints: Sequence[Endpoint],
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    families = {str(endpoint.family or "") for endpoint in endpoints}
    if "" in families or len(families) != len(endpoints):
        raise ValueError("batched endpoints require distinct declared families")
    pricing = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in pricing_usd_per_mtok.items()
    }
    if set(pricing) != families or any(
        set(values) != {"input", "output"}
        or any(value <= 0.0 for value in values.values())
        for values in pricing.values()
    ):
        raise ValueError(
            "batched pricing must exactly cover endpoint families with positive rates"
        )
    return pricing


def _judge_usage_accounting(
    rows: Sequence[Mapping[str, Any]],
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Keep judge tokens/USD separate from deployment-time resource usage."""

    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0}
    )
    for row in rows:
        family = str(row["judge_family"])
        role = str(row["role"])
        usage = dict(row.get("usage") or {})
        input_tokens = float(usage.get("prompt_tokens") or 0)
        output_tokens = float(usage.get("completion_tokens") or 0)
        if input_tokens <= 0.0 or output_tokens < 0.0:
            raise RuntimeError("completed judge call lacks valid reported usage")
        values = grouped[(role, family)]
        values["calls"] += 1.0
        values["input_tokens"] += input_tokens
        values["output_tokens"] += output_tokens
    rows_out = []
    for (role, family), values in sorted(grouped.items()):
        prices = pricing_usd_per_mtok[family]
        usd = (
            values["input_tokens"] / 1_000_000 * float(prices["input"])
            + values["output_tokens"] / 1_000_000 * float(prices["output"])
        )
        rows_out.append(
            {
                "role": role,
                "judge_family": family,
                "calls": int(values["calls"]),
                "input_tokens": int(values["input_tokens"]),
                "output_tokens": int(values["output_tokens"]),
                "api_cost_usd": float(usd),
            }
        )
    return {
        "protocol": "pm-v1.5-judge-usage-accounting-v1",
        "role": "experiment_cost_not_deployment_inference_cost",
        "rows": rows_out,
        "total_calls": sum(row["calls"] for row in rows_out),
        "total_input_tokens": sum(row["input_tokens"] for row in rows_out),
        "total_output_tokens": sum(row["output_tokens"] for row in rows_out),
        "total_api_cost_usd": float(sum(row["api_cost_usd"] for row in rows_out)),
    }


def run_v1_5_batched_schema_order_pilot(
    *,
    evoemo_path: str | Path,
    study_freeze_path: str | Path,
    study_freeze_sha256: str,
    turn_paths: Sequence[str | Path],
    generation_attestation_paths: Sequence[str | Path],
    full_expected_units: Sequence[tuple[str, int, int, str, int]],
    pilot_units: Sequence[tuple[str, int, int, str, int]],
    conditions: Sequence[str],
    endpoints: Sequence[Endpoint],
    contract: Mapping[str, Any],
    out_dir: str | Path,
    run: bool,
    accept_cost_estimate_sha256: str | None,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the stratified multi-unit schema/order-sensitivity pilot."""

    frozen = dict(contract)
    if (
        frozen.get("protocol") != PILOT_PROTOCOL
        or frozen.get("batched_prompt_contract_sha256")
        != batched_prompt_contract_hash()
        or int(frozen.get("candidate_count") or 0) != EXPECTED_CANDIDATE_COUNT
        or not bool(frozen.get("required_before_full_external_client_creation"))
    ):
        raise RuntimeError("study freeze lacks the current batched schema pilot")
    frozen_units = [
        _normalized_unit(value) for value in (frozen.get("units") or [])
    ]
    normalized_pilot_units = sorted(_normalized_unit(value) for value in pilot_units)
    if frozen_units != normalized_pilot_units or not frozen_units:
        raise RuntimeError("batched schema pilot units differ from the freeze")
    order_variants = [int(value) for value in frozen.get("order_variants") or []]
    if order_variants != [0, 1]:
        raise RuntimeError("batched schema pilot must freeze order variants [0, 1]")
    if list(frozen.get("judge_types") or []) != ["quality", "risk"]:
        raise RuntimeError("batched schema pilot must exercise both schemas")
    if len(endpoints) != 2:
        raise RuntimeError("batched schema pilot requires two judge families")
    pricing = _pricing_by_family(
        endpoints, frozen["judge_pricing_usd_per_mtok"]
    )
    api_planning = dict(frozen["api_cost_planning"])
    safety_factor = float(api_planning["input_token_safety_factor"])
    fail_on_overrun = bool(api_planning["fail_on_reported_input_overrun"])
    if safety_factor < 1.0 or not fail_on_overrun:
        raise RuntimeError("batched schema pilot requires fail-closed cost planning")
    full_units = sorted(_normalized_unit(unit) for unit in full_expected_units)
    if any(unit not in full_units for unit in normalized_pilot_units):
        raise RuntimeError("batched schema pilot unit is outside the frozen universe")
    matrix = load_fixed_turns(
        turn_paths,
        conditions=conditions,
        turn_indices=sorted({unit[4] for unit in full_units}),
        expected_units=full_units,
    )
    authorized = _authorized_context_map(evoemo_path)
    endpoint_families = [str(endpoint.family) for endpoint in endpoints]
    frozen_families = [
        str(row["family"]) for row in frozen.get("judge_endpoints") or []
    ]
    if endpoint_families != frozen_families:
        raise RuntimeError("batched schema pilot endpoints differ from the freeze")

    plan: list[dict[str, Any]] = []
    executions: dict[str, dict[str, Any]] = {}
    output_bounds = {
        "quality": int(frozen["estimated_quality_output_tokens"]),
        "risk": int(frozen["estimated_risk_output_tokens"]),
    }
    base_seed = int(frozen["judge_seed"])
    for pilot_index, pilot_unit in enumerate(normalized_pilot_units):
        unit_ordinal = full_units.index(pilot_unit)
        for family_index, endpoint in enumerate(endpoints):
            family = str(endpoint.family)
            for type_index, judge_type in enumerate(("quality", "risk")):
                for order_variant in order_variants:
                    row, execution = _build_call(
                        stage=PILOT_STAGE,
                        unit=pilot_unit,
                        unit_ordinal=unit_ordinal,
                        conditions=conditions,
                        matrix=matrix,
                        authorized_user_context=authorized[
                            (pilot_unit[0], pilot_unit[1])
                        ],
                        endpoint=endpoint,
                        role="schema_order_pilot",
                        judge_type=judge_type,
                        order_variant=order_variant,
                        call_seed=(
                            base_seed
                            + pilot_index * 1000
                            + family_index * 100
                            + type_index * 10
                            + order_variant
                        ),
                        max_output_tokens=output_bounds[judge_type],
                        pricing=pricing[family],
                        input_token_safety_factor=safety_factor,
                        study_freeze_sha256=study_freeze_sha256,
                    )
                    plan.append(row)
                    executions[str(row["logical_call_key"])] = execution
    if len(plan) != int(frozen.get("expected_calls") or 0):
        raise RuntimeError("batched schema pilot call count differs from the freeze")

    out = Path(out_dir)
    raw_path = out / "raw_judge_calls.jsonl"
    summary_path = out / "summary.json"
    attestation_path = out / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        out / "judge_call_ledger.jsonl",
        overwrite=overwrite,
        stage=PILOT_STAGE,
    )
    if overwrite:
        for path in (raw_path, summary_path, attestation_path):
            if path.exists():
                path.unlink()
    execution_result = _execute_plan(
        plan=plan,
        executions=executions,
        endpoints=endpoints,
        out_dir=out,
        stage=PILOT_STAGE,
        run=run,
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
        accept_cost_estimate_sha256=accept_cost_estimate_sha256,
        overwrite=overwrite,
        study_freeze_sha256=study_freeze_sha256,
        input_token_safety_factor=safety_factor,
        fail_on_reported_input_overrun=fail_on_overrun,
        manifest_metadata={
            "protocol": PILOT_PROTOCOL,
            "evoemo_sha256": sha256_file(evoemo_path),
            "study_freeze_file_sha256": sha256_file(study_freeze_path),
            "turn_sha256": {
                str(Path(path).resolve()): sha256_file(path) for path in turn_paths
            },
            "generation_attestation_sha256": {
                str(Path(path).resolve()): sha256_file(path)
                for path in generation_attestation_paths
            },
            "contract_sha256": sha256_text(canonical_json(frozen)),
            "pilot_units": [list(unit) for unit in normalized_pilot_units],
            "conditions": list(conditions),
            "endpoint_contracts": [_endpoint_contract(endpoint) for endpoint in endpoints],
        },
    )
    dry = {
        **{key: value for key, value in execution_result.items() if key != "successful"},
        "protocol": PILOT_PROTOCOL,
        "expected_calls": len(plan),
        "pilot_units": [list(unit) for unit in normalized_pilot_units],
        "pilot_unit_ids": [unit_id(unit) for unit in normalized_pilot_units],
        "conditions": list(conditions),
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "judge_types": ["quality", "risk"],
        "order_variants": order_variants,
        "batched_prompt_contract_sha256": batched_prompt_contract_hash(),
    }
    if not run:
        return dry

    successful = execution_result["successful"]
    raw_rows = [successful[key] for key in sorted(successful)]
    write_jsonl(raw_path, raw_rows)
    judge_usage_accounting = _judge_usage_accounting(
        raw_rows, pricing
    )
    order_values: dict[tuple[str, str, str, str, int], float] = {}
    for row in raw_rows:
        parsed_by_id = {
            str(score["candidate_id"]): score
            for score in row["parsed"]["candidates"]
        }
        for candidate_id, info in row["candidate_mapping"].items():
            score = parsed_by_id[candidate_id]
            if row["judge_type"] == "quality":
                value = CompositeSpec().score(
                    ResponseDimensions(
                        **{name: float(score[name]) for name in RESPONSE_DIMENSIONS}
                    )
                )
            else:
                value = float(np.mean([float(score[name]) for name in RISK_DIMENSIONS])) / 3.0
            order_values[
                (
                    str(row["unit_id"]),
                    str(row["judge_family"]),
                    str(row["judge_type"]),
                    str(info["condition"]),
                    int(row["order_variant"]),
                )
            ] = value
    order_deltas = [
        abs(order_values[(*key, 0)] - order_values[(*key, 1)])
        for key in sorted({key[:4] for key in order_values})
    ]
    mean_order_delta = float(np.mean(order_deltas))
    maximum_order_delta = max(order_deltas)
    mean_threshold = float(frozen["maximum_mean_absolute_order_delta"])
    maximum_threshold = float(frozen["maximum_single_absolute_order_delta"])
    order_checks = {
        "mean_absolute_order_delta": mean_order_delta <= mean_threshold,
        "maximum_absolute_order_delta": maximum_order_delta <= maximum_threshold,
    }
    summary = {
        **dry,
        "status": "PASS" if all(order_checks.values()) else "FAIL",
        "execution_status": "COMPLETE",
        "study_freeze_sha256": study_freeze_sha256,
        "contract_sha256": sha256_text(canonical_json(frozen)),
        "completed_calls": len(raw_rows),
        "schema_success_rate": 1.0,
        "order_diagnostic": {
            "role": "transport_diagnostic_not_efficacy",
            "paired_candidate_scores": len(order_deltas),
            "mean_absolute_order_delta": mean_order_delta,
            "maximum_absolute_order_delta": maximum_order_delta,
            "thresholds": {
                "maximum_mean_absolute_order_delta": mean_threshold,
                "maximum_single_absolute_order_delta": maximum_threshold,
            },
            "checks": order_checks,
            "status": "PASS" if all(order_checks.values()) else "FAIL",
        },
        "judge_usage_accounting": judge_usage_accounting,
        "raw_path": str(raw_path),
        "ledger_path": execution_result["ledger_path"],
        "physical_http_attempts": execution_result["physical_http_attempts"],
    }
    write_json(summary_path, summary)
    create_artifact_attestation(
        attestation_path,
        stage=PILOT_STAGE,
        inputs={
            "study_freeze": study_freeze_path,
            "evoemo": evoemo_path,
            **{f"turn_{index}": path for index, path in enumerate(turn_paths)},
            **{
                f"generation_attestation_{index}": path
                for index, path in enumerate(generation_attestation_paths)
            },
            "run_manifest": execution_result["manifest_path"],
            "cost_estimate": execution_result["estimate_path"],
        },
        outputs={
            "raw_calls": (raw_path, True),
            "call_ledger": (execution_result["ledger_path"], True),
            "summary": (summary_path, False),
        },
        parameters={"contract": frozen},
        expected={"calls": len(plan), "status": summary["status"]},
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary


def require_v1_5_batched_schema_order_pilot_pass(
    summary_path: str | Path,
    attestation_path: str | Path,
    *,
    study_freeze_sha256: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    verification = require_artifact_attestation(
        attestation_path,
        required_stage=PILOT_STAGE,
        required_output_paths={"summary": summary_path},
        expected_freeze_sha256=study_freeze_sha256,
    )
    summary = read_json(summary_path)
    expected_contract_sha = sha256_text(canonical_json(dict(contract)))
    if (
        summary.get("status") != "PASS"
        or summary.get("execution_status") != "COMPLETE"
        or summary.get("protocol") != PILOT_PROTOCOL
        or summary.get("study_freeze_sha256") != study_freeze_sha256
        or summary.get("contract_sha256") != expected_contract_sha
        or summary.get("batched_prompt_contract_sha256")
        != batched_prompt_contract_hash()
        or int(summary.get("completed_calls") or 0)
        != int(contract.get("expected_calls") or 0)
        or float(summary.get("schema_success_rate") or 0.0) != 1.0
        or (summary.get("order_diagnostic") or {}).get("status") != "PASS"
        or not all(
            bool(value)
            for value in (summary.get("order_diagnostic") or {}).get("checks", {}).values()
        )
    ):
        raise RuntimeError("batched schema/order pilot is stale, incomplete, or failed")
    return {
        "status": "PASS",
        "summary_sha256": sha256_file(summary_path),
        "attestation_sha256": verification["attestation_sha256"],
        "contract_sha256": expected_contract_sha,
        "pilot_units": summary["pilot_units"],
        "completed_calls": int(summary["completed_calls"]),
    }


def build_v1_5_batched_evaluation_plan(
    *,
    matrix: Mapping[tuple[str, int, int, str, int, str], Mapping[str, Any]],
    authorized_contexts: Mapping[tuple[str, int], str],
    units: Sequence[tuple[str, int, int, str, int]],
    conditions: Sequence[str],
    endpoints: Sequence[Endpoint],
    contract: Mapping[str, Any],
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
    api_cost_planning: Mapping[str, Any],
    study_freeze_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Build the frozen 192 + 54 quality and 36x2 risk call plan."""

    frozen = dict(contract)
    if (
        frozen.get("protocol") != PROTOCOL
        or frozen.get("batched_prompt_contract_sha256")
        != batched_prompt_contract_hash()
        or int(frozen.get("candidate_count") or 0) != EXPECTED_CANDIDATE_COUNT
        or frozen.get("conditions_sha256")
        != sha256_text(canonical_json(list(conditions)))
        or bool(frozen.get("llm_overall_requested", True))
    ):
        raise RuntimeError("frozen V1.5 batched evaluation contract is stale")
    gate_e = dict(frozen.get("gate_e") or {})
    if (
        gate_e.get("comparator") not in conditions
        or gate_e.get("observed_gate_m") != "PASS"
        or gate_e.get("observed_gate_f") != "PASS"
        or gate_e.get("requires_gate_m") != "PASS"
        or gate_e.get("requires_gate_f") != "PASS"
    ):
        raise RuntimeError("full external scoring is blocked by Gate M/F lineage")
    units = sorted(_normalized_unit(unit) for unit in units)
    if int(frozen.get("expected_scoring_units") or 0) != len(units):
        raise RuntimeError("frozen batched scoring-unit count is stale")
    pricing = _pricing_by_family(endpoints, pricing_usd_per_mtok)
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    api_planning = dict(api_cost_planning)
    if set(api_planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise RuntimeError("batched API cost-planning contract is incomplete")
    safety_factor = float(api_planning["input_token_safety_factor"])
    if safety_factor < 1.0 or not bool(
        api_planning["fail_on_reported_input_overrun"]
    ):
        raise RuntimeError("batched judging requires fail-closed token accounting")

    quality = dict(frozen.get("quality") or {})
    primary_family = str(quality.get("primary_judge_family") or "")
    sensitivity_family = str(quality.get("sensitivity_judge_family") or "")
    if (
        primary_family == sensitivity_family
        or {primary_family, sensitivity_family} != set(endpoint_by_family)
        or quality.get("primary_scope") != "all_scoring_units"
        or quality.get("sensitivity_role") != "sensitivity_only_not_pooled"
        or int(quality.get("primary_judge_count_per_unit") or 0) != 1
        or bool(quality.get("full_two_family_score_pooling"))
    ):
        raise RuntimeError("frozen batched quality endpoint roles are invalid")
    if int(quality.get("expected_primary_units") or 0) != len(units):
        raise RuntimeError("frozen primary quality unit count is stale")
    sensitivity_units = sorted(
        _normalized_unit(value) for value in quality.get("sensitivity_units") or []
    )
    expected_sensitivity = select_stratified_units(
        units,
        units_per_user=int(quality["sensitivity_units_per_user"]),
        seed=int(quality["sensitivity_selection_seed"]),
    )
    if (
        sensitivity_units != expected_sensitivity
        or quality.get("sensitivity_units_sha256")
        != sha256_text(canonical_json([list(unit) for unit in sensitivity_units]))
        or int(quality.get("expected_sensitivity_units") or 0)
        != len(sensitivity_units)
    ):
        raise RuntimeError("frozen Claude sensitivity sample is stale")

    risk = dict(frozen.get("risk_audit") or {})
    if (
        risk.get("protocol") != RISK_AUDIT_PROTOCOL
        or risk.get("role")
        != "preregistered_stratified_audit_not_population_safety_claim"
    ):
        raise RuntimeError("frozen stratified evidence-risk audit is missing")
    risk_families = [str(value) for value in risk.get("judge_families") or []]
    if set(risk_families) != set(endpoint_by_family) or len(risk_families) != len(
        endpoint_by_family
    ):
        raise RuntimeError("risk audit must retain both independent judge families")
    risk_units = sorted(
        _normalized_unit(value) for value in risk.get("units") or []
    )
    expected_risk = select_stratified_units(
        units,
        units_per_user=int(risk["units_per_user"]),
        seed=int(risk["selection_seed"]),
    )
    if (
        risk_units != expected_risk
        or risk.get("units_sha256")
        != sha256_text(canonical_json([list(unit) for unit in risk_units]))
        or int(risk.get("expected_units") or 0) != len(risk_units)
    ):
        raise RuntimeError("frozen stratified risk sample is stale")

    primary_order_variant = int(quality.get("primary_order_variant") or 0)
    sensitivity_order_variant = int(quality.get("sensitivity_order_variant") or 0)
    risk_order_variant = int(risk.get("order_variant") or 0)
    if {primary_order_variant, sensitivity_order_variant, risk_order_variant} != {0}:
        raise RuntimeError("full batched scoring must use frozen cyclic order variant 0")
    quality_output_tokens = int(quality["estimated_output_tokens_per_call"])
    risk_output_tokens = int(risk["estimated_output_tokens_per_call"])
    if quality_output_tokens <= 0 or risk_output_tokens <= 0:
        raise RuntimeError("batched output-token bounds must be positive")
    base_seed = int(frozen["judge_seed"])
    ordinal_by_unit = {unit: index for index, unit in enumerate(units)}

    tasks: list[tuple[tuple[str, int, int, str, int], str, str, str, int, int]] = []
    for unit in units:
        tasks.append(
            (unit, primary_family, "quality_primary", "quality", primary_order_variant, 0)
        )
    for unit in sensitivity_units:
        tasks.append(
            (
                unit,
                sensitivity_family,
                "quality_sensitivity",
                "quality",
                sensitivity_order_variant,
                1,
            )
        )
    for unit in risk_units:
        for family_index, family in enumerate(risk_families):
            tasks.append(
                (unit, family, "risk_audit", "risk", risk_order_variant, 10 + family_index)
            )

    plan: list[dict[str, Any]] = []
    executions: dict[str, dict[str, Any]] = {}
    for task_index, (unit, family, role, judge_type, order_variant, seed_offset) in enumerate(tasks):
        endpoint = endpoint_by_family[family]
        output_tokens = quality_output_tokens if judge_type == "quality" else risk_output_tokens
        row, execution = _build_call(
            stage=STAGE,
            unit=unit,
            unit_ordinal=ordinal_by_unit[unit],
            conditions=conditions,
            matrix=matrix,
            authorized_user_context=authorized_contexts[(unit[0], unit[1])],
            endpoint=endpoint,
            role=role,
            judge_type=judge_type,
            order_variant=order_variant,
            call_seed=base_seed + ordinal_by_unit[unit] * 100 + seed_offset,
            max_output_tokens=output_tokens,
            pricing=pricing[family],
            input_token_safety_factor=safety_factor,
            study_freeze_sha256=study_freeze_sha256,
        )
        plan.append(row)
        executions[str(row["logical_call_key"])] = execution

    expected_counts = {
        "quality_primary": len(units),
        "quality_sensitivity": len(sensitivity_units),
        "risk_audit": len(risk_units) * len(risk_families),
    }
    actual_counts = {
        role: sum(row["role"] == role for row in plan) for role in expected_counts
    }
    if actual_counts != expected_counts:
        raise RuntimeError("batched plan role counts are incomplete")
    if int(frozen.get("expected_api_calls") or 0) != len(plan):
        raise RuntimeError("frozen batched API call count is stale")
    samples = {
        "primary_units": units,
        "sensitivity_units": sensitivity_units,
        "risk_units": risk_units,
        "role_call_counts": expected_counts,
    }
    return plan, executions, samples


def _bootstrap_cluster_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    cluster_field: str,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot bootstrap an empty paired table")
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_field])].append(float(row["delta"]))
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(n_resamples), dtype=float)
    for index in range(int(n_resamples)):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        values = [value for cluster in selected for value in clusters[str(cluster)]]
        samples[index] = float(np.mean(values))
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "estimate": float(np.mean([float(row["delta"]) for row in rows])),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "p_delta_lt_0": float(np.mean(samples < 0.0)),
        "n_clusters": len(cluster_ids),
        "n_resamples": int(n_resamples),
        "confidence_level": float(confidence_level),
    }


def _paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    treatment: str,
    metrics: Sequence[str],
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = str(row["condition"])
        uid = str(row["unit_id"])
        if uid in by_condition[condition]:
            raise RuntimeError(f"duplicate score row for {condition}/{uid}")
        by_condition[condition][uid] = row
    treatment_rows = by_condition[treatment]
    result: dict[str, Any] = {}
    for baseline in conditions:
        if baseline == treatment:
            continue
        baseline_rows = by_condition[str(baseline)]
        if set(treatment_rows) != set(baseline_rows):
            raise RuntimeError(f"paired score universe differs for {baseline}")
        metric_results: dict[str, Any] = {}
        for metric_index, metric in enumerate(metrics):
            paired = [
                {
                    "unit_id": uid,
                    "user_cluster": str(treatment_rows[uid]["user_cluster"]),
                    "scenario_cluster": str(treatment_rows[uid]["scenario_cluster"]),
                    "delta": float(treatment_rows[uid][metric])
                    - float(baseline_rows[uid][metric]),
                }
                for uid in sorted(treatment_rows)
            ]
            user_ci = _bootstrap_cluster_delta(
                paired,
                cluster_field="user_cluster",
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed + metric_index * 2 + 1,
            )
            scenario_ci = _bootstrap_cluster_delta(
                paired,
                cluster_field="scenario_cluster",
                n_resamples=n_resamples,
                confidence_level=confidence_level,
                seed=seed + metric_index * 2,
            )
            metric_results[str(metric)] = {
                "n": len(paired),
                "primary_cluster": "user_id",
                "primary_cluster_ci": user_ci,
                "user_cluster_ci": user_ci,
                "sensitivity_cluster": "scenario",
                "sensitivity_cluster_ci": scenario_ci,
                "scenario_cluster_ci": scenario_ci,
                "wins": sum(row["delta"] > 0 for row in paired),
                "ties": sum(row["delta"] == 0 for row in paired),
                "losses": sum(row["delta"] < 0 for row in paired),
            }
        result[str(baseline)] = metric_results
    return result


def build_external_gate_e(
    *,
    quality_comparisons: Mapping[str, Any],
    risk_comparisons: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen external efficiency gate, PM minus comparator."""

    frozen = dict(contract)
    comparator = str(frozen.get("comparator") or "")
    thresholds = dict(frozen.get("thresholds") or {})
    expected_thresholds = {
        "minimum_quality_delta",
        "maximum_risk_delta",
        "maximum_generator_input_token_delta",
    }
    lineage_hashes = (
        "training_report_sha256",
        "candidate_manifest_sha256",
        "internal_consumption_ledger_sha256",
    )
    if (
        not comparator
        or set(thresholds) != expected_thresholds
        or frozen.get("requires_gate_m") != "PASS"
        or frozen.get("requires_gate_f") != "PASS"
        or any(len(str(frozen.get(key) or "")) != 64 for key in lineage_hashes)
    ):
        raise RuntimeError("external Gate E contract is incomplete")
    try:
        quality_ci = dict(
            quality_comparisons[comparator]["quality_composite"][
                "primary_cluster_ci"
            ]
        )
        token_ci = dict(
            quality_comparisons[comparator]["observed_input_tokens"][
                "primary_cluster_ci"
            ]
        )
        risk_ci = dict(
            risk_comparisons[comparator]["risk_composite"][
                "primary_cluster_ci"
            ]
        )
    except (KeyError, TypeError) as exc:
        raise RuntimeError("external Gate E lacks its paired comparator CIs") from exc
    checks = {
        "internal_gate_m_passed": frozen.get("observed_gate_m") == "PASS",
        "internal_gate_f_passed": frozen.get("observed_gate_f") == "PASS",
        "lineage_complete": all(
            len(str(frozen.get(key) or "")) == 64 for key in lineage_hashes
        ),
        "quality_noninferior": float(quality_ci["lower"])
        >= float(thresholds["minimum_quality_delta"]),
        "evidence_risk_nonincrease": float(risk_ci["upper"])
        <= float(thresholds["maximum_risk_delta"]),
        "generator_input_tokens_strictly_lower": float(token_ci["upper"])
        < float(thresholds["maximum_generator_input_token_delta"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "NOT_SUPPORTED",
        "delta_direction": f"PM_minus_{comparator}",
        "comparator": comparator,
        "thresholds": thresholds,
        "checks": checks,
        "quality_composite_ci": quality_ci,
        "evidence_risk_ci": risk_ci,
        "observed_generator_input_tokens_ci": token_ci,
        "claim_boundary": (
            "quality-preserving generator-input efficiency only; not total USD, "
            "latency, clinical efficacy, or superiority to same-budget fixed"
        ),
    }


def _candidate_by_id(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = {
        str(value["candidate_id"]): value
        for value in (row.get("parsed") or {}).get("candidates") or []
    }
    if set(values) != set(EXPECTED_CANDIDATE_IDS):
        raise RuntimeError("persisted batched judgment has incomplete candidate IDs")
    return values


def _quality_score_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    composite_spec: CompositeSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if raw.get("role") != role:
            continue
        unit = _normalized_unit(raw["unit"])
        parsed = _candidate_by_id(raw)
        mapping = raw.get("candidate_mapping") or {}
        if set(mapping) != set(EXPECTED_CANDIDATE_IDS):
            raise RuntimeError("batched quality mapping is incomplete")
        for candidate_id in EXPECTED_CANDIDATE_IDS:
            score = parsed[candidate_id]
            info = mapping[candidate_id]
            dimensions = ResponseDimensions(
                **{name: float(score[name]) for name in RESPONSE_DIMENSIONS}
            )
            rows.append(
                {
                    "unit_id": str(raw["unit_id"]),
                    "user_id": unit[0],
                    "topic_index": unit[1],
                    "seed": unit[2],
                    "simulator_id": unit[3],
                    "turn_index": unit[4],
                    "user_cluster": unit[0],
                    "scenario_cluster": f"{unit[0]}::{unit[1]}",
                    "condition": str(info["condition"]),
                    "candidate_id": candidate_id,
                    "position": int(info["position"]),
                    "judge_family": str(raw["judge_family"]),
                    "judge_model": str(raw["judge_model"]),
                    **dimensions.model_dump(),
                    "quality_composite": composite_spec.score(dimensions),
                    "observed_input_tokens": int(info["observed_input_tokens"]),
                    "requested_action_id": str(info["requested_action_id"]),
                    "realized_action_id": str(info["realized_action_id"]),
                    "effective_action_id": str(info["effective_action_id"]),
                    "rationale": str(score["rationale"]),
                }
            )
    return rows


def _risk_score_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    expected_families: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        if raw.get("role") != "risk_audit":
            continue
        unit = _normalized_unit(raw["unit"])
        parsed = _candidate_by_id(raw)
        mapping = raw.get("candidate_mapping") or {}
        for candidate_id in EXPECTED_CANDIDATE_IDS:
            score = parsed[candidate_id]
            info = mapping[candidate_id]
            grouped[(str(raw["unit_id"]), str(info["condition"]))].append(
                {
                    "unit": unit,
                    "family": str(raw["judge_family"]),
                    "observed_input_tokens": int(info["observed_input_tokens"]),
                    "requested_action_id": str(info["requested_action_id"]),
                    "realized_action_id": str(info["realized_action_id"]),
                    "effective_action_id": str(info["effective_action_id"]),
                    "scores": {name: float(score[name]) for name in RISK_DIMENSIONS},
                    "rationale": str(score["rationale"]),
                }
            )
    rows: list[dict[str, Any]] = []
    expected = set(str(value) for value in expected_families)
    for (uid, condition), values in sorted(grouped.items()):
        families = {value["family"] for value in values}
        if families != expected or len(values) != len(expected):
            raise RuntimeError(f"risk audit judge families are incomplete for {uid}/{condition}")
        unit_values = {value["unit"] for value in values}
        action_values = {
            (
                value["requested_action_id"],
                value["realized_action_id"],
                value["effective_action_id"],
            )
            for value in values
        }
        token_values = {value["observed_input_tokens"] for value in values}
        if len(unit_values) != 1 or len(action_values) != 1 or len(token_values) != 1:
            raise RuntimeError("risk audit family rows disagree on hidden candidate metadata")
        unit = next(iter(unit_values))
        requested, realized, effective = next(iter(action_values))
        risks = {
            name: float(median(value["scores"][name] for value in values))
            for name in RISK_DIMENSIONS
        }
        mad = {
            name: float(
                median(
                    abs(value["scores"][name] - risks[name]) for value in values
                )
            )
            for name in RISK_DIMENSIONS
        }
        applicable = applicable_risk_fields(requested)
        rows.append(
            {
                "unit_id": uid,
                "user_id": unit[0],
                "topic_index": unit[1],
                "seed": unit[2],
                "simulator_id": unit[3],
                "turn_index": unit[4],
                "user_cluster": unit[0],
                "scenario_cluster": f"{unit[0]}::{unit[1]}",
                "condition": condition,
                **risks,
                "risk_composite": max(risks[name] for name in applicable) / 3.0,
                "risk_composite_all_dimensions_diagnostic": max(risks.values()) / 3.0,
                "risk_dimension_mad": mad,
                "max_risk_dimension_mad": max(mad.values()),
                "observed_input_tokens": next(iter(token_values)),
                "requested_action_id": requested,
                "realized_action_id": realized,
                "effective_action_id": effective,
                "applicable_risk_fields": list(applicable),
                "judge_families": sorted(families),
                "rationales": {
                    value["family"]: value["rationale"] for value in values
                },
            }
        )
    return rows


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    return {
        condition: {
            "n": len(values),
            **{
                metric: float(np.mean([float(row[metric]) for row in values]))
                for metric in metrics
            },
        }
        for condition, values in sorted(grouped.items())
    }


def _position_balance_report(
    rows: Sequence[Mapping[str, Any]], conditions: Sequence[str]
) -> dict[str, Any]:
    counts = {
        condition: {
            str(position): sum(
                row["condition"] == condition and int(row["position"]) == position
                for row in rows
            )
            for position in range(1, EXPECTED_CANDIDATE_COUNT + 1)
        }
        for condition in conditions
    }
    spreads = {
        condition: max(values.values()) - min(values.values())
        for condition, values in counts.items()
    }
    report = {
        "status": "PASS" if max(spreads.values(), default=0) <= 1 else "FAIL",
        "position_counts": counts,
        "maximum_within_condition_count_spread": max(spreads.values(), default=0),
        "required_maximum_spread": 1,
    }
    if report["status"] != "PASS":
        raise RuntimeError("cyclic batched candidate positions are not balanced")
    return report


def _dimension_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    dimensions: Sequence[str],
    prefix: str,
    duplicate_exact_match_rate: float,
    maximum_absolute_dimension_correlation: float,
    reject_constants: bool,
) -> dict[str, Any]:
    matrix = np.asarray(
        [[float(row[name]) for name in dimensions] for row in rows], dtype=float
    )
    duplicates, correlations, constants, prevalence = dimension_health(
        matrix,
        dimensions,
        prefix=prefix,
        duplicate_exact_match_rate=duplicate_exact_match_rate,
        maximum_absolute_dimension_correlation=maximum_absolute_dimension_correlation,
    )
    failures = [*duplicates, *correlations, *(constants if reject_constants else [])]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "duplicate_dimension_pairs": duplicates,
        "high_correlation_dimension_pairs": correlations,
        "constant_dimensions": constants,
        "dimension_prevalence": prevalence,
    }
    if failures:
        raise RuntimeError(
            f"V1.5 batched {prefix} dimension-health gate failed: "
            + canonical_json(report)
        )
    return report


def _rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and array[order[end]] == array[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        ranks[order[index:end]] = rank
        index = end
    return ranks


def _safe_correlation(left: Sequence[float], right: Sequence[float], *, rank: bool) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = _rank(left) if rank else np.asarray(left, dtype=float)
    y = _rank(right) if rank else np.asarray(right, dtype=float)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _quality_sensitivity_report(
    primary_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    comparator: str,
) -> dict[str, Any]:
    primary = {
        (str(row["unit_id"]), str(row["condition"])): row for row in primary_rows
    }
    sensitivity = {
        (str(row["unit_id"]), str(row["condition"])): row
        for row in sensitivity_rows
    }
    if set(sensitivity) - set(primary):
        raise RuntimeError("quality sensitivity rows are outside the primary matrix")
    primary_families = {str(row["judge_family"]) for row in primary_rows}
    sensitivity_families = {str(row["judge_family"]) for row in sensitivity_rows}
    if len(primary_families) != 1 or len(sensitivity_families) != 1:
        raise RuntimeError("quality primary/sensitivity roles do not each use one family")
    keys = sorted(sensitivity)
    primary_values = [float(primary[key]["quality_composite"]) for key in keys]
    sensitivity_values = [float(sensitivity[key]["quality_composite"]) for key in keys]

    def mean_delta(index: Mapping[tuple[str, str], Mapping[str, Any]]) -> float:
        unit_ids = sorted({key[0] for key in sensitivity})
        return float(
            np.mean(
                [
                    float(index[(uid, treatment)]["quality_composite"])
                    - float(index[(uid, comparator)]["quality_composite"])
                    for uid in unit_ids
                ]
            )
        )

    primary_delta = mean_delta(primary)
    sensitivity_delta = mean_delta(sensitivity)
    return {
        "role": "sensitivity_only_not_pooled",
        "interpretation": "not_pooled_into_primary_estimate",
        "primary_judge_family": next(iter(primary_families)),
        "judge_family": next(iter(sensitivity_families)),
        "n_score_pairs": len(keys),
        "n_units": len({key[0] for key in keys}),
        "quality_composite_pearson": _safe_correlation(
            primary_values, sensitivity_values, rank=False
        ),
        "quality_composite_spearman": _safe_correlation(
            primary_values, sensitivity_values, rank=True
        ),
        "primary_treatment_minus_comparator_mean": primary_delta,
        "sensitivity_treatment_minus_comparator_mean": sensitivity_delta,
        "direction_agreement": (
            primary_delta == 0.0
            or sensitivity_delta == 0.0
            or (primary_delta > 0.0) == (sensitivity_delta > 0.0)
        ),
        "gates_primary_claim": False,
        "reporting_required": True,
    }


def run_v1_5_external_batched_evaluation(
    *,
    evoemo_path: str | Path,
    study_freeze_path: str | Path,
    turn_paths: Sequence[str | Path],
    conditions: Sequence[str],
    treatment: str,
    turn_indices: Sequence[int],
    endpoints: Sequence[Endpoint],
    out_dir: str | Path,
    run: bool,
    max_api_calls: int,
    expected_units: Sequence[tuple[str, int, int, str, int]],
    full_expected_units: Sequence[tuple[str, int, int, str, int]],
    composite_spec: CompositeSpec,
    labeling: Mapping[str, Any],
    required_conditions: Sequence[str],
    generation_attestation_paths: Sequence[str | Path],
    study_freeze_sha256: str,
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
    api_cost_planning: Mapping[str, Any],
    primary_bootstrap_cluster: str,
    sensitivity_bootstrap_cluster: str,
    batched_contract: Mapping[str, Any],
    schema_pilot_summary_path: str | Path,
    schema_pilot_attestation_path: str | Path,
    schema_pilot_verification: Mapping[str, Any],
    excluded_unit_ids: Sequence[str],
    full_expected_units_sha256: str,
    observed_cost_match_report: Mapping[str, Any],
    accept_cost_estimate_sha256: str | None = None,
    max_estimated_usd: float = 20.0,
    max_input_tokens_per_call: int = 24_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the frozen V1.5 batched quality analysis and risk audit."""

    if treatment not in conditions:
        raise ValueError("batched treatment must be one of the frozen conditions")
    if list(conditions) != list(required_conditions):
        raise RuntimeError("batched evaluation must score the complete frozen condition matrix")
    if schema_pilot_verification.get("status") != "PASS":
        raise RuntimeError(
            "full batched evaluation requires an attested schema/order pilot"
        )
    if str(primary_bootstrap_cluster) != "user_id" or str(
        sensitivity_bootstrap_cluster
    ) != "scenario":
        raise RuntimeError("batched external cluster contract must remain user/scenario")
    if (
        not isinstance(observed_cost_match_report, Mapping)
        or observed_cost_match_report.get("status") != "PASS"
        or not bool(observed_cost_match_report.get("check"))
    ):
        raise RuntimeError("batched evaluation requires passing observed cost matching")

    full_units = sorted(_normalized_unit(unit) for unit in full_expected_units)
    units = sorted(_normalized_unit(unit) for unit in expected_units)
    excluded = sorted(set(full_units) - set(units))
    if set(units) | set(excluded) != set(full_units):
        raise RuntimeError("batched full/scoring unit partition is invalid")
    if sha256_text(canonical_json(full_units)) != str(full_expected_units_sha256):
        raise RuntimeError("batched full expected-unit hash is stale")
    derived_excluded_ids = sorted(unit_id(unit) for unit in excluded)
    if derived_excluded_ids != sorted(str(value) for value in excluded_unit_ids):
        raise RuntimeError("batched excluded-unit IDs do not bind the canary sample")
    matrix = load_fixed_turns(
        turn_paths,
        conditions=conditions,
        turn_indices=turn_indices,
        expected_units=units,
        full_expected_units=full_units,
        excluded_units=excluded,
    )
    authorized = _authorized_context_map(evoemo_path)
    plan, executions, samples = build_v1_5_batched_evaluation_plan(
        matrix=matrix,
        authorized_contexts=authorized,
        units=units,
        conditions=conditions,
        endpoints=endpoints,
        contract=batched_contract,
        pricing_usd_per_mtok=pricing_usd_per_mtok,
        api_cost_planning=api_cost_planning,
        study_freeze_sha256=study_freeze_sha256,
    )
    out = Path(out_dir)
    raw_path = out / "raw_judge_calls.jsonl"
    primary_score_path = out / "primary_quality_scores.jsonl"
    sensitivity_score_path = out / "claude_sensitivity_scores.jsonl"
    risk_score_path = out / "stratified_risk_audit_scores.jsonl"
    summary_path = out / "summary.json"
    attestation_path = out / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        out / "judge_call_ledger.jsonl", overwrite=overwrite, stage=STAGE
    )
    if overwrite:
        for path in (
            raw_path,
            primary_score_path,
            sensitivity_score_path,
            risk_score_path,
            summary_path,
            attestation_path,
        ):
            if path.exists():
                path.unlink()
    endpoint_contracts = [_endpoint_contract(endpoint) for endpoint in endpoints]
    contract_sha = sha256_text(canonical_json(dict(batched_contract)))
    execution_result = _execute_plan(
        plan=plan,
        executions=executions,
        endpoints=endpoints,
        out_dir=out,
        stage=STAGE,
        run=run,
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
        accept_cost_estimate_sha256=accept_cost_estimate_sha256,
        overwrite=overwrite,
        study_freeze_sha256=study_freeze_sha256,
        input_token_safety_factor=float(
            api_cost_planning["input_token_safety_factor"]
        ),
        fail_on_reported_input_overrun=bool(
            api_cost_planning["fail_on_reported_input_overrun"]
        ),
        manifest_metadata={
            "protocol": PROTOCOL,
            "evoemo_sha256": sha256_file(evoemo_path),
            "study_freeze_file_sha256": sha256_file(study_freeze_path),
            "turn_sha256": {
                str(Path(path).resolve()): sha256_file(path) for path in turn_paths
            },
            "generation_attestation_sha256": {
                str(Path(path).resolve()): sha256_file(path)
                for path in generation_attestation_paths
            },
            "conditions": list(conditions),
            "treatment": treatment,
            "expected_units_sha256": sha256_text(canonical_json(units)),
            "full_expected_units_sha256": full_expected_units_sha256,
            "excluded_unit_ids": list(excluded_unit_ids),
            "endpoint_contracts": endpoint_contracts,
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_hash(composite_spec),
            "batched_contract_sha256": contract_sha,
            "schema_pilot_summary_sha256": sha256_file(schema_pilot_summary_path),
            "schema_pilot_attestation_sha256": sha256_file(
                schema_pilot_attestation_path
            ),
            "schema_pilot_verification": dict(schema_pilot_verification),
            "observed_cost_match_report": dict(observed_cost_match_report),
        },
    )
    dry = {
        **{key: value for key, value in execution_result.items() if key != "successful"},
        "protocol": PROTOCOL,
        "scoring_mode": "anonymous_five_candidate_batched",
        "n_units": len(units),
        "conditions": list(conditions),
        "treatment": treatment,
        "llm_overall_requested": False,
        "quality_primary_units": len(samples["primary_units"]),
        "quality_sensitivity_units": len(samples["sensitivity_units"]),
        "risk_audit_units": len(samples["risk_units"]),
        "role_call_counts": samples["role_call_counts"],
        "excluded_canary_unit_count": len(excluded),
        "batched_contract_sha256": contract_sha,
        "schema_pilot_verification": dict(schema_pilot_verification),
        "observed_cost_match_report": dict(observed_cost_match_report),
        "bootstrap_cluster_contract": {
            "primary": "user_id",
            "sensitivity": "scenario",
        },
    }
    if not run:
        return dry

    successful = execution_result["successful"]
    raw_rows = [successful[key] for key in sorted(successful)]
    write_jsonl(raw_path, raw_rows)
    judge_usage_accounting = _judge_usage_accounting(
        raw_rows, pricing_usd_per_mtok
    )
    primary_rows = _quality_score_rows(
        raw_rows, role="quality_primary", composite_spec=composite_spec
    )
    sensitivity_rows = _quality_score_rows(
        raw_rows, role="quality_sensitivity", composite_spec=composite_spec
    )
    risk_contract = dict(batched_contract["risk_audit"])
    risk_rows = _risk_score_rows(
        raw_rows, expected_families=risk_contract["judge_families"]
    )
    expected_primary_rows = len(units) * len(conditions)
    expected_sensitivity_rows = len(samples["sensitivity_units"]) * len(conditions)
    expected_risk_rows = len(samples["risk_units"]) * len(conditions)
    if (
        len(primary_rows) != expected_primary_rows
        or len(sensitivity_rows) != expected_sensitivity_rows
        or len(risk_rows) != expected_risk_rows
    ):
        raise RuntimeError("batched score matrices are incomplete")
    write_jsonl(primary_score_path, primary_rows)
    write_jsonl(sensitivity_score_path, sensitivity_rows)
    write_jsonl(risk_score_path, risk_rows)

    quality_gate = _dimension_gate(
        primary_rows,
        dimensions=RESPONSE_DIMENSIONS,
        prefix="batched_primary_quality",
        duplicate_exact_match_rate=float(labeling["duplicate_exact_match_rate"]),
        maximum_absolute_dimension_correlation=float(
            labeling["maximum_absolute_dimension_correlation"]
        ),
        reject_constants=bool(labeling["reject_constant_response_dimensions"]),
    )
    position_gate = _position_balance_report(primary_rows, conditions)
    composites = np.asarray(
        [float(row["quality_composite"]) for row in primary_rows], dtype=float
    )
    support = np.asarray(
        [(float(row["emotional_support"]) - 1.0) / 4.0 for row in primary_rows],
        dtype=float,
    )
    exact_rate = float(np.mean(np.isclose(composites, support, atol=1e-12, rtol=0.0)))
    correlation = _safe_correlation(composites, support, rank=False)
    composite_gate = {
        "status": "PASS",
        "composite_support_exact_match_rate": exact_rate,
        "maximum_exact_match_rate": float(
            labeling["composite_support_exact_match_rate"]
        ),
        "composite_support_correlation": correlation,
        "maximum_absolute_correlation": float(
            labeling["maximum_absolute_composite_support_correlation"]
        ),
    }
    if exact_rate >= composite_gate["maximum_exact_match_rate"] or (
        correlation is not None
        and abs(correlation) >= composite_gate["maximum_absolute_correlation"]
    ):
        composite_gate["status"] = "FAIL"
        raise RuntimeError("batched quality composite collapsed to emotional support")

    bootstrap = dict(batched_contract["bootstrap"])
    n_resamples = int(bootstrap["replicates"])
    confidence = float(bootstrap["confidence_level"])
    bootstrap_seed = int(bootstrap["seed"])
    quality_metrics = (*RESPONSE_DIMENSIONS, "quality_composite", "observed_input_tokens")
    primary_condition_summary = _condition_summary(
        primary_rows, metrics=quality_metrics
    )
    primary_comparisons = _paired_comparisons(
        primary_rows,
        conditions=conditions,
        treatment=treatment,
        metrics=quality_metrics,
        n_resamples=n_resamples,
        confidence_level=confidence,
        seed=bootstrap_seed,
    )
    sensitivity_comparisons = _paired_comparisons(
        sensitivity_rows,
        conditions=conditions,
        treatment=treatment,
        metrics=quality_metrics,
        n_resamples=n_resamples,
        confidence_level=confidence,
        seed=bootstrap_seed + 1000,
    )
    comparator = str(risk_contract["primary_comparator"])
    sensitivity_report = {
        **_quality_sensitivity_report(
            primary_rows,
            sensitivity_rows,
            treatment=treatment,
            comparator=comparator,
        ),
        "condition_summary": _condition_summary(
            sensitivity_rows, metrics=quality_metrics
        ),
        "paired_treatment_deltas": sensitivity_comparisons,
    }

    risk_metrics = (
        *RISK_DIMENSIONS,
        "risk_composite",
        "risk_composite_all_dimensions_diagnostic",
    )
    risk_comparisons = _paired_comparisons(
        risk_rows,
        conditions=conditions,
        treatment=treatment,
        metrics=risk_metrics,
        n_resamples=n_resamples,
        confidence_level=confidence,
        seed=bootstrap_seed + 2000,
    )
    mad_threshold = float(risk_contract["reliable_mad_threshold"])
    mad_coverage = {
        name: float(
            np.mean(
                [
                    float(row["risk_dimension_mad"][name]) <= mad_threshold
                    for row in risk_rows
                ]
            )
        )
        for name in RISK_DIMENSIONS
    }
    minimum_mad_coverage = float(risk_contract["minimum_low_mad_coverage"])
    low_coverage = sorted(
        name for name, value in mad_coverage.items() if value < minimum_mad_coverage
    )
    if low_coverage:
        raise RuntimeError(
            "stratified risk audit lacks frozen cross-family agreement: "
            + str(low_coverage)
        )
    risk_ci = risk_comparisons[comparator]["risk_composite"]["primary_cluster_ci"]
    risk_margin = float(risk_contract["nonincrease_margin"])
    no_increase_signal = float(risk_ci["upper"]) <= risk_margin
    risk_audit = {
        "status": "COMPLETE",
        "protocol": RISK_AUDIT_PROTOCOL,
        "role": risk_contract["role"],
        "claim_boundary": (
            "This preregistered stratified audit checks evidence/resource-use "
            "problems only; it is not a clinical, deployment, or population safety claim."
        ),
        "units": len(samples["risk_units"]),
        "users": len({unit[0] for unit in samples["risk_units"]}),
        "judge_families": list(risk_contract["judge_families"]),
        "primary_comparator": comparator,
        "risk_metric": "risk_composite",
        "nonincrease_margin": risk_margin,
        "primary_comparator_delta_ci": risk_ci,
        "finding": (
            "NO_INCREASE_SIGNAL_WITHIN_FROZEN_MARGIN"
            if no_increase_signal
            else "INCONCLUSIVE_OR_INCREASE_SIGNAL"
        ),
        "paper_statement_allowed": no_increase_signal,
        "allowed_statement": (
            "The preregistered stratified audit found no increase in evidence-use "
            "problems for PM-v1.5 relative to the frozen primary comparator."
            if no_increase_signal
            else None
        ),
        "condition_summary": _condition_summary(risk_rows, metrics=risk_metrics),
        "paired_treatment_deltas": risk_comparisons,
        "risk_dimension_mad_coverage": mad_coverage,
        "minimum_low_mad_coverage": minimum_mad_coverage,
    }
    gate_e = build_external_gate_e(
        quality_comparisons=primary_comparisons,
        risk_comparisons=risk_comparisons,
        contract=batched_contract["gate_e"],
    )

    summary = {
        **dry,
        "status": "COMPLETE",
        "execution_status": "COMPLETE",
        "external_estimand": "quality_noninferiority_plus_observed_input_cost_v1",
        "single_external_utility_claim_allowed": False,
        "risk_is_separate_stratified_audit": True,
        "score_rows": len(primary_rows),
        "primary_quality_judge_family": str(
            batched_contract["quality"]["primary_judge_family"]
        ),
        "quality_gate": quality_gate,
        "candidate_position_gate": position_gate,
        "composite_independence_gate": composite_gate,
        "condition_summary": primary_condition_summary,
        "paired_treatment_deltas": primary_comparisons,
        "quality_sensitivity": sensitivity_report,
        "stratified_risk_audit": risk_audit,
        "gate_e": gate_e,
        "judge_usage_accounting": judge_usage_accounting,
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_hash(composite_spec),
        "primary_scores_path": str(primary_score_path),
        "sensitivity_scores_path": str(sensitivity_score_path),
        "risk_audit_scores_path": str(risk_score_path),
        "raw_path": str(raw_path),
        "ledger_path": execution_result["ledger_path"],
        "physical_http_attempts": execution_result["physical_http_attempts"],
        "final_ledger_sha256": sha256_text(
            canonical_json(execution_result["ledger_rows"])
        ),
    }
    write_json(summary_path, summary)
    create_artifact_attestation(
        attestation_path,
        stage=STAGE,
        inputs={
            "study_freeze": study_freeze_path,
            "evoemo": evoemo_path,
            "schema_pilot_summary": schema_pilot_summary_path,
            "schema_pilot_attestation": schema_pilot_attestation_path,
            "run_manifest": execution_result["manifest_path"],
            "cost_estimate": execution_result["estimate_path"],
            **{f"turn_{index}": path for index, path in enumerate(turn_paths)},
            **{
                f"generation_attestation_{index}": path
                for index, path in enumerate(generation_attestation_paths)
            },
        },
        outputs={
            "raw_calls": (raw_path, True),
            "call_ledger": (execution_result["ledger_path"], True),
            "primary_quality_scores": (primary_score_path, True),
            "sensitivity_scores": (sensitivity_score_path, True),
            "risk_audit_scores": (risk_score_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "protocol": PROTOCOL,
            "batched_contract": dict(batched_contract),
            "schema_pilot_verification": dict(schema_pilot_verification),
            "conditions": list(conditions),
            "treatment": treatment,
            "composite_spec": composite_spec.model_dump(mode="json"),
            "observed_cost_match_report": dict(observed_cost_match_report),
            "bootstrap_cluster_contract": {
                "primary": "user_id",
                "sensitivity": "scenario",
            },
        },
        expected={
            "primary_score_rows": expected_primary_rows,
            "sensitivity_score_rows": expected_sensitivity_rows,
            "risk_audit_score_rows": expected_risk_rows,
            "status": "COMPLETE",
        },
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary
