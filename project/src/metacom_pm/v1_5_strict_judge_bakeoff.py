from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from .api import Endpoint, chat_request_payload
from .attempt_ledger import physical_call_key
from .io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
)
from .text import conservative_token_bound


STRICT_BAKEOFF_PROTOCOL = (
    "pm-v1.5-train-only-strict-pairwise-judge-bakeoff-v1"
)
STRICT_BAKEOFF_STAGE = (
    "longitudinal_train_only_strict_pairwise_judge_bakeoff_v1"
)
EXPECTED_PAIR_COUNT = 12
EXPECTED_ORDERED_PAIR_COUNT = 24
EXPECTED_GPT_ANCHOR_CALLS = 16
EXPECTED_CANDIDATES = (
    "anthropic_claude_haiku_4_5",
    "google_gemini_2_5_flash",
    "openai_gpt_5_mini",
)


class StrictBakeoffPairwiseOutput(BaseModel):
    """Provider-constrained preference schema for the strict bake-off.

    Preference values are the scientific output.  Reasons are required for
    auditability, but an arbitrary character ceiling is deliberately absent:
    the provider request already has a frozen total-token ceiling and a long
    explanation must not turn an otherwise valid preference into a false
    schema failure.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    overall_preference: Literal["A", "B", "tie", "insufficient"]
    support_quality_preference: Literal["A", "B", "tie", "insufficient"]
    evidence_handling_preference: Literal["A", "B", "tie", "insufficient"]
    safety_preference: Literal["A", "B", "tie", "insufficient"]
    overall_reason: str
    support_quality_reason: str
    evidence_handling_reason: str
    safety_reason: str

    @field_validator(
        "overall_reason",
        "support_quality_reason",
        "evidence_handling_reason",
        "safety_reason",
    )
    @classmethod
    def _reason_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value


def endpoint_from_record(raw: Mapping[str, Any]) -> Endpoint:
    return Endpoint(
        base_url=str(raw["base_url"]),
        model=str(raw["model"]),
        api_key_env=str(raw["api_key_env"]),
        family=str(raw["family"]),
        transport=str(raw["transport"]),
        supports_strict_json_schema=bool(raw["supports_strict_json_schema"]),
        temperature_mode=str(raw.get("temperature_mode", "explicit")),
        max_output_tokens_parameter=str(
            raw.get("max_output_tokens_parameter", "max_tokens")
        ),
        anthropic_strict_tool_use=bool(
            raw.get("anthropic_strict_tool_use", False)
        ),
        gemini_thinking_budget=(
            int(raw["gemini_thinking_budget"])
            if raw.get("gemini_thinking_budget") is not None
            else None
        ),
        openai_reasoning_effort=(
            str(raw["openai_reasoning_effort"])
            if raw.get("openai_reasoning_effort") is not None
            else None
        ),
    )


def validate_endpoint_contract(record: Mapping[str, Any]) -> None:
    if (
        record.get("protocol")
        != "pm-v1.5-strict-pairwise-judge-bakeoff-endpoints-v1"
    ):
        raise RuntimeError("unexpected strict judge bake-off endpoint protocol")
    candidates = dict(record.get("candidates") or {})
    if tuple(sorted(candidates)) != EXPECTED_CANDIDATES:
        raise RuntimeError("strict judge bake-off candidate set drifted")
    expected = {
        "openai_gpt_5_mini": {
            "model": "gpt-5-mini-2025-08-07",
            "family": "openai_gpt_5_mini",
            "transport": "openai_chat_completions",
            "supports_strict_json_schema": True,
            "temperature_mode": "omit",
            "max_output_tokens_parameter": "max_completion_tokens",
            "openai_reasoning_effort": "minimal",
            "provider_max_output_tokens": 1400,
            "input_usd_per_million_tokens": 0.25,
            "output_usd_per_million_tokens": 2.0,
        },
        "anthropic_claude_haiku_4_5": {
            "model": "claude-haiku-4-5-20251001",
            "family": "anthropic_claude_haiku_4_5",
            "transport": "anthropic_messages",
            "supports_strict_json_schema": True,
            "anthropic_strict_tool_use": True,
            "input_usd_per_million_tokens": 1.0,
            "output_usd_per_million_tokens": 5.0,
        },
        "google_gemini_2_5_flash": {
            "api_key_env": "GEMINI_API_KEY",
            "model": "gemini-2.5-flash",
            "family": "google_gemini_2_5_flash",
            "transport": "gemini_generate_content",
            "supports_strict_json_schema": True,
            "gemini_thinking_budget": 0,
            "input_usd_per_million_tokens": 0.3,
            "output_usd_per_million_tokens": 2.5,
        },
    }
    for key, frozen in expected.items():
        raw = dict(candidates[key])
        for field, value in frozen.items():
            if raw.get(field) != value:
                raise RuntimeError(
                    f"strict judge candidate {key}/{field} drifted"
                )
        endpoint_from_record(raw)
    request = dict(record.get("request_contract") or {})
    if request != {
        "client_internal_retries": 1,
        "input_token_safety_factor": 1.5,
        "maximum_physical_attempts_per_logical_call": 2,
        "pairwise_max_output_tokens": 700,
        "seed": 12091,
        "temperature": 0.0,
    }:
        raise RuntimeError("strict judge bake-off request contract drifted")


def require_gpt_anchor_source(
    source_dir: Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Bind the already-paid GPT-5.6 anchor without issuing it again."""

    results_path = source_dir / "qualification_results.jsonl"
    ledger_path = source_dir / "physical_attempt_ledger.jsonl"
    cost_path = source_dir / "cost_estimate.json"
    if not all(path.is_file() for path in (results_path, ledger_path, cost_path)):
        raise RuntimeError("GPT anchor source is incomplete")
    import json

    selected: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("condition") != "gpt_high_quality_pairwise_anchor":
            continue
        parsed = StrictBakeoffPairwiseOutput.model_validate(
            row.get("parsed"), strict=True
        )
        selected.append(
            {
                "condition": str(row["condition"]),
                "judge_family": str(row["judge_family"]),
                "judge_model": str(row["judge_model"]),
                "record_ids": dict(row["record_ids"]),
                "request_hash": str(row["request_hash"]),
                "parsed": parsed.model_dump(mode="json"),
            }
        )
    selected.sort(key=lambda row: canonical_json(row["record_ids"]))
    if len(selected) != EXPECTED_GPT_ANCHOR_CALLS:
        raise RuntimeError("GPT anchor source must contain exactly 16 calls")
    if any(
        row["judge_family"] != "openai_gpt_5_6_sol"
        or row["judge_model"] != "gpt-5.6-sol"
        for row in selected
    ):
        raise RuntimeError("GPT anchor family/model drifted")
    source_display = source_dir
    if root is not None:
        try:
            source_display = source_dir.resolve().relative_to(root.resolve())
        except ValueError:
            raise RuntimeError("GPT anchor source must be inside the repository")
    return {
        "protocol": "pm-v1.5-existing-gpt56-pairwise-anchor-binding-v1",
        "role": "secondary_high_quality_anchor_not_gold_not_bulk_labeler",
        "source_directory": str(source_display),
        "qualification_results_sha256": sha256_file(results_path),
        "physical_attempt_ledger_sha256": sha256_file(ledger_path),
        "cost_estimate_sha256": sha256_file(cost_path),
        "selected_calls": len(selected),
        "selected_call_keys_sha256": sha256_text(
            canonical_json(
                [
                    {
                        "record_ids": row["record_ids"],
                        "request_hash": row["request_hash"],
                    }
                    for row in selected
                ]
            )
        ),
        "selected_results_sha256": sha256_text(canonical_json(selected)),
    }


def bakeoff_contract_record(
    *,
    endpoint_contract: Mapping[str, Any],
    packet_files: Mapping[str, Any],
    human_anchor: Mapping[str, Any],
    gpt_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    validate_endpoint_contract(endpoint_contract)
    payload = {
        "protocol": STRICT_BAKEOFF_PROTOCOL,
        "status": "READY_FOR_ZERO_API_DRY_RUN",
        "scope": "longitudinal_train_only_pairwise_measurement_qualification",
        "candidate_keys": list(EXPECTED_CANDIDATES),
        "pair_count": EXPECTED_PAIR_COUNT,
        "order_variants": [0, 1],
        "calls_per_candidate": EXPECTED_ORDERED_PAIR_COUNT,
        "planned_new_logical_calls": (
            len(EXPECTED_CANDIDATES) * EXPECTED_ORDERED_PAIR_COUNT
        ),
        "measurement_contract": {
            "pairwise_only": True,
            "combined_vs_split_not_repeated": True,
            "same_ordered_items_for_every_candidate": True,
            "preference_fields_are_scientific_outputs": True,
            "reasons_are_required_audit_text": True,
            "arbitrary_reason_character_ceiling_is_not_a_scientific_gate": True,
            "total_output_tokens_remain_bounded": True,
            "provider_enforced_schema_required": True,
        },
        "decision_boundary": {
            "qualification_only": True,
            "creates_training_labels": False,
            "majority_vote_is_gold": False,
            "human_anchor_is_not_automatic_gold": True,
            "gpt_anchor_is_not_automatic_gold": True,
            "calibration_internal_test_external_outcomes_forbidden": True,
            "one_candidate_may_be_promoted_only_after_researcher_signoff": True,
            "unresolved_items_must_abstain_not_receive_synthetic_labels": True,
        },
        "pre_outcome_criteria": {
            "required_schema_valid_rate": 1.0,
            "minimum_ab_ba_order_consistency": 0.80,
            "minimum_pairwise_informative_rate": 0.50,
            "minimum_gpt_anchor_direction_agreement_secondary": 0.70,
            "human_anchor_review": (
                "descriptive by confidence and dimension; no automatic "
                "promotion threshold and no post-outcome threshold tuning"
            ),
            "failure_disposition": (
                "candidate not promoted; no prompt/schema repair after outcome"
            ),
        },
        "response_schema": {
            "class": StrictBakeoffPairwiseOutput.__name__,
            "sha256": sha256_text(
                canonical_json(
                    StrictBakeoffPairwiseOutput.model_json_schema()
                )
            ),
        },
        "endpoint_contract": dict(endpoint_contract),
        "packet_files": dict(packet_files),
        "human_anchor": dict(human_anchor),
        "gpt_anchor": dict(gpt_anchor),
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    return {
        **payload,
        "contract_sha256": sha256_text(canonical_json(payload)),
    }


def build_call_plan(
    *,
    ordered_pairwise_rows: Sequence[Mapping[str, Any]],
    endpoint_contract: Mapping[str, Any],
    input_token_safety_factor: float | None = None,
) -> list[dict[str, Any]]:
    validate_endpoint_contract(endpoint_contract)
    rows = sorted(
        [dict(row) for row in ordered_pairwise_rows],
        key=lambda row: (str(row["pair_id"]), int(row["order_variant"])),
    )
    if len(rows) != EXPECTED_ORDERED_PAIR_COUNT:
        raise RuntimeError("strict bake-off requires 24 ordered pairwise rows")
    if Counter(int(row["order_variant"]) for row in rows) != {0: 12, 1: 12}:
        raise RuntimeError("strict bake-off AB/BA allocation drifted")
    if len({str(row["pair_id"]) for row in rows}) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("strict bake-off pair coverage drifted")

    request = dict(endpoint_contract["request_contract"])
    frozen_safety_factor = float(request["input_token_safety_factor"])
    if input_token_safety_factor is not None and (
        float(input_token_safety_factor) != frozen_safety_factor
    ):
        raise RuntimeError("strict bake-off input-token safety factor drifted")
    default_max_tokens = int(request["pairwise_max_output_tokens"])
    base_seed = int(request["seed"])
    schema_sha = sha256_text(
        canonical_json(StrictBakeoffPairwiseOutput.model_json_schema())
    )
    plan: list[dict[str, Any]] = []
    for candidate_key in EXPECTED_CANDIDATES:
        raw = dict(endpoint_contract["candidates"][candidate_key])
        endpoint = endpoint_from_record(raw)
        max_tokens = int(
            raw.get("provider_max_output_tokens", default_max_tokens)
        )
        for index, row in enumerate(rows):
            messages = [dict(message) for message in row["messages"]]
            prompt_sha = sha256_text(canonical_json(messages))
            provider_payload = chat_request_payload(
                endpoint,
                messages,
                temperature=float(request["temperature"]),
                max_tokens=max_tokens,
                seed=base_seed + index,
                response_schema=StrictBakeoffPairwiseOutput,
            )
            provider_payload_json = canonical_json(provider_payload)
            input_est = conservative_token_bound(
                provider_payload_json,
                safety_factor=frozen_safety_factor,
            )
            request_parameters: dict[str, Any] = {
                "temperature": (
                    None
                    if endpoint.temperature_mode == "omit"
                    else float(request["temperature"])
                ),
                "temperature_mode": endpoint.temperature_mode,
                "max_output_tokens": max_tokens,
                "max_output_tokens_parameter": (
                    endpoint.max_output_tokens_parameter
                ),
                "seed": base_seed + index,
                "provider_schema_sha256": schema_sha,
                "provider_request_payload_sha256": sha256_text(
                    provider_payload_json
                ),
                "input_token_bound_protocol": (
                    "complete_provider_payload_including_schema_x1.5_v1"
                ),
                "client_internal_retries": int(
                    request["client_internal_retries"]
                ),
            }
            if endpoint.gemini_thinking_budget is not None:
                request_parameters["gemini_thinking_budget"] = int(
                    endpoint.gemini_thinking_budget
                )
            if endpoint.openai_reasoning_effort is not None:
                request_parameters["openai_reasoning_effort"] = str(
                    endpoint.openai_reasoning_effort
                )
            record_ids = {
                "candidate_key": candidate_key,
                "pair_id": str(row["pair_id"]),
                "state_id": str(row["state_id"]),
                "order_variant": int(row["order_variant"]),
            }
            maximum_cost = (
                input_est
                / 1_000_000
                * float(raw["input_usd_per_million_tokens"])
                + max_tokens
                / 1_000_000
                * float(raw["output_usd_per_million_tokens"])
            )
            plan.append(
                {
                    "condition": f"{candidate_key}_pairwise",
                    "candidate_key": candidate_key,
                    "judge_family": endpoint.family,
                    "judge_model": endpoint.model,
                    "schema_kind": "pairwise",
                    "record_ids": record_ids,
                    "messages": messages,
                    "prompt_sha256": prompt_sha,
                    "response_schema_sha256": schema_sha,
                    "request_parameters": request_parameters,
                    "input_tokens_est": input_est,
                    "max_output_tokens": max_tokens,
                    "maximum_single_attempt_cost_usd": maximum_cost,
                    "physical_call_key": physical_call_key(
                        stage=STRICT_BAKEOFF_STAGE,
                        record_ids=record_ids,
                        prompt_sha256=prompt_sha,
                        endpoint=endpoint,
                        request_parameters=request_parameters,
                    ),
                }
            )
    plan.sort(
        key=lambda row: (
            str(row["candidate_key"]),
            str(row["record_ids"]["pair_id"]),
            int(row["record_ids"]["order_variant"]),
        )
    )
    if len(plan) != 72 or len(
        {str(row["physical_call_key"]) for row in plan}
    ) != 72:
        raise RuntimeError("strict bake-off must contain 72 unique calls")
    return plan


def strict_bakeoff_carry_forward(
    *,
    source_dir: Path,
    plan: Sequence[Mapping[str, Any]],
    root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select only exact, already-succeeded calls from an attested prior run."""

    source_dir = source_dir.resolve()
    ledger_path = source_dir / "physical_attempt_ledger.jsonl"
    results_path = source_dir / "qualification_results.jsonl"
    attestation_path = source_dir / "artifact_attestation.json"
    if not all(
        path.is_file()
        for path in (ledger_path, results_path, attestation_path)
    ):
        raise RuntimeError("strict bake-off carry-forward source is incomplete")
    attestation = read_json(attestation_path)
    if (
        attestation.get("protocol")
        != "pm-v1.5-strict-judge-bakeoff-attestation-v1"
        or attestation.get("stage") != STRICT_BAKEOFF_STAGE
        or attestation.get("physical_attempt_ledger_sha256")
        != sha256_file(ledger_path)
        or attestation.get("qualification_results_sha256")
        != sha256_file(results_path)
        or attestation.get("training_labels_created") is not False
    ):
        raise RuntimeError(
            "strict bake-off carry-forward attestation is invalid"
        )
    plan_by_key = {
        str(row["physical_call_key"]): dict(row) for row in plan
    }
    ledger_rows = [dict(row) for row in iter_jsonl(ledger_path)]
    terminals: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        if row.get("event") not in {"SUCCEEDED", "FAILED"}:
            continue
        key = str(row.get("call_key") or "")
        if key not in plan_by_key:
            continue
        if key in terminals:
            raise RuntimeError(
                "strict bake-off carry-forward has duplicate terminal rows"
            )
        terminals[key] = row
    carried_keys = sorted(
        key
        for key, terminal in terminals.items()
        if key in plan_by_key and terminal.get("event") == "SUCCEEDED"
    )
    if not carried_keys:
        raise RuntimeError("strict bake-off carry-forward found no exact successes")
    carried_set = set(carried_keys)
    selected_ledger_rows = [
        row for row in ledger_rows
        if str(row.get("call_key") or "") in carried_set
    ]
    for key in carried_keys:
        rows = [
            row for row in selected_ledger_rows
            if str(row.get("call_key") or "") == key
        ]
        if [row.get("event") for row in rows] != ["STARTED", "SUCCEEDED"]:
            raise RuntimeError(
                "strict bake-off carry-forward call is not one clean attempt"
            )
        record_ids = dict(rows[-1].get("record_ids") or {})
        if record_ids != dict(plan_by_key[key]["record_ids"]):
            raise RuntimeError(
                "strict bake-off carry-forward record IDs drifted"
            )
    by_candidate = Counter(
        str(plan_by_key[key]["candidate_key"]) for key in carried_keys
    )
    source_display = source_dir
    if root is not None:
        try:
            source_display = source_dir.relative_to(root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "strict bake-off carry-forward source must be in repository"
            ) from exc
    record = {
        "protocol": "pm-v1.5-strict-bakeoff-exact-carry-forward-v1",
        "source_directory": str(source_display),
        "source_ledger_sha256": sha256_file(ledger_path),
        "source_results_sha256": sha256_file(results_path),
        "source_attestation_sha256": sha256_file(attestation_path),
        "carried_call_keys": carried_keys,
        "carried_call_keys_sha256": sha256_text(
            canonical_json(carried_keys)
        ),
        "carried_logical_calls": len(carried_keys),
        "carried_by_candidate": dict(sorted(by_candidate.items())),
    }
    return record, selected_ledger_rows


def build_cost_estimate(
    *,
    plan: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    code_manifest: Mapping[str, Any],
    maximum_attempts: int,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    carry_forward: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    carried_keys = set(
        str(key)
        for key in (dict(carry_forward or {}).get("carried_call_keys") or [])
    )
    plan_keys = {str(row["physical_call_key"]) for row in plan}
    if not carried_keys <= plan_keys:
        raise RuntimeError("strict bake-off carry-forward keys are not in plan")
    new_plan = [
        row for row in plan
        if str(row["physical_call_key"]) not in carried_keys
    ]
    one_attempt = math.fsum(
        float(row["maximum_single_attempt_cost_usd"]) for row in new_plan
    )
    by_candidate = {}
    for candidate in EXPECTED_CANDIDATES:
        all_rows = [
            row for row in plan if row["candidate_key"] == candidate
        ]
        rows = [
            row for row in all_rows
            if str(row["physical_call_key"]) not in carried_keys
        ]
        subtotal = math.fsum(
            float(row["maximum_single_attempt_cost_usd"]) for row in rows
        )
        by_candidate[candidate] = {
            "logical_calls": len(all_rows),
            "carried_forward_logical_calls": len(all_rows) - len(rows),
            "new_logical_calls": len(rows),
            "logical_single_attempt_cost_usd": subtotal,
            "maximum_cost_usd": subtotal * maximum_attempts,
        }
    payload = {
        "protocol": "pm-v1.5-strict-judge-bakeoff-cost-v1",
        "stage": STRICT_BAKEOFF_STAGE,
        "qualification_contract_sha256": str(contract["contract_sha256"]),
        "qualification_contract": dict(contract),
        "code_manifest": dict(code_manifest),
        "logical_calls": len(plan),
        "carried_forward_logical_calls": len(carried_keys),
        "new_logical_calls": len(new_plan),
        "maximum_physical_attempts_per_logical_call": maximum_attempts,
        "maximum_physical_attempts": len(new_plan) * maximum_attempts,
        "logical_single_attempt_cost_usd": one_attempt,
        "maximum_cost_usd": one_attempt * maximum_attempts,
        "by_candidate": by_candidate,
        "maximum_input_tokens_per_call_est": (
            max(int(row["input_tokens_est"]) for row in new_plan)
            if new_plan else 0
        ),
        "call_plan_sha256": sha256_text(canonical_json(list(plan))),
        "carry_forward": dict(carry_forward or {}),
        "budget_limits": {
            "max_api_calls": max_api_calls,
            "max_estimated_usd": max_estimated_usd,
            "max_input_tokens_per_call": max_input_tokens_per_call,
        },
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    checks = {
        "api_calls": payload["maximum_physical_attempts"]
        <= max_api_calls,
        "estimated_cost_usd": payload["maximum_cost_usd"]
        <= max_estimated_usd,
        "max_input_tokens_per_call": payload[
            "maximum_input_tokens_per_call_est"
        ]
        <= max_input_tokens_per_call,
    }
    record = {
        **payload,
        "budget_gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
    }
    return {
        **record,
        "cost_estimate_sha256": sha256_text(canonical_json(record)),
    }
