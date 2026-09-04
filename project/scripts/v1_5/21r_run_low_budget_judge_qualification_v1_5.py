#!/usr/bin/env python3
"""Dry-run or execute the train-only low-budget judge qualification matrix."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, Type

from pydantic import BaseModel

from metacom_pm.api import (
    Endpoint,
    StructuredOutputValidationError,
    make_client,
    require_reported_usage,
)
from metacom_pm.attempt_ledger import (
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
)
from metacom_pm.bounded_retry import execute_with_bounded_retry
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
from metacom_pm.pm_v2_judging import ResponseJudgeOutput, RiskJudgeOutput
from metacom_pm.text import conservative_token_bound, estimate_tokens
from metacom_pm.v1_5_judge_qualification import (
    COMBINED_MAX_OUTPUT_TOKENS,
    GPT_ANCHOR_STATE_COUNT,
    PAIRWISE_MAX_OUTPUT_TOKENS,
    QUALITY_MAX_OUTPUT_TOKENS,
    RISK_MAX_OUTPUT_TOKENS,
    CombinedQualityRiskOutput,
    QualificationPairwiseOutput,
    qualification_contract_record,
)
from metacom_pm.v1_5_judge_qualification_analysis import (
    require_tracked_human_anchor,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE = "longitudinal_train_only_judge_qualification_v2"
BACKOFF_SECONDS = (10.0,)


SCHEMAS: dict[str, Type[BaseModel]] = {
    "pairwise": QualificationPairwiseOutput,
    "combined": CombinedQualityRiskOutput,
    "quality": ResponseJudgeOutput,
    "risk": RiskJudgeOutput,
}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def _persist_exact(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise RuntimeError(f"existing qualification artifact drifted: {path}")
    else:
        write_json(path, dict(payload))


def _persist_rows_exact(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    normalized = [dict(row) for row in rows]
    if path.is_file():
        if _rows(path) != normalized:
            raise RuntimeError(f"existing qualification artifact drifted: {path}")
    else:
        write_jsonl(path, normalized)


def _load_delta_carry_forward(
    *,
    source_dir: Path | None,
    plan: Sequence[Mapping[str, Any]],
    maximum_attempts: int,
) -> dict[str, Any]:
    """Reuse only byte-equivalent, provider-succeeded qualification calls."""

    empty = {
        "mechanism": "per_call_exact_physical_identity_v1",
        "source_directory": None,
        "source_call_plan_sha256": None,
        "source_ledger_sha256": None,
        "carried_call_keys": set(),
        "terminal_rows": {},
    }
    if source_dir is None:
        return empty
    old_plan_path = source_dir / "call_plan.jsonl"
    old_ledger_path = source_dir / "physical_attempt_ledger.jsonl"
    if not old_plan_path.is_file() or not old_ledger_path.is_file():
        raise RuntimeError(
            "qualification carry-forward source lacks call plan or ledger"
        )
    old_plan = _rows(old_plan_path)
    old_by_key = {
        str(row["physical_call_key"]): row for row in old_plan
    }
    if len(old_by_key) != len(old_plan):
        raise RuntimeError(
            "qualification carry-forward source has duplicate call keys"
        )
    old_ledger = PersistentAttemptLedger(
        old_ledger_path,
        stage=STAGE,
        expected_calls={
            key: maximum_attempts for key in old_by_key
        },
        maximum_total_attempts=10**9,
    )
    carried: set[str] = set()
    terminals: dict[str, dict[str, Any]] = {}
    for new_row in plan:
        call_key = str(new_row["physical_call_key"])
        old_row = old_by_key.get(call_key)
        if old_row is None:
            continue
        for field in (
            "condition",
            "endpoint_key",
            "judge_family",
            "judge_model",
            "schema_kind",
            "record_ids",
            "messages",
            "prompt_sha256",
            "response_schema_sha256",
            "request_parameters",
        ):
            if old_row.get(field) != new_row.get(field):
                raise RuntimeError(
                    "qualification carry-forward physical identity "
                    f"collided with different {field}: {call_key}"
                )
        if not old_ledger.succeeded(call_key):
            continue
        terminal = old_ledger.terminal_row(call_key) or {}
        parsed = dict(terminal.get("result") or {}).get("parsed")
        usage = terminal.get("usage")
        if not isinstance(parsed, Mapping) or not isinstance(usage, Mapping):
            raise RuntimeError(
                "qualification carry-forward success lacks parsed result "
                f"or usage: {call_key}"
            )
        carried.add(call_key)
        terminals[call_key] = dict(terminal)
    return {
        **empty,
        "source_directory": str(source_dir),
        "source_call_plan_sha256": sha256_file(old_plan_path),
        "source_ledger_sha256": sha256_file(old_ledger_path),
        "carried_call_keys": carried,
        "terminal_rows": terminals,
    }


def _endpoint(
    raw: Mapping[str, Any], *, enable_thinking: bool | None
) -> Endpoint:
    return Endpoint(
        base_url=str(raw["base_url"]),
        model=str(raw["model"]),
        api_key_env=str(raw["api_key_env"]),
        family=str(raw["family"]),
        transport=str(raw["transport"]),
        supports_strict_json_schema=bool(
            raw["supports_strict_json_schema"]
        ),
        enable_thinking=enable_thinking,
        temperature_mode=str(raw.get("temperature_mode", "explicit")),
        max_output_tokens_parameter=str(
            raw.get("max_output_tokens_parameter", "max_tokens")
        ),
    )


def _endpoint_records(
    record: Mapping[str, Any],
) -> tuple[dict[str, Endpoint], dict[str, dict[str, float]]]:
    qwen = dict(record["qwen"])
    gpt = dict(record["gpt_anchor"])
    endpoints = {
        "qwen_nonthinking": _endpoint(qwen, enable_thinking=False),
        "qwen_thinking": _endpoint(qwen, enable_thinking=True),
        "gpt_anchor": _endpoint(gpt, enable_thinking=None),
    }
    prices = {
        "qwen_nonthinking": {
            "input": float(qwen["input_usd_per_million_tokens"]),
            "output": float(qwen["output_usd_per_million_tokens"]),
        },
        "qwen_thinking": {
            "input": float(qwen["input_usd_per_million_tokens"]),
            "output": float(qwen["output_usd_per_million_tokens"]),
        },
        "gpt_anchor": {
            "input": float(gpt["input_usd_per_million_tokens"]),
            "output": float(gpt["output_usd_per_million_tokens"]),
        },
    }
    return endpoints, prices


def _add_call(
    plan: list[dict[str, Any]],
    *,
    condition: str,
    endpoint_key: str,
    endpoint: Endpoint,
    price: Mapping[str, float],
    schema_kind: str,
    messages: Sequence[Mapping[str, str]],
    record_ids: Mapping[str, Any],
    max_output_tokens: int,
    seed: int,
    input_token_safety_factor: float,
) -> None:
    messages_list = _messages_with_loose_json_schema_bridge(
        messages=messages,
        endpoint=endpoint,
        schema_kind=schema_kind,
    )
    messages_json = canonical_json(messages_list)
    if response_schema_requires_json_literal(endpoint) and (
        "json" not in messages_json.casefold()
    ):
        raise RuntimeError(
            "loose json_object response mode requires an explicit JSON "
            "instruction in the model-visible messages"
        )
    prompt_sha256 = sha256_text(messages_json)
    schema = SCHEMAS[schema_kind]
    schema_sha256 = sha256_text(
        canonical_json(schema.model_json_schema())
    )
    input_est = conservative_token_bound(
        messages_json, safety_factor=input_token_safety_factor
    )
    request_parameters = {
        "temperature_mode": endpoint.temperature_mode,
        "temperature": (
            0.0 if endpoint.temperature_mode == "explicit" else None
        ),
        "max_output_tokens_parameter": (
            endpoint.max_output_tokens_parameter
        ),
        "max_output_tokens": int(max_output_tokens),
        "seed": int(seed),
        "schema_kind": schema_kind,
        "response_schema_sha256": schema_sha256,
        "enable_thinking": endpoint.enable_thinking,
    }
    row = {
        "condition": condition,
        "endpoint_key": endpoint_key,
        "judge_family": endpoint.family,
        "judge_model": endpoint.model,
        "schema_kind": schema_kind,
        "record_ids": dict(record_ids),
        "messages": messages_list,
        "prompt_sha256": prompt_sha256,
        "response_schema_sha256": schema_sha256,
        "request_parameters": request_parameters,
        "seed": int(seed),
        "base_input_tokens_est": estimate_tokens(messages_json),
        "input_tokens_est": input_est,
        "max_output_tokens": int(max_output_tokens),
        "maximum_single_attempt_cost_usd": (
            input_est / 1_000_000 * float(price["input"])
            + int(max_output_tokens)
            / 1_000_000
            * float(price["output"])
        ),
    }
    row["physical_call_key"] = physical_call_key(
        stage=STAGE,
        record_ids={
            **dict(record_ids),
            "condition": condition,
            "schema_kind": schema_kind,
        },
        prompt_sha256=prompt_sha256,
        endpoint=endpoint,
        request_parameters=request_parameters,
    )
    plan.append(row)


def response_schema_requires_json_literal(endpoint: Endpoint) -> bool:
    """Return whether this endpoint uses provider loose-JSON response mode."""

    return (
        endpoint.transport == "openai_chat_completions"
        and not endpoint.supports_strict_json_schema
    )


def _messages_with_loose_json_schema_bridge(
    *,
    messages: Sequence[Mapping[str, str]],
    endpoint: Endpoint,
    schema_kind: str,
) -> list[dict[str, str]]:
    """Expose the complete Pydantic schema to loose-json providers."""

    normalized = [dict(row) for row in messages]
    if not response_schema_requires_json_literal(endpoint):
        return normalized
    try:
        schema = SCHEMAS[schema_kind]
    except KeyError as exc:  # pragma: no cover - guarded by SCHEMAS
        raise RuntimeError(
            f"missing loose JSON schema bridge for {schema_kind}"
        ) from exc
    instruction = (
        "LOOSE JSON SCHEMA CONTRACT: return exactly one valid JSON object "
        "matching the complete JSON Schema below. Use every required field "
        "exactly once; do not rename fields, merge a reason into a preference "
        "field, add fields, or include prose outside the JSON object.\n"
        + canonical_json(schema.model_json_schema())
    )
    if not normalized or normalized[-1].get("role") != "user":
        raise RuntimeError(
            "loose JSON schema bridge requires a final user message"
        )
    normalized[-1] = {
        **normalized[-1],
        "content": str(normalized[-1]["content"]).rstrip()
        + "\n\n"
        + instruction,
    }
    return normalized


def build_call_plan(
    *,
    pairwise_rows: Sequence[Mapping[str, Any]],
    equivalence_rows: Sequence[Mapping[str, Any]],
    gpt_anchor_pair_ids: set[str],
    endpoints: Mapping[str, Endpoint],
    prices: Mapping[str, Mapping[str, float]],
    seed: int,
    input_token_safety_factor: float = 1.25,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for index, row in enumerate(
        sorted(
            pairwise_rows,
            key=lambda item: (
                str(item["pair_id"]),
                int(item["order_variant"]),
            ),
        )
    ):
        record_ids = {
            "pair_id": str(row["pair_id"]),
            "state_id": str(row["state_id"]),
            "order_variant": int(row["order_variant"]),
        }
        for endpoint_key in ("qwen_nonthinking", "qwen_thinking"):
            _add_call(
                plan,
                condition=f"{endpoint_key}_pairwise",
                endpoint_key=endpoint_key,
                endpoint=endpoints[endpoint_key],
                price=prices[endpoint_key],
                schema_kind="pairwise",
                messages=row["messages"],
                record_ids=record_ids,
                max_output_tokens=PAIRWISE_MAX_OUTPUT_TOKENS,
                seed=seed + index,
                input_token_safety_factor=input_token_safety_factor,
            )
        if str(row["pair_id"]) in gpt_anchor_pair_ids:
            _add_call(
                plan,
                condition="gpt_high_quality_pairwise_anchor",
                endpoint_key="gpt_anchor",
                endpoint=endpoints["gpt_anchor"],
                price=prices["gpt_anchor"],
                schema_kind="pairwise",
                messages=row["messages"],
                record_ids=record_ids,
                max_output_tokens=PAIRWISE_MAX_OUTPUT_TOKENS,
                seed=seed + index,
                input_token_safety_factor=input_token_safety_factor,
            )
    for index, row in enumerate(
        sorted(equivalence_rows, key=lambda item: str(item["regime"]))
    ):
        record_ids = {
            "equivalence_id": str(row["equivalence_id"]),
            "state_id": str(row["state_id"]),
            "regime": str(row["regime"]),
        }
        for schema_kind, messages_key, maximum in (
            ("combined", "combined_messages", COMBINED_MAX_OUTPUT_TOKENS),
            ("quality", "quality_messages", QUALITY_MAX_OUTPUT_TOKENS),
            ("risk", "risk_messages", RISK_MAX_OUTPUT_TOKENS),
        ):
            condition = {
                "combined": "qwen_nonthinking_combined",
                "quality": "qwen_nonthinking_split_quality",
                "risk": "qwen_nonthinking_split_risk",
            }[schema_kind]
            _add_call(
                plan,
                condition=condition,
                endpoint_key="qwen_nonthinking",
                endpoint=endpoints["qwen_nonthinking"],
                price=prices["qwen_nonthinking"],
                schema_kind=schema_kind,
                messages=row[messages_key],
                record_ids=record_ids,
                max_output_tokens=maximum,
                seed=seed + 100 + index,
                input_token_safety_factor=input_token_safety_factor,
            )
    plan = sorted(
        plan,
        key=lambda row: (
            str(row["condition"]),
            canonical_json(row["record_ids"]),
        ),
    )
    if len(plan) != 91 or len(
        {str(row["physical_call_key"]) for row in plan}
    ) != 91:
        raise RuntimeError("qualification plan must contain 91 unique calls")
    if sum(
        row["condition"] == "gpt_high_quality_pairwise_anchor"
        for row in plan
    ) != GPT_ANCHOR_STATE_COUNT * 2:
        raise RuntimeError("GPT anchor plan coverage drifted")
    return plan


def select_loose_schema_compatibility_pilot(
    plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select one deterministic real call for every loose-schema shape."""

    targets = (
        ("qwen_nonthinking", "pairwise"),
        ("qwen_thinking", "pairwise"),
        ("qwen_nonthinking", "combined"),
        ("qwen_nonthinking", "quality"),
        ("qwen_nonthinking", "risk"),
    )
    selected: list[dict[str, Any]] = []
    for endpoint_key, schema_kind in targets:
        matches = [
            dict(row)
            for row in plan
            if row["endpoint_key"] == endpoint_key
            and row["schema_kind"] == schema_kind
        ]
        if not matches:
            raise RuntimeError(
                "loose-schema compatibility pilot lacks "
                f"{endpoint_key}/{schema_kind}"
            )
        selected.append(matches[0])
    if len({row["physical_call_key"] for row in selected}) != len(targets):
        raise RuntimeError(
            "loose-schema compatibility pilot call keys are not unique"
        )
    return selected


def build_cost_estimate(
    *,
    plan: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    code_manifest: Mapping[str, Any],
    endpoint_contract_path: Path,
    tracked_contract_path: Path,
    packet_dir: Path,
    maximum_attempts: int,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    carry_forward: Mapping[str, Any],
) -> dict[str, Any]:
    carried_call_keys = set(carry_forward["carried_call_keys"])
    pending_plan = [
        row
        for row in plan
        if str(row["physical_call_key"]) not in carried_call_keys
    ]
    one_attempt = math.fsum(
        float(row["maximum_single_attempt_cost_usd"]) for row in pending_plan
    )
    payload = {
        "protocol": "pm-v1.5-low-budget-judge-qualification-cost-v2",
        "stage": STAGE,
        "qualification_contract": dict(contract),
        "qualification_contract_sha256": str(contract["contract_sha256"]),
        "endpoint_contract_file_sha256": sha256_file(
            endpoint_contract_path
        ),
        "tracked_contract_file_sha256": sha256_file(
            tracked_contract_path
        ),
        "packet_files": {
            name: sha256_file(packet_dir / name)
            for name in (
                "qualification_contract.json",
                "qualification_pairs.jsonl",
                "ordered_pairwise_prompts.jsonl",
                "split_equivalence_prompts.jsonl",
                "gpt_anchor_pair_ids.json",
                "human_blind_packet.jsonl",
                "human_annotation_template.jsonl",
            )
        },
        "code_manifest": dict(code_manifest),
        "logical_calls": len(plan),
        "carried_forward_logical_calls": len(carried_call_keys),
        "new_logical_calls": len(pending_plan),
        "carry_forward": {
            key: value
            for key, value in carry_forward.items()
            if key not in {"carried_call_keys", "terminal_rows"}
        }
        | {
            "carried_call_keys_sha256": sha256_text(
                canonical_json(sorted(carried_call_keys))
            )
        },
        "maximum_physical_attempts_per_logical_call": maximum_attempts,
        "maximum_physical_attempts": len(pending_plan) * maximum_attempts,
        "logical_single_attempt_cost_usd": one_attempt,
        "maximum_cost_usd": one_attempt * maximum_attempts,
        "maximum_input_tokens_per_call_est": max(
            (int(row["input_tokens_est"]) for row in pending_plan),
            default=0,
        ),
        "call_plan_sha256": sha256_text(canonical_json(list(plan))),
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
        "api_clients_created": 0,
        "api_calls_made": 0,
        "training_labels_created": False,
    }
    checks = {
        "api_calls": payload["maximum_physical_attempts"]
        <= int(max_api_calls),
        "estimated_cost_usd": payload["maximum_cost_usd"]
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": payload[
            "maximum_input_tokens_per_call_est"
        ]
        <= int(max_input_tokens_per_call),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_low_budget_judge_qualification_packet_v2",
    )
    parser.add_argument(
        "--endpoint-contract",
        type=Path,
        default=ROOT / "configs/pm_v1_5_judge_qualification_v2.json",
    )
    parser.add_argument(
        "--tracked-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_qualification_v2.json",
    )
    parser.add_argument(
        "--human-anchor-binding",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/low_budget_judge_human_anchor_v1.json",
    )
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument(
        "--delta-carry-forward-from",
        type=Path,
        help=(
            "Prior qualification directory. Only exact physical-call "
            "identities with real SUCCEEDED ledger rows are inherited."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-api-calls", type=int, default=500)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument(
        "--loose-schema-compatibility-pilot",
        action="store_true",
        help=(
            "Run only five deterministic train calls covering every Qwen "
            "loose-schema request shape; produces no qualification verdict."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    endpoint_contract = read_json(args.endpoint_contract)
    qualification_pairs = _rows(
        args.packet_dir / "qualification_pairs.jsonl"
    )
    human_packet = _rows(args.packet_dir / "human_blind_packet.jsonl")
    human_anchor = require_tracked_human_anchor(
        root=ROOT,
        binding_path=args.human_anchor_binding,
        pairs=qualification_pairs,
        human_packet=human_packet,
    )
    prepared_contract = read_json(
        args.packet_dir / "qualification_contract.json"
    )
    expected_contract = qualification_contract_record(
        source_lineage=dict(prepared_contract["source_lineage"]),
        endpoint_contract={
            **endpoint_contract,
            "file_sha256": sha256_file(args.endpoint_contract),
            "content_sha256": sha256_text(canonical_json(endpoint_contract)),
        },
        human_anchor=human_anchor,
    )
    if prepared_contract != expected_contract:
        raise RuntimeError("prepared qualification contract drifted from code")
    if read_json(args.tracked_contract) != prepared_contract:
        raise RuntimeError(
            "prepared qualification contract does not match tracked freeze"
        )
    pairwise_rows = _rows(
        args.packet_dir / "ordered_pairwise_prompts.jsonl"
    )
    equivalence_rows = _rows(
        args.packet_dir / "split_equivalence_prompts.jsonl"
    )
    gpt_anchor_pair_ids = set(
        read_json(args.packet_dir / "gpt_anchor_pair_ids.json")["pair_ids"]
    )
    endpoints, prices = _endpoint_records(endpoint_contract)
    request_contract = dict(endpoint_contract["request_contract"])
    maximum_attempts = int(
        request_contract["maximum_physical_attempts_per_logical_call"]
    )
    plan = build_call_plan(
        pairwise_rows=pairwise_rows,
        equivalence_rows=equivalence_rows,
        gpt_anchor_pair_ids=gpt_anchor_pair_ids,
        endpoints=endpoints,
        prices=prices,
        seed=int(request_contract["seed"]),
    )
    if args.loose_schema_compatibility_pilot:
        if args.delta_carry_forward_from is not None:
            raise RuntimeError(
                "loose-schema compatibility pilot may not carry prior results"
            )
        plan = select_loose_schema_compatibility_pilot(plan)
    carry_forward = _load_delta_carry_forward(
        source_dir=args.delta_carry_forward_from,
        plan=plan,
        maximum_attempts=maximum_attempts,
    )
    code_paths = {
        "runner": Path(__file__).resolve(),
        "preparer": ROOT
        / "scripts/v1_5/21q_prepare_low_budget_judge_qualification_v1_5.py",
        "qualification": ROOT
        / "src/metacom_pm/v1_5_judge_qualification.py",
        "qualification_analysis": ROOT
        / "src/metacom_pm/v1_5_judge_qualification_analysis.py",
        "api": ROOT / "src/metacom_pm/api.py",
        "attempt_ledger": ROOT / "src/metacom_pm/attempt_ledger.py",
        "bounded_retry": ROOT / "src/metacom_pm/bounded_retry.py",
    }
    code_manifest = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(code_paths.items())
    }
    estimate = build_cost_estimate(
        plan=plan,
        contract=prepared_contract,
        code_manifest=code_manifest,
        endpoint_contract_path=args.endpoint_contract,
        tracked_contract_path=args.tracked_contract,
        packet_dir=args.packet_dir,
        maximum_attempts=maximum_attempts,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        carry_forward=carry_forward,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _persist_rows_exact(args.out_dir / "call_plan.jsonl", plan)
    _persist_exact(args.out_dir / "cost_estimate.json", estimate)
    if estimate["budget_gate"]["status"] != "PASS":
        raise RuntimeError("qualification dry-run budget gate failed")
    if args.dry_run:
        summary = {
            "status": "DRY_RUN_COMPLETE_ZERO_API",
            "stage": STAGE,
            "cost_estimate_sha256": estimate["cost_estimate_sha256"],
            "call_plan_sha256": estimate["call_plan_sha256"],
            "logical_calls": len(plan),
            "carried_forward_logical_calls": estimate[
                "carried_forward_logical_calls"
            ],
            "new_logical_calls": estimate["new_logical_calls"],
            "maximum_physical_attempts": estimate[
                "maximum_physical_attempts"
            ],
            "logical_single_attempt_cost_usd": estimate[
                "logical_single_attempt_cost_usd"
            ],
            "maximum_cost_usd": estimate["maximum_cost_usd"],
            "api_clients_created": 0,
            "api_calls_made": 0,
            "training_labels_created": False,
            "compatibility_pilot": bool(
                args.loose_schema_compatibility_pilot
            ),
        }
        _persist_exact(args.out_dir / "summary.json", summary)
        print(canonical_json(summary))
        return

    identity = str(estimate["cost_estimate_sha256"])
    if str(args.accept_cost_estimate_sha256 or "") != identity:
        raise RuntimeError("accepted qualification identity does not match dry-run")
    pm_config = read_json(args.pm_v1_5_config) if (
        args.pm_v1_5_config.suffix == ".json"
    ) else None
    if pm_config is None:
        from metacom_pm.config import load_config

        pm_config = load_config(args.pm_v1_5_config)
    require_paid_run_release(
        pm_config,
        config_path=args.pm_v1_5_config,
        stage=STAGE,
        run=True,
        run_identity=identity,
    )
    ledger_path = args.out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path, overwrite=False, stage=STAGE
    )
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=STAGE,
        expected_calls={
            str(row["physical_call_key"]): maximum_attempts for row in plan
        },
        maximum_total_attempts=(
            int(args.max_api_calls)
            + len(carry_forward["carried_call_keys"])
        ),
    )
    for row in plan:
        call_key = str(row["physical_call_key"])
        if (
            call_key not in carry_forward["carried_call_keys"]
            or ledger.succeeded(call_key)
        ):
            continue
        terminal = carry_forward["terminal_rows"][call_key]
        reservation = ledger.reserve(
            call_key,
            record_ids={
                **dict(row["record_ids"]),
                "condition": row["condition"],
            },
            prompt_sha256=str(row["prompt_sha256"]),
        )
        ledger.finish(
            reservation,
            succeeded=True,
            request_hash=terminal.get("request_hash"),
            usage=terminal.get("usage"),
            error=None,
            result=terminal.get("result"),
            metadata={
                "carried_forward": True,
                "carry_forward_mechanism": carry_forward["mechanism"],
                "source_directory": carry_forward["source_directory"],
                "source_call_plan_sha256": carry_forward[
                    "source_call_plan_sha256"
                ],
                "source_ledger_sha256": carry_forward[
                    "source_ledger_sha256"
                ],
            },
        )
    clients: dict[str, Any] = {}
    isolated_schema_failures: list[str] = []
    try:
        for endpoint_key, endpoint in endpoints.items():
            if any(
                str(row["endpoint_key"]) == endpoint_key
                and not ledger.succeeded(str(row["physical_call_key"]))
                for row in plan
            ):
                clients[endpoint_key] = make_client(endpoint)
        for row in plan:
            call_key = str(row["physical_call_key"])
            if ledger.succeeded(call_key):
                continue
            endpoint_key = str(row["endpoint_key"])
            schema = SCHEMAS[str(row["schema_kind"])]

            def call_fn(row=row, endpoint_key=endpoint_key, schema=schema):
                return clients[endpoint_key].chat(
                    list(row["messages"]),
                    temperature=0.0,
                    max_tokens=int(row["max_output_tokens"]),
                    seed=int(row["seed"]),
                    response_schema=schema,
                    retries=1,
                )

            try:
                reservation, result, parsed = execute_with_bounded_retry(
                    ledger,
                    call_key,
                    record_ids={
                        **dict(row["record_ids"]),
                        "condition": row["condition"],
                    },
                    prompt_sha256=str(row["prompt_sha256"]),
                    call_fn=call_fn,
                    max_provider_output_attempts=maximum_attempts,
                    backoff_seconds=BACKOFF_SECONDS,
                )
            except StructuredOutputValidationError:
                # A loose-JSON endpoint can return valid JSON that violates
                # the requested schema. Preserve the failed ledger row and
                # continue so one row cannot erase the rest of the diagnostic
                # matrix. This remains a missing result and can never be
                # aggregated into qualification or training labels.
                isolated_schema_failures.append(call_key)
                continue
            usage = require_reported_usage(result.usage, stage=STAGE)
            if int(usage["prompt_tokens"]) > int(row["input_tokens_est"]):
                raise RuntimeError(
                    "reported input usage exceeds the frozen qualification bound"
                )
            assert parsed is not None
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash=result.request_hash,
                usage=usage,
                error=None,
                result={"parsed": parsed.model_dump(mode="json")},
            )
    finally:
        for client in clients.values():
            client.close()

    results: list[dict[str, Any]] = []
    for row in plan:
        call_key = str(row["physical_call_key"])
        if not ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key) or {}
        results.append(
            {
                "condition": row["condition"],
                "endpoint_key": row["endpoint_key"],
                "judge_family": row["judge_family"],
                "judge_model": row["judge_model"],
                "schema_kind": row["schema_kind"],
                "record_ids": row["record_ids"],
                "request_hash": terminal.get("request_hash"),
                "usage": terminal.get("usage"),
                "parsed": dict(terminal.get("result") or {}).get("parsed"),
            }
        )
    write_jsonl(args.out_dir / "qualification_results.jsonl", results)
    complete = len(results) == len(plan)
    write_json(
        args.out_dir / "summary.json",
        {
            "status": (
                "COMPLETE_LOOSE_SCHEMA_COMPATIBILITY_PILOT_NO_VERDICT"
                if complete and args.loose_schema_compatibility_pilot
                else "COMPLETE_QUALIFICATION_RESULTS_NO_TRAINING_LABELS"
                if complete
                else "INCOMPLETE_LOOSE_SCHEMA_COMPATIBILITY_PILOT_NO_VERDICT"
                if args.loose_schema_compatibility_pilot
                else "INCOMPLETE_NONREPORTABLE_MATRIX"
            ),
            "cost_estimate_sha256": identity,
            "logical_calls": len(plan),
            "completed_calls": len(results),
            "failed_calls": len(plan) - len(results),
            "isolated_schema_validation_failures": len(
                isolated_schema_failures
            ),
            "training_labels_created": False,
            "compatibility_pilot": bool(
                args.loose_schema_compatibility_pilot
            ),
            "requires_separate_preoutcome_frozen_aggregation": (
                complete and not args.loose_schema_compatibility_pilot
            ),
        },
    )
    if not complete:
        raise RuntimeError("qualification matrix is incomplete")


if __name__ == "__main__":
    main()
