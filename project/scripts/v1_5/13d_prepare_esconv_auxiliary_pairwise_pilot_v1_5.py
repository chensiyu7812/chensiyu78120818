#!/usr/bin/env python3
"""Prepare a train-only balanced-order pairwise pilot (zero API calls).

This script intentionally has no ``--run`` mode and never creates an API
client.  It deterministically selects one state from each of the 24 frozen
ESConv-auxiliary train dialogues, binds the already-paid M0+R0/M0+RS
responses, creates both anonymous presentation orders, and writes a
two-family 96-call plan plus a worst-case retry-aware cost estimate.

It never opens calibration/internal-test, audit-only, gold, or judge-output
files and cannot produce training labels.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.attempt_ledger import physical_call_key
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.text import conservative_token_bound, estimate_tokens
from metacom_pm.v1_5_esconv_auxiliary_pairwise import (
    PAIRWISE_MAX_OUTPUT_TOKENS,
    PairwisePreferenceOutput,
    build_pairwise_items,
    build_pairwise_messages,
    pairwise_contract_record,
    select_train_only_pairwise_states,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE = "esconv_auxiliary_train_pairwise_measurement_pilot"
MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL = 10
JUDGE_SEED = 9271


def _load_unique(path: Path, key: str) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    values = [str(row[key]) for row in rows]
    if len(values) != len(set(values)):
        raise RuntimeError(f"{path} contains duplicate {key}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--seed-sources",
        type=Path,
        default=ROOT / "data" / "strategy" / "pm_v1_5_selected_seed_sources.jsonl",
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
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "esconv_auxiliary_pairwise_pilot_v1_5_train",
    )
    parser.add_argument("--max-api-calls", type=int, default=2000)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=8000)
    args = parser.parse_args()

    experiment = load_config(args.experiment_config)
    pm_config = load_config(args.pm_v1_5_config)
    generation_attestation_path = args.generation_dir / "artifact_attestation.json"
    require_artifact_attestation(
        generation_attestation_path,
        required_stage="action_sweep",
        required_output_paths={
            "action_outcomes": args.generation_dir / "action_outcomes.jsonl"
        },
    )
    judging = dict(pm_config["development_judging"])
    endpoint_names = [str(value) for value in judging["judge_endpoints"]]
    endpoints = [endpoint_from_config(experiment, name) for name in endpoint_names]
    if len({str(endpoint.family) for endpoint in endpoints}) != 2:
        raise RuntimeError("pairwise pilot requires exactly two distinct judge families")
    prices = {
        str(family): {
            "input": float(value["input"]),
            "output": float(value["output"]),
        }
        for family, value in dict(judging["pricing_usd_per_mtok"]).items()
    }
    if set(prices) != {str(endpoint.family) for endpoint in endpoints}:
        raise RuntimeError("pairwise pilot pricing does not match endpoint families")

    seeds = list(iter_jsonl(args.seed_sources))
    runtime_rows = _load_unique(args.train_dir / "runtime_states.jsonl", "state_id")
    state_rows = _load_unique(args.train_dir / "pm_v2_states.jsonl", "state_id")
    outcome_rows = list(iter_jsonl(args.generation_dir / "action_outcomes.jsonl"))
    selection = select_train_only_pairwise_states(
        seed_rows=seeds,
        runtime_rows=runtime_rows,
    )
    items = build_pairwise_items(
        selection=selection,
        state_rows=state_rows,
        outcome_rows=outcome_rows,
    )
    state_by_id = {str(row["state_id"]): row for row in state_rows}
    contract = pairwise_contract_record()
    response_schema = PairwisePreferenceOutput.model_json_schema()
    response_schema_sha256 = sha256_text(canonical_json(response_schema))
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])

    plan: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        messages = build_pairwise_messages(
            state=state_by_id[str(item["state_id"])],
            response_a=str(item["response_a"]),
            response_b=str(item["response_b"]),
        )
        prompt_sha256 = sha256_text(canonical_json(messages))
        base_input_tokens = estimate_tokens(canonical_json(messages))
        input_tokens = conservative_token_bound(
            canonical_json(messages), safety_factor=safety_factor
        )
        for endpoint_index, (endpoint_name, endpoint) in enumerate(
            zip(endpoint_names, endpoints)
        ):
            family = str(endpoint.family)
            # Keep the seed identical across the two presentation orders for
            # one state/family. Otherwise an AB/BA difference would confound
            # position sensitivity with provider seed sensitivity.
            state_pair_index = item_index // 2
            seed = JUDGE_SEED + state_pair_index * 10 + endpoint_index
            request_payload_sha256 = sha256_text(
                canonical_json(
                    {
                        "messages": messages,
                        "response_schema": response_schema,
                        "temperature": 0.0,
                        "max_tokens": PAIRWISE_MAX_OUTPUT_TOKENS,
                        "seed": seed,
                    }
                )
            )
            price = prices[family]
            row = {
                "state_id": str(item["state_id"]),
                "dialogue_id": str(item["dialogue_id"]),
                "pair_id": str(item["pair_id"]),
                "order_variant": int(item["order_variant"]),
                "action_a": str(item["action_a"]),
                "action_b": str(item["action_b"]),
                "judge_endpoint": endpoint_name,
                "judge_family": family,
                "judge_model": endpoint.model,
                "seed": seed,
                "prompt_sha256": prompt_sha256,
                "response_schema_sha256": response_schema_sha256,
                "request_payload_sha256": request_payload_sha256,
                "base_input_tokens_est": base_input_tokens,
                "input_tokens_est": input_tokens,
                "max_output_tokens": PAIRWISE_MAX_OUTPUT_TOKENS,
                "maximum_single_attempt_cost_usd": (
                    input_tokens / 1_000_000 * price["input"]
                    + PAIRWISE_MAX_OUTPUT_TOKENS / 1_000_000 * price["output"]
                ),
            }
            row["physical_call_key"] = physical_call_key(
                stage=STAGE,
                record_ids={
                    "state_id": row["state_id"],
                    "pair_id": row["pair_id"],
                    "judge_family": family,
                },
                prompt_sha256=prompt_sha256,
                endpoint=endpoint,
                request_parameters={
                    "temperature": 0.0,
                    "max_tokens": PAIRWISE_MAX_OUTPUT_TOKENS,
                    "seed": seed,
                    "response_schema_sha256": response_schema_sha256,
                },
            )
            plan.append(row)

    logical_cost = math.fsum(
        float(row["maximum_single_attempt_cost_usd"]) for row in plan
    )
    code_paths = {
        "preparation_script": Path(__file__).resolve(),
        "pairwise_contract": (
            ROOT / "src" / "metacom_pm" / "v1_5_esconv_auxiliary_pairwise.py"
        ),
        "api": ROOT / "src" / "metacom_pm" / "api.py",
        "attempt_ledger": ROOT / "src" / "metacom_pm" / "attempt_ledger.py",
        "bounded_retry": ROOT / "src" / "metacom_pm" / "bounded_retry.py",
    }
    code_manifest = {
        name: {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(code_paths.items())
    }
    payload = {
        "stage": STAGE,
        "contract": contract,
        "preparation_only_not_executable": True,
        "paid_runner_status": "NOT_IMPLEMENTED",
        "code_manifest": code_manifest,
        "code_manifest_sha256": sha256_text(canonical_json(code_manifest)),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
        "seed_sources_sha256": sha256_file(args.seed_sources),
        "runtime_states_sha256": sha256_file(args.train_dir / "runtime_states.jsonl"),
        "pm_v2_states_sha256": sha256_file(args.train_dir / "pm_v2_states.jsonl"),
        "generation_outcomes_sha256": sha256_file(
            args.generation_dir / "action_outcomes.jsonl"
        ),
        "generation_attestation_sha256": sha256_file(generation_attestation_path),
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
        "no_api_client_created": True,
    }
    cost_estimate = {
        **payload,
        "cost_estimate_sha256": sha256_text(canonical_json(payload)),
    }
    budget_gate = {
        "status": (
            "PASS"
            if len(plan) <= args.max_api_calls
            and cost_estimate["maximum_cost_usd"] <= args.max_estimated_usd
            and cost_estimate["maximum_input_tokens_per_call_est"]
            <= args.max_input_tokens_per_call
            else "FAIL"
        ),
        "max_api_calls": args.max_api_calls,
        "max_estimated_usd": args.max_estimated_usd,
        "max_input_tokens_per_call": args.max_input_tokens_per_call,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "selection.jsonl", selection)
    write_jsonl(args.out_dir / "pairwise_items.jsonl", items)
    write_jsonl(args.out_dir / "call_plan.jsonl", plan)
    write_json(args.out_dir / "pairwise_contract.json", contract)
    write_json(
        args.out_dir / "cost_estimate.json",
        {**cost_estimate, "budget_gate": budget_gate},
    )
    print({**cost_estimate, "budget_gate": budget_gate})


if __name__ == "__main__":
    main()
