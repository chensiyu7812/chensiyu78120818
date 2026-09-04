#!/usr/bin/env python3
"""Preflight or run the resumable minimum RS clean-pair generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_text,
    utc_now,
    write_json,
)
from metacom_pm.v1_5_mvp_rs_pilot import (
    validate_minimum_rs_execution_inputs,
)


ROOT = Path(__file__).resolve().parents[2]
MVP_RESUMABLE_EXECUTION_PROTOCOL = (
    "pm-v1.5-resumable-development-execution-v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1_execution",
    )
    parser.add_argument("--max-new-calls", type=int, default=98)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    contract = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, contract.generator_endpoint)
    generator_identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }
    report = read_json(args.plan_dir / "plan_report.json")
    call_rows = [
        dict(row) for row in iter_jsonl(args.plan_dir / "call_plan.jsonl")
    ]
    human_path = args.plan_dir / "human_review_binding.json"
    human_binding = read_json(human_path) if human_path.exists() else None
    preflight = validate_minimum_rs_execution_inputs(
        report=report,
        call_rows=call_rows,
        generation_contract=contract,
        generator_identity=generator_identity,
        human_review_binding=human_binding,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "execution_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if not preflight["ready"]:
        raise RuntimeError(
            "RS generation is not ready: "
            + canonical_json(preflight)
        )
    if args.max_new_calls < 1:
        raise ValueError("--max-new-calls must be positive")

    outcomes_path = args.out_dir / "generation_outcomes.jsonl"
    existing: dict[tuple[str, str], dict] = {}
    if outcomes_path.exists():
        for row in iter_jsonl(outcomes_path):
            key = (str(row["pair_id"]), str(row["arm"]))
            if key in existing:
                raise RuntimeError("generation outcomes repeat pair_id+arm")
            existing[key] = dict(row)
    pending: list[dict] = []
    for row in call_rows:
        key = (str(row["pair_id"]), str(row["arm"]))
        prior = existing.get(key)
        if prior is not None:
            if prior.get("prompt_sha256") != row.get("prompt_sha256"):
                raise RuntimeError("completed outcome prompt differs from current plan")
            continue
        pending.append(row)
    if len(pending) > args.max_new_calls:
        raise RuntimeError(
            f"{len(pending)} calls remain but --max-new-calls={args.max_new_calls}"
        )

    client = make_client(endpoint)
    completed_now = 0
    try:
        for row in pending:
            generation = dict(row["generation"])
            result, _ = client.chat(
                list(row["messages"]),
                temperature=float(generation["temperature"]),
                max_tokens=int(generation["max_output_tokens"]),
                seed=int(generation["seed"]),
                response_schema=None,
                retries=3,
            )
            completion_error = contract.completion_gate_error(
                normalized_finish_reason=result.normalized_finish_reason,
                provider_finish_reason=result.provider_finish_reason,
            )
            if completion_error is not None:
                raise RuntimeError(completion_error)
            response = contract.normalize_output(result.text)
            if not response:
                raise RuntimeError("supporter generator returned an empty response")
            usage = require_reported_usage(
                result.usage,
                stage="minimum_rs_clean_pair_generation",
            )
            append_jsonl(
                outcomes_path,
                {
                    "protocol": "pm-v1.5-minimum-rs-generation-outcome-v1",
                    "pair_id": row["pair_id"],
                    "state_id": row["state_id"],
                    "card_id": row["card_id"],
                    "user_id": row["user_id"],
                    "boundary_cue": row["boundary_cue"],
                    "arm": row["arm"],
                    "action_id": row["action_id"],
                    "selected_strategy_card_id": row[
                        "selected_strategy_card_id"
                    ],
                    "selected_strategy_family": row[
                        "selected_strategy_family"
                    ],
                    "response": response,
                    "prompt_sha256": row["prompt_sha256"],
                    "messages_sha256": sha256_text(
                        canonical_json(row["messages"])
                    ),
                    "request_hash": result.request_hash,
                    "provider_finish_reason": result.provider_finish_reason,
                    "normalized_finish_reason": result.normalized_finish_reason,
                    "usage": usage,
                    "latency_ms": result.latency_ms,
                    "model": endpoint.model,
                    "model_family": endpoint.family,
                    "completed_at": utc_now(),
                },
            )
            completed_now += 1
    finally:
        client.close()

    all_outcomes = [dict(row) for row in iter_jsonl(outcomes_path)]
    summary = {
        "protocol": MVP_RESUMABLE_EXECUTION_PROTOCOL,
        "status": (
            "COMPLETE"
            if len(all_outcomes) == len(call_rows)
            else "PARTIAL_RESUMABLE"
        ),
        "planned_calls": len(call_rows),
        "completed_calls": len(all_outcomes),
        "completed_now": completed_now,
        "remaining_calls": len(call_rows) - len(all_outcomes),
        "one_shot_execution_required": False,
        "resume_key": "pair_id+arm",
        "api_calls_made_this_run": completed_now,
    }
    write_json(args.out_dir / "generation_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
