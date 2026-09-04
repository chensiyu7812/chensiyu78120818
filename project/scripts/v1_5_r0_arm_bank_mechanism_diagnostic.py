#!/usr/bin/env python3
"""Third arm for the V1 bank-mechanism diagnostic: same memory, R0 (no strategy).

This is an ad-hoc extension, not part of the PM-v2.2 gated pipeline. It reuses
`select_user_balanced_rs_turns` from `posthoc_v1_bank_probe.py` with the same
turns path / sample size / sample seed / source condition as the original
diagnostic, so it reproduces the identical 40 units. For each unit it builds
the same state and selected memory, but generates a response with an empty
strategy-card list (R0) instead of retrieving from either Strategy Bank.

Purpose: separate "is the bank the problem" (legacy vs full, already tested)
from "is turning on strategy retrieval itself the problem" (RS vs R0, tested
here). Descriptive/mechanistic only, not confirmatory, not a PM-v2 gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import MemoryItem
from metacom_pm.io import append_jsonl, utc_now, write_json
from metacom_pm.posthoc_v1_bank_probe import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    GENERATOR_MAX_OUTPUT_TOKENS,
    GENERATOR_TEMPERATURE,
    _prompt_state,
    select_user_balanced_rs_turns,
)
from metacom_pm.prompts import SELECTIVE_ESMEM_SYSTEM, generation_messages

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--endpoint", default="generator")
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "v1_5_bank_mechanism_diagnostic")
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--max-estimated-usd", type=float, required=True)
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_config(args.config)
    endpoint = endpoint_from_config(experiment_config, args.endpoint)

    samples, _ = select_user_balanced_rs_turns(
        args.turns,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        source_condition="pm",
    )

    if len(samples) > args.max_api_calls:
        raise RuntimeError(
            f"{len(samples)} R0 generations exceed --max-api-calls={args.max_api_calls}"
        )

    planned: list[dict[str, Any]] = []
    estimated_usd = 0.0
    for candidate in samples:
        state = _prompt_state(candidate)
        memories = [MemoryItem.model_validate(row) for row in candidate["selected_memory"]]
        messages = generation_messages(
            state, memories, [], system_prompt=SELECTIVE_ESMEM_SYSTEM
        )
        approx_input_tokens = sum(len(m["content"]) for m in messages) / 4.0
        estimated_usd += (
            approx_input_tokens / 1_000_000 * args.input_usd_per_million_tokens
            + GENERATOR_MAX_OUTPUT_TOKENS / 1_000_000 * args.output_usd_per_million_tokens
        )
        planned.append({"pair_id": candidate["unit_id"], "messages": messages})

    plan_summary = {
        "status": "READY",
        "condition": "r0_no_strategy",
        "planned_calls": len(planned),
        "maximum_estimated_usd": round(estimated_usd, 6),
    }
    if estimated_usd > args.max_estimated_usd:
        raise RuntimeError(
            f"estimated cost ${estimated_usd:.4f} exceeds --max-estimated-usd={args.max_estimated_usd}"
        )
    if args.dry_run:
        print(plan_summary)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(endpoint)
    out_path = args.out_dir / "r0_generations.jsonl"
    raw_log_path = args.out_dir / "r0_raw_api_calls.jsonl"

    for candidate, item in zip(samples, planned):
        generator_seed = int(candidate["seed"]) + int(candidate["turn_index"])
        result, _ = client.chat(
            item["messages"],
            temperature=GENERATOR_TEMPERATURE,
            max_tokens=GENERATOR_MAX_OUTPUT_TOKENS,
            seed=generator_seed,
            response_schema=None,
            retries=3,
        )
        require_reported_usage(result.usage, stage="v1_5_r0_arm_generation")
        append_jsonl(
            raw_log_path,
            {
                "timestamp": utc_now(),
                "pair_id": item["pair_id"],
                "model": endpoint.model,
                "usage": result.usage,
                "normalized_finish_reason": result.normalized_finish_reason,
            },
        )
        append_jsonl(
            out_path,
            {
                "pair_id": item["pair_id"],
                "bank_condition": "r0_no_strategy",
                "response": result.text,
                "normalized_finish_reason": result.normalized_finish_reason,
                "usage": result.usage,
            },
        )

    write_json(
        args.out_dir / "r0_generation_summary.json",
        {
            "status": "COMPLETE",
            "condition": "r0_no_strategy",
            "confirmatory": False,
            "n_generated": len(planned),
            "note": (
                "Same 40 units, same memory, same generator/temperature/cap as "
                "the legacy/full bank arms, but with an empty strategy-card "
                "list (no RS). Descriptive/mechanistic only."
            ),
        },
    )
    print({"status": "COMPLETE", "n_generated": len(planned)})


if __name__ == "__main__":
    main()
