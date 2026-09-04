#!/usr/bin/env python3
"""Plan or run the isolated post-hoc V1 conditional bank diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.posthoc_v1_bank_probe import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    persist_posthoc_v1_bank_dry_run,
    plan_posthoc_v1_bank_probe,
    run_posthoc_v1_bank_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc V1 legacy-vs-full Strategy Bank diagnostic conditional "
            "on an already-frozen PM+RS action. It does not estimate routing "
            "effects or the V1 total effect, is not confirmatory, and is never "
            "a PM-v2 gate."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--legacy-bank", type=Path, required=True)
    parser.add_argument("--full-bank", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="generator",
        help=(
            "config name for the common replay generator treatment; this does "
            "not assert historical original-V1 generator identity"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--input-token-safety-factor", type=float, default=1.25)
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--max-estimated-usd", type=float, required=True)
    parser.add_argument("--max-input-tokens-per-call", type=int, required=True)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run and args.overwrite:
        raise RuntimeError("paid diagnostic runs prohibit --overwrite")
    config = load_config(args.config)
    endpoint = endpoint_from_config(config, args.endpoint)
    estimate, samples, plan = plan_posthoc_v1_bank_probe(
        args.turns,
        args.legacy_bank,
        args.full_bank,
        endpoint=endpoint,
        input_usd_per_million_tokens=args.input_usd_per_million_tokens,
        output_usd_per_million_tokens=args.output_usd_per_million_tokens,
        input_token_safety_factor=args.input_token_safety_factor,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if not estimate["canonical_frozen_v1_inputs"]:
        raise RuntimeError(
            "formal CLI refuses noncanonical inputs; the three frozen SHA-256 "
            "bindings and source_condition=pm are mandatory"
        )
    if int(estimate["maximum_physical_api_attempts"]) > args.max_api_calls:
        raise RuntimeError("dry-run plan exceeds --max-api-calls")
    if float(estimate["maximum_estimated_usd"]) > args.max_estimated_usd:
        raise RuntimeError("dry-run estimate exceeds --max-estimated-usd")
    if any(
        int(row["conservative_input_tokens"]) > args.max_input_tokens_per_call
        for row in plan
    ):
        raise RuntimeError("a planned prompt exceeds --max-input-tokens-per-call")

    if args.dry_run:
        status = persist_posthoc_v1_bank_dry_run(
            args.out_dir,
            estimate,
            samples,
            plan,
            overwrite=args.overwrite,
        )
        print(
            {
                "status": status,
                "diagnostic_label": estimate["diagnostic_label"],
                "confirmatory": False,
                "v2_training_gate": False,
                "sample_size": estimate["sample_size"],
                "planned_calls": estimate["planned_logical_calls"],
                "maximum_estimated_usd": estimate["maximum_estimated_usd"],
                "cost_estimate_sha256": estimate["cost_estimate_sha256"],
            }
        )
        return
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "--run requires --accept-cost-estimate-sha256 from the matching dry-run"
        )
    print(
        run_posthoc_v1_bank_probe(
            args.out_dir,
            estimate,
            samples,
            plan,
            endpoint=endpoint,
            accepted_cost_estimate_sha256=args.accept_cost_estimate_sha256,
            max_api_calls=args.max_api_calls,
            max_estimated_usd=args.max_estimated_usd,
            max_input_tokens_per_call=args.max_input_tokens_per_call,
        )
    )


if __name__ == "__main__":
    main()
