from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review_v9 import (
    run_v9_paired_generation,
    validate_v9_dry_run,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budget-gated V9 paired R0/RS smoke generation."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--v9-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v9",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--max-api-calls", type=int, default=18)
    parser.add_argument("--max-estimated-usd", type=float, default=0.02)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            validate_v9_dry_run(
                out_dir=args.v9_dir,
                max_api_calls=args.max_api_calls,
                max_estimated_usd=args.max_estimated_usd,
                max_input_tokens_per_call=args.max_input_tokens_per_call,
                require_existing=False,
            )
        )
        return
    if not args.accept_cost_estimate_sha256:
        raise RuntimeError("--run requires --accept-cost-estimate-sha256")
    print(
        run_v9_paired_generation(
            out_dir=args.v9_dir,
            experiment_config_path=args.experiment_config,
            accept_cost_estimate_sha256=args.accept_cost_estimate_sha256,
            max_api_calls=args.max_api_calls,
            max_estimated_usd=args.max_estimated_usd,
            max_input_tokens_per_call=args.max_input_tokens_per_call,
        )
    )


if __name__ == "__main__":
    main()
