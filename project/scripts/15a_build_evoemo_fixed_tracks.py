#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

from metacom_pm.config import load_config, endpoint_from_config
from metacom_pm.evoemo import (
    build_fixed_seeker_tracks_v22,
    fixed_seeker_cost_planning_contract,
    persist_fixed_seeker_tracks_v22_dry_run,
    plan_fixed_seeker_tracks_v22,
)
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed PM-v2.2 fixed-seeker generation: dry-run and exact "
            "hash acceptance are mandatory before any client is created."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment.yaml')
    parser.add_argument(
        '--pm-v2-config',
        type=Path,
        default=ROOT / 'configs/pm_v2.yaml',
    )
    parser.add_argument(
        '--seeker-endpoint',
        help=(
            "Optional assertion; must equal the endpoint alias frozen in "
            "fixed_seeker_generation_treatment."
        ),
    )
    parser.add_argument('--simulator-id', default='seeker_main')
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=ROOT / 'outputs/evoemo_fixed_tracks_v22',
    )
    parser.add_argument('--max-turns', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--max-scenarios', type=int)
    parser.add_argument('--max-api-calls', type=int, required=True)
    parser.add_argument('--max-estimated-usd', type=float, required=True)
    parser.add_argument(
        '--max-input-tokens-per-call', type=int, required=True
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--accepted-dry-run-sha256',
        help=(
            "Required with --run; copy the exact hash printed by the matching "
            "--dry-run."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pm_v2_config = load_config(args.pm_v2_config)
    raw_contract = pm_v2_config.get('fixed_seeker_generation_treatment')
    if not isinstance(raw_contract, dict):
        raise RuntimeError(
            "pm_v2.yaml lacks fixed_seeker_generation_treatment"
        )
    contract = FixedSeekerGenerationContract.from_mapping(raw_contract)
    raw_cost_planning = pm_v2_config.get('fixed_seeker_cost_planning')
    if not isinstance(raw_cost_planning, dict):
        raise RuntimeError('pm_v2.yaml lacks fixed_seeker_cost_planning')
    cost_planning = fixed_seeker_cost_planning_contract(raw_cost_planning)
    if (
        args.seeker_endpoint is not None
        and args.seeker_endpoint != contract.seeker_endpoint
    ):
        raise RuntimeError(
            "--seeker-endpoint differs from the frozen fixed-seeker contract"
        )
    endpoint = endpoint_from_config(config, contract.seeker_endpoint)
    seeds = args.seeds
    if seeds is None:
        seeds = [int(x) for x in (config.get('protocol') or {}).get('robustness_seeds', [])]
    if not seeds:
        raise RuntimeError("configs/experiment.yaml protocol.robustness_seeds is empty")
    if args.max_scenarios is not None:
        raise RuntimeError(
            "--max-scenarios is for non-reportable debugging only. "
            "Do not include a subset fixed-track file in study freeze."
        )

    estimate, call_plan = plan_fixed_seeker_tracks_v22(
        ROOT / 'data/external/evo_emo.json',
        seeker_endpoint=endpoint,
        contract=contract,
        cost_planning=cost_planning,
        simulator_id=args.simulator_id,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        max_turns=args.max_turns,
        seeds=seeds,
        max_scenarios=args.max_scenarios,
    )
    if args.dry_run:
        if args.accepted_dry_run_sha256 is not None:
            raise RuntimeError(
                "--accepted-dry-run-sha256 is only valid with --run"
            )
        disposition = persist_fixed_seeker_tracks_v22_dry_run(
            args.out_dir,
            estimate,
            call_plan,
            overwrite=args.overwrite,
        )
        print(json.dumps({
            **estimate,
            "dry_run_disposition": disposition,
            "api_clients_created": 0,
        }, ensure_ascii=False, indent=2))
        if estimate['budget_gate']['status'] != 'PASS':
            raise RuntimeError(
                'fixed-seeker dry-run budget gate failed; do not authorize --run'
            )
        return
    if not args.accepted_dry_run_sha256:
        raise RuntimeError(
            "--run requires --accepted-dry-run-sha256 from the matching dry run"
        )
    print(json.dumps(build_fixed_seeker_tracks_v22(
        ROOT / 'data/external/evo_emo.json',
        args.out_dir,
        seeker_endpoint=endpoint,
        contract=contract,
        cost_planning=cost_planning,
        simulator_id=args.simulator_id,
        accepted_dry_run_sha256=args.accepted_dry_run_sha256,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        max_turns=args.max_turns,
        seeds=seeds,
        max_scenarios=args.max_scenarios,
        overwrite=args.overwrite,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
