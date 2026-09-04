#!/usr/bin/env python3
"""Build isolated, non-truncated fixed seeker tracks for PM-v1.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.evoemo import (
    build_fixed_seeker_tracks_v22,
    fixed_seeker_cost_planning_contract,
    persist_fixed_seeker_tracks_v22_dry_run,
    plan_fixed_seeker_tracks_v22,
)
from metacom_pm.fixed_seeker_contract import (
    FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3,
    FixedSeekerGenerationContract,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Run the bounded V3 compatibility scope only: exactly two "
            "scenarios, one seed, and ten turns. It has its own paid stage "
            "and cannot be mistaken for the formal 102-track artifact."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--fixed-seeker-contract",
        type=Path,
        help=(
            "Optional content-addressed fixed-seeker treatment sidecar. This "
            "keeps an execution-contract repair from mutating the already "
            "qualified PM-v1.5 scientific config. The complete canonical "
            "payload is still bound into every dry-run/call identity."
        ),
    )
    parser.add_argument("--seeker-endpoint")
    parser.add_argument("--simulator-id", default="seeker_main")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "evoemo_fixed_tracks_v1_5",
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--max-estimated-usd", type=float, required=True)
    parser.add_argument("--max-input-tokens-per-call", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--accepted-dry-run-sha256")
    parser.add_argument(
        "--carry-forward-tracks-dir",
        type=Path,
        help=(
            "A prior same-contract run's --out-dir whose already-succeeded, "
            "fully-parsed seeker turns should be recovered read-only into "
            "this run's own ledger under a fresh --out-dir/identity, "
            "instead of being paid for again. Requires the prior "
            "directory's call_plan.jsonl to be byte-identical to this "
            "run's own freshly-computed plan; refuses otherwise. "
            "Cross-contract carry-forward can never match, by design."
        ),
    )
    args = parser.parse_args()

    if args.out_dir.name != "evoemo_fixed_tracks_v1_5" and not (
        args.out_dir.name.startswith("evoemo_fixed_tracks_v1_5_v2")
        or args.out_dir.name.startswith("evoemo_fixed_tracks_v1_5_v3")
    ):
        raise RuntimeError(
            "PM-v1.5 fixed seeker tracks must use an isolated "
            "evoemo_fixed_tracks_v1_5 (v1) or evoemo_fixed_tracks_v1_5_v2* "
            "(v2 retry-contract candidate/repro-check) or "
            "evoemo_fixed_tracks_v1_5_v3* (bounded-surface) directory"
        )
    if args.run and args.overwrite:
        raise RuntimeError("paid fixed-seeker runs prohibit --overwrite")
    if args.pilot:
        if args.max_scenarios != 2 or len(args.seeds or []) != 1:
            raise RuntimeError(
                "fixed-seeker V3 pilot requires --max-scenarios 2 and exactly "
                "one explicit --seeds value"
            )
        if args.max_turns != 10:
            raise RuntimeError("fixed-seeker V3 pilot requires --max-turns 10")
    elif args.max_scenarios is not None:
        raise RuntimeError(
            "subset fixed seeker tracks are debugging-only and cannot enter V1.5"
        )

    experiment = load_config(args.config)
    pm_config = load_config(args.pm_v1_5_config)
    require_paid_run_release(
        pm_config,
        config_path=args.pm_v1_5_config,
        stage=(
            "fixed_seeker_generation_compatibility_pilot"
            if args.pilot
            else "fixed_seeker_generation"
        ),
        run=bool(args.run),
        run_identity=args.accepted_dry_run_sha256,
    )
    if pm_config.get("version") != "pm-v1.5":
        raise RuntimeError("fixed seeker track builder requires PM-v1.5 config")
    raw_contract = (
        json.loads(args.fixed_seeker_contract.read_text(encoding="utf-8"))
        if args.fixed_seeker_contract is not None
        else pm_config.get("fixed_seeker_generation_treatment")
    )
    if not isinstance(raw_contract, dict):
        raise RuntimeError("PM-v1.5 config lacks fixed seeker treatment")
    contract = FixedSeekerGenerationContract.from_mapping(raw_contract)
    uses_v3_contract = (
        contract.version == FIXED_SEEKER_GENERATION_CONTRACT_VERSION_V3
    )
    uses_v3_directory = args.out_dir.name.startswith(
        "evoemo_fixed_tracks_v1_5_v3"
    )
    if uses_v3_contract != uses_v3_directory:
        raise RuntimeError(
            "fixed-seeker output directory generation must match its contract: "
            "V3 uses evoemo_fixed_tracks_v1_5_v3*, legacy contracts do not"
        )
    if args.pilot and not uses_v3_contract:
        raise RuntimeError(
            "fixed-seeker bounded-surface compatibility pilot requires the "
            "V3 contract"
        )
    cost_planning = fixed_seeker_cost_planning_contract(
        pm_config.get("fixed_seeker_cost_planning") or {}
    )
    if (
        args.seeker_endpoint is not None
        and args.seeker_endpoint != contract.seeker_endpoint
    ):
        raise RuntimeError("--seeker-endpoint differs from the frozen treatment")
    endpoint = endpoint_from_config(experiment, contract.seeker_endpoint)
    seeds = args.seeds or [
        int(value)
        for value in (experiment.get("protocol") or {}).get(
            "robustness_seeds", []
        )
    ]
    if not seeds:
        raise RuntimeError("experiment protocol robustness_seeds is empty")

    plan_args = {
        "seeker_endpoint": endpoint,
        "contract": contract,
        "cost_planning": cost_planning,
        "simulator_id": args.simulator_id,
        "max_api_calls": args.max_api_calls,
        "max_estimated_usd": args.max_estimated_usd,
        "max_input_tokens_per_call": args.max_input_tokens_per_call,
        "max_turns": args.max_turns,
        "seeds": seeds,
        "max_scenarios": args.max_scenarios if args.pilot else None,
    }
    evoemo = ROOT / "data" / "external" / "evo_emo.json"
    estimate, call_plan = plan_fixed_seeker_tracks_v22(evoemo, **plan_args)
    if args.dry_run:
        if args.accepted_dry_run_sha256 is not None:
            raise RuntimeError(
                "--accepted-dry-run-sha256 is valid only with --run"
            )
        disposition = persist_fixed_seeker_tracks_v22_dry_run(
            args.out_dir, estimate, call_plan, overwrite=args.overwrite
        )
        print(
            json.dumps(
                {
                    **estimate,
                    "dry_run_disposition": disposition,
                    "api_clients_created": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if estimate["budget_gate"]["status"] != "PASS":
            raise RuntimeError("fixed-seeker dry-run failed its budget gate")
        return

    if not args.accepted_dry_run_sha256:
        raise RuntimeError(
            "--run requires --accepted-dry-run-sha256 from the matching dry run"
        )
    result = build_fixed_seeker_tracks_v22(
        evoemo,
        args.out_dir,
        accepted_dry_run_sha256=args.accepted_dry_run_sha256,
        overwrite=False,
        carry_forward_tracks_dir=args.carry_forward_tracks_dir,
        **plan_args,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
