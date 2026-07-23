#!/usr/bin/env python3
"""Audit fixed-seeker V3 formal/freeze readiness without any API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import require_content_addressed_attestation
from metacom_pm.evoemo import FIXED_SEEKER_V23_STAGE
from metacom_pm.fixed_seeker_contract import require_fixed_seeker_v3_sidecar_contract
from metacom_pm.io import read_json, write_json
from metacom_pm.v1_5_fixed_seeker_readiness import (
    assess_fixed_seeker_v3_promotion,
    load_consumer_sources,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-seeker-contract",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5_fixed_seeker_v3.json",
        help=(
            "V3 sidecar contract (configs/pm_v1_5.yaml itself deliberately "
            "stays on the historical V2 treatment; editing it directly was "
            "shown to invalidate the already-qualified V8.19.2 lineage via a "
            "pm_v1_5_config hash mismatch)."
        ),
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_fixed_tracks_v1_5_v3_pilot_execution_candidate",
    )
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=ROOT / "outputs" / "evoemo_fixed_tracks_v1_5_v3_formal_candidate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "fixed_seeker_v3_promotion_readiness.json",
    )
    args = parser.parse_args()

    pilot = args.pilot_dir
    require_content_addressed_attestation(
        pilot / "artifact_attestation.json",
        required_stage=FIXED_SEEKER_V23_STAGE,
        relocated_inputs={
            "evoemo": ROOT / "data" / "external" / "evo_emo.json",
            "run_manifest": pilot / "run_manifest.json",
            "cost_estimate": pilot / "cost_estimate.json",
            "call_plan": pilot / "call_plan.jsonl",
        },
        relocated_outputs={
            "tracks": pilot / "fixed_seeker_tracks.jsonl",
            "raw_calls": pilot / "raw_seeker_calls.jsonl",
            "physical_attempt_ledger": pilot / "physical_attempt_ledger.jsonl",
            "summary": pilot / "summary.json",
        },
    )
    sidecar_contract = require_fixed_seeker_v3_sidecar_contract(
        args.fixed_seeker_contract
    )
    report = assess_fixed_seeker_v3_promotion(
        fixed_seeker_sidecar_contract=sidecar_contract.payload(),
        pilot_summary=read_json(pilot / "summary.json"),
        pilot_attestation=read_json(pilot / "artifact_attestation.json"),
        consumer_source_text=load_consumer_sources(ROOT),
        formal_bundle_exists=(
            (args.formal_dir / "fixed_seeker_tracks.jsonl").is_file()
            and (args.formal_dir / "artifact_attestation.json").is_file()
        ),
    )
    write_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
