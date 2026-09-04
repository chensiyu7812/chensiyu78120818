#!/usr/bin/env python3
"""Compile existing paid judge results into explicit non-gold weak labels.

This stage is deterministic and zero-API.  It does not open internal-test
labels or external outcomes, and it never upgrades the failed judge
instruments to automatic gold labelers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import require_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.io import canonical_json
from metacom_pm.pm_v2_contracts import PMV2Split
from metacom_pm.v1_5_weak_supervision import (
    compile_auxiliary_weak_labels,
    compile_auxiliary_train_weak_labels,
    compile_longitudinal_weak_labels,
    require_weak_supervision_contract,
    require_weak_supervision_source_hashes,
    write_weak_supervision_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "llm_weak_supervision_v1.json",
    )
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--longitudinal-states",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate"
        / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--longitudinal-outcomes",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_longitudinal_action_sweep_v8_19_2_continuation_v2_dry_run"
        / "action_outcomes.jsonl",
    )
    parser.add_argument(
        "--longitudinal-raw-judges",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_longitudinal_action_judging_v8_19_2_train_calibration_labels"
        / "judge_results.jsonl",
    )
    parser.add_argument(
        "--auxiliary-train-states",
        type=Path,
        default=ROOT
        / "data"
        / "esconv_auxiliary_v1_5"
        / "train"
        / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--auxiliary-train-outcomes",
        type=Path,
        default=ROOT
        / "outputs"
        / "esconv_auxiliary_generation_v1_5_full_train"
        / "action_outcomes.jsonl",
    )
    parser.add_argument(
        "--auxiliary-recovered-labels",
        type=Path,
        default=ROOT
        / "outputs"
        / "esconv_auxiliary_judging_v1_5_full_train"
        / "recovered_action_labels.jsonl",
    )
    parser.add_argument("--auxiliary-calibration-states", type=Path)
    parser.add_argument("--auxiliary-calibration-outcomes", type=Path)
    parser.add_argument("--auxiliary-calibration-labels", type=Path)
    parser.add_argument("--auxiliary-calibration-attestation", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_llm_weak_supervision_v1",
    )
    args = parser.parse_args()

    contract = require_weak_supervision_contract(args.contract)
    frozen_source_paths = {
        "pm_v1_5_config": args.pm_v1_5_config,
        "longitudinal_states": args.longitudinal_states,
        "longitudinal_outcomes": args.longitudinal_outcomes,
        "longitudinal_raw_judges": args.longitudinal_raw_judges,
        "auxiliary_train_states": args.auxiliary_train_states,
        "auxiliary_train_outcomes": args.auxiliary_train_outcomes,
        "auxiliary_recovered_labels": args.auxiliary_recovered_labels,
        "longitudinal_supervision_freeze": ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "longitudinal_supervision_freeze_v1.json",
        "esconv_auxiliary_absolute_instrument_freeze": ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "esconv_auxiliary_absolute_instrument_freeze_v1.json",
    }
    # Verify every source before parsing large outcome matrices.
    require_weak_supervision_source_hashes(contract, frozen_source_paths)
    calibration_args = (
        args.auxiliary_calibration_states,
        args.auxiliary_calibration_outcomes,
        args.auxiliary_calibration_labels,
        args.auxiliary_calibration_attestation,
    )
    if any(value is not None for value in calibration_args) and not all(
        value is not None for value in calibration_args
    ):
        raise RuntimeError(
            "auxiliary calibration weak supervision requires states, outcomes, "
            "labels, and attestation together"
        )
    calibration_source_paths = {}
    if all(value is not None for value in calibration_args):
        require_artifact_attestation(
            args.auxiliary_calibration_attestation,
            required_stage="esconv_auxiliary_weak_judging_full_calibration",
            required_output_paths={
                "labels": args.auxiliary_calibration_labels,
                "raw_results": (
                    args.auxiliary_calibration_attestation.parent
                    / "judge_results.jsonl"
                ),
            },
        )
        calibration_source_paths = {
            "auxiliary_calibration_states": args.auxiliary_calibration_states,
            "auxiliary_calibration_outcomes": args.auxiliary_calibration_outcomes,
            "auxiliary_calibration_labels": args.auxiliary_calibration_labels,
            "auxiliary_calibration_attestation": (
                args.auxiliary_calibration_attestation
            ),
        }
    source_paths = {**frozen_source_paths, **calibration_source_paths}
    config = load_config(args.pm_v1_5_config)
    contract_sha256 = str(contract["contract_sha256"])
    longitudinal_labels = compile_longitudinal_weak_labels(
        states_path=args.longitudinal_states,
        outcomes_path=args.longitudinal_outcomes,
        raw_judges_path=args.longitudinal_raw_judges,
        pm_config=config,
        contract=contract,
        contract_sha256=contract_sha256,
    )
    auxiliary_labels = compile_auxiliary_train_weak_labels(
        states_path=args.auxiliary_train_states,
        outcomes_path=args.auxiliary_train_outcomes,
        recovered_labels_path=args.auxiliary_recovered_labels,
        contract=contract,
        contract_sha256=contract_sha256,
    )
    auxiliary_calibration_labels = (
        compile_auxiliary_weak_labels(
            states_path=args.auxiliary_calibration_states,
            outcomes_path=args.auxiliary_calibration_outcomes,
            labels_path=args.auxiliary_calibration_labels,
            split=PMV2Split.CALIBRATION,
            contract=contract,
            contract_sha256=contract_sha256,
        )
        if calibration_source_paths
        else None
    )
    result = write_weak_supervision_bundle(
        out_dir=args.out_dir,
        contract_path=args.contract,
        source_paths=source_paths,
        longitudinal_labels=longitudinal_labels,
        auxiliary_labels=auxiliary_labels,
        auxiliary_calibration_labels=auxiliary_calibration_labels,
        code_paths={
            "compiler_module": ROOT
            / "src"
            / "metacom_pm"
            / "v1_5_weak_supervision.py",
            "runner": Path(__file__),
            "aggregation": ROOT / "src" / "metacom_pm" / "pm_v2_judging.py",
            "training": ROOT / "src" / "metacom_pm" / "pm_v2_model.py",
        },
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
