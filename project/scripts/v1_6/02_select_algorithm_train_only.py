#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import joblib

from metacom_pm.config import load_config
from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text, write_json
from metacom_pm.pm_v1_6_contracts import AlgorithmCandidate, CandidateFamilyManifest, Step0Observation
from metacom_pm.pm_v1_6_internal import require_no_internal_inputs
from metacom_pm.pm_v1_6_models import OutcomeMode, OverrideConfig, PMV16OutcomeModel
from metacom_pm.pm_v1_6_router import StrongTransparentRouter
from metacom_pm.pm_v1_6_training import (
    ALGORITHM_COMPLEXITY_ORDER,
    calibrate_override_config,
    select_transparent_router_train_only,
    train_only_algorithm_competition,
)
from metacom_pm.pm_v2_contracts import ActionLabel, CompositeSpec
from metacom_pm.pm_v2_data import load_states

ROOT = Path(__file__).resolve().parents[2]


def load_step0(path: Path) -> dict[str, Step0Observation]:
    rows = [Step0Observation.model_validate(row) for row in iter_jsonl(path)]
    result = {row.state_id: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate Step-0 state")
    return result


def load_labels(path: Path) -> list[ActionLabel]:
    return [ActionLabel.model_validate(row) for row in iter_jsonl(path)]


def override_grid(raw: dict) -> list[OverrideConfig]:
    keys = [
        "uncertainty_z",
        "quality_noninferiority_margin",
        "emotional_support_noninferiority_margin",
        "absolute_risk_ceiling",
        "relative_risk_margin",
        "risk_weight",
        "cost_weight",
        "minimum_utility_lcb",
    ]
    return [
        OverrideConfig(**dict(zip(keys, values, strict=True)))
        for values in product(*(raw[key] for key in keys))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pm_v1_6.yaml")
    parser.add_argument("--train-states", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--train-step0", type=Path, required=True)
    parser.add_argument("--calibration-states", type=Path, required=True)
    parser.add_argument("--calibration-labels", type=Path, required=True)
    parser.add_argument("--calibration-step0", type=Path, required=True)
    parser.add_argument(
        "--internal-dataset-sha256",
        required=True,
        help="Precomputed SHA-256 of the sealed internal bundle manifest; contents are not read.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    require_no_internal_inputs([
        args.train_states,
        args.train_labels,
        args.train_step0,
        args.calibration_states,
        args.calibration_labels,
        args.calibration_step0,
    ])
    if len(args.internal_dataset_sha256) != 64:
        raise ValueError("internal dataset SHA-256 must be supplied without reading the bundle")
    config = load_config(args.config)
    if config.get("version") != "pm-v1.6":
        raise RuntimeError("algorithm selection requires PM-v1.6")

    train_states = load_states(args.train_states)
    calibration_states = load_states(args.calibration_states)
    if any(state.split.value != "train" for state in train_states):
        raise RuntimeError("train state file contains a non-train row")
    if any(state.split.value != "calibration" for state in calibration_states):
        raise RuntimeError("calibration state file contains a non-calibration row")
    train_labels = load_labels(args.train_labels)
    calibration_labels = load_labels(args.calibration_labels)
    train_step0 = load_step0(args.train_step0)
    calibration_step0 = load_step0(args.calibration_step0)
    spec = CompositeSpec()

    router_config, router_report = select_transparent_router_train_only(
        states=train_states,
        labels=train_labels,
        step0_by_state=train_step0,
        composite_spec=spec,
    )
    default_override = OverrideConfig()
    competition = train_only_algorithm_competition(
        states=train_states,
        labels=train_labels,
        step0_by_state=train_step0,
        router_config=router_config,
        override_config=default_override,
        composite_spec=spec,
        folds=int(config["algorithm_competition"]["user_group_folds"]),
    )
    selected = AlgorithmCandidate(competition["selected_primary"])
    mode = OutcomeMode(selected.value)
    router = StrongTransparentRouter(router_config)
    rule_actions = (
        {
            state.state_id: router.choose(train_step0[state.state_id]).action_id
            for state in train_states
        }
        if mode is OutcomeMode.RULE_RELATIVE
        else None
    )
    model = PMV16OutcomeModel.train(
        mode=mode,
        states=train_states,
        labels=train_labels,
        step0_by_state=train_step0,
        composite_spec=spec,
        rule_action_by_state=rule_actions,
    )
    calibrated_override, calibration_report = calibrate_override_config(
        model=model,
        router_config=router_config,
        states=calibration_states,
        labels=calibration_labels,
        step0_by_state=calibration_step0,
        candidates=override_grid(config["override_calibration"]["candidate_grid"]),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "pm_v1_6_policy_bundle.joblib"
    report_path = args.out_dir / "train_only_algorithm_report.json"
    manifest_path = args.out_dir / "candidate_family_manifest.json"
    joblib.dump(
        {
            "protocol": "pm-v1.6-policy-bundle-v1",
            "outcome_model": model,
            "router_config": router_config.model_dump(mode="json"),
            "override_config": calibrated_override.model_dump(mode="json"),
        },
        checkpoint_path,
    )
    report = {
        "status": "COMPLETE",
        "version": "pm-v1.6",
        "selected_primary": selected.value,
        "transparent_router_selection": router_report,
        "algorithm_competition": competition,
        "override_calibration": calibration_report,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    write_json(report_path, report)
    train_dataset_sha = sha256_text(canonical_json({
        "states": sha256_file(args.train_states),
        "labels": sha256_file(args.train_labels),
        "step0": sha256_file(args.train_step0),
    }))
    calibration_dataset_sha = sha256_text(canonical_json({
        "states": sha256_file(args.calibration_states),
        "labels": sha256_file(args.calibration_labels),
        "step0": sha256_file(args.calibration_step0),
    }))
    manifest = CandidateFamilyManifest(
        candidate_algorithms=list(ALGORITHM_COMPLEXITY_ORDER),
        selected_primary=selected,
        selection_metric="mean_user_utility",
        one_standard_error_rule=True,
        complexity_order=list(ALGORITHM_COMPLEXITY_ORDER),
        train_dataset_sha256=train_dataset_sha,
        calibration_dataset_sha256=calibration_dataset_sha,
        internal_dataset_sha256=args.internal_dataset_sha256,
        step0_contract_sha256=sha256_text(canonical_json(config["step0"])),
        strong_router_config_sha256=router_config.digest(),
        selected_checkpoint_sha256=sha256_file(checkpoint_path),
        candidate_report_sha256=sha256_file(report_path),
    )
    write_json(manifest_path, manifest.model_dump(mode="json"))


if __name__ == "__main__":
    main()
