#!/usr/bin/env python3
"""Outcome-free transparent-rule grid audit before response generation/sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v1_5_rule_router import (
    RULE_GRID_DIAGNOSTIC_PROTOCOL,
    transparent_rule_candidates,
    transparent_rule_grid_diagnostics,
)
from metacom_pm.pm_v2_contracts import CompositeSpec, PMV2Split
from metacom_pm.pm_v2_data import load_states
from metacom_pm.pm_v2_model import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
STAGE = "pm_v1_5_pre_training_rule_grid_diagnostic"


def _selection_from_config(config) -> SelectionConfig:
    quality = config["quality_composite"]
    composite = CompositeSpec(
        version=str(quality["version"]),
        weights={str(key): float(value) for key, value in quality["weights"].items()},
    )
    return SelectionConfig(**dict(config["selection"]), composite_spec=composite)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-v1-5-config",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_rule_grid_preflight",
    )
    args = parser.parse_args()

    config = load_config(args.pm_v1_5_config)
    contract = dict(config.get("pre_training_rule_grid_diagnostic") or {})
    if (
        contract.get("protocol") != RULE_GRID_DIAGNOSTIC_PROTOCOL
        or contract.get("required_before_response_sweep") is not True
        or contract.get("outcome_labels_used") is not False
        or list(contract.get("state_splits") or []) != ["train", "calibration"]
    ):
        raise RuntimeError("rule-grid preflight config contract is absent or stale")
    states = [
        state
        for state in load_states(args.states)
        if state.split in {PMV2Split.TRAIN, PMV2Split.CALIBRATION}
    ]
    report = transparent_rule_grid_diagnostics(
        states,
        transparent_rule_candidates(config["transparent_rule_router"]["grid"]),
        _selection_from_config(config),
        minimum_unique_policy_mappings=int(
            contract["minimum_unique_policy_mappings"]
        ),
        minimum_maximum_pairwise_disagreement_rate=float(
            contract["minimum_maximum_pairwise_disagreement_rate"]
        ),
    )
    report.update(
        {
            "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
            "states_sha256": sha256_file(args.states),
            "selection_or_retuning_authorized": False,
            "internal_states_used": False,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "rule_grid_report.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    write_json(report_path, report)
    create_artifact_attestation(
        attestation_path,
        stage=STAGE,
        inputs={
            "pm_v1_5_config": args.pm_v1_5_config,
            "states": args.states,
        },
        outputs={"rule_grid_report": (report_path, False)},
        parameters={
            "protocol": RULE_GRID_DIAGNOSTIC_PROTOCOL,
            "outcome_labels_used": False,
            "state_splits": ["train", "calibration"],
            "selection_or_retuning_authorized": False,
        },
        expected={
            "status": report["status"],
            "state_count": report["state_count"],
            "candidate_count": report["candidate_count"],
        },
    )
    print(report)
    if report["status"] != "PASS":
        raise RuntimeError("transparent-rule grid is degenerate; stop before paid sweep")


if __name__ == "__main__":
    main()
