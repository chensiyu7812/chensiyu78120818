#!/usr/bin/env python3
"""Derive PM-v1.5 fixed-policy checkpoints from the frozen learned policy."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from metacom_pm.io import read_json, sha256_file, write_json
from metacom_pm.pm_v2_fixed_model import FixedActionPMV2Model
from metacom_pm.pm_v2_model import PMV2Model


ROOT = Path(__file__).resolve().parents[2]


def _make_fixed(
    base: PMV2Model, *, action_id: str, name: str, source_sha256: str
) -> FixedActionPMV2Model:
    return FixedActionPMV2Model(
        feature_builder=base.feature_builder,
        response_heads=base.response_heads,
        risk_heads=base.risk_heads,
        selection_config=base.selection_config,
        response_conformal_radii=dict(base.response_conformal_radii),
        risk_conformal_radii=dict(base.risk_conformal_radii),
        quality_conformal_radius=float(base.quality_conformal_radius),
        conformal_calibration_report=deepcopy(base.conformal_calibration_report),
        format_version=base.format_version,
        training_report=deepcopy(base.training_report),
        fixed_action=action_id,
        baseline_name=name,
        source_checkpoint_sha256=source_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "pm_v1_5.joblib",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "training_report.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "pm_v1_5_model"
    )
    args = parser.parse_args()

    report = read_json(args.training_report)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if (
        report.get("status") != "COMPLETE"
        or report.get("require_learned_routing_advantage_before_external") is not True
        or report.get("learned_routing_advantage_verified") is not True
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or Path(str(report.get("checkpoint") or "")).name != args.checkpoint.name
    ):
        raise RuntimeError(
            "fixed baselines require the exact reportable PM-v1.5 checkpoint"
        )

    cost_matched_action = str(
        (report.get("calibration_cost_matched_fixed") or {}).get("action_id") or ""
    )
    if not cost_matched_action:
        raise RuntimeError("training report lacks calibration cost-matched action")
    base = PMV2Model.load(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        (
            "cost_matched_fixed",
            cost_matched_action,
            args.out_dir / "cost_matched_fixed.joblib",
        ),
        (
            "event_memory_r0",
            "ME+R0",
            args.out_dir / "me_r0_fixed.joblib",
        ),
    )
    outputs = {}
    for name, action_id, path in specifications:
        model = _make_fixed(
            base,
            action_id=action_id,
            name=name,
            source_sha256=checkpoint_sha256,
        )
        model.save(path)
        outputs[name] = {
            "action_id": action_id,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
        }
    output_report = {
        "status": "COMPLETE",
        "track": "pm-v1.5",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "training_report": str(args.training_report),
        "training_report_sha256": sha256_file(args.training_report),
        "baselines": outputs,
    }
    write_json(args.out_dir / "fixed_baselines.json", output_report)
    print(output_report)


if __name__ == "__main__":
    main()
