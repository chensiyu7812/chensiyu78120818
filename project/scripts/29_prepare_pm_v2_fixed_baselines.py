#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v2_fixed_model import FixedActionPMV2Model
from metacom_pm.pm_v2_model import PMV2Model

ROOT = Path(__file__).resolve().parents[1]


def make_fixed(base: PMV2Model, *, action_id: str, name: str, source_sha: str):
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
        source_checkpoint_sha256=source_sha,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "pm_v2.joblib",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "training_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model",
    )
    args = parser.parse_args()

    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE":
        raise RuntimeError("fixed baselines may only be prepared from a reportable PM-v2")
    action_id = str(report["calibration_cost_matched_fixed"]["action_id"])
    base = PMV2Model.load(args.checkpoint)
    source_sha = sha256_file(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specifications = [
        (
            "cost_matched_fixed",
            action_id,
            args.out_dir / "pm_v2_cost_matched_fixed.joblib",
        ),
        (
            "event_memory_r0",
            "ME+R0",
            args.out_dir / "pm_v2_me_r0_fixed.joblib",
        ),
    ]
    outputs = {}
    for name, fixed_action, path in specifications:
        model = make_fixed(
            base,
            action_id=fixed_action,
            name=name,
            source_sha=source_sha,
        )
        model.save(path)
        outputs[name] = {
            "action_id": fixed_action,
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
        }
    output_report = {
        "status": "COMPLETE",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha,
        "training_report": str(args.training_report),
        "training_report_sha256": sha256_file(args.training_report),
        "baselines": outputs,
    }
    write_json(args.out_dir / "pm_v2_fixed_baselines.json", output_report)
    print(output_report)


if __name__ == "__main__":
    main()
