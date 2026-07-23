#!/usr/bin/env python3
"""Run the frozen PM-v1.5 and rule router on ESConv without any API calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.esconv_v1_5 import choose_esconv_v1_5_policies
from metacom_pm.io import read_json, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "esconv_test_v1_5" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_training" / "pm_v1_5.joblib",
    )
    parser.add_argument(
        "--transparent-rule-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "pm_v1_5_training"
            / "pm_v1_5_transparent_rule.joblib"
        ),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_training" / "training_report.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "esconv_v1_5_preflight"
    )
    args = parser.parse_args()

    training_report = read_json(args.training_report)
    if (
        training_report.get("checkpoint_sha256") != sha256_file(args.checkpoint)
        or training_report.get("transparent_rule_checkpoint_sha256")
        != sha256_file(args.transparent_rule_checkpoint)
    ):
        raise RuntimeError("ESConv preflight checkpoints differ from the training report")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    choices_path = args.out_dir / "policy_choices.jsonl"
    report = choose_esconv_v1_5_policies(
        pm_v2_states_path=args.states,
        learned_checkpoint_path=args.checkpoint,
        transparent_rule_checkpoint_path=args.transparent_rule_checkpoint,
        out_path=choices_path,
    )
    report["training_report_sha256"] = sha256_file(args.training_report)
    write_json(args.out_dir / "summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
