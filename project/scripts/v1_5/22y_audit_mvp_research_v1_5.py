#!/usr/bin/env python3
"""Validate the PM-v1.5 MVP metric definitions and current data boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import read_json, write_json
from metacom_pm.v1_5_mvp_research import (
    audit_v1_5_mvp_data,
    validate_metric_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-registry",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/minimum_publishable_metric_registry_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_publishable_audit_v1.json",
    )
    parser.add_argument(
        "--auxiliary-dir",
        type=Path,
        default=ROOT / "data/esconv_auxiliary_v1_5_visible_v2_candidate",
    )
    parser.add_argument(
        "--external-test-dir",
        type=Path,
        default=ROOT / "data/esconv_test_v1_5_visible_v2_candidate",
    )
    args = parser.parse_args()

    metric_report = validate_metric_registry(read_json(args.metric_registry))
    data_report = audit_v1_5_mvp_data(
        ROOT,
        auxiliary_dir=args.auxiliary_dir,
        external_test_dir=args.external_test_dir,
    )
    report = {
        "protocol": "pm-v1.5-minimum-publishable-research-audit-v1",
        "status": (
            "READY_FOR_ZERO_API_REBUILD"
            if metric_report["status"] == "PASS"
            and data_report["status"] == "NEEDS_ZERO_API_REBUILD_BEFORE_MVP_USE"
            and data_report["blockers"]
            == ["EXISTING_ESCONV_STATES_EXPOSE_DATASET_SITUATION_TO_PM"]
            else (
                "DATA_AND_METRICS_READY_FOR_MVP_PILOT"
                if metric_report["status"] == "PASS"
                and data_report["status"] == "PASS_FOR_MVP_USE"
                else "NEEDS_RESEARCH_REVISION"
            )
        ),
        "metric_registry": metric_report,
        "data_leakage": data_report,
        "hashes_are_scientific_outcomes": False,
        "irreversible_one_shot_execution_required": False,
    }
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
