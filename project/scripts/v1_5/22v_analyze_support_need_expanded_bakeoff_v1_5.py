#!/usr/bin/env python3
"""Analyze the reproducible 39-group SupportNeed representation bakeoff."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.v1_5_support_need_expanded_analysis import (
    analyze_expanded_support_need_bakeoff,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prior",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_support_need_factorized_bakeoff_v1/"
            "report.json"
        ),
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repro", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_expanded_support_need_bakeoff(
        prior_report_path=args.prior,
        candidate_report_path=args.candidate,
        repro_report_path=args.repro,
    )
    write_json(args.out, report)


if __name__ == "__main__":
    main()

