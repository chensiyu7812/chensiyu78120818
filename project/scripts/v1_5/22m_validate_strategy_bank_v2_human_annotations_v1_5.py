#!/usr/bin/env python3
"""Validate exact human review coverage for five Strategy Bank V2 cards."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.v1_5_strategy_bank import (
    validate_strategy_bank_v2_human_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = validate_strategy_bank_v2_human_annotations(
        candidate_dir=args.candidate_dir,
        annotations_path=args.annotations,
    )
    write_json(args.out, report)


if __name__ == "__main__":
    main()
