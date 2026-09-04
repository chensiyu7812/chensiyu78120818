#!/usr/bin/env python3
"""Run the deterministic, zero-API SupportNeedObservation learning canary."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.v1_5_support_need import run_support_need_learning_canary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4311)
    args = parser.parse_args()

    report = run_support_need_learning_canary(seed=args.seed)
    write_json(args.out, report)
    if report["status"] != "PASS":
        raise SystemExit("SupportNeedObservation learning canary did not PASS")


if __name__ == "__main__":
    main()
