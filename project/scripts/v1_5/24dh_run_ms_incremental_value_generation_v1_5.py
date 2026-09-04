#!/usr/bin/env python3
"""Preflight or run the frozen 80-call MS incremental-value repair."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    legacy = runpy.run_path(
        str(ROOT / "scripts/v1_5/24cv_run_d3_ms_replacement_generation_v1_5.py")
    )
    runner = legacy["main"]
    runner.__globals__.update(
        {
            "PLAN_PROTOCOL": "pm-v1.5-ms-incremental-value-generation-plan-v1",
            "PLAN_STATUS": "FROZEN_READY_FOR_80_MS_INCREMENTAL_VALUE_CALLS",
            "PROTOCOL": "pm-v1.5-ms-incremental-value-generation-execution-v1",
            "EXPECTED_CALLS": 80,
        }
    )
    sys.argv = [
        sys.argv[0],
        "--plan-dir",
        str(ROOT / "outputs/pm_v1_5_ms_incremental_value_generation_v1"),
        "--out-dir",
        str(ROOT / "outputs/pm_v1_5_ms_incremental_value_generation_v1_execution"),
        *sys.argv[1:],
    ]
    runner()


if __name__ == "__main__":
    main()
