#!/usr/bin/env python3
"""Freeze the zero-API V5.3 recovery method before selecting real cases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_recovery_qualification import (  # noqa: E402
    build_recovery_qualification_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_3_recovery_qualification_method_v1",
    )
    args = parser.parse_args()
    plan = build_recovery_qualification_plan(ROOT)
    write_json(args.out_dir / "recovery_method.json", plan.model_dump(mode="json"))
    print(
        {
            "protocol": plan.protocol,
            "status": plan.status,
            "cases": len(plan.cases),
            "plan_identity": plan.plan_identity,
            "api_calls": plan.api_calls,
        }
    )


if __name__ == "__main__":
    main()
