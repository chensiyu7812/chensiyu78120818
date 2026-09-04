#!/usr/bin/env python3
"""Audit whether one integrated V5.3 formal runner actually exists."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_runner_wiring_audit import (  # noqa: E402
    audit_formal_runner_wiring,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_3_formal_runner_wiring_audit_v1",
    )
    args = parser.parse_args()
    report = audit_formal_runner_wiring(project_root=ROOT)
    write_json(args.out_dir / "report.json", report)
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "complete_candidate_entrypoints": len(
                report["complete_candidate_entrypoints"]
            ),
            "api_calls": report["api_calls"],
        }
    )


if __name__ == "__main__":
    main()
