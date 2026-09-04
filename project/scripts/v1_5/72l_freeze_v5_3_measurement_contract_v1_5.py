#!/usr/bin/env python3
"""Materialize the zero-API V5.3 measurement contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.io import write_json  # noqa: E402
from metacom_pm.v1_5_v5_3_measurement_freeze import (  # noqa: E402
    build_measurement_freeze,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_3_measurement_freeze_v1",
    )
    args = parser.parse_args()
    freeze = build_measurement_freeze(ROOT)
    write_json(args.out_dir / "measurement_freeze.json", freeze.model_dump(mode="json"))
    print(
        {
            "protocol": freeze.protocol,
            "status": freeze.status,
            "freeze_identity": freeze.freeze_identity,
            "sample_counts_frozen": freeze.sampling["numeric_counts_frozen"],
            "api_calls": freeze.api_calls,
        }
    )


if __name__ == "__main__":
    main()
