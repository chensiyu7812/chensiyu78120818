#!/usr/bin/env python3
"""Build the corrected, zero-API H1-v2 private construction matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_final_dataset_blueprint import (
    H1_V2_BLUEPRINT_PROTOCOL,
    audit_primary_blueprint,
    build_h1_v2_blueprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/pm_v1_5_final_candidate_first_v9_h1_v2/private"),
    )
    args = parser.parse_args()
    rows = build_h1_v2_blueprint()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "construction_blueprint.jsonl"
    write_jsonl(path, rows)
    report = audit_primary_blueprint(rows)
    report.update(
        {
            "protocol": H1_V2_BLUEPRINT_PROTOCOL,
            "blueprint_file": str(path),
            "blueprint_sha256": sha256_file(path),
            "api_calls": 0,
            "human_labels_read": 0,
            "external_lockbox_read": False,
            "next_step": "ZERO_API_H1_V2_CONTROLLED_REALIZATION",
        }
    )
    write_json(args.out_dir / "blueprint_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
