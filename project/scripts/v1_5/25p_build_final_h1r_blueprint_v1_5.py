#!/usr/bin/env python3
"""Freeze the single zero-API 96-state H1R orthogonal blueprint."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_final_dataset_blueprint import (
    H1R_BLUEPRINT_PROTOCOL,
    audit_h1r_blueprint,
    build_h1r_blueprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/pm_v1_5_final_h1r_v1/private"),
    )
    args = parser.parse_args()
    rows = build_h1r_blueprint()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blueprint = args.out_dir / "construction_blueprint.jsonl"
    write_jsonl(blueprint, rows)
    report = audit_h1r_blueprint(rows)
    report.update(
        {
            "blueprint_protocol": H1R_BLUEPRINT_PROTOCOL,
            "blueprint_file": str(blueprint),
            "blueprint_sha256": sha256_file(blueprint),
            "api_calls": 0,
            "human_labels_read": 0,
            "next_step": "REALIZE_VISIBLE_STATES_AND_EXACT_RANK1_CANDIDATES",
        }
    )
    write_json(args.out_dir / "blueprint_report.json", report)
    print(report)


if __name__ == "__main__":
    main()

