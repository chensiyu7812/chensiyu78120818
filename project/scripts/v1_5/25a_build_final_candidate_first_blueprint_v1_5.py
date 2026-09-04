#!/usr/bin/env python3
"""Build the deterministic private P2 construction blueprint (zero API)."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_final_dataset_blueprint import (
    audit_primary_blueprint,
    build_primary_blueprint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="data/pm_v1_5_final_candidate_first_v8/private",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_primary_blueprint()
    blueprint_path = out_dir / "construction_blueprint.jsonl"
    write_jsonl(blueprint_path, rows)
    report = audit_primary_blueprint(rows)
    report.update(
        {
            "blueprint_file": str(blueprint_path),
            "blueprint_sha256": sha256_file(blueprint_path),
            "api_calls": 0,
            "human_labels_read": 0,
            "external_lockbox_read": False,
            "next_step": "GENERATE_RAW_USERS_HISTORIES_AND_RETRIEVE_EXACT_RANK1",
        }
    )
    write_json(out_dir / "blueprint_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
