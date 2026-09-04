#!/usr/bin/env python3
"""Build the zero-API V3 Observation factor-orthogonal private blueprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_v3_observation_orthogonal import (
    PROTOCOL,
    audit_blueprint,
    build_blueprint,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_observation_orthogonal_v1/private",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_blueprint_v1",
    )
    args = parser.parse_args()

    rows = build_blueprint()
    report = audit_blueprint(rows)
    args.blueprint_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blueprint_path = args.blueprint_dir / "construction_blueprint.jsonl"
    write_jsonl(blueprint_path, rows)
    report.update(
        {
            "blueprint_protocol": PROTOCOL,
            "blueprint_path": str(blueprint_path.relative_to(ROOT)),
            "blueprint_sha256": sha256_file(blueprint_path),
        }
    )
    write_json(args.blueprint_dir / "blueprint_report.json", report)
    write_json(args.out_dir / "audit_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
