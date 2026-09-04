#!/usr/bin/env python3
"""Audit V3 claim boundaries and same-stack RAG readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.v1_5_claim_stack_audit import audit_v1_5_v3_claim_stack


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs/pm_v1_5_v3_claim_same_stack_v1.json",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_v1_5_v3_claim_stack(
        project_root=ROOT,
        contract_path=args.contract,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "report.json", report)


if __name__ == "__main__":
    main()

