#!/usr/bin/env python3
"""Audit retained human annotations, templates, and duplicate copies."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.v1_5_human_annotation_audit import (
    audit_historical_human_annotations,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--v2-project-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_historical_human_annotations(
        project_root=args.project_root,
        v2_project_root=args.v2_project_root,
    )
    write_json(args.out, report)


if __name__ == "__main__":
    main()

