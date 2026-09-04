#!/usr/bin/env python3
"""Validate and bind completed train-only SupportNeed human anchors."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.v1_5_support_need_packet import (
    validate_support_need_human_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = validate_support_need_human_annotations(
        packet_dir=args.packet_dir,
        annotations_path=args.annotations,
    )
    write_json(args.out, report)


if __name__ == "__main__":
    main()
