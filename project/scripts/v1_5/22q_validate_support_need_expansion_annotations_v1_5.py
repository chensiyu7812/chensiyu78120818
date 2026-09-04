#!/usr/bin/env python3
"""Validate revised SupportNeed expansion annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.v1_5_support_need_packet import (
    validate_support_need_expansion_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(
        args.out,
        validate_support_need_expansion_annotations(
            packet_dir=args.packet_dir,
            annotations_path=args.annotations,
        ),
    )


if __name__ == "__main__":
    main()
