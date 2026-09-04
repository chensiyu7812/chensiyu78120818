#!/usr/bin/env python3
"""Bind and semantically normalize completed SupportNeed human anchors."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json, write_jsonl
from metacom_pm.v1_5_support_need_packet import (
    normalize_support_need_human_anchors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--normalized-out", type=Path, required=True)
    parser.add_argument("--binding-out", type=Path, required=True)
    args = parser.parse_args()

    result = normalize_support_need_human_anchors(
        packet_dir=args.packet_dir,
        annotations_path=args.annotations,
    )
    write_jsonl(args.normalized_out, result["normalized_rows"])
    write_json(args.binding_out, result["report"])


if __name__ == "__main__":
    main()
