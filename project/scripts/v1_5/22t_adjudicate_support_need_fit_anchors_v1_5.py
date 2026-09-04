#!/usr/bin/env python3
"""Bind two fit reviews, adjudicate them, and prepare the 39-anchor packet."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_support_need_adjudication import (
    build_support_need_fit_adjudication,
    combine_support_need_fit_anchors,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-packet-dir", type=Path, required=True)
    parser.add_argument("--rater-a", type=Path, required=True)
    parser.add_argument("--rater-b", type=Path, required=True)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=(
            ROOT
            / "configs/pm_v1_5_support_need_fit_adjudication_v1.json"
        ),
    )
    parser.add_argument(
        "--initial-packet-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_support_need_packet_v3_lineage_candidate"
        ),
    )
    parser.add_argument(
        "--initial-normalized-anchors",
        type=Path,
        default=(
            ROOT
            / "data/pm_v1_5_contracts/"
            "support_need_human_anchors_normalized_v2.jsonl"
        ),
    )
    parser.add_argument(
        "--initial-normalization-binding",
        type=Path,
        default=(
            ROOT
            / "data/pm_v1_5_contracts/"
            "support_need_human_anchor_binding_v2.json"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    adjudication = build_support_need_fit_adjudication(
        packet_dir=args.fit_packet_dir,
        rater_a_path=args.rater_a,
        rater_b_path=args.rater_b,
        decisions_path=args.decisions,
    )
    fit_packet_rows = [
        dict(row)
        for row in iter_jsonl(
            args.fit_packet_dir / "human_blind_packet.jsonl"
        )
    ]
    combined = combine_support_need_fit_anchors(
        initial_packet_dir=args.initial_packet_dir,
        initial_normalized_path=args.initial_normalized_anchors,
        initial_normalization_binding_path=(
            args.initial_normalization_binding
        ),
        fit_packet_rows=fit_packet_rows,
        fit_normalized_rows=adjudication["normalized_rows"],
        adjudication_report=adjudication["report"],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_a_out = args.out_dir / "raw_rater_a.jsonl"
    raw_b_out = args.out_dir / "raw_rater_b.jsonl"
    shutil.copyfile(args.rater_a, raw_a_out)
    shutil.copyfile(args.rater_b, raw_b_out)
    if (
        sha256_file(raw_a_out) != sha256_file(args.rater_a)
        or sha256_file(raw_b_out) != sha256_file(args.rater_b)
    ):
        raise RuntimeError("support-need raw review copy drifted")
    write_jsonl(
        args.out_dir / "adjudicated_annotations.jsonl",
        adjudication["adjudicated_rows"],
    )
    write_jsonl(
        args.out_dir / "adjudication_trace.jsonl",
        adjudication["trace_rows"],
    )
    write_jsonl(
        args.out_dir / "normalized_fit_anchors.jsonl",
        adjudication["normalized_rows"],
    )
    write_json(
        args.out_dir / "adjudication_report.json",
        adjudication["report"],
    )
    write_jsonl(
        args.out_dir / "combined_human_blind_packet.jsonl",
        combined["combined_packet_rows"],
    )
    write_jsonl(
        args.out_dir / "combined_normalized_anchors.jsonl",
        combined["combined_normalized_rows"],
    )
    write_json(
        args.out_dir / "combined_anchor_report.json",
        combined["report"],
    )


if __name__ == "__main__":
    main()
