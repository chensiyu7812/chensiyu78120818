#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.pm_v1_6_artifacts import finalize_action_sweep

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-outcomes", type=Path, required=True)
    parser.add_argument("--preflight-rows", type=Path, required=True)
    parser.add_argument("--preflight-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summary, rows = finalize_action_sweep(
        source_outcomes_path=args.source_outcomes,
        preflight_rows_path=args.preflight_rows,
        preflight_summary_path=args.preflight_summary,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "formal_action_outcomes.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(rows_path, rows)
    write_json(
        summary_path,
        {
            **summary,
            "formal_action_outcomes_sha256": sha256_file(rows_path),
        },
    )


if __name__ == "__main__":
    main()
