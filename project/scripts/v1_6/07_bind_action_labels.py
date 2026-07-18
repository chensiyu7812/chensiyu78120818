#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import sha256_file, write_json, write_jsonl
from metacom_pm.pm_v1_6_artifacts import bind_action_labels_to_lineage

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--formal-outcomes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    summary, rows = bind_action_labels_to_lineage(
        labels_path=args.labels,
        formal_outcomes_path=args.formal_outcomes,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "action_labels_v1_6.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(rows_path, rows)
    write_json(
        summary_path,
        {
            **summary,
            "bound_action_labels_sha256": sha256_file(rows_path),
        },
    )


if __name__ == "__main__":
    main()
