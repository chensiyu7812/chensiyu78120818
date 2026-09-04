#!/usr/bin/env python3
"""Build a deterministic, zero-API ESConv/EvoEmo understanding profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from metacom_pm.evoemo import (  # noqa: E402
    evoemo_chronology_audit,
    load_evoemo,
)
from metacom_pm.io import iter_jsonl, sha256_file, write_json  # noqa: E402
from metacom_pm.strategy_bank import load_esconv  # noqa: E402
from metacom_pm.v1_5_dataset_understanding import (  # noqa: E402
    build_dataset_understanding_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--esconv", type=Path, default=ROOT / "data/external/ESConv.json"
    )
    parser.add_argument(
        "--evoemo", type=Path, default=ROOT / "data/external/evo_emo.json"
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data/strategy/esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_esconv_evoemo_understanding_v1",
    )
    args = parser.parse_args()

    esconv = load_esconv(args.esconv)
    evoemo = load_evoemo(args.evoemo)
    split_rows = [dict(row) for row in iter_jsonl(args.split_manifest)]
    profile = build_dataset_understanding_profile(
        esconv=esconv,
        split_rows=split_rows,
        evoemo_users=evoemo,
        source_hashes={
            "esconv_sha256": sha256_file(args.esconv),
            "evoemo_sha256": sha256_file(args.evoemo),
            "split_manifest_sha256": sha256_file(args.split_manifest),
        },
        evoemo_chronology_report=evoemo_chronology_audit(args.evoemo),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "profile.json", profile)
    print(
        json.dumps(
            {
                "status": "COMPLETE_ZERO_API",
                "profile_sha256": profile["profile_sha256"],
                "out_dir": str(args.out_dir),
                "api_calls_made": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
