#!/usr/bin/env python3
"""Prepare a zero-API deterministic four-shard longitudinal judging plan.

This script is deliberately not an executor.  It cannot call a provider and
does not alter the frozen call rows.  A later shard-aware runner must require
this contract and verify exact full-plan coverage before any API call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import iter_jsonl, write_json, write_jsonl
from metacom_pm.v1_5_deterministic_sharding import (
    DEFAULT_SHARD_COUNT,
    partition_call_plan,
    sharding_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call-plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise RuntimeError("shard preparation refuses a non-empty output directory")
    rows = list(iter_jsonl(args.call_plan))
    if not rows:
        raise RuntimeError("source call plan is empty")
    shards = partition_call_plan(rows, shard_count=args.shard_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, shard_rows in enumerate(shards):
        write_jsonl(args.out_dir / f"call_plan_shard_{index:02d}.jsonl", shard_rows)
    contract = sharding_contract(
        rows,
        shards,
        shard_count=args.shard_count,
        source_call_plan_path=args.call_plan,
    )
    write_json(args.out_dir / "sharding_contract.json", contract)
    print(contract)


if __name__ == "__main__":
    main()
