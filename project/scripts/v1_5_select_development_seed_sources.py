#!/usr/bin/env python3
"""Freeze the dialogue-level sources used by the 52-user V1.5 development run.

Selection is deterministic and outcome-free: take the first 52 eligible ESConv
train sources in the already-frozen split-manifest order.  Only IDs and split
lineage are published; raw dialogue text remains in the private seed artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "pm-v1.5-selected-development-seed-sources-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data" / "strategy" / "esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "strategy" / "pm_v1_5_selected_seed_sources.jsonl",
    )
    parser.add_argument("--count", type=int, default=52)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count != 52:
        raise RuntimeError("PM-v1.5 freezes exactly 52 development seed sources")
    if args.out.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {args.out}; pass --overwrite")
    eligible = [
        dict(row)
        for row in iter_jsonl(args.split_manifest)
        if row.get("split") == "train"
        and not bool(row.get("excluded_for_evoemo_overlap"))
    ]
    if len(eligible) < args.count:
        raise RuntimeError("split manifest has too few eligible train sources")
    selected = []
    split_sha = sha256_file(args.split_manifest)
    for ordinal, row in enumerate(eligible[: args.count]):
        selected.append(
            {
                "protocol": PROTOCOL,
                "selection_rule": "first_eligible_train_sources_in_manifest_order",
                "selection_ordinal": ordinal,
                "dialogue_id": str(row["dialogue_id"]),
                "split_manifest_index": int(row["index"]),
                "source_split": "train",
                "excluded_for_evoemo_overlap": False,
                "split_manifest_sha256": split_sha,
            }
        )
    ids = [row["dialogue_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("selected seed source IDs are not unique")
    manifest_sha = sha256_text(canonical_json(ids))
    for row in selected:
        row["selected_source_ids_sha256"] = manifest_sha
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, selected)
    print({"status": "FROZEN", "sources": len(selected), "ids_sha256": manifest_sha})


if __name__ == "__main__":
    main()
