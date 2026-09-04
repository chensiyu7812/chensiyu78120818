#!/usr/bin/env python3
"""Rebuild the Strategy Bank with deterministic EvoEmo source-id exclusion.

Fixes a real, confirmed gap in the original bank build: 3 of 84 unique
ESConv dialogues directly named by an EvoEmo session id (`escN`) were not
excluded, because their whole-dialogue Jaccard similarity (0.787/0.632/0.837
for esconv_0539/0585/1212) fell under the 0.88 threshold despite containing
verbatim-identical turns. This script reuses the existing, unmodified
`build_strategy_bank` with `include_deterministic_source_ids=True`.

Writes to NEW `_v1_5` paths. Never overwrites the original frozen
`data/strategy/strategy_cards.jsonl` etc., which remain the accurate
historical record of what V1 (and the PM-v2.2 track, if it hasn't yet
rebuilt) actually used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.strategy_bank import build_strategy_bank

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esconv", type=Path, default=ROOT / "data" / "external" / "ESConv.json")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument(
        "--out-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    )
    parser.add_argument(
        "--out-split",
        type=Path,
        default=ROOT / "data" / "strategy" / "esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--out-audit", type=Path, default=ROOT / "data" / "strategy" / "strategy_bank_audit_v1_5.json"
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--jaccard-threshold", type=float, default=0.88)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.out_bank, args.out_split, args.out_audit):
        if path.exists() and not args.overwrite:
            raise RuntimeError(f"refusing to overwrite existing file without --overwrite: {path}")
    result = build_strategy_bank(
        args.esconv,
        args.evoemo,
        args.out_bank,
        args.out_split,
        args.out_audit,
        seed=args.seed,
        jaccard_threshold=args.jaccard_threshold,
        include_deterministic_source_ids=True,
    )
    print(result)


if __name__ == "__main__":
    main()
