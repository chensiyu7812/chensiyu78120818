from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review_v8 import (
    prepare_generation_semantic_review_v8,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare fail-closed PM V2 V8 semantic-review materials."
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards.jsonl",
    )
    parser.add_argument(
        "--v7-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v7",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v8",
    )
    parser.add_argument(
        "--cases-per-regime",
        type=int,
        choices=(1, 3),
        default=1,
        help="1 creates the 9-case smoke set; 3 creates the 27-case validation set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260749,
        help="Frozen smoke seed selected before annotation; controls ages and row order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_generation_semantic_review_v8(
        strategy_bank_path=args.strategy_bank,
        out_dir=args.out_dir,
        v7_dir=args.v7_dir,
        cases_per_regime=args.cases_per_regime,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
