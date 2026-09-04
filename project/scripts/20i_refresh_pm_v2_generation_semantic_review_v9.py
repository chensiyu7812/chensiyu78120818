from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review_v9 import refresh_v9_review_materials


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild corrected V9 blind reviewer materials from completed paid "
            "outputs without reading the private mapping or calling an API."
        )
    )
    parser.add_argument(
        "--source-v9-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v9",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v9_review_protocol_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        refresh_v9_review_materials(
            source_dir=args.source_v9_dir,
            review_dir=args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
