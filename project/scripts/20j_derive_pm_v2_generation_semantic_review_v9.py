from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review_v9 import derive_v9_marginal_values


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Privately derive V9 Strategy-RAG marginal direction after independent "
            "review, adjudication, and explicit unblinding."
        )
    )
    parser.add_argument("--adjudicated-review", type=Path, required=True)
    parser.add_argument(
        "--private-mapping",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v9"
        / "private_condition_mapping.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v9_private_post_unblind",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        derive_v9_marginal_values(
            adjudicated_review_path=args.adjudicated_review,
            private_mapping_path=args.private_mapping,
            out_dir=args.out_dir,
        )
    )


if __name__ == "__main__":
    main()
