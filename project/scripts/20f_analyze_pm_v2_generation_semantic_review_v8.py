from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review_v8 import (
    analyze_generation_semantic_review_v8,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = (
    ROOT / "outputs" / "pm_v2_generation_pilot_semantic_review_v8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze two independent PM V2 V8 semantic reviews."
    )
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--completed",
        type=Path,
        nargs="+",
        required=True,
        help="At least two independently completed V8 reviewer CSV files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_DIR / "generation_pilot_semantic_review_report.json",
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        default=DEFAULT_DIR / "artifact_attestation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_generation_semantic_review_v8(
        review_dir=args.review_dir,
        completed_paths=args.completed,
        report_path=args.report,
        attestation_path=args.attestation,
    )
    print(report)


if __name__ == "__main__":
    main()
