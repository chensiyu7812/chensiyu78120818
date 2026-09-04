#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.pm_v2_generation_review import (
    analyze_generation_pilot_semantic_review,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = (
    ROOT
    / "outputs"
    / "pm_v2_generation_compatibility_pilot_deterministic_evidence_v7"
    / "artifact_attestation.json"
)
DEFAULT_DIR = ROOT / "outputs" / "pm_v2_generation_pilot_semantic_review_v7"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze two independent generation-pilot semantic review CSVs and "
            "create the fail-closed no-API attestation."
        )
    )
    parser.add_argument("--completed", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--generation-pilot-attestation", type=Path, default=DEFAULT_PILOT
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_DIR / "generation_pilot_semantic_review_packet.csv",
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=DEFAULT_DIR / "generation_pilot_semantic_review_manual.md",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_DIR / "generation_pilot_semantic_review_plan.json",
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = analyze_generation_pilot_semantic_review(
        completed_paths=args.completed,
        pilot_attestation_path=args.generation_pilot_attestation,
        packet_path=args.packet,
        manual_path=args.manual,
        plan_path=args.plan,
        report_path=args.report,
        attestation_path=args.attestation,
        overwrite=args.overwrite,
    )
    print(report)
    if report["status"] != "PASS":
        raise RuntimeError(
            "generation pilot semantic review failed; do not run full generation"
        )


if __name__ == "__main__":
    main()
