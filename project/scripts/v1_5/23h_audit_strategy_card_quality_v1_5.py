#!/usr/bin/env python3
"""Build the bounded quality qualification for the minimum-RS card catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
)
from metacom_pm.v1_5_strategy_card_quality import audit_strategy_card_quality


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1",
    )
    parser.add_argument(
        "--discovery-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_human_sanity_audit_v1_analysis",
    )
    parser.add_argument(
        "--confirmation-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_human_confirmation_v1_candidate",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_card_quality_audit_v1",
    )
    args = parser.parse_args()

    paths = {
        "cards": args.bank_dir / "strategy_cards_v2_candidate.jsonl",
        "annotations": (
            args.bank_dir / "strategy_bank_v2_human_annotations.jsonl"
        ),
        "build_report": args.bank_dir / "build_report.json",
        "human_validation": args.bank_dir / "human_validation_report.json",
        "selected_states": args.pilot_dir / "selected_states.jsonl",
        "discovery_quality": (
            args.discovery_dir / "human_unblinded_quality.jsonl"
        ),
        "discovery_risk": (
            args.discovery_dir / "human_unblinded_risk.jsonl"
        ),
        "confirmation": args.confirmation_dir / "audit_manifest.json",
    }
    report = audit_strategy_card_quality(
        cards=_rows(paths["cards"]),
        annotations=_rows(paths["annotations"]),
        build_report=read_json(paths["build_report"]),
        human_validation_report=read_json(paths["human_validation"]),
        selected_states=_rows(paths["selected_states"]),
        discovery_quality_rows=_rows(paths["discovery_quality"]),
        discovery_risk_rows=_rows(paths["discovery_risk"]),
        confirmation_manifest=read_json(paths["confirmation"]),
    )
    report["lineage"] = {
        name + "_sha256": sha256_file(path) for name, path in paths.items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "card_quality_audit.json"
    write_json(output_path, report)
    print(canonical_json({"output": str(output_path), **report}))


if __name__ == "__main__":
    main()
