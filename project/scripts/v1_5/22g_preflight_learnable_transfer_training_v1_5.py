#!/usr/bin/env python3
"""Build the zero-API, fail-closed V2 formal-fit readiness report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json
from metacom_pm.v1_5_learnable_transfer_readiness import (
    audit_failure_ledger,
    build_formal_fit_readiness,
)


ROOT = Path(__file__).resolve().parents[2]


def _optional_report(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/learnable_transfer_training_data_v2.json",
    )
    parser.add_argument(
        "--root-cause-report",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_training_data_root_cause_audit_v1.json",
    )
    parser.add_argument(
        "--failure-ledger",
        type=Path,
        default=ROOT / "docs/PM_V1_TO_V1_5_GLOBAL_FAILURE_LEDGER_ZH.md",
    )
    parser.add_argument(
        "--legacy-human-annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_factorized_weak_learning_human_anchor_v1"
        / "human_annotation_template.jsonl",
    )
    parser.add_argument(
        "--external-support-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_merged_ood_audit_v8_19_2_v3_production_shape.json",
    )
    parser.add_argument(
        "--clean-contrast-report",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_clean_contrast_integrity_v2.json",
    )
    parser.add_argument(
        "--mechanism-uptake-report",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_clean_mechanism_uptake_v2.json",
    )
    parser.add_argument(
        "--measurement-report",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_mechanism_matched_measurement_v2.json",
    )
    parser.add_argument(
        "--learnability-report",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_train_only_component_learnability_v2.json",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_learnable_transfer_training_readiness_v2.json",
    )
    args = parser.parse_args()

    required = (
        args.contract,
        args.root_cause_report,
        args.failure_ledger,
        args.legacy_human_annotations,
        args.external_support_report,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required readiness inputs are missing: {missing}")

    report = build_formal_fit_readiness(
        contract_path=args.contract,
        root_cause_report=read_json(args.root_cause_report),
        ledger_audit=audit_failure_ledger(args.failure_ledger),
        legacy_human_annotations=list(iter_jsonl(args.legacy_human_annotations)),
        external_support_report=read_json(args.external_support_report),
        clean_contrast_report=_optional_report(args.clean_contrast_report),
        mechanism_uptake_report=_optional_report(args.mechanism_uptake_report),
        measurement_report=_optional_report(args.measurement_report),
        learnability_report=_optional_report(args.learnability_report),
    )
    report["inputs"] = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in {
            "contract": args.contract,
            "root_cause_report": args.root_cause_report,
            "failure_ledger": args.failure_ledger,
            "legacy_human_annotations": args.legacy_human_annotations,
            "external_support_report": args.external_support_report,
            "clean_contrast_report": args.clean_contrast_report,
            "mechanism_uptake_report": args.mechanism_uptake_report,
            "measurement_report": args.measurement_report,
            "learnability_report": args.learnability_report,
        }.items()
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_path, report)
    print(args.out_path)


if __name__ == "__main__":
    main()
