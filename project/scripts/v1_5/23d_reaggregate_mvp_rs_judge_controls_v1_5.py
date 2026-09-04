#!/usr/bin/env python3
"""Reaggregate completed RS judge controls under the frozen material rule."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import canonical_json, iter_jsonl, read_json, write_json
from metacom_pm.v1_5_mvp_judge_runner import qualify_controls


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--qualification-contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/minimum_rs_judge_qualification_v1.json",
    )
    parser.add_argument(
        "--output-name",
        default="qualification_report_material_v2.json",
    )
    args = parser.parse_args()

    report = qualify_controls(
        qualification_contract=read_json(args.qualification_contract),
        call_plan=_rows(args.control_dir / "call_plan.jsonl"),
        results=_rows(args.control_dir / "judge_results.jsonl"),
    )
    output_path = args.control_dir / args.output_name
    if output_path.name == "qualification_report.json":
        raise ValueError("the legacy qualification report must not be overwritten")
    write_json(output_path, report)
    print(canonical_json({"output": str(output_path), **report}))


if __name__ == "__main__":
    main()
