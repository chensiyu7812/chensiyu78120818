#!/usr/bin/env python3
"""Run the full pre-human static gate for orthogonal Observation data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json
from metacom_pm.v1_5_v3_observation_static_gate import audit_pre_human_static_gate


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_observation_orthogonal_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--candidate-rows",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_exact_rank1_v1/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_exact_rank1_v1/materialization_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_pre_human_gate_v1",
    )
    args = parser.parse_args()
    report = audit_pre_human_static_gate(
        blueprint_rows=[dict(row) for row in iter_jsonl(args.blueprint)],
        candidate_rows=[dict(row) for row in iter_jsonl(args.candidate_rows)],
        materialization_report=read_json(args.materialization_report),
    )
    report.update(
        {
            "blueprint_sha256": sha256_file(args.blueprint),
            "candidate_rows_sha256": sha256_file(args.candidate_rows),
            "materialization_report_sha256": sha256_file(args.materialization_report),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "static_gate_report.json", report)
    print(report)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
