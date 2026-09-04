#!/usr/bin/env python3
"""Realize the orthogonal Observation blueprint without APIs or responses."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_v3_observation_realization import audit_states, realize_states


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
        "--out-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_observation_orthogonal_v1/private",
    )
    parser.add_argument("--surface-version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    blueprint = [dict(row) for row in iter_jsonl(args.blueprint)]
    states = realize_states(blueprint, surface_version=args.surface_version)
    report = audit_states(states, blueprint)
    if report["status"] != "PASS":
        raise RuntimeError(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / "raw_states.jsonl"
    write_jsonl(state_path, states)
    report.update(
        {
            "blueprint_sha256": sha256_file(args.blueprint),
            "raw_states_sha256": sha256_file(state_path),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "surface_version": args.surface_version,
        }
    )
    write_json(args.out_dir / "realization_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
