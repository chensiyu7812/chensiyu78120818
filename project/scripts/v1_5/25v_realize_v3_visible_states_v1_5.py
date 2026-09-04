#!/usr/bin/env python3
"""Realize the frozen V3 blueprint as outcome-blind visible states."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_v3_state_realization import (
    audit_v3_states,
    realize_v3_states,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"


def _require_formal_python() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 P2 requires {FORMAL_PYTHON}; got {sys.executable}")


def main() -> None:
    _require_formal_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_visible_states_v2/private",
    )
    args = parser.parse_args()
    blueprint = [dict(row) for row in iter_jsonl(args.blueprint)]
    states = realize_v3_states(blueprint)
    report = audit_v3_states(states, blueprint)
    if report["status"] != "PASS":
        raise RuntimeError(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / "raw_states.jsonl"
    write_jsonl(state_path, states)
    report.update(
        {
            "blueprint_path": str(args.blueprint.relative_to(ROOT)),
            "blueprint_sha256": sha256_file(args.blueprint),
            "raw_states_path": str(state_path.relative_to(ROOT)),
            "raw_states_sha256": sha256_file(state_path),
            "next_step_if_pass": "FORMAL_SAME_STACK_EXACT_RANK1_MATERIALIZATION",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        }
    )
    write_json(args.out_dir / "realization_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
