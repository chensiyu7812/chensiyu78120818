#!/usr/bin/env python3
"""Materialize V3 target-component exact Rank-1 candidates with the frozen stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_v3_candidate_materialization import materialize_v3_candidates


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"


def _require_formal_python() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 P2 requires {FORMAL_PYTHON}; got {sys.executable}")


def _rows(path: Path) -> list[dict]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    _require_formal_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_visible_states_v2/private/raw_states.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4",
    )
    args = parser.parse_args()
    rows, report = materialize_v3_candidates(
        states=_rows(args.states),
        blueprint_rows=_rows(args.blueprint),
        strategy_cards=_rows(args.strategy_cards),
    )
    if report["status"] != "PASS":
        raise RuntimeError(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.out_dir / "candidate_rows_private.jsonl"
    write_jsonl(candidate_path, rows)
    report.update(
        {
            "states_sha256": sha256_file(args.states),
            "blueprint_sha256": sha256_file(args.blueprint),
            "strategy_cards_sha256": sha256_file(args.strategy_cards),
            "candidate_rows_sha256": sha256_file(candidate_path),
            "next_step_if_pass": "VERSIONED_H_ELIGIBILITY_REVIEW_AND_STEP2_EXECUTION_QUALIFICATION",
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        }
    )
    write_json(args.out_dir / "materialization_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
