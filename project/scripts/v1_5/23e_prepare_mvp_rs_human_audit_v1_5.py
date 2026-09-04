#!/usr/bin/env python3
"""Prepare the outcome-blind 12-pair minimum RS human sanity audit."""

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
    write_jsonl,
)
from metacom_pm.v1_5_mvp_human_audit import (
    HUMAN_AUDIT_PAIR_COUNT,
    HUMAN_AUDIT_SEED,
    build_human_audit_packet,
    render_human_audit_html,
)
from metacom_pm.v1_5_mvp_judge_runner import validate_rs_judge_inputs


ROOT = Path(__file__).resolve().parents[2]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1",
    )
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_clean_pair_pilot_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_human_sanity_audit_v1_candidate",
    )
    parser.add_argument("--pair-count", type=int, default=HUMAN_AUDIT_PAIR_COUNT)
    parser.add_argument("--seed", type=int, default=HUMAN_AUDIT_SEED)
    args = parser.parse_args()

    plan_report_path = args.plan_dir / "plan_report.json"
    selected_path = args.plan_dir / "selected_states.jsonl"
    generation_plan_path = args.plan_dir / "call_plan.jsonl"
    outcomes_path = args.generation_dir / "generation_outcomes.jsonl"
    generation_summary_path = args.generation_dir / "generation_summary.json"
    plan_report = read_json(plan_report_path)
    runtime_path = Path(str(plan_report["runtime_states_path"]))
    validated = validate_rs_judge_inputs(
        plan_report=plan_report,
        generation_summary=read_json(generation_summary_path),
        selected_states=_rows(selected_path),
        generation_plan=_rows(generation_plan_path),
        generation_outcomes=_rows(outcomes_path),
        runtime_states=_rows(runtime_path),
    )
    built = build_human_audit_packet(
        validated["pairs"],
        pair_count=args.pair_count,
        seed=args.seed,
    )
    manifest = dict(built["manifest"])
    manifest["lineage"] = {
        "plan_report_sha256": sha256_file(plan_report_path),
        "selected_states_sha256": sha256_file(selected_path),
        "generation_plan_sha256": sha256_file(generation_plan_path),
        "generation_outcomes_sha256": sha256_file(outcomes_path),
        "generation_summary_sha256": sha256_file(
            generation_summary_path
        ),
        "runtime_states_sha256": sha256_file(runtime_path),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "audit_manifest.json", manifest)
    write_jsonl(args.out_dir / "human_blind_packet.jsonl", built["packet"])
    write_jsonl(
        args.out_dir / "human_annotation_template.jsonl",
        built["annotation_template"],
    )
    write_jsonl(
        args.out_dir / "private_blinding_key.jsonl",
        built["private_key"],
    )
    write_jsonl(
        args.out_dir / "outcome_blind_selection.jsonl",
        built["selection_rows"],
    )
    (args.out_dir / "human_blind_review.html").write_text(
        render_human_audit_html(
            manifest=manifest,
            packet=built["packet"],
        ),
        encoding="utf-8",
    )
    print(canonical_json({"output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
