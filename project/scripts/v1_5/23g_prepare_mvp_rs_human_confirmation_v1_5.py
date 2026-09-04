#!/usr/bin/env python3
"""Prepare an outcome-independent second RS human confirmation packet."""

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
    build_human_audit_packet,
    render_human_audit_html,
)
from metacom_pm.v1_5_mvp_judge_runner import validate_rs_judge_inputs


ROOT = Path(__file__).resolve().parents[2]
CONFIRMATION_SEED_MATERIAL = "pm-v1.5-minimum-rs-human-confirmation-v1"
# First four bytes of SHA256(CONFIRMATION_SEED_MATERIAL), interpreted big-endian.
# This makes the default sample derivable rather than researcher-chosen by trying
# multiple seeds.
CONFIRMATION_SEED = 1448680368


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
        "--prior-audit-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_human_sanity_audit_v1_candidate",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_minimum_rs_human_confirmation_v1_candidate",
    )
    parser.add_argument("--pair-count", type=int, default=HUMAN_AUDIT_PAIR_COUNT)
    parser.add_argument("--seed", type=int, default=CONFIRMATION_SEED)
    args = parser.parse_args()

    plan_report_path = args.plan_dir / "plan_report.json"
    selected_path = args.plan_dir / "selected_states.jsonl"
    generation_plan_path = args.plan_dir / "call_plan.jsonl"
    outcomes_path = args.generation_dir / "generation_outcomes.jsonl"
    generation_summary_path = args.generation_dir / "generation_summary.json"
    prior_key_path = args.prior_audit_dir / "private_blinding_key.jsonl"
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
    excluded_pair_ids = {
        str(row["pair_id"]) for row in _rows(prior_key_path)
    }
    built = build_human_audit_packet(
        validated["pairs"],
        pair_count=args.pair_count,
        seed=args.seed,
        excluded_pair_ids=excluded_pair_ids,
    )
    manifest = dict(built["manifest"])
    manifest.update(
        {
            "status": "READY_FOR_INDEPENDENT_BLIND_HUMAN_CONFIRMATION",
            "analysis_role": "confirmatory_measurement_sample",
            "selection_timing": (
                "after_first_human_sanity_signal_before_confirmation_labels"
            ),
            "selection_uses_response_or_judge_outcome": False,
            "selection_inputs": [
                "pair_id",
                "user_id",
                "boundary_cue",
                "selected_strategy_family",
            ],
            "seed_derivation": (
                "uint32_be(first_4_bytes(SHA256("
                + CONFIRMATION_SEED_MATERIAL
                + ")))"
            ),
            "discovery_sample_pair_ids_excluded": len(excluded_pair_ids),
            "claim_boundary": (
                "This second sample can confirm measurement behavior and the "
                "conditional RS pattern; it is not a powered superiority test."
            ),
        }
    )
    manifest["lineage"] = {
        "plan_report_sha256": sha256_file(plan_report_path),
        "selected_states_sha256": sha256_file(selected_path),
        "generation_plan_sha256": sha256_file(generation_plan_path),
        "generation_outcomes_sha256": sha256_file(outcomes_path),
        "generation_summary_sha256": sha256_file(generation_summary_path),
        "runtime_states_sha256": sha256_file(runtime_path),
        "prior_private_blinding_key_sha256": sha256_file(prior_key_path),
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
        render_human_audit_html(manifest=manifest, packet=built["packet"]),
        encoding="utf-8",
    )
    print(canonical_json({"output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
