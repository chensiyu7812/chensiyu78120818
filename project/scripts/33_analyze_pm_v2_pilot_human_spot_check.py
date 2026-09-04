#!/usr/bin/env python3
"""Analyze and attest the fail-closed pilot judge human compatibility gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_pilot_human_spot_check import (
    analyze_pilot_human_spot_check,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "outputs" / "pm_v2_development_pilot_human_spot_check"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate at least two independent integer-rating packets, compute "
            "per-dimension LLM-human and human-human gates, and attest lineage."
        )
    )
    parser.add_argument("--completed", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml"
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_pilot" / "pilot_plan.json",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot"
        / "action_outcomes.jsonl",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot_judging"
        / "action_labels.jsonl",
    )
    parser.add_argument(
        "--raw-results",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot_judging"
        / "judge_results.jsonl",
    )
    parser.add_argument(
        "--judge-compatibility-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot_judging"
        / "summary.json",
    )
    parser.add_argument(
        "--judge-compatibility-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot_judging"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_DIR / "pilot_human_spot_check_packet.csv",
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=DEFAULT_DIR / "pilot_human_spot_check_manual.md",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=DEFAULT_DIR / "pilot_human_spot_check_plan.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_DIR / "pilot_human_spot_check_report.json",
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        default=DEFAULT_DIR / "artifact_attestation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.pm_v2_config)
    if config.get("version") != "pm-v2.2":
        raise RuntimeError("pilot human spot-check requires PM-v2.2")
    states = load_states(args.states)
    evaluator_contexts = load_evaluator_context_index(
        args.evaluator_contexts, states=states, require_exact=True
    )
    report = analyze_pilot_human_spot_check(
        completed_paths=args.completed,
        config=config,
        config_path=args.pm_v2_config,
        states_path=args.states,
        evaluator_contexts_path=args.evaluator_contexts,
        evaluator_contexts=evaluator_contexts,
        pilot_plan_path=args.pilot_plan,
        outcomes_path=args.outcomes,
        labels_path=args.labels,
        raw_results_path=args.raw_results,
        judge_compatibility_summary_path=args.judge_compatibility_summary,
        judge_compatibility_attestation_path=args.judge_compatibility_attestation,
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
            "pilot judge human spot-check failed; do not run the full development API"
        )


if __name__ == "__main__":
    main()
