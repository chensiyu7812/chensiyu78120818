#!/usr/bin/env python3
"""Analyze independent PM-v2 semantic annotations and attest the fail-closed gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_semantic_audit import analyze_semantic_audit


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate two-or-more independent semantic audits, compute frozen "
            "agreement gates, and write a content-bound PASS/FAIL attestation."
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
        "--backend",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "memory_backend.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_packet.csv",
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_manual.md",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_plan.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "artifact_attestation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.pm_v2_config)
    if config.get("version") != "pm-v2.2":
        raise RuntimeError("semantic-sanity audit requires PM-v2.2 config")
    states = load_states(args.states)
    evaluator_contexts = load_evaluator_context_index(
        args.evaluator_contexts, states=states, require_exact=True
    )
    report = analyze_semantic_audit(
        completed_paths=args.completed,
        config=config,
        config_path=args.pm_v2_config,
        states_path=args.states,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
        evaluator_contexts=evaluator_contexts,
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
            "semantic-sanity gate failed; do not run the development API sweep"
        )


if __name__ == "__main__":
    main()
