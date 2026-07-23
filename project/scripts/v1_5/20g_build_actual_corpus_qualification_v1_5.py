#!/usr/bin/env python3
"""Build the zero-API, post-hoc actual-468 corpus qualification addendum."""

from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.io import write_json
from metacom_pm.v1_5_actual_corpus_qualification import (
    ACTUAL_CORPUS_QUALIFICATION_STAGE,
    build_actual_corpus_posthoc_qualification,
)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--repair-overlays", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--pm-v1-5-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml")
    parser.add_argument("--strategy-bank", type=Path, default=ROOT / "data/strategy/strategy_cards_v1_5.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "qualification_report.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    if report_path.exists() or attestation_path.exists():
        raise RuntimeError("qualification output already exists; use a fresh directory")

    report = build_actual_corpus_posthoc_qualification(
        recovered_gate_report_path=args.review_dir / "recovered_gate_report.json",
        recovery_report_path=args.review_dir / "recovery_report.json",
        incomplete_gate_report_path=args.review_dir / "gate_report.json",
        physical_attempt_ledger_path=args.review_dir / "physical_attempt_ledger.jsonl",
        cost_estimate_path=args.review_dir / "cost_estimate.json",
        call_plan_path=args.review_dir / "call_plan.jsonl",
        recompile_verification_path=args.data_dir / "context_grounding_repair_recompile_verification.json",
        development_data_attestation_path=args.data_dir / "artifact_attestation.json",
        classification_path=args.classification,
        repair_overlays_path=args.repair_overlays,
        expected_states_path=args.data_dir / "pm_v2_states.jsonl",
        expected_evaluator_contexts_path=args.data_dir / "evaluator_contexts.jsonl",
        expected_backend_path=args.data_dir / "memory_backend.jsonl",
        expected_runtime_states_path=args.data_dir / "runtime_states.jsonl",
        expected_bundles_path=args.data_dir / "pm_v2_bundles.jsonl",
        expected_data_report_path=args.data_dir / "pm_v2_data_report.json",
        expected_strategy_bank_path=args.strategy_bank,
        expected_pm_config_path=args.pm_v1_5_config,
    )
    write_json(report_path, report)
    create_artifact_attestation(
        attestation_path,
        stage=ACTUAL_CORPUS_QUALIFICATION_STAGE,
        inputs={
            "experiment_config": args.experiment_config,
            "states": args.data_dir / "pm_v2_states.jsonl",
            "evaluator_contexts": args.data_dir / "evaluator_contexts.jsonl",
            "memory_backend": args.data_dir / "memory_backend.jsonl",
            "runtime_states": args.data_dir / "runtime_states.jsonl",
            "bundles": args.data_dir / "pm_v2_bundles.jsonl",
            "data_report": args.data_dir / "pm_v2_data_report.json",
            "strategy_bank": args.strategy_bank,
            "pm_v1_5_config": args.pm_v1_5_config,
            "recovered_gate_report": args.review_dir / "recovered_gate_report.json",
            "recovery_report": args.review_dir / "recovery_report.json",
            "incomplete_gate_report": args.review_dir / "gate_report.json",
            "physical_attempt_ledger": args.review_dir / "physical_attempt_ledger.jsonl",
            "cost_estimate": args.review_dir / "cost_estimate.json",
            "call_plan": args.review_dir / "call_plan.jsonl",
            "recompile_verification": args.data_dir / "context_grounding_repair_recompile_verification.json",
            "development_data_attestation": args.data_dir / "artifact_attestation.json",
            "classification": args.classification,
            "repair_overlays": args.repair_overlays,
            "qualification_contract_code": ROOT / "src/metacom_pm/v1_5_actual_corpus_qualification.py",
        },
        outputs={"qualification_report": (report_path, False)},
        parameters={
            "post_hoc": True,
            "original_gate_status": "FAIL",
            "paid_api_calls": 0,
        },
        expected={"repaired_states": 25, "real_case_rejections": 0},
    )
    print({"status": report["status"], "report": str(report_path), "attestation": str(attestation_path)})


if __name__ == "__main__":
    main()
