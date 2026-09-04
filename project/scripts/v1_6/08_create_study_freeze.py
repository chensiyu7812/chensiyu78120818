#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.io import write_json
from metacom_pm.pm_v1_6_freeze import build_study_freeze

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-config", type=Path, default=ROOT / "configs" / "pm_v1_6.yaml")
    parser.add_argument("--method-contract", type=Path, default=ROOT / "docs" / "PM_V1_6_METHOD_CONTRACT_ZH.md")
    parser.add_argument("--preregistration", type=Path, default=ROOT / "docs" / "PM_V1_6_PREREGISTRATION.json")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--internal-report", type=Path, required=True)
    parser.add_argument("--internal-ledger", type=Path, required=True)
    parser.add_argument("--step0-observations", type=Path, required=True)
    parser.add_argument("--step0-audit-bindings", type=Path, required=True)
    parser.add_argument("--preflight-summary", type=Path, required=True)
    parser.add_argument("--preflight-rows", type=Path, required=True)
    parser.add_argument("--formal-outcomes", type=Path, required=True)
    parser.add_argument("--bound-labels", type=Path, required=True)
    parser.add_argument("--strategy-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl")
    parser.add_argument("--strategy-bank-audit", type=Path, default=ROOT / "data" / "strategy" / "strategy_bank_audit_v1_5.json")
    parser.add_argument("--overlap-audit", type=Path, default=ROOT / "outputs" / "v1_5_strategy_card_evoemo_turn_overlap_audit_v1_5_bank.json")
    parser.add_argument("--fixed-seeker-tracks", type=Path, required=True)
    parser.add_argument("--fixed-seeker-attestation", type=Path, required=True)
    parser.add_argument("--judge-isolation-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "pm_v1_6_study_freeze.json")
    args = parser.parse_args()

    freeze = build_study_freeze(
        method_config_path=args.method_config,
        method_contract_path=args.method_contract,
        preregistration_path=args.preregistration,
        candidate_manifest_path=args.candidate_manifest,
        checkpoint_path=args.checkpoint,
        internal_report_path=args.internal_report,
        internal_ledger_path=args.internal_ledger,
        step0_observations_path=args.step0_observations,
        step0_audit_bindings_path=args.step0_audit_bindings,
        preflight_summary_path=args.preflight_summary,
        preflight_rows_path=args.preflight_rows,
        formal_outcomes_path=args.formal_outcomes,
        bound_labels_path=args.bound_labels,
        strategy_bank_path=args.strategy_bank,
        strategy_bank_audit_path=args.strategy_bank_audit,
        overlap_audit_path=args.overlap_audit,
        fixed_seeker_tracks_path=args.fixed_seeker_tracks,
        fixed_seeker_attestation_path=args.fixed_seeker_attestation,
        judge_isolation_report_path=args.judge_isolation_report,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, freeze.model_dump(mode="json"))


if __name__ == "__main__":
    main()
