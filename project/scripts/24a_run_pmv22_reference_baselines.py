#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.artifacts import require_content_addressed_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.evidence_filter_model import require_evidence_filter_artifacts
from metacom_pm.evoemo import (
    FIXED_SEEKER_V22_STAGE,
    fixed_seeker_cost_planning_contract,
)
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.pm_v22_reference_baselines import (
    PMV22_REFERENCE_BASELINE_STAGE,
    persist_reference_baseline_dry_run,
    plan_reference_baselines,
    run_reference_baselines,
)
from metacom_pm.strategy_bank_approval import (
    require_strategy_bank_human_approval,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixed_seeker_binding(
    *,
    experiment_config: dict,
    pm_v2_config: dict,
    evoemo_path: Path,
    fixed_tracks_path: Path,
    fixed_tracks_attestation_path: Path,
) -> tuple[dict, str]:
    bundle_dir = fixed_tracks_path.resolve().parent
    require_content_addressed_attestation(
        fixed_tracks_attestation_path,
        required_stage=FIXED_SEEKER_V22_STAGE,
        relocated_inputs={
            "evoemo": evoemo_path,
            "run_manifest": bundle_dir / "run_manifest.json",
            "cost_estimate": bundle_dir / "cost_estimate.json",
            "call_plan": bundle_dir / "call_plan.jsonl",
        },
        relocated_outputs={
            "tracks": fixed_tracks_path,
            "raw_calls": bundle_dir / "raw_seeker_calls.jsonl",
            "physical_attempt_ledger": bundle_dir
            / "physical_attempt_ledger.jsonl",
            "summary": bundle_dir / "summary.json",
        },
    )
    attestation = read_json(fixed_tracks_attestation_path)
    parameters = dict(attestation.get("parameters") or {})
    payload = parameters.get("fixed_seeker_generation_contract")
    digest = str(
        parameters.get("fixed_seeker_generation_contract_sha256") or ""
    )
    if not isinstance(payload, dict) or digest != sha256_text(
        canonical_json(payload)
    ):
        raise RuntimeError("fixed-seeker attestation treatment binding is stale")
    raw = pm_v2_config.get("fixed_seeker_generation_treatment")
    if not isinstance(raw, dict):
        raise RuntimeError("pm_v2 config lacks fixed_seeker_generation_treatment")
    contract = FixedSeekerGenerationContract.from_mapping(raw)
    endpoint = endpoint_from_config(experiment_config, contract.seeker_endpoint)
    expected = contract.bind_endpoint(contract.seeker_endpoint, endpoint)
    if payload != expected.payload() or digest != expected.digest():
        raise RuntimeError(
            "fixed-seeker bundle differs from the current PM-v2.2 treatment"
        )
    if int((payload.get("treatment") or {}).get("max_output_tokens") or 0) != 300:
        raise RuntimeError("legacy 60-token-cap fixed tracks are forbidden")
    expected_cost_planning = fixed_seeker_cost_planning_contract(
        pm_v2_config.get("fixed_seeker_cost_planning") or {}
    )
    expected_cost_sha256 = sha256_text(
        canonical_json(expected_cost_planning)
    )
    if (
        parameters.get("fixed_seeker_cost_planning")
        != expected_cost_planning
        or parameters.get("fixed_seeker_cost_planning_sha256")
        != expected_cost_sha256
    ):
        raise RuntimeError("fixed-seeker attestation cost-planning binding is stale")
    summary = read_json(bundle_dir / "summary.json")
    cost_estimate = read_json(bundle_dir / "cost_estimate.json")
    expected_section = dict(attestation.get("expected") or {})
    if (
        summary.get("status") != "COMPLETE"
        or int(summary.get("completion_truncated_count", -1)) != 0
        or summary.get("fixed_seeker_generation_contract") != payload
        or summary.get("fixed_seeker_generation_contract_sha256") != digest
        or summary.get("fixed_seeker_cost_planning")
        != expected_cost_planning
        or summary.get("fixed_seeker_cost_planning_sha256")
        != expected_cost_sha256
        or cost_estimate.get("fixed_seeker_cost_planning")
        != expected_cost_planning
        or cost_estimate.get("fixed_seeker_cost_planning_sha256")
        != expected_cost_sha256
        or (cost_estimate.get("budget_gate") or {}).get("status") != "PASS"
        or (summary.get("planned_budget_gate") or {}).get("status") != "PASS"
        or (summary.get("observed_budget_gate") or {}).get("status") != "PASS"
        or parameters.get("planned_budget_gate")
        != summary.get("planned_budget_gate")
        or parameters.get("observed_budget_gate")
        != summary.get("observed_budget_gate")
        or expected_section.get("planned_budget_gate")
        != summary.get("planned_budget_gate")
        or expected_section.get("observed_budget_gate")
        != summary.get("observed_budget_gate")
    ):
        raise RuntimeError(
            "fixed-seeker bundle is incomplete, truncated, or treatment-mixed"
        )
    return payload, digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Treatment-matched PM-v2.2 external reference baselines. A "
            "deterministic positive-price dry run and exact SHA acceptance are "
            "mandatory before any API client is created."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--pm-v2-config",
        type=Path,
        default=ROOT / "configs" / "pm_v2.yaml",
    )
    parser.add_argument(
        "--evoemo",
        type=Path,
        default=ROOT / "data" / "external" / "evo_emo.json",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards.jsonl",
    )
    parser.add_argument(
        "--strategy-bank-approval",
        type=Path,
        default=ROOT
        / "outputs"
        / "strategy_rag_v1_frozen_candidate"
        / "human_approval.json",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "pm_v2.joblib",
    )
    parser.add_argument(
        "--policy-training-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "training_report.json",
    )
    parser.add_argument(
        "--fixed-tracks",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_fixed_tracks_v22"
        / "fixed_seeker_tracks.jsonl",
    )
    parser.add_argument(
        "--fixed-tracks-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_fixed_tracks_v22"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--evidence-filter-checkpoint",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_evidence_filter"
        / "evidence_filter.joblib",
    )
    parser.add_argument(
        "--evidence-filter-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_evidence_filter"
        / "training_report.json",
    )
    parser.add_argument(
        "--evidence-filter-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_evidence_filter"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / PMV22_REFERENCE_BASELINE_STAGE,
    )
    parser.add_argument("--max-api-calls", type=int, default=1000)
    parser.add_argument("--max-estimated-usd", type=float, default=10.0)
    parser.add_argument(
        "--max-input-tokens-per-call", type=int, default=60000
    )
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError("paid reference-baseline runs prohibit --overwrite")
    if args.dry_run and args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "--accept-cost-estimate-sha256 is valid only with --run"
        )
    if args.run and not args.accept_cost_estimate_sha256:
        raise RuntimeError(
            "--run requires --accept-cost-estimate-sha256 from the matching dry run"
        )
    legacy_names = {"evoemo_selective", "evoemo_fixed_tracks"}
    if args.out_dir.name in legacy_names or args.fixed_tracks.parent.name in legacy_names:
        raise RuntimeError("legacy EvoEmo generation artifacts are forbidden")
    if args.fixed_tracks.parent.name != "evoemo_fixed_tracks_v22":
        raise RuntimeError(
            "PM-v2.2 reference baselines require outputs/evoemo_fixed_tracks_v22"
        )

    experiment = load_config(args.config)
    pm_v2 = load_config(args.pm_v2_config)
    supporter_contract = SupporterGenerationContract.from_config(pm_v2)
    generator = endpoint_from_config(
        experiment, supporter_contract.generator_endpoint
    )
    fixed_payload, fixed_sha256 = _fixed_seeker_binding(
        experiment_config=experiment,
        pm_v2_config=pm_v2,
        evoemo_path=args.evoemo,
        fixed_tracks_path=args.fixed_tracks,
        fixed_tracks_attestation_path=args.fixed_tracks_attestation,
    )
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        pm_v2["evidence_filter"]
    )
    if not evidence_filter_config.enabled:
        raise RuntimeError("PM-v2.2 reference baselines require Evidence Filter")
    evidence_filter_model, evidence_filter_binding = (
        require_evidence_filter_artifacts(
            checkpoint_path=args.evidence_filter_checkpoint,
            report_path=args.evidence_filter_report,
            attestation_path=args.evidence_filter_attestation,
            pm_v2_config_path=args.pm_v2_config,
        )
    )
    external = dict(pm_v2["external_evaluation"])
    retrieval = dict(pm_v2["retrieval"])
    protocol = dict(experiment.get("protocol") or {})
    seeds = [int(value) for value in protocol.get("robustness_seeds") or []]
    if not seeds:
        raise RuntimeError("experiment protocol robustness_seeds is empty")
    if int(protocol.get("evoemo_session_rag_top_k") or 0) < 1:
        raise RuntimeError("experiment session-RAG top-k must be positive")
    if (
        int(external["strategy_top_k"]) != int(retrieval["strategy_top_k"])
        or float(external["memory_min_score"])
        != float(retrieval["memory_min_score"])
        or float(external["strategy_min_score"])
        != float(retrieval["strategy_min_score"])
    ):
        raise RuntimeError("external and PM-v2 retrieval settings differ")
    pricing = {
        str(key): float(value)
        for key, value in dict(
            pm_v2["development_sweep"]["pricing_usd_per_mtok"]
        ).items()
    }
    if pricing != {"input": 0.15, "output": 0.60}:
        raise RuntimeError(
            "reference-baseline planning requires frozen positive 0.15/0.60 pricing"
        )
    safety_factor = float(pm_v2["api_cost_planning"]["input_token_safety_factor"])
    if not bool(pm_v2["api_cost_planning"]["fail_on_reported_input_overrun"]):
        raise RuntimeError("reported input-token overrun must fail closed")

    plan_kwargs = {
        "evoemo_path": args.evoemo,
        "strategy_bank_path": args.strategy_bank,
        "fixed_tracks_path": args.fixed_tracks,
        "policy_checkpoint_path": args.policy_checkpoint,
        "policy_training_report_path": args.policy_training_report,
        "generator_endpoint": generator,
        "supporter_generation_contract": supporter_contract,
        "fixed_seeker_generation_treatment": fixed_payload,
        "fixed_seeker_generation_treatment_sha256": fixed_sha256,
        "evidence_filter_config": evidence_filter_config,
        "evidence_filter_model": evidence_filter_model,
        "evidence_filter_model_binding": evidence_filter_binding,
        "simulator_id": str(external["simulator_id"]),
        "max_turns": int(external["max_turns"]),
        "seeds": seeds,
        "turn_indices": [int(value) for value in external["turn_indices"]],
        "memory_min_score": float(external["memory_min_score"]),
        "strategy_min_score": float(external["strategy_min_score"]),
        "strategy_top_k": int(external["strategy_top_k"]),
        "session_rag_top_k": int(protocol["evoemo_session_rag_top_k"]),
        "input_token_safety_factor": safety_factor,
        "input_usd_per_mtok": pricing["input"],
        "output_usd_per_mtok": pricing["output"],
        "max_api_calls": int(args.max_api_calls),
        "max_estimated_usd": float(args.max_estimated_usd),
        "max_input_tokens_per_call": int(args.max_input_tokens_per_call),
    }
    cost_estimate, call_plan = plan_reference_baselines(**plan_kwargs)
    if args.dry_run:
        disposition = persist_reference_baseline_dry_run(
            args.out_dir,
            cost_estimate=cost_estimate,
            call_plan=call_plan,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    **cost_estimate,
                    "dry_run_disposition": disposition,
                    "api_clients_created": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    strategy_bank_approval = require_strategy_bank_human_approval(
        args.strategy_bank_approval,
        strategy_bank_path=args.strategy_bank,
        split_manifest_path=ROOT
        / "data"
        / "strategy"
        / "esconv_split_manifest.jsonl",
        bank_audit_path=ROOT
        / "data"
        / "strategy"
        / "strategy_bank_audit.json",
        strategy_rag_audit_summary_path=ROOT
        / "outputs"
        / "strategy_rag_v1_audit"
        / "audit_summary.json",
        provisional_candidate_manifest_path=ROOT
        / "outputs"
        / "strategy_rag_v1_frozen_candidate"
        / "strategy_rag_manifest.json",
    )
    print(
        json.dumps(
            run_reference_baselines(
                out_dir=args.out_dir,
                fixed_tracks_attestation_path=args.fixed_tracks_attestation,
                pm_v2_config_path=args.pm_v2_config,
                evidence_filter_checkpoint_path=args.evidence_filter_checkpoint,
                evidence_filter_report_path=args.evidence_filter_report,
                evidence_filter_attestation_path=args.evidence_filter_attestation,
                strategy_bank_approval_path=args.strategy_bank_approval,
                strategy_bank_approval=strategy_bank_approval,
                accepted_cost_estimate_sha256=str(
                    args.accept_cost_estimate_sha256
                ),
                **plan_kwargs,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
