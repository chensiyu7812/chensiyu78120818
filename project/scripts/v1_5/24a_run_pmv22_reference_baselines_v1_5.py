#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.artifacts import require_content_addressed_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.evoemo import (
    FIXED_SEEKER_V23_STAGE,
    evo_memory_global_catalog_digest,
    fixed_seeker_cost_planning_contract,
    load_evoemo,
)
from metacom_pm.fixed_seeker_contract import (
    require_fixed_seeker_v3_formal_bundle,
    require_fixed_seeker_v3_sidecar_contract,
)
from metacom_pm.freeze import require_study_freeze
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.pm_v22_reference_baselines import (
    PMV22_REFERENCE_BASELINE_STAGE,
    persist_reference_baseline_dry_run,
    plan_reference_baselines,
    run_reference_baselines,
)


ROOT = Path(__file__).resolve().parents[2]


def _fixed_seeker_binding(
    *,
    experiment_config: dict,
    fixed_seeker_contract_path: Path,
    evoemo_path: Path,
    fixed_tracks_path: Path,
    fixed_tracks_attestation_path: Path,
) -> tuple[dict, str]:
    bundle_dir = fixed_tracks_path.resolve().parent
    require_content_addressed_attestation(
        fixed_tracks_attestation_path,
        required_stage=FIXED_SEEKER_V23_STAGE,
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
    contract = require_fixed_seeker_v3_sidecar_contract(fixed_seeker_contract_path)
    endpoint = endpoint_from_config(experiment_config, contract.seeker_endpoint)
    expected = contract.bind_endpoint(contract.seeker_endpoint, endpoint)
    if payload != expected.payload() or digest != expected.digest():
        raise RuntimeError(
            "fixed-seeker bundle differs from the current PM-v1.5 treatment"
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
            "Treatment-matched PM-v1.5 external reference baselines. A "
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
        default=ROOT / "configs" / "pm_v1_5.yaml",
    )
    parser.add_argument(
        "--evoemo",
        type=Path,
        default=ROOT / "data" / "external" / "evo_emo.json",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_study_freeze.json",
        help=(
            "PM-v1.5 lightweight freeze (scripts/v1_5_create_freeze.py). Cross-"
            "checked against this script's own computed supporter/fixed-seeker/"
            "evidence-filter treatment so baselines cannot silently drift from "
            "what scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py (PM) actually used."
        ),
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "pm_v1_5.joblib",
    )
    parser.add_argument(
        "--policy-training-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "training_report.json",
    )
    parser.add_argument(
        "--fixed-tracks-bundle-binding",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "fixed_seeker_v3_formal_bundle_v1.json",
        help=(
            "Tracked, content-addressed binding that locates and verifies "
            "the formal V3 fixed-seeker bundle -- never a hardcoded output "
            "directory basename."
        ),
    )
    parser.add_argument(
        "--fixed-seeker-contract",
        type=Path,
        default=ROOT / "configs" / "pm_v1_5_fixed_seeker_v3.json",
        help=(
            "V3 sidecar contract (configs/pm_v1_5.yaml itself deliberately "
            "stays on the historical V2 treatment; editing it directly was "
            "shown to invalidate the already-qualified V8.19.2 lineage via a "
            "pm_v1_5_config hash mismatch)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        # V1.5-specific directory, not the shared PMV22_REFERENCE_BASELINE_STAGE
        # name (that constant still identifies the internal attestation
        # "stage" value, which is intentionally left unchanged below).
        default=ROOT / "outputs" / "evoemo_pm_v1_5_reference_baselines",
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
    fixed_seeker_bundle = require_fixed_seeker_v3_formal_bundle(
        args.fixed_tracks_bundle_binding,
        root=ROOT,
        required_stage=FIXED_SEEKER_V23_STAGE,
    )
    args.fixed_tracks = fixed_seeker_bundle["fixed_tracks_path"]
    args.fixed_tracks_attestation = fixed_seeker_bundle["artifact_attestation_path"]
    legacy_names = {"evoemo_selective", "evoemo_fixed_tracks"}
    if args.out_dir.name in legacy_names or args.fixed_tracks.parent.name in legacy_names:
        raise RuntimeError("legacy EvoEmo generation artifacts are forbidden")

    experiment = load_config(args.config)
    pm_v2 = load_config(args.pm_v2_config)
    require_paid_run_release(
        pm_v2,
        config_path=args.pm_v2_config,
        stage="external_reference_baseline_generation",
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    supporter_contract = SupporterGenerationContract.from_config(pm_v2)
    generator = endpoint_from_config(
        experiment, supporter_contract.generator_endpoint
    )
    fixed_payload, fixed_sha256 = _fixed_seeker_binding(
        experiment_config=experiment,
        fixed_seeker_contract_path=args.fixed_seeker_contract,
        evoemo_path=args.evoemo,
        fixed_tracks_path=args.fixed_tracks,
        fixed_tracks_attestation_path=args.fixed_tracks_attestation,
    )
    # PM-v1.5: Evidence Filter is out of scope (see scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py
    # docstring). Must stay disabled identically across development sweep,
    # external generation, and these reference baselines -- an inconsistency
    # here would repeat the same class of treatment mismatch V1.5 exists to fix.
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        {**pm_v2["evidence_filter"], "enabled": False}
    )
    evidence_filter_model = None
    evidence_filter_binding = {
        "mode": "disabled_for_pm_v1_5",
        "reason": "Evidence Filter is out of scope for PM-v1.5; PM is a pure pre-retrieval router.",
    }

    # PM-v1.5: cross-check this script's own independently-computed treatment
    # against the same lightweight freeze scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py
    # (PM) is bound to, so baselines cannot silently use a different
    # supporter/fixed-seeker/evidence-filter treatment than PM did. This
    # replaces the original script's `require_strategy_bank_human_approval`
    # call, which PM-v1.5 does not perform (see docs/PM_V1_5_CORE_CHAIN_PLAN_ZH.md).
    freeze_verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[args.pm_v2_config, args.evoemo, args.strategy_bank, args.fixed_tracks],
    )
    freeze_sha = str(freeze_verification["freeze_sha256"])
    freeze_data = json.loads(args.freeze.read_text(encoding="utf-8"))
    freeze_contract = (freeze_data.get("notes") or {}).get("generation_contract") or {}
    if not freeze_contract:
        raise RuntimeError("study freeze lacks generation_contract")
    if (
        freeze_contract.get("supporter_generation_treatment") != supporter_contract.payload()
        or freeze_contract.get("supporter_generation_treatment_sha256") != supporter_contract.digest()
    ):
        raise RuntimeError(
            "reference baselines' supporter-generation treatment does not match the PM-v1.5 freeze"
        )
    if (
        freeze_contract.get("fixed_seeker_generation_treatment") != fixed_payload
        or freeze_contract.get("fixed_seeker_generation_treatment_sha256") != fixed_sha256
    ):
        raise RuntimeError(
            "reference baselines' fixed-seeker treatment does not match the PM-v1.5 freeze"
        )
    if (
        freeze_contract.get("evidence_filter") != evidence_filter_config.payload()
        or freeze_contract.get("evidence_filter_model") != evidence_filter_binding
    ):
        raise RuntimeError(
            "reference baselines' Evidence Filter binding does not match the PM-v1.5 freeze"
        )
    # PM-v1.5: the freeze's evoemo_sha256 only pins the raw input file, not
    # what build_evo_memory actually constructs from it (MP/MS/ME item
    # content, chunking, ids). Independently recompute the same digest this
    # script's own memory catalog would have and fail closed on any
    # mismatch, so a memory-builder change that slipped in after the freeze
    # cannot silently produce reference baselines over a different catalog
    # than PM was frozen against.
    live_evo_memory_digest = evo_memory_global_catalog_digest(load_evoemo(args.evoemo))
    if (
        freeze_contract.get("evo_memory_builder_contract_sha256")
        != live_evo_memory_digest["builder_contract_sha256"]
        or freeze_contract.get("evo_memory_global_catalog_sha256")
        != live_evo_memory_digest["global_catalog_sha256"]
    ):
        raise RuntimeError(
            "reference baselines' EvoEmo memory catalog does not match the PM-v1.5 freeze"
        )

    external = dict(pm_v2["external_evaluation"])
    retrieval = dict(pm_v2["retrieval"])
    protocol = dict(experiment.get("protocol") or {})
    seeds = [int(value) for value in protocol.get("robustness_seeds") or []]
    if not seeds:
        raise RuntimeError("experiment protocol robustness_seeds is empty")
    if int(protocol.get("evoemo_session_rag_top_k") or 0) < 1:
        raise RuntimeError("experiment session-RAG top-k must be positive")
    if int(freeze_contract.get("session_rag_top_k") or 0) != int(
        protocol["evoemo_session_rag_top_k"]
    ):
        raise RuntimeError("reference session-RAG top-k differs from the freeze")
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

    # PM-v1.5 has no human Strategy Bank approval. The deterministic
    # leakage/lineage audit plus the escN-overlap rule are its declared bank
    # acceptance procedure, and the freeze binds this run to the exact bank
    # SHA used by the learned policy condition.
    strategy_bank_approval = {
        "protocol": "pm-v1.5-strategy-bank-lineage-overlap-audit-v1",
        "decision": "ACCEPTED_WITHOUT_HUMAN_REVIEW",
        "human_calibration_performed": False,
    }
    print(
        json.dumps(
            run_reference_baselines(
                out_dir=args.out_dir,
                fixed_tracks_attestation_path=args.fixed_tracks_attestation,
                pm_v2_config_path=args.pm_v2_config,
                evidence_filter_checkpoint_path=None,
                evidence_filter_report_path=None,
                evidence_filter_attestation_path=None,
                strategy_bank_approval_path=args.freeze,
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
