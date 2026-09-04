#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.evidence_filter_model import require_evidence_filter_artifacts
from metacom_pm.freeze import require_study_freeze
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.evoemo import fixed_seeker_cost_planning_contract
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, sha256_file, sha256_text
from metacom_pm.pm_v2_evoemo import (
    EVALUATION_UNIT_CONTRACT_PROTOCOL,
    run_pmv2_fixed_evoemo,
)
from metacom_pm.pm_v2_external_eval import expected_external_units

ROOT = Path(__file__).resolve().parents[1]


def resolve_frozen_generator_pricing(
    contract: dict,
    *,
    input_override: float | None,
    output_override: float | None,
) -> dict[str, float]:
    pricing = {
        str(key): float(value)
        for key, value in dict(
            contract.get("generator_pricing_usd_per_mtok") or {}
        ).items()
    }
    if pricing != {"input": 0.15, "output": 0.60}:
        raise RuntimeError(
            "study freeze lacks the exact conservative 0.15/0.60 generator pricing"
        )
    overrides = {"input": input_override, "output": output_override}
    for key, override in overrides.items():
        if override is not None and float(override) != pricing[key]:
            raise RuntimeError(
                f"--{key}-usd-per-mtok must equal the frozen value {pricing[key]}"
            )
    return pricing


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "pm_v2.joblib",
    )
    parser.add_argument(
        "--evidence-filter-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_evidence_filter" / "evidence_filter.joblib",
    )
    parser.add_argument(
        "--evidence-filter-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_evidence_filter" / "training_report.json",
    )
    parser.add_argument(
        "--evidence-filter-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_evidence_filter" / "artifact_attestation.json",
    )
    parser.add_argument("--condition", default="pm_v2")
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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "evoemo_pm_v2")
    parser.add_argument("--max-api-calls", type=int, default=2000)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--input-usd-per-mtok", type=float)
    parser.add_argument("--output-usd-per-mtok", type=float)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs" / "pm_v2_study_freeze.json")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError(
            "paid API runs prohibit --overwrite; use a new output directory"
        )

    config = load_config(args.config)
    pm_v2_config = load_config(args.pm_v2_config)
    supporter_generation_contract = SupporterGenerationContract.from_config(
        pm_v2_config
    )
    fixed_seeker_contract = FixedSeekerGenerationContract.from_mapping(
        pm_v2_config["fixed_seeker_generation_treatment"]
    )
    fixed_seeker_endpoint = endpoint_from_config(
        config, fixed_seeker_contract.seeker_endpoint
    )
    bound_fixed_seeker_contract = fixed_seeker_contract.bind_endpoint(
        fixed_seeker_contract.seeker_endpoint, fixed_seeker_endpoint
    )
    fixed_seeker_cost_planning = fixed_seeker_cost_planning_contract(
        pm_v2_config.get("fixed_seeker_cost_planning") or {}
    )
    fixed_seeker_cost_planning_sha256 = sha256_text(
        canonical_json(fixed_seeker_cost_planning)
    )
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        pm_v2_config["evidence_filter"]
    )
    if not evidence_filter_config.enabled:
        raise RuntimeError("PM-v2 external main conditions require Evidence Filter")
    memory_helpfulness_model, evidence_filter_model_binding = (
        require_evidence_filter_artifacts(
            checkpoint_path=args.evidence_filter_checkpoint,
            report_path=args.evidence_filter_report,
            attestation_path=args.evidence_filter_attestation,
            pm_v2_config_path=args.pm_v2_config,
        )
    )
    evoemo_path = ROOT / "data" / "external" / "evo_emo.json"
    strategy_path = ROOT / "data" / "strategy" / "strategy_cards.jsonl"
    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[
            args.pm_v2_config,
            evoemo_path,
            strategy_path,
            args.checkpoint,
            args.evidence_filter_checkpoint,
            args.evidence_filter_report,
            args.evidence_filter_attestation,
            args.fixed_tracks,
            args.fixed_tracks_attestation,
        ],
    )
    freeze_sha = str(verification["freeze_sha256"])
    freeze_data = json.loads(args.freeze.read_text(encoding="utf-8"))
    notes = freeze_data.get("notes") or {}
    contract = notes.get("generation_contract") or {}
    if not contract:
        raise RuntimeError("study freeze lacks generation_contract")
    if (
        contract.get("supporter_generation_treatment")
        != supporter_generation_contract.payload()
        or contract.get("supporter_generation_treatment_sha256")
        != supporter_generation_contract.digest()
    ):
        raise RuntimeError(
            "study freeze supporter-generation treatment is absent or stale"
        )
    if (
        contract.get("fixed_seeker_generation_treatment")
        != bound_fixed_seeker_contract.payload()
        or contract.get("fixed_seeker_generation_treatment_sha256")
        != bound_fixed_seeker_contract.digest()
    ):
        raise RuntimeError(
            "study freeze fixed-seeker generation treatment is absent or stale"
        )
    if (
        contract.get("fixed_seeker_cost_planning")
        != fixed_seeker_cost_planning
        or contract.get("fixed_seeker_cost_planning_sha256")
        != fixed_seeker_cost_planning_sha256
    ):
        raise RuntimeError(
            "study freeze fixed-seeker cost-planning contract is absent or stale"
        )
    if (
        contract.get("evidence_filter") != evidence_filter_config.payload()
        or contract.get("evidence_filter_config_sha256")
        != evidence_filter_config.digest()
        or contract.get("evidence_filter_model") != evidence_filter_model_binding
    ):
        raise RuntimeError("study freeze Evidence Filter contract is absent or stale")
    generator_pricing = resolve_frozen_generator_pricing(
        contract,
        input_override=args.input_usd_per_mtok,
        output_override=args.output_usd_per_mtok,
    )
    endpoint_name = supporter_generation_contract.generator_endpoint
    endpoint = endpoint_from_config(config, endpoint_name)
    endpoint_sha = sha256_text(
        canonical_json(
            {"model": endpoint.model, "family": endpoint.family, "base_url": endpoint.base_url}
        )
    )
    if endpoint_sha != contract.get("generator_endpoint_sha256"):
        raise RuntimeError("generator endpoint changed after study freeze")
    simulator_id = str(contract["simulator_id"])
    max_turns = int(contract["max_turns"])
    seeds = [int(value) for value in contract["seeds"]]
    evaluation_unit_contract = dict(contract.get("evaluation_unit_contract") or {})
    external_contract = notes.get("external_evaluation_contract") or {}
    if (
        not evaluation_unit_contract
        or evaluation_unit_contract
        != external_contract.get("evaluation_unit_contract")
        or evaluation_unit_contract.get("protocol")
        != EVALUATION_UNIT_CONTRACT_PROTOCOL
    ):
        raise RuntimeError(
            "study freeze lacks one identical generation/external evaluation-unit contract"
        )
    evaluation_turn_indices = [
        int(value)
        for value in evaluation_unit_contract.get("evaluation_turn_indices") or []
    ]
    if evaluation_turn_indices != [
        int(value) for value in external_contract.get("turn_indices") or []
    ]:
        raise RuntimeError("frozen generation and scoring turn indices differ")
    expected_units = expected_external_units(
        evoemo_path,
        seeds=seeds,
        simulator_id=simulator_id,
        turn_indices=evaluation_turn_indices,
    )
    if (
        len(expected_units) != int(evaluation_unit_contract["expected_unit_count"])
        or sha256_text(canonical_json(expected_units))
        != evaluation_unit_contract["expected_units_sha256"]
    ):
        raise RuntimeError(
            "current EvoEmo data does not equal the frozen sparse generation universe"
        )
    strategy_action_tokens = int(contract["strategy_action_tokens"])
    input_token_safety_factor = float(contract["input_token_safety_factor"])
    fail_on_reported_input_overrun = bool(
        contract["fail_on_reported_input_overrun"]
    )
    if input_token_safety_factor < 1.0 or not fail_on_reported_input_overrun:
        raise RuntimeError("study freeze has an invalid API cost-planning contract")
    gates = dict(contract["action_preflight_gates"])
    condition_checkpoint_hashes = {
        "pm_v2": notes.get("checkpoint_sha256"),
        "pm_v2_cost_matched_fixed": (notes.get("fixed_baselines") or {}).get(
            "cost_matched_fixed", {}
        ).get("checkpoint_sha256"),
        "pm_v2_me_r0_fixed": (notes.get("fixed_baselines") or {}).get(
            "event_memory_r0", {}
        ).get("checkpoint_sha256"),
    }
    if args.condition not in condition_checkpoint_hashes:
        raise RuntimeError("condition is not a frozen PM-v2 generation condition")
    if sha256_file(args.checkpoint) != condition_checkpoint_hashes[args.condition]:
        raise RuntimeError("condition/checkpoint pairing does not match study freeze")
    cost_match_reference_dirs = {
        "pm_v2": ROOT / "outputs" / "evoemo_pm_v2_cost_matched_fixed",
        "pm_v2_cost_matched_fixed": ROOT / "outputs" / "evoemo_pm_v2",
    }
    cost_match_reference_dir = cost_match_reference_dirs.get(args.condition)

    result = run_pmv2_fixed_evoemo(
        evoemo_path,
        strategy_path,
        args.checkpoint,
        args.fixed_tracks,
        args.out_dir,
        generator_endpoint=endpoint,
        supporter_generation_contract=supporter_generation_contract,
        fixed_seeker_generation_contract=(
            bound_fixed_seeker_contract.payload()
        ),
        fixed_seeker_generation_contract_sha256=(
            bound_fixed_seeker_contract.digest()
        ),
        simulator_id=simulator_id,
        fixed_tracks_attestation_path=args.fixed_tracks_attestation,
        condition=args.condition,
        max_turns=max_turns,
        seeds=seeds,
        evaluation_unit_contract=evaluation_unit_contract,
        max_scenarios=None,
        overwrite=args.overwrite,
        strategy_action_tokens=strategy_action_tokens,
        study_freeze_sha256=freeze_sha,
        run=bool(args.run),
        accept_cost_estimate_sha256=args.accept_cost_estimate_sha256,
        max_api_calls=args.max_api_calls,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        input_usd_per_mtok=generator_pricing["input"],
        output_usd_per_mtok=generator_pricing["output"],
        input_token_safety_factor=input_token_safety_factor,
        fail_on_reported_input_overrun=fail_on_reported_input_overrun,
        strategy_top_k=int(contract["strategy_top_k"]),
        memory_min_score=contract.get("memory_min_score"),
        strategy_min_score=contract.get("strategy_min_score"),
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        action_preflight_gates=gates,
        maximum_cost_matched_relative_deviation=float(
            contract["maximum_cost_matched_relative_deviation"]
        ),
        cost_match_reference_call_plan_path=(
            cost_match_reference_dir / "call_plan.jsonl"
            if cost_match_reference_dir is not None
            else None
        ),
        cost_match_reference_cost_estimate_path=(
            cost_match_reference_dir / "cost_estimate.json"
            if cost_match_reference_dir is not None
            else None
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
