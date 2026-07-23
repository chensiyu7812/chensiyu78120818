#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.paid_run_release import require_paid_run_release
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.freeze import require_study_freeze
from metacom_pm.fixed_seeker_contract import require_fixed_seeker_v3_sidecar_contract
from metacom_pm.evoemo import (
    FIXED_SEEKER_V23_STAGE,
    evo_memory_global_catalog_digest,
    fixed_seeker_cost_planning_contract,
    load_evoemo,
)
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, read_json, sha256_file, sha256_text
from metacom_pm.pm_v2_evoemo import (
    EVALUATION_UNIT_CONTRACT_PROTOCOL,
    run_pmv2_fixed_evoemo,
)
from metacom_pm.pm_v2_external_eval import expected_external_units
from metacom_pm.response_mechanism_contract import (
    build_response_mechanism_contract,
    require_matching_response_mechanism_contract,
)
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    require_unified_semantic_query_contract,
    semantic_encoder_spec_from_config,
    semantic_runtime_contract_from_config,
)

ROOT = Path(__file__).resolve().parents[2]


def resolve_condition_paths(
    condition: str,
    *,
    checkpoint: Path | None,
    out_dir: Path | None,
) -> tuple[Path, Path]:
    defaults = {
        "pm_v2": (
            ROOT / "outputs" / "pm_v1_5_model" / "pm_v1_5.joblib",
            ROOT / "outputs" / "evoemo_pm_v1_5",
        ),
        "pm_v1_5_transparent_rule_step0": (
            ROOT
            / "outputs"
            / "pm_v1_5_model"
            / "pm_v1_5_transparent_rule.joblib",
            ROOT / "outputs" / "evoemo_pm_v1_5_transparent_rule",
        ),
        "pm_v2_cost_matched_fixed": (
            ROOT / "outputs" / "pm_v1_5_model" / "cost_matched_fixed.joblib",
            ROOT / "outputs" / "evoemo_pm_v1_5_cost_matched_fixed",
        ),
        "pm_v2_me_r0_fixed": (
            ROOT / "outputs" / "pm_v1_5_model" / "me_r0_fixed.joblib",
            ROOT / "outputs" / "evoemo_pm_v1_5_me_r0_fixed",
        ),
    }
    if condition not in defaults:
        raise ValueError(f"unknown PM-v1.5 condition: {condition}")
    default_checkpoint, default_out_dir = defaults[condition]
    return checkpoint or default_checkpoint, out_dir or default_out_dir


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
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--pm-training-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "training_report.json",
    )
    parser.add_argument(
        "--evidence-filter-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "evidence_filter.joblib",
    )
    parser.add_argument(
        "--evidence-filter-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "training_report.json",
    )
    parser.add_argument(
        "--evidence-filter-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_evidence_filter" / "artifact_attestation.json",
    )
    parser.add_argument("--condition", default="pm_v2")
    parser.add_argument(
        "--fixed-tracks",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_fixed_tracks_v1_5_v3_formal_candidate"
        / "fixed_seeker_tracks.jsonl",
    )
    parser.add_argument(
        "--fixed-tracks-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_fixed_tracks_v1_5_v3_formal_candidate"
        / "artifact_attestation.json",
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
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-api-calls", type=int, default=2000)
    parser.add_argument("--max-estimated-usd", type=float, default=5.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--input-usd-per-mtok", type=float)
    parser.add_argument("--output-usd-per-mtok", type=float)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs" / "pm_v1_5_study_freeze.json")
    args = parser.parse_args()

    args.checkpoint, args.out_dir = resolve_condition_paths(
        args.condition, checkpoint=args.checkpoint, out_dir=args.out_dir
    )
    if (
        args.fixed_tracks.parent.name != "evoemo_fixed_tracks_v1_5_v3_formal_candidate"
        or args.fixed_tracks_attestation.parent != args.fixed_tracks.parent
    ):
        raise RuntimeError(
            "PM-v1.5 requires its isolated "
            "outputs/evoemo_fixed_tracks_v1_5_v3_formal_candidate bundle"
        )
    if args.fixed_tracks_attestation.is_file():
        if read_json(args.fixed_tracks_attestation).get("stage") != FIXED_SEEKER_V23_STAGE:
            raise RuntimeError(
                "PM-v1.5 fixed-seeker generation requires a "
                f"{FIXED_SEEKER_V23_STAGE} bundle"
            )

    if args.run and args.overwrite:
        raise RuntimeError(
            "paid API runs prohibit --overwrite; use a new output directory"
        )

    config = load_config(args.config)
    pm_v2_config = load_config(args.pm_v2_config)
    require_unified_semantic_query_contract(pm_v2_config)
    semantic_encoder = None
    semantic_runtime_verification = None
    if args.condition in {"pm_v2", "pm_v1_5_transparent_rule_step0"}:
        # Resolve the exact local snapshot before any paid client can be used.
        # A missing dependency, cache entry, or hash mismatch is a hard NO-RUN.
        semantic_encoder = FrozenTransformerSemanticEncoder.load(
            semantic_encoder_spec_from_config(pm_v2_config)
        )
        semantic_runtime_verification = require_semantic_runtime_contract(
            pm_v2_config, semantic_encoder
        )
    require_paid_run_release(
        pm_v2_config,
        config_path=args.pm_v2_config,
        stage="external_learned_or_rule_generation",
        run=bool(args.run),
        run_identity=args.accept_cost_estimate_sha256,
    )
    supporter_generation_contract = SupporterGenerationContract.from_config(
        pm_v2_config
    )
    fixed_seeker_contract = require_fixed_seeker_v3_sidecar_contract(
        args.fixed_seeker_contract
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
    # PM-v1.5: the Evidence Filter (post-retrieval item-level filtering) is
    # explicitly out of scope for this track -- PM-v1.5 is a pure
    # pre-retrieval source router, matching the original V1 system design
    # (the evidence filter was a separate diagnostic add-on there too, not
    # part of the main system). The original script hard-requires it to be
    # enabled with trained artifacts; this fork disables it unconditionally
    # instead of training a placeholder model.
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        {**pm_v2_config["evidence_filter"], "enabled": False}
    )
    memory_helpfulness_model = None
    evidence_filter_model_binding = {
        "mode": "disabled_for_pm_v1_5",
        "reason": "Evidence Filter is out of scope for PM-v1.5; PM is a pure pre-retrieval router.",
    }
    evoemo_path = ROOT / "data" / "external" / "evo_emo.json"
    strategy_path = ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        # Evidence Filter artifacts are intentionally not required here: EF is
        # disabled for PM-v1.5 (see the evidence_filter_config override
        # above), so there is no trained filter checkpoint/report to freeze.
        required_files=[
            args.pm_v2_config,
            evoemo_path,
            strategy_path,
            args.checkpoint,
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
    if semantic_encoder is not None and (
        contract.get("semantic_encoder_spec")
        != semantic_encoder.spec.model_dump(mode="json")
        or contract.get("semantic_encoder_binding")
        != semantic_encoder.binding.model_dump(mode="json")
    ):
        raise RuntimeError(
            "study freeze semantic encoder contract is absent or stale"
        )
    if semantic_encoder is not None:
        expected_runtime = semantic_runtime_contract_from_config(pm_v2_config)
        if (
            contract.get("semantic_runtime_contract")
            != expected_runtime.model_dump(mode="json")
            or contract.get("semantic_runtime_contract_sha256")
            != expected_runtime.digest()
            or semantic_runtime_verification is None
            or semantic_runtime_verification.get("contract_sha256")
            != expected_runtime.digest()
        ):
            raise RuntimeError(
                "study freeze semantic runtime contract is absent or stale"
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
    # PM-v1.5: the freeze's evoemo_sha256 only pins the raw input file, not
    # what build_evo_memory actually constructs from it (MP/MS/ME item
    # content, chunking, ids). Independently recompute the same digest this
    # run's own memory catalog would have and fail closed on any mismatch,
    # so a memory-builder change that slipped in after the freeze cannot
    # silently generate over a different catalog than was frozen.
    live_evo_memory_digest = evo_memory_global_catalog_digest(load_evoemo(evoemo_path))
    if (
        contract.get("evo_memory_builder_contract_sha256")
        != live_evo_memory_digest["builder_contract_sha256"]
        or contract.get("evo_memory_global_catalog_sha256")
        != live_evo_memory_digest["global_catalog_sha256"]
    ):
        raise RuntimeError(
            "study freeze EvoEmo memory catalog is absent or stale"
        )
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
    live_response_mechanism_contract = build_response_mechanism_contract(
        project_root=ROOT,
        supporter_generation_contract=supporter_generation_contract,
        generator_endpoint_sha256=endpoint_sha,
        strategy_bank_sha256=sha256_file(strategy_path),
        memory_min_score=contract.get("memory_min_score"),
        strategy_min_score=contract.get("strategy_min_score"),
        strategy_top_k=int(contract["strategy_top_k"]),
        evidence_filter_enabled=bool(evidence_filter_config.enabled),
    )
    require_matching_response_mechanism_contract(
        expected=contract.get("response_mechanism_contract") or {},
        actual=live_response_mechanism_contract,
        context="study freeze vs live EvoEmo external generation",
    )
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
        "pm_v1_5_transparent_rule_step0": contract.get(
            "transparent_rule_checkpoint_sha256"
        ),
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
    semantic_condition = args.condition in {
        "pm_v2",
        "pm_v1_5_transparent_rule_step0",
    }
    if semantic_condition and (
        not args.pm_training_report.is_file()
        or sha256_file(args.pm_training_report)
        != contract.get("policy_training_report_sha256")
    ):
        raise RuntimeError(
            "semantic external condition requires the exact frozen training report"
        )
    cost_match_reference_dirs = {
        "pm_v2": ROOT / "outputs" / "evoemo_pm_v1_5_cost_matched_fixed",
        "pm_v2_cost_matched_fixed": ROOT / "outputs" / "evoemo_pm_v1_5",
    }
    cost_match_reference_dir = cost_match_reference_dirs.get(args.condition)

    result = run_pmv2_fixed_evoemo(
        evoemo_path,
        strategy_path,
        args.checkpoint,
        args.fixed_tracks,
        args.out_dir,
        project_root=ROOT,
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
        fixed_seeker_required_stage=FIXED_SEEKER_V23_STAGE,
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
        require_complete_strategy_family_catalog=True,
        evidence_filter_config=evidence_filter_config,
        memory_helpfulness_model=memory_helpfulness_model,
        evidence_filter_model_binding=evidence_filter_model_binding,
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
        semantic_encoder=semantic_encoder,
        semantic_runtime_verification=semantic_runtime_verification,
        development_training_report_path=(
            args.pm_training_report if semantic_condition else None
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
