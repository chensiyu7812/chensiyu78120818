#!/usr/bin/env python3
"""Evaluate the frozen PM-v1.5 external response set.

This fast-track driver enforces study-freeze, treatment and policy-lock
consistency, an attested 12-unit forced-swap judge-sensitivity canary, a real
batched schema/order pilot, and observed cost matching. It trusts each generation
stage's fail-closed ledger instead of duplicating PM-v2.2's larger raw-ledger
cross-validation layer. The shared evaluator's PM-v2.2-specific forced-swap
*efficacy* flag remains disabled because V1.5's paper claims are decided by
the separately persisted paired non-inferiority/cost confidence intervals;
the canary is transport/judge validation, not evidence that PM helps.

The final scoring layer is V1.5-specific: seven anonymous conditions are scored
together. GPT-4o is the full primary quality judge, Claude is a frozen stratified
sensitivity judge, and evidence-use risk is a separate stratified audit. The
legacy pointwise scorer remains in the repository but is not this main path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typing import Any, Mapping, Sequence

from metacom_pm.artifacts import (
    create_artifact_attestation,
    require_artifact_attestation,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.freeze import require_study_freeze
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v2_evoemo import compare_observed_cost_matched_turns
from metacom_pm.pm_v2_external_eval import (
    expected_external_units,
)
from metacom_pm.v1_5_external_batched import (
    require_v1_5_batched_schema_order_pilot_pass,
    run_v1_5_external_batched_evaluation,
)
from metacom_pm.v1_5_external_claims import assess_external_claims
from metacom_pm.v1_5_forced_swap_canary import (
    KEY_CLAIM_GATE,
    require_v1_5_forced_swap_canary,
)
from metacom_pm.v1_5_latency import (
    PROTOCOL as LATENCY_DIAGNOSTIC_PROTOCOL,
    build_descriptive_latency_report,
)
from metacom_pm.pm_v2_judging import (
    composite_spec_from_config,
    labeling_settings_from_config,
)
from metacom_pm.pm_v22_reference_baselines import PMV22_REFERENCE_BASELINE_STAGE

ROOT = Path(__file__).resolve().parents[2]

GENERATION_STAGE = "evoemo_pm_v2_generation"
KNOWN_STAGES = {GENERATION_STAGE, PMV22_REFERENCE_BASELINE_STAGE}


def require_treatment_bound_turn_file(
    turn_path: Path,
    *,
    expected_conditions: Sequence[str],
    expected_units: Sequence[tuple[str, int, int, str, int]],
    treatment: Mapping[str, Any],
    treatment_sha256: str,
    fixed_seeker_treatment: Mapping[str, Any],
    fixed_seeker_treatment_sha256: str,
) -> list[dict[str, Any]]:
    """Reject mixed-treatment or non-complete turns before any judging call."""

    rows = [dict(row) for row in iter_jsonl(turn_path)]
    conditions = {str(value) for value in expected_conditions}
    if not rows or {str(row.get("condition") or "") for row in rows} != conditions:
        raise RuntimeError("generation turn file does not cover its attested conditions")
    expected = sorted(expected_units)
    for condition in sorted(conditions):
        condition_rows = [row for row in rows if str(row.get("condition") or "") == condition]
        units = sorted(
            (
                str(row.get("user_id") or ""),
                int(row.get("topic_index") or 0),
                int(row.get("seed") or 0),
                str(row.get("simulator_id") or ""),
                int(row.get("turn_index") or 0),
            )
            for row in condition_rows
        )
        if units != expected:
            raise RuntimeError(
                f"generation turns for {condition} differ from the frozen unit universe"
            )
        if any(
            row.get("supporter_generation_treatment") != dict(treatment)
            or row.get("supporter_generation_treatment_sha256") != treatment_sha256
            or row.get("fixed_seeker_generation_treatment") != dict(fixed_seeker_treatment)
            or row.get("fixed_seeker_generation_treatment_sha256") != fixed_seeker_treatment_sha256
            or row.get("normalized_finish_reason") != "complete"
            or not str(row.get("supporter_message") or "").strip()
            for row in condition_rows
        ):
            raise RuntimeError(
                f"generation turns for {condition} contain mixed or non-complete output"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml")
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs" / "pm_v1_5_study_freeze.json")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument(
        "--turn-paths",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs" / "evoemo_pm_v1_5_reference_baselines" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v1_5" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v1_5_cost_matched_fixed" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v1_5_me_r0_fixed" / "turns.jsonl",
        ],
    )
    parser.add_argument(
        "--generation-attestations",
        type=Path,
        nargs="+",
        default=[
            ROOT / "outputs" / "evoemo_pm_v1_5_reference_baselines" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v1_5" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v1_5_cost_matched_fixed" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v1_5_me_r0_fixed" / "artifact_attestation.json",
        ],
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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "pm_v1_5_external_response")
    parser.add_argument("--max-api-calls", type=int, default=400)
    parser.add_argument("--max-estimated-usd", type=float, default=20.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=24000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument(
        "--batched-schema-pilot-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_external_batched_schema_order_pilot"
        / "summary.json",
    )
    parser.add_argument(
        "--batched-schema-pilot-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_external_batched_schema_order_pilot"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--forced-swap-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_forced_swap_canary" / "summary.json",
    )
    parser.add_argument(
        "--forced-swap-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v1_5_forced_swap_canary"
        / "artifact_attestation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.run and args.overwrite:
        raise RuntimeError("paid API runs prohibit --overwrite; use a new output directory")

    config = load_config(args.config)
    pm_v1_5_config = load_config(args.pm_v1_5_config)
    if pm_v1_5_config.get("version") != "pm-v1.5":
        raise RuntimeError("this external evaluation driver requires a pm-v1.5 config")

    supporter_generation_contract = SupporterGenerationContract.from_config(pm_v1_5_config)
    fixed_seeker_contract = FixedSeekerGenerationContract.from_mapping(
        pm_v1_5_config["fixed_seeker_generation_treatment"]
    )
    fixed_seeker_endpoint = endpoint_from_config(config, fixed_seeker_contract.seeker_endpoint)
    bound_fixed_seeker_contract = fixed_seeker_contract.bind_endpoint(
        fixed_seeker_contract.seeker_endpoint, fixed_seeker_endpoint
    )
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        {**pm_v1_5_config["evidence_filter"], "enabled": False}
    )

    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[args.pm_v1_5_config, args.evoemo],
    )
    freeze_data = json.loads(args.freeze.read_text(encoding="utf-8"))
    freeze_sha = str(verification["freeze_sha256"])
    notes = freeze_data.get("notes") or {}
    generation_contract = notes.get("generation_contract") or {}
    external_contract = notes.get("external_evaluation_contract") or {}
    if not generation_contract or not external_contract:
        raise RuntimeError("study freeze lacks the PM-v1.5 generation/external contracts")
    if dict(external_contract.get("key_claim_gate") or {}) != KEY_CLAIM_GATE:
        raise RuntimeError(
            "study freeze key_claim_gate does not match the frozen "
            "src/metacom_pm/v1_5_forced_swap_canary.py:KEY_CLAIM_GATE contract"
        )
    if not KEY_CLAIM_GATE["require_v1_5_forced_swap_canary"]:
        raise RuntimeError("PM-v1.5 external evaluation requires the forced-swap canary")

    if (
        generation_contract.get("supporter_generation_treatment") != supporter_generation_contract.payload()
        or generation_contract.get("supporter_generation_treatment_sha256") != supporter_generation_contract.digest()
    ):
        raise RuntimeError("study freeze supporter-generation treatment is stale")
    if (
        generation_contract.get("fixed_seeker_generation_treatment") != bound_fixed_seeker_contract.payload()
        or generation_contract.get("fixed_seeker_generation_treatment_sha256")
        != bound_fixed_seeker_contract.digest()
    ):
        raise RuntimeError("study freeze fixed-seeker generation treatment is stale")
    if (
        generation_contract.get("evidence_filter") != evidence_filter_config.payload()
        or generation_contract.get("evidence_filter_config_sha256") != evidence_filter_config.digest()
    ):
        raise RuntimeError("study freeze evidence-filter treatment is stale")

    # Prove the PM checkpoint was not swapped after the study freeze: the
    # freeze locked its hash (and the reference-baseline attestation's own
    # recorded policy lock, cross-checked below) before any reference-
    # baseline (24a) generation happened.
    frozen_policy_lock = {
        "policy_checkpoint_sha256": sha256_file(args.policy_checkpoint),
        "policy_training_report_sha256": sha256_file(args.policy_training_report),
        "policy_lock_timing": generation_contract.get("policy_lock_timing"),
        "post_generation_policy_tuning_prohibited": generation_contract.get(
            "post_generation_policy_tuning_prohibited"
        ),
    }
    if any(
        generation_contract.get(key) != expected for key, expected in frozen_policy_lock.items()
    ):
        raise RuntimeError(
            "study freeze does not bind the policy that preceded reference "
            "baseline generation"
        )

    conditions = [str(v) for v in external_contract.get("conditions") or []]
    if not conditions:
        raise RuntimeError("study freeze lacks an external condition matrix")
    treatment = str(external_contract.get("treatment") or "")
    if treatment not in conditions:
        raise RuntimeError("study freeze external treatment is not one of its own conditions")
    turn_indices = [int(v) for v in external_contract.get("turn_indices") or []]
    evaluation_unit_contract = dict(external_contract.get("evaluation_unit_contract") or {})
    if evaluation_unit_contract != generation_contract.get("evaluation_unit_contract"):
        raise RuntimeError("generation/external evaluation-unit contracts differ")

    full_expected_units = expected_external_units(
        args.evoemo,
        seeds=[int(v) for v in generation_contract["seeds"]],
        simulator_id=str(generation_contract["simulator_id"]),
        turn_indices=turn_indices,
    )
    if (
        evaluation_unit_contract.get("evaluation_turn_indices") != turn_indices
        or evaluation_unit_contract.get("expected_unit_count") != len(full_expected_units)
        or evaluation_unit_contract.get("expected_units_sha256")
        != sha256_text(canonical_json(full_expected_units))
    ):
        raise RuntimeError("frozen external evaluation-unit universe is stale")

    excluded_units_raw = external_contract.get("excluded_units") or []
    excluded_units = {
        (str(u[0]), int(u[1]), int(u[2]), str(u[3]), int(u[4])) for u in excluded_units_raw
    }
    if not excluded_units:
        raise RuntimeError(
            "study freeze lacks a schema-smoke hold-out unit (see v1_5_create_freeze.py)"
        )
    expected_units = [unit for unit in full_expected_units if unit not in excluded_units]
    if len(expected_units) + len(excluded_units) != len(full_expected_units):
        raise RuntimeError("schema-smoke hold-out partition is not exact")

    if len(args.turn_paths) != len(args.generation_attestations):
        raise ValueError("each turn path requires one generation attestation")
    attested_conditions: set[str] = set()
    turn_path_by_condition: dict[str, Path] = {}
    validated_turn_rows: list[dict[str, Any]] = []
    for turn_path, attestation_path in zip(args.turn_paths, args.generation_attestations):
        raw_attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        stage = str(raw_attestation.get("stage") or "")
        if stage not in KNOWN_STAGES:
            raise RuntimeError(
                "external evaluation accepts only PM-v1.5 generation artifacts; "
                f"unknown stage: {stage!r}"
            )
        require_artifact_attestation(
            attestation_path,
            required_stage=stage,
            required_output_paths={"turns": turn_path},
            expected_freeze_sha256=(freeze_sha if stage == GENERATION_STAGE else None),
        )
        parameters = raw_attestation.get("parameters") or {}
        values = [
            str(value)
            for value in (parameters.get("conditions") or [parameters.get("condition")])
            if value
        ]
        if not values:
            raise RuntimeError("generation attestation has no conditions")
        attested_conditions.update(values)
        if len(values) == 1:
            condition = values[0]
            if condition in turn_path_by_condition:
                raise RuntimeError(f"duplicate generation turn input for {condition}")
            turn_path_by_condition[condition] = turn_path
        if stage == GENERATION_STAGE:
            frozen_parameter_checks = [
                ("supporter_generation_treatment", generation_contract.get("supporter_generation_treatment")),
                (
                    "supporter_generation_treatment_sha256",
                    generation_contract.get("supporter_generation_treatment_sha256"),
                ),
                ("fixed_seeker_generation_treatment", generation_contract.get("fixed_seeker_generation_treatment")),
                (
                    "fixed_seeker_generation_treatment_sha256",
                    generation_contract.get("fixed_seeker_generation_treatment_sha256"),
                ),
                ("simulator_id", generation_contract.get("simulator_id")),
                ("max_turns", generation_contract.get("max_turns")),
                ("seeds", generation_contract.get("seeds")),
                ("evaluation_unit_contract", generation_contract.get("evaluation_unit_contract")),
                ("protocol", generation_contract.get("protocol")),
                ("paid_generation_scope", generation_contract.get("paid_generation_scope")),
                (
                    "action_preflight_gate_scope",
                    generation_contract.get("action_preflight_gate_scope"),
                ),
                (
                    "all_turn_action_preflight_is_diagnostic_only",
                    generation_contract.get(
                        "all_turn_action_preflight_is_diagnostic_only"
                    ),
                ),
                ("generator_endpoint_sha256", generation_contract.get("generator_endpoint_sha256")),
                ("strategy_action_tokens", generation_contract.get("strategy_action_tokens")),
                ("strategy_top_k", generation_contract.get("strategy_top_k")),
                ("memory_min_score", generation_contract.get("memory_min_score")),
                ("strategy_min_score", generation_contract.get("strategy_min_score")),
                ("evidence_filter", generation_contract.get("evidence_filter")),
                (
                    "evidence_filter_config_sha256",
                    generation_contract.get("evidence_filter_config_sha256"),
                ),
                ("evidence_filter_model", generation_contract.get("evidence_filter_model")),
                ("action_preflight_gates", generation_contract.get("action_preflight_gates")),
                (
                    "maximum_cost_matched_relative_deviation",
                    generation_contract.get(
                        "maximum_cost_matched_relative_deviation"
                    ),
                ),
                ("generator_retries", generation_contract.get("generator_retries")),
                (
                    "generator_pricing_usd_per_mtok",
                    generation_contract.get("generator_pricing_usd_per_mtok"),
                ),
                (
                    "input_token_safety_factor",
                    generation_contract.get("input_token_safety_factor"),
                ),
                (
                    "fail_on_reported_input_overrun",
                    generation_contract.get("fail_on_reported_input_overrun"),
                ),
            ]
            for key, expected in frozen_parameter_checks:
                if parameters.get(key) != expected:
                    raise RuntimeError(
                        f"generation attestation {attestation_path} violates frozen {key}"
                    )
        elif stage == PMV22_REFERENCE_BASELINE_STAGE:
            freeze_input = (raw_attestation.get("inputs") or {}).get(
                "strategy_bank_approval"
            ) or {}
            if (
                Path(str(freeze_input.get("path") or "")).resolve()
                != args.freeze.resolve()
                or freeze_input.get("sha256") != sha256_file(args.freeze)
            ):
                raise RuntimeError(
                    "reference-baseline attestation is not bound to the current freeze"
                )
            if any(
                parameters.get(key) != expected
                for key, expected in frozen_policy_lock.items()
            ):
                raise RuntimeError(
                    f"reference-baseline attestation {attestation_path} was generated "
                    "under a different policy lock than the study freeze"
                )
            reference_checks = {
                "supporter_generation_treatment": generation_contract.get(
                    "supporter_generation_treatment"
                ),
                "supporter_generation_treatment_sha256": generation_contract.get(
                    "supporter_generation_treatment_sha256"
                ),
                "fixed_seeker_generation_treatment": generation_contract.get(
                    "fixed_seeker_generation_treatment"
                ),
                "fixed_seeker_generation_treatment_sha256": generation_contract.get(
                    "fixed_seeker_generation_treatment_sha256"
                ),
                "simulator_id": generation_contract.get("simulator_id"),
                "max_turns": generation_contract.get("max_turns"),
                "seeds": generation_contract.get("seeds"),
                "evaluation_unit_contract": generation_contract.get(
                    "evaluation_unit_contract"
                ),
                "paid_generation_scope": generation_contract.get(
                    "paid_generation_scope"
                ),
                "evidence_filter_config_sha256": generation_contract.get(
                    "evidence_filter_config_sha256"
                ),
                "evidence_filter_model": generation_contract.get(
                    "evidence_filter_model"
                ),
            }
            for key, expected in reference_checks.items():
                if parameters.get(key) != expected:
                    raise RuntimeError(
                        f"reference-baseline attestation violates frozen {key}"
                    )
            if parameters.get("retrieval_settings") != {
                "memory_min_score": float(generation_contract["memory_min_score"]),
                "strategy_min_score": float(
                    generation_contract["strategy_min_score"]
                ),
                "strategy_top_k": int(generation_contract["strategy_top_k"]),
                "session_rag_top_k": int(
                    generation_contract["session_rag_top_k"]
                ),
            }:
                raise RuntimeError(
                    "reference-baseline retrieval settings violate the freeze"
                )
            reference_endpoint = parameters.get("generator_endpoint") or {}
            if sha256_text(
                canonical_json(
                    {
                        "model": reference_endpoint.get("model"),
                        "family": reference_endpoint.get("family"),
                        "base_url": reference_endpoint.get("base_url"),
                    }
                )
            ) != generation_contract.get("generator_endpoint_sha256"):
                raise RuntimeError(
                    "reference-baseline generator endpoint violates the freeze"
                )
        validated_turn_rows.extend(
            require_treatment_bound_turn_file(
                turn_path,
                expected_conditions=values,
                expected_units=full_expected_units,
                treatment=generation_contract["supporter_generation_treatment"],
                treatment_sha256=str(
                    generation_contract["supporter_generation_treatment_sha256"]
                ),
                fixed_seeker_treatment=generation_contract[
                    "fixed_seeker_generation_treatment"
                ],
                fixed_seeker_treatment_sha256=str(
                    generation_contract[
                        "fixed_seeker_generation_treatment_sha256"
                    ]
                ),
            )
        )
    if set(conditions) != attested_conditions:
        raise RuntimeError(
            "generation attestations do not exactly cover the frozen conditions"
        )

    latency_contract = external_contract.get("latency_diagnostic") or {}
    if latency_contract != {
        "protocol": LATENCY_DIAGNOSTIC_PROTOCOL,
        "role": "diagnostic_only",
        "confirmatory_latency_claim_allowed": False,
    }:
        raise RuntimeError("study freeze latency diagnostic contract is stale")
    latency_diagnostic = build_descriptive_latency_report(
        validated_turn_rows,
        conditions=conditions,
        treatment=treatment,
        expected_units=expected_units,
    )
    if any(
        latency_diagnostic.get(key) != expected
        for key, expected in latency_contract.items()
    ):
        raise RuntimeError("latency diagnostic output violates the frozen contract")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latency_path = args.out_dir / "latency_diagnostic.json"
    latency_attestation_path = (
        args.out_dir / "latency_diagnostic_attestation.json"
    )
    write_json(latency_path, latency_diagnostic)
    create_artifact_attestation(
        latency_attestation_path,
        stage="pm_v1_5_descriptive_latency",
        inputs={
            "study_freeze": args.freeze,
            **{
                f"turns_{index}": path
                for index, path in enumerate(args.turn_paths)
            },
            **{
                f"generation_attestation_{index}": path
                for index, path in enumerate(args.generation_attestations)
            },
        },
        outputs={"latency_diagnostic": (latency_path, False)},
        parameters={"contract": latency_contract},
        expected={"conditions": len(conditions), "units": len(expected_units)},
        study_freeze_sha256=freeze_sha,
    )

    forced_swap_verification = require_v1_5_forced_swap_canary(
        args.forced_swap_summary,
        args.forced_swap_attestation,
        study_freeze_sha256=freeze_sha,
        full_expected_units=full_expected_units,
        contract=external_contract.get("forced_swap") or {},
    )
    if set(forced_swap_verification["excluded_units"]) != excluded_units:
        raise RuntimeError("external excluded units differ from the frozen canary sample")

    frozen_judges = external_contract.get("judge_endpoints") or []
    frozen_names = [str(row["name"]) for row in frozen_judges]
    endpoints = [endpoint_from_config(config, name) for name in frozen_names]
    for endpoint, frozen in zip(endpoints, frozen_judges):
        current_sha = sha256_text(
            canonical_json(
                {"model": endpoint.model, "family": endpoint.family, "base_url": endpoint.base_url}
            )
        )
        if current_sha != frozen.get("sha256"):
            raise RuntimeError("judge endpoint contract changed after study freeze")

    batched_schema_pilot_contract = (
        external_contract.get("batched_schema_order_pilot") or {}
    )
    batched_schema_pilot_verification = require_v1_5_batched_schema_order_pilot_pass(
        args.batched_schema_pilot_summary,
        args.batched_schema_pilot_attestation,
        study_freeze_sha256=freeze_sha,
        contract=batched_schema_pilot_contract,
    )
    if tuple(batched_schema_pilot_verification["smoke_unit"]) not in excluded_units:
        raise RuntimeError("batched schema pilot unit is not in the frozen canary set")

    cost_match_tolerance = float(external_contract["maximum_cost_matched_relative_deviation"])
    try:
        observed_cost_match = compare_observed_cost_matched_turns(
            turn_path_by_condition[treatment],
            turn_path_by_condition["pm_v2_cost_matched_fixed"],
            maximum_relative_deviation=cost_match_tolerance,
            expected_units=expected_units,
        )
    except KeyError as exc:
        raise RuntimeError(
            "full external evaluation lacks the learned/fixed cost-match turn pair"
        ) from exc
    if observed_cost_match["status"] != "PASS":
        raise RuntimeError(
            "observed PM versus fixed input-token deviation exceeds the frozen limit"
        )

    result = run_v1_5_external_batched_evaluation(
        evoemo_path=args.evoemo,
        study_freeze_path=args.freeze,
        turn_paths=args.turn_paths,
        conditions=conditions,
        treatment=treatment,
        turn_indices=turn_indices,
        endpoints=endpoints,
        out_dir=args.out_dir,
        run=bool(args.run),
        max_api_calls=args.max_api_calls,
        expected_units=expected_units,
        full_expected_units=full_expected_units,
        composite_spec=composite_spec_from_config(pm_v1_5_config),
        labeling=labeling_settings_from_config(pm_v1_5_config),
        required_conditions=conditions,
        generation_attestation_paths=args.generation_attestations,
        study_freeze_sha256=freeze_sha,
        pricing_usd_per_mtok=external_contract["judge_pricing_usd_per_mtok"],
        api_cost_planning=external_contract["api_cost_planning"],
        primary_bootstrap_cluster=str(external_contract["primary_bootstrap_cluster"]),
        sensitivity_bootstrap_cluster=str(external_contract["sensitivity_bootstrap_cluster"]),
        batched_contract=external_contract["batched_evaluation"],
        schema_pilot_summary_path=args.batched_schema_pilot_summary,
        schema_pilot_attestation_path=args.batched_schema_pilot_attestation,
        schema_pilot_verification=batched_schema_pilot_verification,
        excluded_unit_ids=sorted(
            sha256_text(canonical_json(unit))[:24] for unit in excluded_units
        ),
        full_expected_units_sha256=sha256_text(canonical_json(full_expected_units)),
        observed_cost_match_report=observed_cost_match,
        accept_cost_estimate_sha256=args.accept_cost_estimate_sha256,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        overwrite=args.overwrite,
    )
    result = {**result, "latency_diagnostic": latency_diagnostic}
    if args.run:
        claim_assessment = assess_external_claims(
            result, external_contract["claim_assessment"]
        )
        claim_assessment["latency_diagnostic"] = {
            "status": latency_diagnostic["status"],
            "role": latency_diagnostic["role"],
            "confirmatory_latency_claim_allowed": False,
            "path": str(latency_path),
        }
        claim_path = args.out_dir / "claim_assessment.json"
        write_json(claim_path, claim_assessment)
        create_artifact_attestation(
            args.out_dir / "claim_assessment_attestation.json",
            stage="pm_v1_5_external_claim_assessment",
            inputs={
                "study_freeze": args.freeze,
                "external_summary": args.out_dir / "summary.json",
                "external_attestation": args.out_dir / "artifact_attestation.json",
                "latency_diagnostic": latency_path,
                "latency_diagnostic_attestation": latency_attestation_path,
            },
            outputs={"claim_assessment": (claim_path, False)},
            parameters={"contract": external_contract["claim_assessment"]},
            expected={"status": claim_assessment["status"]},
            study_freeze_sha256=freeze_sha,
        )
        result = {**result, "claim_assessment": claim_assessment}
    print(result)


if __name__ == "__main__":
    main()
