#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from metacom_pm.artifacts import (
    require_artifact_attestation,
    require_content_addressed_attestation,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.freeze import require_study_freeze
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text
from metacom_pm.pm_v2_external_eval import (
    expected_external_units,
    run_external_response_evaluation,
)
from metacom_pm.pm_v2_evoemo import compare_observed_cost_matched_turns
from metacom_pm.pm_v2_external_schema_smoke import (
    require_external_pointwise_schema_smoke_pass,
)
from metacom_pm.pm_v2_judging import (
    composite_spec_from_config,
    labeling_settings_from_config,
)
from metacom_pm.pm_v2_forced_swap import require_forced_swap_key_claim
from metacom_pm.pm_v22_reference_baselines import (
    PMV22_REFERENCE_BASELINE_STAGE,
    POLICY_LOCK_TIMING,
    POST_GENERATION_POLICY_TUNING_PROHIBITED,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RAW_GATE_BOOLEAN_CHECKS = (
    "row_count_exact",
    "call_keys_exact",
    "finish_reasons_complete",
    "errors_absent",
    "supporter_generation_treatment_exact",
    "fixed_seeker_generation_treatment_exact",
    "evidence_processing_contract_exact",
    "policy_lock_exact",
)


def require_reference_raw_generation_contract_gate(
    *, summary: Mapping[str, Any], attestation: Mapping[str, Any], row_count: int
) -> dict[str, Any]:
    """Require the exact named raw gate before external judging."""

    gate = dict(summary.get("raw_generation_contract_gate") or {})
    if (
        gate.get("status") != "PASS"
        or int(gate.get("expected_rows", -1)) != int(row_count)
        or int(gate.get("observed_rows", -1)) != int(row_count)
        or not all(gate.get(key) is True for key in REFERENCE_RAW_GATE_BOOLEAN_CHECKS)
        or (attestation.get("parameters") or {}).get(
            "raw_generation_contract_gate"
        )
        != gate
        or (attestation.get("expected") or {}).get(
            "raw_generation_contract_gate"
        )
        != gate
    ):
        raise RuntimeError(
            "reference-baseline summary/attestation lacks the exact PASS raw "
            "generation contract gate"
        )
    return gate


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
        condition_rows = [
            row for row in rows if str(row.get("condition") or "") == condition
        ]
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
            or row.get("fixed_seeker_generation_treatment")
            != dict(fixed_seeker_treatment)
            or row.get("fixed_seeker_generation_treatment_sha256")
            != fixed_seeker_treatment_sha256
            or row.get("normalized_finish_reason") != "complete"
            or not str(row.get("supporter_message") or "").strip()
            for row in condition_rows
        ):
            raise RuntimeError(
                f"generation turns for {condition} contain mixed or non-complete output"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml")
    parser.add_argument("--freeze", type=Path, default=ROOT / "outputs" / "pm_v2_study_freeze.json")
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
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
        "--turn-paths",
        type=Path,
        nargs="+",
        default=[
            ROOT
            / "outputs"
            / "evoemo_pmv22_reference_baselines"
            / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v2" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v2_cost_matched_fixed" / "turns.jsonl",
            ROOT / "outputs" / "evoemo_pm_v2_me_r0_fixed" / "turns.jsonl",
        ],
    )
    parser.add_argument(
        "--generation-attestations",
        type=Path,
        nargs="+",
        default=[
            ROOT
            / "outputs"
            / "evoemo_pmv22_reference_baselines"
            / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v2" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v2_cost_matched_fixed" / "artifact_attestation.json",
            ROOT / "outputs" / "evoemo_pm_v2_me_r0_fixed" / "artifact_attestation.json",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "pm_v2_external_response")
    parser.add_argument("--max-api-calls", type=int, default=6000)
    parser.add_argument("--max-estimated-usd", type=float, default=20.0)
    parser.add_argument("--max-input-tokens-per-call", type=int, default=12000)
    parser.add_argument("--accept-cost-estimate-sha256")
    parser.add_argument(
        "--key-claim-verification",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_forced_swap" / "summary.json",
    )
    parser.add_argument(
        "--key-claim-verification-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_forced_swap"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--pointwise-schema-smoke-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_external_pointwise_schema_smoke"
        / "summary.json",
    )
    parser.add_argument(
        "--pointwise-schema-smoke-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_external_pointwise_schema_smoke"
        / "artifact_attestation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
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
    verification = require_study_freeze(
        args.freeze,
        release_root=ROOT,
        config_path=args.config,
        required_files=[
            args.pm_v2_config,
            args.evoemo,
            args.policy_checkpoint,
            args.policy_training_report,
        ],
    )
    freeze_data = json.loads(args.freeze.read_text(encoding="utf-8"))
    freeze_sha = str(verification["freeze_sha256"])
    external_contract = (freeze_data.get("notes") or {}).get(
        "external_evaluation_contract"
    ) or {}
    generation_contract = (freeze_data.get("notes") or {}).get("generation_contract") or {}
    frozen_generation_treatment = generation_contract.get(
        "supporter_generation_treatment"
    )
    frozen_generation_treatment_sha256 = generation_contract.get(
        "supporter_generation_treatment_sha256"
    )
    frozen_fixed_seeker_treatment = generation_contract.get(
        "fixed_seeker_generation_treatment"
    )
    frozen_fixed_seeker_treatment_sha256 = generation_contract.get(
        "fixed_seeker_generation_treatment_sha256"
    )
    frozen_policy_lock = {
        "policy_checkpoint_sha256": sha256_file(args.policy_checkpoint),
        "policy_training_report_sha256": sha256_file(
            args.policy_training_report
        ),
        "policy_lock_timing": POLICY_LOCK_TIMING,
        "post_generation_policy_tuning_prohibited": (
            POST_GENERATION_POLICY_TUNING_PROHIBITED
        ),
    }
    if any(
        generation_contract.get(key) != expected
        for key, expected in frozen_policy_lock.items()
    ):
        raise RuntimeError(
            "study freeze does not bind the policy that preceded reference "
            "baseline generation"
        )
    if (
        frozen_generation_treatment != supporter_generation_contract.payload()
        or frozen_generation_treatment_sha256
        != supporter_generation_contract.digest()
    ):
        raise RuntimeError("study freeze supporter-generation treatment is stale")
    if (
        frozen_fixed_seeker_treatment != bound_fixed_seeker_contract.payload()
        or frozen_fixed_seeker_treatment_sha256
        != bound_fixed_seeker_contract.digest()
    ):
        raise RuntimeError("study freeze fixed-seeker generation treatment is stale")
    fixed_tracks_contract = (freeze_data.get("notes") or {}).get(
        "fixed_tracks_content_attestation"
    ) or {}
    if (
        fixed_tracks_contract.get("stage")
        != "evoemo_fixed_seeker_tracks_v22"
        or fixed_tracks_contract.get("fixed_seeker_generation_treatment")
        != frozen_fixed_seeker_treatment
        or fixed_tracks_contract.get(
            "fixed_seeker_generation_treatment_sha256"
        )
        != frozen_fixed_seeker_treatment_sha256
    ):
        raise RuntimeError("study freeze fixed-track content contract is stale")
    expected_external_estimand = {
        "external_estimand": "quality_risk_observed_cost_componentwise_pareto_v1",
        "risk_composite_basis": "requested_action_applicable_fields",
        "single_external_utility_claim_allowed": False,
    }
    for key, expected in expected_external_estimand.items():
        if external_contract.get(key) != expected:
            raise RuntimeError(f"study freeze external estimand mismatch: {key}")
    fixed_baseline_artifacts = external_contract.get("fixed_baseline_artifacts") or {}
    if not bool(external_contract.get("require_raw_per_family_health")):
        raise RuntimeError("study freeze does not require raw per-family judge health")
    frozen_conditions = [str(value) for value in external_contract.get("conditions") or []]
    if not frozen_conditions:
        raise RuntimeError("study freeze lacks an external condition matrix")
    conditions = frozen_conditions
    treatment = str(external_contract.get("treatment") or "")
    judge_seed = int(external_contract.get("judge_seed"))
    labeling = labeling_settings_from_config(pm_v2_config)
    for key in (
        "composite_support_exact_match_rate",
        "maximum_absolute_composite_support_correlation",
    ):
        if float(labeling[key]) != float(external_contract[key]):
            raise RuntimeError(f"external composite/support gate changed after freeze: {key}")
    turn_indices = [
        int(value) for value in external_contract.get("turn_indices") or []
    ]
    evaluation_unit_contract = dict(
        external_contract.get("evaluation_unit_contract") or {}
    )
    if evaluation_unit_contract != generation_contract.get(
        "evaluation_unit_contract"
    ):
        raise RuntimeError("generation/external evaluation-unit contracts differ")
    full_expected_units = expected_external_units(
        args.evoemo,
        seeds=[int(value) for value in generation_contract["seeds"]],
        simulator_id=str(generation_contract["simulator_id"]),
        turn_indices=turn_indices,
    )
    if (
        evaluation_unit_contract.get("evaluation_turn_indices") != turn_indices
        or evaluation_unit_contract.get("expected_unit_count")
        != len(full_expected_units)
        or evaluation_unit_contract.get("expected_units_sha256")
        != sha256_text(canonical_json(full_expected_units))
    ):
        raise RuntimeError("frozen external evaluation-unit universe is stale")
    if len(args.turn_paths) != len(args.generation_attestations):
        raise ValueError("each turn path requires one generation attestation")
    attested_conditions = set()
    fixed_conditions_verified: set[str] = set()
    turn_path_by_condition: dict[str, Path] = {}
    for turn_path, attestation_path in zip(args.turn_paths, args.generation_attestations):
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        stage = str(attestation.get("stage") or "")
        if stage == PMV22_REFERENCE_BASELINE_STAGE:
            baseline_dir = turn_path.resolve().parent
            result = require_content_addressed_attestation(
                attestation_path,
                required_stage=PMV22_REFERENCE_BASELINE_STAGE,
                relocated_inputs={
                    "evoemo": args.evoemo,
                    "strategy_bank": ROOT
                    / "data"
                    / "strategy"
                    / "strategy_cards.jsonl",
                    "policy_checkpoint": args.policy_checkpoint,
                    "policy_training_report": args.policy_training_report,
                    "pm_v2_config": args.pm_v2_config,
                    "evidence_filter_checkpoint": ROOT
                    / "outputs"
                    / "pm_v2_evidence_filter"
                    / "evidence_filter.joblib",
                    "evidence_filter_report": ROOT
                    / "outputs"
                    / "pm_v2_evidence_filter"
                    / "training_report.json",
                    "evidence_filter_attestation": ROOT
                    / "outputs"
                    / "pm_v2_evidence_filter"
                    / "artifact_attestation.json",
                    "strategy_bank_approval": ROOT
                    / "outputs"
                    / "strategy_rag_v1_frozen_candidate"
                    / "human_approval.json",
                    "run_manifest": baseline_dir / "run_manifest.json",
                    "cost_estimate": baseline_dir / "cost_estimate.json",
                    "call_plan": baseline_dir / "call_plan.jsonl",
                    "fixed_tracks": ROOT
                    / "outputs"
                    / "evoemo_fixed_tracks_v22"
                    / "fixed_seeker_tracks.jsonl",
                    "fixed_tracks_attestation": ROOT
                    / "outputs"
                    / "evoemo_fixed_tracks_v22"
                    / "artifact_attestation.json",
                },
                relocated_outputs={
                    "turns": turn_path,
                    "raw_calls": baseline_dir / "raw_api_calls.jsonl",
                    "physical_attempt_ledger": baseline_dir
                    / "physical_attempt_ledger.jsonl",
                    "summary": baseline_dir / "generation_summary.json",
                },
            )
        elif stage == "evoemo_pm_v2_generation":
            result = require_artifact_attestation(
                attestation_path,
                required_stage="evoemo_pm_v2_generation",
                required_output_paths={"turns": turn_path},
            )
        else:
            raise RuntimeError(
                "external evaluation accepts only PM-v2.2 generation artifacts; "
                f"legacy or unknown stage is forbidden: {stage!r}"
            )
        parameters = attestation.get("parameters") or {}
        values = [
            str(value)
            for value in (
                parameters.get("conditions") or [parameters.get("condition")]
            )
            if value
        ]
        if not values:
            raise RuntimeError("generation attestation has no conditions")
        attested_conditions.update(str(value) for value in values if value)
        if len(values) == 1 and values[0]:
            condition = str(values[0])
            if condition in turn_path_by_condition:
                raise RuntimeError(f"duplicate generation turn input for {condition}")
            turn_path_by_condition[condition] = turn_path
        if stage == "evoemo_pm_v2_generation" and result.get(
            "study_freeze_sha256"
        ) != freeze_sha:
            raise RuntimeError("PM-v2 generation attestation uses a different study freeze")
        frozen_parameter_checks = [
            (
                "supporter_generation_treatment",
                frozen_generation_treatment,
            ),
            (
                "supporter_generation_treatment_sha256",
                frozen_generation_treatment_sha256,
            ),
            (
                "fixed_seeker_generation_treatment",
                frozen_fixed_seeker_treatment,
            ),
            (
                "fixed_seeker_generation_treatment_sha256",
                frozen_fixed_seeker_treatment_sha256,
            ),
            ("simulator_id", generation_contract.get("simulator_id")),
            ("max_turns", generation_contract.get("max_turns")),
            ("seeds", generation_contract.get("seeds")),
            (
                "evaluation_unit_contract",
                generation_contract.get("evaluation_unit_contract"),
            ),
            (
                "paid_generation_scope",
                generation_contract.get("paid_generation_scope"),
            ),
        ]
        if stage == "evoemo_pm_v2_generation":
            frozen_parameter_checks.extend(
                [
                    (
                        "maximum_cost_matched_relative_deviation",
                        generation_contract.get(
                            "maximum_cost_matched_relative_deviation"
                        ),
                    ),
                    (
                        "generator_retries",
                        generation_contract.get("generator_retries"),
                    ),
                    (
                        "generator_pricing_usd_per_mtok",
                        generation_contract.get(
                            "generator_pricing_usd_per_mtok"
                        ),
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
            )
        for key, expected in frozen_parameter_checks:
            if parameters.get(key) != expected:
                raise RuntimeError(
                    f"generation attestation {attestation_path} violates frozen {key}"
                )
        turn_rows = require_treatment_bound_turn_file(
            turn_path,
            expected_conditions=values,
            expected_units=full_expected_units,
            treatment=frozen_generation_treatment,
            treatment_sha256=str(frozen_generation_treatment_sha256),
            fixed_seeker_treatment=frozen_fixed_seeker_treatment,
            fixed_seeker_treatment_sha256=str(
                frozen_fixed_seeker_treatment_sha256
            ),
        )
        baseline_evidence_contracts: dict[str, dict[str, Any]] | None = None
        baseline_evidence_contracts_sha256: str | None = None
        if stage == PMV22_REFERENCE_BASELINE_STAGE:
            baseline_evidence_contracts = {
                str(key): dict(value)
                for key, value in dict(
                    parameters.get("evidence_processing_contracts") or {}
                ).items()
            }
            baseline_evidence_contracts_sha256 = str(
                parameters.get("evidence_processing_contracts_sha256") or ""
            )
            if (
                set(baseline_evidence_contracts) != set(values)
                or baseline_evidence_contracts_sha256
                != sha256_text(canonical_json(baseline_evidence_contracts))
                or any(
                    parameters.get(key) != expected
                    for key, expected in frozen_policy_lock.items()
                )
                or any(
                    row.get("evidence_processing_contract")
                    != baseline_evidence_contracts[str(row["condition"])]
                    or row.get("evidence_processing_contract_sha256")
                    != sha256_text(
                        canonical_json(
                            baseline_evidence_contracts[str(row["condition"])]
                        )
                    )
                    or any(
                        row.get(key) != expected
                        for key, expected in frozen_policy_lock.items()
                    )
                    for row in turn_rows
                )
            ):
                raise RuntimeError(
                    "reference-baseline evidence-processing contract is stale"
                )
        generation_dir = turn_path.resolve().parent
        manifest_path = generation_dir / "run_manifest.json"
        summary_path = generation_dir / "generation_summary.json"
        raw_path = generation_dir / "raw_api_calls.jsonl"
        ledger_path = generation_dir / "physical_attempt_ledger.jsonl"
        call_plan_path = generation_dir / "call_plan.jsonl"
        cost_estimate_path = generation_dir / "cost_estimate.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cost_estimate = json.loads(cost_estimate_path.read_text(encoding="utf-8"))
        call_plan = [dict(row) for row in iter_jsonl(call_plan_path)]
        raw_rows = [dict(row) for row in iter_jsonl(raw_path)]
        ledger_rows = [dict(row) for row in iter_jsonl(ledger_path)]
        for source_name, source in (
            ("manifest", manifest),
            ("summary", summary),
            ("cost estimate", cost_estimate),
        ):
            if (
                source.get("supporter_generation_treatment")
                != frozen_generation_treatment
                or source.get("supporter_generation_treatment_sha256")
                != frozen_generation_treatment_sha256
                or source.get("fixed_seeker_generation_treatment")
                != frozen_fixed_seeker_treatment
                or source.get("fixed_seeker_generation_treatment_sha256")
                != frozen_fixed_seeker_treatment_sha256
            ):
                raise RuntimeError(
                    f"{source_name} has a mixed supporter-generation treatment"
                )
            if stage == PMV22_REFERENCE_BASELINE_STAGE and (
                source.get("evidence_processing_contracts")
                != baseline_evidence_contracts
                or source.get("evidence_processing_contracts_sha256")
                != baseline_evidence_contracts_sha256
                or any(
                    source.get(key) != expected
                    for key, expected in frozen_policy_lock.items()
                )
            ):
                raise RuntimeError(
                    f"{source_name} has a stale evidence-processing contract"
                )
        if (
            summary.get("status") != "COMPLETE"
            or int(summary.get("completed_turns", -1)) != len(turn_rows)
            or len(call_plan) != len(turn_rows)
            or len(raw_rows) != len(turn_rows)
            or len(
                [row for row in ledger_rows if row.get("event") == "SUCCEEDED"]
            )
            != len(turn_rows)
            or any(row.get("event") == "FAILED" for row in ledger_rows)
        ):
            raise RuntimeError("generation bundle is incomplete or has failed attempts")
        reference_raw_gate: dict[str, Any] | None = None
        if stage == PMV22_REFERENCE_BASELINE_STAGE:
            reference_raw_gate = require_reference_raw_generation_contract_gate(
                summary=summary,
                attestation=attestation,
                row_count=len(turn_rows),
            )
        if (
            stage == "evoemo_pm_v2_generation"
            and (
                (summary.get("raw_generation_contract_gate") or {}).get("status")
                != "PASS"
                or int(summary.get("non_complete_finish_reason_count", -1)) != 0
                or parameters.get("raw_generation_contract_gate")
                != summary.get("raw_generation_contract_gate")
            )
        ) or (
            stage == PMV22_REFERENCE_BASELINE_STAGE
            and (
                int(summary.get("non_complete_finish_reason_count", -1)) != 0
                or reference_raw_gate is None
            )
        ):
            raise RuntimeError(
                "generation summary does not prove a complete raw-call matrix"
            )
        if any(
            row.get("supporter_generation_treatment")
            != frozen_generation_treatment
            or row.get("supporter_generation_treatment_sha256")
            != frozen_generation_treatment_sha256
            or row.get("fixed_seeker_generation_treatment")
            != frozen_fixed_seeker_treatment
            or row.get("fixed_seeker_generation_treatment_sha256")
            != frozen_fixed_seeker_treatment_sha256
            or int(row.get("max_output_tokens") or 0)
            != int(frozen_generation_treatment["max_output_tokens"])
            or (
                stage == PMV22_REFERENCE_BASELINE_STAGE
                and (
                    row.get("evidence_processing_contract")
                    != baseline_evidence_contracts.get(str(row.get("condition") or ""))
                    or row.get("evidence_processing_contract_sha256")
                    != sha256_text(
                        canonical_json(
                            baseline_evidence_contracts.get(
                                str(row.get("condition") or ""), {}
                            )
                        )
                    )
                    or any(
                        row.get(key) != expected
                        for key, expected in frozen_policy_lock.items()
                    )
                )
            )
            for row in call_plan
        ):
            raise RuntimeError("generation call plan has a mixed treatment")
        planned_call_keys = {str(row.get("call_key") or "") for row in call_plan}
        raw_call_keys = {
            str(row.get("physical_call_key") or "") for row in raw_rows
        }
        if (
            raw_call_keys != planned_call_keys
            or any(
                row.get("normalized_finish_reason") != "complete"
                or row.get("supporter_generation_treatment")
                != frozen_generation_treatment
                or row.get("supporter_generation_treatment_sha256")
                != frozen_generation_treatment_sha256
                or row.get("fixed_seeker_generation_treatment")
                != frozen_fixed_seeker_treatment
                or row.get("fixed_seeker_generation_treatment_sha256")
                != frozen_fixed_seeker_treatment_sha256
                or (
                    stage == PMV22_REFERENCE_BASELINE_STAGE
                    and (
                        row.get("evidence_processing_contract")
                        != baseline_evidence_contracts.get(
                            str(row.get("condition") or "")
                        )
                        or row.get("evidence_processing_contract_sha256")
                        != sha256_text(
                            canonical_json(
                                baseline_evidence_contracts.get(
                                    str(row.get("condition") or ""), {}
                                )
                            )
                        )
                        or any(
                            row.get(key) != expected
                            for key, expected in frozen_policy_lock.items()
                        )
                    )
                )
                for row in raw_rows
            )
        ):
            raise RuntimeError("generation raw calls contain a non-complete response")
        if stage == PMV22_REFERENCE_BASELINE_STAGE:
            manifest_payload = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
            if manifest.get("manifest_sha256") != sha256_text(
                canonical_json(manifest_payload)
            ):
                raise RuntimeError("reference-baseline manifest self-hash mismatch")
            endpoint_contract = {
                "model": str(manifest.get("generator_model") or ""),
                "family": str(manifest.get("generator_family") or ""),
                "base_url": str(manifest.get("generator_base_url") or ""),
            }
            for condition in (str(value) for value in values if value):
                frozen_artifact = fixed_baseline_artifacts.get(condition)
                if frozen_artifact is None:
                    continue
                exact_checks = {
                    "stage": stage,
                    "turns_sha256": sha256_file(turn_path),
                    "attestation_sha256": result["attestation_sha256"],
                    "attestation_file_sha256": sha256_file(attestation_path),
                    "run_manifest_sha256": sha256_file(manifest_path),
                    "run_manifest_record_sha256": manifest["manifest_sha256"],
                    "turns_rows": len(turn_rows),
                    "generator_endpoint": endpoint_contract,
                    "generator_endpoint_sha256": sha256_text(
                        canonical_json(endpoint_contract)
                    ),
                    "supporter_generation_treatment": (
                        frozen_generation_treatment
                    ),
                    "supporter_generation_treatment_sha256": (
                        frozen_generation_treatment_sha256
                    ),
                    "fixed_seeker_generation_treatment": (
                        frozen_fixed_seeker_treatment
                    ),
                    "fixed_seeker_generation_treatment_sha256": (
                        frozen_fixed_seeker_treatment_sha256
                    ),
                    "simulator_id": parameters.get("simulator_id"),
                    "max_turns": parameters.get("max_turns"),
                    "seeds": parameters.get("seeds"),
                    "evaluation_unit_contract": parameters.get(
                        "evaluation_unit_contract"
                    ),
                    "paid_generation_scope": parameters.get(
                        "paid_generation_scope"
                    ),
                    "conditions": sorted(values),
                    "evidence_processing_contracts": (
                        baseline_evidence_contracts
                    ),
                    "evidence_processing_contracts_sha256": (
                        baseline_evidence_contracts_sha256
                    ),
                    "evidence_filter_config_sha256": parameters.get(
                        "evidence_filter_config_sha256"
                    ),
                    "evidence_filter_model": parameters.get(
                        "evidence_filter_model"
                    ),
                    "strategy_bank_approval": parameters.get(
                        "strategy_bank_approval"
                    ),
                    **frozen_policy_lock,
                }
                for key, actual in exact_checks.items():
                    if frozen_artifact.get(key) != actual:
                        raise RuntimeError(
                            f"fixed external baseline artifact mismatch: "
                            f"{condition}/{key}"
                        )
                fixed_conditions_verified.add(condition)
    if set(conditions) != attested_conditions:
        raise RuntimeError(
            "generation attestations do not exactly cover the frozen conditions"
        )
    if set(fixed_baseline_artifacts) != fixed_conditions_verified:
        raise RuntimeError(
            "fixed external baselines are not all backed by their frozen PM-v2.2 "
            "turn/attestation artifacts"
        )
    frozen_judges = external_contract.get("judge_endpoints") or []
    frozen_names = [str(row["name"]) for row in frozen_judges]
    endpoints = [endpoint_from_config(config, name) for name in frozen_names]
    for endpoint, frozen in zip(endpoints, frozen_judges):
        current_sha = sha256_text(
            canonical_json(
                {
                    "model": endpoint.model,
                    "family": endpoint.family,
                    "base_url": endpoint.base_url,
                }
            )
        )
        if current_sha != frozen.get("sha256"):
            raise RuntimeError("judge endpoint contract changed after study freeze")
    forced_contract = external_contract.get("forced_swap") or {}
    if not bool(forced_contract.get("exclude_sample_from_full_evaluation")):
        raise RuntimeError("study freeze does not require pilot/full sample separation")
    forced_verification = require_forced_swap_key_claim(
        args.key_claim_verification,
        args.key_claim_verification_attestation,
        study_freeze_sha256=freeze_sha,
        full_expected_units=full_expected_units,
        forced_contract=forced_contract,
    )
    pointwise_schema_smoke_contract = external_contract.get(
        "pointwise_schema_smoke"
    ) or {}
    pointwise_schema_smoke_verification = (
        require_external_pointwise_schema_smoke_pass(
            args.pointwise_schema_smoke_summary,
            args.pointwise_schema_smoke_attestation,
            study_freeze_sha256=freeze_sha,
            contract=pointwise_schema_smoke_contract,
            forced_swap_selected_units_sha256=sha256_text(
                canonical_json(sorted(forced_verification["excluded_units"]))
            ),
        )
    )
    excluded_units = set(forced_verification["excluded_units"])
    expected_units = [
        unit for unit in full_expected_units if unit not in excluded_units
    ]
    if len(expected_units) + len(excluded_units) != len(full_expected_units):
        raise RuntimeError("pilot/full external unit partition is not exact")
    cost_match_tolerance = float(
        external_contract["maximum_cost_matched_relative_deviation"]
    )
    try:
        observed_cost_match = compare_observed_cost_matched_turns(
            turn_path_by_condition["pm_v2"],
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
            "observed PM-v2 versus fixed input-token deviation exceeds the frozen limit"
        )
    result = run_external_response_evaluation(
        evoemo_path=args.evoemo,
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
        composite_spec=composite_spec_from_config(pm_v2_config),
        labeling=labeling,
        minimum_low_mad_coverage_per_dimension=float(
            external_contract["minimum_low_mad_coverage_per_dimension"]
        ),
        minimum_low_mad_coverage_per_condition_dimension=float(
            external_contract[
                "minimum_low_mad_coverage_per_condition_dimension"
            ]
        ),
        required_conditions=frozen_conditions,
        generation_attestation_paths=args.generation_attestations,
        study_freeze_sha256=freeze_sha,
        require_key_claim_verification=bool(
            external_contract.get("require_forced_swap_or_human_check_for_key_claims")
        ),
        key_claim_verification_path=args.key_claim_verification,
        key_claim_verification_attestation_path=(
            args.key_claim_verification_attestation
        ),
        excluded_unit_ids=forced_verification["excluded_unit_ids"],
        pilot_selection_contract_sha256=forced_verification[
            "sample_selection_contract_sha256"
        ],
        full_expected_units_sha256=sha256_text(canonical_json(full_expected_units)),
        observed_cost_match_report=observed_cost_match,
        accept_cost_estimate_sha256=args.accept_cost_estimate_sha256,
        max_estimated_usd=args.max_estimated_usd,
        max_input_tokens_per_call=args.max_input_tokens_per_call,
        pricing_usd_per_mtok=external_contract["judge_pricing_usd_per_mtok"],
        api_cost_planning=external_contract["api_cost_planning"],
        primary_bootstrap_cluster=str(
            external_contract["primary_bootstrap_cluster"]
        ),
        sensitivity_bootstrap_cluster=str(
            external_contract["sensitivity_bootstrap_cluster"]
        ),
        pointwise_schema_smoke_summary_path=args.pointwise_schema_smoke_summary,
        pointwise_schema_smoke_attestation_path=(
            args.pointwise_schema_smoke_attestation
        ),
        pointwise_schema_smoke_verification=pointwise_schema_smoke_verification,
        estimated_response_output_tokens=int(
            pointwise_schema_smoke_contract[
                "estimated_response_output_tokens"
            ]
        ),
        estimated_risk_output_tokens=int(
            pointwise_schema_smoke_contract["estimated_risk_output_tokens"]
        ),
        overwrite=args.overwrite,
        seed=judge_seed,
    )
    print(result)


if __name__ == "__main__":
    main()
