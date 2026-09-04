#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from metacom_pm.artifacts import (
    require_artifact_attestation,
    require_content_addressed_attestation,
)
from metacom_pm.api import require_reported_usage
from metacom_pm.attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    reported_prompt_token_error,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import StrategyCard
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.evidence_filter_model import require_evidence_filter_artifacts
from metacom_pm.freeze import create_study_freeze
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_contracts import ActionLabel
from metacom_pm.pm_v2_development_gate import require_development_pilot_gate
from metacom_pm.pm_v2_evoemo import (
    ACTION_PREFLIGHT_METRIC_SCOPES,
    EVALUATION_UNIT_CONTRACT_PROTOCOL,
)
from metacom_pm.evoemo import (
    FIXED_SEEKER_V22_STAGE,
    fixed_seeker_cost_planning_contract,
)
from metacom_pm.pm_v2_external_eval import expected_external_units
from metacom_pm.pm_v2_generation_pilot import (
    build_generation_compatibility_contract,
    require_generation_compatibility_attestation,
)
from metacom_pm.pm_v2_generation_review_v8 import (
    require_generation_semantic_review_v8,
)
from metacom_pm.pm_v2_forced_swap import (
    forced_swap_unit_id,
    select_forced_swap_units,
)
from metacom_pm.pm_v2_judging import (
    JUDGE_RUBRIC_VERSION,
    composite_spec_from_config,
    composite_weights_hash,
    labeling_settings_from_config,
    prompt_contract_hash,
)
from metacom_pm.pm_v2_model import PMV2Model
from metacom_pm.pm_v22_reference_baselines import (
    PMV22_REFERENCE_BASELINE_STAGE,
    POLICY_LOCK_TIMING,
    POST_GENERATION_POLICY_TUNING_PROHIBITED,
    REFERENCE_BASELINE_CONDITIONS,
    build_reference_evidence_processing_contracts,
    evidence_processing_contracts_sha256,
)
from metacom_pm.pm_v2_semantic_audit import (
    require_pmv2_runtime_state_lineage,
    require_semantic_sanity_pass,
)
from metacom_pm.strategy_bank_approval import require_strategy_bank_human_approval

ROOT = Path(__file__).resolve().parents[1]


def require_complete_only_finish_reason_counts(
    counts: Mapping[str, Any], *, expected_rows: int
) -> dict[str, int]:
    """Accept initialized zero buckets while requiring every row complete."""

    normalized = {
        str(key): int(value) for key, value in dict(counts).items()
    }
    if (
        int(expected_rows) < 0
        or normalized.get("complete", -1) != int(expected_rows)
        or any(
            count != 0
            for reason, count in normalized.items()
            if reason != "complete"
        )
    ):
        raise RuntimeError(
            "fixed seeker finish-reason counts contain incomplete calls"
        )
    return normalized


def require_exact_treatment_turn_rows(
    turn_path: Path,
    *,
    conditions: Sequence[str],
    expected_units: Sequence[tuple[str, int, int, str, int]],
    treatment: Mapping[str, Any],
    treatment_sha256: str,
    fixed_seeker_treatment: Mapping[str, Any],
    fixed_seeker_treatment_sha256: str,
    evidence_processing_contracts: Mapping[str, Mapping[str, Any]],
    policy_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Require one complete, treatment-matched sparse matrix per condition."""

    rows = [dict(row) for row in iter_jsonl(turn_path)]
    expected_conditions = {str(value) for value in conditions}
    if not rows or {str(row.get("condition") or "") for row in rows} != expected_conditions:
        raise RuntimeError("reference baseline turns do not cover the exact conditions")
    expected_unit_rows = sorted(expected_units)
    for condition in sorted(expected_conditions):
        condition_rows = [
            row for row in rows if str(row.get("condition") or "") == condition
        ]
        observed_units = sorted(
            (
                str(row.get("user_id") or ""),
                int(row.get("topic_index") or 0),
                int(row.get("seed") or 0),
                str(row.get("simulator_id") or ""),
                int(row.get("turn_index") or 0),
            )
            for row in condition_rows
        )
        if observed_units != expected_unit_rows:
            raise RuntimeError(
                f"reference baseline {condition} does not equal the sparse unit universe"
            )
        for row in condition_rows:
            expected_evidence_contract = dict(
                evidence_processing_contracts[condition]
            )
            if (
                row.get("supporter_generation_treatment") != dict(treatment)
                or row.get("supporter_generation_treatment_sha256")
                != treatment_sha256
                or row.get("fixed_seeker_generation_treatment")
                != dict(fixed_seeker_treatment)
                or row.get("fixed_seeker_generation_treatment_sha256")
                != fixed_seeker_treatment_sha256
                or row.get("evidence_processing_contract")
                != expected_evidence_contract
                or row.get("evidence_processing_contract_sha256")
                != sha256_text(canonical_json(expected_evidence_contract))
                or any(
                    row.get(key) != expected
                    for key, expected in policy_lock.items()
                )
                or row.get("normalized_finish_reason") != "complete"
                or str(row.get("interaction_mode") or "") != "fixed"
                or not str(row.get("supporter_message") or "").strip()
            ):
                raise RuntimeError(
                    f"reference baseline {condition} contains a mixed, truncated, "
                    "or invalid turn"
                )
    return rows


def require_pmv22_reference_baseline_bundle(
    *,
    attestation_path: Path,
    turns_path: Path,
    manifest_path: Path,
    summary_path: Path,
    evoemo_path: Path,
    strategy_bank_path: Path,
    policy_checkpoint_path: Path,
    policy_training_report_path: Path,
    pm_v2_config_path: Path,
    evidence_filter_checkpoint_path: Path,
    evidence_filter_report_path: Path,
    evidence_filter_attestation_path: Path,
    strategy_bank_approval_path: Path,
    fixed_tracks_path: Path,
    fixed_tracks_attestation_path: Path,
    conditions: Sequence[str],
    expected_units: Sequence[tuple[str, int, int, str, int]],
    treatment: Mapping[str, Any],
    treatment_sha256: str,
    fixed_seeker_treatment: Mapping[str, Any],
    fixed_seeker_treatment_sha256: str,
    evidence_processing_contracts: Mapping[str, Mapping[str, Any]],
    evidence_processing_contracts_sha256: str,
    evidence_filter_config_sha256: str,
    evidence_filter_model_binding: Mapping[str, Any],
    strategy_bank_approval: Mapping[str, Any],
    generator_endpoint: Mapping[str, Any],
    simulator_id: str,
    max_turns: int,
    seeds: Sequence[int],
    evaluation_unit_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the pre-freeze PM-v2.2 reference-baseline bundle contract."""

    bundle_dir = turns_path.resolve().parent
    raw_path = bundle_dir / "raw_api_calls.jsonl"
    ledger_path = bundle_dir / "physical_attempt_ledger.jsonl"
    call_plan_path = bundle_dir / "call_plan.jsonl"
    cost_estimate_path = bundle_dir / "cost_estimate.json"
    verification = require_content_addressed_attestation(
        attestation_path,
        required_stage=PMV22_REFERENCE_BASELINE_STAGE,
        relocated_inputs={
            "evoemo": evoemo_path,
            "strategy_bank": strategy_bank_path,
            "policy_checkpoint": policy_checkpoint_path,
            "policy_training_report": policy_training_report_path,
            "pm_v2_config": pm_v2_config_path,
            "evidence_filter_checkpoint": evidence_filter_checkpoint_path,
            "evidence_filter_report": evidence_filter_report_path,
            "evidence_filter_attestation": evidence_filter_attestation_path,
            "strategy_bank_approval": strategy_bank_approval_path,
            "fixed_tracks": fixed_tracks_path,
            "fixed_tracks_attestation": fixed_tracks_attestation_path,
            "run_manifest": manifest_path,
            "cost_estimate": cost_estimate_path,
            "call_plan": call_plan_path,
        },
        relocated_outputs={
            "turns": turns_path,
            "raw_calls": raw_path,
            "physical_attempt_ledger": ledger_path,
            "summary": summary_path,
        },
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cost_estimate = json.loads(cost_estimate_path.read_text(encoding="utf-8"))
    if attestation.get("stage") != PMV22_REFERENCE_BASELINE_STAGE:
        raise RuntimeError(
            "legacy/V1 baseline bundles cannot satisfy the PM-v2.2 freeze gate"
        )
    manifest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != sha256_text(
        canonical_json(manifest_payload)
    ):
        raise RuntimeError("PM-v2.2 reference-baseline manifest self-hash mismatch")
    expected_conditions = sorted(str(value) for value in conditions)
    expected_unit_rows = sorted(expected_units)
    normalized_evidence_contracts = {
        str(key): dict(value)
        for key, value in evidence_processing_contracts.items()
    }
    if (
        set(normalized_evidence_contracts) != set(expected_conditions)
        or evidence_processing_contracts_sha256
        != sha256_text(canonical_json(normalized_evidence_contracts))
    ):
        raise RuntimeError("reference-baseline evidence-processing contract is invalid")
    policy_lock = {
        "policy_checkpoint_sha256": sha256_file(policy_checkpoint_path),
        "policy_training_report_sha256": sha256_file(
            policy_training_report_path
        ),
        "policy_lock_timing": POLICY_LOCK_TIMING,
        "post_generation_policy_tuning_prohibited": (
            POST_GENERATION_POLICY_TUNING_PROHIBITED
        ),
    }
    common_contract = {
        "conditions": expected_conditions,
        "supporter_generation_treatment": dict(treatment),
        "supporter_generation_treatment_sha256": treatment_sha256,
        "fixed_seeker_generation_treatment": dict(fixed_seeker_treatment),
        "fixed_seeker_generation_treatment_sha256": (
            fixed_seeker_treatment_sha256
        ),
        "evidence_processing_contracts": normalized_evidence_contracts,
        "evidence_processing_contracts_sha256": (
            evidence_processing_contracts_sha256
        ),
        **policy_lock,
        "simulator_id": simulator_id,
        "max_turns": int(max_turns),
        "seeds": [int(value) for value in seeds],
        "evaluation_unit_contract": dict(evaluation_unit_contract),
        "paid_generation_scope": "frozen_evaluation_turns_only",
    }
    for source_name, source in (
        ("attestation", attestation.get("parameters") or {}),
        ("manifest", manifest),
        ("summary", summary),
    ):
        for key, expected in common_contract.items():
            if source.get(key) != expected:
                raise RuntimeError(
                    f"PM-v2.2 reference-baseline {source_name} mismatch: {key}"
                )
        for key, expected in {
            "evidence_filter_config_sha256": evidence_filter_config_sha256,
            "evidence_filter_model": dict(evidence_filter_model_binding),
            "strategy_bank_approval": dict(strategy_bank_approval),
        }.items():
            if source.get(key) != expected:
                raise RuntimeError(
                    f"PM-v2.2 reference-baseline {source_name} mismatch: {key}"
                )
    if (
        manifest.get("stage") != PMV22_REFERENCE_BASELINE_STAGE
        or summary.get("status") != "COMPLETE"
        or int(summary.get("non_complete_finish_reason_count", -1)) != 0
    ):
        raise RuntimeError("PM-v2.2 reference-baseline bundle is not complete")
    observed_endpoint = {
        "model": str(manifest.get("generator_model") or ""),
        "family": str(manifest.get("generator_family") or ""),
        "base_url": str(manifest.get("generator_base_url") or ""),
    }
    if observed_endpoint != dict(generator_endpoint):
        raise RuntimeError("PM-v2.2 reference-baseline generator endpoint mismatch")
    rows = require_exact_treatment_turn_rows(
        turns_path,
        conditions=expected_conditions,
        expected_units=expected_units,
        treatment=treatment,
        treatment_sha256=treatment_sha256,
        fixed_seeker_treatment=fixed_seeker_treatment,
        fixed_seeker_treatment_sha256=fixed_seeker_treatment_sha256,
        evidence_processing_contracts=normalized_evidence_contracts,
        policy_lock=policy_lock,
    )
    output_turn_record = (attestation.get("outputs") or {}).get("turns") or {}
    if (
        int(output_turn_record.get("rows", -1)) != len(rows)
        or int(summary.get("completed_turns", -1)) != len(rows)
    ):
        raise RuntimeError("PM-v2.2 reference-baseline turn counts disagree")
    raw_generation_contract_gate = dict(
        summary.get("raw_generation_contract_gate") or {}
    )
    raw_gate_boolean_keys = {
        "row_count_exact",
        "call_keys_exact",
        "finish_reasons_complete",
        "errors_absent",
        "supporter_generation_treatment_exact",
        "fixed_seeker_generation_treatment_exact",
        "evidence_processing_contract_exact",
        "policy_lock_exact",
    }
    if (
        set(raw_generation_contract_gate)
        != {
            "status",
            "expected_rows",
            "observed_rows",
            *raw_gate_boolean_keys,
        }
        or raw_generation_contract_gate.get("status") != "PASS"
        or int(raw_generation_contract_gate.get("expected_rows", -1))
        != len(rows)
        or int(raw_generation_contract_gate.get("observed_rows", -1))
        != len(rows)
        or any(
            raw_generation_contract_gate.get(key) is not True
            for key in raw_gate_boolean_keys
        )
        or (attestation.get("parameters") or {}).get(
            "raw_generation_contract_gate"
        )
        != raw_generation_contract_gate
        or (attestation.get("expected") or {}).get(
            "raw_generation_contract_gate"
        )
        != raw_generation_contract_gate
    ):
        raise RuntimeError(
            "PM-v2.2 reference-baseline raw-generation contract gate is stale"
        )
    call_plan = [dict(row) for row in iter_jsonl(call_plan_path)]
    if len(call_plan) != len(rows) or any(
        row.get("supporter_generation_treatment") != dict(treatment)
        or row.get("supporter_generation_treatment_sha256") != treatment_sha256
        or row.get("fixed_seeker_generation_treatment")
        != dict(fixed_seeker_treatment)
        or row.get("fixed_seeker_generation_treatment_sha256")
        != fixed_seeker_treatment_sha256
        or row.get("evidence_processing_contract")
        != normalized_evidence_contracts.get(str(row.get("condition") or ""))
        or row.get("evidence_processing_contract_sha256")
        != sha256_text(
            canonical_json(
                normalized_evidence_contracts.get(
                    str(row.get("condition") or ""), {}
                )
            )
        )
        or any(
            row.get(key) != expected for key, expected in policy_lock.items()
        )
        or int(row.get("max_output_tokens") or 0)
        != int(treatment["max_output_tokens"])
        for row in call_plan
    ):
        raise RuntimeError("PM-v2.2 reference-baseline call plan is mixed or incomplete")
    for condition in expected_conditions:
        planned_units = sorted(
            (
                str(row.get("user_id") or ""),
                int(row.get("topic_index") or 0),
                int(row.get("seed") or 0),
                str(row.get("simulator_id") or ""),
                int(row.get("turn_index") or 0),
            )
            for row in call_plan
            if str(row.get("condition") or "") == condition
        )
        if planned_units != expected_unit_rows:
            raise RuntimeError(
                f"PM-v2.2 reference-baseline call plan differs for {condition}"
            )
    cost_payload = {
        key: value
        for key, value in cost_estimate.items()
        if key not in {"cost_estimate_sha256", "budget_gate"}
    }
    if (
        cost_estimate.get("supporter_generation_treatment") != dict(treatment)
        or cost_estimate.get("supporter_generation_treatment_sha256")
        != treatment_sha256
        or cost_estimate.get("fixed_seeker_generation_treatment")
        != dict(fixed_seeker_treatment)
        or cost_estimate.get("fixed_seeker_generation_treatment_sha256")
        != fixed_seeker_treatment_sha256
        or cost_estimate.get("evidence_processing_contracts")
        != normalized_evidence_contracts
        or cost_estimate.get("evidence_processing_contracts_sha256")
        != evidence_processing_contracts_sha256
        or cost_estimate.get("evidence_filter_config_sha256")
        != evidence_filter_config_sha256
        or cost_estimate.get("evidence_filter_model")
        != dict(evidence_filter_model_binding)
        or any(
            cost_estimate.get(key) != expected
            for key, expected in policy_lock.items()
        )
        or (cost_estimate.get("budget_gate") or {}).get("status") != "PASS"
        or cost_estimate.get("call_plan_sha256")
        != sha256_text(canonical_json(call_plan))
        or cost_estimate.get("cost_estimate_sha256")
        != sha256_text(canonical_json(cost_payload))
    ):
        raise RuntimeError("PM-v2.2 reference-baseline cost estimate is stale")
    raw_rows = [dict(row) for row in iter_jsonl(raw_path)]
    planned_call_keys = {str(row.get("call_key") or "") for row in call_plan}
    raw_call_keys = {
        str(row.get("physical_call_key") or "") for row in raw_rows
    }
    if (
        len(raw_rows) != len(rows)
        or "" in planned_call_keys
        or raw_call_keys != planned_call_keys
        or any(
            row.get("normalized_finish_reason") != "complete"
            or row.get("supporter_generation_treatment")
            != dict(treatment)
            or row.get("supporter_generation_treatment_sha256")
            != treatment_sha256
            or row.get("fixed_seeker_generation_treatment")
            != dict(fixed_seeker_treatment)
            or row.get("fixed_seeker_generation_treatment_sha256")
            != fixed_seeker_treatment_sha256
            or row.get("evidence_processing_contract")
            != normalized_evidence_contracts.get(str(row.get("condition") or ""))
            or row.get("evidence_processing_contract_sha256")
            != sha256_text(
                canonical_json(
                    normalized_evidence_contracts.get(
                        str(row.get("condition") or ""), {}
                    )
                )
            )
            or any(
                row.get(key) != expected for key, expected in policy_lock.items()
            )
            for row in raw_rows
        )
    ):
        raise RuntimeError("PM-v2.2 reference-baseline raw calls are not all complete")
    ledger_rows = [dict(row) for row in iter_jsonl(ledger_path)]
    if (
        len([row for row in ledger_rows if row.get("event") == "SUCCEEDED"])
        != len(rows)
        or any(row.get("event") == "FAILED" for row in ledger_rows)
    ):
        raise RuntimeError("PM-v2.2 reference-baseline attempt ledger is incomplete")
    return {
        "stage": PMV22_REFERENCE_BASELINE_STAGE,
        "turns_sha256": sha256_file(turns_path),
        "turns_rows": len(rows),
        "attestation_sha256": verification["attestation_sha256"],
        "attestation_file_sha256": sha256_file(attestation_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "run_manifest_record_sha256": manifest["manifest_sha256"],
        "generator_endpoint": observed_endpoint,
        "generator_endpoint_sha256": sha256_text(
            canonical_json(observed_endpoint)
        ),
        "evidence_filter_config_sha256": evidence_filter_config_sha256,
        "evidence_filter_model": dict(evidence_filter_model_binding),
        "strategy_bank_approval": dict(strategy_bank_approval),
        **common_contract,
    }


def require_complete_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE":
        raise RuntimeError(
            "PM-v2 cannot be frozen unless training status is COMPLETE; "
            f"got {report.get('status')}"
        )
    checks = report.get("reportability_checks") or {}
    if not checks or not all(bool(value) for value in checks.values()):
        raise RuntimeError(
            "PM-v2 cannot be frozen because internal reportability checks did not all pass: "
            + str(checks)
        )
    return report


def require_training_report_checkpoint_binding(
    report: Mapping[str, Any], checkpoint_path: Path
) -> str:
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(
            "training report checkpoint hash does not match the selected "
            "PM-v2 checkpoint"
        )
    return checkpoint_sha256


def require_raw_family_quality_gate_pass(
    report: dict, *, subgroup_name: str
) -> dict:
    gate = report.get("raw_family_quality_gate") or {}
    global_gate = gate.get("global") or {}
    subgroup_gate = gate.get(subgroup_name) or {}
    subgroup_reports = subgroup_gate.get("subgroups") or {}
    family_reports = global_gate.get("families") or {}
    applicable_risk_gate = gate.get("action_applicable_risk_signal") or {}
    require_applicable_risk = subgroup_name == "family_by_action"
    if (
        gate.get("status") != "PASS"
        or global_gate.get("status") != "PASS"
        or subgroup_gate.get("status") != "PASS"
        or subgroup_gate.get("failed_subgroups")
        or not family_reports
        or not subgroup_reports
        or any(row.get("status") != "PASS" for row in family_reports.values())
        or any(row.get("status") != "PASS" for row in subgroup_reports.values())
        or (
            require_applicable_risk
            and (
                applicable_risk_gate.get("status") != "PASS"
                or applicable_risk_gate.get("enforced") is not True
                or applicable_risk_gate.get("failed_panels")
            )
        )
    ):
        raise RuntimeError(
            f"raw family/global/{subgroup_name} quality gates did not all PASS"
        )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "pm_v2_model" / "pm_v2.joblib")
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
    parser.add_argument(
        "--cost-matched-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "pm_v2_cost_matched_fixed.joblib",
    )
    parser.add_argument(
        "--me-r0-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "pm_v2_me_r0_fixed.joblib",
    )
    parser.add_argument(
        "--fixed-baseline-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_model" / "pm_v2_fixed_baselines.json",
    )
    parser.add_argument("--training-report", type=Path, default=ROOT / "outputs" / "pm_v2_model" / "training_report.json")
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl")
    parser.add_argument("--seed-dialogues", type=Path, default=ROOT / "data" / "pm_v2" / "train_seed_dialogues.jsonl")
    parser.add_argument("--data-report", type=Path, default=ROOT / "data" / "pm_v2" / "pm_v2_data_report.json")
    parser.add_argument(
        "--generation-pilot-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_compatibility_pilot_source_grounded_v5"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--generation-pilot-semantic-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_generation_pilot_semantic_review_v8_validation"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--generation-cost-estimate",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "generation_cost_estimate.json",
    )
    parser.add_argument(
        "--generation-call-plan",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "generation_call_plan.jsonl",
    )
    parser.add_argument(
        "--generation-attempt-ledger",
        type=Path,
        default=ROOT
        / "data"
        / "pm_v2"
        / "_generation_physical_attempt_ledger.jsonl",
    )
    parser.add_argument("--bundles", type=Path, default=ROOT / "data" / "pm_v2" / "pm_v2_bundles.jsonl")
    parser.add_argument("--runtime", type=Path, default=ROOT / "data" / "pm_v2" / "runtime_states.jsonl")
    parser.add_argument("--backend", type=Path, default=ROOT / "data" / "pm_v2" / "memory_backend.jsonl")
    parser.add_argument("--evaluator-contexts", type=Path, default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl")
    parser.add_argument("--seed-audit", type=Path, required=True)
    parser.add_argument("--sweep-outcomes", type=Path, default=ROOT / "outputs" / "pm_v2_sweep" / "action_outcomes.jsonl")
    parser.add_argument("--sweep-raw", type=Path, default=ROOT / "outputs" / "pm_v2_sweep" / "raw_api_calls.jsonl")
    parser.add_argument("--sweep-summary", type=Path, default=ROOT / "outputs" / "pm_v2_sweep" / "summary.json")
    parser.add_argument("--sweep-manifest", type=Path, default=ROOT / "outputs" / "pm_v2_sweep" / "run_manifest.json")
    parser.add_argument("--sweep-attestation", type=Path, default=ROOT / "outputs" / "pm_v2_sweep" / "artifact_attestation.json")
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "action_labels.jsonl")
    parser.add_argument("--judge-raw", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "judge_results.jsonl")
    parser.add_argument("--judge-manifest", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "run_manifest.json")
    parser.add_argument("--judge-summary", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "summary.json")
    parser.add_argument(
        "--judge-attestation",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_judging" / "artifact_attestation.json",
    )
    parser.add_argument(
        "--judge-pilot-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_pilot" / "pilot_plan.json",
    )
    parser.add_argument(
        "--pilot-sweep-summary",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_pilot" / "summary.json",
    )
    parser.add_argument(
        "--pilot-sweep-attestation",
        type=Path,
        default=(
            ROOT / "outputs" / "pm_v2_development_pilot" / "artifact_attestation.json"
        ),
    )
    parser.add_argument(
        "--judge-compatibility-manifest",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot_judging"
        / "run_manifest.json",
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
        "--pilot-human-spot-check-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot_human_spot_check"
        / "pilot_human_spot_check_report.json",
    )
    parser.add_argument(
        "--pilot-human-spot-check-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_pilot_human_spot_check"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--judge-schema-smoke-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_judge_schema_smoke"
        / "summary.json",
    )
    parser.add_argument(
        "--judge-schema-smoke-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_development_judge_schema_smoke"
        / "artifact_attestation.json",
    )
    parser.add_argument("--data-label-audit", type=Path, default=ROOT / "outputs" / "pm_v2_judging" / "data_label_audit.json")
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "artifact_attestation.json",
    )
    parser.add_argument("--human-audit", type=Path, default=ROOT / "outputs" / "pm_v2_human_audit" / "human_audit_report.json")
    parser.add_argument("--human-key", type=Path, default=ROOT / "outputs" / "pm_v2_human_audit" / "human_rating_key.json")
    parser.add_argument(
        "--human-manual",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_human_audit" / "human_rating_manual.md",
    )
    parser.add_argument(
        "--human-sample-plan",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_human_audit" / "human_sample_plan.json",
    )
    parser.add_argument("--evoemo", type=Path, default=ROOT / "data" / "external" / "evo_emo.json")
    parser.add_argument("--strategy-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards.jsonl")
    parser.add_argument(
        "--strategy-bank-approval",
        type=Path,
        default=ROOT
        / "outputs"
        / "strategy_rag_v1_frozen_candidate"
        / "human_approval.json",
    )
    parser.add_argument("--fixed-tracks", type=Path, default=ROOT / "outputs" / "evoemo_fixed_tracks_v22" / "fixed_seeker_tracks.jsonl")
    parser.add_argument("--fixed-tracks-attestation", type=Path, default=ROOT / "outputs" / "evoemo_fixed_tracks_v22" / "artifact_attestation.json")
    parser.add_argument(
        "--external-baseline-turns",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_pmv22_reference_baselines"
        / "turns.jsonl",
    )
    parser.add_argument(
        "--external-baseline-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_pmv22_reference_baselines"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--external-baseline-manifest",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_pmv22_reference_baselines"
        / "run_manifest.json",
    )
    parser.add_argument(
        "--external-baseline-summary",
        type=Path,
        default=ROOT
        / "outputs"
        / "evoemo_pmv22_reference_baselines"
        / "generation_summary.json",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "pm_v2_study_freeze.json")
    args = parser.parse_args()

    experiment_config = load_config(args.experiment_config)
    pm_v2_config = load_config(args.pm_v2_config)
    supporter_generation_contract = SupporterGenerationContract.from_config(
        pm_v2_config
    )
    supporter_generation_treatment = supporter_generation_contract.payload()
    supporter_generation_treatment_sha256 = supporter_generation_contract.digest()
    fixed_seeker_contract = FixedSeekerGenerationContract.from_mapping(
        pm_v2_config["fixed_seeker_generation_treatment"]
    )
    fixed_seeker_endpoint = endpoint_from_config(
        experiment_config, fixed_seeker_contract.seeker_endpoint
    )
    bound_fixed_seeker_contract = fixed_seeker_contract.bind_endpoint(
        fixed_seeker_contract.seeker_endpoint, fixed_seeker_endpoint
    )
    fixed_seeker_generation_treatment = bound_fixed_seeker_contract.payload()
    fixed_seeker_generation_treatment_sha256 = bound_fixed_seeker_contract.digest()
    fixed_seeker_cost_planning = fixed_seeker_cost_planning_contract(
        pm_v2_config.get("fixed_seeker_cost_planning") or {}
    )
    fixed_seeker_cost_planning_sha256 = sha256_text(
        canonical_json(fixed_seeker_cost_planning)
    )
    strategy_split_manifest = ROOT / "data" / "strategy" / "esconv_split_manifest.jsonl"
    strategy_bank_audit = ROOT / "data" / "strategy" / "strategy_bank_audit.json"
    strategy_rag_audit_summary = (
        ROOT / "outputs" / "strategy_rag_v1_audit" / "audit_summary.json"
    )
    strategy_candidate_manifest = (
        ROOT
        / "outputs"
        / "strategy_rag_v1_frozen_candidate"
        / "strategy_rag_manifest.json"
    )
    strategy_bank_human_approval = require_strategy_bank_human_approval(
        args.strategy_bank_approval,
        strategy_bank_path=args.strategy_bank,
        split_manifest_path=strategy_split_manifest,
        bank_audit_path=strategy_bank_audit,
        strategy_rag_audit_summary_path=strategy_rag_audit_summary,
        provisional_candidate_manifest_path=strategy_candidate_manifest,
    )
    evidence_filter_config = EvidenceFilterConfig.from_mapping(
        pm_v2_config["evidence_filter"]
    )
    if not evidence_filter_config.enabled:
        raise RuntimeError("PM-v2 study freeze requires the main Evidence Filter")
    frozen_evidence_filter = evidence_filter_config.payload()
    frozen_evidence_filter_sha256 = evidence_filter_config.digest()
    _, evidence_filter_model_binding = require_evidence_filter_artifacts(
        checkpoint_path=args.evidence_filter_checkpoint,
        report_path=args.evidence_filter_report,
        attestation_path=args.evidence_filter_attestation,
        pm_v2_config_path=args.pm_v2_config,
    )
    api_cost_planning = dict(pm_v2_config["api_cost_planning"])
    if set(api_cost_planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("api_cost_planning must contain exactly two frozen keys")
    if float(api_cost_planning["input_token_safety_factor"]) != 1.50:
        raise ValueError("api_cost_planning.input_token_safety_factor must equal 1.50")
    if not bool(api_cost_planning["fail_on_reported_input_overrun"]):
        raise ValueError("api_cost_planning must fail on reported input overrun")
    data_generation_cfg = dict(pm_v2_config["data_generation"])
    generator_pricing = {
        str(key): float(value)
        for key, value in dict(
            data_generation_cfg["pricing_usd_per_mtok"]
        ).items()
    }
    if set(generator_pricing) != {"input", "output"}:
        raise ValueError(
            "data_generation.pricing_usd_per_mtok must contain exactly input/output"
        )
    if generator_pricing != {"input": 0.15, "output": 0.60}:
        raise ValueError(
            "frozen gpt-4o-mini data-generation pricing must equal 0.15/0.60"
        )
    semantic_sanity = require_semantic_sanity_pass(
        report_path=args.semantic_sanity_report,
        attestation_path=args.semantic_sanity_attestation,
        config=pm_v2_config,
        config_path=args.pm_v2_config,
        states_path=args.states,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
    )
    runtime_state_lineage = require_pmv2_runtime_state_lineage(
        args.runtime, load_states(args.states)
    )
    development_pilot_gate = require_development_pilot_gate(
        experiment_config_path=args.experiment_config,
        pm_v2_config_path=args.pm_v2_config,
        states_path=args.states,
        runtime_path=args.runtime,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
        strategy_bank_path=args.strategy_bank,
        semantic_sanity=semantic_sanity,
        semantic_sanity_report_path=args.semantic_sanity_report,
        semantic_sanity_attestation_path=args.semantic_sanity_attestation,
        runtime_state_lineage=runtime_state_lineage,
        pilot_plan_path=args.judge_pilot_plan,
        pilot_sweep_summary_path=args.pilot_sweep_summary,
        pilot_sweep_attestation_path=args.pilot_sweep_attestation,
        judge_compatibility_summary_path=args.judge_compatibility_summary,
        judge_compatibility_attestation_path=args.judge_compatibility_attestation,
        pilot_human_spot_check_report_path=(
            args.pilot_human_spot_check_report
        ),
        pilot_human_spot_check_attestation_path=(
            args.pilot_human_spot_check_attestation
        ),
        judge_schema_smoke_summary_path=args.judge_schema_smoke_summary,
        judge_schema_smoke_attestation_path=(
            args.judge_schema_smoke_attestation
        ),
    )
    semantic_sanity_attestation = json.loads(
        args.semantic_sanity_attestation.read_text(encoding="utf-8")
    )
    semantic_sanity_input_paths = []
    for logical_name, record in (
        semantic_sanity_attestation.get("inputs") or {}
    ).items():
        if not isinstance(record, dict) or not record.get("path"):
            raise RuntimeError(
                f"semantic-sanity attestation has invalid input: {logical_name}"
            )
        semantic_sanity_input_paths.append(Path(str(record["path"])).resolve())
    composite_spec = composite_spec_from_config(pm_v2_config)
    composite_weights_sha256 = composite_weights_hash(composite_spec)
    labeling = labeling_settings_from_config(pm_v2_config)
    composite_support_gates = {
        "composite_support_exact_match_rate": float(
            labeling["composite_support_exact_match_rate"]
        ),
        "maximum_absolute_composite_support_correlation": float(
            labeling["maximum_absolute_composite_support_correlation"]
        ),
    }
    human_thresholds = dict(pm_v2_config["human_label_audit"])
    external_cfg = pm_v2_config["external_evaluation"]
    primary_bootstrap_cluster = str(external_cfg["primary_bootstrap_cluster"])
    sensitivity_bootstrap_cluster = str(
        external_cfg["sensitivity_bootstrap_cluster"]
    )
    if primary_bootstrap_cluster != "user_id":
        raise ValueError("external primary_bootstrap_cluster must equal user_id")
    if sensitivity_bootstrap_cluster != "scenario":
        raise ValueError(
            "external sensitivity_bootstrap_cluster must equal scenario"
        )
    maximum_cost_matched_relative_deviation = float(
        external_cfg["maximum_cost_matched_relative_deviation"]
    )
    if not 0.0 <= maximum_cost_matched_relative_deviation < 1.0:
        raise ValueError(
            "external maximum_cost_matched_relative_deviation must be in [0, 1)"
        )
    external_low_mad_thresholds = {
        "minimum_low_mad_coverage_per_dimension": float(
            external_cfg["minimum_low_mad_coverage_per_dimension"]
        ),
        "minimum_low_mad_coverage_per_condition_dimension": float(
            external_cfg["minimum_low_mad_coverage_per_condition_dimension"]
        ),
    }
    if any(
        not 0.0 <= value <= 1.0
        for value in external_low_mad_thresholds.values()
    ):
        raise ValueError("external low-MAD coverage thresholds must be in [0, 1]")
    development_sweep_cfg = dict(pm_v2_config["development_sweep"])
    expected_development_sweep_keys = {
        "pricing_usd_per_mtok",
        "seed",
        "request_retries",
        "fail_fast",
    }
    if set(development_sweep_cfg) != expected_development_sweep_keys:
        raise ValueError(
            "development_sweep must contain exactly the frozen contract keys"
        )
    sweep_generator_pricing = {
        str(key): float(value)
        for key, value in dict(
            development_sweep_cfg["pricing_usd_per_mtok"]
        ).items()
    }
    if sweep_generator_pricing != {"input": 0.15, "output": 0.60}:
        raise ValueError(
            "development_sweep generator pricing must equal the frozen "
            "positive 0.15/0.60 conservative proxy"
        )
    if int(development_sweep_cfg["request_retries"]) != 1:
        raise ValueError("development_sweep.request_retries must equal 1")
    if not bool(development_sweep_cfg["fail_fast"]):
        raise ValueError("development_sweep.fail_fast must be true")
    sweep_generator_endpoint_name = supporter_generation_contract.generator_endpoint
    sweep_generator_endpoint = endpoint_from_config(
        experiment_config, sweep_generator_endpoint_name
    )
    development_judging_cfg = dict(pm_v2_config["development_judging"])
    development_judge_names = [
        str(value) for value in development_judging_cfg["judge_endpoints"]
    ]
    development_judges = [
        endpoint_from_config(experiment_config, name)
        for name in development_judge_names
    ]
    if (
        len(development_judges) < int(labeling["minimum_families"])
        or len({endpoint.family for endpoint in development_judges})
        != len(development_judges)
    ):
        raise ValueError("development judges must use distinct model families")
    development_judge_pricing = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in dict(
            development_judging_cfg["pricing_usd_per_mtok"]
        ).items()
    }
    if set(development_judge_pricing) != {
        str(endpoint.family) for endpoint in development_judges
    } or any(
        set(values) != {"input", "output"}
        or any(value < 0.0 for value in values.values())
        for values in development_judge_pricing.values()
    ):
        raise ValueError("development judge pricing does not match endpoint families")
    generation_seeds = [
        int(value)
        for value in (experiment_config.get("protocol") or {}).get(
            "robustness_seeds", []
        )
    ]
    if not generation_seeds:
        raise ValueError("experiment protocol.robustness_seeds must be non-empty")
    generator_endpoint = sweep_generator_endpoint
    external_judge_names = [str(value) for value in external_cfg["external_judge_endpoints"]]
    external_judges = [
        endpoint_from_config(experiment_config, name) for name in external_judge_names
    ]
    if (
        len(external_judges) < int(labeling["minimum_families"])
        or len({endpoint.family for endpoint in external_judges})
        != len(external_judges)
    ):
        raise ValueError("frozen external judges must use distinct model families")
    external_judge_pricing = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in dict(
            external_cfg["judge_pricing_usd_per_mtok"]
        ).items()
    }
    if set(external_judge_pricing) != {
        str(endpoint.family) for endpoint in external_judges
    } or any(
        set(values) != {"input", "output"}
        or any(value <= 0.0 for value in values.values())
        for values in external_judge_pricing.values()
    ):
        raise ValueError(
            "external judge pricing must match endpoint families with strictly "
            "positive rates"
        )
    development_families = {endpoint.family for endpoint in development_judges}
    external_families = {endpoint.family for endpoint in external_judges}
    if development_families & external_families:
        raise ValueError(
            "development and external judge families must be completely disjoint"
        )
    action_preflight_gates = {
        str(key): float(value)
        for key, value in external_cfg["action_preflight"].items()
    }
    expected_action_gate_keys = {
        "maximum_severe_ood_fallback_rate",
        "maximum_no_feasible_fallback_rate",
        "minimum_m0_rate",
        "minimum_r0_rate",
        "minimum_m0_r0_rate",
        "minimum_nonfallback_rate",
        "maximum_action_share",
        "minimum_action_entropy_bits",
    }
    if set(action_preflight_gates) != expected_action_gate_keys:
        raise ValueError("external action_preflight keys do not match the frozen contract")
    for name, value in action_preflight_gates.items():
        if name == "minimum_action_entropy_bits":
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        elif not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    simulator_id = str(external_cfg["simulator_id"])
    if not simulator_id:
        raise ValueError("external_evaluation.simulator_id must be non-empty")
    max_turns = int(external_cfg["max_turns"])
    strategy_action_tokens = int(external_cfg["strategy_action_tokens"])
    if max_turns <= 0 or strategy_action_tokens <= 0:
        raise ValueError("external max_turns and strategy_action_tokens must be positive")
    turn_indices = [int(value) for value in external_cfg["turn_indices"]]
    if not turn_indices or turn_indices != sorted(set(turn_indices)):
        raise ValueError(
            "external_evaluation.turn_indices must be non-empty, sorted, and unique"
        )
    if any(value <= 0 or value > max_turns for value in turn_indices):
        raise ValueError("external turn_indices must be within the frozen dialogue length")
    external_expected_units = expected_external_units(
        args.evoemo,
        seeds=generation_seeds,
        simulator_id=simulator_id,
        turn_indices=turn_indices,
    )
    evaluation_unit_contract = {
        "protocol": EVALUATION_UNIT_CONTRACT_PROTOCOL,
        "evaluation_turn_indices": turn_indices,
        "expected_unit_count": len(external_expected_units),
        "expected_units_sha256": sha256_text(
            canonical_json(external_expected_units)
        ),
    }
    external_judge_seed = int(external_cfg["external_judge_seed"])
    retrieval_cfg = pm_v2_config["retrieval"]
    frozen_retrieval = {
        "strategy_top_k": int(external_cfg["strategy_top_k"]),
        "memory_min_score": float(external_cfg["memory_min_score"]),
        "strategy_min_score": float(external_cfg["strategy_min_score"]),
    }
    expected_retrieval = {
        "strategy_top_k": int(retrieval_cfg["strategy_top_k"]),
        "memory_min_score": float(retrieval_cfg["memory_min_score"]),
        "strategy_min_score": float(retrieval_cfg["strategy_min_score"]),
    }
    if frozen_retrieval != expected_retrieval:
        raise ValueError(
            "external retrieval settings must exactly match PM-v2 retrieval settings"
        )
    fixed_bundle_dir = args.fixed_tracks.resolve().parent
    fixed_tracks_verification = require_content_addressed_attestation(
        args.fixed_tracks_attestation,
        required_stage=FIXED_SEEKER_V22_STAGE,
        relocated_inputs={
            "evoemo": args.evoemo,
            "run_manifest": fixed_bundle_dir / "run_manifest.json",
            "cost_estimate": fixed_bundle_dir / "cost_estimate.json",
            "call_plan": fixed_bundle_dir / "call_plan.jsonl",
        },
        relocated_outputs={
            "tracks": args.fixed_tracks,
            "raw_calls": fixed_bundle_dir / "raw_seeker_calls.jsonl",
            "physical_attempt_ledger": fixed_bundle_dir
            / "physical_attempt_ledger.jsonl",
            "summary": fixed_bundle_dir / "summary.json",
        },
    )
    fixed_tracks_attestation = json.loads(
        args.fixed_tracks_attestation.read_text(encoding="utf-8")
    )
    expected_fixed_track_parameters = {
        "simulator_id": simulator_id,
        "max_turns": max_turns,
        "seeds": generation_seeds,
        "fixed_seeker_generation_contract": fixed_seeker_generation_treatment,
        "fixed_seeker_generation_contract_sha256": (
            fixed_seeker_generation_treatment_sha256
        ),
        "fixed_seeker_cost_planning": fixed_seeker_cost_planning,
        "fixed_seeker_cost_planning_sha256": (
            fixed_seeker_cost_planning_sha256
        ),
    }
    for key, expected in expected_fixed_track_parameters.items():
        if (fixed_tracks_attestation.get("parameters") or {}).get(key) != expected:
            raise RuntimeError(f"fixed-track attestation parameter mismatch: {key}")
    fixed_track_rows = sum(1 for _ in iter_jsonl(args.fixed_tracks))
    if (
        int((fixed_tracks_attestation.get("expected") or {}).get("tracks", -1))
        != fixed_track_rows
        or int(
            (fixed_tracks_attestation.get("expected") or {}).get(
                "turns_per_track", -1
            )
        )
        != max_turns
    ):
        raise RuntimeError("fixed-track attestation expected matrix mismatch")
    fixed_tracks_summary = json.loads(
        (fixed_bundle_dir / "summary.json").read_text(encoding="utf-8")
    )
    fixed_tracks_cost_estimate = json.loads(
        (fixed_bundle_dir / "cost_estimate.json").read_text(encoding="utf-8")
    )
    fixed_tracks_manifest = json.loads(
        (fixed_bundle_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    fixed_tracks_expected = fixed_tracks_attestation.get("expected") or {}
    planned_fixed_budget_gate = fixed_tracks_summary.get(
        "planned_budget_gate"
    ) or {}
    observed_fixed_budget_gate = fixed_tracks_summary.get(
        "observed_budget_gate"
    ) or {}
    planned_fixed_checks = planned_fixed_budget_gate.get("checks") or {}
    observed_fixed_checks = observed_fixed_budget_gate.get("checks") or {}
    fixed_track_records = [dict(row) for row in iter_jsonl(args.fixed_tracks)]
    fixed_track_raw_rows = [
        dict(row) for row in iter_jsonl(fixed_bundle_dir / "raw_seeker_calls.jsonl")
    ]
    try:
        require_complete_only_finish_reason_counts(
            fixed_tracks_summary.get("normalized_finish_reason_counts") or {},
            expected_rows=len(fixed_track_raw_rows),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "fixed seeker V2.2 tracks are mixed, truncated, or contract-stale"
        ) from exc
    if (
        fixed_tracks_summary.get("status") != "COMPLETE"
        or fixed_tracks_summary.get("fixed_seeker_generation_contract")
        != fixed_seeker_generation_treatment
        or fixed_tracks_summary.get("fixed_seeker_generation_contract_sha256")
        != fixed_seeker_generation_treatment_sha256
        or fixed_tracks_summary.get("fixed_seeker_cost_planning")
        != fixed_seeker_cost_planning
        or fixed_tracks_summary.get("fixed_seeker_cost_planning_sha256")
        != fixed_seeker_cost_planning_sha256
        or fixed_tracks_cost_estimate.get("fixed_seeker_cost_planning")
        != fixed_seeker_cost_planning
        or fixed_tracks_cost_estimate.get(
            "fixed_seeker_cost_planning_sha256"
        )
        != fixed_seeker_cost_planning_sha256
        or fixed_tracks_manifest.get("fixed_seeker_cost_planning")
        != fixed_seeker_cost_planning
        or fixed_tracks_manifest.get("fixed_seeker_cost_planning_sha256")
        != fixed_seeker_cost_planning_sha256
        or (fixed_tracks_cost_estimate.get("budget_gate") or {}).get("status")
        != "PASS"
        or planned_fixed_budget_gate.get("status") != "PASS"
        or observed_fixed_budget_gate.get("status") != "PASS"
        or not planned_fixed_checks
        or not all(value is True for value in planned_fixed_checks.values())
        or not observed_fixed_checks
        or not all(value is True for value in observed_fixed_checks.values())
        or (fixed_tracks_attestation.get("parameters") or {}).get(
            "planned_budget_gate"
        )
        != planned_fixed_budget_gate
        or (fixed_tracks_attestation.get("parameters") or {}).get(
            "observed_budget_gate"
        )
        != observed_fixed_budget_gate
        or fixed_tracks_expected.get("planned_budget_gate")
        != planned_fixed_budget_gate
        or fixed_tracks_expected.get("observed_budget_gate")
        != observed_fixed_budget_gate
        or fixed_tracks_manifest.get("planned_budget_gate")
        != planned_fixed_budget_gate
        or int(fixed_tracks_summary.get("completion_truncated_count", -1)) != 0
        or any(
            row.get("fixed_seeker_generation_contract")
            != fixed_seeker_generation_treatment
            or row.get("fixed_seeker_generation_contract_sha256")
            != fixed_seeker_generation_treatment_sha256
            or row.get("fixed_seeker_cost_planning")
            != fixed_seeker_cost_planning
            or row.get("fixed_seeker_cost_planning_sha256")
            != fixed_seeker_cost_planning_sha256
            or len(row.get("turn_provenance") or []) != max_turns
            or any(
                turn.get("normalized_finish_reason") != "complete"
                or turn.get("fixed_seeker_generation_contract_sha256")
                != fixed_seeker_generation_treatment_sha256
                or turn.get("fixed_seeker_cost_planning_sha256")
                != fixed_seeker_cost_planning_sha256
                for turn in (row.get("turn_provenance") or [])
            )
            for row in fixed_track_records
        )
        or any(
            row.get("normalized_finish_reason") != "complete"
            or row.get("fixed_seeker_generation_contract_sha256")
            != fixed_seeker_generation_treatment_sha256
            or row.get("fixed_seeker_cost_planning_sha256")
            != fixed_seeker_cost_planning_sha256
            for row in fixed_track_raw_rows
        )
    ):
        raise RuntimeError(
            "fixed seeker V2.2 tracks are mixed, truncated, or contract-stale"
        )
    fixed_tracks_call_plan = [
        dict(row) for row in iter_jsonl(fixed_bundle_dir / "call_plan.jsonl")
    ]
    fixed_cost_payload = {
        key: value
        for key, value in fixed_tracks_cost_estimate.items()
        if key != "dry_run_acceptance_sha256"
    }
    fixed_prices = fixed_seeker_cost_planning["pricing_usd_per_mtok"]
    fixed_safety_factor = float(
        fixed_seeker_cost_planning["input_token_safety_factor"]
    )
    recomputed_fixed_maximum_input_tokens = sum(
        int(row.get("maximum_input_tokens") or 0)
        for row in fixed_tracks_call_plan
    )
    recomputed_fixed_maximum_output_tokens = sum(
        int(row.get("maximum_output_tokens") or 0)
        for row in fixed_tracks_call_plan
    )
    recomputed_fixed_maximum_cost = sum(
        float(row.get("maximum_cost_usd") or 0.0)
        for row in fixed_tracks_call_plan
    )
    if (
        fixed_tracks_cost_estimate.get("dry_run_acceptance_sha256")
        != sha256_text(canonical_json(fixed_cost_payload))
        or fixed_tracks_cost_estimate.get("call_plan_sha256")
        != sha256_text(canonical_json(fixed_tracks_call_plan))
        or int(
            fixed_tracks_cost_estimate.get(
                "maximum_physical_api_attempts", -1
            )
        )
        != len(fixed_tracks_call_plan)
        or int(
            fixed_tracks_cost_estimate.get("maximum_total_input_tokens", -1)
        )
        != recomputed_fixed_maximum_input_tokens
        or int(
            fixed_tracks_cost_estimate.get("maximum_total_output_tokens", -1)
        )
        != recomputed_fixed_maximum_output_tokens
        or int(
            fixed_tracks_cost_estimate.get(
                "maximum_input_tokens_per_call", -1
            )
        )
        != max(
            (
                int(row.get("maximum_input_tokens") or 0)
                for row in fixed_tracks_call_plan
            ),
            default=0,
        )
        or not math.isclose(
            float(fixed_tracks_cost_estimate.get("maximum_estimated_usd", -1.0)),
            recomputed_fixed_maximum_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or len(fixed_track_raw_rows) != len(fixed_tracks_call_plan)
        or any(
            row.get("fixed_seeker_cost_planning_sha256")
            != fixed_seeker_cost_planning_sha256
            or int(row.get("maximum_input_tokens") or 0) <= 0
            or int(row.get("maximum_input_tokens") or 0)
            != math.ceil(
                fixed_safety_factor
                * (
                    int(
                        row.get(
                            "static_request_tokens_with_empty_prior_seeker_content"
                        )
                        or 0
                    )
                    + int(row.get("history_completion_token_cap") or 0)
                )
            )
            or int(row.get("maximum_output_tokens") or 0)
            != fixed_seeker_contract.max_output_tokens
            or float(row.get("maximum_cost_usd") or 0.0) <= 0.0
            or not math.isclose(
                float(row.get("maximum_cost_usd") or 0.0),
                (
                    int(row.get("maximum_input_tokens") or 0)
                    * fixed_prices["input"]
                    + int(row.get("maximum_output_tokens") or 0)
                    * fixed_prices["output"]
                )
                / 1_000_000,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or int(row.get("maximum_physical_attempts") or 0) != 1
            for row in fixed_tracks_call_plan
        )
    ):
        raise RuntimeError(
            "fixed seeker V2.2 cost estimate/call plan is stale or unbounded"
        )
    fixed_plan_by_key = {
        str(row.get("logical_call_key") or ""): row
        for row in fixed_tracks_call_plan
    }
    fixed_raw_by_key = {
        str(row.get("logical_call_key") or ""): row
        for row in fixed_track_raw_rows
    }
    fixed_ledger_rows = [
        dict(row)
        for row in iter_jsonl(
            fixed_bundle_dir / "physical_attempt_ledger.jsonl"
        )
    ]
    fixed_started_keys = [
        str(row.get("call_key") or "")
        for row in fixed_ledger_rows
        if row.get("event") == "STARTED"
    ]
    fixed_succeeded_keys = [
        str(row.get("call_key") or "")
        for row in fixed_ledger_rows
        if row.get("event") == "SUCCEEDED"
    ]
    if (
        "" in fixed_plan_by_key
        or len(fixed_plan_by_key) != len(fixed_tracks_call_plan)
        or "" in fixed_raw_by_key
        or len(fixed_raw_by_key) != len(fixed_track_raw_rows)
        or set(fixed_raw_by_key) != set(fixed_plan_by_key)
        or len(fixed_started_keys) != len(fixed_plan_by_key)
        or len(fixed_succeeded_keys) != len(fixed_plan_by_key)
        or set(fixed_started_keys) != set(fixed_plan_by_key)
        or set(fixed_succeeded_keys) != set(fixed_plan_by_key)
        or any(row.get("event") == "FAILED" for row in fixed_ledger_rows)
    ):
        raise RuntimeError(
            "fixed seeker V2.2 raw/ledger call-key matrix is not exact"
        )
    fixed_observed_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    fixed_observed_prompt_counts: list[int] = []
    fixed_observed_completion_counts: list[int] = []
    for logical_key, plan_row in fixed_plan_by_key.items():
        raw_row = fixed_raw_by_key[logical_key]
        usage = raw_row.get("usage") or {}
        try:
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
            total_tokens = int(usage["total_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "fixed seeker raw usage is missing or invalid"
            ) from exc
        if (
            prompt_tokens <= 0
            or completion_tokens <= 0
            or total_tokens != prompt_tokens + completion_tokens
            or prompt_tokens > int(plan_row["maximum_input_tokens"])
            or completion_tokens > int(plan_row["maximum_output_tokens"])
            or raw_row.get("error") is not None
            or raw_row.get("normalized_finish_reason") != "complete"
            or raw_row.get("fixed_seeker_cost_planning_sha256")
            != fixed_seeker_cost_planning_sha256
        ):
            raise RuntimeError(
                "fixed seeker raw usage exceeds its accepted per-call bound"
            )
        fixed_observed_usage["prompt_tokens"] += prompt_tokens
        fixed_observed_usage["completion_tokens"] += completion_tokens
        fixed_observed_usage["total_tokens"] += total_tokens
        fixed_observed_prompt_counts.append(prompt_tokens)
        fixed_observed_completion_counts.append(completion_tokens)
    recomputed_fixed_observed_cost = (
        fixed_observed_usage["prompt_tokens"] * fixed_prices["input"]
        + fixed_observed_usage["completion_tokens"] * fixed_prices["output"]
    ) / 1_000_000
    fixed_limits = fixed_tracks_cost_estimate.get("budget_limits") or {}
    expected_planned_checks = {
        "api_calls": len(fixed_tracks_call_plan)
        <= int(fixed_limits.get("max_api_calls", 0)),
        "estimated_cost_usd": float(
            fixed_tracks_cost_estimate["maximum_estimated_usd"]
        )
        <= float(fixed_limits.get("max_estimated_usd", 0.0)),
        "max_input_tokens_per_call": max(
            int(row["maximum_input_tokens"])
            for row in fixed_tracks_call_plan
        )
        <= int(fixed_limits.get("max_input_tokens_per_call", 0)),
    }
    recomputed_fixed_observed_checks = {
        "planned_budget_gate_passed": True,
        "physical_attempts": len(fixed_started_keys)
        <= int(fixed_limits.get("max_api_calls", 0)),
        "reported_usage_complete": True,
        "observed_cost_usd": recomputed_fixed_observed_cost
        <= float(fixed_limits.get("max_estimated_usd", 0.0)),
        "observed_cost_within_planned_upper_bound": (
            recomputed_fixed_observed_cost
            <= float(fixed_tracks_cost_estimate["maximum_estimated_usd"])
        ),
        "max_reported_input_tokens_per_call": max(
            fixed_observed_prompt_counts, default=0
        )
        <= int(fixed_limits.get("max_input_tokens_per_call", 0)),
        "max_reported_completion_tokens_per_call": max(
            fixed_observed_completion_counts, default=0
        )
        <= fixed_seeker_contract.max_output_tokens,
    }
    if (
        planned_fixed_budget_gate.get("checks") != expected_planned_checks
        or observed_fixed_budget_gate.get("checks")
        != recomputed_fixed_observed_checks
        or observed_fixed_budget_gate.get("reported_usage")
        != fixed_observed_usage
        or not math.isclose(
            float(observed_fixed_budget_gate.get("observed_cost_usd", -1.0)),
            recomputed_fixed_observed_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(
            observed_fixed_budget_gate.get(
                "max_reported_input_tokens_per_call", -1
            )
        )
        != max(fixed_observed_prompt_counts, default=0)
        or int(
            observed_fixed_budget_gate.get(
                "max_reported_completion_tokens_per_call", -1
            )
        )
        != max(fixed_observed_completion_counts, default=0)
    ):
        raise RuntimeError(
            "fixed seeker V2.2 planned/observed budget gate is not reproducible"
        )
    fixed_bundle_support_paths = [
        fixed_bundle_dir / "cost_estimate.json",
        fixed_bundle_dir / "call_plan.jsonl",
        fixed_bundle_dir / "raw_seeker_calls.jsonl",
        fixed_bundle_dir / "physical_attempt_ledger.jsonl",
    ]
    forced_swap_cfg = external_cfg["forced_swap"]
    expected_forced_swap_keys = {
        "sample_units",
        "order_variants",
        "judge_endpoints",
        "judge_seed",
        "estimated_output_tokens_per_call",
        "exclude_sample_from_full_evaluation",
        "maximum_order_disagreement_rate",
        "minimum_schema_success_rate",
        "minimum_cross_family_support_delta_correlation",
        "require_cross_family_direction_agreement",
        "minimum_support_delta_ci_upper_for_continuation",
    }
    if set(forced_swap_cfg) != expected_forced_swap_keys:
        raise ValueError("external forced_swap keys do not match the frozen contract")
    forced_swap_endpoint_names = [
        str(value) for value in forced_swap_cfg["judge_endpoints"]
    ]
    if not forced_swap_endpoint_names:
        raise ValueError("external forced_swap.judge_endpoints must be non-empty")
    if forced_swap_endpoint_names != external_judge_names:
        raise ValueError(
            "forced-swap must qualify the exact ordered external judge endpoints"
        )
    forced_swap_endpoints = [
        endpoint_from_config(experiment_config, name)
        for name in forced_swap_endpoint_names
    ]
    if len(forced_swap_endpoints) != 2 or len(
        {endpoint.family for endpoint in forced_swap_endpoints}
    ) != 2:
        raise ValueError("forced-swap requires exactly two distinct judge families")
    forced_swap_order_variants = [
        int(value) for value in forced_swap_cfg["order_variants"]
    ]
    if forced_swap_order_variants != [0, 1]:
        raise ValueError("forced_swap.order_variants must be exactly [0, 1]")
    if int(forced_swap_cfg["sample_units"]) <= 0:
        raise ValueError("forced_swap.sample_units must be positive")
    if int(forced_swap_cfg["estimated_output_tokens_per_call"]) <= 0:
        raise ValueError("forced_swap.estimated_output_tokens_per_call must be positive")
    if not bool(forced_swap_cfg["exclude_sample_from_full_evaluation"]):
        raise ValueError("forced-swap pilot units must be excluded from full evaluation")
    for name in (
        "maximum_order_disagreement_rate",
        "minimum_schema_success_rate",
    ):
        if not 0.0 <= float(forced_swap_cfg[name]) <= 1.0:
            raise ValueError(f"forced_swap.{name} must be in [0, 1]")
    if not -1.0 <= float(
        forced_swap_cfg["minimum_cross_family_support_delta_correlation"]
    ) <= 1.0:
        raise ValueError("forced-swap correlation threshold must be in [-1, 1]")
    frozen_forced_swap_units = select_forced_swap_units(
        external_expected_units,
        sample_units=int(forced_swap_cfg["sample_units"]),
        seed=int(forced_swap_cfg["judge_seed"]),
    )
    schema_smoke_unit = frozen_forced_swap_units[0]
    pointwise_schema_smoke_contract = {
        "protocol": "pm-v2-external-pointwise-schema-smoke-v1",
        "purpose": "schema_transport_only_not_raw_family_quality",
        "required_before_full_external_client_creation": True,
        "condition": "pm_v2",
        "condition_identity_visibility_to_judges": "hidden",
        "unit_selection": "first_sorted_frozen_forced_swap_excluded_unit",
        "unit": list(schema_smoke_unit),
        "unit_id": forced_swap_unit_id(schema_smoke_unit),
        "forced_swap_selected_units_sha256": sha256_text(
            canonical_json(frozen_forced_swap_units)
        ),
        "expected_calls": 4,
        "judge_seed": external_judge_seed,
        "estimated_response_output_tokens": 600,
        "estimated_risk_output_tokens": 700,
        "judge_retries": 1,
        "judge_prompt_contract_sha256": prompt_contract_hash(),
        "judge_pricing_usd_per_mtok": external_judge_pricing,
        "api_cost_planning": api_cost_planning,
        "judge_endpoints": [
            {
                "name": name,
                "model": endpoint.model,
                "family": endpoint.family,
                "base_url": endpoint.base_url,
                "sha256": sha256_text(
                    canonical_json(
                        {
                            "model": endpoint.model,
                            "family": endpoint.family,
                            "base_url": endpoint.base_url,
                        }
                    )
                ),
            }
            for name, endpoint in zip(external_judge_names, external_judges)
        ],
    }

    report = require_complete_report(args.training_report)
    require_training_report_checkpoint_binding(report, args.checkpoint)
    require_learned_routing_advantage = bool(
        pm_v2_config["internal_reportability_gate"][
            "require_learned_routing_advantage_before_external"
        ]
    )
    if not require_learned_routing_advantage:
        raise RuntimeError(
            "study freeze requires learned-routing advantage before external work"
        )
    if report.get("require_learned_routing_advantage_before_external") is not True:
        raise RuntimeError(
            "training report does not bind the required learned-routing gate"
        )
    if report.get("learned_routing_advantage_verified") is not True:
        raise RuntimeError(
            "PM-v2 cannot be frozen without verified learned-routing advantage"
        )
    near_duplicate_cfg = dict(pm_v2_config["splits"]["near_duplicate_audit"])
    if set(near_duplicate_cfg) != {
        "method",
        "maximum_word_hash_cosine",
        "maximum_char_hash_cosine",
        "report_top_pairs",
    }:
        raise ValueError("splits.near_duplicate_audit contract keys are stale")
    expected_near_duplicate_contract = {
        "method": str(near_duplicate_cfg["method"]),
        "thresholds": {
            "maximum_word_hash_cosine": float(
                near_duplicate_cfg["maximum_word_hash_cosine"]
            ),
            "maximum_char_hash_cosine": float(
                near_duplicate_cfg["maximum_char_hash_cosine"]
            ),
        },
    }
    cross_split_near_duplicate_audit = (
        report.get("cross_split_near_duplicate_audit") or {}
    )
    if cross_split_near_duplicate_audit.get("status") != "PASS":
        raise RuntimeError("cross-split near-duplicate audit did not PASS")
    for key, expected in expected_near_duplicate_contract.items():
        if cross_split_near_duplicate_audit.get(key) != expected:
            raise RuntimeError(
                f"cross-split near-duplicate audit contract mismatch: {key}"
            )
    if int(cross_split_near_duplicate_audit.get("flagged_pair_count", -1)) != 0:
        raise RuntimeError("cross-split near-duplicate audit contains flagged pairs")
    cross_split_near_duplicate_audit_sha256 = sha256_text(
        canonical_json(cross_split_near_duplicate_audit)
    )
    model = PMV2Model.load(args.checkpoint)
    ood_calibration = report.get("ood_calibration") or {}
    if (
        ood_calibration.get("status") != "PASS"
        or ood_calibration.get("protocol") != "pm-v2-ood-calibration-v1"
        or model.feature_builder.ood_calibration_report != ood_calibration
        or model.feature_builder.semantic_ood_threshold
        != ood_calibration.get("semantic_ood_threshold")
        or model.feature_builder.metadata_ood_threshold
        != ood_calibration.get("metadata_ood_threshold")
    ):
        raise RuntimeError("checkpoint lacks the frozen calibration-split OOD gate")
    if report.get("selection_config_hash") != model.selection_config.digest():
        raise RuntimeError("training report selection hash does not match checkpoint")
    if model.selection_config.composite_spec != composite_spec:
        raise RuntimeError("checkpoint composite spec does not match PM-v2 YAML")
    if report.get("pm_v2_config_sha256") != sha256_file(args.pm_v2_config):
        raise RuntimeError("training report PM-v2 config hash mismatch")
    judge_summary = json.loads(args.judge_summary.read_text(encoding="utf-8"))
    if judge_summary.get("status") != "COMPLETE":
        raise RuntimeError("judge summary is not COMPLETE")
    if (judge_summary.get("quality_gate") or {}).get("status") != "PASS":
        raise RuntimeError("judge quality gate did not pass")
    full_raw_family_quality_gate = require_raw_family_quality_gate_pass(
        judge_summary, subgroup_name="family_by_action"
    )
    if judge_summary.get("prompt_contract_hash") != prompt_contract_hash():
        raise RuntimeError("judge summary prompt contract hash is stale")
    if judge_summary.get("pm_v2_config_sha256") != sha256_file(args.pm_v2_config):
        raise RuntimeError("judge summary PM-v2 config hash mismatch")
    if judge_summary.get("composite_spec") != composite_spec.model_dump(mode="json"):
        raise RuntimeError("judge summary composite spec mismatch")
    if judge_summary.get("composite_weights_sha256") != composite_weights_sha256:
        raise RuntimeError("judge summary composite weights hash mismatch")
    if judge_summary.get("labeling_gates") != labeling:
        raise RuntimeError("judge summary labeling gates mismatch")
    if judge_summary.get("development_judging") != development_judging_cfg:
        raise RuntimeError("judge summary development_judging contract mismatch")
    if (
        judge_summary.get("scope") != "full"
        or judge_summary.get("reportability_status") != "REPORTABLE"
    ):
        raise RuntimeError("judge summary is a nonreportable pilot")
    judge_manifest = json.loads(args.judge_manifest.read_text(encoding="utf-8"))
    if judge_manifest.get("composite_weights_sha256") != composite_weights_sha256:
        raise RuntimeError("judge manifest composite weights hash mismatch")
    expected_development_judge_contracts = [
        {
            "name": name,
            "model": endpoint.model,
            "family": endpoint.family,
            "base_url": endpoint.base_url,
        }
        for name, endpoint in zip(development_judge_names, development_judges)
    ]
    pilot_config = dict(development_judging_cfg["compatibility_pilot"])
    pilot_plan = json.loads(args.judge_pilot_plan.read_text(encoding="utf-8"))
    pilot_payload = {
        key: value for key, value in pilot_plan.items() if key != "pilot_plan_sha256"
    }
    pilot_plan_sha256 = sha256_text(canonical_json(pilot_payload))
    if (
        pilot_plan.get("status") != "READY"
        or pilot_plan.get("pilot_plan_sha256") != pilot_plan_sha256
    ):
        raise RuntimeError("development judge compatibility pilot plan is stale")
    expected_pilot_plan = {
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "states_sha256": sha256_file(args.states),
        "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
        "sample_seed": int(pilot_config["sample_seed"]),
        "states_per_regime": int(pilot_config["states_per_regime"]),
        "actions": [str(value) for value in pilot_config["actions"]],
    }
    for key, expected in expected_pilot_plan.items():
        if pilot_plan.get(key) != expected:
            raise RuntimeError(f"development judge pilot plan mismatch: {key}")
    pilot_expected_keys_sha256 = str(pilot_plan.get("expected_keys_sha256") or "")
    if len(pilot_expected_keys_sha256) != 64:
        raise RuntimeError("development judge pilot lacks expected-key hash")

    compatibility_verification = require_artifact_attestation(
        args.judge_compatibility_attestation,
        required_stage="pm_v2_development_judge_compatibility",
        required_output_paths={"summary": args.judge_compatibility_summary},
    )
    compatibility_attestation_sha256 = str(
        compatibility_verification["attestation_sha256"]
    )
    compatibility_summary = json.loads(
        args.judge_compatibility_summary.read_text(encoding="utf-8")
    )
    compatibility_raw_family_quality_gate = require_raw_family_quality_gate_pass(
        compatibility_summary, subgroup_name="family_by_action"
    )
    expected_compatibility_summary = {
        "status": "PASS",
        "scope": "compatibility_pilot",
        "reportability_status": "COMPATIBILITY_GATE_ONLY",
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "prompt_contract_hash": prompt_contract_hash(),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_sha256,
        "labeling_gates": labeling,
        "development_judging": development_judging_cfg,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "judge_endpoint_descriptors": expected_development_judge_contracts,
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
    }
    for key, expected in expected_compatibility_summary.items():
        if compatibility_summary.get(key) != expected:
            raise RuntimeError(
                f"development judge compatibility summary mismatch: {key}"
            )
    compatibility_gate = compatibility_summary.get("compatibility_gate") or {}
    if compatibility_gate.get("status") != "PASS" or not all(
        bool(value) for value in (compatibility_gate.get("checks") or {}).values()
    ):
        raise RuntimeError("development judge compatibility gate did not fully pass")
    expected_compatibility_thresholds = {
        "minimum_schema_success_rate": float(
            pilot_config["minimum_schema_success_rate"]
        ),
        "minimum_reliable_label_rate": float(
            pilot_config["minimum_reliable_label_rate"]
        ),
        "minimum_low_mad_coverage_per_dimension": float(
            pilot_config["minimum_low_mad_coverage_per_dimension"]
        ),
        "minimum_low_mad_coverage_per_action_dimension": float(
            pilot_config["minimum_low_mad_coverage_per_action_dimension"]
        ),
    }
    if compatibility_gate.get("thresholds") != expected_compatibility_thresholds:
        raise RuntimeError("development judge compatibility thresholds mismatch")
    compatibility_manifest = json.loads(
        args.judge_compatibility_manifest.read_text(encoding="utf-8")
    )
    expected_compatibility_manifest = {
        "stage": "pm_v2_development_judge_compatibility",
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "judge_endpoints": expected_development_judge_contracts,
        "prompt_contract_hash": prompt_contract_hash(),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_sha256,
        "labeling": labeling,
        "development_judging": development_judging_cfg,
        "seed": int(development_judging_cfg["seed"]),
        "response_max_output_tokens": int(
            development_judging_cfg["response_max_output_tokens"]
        ),
        "risk_max_output_tokens": int(
            development_judging_cfg["risk_max_output_tokens"]
        ),
        "scope": "compatibility_pilot",
        "max_outcomes": None,
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": None,
        "judge_retries": 1,
        "pricing_usd_per_mtok": development_judge_pricing,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    }
    for key, expected in expected_compatibility_manifest.items():
        if compatibility_manifest.get(key) != expected:
            raise RuntimeError(
                f"development judge compatibility manifest mismatch: {key}"
            )
    if compatibility_summary.get("run_manifest_sha256") != compatibility_manifest.get(
        "manifest_sha256"
    ):
        raise RuntimeError("compatibility summary/run manifest hash mismatch")
    compatibility_attestation = json.loads(
        args.judge_compatibility_attestation.read_text(encoding="utf-8")
    )
    compatibility_parameters = compatibility_attestation.get("parameters") or {}
    expected_compatibility_parameters = {
        "status": "PASS",
        "scope": "compatibility_pilot",
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "prompt_contract_hash": prompt_contract_hash(),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_sha256,
        "labeling_gates": labeling,
        "development_judging": development_judging_cfg,
        "judge_endpoint_descriptors": expected_development_judge_contracts,
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": None,
        "compatibility_gate": compatibility_gate,
        "raw_family_quality_gate": compatibility_raw_family_quality_gate,
        "judge_retries": 1,
        "pricing_usd_per_mtok": development_judge_pricing,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    }
    for key, expected in expected_compatibility_parameters.items():
        if compatibility_parameters.get(key) != expected:
            raise RuntimeError(
                f"development judge compatibility attestation mismatch: {key}"
            )
    compatibility_output_paths = []
    for logical_name in ("labels", "raw_results", "call_ledger"):
        record = (compatibility_attestation.get("outputs") or {}).get(logical_name)
        if not isinstance(record, dict) or not record.get("path"):
            raise RuntimeError(
                f"compatibility attestation lacks output: {logical_name}"
            )
        compatibility_output_paths.append(Path(str(record["path"])).resolve())
    compatibility_input_paths = []
    for logical_name, record in (compatibility_attestation.get("inputs") or {}).items():
        if not isinstance(record, dict) or not record.get("path"):
            raise RuntimeError(
                f"compatibility attestation has invalid input: {logical_name}"
            )
        compatibility_input_paths.append(Path(str(record["path"])).resolve())

    expected_development_judge_runtime = {
        "development_judging": development_judging_cfg,
        "judge_endpoints": expected_development_judge_contracts,
        "seed": int(development_judging_cfg["seed"]),
        "response_max_output_tokens": int(
            development_judging_cfg["response_max_output_tokens"]
        ),
        "risk_max_output_tokens": int(
            development_judging_cfg["risk_max_output_tokens"]
        ),
        "scope": "full",
        "max_outcomes": None,
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": compatibility_attestation_sha256,
        "judge_retries": 1,
        "pricing_usd_per_mtok": development_judge_pricing,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    }
    for key, expected in expected_development_judge_runtime.items():
        if judge_manifest.get(key) != expected:
            raise RuntimeError(f"judge manifest runtime mismatch: {key}")
    expected_judge_compatibility_bindings = {
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": compatibility_attestation_sha256,
    }
    for key, expected in expected_judge_compatibility_bindings.items():
        if judge_summary.get(key) != expected:
            raise RuntimeError(f"judge summary compatibility mismatch: {key}")
    if judge_summary.get("run_manifest_sha256") != judge_manifest.get("manifest_sha256"):
        raise RuntimeError("judge summary/run manifest hash mismatch")
    judge_attestation_verification = require_artifact_attestation(
        args.judge_attestation,
        required_stage="pm_v2_action_judging",
        required_output_paths={
            "summary": args.judge_summary,
            "labels": args.labels,
            "raw_results": args.judge_raw,
        },
    )
    judge_attestation = json.loads(args.judge_attestation.read_text(encoding="utf-8"))
    judge_attestation_parameters = judge_attestation.get("parameters") or {}
    expected_judge_attestation_parameters = {
        "status": "COMPLETE",
        "scope": "full",
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "prompt_contract_hash": prompt_contract_hash(),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_sha256,
        "labeling_gates": labeling,
        "development_judging": development_judging_cfg,
        "judge_endpoint_descriptors": expected_development_judge_contracts,
        "pilot_plan_sha256": pilot_plan_sha256,
        "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
        "compatibility_attestation_sha256": compatibility_attestation_sha256,
        "compatibility_gate": None,
        "raw_family_quality_gate": full_raw_family_quality_gate,
        "judge_retries": 1,
        "pricing_usd_per_mtok": development_judge_pricing,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    }
    for key, expected in expected_judge_attestation_parameters.items():
        if judge_attestation_parameters.get(key) != expected:
            raise RuntimeError(f"judge attestation parameter mismatch: {key}")
    if len(str(judge_attestation_parameters.get("accepted_cost_estimate_sha256") or "")) != 64:
        raise RuntimeError("judge attestation lacks accepted cost-estimate hash")
    judge_attested_input_paths = []
    for logical_name, record in (judge_attestation.get("inputs") or {}).items():
        if not isinstance(record, dict) or not record.get("path"):
            raise RuntimeError(f"judge attestation has invalid input: {logical_name}")
        judge_attested_input_paths.append(Path(str(record["path"])).resolve())
    judge_attested_output_paths = []
    for logical_name, record in (judge_attestation.get("outputs") or {}).items():
        if not isinstance(record, dict) or not record.get("path"):
            raise RuntimeError(f"judge attestation has invalid output: {logical_name}")
        judge_attested_output_paths.append(Path(str(record["path"])).resolve())
    label_audit = json.loads(args.data_label_audit.read_text(encoding="utf-8"))
    if label_audit.get("status") != "PASS":
        raise RuntimeError("data/label diversity audit did not pass")
    human_audit = json.loads(args.human_audit.read_text(encoding="utf-8"))
    if human_audit.get("status") != "PASS":
        raise RuntimeError("human-versus-LLM judge calibration audit did not pass")
    if human_audit.get("thresholds") != {
        key: human_thresholds[key]
        for key in (
            "minimum_annotators",
            "maximum_response_mae",
            "minimum_response_spearman",
            "maximum_risk_mae",
            "minimum_within_one_rate",
            "minimum_interrater_kappa",
        )
    }:
        raise RuntimeError("human audit thresholds do not exactly match PM-v2 YAML")
    if human_audit.get("pm_v2_config_sha256") != sha256_file(args.pm_v2_config):
        raise RuntimeError("human audit PM-v2 config hash mismatch")
    if human_audit.get("human_key_sha256") != sha256_file(args.human_key):
        raise RuntimeError("human audit key hash mismatch")
    human_key = json.loads(args.human_key.read_text(encoding="utf-8"))
    manual_version = "pm-v2-human-rating-manual-v2-independent-dimensions"
    if (
        Path(str(human_key.get("manual") or "")).resolve()
        != args.human_manual.resolve()
        or human_key.get("manual_version") != manual_version
        or human_key.get("manual_sha256") != sha256_file(args.human_manual)
    ):
        raise RuntimeError("human rating manual path/version/hash mismatch")
    manual_text = args.human_manual.read_text(encoding="utf-8")
    manual_fragments = [manual_version]
    for field in (
        "response_dimension_definitions",
        "response_dimension_boundaries",
        "response_scale_anchors",
        "risk_dimension_definitions",
        "risk_scale_anchors",
    ):
        anchors = human_key.get(field)
        if not isinstance(anchors, dict) or not anchors:
            raise RuntimeError(f"human rating key lacks manual anchors: {field}")
        for name, descriptions in anchors.items():
            manual_fragments.append(str(name))
            manual_fragments.extend(
                str(value)
                for value in (
                    descriptions if isinstance(descriptions, list) else [descriptions]
                )
            )
    if any(fragment not in manual_text for fragment in manual_fragments):
        raise RuntimeError("human rating manual content does not match frozen anchors")

    human_sample_plan = json.loads(
        args.human_sample_plan.read_text(encoding="utf-8")
    )
    sample_wrapper = human_key.get("sample_plan") or {}
    sample_plan_version = "pm-v2-human-sample-plan-v1"
    expected_sample_wrapper = {
        "version": sample_plan_version,
        "sha256": sha256_text(canonical_json(human_sample_plan)),
        "file_sha256": sha256_file(args.human_sample_plan),
        "sample_seed": int(human_thresholds["sample_seed"]),
        "seed_source": "pm_v2.yaml:human_label_audit.sample_seed",
    }
    if Path(str(sample_wrapper.get("path") or "")).resolve() != args.human_sample_plan.resolve():
        raise RuntimeError("human sample-plan path mismatch")
    for key, expected in expected_sample_wrapper.items():
        if sample_wrapper.get(key) != expected:
            raise RuntimeError(f"human sample-plan wrapper mismatch: {key}")
    expected_sample_plan_keys = {
        "version",
        "sample_seed",
        "seed_source",
        "target_items",
        "regime_order",
        "initial_per_regime_quota",
        "candidate_pool_count",
        "candidate_pool_sha256",
        "ordered_selection",
        "input_bindings",
    }
    if set(human_sample_plan) != expected_sample_plan_keys:
        raise RuntimeError("human sample-plan schema mismatch")
    if (
        human_sample_plan.get("version") != sample_plan_version
        or int(human_sample_plan.get("sample_seed", -1))
        != int(human_thresholds["sample_seed"])
        or int(human_sample_plan.get("target_items", -1))
        != int(human_thresholds["items"])
        or len(human_sample_plan.get("ordered_selection") or [])
        != int(human_thresholds["items"])
    ):
        raise RuntimeError("human sample plan violates PM-v2 YAML")
    key_rows_by_item = {
        str(row["item_id"]): row for row in human_key.get("key_rows") or []
    }
    selection_by_item = {
        str(row["item_id"]): row
        for row in human_sample_plan.get("ordered_selection") or []
    }
    if set(key_rows_by_item) != set(selection_by_item):
        raise RuntimeError("human sample plan and rating-key item sets differ")
    for item_id, selection in selection_by_item.items():
        key_row = key_rows_by_item[item_id]
        for field in (
            "state_id",
            "card_id",
            "action_id",
            "regime",
            "outcome_request_hash",
        ):
            if selection.get(field) != key_row.get(field):
                raise RuntimeError(
                    f"human sample plan/rating-key mismatch: {item_id}/{field}"
                )
    sample_inputs = human_sample_plan.get("input_bindings") or {}
    expected_sample_inputs = {
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "states_sha256": sha256_file(args.states),
        "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
        "outcomes_sha256": sha256_file(args.sweep_outcomes),
        "labels_sha256": sha256_file(args.labels),
        "judge_manifest_sha256": sha256_file(args.judge_manifest),
        "llm_rubric_version": JUDGE_RUBRIC_VERSION,
        "llm_prompt_contract_sha256": prompt_contract_hash(),
        "quality_composite_version": composite_spec.version,
        "quality_composite_weights_sha256": composite_weights_sha256,
    }
    for key, expected in expected_sample_inputs.items():
        if sample_inputs.get(key) != expected:
            raise RuntimeError(f"human sample-plan input mismatch: {key}")
    rubric_contract = human_key.get("llm_rubric_contract") or {}
    expected_rubric_contract = {
        "version": JUDGE_RUBRIC_VERSION,
        "prompt_contract_sha256": prompt_contract_hash(),
        "requests_overall_field": False,
        "judge_manifest_sha256": sha256_file(args.judge_manifest),
    }
    if Path(str(rubric_contract.get("judge_manifest") or "")).resolve() != args.judge_manifest.resolve():
        raise RuntimeError("human LLM-rubric judge-manifest path mismatch")
    for key, expected in expected_rubric_contract.items():
        if rubric_contract.get(key) != expected:
            raise RuntimeError(f"human LLM-rubric contract mismatch: {key}")
    expected_human_report_bindings = {
        "manual": str(args.human_manual.resolve()),
        "manual_version": manual_version,
        "manual_sha256": sha256_file(args.human_manual),
        "sample_plan": {
            **expected_sample_wrapper,
            "path": str(args.human_sample_plan.resolve()),
        },
        "llm_rubric_contract": {
            **expected_rubric_contract,
            "judge_manifest": str(args.judge_manifest.resolve()),
        },
    }
    for key, expected in expected_human_report_bindings.items():
        if human_audit.get(key) != expected:
            raise RuntimeError(f"human audit report binding mismatch: {key}")
    human_checks = human_audit.get("checks") or {}
    required_human_checks = {
        "response.emotional_support.mae",
        "response.emotional_support.spearman",
        "response.emotional_support.within_one",
        "response.emotional_support.kappa",
        "quality_composite.mae",
        "quality_composite.spearman",
        "quality_composite.within_one",
        *{f"risk.{name}.kappa" for name in (
            "selected_context_misuse",
            "unnecessary_exposure",
            "stale_or_conflicting_use",
            "unsupported_personal_claim",
            "memory_omission",
            "strategy_overuse",
            "strategy_omission",
        )},
    }
    missing_human_checks = sorted(required_human_checks - set(human_checks))
    if missing_human_checks or not all(bool(human_checks[name]) for name in required_human_checks):
        raise RuntimeError(
            "human audit lacks passing construct-validity checks: "
            + str(missing_human_checks or sorted(
                name for name in required_human_checks if not human_checks[name]
            ))
        )
    seed_audit = json.loads(args.seed_audit.read_text(encoding="utf-8"))
    if seed_audit.get("status") != "COMPLETE" or seed_audit.get("test_or_validation_rows_in_output") != 0:
        raise RuntimeError("seed lineage audit is incomplete or contains non-train rows")
    if seed_audit.get("output_sha256") != sha256_file(args.seed_dialogues):
        raise RuntimeError("seed audit output hash does not match seed dialogues")
    seed_contract_version = "pm-v2-seed-extraction-v1-strict-lineage"
    if seed_audit.get("seed_extraction_contract_version") != seed_contract_version:
        raise RuntimeError("seed extraction contract version is stale")
    seed_code_manifest = seed_audit.get("code_manifest") or {}
    expected_seed_code_paths = {
        "scripts/18_prepare_pm_v2_seed_dialogues.py",
        "src/metacom_pm/io.py",
    }
    if (
        seed_code_manifest.get("contract_version") != seed_contract_version
        or {
            str(row.get("path")) for row in seed_code_manifest.get("files") or []
        }
        != expected_seed_code_paths
    ):
        raise RuntimeError("seed extraction code manifest is incomplete or stale")
    seed_code_manifest_sha256 = sha256_text(canonical_json(seed_code_manifest))
    if seed_audit.get("code_manifest_sha256") != seed_code_manifest_sha256:
        raise RuntimeError("seed extraction code manifest self-hash mismatch")
    seed_lineage_hashes = seed_audit.get("lineage_manifest_hashes") or {}
    if seed_lineage_hashes.get("code_manifest_sha256") != seed_code_manifest_sha256:
        raise RuntimeError("seed lineage does not bind extraction code")
    for row in seed_code_manifest["files"]:
        logical_path = str(row["path"])
        code_path = (ROOT / logical_path).resolve()
        if (
            not code_path.is_relative_to(ROOT.resolve())
            or not code_path.is_file()
            or row.get("sha256") != sha256_file(code_path)
        ):
            raise RuntimeError(f"seed extraction code hash mismatch: {logical_path}")
    seed_dialogue_rows = list(iter_jsonl(args.seed_dialogues))
    for index, row in enumerate(seed_dialogue_rows):
        if row.get("seed_extraction_contract_version") != seed_contract_version:
            raise RuntimeError(f"seed row {index} has a stale extraction contract")
        for name, expected in seed_lineage_hashes.items():
            if row.get(name) != expected:
                raise RuntimeError(
                    f"seed row {index} lineage hash mismatch: {name}"
                )
    seed_source_paths = [Path(value).resolve() for value in seed_audit.get("inputs") or []]
    seed_source_hashes = seed_audit.get("input_sha256") or {}
    if not seed_source_paths:
        raise RuntimeError("seed audit does not bind source inputs")
    for path in seed_source_paths:
        expected = seed_source_hashes.get(str(path)) or seed_source_hashes.get(
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else ""
        )
        if not path.is_file() or expected != sha256_file(path):
            raise RuntimeError(f"seed source hash mismatch: {path}")

    data_report = json.loads(args.data_report.read_text(encoding="utf-8"))
    if data_report.get("status") != "COMPLETE":
        raise RuntimeError("PM-v2 data report is not COMPLETE")
    if data_report.get("pm_v2_config_sha256") != sha256_file(args.pm_v2_config):
        raise RuntimeError("PM-v2 data report config hash mismatch")
    if data_report.get("seed_dialogues_sha256") != sha256_file(args.seed_dialogues):
        raise RuntimeError("PM-v2 data report seed hash mismatch")
    frozen_user_design = {
        "train_users": 24,
        "calibration_users": 12,
        "internal_test_users": 16,
        "cases_per_user": 9,
    }
    for key, expected in frozen_user_design.items():
        if int(data_generation_cfg.get(key, -1)) != expected:
            raise RuntimeError(
                f"data_generation.{key} must remain at the preregistered value "
                f"{expected}"
            )
    if len(data_generation_cfg.get("required_regimes") or []) != 9:
        raise RuntimeError("data_generation.required_regimes must contain nine regimes")
    expected_split_counts = {
        "train": 24 * 9,
        "calibration": 12 * 9,
        "internal_test": 16 * 9,
    }
    expected_total_users = 24 + 12 + 16
    expected_total_states = sum(expected_split_counts.values())
    data_generator_endpoint = endpoint_from_config(
        experiment_config, str(data_generation_cfg["generator_endpoint"])
    )
    expected_generation_compatibility = build_generation_compatibility_contract(
        project_root=ROOT,
        experiment_config_path=args.experiment_config,
        pm_v2_config_path=args.pm_v2_config,
        seed_dialogues_path=args.seed_dialogues,
        endpoint=data_generator_endpoint,
        base_generation_seed=int(data_generation_cfg["base_seed"]),
        full_user_count=expected_total_users,
        input_token_safety_factor=float(
            api_cost_planning["input_token_safety_factor"]
        ),
        fail_on_reported_input_overrun=bool(
            api_cost_planning["fail_on_reported_input_overrun"]
        ),
        input_usd_per_mtok=generator_pricing["input"],
        output_usd_per_mtok=generator_pricing["output"],
    )
    generation_pilot_verification = require_generation_compatibility_attestation(
        args.generation_pilot_attestation,
        expected_contract=expected_generation_compatibility,
    )
    generation_pilot_semantic_verification = (
        require_generation_semantic_review_v8(
            args.generation_pilot_semantic_attestation,
            require_validation=True,
        )
    )
    if data_report.get("generation_pilot_semantic_review") != (
        generation_pilot_semantic_verification
    ):
        raise RuntimeError(
            "PM-v2 data report semantic-review lineage differs from the freeze"
        )
    generation_estimate = json.loads(
        args.generation_cost_estimate.read_text(encoding="utf-8")
    )
    generation_estimate_payload = {
        key: value
        for key, value in generation_estimate.items()
        if key != "cost_estimate_sha256"
    }
    accepted_generation_cost_hash = sha256_text(
        canonical_json(generation_estimate_payload)
    )
    generation_call_plan = list(iter_jsonl(args.generation_call_plan))
    if (
        generation_estimate.get("cost_estimate_sha256")
        != accepted_generation_cost_hash
        or (generation_estimate.get("budget_gate") or {}).get("status") != "PASS"
        or generation_estimate.get("call_plan_sha256")
        != sha256_text(canonical_json(generation_call_plan))
        or generation_estimate.get("generation_run_binding")
        != data_report.get("generation_run_binding")
        or data_report.get("accepted_cost_estimate_sha256")
        != accepted_generation_cost_hash
        or generation_estimate.get("pricing")
        != {
            "input_usd_per_mtok": generator_pricing["input"],
            "output_usd_per_mtok": generator_pricing["output"],
        }
    ):
        raise RuntimeError(
            "full synthetic generation cost plan/accepted hash is stale"
        )
    generation_attempt_plans = {
        str(attempt["call_key"]): attempt
        for row in generation_call_plan
        for attempt in row.get("attempts") or []
    }
    if not generation_attempt_plans or len(generation_attempt_plans) != sum(
        len(row.get("attempts") or []) for row in generation_call_plan
    ):
        raise RuntimeError("full synthetic generation call plan has duplicate keys")
    generation_ledger = PersistentAttemptLedger(
        args.generation_attempt_ledger,
        stage="pm_v2_synthetic_bundle_generation",
        expected_calls={key: 1 for key in generation_attempt_plans},
        maximum_total_attempts=int(generation_estimate["maximum_api_calls"]),
    )
    generation_successes = 0
    for call_key in sorted(generation_ledger.started_call_keys):
        terminal = generation_ledger.terminal_row(call_key)
        if terminal is None:
            raise RuntimeError(
                "full synthetic generation ledger contains an unknown STARTED attempt"
            )
        if terminal.get("event") != "SUCCEEDED":
            continue
        generation_successes += 1
        attempt_plan = generation_attempt_plans[call_key]
        usage_error = reported_prompt_token_error(
            terminal.get("usage"),
            maximum_prompt_tokens=int(attempt_plan["estimated_input_tokens"]),
            stage="persisted full synthetic generation",
            require_positive=True,
        )
        if (
            usage_error is not None
            or not terminal.get("request_hash")
            or not isinstance((terminal.get("result") or {}).get("bundle"), dict)
        ):
            raise RuntimeError(
                "full synthetic generation success lacks auditable usage/result: "
                f"{call_key}: {usage_error}"
            )
    if (
        generation_successes != expected_total_users
        or generation_ledger.started_attempts
        != int(data_report.get("generation_total_physical_api_attempts", -1))
        or sha256_file(args.generation_attempt_ledger)
        != data_report.get("generation_attempt_ledger_sha256")
    ):
        raise RuntimeError(
            "full synthetic generation physical ledger does not match data report"
        )
    generation_run_binding = data_report.get("generation_run_binding") or {}
    expected_generation_binding_values = {
        "protocol": "pm_v2_generation_resume_binding_v2_attempt_ledger",
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "api_cost_planning": api_cost_planning,
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "seed_dialogues_sha256": sha256_file(args.seed_dialogues),
        "generation_compatibility_contract_version": (
            expected_generation_compatibility["version"]
        ),
        "generation_compatibility_contract_sha256": (
            expected_generation_compatibility["contract_sha256"]
        ),
    }
    for key, expected in expected_generation_binding_values.items():
        if generation_run_binding.get(key) != expected:
            raise RuntimeError(f"PM-v2 data generation binding mismatch: {key}")
    if data_report.get("generation_run_binding_sha256") != sha256_text(
        canonical_json(generation_run_binding)
    ):
        raise RuntimeError("PM-v2 data generation binding self-hash mismatch")
    if data_report.get("physical_attempt_ledger_protocol") != (
        PHYSICAL_ATTEMPT_LEDGER_PROTOCOL
    ):
        raise RuntimeError("PM-v2 data report uses a stale attempt-ledger protocol")
    expected_full_state_design = {
        "status": "PASS",
        "cases_per_user": 9,
        "expected_split_state_counts": expected_split_counts,
        "expected_total_states": expected_total_states,
        "unique_normalized_current_user_texts": expected_total_states,
        "normalized_current_user_text_unique_rate": 1.0,
        "semantic_family_union_counts": {
            "train": 14,
            "calibration": 5,
            "internal_test": 5,
        },
    }
    if (
        int(data_report.get("n_users", -1)) != expected_total_users
        or int(data_report.get("n_states", -1)) != expected_total_states
        or data_report.get("split_counts") != expected_split_counts
        or data_report.get("full_state_design") != expected_full_state_design
    ):
        raise RuntimeError(
            "PM-v2 data report does not match the preregistered 24/12/16 x 9 "
            "full-state design"
        )
    data_near_duplicate_audit = (
        data_report.get("cross_split_near_duplicate_audit") or {}
    )
    if data_near_duplicate_audit.get("status") != "PASS":
        raise RuntimeError("data report cross-split near-duplicate audit did not PASS")
    for key, expected in expected_near_duplicate_contract.items():
        if data_near_duplicate_audit.get(key) != expected:
            raise RuntimeError(
                f"data report near-duplicate audit contract mismatch: {key}"
            )
    if (
        int(data_near_duplicate_audit.get("state_count", -1))
        != expected_total_states
        or int(cross_split_near_duplicate_audit.get("state_count", -1))
        != expected_total_states
    ):
        raise RuntimeError("near-duplicate audits do not cover the full state design")
    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(args.strategy_bank)
    ]
    expected_strategy_catalog = {
        "count": len(strategy_cards),
        "estimated_action_tokens": strategy_action_tokens,
        "top_k": int(frozen_retrieval["strategy_top_k"]),
        "strategy_bank_sha256": sha256_file(args.strategy_bank),
    }
    if not strategy_cards or data_report.get("strategy_catalog") != expected_strategy_catalog:
        raise RuntimeError(
            "PM-v2 data report strategy catalog does not match the frozen real bank"
        )

    sweep_summary = json.loads(args.sweep_summary.read_text(encoding="utf-8"))
    if sweep_summary.get("status") != "COMPLETE":
        raise RuntimeError("PM-v2 sweep summary is not COMPLETE")
    require_artifact_attestation(
        args.sweep_attestation,
        required_stage="action_sweep",
        required_output_paths={
            "action_outcomes": args.sweep_outcomes,
            "raw_calls": args.sweep_raw,
            "summary": args.sweep_summary,
        },
    )
    sweep_manifest = json.loads(args.sweep_manifest.read_text(encoding="utf-8"))
    if sweep_manifest.get("runtime_sha256") != sha256_file(args.runtime):
        raise RuntimeError("sweep manifest runtime hash mismatch")
    if sweep_manifest.get("backend_sha256") != sha256_file(args.backend):
        raise RuntimeError("sweep manifest backend hash mismatch")
    sweep_retrieval = {
        "strategy_top_k": int(sweep_manifest["strategy_top_k"]),
        "memory_min_score": float(sweep_manifest["memory_min_score"]),
        "strategy_min_score": float(sweep_manifest["strategy_min_score"]),
    }
    if sweep_retrieval != frozen_retrieval:
        raise RuntimeError(
            "sweep retrieval settings do not exactly match the frozen PM-v2 YAML"
        )
    expected_sweep_binding_values = {
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "pm_v2_version": str(pm_v2_config["version"]),
        "supporter_generation_treatment": supporter_generation_treatment,
        "supporter_generation_treatment_sha256": (
            supporter_generation_treatment_sha256
        ),
        "development_sweep": development_sweep_cfg,
        "retrieval": dict(pm_v2_config["retrieval"]),
        "evidence_filter": frozen_evidence_filter,
        "evidence_filter_config_sha256": frozen_evidence_filter_sha256,
        "evidence_filter_model": evidence_filter_model_binding,
        "semantic_sanity": semantic_sanity,
        "runtime_state_lineage": runtime_state_lineage,
        "api_cost_planning": api_cost_planning,
        "scope": "full",
        "development_pilot_gate": development_pilot_gate,
    }
    sweep_attestation = json.loads(args.sweep_attestation.read_text(encoding="utf-8"))
    sweep_contract_sources = {
        "manifest": sweep_manifest.get("contract_bindings"),
        "summary": sweep_summary.get("contract_bindings"),
        "attestation": (sweep_attestation.get("parameters") or {}).get(
            "contract_bindings"
        ),
    }
    accepted_sweep_cost_hashes: set[str] = set()
    for source_name, raw_bindings in sweep_contract_sources.items():
        if not isinstance(raw_bindings, dict):
            raise RuntimeError(f"sweep {source_name} lacks contract_bindings")
        for key, expected in expected_sweep_binding_values.items():
            if raw_bindings.get(key) != expected:
                raise RuntimeError(
                    f"sweep {source_name} contract binding mismatch: {key}"
                )
        accepted_hash = str(raw_bindings.get("accepted_cost_estimate_sha256") or "")
        if len(accepted_hash) != 64:
            raise RuntimeError(
                f"sweep {source_name} lacks the accepted cost-estimate hash"
            )
        accepted_sweep_cost_hashes.add(accepted_hash)
    if len(accepted_sweep_cost_hashes) != 1:
        raise RuntimeError("sweep artifacts disagree on accepted cost-estimate hash")
    expected_sweep_runtime = {
        "endpoint_model": sweep_generator_endpoint.model,
        "endpoint_base_url": sweep_generator_endpoint.base_url,
        "temperature": supporter_generation_contract.temperature,
        "max_tokens": supporter_generation_contract.max_output_tokens,
        "seed": int(development_sweep_cfg["seed"]),
        "request_retries": 1,
        "fail_fast": True,
        **frozen_retrieval,
        "evidence_filter": frozen_evidence_filter,
        "evidence_filter_config_sha256": frozen_evidence_filter_sha256,
        "evidence_filter_model": {
            "checkpoint_sha256": evidence_filter_model_binding[
                "checkpoint_sha256"
            ],
            "model_contract_sha256": evidence_filter_model_binding[
                "model_contract_sha256"
            ],
        },
        "action_filter": None,
        "supporter_generation_treatment": supporter_generation_treatment,
        "supporter_generation_treatment_sha256": (
            supporter_generation_treatment_sha256
        ),
    }
    for key, expected in expected_sweep_runtime.items():
        if sweep_manifest.get(key) != expected:
            raise RuntimeError(f"sweep manifest runtime mismatch: {key}")
    sweep_attestation_parameters = sweep_attestation.get("parameters") or {}
    expected_sweep_attestation = {
        **expected_sweep_runtime,
        "endpoint_family": sweep_generator_endpoint.family,
        "max_cards": None,
    }
    for key, expected in expected_sweep_attestation.items():
        if sweep_attestation_parameters.get(key) != expected:
            raise RuntimeError(f"sweep attestation runtime mismatch: {key}")
    if (
        sweep_summary.get("supporter_generation_treatment")
        != supporter_generation_treatment
        or sweep_summary.get("supporter_generation_treatment_sha256")
        != supporter_generation_treatment_sha256
        or sweep_summary.get("aborted_on_completion_gate") is not False
        or sweep_summary.get("failures")
    ):
        raise RuntimeError(
            "sweep summary does not prove one complete PM-v2.2 generation treatment"
        )
    sweep_outcome_rows = list(iter_jsonl(args.sweep_outcomes))
    for row in sweep_outcome_rows:
        provenance = row.get("provenance") or {}
        if (
            provenance.get("supporter_generation_treatment")
            != supporter_generation_treatment
            or provenance.get("supporter_generation_treatment_sha256")
            != supporter_generation_treatment_sha256
            or provenance.get("normalized_finish_reason") != "complete"
        ):
            raise RuntimeError(
                "sweep outcome lacks the exact PM-v2.2 treatment or complete finish"
            )
    for raw_row in iter_jsonl(args.sweep_raw):
        if (
            raw_row.get("normalized_finish_reason") != "complete"
            or raw_row.get("supporter_generation_treatment_sha256")
            != supporter_generation_treatment_sha256
        ):
            raise RuntimeError(
                "sweep raw generation log contains a non-complete or mixed-treatment call"
            )

    state_models = load_states(args.states)
    states = [state.model_dump(mode="json") for state in state_models]
    if any(
        int(state["strategy_catalog_count"]) != len(strategy_cards)
        or int(state["strategy_estimated_tokens"]) != strategy_action_tokens
        for state in states
    ):
        raise RuntimeError("PM-v2 states do not bind the frozen real strategy catalog")
    state_ids = {str(row["state_id"]) for row in states}
    card_ids = {str(row["card_id"]) for row in states}
    if len(state_ids) != len(states) or len(card_ids) != len(states):
        raise RuntimeError("PM-v2 states contain duplicate state_id or card_id")
    expected_action_keys = {
        (str(row["card_id"]), str(action_id))
        for row in states
        for action_id in row["allowed_actions"]
    }
    runtime_card_ids = {str(row["card_id"]) for row in iter_jsonl(args.runtime)}
    backend_card_ids = {str(row["card_id"]) for row in iter_jsonl(args.backend)}
    evaluator_index = load_evaluator_context_index(
        args.evaluator_contexts, states=state_models, require_exact=True
    )
    if judge_manifest.get("evaluator_contexts_sha256") != evaluator_index.source_sha256:
        raise RuntimeError("judge manifest evaluator-context source hash mismatch")
    if judge_manifest.get("evaluator_context_map_sha256") != evaluator_index.map_sha256:
        raise RuntimeError("judge manifest evaluator-context map hash mismatch")
    evaluator_state_ids = set(evaluator_index.by_state)
    evaluator_card_ids = set(evaluator_index.by_card)
    if (
        runtime_card_ids != card_ids
        or backend_card_ids != card_ids
        or evaluator_state_ids != state_ids
        or evaluator_card_ids != card_ids
        or len(evaluator_index.by_state) != len(states)
    ):
        raise RuntimeError("states/runtime/backend/evaluator-context lineage mismatch")
    outcome_keys = {
        (str(row["card_id"]), str(row["action_id"]))
        for row in sweep_outcome_rows
    }
    if outcome_keys != expected_action_keys:
        raise RuntimeError("sweep outcomes do not exactly cover state/action lineage")
    state_by_card = {str(row["card_id"]): str(row["state_id"]) for row in states}
    expected_label_keys = {
        (state_by_card[card_id], action_id) for card_id, action_id in expected_action_keys
    }
    label_rows = list(iter_jsonl(args.labels))
    label_models = [ActionLabel.model_validate(row) for row in label_rows]
    label_keys = {
        (label.state_id, label.action_id) for label in label_models
    }
    if label_keys != expected_label_keys:
        raise RuntimeError("action labels do not exactly cover state/action lineage")
    if any(
        label.composite_weights_sha256 != composite_weights_sha256
        for label in label_models
    ):
        raise RuntimeError("action labels do not bind the frozen composite weights")
    judge_raw_rows = list(iter_jsonl(args.judge_raw))
    expected_judge_rows = len(expected_label_keys) * int(labeling["minimum_families"])
    if len(judge_raw_rows) < expected_judge_rows:
        raise RuntimeError("judge raw rows are incomplete for the frozen label matrix")
    if int(data_report.get("n_states", -1)) != len(states):
        raise RuntimeError("data report state count mismatch")
    if sum(1 for _ in iter_jsonl(args.bundles)) != int(data_report.get("n_users", -1)):
        raise RuntimeError("bundle count does not match data report")
    if len(seed_dialogue_rows) != int(
        seed_audit.get("selected_unique_train_seeds", -1)
    ):
        raise RuntimeError("seed dialogue count does not match lineage audit")

    completed_hashes = human_audit.get("completed_inputs_sha256") or {}
    if not completed_hashes:
        raise RuntimeError("human audit does not bind completed annotator files")
    human_completed_paths = [Path(value).resolve() for value in completed_hashes]
    for path in human_completed_paths:
        if not path.is_file() or sha256_file(path) != completed_hashes[str(path)]:
            raise RuntimeError(f"human completed rating hash mismatch: {path}")
    fixed_report = json.loads(args.fixed_baseline_report.read_text(encoding="utf-8"))
    if fixed_report.get("status") != "COMPLETE":
        raise RuntimeError("fixed baseline checkpoint report is not COMPLETE")
    if fixed_report.get("source_checkpoint_sha256") != sha256_file(args.checkpoint):
        raise RuntimeError("fixed baselines were not derived from the frozen PM-v2 checkpoint")
    expected_fixed = {
        "cost_matched_fixed": args.cost_matched_checkpoint,
        "event_memory_r0": args.me_r0_checkpoint,
    }
    for name, path in expected_fixed.items():
        expected_sha = fixed_report["baselines"][name]["checkpoint_sha256"]
        if expected_sha != sha256_file(path):
            raise RuntimeError(f"fixed baseline checkpoint hash mismatch: {name}")

    generation_contract = {
        "supporter_generation_treatment": supporter_generation_treatment,
        "supporter_generation_treatment_sha256": (
            supporter_generation_treatment_sha256
        ),
        "fixed_seeker_generation_treatment": fixed_seeker_generation_treatment,
        "fixed_seeker_generation_treatment_sha256": (
            fixed_seeker_generation_treatment_sha256
        ),
        "fixed_seeker_cost_planning": fixed_seeker_cost_planning,
        "fixed_seeker_cost_planning_sha256": (
            fixed_seeker_cost_planning_sha256
        ),
        "policy_checkpoint_sha256": sha256_file(args.checkpoint),
        "policy_training_report_sha256": sha256_file(args.training_report),
        "policy_lock_timing": POLICY_LOCK_TIMING,
        "post_generation_policy_tuning_prohibited": (
            POST_GENERATION_POLICY_TUNING_PROHIBITED
        ),
        "generator_endpoint": {
            "model": generator_endpoint.model,
            "family": generator_endpoint.family,
            "base_url": generator_endpoint.base_url,
        },
        "generator_endpoint_sha256": sha256_text(
            canonical_json(
                {
                    "model": generator_endpoint.model,
                    "family": generator_endpoint.family,
                    "base_url": generator_endpoint.base_url,
                }
            )
        ),
        "simulator_id": simulator_id,
        "max_turns": max_turns,
        "seeds": generation_seeds,
        "evaluation_unit_contract": evaluation_unit_contract,
        "paid_generation_scope": "frozen_evaluation_turns_only",
        "all_turn_action_preflight_is_diagnostic_only": True,
        "strategy_action_tokens": strategy_action_tokens,
        **frozen_retrieval,
        "evidence_filter": frozen_evidence_filter,
        "evidence_filter_config_sha256": frozen_evidence_filter_sha256,
        "evidence_filter_model": evidence_filter_model_binding,
        "action_preflight_gates": action_preflight_gates,
        "action_preflight_metric_scopes": ACTION_PREFLIGHT_METRIC_SCOPES,
        "maximum_cost_matched_relative_deviation": (
            maximum_cost_matched_relative_deviation
        ),
        "generator_retries": 1,
        "generator_pricing_usd_per_mtok": sweep_generator_pricing,
        "api_cost_planning": api_cost_planning,
        "input_token_safety_factor": float(
            api_cost_planning["input_token_safety_factor"]
        ),
        "fail_on_reported_input_overrun": bool(
            api_cost_planning["fail_on_reported_input_overrun"]
        ),
    }
    baseline_condition_map = {
        "M0+R0": "no_memory_r0",
        "ME+R0": "pm_v2_me_r0_fixed",
        "cost_matched_fixed": "pm_v2_cost_matched_fixed",
        "structured_high_resource": "best_fixed",
        "raw_session_top_k": "session_rag_rs",
        "full_history": "full_history_rs",
    }
    frozen_required_baselines = list(
        pm_v2_config["external_evaluation"]["required_baselines"]
    )
    if set(frozen_required_baselines) - set(baseline_condition_map):
        raise ValueError("PM-v2 required baseline lacks a frozen condition mapping")
    reference_baseline_conditions = {
        baseline_condition_map[name]
        for name in (
            "M0+R0",
            "structured_high_resource",
            "raw_session_top_k",
            "full_history",
        )
    }
    if reference_baseline_conditions != set(REFERENCE_BASELINE_CONDITIONS):
        raise RuntimeError("reference-baseline condition contract changed")
    reference_evidence_processing_contracts = (
        build_reference_evidence_processing_contracts(
            evidence_filter_config=evidence_filter_config,
            evidence_filter_model_binding=evidence_filter_model_binding,
            memory_min_score=float(frozen_retrieval["memory_min_score"]),
            strategy_min_score=float(frozen_retrieval["strategy_min_score"]),
            strategy_top_k=int(frozen_retrieval["strategy_top_k"]),
            session_rag_top_k=int(
                (experiment_config.get("protocol") or {})[
                    "evoemo_session_rag_top_k"
                ]
            ),
        )
    )
    reference_evidence_processing_contracts_sha256 = (
        evidence_processing_contracts_sha256(
            reference_evidence_processing_contracts
        )
    )
    reference_baseline_shared_contract = require_pmv22_reference_baseline_bundle(
        attestation_path=args.external_baseline_attestation,
        turns_path=args.external_baseline_turns,
        manifest_path=args.external_baseline_manifest,
        summary_path=args.external_baseline_summary,
        evoemo_path=args.evoemo,
        strategy_bank_path=args.strategy_bank,
        policy_checkpoint_path=args.checkpoint,
        policy_training_report_path=args.training_report,
        pm_v2_config_path=args.pm_v2_config,
        evidence_filter_checkpoint_path=args.evidence_filter_checkpoint,
        evidence_filter_report_path=args.evidence_filter_report,
        evidence_filter_attestation_path=args.evidence_filter_attestation,
        strategy_bank_approval_path=args.strategy_bank_approval,
        fixed_tracks_path=args.fixed_tracks,
        fixed_tracks_attestation_path=args.fixed_tracks_attestation,
        conditions=sorted(reference_baseline_conditions),
        expected_units=external_expected_units,
        treatment=supporter_generation_treatment,
        treatment_sha256=supporter_generation_treatment_sha256,
        fixed_seeker_treatment=fixed_seeker_generation_treatment,
        fixed_seeker_treatment_sha256=(
            fixed_seeker_generation_treatment_sha256
        ),
        evidence_processing_contracts=(
            reference_evidence_processing_contracts
        ),
        evidence_processing_contracts_sha256=(
            reference_evidence_processing_contracts_sha256
        ),
        evidence_filter_config_sha256=frozen_evidence_filter_sha256,
        evidence_filter_model_binding=evidence_filter_model_binding,
        strategy_bank_approval=strategy_bank_human_approval,
        generator_endpoint={
            "model": generator_endpoint.model,
            "family": str(generator_endpoint.family or ""),
            "base_url": generator_endpoint.base_url,
        },
        simulator_id=simulator_id,
        max_turns=max_turns,
        seeds=generation_seeds,
        evaluation_unit_contract=evaluation_unit_contract,
    )
    fixed_external_baseline_artifacts = {
        condition: {
            "condition": condition,
            **reference_baseline_shared_contract,
        }
        for condition in sorted(reference_baseline_conditions)
    }
    reference_baseline_dir = args.external_baseline_turns.resolve().parent
    reference_baseline_support_paths = [
        reference_baseline_dir / "raw_api_calls.jsonl",
        reference_baseline_dir / "physical_attempt_ledger.jsonl",
        reference_baseline_dir / "call_plan.jsonl",
        reference_baseline_dir / "cost_estimate.json",
    ]
    generation_pilot_dir = args.generation_pilot_attestation.resolve().parent
    generation_pilot_paths = [
        args.generation_pilot_attestation,
        generation_pilot_dir / "cost_estimate.json",
        generation_pilot_dir / "call_plan.jsonl",
        generation_pilot_dir / "physical_attempt_ledger.jsonl",
        generation_pilot_dir / "pilot_bundle.json",
        generation_pilot_dir / "summary.json",
        generation_pilot_dir / "run_manifest.json",
    ]
    required = [
        args.experiment_config,
        args.pm_v2_config,
        args.checkpoint,
        args.evidence_filter_checkpoint,
        args.evidence_filter_report,
        args.evidence_filter_attestation,
        args.cost_matched_checkpoint,
        args.me_r0_checkpoint,
        args.fixed_baseline_report,
        args.training_report,
        args.states,
        args.seed_dialogues,
        args.data_report,
        *generation_pilot_paths,
        args.generation_cost_estimate,
        args.generation_call_plan,
        args.generation_attempt_ledger,
        args.bundles,
        args.runtime,
        args.backend,
        args.evaluator_contexts,
        args.seed_audit,
        *seed_source_paths,
        args.sweep_outcomes,
        args.sweep_raw,
        args.sweep_summary,
        args.sweep_manifest,
        args.sweep_attestation,
        args.labels,
        args.judge_raw,
        args.judge_manifest,
        args.judge_summary,
        args.judge_attestation,
        args.judge_pilot_plan,
        args.pilot_sweep_summary,
        args.pilot_sweep_attestation,
        args.judge_compatibility_manifest,
        args.judge_compatibility_summary,
        args.judge_compatibility_attestation,
        *compatibility_input_paths,
        *compatibility_output_paths,
        *judge_attested_input_paths,
        *judge_attested_output_paths,
        args.data_label_audit,
        args.semantic_sanity_report,
        args.semantic_sanity_attestation,
        *semantic_sanity_input_paths,
        args.human_audit,
        args.human_key,
        args.human_manual,
        args.human_sample_plan,
        *human_completed_paths,
        args.evoemo,
        args.strategy_bank,
        args.strategy_bank_approval,
        strategy_split_manifest,
        strategy_bank_audit,
        strategy_rag_audit_summary,
        strategy_candidate_manifest,
        args.fixed_tracks,
        args.fixed_tracks_attestation,
        fixed_bundle_dir / "run_manifest.json",
        fixed_bundle_dir / "summary.json",
        *fixed_bundle_support_paths,
        args.external_baseline_turns,
        args.external_baseline_attestation,
        args.external_baseline_manifest,
        args.external_baseline_summary,
        *reference_baseline_support_paths,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    frozen = create_study_freeze(
        release_root=ROOT,
        config_path=args.experiment_config,
        checkpoint_paths=[
            args.checkpoint,
            args.cost_matched_checkpoint,
            args.me_r0_checkpoint,
            args.evidence_filter_checkpoint,
        ],
        data_paths=[
            args.pm_v2_config,
            args.evidence_filter_report,
            args.evidence_filter_attestation,
            args.fixed_baseline_report,
            args.training_report,
            args.states,
            args.seed_dialogues,
            args.data_report,
            *generation_pilot_paths,
            args.generation_cost_estimate,
            args.generation_call_plan,
            args.generation_attempt_ledger,
            args.bundles,
            args.runtime,
            args.backend,
            args.evaluator_contexts,
            args.seed_audit,
            *seed_source_paths,
            args.sweep_outcomes,
            args.sweep_raw,
            args.sweep_summary,
            args.sweep_manifest,
            args.sweep_attestation,
            args.labels,
            args.judge_raw,
            args.judge_manifest,
            args.judge_summary,
            args.judge_attestation,
            args.judge_pilot_plan,
            args.pilot_sweep_summary,
            args.pilot_sweep_attestation,
            args.judge_compatibility_manifest,
            args.judge_compatibility_summary,
            args.judge_compatibility_attestation,
            *compatibility_input_paths,
            *compatibility_output_paths,
            *judge_attested_input_paths,
            *judge_attested_output_paths,
            args.data_label_audit,
            args.semantic_sanity_report,
            args.semantic_sanity_attestation,
            *semantic_sanity_input_paths,
            args.human_audit,
            args.human_key,
            args.human_manual,
            args.human_sample_plan,
            *human_completed_paths,
            args.evoemo,
            args.strategy_bank,
            args.strategy_bank_approval,
            strategy_split_manifest,
            strategy_bank_audit,
            strategy_rag_audit_summary,
            strategy_candidate_manifest,
            args.fixed_tracks,
            args.fixed_tracks_attestation,
            fixed_bundle_dir / "run_manifest.json",
            fixed_bundle_dir / "summary.json",
            *fixed_bundle_support_paths,
            args.external_baseline_turns,
            args.external_baseline_attestation,
            args.external_baseline_manifest,
            args.external_baseline_summary,
            *reference_baseline_support_paths,
        ],
        prompt_files=[args.pm_v2_config, args.judge_manifest],
        out_path=args.out,
        notes={
            "pm_version": model.format_version,
            "selection_config_hash": model.selection_config.digest(),
            "feature_config_hash": model.feature_builder.config_hash(),
            "ood_calibration": ood_calibration,
            "judge_prompt_contract_hash": prompt_contract_hash(),
            "quality_composite_version": model.selection_config.composite_spec.version,
            "quality_composite": composite_spec.model_dump(mode="json"),
            "quality_composite_sha256": sha256_text(
                canonical_json(composite_spec.model_dump(mode="json"))
            ),
            "quality_composite_weights_sha256": sha256_text(
                canonical_json(composite_spec.weights)
            ),
            "labeling_gates": labeling,
            "api_cost_planning": api_cost_planning,
            "evidence_filter_model": evidence_filter_model_binding,
            "data_generation_contract": {
                "generator_endpoint": str(
                    data_generation_cfg["generator_endpoint"]
                ),
                "generator_pricing_usd_per_mtok": generator_pricing,
                "api_cost_planning": api_cost_planning,
                "generation_compatibility_contract_sha256": (
                    expected_generation_compatibility["contract_sha256"]
                ),
                "generation_run_binding_sha256": data_report[
                    "generation_run_binding_sha256"
                ],
            },
            "development_sweep_contract": development_sweep_cfg,
            "generation_compatibility_pilot_artifacts": {
                **generation_pilot_verification,
                "attestation_file_sha256": sha256_file(
                    args.generation_pilot_attestation
                ),
            },
            "generation_pilot_semantic_review_artifacts": {
                **generation_pilot_semantic_verification,
                "attestation_file_sha256": sha256_file(
                    args.generation_pilot_semantic_attestation
                ),
            },
            "full_generation_cost_and_ledger": {
                "accepted_cost_estimate_sha256": accepted_generation_cost_hash,
                "cost_estimate_file_sha256": sha256_file(
                    args.generation_cost_estimate
                ),
                "call_plan_file_sha256": sha256_file(args.generation_call_plan),
                "physical_attempt_ledger_sha256": sha256_file(
                    args.generation_attempt_ledger
                ),
                "physical_attempts": generation_ledger.started_attempts,
                "successful_users": generation_successes,
                "physical_attempt_ledger_protocol": (
                    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL
                ),
            },
            "development_pilot_gate": development_pilot_gate,
            "development_judge_compatibility": {
                "pilot_plan_sha256": pilot_plan_sha256,
                "pilot_expected_keys_sha256": pilot_expected_keys_sha256,
                "attestation_sha256": compatibility_attestation_sha256,
                "judge_endpoints": expected_development_judge_contracts,
                "pricing_usd_per_mtok": development_judge_pricing,
                "api_cost_planning": api_cost_planning,
                "gate": compatibility_gate,
            },
            "development_judge_attestation_sha256": str(
                judge_attestation_verification["attestation_sha256"]
            ),
            "evaluator_context_map_sha256": evaluator_index.map_sha256,
            "strategy_catalog": expected_strategy_catalog,
            "internal_reportability_checks": report["reportability_checks"],
            "learned_routing_advantage_verified": True,
            "require_learned_routing_advantage_before_external": True,
            "preregistered_data_design": {
                **frozen_user_design,
                "split_counts": expected_split_counts,
                "total_users": expected_total_users,
                "total_states": expected_total_states,
                "full_state_design": expected_full_state_design,
            },
            "cross_split_near_duplicate_audit": {
                "report": cross_split_near_duplicate_audit,
                "report_sha256": cross_split_near_duplicate_audit_sha256,
                "config": near_duplicate_cfg,
            },
            "semantic_sanity": {
                **semantic_sanity,
                "attested_input_sha256": {
                    str(path): sha256_file(path)
                    for path in semantic_sanity_input_paths
                },
            },
            "seed_extraction_contract": {
                "version": seed_contract_version,
                "code_manifest": seed_code_manifest,
                "code_manifest_sha256": seed_code_manifest_sha256,
                "lineage_manifest_hashes": seed_lineage_hashes,
            },
            "human_label_audit_checks": human_audit.get("checks"),
            "human_label_audit_thresholds": human_audit.get("thresholds"),
            "human_label_audit_artifacts": {
                "manual_version": manual_version,
                "manual_sha256": sha256_file(args.human_manual),
                "sample_plan": expected_sample_wrapper,
                "llm_rubric_contract": expected_rubric_contract,
            },
            "generation_contract": generation_contract,
            "strategy_bank_human_approval": strategy_bank_human_approval,
            "fixed_tracks_content_attestation": {
                "stage": FIXED_SEEKER_V22_STAGE,
                "fixed_seeker_generation_treatment": (
                    fixed_seeker_generation_treatment
                ),
                "fixed_seeker_generation_treatment_sha256": (
                    fixed_seeker_generation_treatment_sha256
                ),
                "fixed_seeker_cost_planning": fixed_seeker_cost_planning,
                "fixed_seeker_cost_planning_sha256": (
                    fixed_seeker_cost_planning_sha256
                ),
                "planned_budget_gate": planned_fixed_budget_gate,
                "observed_budget_gate": observed_fixed_budget_gate,
                "attestation_sha256": fixed_tracks_verification[
                    "attestation_sha256"
                ],
                "relocated_records": fixed_tracks_verification[
                    "relocated_records"
                ],
            },
            "fixed_external_baseline_artifacts": fixed_external_baseline_artifacts,
            "external_evaluation_contract": {
                "external_estimand": (
                    "quality_risk_observed_cost_componentwise_pareto_v1"
                ),
                "risk_composite_basis": "requested_action_applicable_fields",
                "single_external_utility_claim_allowed": False,
                "primary_bootstrap_cluster": primary_bootstrap_cluster,
                "sensitivity_bootstrap_cluster": sensitivity_bootstrap_cluster,
                "judge_pricing_usd_per_mtok": external_judge_pricing,
                "api_cost_planning": api_cost_planning,
                "judge_endpoints": [
                    {
                        "name": name,
                        "model": endpoint.model,
                        "family": endpoint.family,
                        "base_url": endpoint.base_url,
                        "sha256": sha256_text(
                            canonical_json(
                                {
                                    "model": endpoint.model,
                                    "family": endpoint.family,
                                    "base_url": endpoint.base_url,
                                }
                            )
                        ),
                    }
                    for name, endpoint in zip(external_judge_names, external_judges)
                ],
                "turn_indices": turn_indices,
                "evaluation_unit_contract": evaluation_unit_contract,
                "treatment": "pm_v2",
                "judge_seed": external_judge_seed,
                "required_baselines": list(
                    frozen_required_baselines
                ),
                "baseline_condition_map": baseline_condition_map,
                "fixed_baseline_artifacts": fixed_external_baseline_artifacts,
                "conditions": [
                    "pm_v2",
                    *[baseline_condition_map[name] for name in frozen_required_baselines],
                ],
                "primary_resource_metric": pm_v2_config["external_evaluation"][
                    "primary_resource_metric"
                ],
                "maximum_cost_matched_relative_deviation": (
                    maximum_cost_matched_relative_deviation
                ),
                **external_low_mad_thresholds,
                **composite_support_gates,
                "require_raw_per_family_health": True,
                "require_forced_swap_or_human_check_for_key_claims": bool(
                    pm_v2_config["external_evaluation"][
                        "require_forced_swap_or_human_check_for_key_claims"
                    ]
                ),
                "pointwise_schema_smoke": pointwise_schema_smoke_contract,
                "forced_swap": {
                    "judge_pricing_usd_per_mtok": external_judge_pricing,
                    "api_cost_planning": api_cost_planning,
                    "treatment": "pm_v2",
                    "baseline": "pm_v2_cost_matched_fixed",
                    "sample_units": int(forced_swap_cfg["sample_units"]),
                    "order_variants": forced_swap_order_variants,
                    "judge_seed": int(forced_swap_cfg["judge_seed"]),
                    "estimated_output_tokens_per_call": int(
                        forced_swap_cfg["estimated_output_tokens_per_call"]
                    ),
                    "exclude_sample_from_full_evaluation": True,
                    "maximum_order_disagreement_rate": float(
                        forced_swap_cfg["maximum_order_disagreement_rate"]
                    ),
                    "minimum_schema_success_rate": float(
                        forced_swap_cfg["minimum_schema_success_rate"]
                    ),
                    "minimum_cross_family_support_delta_correlation": float(
                        forced_swap_cfg[
                            "minimum_cross_family_support_delta_correlation"
                        ]
                    ),
                    "require_cross_family_direction_agreement": bool(
                        forced_swap_cfg["require_cross_family_direction_agreement"]
                    ),
                    "minimum_support_delta_ci_upper_for_continuation": float(
                        forced_swap_cfg[
                            "minimum_support_delta_ci_upper_for_continuation"
                        ]
                    ),
                    "minimum_support_delta_ci_lower_for_advantage": float(
                        forced_swap_cfg[
                            "minimum_support_delta_ci_lower_for_advantage"
                        ]
                    ),
                    "require_positive_support_delta_every_family": bool(
                        forced_swap_cfg[
                            "require_positive_support_delta_every_family"
                        ]
                    ),
                    "minimum_resolved_preference_margin": int(
                        forced_swap_cfg["minimum_resolved_preference_margin"]
                    ),
                    "maximum_cost_matched_relative_deviation": (
                        maximum_cost_matched_relative_deviation
                    ),
                    "judge_endpoints": [
                        {
                            "name": name,
                            "model": endpoint.model,
                            "family": endpoint.family,
                            "base_url": endpoint.base_url,
                            "sha256": sha256_text(
                                canonical_json(
                                    {
                                        "model": endpoint.model,
                                        "family": endpoint.family,
                                        "base_url": endpoint.base_url,
                                    }
                                )
                            ),
                        }
                        for name, endpoint in zip(
                            forced_swap_endpoint_names, forced_swap_endpoints
                        )
                    ],
                },
            },
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "fixed_baselines": fixed_report["baselines"],
            "external_use": "fixed-input EvoEmo only; no calibration on external results",
        },
    )
    print(frozen)


if __name__ == "__main__":
    main()
