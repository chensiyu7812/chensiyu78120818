from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .artifacts import require_artifact_attestation
from .config import endpoint_from_config, load_config
from .generation_contract import SupporterGenerationContract
from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text
from .pm_v2_contracts import ResourceNeedRegime
from .pm_v2_audit import audit_deployable_feature_observability
from .pm_v2_data import load_evaluator_context_index, load_states
from .pm_v2_judging import (
    composite_spec_from_config,
    composite_weights_hash,
    labeling_settings_from_config,
    prompt_contract_hash,
)
from .pm_v2_judge_schema_smoke import (
    require_development_judge_schema_smoke_pass,
)
from .pm_v2_pilot_human_spot_check import (
    require_pilot_human_spot_check_pass,
)


DEVELOPMENT_PILOT_GATE_PROTOCOL = "pm-v2-development-pilot-gate-v3-treatment-bound"


def _manifest_is_self_consistent(manifest: Mapping[str, Any]) -> bool:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return manifest.get("manifest_sha256") == sha256_text(canonical_json(payload))


def _plan_is_self_consistent(plan: Mapping[str, Any]) -> bool:
    payload = {key: value for key, value in plan.items() if key != "pilot_plan_sha256"}
    return plan.get("pilot_plan_sha256") == sha256_text(canonical_json(payload))


def _recorded_path(
    attestation: Mapping[str, Any], section: str, logical_name: str
) -> Path:
    record = (attestation.get(section) or {}).get(logical_name)
    if not isinstance(record, Mapping) or not record.get("path"):
        raise RuntimeError(
            f"development pilot attestation lacks {section}.{logical_name}"
        )
    return Path(str(record["path"])).resolve()


def _require_recorded_path(
    attestation: Mapping[str, Any],
    section: str,
    logical_name: str,
    expected: str | Path,
) -> None:
    if _recorded_path(attestation, section, logical_name) != Path(expected).resolve():
        raise RuntimeError(
            f"development pilot attestation path mismatch: {section}.{logical_name}"
        )


def _current_judge_descriptors(
    experiment_config: Mapping[str, Any], pm_v2_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    names = [
        str(value)
        for value in pm_v2_config["development_judging"]["judge_endpoints"]
    ]
    endpoints = [endpoint_from_config(experiment_config, name) for name in names]
    return [
        {
            "name": name,
            "model": endpoint.model,
            "family": endpoint.family,
            "base_url": endpoint.base_url,
        }
        for name, endpoint in zip(names, endpoints)
    ]


def require_development_pilot_gate(
    *,
    experiment_config_path: str | Path,
    pm_v2_config_path: str | Path,
    states_path: str | Path,
    runtime_path: str | Path,
    backend_path: str | Path,
    evaluator_contexts_path: str | Path,
    strategy_bank_path: str | Path,
    semantic_sanity: Mapping[str, Any],
    semantic_sanity_report_path: str | Path,
    semantic_sanity_attestation_path: str | Path,
    runtime_state_lineage: Mapping[str, Any],
    pilot_plan_path: str | Path,
    pilot_sweep_summary_path: str | Path,
    pilot_sweep_attestation_path: str | Path,
    judge_compatibility_summary_path: str | Path,
    judge_compatibility_attestation_path: str | Path,
    pilot_human_spot_check_report_path: str | Path,
    pilot_human_spot_check_attestation_path: str | Path,
    judge_schema_smoke_summary_path: str | Path,
    judge_schema_smoke_attestation_path: str | Path,
) -> dict[str, Any]:
    """Require the complete action + judge pilot chain before a full sweep.

    The returned compact hash binding is suitable for inclusion in a full
    sweep cost plan, run manifest and output attestation.
    """

    experiment_config_path = Path(experiment_config_path).resolve()
    pm_v2_config_path = Path(pm_v2_config_path).resolve()
    states_path = Path(states_path).resolve()
    runtime_path = Path(runtime_path).resolve()
    backend_path = Path(backend_path).resolve()
    evaluator_contexts_path = Path(evaluator_contexts_path).resolve()
    strategy_bank_path = Path(strategy_bank_path).resolve()
    semantic_sanity_report_path = Path(semantic_sanity_report_path).resolve()
    semantic_sanity_attestation_path = Path(
        semantic_sanity_attestation_path
    ).resolve()
    pilot_plan_path = Path(pilot_plan_path).resolve()
    pilot_sweep_summary_path = Path(pilot_sweep_summary_path).resolve()
    pilot_sweep_attestation_path = Path(pilot_sweep_attestation_path).resolve()
    judge_compatibility_summary_path = Path(
        judge_compatibility_summary_path
    ).resolve()
    judge_compatibility_attestation_path = Path(
        judge_compatibility_attestation_path
    ).resolve()
    pilot_human_spot_check_report_path = Path(
        pilot_human_spot_check_report_path
    ).resolve()
    pilot_human_spot_check_attestation_path = Path(
        pilot_human_spot_check_attestation_path
    ).resolve()
    judge_schema_smoke_summary_path = Path(
        judge_schema_smoke_summary_path
    ).resolve()
    judge_schema_smoke_attestation_path = Path(
        judge_schema_smoke_attestation_path
    ).resolve()

    experiment_config = load_config(experiment_config_path)
    pm_v2_config = load_config(pm_v2_config_path)
    if pm_v2_config.get("version") != "pm-v2.2":
        raise RuntimeError("development pilot gate requires PM-v2.2")
    supporter_generation_contract = SupporterGenerationContract.from_config(
        pm_v2_config
    )
    pilot_config = dict(
        pm_v2_config["development_judging"]["compatibility_pilot"]
    )
    expected_actions = [str(value) for value in pilot_config["actions"]]
    expected_regimes = {regime.value for regime in ResourceNeedRegime}
    expected_state_count = len(expected_regimes) * int(
        pilot_config["states_per_regime"]
    )
    expected_outcome_count = expected_state_count * len(expected_actions)

    plan = read_json(pilot_plan_path)
    if plan.get("status") != "READY" or not _plan_is_self_consistent(plan):
        raise RuntimeError("development pilot plan is absent, stale, or not READY")
    expected_plan_values = {
        "protocol": "pm_v2_development_compatibility_pilot_v2_treatment_bound",
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "states_sha256": sha256_file(states_path),
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "semantic_sanity": dict(semantic_sanity),
        "runtime_state_lineage": dict(runtime_state_lineage),
        "sample_seed": int(pilot_config["sample_seed"]),
        "states_per_regime": int(pilot_config["states_per_regime"]),
        "actions": expected_actions,
        "expected_state_count": expected_state_count,
        "expected_outcome_count": expected_outcome_count,
    }
    for key, expected in expected_plan_values.items():
        if plan.get(key) != expected:
            raise RuntimeError(f"development pilot plan lineage mismatch: {key}")
    selected_states = list(plan.get("selected_states") or [])
    if len(selected_states) != expected_state_count:
        raise RuntimeError("development pilot plan state count mismatch")
    if Counter(str(row.get("regime")) for row in selected_states) != Counter(
        {
            regime: int(pilot_config["states_per_regime"])
            for regime in expected_regimes
        }
    ):
        raise RuntimeError("development pilot plan is not regime balanced")
    if any(row.get("actions") != expected_actions for row in selected_states):
        raise RuntimeError("development pilot plan action matrix changed")
    if any(str(row.get("split")) != "train" for row in selected_states):
        raise RuntimeError(
            "development pilot must use train states only; calibration and "
            "internal-test labels are forbidden before the full sweep"
        )
    deployable_feature_observability = plan.get(
        "deployable_feature_observability"
    ) or {}
    expected_observability_settings = dict(
        pilot_config["deployable_feature_observability"]
    )
    observability_states = load_states(states_path)
    observability_contexts = load_evaluator_context_index(
        evaluator_contexts_path,
        states=observability_states,
        require_exact=True,
    )
    recomputed_observability = audit_deployable_feature_observability(
        observability_states,
        evaluator_contexts=observability_contexts,
        settings=expected_observability_settings,
    )
    observability_checks = deployable_feature_observability.get("checks") or {}
    if (
        deployable_feature_observability.get("status") != "PASS"
        or deployable_feature_observability != recomputed_observability
        or deployable_feature_observability.get("train_only") is not True
        or deployable_feature_observability.get("settings")
        != expected_observability_settings
        or set(observability_checks)
        != {
            "regime_macro_f1",
            "needed_sources_macro_f1",
            "memory_need_macro_f1",
            "strategy_direction_macro_f1",
            "memory_direction_macro_f1",
        }
        or not all(bool(value) for value in observability_checks.values())
    ):
        raise RuntimeError(
            "development pilot deployable-feature observability did not PASS"
        )
    expected_outcome_keys = sorted(
        [str(row["card_id"]), action]
        for row in selected_states
        for action in expected_actions
    )
    if plan.get("expected_keys_sha256") != sha256_text(
        canonical_json(expected_outcome_keys)
    ):
        raise RuntimeError("development pilot expected-key hash mismatch")

    generation_endpoint = endpoint_from_config(
        experiment_config,
        supporter_generation_contract.generator_endpoint,
    )
    judge_schema_smoke = require_development_judge_schema_smoke_pass(
        summary_path=judge_schema_smoke_summary_path,
        attestation_path=judge_schema_smoke_attestation_path,
        experiment_config_path=experiment_config_path,
        pm_v2_config_path=pm_v2_config_path,
        states_path=states_path,
        backend_path=backend_path,
        evaluator_contexts_path=evaluator_contexts_path,
        pilot_plan_path=pilot_plan_path,
        semantic_sanity_report_path=semantic_sanity_report_path,
        semantic_sanity_attestation_path=semantic_sanity_attestation_path,
    )
    expected_sweep_binding_base = {
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "pm_v2_version": "pm-v2.2",
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "development_sweep": dict(pm_v2_config["development_sweep"]),
        "retrieval": dict(pm_v2_config["retrieval"]),
        "semantic_sanity": dict(semantic_sanity),
        "runtime_state_lineage": dict(runtime_state_lineage),
        "api_cost_planning": dict(pm_v2_config["api_cost_planning"]),
        "scope": "compatibility_pilot",
        "pilot_plan_sha256": str(plan["pilot_plan_sha256"]),
        "pilot_expected_keys_sha256": str(plan["expected_keys_sha256"]),
        "deployable_feature_observability": deployable_feature_observability,
        "judge_schema_smoke": judge_schema_smoke,
    }
    pilot_sweep_verification = require_artifact_attestation(
        pilot_sweep_attestation_path,
        required_stage="action_sweep",
        required_output_paths={"summary": pilot_sweep_summary_path},
    )
    pilot_sweep_attestation = read_json(pilot_sweep_attestation_path)
    pilot_outcomes_path = _recorded_path(
        pilot_sweep_attestation, "outputs", "action_outcomes"
    )
    pilot_raw_calls_path = _recorded_path(
        pilot_sweep_attestation, "outputs", "raw_calls"
    )
    _recorded_path(
        pilot_sweep_attestation, "outputs", "physical_attempt_ledger"
    )
    pilot_sweep_manifest_path = _recorded_path(
        pilot_sweep_attestation, "inputs", "run_manifest"
    )
    for logical_name, expected_path in (
        ("runtime", runtime_path),
        ("backend", backend_path),
        ("strategy_bank", strategy_bank_path),
    ):
        _require_recorded_path(
            pilot_sweep_attestation, "inputs", logical_name, expected_path
        )
    pilot_manifest = read_json(pilot_sweep_manifest_path)
    if not _manifest_is_self_consistent(pilot_manifest):
        raise RuntimeError("development pilot sweep manifest self-hash mismatch")
    observed_sweep_bindings = pilot_manifest.get("contract_bindings") or {}
    accepted_sweep_cost_hash = str(
        observed_sweep_bindings.get("accepted_cost_estimate_sha256") or ""
    )
    expected_sweep_pricing = {
        "input_usd_per_mtok": float(
            pm_v2_config["development_sweep"]["pricing_usd_per_mtok"]["input"]
        ),
        "output_usd_per_mtok": float(
            pm_v2_config["development_sweep"]["pricing_usd_per_mtok"]["output"]
        ),
    }
    expected_sweep_bindings = {
        **expected_sweep_binding_base,
        "accepted_cost_estimate_sha256": accepted_sweep_cost_hash,
        "pricing": expected_sweep_pricing,
    }
    if (
        len(accepted_sweep_cost_hash) != 64
        or observed_sweep_bindings != expected_sweep_bindings
    ):
        raise RuntimeError(
            "development pilot sweep cost/pricing bindings are incomplete"
        )
    expected_pilot_manifest_values = {
        "stage": "action_sweep",
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "endpoint_model": generation_endpoint.model,
        "endpoint_base_url": generation_endpoint.base_url,
        "temperature": supporter_generation_contract.temperature,
        "max_tokens": supporter_generation_contract.max_output_tokens,
        "seed": int(pm_v2_config["development_sweep"]["seed"]),
        "request_retries": 1,
        "fail_fast": True,
        "input_token_safety_factor": float(
            pm_v2_config["api_cost_planning"]["input_token_safety_factor"]
        ),
        "fail_on_reported_input_overrun": True,
        "strategy_top_k": int(pm_v2_config["retrieval"]["strategy_top_k"]),
        "memory_min_score": float(pm_v2_config["retrieval"]["memory_min_score"]),
        "strategy_min_score": float(
            pm_v2_config["retrieval"]["strategy_min_score"]
        ),
        "action_filter": sorted(expected_actions),
        "card_filter": sorted(str(row["card_id"]) for row in selected_states),
        "system_prompt_sha256": (
            supporter_generation_contract.system_prompt_sha256
        ),
        "contract_bindings": expected_sweep_bindings,
        "planned_maximum_physical_api_attempts": expected_outcome_count,
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
    }
    for key, expected in expected_pilot_manifest_values.items():
        if pilot_manifest.get(key) != expected:
            raise RuntimeError(f"development pilot sweep manifest mismatch: {key}")
    pilot_summary = read_json(pilot_sweep_summary_path)
    expected_pilot_summary_values = {
        "status": "COMPLETE",
        "n_cards": expected_state_count,
        "expected_outcomes": expected_outcome_count,
        "completed_outcomes": expected_outcome_count,
        "total_physical_http_attempts": expected_outcome_count,
        "request_retries": 1,
        "fail_fast": True,
        "aborted_on_completion_gate": False,
        "failures": [],
        "endpoint_model": generation_endpoint.model,
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "contract_bindings": expected_sweep_bindings,
    }
    for key, expected in expected_pilot_summary_values.items():
        if pilot_summary.get(key) != expected:
            raise RuntimeError(f"development pilot sweep summary mismatch: {key}")
    outcome_rows = list(iter_jsonl(pilot_outcomes_path))
    actual_outcome_keys = sorted(
        [str(row.get("card_id")), str(row.get("action_id"))]
        for row in outcome_rows
    )
    if actual_outcome_keys != expected_outcome_keys:
        raise RuntimeError("development pilot sweep output matrix is not exact")
    if any(
        (row.get("provenance") or {}).get(
            "supporter_generation_treatment_sha256"
        )
        != supporter_generation_contract.digest()
        or (row.get("provenance") or {}).get(
            "supporter_generation_treatment"
        )
        != supporter_generation_contract.payload()
        or (row.get("provenance") or {}).get("normalized_finish_reason")
        != "complete"
        for row in outcome_rows
    ):
        raise RuntimeError(
            "development pilot outcomes are mixed-treatment or not complete"
        )
    raw_call_rows = list(iter_jsonl(pilot_raw_calls_path))
    if not raw_call_rows or any(
        row.get("error") is not None
        or row.get("normalized_finish_reason") != "complete"
        or row.get("completion_truncated") is not False
        for row in raw_call_rows
    ):
        raise RuntimeError(
            "development pilot raw calls include a failed or truncated completion"
        )
    pilot_parameters = pilot_sweep_attestation.get("parameters") or {}
    expected_pilot_parameters = {
        "endpoint_model": generation_endpoint.model,
        "endpoint_family": generation_endpoint.family,
        "endpoint_base_url": generation_endpoint.base_url,
        "request_retries": 1,
        "fail_fast": True,
        "input_token_safety_factor": float(
            pm_v2_config["api_cost_planning"]["input_token_safety_factor"]
        ),
        "fail_on_reported_input_overrun": True,
        "action_filter": sorted(expected_actions),
        "card_filter": sorted(str(row["card_id"]) for row in selected_states),
        "contract_bindings": expected_sweep_bindings,
        "maximum_physical_api_attempts_planned": expected_outcome_count,
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
    }
    for key, expected in expected_pilot_parameters.items():
        if pilot_parameters.get(key) != expected:
            raise RuntimeError(f"development pilot sweep attestation mismatch: {key}")

    judge_verification = require_artifact_attestation(
        judge_compatibility_attestation_path,
        required_stage="pm_v2_development_judge_compatibility",
        required_output_paths={"summary": judge_compatibility_summary_path},
    )
    judge_attestation = read_json(judge_compatibility_attestation_path)
    judge_output_paths = {
        required_output: _recorded_path(
            judge_attestation, "outputs", required_output
        )
        for required_output in (
            "summary",
            "labels",
            "raw_results",
            "call_ledger",
        )
    }
    for logical_name, expected_path in (
        ("experiment_config", experiment_config_path),
        ("pm_v2_config", pm_v2_config_path),
        ("states", states_path),
        ("outcomes", pilot_outcomes_path),
        ("evaluator_contexts", evaluator_contexts_path),
        ("sweep_manifest", pilot_sweep_manifest_path),
        ("pilot_plan", pilot_plan_path),
    ):
        _require_recorded_path(
            judge_attestation, "inputs", logical_name, expected_path
        )
    judge_summary = read_json(judge_compatibility_summary_path)
    composite_spec = composite_spec_from_config(pm_v2_config)
    composite_hash = composite_weights_hash(composite_spec)
    labeling = labeling_settings_from_config(pm_v2_config)
    judging_config = dict(pm_v2_config["development_judging"])
    judge_descriptors = _current_judge_descriptors(
        experiment_config, pm_v2_config
    )
    judge_families = {str(row["family"]) for row in judge_descriptors}
    if None in {row["family"] for row in judge_descriptors} or len(
        judge_families
    ) != len(judge_descriptors):
        raise RuntimeError("development pilot judges are not family-distinct")
    judge_pairs = expected_outcome_count * len(judge_descriptors)
    judge_calls = judge_pairs * 2
    compatibility_gate = judge_summary.get("compatibility_gate") or {}
    if compatibility_gate.get("status") != "PASS" or not all(
        bool(value) for value in (compatibility_gate.get("checks") or {}).values()
    ):
        raise RuntimeError("development judge compatibility gate did not PASS")
    expected_compatibility_thresholds = {
        key: float(pilot_config[key])
        for key in (
            "minimum_schema_success_rate",
            "minimum_reliable_label_rate",
            "minimum_low_mad_coverage_per_dimension",
            "minimum_low_mad_coverage_per_action_dimension",
        )
    }
    if compatibility_gate.get("thresholds") != expected_compatibility_thresholds:
        raise RuntimeError("development judge compatibility thresholds changed")
    label_value_feasibility = judge_summary.get("label_value_feasibility") or {}
    expected_feasibility_thresholds = dict(
        pilot_config["label_value_feasibility"]
    )
    expected_feasibility_checks = {
        "distinct_oracle_actions",
        "maximum_oracle_action_share",
        "m0_oracle_share",
        "r0_oracle_share",
        "rs_oracle_share",
        "per_regime_oracle_diversity",
        "within_state_quality_range",
        "within_state_quality_variance",
        "strategy_helpful_separation",
        "strategy_harmful_separation",
        "memory_helpful_separation",
        "memory_harmful_separation",
        "oracle_utility_headroom",
        "oracle_quality_headroom",
        "oracle_emotional_support_headroom",
    }
    feasibility_checks = label_value_feasibility.get("checks") or {}
    if (
        label_value_feasibility.get("status") != "PASS"
        or set(feasibility_checks) != expected_feasibility_checks
        or not all(bool(value) for value in feasibility_checks.values())
        or label_value_feasibility.get("thresholds")
        != expected_feasibility_thresholds
        or label_value_feasibility.get("n_states") != expected_state_count
        or label_value_feasibility.get("n_labels") != expected_outcome_count
    ):
        raise RuntimeError(
            "development pilot label-value feasibility did not exactly PASS"
        )
    feasibility_by_regime = label_value_feasibility.get(
        "distinct_oracle_actions_by_regime"
    ) or {}
    if (
        set(feasibility_by_regime) != expected_regimes
        or any(
            int(value)
            < int(
                expected_feasibility_thresholds[
                    "minimum_distinct_oracle_actions_per_regime"
                ]
            )
            for value in feasibility_by_regime.values()
        )
    ):
        raise RuntimeError(
            "development pilot label-value regime coverage is incomplete"
        )
    feasibility_rows = list(label_value_feasibility.get("rows") or [])
    if (
        len(feasibility_rows) != expected_state_count
        or {str(row.get("state_id")) for row in feasibility_rows}
        != {str(row["state_id"]) for row in selected_states}
        or Counter(str(row.get("regime")) for row in feasibility_rows)
        != Counter(str(row["regime"]) for row in selected_states)
    ):
        raise RuntimeError(
            "development pilot label-value rows do not match the frozen pilot"
        )
    expected_judge_summary_values = {
        "status": "PASS",
        "scope": "compatibility_pilot",
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "prompt_contract_hash": prompt_contract_hash(),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_hash,
        "labeling_gates": labeling,
        "development_judging": judging_config,
        "judge_endpoint_descriptors": judge_descriptors,
        "pilot_plan_sha256": plan["pilot_plan_sha256"],
        "pilot_expected_keys_sha256": plan["expected_keys_sha256"],
        "compatibility_attestation_sha256": None,
        "full_expected_api_calls": judge_calls,
        "completed_judge_pairs": judge_pairs,
        "remaining_judge_pairs": 0,
        "remaining_api_calls": 0,
        "label_rows": expected_outcome_count,
        "physical_http_attempts": judge_calls,
        "label_value_feasibility": label_value_feasibility,
    }
    for key, expected in expected_judge_summary_values.items():
        if judge_summary.get(key) != expected:
            raise RuntimeError(f"development judge compatibility mismatch: {key}")
    judge_parameters = judge_attestation.get("parameters") or {}
    expected_judge_parameters = {
        "status": "PASS",
        "scope": "compatibility_pilot",
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "prompt_contract_hash": prompt_contract_hash(),
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_hash,
        "labeling_gates": labeling,
        "development_judging": judging_config,
        "judge_endpoint_descriptors": judge_descriptors,
        "pilot_plan_sha256": plan["pilot_plan_sha256"],
        "pilot_expected_keys_sha256": plan["expected_keys_sha256"],
        "compatibility_attestation_sha256": None,
        "compatibility_gate": compatibility_gate,
        "label_value_feasibility": label_value_feasibility,
        "judge_retries": 1,
        "api_cost_planning": dict(pm_v2_config["api_cost_planning"]),
        "pricing_usd_per_mtok": {
            str(family): {
                "input": float(values["input"]),
                "output": float(values["output"]),
            }
            for family, values in dict(
                judging_config["pricing_usd_per_mtok"]
            ).items()
        },
        "physical_http_attempts": judge_calls,
    }
    for key, expected in expected_judge_parameters.items():
        if judge_parameters.get(key) != expected:
            raise RuntimeError(
                f"development judge compatibility attestation mismatch: {key}"
            )
    if len(str(judge_parameters.get("accepted_cost_estimate_sha256") or "")) != 64:
        raise RuntimeError(
            "development judge compatibility lacks accepted cost-estimate hash"
        )

    human_spot_check = require_pilot_human_spot_check_pass(
        report_path=pilot_human_spot_check_report_path,
        attestation_path=pilot_human_spot_check_attestation_path,
        config=pm_v2_config,
        config_path=pm_v2_config_path,
        states_path=states_path,
        evaluator_contexts_path=evaluator_contexts_path,
        pilot_plan_path=pilot_plan_path,
        outcomes_path=pilot_outcomes_path,
        labels_path=judge_output_paths["labels"],
        raw_results_path=judge_output_paths["raw_results"],
        judge_compatibility_summary_path=judge_compatibility_summary_path,
        judge_compatibility_attestation_path=(
            judge_compatibility_attestation_path
        ),
    )

    result = {
        "status": "PASS",
        "protocol": DEVELOPMENT_PILOT_GATE_PROTOCOL,
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "generator_endpoint_sha256": sha256_text(
            canonical_json(
                {
                    "model": generation_endpoint.model,
                    "family": generation_endpoint.family,
                    "base_url": generation_endpoint.base_url,
                }
            )
        ),
        "semantic_sanity_attestation_sha256": semantic_sanity.get(
            "attestation_sha256"
        ),
        "deployable_feature_observability": deployable_feature_observability,
        "judge_schema_smoke": judge_schema_smoke,
        "pilot_plan_sha256": plan["pilot_plan_sha256"],
        "pilot_expected_keys_sha256": plan["expected_keys_sha256"],
        "pilot_sweep_summary_sha256": sha256_file(pilot_sweep_summary_path),
        "pilot_sweep_manifest_sha256": sha256_file(pilot_sweep_manifest_path),
        "pilot_sweep_outcomes_sha256": sha256_file(pilot_outcomes_path),
        "pilot_sweep_attestation_sha256": pilot_sweep_verification[
            "attestation_sha256"
        ],
        "judge_compatibility_summary_sha256": sha256_file(
            judge_compatibility_summary_path
        ),
        "judge_compatibility_attestation_sha256": judge_verification[
            "attestation_sha256"
        ],
        "label_value_feasibility_sha256": sha256_text(
            canonical_json(label_value_feasibility)
        ),
        "human_spot_check": human_spot_check,
        "expected_pilot_outcomes": expected_outcome_count,
        "expected_judge_physical_calls": judge_calls,
    }
    result["binding_sha256"] = sha256_text(canonical_json(result))
    return result
