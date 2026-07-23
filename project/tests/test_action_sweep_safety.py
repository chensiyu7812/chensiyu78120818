from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from metacom_pm.api import CallResult, Endpoint, ProviderRequestError
from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import MemoryBackendRecord
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_contracts import ResourceNeedRegime
from metacom_pm.pm_v2_development_gate import require_development_pilot_gate
from metacom_pm.pm_v2_judging import (
    composite_spec_from_config,
    composite_weights_hash,
    labeling_settings_from_config,
    prompt_contract_hash,
)
from metacom_pm.sweep import plan_action_sweep, run_action_sweep
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_development_pilot_gate_fixture(tmp_path: Path) -> tuple[dict, dict]:
    experiment_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    pm_v2_config_path = PROJECT_ROOT / "configs" / "pm_v2.yaml"
    experiment_config = load_config(experiment_config_path)
    pm_v2_config = load_config(pm_v2_config_path)
    supporter_generation_contract = SupporterGenerationContract.from_config(
        pm_v2_config
    )
    pilot_config = pm_v2_config["development_judging"]["compatibility_pilot"]
    actions = [str(value) for value in pilot_config["actions"]]

    states_path = tmp_path / "states.jsonl"
    runtime_path = tmp_path / "runtime.jsonl"
    backend_path = tmp_path / "backend.jsonl"
    evaluator_contexts_path = tmp_path / "evaluator_contexts.jsonl"
    strategy_bank_path = tmp_path / "strategy.jsonl"
    for path, marker in (
        (states_path, "states"),
        (runtime_path, "runtime"),
        (backend_path, "backend"),
        (evaluator_contexts_path, "evaluator"),
        (strategy_bank_path, "strategy"),
    ):
        write_jsonl(path, [{"fixture": marker}])
    semantic_sanity = {
        "status": "PASS",
        "attestation_sha256": "a" * 64,
        "lineage": "semantic-fixture",
    }
    runtime_state_lineage = {
        "status": "PASS",
        "lineage_sha256": "b" * 64,
    }
    observability = {
        "status": "PASS",
        "checks": {
            "regime_macro_f1": True,
            "needed_sources_macro_f1": True,
            "memory_need_macro_f1": True,
            "strategy_direction_macro_f1": True,
            "memory_direction_macro_f1": True,
        },
        "settings": dict(
            pilot_config["deployable_feature_observability"]
        ),
        "train_only": True,
        "n_states": 216,
        "n_users": 24,
        "class_counts": {},
        "metrics": {},
        "folds": [],
        "feature_builder_config_sha256": "4" * 64,
        "interpretation": "fixture",
    }
    schema_smoke = {
        "status": "PASS",
        "protocol": "pm-v2-development-judge-schema-smoke-v1",
        "summary_sha256": "5" * 64,
        "attestation_sha256": "6" * 64,
        "contract_sha256": "7" * 64,
        "physical_attempts": 4,
    }

    selected_states = []
    index = 0
    for regime in sorted(regime.value for regime in ResourceNeedRegime):
        for _ in range(int(pilot_config["states_per_regime"])):
            selected_states.append(
                {
                    "regime": regime,
                    "state_id": f"state_{index:03d}",
                    "card_id": f"card_{index:03d}",
                    "split": "train",
                    "actions": actions,
                }
            )
            index += 1
    expected_keys = sorted(
        [row["card_id"], action]
        for row in selected_states
        for action in actions
    )
    pilot_plan = {
        "status": "READY",
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
        "evaluator_contexts_map_sha256": "c" * 64,
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "semantic_sanity": semantic_sanity,
        "runtime_state_lineage": runtime_state_lineage,
        "deployable_feature_observability": observability,
        "sample_seed": int(pilot_config["sample_seed"]),
        "states_per_regime": int(pilot_config["states_per_regime"]),
        "actions": actions,
        "selected_states": selected_states,
        "expected_state_count": len(selected_states),
        "expected_outcome_count": len(expected_keys),
        "expected_keys_sha256": sha256_text(canonical_json(expected_keys)),
    }
    pilot_plan["pilot_plan_sha256"] = sha256_text(canonical_json(pilot_plan))
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    pilot_plan_path = pilot_dir / "pilot_plan.json"
    write_json(pilot_plan_path, pilot_plan)

    sweep_bindings = {
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
        "semantic_sanity": semantic_sanity,
        "runtime_state_lineage": runtime_state_lineage,
        "api_cost_planning": dict(pm_v2_config["api_cost_planning"]),
        "scope": "compatibility_pilot",
        "pilot_plan_sha256": pilot_plan["pilot_plan_sha256"],
        "pilot_expected_keys_sha256": pilot_plan["expected_keys_sha256"],
        "deployable_feature_observability": observability,
        "judge_schema_smoke": schema_smoke,
        "accepted_cost_estimate_sha256": "0" * 64,
        "pricing": {
            "input_usd_per_mtok": float(
                pm_v2_config["development_sweep"]["pricing_usd_per_mtok"][
                    "input"
                ]
            ),
            "output_usd_per_mtok": float(
                pm_v2_config["development_sweep"]["pricing_usd_per_mtok"][
                    "output"
                ]
            ),
        },
    }
    generator = endpoint_from_config(
        experiment_config,
        supporter_generation_contract.generator_endpoint,
    )
    card_filter = sorted(row["card_id"] for row in selected_states)
    pilot_manifest = {
        "stage": "action_sweep",
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "endpoint_model": generator.model,
        "endpoint_base_url": generator.base_url,
        "temperature": supporter_generation_contract.temperature,
        "max_tokens": supporter_generation_contract.max_output_tokens,
        "seed": int(pm_v2_config["development_sweep"]["seed"]),
        "request_retries": 1,
        "fail_fast": True,
        "input_token_safety_factor": 1.5,
        "fail_on_reported_input_overrun": True,
        "strategy_top_k": int(pm_v2_config["retrieval"]["strategy_top_k"]),
        "memory_min_score": float(pm_v2_config["retrieval"]["memory_min_score"]),
        "strategy_min_score": float(
            pm_v2_config["retrieval"]["strategy_min_score"]
        ),
        "action_filter": sorted(actions),
        "card_filter": card_filter,
        "system_prompt_sha256": (
            supporter_generation_contract.system_prompt_sha256
        ),
        "physical_attempt_ledger_protocol": "pm-v2-physical-http-attempt-ledger-v1",
        "planned_maximum_physical_api_attempts": len(expected_keys),
        "runtime_maximum_physical_api_attempts": len(expected_keys),
        "call_key_matrix_sha256": "e" * 64,
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "contract_bindings": sweep_bindings,
    }
    pilot_manifest["manifest_sha256"] = sha256_text(
        canonical_json(pilot_manifest)
    )
    pilot_manifest_path = pilot_dir / "run_manifest.json"
    write_json(pilot_manifest_path, pilot_manifest)
    outcomes_path = pilot_dir / "action_outcomes.jsonl"
    write_jsonl(
        outcomes_path,
        [
            {
                "card_id": card_id,
                "action_id": action_id,
                "provenance": {
                    "supporter_generation_treatment": (
                        supporter_generation_contract.payload()
                    ),
                    "supporter_generation_treatment_sha256": (
                        supporter_generation_contract.digest()
                    ),
                    "normalized_finish_reason": "complete",
                },
            }
            for card_id, action_id in expected_keys
        ],
    )
    raw_calls_path = pilot_dir / "raw_api_calls.jsonl"
    ledger_path = pilot_dir / "physical_attempt_ledger.jsonl"
    write_jsonl(
        raw_calls_path,
        [
            {
                "fixture": "raw",
                "error": None,
                "normalized_finish_reason": "complete",
                "completion_truncated": False,
            }
        ],
    )
    write_jsonl(ledger_path, [{"fixture": "ledger"}])
    pilot_summary = {
        "status": "COMPLETE",
        "n_cards": len(selected_states),
        "expected_outcomes": len(expected_keys),
        "completed_outcomes": len(expected_keys),
        "new_api_calls": len(expected_keys),
        "new_physical_http_attempts": len(expected_keys),
        "historical_physical_http_attempts": 0,
        "total_physical_http_attempts": len(expected_keys),
        "request_retries": 1,
        "fail_fast": True,
        "aborted_on_completion_gate": False,
        "failures": [],
        "endpoint_model": generator.model,
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "contract_bindings": sweep_bindings,
    }
    pilot_summary_path = pilot_dir / "summary.json"
    write_json(pilot_summary_path, pilot_summary)
    pilot_attestation_path = pilot_dir / "artifact_attestation.json"
    create_artifact_attestation(
        pilot_attestation_path,
        stage="action_sweep",
        inputs={
            "runtime": runtime_path,
            "backend": backend_path,
            "strategy_bank": strategy_bank_path,
            "run_manifest": pilot_manifest_path,
        },
        outputs={
            "action_outcomes": (outcomes_path, True),
            "raw_calls": (raw_calls_path, True),
            "physical_attempt_ledger": (ledger_path, True),
            "summary": (pilot_summary_path, False),
        },
        parameters={
            "endpoint_model": generator.model,
            "endpoint_family": generator.family,
            "endpoint_base_url": generator.base_url,
            "request_retries": 1,
            "fail_fast": True,
            "input_token_safety_factor": 1.5,
            "fail_on_reported_input_overrun": True,
            "maximum_physical_api_attempts_planned": len(expected_keys),
            "action_filter": sorted(actions),
            "card_filter": card_filter,
            "supporter_generation_treatment": (
                supporter_generation_contract.payload()
            ),
            "supporter_generation_treatment_sha256": (
                supporter_generation_contract.digest()
            ),
            "contract_bindings": sweep_bindings,
        },
        expected={"cards": len(selected_states), "outcomes": len(expected_keys)},
    )

    judging_dir = tmp_path / "judging"
    judging_dir.mkdir()
    judging_config = dict(pm_v2_config["development_judging"])
    judge_names = [str(value) for value in judging_config["judge_endpoints"]]
    judge_endpoints = [
        endpoint_from_config(experiment_config, name) for name in judge_names
    ]
    judge_descriptors = [
        {
            "name": name,
            "model": endpoint.model,
            "family": endpoint.family,
            "base_url": endpoint.base_url,
        }
        for name, endpoint in zip(judge_names, judge_endpoints)
    ]
    composite = composite_spec_from_config(pm_v2_config)
    labeling = labeling_settings_from_config(pm_v2_config)
    judge_pairs = len(expected_keys) * len(judge_endpoints)
    judge_calls = judge_pairs * 2
    compatibility_gate = {
        "status": "PASS",
        "checks": {
            "exact_raw_matrix": True,
            "schema_success_rate": True,
            "complete_two_family_labels": True,
            "dimension_quality_gate": True,
            "raw_family_dimension_quality_gate": True,
            "label_value_feasibility": True,
        },
        "thresholds": {
            key: float(pilot_config[key])
            for key in (
                "minimum_schema_success_rate",
                "minimum_reliable_label_rate",
                "minimum_low_mad_coverage_per_dimension",
                "minimum_low_mad_coverage_per_action_dimension",
            )
        },
    }
    feasibility_checks = {
        "distinct_oracle_actions": True,
        "maximum_oracle_action_share": True,
        "m0_oracle_share": True,
        "r0_oracle_share": True,
        "rs_oracle_share": True,
        "per_regime_oracle_diversity": True,
        "within_state_quality_range": True,
        "within_state_quality_variance": True,
        "strategy_helpful_separation": True,
        "strategy_harmful_separation": True,
        "memory_helpful_separation": True,
        "memory_harmful_separation": True,
        "oracle_utility_headroom": True,
        "oracle_quality_headroom": True,
        "oracle_emotional_support_headroom": True,
    }
    label_value_feasibility = {
        "status": "PASS",
        "checks": feasibility_checks,
        "thresholds": dict(pilot_config["label_value_feasibility"]),
        "n_states": len(selected_states),
        "n_labels": len(expected_keys),
        "oracle_action_distribution": {
            action: 3 for action in actions[:6]
        },
        "distinct_oracle_actions_by_regime": {
            regime.value: 2 for regime in ResourceNeedRegime
        },
        "rows": [
            {
                "state_id": row["state_id"],
                "regime": row["regime"],
                "oracle_action": actions[index % 6],
            }
            for index, row in enumerate(selected_states)
        ],
    }
    judge_summary = {
        "status": "PASS",
        "scope": "compatibility_pilot",
        "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
        "prompt_contract_hash": prompt_contract_hash(),
        "composite_spec": composite.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_hash(composite),
        "labeling_gates": labeling,
        "development_judging": judging_config,
        "judge_endpoint_descriptors": judge_descriptors,
        "pilot_plan_sha256": pilot_plan["pilot_plan_sha256"],
        "pilot_expected_keys_sha256": pilot_plan["expected_keys_sha256"],
        "compatibility_attestation_sha256": None,
        "full_expected_api_calls": judge_calls,
        "completed_judge_pairs": judge_pairs,
        "remaining_judge_pairs": 0,
        "remaining_api_calls": 0,
        "label_rows": len(expected_keys),
        "physical_http_attempts": judge_calls,
        "compatibility_gate": compatibility_gate,
        "label_value_feasibility": label_value_feasibility,
    }
    judge_summary_path = judging_dir / "summary.json"
    write_json(judge_summary_path, judge_summary)
    judge_cost_path = judging_dir / "cost_estimate.json"
    judge_plan_path = judging_dir / "call_plan.jsonl"
    judge_labels_path = judging_dir / "action_labels.jsonl"
    judge_raw_path = judging_dir / "judge_results.jsonl"
    judge_ledger_path = judging_dir / "judge_call_ledger.jsonl"
    write_json(judge_cost_path, {"cost_estimate_sha256": "f" * 64})
    write_jsonl(judge_plan_path, [{"fixture": "judge-plan"}])
    write_jsonl(judge_labels_path, [{"fixture": "labels"}])
    write_jsonl(judge_raw_path, [{"fixture": "judge-raw"}])
    write_jsonl(judge_ledger_path, [{"fixture": "judge-ledger"}])
    judge_attestation_path = judging_dir / "artifact_attestation.json"
    normalized_pricing = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in judging_config["pricing_usd_per_mtok"].items()
    }
    create_artifact_attestation(
        judge_attestation_path,
        stage="pm_v2_development_judge_compatibility",
        inputs={
            "experiment_config": experiment_config_path,
            "pm_v2_config": pm_v2_config_path,
            "states": states_path,
            "outcomes": outcomes_path,
            "evaluator_contexts": evaluator_contexts_path,
            "sweep_manifest": pilot_manifest_path,
            "cost_estimate": judge_cost_path,
            "call_plan": judge_plan_path,
            "pilot_plan": pilot_plan_path,
        },
        outputs={
            "summary": (judge_summary_path, False),
            "labels": (judge_labels_path, True),
            "raw_results": (judge_raw_path, True),
            "call_ledger": (judge_ledger_path, True),
        },
        parameters={
            "status": "PASS",
            "scope": "compatibility_pilot",
            "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
            "prompt_contract_hash": prompt_contract_hash(),
            "composite_spec": composite.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_hash(composite),
            "labeling_gates": labeling,
            "development_judging": judging_config,
            "judge_endpoint_descriptors": judge_descriptors,
            "pilot_plan_sha256": pilot_plan["pilot_plan_sha256"],
            "pilot_expected_keys_sha256": pilot_plan["expected_keys_sha256"],
            "compatibility_attestation_sha256": None,
            "compatibility_gate": compatibility_gate,
            "label_value_feasibility": label_value_feasibility,
            "accepted_cost_estimate_sha256": "f" * 64,
            "judge_retries": 1,
            "api_cost_planning": dict(pm_v2_config["api_cost_planning"]),
            "pricing_usd_per_mtok": normalized_pricing,
            "physical_http_attempts": judge_calls,
        },
        expected={"outcomes": len(expected_keys), "judge_pairs": judge_pairs},
    )
    kwargs = {
        "experiment_config_path": experiment_config_path,
        "pm_v2_config_path": pm_v2_config_path,
        "states_path": states_path,
        "runtime_path": runtime_path,
        "backend_path": backend_path,
        "evaluator_contexts_path": evaluator_contexts_path,
        "strategy_bank_path": strategy_bank_path,
        "semantic_sanity": semantic_sanity,
        "runtime_state_lineage": runtime_state_lineage,
        "semantic_sanity_report_path": tmp_path / "semantic_report.json",
        "semantic_sanity_attestation_path": (
            tmp_path / "semantic_attestation.json"
        ),
        "pilot_plan_path": pilot_plan_path,
        "pilot_sweep_summary_path": pilot_summary_path,
        "pilot_sweep_attestation_path": pilot_attestation_path,
        "judge_compatibility_summary_path": judge_summary_path,
        "judge_compatibility_attestation_path": judge_attestation_path,
        "pilot_human_spot_check_report_path": (
            tmp_path / "pilot_human_spot_check_report.json"
        ),
        "pilot_human_spot_check_attestation_path": (
            tmp_path / "pilot_human_spot_check_attestation.json"
        ),
        "judge_schema_smoke_summary_path": tmp_path / "schema_smoke_summary.json",
        "judge_schema_smoke_attestation_path": (
            tmp_path / "schema_smoke_attestation.json"
        ),
    }
    paths = {
        "pilot_plan": pilot_plan_path,
        "pilot_summary": pilot_summary_path,
        "pilot_manifest": pilot_manifest_path,
        "pilot_outcomes": outcomes_path,
        "pilot_raw": raw_calls_path,
        "pilot_attestation": pilot_attestation_path,
        "observability": observability,
        "schema_smoke": schema_smoke,
        "judge_summary": judge_summary_path,
        "judge_attestation": judge_attestation_path,
    }
    return kwargs, paths


def test_judging_requires_content_addressed_sweep_chain(tmp_path: Path) -> None:
    _, paths = _build_development_pilot_gate_fixture(tmp_path)
    script = PROJECT_ROOT / "scripts" / "21_judge_pm_v2_action_sweep.py"
    spec = importlib.util.spec_from_file_location("pm_v2_judge_source_gate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.require_action_sweep_source_chain(
        outcomes_path=paths["pilot_outcomes"],
        summary_path=paths["pilot_summary"],
        manifest_path=paths["pilot_manifest"],
        attestation_path=paths["pilot_attestation"],
    )
    assert result["status"] == "PASS"
    assert result["expected_outcomes"] == 180

    original_raw = paths["pilot_raw"].read_text(encoding="utf-8")
    paths["pilot_raw"].write_text(original_raw + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outputs hash mismatch"):
        module.require_action_sweep_source_chain(
            outcomes_path=paths["pilot_outcomes"],
            summary_path=paths["pilot_summary"],
            manifest_path=paths["pilot_manifest"],
            attestation_path=paths["pilot_attestation"],
        )

    gate = {"status": "PASS", "binding_sha256": "9" * 64}
    assert module.require_full_sweep_development_binding(
        {"contract_bindings": {"development_pilot_gate": gate}}, gate
    ) == gate
    with pytest.raises(RuntimeError, match="not bound"):
        module.require_full_sweep_development_binding(
            {"contract_bindings": {}}, gate
        )
    with pytest.raises(RuntimeError, match="not bound"):
        module.require_full_sweep_development_binding(
            {
                "contract_bindings": {
                    "development_pilot_gate": {
                        **gate,
                        "binding_sha256": "8" * 64,
                    }
                }
            },
            gate,
        )


def _load_script_module(name: str, filename: str):
    script = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script06_spent_ledger_makes_dry_run_exact_validate_only(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module(
        "pm_v2_action_sweep_dry_run", "06_run_action_sweep.py"
    )
    out_dir = tmp_path / "sweep"
    result = {
        "cost_estimate_sha256": "a" * 64,
        "call_plan_sha256": "b" * 64,
        "budget_gate": {"status": "PASS", "limits": {"max_api_calls": 2}},
    }
    rows = [{"physical_call_key": "one"}, {"physical_call_key": "two"}]
    assert module._persist_or_validate_dry_run(out_dir, result, rows) == "WRITTEN"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    ledger_path.write_text('{"spent":true}\n', encoding="utf-8")
    estimate_before = (out_dir / "cost_estimate.json").read_bytes()
    plan_before = (out_dir / "call_plan.jsonl").read_bytes()

    def forbid_write(*args, **kwargs):
        raise AssertionError("spent dry-run attempted to rewrite an artifact")

    monkeypatch.setattr(module, "write_json", forbid_write)
    monkeypatch.setattr(module, "write_jsonl", forbid_write)
    assert (
        module._persist_or_validate_dry_run(out_dir, result, rows)
        == "VALIDATED_EXISTING"
    )
    with pytest.raises(RuntimeError, match="original exact estimate"):
        module._persist_or_validate_dry_run(
            out_dir,
            {
                **result,
                "budget_gate": {
                    "status": "PASS",
                    "limits": {"max_api_calls": 3},
                },
            },
            rows,
        )
    assert (out_dir / "cost_estimate.json").read_bytes() == estimate_before
    assert (out_dir / "call_plan.jsonl").read_bytes() == plan_before


def test_script21_spent_ledger_freezes_full_dry_run_and_client_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script_module(
        "pm_v2_judge_dry_run", "21_judge_pm_v2_action_sweep.py"
    )
    out_dir = tmp_path / "judge"
    out_dir.mkdir()
    ledger_path = out_dir / "judge_call_ledger.jsonl"
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    estimate = {"cost_estimate_sha256": "c" * 64, "full_logical_api_calls": 4}
    gate = {"status": "PASS", "limits": {"max_api_calls": 4}}
    plan = [{"physical_call_key": str(index)} for index in range(4)]
    assert module.persist_or_validate_judge_dry_run(
        ledger_path=ledger_path,
        estimate_path=estimate_path,
        call_plan_path=plan_path,
        cost_estimate=estimate,
        budget_gate=gate,
        call_plan=plan,
    ) == "WRITTEN"
    ledger_path.write_text('{"spent":true}\n', encoding="utf-8")
    estimate_before = estimate_path.read_bytes()
    plan_before = plan_path.read_bytes()

    def forbid_write(*args, **kwargs):
        raise AssertionError("spent judge dry-run attempted to rewrite an artifact")

    monkeypatch.setattr(module, "write_json", forbid_write)
    monkeypatch.setattr(module, "write_jsonl", forbid_write)
    assert module.persist_or_validate_judge_dry_run(
        ledger_path=ledger_path,
        estimate_path=estimate_path,
        call_plan_path=plan_path,
        cost_estimate=estimate,
        budget_gate=gate,
        call_plan=plan,
    ) == "VALIDATED_EXISTING"
    with pytest.raises(RuntimeError, match="original exact estimate"):
        module.persist_or_validate_judge_dry_run(
            ledger_path=ledger_path,
            estimate_path=estimate_path,
            call_plan_path=plan_path,
            cost_estimate=estimate,
            budget_gate={"status": "PASS", "limits": {"max_api_calls": 5}},
            call_plan=plan,
        )
    assert estimate_path.read_bytes() == estimate_before
    assert plan_path.read_bytes() == plan_before

    first_key = "PM_V2_PREFLIGHT_FIRST"
    second_key = "PM_V2_PREFLIGHT_SECOND"
    monkeypatch.setenv(first_key, "first-secret")
    monkeypatch.delenv(second_key, raising=False)
    endpoints = {
        "a": Endpoint("https://invalid.example", "a", first_key, family="a"),
        "b": Endpoint("https://invalid.example", "b", second_key, family="b"),
    }
    built = []
    with pytest.raises(RuntimeError, match=second_key):
        module.preflight_development_judge_clients(
            endpoints, client_factory=lambda endpoint: built.append(endpoint)
        )
    assert built == []

    monkeypatch.setenv(second_key, "second-secret")

    class FirstClient:
        closed = False

        def close(self):
            self.closed = True

    first_client = FirstClient()

    def partial_factory(endpoint):
        if endpoint.family == "a":
            return first_client
        raise RuntimeError("second client construction failed")

    with pytest.raises(RuntimeError, match="second client construction failed"):
        module.preflight_development_judge_clients(
            endpoints, client_factory=partial_factory
        )
    assert first_client.closed is True
    assert ledger_path.read_text(encoding="utf-8") == '{"spent":true}\n'


def test_full_sweep_pilot_gate_is_exact_and_tamper_evident(
    tmp_path: Path, monkeypatch
) -> None:
    import metacom_pm.pm_v2_development_gate as development_gate_module

    human_result = {
        "status": "PASS",
        "protocol": "pm-v2-development-judge-human-spot-check-v1",
        "report_sha256": "1" * 64,
        "attestation_sha256": "2" * 64,
        "human_plan_sha256": "3" * 64,
        "item_count": 18,
        "judge_families": ["fixture_a", "fixture_b"],
    }
    monkeypatch.setattr(
        development_gate_module,
        "require_pilot_human_spot_check_pass",
        lambda **kwargs: human_result,
    )
    kwargs, paths = _build_development_pilot_gate_fixture(tmp_path)
    monkeypatch.setattr(development_gate_module, "load_states", lambda path: [])
    monkeypatch.setattr(
        development_gate_module,
        "load_evaluator_context_index",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        development_gate_module,
        "audit_deployable_feature_observability",
        lambda *args, **kwargs: paths["observability"],
    )
    monkeypatch.setattr(
        development_gate_module,
        "require_development_judge_schema_smoke_pass",
        lambda **kwargs: paths["schema_smoke"],
    )
    result = require_development_pilot_gate(**kwargs)
    assert result["status"] == "PASS"
    assert result["expected_pilot_outcomes"] == 180
    assert result["expected_judge_physical_calls"] == 720
    assert len(result["binding_sha256"]) == 64
    assert result["human_spot_check"] == human_result

    pilot_plan = read_json(paths["pilot_plan"])
    contaminated_plan = json.loads(json.dumps(pilot_plan))
    contaminated_plan["selected_states"][0]["split"] = "calibration"
    contaminated_payload = {
        key: value
        for key, value in contaminated_plan.items()
        if key != "pilot_plan_sha256"
    }
    contaminated_plan["pilot_plan_sha256"] = sha256_text(
        canonical_json(contaminated_payload)
    )
    write_json(paths["pilot_plan"], contaminated_plan)
    with pytest.raises(RuntimeError, match="train states only"):
        require_development_pilot_gate(**kwargs)
    write_json(paths["pilot_plan"], pilot_plan)

    missing = dict(kwargs)
    missing["judge_compatibility_attestation_path"] = (
        tmp_path / "missing-attestation.json"
    )
    with pytest.raises(RuntimeError, match="Artifact attestation failed"):
        require_development_pilot_gate(**missing)

    judge_summary = read_json(paths["judge_summary"])
    write_json(paths["judge_summary"], {**judge_summary, "status": "FAIL"})
    with pytest.raises(RuntimeError, match="outputs hash mismatch"):
        require_development_pilot_gate(**kwargs)


def test_action_sweep_plan_is_exact_and_does_not_need_api_key(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    monkeypatch.delenv("UNSET_TEST_API_KEY", raising=False)
    endpoint = Endpoint(
        base_url="https://invalid.example",
        model="never-called",
        api_key_env="UNSET_TEST_API_KEY",
        family="test",
    )

    estimate, rows = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )

    assert estimate["expected_api_calls"] == len(tiny_state.allowed_actions)
    assert estimate["request_retries"] == 3
    assert estimate["maximum_physical_api_attempts"] == 3 * len(
        tiny_state.allowed_actions
    )
    assert estimate["estimated_cost_usd"] == 3 * estimate[
        "logical_estimated_cost_usd"
    ]
    assert len(rows) == len(tiny_state.allowed_actions)
    assert {row["action_id"] for row in rows} == set(tiny_state.allowed_actions)
    assert all(len(row["call_key"]) == 64 for row in rows)
    assert all(row["max_http_attempts"] == 3 for row in rows)
    assert len({row["call_key"] for row in rows}) == len(rows)
    assert estimate["max_input_tokens"] > 0
    assert estimate["estimated_cost_usd"] > 0
    assert len(estimate["cost_estimate_sha256"]) == 64


def test_action_sweep_caches_identical_strategy_query_without_changing_rows(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    import metacom_pm.sweep as sweep_module

    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [MemoryBackendRecord(card_id=tiny_state.card_id, items=tiny_memories).model_dump(mode="json")],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    original_retrieve = sweep_module.StrategyRetriever.retrieve
    calls = 0

    def counted_retrieve(self, query):
        nonlocal calls
        calls += 1
        return original_retrieve(self, query)

    monkeypatch.setattr(sweep_module.StrategyRetriever, "retrieve", counted_retrieve)
    endpoint = Endpoint("https://invalid.example", "fixture", "UNSET", family="test")
    estimate, rows = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        request_retries=1,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    assert any(row["action_id"].endswith("+RS") for row in rows)
    assert calls == 1
    assert estimate["logical_api_calls"] == len(rows)


def test_action_sweep_deduplicates_filter_collapsed_prompts(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    filter_config = EvidenceFilterConfig.from_mapping(
        load_config(PROJECT_ROOT / "configs/pm_v2.yaml")["evidence_filter"]
    )
    endpoint = Endpoint(
        "https://invalid.example", "fixture", "UNSET", family="test"
    )
    estimate, rows = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        request_retries=1,
        evidence_filter_config=filter_config,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    assert estimate["logical_api_calls"] == len(tiny_state.allowed_actions)
    assert estimate["expected_api_calls"] < estimate["logical_api_calls"]
    assert estimate["prompt_alias_outcomes"] > 0
    assert len({row["call_key"] for row in rows}) == estimate[
        "expected_api_calls"
    ]

    class SuccessfulClient:
        calls = 0

        def __init__(self, endpoint):
            pass

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            type(self).calls += 1
            return (
                CallResult(
                    text="A supportive response.",
                    raw_response={"fixture": True},
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 5,
                        "total_tokens": 105,
                    },
                    latency_ms=1.0,
                    request_hash="prompt-equivalence-success",
                ),
                None,
            )

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", SuccessfulClient)
    out_dir = tmp_path / "out"
    summary = run_action_sweep(
        runtime,
        backend,
        strategies,
        out_dir / "outcomes.jsonl",
        out_dir / "raw.jsonl",
        out_dir / "summary.json",
        endpoint=endpoint,
        request_retries=1,
        evidence_filter_config=filter_config,
    )
    assert summary["status"] == "COMPLETE"
    assert SuccessfulClient.calls == estimate["expected_api_calls"]
    outcomes = list(iter_jsonl(out_dir / "outcomes.jsonl"))
    assert len(outcomes) == len(tiny_state.allowed_actions)
    assert len(list(iter_jsonl(out_dir / "raw.jsonl"))) == estimate[
        "expected_api_calls"
    ]
    aliases = [
        row
        for row in outcomes
        if (row.get("provenance") or {}).get("prompt_equivalence_alias")
    ]
    assert aliases


def test_action_sweep_cost_hash_binds_pricing(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    endpoint = Endpoint("https://invalid.example", "never-called", "UNSET", family="test")

    first, _ = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    second, _ = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        input_usd_per_mtok=1.1,
        output_usd_per_mtok=2.0,
    )

    assert first["cost_estimate_sha256"] != second["cost_estimate_sha256"]


def test_action_sweep_cost_hash_binds_physical_attempt_cap(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    endpoint = Endpoint("https://invalid.example", "never-called", "UNSET", family="test")

    one_attempt, _ = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        request_retries=1,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    three_attempts, _ = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        request_retries=3,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )

    assert one_attempt["logical_api_calls"] == three_attempts["logical_api_calls"]
    assert three_attempts["maximum_physical_api_attempts"] == 3 * one_attempt[
        "maximum_physical_api_attempts"
    ]
    assert three_attempts["estimated_cost_usd"] == 3 * one_attempt[
        "estimated_cost_usd"
    ]
    assert one_attempt["cost_estimate_sha256"] != three_attempts[
        "cost_estimate_sha256"
    ]
    fail_fast, _ = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        request_retries=1,
        fail_fast=True,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    assert fail_fast["cost_estimate_sha256"] != one_attempt[
        "cost_estimate_sha256"
    ]
    safety_margin, safety_rows = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        request_retries=1,
        input_token_safety_factor=1.5,
        fail_on_reported_input_overrun=True,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    assert safety_margin["input_token_safety_factor"] == 1.5
    assert safety_margin["fail_on_reported_input_overrun"] is True
    assert safety_margin["cost_estimate_sha256"] != one_attempt[
        "cost_estimate_sha256"
    ]
    assert all(
        row["estimated_input_tokens"] > row["raw_estimated_input_tokens"]
        for row in safety_rows
    )


def test_action_sweep_transport_attempt_cap_is_separate_from_request_retries(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    endpoint = Endpoint(
        "https://invalid.example", "never-called", "UNSET", family="test"
    )

    estimate, rows = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        request_retries=1,
        transport_max_attempts_per_call=4,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )

    assert estimate["request_retries"] == 1
    assert estimate["transport_max_attempts_per_call"] == 4
    assert estimate["maximum_physical_api_attempts"] == 4 * estimate[
        "expected_api_calls"
    ]
    assert estimate["estimated_cost_usd"] == pytest.approx(
        4 * estimate["logical_estimated_cost_usd"]
    )
    assert {row["max_http_attempts"] for row in rows} == {4}


def test_v1_5_longitudinal_transport_contract_is_separate_and_self_hashed() -> None:
    module = _load_script_module(
        "pm_v1_5_longitudinal_transport_contract",
        "v1_5/06_run_action_sweep_v1_5.py",
    )
    contract = module._longitudinal_transport_execution_contract()
    body = {key: value for key, value in contract.items() if key != "contract_sha256"}

    assert contract["transport_max_attempts_per_call"] == 4
    assert contract["transport_retry_policy"] == "bounded_transport"
    assert contract["continue_after_isolated_terminal_failure"] is True
    assert contract["terminal_content_failures_are_not_blindly_retried"] is True
    assert contract["legacy_config_request_retries"] == 1
    assert contract["legacy_config_fail_fast"] is True
    assert contract["scientific_treatment_unchanged"] is True
    assert contract["retryable_classes"] == [
        "http_5xx",
        "network_timeout",
        "rate_limited_429",
        "request_timeout_408",
    ]
    assert contract["contract_sha256"] == sha256_text(canonical_json(body))


def test_bounded_sweep_circuit_breaker_stops_repeated_same_class_failures(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])

    class RepeatedTerminalClient:
        calls = 0

        def __init__(self, endpoint):
            self.endpoint = endpoint

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            type(self).calls += 1
            raise ProviderRequestError(
                status_code=401,
                detail="fixture terminal failure",
                schema_mode=False,
            )

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(
        sweep_module, "OpenAICompatibleClient", RepeatedTerminalClient
    )
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="circuit breaker"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            out_dir / "outcomes.jsonl",
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=Endpoint(
                "https://invalid.example", "never-called", "UNSET", family="test"
            ),
            action_filter={"M0+R0", "M0+RS", "MP+R0"},
            request_retries=1,
            transport_max_attempts_per_call=4,
            fail_fast=False,
            transport_retry_policy="bounded_transport",
            transport_backoff_seconds=(0.0, 0.0, 0.0),
            consecutive_same_class_circuit_breaker=2,
        )

    assert RepeatedTerminalClient.calls == 2
    events = [
        row["event"]
        for row in iter_jsonl(out_dir / "physical_attempt_ledger.jsonl")
    ]
    assert events == ["STARTED", "FAILED", "STARTED", "FAILED"]


def test_action_sweep_missing_api_key_does_not_reserve_attempt(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])

    key_name = "PM_V2_TEST_MISSING_SWEEP_KEY"
    monkeypatch.delenv(key_name, raising=False)
    endpoint = Endpoint(
        "https://invalid.example", "fixture", key_name, family="test"
    )
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match=key_name):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            out_dir / "outcomes.jsonl",
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=endpoint,
            action_filter={"M0+R0"},
            request_retries=1,
            fail_fast=True,
        )
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    assert not ledger_path.exists() or list(iter_jsonl(ledger_path)) == []


def test_action_sweep_reported_input_overrun_is_ledgered_then_aborts(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])

    class OverrunClient:
        calls = 0

        def __init__(self, endpoint):
            pass

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            type(self).calls += 1
            return (
                CallResult(
                    text="A response",
                    raw_response={"fixture": True},
                    usage={
                        "prompt_tokens": 999_999,
                        "completion_tokens": 2,
                        "total_tokens": 1_000_001,
                    },
                    latency_ms=1.0,
                    request_hash="request-overrun",
                ),
                None,
            )

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", OverrunClient)
    endpoint = Endpoint("https://invalid.example", "fixture", "UNSET", family="test")
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            out_dir / "outcomes.jsonl",
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=endpoint,
            action_filter={"M0+R0"},
            request_retries=1,
            input_token_safety_factor=1.5,
            fail_on_reported_input_overrun=True,
        )
    assert OverrunClient.calls == 1
    assert not (out_dir / "outcomes.jsonl").exists()
    ledger = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger] == ["STARTED", "FAILED"]
    assert ledger[-1]["usage"]["prompt_tokens"] == 999_999
    summary = read_json(out_dir / "summary.json")
    assert summary["aborted_on_input_token_overrun"] is True
    assert summary["failures"][-1]["reported_input_token_overrun"] is True


def test_action_sweep_missing_provider_usage_is_ledgered_then_aborts(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])

    class MissingUsageClient:
        calls = 0

        def __init__(self, endpoint):
            pass

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            type(self).calls += 1
            return (
                CallResult(
                    text="A response",
                    raw_response={"fixture": True},
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    latency_ms=1.0,
                    request_hash="request-missing-usage",
                ),
                None,
            )

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", MissingUsageClient)
    endpoint = Endpoint("https://invalid.example", "fixture", "UNSET", family="test")
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            out_dir / "outcomes.jsonl",
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=endpoint,
            action_filter={"M0+R0"},
            request_retries=1,
            fail_fast=True,
        )
    assert MissingUsageClient.calls == 1
    ledger = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger] == ["STARTED", "FAILED"]
    assert ledger[-1]["usage"]["prompt_tokens"] == 0
    summary = read_json(out_dir / "summary.json")
    assert summary["aborted_on_missing_reported_usage"] is True
    assert summary["failures"][-1]["missing_reported_usage"] is True
    assert not (out_dir / "outcomes.jsonl").exists()


def test_action_sweep_recovers_paid_success_from_ledger_without_second_call(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])

    class SuccessfulClient:
        calls = 0

        def __init__(self, endpoint):
            pass

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            type(self).calls += 1
            return (
                CallResult(
                    text="A grounded and supportive response.",
                    raw_response={"fixture": True},
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 8,
                        "total_tokens": 108,
                    },
                    latency_ms=1.0,
                    request_hash="request-success-crash-recovery",
                ),
                None,
            )

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", SuccessfulClient)
    real_append = sweep_module.append_jsonl
    out_dir = tmp_path / "out"
    outcome_path = out_dir / "outcomes.jsonl"
    crashed = {"value": False}

    def crash_after_success_ledger(path, row):
        if Path(path) == outcome_path and not crashed["value"]:
            crashed["value"] = True
            raise OSError("simulated crash after SUCCEEDED ledger event")
        return real_append(path, row)

    monkeypatch.setattr(sweep_module, "append_jsonl", crash_after_success_ledger)
    endpoint = Endpoint("https://invalid.example", "fixture", "UNSET", family="test")
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            outcome_path,
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=endpoint,
            action_filter={"M0+R0"},
            request_retries=1,
            fail_fast=True,
        )
    assert SuccessfulClient.calls == 1
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows] == ["STARTED", "SUCCEEDED"]
    assert ledger_rows[-1]["result"]["action_outcome"]["action_id"] == "M0+R0"

    monkeypatch.setattr(sweep_module, "append_jsonl", real_append)
    summary = run_action_sweep(
        runtime,
        backend,
        strategies,
        outcome_path,
        out_dir / "raw.jsonl",
        out_dir / "summary.json",
        endpoint=endpoint,
        action_filter={"M0+R0"},
        request_retries=1,
        fail_fast=True,
    )
    assert SuccessfulClient.calls == 1
    assert summary["status"] == "COMPLETE"
    assert summary["new_physical_http_attempts"] == 0
    assert summary["historical_physical_http_attempts"] == 1
    assert len(list(iter_jsonl(outcome_path))) == 1


def test_action_sweep_fail_fast_stops_after_first_failed_logical_call(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])

    class AlwaysFailClient:
        calls = 0

        def __init__(self, endpoint):
            self.endpoint = endpoint

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            type(self).calls += 1
            raise RuntimeError("fixture failure")

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", AlwaysFailClient)
    endpoint = Endpoint("https://invalid.example", "never-called", "UNSET", family="test")
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            out_dir / "outcomes.jsonl",
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=endpoint,
            request_retries=1,
            fail_fast=True,
        )
    assert AlwaysFailClient.calls == 1
    summary = read_json(out_dir / "summary.json")
    assert summary["aborted_on_first_failure"] is True
    assert summary["fail_fast"] is True
    assert len(list(iter_jsonl(out_dir / "raw.jsonl"))) == 1
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows] == ["STARTED", "FAILED"]
    assert ledger_rows[0]["attempt_key"] == ledger_rows[1]["attempt_key"]
    assert summary["new_physical_http_attempts"] == 1
    assert summary["total_physical_http_attempts"] == 1

    # Resume consumes the historical STARTED reservation.  The failed call has
    # no attempt slot left and must never be paid for a second time.
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            out_dir / "outcomes.jsonl",
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=endpoint,
            request_retries=1,
            fail_fast=True,
        )
    assert AlwaysFailClient.calls == 1
    resumed_summary = read_json(out_dir / "summary.json")
    assert resumed_summary["new_physical_http_attempts"] == 0
    assert resumed_summary["historical_physical_http_attempts"] == 1
    assert resumed_summary["total_physical_http_attempts"] == 1
    assert list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl")) == ledger_rows

    # Neither run nor dry-run overwrite may erase a spent attempt ledger.
    with pytest.raises(RuntimeError, match="refuses --overwrite"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            out_dir / "outcomes.jsonl",
            out_dir / "raw.jsonl",
            out_dir / "summary.json",
            endpoint=endpoint,
            request_retries=1,
            fail_fast=True,
            overwrite=True,
        )
    assert AlwaysFailClient.calls == 1
    assert list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl")) == ledger_rows


def test_action_sweep_cost_hash_binds_pmv2_contract_and_abstention(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    endpoint = Endpoint("https://invalid.example", "never-called", "UNSET", family="test")

    first, _ = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        memory_min_score=0.0,
        strategy_min_score=0.0,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
        contract_bindings={"pm_v2_config_sha256": "a" * 64},
    )
    second, _ = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        memory_min_score=0.0,
        strategy_min_score=0.0,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
        contract_bindings={"pm_v2_config_sha256": "b" * 64},
    )

    assert first["contract_bindings"]["pm_v2_config_sha256"] == "a" * 64
    assert first["cost_estimate_sha256"] != second["cost_estimate_sha256"]


def test_pmv2_script06_locks_one_physical_attempt_in_config(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [tiny_state.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(
                card_id=tiny_state.card_id, items=tiny_memories
            ).model_dump(mode="json")
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    pm_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/pm_v2.yaml").read_text(encoding="utf-8")
    )
    pm_config["development_sweep"]["request_retries"] = 1
    pm_config["development_sweep"]["fail_fast"] = True
    pm_config_path = tmp_path / "pm_v2.yaml"
    pm_config_path.write_text(
        yaml.safe_dump(pm_config, sort_keys=False), encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/06_run_action_sweep.py"),
        "--dry-run",
        "--config",
        str(PROJECT_ROOT / "configs/experiment.yaml"),
        "--pm-v2-config",
        str(pm_config_path),
        "--runtime",
        str(runtime),
        "--backend",
        str(backend),
        "--strategy-bank",
        str(strategies),
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        "100",
        "--max-estimated-usd",
        "100",
    ]
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    estimate = json.loads(
        (out_dir / "cost_estimate.json").read_text(encoding="utf-8")
    )
    assert estimate["request_retries"] == 1
    assert estimate["fail_fast"] is True
    assert estimate["input_token_safety_factor"] == 1.5
    assert estimate["fail_on_reported_input_overrun"] is True
    assert estimate["pricing"] == {
        "input_usd_per_mtok": 0.15,
        "output_usd_per_mtok": 0.60,
    }
    assert estimate["maximum_physical_api_attempts"] == estimate[
        "expected_api_calls"
    ]
    assert estimate["expected_api_calls"] <= estimate["logical_api_calls"]
    assert estimate["budget_gate"]["checks"]["physical_api_attempts"]
    assert estimate["contract_bindings"]["development_sweep"][
        "request_retries"
    ] == 1
    assert estimate["contract_bindings"]["development_sweep"]["fail_fast"] is True

    override = subprocess.run(
        [*command, "--request-retries", "2"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert override.returncode != 0
    assert "request_retries override differs" in override.stderr

    pricing_override = subprocess.run(
        [*command, "--input-usd-per-mtok", "0"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert pricing_override.returncode != 0
    assert "input pricing override differs" in pricing_override.stderr

    pm_config["development_sweep"]["request_retries"] = 2
    invalid_config = tmp_path / "pm_v2_invalid.yaml"
    invalid_config.write_text(
        yaml.safe_dump(pm_config, sort_keys=False), encoding="utf-8"
    )
    invalid = subprocess.run(
        [
            *command[: command.index("--pm-v2-config") + 1],
            str(invalid_config),
            *command[command.index("--runtime") :],
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid.returncode != 0
    assert "request_retries must equal 1" in invalid.stderr

    pm_config["development_sweep"]["request_retries"] = 1
    pm_config["development_sweep"]["fail_fast"] = False
    non_fail_fast_config = tmp_path / "pm_v2_non_fail_fast.yaml"
    non_fail_fast_config.write_text(
        yaml.safe_dump(pm_config, sort_keys=False), encoding="utf-8"
    )
    non_fail_fast = subprocess.run(
        [
            *command[: command.index("--pm-v2-config") + 1],
            str(non_fail_fast_config),
            *command[command.index("--runtime") :],
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert non_fail_fast.returncode != 0
    assert "fail_fast must be true" in non_fail_fast.stderr

    pmv2_runtime = tiny_state.model_dump(mode="json")
    pmv2_runtime.setdefault("provenance", {})["pm_v2_state_id"] = "state_fixture"
    write_jsonl(runtime, [pmv2_runtime])
    for ad_hoc_filter in (
        ["--actions", "M0+R0"],
        ["--max-cards", "999999"],
    ):
        bypass = subprocess.run(
            [*command, *ad_hoc_filter],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert bypass.returncode != 0
        assert "require the exact frozen --pilot-plan" in bypass.stderr
