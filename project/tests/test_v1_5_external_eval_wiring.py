"""End-to-end, no-API wiring tests for the PM-v1.5 freeze -> external-eval chain.

These do not re-test src/metacom_pm/pm_v2_external_eval.py's internals (already
covered by tests/test_pm_v2_eval_hardening.py); they test that this session's
new scripts (scripts/v1_5_create_freeze.py, scripts/v1_5/25_eval_pm_v2_external_v1_5.py,
scripts/v1_5/35_run_pm_v2_external_pointwise_schema_smoke_v1_5.py) assemble the
legacy fallback and the current
scripts/v1_5/36_run_external_batched_schema_order_pilot_v1_5.py path, and assemble the
freeze/treatment/policy-lock consistency checks correctly and fail closed on
tampered or mismatched inputs, using real configs/pm_v1_5.yaml and the real
EvoEmo/strategy-bank data (both small enough to load directly; no fixtures to
maintain in parallel with the real schema).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evoemo import (
    FIXED_SEEKER_V22_STAGE,
    fixed_seeker_cost_planning_contract,
    load_evoemo,
)
from metacom_pm.fixed_seeker_contract import FixedSeekerGenerationContract
from metacom_pm.io import (
    canonical_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v22_reference_baselines import PMV22_REFERENCE_BASELINE_STAGE

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "outputs" / "v1_5_test_fixtures"


def _load_module(relpath: str, name: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def workdir(request):
    d = FIXTURE_ROOT / request.node.name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _placeholder(path: Path, content: str = "placeholder") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _fixed_track_fixture(workdir: Path) -> tuple[Path, Path]:
    bundle = workdir / "evoemo_fixed_tracks_v1_5"
    bundle.mkdir(parents=True)
    experiment = load_config(ROOT / "configs" / "experiment.yaml")
    pm_config = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    contract = FixedSeekerGenerationContract.from_mapping(
        pm_config["fixed_seeker_generation_treatment"]
    )
    endpoint = endpoint_from_config(experiment, contract.seeker_endpoint)
    bound = contract.bind_endpoint(contract.seeker_endpoint, endpoint)
    treatment = bound.payload()
    treatment_sha256 = bound.digest()
    cost_planning = fixed_seeker_cost_planning_contract(
        pm_config["fixed_seeker_cost_planning"]
    )
    cost_planning_sha256 = sha256_text(canonical_json(cost_planning))
    seeds = [int(value) for value in experiment["protocol"]["robustness_seeds"]]
    simulator_id = str(pm_config["external_evaluation"]["simulator_id"])
    max_turns = int(pm_config["external_evaluation"]["max_turns"])
    rows = [
        {
            "track_id": f"fixture-{user['id']}-{topic['idx']}-{seed}",
            "user_id": str(user["id"]),
            "topic_index": int(topic["idx"]),
            "seed": seed,
            "simulator_id": simulator_id,
            "seeker_turns": ["fixture seeker turn"] * max_turns,
            "fixed_seeker_generation_contract": treatment,
            "fixed_seeker_generation_contract_sha256": treatment_sha256,
            "fixed_seeker_cost_planning": cost_planning,
            "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
        }
        for user in load_evoemo(ROOT / "data" / "external" / "evo_emo.json")
        for topic in (user.get("subsequent_topics") or [])
        for seed in seeds
    ]
    tracks = bundle / "fixed_seeker_tracks.jsonl"
    write_jsonl(tracks, rows)
    write_json(bundle / "run_manifest.json", {"fixture": True})
    write_json(bundle / "cost_estimate.json", {"fixture": True})
    write_jsonl(bundle / "call_plan.jsonl", [])
    write_jsonl(bundle / "raw_seeker_calls.jsonl", [])
    write_jsonl(bundle / "physical_attempt_ledger.jsonl", [])
    budget_gate = {"status": "PASS", "checks": {"fixture": True}}
    summary = bundle / "summary.json"
    write_json(
        summary,
        {
            "status": "COMPLETE",
            "expected_tracks": len(rows),
            "completed_tracks": len(rows),
            "max_turns": max_turns,
            "completion_truncated_count": 0,
            "failures": [],
            "fixed_seeker_generation_contract": treatment,
            "fixed_seeker_generation_contract_sha256": treatment_sha256,
            "fixed_seeker_cost_planning": cost_planning,
            "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
            "planned_budget_gate": budget_gate,
            "observed_budget_gate": budget_gate,
        },
    )
    attestation = bundle / "artifact_attestation.json"
    create_artifact_attestation(
        attestation,
        stage=FIXED_SEEKER_V22_STAGE,
        inputs={
            "evoemo": ROOT / "data" / "external" / "evo_emo.json",
            "run_manifest": bundle / "run_manifest.json",
            "cost_estimate": bundle / "cost_estimate.json",
            "call_plan": bundle / "call_plan.jsonl",
        },
        outputs={
            "tracks": (tracks, True),
            "raw_calls": (bundle / "raw_seeker_calls.jsonl", True),
            "physical_attempt_ledger": (
                bundle / "physical_attempt_ledger.jsonl",
                True,
            ),
            "summary": (summary, False),
        },
        parameters={
            "simulator_id": simulator_id,
            "max_turns": max_turns,
            "seeds": seeds,
            "fixed_seeker_generation_contract": treatment,
            "fixed_seeker_generation_contract_sha256": treatment_sha256,
            "fixed_seeker_cost_planning": cost_planning,
            "fixed_seeker_cost_planning_sha256": cost_planning_sha256,
        },
        expected={
            "tracks": len(rows),
            "turns_per_track": max_turns,
            "completion_truncated_count": 0,
            "planned_budget_gate_status": "PASS",
            "observed_budget_gate_status": "PASS",
        },
    )
    return tracks, attestation


def _build_freeze(workdir: Path, monkeypatch, *, pm_checkpoint_content: str = "pm-checkpoint"):
    module = _load_module("scripts/v1_5_create_freeze.py", "v1_5_create_freeze_test")
    pm_checkpoint = _placeholder(workdir / "pm_v1_5.joblib", pm_checkpoint_content)
    states = workdir / "pm_v2_states.jsonl"
    labels = workdir / "action_labels.jsonl"
    runtime = workdir / "runtime_states.jsonl"
    backend = workdir / "memory_backend.jsonl"
    evaluator_contexts = workdir / "evaluator_contexts.jsonl"
    outcomes = workdir / "action_outcomes.jsonl"
    write_jsonl(states, ({"fixture_state": index} for index in range(468)))
    write_jsonl(runtime, ({"fixture_runtime": index} for index in range(468)))
    write_jsonl(backend, ({"fixture_backend": index} for index in range(468)))
    write_jsonl(
        evaluator_contexts,
        ({"fixture_context": index} for index in range(468)),
    )
    write_jsonl(
        outcomes, ({"fixture_outcome": index} for index in range(7_488))
    )
    write_jsonl(labels, ({"fixture_label": index} for index in range(7_488)))
    training_report = workdir / "training_report.json"
    write_json(
        training_report,
        {
            "status": "COMPLETE",
            "checkpoint": str(pm_checkpoint),
            "checkpoint_sha256": sha256_file(pm_checkpoint),
            "pm_v2_config_sha256": sha256_file(ROOT / "configs" / "pm_v1_5.yaml"),
            "states_sha256": sha256_file(states),
            "labels_sha256": sha256_file(labels),
            "require_learned_routing_advantage_before_external": True,
            "learned_routing_advantage_verified": True,
            "reportability_checks": {"fixture_gate": True},
        },
    )
    cost_matched_checkpoint = _placeholder(workdir / "cost_matched_fixed.joblib")
    me_r0_checkpoint = _placeholder(workdir / "me_r0_fixed.joblib")
    fixed_baselines_report = workdir / "fixed_baselines.json"
    write_json(
        fixed_baselines_report,
        {
            "status": "COMPLETE",
            "track": "pm-v1.5",
            "source_checkpoint_sha256": sha256_file(pm_checkpoint),
            "training_report_sha256": sha256_file(training_report),
            "baselines": {
                "cost_matched_fixed": {
                    "action_id": "M0+R0",
                    "checkpoint_sha256": sha256_file(cost_matched_checkpoint),
                },
                "event_memory_r0": {
                    "action_id": "ME+R0",
                    "checkpoint_sha256": sha256_file(me_r0_checkpoint),
                },
            },
        },
    )

    data_report = workdir / "data_report.json"
    expected_split_states = {
        "train": 216,
        "calibration": 108,
        "internal_test": 144,
    }
    write_json(
        data_report,
        {
            "status": "COMPLETE",
            "n_users": 52,
            "n_states": 468,
            "split_counts": expected_split_states,
            "all_states_have_16_actions": True,
            "full_state_design": {
                "status": "PASS",
                "cases_per_user": 9,
                "expected_split_state_counts": expected_split_states,
                "expected_total_states": 468,
            },
        },
    )
    data_attestation = workdir / "data_attestation.json"
    create_artifact_attestation(
        data_attestation,
        stage="pm_v1_5_development_data",
        inputs={
            "pm_v1_5_config": ROOT / "configs" / "pm_v1_5.yaml",
            "seed_dialogues": ROOT
            / "data"
            / "pm_v2"
            / "train_seed_dialogues_v1_5.jsonl",
            "strategy_bank": ROOT
            / "data"
            / "strategy"
            / "strategy_cards_v1_5.jsonl",
        },
        outputs={
            "states": (states, True),
            "runtime": (runtime, True),
            "backend": (backend, True),
            "evaluator_contexts": (evaluator_contexts, True),
            "data_report": (data_report, False),
        },
        parameters={"track": "pm-v1.5"},
        expected={"states": 468, "users": 52},
    )
    semantic_review = {
        "status": "PASS_VIA_AUTOMATED_MULTI_FAMILY_REVIEW",
        "automated_review_report_sha256": "a" * 64,
        "automated_review_attestation_sha256": "b" * 64,
    }
    full_sweep_gate = {
        "protocol": "pm-v1.5-full-sweep-gate-v1",
        "status": "PASS",
        "scope": "full",
        "human_calibration_performed": False,
        "automated_review_report_sha256": "a" * 64,
        "automated_review_attestation_sha256": "b" * 64,
    }
    sweep_bindings = {
        "scope": "full",
        "pm_v2_config_sha256": sha256_file(
            ROOT / "configs" / "pm_v1_5.yaml"
        ),
        "pm_v2_version": "pm-v1.5",
        "retrieval": load_config(ROOT / "configs" / "pm_v1_5.yaml")[
            "retrieval"
        ],
        "evidence_filter": {"enabled": False},
        "evidence_filter_model": {"mode": "disabled_for_pm_v1_5"},
        "semantic_sanity": semantic_review,
        "v1_5_full_sweep_gate": full_sweep_gate,
    }
    sweep_summary = workdir / "sweep_summary.json"
    write_json(
        sweep_summary,
        {
            "status": "COMPLETE",
            "n_cards": 468,
            "expected_outcomes": 7_488,
            "completed_outcomes": 7_488,
            "failures": [],
            "strategy_bank_sha256": sha256_file(
                ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
            ),
            "contract_bindings": sweep_bindings,
        },
    )
    sweep_attestation = workdir / "sweep_attestation.json"
    create_artifact_attestation(
        sweep_attestation,
        stage="action_sweep",
        inputs={
            "runtime": runtime,
            "backend": backend,
            "strategy_bank": ROOT
            / "data"
            / "strategy"
            / "strategy_cards_v1_5.jsonl",
        },
        outputs={
            "summary": (sweep_summary, False),
            "action_outcomes": (outcomes, True),
        },
        parameters={
            "max_cards": None,
            "action_filter": None,
            "card_filter": None,
            "contract_bindings": sweep_bindings,
        },
        expected={"cards": 468, "outcomes": 7_488},
    )
    judging_summary = workdir / "judging_summary.json"
    write_json(
        judging_summary,
        {
            "status": "COMPLETE",
            "scope": "full",
            "reportability_status": "REPORTABLE",
            "outcomes": 7_488,
            "label_rows": 7_488,
            "remaining_judge_pairs": 0,
            "remaining_api_calls": 0,
            "quality_gate": {"status": "PASS"},
            "raw_family_quality_gate": {"status": "PASS"},
        },
    )
    judging_attestation = workdir / "judging_attestation.json"
    create_artifact_attestation(
        judging_attestation,
        stage="pm_v2_action_judging",
        inputs={
            "states": states,
            "outcomes": outcomes,
            "evaluator_contexts": evaluator_contexts,
            "sweep_summary": sweep_summary,
            "sweep_attestation": sweep_attestation,
        },
        outputs={"summary": (judging_summary, False), "labels": (labels, True)},
        parameters={
            "status": "COMPLETE",
            "scope": "full",
            "pm_v2_config_sha256": sha256_file(
                ROOT / "configs" / "pm_v1_5.yaml"
            ),
        },
    )

    decision_quality_report = workdir / "decision_quality_report.json"
    write_json(
        decision_quality_report,
        {
            "status": "COMPLETE",
            "protocol": "pm-v1.5-internal-decision-quality-v1",
            "split": "internal_test",
            "n_states": 1,
            "pm_v1_5_config_sha256": sha256_file(ROOT / "configs" / "pm_v1_5.yaml"),
            "checkpoint_sha256": sha256_file(pm_checkpoint),
            "training_report_sha256": sha256_file(training_report),
            "claim_boundary": "fixture-not-a-claim",
        },
    )
    decision_quality_attestation = workdir / "decision_quality_attestation.json"
    create_artifact_attestation(
        decision_quality_attestation,
        stage="pm_v1_5_decision_quality",
        inputs={"training_report": training_report},
        outputs={"report": (decision_quality_report, False)},
        parameters={"track": "pm-v1.5"},
    )
    fixed_tracks, fixed_tracks_attestation = _fixed_track_fixture(workdir)
    out = workdir / "pm_v1_5_study_freeze.json"

    argv = [
        "v1_5_create_freeze.py",
        "--config", str(ROOT / "configs" / "experiment.yaml"),
        "--pm-v1-5-config", str(ROOT / "configs" / "pm_v1_5.yaml"),
        "--evoemo", str(ROOT / "data" / "external" / "evo_emo.json"),
        "--strategy-bank", str(ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"),
        "--fixed-tracks", str(fixed_tracks),
        "--fixed-tracks-attestation", str(fixed_tracks_attestation),
        "--pm-checkpoint", str(pm_checkpoint),
        "--pm-training-report", str(training_report),
        "--cost-matched-fixed-checkpoint", str(cost_matched_checkpoint),
        "--me-r0-fixed-checkpoint", str(me_r0_checkpoint),
        "--fixed-baselines-report", str(fixed_baselines_report),
        "--development-data-attestation", str(data_attestation),
        "--development-data-report", str(data_report),
        "--states", str(states),
        "--sweep-summary", str(sweep_summary),
        "--sweep-attestation", str(sweep_attestation),
        "--judging-summary", str(judging_summary),
        "--judging-labels", str(labels),
        "--judging-attestation", str(judging_attestation),
        "--decision-quality-report", str(decision_quality_report),
        "--decision-quality-attestation", str(decision_quality_attestation),
        "--out", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return out, {
        "pm_checkpoint": pm_checkpoint,
        "training_report": training_report,
        "cost_matched_checkpoint": cost_matched_checkpoint,
        "me_r0_checkpoint": me_r0_checkpoint,
        "states": states,
        "labels": labels,
    }


def _fake_generation_artifact(
    workdir: Path,
    name: str,
    *,
    stage: str,
    parameters: dict,
    study_freeze_sha256: str | None = None,
    inputs: dict[str, Path] | None = None,
) -> tuple[Path, Path]:
    turns_path = workdir / f"{name}_turns.jsonl"
    write_jsonl(turns_path, [{"placeholder": True}])
    attestation_path = workdir / f"{name}_attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage=stage,
        inputs=inputs
        or {"evoemo": ROOT / "data" / "external" / "evo_emo.json"},
        outputs={"turns": (turns_path, True)},
        parameters=parameters,
        study_freeze_sha256=study_freeze_sha256,
    )
    return turns_path, attestation_path


def _reference_parameters(freeze: dict, *, checkpoint_sha256: str | None = None) -> dict:
    generation = freeze["notes"]["generation_contract"]
    experiment = load_config(ROOT / "configs" / "experiment.yaml")
    supporter = endpoint_from_config(experiment, "generator")
    return {
        "conditions": [
            "no_memory_r0",
            "best_fixed",
            "session_rag_rs",
            "full_history_rs",
        ],
        "supporter_generation_treatment": generation[
            "supporter_generation_treatment"
        ],
        "supporter_generation_treatment_sha256": generation[
            "supporter_generation_treatment_sha256"
        ],
        "fixed_seeker_generation_treatment": generation[
            "fixed_seeker_generation_treatment"
        ],
        "fixed_seeker_generation_treatment_sha256": generation[
            "fixed_seeker_generation_treatment_sha256"
        ],
        "simulator_id": generation["simulator_id"],
        "max_turns": generation["max_turns"],
        "seeds": generation["seeds"],
        "evaluation_unit_contract": generation["evaluation_unit_contract"],
        "paid_generation_scope": generation["paid_generation_scope"],
        "evidence_filter_config_sha256": generation[
            "evidence_filter_config_sha256"
        ],
        "evidence_filter_model": generation["evidence_filter_model"],
        "retrieval_settings": {
            "memory_min_score": float(generation["memory_min_score"]),
            "strategy_min_score": float(generation["strategy_min_score"]),
            "strategy_top_k": int(generation["strategy_top_k"]),
            "session_rag_top_k": int(generation["session_rag_top_k"]),
        },
        "generator_endpoint": {
            "model": supporter.model,
            "family": supporter.family,
            "base_url": supporter.base_url,
        },
        "policy_checkpoint_sha256": (
            checkpoint_sha256
            if checkpoint_sha256 is not None
            else generation["policy_checkpoint_sha256"]
        ),
        "policy_training_report_sha256": generation[
            "policy_training_report_sha256"
        ],
        "policy_lock_timing": generation["policy_lock_timing"],
        "post_generation_policy_tuning_prohibited": generation[
            "post_generation_policy_tuning_prohibited"
        ],
    }


def test_v1_5_freeze_creation_produces_well_formed_external_contract(workdir, monkeypatch):
    out, checkpoints = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    notes = freeze["notes"]
    generation_contract = notes["generation_contract"]
    external_contract = notes["external_evaluation_contract"]

    assert notes["retrieval_consistency"] == {
        "status": "PASS",
        "protocol": "pm-v1.5-development-external-retrieval-lock-v1",
        "strategy_top_k": 3,
        "memory_min_score": 0.0,
        "strategy_min_score": 0.0,
    }
    assert notes["bank_seed_lineage"]["strategy_bank_cards"] == 12_403
    assert notes["bank_seed_lineage"]["excluded_esconv_sources"] == 84
    assert notes["bank_seed_lineage"]["seed_dialogues"] == 875
    assert notes["bank_seed_lineage"]["deterministic_source_findings"] == 0
    assert notes["full_development_chain"]["states"] == 468
    assert notes["full_development_chain"]["actions_per_state"] == 16
    assert notes["full_development_chain"]["outcomes"] == 7_488
    assert notes["full_development_chain"]["labels"] == 7_488

    assert external_contract["treatment"] == "pm_v2"
    assert set(external_contract["conditions"]) >= {
        "pm_v2",
        "pm_v2_cost_matched_fixed",
        "pm_v2_me_r0_fixed",
        "no_memory_r0",
        "best_fixed",
        "session_rag_rs",
        "full_history_rs",
    }
    smoke = external_contract["pointwise_schema_smoke"]
    assert smoke["expected_calls"] == 4
    assert len(smoke["judge_endpoints"]) == 2
    assert smoke["legacy_pointwise_fallback_only"] is True
    assert smoke["required_before_full_external_client_creation"] is False
    batched_pilot = external_contract["batched_schema_order_pilot"]
    assert batched_pilot["expected_calls"] == 8
    assert batched_pilot["judge_types"] == ["quality", "risk"]
    assert batched_pilot["required_before_full_external_client_creation"] is True
    batched = external_contract["batched_evaluation"]
    assert batched["expected_scoring_units"] == 192
    assert batched["quality"]["primary_judge_count_per_unit"] == 1
    assert batched["quality"]["full_two_family_score_pooling"] is False
    assert batched["quality"]["expected_sensitivity_units"] == 54
    assert batched["risk_audit"]["expected_units"] == 36
    assert batched["expected_api_calls"] == 318
    from metacom_pm.v1_5_forced_swap_canary import KEY_CLAIM_GATE

    assert external_contract["key_claim_gate"] == KEY_CLAIM_GATE
    assert len(external_contract["excluded_units"]) == 12
    assert smoke["unit"] == sorted(external_contract["excluded_units"])[0]
    assert external_contract["forced_swap"]["sample_units"] == 12

    from metacom_pm.io import sha256_file

    assert generation_contract["policy_checkpoint_sha256"] == sha256_file(checkpoints["pm_checkpoint"])
    assert generation_contract["policy_training_report_sha256"] == sha256_file(checkpoints["training_report"])
    assert generation_contract["post_generation_policy_tuning_prohibited"] is True
    required_by_forced_swap_driver = {
        "protocol",
        "simulator_id",
        "max_turns",
        "seeds",
        "strategy_action_tokens",
        "strategy_top_k",
        "memory_min_score",
        "strategy_min_score",
        "action_preflight_gates",
        "maximum_cost_matched_relative_deviation",
        "generator_retries",
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
        "evaluation_unit_contract",
        "paid_generation_scope",
    }
    assert required_by_forced_swap_driver <= set(generation_contract)
    assert generation_contract["paid_generation_scope"] == (
        "frozen_evaluation_turns_only"
    )


def test_v1_5_freeze_rejects_development_external_retrieval_drift():
    module = _load_module(
        "scripts/v1_5_create_freeze.py", "v1_5_retrieval_lock_test"
    )
    config = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    config["external_evaluation"]["strategy_top_k"] += 1
    with pytest.raises(RuntimeError, match="configuration-generalization"):
        module.require_v1_5_retrieval_consistency(config)


def test_v1_5_external_generation_defaults_are_condition_isolated():
    module = _load_module(
        "scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py",
        "v1_5_condition_paths",
    )
    learned_checkpoint, learned_out = module.resolve_condition_paths(
        "pm_v2", checkpoint=None, out_dir=None
    )
    fixed_checkpoint, fixed_out = module.resolve_condition_paths(
        "pm_v2_cost_matched_fixed", checkpoint=None, out_dir=None
    )
    me_checkpoint, me_out = module.resolve_condition_paths(
        "pm_v2_me_r0_fixed", checkpoint=None, out_dir=None
    )
    assert learned_checkpoint.name == "pm_v1_5.joblib"
    assert fixed_checkpoint.name == "cost_matched_fixed.joblib"
    assert me_checkpoint.name == "me_r0_fixed.joblib"
    assert len({learned_out, fixed_out, me_out}) == 3
    assert all("pm_v1_5" in str(path) for path in (learned_out, fixed_out, me_out))


def test_v1_5_judging_requires_honest_full_sweep_binding():
    module = _load_module(
        "scripts/v1_5/21_judge_pm_v2_action_sweep_v1_5.py",
        "v1_5_full_sweep_binding",
    )
    report_sha = "a" * 64
    attestation_sha = "b" * 64
    gate = {
        "protocol": "pm-v1.5-full-sweep-gate-v1",
        "status": "PASS",
        "scope": "full",
        "human_calibration_performed": False,
        "automated_review_attestation_sha256": attestation_sha,
        "automated_review_report_sha256": report_sha,
    }
    chain = {"contract_bindings": {"scope": "full", "v1_5_full_sweep_gate": gate}}
    assert module.require_v1_5_full_sweep_binding(
        chain,
        automated_review_report_sha256=report_sha,
        automated_review_attestation_sha256=attestation_sha,
    ) == gate

    stale = {
        "contract_bindings": {
            "scope": "pilot",
            "v1_5_full_sweep_gate": gate,
        }
    }
    with pytest.raises(RuntimeError, match="exact PM-v1.5 full matrix"):
        module.require_v1_5_full_sweep_binding(
            stale,
            automated_review_report_sha256=report_sha,
            automated_review_attestation_sha256=attestation_sha,
        )


def test_v1_5_external_eval_rejects_unknown_generation_stage(workdir, monkeypatch):
    out, checkpoints = _build_freeze(workdir, monkeypatch)
    freeze_sha = json.loads(out.read_text(encoding="utf-8"))["freeze_sha256"]
    turns_path, attestation_path = _fake_generation_artifact(
        workdir,
        "bogus",
        stage="some_unrelated_stage",
        parameters={"condition": "pm_v2"},
        study_freeze_sha256=freeze_sha,
    )

    module = _load_module(
        "scripts/v1_5/25_eval_pm_v2_external_v1_5.py", "v1_5_external_eval_test_a"
    )
    argv = [
        "25_eval_pm_v2_external_v1_5.py",
        "--dry-run",
        "--freeze", str(out),
        "--turn-paths", str(turns_path),
        "--generation-attestations", str(attestation_path),
        "--policy-checkpoint", str(checkpoints["pm_checkpoint"]),
        "--policy-training-report", str(checkpoints["training_report"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="unknown stage"):
        module.main()


def test_v1_5_external_eval_rejects_stale_supporter_treatment(workdir, monkeypatch):
    out, checkpoints = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    freeze_sha = freeze["freeze_sha256"]
    generation_contract = freeze["notes"]["generation_contract"]

    parameters = {
        "condition": "pm_v2",
        "supporter_generation_treatment": generation_contract["supporter_generation_treatment"],
        "supporter_generation_treatment_sha256": "0" * 64,  # tampered
        "fixed_seeker_generation_treatment": generation_contract["fixed_seeker_generation_treatment"],
        "fixed_seeker_generation_treatment_sha256": generation_contract[
            "fixed_seeker_generation_treatment_sha256"
        ],
        "simulator_id": generation_contract["simulator_id"],
        "max_turns": generation_contract["max_turns"],
        "seeds": generation_contract["seeds"],
        "evaluation_unit_contract": generation_contract["evaluation_unit_contract"],
    }
    turns_path, attestation_path = _fake_generation_artifact(
        workdir,
        "pm_v2",
        stage="evoemo_pm_v2_generation",
        parameters=parameters,
        study_freeze_sha256=freeze_sha,
    )

    module = _load_module(
        "scripts/v1_5/25_eval_pm_v2_external_v1_5.py", "v1_5_external_eval_test_b"
    )
    argv = [
        "25_eval_pm_v2_external_v1_5.py",
        "--dry-run",
        "--freeze", str(out),
        "--turn-paths", str(turns_path),
        "--generation-attestations", str(attestation_path),
        "--policy-checkpoint", str(checkpoints["pm_checkpoint"]),
        "--policy-training-report", str(checkpoints["training_report"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="violates frozen supporter_generation_treatment_sha256"):
        module.main()


def test_v1_5_external_eval_rejects_swapped_policy_checkpoint(workdir, monkeypatch):
    out, checkpoints = _build_freeze(workdir, monkeypatch)
    swapped_checkpoint = _placeholder(workdir / "swapped.joblib", "not-the-frozen-checkpoint")

    module = _load_module(
        "scripts/v1_5/25_eval_pm_v2_external_v1_5.py", "v1_5_external_eval_test_c"
    )
    argv = [
        "25_eval_pm_v2_external_v1_5.py",
        "--dry-run",
        "--freeze", str(out),
        "--policy-checkpoint", str(swapped_checkpoint),
        "--policy-training-report", str(checkpoints["training_report"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="does not bind the policy"):
        module.main()


def test_v1_5_external_eval_rejects_reference_baseline_policy_lock_mismatch(workdir, monkeypatch):
    out, checkpoints = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    freeze_sha = freeze["freeze_sha256"]
    generation_contract = freeze["notes"]["generation_contract"]

    parameters = _reference_parameters(freeze, checkpoint_sha256="f" * 64)
    turns_path, attestation_path = _fake_generation_artifact(
        workdir,
        "reference_baselines",
        stage=PMV22_REFERENCE_BASELINE_STAGE,
        parameters=parameters,
        study_freeze_sha256=None,
        inputs={"strategy_bank_approval": out},
    )

    module = _load_module(
        "scripts/v1_5/25_eval_pm_v2_external_v1_5.py", "v1_5_external_eval_test_d"
    )

    argv = [
        "25_eval_pm_v2_external_v1_5.py",
        "--dry-run",
        "--freeze", str(out),
        "--turn-paths", str(turns_path),
        "--generation-attestations", str(attestation_path),
        "--policy-checkpoint", str(checkpoints["pm_checkpoint"]),
        "--policy-training-report", str(checkpoints["training_report"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="different policy lock"):
        module.main()


def test_v1_5_schema_smoke_rejects_changed_judge_endpoint(workdir, monkeypatch):
    out, _checkpoints = _build_freeze(workdir, monkeypatch)

    module = _load_module(
        "scripts/v1_5/35_run_pm_v2_external_pointwise_schema_smoke_v1_5.py",
        "v1_5_schema_smoke_test",
    )

    from metacom_pm.config import Endpoint, load_config
    from metacom_pm.generation_contract import SupporterGenerationContract

    monkeypatch.setattr(
        module,
        "endpoint_from_config",
        lambda config, name: Endpoint(
            "https://changed.invalid/v1", "changed-model", "UNSET", family="changed-family"
        ),
    )

    freeze = json.loads(out.read_text(encoding="utf-8"))
    generation_contract = freeze["notes"]["generation_contract"]
    excluded_units = [
        (str(u[0]), int(u[1]), int(u[2]), str(u[3]), int(u[4]))
        for u in freeze["notes"]["external_evaluation_contract"]["excluded_units"]
    ]
    monkeypatch.setattr(
        module,
        "require_v1_5_forced_swap_canary",
        lambda *args, **kwargs: {"excluded_units": excluded_units},
    )
    supporter_version = SupporterGenerationContract.from_config(
        load_config(ROOT / "configs" / "pm_v1_5.yaml")
    ).version

    # Match the frozen generation contract exactly so the earlier
    # generation-attestation parameter check passes and execution actually
    # reaches the judge-endpoint contract check this test isolates.
    turns_path = workdir / "turns.jsonl"
    write_jsonl(turns_path, [{"placeholder": True}])
    attestation_path = workdir / "attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage="evoemo_pm_v2_generation",
        inputs={"strategy_bank_approval": out},
        outputs={"turns": (turns_path, True)},
        parameters={
            "condition": "pm_v2",
            "protocol": supporter_version,
            "simulator_id": generation_contract["simulator_id"],
            "max_turns": generation_contract["max_turns"],
            "seeds": generation_contract["seeds"],
            "evaluation_unit_contract": generation_contract["evaluation_unit_contract"],
            "paid_generation_scope": "frozen_evaluation_turns_only",
        },
        study_freeze_sha256=freeze["freeze_sha256"],
    )

    argv = [
        "35_run_pm_v2_external_pointwise_schema_smoke_v1_5.py",
        "--dry-run",
        "--freeze", str(out),
        "--turns", str(turns_path),
        "--generation-attestation", str(attestation_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="endpoint changed after study freeze"):
        module.main()


def _schema_smoke_generation_artifact(workdir: Path, freeze: dict) -> tuple[Path, Path]:
    generation_contract = freeze["notes"]["generation_contract"]
    turns_path = workdir / "schema_smoke_turns.jsonl"
    write_jsonl(turns_path, [{"placeholder": True}])
    attestation_path = workdir / "schema_smoke_generation_attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage="evoemo_pm_v2_generation",
        inputs={"evoemo": ROOT / "data" / "external" / "evo_emo.json"},
        outputs={"turns": (turns_path, True)},
        parameters={
            "condition": "pm_v2",
            "protocol": generation_contract["protocol"],
            "simulator_id": generation_contract["simulator_id"],
            "max_turns": generation_contract["max_turns"],
            "seeds": generation_contract["seeds"],
            "evaluation_unit_contract": generation_contract["evaluation_unit_contract"],
            "paid_generation_scope": "frozen_evaluation_turns_only",
        },
        study_freeze_sha256=freeze["freeze_sha256"],
    )
    return turns_path, attestation_path


def test_v1_5_schema_smoke_happy_path_reaches_shared_runner(workdir, monkeypatch):
    out, _ = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    turns_path, attestation_path = _schema_smoke_generation_artifact(workdir, freeze)
    module = _load_module(
        "scripts/v1_5/35_run_pm_v2_external_pointwise_schema_smoke_v1_5.py",
        "v1_5_schema_smoke_happy_path",
    )
    excluded = [
        (str(u[0]), int(u[1]), int(u[2]), str(u[3]), int(u[4]))
        for u in freeze["notes"]["external_evaluation_contract"]["excluded_units"]
    ]
    monkeypatch.setattr(
        module,
        "require_v1_5_forced_swap_canary",
        lambda *args, **kwargs: {"excluded_units": excluded},
    )
    calls = []
    monkeypatch.setattr(
        module,
        "run_external_pointwise_schema_smoke",
        lambda **kwargs: calls.append(kwargs) or {"status": "DRY_RUN_COMPLETE"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "35_run_pm_v2_external_pointwise_schema_smoke_v1_5.py",
            "--dry-run",
            "--freeze",
            str(out),
            "--turns",
            str(turns_path),
            "--generation-attestation",
            str(attestation_path),
        ],
    )
    module.main()
    assert len(calls) == 1
    assert calls[0]["smoke_unit"] == sorted(excluded)[0]


def test_v1_5_schema_smoke_rejects_tampered_generation_output(workdir, monkeypatch):
    out, _ = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    turns_path, attestation_path = _schema_smoke_generation_artifact(workdir, freeze)
    with turns_path.open("a", encoding="utf-8") as handle:
        handle.write('{"tampered":true}\n')
    module = _load_module(
        "scripts/v1_5/35_run_pm_v2_external_pointwise_schema_smoke_v1_5.py",
        "v1_5_schema_smoke_tamper",
    )
    excluded = [
        (str(u[0]), int(u[1]), int(u[2]), str(u[3]), int(u[4]))
        for u in freeze["notes"]["external_evaluation_contract"]["excluded_units"]
    ]
    monkeypatch.setattr(
        module,
        "require_v1_5_forced_swap_canary",
        lambda *args, **kwargs: {"excluded_units": excluded},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "35_run_pm_v2_external_pointwise_schema_smoke_v1_5.py",
            "--dry-run",
            "--freeze",
            str(out),
            "--turns",
            str(turns_path),
            "--generation-attestation",
            str(attestation_path),
        ],
    )
    with pytest.raises(RuntimeError, match="Artifact attestation failed"):
        module.main()


def test_v1_5_batched_schema_pilot_happy_path_reaches_v1_5_runner(
    workdir, monkeypatch
):
    out, _ = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    notes = freeze["notes"]
    generation = notes["generation_contract"]
    external = notes["external_evaluation_contract"]
    turn_paths: list[Path] = []
    attestations: list[Path] = []

    reference_turns = workdir / "batched_reference_turns.jsonl"
    write_jsonl(reference_turns, [{"placeholder": True}])
    reference_attestation = workdir / "batched_reference_attestation.json"
    create_artifact_attestation(
        reference_attestation,
        stage=PMV22_REFERENCE_BASELINE_STAGE,
        inputs={"strategy_bank_approval": out},
        outputs={"turns": (reference_turns, True)},
        parameters={"conditions": [
            "no_memory_r0",
            "best_fixed",
            "session_rag_rs",
            "full_history_rs",
        ]},
    )
    turn_paths.append(reference_turns)
    attestations.append(reference_attestation)
    for condition in (
        "pm_v2",
        "pm_v2_cost_matched_fixed",
        "pm_v2_me_r0_fixed",
    ):
        turns = workdir / f"batched_{condition}_turns.jsonl"
        write_jsonl(turns, [{"placeholder": True}])
        attestation = workdir / f"batched_{condition}_attestation.json"
        create_artifact_attestation(
            attestation,
            stage="evoemo_pm_v2_generation",
            inputs={"freeze": out},
            outputs={"turns": (turns, True)},
            parameters={"condition": condition},
            study_freeze_sha256=freeze["freeze_sha256"],
        )
        turn_paths.append(turns)
        attestations.append(attestation)

    module = _load_module(
        "scripts/v1_5/36_run_external_batched_schema_order_pilot_v1_5.py",
        "v1_5_batched_schema_pilot_happy_path",
    )
    excluded = [
        (str(unit[0]), int(unit[1]), int(unit[2]), str(unit[3]), int(unit[4]))
        for unit in external["excluded_units"]
    ]
    monkeypatch.setattr(
        module,
        "require_v1_5_forced_swap_canary",
        lambda *args, **kwargs: {"excluded_units": excluded},
    )
    monkeypatch.setattr(module, "_require_treatment_bound_turns", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(
        module,
        "run_v1_5_batched_schema_order_pilot",
        lambda **kwargs: calls.append(kwargs) or {"status": "DRY_RUN_COMPLETE"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "36_run_external_batched_schema_order_pilot_v1_5.py",
            "--dry-run",
            "--freeze",
            str(out),
            "--turn-paths",
            *(str(path) for path in turn_paths),
            "--generation-attestations",
            *(str(path) for path in attestations),
            "--out-dir",
            str(workdir / "batched_schema_pilot"),
        ],
    )
    module.main()
    assert len(calls) == 1
    assert calls[0]["conditions"] == external["conditions"]
    assert calls[0]["smoke_unit"] == sorted(excluded)[0]
    assert calls[0]["contract"]["expected_calls"] == 8


def test_v1_5_external_eval_happy_path_reaches_shared_dry_runner(workdir, monkeypatch):
    out, checkpoints = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    freeze_sha = freeze["freeze_sha256"]
    notes = freeze["notes"]
    generation = notes["generation_contract"]
    external = notes["external_evaluation_contract"]
    reference_conditions = [
        "no_memory_r0",
        "best_fixed",
        "session_rag_rs",
        "full_history_rs",
    ]
    turn_paths: list[Path] = []
    attestation_paths: list[Path] = []
    reference_turns = workdir / "reference_turns.jsonl"
    write_jsonl(reference_turns, [{"placeholder": True}])
    reference_attestation = workdir / "reference_attestation.json"
    create_artifact_attestation(
        reference_attestation,
        stage=PMV22_REFERENCE_BASELINE_STAGE,
        inputs={"strategy_bank_approval": out},
        outputs={"turns": (reference_turns, True)},
        parameters=_reference_parameters(freeze),
    )
    turn_paths.append(reference_turns)
    attestation_paths.append(reference_attestation)
    for condition in (
        "pm_v2",
        "pm_v2_cost_matched_fixed",
        "pm_v2_me_r0_fixed",
    ):
        turns = workdir / f"{condition}_turns.jsonl"
        write_jsonl(turns, [{"placeholder": True}])
        attestation = workdir / f"{condition}_attestation.json"
        create_artifact_attestation(
            attestation,
            stage="evoemo_pm_v2_generation",
            inputs={"freeze": out},
            outputs={"turns": (turns, True)},
            parameters={
                "condition": condition,
                "supporter_generation_treatment": generation[
                    "supporter_generation_treatment"
                ],
                "supporter_generation_treatment_sha256": generation[
                    "supporter_generation_treatment_sha256"
                ],
                "fixed_seeker_generation_treatment": generation[
                    "fixed_seeker_generation_treatment"
                ],
                "fixed_seeker_generation_treatment_sha256": generation[
                    "fixed_seeker_generation_treatment_sha256"
                ],
                "simulator_id": generation["simulator_id"],
                "max_turns": generation["max_turns"],
                "seeds": generation["seeds"],
                "evaluation_unit_contract": generation["evaluation_unit_contract"],
                **{
                    key: generation[key]
                    for key in (
                        "protocol",
                        "paid_generation_scope",
                        "action_preflight_gate_scope",
                        "all_turn_action_preflight_is_diagnostic_only",
                        "generator_endpoint_sha256",
                        "strategy_action_tokens",
                        "strategy_top_k",
                        "memory_min_score",
                        "strategy_min_score",
                        "evidence_filter",
                        "evidence_filter_config_sha256",
                        "evidence_filter_model",
                        "action_preflight_gates",
                        "maximum_cost_matched_relative_deviation",
                        "generator_retries",
                        "generator_pricing_usd_per_mtok",
                        "input_token_safety_factor",
                        "fail_on_reported_input_overrun",
                    )
                },
            },
            study_freeze_sha256=freeze_sha,
        )
        turn_paths.append(turns)
        attestation_paths.append(attestation)

    module = _load_module(
        "scripts/v1_5/25_eval_pm_v2_external_v1_5.py",
        "v1_5_external_eval_happy_path",
    )
    excluded = {
        (str(u[0]), int(u[1]), int(u[2]), str(u[3]), int(u[4]))
        for u in external["excluded_units"]
    }
    monkeypatch.setattr(module, "require_treatment_bound_turn_file", lambda *a, **k: [])
    monkeypatch.setattr(
        module,
        "build_descriptive_latency_report",
        lambda *a, **k: {
            "status": "COMPLETE",
            "protocol": "pm-v1.5-noninterleaved-latency-diagnostic-v1",
            "role": "diagnostic_only",
            "confirmatory_latency_claim_allowed": False,
        },
    )
    monkeypatch.setattr(
        module,
        "require_v1_5_forced_swap_canary",
        lambda *a, **k: {"excluded_units": sorted(excluded)},
    )
    monkeypatch.setattr(
        module,
        "require_v1_5_batched_schema_order_pilot_pass",
        lambda *a, **k: {
            "status": "PASS",
            "smoke_unit": list(sorted(excluded)[0]),
        },
    )
    monkeypatch.setattr(
        module,
        "compare_observed_cost_matched_turns",
        lambda *a, **k: {"status": "PASS"},
    )
    calls = []
    monkeypatch.setattr(
        module,
        "run_v1_5_external_batched_evaluation",
        lambda **kwargs: calls.append(kwargs) or {"status": "DRY_RUN_COMPLETE"},
    )
    argv = [
        "25_eval_pm_v2_external_v1_5.py",
        "--dry-run",
        "--freeze",
        str(out),
        "--policy-checkpoint",
        str(checkpoints["pm_checkpoint"]),
        "--policy-training-report",
        str(checkpoints["training_report"]),
        "--turn-paths",
        *(str(path) for path in turn_paths),
        "--generation-attestations",
        *(str(path) for path in attestation_paths),
        # Without this, args.out_dir defaults to the real production path
        # outputs/pm_v1_5_external_response -- the latency-diagnostic write
        # (not monkeypatched, unlike run_external_response_evaluation) would
        # otherwise leave mock artifacts there.
        "--out-dir",
        str(workdir / "external_response"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert len(calls) == 1
    assert calls[0]["conditions"] == external["conditions"]
    assert set(calls[0]["expected_units"]).isdisjoint(excluded)
