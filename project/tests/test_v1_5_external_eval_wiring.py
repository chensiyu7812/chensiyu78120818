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
import sys
import tempfile
from pathlib import Path

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.evoemo import (
    FIXED_SEEKER_V23_STAGE,
    fixed_seeker_cost_planning_contract,
    load_evoemo,
)
from metacom_pm.fixed_seeker_contract import (
    FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL,
    require_fixed_seeker_v3_sidecar_contract,
)
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.response_mechanism_contract import build_response_mechanism_contract
from metacom_pm.io import (
    canonical_json,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.internal_holdout import (
    begin_internal_test_consumption,
    finish_internal_test_consumption,
    freeze_candidate_manifest,
    seal_internal_label_bundle,
)
from metacom_pm.pm_v22_reference_baselines import PMV22_REFERENCE_BASELINE_STAGE
from metacom_pm.pm_v1_5_semantic import (
    SemanticEncoderBinding,
    semantic_encoder_spec_from_config,
)
from metacom_pm.v1_5_judge_isolation import require_judge_role_isolation

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
def workdir() -> Path:
    """Give every pytest process an isolated directory.

    Codex review sessions commonly run this file concurrently in the shared
    checkout.  A repository-relative directory keyed only by the test name let
    one process remove another process's live fixture.
    """

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="external_wiring_", dir=FIXTURE_ROOT
    ) as directory:
        yield Path(directory)


def _placeholder(path: Path, content: str = "placeholder") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _seed_lineage_fixture(workdir: Path) -> tuple[Path, Path]:
    """Build the exact public-lineage seed inputs without a private run bundle."""

    split_manifest = (
        ROOT / "data" / "strategy" / "esconv_split_manifest_v1_5.jsonl"
    )
    split_rows = [
        json.loads(line)
        for line in split_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row
        for row in split_rows
        if row.get("split") == "train"
        and not bool(row.get("excluded_for_evoemo_overlap"))
    ][:875]
    assert len(selected) == 875
    seeds = workdir / "train_seed_dialogues_v1_5.jsonl"
    write_jsonl(
        seeds,
        (
            {
                "dialogue_id": str(row["dialogue_id"]),
                "source_split": "train",
                "excluded_for_evoemo_overlap": False,
            }
            for row in selected
        ),
    )
    audit = workdir / "train_seed_dialogues_v1_5.jsonl.audit.json"
    write_json(
        audit,
        {
            "status": "COMPLETE",
            "output_sha256": sha256_file(seeds),
            "selected_unique_train_seeds": 875,
            "evoemo_overlap_excluded_total": 84,
            "train_manifest": {"file_sha256": sha256_file(split_manifest)},
            "split_manifest": {"sha256": sha256_file(split_manifest)},
        },
    )
    return seeds, audit


def _write_formal_bundle_binding(
    binding_path: Path,
    *,
    bundle_dir: Path,
    stage: str = FIXED_SEEKER_V23_STAGE,
    expected_tracks: int,
    expected_logical_calls: int,
) -> Path:
    """Tracked, content-addressed binding for a test-built formal bundle.

    Mirrors the real data/pm_v1_5_contracts/fixed_seeker_v3_formal_bundle_v1.
    json shape exactly, so consumers under test locate and verify the bundle
    the same way they would locate the real one -- never via a hardcoded
    output-directory basename.
    """

    payload = {
        "protocol": FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL,
        "stage": stage,
        "output_directory": str(bundle_dir),
        "approval_identity": "test-fixture-identity",
        "sidecar_sha256": sha256_file(
            ROOT / "configs" / "pm_v1_5_fixed_seeker_v3.json"
        ),
        "call_plan_sha256": "0" * 64,
        "expected_tracks": expected_tracks,
        "expected_logical_calls": expected_logical_calls,
        "fixed_seeker_tracks_sha256": sha256_file(
            bundle_dir / "fixed_seeker_tracks.jsonl"
        ),
        "artifact_attestation_sha256": sha256_file(
            bundle_dir / "artifact_attestation.json"
        ),
    }
    payload["binding_sha256"] = sha256_text(canonical_json(payload))
    write_json(binding_path, payload)
    return binding_path


def _fixed_track_fixture(workdir: Path) -> tuple[Path, Path, Path]:
    bundle = workdir / "evoemo_fixed_tracks_v1_5_v3_formal_candidate"
    bundle.mkdir(parents=True)
    experiment = load_config(ROOT / "configs" / "experiment.yaml")
    pm_config = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    # V3 is read only from the separately-tracked sidecar, never from
    # configs/pm_v1_5.yaml (which stays on the historical V2 treatment) --
    # matching exactly what the real, migrated V1.5 consumers now do.
    contract = require_fixed_seeker_v3_sidecar_contract(
        ROOT / "configs" / "pm_v1_5_fixed_seeker_v3.json"
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
            "expected_logical_calls": len(rows) * max_turns,
            "successful_logical_calls": len(rows) * max_turns,
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
        stage=FIXED_SEEKER_V23_STAGE,
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
    binding = bundle.parent / "fixed_seeker_v3_formal_bundle_binding.json"
    _write_formal_bundle_binding(
        binding,
        bundle_dir=bundle,
        expected_tracks=len(rows),
        expected_logical_calls=len(rows) * max_turns,
    )
    return tracks, attestation, binding


def _esconv_v1_5_freeze_fixture(
    workdir: Path,
    *,
    pm_checkpoint: Path,
    transparent_rule_checkpoint: Path,
    training_report: Path,
    development_observable_support: dict,
) -> dict[str, Path]:
    config = load_config(ROOT / "configs" / "pm_v1_5.yaml")[
        "esconv_external_evaluation"
    ]
    bundle = workdir / "esconv_test_v1_5"
    bundle.mkdir()
    paths = {
        "runtime_states": bundle / "runtime_states.jsonl",
        "pm_v2_states": bundle / "pm_v2_states.jsonl",
        "memory_backend": bundle / "memory_backend.jsonl",
        "audit_only": bundle / "audit_only.jsonl",
        "split_audit": bundle / "split_audit.json",
    }
    state_count = int(config["expected_primary_states"])
    write_jsonl(
        paths["runtime_states"],
        (
            {
                "state_id": f"esconv-state-{index}",
                "card_id": f"esconv-card-{index}",
            }
            for index in range(state_count)
        ),
    )
    write_jsonl(
        paths["pm_v2_states"],
        ({"state_id": f"esconv-state-{index}"} for index in range(state_count)),
    )
    write_jsonl(
        paths["memory_backend"],
        ({"state_id": f"esconv-state-{index}"} for index in range(state_count)),
    )
    write_jsonl(
        paths["audit_only"],
        ({"state_id": f"esconv-state-{index}"} for index in range(state_count)),
    )
    write_json(
        paths["split_audit"],
        {
            "status": "PASS",
            "protocol": config["split_protocol"],
            "outcomes_used_for_split": False,
            "eligible_test_dialogues": config["nonoverlap_dialogue_counts"][
                "test"
            ],
            "checks": {"fixture_isolation": True},
        },
    )
    build_report = bundle / "build_report.json"
    write_json(
        build_report,
        {
            "status": "COMPLETE",
            "protocol": config["protocol"],
            "turn_selection_protocol": config["turn_selection_protocol"],
            "same_pm_checkpoint_required": True,
            "retraining_or_esconv_outcome_tuning_authorized": False,
            "turn_selection_uses_response_or_judge_outcomes": False,
            "all_support_eligible_test_turns_included": True,
            "legal_actions": config["legal_actions"],
            "test_dialogues": config["nonoverlap_dialogue_counts"]["test"],
            "test_turns": state_count,
            "observable_state_support": {"status": "PASS"},
            "development_observable_state_support_sha256": sha256_text(
                canonical_json(development_observable_support)
            ),
            "outputs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
        },
    )
    policy_dir = workdir / "esconv_v1_5_preflight"
    policy_dir.mkdir()
    choices = policy_dir / "policy_choices.jsonl"
    write_jsonl(
        choices,
        (
            {
                "state_id": f"esconv-state-{index}",
                "learned_action": "M0+R0",
                "transparent_rule_action": "M0+RS",
                "always_r0_action": "M0+R0",
                "always_rs_action": "M0+RS",
                "esconv_outcome_used_for_choice": False,
            }
            for index in range(state_count)
        ),
    )
    summary = policy_dir / "summary.json"
    write_json(
        summary,
        {
            "status": "COMPLETE",
            "protocol": config["protocol"],
            "same_frozen_checkpoint": True,
            "esconv_train_validation_or_test_outcomes_used_for_choice": False,
            "learned_checkpoint_sha256": sha256_file(pm_checkpoint),
            "transparent_rule_checkpoint_sha256": sha256_file(
                transparent_rule_checkpoint
            ),
            "training_report_sha256": sha256_file(training_report),
            "pm_v2_states_sha256": sha256_file(paths["pm_v2_states"]),
            "policy_choices_sha256": sha256_file(choices),
            "state_count": state_count,
        },
    )
    return {
        "build_report": build_report,
        **paths,
        "policy_summary": summary,
        "policy_choices": choices,
    }


def _build_freeze(workdir: Path, monkeypatch, *, pm_checkpoint_content: str = "pm-checkpoint"):
    module = _load_module("scripts/v1_5_create_freeze.py", "v1_5_create_freeze_test")
    # These are contract-wiring tests, not model-distribution tests.  CI uses a
    # clean, offline Hugging Face cache, so bind the exact frozen spec/tree to a
    # deterministic fixture record without resolving or loading model weights.
    # Production freeze creation still calls the real local-only resolver and
    # fails closed if the pinned snapshot is absent or its tree hash drifts.
    semantic_spec = semantic_encoder_spec_from_config(
        load_config(ROOT / "configs" / "pm_v1_5.yaml")
    )
    fixture_semantic_binding = SemanticEncoderBinding(
        spec_sha256=semantic_spec.digest(),
        snapshot_tree_sha256=semantic_spec.snapshot_tree_sha256,
        snapshot_file_count=1,
        implementation="transformers-auto-model-cls-float32",
    )

    def _fixture_semantic_resolver(requested_spec):
        assert requested_spec == semantic_spec
        return workdir / "offline-semantic-snapshot-fixture", fixture_semantic_binding

    monkeypatch.setattr(
        module,
        "resolve_semantic_encoder_binding",
        _fixture_semantic_resolver,
    )
    pm_checkpoint = _placeholder(workdir / "pm_v1_5.joblib", pm_checkpoint_content)
    states = workdir / "pm_v2_states.jsonl"
    labels = workdir / "action_labels.jsonl"
    train_calibration_labels = workdir / "action_labels_train_calibration.jsonl"
    internal_test_labels = workdir / "action_labels_internal_test.jsonl"
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
    write_jsonl(
        train_calibration_labels,
        ({"fixture_label": index} for index in range(5_184)),
    )
    write_jsonl(
        internal_test_labels,
        (
            {"state_id": f"fixture_state_{index // 16}", "fixture_label": index}
            for index in range(2_304)
        ),
    )
    transparent_rule_checkpoint = _placeholder(
        workdir / "pm_v1_5_transparent_rule.joblib"
    )
    no_step0_checkpoint = _placeholder(workdir / "pm_v1_5_no_step0.joblib")
    no_state_bge_checkpoint = _placeholder(
        workdir / "pm_v1_5_no_state_bge.joblib"
    )
    lexical_only_checkpoint = _placeholder(
        workdir / "pm_v1_5_lexical_only.joblib"
    )
    rule_grid_report = workdir / "rule_grid_report.json"
    write_json(
        rule_grid_report,
        {
            "protocol": "pm-v1.5-transparent-rule-grid-diagnostic-v1",
            "status": "PASS",
            "outcome_labels_used": False,
            "internal_states_used": False,
            "selection_or_retuning_authorized": False,
            "pm_v1_5_config_sha256": sha256_file(
                ROOT / "configs" / "pm_v1_5.yaml"
            ),
            "states_sha256": sha256_file(states),
        },
    )
    rule_grid_attestation = workdir / "rule_grid_attestation.json"
    create_artifact_attestation(
        rule_grid_attestation,
        stage="pm_v1_5_pre_training_rule_grid_diagnostic",
        inputs={
            "pm_v1_5_config": ROOT / "configs" / "pm_v1_5.yaml",
            "states": states,
        },
        outputs={"rule_grid_report": (rule_grid_report, False)},
        parameters={"outcome_labels_used": False},
    )
    candidate_manifest = workdir / "candidate_manifest.json"
    sealed_internal_bundle = workdir / "sealed_internal_bundle.json"
    seal_internal_label_bundle(
        sealed_internal_bundle,
        internal_labels_path=internal_test_labels,
    )
    freeze_candidate_manifest(
        candidate_manifest,
        run_identity="fixture-protocol-repair-run",
        artifacts={
            "primary_checkpoint": pm_checkpoint,
            "transparent_rule_checkpoint": transparent_rule_checkpoint,
            "no_step0_checkpoint": no_step0_checkpoint,
            "no_state_bge_checkpoint": no_state_bge_checkpoint,
            "lexical_only_checkpoint": lexical_only_checkpoint,
            "rule_grid_preflight_report": rule_grid_report,
            "rule_grid_preflight_attestation": rule_grid_attestation,
            "sealed_internal_bundle": sealed_internal_bundle,
        },
        parameters={
            "semantic_diagnostics_protocol": "pm-v1.5-semantic-diagnostics-v1",
            "live_training_runtime_contract_sha256": sha256_text(
                canonical_json(
                    load_config(ROOT / "configs" / "pm_v1_5.yaml")[
                        "semantic_runtime"
                    ]
                )
            ),
            "internal_ablation_results_may_select_candidate": False,
        },
    )
    internal_consumption_ledger = workdir / "internal_consumption.jsonl"
    begin_internal_test_consumption(
        internal_consumption_ledger,
        candidate_manifest_path=candidate_manifest,
        internal_labels_path=internal_test_labels,
    )
    development_observable_support = {
        "protocol": "pm-v1.5-development-external-observable-state-support-v1",
        "status": "PASS",
        "outcome_labels_used": False,
        "evoemo_content_used": False,
        "history_turn_targets": [2, 4, 6, 8],
        "summary_treatments": ["present", "absent"],
    }
    training_report = workdir / "training_report.json"
    write_json(
        training_report,
        {
            "status": "COMPLETE",
            "checkpoint": str(pm_checkpoint),
            "checkpoint_sha256": sha256_file(pm_checkpoint),
            "pm_v2_config_sha256": sha256_file(ROOT / "configs" / "pm_v1_5.yaml"),
            "states_sha256": sha256_file(states),
            "train_calibration_labels_sha256": sha256_file(
                train_calibration_labels
            ),
            "internal_test_labels_sha256": sha256_file(internal_test_labels),
            "candidate_manifest_sha256": sha256_file(candidate_manifest),
            "pre_training_rule_grid_diagnostic": read_json(rule_grid_report),
            "pre_training_rule_grid_attestation_sha256": read_json(
                rule_grid_attestation
            )["attestation_sha256"],
            "transparent_rule_checkpoint_sha256": sha256_file(
                transparent_rule_checkpoint
            ),
            "no_step0_checkpoint_sha256": sha256_file(no_step0_checkpoint),
            "no_state_bge_checkpoint_sha256": sha256_file(
                no_state_bge_checkpoint
            ),
            "lexical_only_checkpoint_sha256": sha256_file(
                lexical_only_checkpoint
            ),
            "semantic_runtime_verification": {
                "status": "PASS",
                "contract": load_config(ROOT / "configs" / "pm_v1_5.yaml")[
                    "semantic_runtime"
                ],
                "contract_sha256": sha256_text(
                    canonical_json(
                        load_config(ROOT / "configs" / "pm_v1_5.yaml")[
                            "semantic_runtime"
                        ]
                    )
                ),
            },
            "live_training_runtime_verification": {
                "status": "PASS",
                "contract": load_config(ROOT / "configs" / "pm_v1_5.yaml")[
                    "semantic_runtime"
                ],
                "contract_sha256": sha256_text(
                    canonical_json(
                        load_config(ROOT / "configs" / "pm_v1_5.yaml")[
                            "semantic_runtime"
                        ]
                    )
                ),
            },
            "internal_ablation_results_may_select_or_retune_candidate": False,
            "step0_shortcut_audit": {"status": "PASS"},
            "selected_routing_algorithm": "state_centered_paired_delta_hgb",
            "algorithm_selection": {
                "selected_algorithm": "state_centered_paired_delta_hgb"
            },
            "gate_m": {"status": "PASS"},
            "gate_f": {"status": "PASS"},
            "reportability_checks": {"fixture_gate": True},
            "development_observable_state_support": (
                development_observable_support
            ),
        },
    )
    finish_internal_test_consumption(
        internal_consumption_ledger,
        report_path=training_report,
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
    seed_dialogues, seed_audit = _seed_lineage_fixture(workdir)
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
            "observable_state_support": development_observable_support,
        },
    )
    data_attestation = workdir / "data_attestation.json"
    create_artifact_attestation(
        data_attestation,
        stage="pm_v1_5_development_data",
        inputs={
            "pm_v1_5_config": ROOT / "configs" / "pm_v1_5.yaml",
            "seed_dialogues": seed_dialogues,
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
        "status": "PASS",
        "actual_corpus_review_report_sha256": "c" * 64,
        "actual_corpus_review_attestation_sha256": "d" * 64,
        "step0_shortcut_audit_report_sha256": "e" * 64,
        "step0_shortcut_audit_attestation_sha256": "f" * 64,
    }
    full_sweep_gate = {
        "protocol": "pm-v1.5-full-sweep-gate-v3",
        "status": "PASS",
        "scope": "full",
        "human_calibration_performed": False,
        "actual_corpus_review_report_sha256": "c" * 64,
        "actual_corpus_review_attestation_sha256": "d" * 64,
        "step0_shortcut_audit_report_sha256": "e" * 64,
        "step0_shortcut_audit_attestation_sha256": "f" * 64,
    }
    pm_v1_5_config_for_fixture = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    fixture_supporter_contract = SupporterGenerationContract.from_config(
        pm_v1_5_config_for_fixture
    )
    fixture_generator_endpoint = endpoint_from_config(
        load_config(ROOT / "configs" / "experiment.yaml"),
        fixture_supporter_contract.generator_endpoint,
    )
    fixture_generator_endpoint_sha256 = sha256_text(
        canonical_json(
            {
                "model": fixture_generator_endpoint.model,
                "family": fixture_generator_endpoint.family,
                "base_url": fixture_generator_endpoint.base_url,
            }
        )
    )
    fixture_retrieval = pm_v1_5_config_for_fixture["retrieval"]
    fixture_response_mechanism_contract = build_response_mechanism_contract(
        project_root=ROOT,
        supporter_generation_contract=fixture_supporter_contract,
        generator_endpoint_sha256=fixture_generator_endpoint_sha256,
        strategy_bank_sha256=sha256_file(
            ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"
        ),
        memory_min_score=fixture_retrieval["memory_min_score"],
        strategy_min_score=fixture_retrieval["strategy_min_score"],
        strategy_top_k=fixture_retrieval["strategy_top_k"],
        evidence_filter_enabled=False,
    )
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
        "response_mechanism_contract": fixture_response_mechanism_contract,
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
    judge_role_isolation = require_judge_role_isolation(
        load_config(ROOT / "configs" / "experiment.yaml"),
        load_config(ROOT / "configs" / "pm_v1_5.yaml"),
    )
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
            "judge_role_isolation": judge_role_isolation,
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
            "judge_role_isolation": judge_role_isolation,
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
    fixed_tracks, fixed_tracks_attestation, fixed_tracks_binding = (
        _fixed_track_fixture(workdir)
    )
    esconv_fixture = _esconv_v1_5_freeze_fixture(
        workdir,
        pm_checkpoint=pm_checkpoint,
        transparent_rule_checkpoint=transparent_rule_checkpoint,
        training_report=training_report,
        development_observable_support=development_observable_support,
    )
    out = workdir / "pm_v1_5_study_freeze.json"

    argv = [
        "v1_5_create_freeze.py",
        "--config", str(ROOT / "configs" / "experiment.yaml"),
        "--pm-v1-5-config", str(ROOT / "configs" / "pm_v1_5.yaml"),
        "--evoemo", str(ROOT / "data" / "external" / "evo_emo.json"),
        "--strategy-bank", str(ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl"),
        "--seed-dialogues", str(seed_dialogues),
        "--seed-audit", str(seed_audit),
        "--fixed-tracks-bundle-binding", str(fixed_tracks_binding),
        "--pm-checkpoint", str(pm_checkpoint),
        "--pm-training-report", str(training_report),
        "--candidate-manifest", str(candidate_manifest),
        "--internal-consumption-ledger", str(internal_consumption_ledger),
        "--transparent-rule-checkpoint", str(transparent_rule_checkpoint),
        "--no-step0-checkpoint", str(no_step0_checkpoint),
        "--no-state-bge-checkpoint", str(no_state_bge_checkpoint),
        "--lexical-only-checkpoint", str(lexical_only_checkpoint),
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
        "--train-calibration-labels", str(train_calibration_labels),
        "--internal-test-labels", str(internal_test_labels),
        "--judging-attestation", str(judging_attestation),
        "--decision-quality-report", str(decision_quality_report),
        "--decision-quality-attestation", str(decision_quality_attestation),
        "--esconv-build-report", str(esconv_fixture["build_report"]),
        "--esconv-runtime-states", str(esconv_fixture["runtime_states"]),
        "--esconv-pm-states", str(esconv_fixture["pm_v2_states"]),
        "--esconv-memory-backend", str(esconv_fixture["memory_backend"]),
        "--esconv-audit-only", str(esconv_fixture["audit_only"]),
        "--esconv-split-audit", str(esconv_fixture["split_audit"]),
        "--esconv-policy-summary", str(esconv_fixture["policy_summary"]),
        "--esconv-policy-choices", str(esconv_fixture["policy_choices"]),
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
        "evo_memory_builder_contract_sha256": generation[
            "evo_memory_builder_contract_sha256"
        ],
        "evo_memory_global_catalog_sha256": generation[
            "evo_memory_global_catalog_sha256"
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
    esconv_binding = notes["esconv_external_binding"]

    assert esconv_binding["status"] == "PASS"
    assert esconv_binding["test_dialogues"] == 169
    assert esconv_binding["test_turns"] == 2112
    assert esconv_binding["same_checkpoint_sha256"] == sha256_file(
        checkpoints["pm_checkpoint"]
    )
    assert esconv_binding["external_outcomes_used_for_tuning"] is False

    esconv_generation_contract = notes["esconv_generation_contract"]
    assert esconv_generation_contract["protocol"] == (
        "pm-v1.5-esconv-action-first-generation-contract-v1"
    )
    assert esconv_generation_contract["legal_actions"] == ["M0+R0", "M0+RS"]
    assert esconv_generation_contract["expected_state_count"] == 2112
    # Action-first: exactly 2 legal actions per state, never one generation
    # per policy condition (which would wrongly be 2112 * 4 = 8448).
    assert esconv_generation_contract["expected_logical_action_outcomes"] == (
        2112 * 2
    )
    assert esconv_generation_contract["audit_only_referenced"] is False
    # The ESConv contract reuses the exact same response_mechanism_contract
    # object already bound to the internal sweep and EvoEmo external
    # generation -- not a separately-built one that merely happens to match.
    assert (
        esconv_generation_contract["response_mechanism_contract"]
        == generation_contract["response_mechanism_contract"]
    )

    assert notes["retrieval_consistency"] == {
        "status": "PASS",
        "protocol": "pm-v1.5-development-external-retrieval-lock-v1",
        "strategy_top_k": 3,
        "memory_min_score": 0.0,
        "strategy_min_score": 0.0,
    }
    semantic_spec = semantic_encoder_spec_from_config(
        load_config(ROOT / "configs" / "pm_v1_5.yaml")
    )
    assert generation_contract["semantic_encoder_spec"] == semantic_spec.model_dump(
        mode="json"
    )
    assert generation_contract["semantic_encoder_binding"] == (
        SemanticEncoderBinding(
            spec_sha256=semantic_spec.digest(),
            snapshot_tree_sha256=semantic_spec.snapshot_tree_sha256,
            snapshot_file_count=1,
            implementation="transformers-auto-model-cls-float32",
        ).model_dump(mode="json")
    )
    config = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    assert generation_contract["semantic_runtime_contract"] == config[
        "semantic_runtime"
    ]
    assert notes["bank_seed_lineage"]["strategy_bank_cards"] == 11_590
    assert notes["bank_seed_lineage"]["strategy_source_dialogues"] == 823
    assert notes["bank_seed_lineage"]["selected_seed_sources"] == 52
    assert notes["bank_seed_lineage"]["seed_strategy_source_intersection"] == []
    assert notes["bank_seed_lineage"]["excluded_esconv_sources"] == 84
    assert notes["bank_seed_lineage"]["seed_dialogues"] == 875
    assert notes["bank_seed_lineage"]["deterministic_source_findings"] == 0
    assert notes["full_development_chain"]["states"] == 468
    assert notes["full_development_chain"]["actions_per_state"] == 16
    assert notes["full_development_chain"]["outcomes"] == 7_488
    assert notes["full_development_chain"]["labels"] == 7_488

    assert external_contract["treatment"] == "pm_v2"
    assert set(external_contract["conditions"]) == {
        "pm_v2",
        "pm_v1_5_transparent_rule_step0",
        "pm_v2_cost_matched_fixed",
        "pm_v2_me_r0_fixed",
        "best_fixed",
    }
    assert set(external_contract["secondary_conditions"]) == {
        "no_memory_r0",
        "session_rag_rs",
        "full_history_rs",
    }
    smoke = external_contract["pointwise_schema_smoke"]
    assert smoke["expected_calls"] == 4
    assert len(smoke["judge_endpoints"]) == 2
    assert smoke["legacy_pointwise_fallback_only"] is True
    assert smoke["required_before_full_external_client_creation"] is False
    batched_pilot = external_contract["batched_schema_order_pilot"]
    assert batched_pilot["expected_calls"] == 24
    assert len(batched_pilot["units"]) == 3
    assert batched_pilot["judge_types"] == ["quality", "risk"]
    assert batched_pilot["required_before_full_external_client_creation"] is True
    batched = external_contract["batched_evaluation"]
    assert batched["candidate_count"] == 5
    assert batched["gate_e"]["observed_gate_m"] == "PASS"
    assert batched["gate_e"]["observed_gate_f"] == "PASS"
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


def _response_mechanism_base_kwargs() -> dict:
    pm_v1_5_config = load_config(ROOT / "configs" / "pm_v1_5.yaml")
    supporter_contract = SupporterGenerationContract.from_config(pm_v1_5_config)
    retrieval = pm_v1_5_config["retrieval"]
    return {
        "project_root": ROOT,
        "supporter_generation_contract": supporter_contract,
        "generator_endpoint_sha256": "a" * 64,
        "strategy_bank_sha256": "b" * 64,
        "memory_min_score": retrieval["memory_min_score"],
        "strategy_min_score": retrieval["strategy_min_score"],
        "strategy_top_k": retrieval["strategy_top_k"],
        "evidence_filter_enabled": False,
    }


@pytest.mark.parametrize(
    "override,expected_match",
    [
        (
            {"generator_endpoint_sha256": "c" * 64},
            "generator endpoint differs",
        ),  # model
        (
            {"strategy_bank_sha256": "d" * 64},
            "response mechanism contract",
        ),  # Bank
        ({"strategy_top_k": 99}, "response mechanism contract"),  # top-k
        (
            {"evidence_filter_enabled": True},
            "response mechanism contract",
        ),  # filter
    ],
)
def test_require_v1_5_response_mechanism_consistency_fails_closed_on_drift(
    override, expected_match
):
    module = _load_module(
        "scripts/v1_5_create_freeze.py", "v1_5_response_mechanism_drift_test"
    )
    base_kwargs = _response_mechanism_base_kwargs()
    freeze_contract = build_response_mechanism_contract(**base_kwargs)
    sweep_contract = build_response_mechanism_contract(**{**base_kwargs, **override})
    with pytest.raises(RuntimeError, match=expected_match):
        module.require_v1_5_response_mechanism_consistency(
            freeze_contract=freeze_contract, sweep_contract=sweep_contract
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "src/metacom_pm/prompts.py",  # prompt
        "src/metacom_pm/retrieval.py",  # query builder
    ],
)
def test_require_v1_5_response_mechanism_consistency_fails_closed_on_code_drift(
    tmp_path, relative_path
):
    import shutil

    from metacom_pm.response_mechanism_contract import MECHANISM_CODE_RELATIVE_PATHS

    module = _load_module(
        "scripts/v1_5_create_freeze.py", "v1_5_response_mechanism_code_drift_test"
    )
    fake_root = tmp_path / "project"
    (fake_root / "src" / "metacom_pm").mkdir(parents=True)
    for relative in MECHANISM_CODE_RELATIVE_PATHS:
        dest = fake_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)

    base_kwargs = {**_response_mechanism_base_kwargs(), "project_root": fake_root}
    freeze_contract = build_response_mechanism_contract(**base_kwargs)

    perturbed_path = fake_root / relative_path
    perturbed_path.write_text(
        perturbed_path.read_text(encoding="utf-8") + "\n# perturbed\n",
        encoding="utf-8",
    )
    sweep_contract = build_response_mechanism_contract(**base_kwargs)
    with pytest.raises(RuntimeError, match="response mechanism contract"):
        module.require_v1_5_response_mechanism_consistency(
            freeze_contract=freeze_contract, sweep_contract=sweep_contract
        )


def test_esconv_generation_contract_expected_keys_change_with_legal_actions_or_states(
    tmp_path,
):
    module = _load_module(
        "scripts/v1_5_create_freeze.py", "v1_5_esconv_generation_contract_test"
    )
    states_path = tmp_path / "runtime_states.jsonl"
    write_jsonl(
        states_path,
        ({"card_id": f"card_{i}"} for i in range(5)),
    )
    choices_path = tmp_path / "policy_choices.jsonl"
    write_jsonl(choices_path, [{"dummy": 1}])
    base_kwargs = dict(
        response_mechanism_contract={"contract_sha256": "a" * 64},
        runtime_states_path=states_path,
        policy_choices_path=choices_path,
        legal_actions=("M0+R0", "M0+RS"),
    )
    baseline = module.build_v1_5_esconv_generation_contract(**base_kwargs)
    assert baseline["expected_state_count"] == 5
    assert baseline["expected_logical_action_outcomes"] == 10

    fewer_actions = module.build_v1_5_esconv_generation_contract(
        **{**base_kwargs, "legal_actions": ("M0+R0",)}
    )
    assert fewer_actions["expected_logical_action_outcomes"] == 5
    assert (
        fewer_actions["expected_action_keys_sha256"]
        != baseline["expected_action_keys_sha256"]
    )

    more_states_path = tmp_path / "runtime_states_more.jsonl"
    write_jsonl(
        more_states_path,
        ({"card_id": f"card_{i}"} for i in range(6)),
    )
    more_states = module.build_v1_5_esconv_generation_contract(
        **{**base_kwargs, "runtime_states_path": more_states_path}
    )
    assert more_states["expected_state_count"] == 6
    assert (
        more_states["expected_action_keys_sha256"]
        != baseline["expected_action_keys_sha256"]
    )


def test_v1_5_external_generation_defaults_are_condition_isolated():
    module = _load_module(
        "scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py",
        "v1_5_condition_paths",
    )
    learned_checkpoint, learned_out = module.resolve_condition_paths(
        "pm_v2", checkpoint=None, out_dir=None
    )
    rule_checkpoint, rule_out = module.resolve_condition_paths(
        "pm_v1_5_transparent_rule_step0", checkpoint=None, out_dir=None
    )
    fixed_checkpoint, fixed_out = module.resolve_condition_paths(
        "pm_v2_cost_matched_fixed", checkpoint=None, out_dir=None
    )
    me_checkpoint, me_out = module.resolve_condition_paths(
        "pm_v2_me_r0_fixed", checkpoint=None, out_dir=None
    )
    assert learned_checkpoint.name == "pm_v1_5.joblib"
    assert rule_checkpoint.name == "pm_v1_5_transparent_rule.joblib"
    assert fixed_checkpoint.name == "cost_matched_fixed.joblib"
    assert me_checkpoint.name == "me_r0_fixed.joblib"
    assert len({learned_out, rule_out, fixed_out, me_out}) == 4
    assert all(
        "pm_v1_5" in str(path)
        for path in (learned_out, rule_out, fixed_out, me_out)
    )


def test_v1_5_judging_requires_honest_full_sweep_binding():
    module = _load_module(
        "scripts/v1_5/21_judge_pm_v2_action_sweep_v1_5.py",
        "v1_5_full_sweep_binding",
    )
    actual_report_sha = "c" * 64
    actual_attestation_sha = "d" * 64
    shortcut_report_sha = "e" * 64
    shortcut_attestation_sha = "f" * 64
    gate = {
        "protocol": "pm-v1.5-full-sweep-gate-v3",
        "status": "PASS",
        "scope": "full",
        "human_calibration_performed": False,
        "actual_corpus_review_attestation_sha256": actual_attestation_sha,
        "actual_corpus_review_report_sha256": actual_report_sha,
        "step0_shortcut_audit_report_sha256": shortcut_report_sha,
        "step0_shortcut_audit_attestation_sha256": shortcut_attestation_sha,
    }
    chain = {"contract_bindings": {"scope": "full", "v1_5_full_sweep_gate": gate}}
    assert module.require_v1_5_full_sweep_binding(
        chain,
        actual_corpus_review_report_sha256=actual_report_sha,
        actual_corpus_review_attestation_sha256=actual_attestation_sha,
        step0_shortcut_audit_report_sha256=shortcut_report_sha,
        step0_shortcut_audit_attestation_sha256=shortcut_attestation_sha,
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
            actual_corpus_review_report_sha256=actual_report_sha,
            actual_corpus_review_attestation_sha256=actual_attestation_sha,
            step0_shortcut_audit_report_sha256=shortcut_report_sha,
            step0_shortcut_audit_attestation_sha256=shortcut_attestation_sha,
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


def test_v1_5_external_eval_rejects_stale_evo_memory_catalog(workdir, monkeypatch):
    # Guards the checkpoint added to close the gap where a memory-builder
    # change after the freeze (chunking, splitting, id scheme) could
    # otherwise slip past external evaluation undetected: evoemo_sha256
    # alone only pins the raw input file, not what build_evo_memory
    # actually constructs from it.
    out, checkpoints = _build_freeze(workdir, monkeypatch)
    freeze = json.loads(out.read_text(encoding="utf-8"))
    freeze_sha = freeze["freeze_sha256"]
    generation_contract = freeze["notes"]["generation_contract"]

    parameters = {
        "condition": "pm_v2",
        "supporter_generation_treatment": generation_contract["supporter_generation_treatment"],
        "supporter_generation_treatment_sha256": generation_contract[
            "supporter_generation_treatment_sha256"
        ],
        "evo_memory_builder_contract_sha256": "0" * 64,  # tampered
        "evo_memory_global_catalog_sha256": generation_contract[
            "evo_memory_global_catalog_sha256"
        ],
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
        "scripts/v1_5/25_eval_pm_v2_external_v1_5.py", "v1_5_external_eval_test_evo_memory"
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
    with pytest.raises(RuntimeError, match="violates frozen evo_memory_builder_contract_sha256"):
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
        "pm_v1_5_transparent_rule_step0",
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
    assert calls[0]["pilot_units"] == sorted(excluded)[:3]
    assert calls[0]["contract"]["expected_calls"] == 24


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
        "pm_v1_5_transparent_rule_step0",
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
                "evo_memory_builder_contract_sha256": generation[
                    "evo_memory_builder_contract_sha256"
                ],
                "evo_memory_global_catalog_sha256": generation[
                    "evo_memory_global_catalog_sha256"
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
        "build_separated_resource_report",
        lambda *a, **k: {
            "status": "COMPLETE",
            "protocol": "pm-v1.5-separated-resource-accounting-v1",
            "aggregation": "separate_metrics_no_composite_cost",
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
            "pilot_units": [list(unit) for unit in sorted(excluded)[:3]],
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


def _pre_v3_fixed_seeker_bundle(workdir: Path) -> Path:
    """A bare, pre-V3 (V22-stage) bundle plus its own tracked binding.

    Every migrated consumer now locates and verifies its formal bundle only
    through a content-addressed binding, never a hardcoded output-directory
    basename -- so the reverse-direction check is that a binding honestly
    declaring an old V2/V22-stage bundle is still refused, exactly like
    tests/test_v1_5_fixed_seeker_v3_readiness.py's unit-level
    ``test_require_fixed_seeker_v3_formal_bundle_fails_closed_on_old_v2_binding``,
    but exercised end-to-end through each real consumer script.
    """

    old_style_dir = workdir / "evoemo_fixed_tracks_v1_5"
    old_style_dir.mkdir(parents=True)
    fixed_tracks = old_style_dir / "fixed_seeker_tracks.jsonl"
    fixed_tracks.write_text('{"track_id": "old-v2"}\n', encoding="utf-8")
    fixed_tracks_attestation = old_style_dir / "artifact_attestation.json"
    write_json(fixed_tracks_attestation, {"stage": "evoemo_fixed_seeker_tracks_v22"})
    write_json(
        old_style_dir / "summary.json",
        {
            "expected_tracks": 1,
            "completed_tracks": 1,
            "expected_logical_calls": 1,
            "successful_logical_calls": 1,
        },
    )
    binding = workdir / "old_v2_bundle_binding.json"
    payload = {
        "protocol": FIXED_SEEKER_V3_FORMAL_BUNDLE_BINDING_PROTOCOL,
        "stage": "evoemo_fixed_seeker_tracks_v22",
        "output_directory": str(old_style_dir),
        "approval_identity": "test-old-v2-identity",
        "sidecar_sha256": sha256_file(
            ROOT / "configs" / "pm_v1_5_fixed_seeker_v3.json"
        ),
        "call_plan_sha256": "0" * 64,
        "expected_tracks": 1,
        "expected_logical_calls": 1,
        "fixed_seeker_tracks_sha256": sha256_file(fixed_tracks),
        "artifact_attestation_sha256": sha256_file(fixed_tracks_attestation),
    }
    payload["binding_sha256"] = sha256_text(canonical_json(payload))
    write_json(binding, payload)
    return binding


def test_v1_5_freeze_rejects_pre_v3_fixed_seeker_bundle_directory_name(
    workdir, monkeypatch
):
    """Reverse-direction check for the fixed-seeker V3 atomic migration:

    scripts/v1_5_create_freeze.py must still refuse a binding that honestly
    declares an old V2/V22-stage bundle, not silently promote it.
    """

    module = _load_module(
        "scripts/v1_5_create_freeze.py", "v1_5_create_freeze_reverse_test"
    )
    binding = _pre_v3_fixed_seeker_bundle(workdir)
    placeholder = lambda name: str(workdir / name)  # noqa: E731
    argv = [
        "v1_5_create_freeze.py",
        "--fixed-tracks-bundle-binding", str(binding),
        "--pm-checkpoint", placeholder("pm.joblib"),
        "--pm-training-report", placeholder("training_report.json"),
        "--candidate-manifest", placeholder("candidate_manifest.json"),
        "--internal-consumption-ledger", placeholder("consumption_ledger.jsonl"),
        "--transparent-rule-checkpoint", placeholder("transparent_rule.joblib"),
        "--no-step0-checkpoint", placeholder("no_step0.joblib"),
        "--no-state-bge-checkpoint", placeholder("no_state_bge.joblib"),
        "--lexical-only-checkpoint", placeholder("lexical_only.joblib"),
        "--cost-matched-fixed-checkpoint", placeholder("cost_matched_fixed.joblib"),
        "--me-r0-fixed-checkpoint", placeholder("me_r0_fixed.joblib"),
        "--out", placeholder("study_freeze.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="binding stage does not match"):
        module.main()


def test_v1_5_evoemo_generation_rejects_pre_v3_fixed_seeker_bundle_directory_name(
    workdir, monkeypatch
):
    """Same reverse-direction check for scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py."""

    module = _load_module(
        "scripts/v1_5/24_run_pm_v2_evoemo_v1_5.py",
        "v1_5_24_run_pm_v2_evoemo_reverse_test",
    )
    binding = _pre_v3_fixed_seeker_bundle(workdir)
    argv = [
        "24_run_pm_v2_evoemo_v1_5.py",
        "--dry-run",
        "--fixed-tracks-bundle-binding", str(binding),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="binding stage does not match"):
        module.main()


def test_v1_5_reference_baselines_rejects_pre_v3_fixed_seeker_bundle_directory_name(
    workdir, monkeypatch
):
    """Same reverse-direction check for

    scripts/v1_5/24a_run_pmv22_reference_baselines_v1_5.py.
    """

    module = _load_module(
        "scripts/v1_5/24a_run_pmv22_reference_baselines_v1_5.py",
        "v1_5_24a_run_pmv22_reference_baselines_reverse_test",
    )
    binding = _pre_v3_fixed_seeker_bundle(workdir)
    argv = [
        "24a_run_pmv22_reference_baselines_v1_5.py",
        "--dry-run",
        "--fixed-tracks-bundle-binding", str(binding),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="binding stage does not match"):
        module.main()


def test_run_pmv2_fixed_evoemo_rejects_an_unsupported_fixed_seeker_stage_value():
    """Reverse-direction check for the shared runner's new

    fixed_seeker_required_stage parameter (pm_v2_evoemo.run_pmv2_fixed_evoemo):
    only the two known stages may ever be requested, never an arbitrary or
    mistyped string that could silently accept anything.
    """

    from metacom_pm.pm_v2_evoemo import run_pmv2_fixed_evoemo

    with pytest.raises(ValueError, match="unsupported fixed-seeker required stage"):
        run_pmv2_fixed_evoemo(
            "evoemo.json",
            "strategy.jsonl",
            "checkpoint.joblib",
            "fixed_tracks.jsonl",
            "out_dir",
            project_root=ROOT,
            generator_endpoint=None,
            supporter_generation_contract=None,
            fixed_seeker_generation_contract={},
            fixed_seeker_generation_contract_sha256="",
            simulator_id="sim",
            fixed_seeker_required_stage="not_a_real_stage",
            evaluation_unit_contract={},
            action_preflight_gates={},
            maximum_cost_matched_relative_deviation=0.1,
            input_usd_per_mtok=0.15,
            output_usd_per_mtok=0.60,
        )
