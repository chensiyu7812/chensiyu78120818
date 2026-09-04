from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.api import CallResult, Endpoint
from metacom_pm.config import load_config
from metacom_pm.contracts import StrategyCard
from metacom_pm.evidence_filter import EvidenceFilterConfig
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.pm_v22_reference_baselines import (
    PMV22_REFERENCE_BASELINE_STAGE,
    POLICY_LOCK_FIELDS,
    REFERENCE_BASELINE_CONDITIONS,
    build_raw_generation_contract_gate,
    persist_reference_baseline_dry_run,
    plan_reference_baselines,
    run_reference_baselines,
)


ROOT = Path(__file__).resolve().parents[1]


class HelpfulModel:
    threshold = 0.5
    checkpoint_sha256 = "1" * 64

    def predict_helpfulness(self, **_: object) -> float:
        return 1.0

    def contract_hash(self) -> str:
        return "2" * 64


class OneResultClient:
    def __init__(self, result: CallResult):
        self.result = result
        self.calls = 0
        self.closed = False

    def chat(self, *args, **kwargs):
        self.calls += 1
        return self.result, None

    def close(self) -> None:
        self.closed = True


def test_policy_training_producer_writes_consumer_checkpoint_hash_contract() -> None:
    producer = (ROOT / "scripts" / "22_train_pm_v2.py").read_text(
        encoding="utf-8"
    )

    assert '"checkpoint_sha256": sha256_file(checkpoint)' in producer


def _user_and_topic() -> tuple[dict, dict]:
    topic = {
        "idx": 1,
        "topic": "work stress",
        "psychological_condition": "worried",
        "physical_condition": "tired",
        "more_details": "A deadline is approaching.",
    }
    user = {
        "id": "u-test",
        "basic_info": {"occupation": "stressful project manager"},
        "dialog_history": [
            {
                "id": "past-1",
                "timestamp": "2025-01-01",
                "summary": "The user felt stress about a work deadline.",
                "dialogue": [
                    {
                        "role": "seeker",
                        "content": "My work deadline makes me feel stressed.",
                    },
                    {
                        "role": "supporter",
                        "content": "That sounds like a lot of pressure.",
                    },
                ],
            }
        ],
        "subsequent_topics": [topic],
    }
    return user, topic


def _fixed_treatment() -> tuple[dict, str]:
    payload = {
        "treatment": {
            "version": "pm-v2.2-fixed-seeker-generation-v1",
            "max_output_tokens": 300,
            "accepted_normalized_finish_reasons": ["complete"],
        },
        "endpoint": {
            "endpoint_id": "seeker",
            "base_url": "https://example.invalid",
            "model": "fake-seeker",
            "family": "fake-seeker-family",
        },
    }
    from metacom_pm.io import canonical_json, sha256_text

    return payload, sha256_text(canonical_json(payload))


def _write_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    user, topic = _user_and_topic()
    monkeypatch.setattr(
        "metacom_pm.pm_v22_reference_baselines._scenarios",
        lambda _path: [(user, topic)],
    )
    evoemo = tmp_path / "evoemo.json"
    evoemo.write_text("[]\n", encoding="utf-8")
    strategy = tmp_path / "strategy.jsonl"
    write_jsonl(
        strategy,
        [
            StrategyCard(
                strategy_id="strat_aaaaaaaaaaaa",
                strategy_label="reflect",
                retrieval_text="work deadline stress worried",
                guidance_text="Reflect the user's work stress before advice.",
                example_response="It makes sense that this deadline feels heavy.",
                source_dialogue_id="train-dialogue",
                source_turn_index=1,
            ).model_dump(mode="json")
        ],
    )
    fixed_payload, fixed_sha = _fixed_treatment()
    fixed_tracks = tmp_path / "fixed_seeker_tracks.jsonl"
    write_jsonl(
        fixed_tracks,
        [
            {
                "track_id": "track-test",
                "user_id": "u-test",
                "topic_index": 1,
                "seed": 101,
                "simulator_id": "seeker_main",
                "initial_greeting": (
                    "Hi, I'm here with you. What would you like to talk about today?"
                ),
                "seeker_turns": ["I feel stressed about my work deadline."],
                "turn_provenance": [
                    {
                        "turn_index": 1,
                        "normalized_finish_reason": "complete",
                        "fixed_seeker_generation_contract_sha256": fixed_sha,
                    }
                ],
                "fixed_seeker_generation_contract": fixed_payload,
                "fixed_seeker_generation_contract_sha256": fixed_sha,
            }
        ],
    )
    config_path = tmp_path / "pm_v2.yaml"
    config_path.write_text("version: pm-v2.2\n", encoding="utf-8")
    fixed_attestation = tmp_path / "fixed_attestation.json"
    write_json(fixed_attestation, {"status": "test"})
    evidence_checkpoint = tmp_path / "filter.joblib"
    evidence_report = tmp_path / "filter_report.json"
    evidence_attestation = tmp_path / "filter_attestation.json"
    strategy_approval = tmp_path / "strategy_approval.json"
    for path in (
        evidence_checkpoint,
        evidence_report,
        evidence_attestation,
        strategy_approval,
    ):
        path.write_text("test\n", encoding="utf-8")
    policy_checkpoint = tmp_path / "pm_v2.joblib"
    policy_checkpoint.write_bytes(b"locked policy checkpoint")
    policy_training_report = tmp_path / "training_report.json"
    write_json(
        policy_training_report,
        {
            "status": "COMPLETE",
            "checkpoint": str(policy_checkpoint),
            "checkpoint_sha256": sha256_file(policy_checkpoint),
            "require_learned_routing_advantage_before_external": True,
            "learned_routing_advantage_verified": True,
        },
    )
    config = load_config(ROOT / "configs" / "pm_v2.yaml")
    return {
        "evoemo_path": evoemo,
        "strategy_bank_path": strategy,
        "fixed_tracks_path": fixed_tracks,
        "policy_checkpoint_path": policy_checkpoint,
        "policy_training_report_path": policy_training_report,
        "fixed_tracks_attestation_path": fixed_attestation,
        "pm_v2_config_path": config_path,
        "evidence_filter_checkpoint_path": evidence_checkpoint,
        "evidence_filter_report_path": evidence_report,
        "evidence_filter_attestation_path": evidence_attestation,
        "strategy_bank_approval_path": strategy_approval,
        "generator_endpoint": Endpoint(
            base_url="https://example.invalid",
            model="fake-generator",
            api_key_env="NEVER_USED",
            family="fake-generator-family",
        ),
        "supporter_generation_contract": (
            SupporterGenerationContract.from_config(config)
        ),
        "fixed_seeker_generation_treatment": fixed_payload,
        "fixed_seeker_generation_treatment_sha256": fixed_sha,
        "evidence_filter_config": EvidenceFilterConfig.from_mapping(
            config["evidence_filter"]
        ),
        "evidence_filter_model": HelpfulModel(),
        "evidence_filter_model_binding": {
            "checkpoint_sha256": "1" * 64,
            "model_contract_sha256": "2" * 64,
        },
        "simulator_id": "seeker_main",
        "max_turns": 1,
        "seeds": [101],
        "turn_indices": [1],
        "memory_min_score": 0.0,
        "strategy_min_score": 0.0,
        "strategy_top_k": 1,
        "session_rag_top_k": 1,
        "input_token_safety_factor": 1.5,
        "input_usd_per_mtok": 0.15,
        "output_usd_per_mtok": 0.60,
        "max_api_calls": 4,
        "max_estimated_usd": 5.0,
        "max_input_tokens_per_call": 10000,
    }


def _plan_kwargs(values: dict) -> dict:
    omitted = {
        "fixed_tracks_attestation_path",
        "pm_v2_config_path",
        "evidence_filter_checkpoint_path",
        "evidence_filter_report_path",
        "evidence_filter_attestation_path",
        "strategy_bank_approval_path",
    }
    return {key: value for key, value in values.items() if key not in omitted}


def test_dry_run_is_exact_four_condition_sparse_matrix_and_positive_price(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)

    first, plan = plan_reference_baselines(**_plan_kwargs(values))
    second, repeated_plan = plan_reference_baselines(**_plan_kwargs(values))

    assert first == second
    assert plan == repeated_plan
    assert len(plan) == 4
    assert tuple(row["condition"] for row in plan) == REFERENCE_BASELINE_CONDITIONS
    assert first["expected_api_calls"] == 4
    assert first["estimated_cost_usd"] > 0.0
    assert first["budget_gate"]["status"] == "PASS"
    assert first["evaluation_unit_contract"]["expected_unit_count"] == 1
    assert all(row["max_output_tokens"] == 300 for row in plan)
    assert all(row["max_http_attempts"] == 1 for row in plan)
    assert all(
        row["supporter_generation_treatment_sha256"]
        == values["supporter_generation_contract"].digest()
        for row in plan
    )
    assert all(
        row["fixed_seeker_generation_treatment_sha256"]
        == values["fixed_seeker_generation_treatment_sha256"]
        for row in plan
    )
    policy_sha256 = sha256_file(values["policy_checkpoint_path"])
    assert first["policy_checkpoint_sha256"] == policy_sha256
    assert first["policy_lock_timing"] == (
        "before_first_reference_baseline_api_call"
    )
    assert first["post_generation_policy_tuning_prohibited"] is True
    assert all(row["policy_checkpoint_sha256"] == policy_sha256 for row in plan)
    assert all(
        row["post_generation_policy_tuning_prohibited"] is True for row in plan
    )
    contracts = first["evidence_processing_contracts"]
    assert contracts["best_fixed"]["memory_processing"] == (
        "full_shared_supervised_evidence_filter"
    )
    assert contracts["session_rag_rs"]["memory_processing"].startswith(
        "preserve_all_retrieved_raw_session"
    )
    assert contracts["full_history_rs"]["memory_processing"].startswith(
        "preserve_all_authorized_raw_session"
    )


def test_raw_generation_gate_rejects_noncomplete_error_and_mixed_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)
    estimate, plan = plan_reference_baselines(**_plan_kwargs(values))
    raw_rows = [
        {
            "condition": row["condition"],
            "physical_call_key": row["call_key"],
            "normalized_finish_reason": "complete",
            "error": None,
            "supporter_generation_treatment": row[
                "supporter_generation_treatment"
            ],
            "supporter_generation_treatment_sha256": row[
                "supporter_generation_treatment_sha256"
            ],
            "fixed_seeker_generation_treatment": row[
                "fixed_seeker_generation_treatment"
            ],
            "fixed_seeker_generation_treatment_sha256": row[
                "fixed_seeker_generation_treatment_sha256"
            ],
            "evidence_processing_contract": row[
                "evidence_processing_contract"
            ],
            "evidence_processing_contract_sha256": row[
                "evidence_processing_contract_sha256"
            ],
            **{key: row[key] for key in POLICY_LOCK_FIELDS},
        }
        for row in plan
    ]
    kwargs = {
        "call_plan": plan,
        "supporter_generation_treatment": estimate[
            "supporter_generation_treatment"
        ],
        "supporter_generation_treatment_sha256": estimate[
            "supporter_generation_treatment_sha256"
        ],
        "fixed_seeker_generation_treatment": estimate[
            "fixed_seeker_generation_treatment"
        ],
        "fixed_seeker_generation_treatment_sha256": estimate[
            "fixed_seeker_generation_treatment_sha256"
        ],
        "evidence_processing_contracts": estimate[
            "evidence_processing_contracts"
        ],
        "policy_lock": {key: estimate[key] for key in POLICY_LOCK_FIELDS},
    }
    passing = build_raw_generation_contract_gate(raw_rows=raw_rows, **kwargs)
    assert passing["status"] == "PASS"
    assert all(
        passing[key] is True
        for key in (
            "row_count_exact",
            "call_keys_exact",
            "finish_reasons_complete",
            "errors_absent",
            "supporter_generation_treatment_exact",
            "fixed_seeker_generation_treatment_exact",
            "evidence_processing_contract_exact",
            "policy_lock_exact",
        )
    )

    mixed = [dict(row) for row in raw_rows]
    mixed[0]["normalized_finish_reason"] = "length"
    mixed[1]["error"] = "provider error"
    mixed[2]["supporter_generation_treatment_sha256"] = "0" * 64
    mixed[3]["policy_checkpoint_sha256"] = "f" * 64
    failed = build_raw_generation_contract_gate(raw_rows=mixed, **kwargs)
    assert failed["status"] == "FAIL"
    assert failed["finish_reasons_complete"] is False
    assert failed["errors_absent"] is False
    assert failed["supporter_generation_treatment_exact"] is False
    assert failed["policy_lock_exact"] is False


def test_truncated_response_is_failed_once_raw_retained_and_no_turn_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)
    plan_kwargs = _plan_kwargs(values)
    estimate, plan = plan_reference_baselines(**plan_kwargs)
    out_dir = tmp_path / PMV22_REFERENCE_BASELINE_STAGE
    persist_reference_baseline_dry_run(
        out_dir, cost_estimate=estimate, call_plan=plan
    )
    client = OneResultClient(
        CallResult(
            text="This response was cut off",
            raw_response={
                "choices": [{"finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            },
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
            latency_ms=1.0,
            request_hash="request-hash",
            provider_finish_reason="length",
            normalized_finish_reason="length",
        )
    )

    with pytest.raises(RuntimeError, match="output-token limit"):
        run_reference_baselines(
            out_dir=out_dir,
            fixed_tracks_attestation_path=values[
                "fixed_tracks_attestation_path"
            ],
            pm_v2_config_path=values["pm_v2_config_path"],
            evidence_filter_checkpoint_path=values[
                "evidence_filter_checkpoint_path"
            ],
            evidence_filter_report_path=values[
                "evidence_filter_report_path"
            ],
            evidence_filter_attestation_path=values[
                "evidence_filter_attestation_path"
            ],
            strategy_bank_approval_path=values[
                "strategy_bank_approval_path"
            ],
            strategy_bank_approval={"status": "APPROVED_FOR_TEST"},
            accepted_cost_estimate_sha256=estimate[
                "cost_estimate_sha256"
            ],
            client_factory=lambda _endpoint: client,
            **plan_kwargs,
        )

    assert client.calls == 1
    assert client.closed is True
    assert not (out_dir / "turns.jsonl").exists()
    raw_rows = list(iter_jsonl(out_dir / "raw_api_calls.jsonl"))
    assert len(raw_rows) == 1
    assert raw_rows[0]["normalized_finish_reason"] == "length"
    assert raw_rows[0]["error"]
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows] == ["STARTED", "FAILED"]
    assert ledger_rows[-1]["usage"]["total_tokens"] == 25


def test_run_rejects_wrong_accepted_hash_before_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)
    plan_kwargs = _plan_kwargs(values)
    estimate, plan = plan_reference_baselines(**plan_kwargs)
    out_dir = tmp_path / PMV22_REFERENCE_BASELINE_STAGE
    persist_reference_baseline_dry_run(
        out_dir, cost_estimate=estimate, call_plan=plan
    )
    created = 0

    def factory(_endpoint):
        nonlocal created
        created += 1
        raise AssertionError("must not create client")

    with pytest.raises(RuntimeError, match="must exactly equal"):
        run_reference_baselines(
            out_dir=out_dir,
            fixed_tracks_attestation_path=values[
                "fixed_tracks_attestation_path"
            ],
            pm_v2_config_path=values["pm_v2_config_path"],
            evidence_filter_checkpoint_path=values[
                "evidence_filter_checkpoint_path"
            ],
            evidence_filter_report_path=values[
                "evidence_filter_report_path"
            ],
            evidence_filter_attestation_path=values[
                "evidence_filter_attestation_path"
            ],
            strategy_bank_approval_path=values[
                "strategy_bank_approval_path"
            ],
            strategy_bank_approval={"status": "APPROVED_FOR_TEST"},
            accepted_cost_estimate_sha256="0" * 64,
            client_factory=factory,
            **plan_kwargs,
        )
    assert created == 0


def test_missing_policy_checkpoint_blocks_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)
    values["policy_checkpoint_path"].unlink()

    with pytest.raises(FileNotFoundError, match="final PM-v2 policy checkpoint"):
        plan_reference_baselines(**_plan_kwargs(values))


def test_policy_checkpoint_drift_after_dry_run_blocks_before_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)
    plan_kwargs = _plan_kwargs(values)
    estimate, plan = plan_reference_baselines(**plan_kwargs)
    out_dir = tmp_path / PMV22_REFERENCE_BASELINE_STAGE
    persist_reference_baseline_dry_run(
        out_dir, cost_estimate=estimate, call_plan=plan
    )
    values["policy_checkpoint_path"].write_bytes(b"post-plan policy drift")
    created = 0

    def factory(_endpoint):
        nonlocal created
        created += 1
        raise AssertionError("must not create client")

    with pytest.raises(
        RuntimeError,
        match="training report does not bind the selected checkpoint SHA-256",
    ):
        run_reference_baselines(
            out_dir=out_dir,
            fixed_tracks_attestation_path=values[
                "fixed_tracks_attestation_path"
            ],
            pm_v2_config_path=values["pm_v2_config_path"],
            evidence_filter_checkpoint_path=values[
                "evidence_filter_checkpoint_path"
            ],
            evidence_filter_report_path=values[
                "evidence_filter_report_path"
            ],
            evidence_filter_attestation_path=values[
                "evidence_filter_attestation_path"
            ],
            strategy_bank_approval_path=values[
                "strategy_bank_approval_path"
            ],
            strategy_bank_approval={"status": "APPROVED_FOR_TEST"},
            accepted_cost_estimate_sha256=estimate[
                "cost_estimate_sha256"
            ],
            client_factory=factory,
            **plan_kwargs,
        )
    assert created == 0


def test_complete_fake_run_attests_every_condition_and_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)
    plan_kwargs = _plan_kwargs(values)
    estimate, plan = plan_reference_baselines(**plan_kwargs)
    out_dir = tmp_path / PMV22_REFERENCE_BASELINE_STAGE
    persist_reference_baseline_dry_run(
        out_dir, cost_estimate=estimate, call_plan=plan
    )
    client = OneResultClient(
        CallResult(
            text="That deadline sounds exhausting. We can take this one step at a time.",
            raw_response={
                "choices": [{"finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 12,
                    "total_tokens": 32,
                },
            },
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 12,
                "total_tokens": 32,
            },
            latency_ms=1.0,
            request_hash="request-hash",
            provider_finish_reason="stop",
            normalized_finish_reason="complete",
        )
    )

    summary = run_reference_baselines(
        out_dir=out_dir,
        fixed_tracks_attestation_path=values[
            "fixed_tracks_attestation_path"
        ],
        pm_v2_config_path=values["pm_v2_config_path"],
        evidence_filter_checkpoint_path=values[
            "evidence_filter_checkpoint_path"
        ],
        evidence_filter_report_path=values[
            "evidence_filter_report_path"
        ],
        evidence_filter_attestation_path=values[
            "evidence_filter_attestation_path"
        ],
        strategy_bank_approval_path=values["strategy_bank_approval_path"],
        strategy_bank_approval={"status": "APPROVED_FOR_TEST"},
        accepted_cost_estimate_sha256=estimate["cost_estimate_sha256"],
        client_factory=lambda _endpoint: client,
        **plan_kwargs,
    )

    assert summary["status"] == "COMPLETE"
    assert summary["completed_turns"] == 4
    assert summary["non_complete_finish_reason_count"] == 0
    assert summary["policy_checkpoint_sha256"] == sha256_file(
        values["policy_checkpoint_path"]
    )
    assert summary["post_generation_policy_tuning_prohibited"] is True
    raw_gate = summary["raw_generation_contract_gate"]
    assert raw_gate["status"] == "PASS"
    assert raw_gate["observed_rows"] == 4
    assert client.calls == 4
    assert client.closed is True
    turns = list(iter_jsonl(out_dir / "turns.jsonl"))
    assert {row["condition"] for row in turns} == set(
        REFERENCE_BASELINE_CONDITIONS
    )
    assert all(row["normalized_finish_reason"] == "complete" for row in turns)
    assert all(row["evidence_processing_contract"] for row in turns)


def test_disabled_evidence_filter_omits_model_files_from_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _write_inputs(tmp_path, monkeypatch)
    plan_kwargs = _plan_kwargs(values)
    plan_kwargs.update(
        {
            "evidence_filter_config": EvidenceFilterConfig.from_mapping(
                {
                    **load_config(ROOT / "configs" / "pm_v2.yaml")[
                        "evidence_filter"
                    ],
                    "enabled": False,
                }
            ),
            "evidence_filter_model": None,
            "evidence_filter_model_binding": {
                "mode": "disabled_for_pm_v1_5"
            },
        }
    )
    estimate, plan = plan_reference_baselines(**plan_kwargs)
    out_dir = tmp_path / "disabled_filter_reference_baselines"
    persist_reference_baseline_dry_run(
        out_dir, cost_estimate=estimate, call_plan=plan
    )
    client = OneResultClient(
        CallResult(
            text="That sounds difficult, and I am here with you.",
            raw_response={
                "choices": [{"finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            },
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
            latency_ms=1.0,
            request_hash="request-hash",
            provider_finish_reason="stop",
            normalized_finish_reason="complete",
        )
    )
    summary = run_reference_baselines(
        out_dir=out_dir,
        fixed_tracks_attestation_path=values["fixed_tracks_attestation_path"],
        pm_v2_config_path=values["pm_v2_config_path"],
        evidence_filter_checkpoint_path=None,
        evidence_filter_report_path=None,
        evidence_filter_attestation_path=None,
        strategy_bank_approval_path=values["strategy_bank_approval_path"],
        strategy_bank_approval={
            "protocol": "pm-v1.5-strategy-bank-lineage-overlap-audit-v1",
            "decision": "ACCEPTED_WITHOUT_HUMAN_REVIEW",
            "human_calibration_performed": False,
        },
        accepted_cost_estimate_sha256=estimate["cost_estimate_sha256"],
        client_factory=lambda _endpoint: client,
        **plan_kwargs,
    )
    from metacom_pm.io import read_json

    attestation = read_json(out_dir / "artifact_attestation.json")
    assert not any(
        name.startswith("evidence_filter_") for name in attestation["inputs"]
    )
    raw_gate = summary["raw_generation_contract_gate"]
    attestation = load_config(out_dir / "artifact_attestation.json")
    assert attestation["stage"] == PMV22_REFERENCE_BASELINE_STAGE
    assert attestation["outputs"]["turns"]["rows"] == 4
    assert "policy_checkpoint" in attestation["inputs"]
    assert "policy_training_report" in attestation["inputs"]
    assert attestation["parameters"]["policy_lock_timing"] == (
        "before_first_reference_baseline_api_call"
    )
    assert attestation["parameters"]["raw_generation_contract_gate"] == raw_gate
    assert attestation["expected"]["raw_generation_contract_gate"] == raw_gate
