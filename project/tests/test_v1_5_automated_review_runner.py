from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.bounded_retry import TERMINAL_DISPOSITION, failure_metadata
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.v1_5_automated_semantic_review import (
    AUTOMATED_CONTROL_PROTOCOL,
    AUTOMATED_REVIEW_PROTOCOL,
    RATING_FIELDS,
    aggregate_gate,
    require_automated_semantic_review_pass,
)


ROOT = Path(__file__).resolve().parents[1]


def _all_affirming_ratings() -> dict[str, int]:
    return {field: 1 for field in RATING_FIELDS}


def test_aggregate_gate_requires_both_of_two_families_to_catch_a_control():
    families = ["deepseek", "google_gemini"]
    real_case_results = {
        "case_1": {
            family: {"ratings": _all_affirming_ratings()} for family in families
        }
    }
    controls = [
        {
            "item_id": f"case_1__control_{field}_{replica}",
            "case_item_id": "case_1",
            "corrupted_field": field,
            "rating_field": field,
            "override": {field: replica},
            "case_text": f"corrupted {field} {replica}",
        }
        for field in RATING_FIELDS
        for replica in (1, 2)
    ]

    # Both families catch the planted error (rate it 0): majority-of-2 is 2, PASS.
    both_catch = {
        control["item_id"]: {
            family: {
                "ratings": {
                    **_all_affirming_ratings(),
                    control["rating_field"]: 0,
                }
            }
            for family in families
        }
        for control in controls
    }
    gate = aggregate_gate(
        real_case_results=real_case_results,
        control_results=both_catch,
        controls=controls,
        judge_family_names=families,
        control_protocol="test-controls-v2",
        required_control_fields=RATING_FIELDS,
        controls_per_field=2,
    )
    assert gate["status"] == "PASS"
    assert gate["control_misses"] == []

    # Only one of two families catches it: majority-of-2 is 2, so 1/2 must FAIL,
    # not be treated as a passing majority the way 1/3 or 2/3 would be scaled.
    only_one_catches = dict(both_catch)
    first = controls[0]
    only_one_catches[first["item_id"]] = {
        "deepseek": {
            "ratings": {
                **_all_affirming_ratings(),
                first["rating_field"]: 0,
            }
        },
        "google_gemini": {"ratings": _all_affirming_ratings()},
    }
    gate = aggregate_gate(
        real_case_results=real_case_results,
        control_results=only_one_catches,
        controls=controls,
        judge_family_names=families,
        control_protocol="test-controls-v2",
        required_control_fields=RATING_FIELDS,
        controls_per_field=2,
    )
    assert gate["status"] == "FAIL"
    assert len(gate["control_misses"]) == 1
    assert gate["control_misses"][0]["caught_by"] == ["deepseek"]


def test_aggregate_gate_rejects_empty_or_incomplete_control_matrix():
    families = ["deepseek", "google_gemini"]
    real_case_results = {
        "case_1": {
            family: {"ratings": _all_affirming_ratings()} for family in families
        }
    }
    gate = aggregate_gate(
        real_case_results=real_case_results,
        control_results={},
        controls=[],
        judge_family_names=families,
        control_protocol="test-controls-v2",
        required_control_fields=RATING_FIELDS,
        controls_per_field=2,
    )
    assert gate["status"] == "FAIL"
    assert gate["control_contract_errors"]


def test_pilot_review_pass_binds_current_config_and_strategy_bank(tmp_path: Path):
    config = tmp_path / "pm.yaml"
    experiment = tmp_path / "experiment.yaml"
    bank = tmp_path / "strategy.jsonl"
    endpoint_names = ["judge_a", "judge_b"]
    descriptors = [
        {
            "alias": "judge_a",
            "family": "family_a",
            "model": "model-a",
            "base_url": "https://a.example.invalid",
            "transport": "openai_chat_completions",
        },
        {
            "alias": "judge_b",
            "family": "family_b",
            "model": "model-b",
            "base_url": "https://b.example.invalid",
            "transport": "openai_chat_completions",
        },
    ]
    config.write_text(
        "version: pm-v1.5\nautomated_semantic_review:\n"
        "  judge_endpoints: [judge_a, judge_b]\n",
        encoding="utf-8",
    )
    experiment_text = (
        "endpoints:\n"
        "  judge_a:\n"
        "    base_url: https://a.example.invalid\n"
        "    model: model-a\n"
        "    api_key_env: A_KEY\n"
        "    family: family_a\n"
        "  judge_b:\n"
        "    base_url: https://b.example.invalid\n"
        "    model: model-b\n"
        "    api_key_env: B_KEY\n"
        "    family: family_b\n"
    )
    experiment.write_text(experiment_text, encoding="utf-8")
    bank.write_text('{"strategy_id":"strat_000000000000"}\n', encoding="utf-8")
    pilot_attestation = tmp_path / "pilot_attestation.json"
    write_json(
        pilot_attestation,
        {"parameters": {"compatibility_contract": {"contract_sha256": "c" * 64}}},
    )
    control_manifest = [
        {
            "item_id": str(index),
            "case_item_id": "case",
            "corrupted_field": RATING_FIELDS[index // 2],
            "rating_field": RATING_FIELDS[index // 2],
            "override": {},
            "case_text_sha256": f"{index:064x}",
        }
        for index in range(24)
    ]
    matrix_sha256 = sha256_text(canonical_json(control_manifest))
    report_path = tmp_path / "gate.json"
    write_json(
        report_path,
        {
            "protocol": AUTOMATED_REVIEW_PROTOCOL,
            "status": "PASS",
            "human_calibration_performed": False,
            "control_protocol": AUTOMATED_CONTROL_PROTOCOL,
            "required_control_fields": list(RATING_FIELDS),
            "controls_per_field": 2,
            "n_controls": 24,
            "control_field_counts": {field: 2 for field in RATING_FIELDS},
            "control_contract_errors": [],
            "control_matrix_sha256": matrix_sha256,
            "control_manifest": control_manifest,
            "control_catches": [{"item_id": str(index)} for index in range(24)],
            "control_misses": [],
            "judge_families": endpoint_names,
            "judge_endpoint_descriptors": descriptors,
            "n_real_cases": 36,
            "n_paid_pilot_cases": 9,
            "paid_pilot_audit": {
                "pilot_attestation_sha256": sha256_file(pilot_attestation),
                "pilot_contract_sha256": "c" * 64,
            },
        },
    )
    real_judgments = tmp_path / "real.json"
    control_judgments = tmp_path / "control.json"
    controls_path = tmp_path / "controls.json"
    ledger = tmp_path / "ledger.jsonl"
    write_json(real_judgments, {})
    write_json(control_judgments, {})
    write_json(controls_path, control_manifest)
    ledger.write_text("{}\n", encoding="utf-8")
    attestation = tmp_path / "attestation.json"
    create_artifact_attestation(
        attestation,
        stage="pm_v1_5_automated_semantic_review",
        inputs={
            "experiment_config": experiment,
            "pm_v1_5_config": config,
            "strategy_bank": bank,
            "generation_pilot_attestation": pilot_attestation,
        },
        outputs={
            "real_case_judgments": (real_judgments, False),
            "control_judgments": (control_judgments, False),
            "controls": (controls_path, False),
            "gate_report": (report_path, False),
            "physical_attempt_ledger": (ledger, True),
        },
        parameters={
            "judge_endpoint_descriptors": descriptors,
            "paid_pilot_attestation_sha256": sha256_file(pilot_attestation),
        },
    )
    assert require_automated_semantic_review_pass(
        report_path,
        attestation,
        expected_experiment_config_path=experiment,
        expected_pm_config_path=config,
        expected_strategy_bank_path=bank,
        expected_generation_pilot_attestation_path=pilot_attestation,
    )["report"]["status"] == "PASS"
    bank.write_text('{"strategy_id":"strat_changed000000"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch|does not bind"):
        require_automated_semantic_review_pass(
            report_path,
            attestation,
            expected_experiment_config_path=experiment,
            expected_pm_config_path=config,
            expected_strategy_bank_path=bank,
            expected_generation_pilot_attestation_path=pilot_attestation,
        )
    bank.write_text('{"strategy_id":"strat_000000000000"}\n', encoding="utf-8")
    experiment.write_text(
        experiment_text.replace(
            "https://a.example.invalid", "https://changed.example.invalid"
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="hash mismatch|does not bind"):
        require_automated_semantic_review_pass(
            report_path,
            attestation,
            expected_experiment_config_path=experiment,
            expected_pm_config_path=config,
            expected_strategy_bank_path=bank,
            expected_generation_pilot_attestation_path=pilot_attestation,
        )

    experiment.write_text(
        experiment_text.replace("model-a", "model-a-v2"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="hash mismatch|does not bind"):
        require_automated_semantic_review_pass(
            report_path,
            attestation,
            expected_experiment_config_path=experiment,
            expected_pm_config_path=config,
            expected_strategy_bank_path=bank,
            expected_generation_pilot_attestation_path=pilot_attestation,
        )


def test_runner_rejects_non_frozen_judge_alias_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_runner()
    pilot = _patch_paid_pilot(module, monkeypatch, tmp_path)
    argv = _argv(tmp_path / "review", "--dry-run", pilot) + [
        "--judge-endpoints",
        "training_judge_deepseek_flash",
        "training_judge_gemini_flash_lite",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="judge endpoints differ from frozen config"):
        module.main()


def _load_runner():
    path = ROOT / "scripts" / "v1_5_run_automated_semantic_review.py"
    spec = importlib.util.spec_from_file_location("v1_5_automated_review_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_paid_pilot(module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    pilot_attestation = tmp_path / "pilot_attestation.json"
    pilot_bundle = tmp_path / "pilot_bundle.json"
    write_json(pilot_attestation, {"fixture": True})
    write_json(pilot_bundle, {"fixture": True})
    rows = [
        {
            "kind": "real",
            "real_source": "paid_compatibility_pilot",
            "item_id": f"paid_pilot_case_{index:02d}",
            "text": f"Paid pilot semantic case {index}",
        }
        for index in range(9)
    ]
    audit = {
        "status": "PASS",
        "pilot_attestation_path": str(pilot_attestation),
        "pilot_attestation_sha256": sha256_file(pilot_attestation),
        "pilot_bundle_path": str(pilot_bundle),
        "pilot_bundle_sha256": sha256_file(pilot_bundle),
        "pilot_contract_sha256": "c" * 64,
        "pilot_verification_attestation_sha256": "a" * 64,
        "item_count": 9,
    }
    monkeypatch.setattr(
        module,
        "build_generation_pilot_review_items",
        lambda **_kwargs: (rows, audit),
    )
    return pilot_attestation


def _argv(out_dir: Path, mode: str, pilot_attestation: Path) -> list[str]:
    return [
        "v1_5_run_automated_semantic_review.py",
        mode,
        "--out-dir",
        str(out_dir),
        "--generation-pilot-attestation",
        str(pilot_attestation),
        "--max-api-calls",
        "400",
        "--max-estimated-usd",
        "5",
        "--max-input-tokens-per-call",
        "12000",
        "--input-usd-per-million-tokens",
        "1",
        "--output-usd-per-million-tokens",
        "1",
    ]


def test_automated_review_dry_run_freezes_the_durable_retry_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_runner()
    out_dir = tmp_path / "review"
    pilot = _patch_paid_pilot(module, monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run", pilot))
    module.main()
    first = read_json(out_dir / "cost_estimate.json")
    plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    assert first["budget_gate"]["status"] == "PASS"
    # 27 deterministic real + exact 9 paid-pilot real + 24 controls, across
    # 2 development-only families.
    # Final judges are hard-isolated from this gate. Each logical call budgets
    # up to 3 physical attempts, so the worst-case cap is 120 x 3 = 360.
    assert first["n_logical_calls"] == 120
    assert first["n_real_cases"] == 36
    assert first["maximum_physical_api_attempts"] == 360
    assert len(plan) == 120
    assert all(row["maximum_physical_attempts"] == 3 for row in plan)
    assert len({row["physical_call_key"] for row in plan}) == 120
    assert {row["judge_family"] for row in plan} == {"deepseek", "google_gemini"}
    assert first["call_order_protocol"] == (
        "frozen-endpoint-order-native-gemini-first-v1"
    )
    assert plan[0]["endpoint_name"] == "training_judge_gemini_flash_lite"
    assert first["judge_endpoint_descriptors"][0]["transport"] == (
        "gemini_generate_content"
    )
    retry_contract = first["retry_contract"]
    assert retry_contract["protocol"] == "pm-v1.5-bounded-retry-v2"
    assert retry_contract["cross_process_eligibility_source"] == (
        "physical_attempt_ledger"
    )
    assert "request_timeout_408" in retry_contract["retryable_up_to_full_budget"]
    assert retry_contract[
        "missing_field_maximum_additional_physical_attempts"
    ] == 1
    assert "consecutive_failure_circuit_breaker_limit" not in retry_contract

    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run", pilot))
    module.main()
    assert read_json(out_dir / "cost_estimate.json") == first
    assert list(iter_jsonl(out_dir / "call_plan.jsonl")) == plan


def test_automated_review_run_requires_accepted_dry_run_hash_before_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_runner()
    out_dir = tmp_path / "review"
    pilot = _patch_paid_pilot(module, monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run", pilot))
    module.main()
    monkeypatch.setattr(
        module,
        "require_paid_run_release",
        lambda *args, **kwargs: {"status": "PAID_RUN_RELEASED"},
    )
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--run", pilot))
    with pytest.raises(RuntimeError, match="requires exact --accept-cost-estimate-sha256"):
        module.main()


def test_run_refuses_persisted_terminal_failure_before_loading_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_runner()
    out_dir = tmp_path / "review"
    pilot = _patch_paid_pilot(module, monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run", pilot))
    module.main()
    monkeypatch.setattr(
        module,
        "require_paid_run_release",
        lambda *args, **kwargs: {"status": "PAID_RUN_RELEASED"},
    )
    estimate = read_json(out_dir / "cost_estimate.json")
    plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    expected_calls = {
        str(row["physical_call_key"]): module.MAX_PHYSICAL_ATTEMPTS_PER_CALL
        for row in plan
    }
    ledger = PersistentAttemptLedger(
        out_dir / "physical_attempt_ledger.jsonl",
        stage="pm_v1_5_automated_semantic_review",
        expected_calls=expected_calls,
        maximum_total_attempts=len(plan) * module.MAX_PHYSICAL_ATTEMPTS_PER_CALL,
    )
    first = plan[0]
    call_key = str(first["physical_call_key"])
    reservation = ledger.reserve(
        call_key,
        record_ids={
            "item_id": first["item_id"],
            "judge_family": first["judge_family"],
            "kind": first["kind"],
        },
        prompt_sha256=str(first["prompt_sha256"]),
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash="a" * 64,
        usage=None,
        error="ProviderRequestError: injected 403",
        metadata=failure_metadata(
            retry_class="provider_request_error_4xx",
            retry_disposition=TERMINAL_DISPOSITION,
            status_code=403,
        ),
    )

    monkeypatch.setattr(
        module,
        "make_client",
        lambda _endpoint: pytest.fail("credentials/client path must not be reached"),
    )
    argv = _argv(out_dir, "--run", pilot) + [
        "--accept-cost-estimate-sha256",
        estimate["cost_estimate_sha256"],
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="persisted ledger"):
        module.main()
    assert ledger.attempts_for(call_key) == 1
