from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from metacom_pm.api import ProviderRequestError, RetryableProviderError
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.bounded_retry import (
    EXHAUSTED_DISPOSITION,
    RETRYABLE_DISPOSITION,
    failure_metadata,
)
from metacom_pm.io import canonical_json, sha256_file, sha256_text, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v1_5" / "21_judge_pm_v2_action_sweep_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "v1_5_development_judging_transport", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger(tmp_path: Path) -> PersistentAttemptLedger:
    return PersistentAttemptLedger(
        tmp_path / "ledger.jsonl",
        stage="pm_v2_action_judging",
        expected_calls={"call": 4},
        maximum_total_attempts=4,
    )


def test_transport_contract_is_self_hashed_and_scientifically_separate() -> None:
    module = _module()
    attempts = {"deepseek_official": 3, "google_gemini": 2}
    contract = module.development_judging_transport_contract(
        provider_output_attempts_by_family=attempts
    )
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}

    assert contract["contract_sha256"] == sha256_text(canonical_json(payload))
    assert contract["maximum_physical_attempts_per_logical_call"] == 4
    assert contract["provider_output_maximum_attempts_by_family"] == attempts
    assert contract["scientific_judge_contract_unchanged"] is True
    assert contract["terminal_content_or_schema_failure_is_not_blindly_retried"]
    assert contract["continue_after_isolated_provider_failure"] is True
    assert contract["code_manifest"]["runner"]["sha256"]

    with pytest.raises(ValueError, match="fit the physical-attempt bound"):
        module.development_judging_transport_contract(
            provider_output_attempts_by_family={"deepseek_official": 5}
        )


def test_formal_raw_gate_uses_applicability_and_sparse_zero_contract(
    monkeypatch,
) -> None:
    module = _module()
    captured = {}

    def global_gate(rows, **kwargs):
        captured["global"] = kwargs
        return {"status": "FAIL", "families": {}}

    def action_gate(rows, **kwargs):
        captured["action"] = kwargs
        return {"status": "PASS", "subgroups": {}}

    def risk_gate(rows, **kwargs):
        captured["risk"] = kwargs
        return {"status": "PASS", "actions": {}}

    monkeypatch.setattr(module, "validate_raw_judge_family_health", global_gate)
    monkeypatch.setattr(
        module, "validate_raw_judge_family_subgroup_health", action_gate
    )
    monkeypatch.setattr(
        module, "validate_action_applicable_risk_signal", risk_gate
    )
    labeling = {
        "duplicate_exact_match_rate": 0.98,
        "maximum_absolute_dimension_correlation": 0.95,
        "composite_support_exact_match_rate": 0.98,
        "maximum_absolute_composite_support_correlation": 0.995,
        "reject_constant_response_dimensions": True,
        "reject_constant_risk_dimensions": True,
        "minimum_action_applicable_risk_signal_rate": 0.005,
        "minimum_action_applicable_risk_distinct_values": 2,
    }
    report = module.evaluate_raw_judge_gates(
        [{"judge_family": "family", "action_id": "M0+R0"}],
        outcomes=[SimpleNamespace(action_id="M0+R0")],
        endpoints=[SimpleNamespace(family="family")],
        labeling=labeling,
        composite_spec=SimpleNamespace(),
        compatibility_pilot=False,
    )

    assert report["status"] == "FAIL"
    assert captured["global"]["raise_on_failure"] is False
    assert captured["global"]["minimum_nonzero_observations"] == 10
    assert captured["global"]["split_correlation_by_sign"] is True
    assert captured["global"]["inapplicable_risk_dimensions"]
    assert captured["global"]["inapplicable_risk_dimensions_by_action"]
    assert captured["action"]["raise_on_failure"] is False
    assert captured["action"]["minimum_nonzero_observations"] == 10
    assert captured["action"]["split_correlation_by_sign"] is True
    assert captured["risk"]["raise_on_failure"] is False


def test_formal_instrument_failure_is_nonreportable_and_creates_no_labels() -> None:
    module = _module()
    failed = module.formal_instrument_decision(
        raw_family_quality_gate={"status": "FAIL"},
        quality_gate={"status": "PASS"},
        compatibility_pilot=False,
        sealed_holdout_scope=False,
    )
    passed = module.formal_instrument_decision(
        raw_family_quality_gate={"status": "PASS"},
        quality_gate={"status": "PASS"},
        compatibility_pilot=False,
        sealed_holdout_scope=False,
    )

    assert failed == {
        "formal_instrument_supported": False,
        "training_labels_created": False,
        "status": "LONGITUDINAL_JUDGE_INSTRUMENT_NOT_SUPPORTED",
        "reportability_status": "NONREPORTABLE_MEASUREMENT_INSTRUMENT",
    }
    assert passed["formal_instrument_supported"] is True
    assert passed["training_labels_created"] is True
    assert passed["status"] == "COMPLETE"


def test_attestation_binds_posthoc_qualification_instead_of_stale_original() -> None:
    module = _module()
    qualification = Path("/tmp/qualification.json")
    qualification_attestation = Path("/tmp/qualification_attestation.json")
    inputs = module.actual_corpus_attestation_inputs(
        SimpleNamespace(
            actual_corpus_qualification_report=qualification,
            actual_corpus_qualification_attestation=qualification_attestation,
            actual_corpus_semantic_review_report=Path("/tmp/stale.json"),
            actual_corpus_semantic_review_attestation=Path(
                "/tmp/stale_attestation.json"
            ),
        )
    )

    assert inputs == {
        "actual_corpus_qualification": qualification,
        "actual_corpus_qualification_attestation": qualification_attestation,
    }
    assert "actual_corpus_semantic_review" not in inputs


def test_worst_case_cost_and_attempt_bounds_scale_exactly_fourfold() -> None:
    module = _module()
    bounds = module.development_judging_cost_bounds(
        [
            {
                "input_tokens_est": 100,
                "max_output_tokens": 25,
                "maximum_cost_usd": 0.1,
            },
            {
                "input_tokens_est": 200,
                "max_output_tokens": 75,
                "maximum_cost_usd": 0.2,
            },
        ]
    )

    assert bounds == {
        "logical_input_tokens": 300,
        "logical_output_tokens": 100,
        "logical_cost_usd": pytest.approx(0.3),
        "maximum_physical_attempts": 8,
        "maximum_input_tokens": 1200,
        "maximum_output_tokens": 400,
        "maximum_cost_usd": pytest.approx(1.2),
    }


def test_only_known_provider_surface_failures_are_isolatable() -> None:
    module = _module()
    transient = RetryableProviderError(
        "temporary",
        last_retry_class="http_5xx",
        last_status_code=503,
        attempts_tried=4,
    )
    terminal_4xx = ProviderRequestError(
        status_code=400,
        detail="bad request",
        schema_mode=True,
    )

    assert module._isolatable_provider_failure_class(transient) == "http_5xx"
    unknown = RetryableProviderError(
        "unknown",
        last_retry_class="other",
        last_status_code=None,
        attempts_tried=1,
    )
    truncated = RetryableProviderError(
        "length",
        last_retry_class="output_token_limit",
        last_status_code=None,
        attempts_tried=1,
    )
    assert module._isolatable_provider_failure_class(truncated) == "output_token_limit"
    assert module._isolatable_provider_failure_class(unknown) is None
    assert module._isolatable_provider_failure_class(terminal_4xx) is None
    assert module._isolatable_provider_failure_class(RuntimeError("local")) is None


def test_persisted_retryable_or_exhausted_transport_failure_can_be_skipped(
    tmp_path: Path,
) -> None:
    module = _module()
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(
        "call", record_ids={"row": 1}, prompt_sha256="a" * 64
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash="b" * 64,
        usage=None,
        error="503",
        metadata=failure_metadata(
            retry_class="http_5xx",
            retry_disposition=EXHAUSTED_DISPOSITION,
            status_code=503,
        ),
    )
    assert module._persisted_isolatable_failure_class(ledger, "call") == "http_5xx"

    other = PersistentAttemptLedger(
        tmp_path / "terminal.jsonl",
        stage="pm_v2_action_judging",
        expected_calls={"call": 4},
        maximum_total_attempts=4,
    )
    reservation = other.reserve(
        "call", record_ids={"row": 1}, prompt_sha256="a" * 64
    )
    other.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="local",
        metadata=failure_metadata(
            retry_class="stage_postcondition_failure",
            retry_disposition=RETRYABLE_DISPOSITION,
        ),
    )
    assert module._persisted_isolatable_failure_class(other, "call") is None


def test_circuit_breaker_counts_only_consecutive_same_class_failures() -> None:
    module = _module()
    previous = None
    count = 0
    for retry_class in ("http_5xx", "http_5xx", "network_timeout", "http_5xx"):
        previous, count = module.advance_development_judging_failure_streak(
            previous_class=previous,
            previous_count=count,
            retry_class=retry_class,
        )
    assert (previous, count) == ("http_5xx", 1)

    previous = None
    count = 0
    for _ in range(4):
        previous, count = module.advance_development_judging_failure_streak(
            previous_class=previous,
            previous_count=count,
            retry_class="http_5xx",
        )
    with pytest.raises(RuntimeError, match="5 times in a row"):
        module.advance_development_judging_failure_streak(
            previous_class=previous,
            previous_count=count,
            retry_class="http_5xx",
        )


def test_carry_forward_requires_exact_plan_and_complete_success_provenance(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "prior"
    source.mkdir()
    plan = [
        {
            "physical_call_key": "call",
            "state_id": "state",
            "action_id": "M0+R0",
            "judge_family": "google_gemini",
            "judge_type": "response",
            "prompt_hash": "a" * 64,
        }
    ]
    write_jsonl(source / "call_plan.jsonl", plan)
    ledger = PersistentAttemptLedger(
        source / "judge_call_ledger.jsonl",
        stage="pm_v2_action_judging",
        expected_calls={"call": 4},
        maximum_total_attempts=4,
    )
    reservation = ledger.reserve(
        "call",
        record_ids={
            "state_id": "state",
            "action_id": "M0+R0",
            "judge_family": "google_gemini",
            "judge_type": "response",
        },
        prompt_sha256="a" * 64,
    )
    ledger.finish(
        reservation,
        succeeded=True,
        request_hash="b" * 64,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        error=None,
        result={"parsed": {"fixture": True}},
    )

    carried = module.load_development_judging_carry_forward(
        carry_forward_dir=source,
        call_plan=plan,
        stage="pm_v2_action_judging",
    )
    assert carried["carried_call_keys"] == {"call"}
    assert carried["source_ledger_sha256"] == sha256_file(
        source / "judge_call_ledger.jsonl"
    )

    with pytest.raises(RuntimeError, match="not byte-equivalent"):
        module.load_development_judging_carry_forward(
            carry_forward_dir=source,
            call_plan=[{**plan[0], "prompt_hash": "c" * 64}],
            stage="pm_v2_action_judging",
        )


def test_formal_runner_wires_bounded_transport_and_nonreportable_incomplete() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "execute_with_bounded_retry(" in source
    assert "DEVELOPMENT_JUDGING_MAXIMUM_PHYSICAL_ATTEMPTS = 4" in source
    assert '"max_http_attempts": (' in source
    assert '"transport_execution_contract": transport_execution_contract' in source
    assert '"NONREPORTABLE_INCOMPLETE_MATRIX"' in source
    assert "load_development_judging_carry_forward(" in source
    assert '"carry_forward_source_ledger_sha256"' in source
    assert "the one-attempt protocol forbids reissuing it" not in source
    assert "development judge HTTP call failed after its ledger row was saved" not in source


def test_formal_judging_scope_selects_splits_without_label_access() -> None:
    module = _module()
    state_by_card = {
        "train": SimpleNamespace(split=SimpleNamespace(value="train")),
        "calibration": SimpleNamespace(split=SimpleNamespace(value="calibration")),
        "internal": SimpleNamespace(split=SimpleNamespace(value="internal_test")),
    }
    outcomes = [
        SimpleNamespace(card_id="train"),
        SimpleNamespace(card_id="calibration"),
        SimpleNamespace(card_id="internal"),
    ]

    development = module.select_formal_judging_outcomes(
        outcomes,
        state_by_card=state_by_card,
        label_scope=module.TRAIN_CALIBRATION_SCOPE,
    )
    internal = module.select_formal_judging_outcomes(
        outcomes,
        state_by_card=state_by_card,
        label_scope=module.SEALED_INTERNAL_TEST_SCOPE,
    )

    assert [row.card_id for row in development] == ["train", "calibration"]
    assert [row.card_id for row in internal] == ["internal"]
    with pytest.raises(ValueError, match="unsupported formal judging scope"):
        module.select_formal_judging_outcomes(
            outcomes,
            state_by_card=state_by_card,
            label_scope="all",
        )


def test_internal_scope_is_sealed_before_any_outcome_aggregate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '--label-scope' in source
    assert '"development_action_judging_train_calibration"' in source
    assert '"development_action_judging_internal_test"' in source
    assert '"NOT_EVALUATED_SEALED_HOLDOUT"' in source
    assert '"SEALED_INTERNAL_TEST_COMPLETE"' in source
    assert '"SEALED_HOLDOUT_NOT_YET_CONSUMED"' in source
    assert "if sealed_holdout_scope:" in source
    assert "seal_internal_label_bundle(" in source
