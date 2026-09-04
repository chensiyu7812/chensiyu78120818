from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts"
    / "v1_5"
    / "13e_run_esconv_auxiliary_pairwise_pilot_v1_5.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location("pairwise_pilot_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transport_contract_costs_every_authorized_attempt_and_is_content_addressed():
    module = _runner()
    manifest = {"runner": {"sha256": "a" * 64}}
    record = module._transport_contract(
        provider_output_attempts_by_family={
            "google_gemini": 2,
            "deepseek_official": 3,
        },
        code_manifest=manifest,
    )
    assert record["maximum_physical_attempts_per_logical_call"] == 10
    assert record["provider_output_maximum_attempts_by_family"] == {
        "deepseek_official": 3,
        "google_gemini": 2,
    }
    assert record["client_internal_retries"] == 1
    assert len(record["contract_sha256"]) == 64


def test_transport_contract_rejects_provider_output_budget_above_physical_bound():
    module = _runner()
    with pytest.raises(ValueError, match="retry budget"):
        module._transport_contract(
            provider_output_attempts_by_family={"family": 11},
            code_manifest={"runner": {"sha256": "a" * 64}},
        )


def test_circuit_breaker_counts_only_consecutive_same_class_failures():
    module = _runner()
    previous = None
    count = 0
    for value in ("http_5xx", "http_5xx", "network_timeout", "http_5xx"):
        previous, count = module._advance_failure_streak(
            previous_class=previous,
            previous_count=count,
            retry_class=value,
        )
    assert (previous, count) == ("http_5xx", 1)

    previous = None
    count = 0
    for _ in range(4):
        previous, count = module._advance_failure_streak(
            previous_class=previous,
            previous_count=count,
            retry_class="http_5xx",
        )
    with pytest.raises(RuntimeError, match="circuit breaker"):
        module._advance_failure_streak(
            previous_class=previous,
            previous_count=count,
            retry_class="http_5xx",
        )


def test_circuit_breaker_is_per_family_when_successes_interleave():
    module = _runner()
    streaks = {"google_gemini": (None, 0), "deepseek_official": (None, 0)}
    for _ in range(4):
        module._record_family_success(streaks, family="google_gemini")
        module._record_family_failure(
            streaks,
            family="deepseek_official",
            retry_class="structured_output_validation_error",
        )
    assert streaks["google_gemini"] == (None, 0)
    assert streaks["deepseek_official"] == (
        "structured_output_validation_error",
        4,
    )
    module._record_family_success(streaks, family="google_gemini")
    with pytest.raises(RuntimeError, match="circuit breaker"):
        module._record_family_failure(
            streaks,
            family="deepseek_official",
            retry_class="structured_output_validation_error",
        )


def test_runner_is_pilot_only_and_cannot_create_action_labels():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'STAGE = "esconv_auxiliary_train_pairwise_measurement_pilot"' in source
    assert "aggregate_pairwise_pilot" in source
    assert '"training_labels_created": False' in source
    assert "build_action_label" not in source
    assert "ActionLabel" not in source
    assert "internal_test" not in source.replace(
        "It never reads calibration/internal-test data", ""
    )
