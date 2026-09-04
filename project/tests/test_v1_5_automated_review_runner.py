from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.bounded_retry import TERMINAL_DISPOSITION, failure_metadata
from metacom_pm.io import iter_jsonl, read_json
from metacom_pm.v1_5_automated_semantic_review import RATING_FIELDS, aggregate_gate


ROOT = Path(__file__).resolve().parents[1]


def _all_affirming_ratings() -> dict[str, int]:
    return {field: 1 for field in RATING_FIELDS}


def test_aggregate_gate_requires_both_of_two_families_to_catch_a_control():
    families = ["deepseek", "openai_gpt4o"]
    real_case_results = {
        "case_1": {
            family: {"ratings": _all_affirming_ratings()} for family in families
        }
    }
    controls = [{"item_id": "case_1__control_regime", "rating_field": "regime_match"}]

    # Both families catch the planted error (rate it 0): majority-of-2 is 2, PASS.
    both_catch = {
        "case_1__control_regime": {
            family: {
                "ratings": {**_all_affirming_ratings(), "regime_match": 0}
            }
            for family in families
        }
    }
    gate = aggregate_gate(
        real_case_results=real_case_results,
        control_results=both_catch,
        controls=controls,
        judge_family_names=families,
    )
    assert gate["status"] == "PASS"
    assert gate["control_misses"] == []

    # Only one of two families catches it: majority-of-2 is 2, so 1/2 must FAIL,
    # not be treated as a passing majority the way 1/3 or 2/3 would be scaled.
    only_one_catches = {
        "case_1__control_regime": {
            "deepseek": {
                "ratings": {**_all_affirming_ratings(), "regime_match": 0}
            },
            "openai_gpt4o": {"ratings": _all_affirming_ratings()},
        }
    }
    gate = aggregate_gate(
        real_case_results=real_case_results,
        control_results=only_one_catches,
        controls=controls,
        judge_family_names=families,
    )
    assert gate["status"] == "FAIL"
    assert len(gate["control_misses"]) == 1
    assert gate["control_misses"][0]["caught_by"] == ["deepseek"]


def _load_runner():
    path = ROOT / "scripts" / "v1_5_run_automated_semantic_review.py"
    spec = importlib.util.spec_from_file_location("v1_5_automated_review_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(out_dir: Path, mode: str) -> list[str]:
    return [
        "v1_5_run_automated_semantic_review.py",
        mode,
        "--out-dir",
        str(out_dir),
        "--max-api-calls",
        "200",
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
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run"))
    module.main()
    first = read_json(out_dir / "cost_estimate.json")
    plan = list(iter_jsonl(out_dir / "call_plan.jsonl"))
    assert first["budget_gate"]["status"] == "PASS"
    # 33 real+control cases x 2 families (qwen dropped, see module docstring
    # amendment: two independent infra/compat failures, not a semantic-score
    # search) = 66 logical calls, not the original 3-family 99. Each logical
    # call now budgets up to 3 physical attempts (see the second docstring
    # amendment: bounded retry on transient infra failures), so the worst-case
    # physical-attempt cap is 66 x 3 = 198.
    assert first["n_logical_calls"] == 66
    assert first["maximum_physical_api_attempts"] == 198
    assert len(plan) == 66
    assert all(row["maximum_physical_attempts"] == 3 for row in plan)
    assert len({row["physical_call_key"] for row in plan}) == 66
    assert {row["judge_family"] for row in plan} == {"deepseek", "openai_gpt4o"}
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

    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run"))
    module.main()
    assert read_json(out_dir / "cost_estimate.json") == first
    assert list(iter_jsonl(out_dir / "call_plan.jsonl")) == plan


def test_automated_review_run_requires_accepted_dry_run_hash_before_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_runner()
    out_dir = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run"))
    module.main()
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--run"))
    with pytest.raises(RuntimeError, match="requires exact --accept-cost-estimate-sha256"):
        module.main()


def test_run_refuses_persisted_terminal_failure_before_loading_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_runner()
    out_dir = tmp_path / "review"
    monkeypatch.setattr(sys, "argv", _argv(out_dir, "--dry-run"))
    module.main()
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
    argv = _argv(out_dir, "--run") + [
        "--accept-cost-estimate-sha256",
        estimate["cost_estimate_sha256"],
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="persisted ledger"):
        module.main()
    assert ledger.attempts_for(call_key) == 1
