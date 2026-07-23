from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.api import CallResult, Endpoint
from metacom_pm.config import load_config
from metacom_pm.contracts import MemoryBackendRecord
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import iter_jsonl, read_json, write_jsonl
from metacom_pm.sweep import plan_action_sweep, run_action_sweep


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> SupporterGenerationContract:
    return SupporterGenerationContract.from_config(
        load_config(PROJECT_ROOT / "configs" / "pm_v2.yaml")
    )


def _inputs(tmp_path, tiny_state, tiny_memories, tiny_strategy):
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
    return runtime, backend, strategies


def test_pmv22_action_sweep_plan_binds_full_generation_treatment(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies = _inputs(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    contract = _contract()
    endpoint = Endpoint(
        "https://invalid.example", "fixture", "UNSET", family="test"
    )
    estimate, rows = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        action_filter={"M0+R0"},
        temperature=contract.temperature,
        max_tokens=contract.max_output_tokens,
        supporter_generation_contract=contract,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    assert estimate["supporter_generation_treatment"] == contract.payload()
    assert estimate["supporter_generation_treatment_sha256"] == contract.digest()
    assert rows[0]["supporter_generation_treatment"] == contract.payload()
    assert rows[0]["supporter_generation_treatment_sha256"] == contract.digest()

    legacy_estimate, legacy_rows = plan_action_sweep(
        runtime,
        backend,
        strategies,
        endpoint=endpoint,
        action_filter={"M0+R0"},
        temperature=contract.temperature,
        max_tokens=contract.max_output_tokens,
        system_prompt=contract.system_prompt,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    assert rows[0]["prompt_sha256"] == legacy_rows[0]["prompt_sha256"]
    assert rows[0]["call_key"] != legacy_rows[0]["call_key"]
    assert estimate["cost_estimate_sha256"] != legacy_estimate[
        "cost_estimate_sha256"
    ]

    with pytest.raises(ValueError, match="max_tokens differs"):
        plan_action_sweep(
            runtime,
            backend,
            strategies,
            endpoint=endpoint,
            action_filter={"M0+R0"},
            temperature=contract.temperature,
            max_tokens=contract.max_output_tokens - 1,
            supporter_generation_contract=contract,
            input_usd_per_mtok=1.0,
            output_usd_per_mtok=2.0,
        )


def test_plan_action_sweep_is_reproducible_across_separate_invocations(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    """call_plan_sha256/cost_estimate_sha256 must be byte-identical across
    two separate calls with identical inputs -- a real dry-run and a later
    separate --run process recompute this same plan and require an exact
    match (scripts/v1_5/06, 14; the new ESConv-auxiliary 13b). Real wall-clock
    retrieval latency must never leak into these hashes."""

    runtime, backend, strategies = _inputs(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    contract = _contract()
    endpoint = Endpoint(
        "https://invalid.example", "fixture", "UNSET", family="test"
    )
    kwargs = dict(
        endpoint=endpoint,
        action_filter={"MP+RS"},
        temperature=contract.temperature,
        max_tokens=contract.max_output_tokens,
        supporter_generation_contract=contract,
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=2.0,
    )
    first_estimate, first_rows = plan_action_sweep(
        runtime, backend, strategies, **kwargs
    )
    second_estimate, second_rows = plan_action_sweep(
        runtime, backend, strategies, **kwargs
    )
    assert first_rows == second_rows
    assert first_estimate == second_estimate
    assert (
        first_estimate["cost_estimate_sha256"]
        == second_estimate["cost_estimate_sha256"]
    )
    called_attempts = [
        attempt
        for row in first_rows
        for attempt in row["retrieval_attempts"]
        if attempt["called"]
    ]
    assert called_attempts
    assert all(attempt["latency_ms"] == 0.0 for attempt in called_attempts)


def test_pmv22_action_sweep_rejects_truncation_before_outcome_write(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies = _inputs(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    contract = _contract()

    class TruncatedClient:
        def __init__(self, endpoint):
            pass

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            return (
                CallResult(
                    text="This response was cut off",
                    raw_response={
                        "choices": [
                            {
                                "message": {"content": "This response was cut off"},
                                "finish_reason": "length",
                            }
                        ]
                    },
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": contract.max_output_tokens,
                        "total_tokens": 100 + contract.max_output_tokens,
                    },
                    latency_ms=1.0,
                    request_hash="truncated-request",
                    provider_finish_reason="length",
                    normalized_finish_reason="length",
                ),
                None,
            )

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", TruncatedClient)
    endpoint = Endpoint(
        "https://invalid.example", "fixture", "UNSET", family="test"
    )
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
            temperature=contract.temperature,
            max_tokens=contract.max_output_tokens,
            request_retries=1,
            fail_fast=True,
            supporter_generation_contract=contract,
        )

    assert not (out_dir / "outcomes.jsonl").exists()
    raw_rows = list(iter_jsonl(out_dir / "raw.jsonl"))
    assert len(raw_rows) == 1
    assert raw_rows[0]["raw_response"]["choices"][0]["finish_reason"] == "length"
    assert raw_rows[0]["normalized_finish_reason"] == "length"
    assert raw_rows[0]["completion_truncated"] is True
    assert "output-token limit" in raw_rows[0]["error"]
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    assert [row["event"] for row in ledger_rows] == ["STARTED", "FAILED"]
    assert ledger_rows[-1]["usage"]["completion_tokens"] == (
        contract.max_output_tokens
    )
    summary = read_json(out_dir / "summary.json")
    assert summary["aborted_on_completion_gate"] is True
    assert summary["failures"][-1]["supporter_completion_gate_rejected"] is True


def test_pmv22_action_sweep_normalizes_completed_output_and_records_treatment(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies = _inputs(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    contract = _contract()

    class CompleteClient:
        def __init__(self, endpoint):
            pass

        def close(self):
            pass

        def chat(self, *args, **kwargs):
            return (
                CallResult(
                    text="  A   calm\nresponse.  ",
                    raw_response={
                        "choices": [
                            {
                                "message": {"content": "A calm response."},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 4,
                        "total_tokens": 104,
                    },
                    latency_ms=1.0,
                    request_hash="complete-request",
                    provider_finish_reason="stop",
                    normalized_finish_reason="complete",
                ),
                None,
            )

    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", CompleteClient)
    endpoint = Endpoint(
        "https://invalid.example", "fixture", "UNSET", family="test"
    )
    out_dir = tmp_path / "out"
    summary = run_action_sweep(
        runtime,
        backend,
        strategies,
        out_dir / "outcomes.jsonl",
        out_dir / "raw.jsonl",
        out_dir / "summary.json",
        endpoint=endpoint,
        action_filter={"M0+R0"},
        temperature=contract.temperature,
        max_tokens=contract.max_output_tokens,
        request_retries=1,
        fail_fast=True,
        supporter_generation_contract=contract,
    )
    assert summary["status"] == "COMPLETE"
    assert summary["supporter_generation_treatment"] == contract.payload()
    outcome = list(iter_jsonl(out_dir / "outcomes.jsonl"))[0]
    assert outcome["response"] == "A calm response."
    provenance = outcome["provenance"]
    assert provenance["normalized_finish_reason"] == "complete"
    assert provenance["supporter_generation_treatment"] == contract.payload()
    assert provenance["supporter_generation_treatment_sha256"] == contract.digest()
