from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from metacom_pm.api import CallResult, Endpoint, ProviderRequestError, RetryableProviderError
from metacom_pm.contracts import MemoryBackendRecord
from metacom_pm.io import iter_jsonl, read_json, write_jsonl
from metacom_pm.sweep import run_action_sweep


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_v1_5_runner():
    path = PROJECT_ROOT / "scripts" / "v1_5" / "06_run_action_sweep_v1_5.py"
    spec = importlib.util.spec_from_file_location("v1_5_sweep_continuation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _endpoint() -> Endpoint:
    return Endpoint("https://invalid.example", "never-called", "UNSET", family="test")


def _write_two_card_fixture(
    tmp_path: Path, tiny_state, tiny_memories, tiny_strategy
) -> tuple[Path, Path, Path, list[tuple[str, str]]]:
    """Two cards, action_filter narrowed to 2 actions each -- 4 logical calls."""

    card_a = tiny_state.model_copy(
        update={"card_id": "card_a0000000000000", "state_id": "state_a0000000000000"}
    )
    card_b = tiny_state.model_copy(
        update={"card_id": "card_b0000000000000", "state_id": "state_b0000000000000"}
    )
    runtime = tmp_path / "runtime.jsonl"
    backend = tmp_path / "backend.jsonl"
    strategies = tmp_path / "strategies.jsonl"
    write_jsonl(runtime, [card_a.model_dump(mode="json"), card_b.model_dump(mode="json")])
    write_jsonl(
        backend,
        [
            MemoryBackendRecord(card_id=card_a.card_id, items=tiny_memories).model_dump(
                mode="json"
            ),
            MemoryBackendRecord(card_id=card_b.card_id, items=tiny_memories).model_dump(
                mode="json"
            ),
        ],
    )
    write_jsonl(strategies, [tiny_strategy.model_dump(mode="json")])
    actions = ["M0+R0", "M0+RS"]
    expected_keys = [
        (card_id, action_id)
        for card_id in (card_a.card_id, card_b.card_id)
        for action_id in actions
    ]
    return runtime, backend, strategies, expected_keys


class ScriptedClient:
    """Fake OpenAICompatibleClient: chat() outcomes are scripted per *physical*
    call, in strict call order, so transient-failure-then-recovery and
    terminal-failure scenarios can be exercised deterministically."""

    script: list = []
    call_log: list = []

    def __init__(self, endpoint):
        self.endpoint = endpoint

    def close(self) -> None:
        pass

    def chat(self, messages, *, temperature, max_tokens, seed, response_schema, retries):
        index = len(type(self).call_log)
        type(self).call_log.append(messages)
        action = type(self).script[index]
        if isinstance(action, BaseException):
            raise action
        return action, None


def _success_result(tag: str) -> CallResult:
    return CallResult(
        text=f"A supportive response ({tag}).",
        raw_response={"fixture": tag},
        usage={"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
        latency_ms=1.0,
        request_hash=f"prompt-equivalence-{tag}",
    )


def test_bounded_transport_retry_recovers_after_transient_5xx_and_completes_exactly(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies, expected_keys = _write_two_card_fixture(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    ScriptedClient.script = [
        RetryableProviderError(
            "transient", last_retry_class="http_5xx", last_status_code=503,
            attempts_tried=1,
        ),
        RetryableProviderError(
            "transient", last_retry_class="network_timeout", last_status_code=None,
            attempts_tried=2,
        ),
        _success_result("call1-attempt3"),
        _success_result("call2"),
        _success_result("call3"),
        _success_result("call4"),
    ]
    ScriptedClient.call_log = []
    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", ScriptedClient)
    out_dir = tmp_path / "out"
    summary = run_action_sweep(
        runtime,
        backend,
        strategies,
        out_dir / "outcomes.jsonl",
        out_dir / "raw.jsonl",
        out_dir / "summary.json",
        endpoint=_endpoint(),
        action_filter={"M0+R0", "M0+RS"},
        request_retries=3,
        fail_fast=False,
        transport_retry_policy="bounded_transport",
        transport_backoff_seconds=(0.0, 0.0),
    )
    assert summary["status"] == "COMPLETE"
    assert len(ScriptedClient.call_log) == 6
    outcomes = list(iter_jsonl(out_dir / "outcomes.jsonl"))
    assert {(row["card_id"], row["action_id"]) for row in outcomes} == set(expected_keys)
    assert len(outcomes) == len(expected_keys)
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    events_by_key: dict[str, list[str]] = {}
    for row in ledger_rows:
        events_by_key.setdefault(row["call_key"], []).append(row["event"])
    # Exactly one call_key needed 3 physical attempts (2 transient failures
    # then a success); the other three succeeded on their first attempt.
    attempt_counts = sorted(len(events) // 2 for events in events_by_key.values())
    assert attempt_counts == [1, 1, 1, 3]
    recovered = [key for key, events in events_by_key.items() if len(events) == 6]
    assert len(recovered) == 1
    assert events_by_key[recovered[0]] == [
        "STARTED", "FAILED", "STARTED", "FAILED", "STARTED", "SUCCEEDED",
    ]


def test_bounded_transport_retry_never_double_bills_a_succeeded_call(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies, expected_keys = _write_two_card_fixture(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    ScriptedClient.script = [
        _success_result("call1"),
        _success_result("call2"),
        _success_result("call3"),
        _success_result("call4"),
    ]
    ScriptedClient.call_log = []
    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", ScriptedClient)
    out_dir = tmp_path / "out"
    kwargs = dict(
        endpoint=_endpoint(),
        action_filter={"M0+R0", "M0+RS"},
        request_retries=3,
        fail_fast=False,
        transport_retry_policy="bounded_transport",
        transport_backoff_seconds=(0.0, 0.0),
    )
    summary = run_action_sweep(
        runtime, backend, strategies,
        out_dir / "outcomes.jsonl", out_dir / "raw.jsonl", out_dir / "summary.json",
        **kwargs,
    )
    assert summary["status"] == "COMPLETE"
    assert len(ScriptedClient.call_log) == 4

    def _forbid_more_calls(*args, **kwargs):
        raise AssertionError("a succeeded call must never be re-issued")

    monkeypatch.setattr(ScriptedClient, "chat", _forbid_more_calls)
    resumed_summary = run_action_sweep(
        runtime, backend, strategies,
        out_dir / "outcomes.jsonl", out_dir / "raw.jsonl", out_dir / "summary.json",
        **kwargs,
    )
    assert resumed_summary["status"] == "COMPLETE"
    assert resumed_summary["new_physical_http_attempts"] == 0
    assert resumed_summary["historical_physical_http_attempts"] == 4
    outcomes = list(iter_jsonl(out_dir / "outcomes.jsonl"))
    assert len(outcomes) == len(expected_keys)


def test_bounded_transport_retry_exhausts_the_per_call_attempt_cap(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies, expected_keys = _write_two_card_fixture(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    always_transient = RetryableProviderError(
        "transient", last_retry_class="http_5xx", last_status_code=503,
        attempts_tried=1,
    )
    ScriptedClient.script = [
        always_transient, always_transient,  # call 1: exhausts a 2-attempt cap
        _success_result("call2"),
        _success_result("call3"),
        _success_result("call4"),
    ]
    ScriptedClient.call_log = []
    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", ScriptedClient)
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime, backend, strategies,
            out_dir / "outcomes.jsonl", out_dir / "raw.jsonl", out_dir / "summary.json",
            endpoint=_endpoint(),
            action_filter={"M0+R0", "M0+RS"},
            request_retries=2,
            fail_fast=False,
            transport_retry_policy="bounded_transport",
            transport_backoff_seconds=(0.0, 0.0),
        )
    assert len(ScriptedClient.call_log) == 5
    summary = read_json(out_dir / "summary.json")
    assert summary["status"] == "INCOMPLETE"
    assert summary["completed_outcomes"] == 3
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    exhausted_key_events = [
        row for row in ledger_rows if row["call_key"] == ledger_rows[0]["call_key"]
    ]
    assert [row["event"] for row in exhausted_key_events] == [
        "STARTED", "FAILED", "STARTED", "FAILED",
    ]

    # A further invocation must not reserve a 3rd physical attempt for the
    # already-exhausted call: the runtime cap is fail-closed, not silently
    # extended by re-running the script.
    ScriptedClient.script = [always_transient] * 10
    ScriptedClient.call_log = []
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime, backend, strategies,
            out_dir / "outcomes.jsonl", out_dir / "raw.jsonl", out_dir / "summary.json",
            endpoint=_endpoint(),
            action_filter={"M0+R0", "M0+RS"},
            request_retries=2,
            fail_fast=False,
            transport_retry_policy="bounded_transport",
            transport_backoff_seconds=(0.0, 0.0),
        )
    assert len(ScriptedClient.call_log) == 0


def test_bounded_transport_retry_never_blindly_retries_terminal_content_errors(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies, expected_keys = _write_two_card_fixture(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    truncated = RetryableProviderError(
        "truncated", last_retry_class="output_token_limit", last_status_code=None,
        attempts_tried=1,
    )
    rejected = ProviderRequestError(
        status_code=401, detail="bad key", schema_mode=False,
    )
    ScriptedClient.script = [
        truncated,
        rejected,
        _success_result("call3"),
        _success_result("call4"),
    ]
    ScriptedClient.call_log = []
    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", ScriptedClient)
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime, backend, strategies,
            out_dir / "outcomes.jsonl", out_dir / "raw.jsonl", out_dir / "summary.json",
            endpoint=_endpoint(),
            action_filter={"M0+R0", "M0+RS"},
            # A generous per-call budget: a terminal error must consume only
            # ONE attempt, never the full 5-attempt allowance.
            request_retries=5,
            fail_fast=False,
            transport_retry_policy="bounded_transport",
            transport_backoff_seconds=(0.0, 0.0),
        )
    assert len(ScriptedClient.call_log) == 4
    summary = read_json(out_dir / "summary.json")
    assert summary["completed_outcomes"] == 2
    ledger_rows = list(iter_jsonl(out_dir / "physical_attempt_ledger.jsonl"))
    by_key: dict[str, list[str]] = {}
    for row in ledger_rows:
        by_key.setdefault(row["call_key"], []).append(row["event"])
    terminal_single_attempt = [
        events for events in by_key.values() if events == ["STARTED", "FAILED"]
    ]
    assert len(terminal_single_attempt) == 2

    # Re-invoking must never issue a second physical attempt for either
    # terminal call -- both are permanently blocked, not merely "unlucky".
    def _forbid_more_calls(*args, **kwargs):
        raise AssertionError("a terminal failure must never be retried")

    monkeypatch.setattr(ScriptedClient, "chat", _forbid_more_calls)
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime, backend, strategies,
            out_dir / "outcomes.jsonl", out_dir / "raw.jsonl", out_dir / "summary.json",
            endpoint=_endpoint(),
            action_filter={"M0+R0", "M0+RS"},
            request_retries=5,
            fail_fast=False,
            transport_retry_policy="bounded_transport",
            transport_backoff_seconds=(0.0, 0.0),
        )
    resumed_summary = read_json(out_dir / "summary.json")
    assert resumed_summary["completed_outcomes"] == 2
    assert resumed_summary["new_physical_http_attempts"] == 0


def test_bounded_transport_retry_covers_state_by_actions_matrix_exactly(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies, expected_keys = _write_two_card_fixture(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    ScriptedClient.script = [
        RetryableProviderError(
            "transient", last_retry_class="rate_limited_429", last_status_code=429,
            attempts_tried=1,
        ),
        _success_result("call1-attempt2"),
        _success_result("call2"),
        _success_result("call3"),
        _success_result("call4"),
    ]
    ScriptedClient.call_log = []
    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", ScriptedClient)
    out_dir = tmp_path / "out"
    summary = run_action_sweep(
        runtime, backend, strategies,
        out_dir / "outcomes.jsonl", out_dir / "raw.jsonl", out_dir / "summary.json",
        endpoint=_endpoint(),
        action_filter={"M0+R0", "M0+RS"},
        request_retries=3,
        fail_fast=False,
        transport_retry_policy="bounded_transport",
        transport_backoff_seconds=(0.0, 0.0),
    )
    assert summary["status"] == "COMPLETE"
    assert summary["expected_outcomes"] == len(expected_keys) == 4
    assert summary["completed_outcomes"] == 4
    outcomes = list(iter_jsonl(out_dir / "outcomes.jsonl"))
    outcome_keys = [(row["card_id"], row["action_id"]) for row in outcomes]
    assert sorted(outcome_keys) == sorted(expected_keys)
    assert len(outcome_keys) == len(set(outcome_keys))


def test_exact_plan_carry_forward_only_reissues_the_missing_call(
    tmp_path, monkeypatch, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies, expected_keys = _write_two_card_fixture(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    transient = RetryableProviderError(
        "transient",
        last_retry_class="http_5xx",
        last_status_code=503,
        attempts_tried=1,
    )
    ScriptedClient.script = [
        transient,
        transient,
        _success_result("source-call2"),
        _success_result("source-call3"),
        _success_result("source-call4"),
    ]
    ScriptedClient.call_log = []
    import metacom_pm.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "OpenAICompatibleClient", ScriptedClient)
    source = tmp_path / "source"
    with pytest.raises(RuntimeError, match="action sweep incomplete"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            source / "outcomes.jsonl",
            source / "raw.jsonl",
            source / "summary.json",
            endpoint=_endpoint(),
            action_filter={"M0+R0", "M0+RS"},
            request_retries=2,
            fail_fast=False,
            transport_retry_policy="bounded_transport",
            transport_backoff_seconds=(0.0,),
        )
    source_ledger_rows = list(
        iter_jsonl(source / "physical_attempt_ledger.jsonl")
    )
    carried = {
        str(row["call_key"]): row
        for row in source_ledger_rows
        if row["event"] == "SUCCEEDED"
    }
    assert len(carried) == 3

    ScriptedClient.script = [_success_result("continuation-missing-call")]
    ScriptedClient.call_log = []
    continuation = tmp_path / "continuation"
    binding = {
        "protocol": "fixture-exact-plan-carry-forward-v1",
        "source_ledger_sha256": "a" * 64,
    }
    summary = run_action_sweep(
        runtime,
        backend,
        strategies,
        continuation / "outcomes.jsonl",
        continuation / "raw.jsonl",
        continuation / "summary.json",
        endpoint=_endpoint(),
        action_filter={"M0+R0", "M0+RS"},
        request_retries=2,
        fail_fast=False,
        transport_retry_policy="bounded_transport",
        transport_backoff_seconds=(0.0,),
        max_physical_api_attempts=2,
        carry_forward_terminal_rows=carried,
        carry_forward_binding=binding,
    )
    assert summary["status"] == "COMPLETE"
    assert summary["completed_outcomes"] == len(expected_keys) == 4
    assert summary["carried_forward_logical_calls"] == 3
    assert summary["historical_physical_http_attempts"] == 3
    assert summary["new_physical_http_attempts"] == 1
    assert summary["maximum_physical_api_attempts_planned"] == 2
    assert len(ScriptedClient.call_log) == 1
    continuation_rows = list(
        iter_jsonl(continuation / "physical_attempt_ledger.jsonl")
    )
    carried_successes = [
        row
        for row in continuation_rows
        if row["event"] == "SUCCEEDED"
        and (row.get("metadata") or {}).get("carried_forward") is True
    ]
    assert len(carried_successes) == 3


def test_carry_forward_requires_binding_and_rejects_unknown_call(
    tmp_path, tiny_state, tiny_memories, tiny_strategy
):
    runtime, backend, strategies, _ = _write_two_card_fixture(
        tmp_path, tiny_state, tiny_memories, tiny_strategy
    )
    with pytest.raises(ValueError, match="content-addressed binding"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            tmp_path / "out" / "outcomes.jsonl",
            tmp_path / "out" / "raw.jsonl",
            tmp_path / "out" / "summary.json",
            endpoint=_endpoint(),
            action_filter={"M0+R0", "M0+RS"},
            carry_forward_terminal_rows={"unknown": {}},
        )
    with pytest.raises(RuntimeError, match="outside the current plan"):
        run_action_sweep(
            runtime,
            backend,
            strategies,
            tmp_path / "out2" / "outcomes.jsonl",
            tmp_path / "out2" / "raw.jsonl",
            tmp_path / "out2" / "summary.json",
            endpoint=_endpoint(),
            action_filter={"M0+R0", "M0+RS"},
            carry_forward_terminal_rows={"unknown": {}},
            carry_forward_binding={"protocol": "fixture"},
        )


def test_continuation_cost_identity_prices_only_remaining_calls():
    module = _load_v1_5_runner()
    rows = [
        {
            "call_key": "call-a",
            "estimated_input_tokens": 100,
            "maximum_output_tokens": 20,
        },
        {
            "call_key": "call-b",
            "estimated_input_tokens": 300,
            "maximum_output_tokens": 40,
        },
    ]
    estimate = {
        "protocol": "fixture",
        "expected_api_calls": 2,
        "logical_api_calls": 2,
        "maximum_physical_api_attempts": 8,
        "transport_max_attempts_per_call": 4,
        "pricing": {
            "input_usd_per_mtok": 1.0,
            "output_usd_per_mtok": 2.0,
        },
        "contract_bindings": {"science": "frozen"},
        "cost_estimate_sha256": "old",
    }
    carry_forward = {
        "binding": {"protocol": "fixture-carry", "binding_sha256": "b" * 64},
        "carried_call_keys": {"call-a"},
        "carried_terminal_rows": {"call-a": {}},
    }
    continuation = module._continuation_cost_estimate(
        estimate, rows, carry_forward
    )
    assert continuation["historical_carried_forward_calls"] == 1
    assert continuation["remaining_new_logical_calls"] == 1
    assert continuation["maximum_physical_api_attempts"] == 4
    assert continuation["estimated_total_input_tokens"] == 300
    assert continuation["maximum_total_output_tokens"] == 40
    assert continuation["estimated_cost_usd"] == pytest.approx(
        4 * ((300 / 1_000_000) + (40 / 1_000_000 * 2))
    )
    body = {
        key: value
        for key, value in continuation.items()
        if key != "cost_estimate_sha256"
    }
    assert continuation["cost_estimate_sha256"] == module.sha256_text(
        module.canonical_json(body)
    )
