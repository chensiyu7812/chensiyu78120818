from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.api import (
    CallResult,
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
)
from metacom_pm.attempt_ledger import PersistentAttemptLedger
from metacom_pm.bounded_retry import (
    EXHAUSTED_DISPOSITION,
    RETRYABLE_DISPOSITION,
    TERMINAL_DISPOSITION,
    RetryBlockedError,
    call_retry_blocker,
    execute_with_bounded_retry,
    retry_ledger_summary,
)


def _ledger(tmp_path: Path, *, expected_attempts: int = 3) -> PersistentAttemptLedger:
    return PersistentAttemptLedger(
        tmp_path / "ledger.jsonl",
        stage="test_bounded_retry",
        expected_calls={"call_1": expected_attempts},
        maximum_total_attempts=expected_attempts,
    )


def _retryable(
    retry_class: str,
    status_code: int | None,
    *,
    retry_after_seconds: float | None = None,
    usage: dict[str, int] | None = None,
) -> RetryableProviderError:
    return RetryableProviderError(
        f"API call failed after strict retries: {retry_class}",
        last_retry_class=retry_class,
        last_status_code=status_code,
        attempts_tried=1,
        request_hash="b" * 64,
        usage=usage,
        response_diagnostics={
            "status_code": status_code,
            "response_body_sha256": "c" * 64,
        },
        retry_after_seconds=retry_after_seconds,
    )


def _execute(ledger: PersistentAttemptLedger, call_fn, **kwargs):
    return execute_with_bounded_retry(
        ledger,
        "call_1",
        record_ids={"item": "x"},
        prompt_sha256="a" * 64,
        call_fn=call_fn,
        sleep=kwargs.pop("sleep", lambda _seconds: None),
        **kwargs,
    )


def test_http_503_then_success_retries_and_ledgers_both_attempts(tmp_path: Path):
    ledger = _ledger(tmp_path)
    sleeps: list[float] = []
    calls = [
        lambda: (_ for _ in ()).throw(_retryable("http_5xx", 503)),
        lambda: ("ok", "parsed"),
    ]

    reservation, result, parsed = _execute(
        ledger, lambda: calls.pop(0)(), sleep=sleeps.append
    )

    assert (result, parsed) == ("ok", "parsed")
    assert reservation.attempt_index == 2
    assert ledger.attempts_for("call_1") == 2
    first = ledger.terminal_row("call_1", 1)
    assert first is not None
    assert first["metadata"]["retry_disposition"] == RETRYABLE_DISPOSITION
    assert first["request_hash"] == "b" * 64
    assert first["result"]["provider_response_diagnostics"][
        "response_body_sha256"
    ] == "c" * 64
    assert sleeps == [10.0]


def test_retry_after_header_overrides_shorter_policy_backoff(tmp_path: Path):
    ledger = _ledger(tmp_path)
    sleeps: list[float] = []
    calls = [
        lambda: (_ for _ in ()).throw(
            _retryable("rate_limited_429", 429, retry_after_seconds=47.0)
        ),
        lambda: ("ok", "parsed"),
    ]
    _execute(ledger, lambda: calls.pop(0)(), sleep=sleeps.append)
    assert sleeps == [47.0]


def test_request_timeout_408_is_in_the_bounded_transient_set(tmp_path: Path):
    ledger = _ledger(tmp_path)
    calls = [
        lambda: (_ for _ in ()).throw(_retryable("request_timeout_408", 408)),
        lambda: ("ok", "parsed"),
    ]
    reservation, result, parsed = _execute(ledger, lambda: calls.pop(0)())
    assert reservation.attempt_index == 2
    assert (result, parsed) == ("ok", "parsed")


def test_403_is_terminal_across_process_reload_and_preserves_diagnostics(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)

    def forbidden():
        raise ProviderRequestError(
            status_code=403,
            detail="forbidden",
            schema_mode=False,
            request_hash="d" * 64,
            response_diagnostics={"response_body_sha256": "e" * 64},
        )

    with pytest.raises(ProviderRequestError):
        _execute(ledger, forbidden)
    terminal = ledger.terminal_row("call_1", 1)
    assert terminal is not None
    assert terminal["request_hash"] == "d" * 64
    assert terminal["metadata"]["retry_disposition"] == TERMINAL_DISPOSITION

    reloaded = _ledger(tmp_path)
    calls = 0

    def must_not_call():
        nonlocal calls
        calls += 1
        return "wrong", "wrong"

    with pytest.raises(RetryBlockedError, match="terminal_nonretryable"):
        _execute(reloaded, must_not_call)
    assert calls == 0
    assert reloaded.attempts_for("call_1") == 1


class _FakeValidationError:
    def errors(self, **_kwargs):
        return [{"type": "value_error", "loc": [], "msg": "bad"}]

    def __str__(self) -> str:
        return "injected semantic validation failure"


def test_structured_rejection_preserves_paid_response_and_is_terminal_on_reload(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path)
    paid_call = CallResult(
        text='{"score": 2}',
        raw_response={"id": "paid-id", "choices": [{"message": {"content": "x"}}]},
        usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        latency_ms=12.0,
        request_hash="f" * 64,
    )

    def rejected():
        raise StructuredOutputValidationError(
            call=paid_call,
            parsed_payload={"score": 2},
            response_schema=object,
            validation_error=_FakeValidationError(),
        )

    with pytest.raises(StructuredOutputValidationError):
        _execute(ledger, rejected)
    terminal = ledger.terminal_row("call_1", 1)
    assert terminal is not None
    assert terminal["request_hash"] == "f" * 64
    assert terminal["usage"] == paid_call.usage
    assert terminal["result"]["provider_response"]["id"] == "paid-id"
    assert terminal["result"]["parsed_payload"] == {"score": 2}
    assert terminal["metadata"]["retry_disposition"] == TERMINAL_DISPOSITION

    reloaded = _ledger(tmp_path)
    with pytest.raises(RetryBlockedError):
        _execute(reloaded, lambda: ("wrong", "wrong"))
    assert reloaded.attempts_for("call_1") == 1


def test_missing_field_is_retried_once_and_never_resets_after_reload(tmp_path: Path):
    ledger = _ledger(tmp_path, expected_attempts=3)

    with pytest.raises(RetryableProviderError):
        _execute(
            ledger,
            lambda: (_ for _ in ()).throw(
                _retryable(
                    "missing_field",
                    None,
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                )
            ),
        )
    assert ledger.attempts_for("call_1") == 2
    assert not ledger.exhausted("call_1")
    assert ledger.terminal_row("call_1", 1)["metadata"][
        "retry_disposition"
    ] == RETRYABLE_DISPOSITION
    assert ledger.terminal_row("call_1", 2)["metadata"][
        "retry_disposition"
    ] == TERMINAL_DISPOSITION
    assert ledger.terminal_row("call_1", 2)["usage"]["total_tokens"] == 12

    reloaded = _ledger(tmp_path, expected_attempts=3)
    calls = 0

    def must_not_call():
        nonlocal calls
        calls += 1
        return "wrong", "wrong"

    with pytest.raises(RetryBlockedError, match="terminal_nonretryable.*missing_field"):
        _execute(reloaded, must_not_call)
    assert calls == 0
    assert reloaded.attempts_for("call_1") == 2


def test_crash_during_the_one_missing_field_retry_cannot_open_a_third_attempt(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path, expected_attempts=3)

    class StopAfterFirstFailure(RuntimeError):
        pass

    with pytest.raises(StopAfterFirstFailure):
        _execute(
            ledger,
            lambda: (_ for _ in ()).throw(_retryable("missing_field", None)),
            sleep=lambda _seconds: (_ for _ in ()).throw(StopAfterFirstFailure()),
        )
    second = ledger.reserve(
        "call_1", record_ids={"item": "x"}, prompt_sha256="a" * 64
    )
    assert second.attempt_index == 2
    # Simulate process death after STARTED was fsynced but before a terminal
    # row could be written. The unknown second attempt still consumes the one
    # extra attempt authorized by the missing-field policy.
    reloaded = _ledger(tmp_path, expected_attempts=3)
    assert call_retry_blocker(reloaded, "call_1") == (
        "provider-output repair allowance has an unknown spent attempt"
    )
    with pytest.raises(RetryBlockedError, match="unknown spent attempt"):
        _execute(reloaded, lambda: ("wrong", "wrong"))
    assert reloaded.attempts_for("call_1") == 2


def test_invalid_provider_json_gets_exactly_one_ledger_visible_repair(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path, expected_attempts=3)
    calls = [
        lambda: (_ for _ in ()).throw(_retryable("provider_output_format", None)),
        lambda: ("ok", "parsed"),
    ]
    reservation, result, parsed = _execute(ledger, lambda: calls.pop(0)())
    assert reservation.attempt_index == 2
    assert (result, parsed) == ("ok", "parsed")
    first = ledger.terminal_row("call_1", 1)
    assert first is not None
    assert first["metadata"]["retry_class"] == "provider_output_format"
    assert first["metadata"]["retry_disposition"] == RETRYABLE_DISPOSITION


def test_invalid_provider_json_cannot_open_a_third_attempt_after_failed_repair(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path, expected_attempts=3)
    with pytest.raises(RetryableProviderError):
        _execute(
            ledger,
            lambda: (_ for _ in ()).throw(
                _retryable("provider_output_format", None)
            ),
        )
    assert ledger.attempts_for("call_1") == 2
    assert ledger.terminal_row("call_1", 2)["metadata"][
        "retry_disposition"
    ] == TERMINAL_DISPOSITION
    reloaded = _ledger(tmp_path, expected_attempts=3)
    assert call_retry_blocker(reloaded, "call_1") == (
        "persisted terminal_nonretryable failure (provider_output_format)"
    )


def test_transport_failures_do_not_consume_provider_output_repair_allowance(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path, expected_attempts=6)
    calls = [
        lambda: (_ for _ in ()).throw(_retryable("http_5xx", 503)),
        lambda: (_ for _ in ()).throw(_retryable("provider_output_format", None)),
        lambda: (_ for _ in ()).throw(_retryable("http_5xx", 503)),
        lambda: ("ok", "parsed"),
    ]
    reservation, result, parsed = _execute(ledger, lambda: calls.pop(0)())
    assert reservation.attempt_index == 4
    assert (result, parsed) == ("ok", "parsed")
    assert [
        ledger.terminal_row("call_1", index)["metadata"]["retry_class"]
        for index in (1, 2, 3)
    ] == ["http_5xx", "provider_output_format", "http_5xx"]
    assert all(
        ledger.terminal_row("call_1", index)["metadata"]["retry_disposition"]
        == RETRYABLE_DISPOSITION
        for index in (1, 2, 3)
    )


def test_exhausting_repeated_5xx_marks_the_final_failure_exhausted(tmp_path: Path):
    ledger = _ledger(tmp_path, expected_attempts=3)
    with pytest.raises(RetryableProviderError):
        _execute(
            ledger,
            lambda: (_ for _ in ()).throw(_retryable("http_5xx", 500)),
        )
    assert ledger.attempts_for("call_1") == 3
    assert ledger.exhausted("call_1")
    assert ledger.terminal_row("call_1", 3)["metadata"][
        "retry_disposition"
    ] == EXHAUSTED_DISPOSITION


def test_resume_after_reload_continues_only_an_explicitly_retryable_failure(
    tmp_path: Path,
):
    ledger = _ledger(tmp_path, expected_attempts=3)

    class SimulatedProcessExit(RuntimeError):
        pass

    with pytest.raises(SimulatedProcessExit):
        _execute(
            ledger,
            lambda: (_ for _ in ()).throw(_retryable("http_5xx", 503)),
            sleep=lambda _seconds: (_ for _ in ()).throw(SimulatedProcessExit()),
        )
    assert ledger.attempts_for("call_1") == 1
    assert call_retry_blocker(ledger, "call_1") is None

    reloaded = _ledger(tmp_path, expected_attempts=3)
    reservation, result, parsed = _execute(
        reloaded, lambda: ("ok", "parsed")
    )
    assert reservation.attempt_index == 2
    assert (result, parsed) == ("ok", "parsed")


def test_legacy_failure_without_disposition_is_fail_closed(tmp_path: Path):
    ledger = _ledger(tmp_path)
    reservation = ledger.reserve(
        "call_1", record_ids={"item": "x"}, prompt_sha256="a" * 64
    )
    ledger.finish(
        reservation,
        succeeded=False,
        request_hash=None,
        usage=None,
        error="legacy failure",
    )
    assert call_retry_blocker(ledger, "call_1") == (
        "historical failure has no retry disposition"
    )
    with pytest.raises(RetryBlockedError):
        _execute(ledger, lambda: ("wrong", "wrong"))


def test_retry_summary_reports_recovery(tmp_path: Path):
    ledger = _ledger(tmp_path)
    calls = [
        lambda: (_ for _ in ()).throw(_retryable("http_5xx", 503)),
        lambda: ("ok", "parsed"),
    ]
    reservation, _, _ = _execute(ledger, lambda: calls.pop(0)())
    ledger.finish(
        reservation,
        succeeded=True,
        request_hash="a" * 64,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        error=None,
    )
    summary = retry_ledger_summary(ledger, ["call_1"])
    assert summary["physical_attempts"] == 2
    assert summary["failed_attempts_by_class"] == {"http_5xx": 1}
    assert summary["logical_calls_recovered_after_retry"] == 1
