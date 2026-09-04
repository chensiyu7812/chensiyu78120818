"""Durable, ledger-visible retries for transient provider failures.

The PM-v1.5 semantic-review runner calls the provider client with ``retries=1``
so every physical HTTP attempt has its own append-only ledger reservation.  A
later process may continue only failures explicitly recorded as retryable.
Deterministic failures remain terminal across process restarts.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Mapping

from .api import (
    ProviderRequestError,
    RetryableProviderError,
    StructuredOutputValidationError,
)
from .attempt_ledger import AttemptReservation, PersistentAttemptLedger

RETRY_CONTRACT_PROTOCOL = "pm-v1.5-bounded-retry-v4-independent-transport-format"
RETRYABLE_UP_TO_FULL_BUDGET = frozenset(
    {
        "rate_limited_429",
        "request_timeout_408",
        "http_5xx",
        "network_timeout",
    }
)
DEFAULT_BACKOFF_SECONDS: tuple[float, ...] = (10.0, 30.0)
BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES = frozenset(
    {"missing_field", "provider_output_format"}
)
DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS = 2  # initial response plus one repair
# Compatibility alias for old imports; the v3 policy applies one shared cap to
# every provider-output-format class rather than only to missing fields.
DEFAULT_MAX_MISSING_FIELD_ATTEMPTS = DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS

RETRYABLE_DISPOSITION = "retryable_transient"
TERMINAL_DISPOSITION = "terminal_nonretryable"
EXHAUSTED_DISPOSITION = "retry_budget_exhausted"


class RetryBlockedError(RuntimeError):
    """A persisted terminal failure makes another HTTP attempt unauthorized."""


def failure_metadata(
    *,
    retry_class: str,
    retry_disposition: str,
    status_code: int | None = None,
    provider_attempts_tried: int | None = None,
    retry_after_seconds: float | None = None,
    response_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen metadata that controls cross-process retry eligibility."""

    return {
        "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
        "retry_class": str(retry_class),
        "retry_disposition": str(retry_disposition),
        "status_code": int(status_code) if status_code is not None else None,
        "provider_attempts_tried": (
            int(provider_attempts_tried)
            if provider_attempts_tried is not None
            else None
        ),
        "retry_after_seconds": (
            float(retry_after_seconds)
            if retry_after_seconds is not None
            else None
        ),
        "response_diagnostics": (
            dict(response_diagnostics) if response_diagnostics is not None else None
        ),
    }


def _call_failures(
    ledger: PersistentAttemptLedger, call_key: str
) -> list[dict[str, Any]]:
    return [
        row for row in ledger.failures() if str(row.get("call_key")) == call_key
    ]


def _provider_output_failures(
    ledger: PersistentAttemptLedger, call_key: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in _call_failures(ledger, call_key)
        if isinstance(row.get("metadata"), Mapping)
        and str(row["metadata"].get("retry_class") or "")
        in BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES
    ]


def call_retry_blocker(
    ledger: PersistentAttemptLedger,
    call_key: str,
    *,
    max_provider_output_attempts: int = DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS,
) -> str | None:
    """Explain why a non-successful call may not reserve another attempt.

    This decision is derived entirely from the persisted ledger.  It therefore
    survives process restarts and cannot accidentally reset a local counter.
    Historical failure rows created before this protocol are fail-closed.
    """

    if call_key not in ledger.expected_calls:
        return "call key is absent from the frozen attempt plan"
    if ledger.succeeded(call_key):
        return None
    failures = _call_failures(ledger, call_key)
    for row in failures:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            return "historical failure has no retry disposition"
        if metadata.get("retry_contract_protocol") != RETRY_CONTRACT_PROTOCOL:
            return "historical failure uses an unknown retry contract"
        retry_class = str(metadata.get("retry_class") or "")
        disposition = str(metadata.get("retry_disposition") or "")
        if disposition in {TERMINAL_DISPOSITION, EXHAUSTED_DISPOSITION}:
            return f"persisted {disposition} failure ({retry_class or 'unknown'})"
        if disposition != RETRYABLE_DISPOSITION:
            return "historical failure has an unknown retry disposition"
    provider_output_failures = _provider_output_failures(ledger, call_key)
    if len(provider_output_failures) >= int(max_provider_output_attempts):
        return "provider-output repair allowance is already spent"
    # If the process died after reserving an attempt that followed a malformed
    # provider surface, that unknown paid attempt may have been the one allowed
    # repair.  Fail closed rather than silently opening another repair draw.
    if provider_output_failures:
        last_output_failure_index = max(
            int(row.get("attempt_index") or 0) for row in provider_output_failures
        )
        for attempt_index in range(last_output_failure_index + 1, ledger.attempts_for(call_key) + 1):
            if ledger.terminal_event(call_key, attempt_index) is None:
                return "provider-output repair allowance has an unknown spent attempt"
    if ledger.exhausted(call_key):
        return "physical-attempt bound is exhausted"
    return None


def require_call_retryable(
    ledger: PersistentAttemptLedger,
    call_key: str,
    *,
    max_provider_output_attempts: int = DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS,
) -> None:
    blocker = call_retry_blocker(
        ledger,
        call_key,
        max_provider_output_attempts=max_provider_output_attempts,
    )
    if blocker is not None:
        raise RetryBlockedError(f"call {call_key} cannot be retried: {blocker}")


def _response_result(
    response_diagnostics: Mapping[str, Any] | None,
    *,
    provider_text: str | None = None,
) -> dict[str, Any] | None:
    if response_diagnostics is None and provider_text is None:
        return None
    result: dict[str, Any] = {
        "provider_response_diagnostics": (
            dict(response_diagnostics) if response_diagnostics is not None else None
        )
    }
    if provider_text is not None:
        bounded = str(provider_text)[:8192]
        result.update(
            {
                "provider_output_text": bounded,
                "provider_output_text_sha256": hashlib.sha256(
                    str(provider_text).encode("utf-8")
                ).hexdigest(),
                "provider_output_text_truncated": len(str(provider_text)) > len(bounded),
            }
        )
    return result


def execute_with_bounded_retry(
    ledger: PersistentAttemptLedger,
    call_key: str,
    *,
    record_ids: Mapping[str, Any],
    prompt_sha256: str,
    call_fn: Callable[[], tuple[Any, Any]],
    max_provider_output_attempts: int = DEFAULT_MAX_PROVIDER_OUTPUT_ATTEMPTS,
    backoff_seconds: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    capture_provider_output_text: bool = False,
) -> tuple[AttemptReservation, Any, Any]:
    """Retry only persisted transient failures, ledgering every attempt.

    The successful reservation remains open for the caller's stage-specific
    usage and postcondition checks.  All failed reservations are finalized
    here with enough metadata to make a restart obey the same decision.
    """

    if max_provider_output_attempts < 1:
        raise ValueError("max_provider_output_attempts must be positive")
    while True:
        require_call_retryable(
            ledger,
            call_key,
            max_provider_output_attempts=max_provider_output_attempts,
        )
        reservation = ledger.reserve(
            call_key, record_ids=record_ids, prompt_sha256=prompt_sha256
        )
        try:
            call_result, parsed = call_fn()
        except ProviderRequestError as exc:
            metadata = failure_metadata(
                retry_class="provider_request_error_4xx",
                retry_disposition=TERMINAL_DISPOSITION,
                status_code=exc.status_code,
                response_diagnostics=exc.response_diagnostics,
            )
            ledger.finish(
                reservation,
                succeeded=False,
                request_hash=exc.request_hash,
                usage=exc.usage,
                error=f"{type(exc).__name__}: {exc}",
                result=_response_result(exc.response_diagnostics),
                metadata=metadata,
            )
            raise
        except StructuredOutputValidationError as exc:
            call = exc.call
            result = {
                "provider_response": call.raw_response,
                "parsed_payload": exc.parsed_payload,
                "validation_errors": exc.validation_errors,
                "response_schema": getattr(exc.response_schema, "__name__", None),
            }
            ledger.finish(
                reservation,
                succeeded=False,
                request_hash=call.request_hash,
                usage=call.usage,
                error=f"{type(exc).__name__}: {exc}",
                result=result,
                metadata=failure_metadata(
                    retry_class="structured_output_validation_error",
                    retry_disposition=TERMINAL_DISPOSITION,
                ),
            )
            raise
        except RetryableProviderError as exc:
            effective_attempt_cap = ledger.expected_calls[call_key]
            if reservation.attempt_index >= effective_attempt_cap:
                disposition = EXHAUSTED_DISPOSITION
            elif (
                exc.last_retry_class in BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES
                and len(_provider_output_failures(ledger, call_key)) + 1
                >= int(max_provider_output_attempts)
            ):
                disposition = TERMINAL_DISPOSITION
            elif (
                exc.last_retry_class in BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES
                or exc.last_retry_class in RETRYABLE_UP_TO_FULL_BUDGET
            ):
                disposition = RETRYABLE_DISPOSITION
            else:
                disposition = TERMINAL_DISPOSITION
            metadata = failure_metadata(
                retry_class=exc.last_retry_class,
                retry_disposition=disposition,
                status_code=exc.last_status_code,
                provider_attempts_tried=exc.attempts_tried,
                retry_after_seconds=exc.retry_after_seconds,
                response_diagnostics=exc.response_diagnostics,
            )
            ledger.finish(
                reservation,
                succeeded=False,
                request_hash=exc.request_hash,
                usage=exc.usage,
                error=f"{type(exc).__name__}: {exc}",
                result=_response_result(
                    exc.response_diagnostics,
                    provider_text=(
                        exc.provider_text
                        if capture_provider_output_text
                        and exc.last_retry_class
                        in BOUNDED_PROVIDER_OUTPUT_RETRY_CLASSES
                        else None
                    ),
                ),
                metadata=metadata,
            )
            if disposition != RETRYABLE_DISPOSITION:
                raise
            wait_index = min(
                reservation.attempt_index - 1,
                max(len(backoff_seconds) - 1, 0),
            )
            policy_wait = backoff_seconds[wait_index] if backoff_seconds else 0.0
            sleep(max(float(policy_wait), float(exc.retry_after_seconds or 0.0)))
            continue
        except Exception as exc:  # noqa: BLE001 - every failure must be terminal
            ledger.finish(
                reservation,
                succeeded=False,
                request_hash=None,
                usage=None,
                error=f"{type(exc).__name__}: {exc}",
                metadata=failure_metadata(
                    retry_class="unclassified_local_error",
                    retry_disposition=TERMINAL_DISPOSITION,
                ),
            )
            raise
        return reservation, call_result, parsed


def retry_ledger_summary(
    ledger: PersistentAttemptLedger, call_keys: list[str]
) -> dict[str, Any]:
    """Summarize actual transport retries for the final gate report."""

    failures = [
        row for row in ledger.failures() if str(row.get("call_key")) in call_keys
    ]
    failures_by_class: dict[str, int] = {}
    for row in failures:
        metadata = row.get("metadata") or {}
        retry_class = str(metadata.get("retry_class") or "unclassified")
        failures_by_class[retry_class] = failures_by_class.get(retry_class, 0) + 1
    retried = [key for key in call_keys if ledger.attempts_for(key) > 1]
    recovered = [key for key in retried if ledger.succeeded(key)]
    return {
        "retry_contract_protocol": RETRY_CONTRACT_PROTOCOL,
        "logical_calls": len(call_keys),
        "physical_attempts": sum(ledger.attempts_for(key) for key in call_keys),
        "failed_physical_attempts": len(failures),
        "failed_attempts_by_class": dict(sorted(failures_by_class.items())),
        "logical_calls_retried": len(retried),
        "logical_calls_recovered_after_retry": len(recovered),
    }
