from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .api import Endpoint, require_reported_usage
from .io import append_jsonl, canonical_json, iter_jsonl, sha256_text, utc_now


PHYSICAL_ATTEMPT_LEDGER_PROTOCOL = "pm-v2-physical-http-attempt-ledger-v1"
_TERMINAL_EVENTS = {"SUCCEEDED", "FAILED"}


def forbid_overwrite_of_spent_attempts(
    path: str | Path, *, overwrite: bool, stage: str
) -> None:
    """Prevent any overwrite mode from erasing paid-attempt history.

    This check intentionally runs before dry-run/run branching.  Otherwise a
    caller could use ``--dry-run --overwrite`` to delete the durable ledger and
    then submit the same physical call again in a later API run.
    """

    ledger_path = Path(path)
    if overwrite and ledger_path.is_file() and ledger_path.stat().st_size > 0:
        raise RuntimeError(
            f"{stage} refuses --overwrite because the physical-attempt ledger "
            f"is non-empty: {ledger_path}. Paid attempt history is immutable; "
            "resume in place or use a new protocol-reviewed output directory."
        )


def reported_prompt_token_error(
    usage: Mapping[str, Any] | None,
    *,
    maximum_prompt_tokens: int,
    stage: str,
    require_positive: bool,
) -> str | None:
    """Return a fail-closed accounting error for provider-reported usage."""

    if require_positive:
        try:
            normalized = require_reported_usage(usage, stage=stage)
        except RuntimeError as exc:
            return str(exc)
        reported = normalized["prompt_tokens"]
    else:
        reported = int((usage or {}).get("prompt_tokens") or 0)
    if reported > int(maximum_prompt_tokens):
        return (
            f"{stage} reported prompt_tokens exceed the frozen conservative "
            f"bound: reported={reported}, bound={int(maximum_prompt_tokens)}"
        )
    return None


def physical_call_key(
    *,
    stage: str,
    record_ids: Mapping[str, Any],
    prompt_sha256: str,
    endpoint: Endpoint,
    request_parameters: Mapping[str, Any],
) -> str:
    """Return the immutable logical-call key used by dry-run and resume."""

    return sha256_text(
        canonical_json(
            {
                "protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
                "stage": str(stage),
                "record_ids": dict(record_ids),
                "prompt_sha256": str(prompt_sha256),
                "endpoint": {
                    "base_url": endpoint.base_url,
                    "model": endpoint.model,
                    "family": endpoint.family,
                },
                "request_parameters": dict(request_parameters),
            }
        )
    )


def _attempt_key(call_key: str, attempt_index: int) -> str:
    return sha256_text(f"{call_key}:{int(attempt_index)}")


@dataclass(frozen=True)
class AttemptReservation:
    call_key: str
    attempt_index: int
    attempt_key: str


class PersistentAttemptLedger:
    """Append-only, crash-conservative accounting for physical HTTP attempts.

    A STARTED event is fsynced before the HTTP request.  Consequently an
    interrupted/unknown attempt is treated as spent on resume and can never be
    issued again under the same attempt key.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stage: str,
        expected_calls: Mapping[str, int],
        maximum_total_attempts: int,
    ) -> None:
        self.path = Path(path)
        self.stage = str(stage)
        self.expected_calls = {
            str(key): int(value) for key, value in expected_calls.items()
        }
        if any(value < 1 for value in self.expected_calls.values()):
            raise ValueError("per-call physical-attempt bounds must be positive")
        self.maximum_total_attempts = int(maximum_total_attempts)
        if self.maximum_total_attempts < 1:
            raise ValueError("maximum_total_attempts must be positive")
        self._events: dict[tuple[str, int], list[dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line_no, row in enumerate(iter_jsonl(self.path), 1):
            if row.get("protocol") != PHYSICAL_ATTEMPT_LEDGER_PROTOCOL:
                raise RuntimeError(f"attempt ledger protocol mismatch at line {line_no}")
            if row.get("stage") != self.stage:
                raise RuntimeError(f"attempt ledger stage mismatch at line {line_no}")
            call_key = str(row.get("call_key") or "")
            if call_key not in self.expected_calls:
                raise RuntimeError(
                    f"attempt ledger has an unplanned call key at line {line_no}"
                )
            attempt_index = int(row.get("attempt_index") or 0)
            if not 1 <= attempt_index <= self.expected_calls[call_key]:
                raise RuntimeError(
                    f"attempt ledger index exceeds the frozen bound at line {line_no}"
                )
            if row.get("attempt_key") != _attempt_key(call_key, attempt_index):
                raise RuntimeError(f"attempt ledger key mismatch at line {line_no}")
            event = str(row.get("event") or "")
            if event not in {"STARTED", *_TERMINAL_EVENTS}:
                raise RuntimeError(f"unknown attempt-ledger event at line {line_no}")
            key = (call_key, attempt_index)
            prior = self._events.setdefault(key, [])
            if not prior and event != "STARTED":
                raise RuntimeError(
                    f"attempt ledger terminal event precedes STARTED at line {line_no}"
                )
            if prior:
                if prior[0].get("event") != "STARTED" or len(prior) != 1:
                    raise RuntimeError(f"duplicate attempt-ledger event at line {line_no}")
                if event not in _TERMINAL_EVENTS:
                    raise RuntimeError(f"duplicate STARTED event at line {line_no}")
            prior.append(row)

        by_call: dict[str, list[int]] = {}
        for call_key, attempt_index in self._events:
            by_call.setdefault(call_key, []).append(attempt_index)
        for call_key, indices in by_call.items():
            ordered = sorted(indices)
            if ordered != list(range(1, max(ordered) + 1)):
                raise RuntimeError(f"attempt ledger has a gap for call {call_key}")
            successful = [
                index
                for index in ordered
                if self.terminal_event(call_key, index) == "SUCCEEDED"
            ]
            if successful and max(ordered) != successful[0]:
                raise RuntimeError(
                    f"attempt ledger continued after success for call {call_key}"
                )
        if self.started_attempts > self.maximum_total_attempts:
            raise RuntimeError("historical attempts exceed the runtime physical-attempt cap")

    @property
    def started_attempts(self) -> int:
        return len(self._events)

    @property
    def remaining_attempts(self) -> int:
        return self.maximum_total_attempts - self.started_attempts

    @property
    def event_rows(self) -> list[dict[str, Any]]:
        """Return all persisted events in deterministic attempt/event order."""

        rows: list[dict[str, Any]] = []
        for key in sorted(self._events):
            rows.extend(dict(row) for row in self._events[key])
        return rows

    @property
    def started_call_keys(self) -> set[str]:
        return {call_key for call_key, _ in self._events}

    def attempts_for(self, call_key: str) -> int:
        return sum(1 for key, _ in self._events if key == call_key)

    def terminal_event(self, call_key: str, attempt_index: int) -> str | None:
        events = self._events.get((call_key, int(attempt_index))) or []
        return str(events[-1]["event"]) if len(events) == 2 else None

    def terminal_row(
        self, call_key: str, attempt_index: int | None = None
    ) -> dict[str, Any] | None:
        """Return a terminal event, preferring the latest attempt when omitted."""

        if attempt_index is not None:
            events = self._events.get((call_key, int(attempt_index))) or []
            return dict(events[-1]) if len(events) == 2 else None
        for index in range(self.attempts_for(call_key), 0, -1):
            row = self.terminal_row(call_key, index)
            if row is not None:
                return row
        return None

    def succeeded(self, call_key: str) -> bool:
        return any(
            self.terminal_event(call_key, index) == "SUCCEEDED"
            for index in range(1, self.attempts_for(call_key) + 1)
        )

    def exhausted(self, call_key: str) -> bool:
        return self.attempts_for(call_key) >= self.expected_calls[call_key]

    def failures(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (call_key, attempt_index), events in sorted(self._events.items()):
            if len(events) == 2 and events[-1].get("event") == "FAILED":
                rows.append(dict(events[-1]))
        return rows

    def reserve(
        self,
        call_key: str,
        *,
        record_ids: Mapping[str, Any],
        prompt_sha256: str,
    ) -> AttemptReservation:
        if call_key not in self.expected_calls:
            raise RuntimeError("cannot reserve an unplanned physical call")
        if self.succeeded(call_key):
            raise RuntimeError("cannot reserve a call that already succeeded")
        if self.exhausted(call_key):
            raise RuntimeError("logical call exhausted its physical-attempt bound")
        if self.started_attempts >= self.maximum_total_attempts:
            raise RuntimeError("runtime physical-attempt cap exhausted")
        attempt_index = self.attempts_for(call_key) + 1
        reservation = AttemptReservation(
            call_key=call_key,
            attempt_index=attempt_index,
            attempt_key=_attempt_key(call_key, attempt_index),
        )
        row = {
            "protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "stage": self.stage,
            "event": "STARTED",
            "timestamp": utc_now(),
            "call_key": call_key,
            "attempt_index": attempt_index,
            "attempt_key": reservation.attempt_key,
            "record_ids": dict(record_ids),
            "prompt_sha256": str(prompt_sha256),
        }
        append_jsonl(self.path, row)
        self._events[(call_key, attempt_index)] = [row]
        return reservation

    def finish(
        self,
        reservation: AttemptReservation,
        *,
        succeeded: bool,
        request_hash: str | None,
        usage: Mapping[str, Any] | None,
        error: str | None,
        result: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        key = (reservation.call_key, reservation.attempt_index)
        events = self._events.get(key)
        if events is None or len(events) != 1:
            raise RuntimeError("attempt reservation is absent or already terminal")
        row = {
            "protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "stage": self.stage,
            "event": "SUCCEEDED" if succeeded else "FAILED",
            "timestamp": utc_now(),
            "call_key": reservation.call_key,
            "attempt_index": reservation.attempt_index,
            "attempt_key": reservation.attempt_key,
            "record_ids": dict(events[0].get("record_ids") or {}),
            "prompt_sha256": events[0].get("prompt_sha256"),
            "request_hash": request_hash,
            "usage": dict(usage) if usage is not None else None,
            "error": error,
            "result": dict(result) if result is not None else None,
            "metadata": dict(metadata) if metadata is not None else None,
        }
        append_jsonl(self.path, row)
        events.append(row)
