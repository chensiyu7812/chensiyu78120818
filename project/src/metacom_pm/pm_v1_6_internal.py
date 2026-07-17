from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .io import append_jsonl, canonical_json, iter_jsonl, sha256_file, sha256_text, utc_now
from .pm_v1_6_contracts import (
    CandidateFamilyManifest,
    INTERNAL_LEDGER_PROTOCOL,
)


@dataclass(frozen=True)
class InternalConsumptionReservation:
    consumption_key: str
    manifest_sha256: str
    internal_dataset_sha256: str
    ledger_path: str


def internal_consumption_key(
    *,
    manifest_sha256: str,
    internal_dataset_sha256: str,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "protocol": INTERNAL_LEDGER_PROTOCOL,
                "manifest_sha256": manifest_sha256,
                "internal_dataset_sha256": internal_dataset_sha256,
            }
        )
    )


def reserve_internal_test_once(
    *,
    manifest_path: str | Path,
    internal_dataset_path: str | Path,
    ledger_path: str | Path,
) -> InternalConsumptionReservation:
    """Reserve one frozen candidate family against one internal dataset hash."""

    manifest_path = Path(manifest_path)
    internal_dataset_path = Path(internal_dataset_path)
    ledger_path = Path(ledger_path)
    manifest = CandidateFamilyManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest_sha = sha256_file(manifest_path)
    internal_sha = sha256_file(internal_dataset_path)
    if manifest.internal_dataset_sha256 != internal_sha:
        raise RuntimeError("candidate manifest is not bound to this internal dataset")
    existing = list(iter_jsonl(ledger_path)) if ledger_path.exists() else []
    consumed_hashes = {
        str(row.get("internal_dataset_sha256") or "")
        for row in existing
        if row.get("event") in {"RESERVED", "COMPLETE", "FAILED"}
    }
    if internal_sha in consumed_hashes:
        raise RuntimeError(
            "this internal dataset SHA has already been reserved or consumed"
        )
    key = internal_consumption_key(
        manifest_sha256=manifest_sha,
        internal_dataset_sha256=internal_sha,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        ledger_path,
        {
            "protocol": INTERNAL_LEDGER_PROTOCOL,
            "event": "RESERVED",
            "timestamp": utc_now(),
            "consumption_key": key,
            "manifest_sha256": manifest_sha,
            "internal_dataset_sha256": internal_sha,
            "selected_primary": manifest.selected_primary.value,
            "selected_checkpoint_sha256": manifest.selected_checkpoint_sha256,
        },
    )
    return InternalConsumptionReservation(
        consumption_key=key,
        manifest_sha256=manifest_sha,
        internal_dataset_sha256=internal_sha,
        ledger_path=str(ledger_path),
    )


def finish_internal_test(
    reservation: InternalConsumptionReservation,
    *,
    status: str,
    report_path: str | Path | None,
    error: str | None = None,
) -> None:
    if status not in {"COMPLETE", "FAILED"}:
        raise ValueError("internal-test status must be COMPLETE or FAILED")
    ledger_path = Path(reservation.ledger_path)
    rows = list(iter_jsonl(ledger_path))
    matching = [
        row
        for row in rows
        if row.get("consumption_key") == reservation.consumption_key
    ]
    if not matching or matching[-1].get("event") != "RESERVED":
        raise RuntimeError("internal-test reservation is missing or already terminal")
    report_sha = (
        sha256_file(report_path)
        if status == "COMPLETE" and report_path is not None
        else None
    )
    if status == "COMPLETE" and report_sha is None:
        raise ValueError("COMPLETE internal test requires a report")
    append_jsonl(
        ledger_path,
        {
            "protocol": INTERNAL_LEDGER_PROTOCOL,
            "event": status,
            "timestamp": utc_now(),
            "consumption_key": reservation.consumption_key,
            "manifest_sha256": reservation.manifest_sha256,
            "internal_dataset_sha256": reservation.internal_dataset_sha256,
            "report_sha256": report_sha,
            "error": error,
        },
    )


def require_no_internal_inputs(paths: Sequence[str | Path]) -> None:
    """Training/calibration scripts call this on all declared inputs."""

    forbidden = [
        str(Path(path))
        for path in paths
        if "internal" in Path(path).name.lower()
        or "internal_test" in str(path).lower()
    ]
    if forbidden:
        raise RuntimeError(
            "train/calibration stage cannot read internal-test paths: "
            + ", ".join(forbidden)
        )
