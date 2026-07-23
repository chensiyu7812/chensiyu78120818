from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .io import canonical_json, sha256_file, sha256_text, utc_now, write_json


CANDIDATE_MANIFEST_PROTOCOL = "pm-v1.5-frozen-candidate-family-v1"
INTERNAL_LEDGER_PROTOCOL = "pm-v1.5-internal-test-consumption-ledger-v1"
SEALED_INTERNAL_BUNDLE_PROTOCOL = "pm-v1.5-sealed-internal-label-bundle-v1"


def seal_internal_label_bundle(
    path: str | Path,
    *,
    internal_labels_path: str | Path,
) -> dict[str, Any]:
    """Create or exactly validate the pre-training opaque internal seal."""

    labels_path = Path(internal_labels_path)
    state_ids: set[str] = set()
    schema_keys: set[tuple[str, ...]] = set()
    row_count = 0
    with labels_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not str(row.get("state_id") or ""):
                raise RuntimeError(
                    f"invalid internal label row while sealing: {line_number}"
                )
            state_ids.add(str(row["state_id"]))
            schema_keys.add(tuple(sorted(str(key) for key in row)))
            row_count += 1
    if not row_count or len(schema_keys) != 1:
        raise RuntimeError("internal label bundle is empty or has schema drift")
    core = {
        "protocol": SEALED_INTERNAL_BUNDLE_PROTOCOL,
        "status": "SEALED_BEFORE_TRAINING",
        "internal_labels_sha256": sha256_file(labels_path),
        "row_count": row_count,
        "state_count": len(state_ids),
        "state_universe_sha256": sha256_text(canonical_json(sorted(state_ids))),
        "schema_sha256": sha256_text(canonical_json(list(next(iter(schema_keys))))),
    }
    payload = {**core, "seal_sha256": sha256_text(canonical_json(core))}
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("sealed internal bundle already exists with different content")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return payload


def require_sealed_internal_label_bundle(
    path: str | Path,
    *,
    internal_labels_path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "seal_sha256"}
    if (
        payload.get("protocol") != SEALED_INTERNAL_BUNDLE_PROTOCOL
        or payload.get("status") != "SEALED_BEFORE_TRAINING"
        or payload.get("internal_labels_sha256") != sha256_file(internal_labels_path)
        or payload.get("seal_sha256") != sha256_text(canonical_json(core))
        or int(payload.get("row_count") or 0) < 1
        or int(payload.get("state_count") or 0) < 1
    ):
        raise RuntimeError("sealed internal label bundle is missing, stale, or invalid")
    return payload


def freeze_candidate_manifest(
    path: str | Path,
    *,
    run_identity: str,
    artifacts: Mapping[str, str | Path],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Write or exactly validate the pre-internal-test candidate manifest."""

    path = Path(path)
    artifact_rows = {
        name: {"path": str(value), "sha256": sha256_file(value)}
        for name, value in sorted(artifacts.items())
    }
    core = {
        "protocol": CANDIDATE_MANIFEST_PROTOCOL,
        "status": "FROZEN_BEFORE_INTERNAL_TEST",
        "run_identity": str(run_identity),
        "artifacts": artifact_rows,
        "parameters": dict(parameters),
    }
    payload = {
        **core,
        "candidate_manifest_sha256": sha256_text(canonical_json(core)),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("candidate manifest already exists with different content")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return payload


def _read_ledger(handle) -> list[dict[str, Any]]:
    handle.seek(0)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid internal-test ledger row {line_number}"
            ) from exc
        if row.get("protocol") != INTERNAL_LEDGER_PROTOCOL:
            raise RuntimeError("internal-test ledger protocol mismatch")
        rows.append(row)
    return rows


def begin_internal_test_consumption(
    ledger_path: str | Path,
    *,
    candidate_manifest_path: str | Path,
    internal_labels_path: str | Path,
    sealed_artifact_name: str = "sealed_internal_bundle",
    consumption_domain: str = "default",
) -> dict[str, Any]:
    """Atomically spend the one permitted internal-test outcome read.

    The STARTED event is fsynced before the caller is allowed to open the label
    file.  A crash therefore consumes the run rather than enabling an invisible
    retry; resuming requires a new run identity and candidate manifest.
    """

    manifest = json.loads(Path(candidate_manifest_path).read_text(encoding="utf-8"))
    if (
        manifest.get("protocol") != CANDIDATE_MANIFEST_PROTOCOL
        or manifest.get("status") != "FROZEN_BEFORE_INTERNAL_TEST"
    ):
        raise RuntimeError("internal-test consumption requires a frozen candidate manifest")
    manifest_sha256 = sha256_file(candidate_manifest_path)
    labels_sha256 = sha256_file(internal_labels_path)
    if not sealed_artifact_name or not consumption_domain:
        raise ValueError("internal-test seal name and consumption domain are required")
    sealed_record = (manifest.get("artifacts") or {}).get(sealed_artifact_name) or {}
    sealed_path = Path(str(sealed_record.get("path") or ""))
    sealed = require_sealed_internal_label_bundle(
        sealed_path, internal_labels_path=internal_labels_path
    )
    if sealed_record.get("sha256") != sha256_file(sealed_path):
        raise RuntimeError("candidate manifest does not bind the sealed internal bundle")
    if sealed["internal_labels_sha256"] != labels_sha256:
        raise RuntimeError("internal labels differ from the pre-training seal")
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ledger_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        with os.fdopen(fd, "r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            rows = _read_ledger(handle)
            if rows:
                raise RuntimeError(
                    "internal-test outcome has already been consumed or consumption started"
                )
            event = {
                "protocol": INTERNAL_LEDGER_PROTOCOL,
                "event": "STARTED",
                "run_identity": manifest["run_identity"],
                "candidate_manifest_sha256": manifest_sha256,
                "internal_labels_sha256": labels_sha256,
                "sealed_artifact_name": sealed_artifact_name,
                "consumption_domain": consumption_domain,
                "started_at": utc_now(),
            }
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return event
    except Exception:
        # fd is owned by fdopen after successful construction.
        raise


def finish_internal_test_consumption(
    ledger_path: str | Path,
    *,
    report_path: str | Path,
    expected_domain: str = "default",
) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    fd = os.open(ledger_path, os.O_RDWR)
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        rows = _read_ledger(handle)
        if len(rows) != 1 or rows[0].get("event") != "STARTED":
            raise RuntimeError("internal-test ledger is not awaiting completion")
        if str(rows[0].get("consumption_domain") or "default") != expected_domain:
            raise RuntimeError("internal-test ledger consumption domain mismatch")
        event = {
            "protocol": INTERNAL_LEDGER_PROTOCOL,
            "event": "COMPLETED",
            "run_identity": rows[0]["run_identity"],
            "candidate_manifest_sha256": rows[0]["candidate_manifest_sha256"],
            "internal_labels_sha256": rows[0]["internal_labels_sha256"],
            "sealed_artifact_name": str(
                rows[0].get("sealed_artifact_name") or "sealed_internal_bundle"
            ),
            "consumption_domain": expected_domain,
            "report_sha256": sha256_file(report_path),
            "completed_at": utc_now(),
        }
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event


def require_completed_internal_consumption(
    ledger_path: str | Path,
    *,
    candidate_manifest_path: str | Path,
    report_path: str | Path,
    expected_domain: str = "default",
) -> dict[str, Any]:
    with Path(ledger_path).open("r", encoding="utf-8") as handle:
        rows = _read_ledger(handle)
    if len(rows) != 2 or [row.get("event") for row in rows] != [
        "STARTED",
        "COMPLETED",
    ]:
        raise RuntimeError("internal-test consumption is not exactly completed once")
    if rows[1]["candidate_manifest_sha256"] != sha256_file(candidate_manifest_path):
        raise RuntimeError("internal-test ledger candidate manifest hash mismatch")
    if rows[1]["report_sha256"] != sha256_file(report_path):
        raise RuntimeError("internal-test ledger report hash mismatch")
    if any(
        str(row.get("consumption_domain") or "default") != expected_domain
        for row in rows
    ):
        raise RuntimeError("internal-test ledger consumption domain mismatch")
    return {"status": "PASS", "events": rows}
