from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, utc_now, write_json


def _file_record(path: str | Path, *, jsonl: bool = False) -> dict[str, Any]:
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    record: dict[str, Any] = {
        "path": str(p),
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }
    if jsonl:
        record["rows"] = sum(1 for _ in iter_jsonl(p))
    return record


def create_artifact_attestation(
    out_path: str | Path,
    *,
    stage: str,
    inputs: Mapping[str, str | Path],
    outputs: Mapping[str, tuple[str | Path, bool]],
    parameters: Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
    study_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind generated artifacts to exact inputs, parameters and optional freeze.

    ``outputs`` maps a logical name to ``(path, is_jsonl)``.  The attestation is
    written only after all outputs exist, so an evaluator can reject stale or
    partially resumed files instead of trusting directory names.
    """
    value = {
        "status": "ATTESTED",
        "stage": stage,
        "created_at": utc_now(),
        "study_freeze_sha256": study_freeze_sha256,
        "inputs": {
            name: _file_record(path) for name, path in sorted(inputs.items())
        },
        "outputs": {
            name: _file_record(path, jsonl=is_jsonl)
            for name, (path, is_jsonl) in sorted(outputs.items())
        },
        "parameters": dict(parameters),
        "expected": dict(expected or {}),
    }
    value["attestation_sha256"] = sha256_text(canonical_json(value))
    write_json(out_path, value)
    return value


def verify_artifact_attestation(
    attestation_path: str | Path,
    *,
    required_stage: str | None = None,
    required_output_paths: Mapping[str, str | Path] | None = None,
    expected_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(attestation_path).resolve()
    if not path.is_file():
        return {"ok": False, "errors": [f"missing attestation: {path}"]}
    value = read_json(path)
    errors: list[str] = []
    expected_self = value.get("attestation_sha256")
    without_self = {k: v for k, v in value.items() if k != "attestation_sha256"}
    if expected_self != sha256_text(canonical_json(without_self)):
        errors.append("attestation record hash mismatch")
    if value.get("status") != "ATTESTED":
        errors.append("attestation status is not ATTESTED")
    if required_stage and value.get("stage") != required_stage:
        errors.append(
            f"stage mismatch: expected {required_stage}, got {value.get('stage')}"
        )
    if expected_freeze_sha256 is not None and value.get(
        "study_freeze_sha256"
    ) != expected_freeze_sha256:
        errors.append("study freeze hash mismatch")

    for section in ("inputs", "outputs"):
        records = value.get(section)
        if not isinstance(records, dict) or not records:
            errors.append(f"missing or empty {section}")
            continue
        for name, record in records.items():
            p = Path(str(record.get("path", "")))
            if not p.is_file():
                errors.append(f"missing {section} file {name}: {p}")
                continue
            if sha256_file(p) != record.get("sha256"):
                errors.append(f"{section} hash mismatch: {name}")
            if "rows" in record:
                try:
                    rows = sum(1 for _ in iter_jsonl(p))
                except Exception as exc:
                    errors.append(f"cannot validate JSONL {name}: {exc}")
                else:
                    if rows != int(record["rows"]):
                        errors.append(f"row count mismatch: {name}")

    if required_output_paths:
        output_records = value.get("outputs") or {}
        for name, requested in required_output_paths.items():
            record = output_records.get(name)
            if not isinstance(record, dict):
                errors.append(f"attestation lacks required output: {name}")
                continue
            if Path(str(record.get("path", ""))).resolve() != Path(requested).resolve():
                errors.append(f"required output path mismatch: {name}")
    return {
        "ok": not errors,
        "errors": errors,
        "attestation_sha256": expected_self,
        "stage": value.get("stage"),
        "study_freeze_sha256": value.get("study_freeze_sha256"),
        "path": str(path),
    }


def require_artifact_attestation(*args, **kwargs) -> dict[str, Any]:
    result = verify_artifact_attestation(*args, **kwargs)
    if not result["ok"]:
        raise RuntimeError(
            "Artifact attestation failed:\n- " + "\n- ".join(result["errors"])
        )
    return result


def verify_content_addressed_attestation(
    attestation_path: str | Path,
    *,
    required_stage: str,
    relocated_inputs: Mapping[str, str | Path],
    relocated_outputs: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Verify a relocated artifact bundle by logical name and content hash.

    Older release bundles recorded absolute build-host paths.  Those paths are
    provenance, not portable identities.  Relocation is safe only when the
    attestation self-hash is intact and every replay-critical local file matches
    the recorded digest (and JSONL row count, when present).  Unrequested bulky
    raw-call outputs may remain provenance-only.
    """

    path = Path(attestation_path).resolve()
    if not path.is_file():
        return {"ok": False, "errors": [f"missing attestation: {path}"]}
    value = read_json(path)
    errors: list[str] = []
    expected_self = value.get("attestation_sha256")
    without_self = {key: item for key, item in value.items() if key != "attestation_sha256"}
    if expected_self != sha256_text(canonical_json(without_self)):
        errors.append("attestation record hash mismatch")
    if value.get("status") != "ATTESTED":
        errors.append("attestation status is not ATTESTED")
    if value.get("stage") != required_stage:
        errors.append(
            f"stage mismatch: expected {required_stage}, got {value.get('stage')}"
        )
    if not relocated_inputs or not relocated_outputs:
        errors.append("content-addressed relocation requires inputs and outputs")

    relocated_records: dict[str, dict[str, Any]] = {}
    for section_name, requested in (
        ("inputs", relocated_inputs),
        ("outputs", relocated_outputs),
    ):
        records = value.get(section_name) or {}
        for logical_name, local_path in requested.items():
            record = records.get(logical_name)
            local = Path(local_path).resolve()
            if not isinstance(record, dict):
                errors.append(
                    f"attestation lacks replay-critical {section_name} record: "
                    f"{logical_name}"
                )
                continue
            if not local.is_file():
                errors.append(f"missing relocated {section_name} file: {local}")
                continue
            actual_sha256 = sha256_file(local)
            if actual_sha256 != record.get("sha256"):
                errors.append(
                    f"relocated {section_name} hash mismatch: {logical_name}"
                )
            if "bytes" in record and local.stat().st_size != int(record["bytes"]):
                errors.append(
                    f"relocated {section_name} byte-count mismatch: {logical_name}"
                )
            if "rows" in record:
                try:
                    actual_rows = sum(1 for _ in iter_jsonl(local))
                except Exception as exc:
                    errors.append(
                        f"cannot validate relocated JSONL {logical_name}: {exc}"
                    )
                else:
                    if actual_rows != int(record["rows"]):
                        errors.append(
                            f"relocated {section_name} row-count mismatch: "
                            f"{logical_name}"
                        )
            relocated_records[f"{section_name}.{logical_name}"] = {
                "local_path": str(local),
                "recorded_path": str(record.get("path") or ""),
                "sha256": actual_sha256,
            }
    return {
        "ok": not errors,
        "errors": errors,
        "attestation_sha256": expected_self,
        "stage": value.get("stage"),
        "path": str(path),
        "content_addressed_relocation": True,
        "relocated_records": relocated_records,
    }


def require_content_addressed_attestation(*args, **kwargs) -> dict[str, Any]:
    result = verify_content_addressed_attestation(*args, **kwargs)
    if not result["ok"]:
        raise RuntimeError(
            "Content-addressed artifact attestation failed:\n- "
            + "\n- ".join(result["errors"])
        )
    return result
