from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .io import canonical_json, read_json, sha256_file, sha256_text, utc_now
from .pm_v1_6_contracts import CandidateFamilyManifest, PROTOCOL_VERSION

FREEZE_PROTOCOL = "pm-v1.6-study-freeze-v1"
EXTERNAL_CONDITIONS = (
    "learned_pm_step0",
    "strong_rule_step0",
    "cost_matched_fixed",
    "best_high_resource_fixed",
    "me_r0_legacy_anchor",
    "m0_r0",
    "session_rag_rs",
)


class FrozenFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class PMV16StudyFreeze(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: str = FREEZE_PROTOCOL
    version: str = PROTOCOL_VERSION
    status: str = "FROZEN_FOR_EXTERNAL_EXECUTION"
    created_at: str
    files: dict[str, FrozenFile]
    external_conditions: list[str]
    primary_cluster: str
    development_judge_families: list[str]
    final_judge_families: list[str]
    internal_gate_status: str
    internal_external_generation_allowed: bool
    selected_primary_algorithm: str
    selected_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    freeze_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coherent(self) -> "PMV16StudyFreeze":
        if self.external_conditions != list(EXTERNAL_CONDITIONS):
            raise ValueError("external conditions differ from frozen PM-v1.6 matrix")
        if self.internal_gate_status != "PASS":
            raise ValueError("study freeze requires an internal PASS")
        if not self.internal_external_generation_allowed:
            raise ValueError("internal report forbids external generation")
        if set(self.development_judge_families) & set(self.final_judge_families):
            raise ValueError("development and final judge families overlap")
        payload = self.model_dump(mode="json", exclude={"freeze_payload_sha256"})
        if self.freeze_payload_sha256 != sha256_text(canonical_json(payload)):
            raise ValueError("freeze payload SHA-256 mismatch")
        return self


def _frozen_file(path: str | Path) -> FrozenFile:
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return FrozenFile(
        path=str(value),
        sha256=sha256_file(value),
        size_bytes=value.stat().st_size,
    )


def build_study_freeze(
    *,
    method_config_path: str | Path,
    method_contract_path: str | Path,
    preregistration_path: str | Path,
    candidate_manifest_path: str | Path,
    checkpoint_path: str | Path,
    internal_report_path: str | Path,
    internal_ledger_path: str | Path,
    step0_observations_path: str | Path,
    step0_audit_bindings_path: str | Path,
    preflight_summary_path: str | Path,
    preflight_rows_path: str | Path,
    formal_outcomes_path: str | Path,
    bound_labels_path: str | Path,
    strategy_bank_path: str | Path,
    strategy_bank_audit_path: str | Path,
    overlap_audit_path: str | Path,
    fixed_seeker_tracks_path: str | Path,
    fixed_seeker_attestation_path: str | Path,
    judge_isolation_report_path: str | Path,
) -> PMV16StudyFreeze:
    file_paths = {
        "method_config": method_config_path,
        "method_contract": method_contract_path,
        "preregistration": preregistration_path,
        "candidate_manifest": candidate_manifest_path,
        "checkpoint": checkpoint_path,
        "internal_report": internal_report_path,
        "internal_ledger": internal_ledger_path,
        "step0_observations": step0_observations_path,
        "step0_audit_bindings": step0_audit_bindings_path,
        "preflight_summary": preflight_summary_path,
        "preflight_rows": preflight_rows_path,
        "formal_outcomes": formal_outcomes_path,
        "bound_labels": bound_labels_path,
        "strategy_bank": strategy_bank_path,
        "strategy_bank_audit": strategy_bank_audit_path,
        "overlap_audit": overlap_audit_path,
        "fixed_seeker_tracks": fixed_seeker_tracks_path,
        "fixed_seeker_attestation": fixed_seeker_attestation_path,
        "judge_isolation_report": judge_isolation_report_path,
    }
    files = {name: _frozen_file(path) for name, path in file_paths.items()}

    config = read_json(method_config_path) if str(method_config_path).endswith(".json") else None
    if config is None:
        from .config import load_config

        config = load_config(method_config_path)
    if config.get("version") != PROTOCOL_VERSION:
        raise RuntimeError("study freeze requires PM-v1.6 method config")
    if list(config["external_evaluation"]["condition_matrix"]) != list(
        EXTERNAL_CONDITIONS
    ):
        raise RuntimeError("method config external condition matrix drifted")

    candidate = CandidateFamilyManifest.model_validate_json(
        Path(candidate_manifest_path).read_text(encoding="utf-8")
    )
    if candidate.selected_checkpoint_sha256 != files["checkpoint"].sha256:
        raise RuntimeError("candidate manifest/checkpoint SHA-256 mismatch")

    internal = read_json(internal_report_path)
    if (
        internal.get("status") != "PASS"
        or internal.get("external_generation_allowed") is not True
        or internal.get("checkpoint_sha256") != files["checkpoint"].sha256
        or internal.get("candidate_manifest_sha256")
        != files["candidate_manifest"].sha256
    ):
        raise RuntimeError("internal report is not a matching PASS")

    preflight = read_json(preflight_summary_path)
    if (
        preflight.get("status") != "PASS"
        or int(preflight.get("required_hit_failures", -1)) != 0
        or preflight.get("action_preflight_sha256")
        not in {None, files["preflight_rows"].sha256}
    ):
        raise RuntimeError("required-hit preflight is not a matching PASS")

    judge = read_json(judge_isolation_report_path)
    if judge.get("status") != "PASS":
        raise RuntimeError("judge-isolation report is not PASS")
    resolved = judge.get("resolved_endpoints") or []
    development_families = sorted({str(row.get("family") or "") for row in resolved})
    expected_dev = sorted(config["judge_isolation"]["development_families"])
    final_families = sorted(config["judge_isolation"]["final_families"])
    if development_families != expected_dev:
        raise RuntimeError("judge-isolation resolved families drifted")

    payload: dict[str, Any] = {
        "protocol": FREEZE_PROTOCOL,
        "version": PROTOCOL_VERSION,
        "status": "FROZEN_FOR_EXTERNAL_EXECUTION",
        "created_at": utc_now(),
        "files": {
            name: record.model_dump(mode="json") for name, record in files.items()
        },
        "external_conditions": list(EXTERNAL_CONDITIONS),
        "primary_cluster": str(config["external_evaluation"]["primary_cluster"]),
        "development_judge_families": development_families,
        "final_judge_families": final_families,
        "internal_gate_status": str(internal["status"]),
        "internal_external_generation_allowed": bool(
            internal["external_generation_allowed"]
        ),
        "selected_primary_algorithm": candidate.selected_primary.value,
        "selected_checkpoint_sha256": files["checkpoint"].sha256,
        "method_contract_sha256": files["method_contract"].sha256,
    }
    payload["freeze_payload_sha256"] = sha256_text(canonical_json(payload))
    return PMV16StudyFreeze.model_validate(payload)


def require_study_freeze(
    freeze_path: str | Path,
    *,
    expected_files: Mapping[str, str | Path] | None = None,
) -> PMV16StudyFreeze:
    freeze = PMV16StudyFreeze.model_validate_json(
        Path(freeze_path).read_text(encoding="utf-8")
    )
    for name, record in freeze.files.items():
        path = Path(record.path)
        if not path.is_file() or sha256_file(path) != record.sha256:
            raise RuntimeError(f"frozen file drifted or disappeared: {name}")
    for name, path in (expected_files or {}).items():
        if name not in freeze.files:
            raise RuntimeError(f"freeze lacks expected file: {name}")
        if sha256_file(path) != freeze.files[name].sha256:
            raise RuntimeError(f"runtime file differs from freeze: {name}")
    return freeze
