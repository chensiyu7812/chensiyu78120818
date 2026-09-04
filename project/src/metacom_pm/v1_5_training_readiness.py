from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .artifacts import require_artifact_attestation
from .io import read_json, sha256_file


TRAIN_CALIBRATION_JUDGING_STAGE = "pm_v2_action_judging_train_calibration"
SEALED_INTERNAL_JUDGING_STAGE = "pm_v2_action_judging_sealed_internal_test"


def _exact_nonnegative_int(
    payload: dict[str, Any], field: str, errors: list[str]
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"judging summary has invalid {field}")
        return -1
    return value


def require_longitudinal_judging_contract(
    *,
    summary_path: str | Path,
    attestation_path: str | Path,
    labels_path: str | Path,
    label_scope: Literal["train_calibration", "sealed_internal_test"],
    expected_label_rows: int | None = None,
    sealed_internal_bundle_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require the exact reportable judging artifact before PM training.

    Structural validation of ``ActionLabel`` rows remains the training
    runner's responsibility.  This guard supplies the missing provenance
    boundary: labels must be the content-addressed output of the formal
    longitudinal judging stage, and the corresponding matrix/gates must have
    reached the status appropriate for their split role.
    """

    summary_path = Path(summary_path)
    attestation_path = Path(attestation_path)
    labels_path = Path(labels_path)
    if label_scope == "train_calibration":
        required_stage = TRAIN_CALIBRATION_JUDGING_STAGE
        required_outputs: dict[str, str | Path] = {
            "summary": summary_path,
            "train_calibration_labels": labels_path,
        }
    elif label_scope == "sealed_internal_test":
        required_stage = SEALED_INTERNAL_JUDGING_STAGE
        if sealed_internal_bundle_path is None:
            raise ValueError(
                "sealed internal judging requires its sealed bundle path"
            )
        required_outputs = {
            "summary": summary_path,
            "internal_test_labels": labels_path,
            "sealed_internal_bundle": sealed_internal_bundle_path,
        }
    else:  # pragma: no cover - Literal protects typed callers.
        raise ValueError(f"unsupported longitudinal judging scope: {label_scope}")

    attestation = require_artifact_attestation(
        attestation_path,
        required_stage=required_stage,
        required_output_paths=required_outputs,
    )
    summary = read_json(summary_path)
    attestation_payload = read_json(attestation_path)
    errors: list[str] = []
    if summary.get("label_scope") != label_scope:
        errors.append("summary label scope mismatch")
    remaining_pairs = _exact_nonnegative_int(
        summary, "remaining_judge_pairs", errors
    )
    remaining_calls = _exact_nonnegative_int(
        summary, "remaining_api_calls", errors
    )
    label_rows = _exact_nonnegative_int(summary, "label_rows", errors)
    completed_pairs = _exact_nonnegative_int(
        summary, "completed_judge_pairs", errors
    )
    full_calls = _exact_nonnegative_int(
        summary, "full_expected_api_calls", errors
    )
    if remaining_pairs != 0:
        errors.append("judging matrix still has missing judge pairs")
    if remaining_calls != 0:
        errors.append("judging matrix still has missing API calls")
    if label_rows < 1:
        errors.append("judging summary has no label rows")
    if label_rows >= 0 and completed_pairs != label_rows * 2:
        errors.append("judging summary does not contain two families per label")
    if label_rows >= 0 and full_calls != label_rows * 4:
        errors.append("judging summary logical-call matrix is not exact")
    if expected_label_rows is not None and label_rows != int(expected_label_rows):
        errors.append(
            "judging label-row count mismatch: "
            f"expected={expected_label_rows}, observed={label_rows}"
        )
    attestation_parameters = attestation_payload.get("parameters") or {}
    if attestation_parameters.get("label_scope") != label_scope:
        errors.append("attestation label scope mismatch")
    if attestation_parameters.get("scope") != "full":
        errors.append("training refuses a non-full judging attestation")
    if attestation_parameters.get("status") != summary.get("status"):
        errors.append("attestation and summary status disagree")

    if label_scope == "train_calibration":
        if summary.get("status") != "COMPLETE":
            errors.append("train/calibration judging status is not COMPLETE")
        if summary.get("reportability_status") != "REPORTABLE":
            errors.append("train/calibration judging is not reportable")
        if (summary.get("quality_gate") or {}).get("status") != "PASS":
            errors.append("train/calibration label quality gate did not PASS")
        if (summary.get("raw_family_quality_gate") or {}).get("status") != "PASS":
            errors.append("train/calibration raw-family quality gate did not PASS")
    else:
        if summary.get("status") != "SEALED_INTERNAL_TEST_COMPLETE":
            errors.append("internal-test judging status is not sealed-complete")
        if (
            summary.get("reportability_status")
            != "SEALED_HOLDOUT_NOT_YET_CONSUMED"
        ):
            errors.append("internal-test judging is not an opaque sealed holdout")
        for gate_name in ("quality_gate", "raw_family_quality_gate"):
            gate = summary.get(gate_name) or {}
            if gate.get("status") != "NOT_EVALUATED_SEALED_HOLDOUT":
                errors.append(
                    f"internal-test {gate_name} was evaluated before consumption"
                )

    if errors:
        raise RuntimeError(
            "Longitudinal judging contract failed:\n- " + "\n- ".join(errors)
        )
    return {
        "status": "PASS",
        "label_scope": label_scope,
        "summary_sha256": sha256_file(summary_path),
        "attestation_sha256": attestation["attestation_sha256"],
        "labels_sha256": sha256_file(labels_path),
        "label_rows": label_rows,
    }
