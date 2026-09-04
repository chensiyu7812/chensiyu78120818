from __future__ import annotations

import json
from pathlib import Path

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.io import write_json
from metacom_pm.v1_5_training_readiness import (
    SEALED_INTERNAL_JUDGING_STAGE,
    TRAIN_CALIBRATION_JUDGING_STAGE,
    require_longitudinal_judging_contract,
)


def _attest(
    tmp_path: Path,
    *,
    stage: str,
    summary: Path,
    labels: Path,
    internal: bool = False,
) -> tuple[Path, Path | None]:
    source = tmp_path / f"{stage}_source.json"
    write_json(source, {"frozen": True})
    seal = None
    outputs: dict[str, tuple[Path, bool]] = {
        "summary": (summary, False),
    }
    if internal:
        seal = tmp_path / "sealed_internal_bundle.json"
        write_json(seal, {"status": "SEALED_BEFORE_TRAINING"})
        outputs.update(
            {
                "internal_test_labels": (labels, True),
                "sealed_internal_bundle": (seal, False),
            }
        )
    else:
        outputs["train_calibration_labels"] = (labels, True)
    attestation = tmp_path / f"{stage}_attestation.json"
    create_artifact_attestation(
        attestation,
        stage=stage,
        inputs={"source": source},
        outputs=outputs,
        parameters={
            "status": json.loads(summary.read_text(encoding="utf-8"))["status"],
            "scope": "full",
            "label_scope": (
                "sealed_internal_test" if internal else "train_calibration"
            ),
        },
    )
    return attestation, seal


def _labels(path: Path, count: int) -> None:
    path.write_text(
        "".join(f'{{"state_id":"s{index}"}}\n' for index in range(count)),
        encoding="utf-8",
    )


def test_train_calibration_requires_reportable_pass_and_bound_labels(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "train_labels.jsonl"
    _labels(labels, 2)
    summary = tmp_path / "train_summary.json"
    write_json(
        summary,
        {
            "status": "COMPLETE",
            "reportability_status": "REPORTABLE",
            "label_scope": "train_calibration",
            "remaining_judge_pairs": 0,
            "remaining_api_calls": 0,
            "label_rows": 2,
            "completed_judge_pairs": 4,
            "full_expected_api_calls": 8,
            "quality_gate": {"status": "PASS"},
            "raw_family_quality_gate": {"status": "PASS"},
        },
    )
    attestation, _ = _attest(
        tmp_path,
        stage=TRAIN_CALIBRATION_JUDGING_STAGE,
        summary=summary,
        labels=labels,
    )

    result = require_longitudinal_judging_contract(
        summary_path=summary,
        attestation_path=attestation,
        labels_path=labels,
        label_scope="train_calibration",
        expected_label_rows=2,
    )
    assert result["status"] == "PASS"
    assert result["label_rows"] == 2


def test_train_calibration_refuses_failed_quality_gate(tmp_path: Path) -> None:
    labels = tmp_path / "train_labels.jsonl"
    _labels(labels, 1)
    summary = tmp_path / "train_summary.json"
    write_json(
        summary,
        {
            "status": "COMPLETE",
            "reportability_status": "REPORTABLE",
            "label_scope": "train_calibration",
            "remaining_judge_pairs": 0,
            "remaining_api_calls": 0,
            "label_rows": 1,
            "completed_judge_pairs": 2,
            "full_expected_api_calls": 4,
            "quality_gate": {"status": "FAIL"},
            "raw_family_quality_gate": {"status": "PASS"},
        },
    )
    attestation, _ = _attest(
        tmp_path,
        stage=TRAIN_CALIBRATION_JUDGING_STAGE,
        summary=summary,
        labels=labels,
    )

    with pytest.raises(RuntimeError, match="quality gate did not PASS"):
        require_longitudinal_judging_contract(
            summary_path=summary,
            attestation_path=attestation,
            labels_path=labels,
            label_scope="train_calibration",
        )


def test_training_refuses_labels_not_bound_by_attestation(tmp_path: Path) -> None:
    labels = tmp_path / "train_labels.jsonl"
    _labels(labels, 1)
    other_labels = tmp_path / "other_labels.jsonl"
    _labels(other_labels, 1)
    summary = tmp_path / "train_summary.json"
    write_json(
        summary,
        {
            "status": "COMPLETE",
            "reportability_status": "REPORTABLE",
            "label_scope": "train_calibration",
            "remaining_judge_pairs": 0,
            "remaining_api_calls": 0,
            "label_rows": 1,
            "completed_judge_pairs": 2,
            "full_expected_api_calls": 4,
            "quality_gate": {"status": "PASS"},
            "raw_family_quality_gate": {"status": "PASS"},
        },
    )
    attestation, _ = _attest(
        tmp_path,
        stage=TRAIN_CALIBRATION_JUDGING_STAGE,
        summary=summary,
        labels=labels,
    )

    with pytest.raises(RuntimeError, match="required output path mismatch"):
        require_longitudinal_judging_contract(
            summary_path=summary,
            attestation_path=attestation,
            labels_path=other_labels,
            label_scope="train_calibration",
        )


def test_training_refuses_incomplete_matrix_accounting(tmp_path: Path) -> None:
    labels = tmp_path / "train_labels.jsonl"
    _labels(labels, 1)
    summary = tmp_path / "train_summary.json"
    write_json(
        summary,
        {
            "status": "COMPLETE",
            "reportability_status": "REPORTABLE",
            "label_scope": "train_calibration",
            "remaining_judge_pairs": 0,
            "remaining_api_calls": 0,
            "label_rows": 1,
            "completed_judge_pairs": 1,
            "full_expected_api_calls": 4,
            "quality_gate": {"status": "PASS"},
            "raw_family_quality_gate": {"status": "PASS"},
        },
    )
    attestation, _ = _attest(
        tmp_path,
        stage=TRAIN_CALIBRATION_JUDGING_STAGE,
        summary=summary,
        labels=labels,
    )

    with pytest.raises(RuntimeError, match="two families per label"):
        require_longitudinal_judging_contract(
            summary_path=summary,
            attestation_path=attestation,
            labels_path=labels,
            label_scope="train_calibration",
        )


def test_internal_contract_requires_opaque_unevaluated_holdout(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "internal_labels.jsonl"
    _labels(labels, 3)
    summary = tmp_path / "internal_summary.json"
    write_json(
        summary,
        {
            "status": "SEALED_INTERNAL_TEST_COMPLETE",
            "reportability_status": "SEALED_HOLDOUT_NOT_YET_CONSUMED",
            "label_scope": "sealed_internal_test",
            "remaining_judge_pairs": 0,
            "remaining_api_calls": 0,
            "label_rows": 3,
            "completed_judge_pairs": 6,
            "full_expected_api_calls": 12,
            "quality_gate": {"status": "NOT_EVALUATED_SEALED_HOLDOUT"},
            "raw_family_quality_gate": {
                "status": "NOT_EVALUATED_SEALED_HOLDOUT"
            },
        },
    )
    attestation, seal = _attest(
        tmp_path,
        stage=SEALED_INTERNAL_JUDGING_STAGE,
        summary=summary,
        labels=labels,
        internal=True,
    )
    assert seal is not None

    result = require_longitudinal_judging_contract(
        summary_path=summary,
        attestation_path=attestation,
        labels_path=labels,
        label_scope="sealed_internal_test",
        expected_label_rows=3,
        sealed_internal_bundle_path=seal,
    )
    assert result["status"] == "PASS"


def test_internal_contract_refuses_preconsumption_gate_evaluation(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "internal_labels.jsonl"
    _labels(labels, 1)
    summary = tmp_path / "internal_summary.json"
    write_json(
        summary,
        {
            "status": "SEALED_INTERNAL_TEST_COMPLETE",
            "reportability_status": "SEALED_HOLDOUT_NOT_YET_CONSUMED",
            "label_scope": "sealed_internal_test",
            "remaining_judge_pairs": 0,
            "remaining_api_calls": 0,
            "label_rows": 1,
            "completed_judge_pairs": 2,
            "full_expected_api_calls": 4,
            "quality_gate": {"status": "PASS"},
            "raw_family_quality_gate": {
                "status": "NOT_EVALUATED_SEALED_HOLDOUT"
            },
        },
    )
    attestation, seal = _attest(
        tmp_path,
        stage=SEALED_INTERNAL_JUDGING_STAGE,
        summary=summary,
        labels=labels,
        internal=True,
    )
    assert seal is not None

    with pytest.raises(RuntimeError, match="quality_gate was evaluated"):
        require_longitudinal_judging_contract(
            summary_path=summary,
            attestation_path=attestation,
            labels_path=labels,
            label_scope="sealed_internal_test",
            sealed_internal_bundle_path=seal,
        )
