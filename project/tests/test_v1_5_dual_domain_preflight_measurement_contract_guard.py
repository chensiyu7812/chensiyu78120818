"""Guard tests for
scripts/v1_5/21a_preflight_dual_domain_training_v1_5.py's
``_require_frozen_esconv_auxiliary_measurement_contract``.

This is the mechanism that binds the dual-domain training preflight to the
frozen ESConv-auxiliary measurement contract (applicability contract,
conservative-utility protocol/lambda/clamp ranges, and the code hashes that
define them) attested by 13c_judge_esconv_auxiliary_v1_5.py's train-split
run. Only the guard function itself is tested here (with a minimal, real,
on-disk code file to hash), not the full preflight script end-to-end.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.io import canonical_json, sha256_file, sha256_text, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "v1_5" / "21a_preflight_dual_domain_training_v1_5.py"
REAL_CODE_FILE_RELATIVE = "src/metacom_pm/artifacts.py"


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location(
        "v1_5_21a_preflight_dual_domain_training", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_measurement_contract() -> dict:
    real_code_path = ROOT / REAL_CODE_FILE_RELATIVE
    return {
        "protocol": "pm-v1.5-judge-uncertainty-robust-measurement-contract-v1",
        "conservative_utility_protocol": "pm-v1.5-mad-adjusted-conservative-utility-v1",
        "conservative_utility_lambda": 1.0,
        "conservative_utility_lambda_is_fixed_never_tuned": True,
        "response_dimension_clamp_range": [1.0, 5.0],
        "risk_dimension_clamp_range": [0.0, 3.0],
        "code_manifest": {
            "pm_v2_model": {
                "relative_path": REAL_CODE_FILE_RELATIVE,
                "sha256": sha256_file(real_code_path),
            }
        },
    }


def _write_attestation(tmp_path: Path, *, labels_path: Path, measurement_contract: dict) -> Path:
    attestation_path = tmp_path / "attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage="esconv_auxiliary_judging_full_train",
        inputs={"config": labels_path},
        outputs={"labels": (labels_path, True)},
        parameters={"measurement_contract": measurement_contract},
    )
    return attestation_path


def test_accepts_a_consistent_frozen_measurement_contract(tmp_path):
    module = _load_preflight_module()
    labels_path = tmp_path / "action_labels.jsonl"
    write_jsonl(labels_path, [{"a": 1}])
    attestation_path = _write_attestation(
        tmp_path, labels_path=labels_path, measurement_contract=_valid_measurement_contract()
    )
    result = module._require_frozen_esconv_auxiliary_measurement_contract(
        attestation_path, expected_train_labels_path=labels_path
    )
    assert result["conservative_utility_lambda"] == 1.0


def test_rejects_missing_measurement_contract(tmp_path):
    module = _load_preflight_module()
    labels_path = tmp_path / "action_labels.jsonl"
    write_jsonl(labels_path, [{"a": 1}])
    attestation_path = tmp_path / "attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage="esconv_auxiliary_judging_full_train",
        inputs={"config": labels_path},
        outputs={"labels": (labels_path, True)},
        parameters={},
    )
    with pytest.raises(RuntimeError, match="measurement_contract"):
        module._require_frozen_esconv_auxiliary_measurement_contract(
            attestation_path, expected_train_labels_path=labels_path
        )


def test_rejects_wrong_stage(tmp_path):
    module = _load_preflight_module()
    labels_path = tmp_path / "action_labels.jsonl"
    write_jsonl(labels_path, [{"a": 1}])
    attestation_path = tmp_path / "attestation.json"
    create_artifact_attestation(
        attestation_path,
        stage="esconv_auxiliary_judging_full_calibration",
        inputs={"config": labels_path},
        outputs={"labels": (labels_path, True)},
        parameters={"measurement_contract": _valid_measurement_contract()},
    )
    with pytest.raises(RuntimeError):
        module._require_frozen_esconv_auxiliary_measurement_contract(
            attestation_path, expected_train_labels_path=labels_path
        )


def test_rejects_mismatched_labels_path(tmp_path):
    module = _load_preflight_module()
    attested_labels_path = tmp_path / "attested_labels.jsonl"
    write_jsonl(attested_labels_path, [{"a": 1}])
    other_labels_path = tmp_path / "other_labels.jsonl"
    write_jsonl(other_labels_path, [{"a": 1}])
    attestation_path = _write_attestation(
        tmp_path,
        labels_path=attested_labels_path,
        measurement_contract=_valid_measurement_contract(),
    )
    with pytest.raises(RuntimeError):
        module._require_frozen_esconv_auxiliary_measurement_contract(
            attestation_path, expected_train_labels_path=other_labels_path
        )


def test_rejects_non_fixed_lambda(tmp_path):
    module = _load_preflight_module()
    labels_path = tmp_path / "action_labels.jsonl"
    write_jsonl(labels_path, [{"a": 1}])
    contract = _valid_measurement_contract()
    contract["conservative_utility_lambda"] = 0.5
    attestation_path = _write_attestation(
        tmp_path, labels_path=labels_path, measurement_contract=contract
    )
    with pytest.raises(RuntimeError, match="lambda"):
        module._require_frozen_esconv_auxiliary_measurement_contract(
            attestation_path, expected_train_labels_path=labels_path
        )


def test_rejects_wrong_clamp_range(tmp_path):
    module = _load_preflight_module()
    labels_path = tmp_path / "action_labels.jsonl"
    write_jsonl(labels_path, [{"a": 1}])
    contract = _valid_measurement_contract()
    contract["risk_dimension_clamp_range"] = [0.0, 5.0]
    attestation_path = _write_attestation(
        tmp_path, labels_path=labels_path, measurement_contract=contract
    )
    with pytest.raises(RuntimeError, match="clamp"):
        module._require_frozen_esconv_auxiliary_measurement_contract(
            attestation_path, expected_train_labels_path=labels_path
        )


def test_rejects_code_drift_since_freeze(tmp_path):
    module = _load_preflight_module()
    labels_path = tmp_path / "action_labels.jsonl"
    write_jsonl(labels_path, [{"a": 1}])
    contract = _valid_measurement_contract()
    contract["code_manifest"]["pm_v2_model"]["sha256"] = "0" * 64
    attestation_path = _write_attestation(
        tmp_path, labels_path=labels_path, measurement_contract=contract
    )
    with pytest.raises(RuntimeError, match="changed since"):
        module._require_frozen_esconv_auxiliary_measurement_contract(
            attestation_path, expected_train_labels_path=labels_path
        )


def _self_hashed_contract(status: str) -> dict:
    payload = {"status": status, "note": "test fixture"}
    return {**payload, "contract_sha256": sha256_text(canonical_json(payload))}


def test_instrument_freeze_guard_fails_closed_on_a_missing_contract(tmp_path):
    module = _load_preflight_module()
    missing_path = tmp_path / "no_such_contract.json"
    assert not missing_path.exists()
    # Unlike an earlier, more permissive version of this guard, a missing
    # contract must now itself fail closed -- dual-domain training may not
    # silently proceed just because no freeze record has been produced yet.
    with pytest.raises(RuntimeError, match="missing"):
        module._require_content_addressed_instrument_freeze_contract(
            missing_path,
            blocking_status="SOME_BLOCKING_STATUS",
            contract_label="test instrument",
        )


def test_instrument_freeze_guard_fails_closed_on_hash_mismatch(tmp_path):
    module = _load_preflight_module()
    contract_path = tmp_path / "contract.json"
    contract = _self_hashed_contract("SOME_OTHER_STATUS")
    contract["contract_sha256"] = "0" * 64
    write_json(contract_path, contract)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        module._require_content_addressed_instrument_freeze_contract(
            contract_path,
            blocking_status="SOME_BLOCKING_STATUS",
            contract_label="test instrument",
        )


def test_instrument_freeze_guard_passes_when_status_is_not_the_blocking_value(tmp_path):
    module = _load_preflight_module()
    contract_path = tmp_path / "contract.json"
    write_json(contract_path, _self_hashed_contract("SOME_OTHER_STATUS"))
    result = module._require_content_addressed_instrument_freeze_contract(
        contract_path,
        blocking_status="SOME_BLOCKING_STATUS",
        contract_label="test instrument",
    )
    assert result["status"] == "SOME_OTHER_STATUS"


def test_instrument_freeze_guard_fails_closed_on_the_blocking_status(tmp_path):
    module = _load_preflight_module()
    contract_path = tmp_path / "contract.json"
    write_json(contract_path, _self_hashed_contract("SOME_BLOCKING_STATUS"))
    with pytest.raises(RuntimeError, match="SOME_BLOCKING_STATUS"):
        module._require_content_addressed_instrument_freeze_contract(
            contract_path,
            blocking_status="SOME_BLOCKING_STATUS",
            contract_label="test instrument",
        )


def test_instrument_freeze_guard_matches_the_real_frozen_absolute_contract():
    module = _load_preflight_module()
    real_contract_path = (
        ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "esconv_auxiliary_absolute_instrument_freeze_v1.json"
    )
    assert real_contract_path.is_file()
    with pytest.raises(RuntimeError, match="NOT_SUPPORTED"):
        module._require_content_addressed_instrument_freeze_contract(
            real_contract_path,
            blocking_status=module.ESCONV_AUXILIARY_ABSOLUTE_LABEL_INSTRUMENT_NOT_SUPPORTED,
            contract_label="ESConv-auxiliary absolute-label instrument freeze",
        )


def test_instrument_freeze_guard_matches_the_real_frozen_pairwise_contract():
    module = _load_preflight_module()
    real_contract_path = (
        ROOT
        / "data"
        / "pm_v1_5_contracts"
        / "esconv_auxiliary_pairwise_instrument_freeze_v1.json"
    )
    assert real_contract_path.is_file()
    with pytest.raises(RuntimeError, match="NOT_SUPPORTED"):
        module._require_content_addressed_instrument_freeze_contract(
            real_contract_path,
            blocking_status=module.ESCONV_AUXILIARY_PAIRWISE_INSTRUMENT_NOT_SUPPORTED,
            contract_label="ESConv-auxiliary pairwise instrument freeze",
        )
