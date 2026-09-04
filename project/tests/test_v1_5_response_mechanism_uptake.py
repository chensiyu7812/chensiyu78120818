from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

from metacom_pm.io import canonical_json, read_json, sha256_text
from metacom_pm.v1_5_response_mechanism_uptake import (
    UPTAKE_MEASUREMENT_PROTOCOL,
    _directional_status,
    uptake_measurement_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/longitudinal_response_mechanism_pilot_v1.json"
)
MEASUREMENT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/"
    "longitudinal_response_mechanism_uptake_measurement_v1.json"
)
FREEZE_CONTRACT_PATH = (
    PROJECT_ROOT
    / "data/pm_v1_5_contracts/"
    "longitudinal_response_mechanism_uptake_freeze_v1.json"
)


def _runner_module():
    path = (
        PROJECT_ROOT
        / "scripts/v1_5/21f_run_longitudinal_response_mechanism_pilot_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location("uptake_bound_pilot_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_uptake_contract_is_self_hashing_and_report_only() -> None:
    contract = read_json(MEASUREMENT_CONTRACT_PATH)
    without_sha = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    assert contract["protocol"] == UPTAKE_MEASUREMENT_PROTOCOL
    assert contract["status"] == "FROZEN_BEFORE_PILOT_OUTCOMES"
    assert contract["contract_sha256"] == sha256_text(canonical_json(without_sha))
    assert contract["api_judges_used"] is False
    assert contract["training_labels_created"] is False
    assert "PM training authorization" in contract["forbidden_consequences"]


def test_contract_factory_binds_pilot_and_encoder() -> None:
    pilot = read_json(PILOT_CONTRACT_PATH)
    contract = uptake_measurement_contract(
        pilot_contract_sha256=pilot["contract_sha256"],
        semantic_encoder_spec_sha256="encoder-contract",
    )
    assert contract["pilot_contract_sha256"] == pilot["contract_sha256"]
    assert contract["semantic_encoder_spec_sha256"] == "encoder-contract"
    assert contract["directional_signal_rule"] == {
        "helpful_pair_count": 10,
        "minimum_helpful_positive_pairs": 6,
        "helpful_mean_delta_must_be": "greater_than_zero",
        "harmful_pair_count": 4,
        "minimum_harmful_negative_pairs": 3,
        "harmful_mean_delta_must_be": "less_than_zero",
        "primary_metric_only": True,
        "lexical_and_guardrail_metrics": "reported_not_gating",
    }


def test_real_pilot_freeze_is_self_hashing_and_cannot_authorize_training() -> None:
    freeze = read_json(FREEZE_CONTRACT_PATH)
    without_sha = {
        key: value for key, value in freeze.items() if key != "contract_sha256"
    }
    assert freeze["contract_sha256"] == sha256_text(canonical_json(without_sha))
    assert freeze["status"] == "MECHANISM_SIGNAL_NOT_ESTABLISHED_REPORT_ONLY"
    assert freeze["decision"] == (
        "NO_GO_FOR_TOP1_EVIDENCE_SURFACE_AS_LONGITUDINAL_SUPERVISION_RESCUE"
    )
    assert freeze["training_labels_created"] is False
    assert freeze["pm_training_authorized"] is False
    assert freeze["internal_test_opened"] is False


def test_directional_status_uses_only_predeclared_primary_rule() -> None:
    helpful = {
        "pairs": 10,
        "mean_target_evidence_bge_cosine_delta": 0.01,
        "positive_bge_pairs": 6,
    }
    harmful = {
        "pairs": 4,
        "mean_target_evidence_bge_cosine_delta": -0.01,
        "negative_bge_pairs": 3,
    }
    assert _directional_status(helpful, harmful) == (
        "MECHANISM_SIGNAL_PRESENT_REPORT_ONLY"
    )
    assert _directional_status(
        {**helpful, "positive_bge_pairs": 5}, harmful
    ) == "MECHANISM_SIGNAL_NOT_ESTABLISHED_REPORT_ONLY"
    assert _directional_status(
        helpful, {**harmful, "mean_target_evidence_bge_cosine_delta": 0.0}
    ) == "MECHANISM_SIGNAL_NOT_ESTABLISHED_REPORT_ONLY"


def test_real_run_entry_binds_both_config_paths_and_measurement_contract() -> None:
    module = _runner_module()
    source = inspect.getsource(module.main)
    assert "experiment_config_path=args.experiment_config" in source
    assert (
        "uptake_measurement_contract_path=args.uptake_measurement_contract"
        in source
    )
    assert module.CANONICAL_UPTAKE_MEASUREMENT_CONTRACT_PATH == (
        MEASUREMENT_CONTRACT_PATH
    )
    assert module.UPTAKE_MEASUREMENT_CODE_PATH.name == (
        "v1_5_response_mechanism_uptake.py"
    )
    assert module.UPTAKE_ANALYZER_PATH.name == (
        "21h_analyze_longitudinal_response_mechanism_uptake_v1_5.py"
    )
