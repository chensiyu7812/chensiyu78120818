"""Tests for scripts/v1_5/13c_judge_esconv_auxiliary_v1_5.py's
``measurement_contract_record``.

This is the freeze record 21a_preflight_dual_domain_training_v1_5.py's
``_require_frozen_esconv_auxiliary_measurement_contract`` requires and
re-verifies before dual-domain training may proceed (see
test_v1_5_dual_domain_preflight_measurement_contract_guard.py). Only the
record-building function is tested here, not the full judging script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "v1_5" / "13c_judge_esconv_auxiliary_v1_5.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "v1_5_13c_judge_esconv_auxiliary", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_measurement_contract_binds_lambda_clamp_ranges_and_applicability():
    module = _load_module()
    action_ids = ["M0+R0", "M0+RS"]
    inapplicable = module.dimensions_inapplicable_to_every_action(action_ids)
    by_action = module.dimension_applicability_by_action(action_ids)
    record = module.measurement_contract_record(
        action_ids_present=action_ids,
        inapplicable_risk_dimensions=inapplicable,
        inapplicable_risk_dimensions_by_action=by_action,
    )
    assert record["protocol"] == module.MEASUREMENT_CONTRACT_PROTOCOL
    assert record["conservative_utility_lambda"] == 1.0
    assert record["conservative_utility_lambda_is_fixed_never_tuned"] is True
    assert record["response_dimension_clamp_range"] == [1.0, 5.0]
    assert record["risk_dimension_clamp_range"] == [0.0, 3.0]
    assert set(record["inapplicable_risk_dimensions"]) == {
        "risk.selected_context_misuse",
        "risk.stale_or_conflicting_use",
        "risk.unnecessary_exposure",
    }
    assert "risk.strategy_overuse" in record["inapplicable_risk_dimensions_by_action"]["M0+R0"]
    assert (
        "risk.strategy_overuse"
        not in record["inapplicable_risk_dimensions_by_action"]["M0+RS"]
    )
    # Every code_manifest entry's recorded hash must match the real file on
    # disk right now (this test's own run *is* "no drift since the freeze").
    for entry in record["code_manifest"].values():
        assert module.sha256_file(ROOT / entry["relative_path"]) == entry["sha256"]


def test_measurement_contract_hash_is_order_independent_and_sensitive():
    module = _load_module()
    forward = ["M0+R0", "M0+RS"]
    backward = ["M0+RS", "M0+R0"]
    by_action_forward = module.dimension_applicability_by_action(forward)
    by_action_backward = module.dimension_applicability_by_action(backward)
    inapplicable = module.dimensions_inapplicable_to_every_action(forward)
    record_forward = module.measurement_contract_record(
        action_ids_present=forward,
        inapplicable_risk_dimensions=inapplicable,
        inapplicable_risk_dimensions_by_action=by_action_forward,
    )
    record_backward = module.measurement_contract_record(
        action_ids_present=backward,
        inapplicable_risk_dimensions=inapplicable,
        inapplicable_risk_dimensions_by_action=by_action_backward,
    )
    assert record_forward["contract_sha256"] == record_backward["contract_sha256"]

    single_action_record = module.measurement_contract_record(
        action_ids_present=["M0+R0"],
        inapplicable_risk_dimensions=module.dimensions_inapplicable_to_every_action(
            ["M0+R0"]
        ),
        inapplicable_risk_dimensions_by_action=module.dimension_applicability_by_action(
            ["M0+R0"]
        ),
    )
    assert single_action_record["contract_sha256"] != record_forward["contract_sha256"]
