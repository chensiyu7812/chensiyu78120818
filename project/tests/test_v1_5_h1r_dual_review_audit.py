from __future__ import annotations

import importlib.util
from pathlib import Path

from metacom_pm.io import read_json


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/25t_audit_final_h1r_reviews_v1_5.py"
AUDIT = ROOT / "outputs/pm_v1_5_final_h1r_dual_review_audit_v1/audit_report.json"


def _module():
    spec = importlib.util.spec_from_file_location("h1r_dual_review_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h1r_derived_decision_keeps_two_gate_contract() -> None:
    module = _module()
    assert module._derived({"applicability_safe": "yes", "incremental_over_r0": "yes"}) == "on"
    assert module._derived({"applicability_safe": "yes", "incremental_over_r0": "no"}) == "off"
    assert module._derived({"applicability_safe": "uncertain", "incremental_over_r0": "yes"}) == "abstain"


def test_saved_h1r_audit_attributes_failure_before_pm_and_generator() -> None:
    report = read_json(AUDIT)
    assert report["status"] == "HOLD_DO_NOT_FREEZE_GOLD_OR_TRAIN"
    assert report["schema"]["errors"] == []
    assert report["pre_adjudication_iaa"]["n_component_pairs"] == 96
    assert report["pre_adjudication_iaa"]["decision_raw_agreement"] == 80 / 96
    assert report["pre_adjudication_iaa"]["per_component"]["ME"]["decision_raw_agreement"] == 14 / 24
    assert "PM_model_predictions" in report["not_assessed"]
    assert "STEP2_generator_execution" in report["not_assessed"]
