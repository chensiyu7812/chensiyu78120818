from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24ci_audit_ms_external_subdomain_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("ms_external_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ms_audit_finds_construct_drift_without_outcomes():
    report = _module().build_audit()
    assert report["status"] == "D1B_REQUIRED_BEFORE_D2_OR_RETRAINING"
    assert report["api_calls_made"] == 0
    assert not any(report["outcomes_read"].values())
    findings = report["implementation_findings"]
    assert findings["internal"]["fallback_rate"] == 0.5
    assert findings["external"]["supplied_summary_rate"] == 1.0
    assert not findings["cross_session_pattern_compiler_implemented"]


def test_old_ms_pairs_are_not_silently_reused():
    report = _module().build_audit()
    pairs = report["implementation_findings"]["old_ms_pair_lineage"]
    assert pairs["pairs"] == 64
    assert pairs["pairs_with_fallback_item"] == 59
    assert pairs["pairs_with_only_supplied_summaries"] == 5
    assert pairs["pairs_clean_under_narrowed_v1_5_contract"] == 4
    repairs = report["required_repairs_before_D2"]
    assert any("Downgrade all 64 old MS pairs" in row for row in repairs)


def test_evoemo_is_feature_subdomain_not_content_subset():
    report = _module().build_audit()
    contract = report["external_subdomain_contract"]
    assert contract["current_exact_mechanism_pass"] is True
    assert contract["semantic_subtype_gate"] == "NOT_YET_PASSED"
    assert contract["current_descriptor_support"] == {
        "MP": 1.0,
        "MS": 0.8676470588235294,
        "ME": 0.8921568627450981,
    }
    assert report["revised_D2_budget"]["new_response_api_calls"] == 144
    assert report["revised_D2_budget"]["total_human_decisions"] == 104
