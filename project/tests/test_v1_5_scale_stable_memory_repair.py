from __future__ import annotations

import importlib.util
from pathlib import Path

from metacom_pm.text import content_word_match_level


ROOT = Path(__file__).resolve().parents[1]


def _module(filename: str):
    path = ROOT / "scripts/v1_5" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_content_word_level_is_bounded_and_length_stable() -> None:
    assert content_word_match_level("the and but", "the and but") == 0.0
    assert content_word_match_level("choir tonight", "choir rehearsal") == 0.5
    assert content_word_match_level(
        "choir tonight tense " + "and the " * 100,
        "choir rehearsal felt tense",
    ) == 1.0


def test_v3_is_confirmation_only_balanced_and_outcome_blind() -> None:
    report = _module(
        "24eb_freeze_scale_stable_memory_confirmation_v1_5.py"
    ).build(root=ROOT)
    assert report["status"] == (
        "PASS_FROZEN_UNTOUCHED_SCALE_STABLE_CONFIRMATION_V3"
    )
    assert report["checks"]["expected_new_users_and_roles"]
    assert report["checks"]["expected_formal_rows"]
    assert report["checks"]["zero_api_zero_outcome"]
    assert set(report["counts"]) == {
        "untouched_confirmation:MP",
        "untouched_confirmation:MS",
        "untouched_confirmation:ME",
    }


def test_one_repair_result_is_partial_and_not_retuned_to_pass() -> None:
    _module("24ec_recompile_scale_stable_memory_fit_v1_5.py").build(root=ROOT)
    report = _module("24ed_train_scale_stable_memory_heads_v1_5.py").build(
        root=ROOT
    )
    assert report["repair_budget"] == (
        "ONE_SCALE_STABLE_REPRESENTATION_REPAIR_CONSUMED"
    )
    assert report["status"] == "FAIL_SCALE_STABLE_MEMORY_HEADS"
    assert report["components"]["ME"]["status"] == "PASS_PROMOTED"
    assert report["components"]["MP"]["status"] == "FAIL_NOT_PROMOTED"
    assert report["components"]["MS"]["status"] == "FAIL_NOT_PROMOTED"
    assert report["v1_v2_confirmation_reused"] is False
    assert report["external_quality_risk_or_response_outcome_read"] is False


def test_external_transport_uses_eligible_denominator_and_fails_closed() -> None:
    report = _module(
        "24ee_audit_scale_stable_evoemo_transport_v1_5.py"
    ).build(root=ROOT)
    assert report["status"] == "PASS_REPRESENTATION_TRANSPORT_PARTIAL_HEADS_ONLY"
    assert report["promoted_heads"] == ["ME"]
    assert report["unpromoted_heads_fail_closed"] == ["MP", "MS"]
    assert report["checks"][
        "minimum_0_80_outcome_blind_transport_coverage_each_source"
    ]
    assert all(
        report["components"][component]["head_eligible_in_support_rate"] >= 0.8
        for component in ("MP", "MS", "ME")
    )
    assert report["components"]["MP"]["predicted_on"] == 0
    assert report["components"]["MS"]["predicted_on"] == 0
