from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_strategy_g2_controlled_retrieval_v1"


def test_g2_control_suite_is_not_misreported_as_real_qualification() -> None:
    report = json.loads(
        (OUT / "controlled_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == (
        "CONTROL_SUITE_COMPLETE_REAL_G2_QUALIFICATION_NOT_STARTED"
    )
    assert report["bank_card_count"] == 6
    assert report["synthetic_positive_cases"] == 42
    assert report["synthetic_boundary_or_negative_controls"] == 12
    assert report["synthetic_cases_count_toward_real_qualification_n"] is False
    assert report["promotion_authorized"] is False
    assert report["external_or_test_outcomes_used"] is False
    assert report["query_includes_expected_move_id"] is False


def test_g2_eligibility_controls_all_pass() -> None:
    report = json.loads(
        (OUT / "controlled_report.json").read_text(encoding="utf-8")
    )
    assert report["eligibility_expected_set_passes"] == 54
    assert report["eligibility_expected_set_denominator"] == 54
    assert report["lexical_control_top1_passes"] == 12
    assert report["bge_control_top1_passes"] == 12
