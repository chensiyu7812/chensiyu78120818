from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_strategy_g2_real_qualification_v1"


def _report() -> dict:
    return json.loads(
        (OUT / "real_qualification_report.json").read_text(encoding="utf-8")
    )


def test_real_g2_uses_strict_g1_lineage_disjoint_train_holdout() -> None:
    report = _report()
    assert report["universe"]["validation_test_evoemo_rows"] == 0
    assert report["universe"]["primary_strict_holdout_dialogues"] >= 30
    assert report["qualification_checks"][
        "strict_holdout_has_zero_g1_lineage_overlap"
    ]
    assert report["external_or_test_outcomes_used"] is False


def test_real_g2_freeze_is_top1_fail_closed_and_boundary_clean() -> None:
    report = _report()
    assert report["status"] == "RAG_QUALIFIED_FOR_G3_CLEAN_TREATMENT"
    assert report["promotion_authorized_for_g3"] is True
    assert report["formal_response_quality_benefit_proven"] is False
    assert report["qualification_checks"]["top1_cap_is_enforced"]
    assert report["qualification_checks"][
        "selected_ranker_has_zero_boundary_violations"
    ]
    assert report["qualification_checks"][
        "strict_dialogue_activation_rate_at_least_5_percent"
    ]
    freeze = json.loads(
        (OUT / "runtime_retrieval_freeze.json").read_text(encoding="utf-8")
    )
    assert freeze["status"] == "FROZEN_FOR_G3"
    assert freeze["top_k"] == 1
    assert freeze["no_eligible_or_below_floor"].startswith("abstain")
    assert freeze["raw_source_responses_available"] is False


def test_bge_requires_preregistered_material_gain() -> None:
    report = _report()
    decision = report["ranker_decision"]
    if decision["selected"] == "bge":
        assert decision["bge_promoted"] is True
        assert (
            decision["observed_bge_minus_lexical_proxy_agreement"]
            >= decision["bge_required_absolute_proxy_gain"]
        )
    else:
        assert decision["selected"] == "lexical"
        assert decision["bge_promoted"] is False
