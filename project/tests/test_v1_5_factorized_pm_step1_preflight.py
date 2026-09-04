from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24x_preflight_factorized_pm_step1_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("factorized_preflight", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_minimal_supplement_can_be_prepared_but_model_cannot_promote() -> None:
    report = _module().build_report(ROOT)
    assert (
        report["status"]
        == "READY_TO_PREPARE_CORRECTED_SUPPLEMENT_MODEL_PROMOTION_BLOCKED"
    )
    assert (
        report["data_collection_gate"]["status"]
        == "READY_TO_PREPARE_CORRECTED_16_PAIR_SUPPLEMENT"
    )
    assert (
        report["model_promotion_gate"]["status"]
        == "BLOCKED_BEFORE_MODEL_CONFIRMATION"
    )
    failed = set(report["model_promotion_gate"]["failed_checks"])
    assert "effect_pilot_has_minimum_independent_groups" in failed
    assert "primary_feature_dimension_within_capacity" in failed
    assert "primary_model_beats_prevalence_on_proper_scores" in failed
    assert "quality_risk_cost_are_not_collapsed_into_one_training_head" in failed
    assert "atomic_risk_heads_have_event_and_nonevent_support" in failed


def test_preflight_preserves_unseen_wave2_and_rejects_baai_promotion() -> None:
    report = _module().build_report(ROOT)
    checks = {item["check_id"]: item for item in report["checks"]}
    assert checks["contract_frozen_before_wave2_outcomes"]["passed"]
    assert checks["full_step1_opportunity_layer_exists"]["passed"]
    assert checks["corrected_atomic_risk_review_contract_frozen"]["passed"]
    assert checks["wave1_has_both_quality_effect_directions"]["passed"]
    assert checks["fresh_pool_can_supply_minimal_16_group_supplement"]["passed"]
    assert checks["baai_not_promoted_after_negative_diagnostic"]["passed"]
    assert checks["old_wave2_freeze_is_secondary_only"]["passed"]
    assert checks["wave2_plan_does_not_claim_execution_authority"]["passed"]
    assert not (
        ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v2_confirmation_execution"
        / "generation_outcomes.jsonl"
    ).exists()
