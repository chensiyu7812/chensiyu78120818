from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.io import sha256_file, write_json
from metacom_pm.pm_v1_6_claims import (
    ComparatorGate,
    HierarchicalClaimContract,
    assess_hierarchical_claims,
)
from metacom_pm.pm_v1_6_contracts import (
    AlgorithmCandidate,
    CandidateFamilyManifest,
)
from metacom_pm.pm_v1_6_internal import (
    finish_internal_test,
    require_no_internal_inputs,
    reserve_internal_test_once,
)


def test_internal_dataset_sha_can_only_be_reserved_once(tmp_path: Path) -> None:
    internal_bundle = tmp_path / "sealed_internal_bundle.json"
    write_json(internal_bundle, {"protocol": "fixture", "content": "sealed"})
    internal_sha = sha256_file(internal_bundle)

    report = tmp_path / "candidate_report.json"
    checkpoint = tmp_path / "checkpoint.joblib"
    write_json(report, {"status": "COMPLETE"})
    checkpoint.write_bytes(b"fixture checkpoint")
    manifest = CandidateFamilyManifest(
        candidate_algorithms=[
            AlgorithmCandidate.ABSOLUTE_HGB,
            AlgorithmCandidate.STATE_CENTERED_HGB,
            AlgorithmCandidate.RULE_RELATIVE_HGB,
        ],
        selected_primary=AlgorithmCandidate.ABSOLUTE_HGB,
        selection_metric="mean_user_utility",
        one_standard_error_rule=True,
        complexity_order=[
            AlgorithmCandidate.ABSOLUTE_HGB,
            AlgorithmCandidate.STATE_CENTERED_HGB,
            AlgorithmCandidate.RULE_RELATIVE_HGB,
        ],
        train_dataset_sha256="1" * 64,
        calibration_dataset_sha256="2" * 64,
        internal_dataset_sha256=internal_sha,
        step0_contract_sha256="3" * 64,
        strong_router_config_sha256="4" * 64,
        selected_checkpoint_sha256=sha256_file(checkpoint),
        candidate_report_sha256=sha256_file(report),
    )
    manifest_path = tmp_path / "candidate_manifest.json"
    write_json(manifest_path, manifest.model_dump(mode="json"))
    ledger = tmp_path / "internal_consumption.jsonl"

    reservation = reserve_internal_test_once(
        manifest_path=manifest_path,
        internal_dataset_path=internal_bundle,
        ledger_path=ledger,
    )
    with pytest.raises(RuntimeError, match="already been reserved or consumed"):
        reserve_internal_test_once(
            manifest_path=manifest_path,
            internal_dataset_path=internal_bundle,
            ledger_path=ledger,
        )

    internal_report = tmp_path / "internal_report.json"
    write_json(internal_report, {"status": "INTERNAL_NOT_SUPPORTED"})
    finish_internal_test(
        reservation,
        status="COMPLETE",
        report_path=internal_report,
    )


def test_train_stage_rejects_internal_paths() -> None:
    with pytest.raises(RuntimeError, match="cannot read internal-test"):
        require_no_internal_inputs(["data/train.jsonl", "data/internal_test.jsonl"])


def _metric(lower: float, upper: float, estimate: float = 0.0) -> dict:
    return {
        "primary_cluster": "user_id",
        "primary_cluster_ci": {
            "estimate": estimate,
            "lower": lower,
            "upper": upper,
        },
    }


def _comparison(
    *,
    quality=(-0.01, 0.02),
    support=(-0.01, 0.02),
    risk=(-0.02, 0.0),
    utility=(0.01, 0.04),
    cost=(-0.20, -0.05),
) -> dict:
    return {
        "quality_composite": _metric(*quality),
        "emotional_support": _metric(*support),
        "resource_use_risk": _metric(*risk),
        "total_utility": _metric(*utility),
        "total_variable_cost": _metric(*cost),
    }


def _contract() -> HierarchicalClaimContract:
    learned_gate = ComparatorGate(
        quality_noninferiority_margin=0.02,
        emotional_support_noninferiority_margin=0.02,
        maximum_risk_delta_ci_upper=0.0,
        require_utility_strict_advantage=True,
        require_cost_strict_reduction=False,
    )
    high_gate = ComparatorGate(
        quality_noninferiority_margin=0.02,
        emotional_support_noninferiority_margin=0.02,
        maximum_risk_delta_ci_upper=0.0,
        require_utility_strict_advantage=False,
        require_cost_strict_reduction=True,
    )
    secondary = ComparatorGate(
        quality_noninferiority_margin=0.02,
        emotional_support_noninferiority_margin=0.02,
        maximum_risk_delta_ci_upper=0.0,
        require_utility_strict_advantage=False,
        require_cost_strict_reduction=False,
    )
    return HierarchicalClaimContract(
        learned_vs_rule=learned_gate,
        learned_vs_high_resource=high_gate,
        learned_vs_cost_matched=secondary,
        legacy_regression_quality_margin=0.02,
        legacy_regression_utility_margin=0.0,
    )


def _summary() -> dict:
    conditions = [
        "learned_pm_step0",
        "strong_rule_step0",
        "cost_matched_fixed",
        "best_high_resource_fixed",
        "me_r0_legacy_anchor",
        "m0_r0",
        "session_rag_rs",
    ]
    return {
        "status": "COMPLETE",
        "primary_cluster": "user_id",
        "conditions": conditions,
        "paired_treatment_deltas": {
            "strong_rule_step0": _comparison(),
            "cost_matched_fixed": _comparison(
                utility=(-0.01, 0.03), cost=(-0.05, 0.02)
            ),
            "best_high_resource_fixed": _comparison(
                utility=(-0.01, 0.03), cost=(-0.20, -0.05)
            ),
            "me_r0_legacy_anchor": _comparison(
                utility=(0.0, 0.03), cost=(-0.05, 0.02)
            ),
        },
    }


def test_hierarchical_claims_require_rule_high_resource_and_legacy_gates() -> None:
    result = assess_hierarchical_claims(_summary(), _contract())
    assert result["status"] == "SUPPORTED"
    assert result["claim_a_learned_routing_vs_rule"]["status"] == "SUPPORTED"
    assert result["claim_b_efficiency_vs_high_resource"]["status"] == "SUPPORTED"
    assert result["legacy_me_r0_regression_guard"]["status"] == "PASS"

    failed = _summary()
    failed["paired_treatment_deltas"]["strong_rule_step0"]["total_utility"] = _metric(
        -0.03, 0.01
    )
    result = assess_hierarchical_claims(failed, _contract())
    assert result["status"] == "NOT_SUPPORTED"
    assert result["claim_a_learned_routing_vs_rule"]["status"] == "NOT_SUPPORTED"
