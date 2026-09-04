from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.io import canonical_json, sha256_text, write_json
from metacom_pm.pm_v2_forced_swap import forced_swap_unit_id, select_forced_swap_units
from metacom_pm.v1_5_external_batched import PROTOCOL as BATCHED_PROTOCOL
from metacom_pm.v1_5_external_claims import PROTOCOL as CLAIM_PROTOCOL
from metacom_pm.v1_5_external_claims import assess_external_claims
from metacom_pm.v1_5_forced_swap_canary import PROTOCOL as CANARY_PROTOCOL
from metacom_pm.v1_5_forced_swap_canary import require_v1_5_forced_swap_canary
from metacom_pm.v1_5_latency import (
    build_descriptive_latency_report,
    build_separated_resource_report,
)


def _claim_summary(
    quality_lower: float, cost_upper: float, *, primary_cluster: str = "user_id"
) -> dict:
    return {
        "status": "COMPLETE",
        "protocol": BATCHED_PROTOCOL,
        "scoring_mode": "anonymous_five_candidate_batched",
        "primary_quality_judge_family": "openai_gpt4o",
        "quality_gate": {"status": "PASS"},
        "candidate_position_gate": {"status": "PASS"},
        "composite_independence_gate": {"status": "PASS"},
        "quality_sensitivity": {
            "role": "sensitivity_only_not_pooled",
            "judge_family": "anthropic_claude",
            "reporting_required": True,
            "gates_primary_claim": False,
            "direction_agreement": True,
            "quality_composite_spearman": 0.5,
        },
        "stratified_risk_audit": {
            "status": "COMPLETE",
            "role": "preregistered_stratified_audit_not_population_safety_claim",
            "finding": "NO_INCREASE_SIGNAL_WITHIN_FROZEN_MARGIN",
            "paper_statement_allowed": True,
            "allowed_statement": "bounded statement",
            "claim_boundary": "not a population safety claim",
        },
        "gate_e": {
            "status": (
                "PASS"
                if quality_lower >= -0.02 and cost_upper < 0.0
                else "NOT_SUPPORTED"
            ),
            "comparator": "best_fixed",
            "checks": {
                "quality_noninferior": quality_lower >= -0.02,
                "evidence_risk_nonincrease": True,
                "generator_input_tokens_strictly_lower": cost_upper < 0.0,
                "internal_gate_m_passed": True,
                "internal_gate_f_passed": True,
                "lineage_complete": True,
            },
        },
        "paired_treatment_deltas": {
            "best_fixed": {
                "quality_composite": {
                    "primary_cluster": primary_cluster,
                    "primary_cluster_ci": {
                        "estimate": 0.0,
                        "lower": quality_lower,
                        "upper": 0.04,
                    },
                },
                "observed_input_tokens": {
                    "primary_cluster": primary_cluster,
                    "primary_cluster_ci": {
                        "estimate": -25.0,
                        "lower": -40.0,
                        "upper": cost_upper,
                    },
                },
            }
        },
    }


def _claim_contract() -> dict:
    return {
        "protocol": CLAIM_PROTOCOL,
        "quality_judging_mode": "anonymous_five_candidate_batched",
        "primary_quality_judge_family": "openai_gpt4o",
        "sensitivity_judge_family": "anthropic_claude",
        "sensitivity_role": "sensitivity_only_not_pooled",
        "risk_audit_role": "preregistered_stratified_audit_not_population_safety_claim",
        "primary_quality_and_cost_comparator": "best_fixed",
        "quality_metric": "quality_composite",
        "quality_noninferiority_margin": 0.02,
        "cost_metric": "observed_input_tokens",
        "maximum_cost_delta_ci_upper": 0.0,
        "primary_cluster": "user_id",
    }


def test_external_claim_requires_both_quality_noninferiority_and_cost_superiority():
    supported = assess_external_claims(_claim_summary(-0.02, -1.0), _claim_contract())
    assert supported["status"] == "SUPPORTED"
    quality_fail = assess_external_claims(
        _claim_summary(-0.021, -1.0), _claim_contract()
    )
    cost_fail = assess_external_claims(_claim_summary(-0.01, 0.0), _claim_contract())
    assert quality_fail["status"] == "NOT_SUPPORTED"
    assert cost_fail["status"] == "NOT_SUPPORTED"
    assert supported["latency_claim_status"] == "DESCRIPTIVE_ONLY_NOT_CONFIRMATORY"
    assert supported["primary_comparator_interpretation"].startswith(
        "structured_high_resource"
    )
    assert supported["quality_sensitivity"]["affects_primary_verdict"] is False
    assert supported["risk_audit"]["paper_statement_allowed"] is True


def test_external_claim_rejects_a_ci_not_stamped_as_the_frozen_primary_cluster():
    summary = _claim_summary(-0.02, -1.0, primary_cluster="scenario")
    with pytest.raises(RuntimeError, match="not stamped as the frozen primary_cluster"):
        assess_external_claims(summary, _claim_contract())


def test_external_claim_rejects_comparator_semantic_drift():
    contract = {**_claim_contract(), "primary_quality_and_cost_comparator": "other"}
    with pytest.raises(RuntimeError, match="structured-high-resource"):
        assess_external_claims(_claim_summary(-0.02, -1.0), contract)


def test_latency_report_is_complete_but_explicitly_nonconfirmatory():
    units = [("u1", 0, 101, "seeker", 3), ("u2", 0, 101, "seeker", 3)]
    rows = []
    for condition, base in (("pm", 80.0), ("fixed", 100.0)):
        for user_id, topic, seed, simulator, turn in units:
            rows.append(
                {
                    "condition": condition,
                    "user_id": user_id,
                    "topic_index": topic,
                    "seed": seed,
                    "simulator_id": simulator,
                    "turn_index": turn,
                    "latency_ms": base,
                    "cost": {
                        "latency_ms": base,
                        "generation_latency_ms": base - 10.0,
                        "retrieval_latency_ms": 4.0,
                        "pm_inference_ms": 2.0,
                        "pre_evidence_compute_ms": 1.0,
                    },
                }
            )
    report = build_descriptive_latency_report(
        rows, conditions=["pm", "fixed"], treatment="pm", expected_units=units
    )
    assert report["status"] == "COMPLETE"
    assert report["confirmatory_latency_claim_allowed"] is False
    assert report["paired_point_deltas"]["fixed"]["total_latency_ms"][
        "mean"
    ] == -20.0


def test_resource_accounting_keeps_compute_retrieval_tokens_latency_and_usd_separate():
    units = [("u1", 0, 101, "seeker", 3), ("u2", 0, 101, "seeker", 3)]
    rows = []
    for condition in ("pm", "fixed"):
        for user_id, topic, seed, simulator, turn in units:
            rows.append(
                {
                    "condition": condition,
                    "user_id": user_id,
                    "topic_index": topic,
                    "seed": seed,
                    "simulator_id": simulator,
                    "turn_index": turn,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "latency_ms": 50.0,
                    "cost": {
                        "step0_memory_comparisons": 3 if condition == "pm" else 0,
                        "step0_strategy_family_comparisons": 8 if condition == "pm" else 0,
                        "step0_latency_ms": 1.0 if condition == "pm" else 0.0,
                        "retrieval_calls": 1,
                        "retrieval_latency_ms": 2.0,
                    },
                }
            )
    report = build_separated_resource_report(
        rows,
        conditions=["pm", "fixed"],
        expected_units=units,
        generator_pricing_usd_per_mtok={"input": 0.15, "output": 0.60},
    )
    assert report["aggregation"] == "separate_metrics_no_composite_cost"
    pm = report["condition_summaries"]["pm"]
    assert pm["step0_memory_comparisons"]["total"] == 6.0
    assert pm["generator_input_tokens"]["total"] == 200.0
    assert pm["generator_api_cost_usd"]["total"] > 0.0


def test_forced_swap_canary_verifier_accepts_compatibility_without_efficacy(
    tmp_path: Path,
):
    units = [(f"u{i}", 1, 101, "seeker_main", 3) for i in range(20)]
    contract = {
        "protocol": CANARY_PROTOCOL,
        "treatment": "pm_v2",
        "baseline": "pm_v2_cost_matched_fixed",
        "sample_units": 4,
        "judge_seed": 17,
    }
    selected = select_forced_swap_units(units, sample_units=4, seed=17)
    gate = {"status": "PASS", "checks": {"schema": True, "order": True}}
    freeze_sha = "a" * 64
    summary = {
        "status": "PASS",
        "study_freeze_sha256": freeze_sha,
        "treatment": contract["treatment"],
        "baseline": contract["baseline"],
        "sample_units": 4,
        "sample_units_sha256": sha256_text(canonical_json(selected)),
        "excluded_unit_ids": [forced_swap_unit_id(unit) for unit in selected],
        "order_variants": [0, 1],
        "judge_seed": 17,
        "full_expected_units_sha256": sha256_text(canonical_json(units)),
        "compatibility_gate": gate,
        "support_key_claim_verified": False,
    }
    summary_path = tmp_path / "summary.json"
    input_path = tmp_path / "input.json"
    attestation_path = tmp_path / "attestation.json"
    write_json(summary_path, summary)
    write_json(input_path, {"fixture": True})
    create_artifact_attestation(
        attestation_path,
        stage="pm_v2_forced_swap_key_claim",
        inputs={"fixture": input_path},
        outputs={"summary": (summary_path, False)},
        parameters={"compatibility_gate": gate},
        study_freeze_sha256=freeze_sha,
    )
    verified = require_v1_5_forced_swap_canary(
        summary_path,
        attestation_path,
        study_freeze_sha256=freeze_sha,
        full_expected_units=units,
        contract=contract,
    )
    assert verified["status"] == "PASS"
    assert verified["efficacy_is_not_required_from_canary"] is True


def test_forced_swap_canary_rejects_nonpassing_measurement_gate(tmp_path: Path):
    units = [(f"u{i}", 1, 101, "seeker_main", 3) for i in range(5)]
    contract = {
        "protocol": CANARY_PROTOCOL,
        "treatment": "pm_v2",
        "baseline": "pm_v2_cost_matched_fixed",
        "sample_units": 2,
        "judge_seed": 9,
    }
    selected = select_forced_swap_units(units, sample_units=2, seed=9)
    gate = {"status": "NONREPORTABLE", "checks": {"schema": True, "order": False}}
    freeze_sha = "b" * 64
    summary = {
        "status": "PASS",
        "study_freeze_sha256": freeze_sha,
        "treatment": contract["treatment"],
        "baseline": contract["baseline"],
        "sample_units": 2,
        "sample_units_sha256": sha256_text(canonical_json(selected)),
        "excluded_unit_ids": [forced_swap_unit_id(unit) for unit in selected],
        "order_variants": [0, 1],
        "judge_seed": 9,
        "full_expected_units_sha256": sha256_text(canonical_json(units)),
        "compatibility_gate": gate,
    }
    summary_path = tmp_path / "summary.json"
    input_path = tmp_path / "input.json"
    attestation_path = tmp_path / "attestation.json"
    write_json(summary_path, summary)
    write_json(input_path, {})
    create_artifact_attestation(
        attestation_path,
        stage="pm_v2_forced_swap_key_claim",
        inputs={"fixture": input_path},
        outputs={"summary": (summary_path, False)},
        parameters={"compatibility_gate": gate},
        study_freeze_sha256=freeze_sha,
    )
    with pytest.raises(RuntimeError, match="did not PASS"):
        require_v1_5_forced_swap_canary(
            summary_path,
            attestation_path,
            study_freeze_sha256=freeze_sha,
            full_expected_units=units,
            contract=contract,
        )
