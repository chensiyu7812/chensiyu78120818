"""Frozen PM-v1.5 claim assessment over the external paired-CI summary."""

from __future__ import annotations

from typing import Any, Mapping

from .v1_5_external_batched import PROTOCOL as BATCHED_EXTERNAL_PROTOCOL


PROTOCOL = "pm-v1.5-external-claim-assessment-v1"


def assess_external_claims(
    summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    if summary.get("status") != "COMPLETE":
        raise RuntimeError("external claim assessment requires a COMPLETE summary")
    if (
        summary.get("protocol") != BATCHED_EXTERNAL_PROTOCOL
        or summary.get("scoring_mode") != contract.get("quality_judging_mode")
        or summary.get("primary_quality_judge_family")
        != contract.get("primary_quality_judge_family")
        or (summary.get("quality_gate") or {}).get("status") != "PASS"
        or (summary.get("candidate_position_gate") or {}).get("status") != "PASS"
        or (summary.get("composite_independence_gate") or {}).get("status")
        != "PASS"
    ):
        raise RuntimeError(
            "external claim assessment requires the frozen anonymous batched primary analysis"
        )
    if contract.get("protocol") != PROTOCOL:
        raise RuntimeError("external claim-assessment contract is missing or stale")
    comparator = str(contract["primary_quality_and_cost_comparator"])
    if comparator != "best_fixed":
        raise RuntimeError(
            "PM-v1.5 primary comparator must be best_fixed, the frozen "
            "structured-high-resource condition"
        )
    comparisons = summary.get("paired_treatment_deltas") or {}
    if comparator not in comparisons:
        raise RuntimeError(f"external summary lacks primary comparator: {comparator}")
    metrics = comparisons[comparator]
    quality_metric = str(contract["quality_metric"])
    cost_metric = str(contract["cost_metric"])
    primary_cluster = str(contract["primary_cluster"])
    try:
        quality_metrics = metrics[quality_metric]
        cost_metrics = metrics[cost_metric]
        quality_ci = dict(quality_metrics["primary_cluster_ci"])
        cost_ci = dict(cost_metrics["primary_cluster_ci"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("external summary lacks frozen primary-cluster CIs") from exc
    # Defense in depth, not the only gate: the V1.5 batched evaluator already
    # hard-requires primary_bootstrap_cluster == "user_id" before it
    # will compute anything. This additionally proves the specific CI this
    # function reads is stamped as having come from that same cluster field,
    # so a future change to the shared evaluator's metric dict shape (e.g.
    # reordering which CI is "primary") cannot silently swap in a
    # differently-clustered interval without this function noticing.
    if (
        quality_metrics.get("primary_cluster") != primary_cluster
        or cost_metrics.get("primary_cluster") != primary_cluster
    ):
        raise RuntimeError(
            "external summary's primary-cluster CI is not stamped as the "
            f"frozen primary_cluster={primary_cluster!r}"
        )
    quality_margin = float(contract["quality_noninferiority_margin"])
    quality_lower = float(quality_ci["lower"])
    cost_upper = float(cost_ci["upper"])
    checks = {
        "quality_noninferior_to_primary_comparator": quality_lower
        >= -quality_margin,
        "observed_input_token_cost_strictly_lower": cost_upper
        < float(contract["maximum_cost_delta_ci_upper"]),
    }
    gate_e = dict(summary.get("gate_e") or {})
    if (
        gate_e.get("comparator") != comparator
        or not isinstance(gate_e.get("checks"), Mapping)
    ):
        raise RuntimeError("external summary lacks the frozen Gate E verdict")
    checks["gate_e_quality_noninferior"] = bool(
        gate_e["checks"].get("quality_noninferior")
    )
    checks["gate_e_evidence_risk_nonincrease"] = bool(
        gate_e["checks"].get("evidence_risk_nonincrease")
    )
    checks["gate_e_generator_input_tokens_strictly_lower"] = bool(
        gate_e["checks"].get("generator_input_tokens_strictly_lower")
    )
    checks["internal_gate_m_and_f_lineage"] = bool(
        gate_e["checks"].get("internal_gate_m_passed")
        and gate_e["checks"].get("internal_gate_f_passed")
        and gate_e["checks"].get("lineage_complete")
    )
    supported = gate_e.get("status") == "PASS" and all(checks.values())
    sensitivity = summary.get("quality_sensitivity") or {}
    if (
        sensitivity.get("role") != contract.get("sensitivity_role")
        or sensitivity.get("judge_family")
        != contract.get("sensitivity_judge_family")
        or not bool(sensitivity.get("reporting_required"))
        or bool(sensitivity.get("gates_primary_claim"))
    ):
        raise RuntimeError("external summary misstates the Claude sensitivity role")
    risk_audit = summary.get("stratified_risk_audit") or {}
    if (
        risk_audit.get("status") != "COMPLETE"
        or risk_audit.get("role") != contract.get("risk_audit_role")
    ):
        raise RuntimeError("external summary lacks the frozen bounded risk-audit role")
    risk_statement_allowed = bool(risk_audit.get("paper_statement_allowed"))
    return {
        "status": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "protocol": PROTOCOL,
        "primary_comparator": comparator,
        "primary_comparator_interpretation": (
            "structured_high_resource_fixed_MPMSME_plus_RS"
        ),
        "primary_cluster": str(contract["primary_cluster"]),
        "checks": checks,
        "quality": {
            "metric": quality_metric,
            "noninferiority_margin": quality_margin,
            "delta_ci": quality_ci,
        },
        "cost": {
            "metric": cost_metric,
            "maximum_delta_ci_upper": float(
                contract["maximum_cost_delta_ci_upper"]
            ),
            "delta_ci": cost_ci,
        },
        "gate_e": gate_e,
        "latency_claim_status": "DESCRIPTIVE_ONLY_NOT_CONFIRMATORY",
        "quality_sensitivity": {
            "judge_family": str(contract["sensitivity_judge_family"]),
            "role": str(contract["sensitivity_role"]),
            "direction_agreement": sensitivity.get("direction_agreement"),
            "quality_composite_spearman": sensitivity.get(
                "quality_composite_spearman"
            ),
            "affects_primary_verdict": False,
        },
        "risk_audit": {
            "role": str(contract["risk_audit_role"]),
            "finding": risk_audit.get("finding"),
            "paper_statement_allowed": risk_statement_allowed,
            "allowed_statement": (
                risk_audit.get("allowed_statement")
                if risk_statement_allowed
                else None
            ),
            "claim_boundary": risk_audit.get("claim_boundary"),
            "affects_gate_e_efficiency_verdict": True,
        },
        "claim": (
            "PM-v1.5 is externally non-inferior in frozen quality composite and "
            "uses fewer observed input tokens than the structured high-resource "
            "fixed comparator"
        ),
    }
