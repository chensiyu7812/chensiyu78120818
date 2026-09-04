from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .pm_v1_6_contracts import CLAIM_PROTOCOL


class ComparatorGate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    quality_noninferiority_margin: float = Field(ge=0.0)
    emotional_support_noninferiority_margin: float = Field(ge=0.0)
    maximum_risk_delta_ci_upper: float
    require_utility_strict_advantage: bool
    require_cost_strict_reduction: bool


class HierarchicalClaimContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: str = CLAIM_PROTOCOL
    learned_condition: str = "learned_pm_step0"
    rule_condition: str = "strong_rule_step0"
    cost_matched_condition: str = "cost_matched_fixed"
    high_resource_condition: str = "best_high_resource_fixed"
    legacy_me_r0_condition: str = "me_r0_legacy_anchor"
    minimal_condition: str = "m0_r0"
    empirical_session_rag_condition: str = "session_rag_rs"
    primary_cluster: str = "user_id"
    learned_vs_rule: ComparatorGate
    learned_vs_high_resource: ComparatorGate
    learned_vs_cost_matched: ComparatorGate
    legacy_regression_quality_margin: float = Field(ge=0.0)
    legacy_regression_utility_margin: float = Field(ge=0.0)

    @model_validator(mode="after")
    def unique_conditions(self) -> "HierarchicalClaimContract":
        values = [
            self.learned_condition,
            self.rule_condition,
            self.cost_matched_condition,
            self.high_resource_condition,
            self.legacy_me_r0_condition,
            self.minimal_condition,
            self.empirical_session_rag_condition,
        ]
        if len(values) != len(set(values)):
            raise ValueError("external condition IDs must be unique")
        return self


def _metric_ci(
    summary: Mapping[str, Any],
    comparator: str,
    metric: str,
    *,
    primary_cluster: str,
) -> dict[str, float]:
    try:
        row = summary["paired_treatment_deltas"][comparator][metric]
        if row.get("primary_cluster") != primary_cluster:
            raise RuntimeError("paired CI cluster differs from frozen contract")
        ci = row["primary_cluster_ci"]
        return {
            "estimate": float(ci["estimate"]),
            "lower": float(ci["lower"]),
            "upper": float(ci["upper"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"external summary lacks {comparator}/{metric} primary CI"
        ) from exc


def _assess_comparator(
    summary: Mapping[str, Any],
    *,
    comparator: str,
    gate: ComparatorGate,
    primary_cluster: str,
) -> dict[str, Any]:
    quality = _metric_ci(
        summary, comparator, "quality_composite", primary_cluster=primary_cluster
    )
    support = _metric_ci(
        summary, comparator, "emotional_support", primary_cluster=primary_cluster
    )
    risk = _metric_ci(
        summary, comparator, "resource_use_risk", primary_cluster=primary_cluster
    )
    utility = _metric_ci(
        summary, comparator, "total_utility", primary_cluster=primary_cluster
    )
    cost = _metric_ci(
        summary, comparator, "total_variable_cost", primary_cluster=primary_cluster
    )
    checks = {
        "quality_noninferior": quality["lower"]
        >= -gate.quality_noninferiority_margin,
        "emotional_support_noninferior": support["lower"]
        >= -gate.emotional_support_noninferiority_margin,
        "risk_not_increased": risk["upper"] <= gate.maximum_risk_delta_ci_upper,
        "utility_strict_advantage": (
            utility["lower"] > 0.0
            if gate.require_utility_strict_advantage
            else True
        ),
        "cost_strict_reduction": (
            cost["upper"] < 0.0 if gate.require_cost_strict_reduction else True
        ),
    }
    return {
        "status": "SUPPORTED" if all(checks.values()) else "NOT_SUPPORTED",
        "checks": checks,
        "quality": quality,
        "emotional_support": support,
        "resource_use_risk": risk,
        "total_utility": utility,
        "total_variable_cost": cost,
    }


def assess_hierarchical_claims(
    summary: Mapping[str, Any],
    contract: HierarchicalClaimContract,
) -> dict[str, Any]:
    if summary.get("status") != "COMPLETE":
        raise RuntimeError("claim assessment requires a COMPLETE external summary")
    if summary.get("primary_cluster") != contract.primary_cluster:
        raise RuntimeError("external primary cluster differs from claim contract")
    conditions = set(summary.get("conditions") or [])
    required = {
        contract.learned_condition,
        contract.rule_condition,
        contract.cost_matched_condition,
        contract.high_resource_condition,
        contract.legacy_me_r0_condition,
        contract.minimal_condition,
        contract.empirical_session_rag_condition,
    }
    if conditions != required:
        raise RuntimeError(
            "external condition matrix differs from the frozen seven-condition contract"
        )
    learned_vs_rule = _assess_comparator(
        summary,
        comparator=contract.rule_condition,
        gate=contract.learned_vs_rule,
        primary_cluster=contract.primary_cluster,
    )
    learned_vs_high = _assess_comparator(
        summary,
        comparator=contract.high_resource_condition,
        gate=contract.learned_vs_high_resource,
        primary_cluster=contract.primary_cluster,
    )
    learned_vs_cost = _assess_comparator(
        summary,
        comparator=contract.cost_matched_condition,
        gate=contract.learned_vs_cost_matched,
        primary_cluster=contract.primary_cluster,
    )
    legacy_quality = _metric_ci(
        summary,
        contract.legacy_me_r0_condition,
        "quality_composite",
        primary_cluster=contract.primary_cluster,
    )
    legacy_utility = _metric_ci(
        summary,
        contract.legacy_me_r0_condition,
        "total_utility",
        primary_cluster=contract.primary_cluster,
    )
    legacy_checks = {
        "quality_not_regressed": legacy_quality["lower"]
        >= -contract.legacy_regression_quality_margin,
        "utility_not_regressed": legacy_utility["lower"]
        >= -contract.legacy_regression_utility_margin,
    }
    legacy = {
        "status": "PASS" if all(legacy_checks.values()) else "FAIL",
        "checks": legacy_checks,
        "quality": legacy_quality,
        "utility": legacy_utility,
        "role": "legacy_regression_guard_not_primary_algorithm_baseline",
    }
    return {
        "status": (
            "SUPPORTED"
            if learned_vs_rule["status"] == "SUPPORTED"
            and learned_vs_high["status"] == "SUPPORTED"
            and legacy["status"] == "PASS"
            else "NOT_SUPPORTED"
        ),
        "protocol": CLAIM_PROTOCOL,
        "claim_a_learned_routing_vs_rule": learned_vs_rule,
        "claim_b_efficiency_vs_high_resource": learned_vs_high,
        "claim_c_cost_matched_secondary": learned_vs_cost,
        "legacy_me_r0_regression_guard": legacy,
        "claim_boundaries": {
            "supported_domain": (
                "frozen synthetic routing matrix and EvoEmo simulated external evaluation"
            ),
            "not_supported": [
                "clinical efficacy",
                "real-user emotional improvement",
                "deployment safety",
                "reinforcement learning",
                "POMDP optimization",
                "universal optimality",
                "total latency reduction unless separately confirmed",
                "fully independent non-ESConv external generalization",
            ],
        },
    }
