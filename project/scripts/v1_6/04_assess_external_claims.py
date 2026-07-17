#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.io import read_json, write_json
from metacom_pm.pm_v1_6_claims import (
    ComparatorGate,
    HierarchicalClaimContract,
    assess_hierarchical_claims,
)

ROOT = Path(__file__).resolve().parents[2]


def _gate(raw: dict) -> ComparatorGate:
    return ComparatorGate.model_validate(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "pm_v1_6.yaml"
    )
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("version") != "pm-v1.6":
        raise RuntimeError("external claim assessment requires PM-v1.6")
    external = config["external_evaluation"]
    claims = config["external_claims"]
    conditions = list(external["condition_matrix"])
    if len(conditions) != 7 or len(conditions) != len(set(conditions)):
        raise RuntimeError("PM-v1.6 external condition matrix must contain seven unique conditions")

    contract = HierarchicalClaimContract(
        learned_condition=conditions[0],
        rule_condition=conditions[1],
        cost_matched_condition=conditions[2],
        high_resource_condition=conditions[3],
        legacy_me_r0_condition=conditions[4],
        minimal_condition=conditions[5],
        empirical_session_rag_condition=conditions[6],
        primary_cluster=str(external["primary_cluster"]),
        learned_vs_rule=_gate(claims["claim_a_learned_vs_rule"]),
        learned_vs_high_resource=_gate(
            claims["claim_b_efficiency_vs_high_resource"]
        ),
        learned_vs_cost_matched=_gate(
            claims["claim_c_cost_matched_secondary"]
        ),
        legacy_regression_quality_margin=float(
            claims["legacy_me_r0_regression_guard"][
                "quality_noninferiority_margin"
            ]
        ),
        legacy_regression_utility_margin=float(
            claims["legacy_me_r0_regression_guard"][
                "utility_noninferiority_margin"
            ]
        ),
    )
    result = assess_hierarchical_claims(
        read_json(args.external_summary),
        contract,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)


if __name__ == "__main__":
    main()
