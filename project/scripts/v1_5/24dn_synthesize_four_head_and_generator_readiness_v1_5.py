#!/usr/bin/env python3
"""Synthesize four-head routing evidence and generator-use readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metacom_pm.io import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-four-head-generator-readiness-synthesis-v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(*, root: Path = ROOT) -> dict[str, Any]:
    rs_path = root / "outputs/pm_v1_5_same_bank_rs_opportunity_router_fit_v1/fit_report.json"
    ms_path = root / "outputs/pm_v1_5_ms_opportunity_router_fit_v1/fit_report.json"
    mp_me_path = root / "outputs/pm_v1_5_mp_me_opportunity_router_fit_v1/fit_report.json"
    h2_path = root / "outputs/pm_v1_5_h2_retrieval_qualification_v1/summary.json"
    old_ms_path = root / "outputs/pm_v1_5_d3_ms_nonuse_root_cause_v1/report_evidence.json"
    alignment_path = root / "data/pm_v1_5_contracts/generator_resource_use_alignment_v1.json"
    routing_contract_path = root / "data/pm_v1_5_contracts/resource_routing_measurement_layers_v1.json"

    rs = _json(rs_path)
    ms = _json(ms_path)
    mp_me = _json(mp_me_path)
    h2 = _json(h2_path)
    old_ms = _json(old_ms_path)
    alignment = _json(alignment_path)
    routing_contract = _json(routing_contract_path)

    rs_metric = rs["seed_reports"][0]["human_h2_transfer"]
    ms_metric = ms["evaluation"][0]["content_theme_holdout"]
    mp_metric = mp_me["component_reports"]["MP"]["seed_reports"][0][
        "content_theme_holdout"
    ]
    me_metric = mp_me["component_reports"]["ME"]["seed_reports"][0][
        "content_theme_holdout"
    ]
    components = [
        {
            "component": "RS",
            "routing_status": "basic_pass",
            "balanced_accuracy": rs_metric["balanced_accuracy"],
            "positive_recall": rs_metric["positive_recall"],
            "evidence_level": "human_H2_development_informed",
            "independent_groups": sum(
                int(value) for value in rs["counts"]["h2_human_classes"].values()
            ),
            "retrieval_status": "not_qualified_top1_19_of_31",
            "external_status": "not_yet_run_under_new_router",
        },
        {
            "component": "MS",
            "routing_status": "controlled_basic_pass",
            "balanced_accuracy": ms_metric["balanced_accuracy"],
            "positive_recall": ms_metric["positive_recall"],
            "evidence_level": "controlled_real_candidate_states",
            "independent_groups": 32,
            "retrieval_status": "constructed_set_64_of_64_relevant",
            "external_status": "EvoEmo_transport_pending",
        },
        {
            "component": "MP",
            "routing_status": "controlled_basic_pass",
            "balanced_accuracy": mp_metric["balanced_accuracy"],
            "positive_recall": mp_metric["positive_recall"],
            "evidence_level": "controlled_micro_world",
            "independent_groups": mp_me["component_reports"]["MP"]["counts"][
                "independent_users"
            ],
            "retrieval_status": "typed_preference_profile_real_fit_pending",
            "external_status": "EvoEmo_profile_only_pending",
        },
        {
            "component": "ME",
            "routing_status": "controlled_basic_pass",
            "balanced_accuracy": me_metric["balanced_accuracy"],
            "positive_recall": me_metric["positive_recall"],
            "evidence_level": "controlled_micro_world",
            "independent_groups": mp_me["component_reports"]["ME"]["counts"][
                "independent_users"
            ],
            "retrieval_status": "real_candidate_qualification_pending",
            "external_status": "EvoEmo_transport_pending",
        },
    ]
    all_basic = all(row["routing_status"].endswith("pass") for row in components)
    report = {
        "protocol": PROTOCOL,
        "status": (
            "FOUR_HEAD_BASIC_ROUTING_READY_GENERATOR_ALIGNMENT_NEXT"
            if all_basic
            else "FOUR_HEAD_BASIC_ROUTING_INCOMPLETE"
        ),
        "plain_language": (
            "All four component bits now have at least a basic outcome-blind "
            "learnability result, but their evidence levels differ. The next "
            "causal bottleneck is candidate qualification and source-specific "
            "generator use, not another round of single-response winner labels."
        ),
        "components": components,
        "action_space": {
            "legal_actions": 16,
            "training_form": "four independent binary heads plus deterministic hard gates",
            "combination": "The four post-gate bits map mechanically to the existing 16 action ids.",
            "not_used": "direct 16-class classification",
        },
        "comparability_warning": (
            "Balanced accuracy values are displayed together for status, not "
            "pooled or ranked: RS uses human H2 transfer, MS controlled candidate "
            "states, and MP/ME controlled micro-worlds."
        ),
        "candidate_layer": {
            "RS": {
                "human_opportunities": h2["methods"]["transparent"]["opportunity_items"],
                "top1_acceptable": h2["methods"]["transparent"]["top1_correct"],
                "top1_acceptable_rate": h2["methods"]["transparent"]["top1_acceptable_rate"],
                "decision": "repair_or_abstain_at_item_ranker; do not relabel router",
            },
            "memory": {
                "decision": (
                    "Use owner/time/conflict/duplicate hard gates, then qualify "
                    "source-specific Top-k and report external in-support/OOD coverage."
                )
            },
        },
        "generator_layer": {
            "known_old_result": {
                "MS_pairs": old_ms["dataset_grain"]["pair_realizations"],
                "quality_treatment_wins": old_ms["final_quality_verdict_counts"]["treatment"],
                "quality_control_wins": old_ms["final_quality_verdict_counts"]["control"],
                "quality_ties": old_ms["final_quality_verdict_counts"]["tie"],
                "interpretation": (
                    "The old source-agnostic prompt plus mostly redundant MS evidence "
                    "did not yield material response gains; it does not invalidate "
                    "opportunity routing or prove generator capability failure."
                ),
            },
            "new_contract_status": alignment["status"],
            "next_test": {
                "scope": "Upstream-qualified independent states only",
                "minimum_per_component": alignment["minimum_generator_diagnosis"][
                    "minimum_per_component"
                ],
                "labels": alignment["measurement_layers"][2]["labels"],
                "stop_rule": alignment["minimum_generator_diagnosis"]["decision"],
            },
        },
        "data_quality": {
            "same_stack_RS": True,
            "response_outcome_not_training_gold": True,
            "dataset_identity_not_model_input": True,
            "formal_ESConv_test_excluded_from_RS_fit": True,
            "EvoEmo_content_excluded_from_internal_fit": True,
            "known_limitations": [
                "H2 is development-informed and already consumed; not an external confirmation set.",
                "MP/ME current results are controlled learnability, not real-candidate accuracy.",
                "RS candidate ranking remains below the old preregistered qualification gate.",
                "The new source-specific generator prompt has not yet been executed.",
            ],
        },
        "next_execution_order": [
            "Freeze/load four heads and deterministic hard gates; preserve all 16 actions.",
            "Qualify selected-item retrieval separately, with abstention when no safe candidate exists.",
            "Run at least eight upstream-qualified states per component through the source-specific generator prompt.",
            "Measure grounded functional use versus correct ignore, surface-only use, missed use, and misuse.",
            "Only after upstream layers pass, run blinded quality-risk-cost paired evaluation.",
            "Evaluate frozen stack on ESConv and EvoEmo with component/subtype/OOD reporting.",
        ],
        "claim_boundary": routing_contract["claim_boundary"],
        "inputs": {
            path.name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in (
                rs_path,
                ms_path,
                mp_me_path,
                h2_path,
                old_ms_path,
                alignment_path,
                routing_contract_path,
            )
        },
    }
    out_dir = root / "outputs/pm_v1_5_four_head_generator_readiness_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "report_evidence.json", report)
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "components": [
                {
                    "component": row["component"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "evidence_level": row["evidence_level"],
                }
                for row in report["components"]
            ],
        }
    )


if __name__ == "__main__":
    main()
