#!/usr/bin/env python3
"""Synthesize the V1.5 Strategy RAG / RS evidence and freeze the next protocol.

This script is intentionally read-only with respect to prior experiments.  It
recomputes a compact evidence ledger from their reports and writes a technical
report artifact plus a machine-readable next-stage protocol.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pm_v1_5_strategy_rs_evidence_synthesis_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
) -> dict[str, Any]:
    row = {
        "id": source_id,
        "label": f"{label} — {description}",
        "path": path,
    }
    if not path.endswith(".py"):
        row["query"] = {
            "engine": "duckdb",
            "description": (
                "Load the reviewed local JSON/JSONL evidence named in path."
            ),
            "sql": f"SELECT * FROM read_json_auto('{path}')",
            "tables_used": [path],
        }
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    bank_path = (
        ROOT
        / "outputs/pm_v1_5_strategy_rag_v4_candidate_v1"
        / "strategy_cards_v4_candidate.jsonl"
    )
    cards = read_jsonl(bank_path)
    active_cards = [
        card
        for card in cards
        if card["source_support"]["provisional_source_support_pass"]
    ]

    round_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    exposed_cards: set[str] = set()
    exposed_cores: set[str] = set()
    positive_cores: set[str] = set()
    direct_users: set[str] = set()

    for round_id in (1, 2, 4):
        selected_path = (
            ROOT
            / f"outputs/pm_v1_5_strategy_rag_v4_direct_effect_v{round_id}"
            / "selected_states.jsonl"
        )
        selected = read_jsonl(selected_path)
        selected_by_pair = {row["pair_id"]: row for row in selected}
        direct_users.update(str(row["user_id"]) for row in selected)
        exposed_cards.update(str(row["selected_card_id"]) for row in selected)
        exposed_cores.update(
            str(row["selected_core_submove_id"]) for row in selected
        )
        report_name = (
            "retrospective_risk_first_report.json"
            if round_id == 1
            else "independent_confirmation_report.json"
        )
        report = read_json(
            ROOT
            / (
                "outputs/pm_v1_5_strategy_rag_v4_direct_effect_"
                f"v{round_id}_independent_blind"
            )
            / report_name
        )
        outcomes = read_jsonl(
            ROOT
            / (
                "outputs/pm_v1_5_strategy_rag_v4_direct_effect_"
                f"v{round_id}_execution"
            )
            / "generation_outcomes.jsonl"
        )
        arm_total_tokens = {
            arm: sum(
                int(row["usage"]["total_tokens"])
                for row in outcomes
                if row["arm"] == arm
            )
            for arm in ("R0", "RS")
        }
        token_delta_rate = (
            arm_total_tokens["RS"] / arm_total_tokens["R0"] - 1.0
        )
        quality = report["quality_counts"]
        risk_first = report["risk_first_action_counts"]
        prompt_status = {
            1: "legacy prompt; cross-round user overlap",
            2: "long prompt; cross-round user overlap",
            4: "current compact prompt; fresh users",
        }[round_id]
        round_rows.append(
            {
                "round": f"V{round_id}",
                "pairs": int(report["pair_count"]),
                "quality_R0": int(quality["R0"]),
                "quality_RS": int(quality["RS"]),
                "quality_tie": int(quality["tie"]),
                "risk_first_R0": int(risk_first["R0"]),
                "risk_first_RS": int(risk_first["RS"]),
                "R0_total_tokens": arm_total_tokens["R0"],
                "RS_total_tokens": arm_total_tokens["RS"],
                "RS_token_delta_rate": round(token_delta_rate, 6),
                "prompt_and_sampling": prompt_status,
                "poolable_for_training": False,
            }
        )
        exposure = Counter(
            str(row["selected_strategy_family"]) for row in selected
        )
        on_counts = Counter(
            str(decision["strategy_family"])
            for decision in report["pair_decisions"]
            if decision["risk_first_action"] == "RS"
        )
        for decision in report["pair_decisions"]:
            if decision["risk_first_action"] == "RS":
                positive_cores.add(
                    str(
                        selected_by_pair[decision["pair_id"]][
                            "selected_core_submove_id"
                        ]
                    )
                )
        for family in sorted(exposure):
            family_rows.append(
                {
                    "round": f"V{round_id}",
                    "family": family,
                    "exposures": int(exposure[family]),
                    "risk_first_RS": int(on_counts[family]),
                    "risk_first_R0": int(exposure[family] - on_counts[family]),
                    "descriptive_only": True,
                }
            )

    direct_quality_rows = [
        {"round": row["round"], "decision": decision, "count": row[key]}
        for row in round_rows
        for decision, key in (
            ("R0", "quality_R0"),
            ("RS", "quality_RS"),
            ("tie", "quality_tie"),
        )
    ]
    risk_first_rows = [
        {"round": row["round"], "action": action, "count": row[key]}
        for row in round_rows
        for action, key in (
            ("R0", "risk_first_R0"),
            ("RS", "risk_first_RS"),
        )
    ]
    cost_rows = [
        {
            "round": row["round"],
            "RS_token_delta_rate": row["RS_token_delta_rate"],
            "R0_total_tokens": row["R0_total_tokens"],
            "RS_total_tokens": row["RS_total_tokens"],
        }
        for row in round_rows
    ]

    family_aggregate: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in family_rows}):
        rows = [row for row in family_rows if row["family"] == family]
        family_aggregate.append(
            {
                "family": family,
                "exposures": sum(row["exposures"] for row in rows),
                "risk_first_RS": sum(row["risk_first_RS"] for row in rows),
                "risk_first_R0": sum(row["risk_first_R0"] for row in rows),
                "rounds_with_RS": sum(
                    row["risk_first_RS"] > 0 for row in rows
                ),
                "rounds_observed": len(rows),
                "interpretation": "descriptive only; rounds are not poolable",
            }
        )

    bank_by_family: list[dict[str, Any]] = []
    for family in sorted({str(card["strategy_family"]) for card in cards}):
        family_cards = [
            card for card in cards if card["strategy_family"] == family
        ]
        active_family = [
            card
            for card in family_cards
            if card["source_support"]["provisional_source_support_pass"]
        ]
        bank_by_family.append(
            {
                "family": family,
                "candidate_variants": len(family_cards),
                "candidate_core_submoves": len(
                    {str(card["core_submove_id"]) for card in family_cards}
                ),
                "weak_source_pass_variants": len(active_family),
                "weak_source_pass_cores": len(
                    {str(card["core_submove_id"]) for card in active_family}
                ),
                "direct_effect_exposed_cores": len(
                    {
                        str(card["core_submove_id"])
                        for card in family_cards
                        if str(card["core_submove_id"]) in exposed_cores
                    }
                ),
                "risk_first_positive_cores": len(
                    {
                        str(card["core_submove_id"])
                        for card in family_cards
                        if str(card["core_submove_id"]) in positive_cores
                    }
                ),
            }
        )

    old_human = read_json(
        ROOT
        / "outputs/pm_v1_5_minimum_rs_human_sanity_audit_v1_analysis"
        / "human_audit_analysis.json"
    )
    old_judge = read_json(
        ROOT
        / "outputs/pm_v1_5_minimum_rs_judging_v1"
        / "measurement_report.json"
    )
    g0 = read_json(
        ROOT
        / "outputs/pm_v1_5_strategy_g0_freeze_v1"
        / "g0_freeze_report.json"
    )
    g1 = read_json(
        ROOT
        / "outputs/pm_v1_5_strategy_g1_final_bank_v1"
        / "g1_dual_human_freeze_report.json"
    )
    g2 = read_json(
        ROOT
        / "outputs/pm_v1_5_strategy_g2_real_qualification_v1"
        / "real_qualification_report.json"
    )
    bank_build = read_json(
        ROOT
        / "outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1"
        / "build_report.json"
    )

    evidence_ledger = [
        {
            "evidence": "Old five-card LLM-judged clean-pair pilot",
            "n": 49,
            "result": "RS 7 / R0 16 / tie 26; real-pair AB/BA consistency 0.531",
            "valid_use": "cost/mechanism diagnostics only",
            "not_valid_for": "PM hard training labels or general RS benefit",
        },
        {
            "evidence": "Old five-card single blind sanity review",
            "n": int(old_human["data_quality"]["received_pairs"]),
            "result": "forced preference RS 8 / R0 4; no tie option used",
            "valid_use": "plausibility and instrument calibration",
            "not_valid_for": "training labels; annotator identity and IRR absent",
        },
        {
            "evidence": "G0 bottom-up ESConv open coding",
            "n": int(g0["human_item_count"]),
            "result": (
                f"{g0['field_distributions_after_resolution']['meaningful_support_action']['true']} meaningful; "
                f"{g0['field_distributions_after_resolution']['reusable_as_general_technique']['true']} reusable; "
                f"{g0['field_distributions_after_resolution']['clear_risk_or_boundary_problem']['true']} risk/boundary"
            ),
            "valid_use": "define safe topic-agnostic move space",
            "not_valid_for": "move prevalence or response-effect claims",
        },
        {
            "evidence": "G1 dual-review six-card seed bank",
            "n": int(g1["review_row_count_each"]),
            "result": "corrected definitions retained; source promotion failed",
            "valid_use": "safe definitions and exclusions",
            "not_valid_for": "gold source grounding or formal RS benefit",
        },
        {
            "evidence": "G2 six-card real retrieval qualification",
            "n": int(g2["primary_strict"]["states"]),
            "result": "lexical opportunity gate qualified for clean treatment only",
            "valid_use": "Top-1 treatment construction",
            "not_valid_for": "response-quality benefit or 100-card retrieval",
        },
        {
            "evidence": "V1/V2/V4 100-card direct-effect blind rounds",
            "n": sum(row["pairs"] for row in round_rows),
            "result": (
                f"descriptive risk-first RS "
                f"{sum(row['risk_first_RS'] for row in round_rows)} / "
                f"R0 {sum(row['risk_first_R0'] for row in round_rows)}"
            ),
            "valid_use": "prove non-degenerate RS opportunities exist; find mechanisms",
            "not_valid_for": "pooled training; prompts/samples differ and V1/V2 overlap",
        },
        {
            "evidence": "V3 technical safety audit",
            "n": 6,
            "result": "one semantic inversion around 'give up' stopped blind scoring",
            "valid_use": "hard-off high-stakes gate and generator failure mode",
            "not_valid_for": "quality-effect counts",
        },
    ]

    next_protocol = {
        "protocol": "pm-v1.5-rs-positive-expansion-and-gated-retrieval-v1",
        "research_target": (
            "Learn whether one naturally retrieved, safe strategy card should "
            "be injected for the current state, under a frozen generator and bank."
        ),
        "runtime_order": [
            "hard-off gate for phatic/stop/high-stakes/non-substantive states",
            "profile selection: minimal for explicit low burden, otherwise dialogic",
            "family eligibility from visible cues",
            (
                "natural within-family lexical-cosine Top-1 retrieval over only "
                "the 62 provisional-pass variants; no target submove id in query"
            ),
            "card applicability/exclusion check; abstain if no valid candidate",
            (
                "PM_RS predicts the risk-first material-benefit probability for "
                "the state-card pair"
            ),
            "inject Top-1 only when threshold, risk gate, and cost cap all pass",
        ],
        "do_not_use_as_runtime_retrieval": (
            "The current transparent_rule direct-effect route: it hard-codes "
            "target_submove_id and assigns score 1.0, so it is an oracle-style "
            "treatment constructor rather than a natural ranker."
        ),
        "bank_freeze": {
            "audit_catalog_variants": len(cards),
            "active_variants": len(active_cards),
            "active_core_submoves": len(
                {str(card["core_submove_id"]) for card in active_cards}
            ),
            "quarantined_weak_source_fail_variants": len(cards)
            - len(active_cards),
            "injection_top_k": 1,
            "raw_source_examples_exposed": False,
            "external_test_bank_changes_allowed": False,
        },
        "positive_expansion_wave": {
            "new_independent_train_dialogue_groups": 32,
            "selection_is_outcome_blind": True,
            "exclude_prior_direct_effect_users": len(direct_users),
            "strata": [
                {
                    "stratum": "focused clarification opportunity",
                    "groups": 12,
                    "observable_basis": (
                        "explicit uncertainty, competing priorities, or a "
                        "missing goal/feeling distinction"
                    ),
                },
                {
                    "stratum": "grounded reflection opportunity",
                    "groups": 10,
                    "observable_basis": (
                        "explicit emotion or explicit value-obstacle tension; "
                        "avoid multi-entity third-party inference"
                    ),
                },
                {
                    "stratum": "welcomed single microstep",
                    "groups": 8,
                    "observable_basis": (
                        "explicit request for advice plus a visible goal and "
                        "a low-domain-risk action space"
                    ),
                },
                {
                    "stratum": "redundancy negative control",
                    "groups": 2,
                    "observable_basis": (
                        "a simple restatement opportunity where the base model "
                        "is likely already sufficient"
                    ),
                },
            ],
            "profile_quota": {
                "minimal": 6,
                "dialogic": 26,
                "reason": (
                    "The initial target of ten was reduced before generation "
                    "after a visible-only capacity audit. Six safe, distinct "
                    "minimal-profile states were naturally available inside "
                    "the frozen positive-opportunity strata."
                ),
            },
            "stage_1": (
                "one frozen R0/RS pair per state; blind review of material "
                "quality, five material-risk categories, and treatment uptake"
            ),
            "stage_2": (
                "two additional generation seeds only for provisional RS wins "
                "and invalid/mixed cases; negatives are not relabeled by seed shopping"
            ),
            "expected_human_load": (
                "32 initial pair reviews plus approximately 12-20 replicate "
                "reviews, rather than another full-bank audit"
            ),
        },
        "label_rule": {
            "positive": (
                "RS is the risk-first action in at least 2 of 3 valid seeds: "
                "RS is free of material risk and either removes an R0 material "
                "risk or provides a material quality gain; incremental cost is "
                "within the frozen cap."
            ),
            "negative": (
                "no material RS benefit, R0 material win, or reproducible "
                "RS-induced material risk; a tie is OFF because it adds cost."
            ),
            "unknown": (
                "retrieval not applicable, treatment not executed, generation "
                "invalid, or seed results remain materially contradictory."
            ),
            "material_risk_categories": [
                "unsupported inference",
                "request or boundary mismatch",
                "excessive burden or directiveness",
                "false reassurance or minimization",
                "domain or high-stakes overreach",
            ],
            "cost_rule": (
                "risk is a veto, material benefit is required, and cost breaks "
                "ties; do not invent an opaque weighted composite score."
            ),
        },
        "pm_model": {
            "model": "L2-regularized logistic regression",
            "loss": "class-weighted binary cross-entropy / log-loss",
            "unit": "independent dialogue group and frozen state-card pair",
            "features": [
                "visible hard/soft opportunity flags",
                "eligible family count",
                "selected family and core submove",
                "minimal versus dialogic profile",
                "lexical cosine Top-1 score and Top-1 minus Top-2 margin",
                "weak source-support count and pass flag",
                "dialogue length and incremental prompt-token estimate",
                "multi-entity/third-party complexity flag",
                "prior-attempt-failed or value-obstacle cue",
            ],
            "excluded_features": [
                "external-test outcomes",
                "judge preferences",
                "target_submove_id embedded in the retrieval query",
                "raw ESConv supporter responses",
                "blind answer keys",
            ],
            "validation": (
                "grouped OOF by dialogue; freeze C by a small predeclared grid "
                "and one-standard-error preference for stronger regularization"
            ),
            "threshold_gate": (
                "OOF must produce at least 8 ON and 8 OFF decisions, predicted "
                "ON coverage between 20% and 60%, no higher material-risk rate "
                "than always-off, and better risk-first utility than always-off."
            ),
            "qualification_metrics": [
                "ON precision and recall",
                "balanced accuracy",
                "log-loss and Brier score",
                "material-risk rate among predicted ON",
                "material-win rate among predicted ON",
                "incremental tokens/cost versus always-off",
                "coverage-risk curve with abstention",
            ],
            "fallback": (
                "If the head degenerates to always-off or misses the gates, "
                "report RS as an identifiable but not yet learnable component; "
                "do not lower the standard or add BAAI to force a positive result."
            ),
        },
        "retrieval_bakeoff": {
            "primary": "family-gated lexical cosine Top-1",
            "baseline": "transparent-rule oracle route (diagnostic only)",
            "challenger": (
                "BAAI/bge-small-en-v1.5 only as an offline challenger on the "
                "same frozen applicability labels"
            ),
            "promotion_rule": (
                "BAAI is promoted only if it improves blinded Top-1 card-fit "
                "without more abstention or exclusion violations; otherwise "
                "keep lexical for V1.5."
            ),
        },
        "evaluation_firewall": {
            "training": (
                "ESConv official train visible-only pre-response universe; "
                "hidden next supporter response and native strategy labels excluded"
            ),
            "threshold_and_debug": (
                "development/calibration groups only; prior V1-V4 are pilot evidence"
            ),
            "final": (
                "one frozen shared bank and PM on untouched ESConv formal test "
                "and EvoEmo; no domain-specific bank swap"
            ),
            "claim_boundary": (
                "topic-agnostic emotional-support technique RAG only; no claim "
                "for medical, legal, crisis, autism-specific, or factual guidance"
            ),
        },
    }

    summary = {
        "protocol": "pm-v1.5-strategy-rs-evidence-synthesis-v1",
        "generated_at": generated_at,
        "conclusion": (
            "RS has real but sparse conditional value.  The current evidence "
            "qualifies the research question and a non-degenerate action space, "
            "not a general RS advantage or a train-ready label pool."
        ),
        "direct_effect_rounds": round_rows,
        "descriptive_direct_effect_totals_not_poolable": {
            "pairs": sum(row["pairs"] for row in round_rows),
            "quality_R0": sum(row["quality_R0"] for row in round_rows),
            "quality_RS": sum(row["quality_RS"] for row in round_rows),
            "quality_tie": sum(row["quality_tie"] for row in round_rows),
            "risk_first_R0": sum(row["risk_first_R0"] for row in round_rows),
            "risk_first_RS": sum(row["risk_first_RS"] for row in round_rows),
        },
        "bank": {
            "candidate_variants": len(cards),
            "candidate_core_submoves": len(
                {str(card["core_submove_id"]) for card in cards}
            ),
            "weak_source_pass_variants": len(active_cards),
            "weak_source_pass_core_submoves": len(
                {str(card["core_submove_id"]) for card in active_cards}
            ),
            "direct_effect_exposures": sum(row["pairs"] for row in round_rows)
            + 6,
            "direct_effect_distinct_cards_all_v1_v2_v3_v4": len(
                exposed_cards
                | {
                    str(row["selected_card_id"])
                    for row in read_jsonl(
                        ROOT
                        / "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v3"
                        / "selected_states.jsonl"
                    )
                }
            ),
            "direct_effect_distinct_core_submoves_all_v1_v2_v3_v4": len(
                exposed_cores
                | {
                    str(row["selected_core_submove_id"])
                    for row in read_jsonl(
                        ROOT
                        / "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v3"
                        / "selected_states.jsonl"
                    )
                }
            ),
            "minimal_profile_direct_effect_exposures": 0,
            "risk_first_positive_core_submoves_v1_v2_v4": len(positive_cores),
            "formal_rs_qualified": False,
        },
        "retrieval_truth": {
            "direct_effect_ranker": "transparent_rule",
            "score": 1.0,
            "target_submove_hard_routed": True,
            "natural_100_card_top1_tested": False,
            "bge_promoted": False,
            "existing_qualified_scope": (
                "six-card G2 opportunity construction only; response benefit unproven"
            ),
        },
        "family_descriptive": family_aggregate,
        "evidence_ledger": evidence_ledger,
        "next_protocol_file": "rs_positive_expansion_protocol.json",
            "source_note": (
            "All aggregate V1/V2/V4 totals are descriptive.  They are retained "
            "to expose the evidence landscape, not to create pooled labels."
        ),
    }
    write_json(OUT / "evidence_summary.json", summary)
    write_json(OUT / "rs_positive_expansion_protocol.json", next_protocol)

    sources = [
        source(
            "synthesis",
            "Recomputed RS evidence synthesis",
            "outputs/pm_v1_5_strategy_rs_evidence_synthesis_v1/evidence_summary.json",
            (
                "Mechanically recomputed round, family, bank-coverage, and "
                "evidence-ledger rows used by this report."
            ),
        ),
        source(
            "direct_effect",
            "V1/V2/V4 independent blind reports and selected states",
            "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v4_independent_blind/independent_confirmation_report.json",
            (
                "Recomputed round-level quality, material-risk, risk-first, "
                "family, card, and core-submove counts from V1, V2, and V4."
            ),
        ),
        source(
            "bank_v4",
            "Strategy RAG V4 candidate catalog",
            "outputs/pm_v1_5_strategy_rag_v4_candidate_v1/strategy_cards_v4_candidate.jsonl",
            (
                "One hundred technique-only candidate variants with weak "
                "source-support status and no raw source responses in prompts."
            ),
        ),
        source(
            "bank_v3",
            "Strategy Bank V3 source-mapping report",
            "outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1/build_report.json",
            (
                "Outcome-blind ESConv-train source mapping used to construct "
                "the fifty core submoves."
            ),
        ),
        source(
            "g0_g1",
            "G0/G1 bottom-up coding and dual-review reports",
            "outputs/pm_v1_5_strategy_g1_final_bank_v1/g1_dual_human_freeze_report.json",
            (
                "Open-coding move definitions, reviewer agreement, corrections, "
                "and source-qualification boundary."
            ),
        ),
        source(
            "g2",
            "G2 real retrieval qualification",
            "outputs/pm_v1_5_strategy_g2_real_qualification_v1/real_qualification_report.json",
            (
                "Six-card opportunity qualification comparing lexical and BGE "
                "retrieval under strict lineage disjointness."
            ),
        ),
        source(
            "old_rs",
            "Old minimum RS judge and human sanity reports",
            "outputs/pm_v1_5_minimum_rs_judging_v1/measurement_report.json",
            (
                "Legacy five-card quality, risk, cost, judge-consistency, and "
                "small blind-review evidence."
            ),
        ),
        source(
            "direct_code",
            "Direct-effect treatment-construction code",
            "scripts/v1_5/24g_prepare_strategy_rag_v4_direct_effect_v1_5.py",
            (
                "Implementation of visible-cue routing, target-submove routing, "
                "candidate filtering, and R0/RS matched prompt construction."
            ),
        ),
    ]

    title = "PM V1.5：RS 证据总账、开关规律与正例扩展方案"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "对当前所有 Strategy RAG / RS 实验、人评、检索和卡库证据的 "
                "研究级合并审计，以及可执行的下一轮 PM_RS 训练协议。"
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "direct_pairs",
                    "description": (
                        "V1、V2、V4 三轮盲评总数；仅作描述，不能直接合并训练。"
                    ),
                    "dataset": "headline",
                    "sourceId": "synthesis",
                    "metrics": [
                        {
                            "label": "已盲评 direct-effect 对",
                            "field": "direct_pairs",
                            "format": "number",
                        },
                        {
                            "label": "其中 risk-first RS",
                            "field": "direct_rs",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "active_bank",
                    "description": (
                        "通过弱来源支持门的候选变体；对应 31 个核心子策略。"
                    ),
                    "dataset": "headline",
                    "sourceId": "synthesis",
                    "metrics": [
                        {
                            "label": "当前可进入下一轮的卡片变体",
                            "field": "active_variants",
                            "format": "number",
                        },
                        {
                            "label": "对应核心子策略",
                            "field": "active_cores",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "coverage",
                    "description": (
                        "V1-V4 direct-effect 实际覆盖的核心子策略；minimal 档为零。"
                    ),
                    "dataset": "headline",
                    "sourceId": "synthesis",
                    "metrics": [
                        {
                            "label": "已做 direct-effect 的核心子策略",
                            "field": "exposed_cores",
                            "format": "number",
                        },
                        {
                            "label": "minimal 档曝光",
                            "field": "minimal_exposures",
                            "format": "number",
                        },
                    ],
                },
            ],
            "charts": [
                {
                    "id": "quality_by_round",
                    "title": "各轮盲评质量偏好",
                    "subtitle": (
                        "三轮都没有显示 RS 的总体优势；V4 的 tie 比例最高。"
                    ),
                    "type": "bar",
                    "dataset": "direct_quality",
                    "sourceId": "synthesis",
                    "encodings": {
                        "x": {
                            "field": "round",
                            "type": "ordinal",
                            "label": "实验轮次",
                        },
                        "y": {
                            "field": "count",
                            "type": "quantitative",
                            "label": "盲评对数",
                        },
                        "color": {
                            "field": "decision",
                            "type": "nominal",
                            "label": "偏好",
                        },
                    },
                    "layout": "full",
                },
                {
                    "id": "risk_first_by_round",
                    "title": "各轮 risk-first 最终动作",
                    "subtitle": (
                        "tie 默认关闭、RS 物质风险否决后，仍有少量明确 ON。"
                    ),
                    "type": "bar",
                    "dataset": "risk_first",
                    "sourceId": "synthesis",
                    "encodings": {
                        "x": {
                            "field": "round",
                            "type": "ordinal",
                            "label": "实验轮次",
                        },
                        "y": {
                            "field": "count",
                            "type": "quantitative",
                            "label": "状态数",
                        },
                        "color": {
                            "field": "action",
                            "type": "nominal",
                            "label": "动作",
                        },
                    },
                    "layout": "full",
                },
                {
                    "id": "cost_by_round",
                    "title": "RS 的总 token 增量",
                    "subtitle": (
                        "RS 每轮都增加 token；无实质收益时关闭是必要的成本规则。"
                    ),
                    "type": "bar",
                    "dataset": "cost_by_round",
                    "sourceId": "synthesis",
                    "encodings": {
                        "x": {
                            "field": "round",
                            "type": "ordinal",
                            "label": "实验轮次",
                        },
                        "y": {
                            "field": "RS_token_delta_rate",
                            "type": "quantitative",
                            "label": "RS 相对 R0 的总 token 增量",
                            "format": "percent",
                        },
                    },
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "bank_coverage",
                    "title": "100 卡候选库的真实覆盖层级",
                    "dataset": "bank_coverage",
                    "sourceId": "synthesis",
                    "defaultSort": {
                        "field": "family",
                        "direction": "asc",
                    },
                    "columns": [
                        {"field": "family", "label": "策略族", "type": "text"},
                        {
                            "field": "candidate_variants",
                            "label": "候选变体",
                            "format": "number",
                        },
                        {
                            "field": "weak_source_pass_variants",
                            "label": "弱来源通过",
                            "format": "number",
                        },
                        {
                            "field": "weak_source_pass_cores",
                            "label": "通过核心子策略",
                            "format": "number",
                        },
                        {
                            "field": "direct_effect_exposed_cores",
                            "label": "效应已曝光核心",
                            "format": "number",
                        },
                        {
                            "field": "risk_first_positive_cores",
                            "label": "曾产生 ON 的核心",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "family_pattern",
                    "title": "跨轮策略族描述性模式",
                    "dataset": "family_pattern",
                    "sourceId": "synthesis",
                    "defaultSort": {
                        "field": "risk_first_RS",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "family", "label": "策略族", "type": "text"},
                        {
                            "field": "exposures",
                            "label": "总曝光",
                            "format": "number",
                        },
                        {
                            "field": "risk_first_RS",
                            "label": "描述性 ON",
                            "format": "number",
                        },
                        {
                            "field": "risk_first_R0",
                            "label": "描述性 OFF",
                            "format": "number",
                        },
                        {
                            "field": "rounds_with_RS",
                            "label": "出现 ON 的轮数",
                            "format": "number",
                        },
                        {
                            "field": "interpretation",
                            "label": "解释边界",
                            "type": "text",
                        },
                    ],
                },
                {
                    "id": "evidence_ledger",
                    "title": "哪些证据能用、不能怎么用",
                    "dataset": "evidence_ledger",
                    "sourceId": "synthesis",
                    "defaultSort": {"field": "n", "direction": "desc"},
                    "columns": [
                        {"field": "evidence", "label": "证据", "type": "text"},
                        {"field": "n", "label": "N", "format": "number"},
                        {"field": "result", "label": "结果", "type": "text"},
                        {
                            "field": "valid_use",
                            "label": "可以支持",
                            "type": "text",
                        },
                        {
                            "field": "not_valid_for",
                            "label": "不能支持",
                            "type": "text",
                        },
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": (
                        "## 结论：RS 值得学，但只能学“条件性开启”\n\n"
                        "现有证据已经排除了两个极端：不是“RS 总有用”，也不是"
                        "“RS 永远该关”。在相同 generator 下，RS 偶尔能抑制基础"
                        "回复的虚假安慰、过度建议或错过关键张力；更多时候它只是"
                        "重复基础模型已经会做的事，或者诱发未经支持的推断。因而"
                        "正确研究对象是**状态—候选卡对的边际效应开关**。\n\n"
                        "但当前标签还不能直接训练：V1/V2/V4 的 prompt 与抽样不同，"
                        "V1/V2 还复用了部分用户；盲评身份未形成可引用的人类 gold；"
                        "最重要的是 direct-effect 使用了硬编码 submove 的透明路由，"
                        "尚未检验 100 卡自然 Top‑1 召回。"
                    ),
                },
                {
                    "id": "metrics",
                    "type": "metric-strip",
                    "cardIds": ["direct_pairs", "active_bank", "coverage"],
                },
                {
                    "id": "quality_head",
                    "type": "markdown",
                    "body": (
                        "## RS 没有总体优势，正例来自少量可解释机制\n\n"
                        "三轮共 49 对的合计只能作为地图，不能作为一个统计样本。"
                        "各轮的 R0 胜出都多于 RS；这正说明把所有“有检索机会”的"
                        "状态都开 RS 会退化。可复现的候选机制主要是：一次聚焦澄清、"
                        "有据的价值—障碍反映，以及用户明确欢迎时的单一小步骤。"
                    ),
                },
                {
                    "id": "quality_chart",
                    "type": "chart",
                    "chartId": "quality_by_round",
                    "layout": "full",
                },
                {
                    "id": "risk_head",
                    "type": "markdown",
                    "body": (
                        "## risk-first 规则让“tie 就关”成为可审计标签\n\n"
                        "这里的 RS risk 不是抽象的“记忆风险”，而是资源注入诱发的"
                        "可见回复风险：未经支持的推断、请求/边界错配、负担或指令"
                        "过强、虚假安慰/最小化，以及领域或高风险越界。RS 回复一旦"
                        "出现 material risk 就否决；两边都安全但无实质质量差异时，"
                        "因 RS 增加成本而关闭。只有 RS 消除 R0 风险，或带来实质"
                        "质量提升，才标 ON。"
                    ),
                },
                {
                    "id": "risk_chart",
                    "type": "chart",
                    "chartId": "risk_first_by_round",
                    "layout": "full",
                },
                {
                    "id": "cost_head",
                    "type": "markdown",
                    "body": (
                        "## RS 成本不是理论问题：三轮都增加了 token\n\n"
                        "相对同轮 R0，RS 总 token 在 V1、V2、V4 分别增加约 "
                        "21.8%、52.7% 和 35.5%。prompt 版本会改变幅度，因此不把"
                        "三个百分比平均成一个常数；但方向完全一致。由此可以预先"
                        "冻结一个简单原则：没有 material benefit 就关闭，不能用"
                        "轻微偏好为持续成本买单。"
                    ),
                },
                {
                    "id": "cost_chart",
                    "type": "chart",
                    "chartId": "cost_by_round",
                    "layout": "full",
                },
                {
                    "id": "bank_head",
                    "type": "markdown",
                    "body": (
                        "## 100 卡是开发目录，不是 100 张已合格知识卡\n\n"
                        "候选库由 50 个核心子策略各生成 minimal/dialogic 两个执行"
                        "档位。弱来源门仅让 62 个变体、31 个核心子策略进入下一轮；"
                        "这个门来自 ESConv train 的 outcome-blind 弱映射，不是人工"
                        "gold。direct-effect 到目前只覆盖 16 个核心子策略，且全部为"
                        "dialogic。合理做法是保留 100 卡作审计目录，隔离 38 个弱来源"
                        "失败变体，只让 62 个候选进入下一轮，并让实际效应而非卡片"
                        "数量决定最终资格。"
                    ),
                },
                {
                    "id": "bank_table",
                    "type": "table",
                    "tableId": "bank_coverage",
                    "layout": "full",
                },
                {
                    "id": "retrieval_head",
                    "type": "markdown",
                    "body": (
                        "## 当前实验验证了给定卡的效应，尚未验证自然召回\n\n"
                        "直接效应脚本先根据词面 cue 决定 target family 与 "
                        "target_submove_id，再只保留完全匹配的卡并赋分 1.0。这种"
                        "做法适合隔离 treatment，却不是自然 RAG。下一轮应去掉 query"
                        "中的 submove id：先由可见规则做安全 family gate，再在 62 个"
                        "活跃变体中按 profile 选择候选集，用词法余弦做 family 内"
                        " Top‑1，并允许低分或违反 when-not-to-use 时 abstain。BAAI "
                        "只在同一批 card-fit 标签上做离线挑战者；现有证据不足以替换"
                        "词法基线。Top‑2/3 可用于审计，不应一起注入 generator。"
                    ),
                },
                {
                    "id": "family_head",
                    "type": "markdown",
                    "body": (
                        "## Question/Reflection 是正例扩展优先层，Restatement 是负对照\n\n"
                        "Question 三轮都出现 ON；Reflection 在后两轮出现 ON；"
                        "Suggestions 偶发；Affirmation 的早期优势没有延续；"
                        "Restatement 三轮均为 OFF。因为轮次不可合并，这些只能用于"
                        "预注册下一轮的 outcome-blind 分层抽样，不能直接当 family "
                        "标签或删卡依据。"
                    ),
                },
                {
                    "id": "family_table",
                    "type": "table",
                    "tableId": "family_pattern",
                    "layout": "full",
                },
                {
                    "id": "evidence_head",
                    "type": "markdown",
                    "body": (
                        "## 旧证据保留用途，但不得混成训练 gold\n\n"
                        "旧五卡人评证明过 RS 有可能有用；自动 judge 揭示过成本和"
                        "不稳定性；G0/G1 定义了安全动作；G2 仅资格化了六卡机会"
                        "构造；V3 找到安全门漏洞。各自都重要，但测量对象不同。"
                    ),
                },
                {
                    "id": "evidence_table",
                    "type": "table",
                    "tableId": "evidence_ledger",
                    "layout": "full",
                },
                {
                    "id": "rules",
                    "type": "markdown",
                    "body": (
                        "## 可部署的 RS 开关规则\n\n"
                        "1. 寒暄、告别、明确停止、非实质内容、高风险线索：硬关。\n"
                        "2. 没有通过来源门且符合当前边界的 Top‑1 卡：关。\n"
                        "3. 卡片没有真正执行、生成失败或证据矛盾：训练记 unknown，"
                        "推理关。\n"
                        "4. RS 有任何 material risk：关。\n"
                        "5. 两边安全但 RS 没有实质质量收益：关。\n"
                        "6. 增量成本超冻结预算：关。\n"
                        "7. 只有其余条件全通过才开。\n\n"
                        "这不是把 quality、risk、cost 随意加权成一个不透明分数，"
                        "而是风险否决、收益必要、成本打破平局的词典序决策。"
                    ),
                },
                {
                    "id": "expansion",
                    "type": "markdown",
                    "body": (
                        "## 下一轮只做 32 个新独立组，定向扩正例但不看结果选样\n\n"
                        "从 ESConv official train 的 visible-only 母池中选 32 个"
                        "与旧 direct-effect "
                        "用户零重叠的自然状态：12 个聚焦澄清、10 个有据反映、"
                        "8 个明确欢迎单一小步骤、2 个冗余负对照；其中 6 个走"
                        " minimal 档。原目标 10 个，但 visible-only 容量审计在生成"
                        "前发现安全自然候选不足，因此透明下调。分层依据只用可见 cue，"
                        "不用 judge、生成结果或"
                        "外部测试标签，因此属于合法的 case-control 训练抽样，不"
                        "估计自然流行率。\n\n"
                        "先每状态生成一对；只有初步 RS 胜出和 invalid/mixed 状态再"
                        "补两个 seed。最终 ON 必须在至少 2/3 个有效 seed 中成为"
                        " risk-first 动作。这样既扩大正例，又防止偶然生成或挑 seed "
                        "制造正例。预计人评 44–52 对，不再审核整套 Bank。"
                    ),
                },
                {
                    "id": "model",
                    "type": "markdown",
                    "body": (
                        "## PM_RS 用小模型学习 state-card 边际效应\n\n"
                        "V1.5 使用 L2 正则 logistic regression，loss 为 class-weighted "
                        "binary cross-entropy。输入只含可见 cue、候选 family/submove、"
                        "profile、Top‑1 词法余弦与 margin、来源支持、第三方复杂度和"
                        "增量 token 估计；不含外测结果、judge 偏好、答案 key、原始"
                        " supporter 回复或硬编码 target submove。\n\n"
                        "按 dialogue 做 grouped OOF。及格门预先冻结为：OOF 至少产生"
                        " 8 个 ON 和 8 个 OFF，ON 覆盖 20%–60%，预测 ON 的 material "
                        "risk 不高于 always-off，并在 risk-first utility 上优于"
                        " always-off。若仍退化成全关，就诚实结论为“RS 条件存在但"
                        "当前数据不可学习”，不靠换 BAAI、复杂 loss 或降低门槛救结果。"
                    ),
                },
                {
                    "id": "firewall",
                    "type": "markdown",
                    "body": (
                        "## 数据防火墙与论文主张保持不变\n\n"
                        "训练只用 ESConv official train 的 visible-only pre-response "
                        "投影；开发轮次只作阈值与机制研究；"
                        "正式 ESConv test 与 EvoEmo 在 Bank、retriever、generator、"
                        "PM 阈值全部冻结后各运行一次，而且两边使用同一套 RAG。"
                        "不按外部数据集更换知识库，也不从外测失败反向补卡。论文"
                        "主张限定为 topic-agnostic 情绪支持技术资源的条件性调度，"
                        "不覆盖医疗、法律、危机、临床或特定高敏感人群知识。"
                    ),
                },
                {
                    "id": "next",
                    "type": "markdown",
                    "body": (
                        "## 立即执行顺序\n\n"
                        "1. 冻结 62 个活跃变体、自然词法 Top‑1 与上述标签规则。\n"
                        "2. 构造 32 组 outcome-blind 检索包，先审核 card-fit，"
                        "不合格候选直接 abstain。\n"
                        "3. 运行 32 对 R0/RS，盲评后仅对候选正例/混合项补 seed。\n"
                        "4. 达到至少 8 个稳定 ON 后训练 logistic PM_RS；否则停止"
                        "扩张并披露局限。\n"
                        "5. grouped OOF 过门后再冻结并进入正式双外测。\n\n"
                        "当前最需要解决的不是再加卡或换 embedding，而是把"
                        "“自然召回正确”与“正确卡有边际收益”这两个资格门补齐。"
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## 尚待结果回答的问题\n\n"
                        "- minimal 档能否在低负担状态保持收益而不增加机械感？\n"
                        "- Question/Reflection 的候选规律在新独立组中是否复现？\n"
                        "- 自然词法 Top‑1 相比透明 oracle 路由损失多少 card-fit？\n"
                        "- 固定 Bank 下学到的开关能否在 EvoEmo 保持风险和成本优势？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "direct_pairs": sum(row["pairs"] for row in round_rows),
                        "direct_rs": sum(
                            row["risk_first_RS"] for row in round_rows
                        ),
                        "active_variants": len(active_cards),
                        "active_cores": len(
                            {
                                str(card["core_submove_id"])
                                for card in active_cards
                            }
                        ),
                        "exposed_cores": summary["bank"][
                            "direct_effect_distinct_core_submoves_all_v1_v2_v3_v4"
                        ],
                        "minimal_exposures": 0,
                    }
                ],
                "direct_quality": direct_quality_rows,
                "risk_first": risk_first_rows,
                "cost_by_round": cost_rows,
                "bank_coverage": bank_by_family,
                "family_pattern": family_aggregate,
                "evidence_ledger": evidence_ledger,
            },
        },
        "sources": sources,
    }
    write_json(OUT / "artifact.json", artifact)

    markdown = f"""# {title}

## 结论

RS 不是总体有益资源，也不是应该永久关闭。它在少量可解释的状态—候选卡组合上有实质价值；正确目标是学习条件性开启。当前证据通过了“研究问题非退化”的最低门，但尚未形成可直接合并的训练标签，也尚未验证 100 卡自然 Top-1 召回。

## 当前证据的关键数字

- V1/V2/V4 共盲评 {sum(row["pairs"] for row in round_rows)} 对：描述性 risk-first RS {sum(row["risk_first_RS"] for row in round_rows)}、R0 {sum(row["risk_first_R0"] for row in round_rows)}；轮次不可合并训练。
- 候选库 100 个变体 = 50 个核心子策略 × 2 profiles；62 个变体、31 个核心子策略通过弱来源门。
- V1-V4 direct-effect 只覆盖 {summary["bank"]["direct_effect_distinct_core_submoves_all_v1_v2_v3_v4"]} 个核心子策略，minimal profile 曝光为 0。
- direct-effect 使用 transparent_rule 硬路由 target_submove_id，得分固定 1.0；它不是 100 卡自然召回测试。

## 下一轮冻结方案

1. 活跃库只保留 62 个弱来源通过变体；Top-1 注入，不把 Top-2/3 一起塞给 generator。
2. 可见规则做 hard-off 与 family gate；family 内使用不含 target_submove_id 的词法余弦自然召回，低分/违规时 abstain。
3. 从 ESConv official train 的 visible-only 母池选 32 个 outcome-blind、旧用户零重叠状态：12 clarification、10 grounded reflection、8 welcomed microstep、2 redundancy controls；6 个走 minimal。原目标 10 个，因生成前容量审计不足而透明下调。
4. 初轮一状态一对；仅 provisional RS wins 与 invalid/mixed 补两个 seed。稳定 ON 要求至少 2/3 seed 的 risk-first 结果为 RS。
5. 用 L2 logistic regression + class-weighted log-loss；grouped OOF 至少 8 ON/8 OFF、ON coverage 20%–60%、风险不高于 always-off、risk-first utility 优于 always-off。
6. 训练/开发完成后冻结同一 Bank、retriever、generator、PM，再一次性跑 ESConv formal test 与 EvoEmo。

完整证据、定义、表格和可执行协议见同目录 `report.html`、`evidence_summary.json` 与 `rs_positive_expansion_protocol.json`。
"""
    (OUT / "RS_EVIDENCE_SYNTHESIS_AND_NEXT_PLAN_ZH.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "out_dir": str(OUT),
                "direct_pairs_descriptive": sum(
                    row["pairs"] for row in round_rows
                ),
                "active_variants": len(active_cards),
                "active_cores": len(
                    {str(card["core_submove_id"]) for card in active_cards}
                ),
                "next_groups": 32,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
