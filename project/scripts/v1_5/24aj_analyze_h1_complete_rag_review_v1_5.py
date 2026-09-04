#!/usr/bin/env python3
"""Bind and analyze the complete PM V1.5 H1 RAG human review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h1-complete-rag-human-review-v1"
REPORT_TITLE = "PM V1.5 H1 RAG Audit"


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _acceptable_ranks(value: object) -> set[int]:
    return {
        int(token)
        for token in re.findall(r"[1-5]", str(value or ""))
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _source() -> dict[str, Any]:
    return {
        "id": "h1_review_source",
        "label": "PM V1.5 H1 complete RAG review",
        "path": "outputs/pm_v1_5_h1_complete_rag_review_v1_analysis/h1_analysis.json",
        "query": {
            "engine": "DuckDB",
            "sql": (
                "SELECT * FROM read_json_auto("
                "'outputs/pm_v1_5_h1_complete_rag_review_v1_analysis/"
                "report_source_rows.jsonl')"
            ),
            "description": (
                "Exact-ID join of 130 independent human annotations to the "
                "frozen H1 review packet and private retrieval audit."
            ),
            "language": "Python",
            "tables_used": [
                "h1_complete_rag_annotations.jsonl",
                "h1_review_packet.json",
                "private_retrieval_audit.jsonl",
            ],
            "filters": [
                "protocol = pm-v1.5-h1-complete-rag-human-review-v1",
                "50 card-core items and 80 retrieval items",
                "retrieval metrics use 64 ranked items unless stated",
            ],
            "metric_definitions": [
                "Top-1 fit rate = human top1_fit=yes / ranked retrieval items.",
                "Opportunity-conditioned Top-5 recall = RS-opportunity items with at least one acceptable rank / RS-opportunity items.",
                "Source-gate precision = human source-approved cores / cores that passed the old provisional source-count gate.",
                "Coarse opportunity-gate correctness counts 57 correctly ranked opportunity states plus 16 correctly abstained negative controls over all 80 retrieval items.",
            ],
        },
    }


def _artifact(
    *,
    summary: dict[str, Any],
    family_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    source_gate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source = _source()
    rates = summary["retrieval"]["rates"]
    qualification_rows = [
        {
            "metric": "Top-1 suitable, all ranked",
            "rate": rates["top1_fit_all_ranked"],
            "numerator": 33,
            "denominator": 64,
            "registered_gate": 0.80,
            "gate_passed": False,
        },
        {
            "metric": "Top-1 suitable, opportunity only",
            "rate": rates["top1_fit_given_opportunity"],
            "numerator": 33,
            "denominator": 57,
            "registered_gate": 0.80,
            "gate_passed": False,
        },
        {
            "metric": "Top-5 recall, opportunity only",
            "rate": rates["top5_acceptable_recall_given_opportunity"],
            "numerator": 50,
            "denominator": 57,
            "registered_gate": 0.90,
            "gate_passed": False,
        },
        {
            "metric": "Negative-control abstention",
            "rate": rates["negative_abstention_accuracy"],
            "numerator": 16,
            "denominator": 16,
            "registered_gate": 1.0,
            "gate_passed": True,
        },
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": REPORT_TITLE,
        "description": "Human-audited card evidence and retrieval qualification.",
        "generatedAt": "2026-07-30T00:00:00+09:00",
        "sources": [source],
        "charts": [
            {
                "id": "qualification_rates_chart",
                "title": "H1 retrieval qualification rates",
                "subtitle": "Top-1 and Top-5 miss their preregistered gates; deterministic negative abstention passes.",
                "showDescription": True,
                "intent": "comparison",
                "question": "Which H1 retrieval gates passed?",
                "rationale": "A bar chart makes four bounded rates and their denominators directly comparable.",
                "comparisonContext": {
                    "basis": "registered gate",
                    "reference": "0.80 Top-1, 0.90 Top-5, 1.00 hard-off abstention",
                },
                "type": "bar",
                "dataset": "qualification_rates",
                "sourceId": "h1_review_source",
                "encodings": {
                    "x": {
                        "field": "metric",
                        "type": "nominal",
                        "label": "Metric",
                    },
                    "y": {
                        "field": "rate",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "Rate",
                    },
                    "tooltip": [
                        {"field": "numerator", "type": "quantitative", "label": "Numerator"},
                        {"field": "denominator", "type": "quantitative", "label": "Denominator"},
                        {"field": "registered_gate", "type": "quantitative", "format": "percent", "label": "Gate"},
                    ],
                },
                "valueFormat": "percent",
                "layout": "full",
            }
        ],
        "tables": [
            {
                "id": "family_table",
                "title": "Card evidence and retrieval results by strategy family",
                "subtitle": "Card cores and ranked retrieval states in the completed H1 review.",
                "dataset": "family_summary",
                "defaultSort": {"field": "top1_fit_rate", "direction": "desc"},
                "density": "spacious",
                "sourceId": "h1_review_source",
                "columns": [
                    {"field": "family", "label": "Strategy family", "type": "text"},
                    {"field": "cores", "label": "Cores", "type": "number"},
                    {"field": "source_approved", "label": "Source-approved", "type": "number"},
                    {"field": "variants_approved", "label": "Variants approved", "type": "number"},
                    {"field": "top1_items", "label": "Top-1 items", "type": "number"},
                    {"field": "top1_fit", "label": "Top-1 fit", "type": "number"},
                    {"field": "top1_fit_rate", "label": "Top-1 fit rate", "format": "percent"},
                    {"field": "hard_exclusions", "label": "Hard exclusions", "type": "number"},
                ],
            },
            {
                "id": "failure_table",
                "title": "Top-1 outcome decomposition",
                "subtitle": "All 64 ranked retrieval states, mutually exclusive.",
                "dataset": "failure_decomposition",
                "defaultSort": {"field": "count", "direction": "desc"},
                "density": "spacious",
                "sourceId": "h1_review_source",
                "columns": [
                    {"field": "outcome", "label": "Outcome", "type": "text"},
                    {"field": "count", "label": "States", "type": "number"},
                    {"field": "share", "label": "Share", "format": "percent"},
                    {"field": "interpretation", "label": "Interpretation", "type": "text"},
                ],
            },
            {
                "id": "source_gate_table",
                "title": "Old source-count gate against human source review",
                "subtitle": "Fifty core cards; the old gate measured volume, not source validity.",
                "dataset": "source_gate",
                "defaultSort": {"field": "count", "direction": "desc"},
                "density": "spacious",
                "sourceId": "h1_review_source",
                "columns": [
                    {"field": "old_gate", "label": "Old gate", "type": "text"},
                    {"field": "human_source", "label": "Human source", "type": "text"},
                    {"field": "count", "label": "Cores", "type": "number"},
                ],
            },
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "h1_review_source",
                "body": (
                    "## Technical summary\n\n"
                    "**The RS opportunity construct is supported, but the current Bank and retriever are not qualified.** "
                    "Human review found an RS opportunity in 57/64 enriched ranked states, while all 16 hard-off controls were correctly abstained. "
                    "However, Top-1 was suitable in only 33/64 states, eight Top-1 cards triggered hard exclusions, and seven true-opportunity states had no safe card in Top-5.\n\n"
                    "**Card definitions are stronger than their provenance.** All 50 core moves passed clarity, use-condition, and boundary review, but only 11 cores had valid displayed source evidence. "
                    "Three cores were substantive duplicates. The defensible immediate Bank is therefore 22 approved variants, with 72 variants awaiting source relinking and six variants rejected."
                ),
            },
            {
                "id": "qualification_heading",
                "type": "markdown",
                "body": (
                    "## The opportunity gate is promising; retrieval is the bottleneck\n\n"
                    "The coarse gate made the right rank-versus-abstain decision for 73/80 H1 items, but the ranked arm was deliberately opportunity-enriched and is not a prevalence estimate. "
                    "Within the 64 ranked states, 17 failures were ranking errors with a safe lower-ranked alternative, seven were true card-coverage gaps, and seven should have abstained before ranking."
                ),
            },
            {"id": "qualification_chart", "type": "chart", "chartId": "qualification_rates_chart"},
            {
                "id": "card_heading",
                "type": "markdown",
                "body": (
                    "## The taxonomy survives, but the old source gate does not\n\n"
                    "The human reviewer accepted the definitions and boundary clauses for every core. "
                    "The old count-based source gate had only 9/31 precision against human source review: 22 nominal passes lacked direct action evidence. "
                    "It must be retired as a quality gate; the 36 nonduplicate revised cores need new literal train-only evidence, not rewritten definitions."
                ),
            },
            {"id": "family_table_block", "type": "table", "tableId": "family_table"},
            {"id": "source_gate_table_block", "type": "table", "tableId": "source_gate_table"},
            {
                "id": "failure_heading",
                "type": "markdown",
                "body": (
                    "## Failures separate cleanly into routing, ranking, and coverage\n\n"
                    "Top-1 failure is not one undifferentiated model problem. "
                    "Seven ranked states had no RS opportunity, including two malformed states ending on a supporter turn; seven opportunity states lacked any safe Top-5 card; and 17 had a safe candidate but the ranker placed another card first. "
                    "Four additional opportunity states had a Top-1 hard exclusion that a lower rank could avoid."
                ),
            },
            {"id": "failure_table_block", "type": "table", "tableId": "failure_table"},
            {
                "id": "scope_methods",
                "type": "markdown",
                "body": (
                    "## Scope and method\n\n"
                    "The audit joins 130 unique human records to the frozen H1 packet: 50 core-card judgments, 64 ranked train-only retrieval states, and 16 deterministic abstention controls. "
                    "Top-5 recall uses only the 57 states the reviewer judged to contain an RS opportunity. "
                    "Source examples are evidence for whether an atomic move appears; they are not response-quality labels or PM targets."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and robustness\n\n"
                    "H1 uses one independent reviewer, so it establishes an adjudicated development decision rather than inter-rater reliability. "
                    "The 64 ranked states were selected to exercise active submoves, so the 89% opportunity share cannot estimate natural ESConv prevalence. "
                    "Two states ending in a supporter turn expose a packet-construction defect and must be replaced. "
                    "Any repair learned from H1 requires a fresh, sealed qualification slice; H1 replay cannot be reported as unbiased accuracy."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Delete the three duplicate cores and their six variants.\n"
                    "2. Freeze the 11 source-approved cores (22 variants); re-source the 36 revised cores with at least two independent literal train examples each, in one 36-item correction packet.\n"
                    "3. Require the latest visible turn to be a seeker turn and extend general hard-off guards for violence, substance/child safety, legal or occupational misconduct, factual-resource requests, and routine closing.\n"
                    "4. Use H1 acceptable ranks to repair transparent applicability and compare lexical versus frozen BGE scoring offline; do not tune on response outcomes.\n"
                    "5. Use the sealed H2 internal RS slice as the fresh post-repair retrieval qualification, avoiding another standalone human packet."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "Can at least 30 distinct cores obtain two direct train-only source examples without lowering evidence standards? "
                    "After source relinking and hard-off repair, does a transparent reranker reach the 0.80 Top-1 gate on the sealed H2 slice? "
                    "Does frozen BGE improve ordering among already-applicable cards, rather than merely increasing semantic similarity?"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-07-30T00:00:00+09:00",
            "status": "ready",
            "datasets": {
                "qualification_rates": qualification_rows,
                "family_summary": family_rows,
                "failure_decomposition": decomposition_rows,
                "source_gate": source_gate_rows,
            },
        },
        "sources": [source],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/h1_review_packet.json",
    )
    parser.add_argument(
        "--private-audit",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/private_retrieval_audit.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h1_complete_rag_review_v1_analysis",
    )
    args = parser.parse_args()

    annotations = [dict(row) for row in iter_jsonl(args.annotations)]
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    items = {
        str(row["review_item_id"]): row
        for row in packet["card_items"] + packet["retrieval_items"]
    }
    annotation_by_id = {
        str(row["review_item_id"]): row for row in annotations
    }
    checks = {
        "annotation_rows_130": len(annotations) == 130,
        "annotation_ids_unique": len(annotation_by_id) == len(annotations),
        "packet_ids_unique": len(items) == 130,
        "exact_id_join": set(annotation_by_id) == set(items),
        "protocol_exact": all(row.get("protocol") == PROTOCOL for row in annotations),
        "annotator_present": all(str(row.get("annotator_id", "")).strip() for row in annotations),
        "card_rows_50": sum(row.get("item_type") == "card" for row in annotations) == 50,
        "retrieval_rows_80": sum(row.get("item_type") == "retrieval" for row in annotations) == 80,
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))

    cards = [row for row in annotations if row["item_type"] == "card"]
    ranked = [row for row in annotations if row.get("top1_fit") in {"yes", "no"}]
    abstentions = [row for row in annotations if row.get("abstention_correct")]
    if len(ranked) != 64 or len(abstentions) != 16:
        raise RuntimeError("retrieval response cardinality mismatch")

    card_status = Counter()
    family_rows: list[dict[str, Any]] = []
    families = sorted(
        {str(items[row["review_item_id"]]["strategy_family"]) for row in cards}
    )
    for row in cards:
        if (
            row["minimal_variant"] == "reject"
            and row["dialogic_variant"] == "reject"
        ):
            card_status["reject_core"] += 1
        elif (
            row["minimal_variant"] == "approve"
            and row["dialogic_variant"] == "approve"
            and row["source_support_valid"] == "yes"
            and row["distinctness"] == "distinct"
        ):
            card_status["approve_core"] += 1
        else:
            card_status["revise_core"] += 1

    for family in families:
        family_cards = [
            row
            for row in cards
            if items[row["review_item_id"]]["strategy_family"] == family
        ]
        family_ranked = [
            row
            for row in ranked
            if items[row["review_item_id"]]["top5"][0]["card"]["strategy_family"]
            == family
        ]
        top1_fit = sum(row["top1_fit"] == "yes" for row in family_ranked)
        family_rows.append(
            {
                "family": family,
                "cores": len(family_cards),
                "source_approved": sum(
                    row["source_support_valid"] == "yes" for row in family_cards
                ),
                "variants_approved": 2
                * sum(row["minimal_variant"] == "approve" for row in family_cards),
                "variants_revise": 2
                * sum(row["minimal_variant"] == "revise" for row in family_cards),
                "variants_reject": 2
                * sum(row["minimal_variant"] == "reject" for row in family_cards),
                "top1_items": len(family_ranked),
                "top1_fit": top1_fit,
                "top1_fit_rate": _rate(top1_fit, len(family_ranked)),
                "hard_exclusions": sum(
                    row["hard_exclusion_triggered"] == "yes"
                    for row in family_ranked
                ),
            }
        )

    old_source_cross = Counter(
        (
            "pass"
            if items[row["review_item_id"]]["source_support"][
                "provisional_source_support_pass"
            ]
            else "fail",
            row["source_support_valid"],
        )
        for row in cards
    )
    source_gate_rows = [
        {
            "old_gate": old_gate,
            "human_source": human,
            "count": old_source_cross[(old_gate, human)],
        }
        for old_gate in ("pass", "fail")
        for human in ("yes", "no")
    ]
    source_tp = old_source_cross[("pass", "yes")]
    source_fp = old_source_cross[("pass", "no")]
    source_fn = old_source_cross[("fail", "yes")]
    source_tn = old_source_cross[("fail", "no")]

    opportunity_yes = [row for row in ranked if row["rs_opportunity"] == "yes"]
    opportunity_no = [row for row in ranked if row["rs_opportunity"] == "no"]
    top1_yes = [row for row in ranked if row["top1_fit"] == "yes"]
    hard = [
        row for row in ranked if row["hard_exclusion_triggered"] == "yes"
    ]
    top5_supported = [
        row for row in opportunity_yes if _acceptable_ranks(row.get("acceptable_ranks"))
    ]
    ranking_failure = [
        row
        for row in ranked
        if row["rs_opportunity"] == "yes"
        and row["top1_fit"] == "no"
        and row["best_candidate"] != "no_safe_card"
    ]
    coverage_gap = [
        row
        for row in ranked
        if row["rs_opportunity"] == "yes"
        and row["best_candidate"] == "no_safe_card"
    ]
    last_supporter = [
        row
        for row in ranked
        if items[row["review_item_id"]]["visible_dialogue"][-1]["speaker"]
        == "supporter"
    ]
    decomposition_rows = [
        {
            "outcome": "Top-1 suitable",
            "count": len(top1_yes),
            "share": _rate(len(top1_yes), len(ranked)),
            "interpretation": "Current card is usable.",
        },
        {
            "outcome": "Ranking failure",
            "count": len(ranking_failure),
            "share": _rate(len(ranking_failure), len(ranked)),
            "interpretation": "RS opportunity and safe lower-ranked card exist.",
        },
        {
            "outcome": "Top-5 coverage gap",
            "count": len(coverage_gap),
            "share": _rate(len(coverage_gap), len(ranked)),
            "interpretation": "RS opportunity exists but no safe Top-5 card.",
        },
        {
            "outcome": "Should abstain before ranking",
            "count": len(opportunity_no),
            "share": _rate(len(opportunity_no), len(ranked)),
            "interpretation": "No ordinary RS opportunity in the reviewed state.",
        },
    ]
    if sum(row["count"] for row in decomposition_rows) != 64:
        raise RuntimeError("failure decomposition is not exhaustive")

    duplicate_rows = [
        {
            "core_submove_id": items[row["review_item_id"]]["core_submove_id"],
            "duplicate_of": row["duplicate_of"],
            "required_corrections": row["required_corrections"],
        }
        for row in cards
        if row["distinctness"] == "duplicate"
    ]
    hard_rows = [
        {
            "review_item_id": row["review_item_id"],
            "top1_core_submove_id": items[row["review_item_id"]]["top5"][0][
                "card"
            ]["core_submove_id"],
            "rs_opportunity": row["rs_opportunity"],
            "best_candidate": row["best_candidate"],
            "notes": row["notes"],
        }
        for row in hard
    ]

    summary = {
        "protocol": PROTOCOL,
        "status": "H1_COMPLETE_BANK_AND_RETRIEVER_NOT_QUALIFIED_REPAIR_ONCE",
        "data_quality": {
            "checks": checks,
            "grain": "one independent reviewer judgment per frozen H1 review_item_id",
            "completeness": "130/130",
            "join_coverage": "130/130",
            "duplicate_annotation_ids": 0,
            "confidence": "high for descriptive H1 decisions; no IAA claim",
        },
        "cards": {
            "core_counts": dict(card_status),
            "variant_counts": {
                "approve": 2 * card_status["approve_core"],
                "revise": 2 * card_status["revise_core"],
                "reject": 2 * card_status["reject_core"],
            },
            "definition_clear": sum(
                row["support_move_clear"] == "yes" for row in cards
            ),
            "when_to_use_valid": sum(
                row["when_to_use_valid"] == "yes" for row in cards
            ),
            "boundary_risk_valid": sum(
                row["boundary_risk_valid"] == "yes" for row in cards
            ),
            "source_evidence_valid": sum(
                row["source_support_valid"] == "yes" for row in cards
            ),
            "source_evidence_invalid": sum(
                row["source_support_valid"] == "no" for row in cards
            ),
            "duplicate_cores": duplicate_rows,
            "family_summary": family_rows,
            "old_source_gate": {
                "confusion": {
                    "true_positive": source_tp,
                    "false_positive": source_fp,
                    "false_negative": source_fn,
                    "true_negative": source_tn,
                },
                "precision": _rate(source_tp, source_tp + source_fp),
                "recall": _rate(source_tp, source_tp + source_fn),
                "specificity": _rate(source_tn, source_tn + source_fp),
                "accuracy": _rate(source_tp + source_tn, len(cards)),
                "decision": "RETIRE_AS_SOURCE_QUALITY_GATE",
            },
        },
        "retrieval": {
            "ranked_items": len(ranked),
            "negative_controls": len(abstentions),
            "opportunity_yes": len(opportunity_yes),
            "opportunity_no": len(opportunity_no),
            "top1_fit": len(top1_yes),
            "top1_not_fit": len(ranked) - len(top1_yes),
            "hard_exclusions": len(hard),
            "top5_supported_opportunity_items": len(top5_supported),
            "top5_no_safe_opportunity_items": len(coverage_gap),
            "ranking_failures_with_safe_alternative": len(ranking_failure),
            "last_visible_turn_supporter_defects": len(last_supporter),
            "negative_abstention_correct": sum(
                row["abstention_correct"] == "yes" for row in abstentions
            ),
            "rates": {
                "top1_fit_all_ranked": _rate(len(top1_yes), len(ranked)),
                "top1_fit_given_opportunity": _rate(
                    len(top1_yes), len(opportunity_yes)
                ),
                "top5_acceptable_recall_given_opportunity": _rate(
                    len(top5_supported), len(opportunity_yes)
                ),
                "hard_exclusion_all_ranked": _rate(len(hard), len(ranked)),
                "negative_abstention_accuracy": _rate(
                    sum(row["abstention_correct"] == "yes" for row in abstentions),
                    len(abstentions),
                ),
                "coarse_rank_or_abstain_correctness_enriched_H1": _rate(
                    len(opportunity_yes)
                    + sum(
                        row["abstention_correct"] == "yes"
                        for row in abstentions
                    ),
                    len(ranked) + len(abstentions),
                ),
            },
            "failure_decomposition": decomposition_rows,
            "family_summary": family_rows,
            "hard_exclusion_items": hard_rows,
        },
        "scientific_decision": {
            "rs_opportunity_construct_supported": True,
            "current_100_variant_bank_qualified": False,
            "current_retriever_qualified": False,
            "baai_required_before_repair": False,
            "pm_rs_training_should_be_abandoned": False,
            "why": (
                "Opportunity versus no-opportunity is largely separable in "
                "this enriched audit, while most failures occur after the "
                "gate: source provenance, card coverage, and ranking."
            ),
        },
        "next": {
            "delete_duplicate_cores": 3,
            "freeze_source_approved_cores": 11,
            "freeze_approved_variants": 22,
            "resource_relink_cores": 36,
            "maximum_final_variants_after_duplicate_removal": 94,
            "source_correction_packet": (
                "One item per revised core; show five fresh train-only source "
                "candidates and require at least two independent literal "
                "supports. Do not re-review definitions."
            ),
            "retrieval_repair": [
                "require latest visible turn speaker=seeker",
                "extend general hard-off and factual/resource-request guards",
                "use H1 acceptable ranks to repair transparent applicability",
                "compare lexical and frozen BGE only on H1 development labels",
                "qualify once on sealed H2 internal RS slice",
            ],
            "additional_standalone_human_retrieval_packet": False,
        },
        "lineage": {
            "annotations_sha256": sha256_file(args.annotations),
            "packet_sha256": sha256_file(args.packet),
            "private_audit_sha256": sha256_file(args.private_audit),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bound_rows = [
        {
            **items[row["review_item_id"]],
            "human_annotation": row,
        }
        for row in annotations
    ]
    _write_json(args.out_dir / "h1_analysis.json", summary)
    _write_jsonl(args.out_dir / "h1_annotations_bound.jsonl", bound_rows)
    _write_jsonl(args.out_dir / "h1_annotations_frozen.jsonl", annotations)
    artifact = _artifact(
        summary=summary,
        family_rows=family_rows,
        decomposition_rows=decomposition_rows,
        source_gate_rows=source_gate_rows,
    )
    source_rows: list[dict[str, Any]] = []
    for dataset, rows in artifact["snapshot"]["datasets"].items():
        source_rows.extend({"dataset": dataset, **row} for row in rows)
    _write_jsonl(args.out_dir / "report_source_rows.jsonl", source_rows)
    _write_json(
        args.out_dir / "artifact.json",
        artifact,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
