#!/usr/bin/env python3
"""Aggregate the human audit of natural Strategy-RAG Top-1 retrieval.

This analysis is deliberately a development diagnostic.  It validates the
annotation lineage, separates catalog coverage from ranking quality, and
prevents response generation when the reviewed Top-1 treatment is not clean.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-natural-retrieval-fit-review-analysis-v1"
EXPECTED_SOURCE_PROTOCOL = "pm-v1.5-rs-natural-retrieval-fit-review-v1"


def _wilson(k: int, n: int, z: float = 1.96) -> dict[str, float]:
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return {
        "rate": round(p, 6),
        "wilson_95_low": round(center - radius, 6),
        "wilson_95_high": round(center + radius, 6),
    }


def _numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def _auc(rows: list[dict[str, Any]], field: str) -> float:
    positive = [float(row[field]) for row in rows if row["clean_top1"]]
    negative = [float(row[field]) for row in rows if not row["clean_top1"]]
    comparisons = [
        float(left > right) + 0.5 * float(left == right)
        for left in positive
        for right in negative
    ]
    return round(sum(comparisons) / len(comparisons), 6)


def _segment(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    output: list[dict[str, Any]] = []
    for value, group in sorted(grouped.items()):
        output.append(
            {
                field: value,
                "n": len(group),
                "clean_top1": sum(bool(row["clean_top1"]) for row in group),
                "clean_top1_rate": round(
                    sum(bool(row["clean_top1"]) for row in group) / len(group),
                    6,
                ),
                "hard_exclusion": sum(
                    row["hard_exclusion_triggered"] == "yes" for row in group
                ),
                "better_candidate_counts": dict(
                    sorted(Counter(row["better_candidate"] for row in group).items())
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_candidate"
            / "retrieval_review_packet.jsonl"
        ),
    )
    parser.add_argument(
        "--private-audit",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_candidate"
            / "private_selection_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_analysis"
        ),
    )
    args = parser.parse_args()

    annotations = [dict(row) for row in iter_jsonl(args.annotations)]
    packet = {
        str(row["review_item_id"]): dict(row) for row in iter_jsonl(args.packet)
    }
    private = {
        str(row["review_item_id"]): dict(row)
        for row in iter_jsonl(args.private_audit)
    }
    annotation_ids = [str(row.get("review_item_id", "")) for row in annotations]
    required = {
        "protocol",
        "review_item_id",
        "top1_fit",
        "hard_exclusion_triggered",
        "better_candidate",
        "annotator_id",
    }
    allowed_fit = {"yes", "no", "uncertain"}
    allowed_binary = {"yes", "no", "uncertain"}
    allowed_better = {"none", "rank2", "rank3", "no_safe_card", "uncertain"}
    checks = {
        "exactly_32_annotations": len(annotations) == 32,
        "unique_annotation_ids": len(annotation_ids) == len(set(annotation_ids)),
        "ids_match_packet": set(annotation_ids) == set(packet),
        "ids_match_private_audit": set(annotation_ids) == set(private),
        "source_protocol_matches": all(
            row.get("protocol") == EXPECTED_SOURCE_PROTOCOL for row in annotations
        ),
        "required_fields_complete": all(
            required <= set(row)
            and all(str(row.get(field, "")).strip() for field in required)
            for row in annotations
        ),
        "values_in_schema": all(
            row.get("top1_fit") in allowed_fit
            and row.get("hard_exclusion_triggered") in allowed_binary
            and row.get("better_candidate") in allowed_better
            for row in annotations
        ),
        "one_state_per_dialogue": len(
            {
                str(private[item_id]["source_dialogue_id"])
                for item_id in annotation_ids
            }
        )
        == len(annotations),
    }
    if not all(checks.values()):
        raise RuntimeError(canonical_json(checks))

    joined: list[dict[str, Any]] = []
    for annotation in annotations:
        item_id = str(annotation["review_item_id"])
        visible = packet[item_id]
        audit = private[item_id]
        clean_top1 = (
            annotation["top1_fit"] == "yes"
            and annotation["hard_exclusion_triggered"] == "no"
            and annotation["better_candidate"] == "none"
        )
        top3 = list(audit["top3"])
        preferred_rank: int | None
        preferred_card_id: str | None
        if clean_top1:
            preferred_rank = 1
            preferred_card_id = str(top3[0]["card_id"])
        elif annotation["better_candidate"] in {"rank2", "rank3"}:
            preferred_rank = int(str(annotation["better_candidate"])[-1])
            preferred_card_id = str(top3[preferred_rank - 1]["card_id"])
        else:
            preferred_rank = None
            preferred_card_id = None
        joined.append(
            {
                **annotation,
                "state_id": str(audit["state_id"]),
                "source_dialogue_id": str(audit["source_dialogue_id"]),
                "source_turn_index": int(audit["source_turn_index"]),
                "stratum": str(audit["stratum"]),
                "execution_profile": str(audit["execution_profile"]),
                "top1_card_id": str(top3[0]["card_id"]),
                "top1_core_submove_id": str(top3[0]["core_submove_id"]),
                "top1_strategy_family": str(top3[0]["strategy_family"]),
                "retrieval_score": float(audit["retrieval_score"]),
                "top_vs_second_margin": float(audit["top_vs_second_margin"]),
                "clean_top1": clean_top1,
                "preferred_rank": preferred_rank,
                "preferred_card_id": preferred_card_id,
                "visible_dialogue_sha256": str(
                    audit["visible_dialogue_sha256"]
                ),
            }
        )

    logical_checks = {
        "every_yes_is_clean_and_none": all(
            row["top1_fit"] != "yes" or row["clean_top1"] for row in joined
        ),
        "every_no_names_alternative_or_no_safe_card": all(
            row["top1_fit"] != "no"
            or row["better_candidate"] in {"rank2", "rank3", "no_safe_card"}
            for row in joined
        ),
        "no_clean_top1_has_hard_exclusion": all(
            not row["clean_top1"]
            or row["hard_exclusion_triggered"] == "no"
            for row in joined
        ),
    }
    if not all(logical_checks.values()):
        raise RuntimeError(canonical_json(logical_checks))

    n = len(joined)
    clean = sum(bool(row["clean_top1"]) for row in joined)
    exclusions = sum(
        row["hard_exclusion_triggered"] == "yes" for row in joined
    )
    better_rank23 = sum(
        row["better_candidate"] in {"rank2", "rank3"} for row in joined
    )
    no_safe = sum(row["better_candidate"] == "no_safe_card" for row in joined)
    safe_rows = [row for row in joined if row["clean_top1"]]
    failed_rows = [row for row in joined if not row["clean_top1"]]
    reciprocal_rank = sum(
        1.0 / int(row["preferred_rank"])
        if row["preferred_rank"] is not None
        else 0.0
        for row in joined
    ) / n

    report = {
        "protocol": PROTOCOL,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": (
            "DEVELOPMENT_AUDIT_COMPLETE_NATURAL_TOP1_FAILED_"
            "STOP_BEFORE_GENERATION"
        ),
        "checks": {**checks, **logical_checks},
        "counts": {
            "review_items": n,
            "clean_top1": clean,
            "hard_exclusion": exclusions,
            "better_rank2_or_rank3": better_rank23,
            "no_safe_card_in_visible_top3": no_safe,
            "diagnostic_top3_candidate_signal": clean + better_rank23,
        },
        "rates_with_wilson_95": {
            "clean_top1": _wilson(clean, n),
            "hard_exclusion": _wilson(exclusions, n),
            "diagnostic_top3_candidate_signal": _wilson(
                clean + better_rank23, n
            ),
            "no_safe_card_in_visible_top3": _wilson(no_safe, n),
        },
        "rank_diagnostics": {
            "preferred_rank_counts": {
                "rank1": clean,
                "rank2": sum(
                    row["better_candidate"] == "rank2" for row in joined
                ),
                "rank3": sum(
                    row["better_candidate"] == "rank3" for row in joined
                ),
                "none": no_safe,
            },
            "diagnostic_reciprocal_rank": round(reciprocal_rank, 6),
            "important_caveat": (
                "Rank2/3 choices indicate a better visible candidate but do "
                "not independently certify that candidate against every hard "
                "exclusion."
            ),
        },
        "score_diagnostics": {
            "clean_top1_score": _numeric_summary(
                [float(row["retrieval_score"]) for row in safe_rows]
            ),
            "failed_top1_score": _numeric_summary(
                [float(row["retrieval_score"]) for row in failed_rows]
            ),
            "clean_top1_margin": _numeric_summary(
                [float(row["top_vs_second_margin"]) for row in safe_rows]
            ),
            "failed_top1_margin": _numeric_summary(
                [float(row["top_vs_second_margin"]) for row in failed_rows]
            ),
            "score_auc_for_clean_top1": _auc(joined, "retrieval_score"),
            "margin_auc_for_clean_top1": _auc(
                joined, "top_vs_second_margin"
            ),
            "interpretation": (
                "Neither raw lexical score nor Top1-Top2 margin is a valid "
                "applicability confidence signal in this development batch."
            ),
        },
        "by_stratum": _segment(joined, "stratum"),
        "by_profile": _segment(joined, "execution_profile"),
        "by_family": _segment(joined, "top1_strategy_family"),
        "by_core_submove": _segment(joined, "top1_core_submove_id"),
        "research_use": {
            "valid_for": [
                "development diagnosis of natural retrieval",
                "transparent card-applicability rule repair",
                "ranking challenger construction",
            ],
            "not_valid_for": [
                "PM component-effect training labels",
                "final retrieval accuracy claims",
                "general Strategy-RAG benefit claims",
                "EvoEmo coverage claims",
            ],
            "human_evidence_level": (
                "single independent reviewer; complete and internally "
                "consistent, but no inter-rater reliability"
            ),
        },
        "decision": {
            "generate_r0_rs_pairs_now": False,
            "reason": (
                "Only 13 of 32 Top1 treatments are clean; generating only "
                "those cannot satisfy the planned 8-positive/8-nonpositive "
                "minimum, while generating rejected treatments would "
                "confound retrieval failure with PM component effect."
            ),
            "next": (
                "Repair card-level applicability without response outcomes, "
                "freeze it, then run a fresh disjoint holdout review."
            ),
        },
        "proposed_fresh_holdout_gate": {
            "independent_dialogues": 16,
            "minimum_clean_top1": 12,
            "maximum_hard_exclusion": 0,
            "minimum_represented_family_clean_rate": 0.5,
            "tuning_on_holdout_forbidden": True,
        },
        "source_lineage": {
            "annotations": str(args.annotations),
            "annotations_sha256": sha256_file(args.annotations),
            "packet": str(args.packet.relative_to(ROOT)),
            "packet_sha256": sha256_file(args.packet),
            "private_audit": str(args.private_audit.relative_to(ROOT)),
            "private_audit_sha256": sha256_file(args.private_audit),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "analysis_report.json", report)
    write_jsonl(args.out_dir / "normalized_annotations.jsonl", annotations)
    write_jsonl(args.out_dir / "joined_review_decisions.jsonl", joined)
    (args.out_dir / "NEXT_DECISION_ZH.md").write_text(
        "\n".join(
            [
                "# PM V1.5 自然 Strategy-RAG 检索审计结论",
                "",
                "状态：`STOP_BEFORE_GENERATION / DEVELOPMENT BATCH`",
                "",
                f"- 安全合适 Top-1：{clean}/{n}",
                f"- 硬排除：{exclusions}/{n}",
                f"- 审阅者指出 Rank-2/3 更好：{better_rank23}/{n}",
                f"- 可见 Top-3 无安全卡：{no_safe}/{n}",
                "",
                "当前 32 条只能用于修复检索，不能用作 PM 收益标签。",
                "修复后必须冻结方法，再在零重叠的 16 个 ESConv-train",
                "dialogue 上做一次轻量留出复核。ESConv formal test 与",
                "EvoEmo 不得参与修复。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "clean_top1": f"{clean}/{n}",
            "hard_exclusion": f"{exclusions}/{n}",
            "better_rank2_or_rank3": f"{better_rank23}/{n}",
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
