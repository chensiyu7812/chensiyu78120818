#!/usr/bin/env python3
"""Analyze the frozen 16-dialogue natural-retrieval holdout review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-natural-retrieval-holdout-analysis-v1"
SOURCE_PROTOCOL = "pm-v1.5-rs-natural-retrieval-fit-holdout-v3-final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v3_final_candidate"
            / "retrieval_review_packet.jsonl"
        ),
    )
    parser.add_argument(
        "--private-audit",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v3_final_candidate"
            / "private_selection_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v3_analysis"
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
    ids = [str(row.get("review_item_id", "")) for row in annotations]
    checks = {
        "exactly_16_annotations": len(annotations) == 16,
        "unique_ids": len(ids) == len(set(ids)),
        "ids_match_packet": set(ids) == set(packet),
        "ids_match_private_audit": set(ids) == set(private),
        "source_protocol_matches": all(
            row.get("protocol") == SOURCE_PROTOCOL for row in annotations
        ),
        "all_required_decisions_present": all(
            row.get("top1_fit") in {"yes", "no", "uncertain"}
            and row.get("hard_exclusion_triggered") in {
                "yes",
                "no",
                "uncertain",
            }
            and row.get("better_candidate")
            in {"none", "rank2", "rank3", "no_safe_card", "uncertain"}
            and str(row.get("annotator_id", "")).strip()
            for row in annotations
        ),
        "one_dialogue_per_item": len(
            {str(private[item_id]["source_dialogue_id"]) for item_id in ids}
        )
        == len(annotations),
    }
    if not all(checks.values()):
        raise RuntimeError(canonical_json(checks))

    joined: list[dict[str, Any]] = []
    for annotation in annotations:
        item_id = str(annotation["review_item_id"])
        audit = private[item_id]
        clean = (
            annotation["top1_fit"] == "yes"
            and annotation["hard_exclusion_triggered"] == "no"
            and annotation["better_candidate"] == "none"
        )
        joined.append(
            {
                **annotation,
                "source_dialogue_id": str(audit["source_dialogue_id"]),
                "state_id": str(audit["state_id"]),
                "stratum": str(audit["stratum"]),
                "execution_profile": str(audit["execution_profile"]),
                "selected_strategy_family": str(
                    audit["selected_strategy_family"]
                ),
                "selected_core_submove_id": str(
                    audit["selected_core_submove_id"]
                ),
                "selected_card_id": str(audit["selected_card_id"]),
                "clean_top1": clean,
            }
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_family[str(row["selected_strategy_family"])].append(row)
    family_results = []
    for family, rows in sorted(by_family.items()):
        clean = sum(bool(row["clean_top1"]) for row in rows)
        family_results.append(
            {
                "family": family,
                "n": len(rows),
                "clean_top1": clean,
                "clean_rate": round(clean / len(rows), 6),
            }
        )

    clean = sum(bool(row["clean_top1"]) for row in joined)
    exclusions = sum(
        row["hard_exclusion_triggered"] == "yes" for row in joined
    )
    gate = {
        "at_least_12_of_16_clean_top1": clean >= 12,
        "zero_hard_exclusion": exclusions == 0,
        "each_represented_family_at_least_half_clean": all(
            row["clean_rate"] >= 0.5 for row in family_results
        ),
    }
    passed = all(gate.values())
    report = {
        "protocol": PROTOCOL,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": (
            "NATURAL_RETRIEVER_QUALIFIED_FOR_TRAIN_ONLY_CLEAN_PAIRS"
            if passed
            else "NATURAL_RETRIEVER_HOLDOUT_FAILED_NO_RETUNING_ON_HOLDOUT"
        ),
        "checks": checks,
        "gate": gate,
        "passed": passed,
        "counts": {
            "review_items": len(joined),
            "clean_top1": clean,
            "hard_exclusion": exclusions,
            "top1_fit": dict(sorted(Counter(
                row["top1_fit"] for row in joined
            ).items())),
            "better_candidate": dict(sorted(Counter(
                row["better_candidate"] for row in joined
            ).items())),
        },
        "by_family": family_results,
        "data_firewall": {
            "development_32_used_for_holdout_selection": False,
            "pre_freeze_dry_run_16_excluded": True,
            "response_outcomes_used": False,
            "formal_esconv_test_used": False,
            "evoemo_used": False,
            "retuning_on_this_holdout_allowed": False,
        },
        "next": (
            "Select train-only state-card pairs outcome-blind and begin R0/RS "
            "matched generation."
            if passed
            else (
                "Do not retune on these 16 labels. Fall back to the narrow "
                "six-card runtime or report expanded natural retrieval as a "
                "V1.5 limitation."
            )
        ),
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
    write_json(args.out_dir / "holdout_analysis_report.json", report)
    write_jsonl(args.out_dir / "joined_holdout_decisions.jsonl", joined)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "clean_top1": f"{clean}/16",
            "hard_exclusion": f"{exclusions}/16",
            "gate": gate,
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
