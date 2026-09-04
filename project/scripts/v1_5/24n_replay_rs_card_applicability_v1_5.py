#!/usr/bin/env python3
"""Replay the frozen 32-state development audit with card-level filtering.

The replay is an explicitly biased development diagnostic because the human
labels informed the transparent rules.  Its output must never be reported as
held-out retrieval accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_strategy_rag_v4 import (
    rank_applicable_v4_cards,
    validate_v4_candidate_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-card-applicability-development-replay-v1"
MINIMUM_SCORE = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joined-review",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_analysis"
            / "joined_review_decisions.jsonl"
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
        "--cards",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_strategy_rag_v4_candidate_v1"
            / "strategy_cards_v4_candidate.jsonl"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_card_applicability_development_replay_v1"
        ),
    )
    args = parser.parse_args()

    human = {
        str(row["review_item_id"]): dict(row)
        for row in iter_jsonl(args.joined_review)
    }
    states = {
        str(row["review_item_id"]): dict(row)
        for row in iter_jsonl(args.private_audit)
    }
    if set(human) != set(states) or len(human) != 32:
        raise RuntimeError("development replay requires the exact 32 reviewed states")
    cards = validate_v4_candidate_cards(
        [dict(row) for row in iter_jsonl(args.cards)]
    )
    active_cards = [
        card
        for card in cards
        if card["source_support"]["provisional_source_support_pass"]
    ]

    output: list[dict[str, Any]] = []
    for item_id in sorted(human):
        label = human[item_id]
        state = states[item_id]
        ranked = rank_applicable_v4_cards(
            query=str(state["natural_query"]),
            current_user_text=str(state["current_user_text"]),
            recent_user_text=" ".join(
                str(turn["content"])
                for turn in state["visible_dialogue"]
                if str(turn["speaker"]) == "seeker"
            ),
            flags=dict(state["observable_flags"]),
            cards=active_cards,
        )
        selected = (
            ranked[0]
            if ranked and float(ranked[0]["score"]) >= MINIMUM_SCORE
            else None
        )
        selected_id = str(selected["card_id"]) if selected else None
        preferred_id = label.get("preferred_card_id")
        original_id = str(label["top1_card_id"])
        if selected_id is None:
            relation = (
                "correct_abstention_signal"
                if label["better_candidate"] == "no_safe_card"
                else "unreviewed_abstention"
            )
        elif selected_id == preferred_id:
            relation = "matches_human_preferred_visible_candidate"
        elif selected_id == original_id and bool(label["clean_top1"]):
            relation = "retains_reviewed_clean_top1"
        else:
            relation = "new_unreviewed_top1"
        output.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": item_id,
                "state_id": str(state["state_id"]),
                "source_dialogue_id": str(state["source_dialogue_id"]),
                "stratum": str(state["stratum"]),
                "execution_profile": str(state["execution_profile"]),
                "human_clean_original_top1": bool(label["clean_top1"]),
                "human_hard_exclusion_original_top1": (
                    label["hard_exclusion_triggered"] == "yes"
                ),
                "human_better_candidate": str(label["better_candidate"]),
                "human_preferred_card_id": preferred_id,
                "original_top1_card_id": original_id,
                "replay_status": "retrieved_top1" if selected else "abstain",
                "replay_top1": selected,
                "replay_top3": ranked[:3],
                "relation_to_review": relation,
            }
        )

    relations = Counter(row["relation_to_review"] for row in output)
    core_counts = Counter(
        str(row["replay_top1"]["core_submove_id"])
        for row in output
        if row["replay_top1"]
    )
    report = {
        "protocol": PROTOCOL,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "DEVELOPMENT_REPLAY_COMPLETE_NOT_HELD_OUT_EVIDENCE",
        "review_items": len(output),
        "retrieved": sum(row["replay_top1"] is not None for row in output),
        "abstained": sum(row["replay_top1"] is None for row in output),
        "relation_counts": dict(sorted(relations.items())),
        "original_hard_exclusion_top1_retained": sum(
            row["human_hard_exclusion_original_top1"]
            and row["replay_top1"] is not None
            and row["replay_top1"]["card_id"] == row["original_top1_card_id"]
            for row in output
        ),
        "replay_top1_core_counts": dict(sorted(core_counts.items())),
        "diagnostic_only": True,
        "interpretation_rule": (
            "Matches to the human-preferred visible candidate are useful "
            "debug evidence. New Top1 cards remain unreviewed; this replay "
            "cannot establish post-repair accuracy."
        ),
        "data_firewall": {
            "response_generation_outcomes_used": False,
            "hidden_next_supporter_response_used": False,
            "formal_esconv_test_used": False,
            "evoemo_used": False,
        },
        "source_lineage": {
            "joined_review": str(args.joined_review.relative_to(ROOT)),
            "joined_review_sha256": sha256_file(args.joined_review),
            "private_audit": str(args.private_audit.relative_to(ROOT)),
            "private_audit_sha256": sha256_file(args.private_audit),
            "cards": str(args.cards.relative_to(ROOT)),
            "cards_sha256": sha256_file(args.cards),
        },
        "next_gate": (
            "Freeze the applicability rules and evaluate a newly selected "
            "zero-overlap 16-dialogue holdout. Do not tune on that holdout."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "development_replay_decisions.jsonl", output)
    write_json(args.out_dir / "development_replay_report.json", report)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "relations": dict(relations),
            "retrieved": report["retrieved"],
            "abstained": report["abstained"],
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
