#!/usr/bin/env python3
"""Build an outcome-blind audited RS opportunity rule baseline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.contracts import StrategyCard
from metacom_pm.io import (
    iter_jsonl,
    sha256_file,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_strategy_rag_runtime import QualifiedStrategyRAG


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-outcome-blind-opportunity-audit-v1"
SEED = "pm-v1.5-rs-opportunity-audit-selection-v1"
POSITIVE_PER_MOVE = 8


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _development_dialogues(root: Path) -> tuple[set[str], list[str]]:
    paths = sorted(
        (root / "outputs").glob(
            "pm_v1_5_strategy_rag_v4_direct_effect_v*/selected_states.jsonl"
        )
    )
    paths.extend(
        sorted(
            (root / "outputs").glob(
                "pm_v1_5_rs_six_card_clean_pair_v*/selected_states.jsonl"
            )
        )
    )
    paths.extend(
        [
            root
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_candidate"
            / "private_selection_audit.jsonl",
            root
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v2_candidate"
            / "private_selection_audit.jsonl",
            root
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v3_final_candidate"
            / "private_selection_audit.jsonl",
        ]
    )
    dialogues: set[str] = set()
    used: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        used.append(str(path.relative_to(root)))
        for row in _rows(path):
            value = row.get("source_dialogue_id") or row.get("user_id")
            if value:
                dialogues.add(str(value))
    return dialogues, used


def _one_per_dialogue(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dialogue[str(row["source_dialogue_id"])].append(row)
    chosen = []
    for dialogue_id, candidates in by_dialogue.items():
        candidates.sort(
            key=lambda row: stable_hex(
                SEED,
                role,
                dialogue_id,
                row["source_turn_index"],
                n=32,
            )
        )
        chosen.append(candidates[0])
    return chosen


def build(root: Path = ROOT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    universe_path = (
        root
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
        / "clean_train_strategy_universe.jsonl"
    )
    bank_path = root / "data/strategy/strategy_cards_v1_5_minimal.jsonl"
    accepted_path = (
        root
        / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1"
        / "combined_private_weak_sources.jsonl"
    )
    formal_test_path = (
        root
        / "data/esconv_test_v1_5_visible_v2_candidate/runtime_states.jsonl"
    )
    cards = [StrategyCard.model_validate(row) for row in _rows(bank_path)]
    rag = QualifiedStrategyRAG(cards)
    accepted = {
        str(row["source_dialogue_id"]) for row in _rows(accepted_path)
    }
    formal = {str(row["user_id"]) for row in _rows(formal_test_path)}
    development, development_paths = _development_dialogues(root)
    # This artifact audits the already frozen transparent rule; it does not
    # fit a learned opportunity model.  Replaying prior development dialogue
    # is therefore allowed and explicitly counted.  A future learned
    # opportunity head must use a fresh split.
    blocked = accepted | formal

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    invalid_last_turn = 0
    for source in _rows(universe_path):
        dialogue_id = str(source["source_dialogue_id"])
        if dialogue_id in blocked:
            continue
        dialogue = list(source.get("recent_dialogue") or [])
        if not dialogue or str(dialogue[-1].get("speaker")) != "seeker":
            invalid_last_turn += 1
            continue
        decision = rag.retrieve(dialogue)
        selected_move = (
            decision.selected_cards[0].strategy_label
            if decision.selected_cards
            else None
        )
        row = {
            "protocol": PROTOCOL,
            "source_dialogue_id": dialogue_id,
            "source_turn_index": int(source["source_turn_index"]),
            "opportunity_y": int(selected_move is not None),
            "opportunity_status": str(decision.status),
            "selected_move_id": selected_move,
            "eligible_move_ids": list(decision.eligible_move_ids),
            "observable_flags": dict(decision.observable_flags),
            "current_user_text": str(dialogue[-1].get("content") or ""),
            "recent_dialogue": dialogue,
            "selection_uses_hidden_next_response": False,
            "selection_uses_native_strategy_label": False,
            "selection_uses_response_outcome": False,
        }
        (positives if selected_move is not None else negatives).append(row)

    positive_unique = _one_per_dialogue(positives, "positive")
    by_move: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positive_unique:
        by_move[str(row["selected_move_id"])].append(row)
    positive_selected: list[dict[str, Any]] = []
    for move in sorted(by_move):
        ranked = sorted(
            by_move[move],
            key=lambda row: stable_hex(
                SEED,
                "positive",
                move,
                row["source_dialogue_id"],
                row["source_turn_index"],
                n=32,
            ),
        )
        positive_selected.extend(ranked[:POSITIVE_PER_MOVE])

    positive_dialogues = {
        str(row["source_dialogue_id"]) for row in positive_selected
    }
    negative_unique = _one_per_dialogue(
        [
            row
            for row in negatives
            if str(row["source_dialogue_id"]) not in positive_dialogues
        ],
        "negative",
    )
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in negative_unique:
        by_status[str(row["opportunity_status"])].append(row)
    for status in by_status:
        by_status[status].sort(
            key=lambda row: stable_hex(
                SEED,
                "negative",
                status,
                row["source_dialogue_id"],
                row["source_turn_index"],
                n=32,
            )
        )
    negative_target = len(positive_selected)
    negative_selected: list[dict[str, Any]] = []
    statuses = sorted(by_status)
    while len(negative_selected) < negative_target:
        progressed = False
        for status in statuses:
            if by_status[status]:
                negative_selected.append(by_status[status].pop(0))
                progressed = True
                if len(negative_selected) == negative_target:
                    break
        if not progressed:
            break

    selected = positive_selected + negative_selected
    selected.sort(
        key=lambda row: (
            row["opportunity_y"],
            row["selected_move_id"] or "",
            row["opportunity_status"],
            row["source_dialogue_id"],
        )
    )
    groups = {row["source_dialogue_id"] for row in selected}
    prior_development_overlap = len(groups & development)
    positive_counts = Counter(
        str(row["selected_move_id"])
        for row in selected
        if row["opportunity_y"] == 1
    )
    negative_counts = Counter(
        str(row["opportunity_status"])
        for row in selected
        if row["opportunity_y"] == 0
    )
    five_active_moves = {
        "AM01_invite_open_expression",
        "AM02_ask_one_focused_clarification",
        "AM04_tentative_paraphrase_check",
        "AM05_grounded_validation",
        "AM10_offer_one_optional_micro_step",
    }
    passed = (
        60 <= len(selected) <= 100
        and len(groups) == len(selected)
        and sum(positive_counts.values()) == sum(negative_counts.values())
        and set(positive_counts) == five_active_moves
        and min(positive_counts.values()) >= 6
        and not (groups & blocked)
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "AUDITED_TRANSPARENT_OPPORTUNITY_BASELINE_READY"
            if passed
            else "BLOCKED_INSUFFICIENT_OUTCOME_BLIND_COVERAGE"
        ),
        "role": (
            "Transparent outcome-blind Step1 opportunity baseline; not a "
            "realized-benefit label, not a learned opportunity dataset, and "
            "not a claim of latent-need diagnosis."
        ),
        "independent_dialogue_groups": len(groups),
        "class_counts": {
            "no_opportunity": sum(negative_counts.values()),
            "opportunity": sum(positive_counts.values()),
        },
        "positive_by_active_move": dict(sorted(positive_counts.items())),
        "negative_by_runtime_status": dict(sorted(negative_counts.items())),
        "invalid_last_turn_rows_excluded": invalid_last_turn,
        "selection": {
            "one_state_per_dialogue": True,
            "uses_hidden_next_supporter_response": False,
            "uses_native_strategy_label": False,
            "uses_response_or_judge_outcome": False,
            "accepted_g1_sources_excluded": len(accepted),
            "prior_rs_development_dialogues_available_for_rule_replay": len(
                development
            ),
            "selected_prior_development_overlap": prior_development_overlap,
            "formal_esconv_test_dialogues_excluded": len(formal),
            "prior_development_paths": development_paths,
        },
        "interpretation": {
            "may_support": [
                "audited deterministic opportunity gate",
                "transparent rule baseline for a future learned opportunity head",
                "end-to-end negative controls such as routine no-opportunity states"
            ],
            "may_not_support": [
                "RS realized benefit",
                "human need gold",
                "an independently fitted learned opportunity head",
                "PM quality-effect training labels",
                "external efficacy"
            ],
        },
        "checks": {
            "sixty_to_one_hundred_unique_groups": (
                60 <= len(selected) <= 100 and len(groups) == len(selected)
            ),
            "balanced_binary_classes": (
                sum(positive_counts.values()) == sum(negative_counts.values())
            ),
            "all_active_moves_with_at_least_six_groups": (
                set(positive_counts) == five_active_moves
                and all(
                    count >= 6
                    for count in positive_counts.values()
                )
            ),
            "zero_blocked_dialogue_overlap": not (groups & blocked),
        },
        "lineage": {
            "universe_sha256": sha256_file(universe_path),
            "bank_sha256": sha256_file(bank_path),
            "accepted_g1_sources_sha256": sha256_file(accepted_path),
            "formal_test_states_sha256": sha256_file(formal_test_path),
        },
    }
    return report, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_opportunity_layer_v1",
    )
    args = parser.parse_args()
    report, rows = build()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "opportunity_report.json", report)
    write_jsonl(args.out_dir / "opportunity_states.jsonl", rows)
    print(report)


if __name__ == "__main__":
    main()
