#!/usr/bin/env python3
"""Bind two G1 reviews and freeze the minimal six-card G2 development Bank.

The two narrow exceptions rejected by both reviewers are removed. For the
remaining six cards, human agreement on the core move definition is kept
separate from the failed weak-source qualification claim. Source responses
are never compiled into the runtime Bank.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g1-dual-human-bank-freeze-v1"
DEFAULT_REVIEW_1 = Path(
    "/home/tokkio/.codex/attachments/"
    "2cf0727b-a240-4dd9-9fb0-3cb100150736/pasted-text.txt"
)
DEFAULT_REVIEW_2 = Path(
    "/home/tokkio/.codex/attachments/"
    "f4d56f18-b4ab-4f44-9149-28d4d6e870ab/pasted-text.txt"
)
INCLUDED_MOVES = [
    "AM01_invite_open_expression",
    "AM02_ask_one_focused_clarification",
    "AM04_tentative_paraphrase_check",
    "AM05_grounded_validation",
    "AM10_offer_one_optional_micro_step",
    "AM14_supportive_transition",
]
REJECTED_MOVES = [
    "AM07_acknowledge_effort_strength_or_resource",
    "AM15_explore_interpersonal_boundary",
]
REVIEW_FIELDS = [
    "support_move_clear",
    "when_to_use_valid",
    "when_not_to_use_valid",
    "eligibility_metadata_valid",
    "burden_and_risk_valid",
    "safe_topic_agnostic_technique",
    "source_gate_or_exception_acceptable",
    "approve_for_g2_candidate",
]
CORE_FIELDS = REVIEW_FIELDS[:6]
HIGH_STAKES_RE = re.compile(
    r"\b(?:suicid\w*|self[- ]?harm\w*|kill(?:ing|ed)? "
    r"(?:myself|him|her|them)|want to die|feel like dying|violence|"
    r"violent|abuse\w*|aggressive|slapped?|hit me|hurt me|threat\w*|"
    r"emergency|crisis|homeless|evict\w*|can(?:not|'t) accept this "
    r"pain)\b",
    flags=re.IGNORECASE,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _by_move(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result = {str(row["move_id"]): row for row in rows}
    if len(result) != len(rows) or len(rows) != 8:
        raise ValueError(f"{label} requires exactly eight unique moves")
    return result


def _correct_card(card: dict[str, Any]) -> dict[str, Any]:
    row = dict(card)
    move_id = str(row["move_id"])
    global_rule = (
        "Do not use an ordinary technique card in place of responding to an "
        "active self-harm, violence, abuse, or other acute crisis cue; defer "
        "to the upstream safety flow."
    )
    corrections = {
        "AM01_invite_open_expression": (
            "Do not issue a second open invitation before the user has had a "
            "turn to respond. Do not repeat it after the concern is clear, "
            "stack it with other questions, or reopen after an explicit stop "
            "request."
        ),
        "AM02_ask_one_focused_clarification": (
            "Ask at most one missing detail. Do not stack questions, attribute "
            "an unsupported motive, repeat already answered information, or "
            "probe under a listen-only or explicit-stop boundary."
        ),
        "AM04_tentative_paraphrase_check": (
            "Do not introduce motives, diagnoses, facts, causal claims, or "
            "stronger conclusions than the user stated. Use one brief "
            "correction check only; do not append an additional exploratory "
            "question as part of this card."
        ),
        "AM05_grounded_validation": (
            "Do not invent or intensify emotion, diagnose, minimize, tell the "
            "user not to feel something, promise an outcome, infer an unstated "
            "goal, or validate a harmful factual claim as true."
        ),
        "AM10_offer_one_optional_micro_step": (
            "This card is ineligible unless the visible dialogue explicitly "
            "welcomes advice or action help. Offer exactly one concrete, "
            "low-risk, low-burden, goal-linked and easily declined step; do "
            "not stack tasks, impose a deadline, or use specialist guidance."
        ),
        "AM14_supportive_transition": (
            "If the user explicitly asks to stop or close, acknowledge and "
            "close without asking whether they want to continue. Do not close "
            "abruptly while they are still engaging or promise permanent "
            "availability."
        ),
    }
    row["when_not_to_use"] = corrections[move_id] + " " + global_rule
    if move_id == "AM10_offer_one_optional_micro_step":
        row["when_to_use"] = (
            "Use only when advice or action help is explicitly welcomed in "
            "the visible dialogue, the goal is clear, and one low-risk, "
            "low-burden step fits that goal."
        )
    if move_id == "AM14_supportive_transition":
        row["when_to_use"] = (
            "Use when the user signals fatigue, completion, a wish to change "
            "focus, or an explicit wish to stop. Under an explicit stop "
            "request, use the acknowledge-and-close variant only."
        )
    row["global_eligibility_rules_applied_before_ranking"] = True
    row["source_qualification_status"] = (
        "FORMATIVE_PROVENANCE_ONLY_WEAK_SOURCE_GATE_NOT_QUALIFIED"
    )
    row["human_definition_status"] = (
        "DUAL_REVIEW_DEFINITION_QUALIFIED_AFTER_EXPLICIT_CORRECTIONS"
    )
    row["eligible_for_g2_development"] = True
    row["eligible_for_formal_rs"] = False
    row["raw_source_responses_exposed_to_generator"] = False
    return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-1", type=Path, default=DEFAULT_REVIEW_1)
    parser.add_argument("--review-2", type=Path, default=DEFAULT_REVIEW_2)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_g1_final_bank_review_candidate_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_g1_final_bank_v1",
    )
    args = parser.parse_args()

    review_1_rows = _read_jsonl(args.review_1)
    review_2_rows = _read_jsonl(args.review_2)
    review_1 = _by_move(review_1_rows, label="review_1")
    review_2 = _by_move(review_2_rows, label="review_2")
    if set(review_1) != set(review_2):
        raise ValueError("review move sets differ")
    if any(
        row["protocol"] != "pm-v1.5-strategy-g1-final-bank-human-review-v1"
        for row in review_1_rows + review_2_rows
    ):
        raise ValueError("unexpected human review protocol")

    packet_rows = _read_jsonl(args.candidate_dir / "human_review_packet.jsonl")
    packet = _by_move(packet_rows, label="review_packet")
    cards = _by_move(
        _read_jsonl(args.candidate_dir / "strategy_cards_g1_candidate.jsonl"),
        label="candidate_cards",
    )
    if set(packet) != set(review_1) or set(cards) != set(review_1):
        raise ValueError("candidate/review move sets differ")
    for move_id in review_1:
        expected_ids = {
            str(row["blind_item_id"])
            for row in packet[move_id]["source_examples"]
        }
        for label, review in (
            ("review_1", review_1),
            ("review_2", review_2),
        ):
            actual_ids = {
                str(row["blind_item_id"])
                for row in review[move_id]["source_example_review"]
            }
            if expected_ids != actual_ids:
                raise ValueError(f"{label}/{move_id}: source example IDs drift")

    example_entries = [
        (str(example["blind_item_id"]), str(example["source_dialogue_id"]), move_id)
        for move_id, row in packet.items()
        for example in row["source_examples"]
    ]
    duplicate_items = {
        item_id: [
            {"source_dialogue_id": dialogue_id, "move_id": move_id}
            for candidate_id, dialogue_id, move_id in example_entries
            if candidate_id == item_id
        ]
        for item_id, count in Counter(row[0] for row in example_entries).items()
        if count > 1
    }
    dialogue_moves: dict[str, set[str]] = defaultdict(set)
    for _, dialogue_id, move_id in example_entries:
        dialogue_moves[dialogue_id].add(move_id)
    cross_card_dialogues = {
        dialogue_id: sorted(move_ids)
        for dialogue_id, move_ids in dialogue_moves.items()
        if len(move_ids) > 1
    }

    high_stakes_fixed_examples: list[dict[str, Any]] = []
    for move_id, row in packet.items():
        for example in row["source_examples"]:
            visible = "\n".join(
                str(turn["content"])
                for turn in example["recent_visible_dialogue"]
            )
            text = visible + "\n" + str(example["supporter_response_to_label"])
            hits = sorted(
                {
                    match.group(0).casefold()
                    for match in HIGH_STAKES_RE.finditer(text)
                }
            )
            if hits:
                high_stakes_fixed_examples.append(
                    {
                        "move_id": move_id,
                        "blind_item_id": example["blind_item_id"],
                        "source_dialogue_id": example["source_dialogue_id"],
                        "lexical_review_hits_not_automatic_clinical_labels": hits,
                    }
                )

    field_agreement = {
        field: {
            "exact_agreement_count": sum(
                review_1[move_id][field] == review_2[move_id][field]
                for move_id in review_1
            ),
            "denominator": 8,
        }
        for field in REVIEW_FIELDS
    }
    move_fit_agreements: list[bool] = []
    exclusion_agreements: list[bool] = []
    for move_id in review_1:
        first = {
            str(row["blind_item_id"]): row
            for row in review_1[move_id]["source_example_review"]
        }
        second = {
            str(row["blind_item_id"]): row
            for row in review_2[move_id]["source_example_review"]
        }
        for item_id in first:
            move_fit_agreements.append(
                first[item_id]["move_fit"] == second[item_id]["move_fit"]
            )
            exclusion_agreements.append(
                first[item_id]["missed_hard_exclusion"]
                == second[item_id]["missed_hard_exclusion"]
            )

    final_cards = [_correct_card(cards[move_id]) for move_id in INCLUDED_MOVES]
    runtime_cards: list[dict[str, Any]] = []
    for card in final_cards:
        move_id = str(card["move_id"])
        strategy_id = "strat_" + hashlib.sha256(
            f"{PROTOCOL}|{move_id}".encode()
        ).hexdigest()[:20]
        retrieval_text = (
            f"Support move: {card['name']}. "
            f"Use conditions: {card['when_to_use']} "
            f"Observable goals: {', '.join(card['goal_types'])}. "
            f"Do not use: {card['when_not_to_use']}"
        )
        guidance_text = (
            f"{card['support_move']} Use conditions: {card['when_to_use']} "
            f"Boundaries: {card['when_not_to_use']} Use only information "
            "visible in the current prompt. Compose new wording; never copy "
            "a source response or fabricate first-person experience."
        )
        runtime_cards.append(
            {
                "strategy_id": strategy_id,
                "strategy_label": move_id,
                "retrieval_text": retrieval_text,
                "guidance_text": guidance_text,
                "example_response": (
                    "No source response is provided. Compose a new response "
                    "from the visible dialogue and this guidance only."
                ),
                "source_dialogue_id": "g1_dual_human_definition_bank",
                "source_turn_index": 0,
            }
        )

    global_rules = {
        "protocol": PROTOCOL,
        "apply_before_ranking": True,
        "rules": [
            {
                "rule_id": "active_high_stakes_override",
                "if": (
                    "visible dialogue contains an active self-harm, violence, "
                    "abuse, or acute crisis cue not yet handled"
                ),
                "effect": (
                    "ordinary strategy Bank ineligible; use upstream safety "
                    "flow before any ordinary support move"
                ),
            },
            {
                "rule_id": "explicit_stop_boundary",
                "if": "user explicitly asks to stop, close, or end",
                "effect": (
                    "only AM14 acknowledge-and-close variant eligible; never "
                    "ask whether the user wants to continue"
                ),
            },
            {
                "rule_id": "advice_permission",
                "if": "advice or action help is not explicitly welcomed",
                "effect": "AM10 ineligible",
            },
            {
                "rule_id": "question_repetition",
                "if": (
                    "an open invitation or clarification was already asked "
                    "without an intervening user response, or the answer is "
                    "already visible"
                ),
                "effect": "AM01 and AM02 ineligible",
            },
            {
                "rule_id": "single_card_cap",
                "if": "more than one card remains eligible",
                "effect": "rank eligible cards and inject Top-1 only",
            },
        ],
        "raw_source_responses_available_to_runtime": False,
        "source_example_labels_used_as_PM_or_permission_gold": False,
    }

    per_move_decision = {}
    for move_id in review_1:
        per_move_decision[move_id] = {
            "reviewer_1_overall": review_1[move_id][
                "approve_for_g2_candidate"
            ],
            "reviewer_2_overall": review_2[move_id][
                "approve_for_g2_candidate"
            ],
            "reviewer_1_core_fields": {
                field: review_1[move_id][field] for field in CORE_FIELDS
            },
            "reviewer_2_core_fields": {
                field: review_2[move_id][field] for field in CORE_FIELDS
            },
            "final_g1_decision": (
                "INCLUDE_DEFINITION_AFTER_CORRECTIONS_SOURCE_NOT_QUALIFIED"
                if move_id in INCLUDED_MOVES
                else "REJECT_NARROW_EXCEPTION_NO_REPLACEMENT"
            ),
            "decision_basis": (
                "Both reviewers rejected the below-20-dialogue exception."
                if move_id in REJECTED_MOVES
                else (
                    "Core move/use/eligibility fields were human-supported; "
                    "overall approval was confounded by weak-source failures. "
                    "The card is retained only as a corrected human-definition "
                    "development resource, with source qualification withdrawn."
                )
            ),
        }

    report = {
        "protocol": PROTOCOL,
        "status": "G1_COMPLETE_SIX_CARD_BANK_FROZEN_FOR_G2_DEVELOPMENT",
        "review_1_sha256": _sha256(args.review_1),
        "review_2_sha256": _sha256(args.review_2),
        "review_row_count_each": 8,
        "reviewer_1_overall_counts": dict(
            Counter(
                row["approve_for_g2_candidate"] for row in review_1_rows
            )
        ),
        "reviewer_2_overall_counts": dict(
            Counter(
                row["approve_for_g2_candidate"] for row in review_2_rows
            )
        ),
        "field_agreement": field_agreement,
        "source_example_agreement": {
            "move_fit_exact": {
                "count": sum(move_fit_agreements),
                "denominator": len(move_fit_agreements),
                "rate": round(
                    sum(move_fit_agreements) / len(move_fit_agreements), 4
                ),
            },
            "missed_hard_exclusion_exact": {
                "count": sum(exclusion_agreements),
                "denominator": len(exclusion_agreements),
                "rate": round(
                    sum(exclusion_agreements) / len(exclusion_agreements), 4
                ),
            },
        },
        "source_packet_grain_correction": {
            "card_example_judgments": len(example_entries),
            "unique_blind_items": len({row[0] for row in example_entries}),
            "unique_source_dialogues": len({row[1] for row in example_entries}),
            "duplicate_blind_items": duplicate_items,
            "cross_card_source_dialogues": cross_card_dialogues,
        },
        "automated_high_stakes_rescan": {
            "role": "lexical_review_candidates_not_clinical_labels",
            "fixed_examples_flagged": len(high_stakes_fixed_examples),
            "examples": high_stakes_fixed_examples,
        },
        "source_qualification_conclusion": (
            "FAILED_FOR_PROMOTION_GOLD; weak-source counts and five-example "
            "sets are formative provenance only"
        ),
        "included_moves": INCLUDED_MOVES,
        "rejected_moves": REJECTED_MOVES,
        "final_card_count": len(final_cards),
        "runtime_source_response_count": 0,
        "eligible_for_formal_rs": False,
        "g1_human_review_complete": True,
        "additional_g1_human_review_authorized": False,
        "per_move_decision": per_move_decision,
        "next_gate": "G2 transparent eligibility and lexical-vs-BGE retrieval qualification",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "human_review_1_bound.jsonl", review_1_rows)
    _write_jsonl(args.out_dir / "human_review_2_bound.jsonl", review_2_rows)
    _write_jsonl(args.out_dir / "strategy_cards_g1_final.jsonl", final_cards)
    _write_jsonl(
        args.out_dir / "strategy_cards_g1_runtime.jsonl", runtime_cards
    )
    _write_json(args.out_dir / "global_eligibility_rules.json", global_rules)
    _write_json(args.out_dir / "g1_dual_human_freeze_report.json", report)
    print(json.dumps({"output": str(args.out_dir), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
