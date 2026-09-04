#!/usr/bin/env python3
"""Run the outcome-blind G2 eligibility and retrieval control suite.

The 42 positive cases and 12 negative/boundary controls are synthetic contract
tests. They do not count as the required real, independent qualification
states and cannot promote BGE or the Bank to formal RS.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from metacom_pm.text import lexical_score


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g2-controlled-retrieval-v1"
DEFAULT_MODEL = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
MOVE_TEXTS = {
    "AM01_invite_open_expression": [
        "There is something bothering me, but I do not know where to start.",
        "I want to talk, though I am not sure how to explain it.",
        "A lot has happened and I need somewhere to begin.",
        "I feel off and want to get something out.",
        "It is complicated. Can I talk through it?",
        "I need someone to listen, but the concern is hard to put into words.",
        "Something is wrong and I have not said what it is yet.",
    ],
    "AM02_ask_one_focused_clarification": [
        "I need to decide today, but the timing is the part I have not explained.",
        "The situation changed after one event, though I have not said when.",
        "I am unsure what to do; one missing detail is who is involved.",
        "I keep saying work is difficult without explaining which task.",
        "The plan depends on whether this happens once or every week.",
        "I want help making sense of it, but my immediate goal is still unclear.",
        "There are two options and the key difference has not been stated.",
    ],
    "AM04_tentative_paraphrase_check": [
        "I am torn: leaving would bring relief, but I would also feel guilty.",
        "I want support, yet I am afraid advice will make me feel pushed.",
        "Part of me wants to try again and part of me expects another failure.",
        "The problem is not the workload alone; it is feeling unnoticed.",
        "I am angry about what happened but mostly disappointed in myself.",
        "I said I am fine, but what I mean is that I can keep functioning.",
        "I want distance without ending the relationship.",
    ],
    "AM05_grounded_validation": [
        "I have been exhausted and discouraged after repeated rejections.",
        "It hurts that my friend dismissed something important to me.",
        "I feel overwhelmed by everything that has piled up this week.",
        "I am embarrassed that this simple task has become so difficult.",
        "I feel lonely even when other people are around.",
        "I am frustrated because I tried carefully and it still went wrong.",
        "I am scared about the uncertainty and cannot settle down.",
    ],
    "AM10_offer_one_optional_micro_step": [
        "Could you give me one small thing I can try today?",
        "I would like advice, but only one easy first step.",
        "What is one low-risk action I could take next?",
        "Please suggest a small step that I can decline if it does not fit.",
        "I want practical help without a long plan.",
        "Can you help me start with one manageable action?",
        "Advice is welcome; I only have energy for one tiny step.",
    ],
    "AM14_supportive_transition": [
        "I am getting tired and would like to wrap this up.",
        "I think I am done talking about this for now.",
        "Can we change the subject?",
        "I have said what I needed and want to close.",
        "I would rather pause and continue another time.",
        "This topic is enough for today; I want to move on.",
        "Please end the conversation now.",
    ],
}


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


def _encode(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            chunks.append(hidden.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _eligibility(flags: dict[str, bool]) -> list[str]:
    if flags.get("active_high_stakes"):
        return []
    if flags.get("explicit_stop"):
        return ["AM14_supportive_transition"]
    eligible: list[str] = []
    repeated = flags.get("question_repetition_block", False)
    if flags.get("open_expression_opportunity") and not repeated:
        eligible.append("AM01_invite_open_expression")
    if (
        flags.get("focused_clarification_opportunity")
        and not repeated
        and not flags.get("listen_only")
    ):
        eligible.append("AM02_ask_one_focused_clarification")
    if flags.get("paraphrase_check_opportunity"):
        eligible.append("AM04_tentative_paraphrase_check")
    if flags.get("grounded_validation_opportunity"):
        eligible.append("AM05_grounded_validation")
    if (
        flags.get("explicit_advice_welcome")
        and flags.get("one_low_risk_step_available")
        and not flags.get("listen_only")
    ):
        eligible.append("AM10_offer_one_optional_micro_step")
    if flags.get("transition_opportunity"):
        eligible.append("AM14_supportive_transition")
    return eligible


def _query(case: dict[str, Any]) -> str:
    cues = [
        key.replace("_", " ")
        for key, value in case["observable_flags"].items()
        if value
        and key
        not in {
            "active_high_stakes",
            "question_repetition_block",
        }
    ]
    return (
        f"Latest visible user text: {case['visible_user_text']}\n"
        f"Observable interaction cues: {', '.join(cues) or 'none'}."
    )


def _positive_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for move_id, texts in MOVE_TEXTS.items():
        for index, text in enumerate(texts, start=1):
            flags = {
                "active_high_stakes": False,
                "explicit_stop": False,
                "listen_only": False,
                "question_repetition_block": False,
                "open_expression_opportunity": False,
                "focused_clarification_opportunity": False,
                "paraphrase_check_opportunity": False,
                "grounded_validation_opportunity": False,
                "explicit_advice_welcome": False,
                "one_low_risk_step_available": False,
                "transition_opportunity": False,
            }
            if move_id == "AM01_invite_open_expression":
                flags["open_expression_opportunity"] = True
                flags["grounded_validation_opportunity"] = index % 2 == 0
            elif move_id == "AM02_ask_one_focused_clarification":
                flags["focused_clarification_opportunity"] = True
                flags["paraphrase_check_opportunity"] = index % 2 == 0
                flags["grounded_validation_opportunity"] = index % 3 == 0
            elif move_id == "AM04_tentative_paraphrase_check":
                flags["paraphrase_check_opportunity"] = True
                flags["grounded_validation_opportunity"] = True
            elif move_id == "AM05_grounded_validation":
                flags["grounded_validation_opportunity"] = True
                flags["paraphrase_check_opportunity"] = index % 3 == 0
            elif move_id == "AM10_offer_one_optional_micro_step":
                flags["explicit_advice_welcome"] = True
                flags["one_low_risk_step_available"] = True
                flags["grounded_validation_opportunity"] = index % 2 == 0
            elif move_id == "AM14_supportive_transition":
                if index == len(texts):
                    flags["explicit_stop"] = True
                else:
                    flags["transition_opportunity"] = True
                    flags["grounded_validation_opportunity"] = index % 2 == 0
            cases.append(
                {
                    "case_id": f"g2_control_{move_id[:4].lower()}_{index:02d}",
                    "case_type": "synthetic_positive_control",
                    "visible_user_text": text,
                    "observable_flags": flags,
                    "expected_top1_move_id": move_id,
                }
            )
    return cases


def _negative_cases() -> list[dict[str, Any]]:
    rows = [
        (
            "I want to die and I may hurt myself tonight.",
            {"active_high_stakes": True, "explicit_advice_welcome": True},
            None,
        ),
        (
            "My partner threatened to hit me and I am in immediate danger.",
            {"active_high_stakes": True, "transition_opportunity": True},
            None,
        ),
        (
            "Please stop and end this conversation.",
            {"explicit_stop": True, "open_expression_opportunity": True},
            "AM14_supportive_transition",
        ),
        (
            "I cannot chat any more. Close this now.",
            {"explicit_stop": True, "explicit_advice_welcome": True},
            "AM14_supportive_transition",
        ),
        (
            "You already asked me to explain; please let me answer.",
            {
                "open_expression_opportunity": True,
                "question_repetition_block": True,
            },
            None,
        ),
        (
            "That question was already answered in my last message.",
            {
                "focused_clarification_opportunity": True,
                "question_repetition_block": True,
            },
            None,
        ),
        (
            "Please just listen. I do not want advice.",
            {
                "listen_only": True,
                "explicit_advice_welcome": False,
                "one_low_risk_step_available": True,
                "grounded_validation_opportunity": True,
            },
            "AM05_grounded_validation",
        ),
        (
            "I am not asking what to do; I only want to say this is painful.",
            {
                "listen_only": True,
                "one_low_risk_step_available": True,
                "grounded_validation_opportunity": True,
            },
            "AM05_grounded_validation",
        ),
        ("Hello, nice weather today.", {}, None),
        ("Thanks.", {}, None),
        (
            "Give me medical instructions for changing my medication dose.",
            {"active_high_stakes": True, "explicit_advice_welcome": True},
            None,
        ),
        (
            "I need legal advice before an eviction tomorrow.",
            {"active_high_stakes": True, "explicit_advice_welcome": True},
            None,
        ),
    ]
    keys = [
        "active_high_stakes",
        "explicit_stop",
        "listen_only",
        "question_repetition_block",
        "open_expression_opportunity",
        "focused_clarification_opportunity",
        "paraphrase_check_opportunity",
        "grounded_validation_opportunity",
        "explicit_advice_welcome",
        "one_low_risk_step_available",
        "transition_opportunity",
    ]
    cases = []
    for index, (text, partial, expected) in enumerate(rows, start=1):
        flags = {key: bool(partial.get(key, False)) for key in keys}
        cases.append(
            {
                "case_id": f"g2_negative_{index:02d}",
                "case_type": "synthetic_boundary_or_negative_control",
                "visible_user_text": text,
                "observable_flags": flags,
                "expected_top1_move_id": expected,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_g1_final_bank_v1"
        / "strategy_cards_g1_runtime.jsonl",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_g2_controlled_retrieval_v1",
    )
    args = parser.parse_args()

    cards = _read_jsonl(args.bank)
    if len(cards) != 6:
        raise ValueError("G2 control suite requires the frozen six-card Bank")
    card_by_move = {str(row["strategy_label"]): row for row in cards}
    cases = _positive_cases() + _negative_cases()
    if len(cases) != 54:
        raise ValueError("expected 42 positive plus 12 negative controls")
    for case in cases:
        case["eligible_move_ids"] = _eligibility(case["observable_flags"])
        expected = case["expected_top1_move_id"]
        if expected is not None and expected not in case["eligible_move_ids"]:
            raise ValueError(f"{case['case_id']}: expected move is ineligible")
        case["query_text"] = _query(case)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(device)
    card_texts = [str(row["retrieval_text"]) for row in cards]
    query_texts = [str(row["query_text"]) for row in cases]
    vectors = _encode(
        card_texts + query_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    card_vectors = vectors[: len(cards)]
    query_vectors = vectors[len(cards) :]

    results: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        eligible = [
            card_by_move[move_id] for move_id in case["eligible_move_ids"]
        ]
        lexical_ranked = sorted(
            (
                (
                    lexical_score(case["query_text"], card["retrieval_text"]),
                    str(card["strategy_label"]),
                )
                for card in eligible
            ),
            reverse=True,
        )
        semantic_ranked = sorted(
            (
                (
                    float(
                        query_vectors[case_index]
                        @ card_vectors[
                            next(
                                index
                                for index, row in enumerate(cards)
                                if row["strategy_label"]
                                == card["strategy_label"]
                            )
                        ]
                    ),
                    str(card["strategy_label"]),
                )
                for card in eligible
            ),
            reverse=True,
        )
        lexical_top1 = lexical_ranked[0][1] if lexical_ranked else None
        bge_top1 = semantic_ranked[0][1] if semantic_ranked else None
        expected = case["expected_top1_move_id"]
        results.append(
            {
                **case,
                "lexical_ranking": [
                    {"move_id": move_id, "score": round(score, 8)}
                    for score, move_id in lexical_ranked
                ],
                "bge_ranking": [
                    {"move_id": move_id, "cosine": round(score, 8)}
                    for score, move_id in semantic_ranked
                ],
                "lexical_top1": lexical_top1,
                "bge_top1": bge_top1,
                "lexical_expected_top1_pass": lexical_top1 == expected,
                "bge_expected_top1_pass": bge_top1 == expected,
            }
        )

    positive = [
        row for row in results if row["case_type"] == "synthetic_positive_control"
    ]
    negative = [
        row
        for row in results
        if row["case_type"] == "synthetic_boundary_or_negative_control"
    ]
    per_move = {}
    for move_id in MOVE_TEXTS:
        rows = [
            row
            for row in positive
            if row["expected_top1_move_id"] == move_id
        ]
        per_move[move_id] = {
            "cases": len(rows),
            "lexical_top1_passes": sum(
                row["lexical_expected_top1_pass"] for row in rows
            ),
            "bge_top1_passes": sum(
                row["bge_expected_top1_pass"] for row in rows
            ),
        }
    report = {
        "protocol": PROTOCOL,
        "status": "CONTROL_SUITE_COMPLETE_REAL_G2_QUALIFICATION_NOT_STARTED",
        "bank_card_count": len(cards),
        "synthetic_positive_cases": len(positive),
        "synthetic_boundary_or_negative_controls": len(negative),
        "eligibility_expected_set_passes": sum(
            _eligibility(row["observable_flags"]) == row["eligible_move_ids"]
            for row in results
        ),
        "eligibility_expected_set_denominator": len(results),
        "lexical_positive_top1_passes": sum(
            row["lexical_expected_top1_pass"] for row in positive
        ),
        "bge_positive_top1_passes": sum(
            row["bge_expected_top1_pass"] for row in positive
        ),
        "positive_denominator": len(positive),
        "lexical_control_top1_passes": sum(
            row["lexical_expected_top1_pass"] for row in negative
        ),
        "bge_control_top1_passes": sum(
            row["bge_expected_top1_pass"] for row in negative
        ),
        "control_denominator": len(negative),
        "per_move": per_move,
        "bge_model": "BAAI/bge-small-en-v1.5",
        "bge_revision": args.model.name,
        "bge_pooling": "cls",
        "bge_normalized": True,
        "bge_score": "normalized_dot_product_equal_to_cosine",
        "lexical_score": "existing_term_frequency_cosine",
        "query_includes_expected_move_id": False,
        "external_or_test_outcomes_used": False,
        "synthetic_cases_count_toward_real_qualification_n": False,
        "promotion_authorized": False,
        "next_gate": (
            "Build real ESConv-train, Bank-source-disjoint query states and "
            "run the same frozen eligibility/ranking stack; do not request "
            "additional G1 human review."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "controlled_cases.jsonl", cases)
    _write_jsonl(args.out_dir / "controlled_results.jsonl", results)
    _write_json(args.out_dir / "controlled_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
