#!/usr/bin/env python3
"""Build a bounded G1 source-candidate pool from all 9,148 ESConv train turns.

BGE and TF-IDF are used only for candidate recall. Scores are not move labels,
card-quality judgments, PM features, or evidence that a card is useful.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g1-full-universe-candidate-pool-v1"
DEFAULT_MODEL = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
_HIGH_STAKES_RE = re.compile(
    r"\b(?:suicid|kill|self[- ]?harm|abuse|violent|domestic violence|"
    r"medication|dose|diagnos|vaccine|lawyer|legal|evict|homeless|"
    r"investment|loan|police|emergency|pregnan|miscarriage)\w*\b",
    flags=re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    *,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
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


def _candidate_id(row: dict[str, Any]) -> str:
    source = (
        f"{PROTOCOL}|{row['source_dialogue_id']}|"
        f"{row['source_turn_index']}|{row['strategy_id']}"
    )
    return "strategy_g1_" + hashlib.sha256(source.encode()).hexdigest()[:20]


def _source_text(row: dict[str, Any]) -> str:
    recent = list(row.get("recent_dialogue") or [])
    last_seeker = next(
        (
            str(turn.get("content") or "")
            for turn in reversed(recent)
            if str(turn.get("speaker") or "") == "seeker"
        ),
        "",
    )
    return (
        f"Latest user context: {last_seeker}\n"
        f"Supporter response: {row['supporter_response']}"
    )


def _move_text(move: dict[str, Any]) -> str:
    return (
        f"Support move: {move['name']}. "
        f"Definition: {move['definition']}. "
        f"Execution constraints: grounded, bounded, topic-agnostic."
    )


def _top_distinct_dialogues(
    scores: np.ndarray,
    rows: list[dict[str, Any]],
    count: int,
) -> list[int]:
    selected: list[int] = []
    seen_dialogues: set[str] = set()
    for index in np.argsort(-scores):
        source_index = int(index)
        dialogue_id = str(rows[source_index]["source_dialogue_id"])
        if dialogue_id in seen_dialogues:
            continue
        selected.append(source_index)
        seen_dialogues.add(dialogue_id)
        if len(selected) == count:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
        / "clean_train_strategy_universe.jsonl",
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=ROOT
        / "data/strategy"
        / "pm_v1_5_strategy_atomic_move_codebook_v1.json",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--semantic-per-move", type=int, default=120)
    parser.add_argument("--lexical-per-move", type=int, default=40)
    parser.add_argument("--negative-controls-per-move", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_g1_candidate_pool_v1",
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.universe)
    if len(rows) != 9148:
        raise ValueError(f"expected 9148 clean train rows, found {len(rows)}")
    source_keys = [
        (str(row["source_dialogue_id"]), int(row["source_turn_index"]))
        for row in rows
    ]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("clean universe repeats a dialogue-turn source key")
    if len({key[0] for key in source_keys}) != 748:
        raise ValueError("expected 748 independent train source dialogues")

    codebook = _read_json(args.codebook)
    moves = list(codebook["moves"])
    if len(moves) != 17:
        raise ValueError("G1 requires the frozen 17-move G0 codebook")
    source_texts = [_source_text(row) for row in rows]
    move_texts = [_move_text(move) for move in moves]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(device)
    source_embeddings = _encode(
        texts=source_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    move_embeddings = _encode(
        texts=move_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    semantic_scores = source_embeddings @ move_embeddings.T

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(source_texts + move_texts)
    lexical_scores = (tfidf[: len(rows)] @ tfidf[len(rows) :].T).toarray()

    proposals: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    negative_controls: dict[int, set[str]] = defaultdict(set)
    per_move_selection: dict[str, dict[str, Any]] = {}
    for move_index, move in enumerate(moves):
        move_id = str(move["move_id"])
        semantic_indices = _top_distinct_dialogues(
            semantic_scores[:, move_index],
            rows,
            args.semantic_per_move,
        )
        lexical_indices = _top_distinct_dialogues(
            lexical_scores[:, move_index],
            rows,
            args.lexical_per_move,
        )
        for rank, source_index in enumerate(semantic_indices, start=1):
            proposals[source_index].setdefault(
                move_id,
                {
                    "move_id": move_id,
                    "semantic_rank": None,
                    "semantic_score": None,
                    "lexical_rank": None,
                    "lexical_score": None,
                },
            )
            proposals[source_index][move_id].update(
                {
                    "semantic_rank": rank,
                    "semantic_score": round(
                        float(semantic_scores[source_index, move_index]), 8
                    ),
                }
            )
        for rank, source_index in enumerate(lexical_indices, start=1):
            proposals[source_index].setdefault(
                move_id,
                {
                    "move_id": move_id,
                    "semantic_rank": None,
                    "semantic_score": None,
                    "lexical_rank": None,
                    "lexical_score": None,
                },
            )
            proposals[source_index][move_id].update(
                {
                    "lexical_rank": rank,
                    "lexical_score": round(
                        float(lexical_scores[source_index, move_index]), 8
                    ),
                }
            )
        per_move_selection[move_id] = {
            "semantic_distinct_dialogues": len(semantic_indices),
            "lexical_distinct_dialogues": len(lexical_indices),
            "union_source_rows": len(set(semantic_indices) | set(lexical_indices)),
        }
        seen_control_dialogues: set[str] = set()
        for index in np.argsort(semantic_scores[:, move_index]):
            source_index = int(index)
            source = rows[source_index]
            dialogue_id = str(source["source_dialogue_id"])
            if dialogue_id in seen_control_dialogues:
                continue
            if bool(source["first_person_language_flag"]):
                continue
            if bool(source["domain_claim_keyword_flag"]):
                continue
            if _HIGH_STAKES_RE.search(source_texts[source_index]):
                continue
            negative_controls[source_index].add(move_id)
            seen_control_dialogues.add(dialogue_id)
            if len(seen_control_dialogues) == args.negative_controls_per_move:
                break
        per_move_selection[move_id]["low_score_negative_controls"] = len(
            seen_control_dialogues
        )

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for source_index in sorted(set(proposals) | set(negative_controls)):
        row = rows[source_index]
        item_id = _candidate_id(row)
        proposal_rows = sorted(
            proposals[source_index].values(),
            key=lambda value: str(value["move_id"]),
        )
        visible_context = {
            "blind_item_id": item_id,
            "recent_visible_dialogue": row["recent_dialogue"],
            "supporter_response_to_label": row["supporter_response"],
        }
        public_rows.append(visible_context)
        combined_text = " ".join(
            [
                str(row["supporter_response"]),
                *[
                    str(turn.get("content") or "")
                    for turn in row.get("recent_dialogue") or []
                ],
            ]
        )
        private_rows.append(
            {
                "blind_item_id": item_id,
                "source_dialogue_id": row["source_dialogue_id"],
                "source_turn_index": row["source_turn_index"],
                "strategy_id": row["strategy_id"],
                "native_strategy_label_audit_only": row["strategy_label"],
                "problem_type_audit_only": row["problem_type"],
                "emotion_type_audit_only": row["emotion_type"],
                "experience_type_audit_only": row["experience_type"],
                "first_person_language_flag": bool(
                    row["first_person_language_flag"]
                ),
                "domain_claim_keyword_flag": bool(
                    row["domain_claim_keyword_flag"]
                ),
                "high_stakes_context_review_flag": bool(
                    _HIGH_STAKES_RE.search(combined_text)
                ),
                "candidate_move_proposals": proposal_rows,
                "low_score_control_for_move_ids": sorted(
                    negative_controls.get(source_index, set())
                ),
            }
        )

    selection_count_by_move = Counter(
        proposal["move_id"]
        for row in private_rows
        for proposal in row["candidate_move_proposals"]
    )
    report = {
        "protocol": PROTOCOL,
        "status": "G1_CANDIDATE_POOL_READY_WEAK_LABEL_AND_HUMAN_SOURCE_REVIEW_PENDING",
        "full_train_universe_rows_scored": len(rows),
        "full_train_universe_dialogues": len(set(key[0] for key in source_keys)),
        "frozen_atomic_moves": len(moves),
        "semantic_candidates_per_move": args.semantic_per_move,
        "lexical_candidates_per_move": args.lexical_per_move,
        "candidate_union_rows": len(public_rows),
        "candidate_union_dialogues": len(
            {str(row["source_dialogue_id"]) for row in private_rows}
        ),
        "candidate_move_memberships": sum(selection_count_by_move.values()),
        "low_score_control_memberships": sum(
            len(row["low_score_control_for_move_ids"]) for row in private_rows
        ),
        "rows_used_as_any_low_score_control": sum(
            bool(row["low_score_control_for_move_ids"]) for row in private_rows
        ),
        "selection_count_by_move": dict(sorted(selection_count_by_move.items())),
        "per_move_selection": per_move_selection,
        "high_stakes_review_rows": sum(
            row["high_stakes_context_review_flag"] for row in private_rows
        ),
        "first_person_review_rows": sum(
            row["first_person_language_flag"] for row in private_rows
        ),
        "domain_claim_review_rows": sum(
            row["domain_claim_keyword_flag"] for row in private_rows
        ),
        "native_labels_visible_to_weak_labeler": False,
        "problem_and_emotion_labels_visible_to_weak_labeler": False,
        "ranking_role": "candidate_recall_only",
        "automatic_move_label_or_card_promotion_authorized": False,
        "next_step": (
            "Use qualified Coder B for weak multi-label proposals on this bounded "
            "public packet, include fixed low-score controls, then human-audit "
            "literal sources for moves that may reach the 20-dialogue source gate."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "weak_label_public_packet.jsonl", public_rows)
    _write_jsonl(args.out_dir / "private_candidate_lineage.jsonl", private_rows)
    _write_json(args.out_dir / "candidate_pool_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
