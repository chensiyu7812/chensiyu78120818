#!/usr/bin/env python3
"""Diagnose BGE-small as a Top-3 Strategy-card reranker.

This is post-hoc diagnostic evidence after the expanded natural retriever
failed its frozen holdout.  It cannot promote a new V1.5 formal ranker or
justify another holdout.  It answers only whether BGE is promising enough to
keep for later work while the V1.5 training path proceeds with the qualified
six-card fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-bge-top3-reranking-posthoc-diagnostic-v1"
DEFAULT_MODEL = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _encode(
    texts: list[str],
    *,
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


def _preferred(
    label: dict[str, Any],
    top3: list[dict[str, Any]],
) -> tuple[str | None, int | None]:
    if label.get("clean_top1") or (
        label.get("top1_fit") == "yes"
        and label.get("hard_exclusion_triggered") == "no"
        and label.get("better_candidate") == "none"
    ):
        return str(top3[0]["card_id"]), 1
    better = str(label.get("better_candidate"))
    if better in {"rank2", "rank3"}:
        rank = int(better[-1])
        return str(top3[rank - 1]["card_id"]), rank
    return None, None


def _metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    preferred_rows = [row for row in rows if row["preferred_card_id"]]
    exact = sum(
        row[f"{variant}_top1_card_id"] == row["preferred_card_id"]
        for row in preferred_rows
    )
    reciprocal_rank = sum(
        1.0 / int(row[f"{variant}_preferred_rank"])
        for row in preferred_rows
    )
    hard_original_retained = sum(
        row["original_hard_exclusion"]
        and row[f"{variant}_top1_card_id"] == row["lexical_top1_card_id"]
        for row in rows
    )
    return {
        "items": len(rows),
        "items_with_human_preferred_candidate": len(preferred_rows),
        "exact_preferred_top1": exact,
        "exact_preferred_top1_rate": round(exact / len(preferred_rows), 6),
        "mean_reciprocal_rank": round(
            reciprocal_rank / len(preferred_rows), 6
        ),
        "original_hard_exclusion_top1_retained": hard_original_retained,
        "top1_core_counts": dict(
            sorted(Counter(row[f"{variant}_top1_core"] for row in rows).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=64)
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
        "--development-labels",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_analysis"
            / "joined_review_decisions.jsonl"
        ),
    )
    parser.add_argument(
        "--development-audit",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_candidate"
            / "private_selection_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--holdout-labels",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_fit_holdout_v3_analysis"
            / "joined_holdout_decisions.jsonl"
        ),
    )
    parser.add_argument(
        "--holdout-audit",
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
            ROOT / "outputs/pm_v1_5_rs_bge_top3_reranking_diagnostic_v1"
        ),
    )
    args = parser.parse_args()

    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    cards = {
        str(row["card_id"]): dict(row) for row in iter_jsonl(args.cards)
    }
    sources = []
    for split, label_path, audit_path in (
        (
            "development32",
            args.development_labels,
            args.development_audit,
        ),
        ("frozen_holdout16", args.holdout_labels, args.holdout_audit),
    ):
        labels = {
            str(row["review_item_id"]): dict(row)
            for row in iter_jsonl(label_path)
        }
        audits = {
            str(row["review_item_id"]): dict(row)
            for row in iter_jsonl(audit_path)
        }
        if set(labels) != set(audits):
            raise RuntimeError(f"{split} label/audit ids drift")
        for item_id in sorted(labels):
            top3 = [dict(row) for row in audits[item_id]["top3"]]
            preferred_id, preferred_lexical_rank = _preferred(
                labels[item_id], top3
            )
            sources.append(
                {
                    "split": split,
                    "review_item_id": item_id,
                    "natural_query": str(audits[item_id]["natural_query"]),
                    "top3": top3,
                    "preferred_card_id": preferred_id,
                    "preferred_lexical_rank": preferred_lexical_rank,
                    "lexical_top1_card_id": str(top3[0]["card_id"]),
                    "original_hard_exclusion": (
                        labels[item_id]["hard_exclusion_triggered"] == "yes"
                    ),
                }
            )

    card_ids = sorted(
        {
            str(candidate["card_id"])
            for row in sources
            for candidate in row["top3"]
        }
    )
    queries = [row["natural_query"] for row in sources]
    card_texts = [str(cards[card_id]["retrieval_text"]) for card_id in card_ids]
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True
    ).to(device)
    card_vectors = _encode(
        card_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    plain_query_vectors = _encode(
        queries,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    prefixed_query_vectors = _encode(
        [QUERY_PREFIX + query for query in queries],
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    card_index = {card_id: index for index, card_id in enumerate(card_ids)}

    decisions: list[dict[str, Any]] = []
    for row_index, source in enumerate(sources):
        candidate_ids = [
            str(candidate["card_id"]) for candidate in source["top3"]
        ]
        decision = dict(source)
        for variant, query_vector in (
            ("bge_plain", plain_query_vectors[row_index]),
            ("bge_prefixed", prefixed_query_vectors[row_index]),
        ):
            ranking = sorted(
                (
                    {
                        "card_id": card_id,
                        "core_submove_id": str(
                            cards[card_id]["core_submove_id"]
                        ),
                        "score": round(
                            float(query_vector @ card_vectors[card_index[card_id]]),
                            8,
                        ),
                    }
                    for card_id in candidate_ids
                ),
                key=lambda item: (item["score"], item["card_id"]),
                reverse=True,
            )
            preferred_id = source["preferred_card_id"]
            preferred_rank = (
                next(
                    index + 1
                    for index, item in enumerate(ranking)
                    if item["card_id"] == preferred_id
                )
                if preferred_id is not None
                else None
            )
            decision[f"{variant}_ranking"] = ranking
            decision[f"{variant}_top1_card_id"] = ranking[0]["card_id"]
            decision[f"{variant}_top1_core"] = ranking[0]["core_submove_id"]
            decision[f"{variant}_preferred_rank"] = preferred_rank
        decisions.append(decision)

    split_reports = {}
    for split in ("development32", "frozen_holdout16", "combined48"):
        subset = (
            decisions
            if split == "combined48"
            else [row for row in decisions if row["split"] == split]
        )
        baseline_exact = sum(
            row["lexical_top1_card_id"] == row["preferred_card_id"]
            for row in subset
            if row["preferred_card_id"]
        )
        preferred_count = sum(bool(row["preferred_card_id"]) for row in subset)
        split_reports[split] = {
            "lexical_visible_top3_baseline": {
                "exact_preferred_top1": baseline_exact,
                "items_with_human_preferred_candidate": preferred_count,
                "exact_preferred_top1_rate": round(
                    baseline_exact / preferred_count, 6
                ),
            },
            "bge_plain": _metrics(subset, "bge_plain"),
            "bge_prefixed": _metrics(subset, "bge_prefixed"),
        }

    failed_holdout = [
        row
        for row in decisions
        if row["split"] == "frozen_holdout16"
        and row["lexical_top1_card_id"] != row["preferred_card_id"]
    ]
    failure_rescue = []
    for row in failed_holdout:
        failure_rescue.append(
            {
                "review_item_id": row["review_item_id"],
                "preferred_lexical_rank": row["preferred_lexical_rank"],
                "bge_plain_rescues": (
                    row["bge_plain_top1_card_id"] == row["preferred_card_id"]
                ),
                "bge_prefixed_rescues": (
                    row["bge_prefixed_top1_card_id"]
                    == row["preferred_card_id"]
                ),
                "original_hard_exclusion": row["original_hard_exclusion"],
            }
        )

    report = {
        "protocol": PROTOCOL,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "status": "POSTHOC_DIAGNOSTIC_COMPLETE_NOT_A_V1_5_PROMOTION_TEST",
        "model": {
            "model_id": "BAAI/bge-small-en-v1.5",
            "local_revision": args.model.name,
            "device": str(device),
            "pooling": "CLS",
            "normalization": "L2",
            "score": "dot product equal to cosine",
            "max_length": 256,
        },
        "task": (
            "Rerank only the three already human-visible lexical candidates; "
            "this does not test full-bank candidate recall."
        ),
        "split_reports": split_reports,
        "frozen_holdout_failure_rescue": failure_rescue,
        "decision_policy": {
            "blocks_six_card_training_fallback": False,
            "can_promote_bge_for_formal_v1_5": False,
            "reason": (
                "The frozen holdout labels were already observed before this "
                "post-hoc challenger was run. A favorable result may motivate "
                "V2, but cannot replace the failed frozen V1.5 ranker."
            ),
        },
        "source_lineage": {
            "model_files_local_only": True,
            "cards": str(args.cards.relative_to(ROOT)),
            "cards_sha256": sha256_file(args.cards),
            "development_labels": str(
                args.development_labels.relative_to(ROOT)
            ),
            "development_labels_sha256": sha256_file(
                args.development_labels
            ),
            "holdout_labels": str(args.holdout_labels.relative_to(ROOT)),
            "holdout_labels_sha256": sha256_file(args.holdout_labels),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "bge_reranking_decisions.jsonl", decisions)
    write_json(args.out_dir / "bge_reranking_report.json", report)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "holdout": split_reports["frozen_holdout16"],
            "failure_rescue": failure_rescue,
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
