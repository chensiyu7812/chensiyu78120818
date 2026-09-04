#!/usr/bin/env python3
"""Compare post-H1 transparent and BGE rerankers on reviewed H1 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from metacom_pm.io import iter_jsonl, sha256_file
from metacom_pm.text import lexical_score, normalize_space
from metacom_pm.v1_5_strategy_rag_repair import (
    REPAIR_PROTOCOL,
    repaired_observable_opportunity_flags,
)
from metacom_pm.v1_5_strategy_rag_v4 import assess_v4_card_applicability


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-post-h1-retrieval-bakeoff-v1"
DEFAULT_BGE = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)


def _acceptable_ranks(value: object) -> set[int]:
    return {int(token) for token in re.findall(r"[1-5]", str(value or ""))}


def _seeker_texts(dialogue: list[dict[str, Any]]) -> list[str]:
    return [
        normalize_space(turn.get("content", ""))
        for turn in dialogue
        if str(turn.get("speaker")) == "seeker"
        and normalize_space(turn.get("content", ""))
    ]


def _query(dialogue: list[dict[str, Any]], flags: dict[str, Any]) -> str:
    excluded = {
        "substantive",
        "pure_phatic",
        "routine_closing",
        "active_high_stakes",
        "explicit_stop",
        "ordinary_rag_hard_off",
    }
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool) and value and key not in excluded
    ]
    recent = " ".join(_seeker_texts(dialogue)[-3:])
    return (
        "Represent this sentence for searching relevant passages: "
        "Choose one safe, topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cues) or 'none'}. "
        f"Recent seeker context: {recent}"
    )


def _card_text(card: dict[str, Any]) -> str:
    return " ".join(
        [
            str(card["strategy_family"]),
            str(card["support_move"]),
            str(card["when_to_use"]),
            "Goals:",
            ", ".join(card.get("goal_types") or []),
        ]
    )


def _encode_bge(
    texts: list[str],
    *,
    model_path: Path,
    batch_size: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if torch.cuda.is_available():
        device_index = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.get_device_properties(index).total_memory,
        )
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            vectors = model(**batch).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            chunks.append(vectors.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _bootstrap_difference(
    left: list[int],
    right: list[int],
    *,
    seed: int = 101,
    draws: int = 10000,
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        return {"difference": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(1)
    return {
        "difference": round(float(delta.mean()), 8),
        "ci95_low": round(float(np.quantile(sampled, 0.025)), 8),
        "ci95_high": round(float(np.quantile(sampled, 0.975)), 8),
    }


def _confusion(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        actual = bool(row["human_opportunity"])
        predicted = bool(row[f"{method}_on"])
        counts[
            ("tp" if actual and predicted else
             "tn" if not actual and not predicted else
             "fp" if not actual and predicted else "fn")
        ] += 1
    total = len(rows)
    return {
        "tp": counts["tp"],
        "tn": counts["tn"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "accuracy": round(
            (counts["tp"] + counts["tn"]) / total if total else 0.0, 8
        ),
        "balanced_accuracy": round(
            (
                counts["tp"] / max(1, counts["tp"] + counts["fn"])
                + counts["tn"] / max(1, counts["tn"] + counts["fp"])
            )
            / 2,
            8,
        ),
    }


def _render_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    methods = summary["methods"]
    table = "\n".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{value['gate_accuracy']:.1%}</td>"
        f"<td>{value['gate_balanced_accuracy']:.1%}</td>"
        f"<td>{value['top1_accuracy']:.1%} "
        f"({value['top1_correct']}/{value['top1_evaluable']})</td>"
        "</tr>"
        for name, value in methods.items()
    )
    hardoffs = Counter(
        reason
        for row in rows
        for reason in row["repair_hard_off_reasons"]
    )
    hardoff_list = "".join(
        f"<li>{html.escape(reason)}: {count}</li>"
        for reason, count in hardoffs.most_common()
    )
    decision = html.escape(summary["decision"]["plain_language"])
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>PM V1.5 H1 后检索修复比较</title><style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:auto;padding:24px;line-height:1.55;color:#17202a}}
.box{{background:#f3f6f9;border-left:5px solid #245f9e;padding:14px;margin:14px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd5df;padding:9px;text-align:left}}
</style></head><body><h1>PM V1.5 H1 后检索修复比较</h1>
<div class="box"><b>结论：</b>{decision}</div>
<p>这是 H1 开发集上的诊断，不是最终资格结论。所有方法都只在同一批已经人审过的
Top‑5 卡片内比较；H2 才用于冻结。</p>
<table><thead><tr><th>方法</th><th>开关准确率</th><th>开关 balanced accuracy</th>
<th>可评状态 Top‑1</th></tr></thead><tbody>{table}</tbody></table>
<h2>新增普通 RAG hard-off 触发</h2><ul>{hardoff_list}</ul>
<h2>解释边界</h2><p>BAAI 只参与同一安全/适用性层级内的排序，不能覆盖 hard-off，
不能把事实资源请求变成普通技术卡，也不产出 PM 训练标签。无合适卡的真实机会仍记为
Bank coverage gap，而不是强迫检索器猜一张。</p></body></html>"""


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h1-packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/h1_review_packet.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_analysis/h1_annotations_frozen.jsonl",
    )
    parser.add_argument("--bge-model", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_post_h1_retrieval_bakeoff_v1",
    )
    args = parser.parse_args()

    packet = json.loads(args.h1_packet.read_text(encoding="utf-8"))
    annotations = {
        str(row["review_item_id"]): dict(row)
        for row in iter_jsonl(args.annotations)
        if row.get("item_type") == "retrieval"
    }
    item_rows: list[dict[str, Any]] = []
    all_texts: list[str] = []
    text_index: dict[str, int] = {}

    def bind_text(text: str) -> int:
        if text not in text_index:
            text_index[text] = len(all_texts)
            all_texts.append(text)
        return text_index[text]

    for item in packet["retrieval_items"]:
        item_id = str(item["review_item_id"])
        annotation = annotations[item_id]
        dialogue = list(item["visible_dialogue"])
        seekers = _seeker_texts(dialogue)
        latest = seekers[-1] if seekers else ""
        recent = " ".join(seekers[-3:])
        flags = repaired_observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=recent,
            visible_dialogue=dialogue,
        )
        human_opportunity = (
            str(annotation.get("rs_opportunity") or "no") == "yes"
            if item["expected_behavior"] == "rank_cards"
            else False
        )
        candidates: list[dict[str, Any]] = []
        acceptable_ids: set[str] = set()
        if item["expected_behavior"] == "rank_cards":
            ranks = _acceptable_ranks(annotation.get("acceptable_ranks"))
            for rank, old in enumerate(item["top5"], start=1):
                card = dict(old["card"])
                card["retrieval_text"] = _card_text(card)
                if rank in ranks:
                    acceptable_ids.add(str(card["core_submove_id"]))
                applicability = assess_v4_card_applicability(
                    card,
                    current_user_text=latest,
                    recent_user_text=recent,
                    flags=flags,
                )
                candidates.append(
                    {
                        "old_rank": rank,
                        "core_submove_id": str(card["core_submove_id"]),
                        "card": card,
                        "eligible": (
                            not flags["ordinary_rag_hard_off"]
                            and bool(applicability["eligible"])
                        ),
                        "compatibility_tier": int(
                            applicability["compatibility_tier"]
                        ),
                        "lexical_score": lexical_score(
                            _query(dialogue, flags), _card_text(card)
                        ),
                        "card_text_index": bind_text(_card_text(card)),
                    }
                )
        query_index = bind_text(_query(dialogue, flags))
        item_rows.append(
            {
                "review_item_id": item_id,
                "expected_behavior": item["expected_behavior"],
                "human_opportunity": human_opportunity,
                "human_acceptable_core_ids": sorted(acceptable_ids),
                "human_hard_exclusion_at_old_top1": (
                    annotation.get("hard_exclusion_triggered") == "yes"
                ),
                "repair_hard_off": bool(flags["ordinary_rag_hard_off"]),
                "repair_hard_off_reasons": list(
                    flags["ordinary_rag_hard_off_reasons"]
                ),
                "query_text_index": query_index,
                "candidates": candidates,
            }
        )

    embeddings = _encode_bge(
        all_texts,
        model_path=args.bge_model,
        batch_size=args.batch_size,
    )
    for row in item_rows:
        query_vector = embeddings[row["query_text_index"]]
        for candidate in row["candidates"]:
            candidate["bge_cosine"] = float(
                query_vector
                @ embeddings[candidate["card_text_index"]]
            )
        eligible = [
            candidate for candidate in row["candidates"]
            if candidate["eligible"]
        ]
        methods = {
            "original_h1": sorted(
                row["candidates"], key=lambda value: value["old_rank"]
            ),
            "repaired_transparent": sorted(
                eligible,
                key=lambda value: (
                    value["compatibility_tier"],
                    value["lexical_score"],
                    -value["old_rank"],
                ),
                reverse=True,
            ),
            "repaired_bge": sorted(
                eligible,
                key=lambda value: (
                    value["compatibility_tier"],
                    value["bge_cosine"],
                    -value["old_rank"],
                ),
                reverse=True,
            ),
        }
        acceptable = set(row["human_acceptable_core_ids"])
        for method, ranked in methods.items():
            on = bool(ranked) and (
                method == "original_h1" or not row["repair_hard_off"]
            )
            top1 = ranked[0]["core_submove_id"] if on else None
            row[f"{method}_on"] = on
            row[f"{method}_top1_core_submove_id"] = top1
            row[f"{method}_top1_acceptable"] = (
                top1 in acceptable if top1 is not None and acceptable else None
            )
        for candidate in row["candidates"]:
            candidate.pop("card", None)
            candidate.pop("card_text_index", None)
        row.pop("query_text_index", None)

    methods_summary: dict[str, Any] = {}
    for method in (
        "original_h1",
        "repaired_transparent",
        "repaired_bge",
    ):
        confusion = _confusion(item_rows, method)
        evaluable = [
            row for row in item_rows
            if row["human_opportunity"]
            and row["human_acceptable_core_ids"]
            and row[f"{method}_on"]
        ]
        correct = [
            int(bool(row[f"{method}_top1_acceptable"]))
            for row in evaluable
        ]
        methods_summary[method] = {
            "gate_accuracy": confusion["accuracy"],
            "gate_balanced_accuracy": confusion["balanced_accuracy"],
            "gate_confusion": confusion,
            "top1_evaluable": len(evaluable),
            "top1_correct": sum(correct),
            "top1_accuracy": round(
                sum(correct) / len(correct) if correct else 0.0, 8
            ),
        }
    paired = [
        row for row in item_rows
        if row["human_opportunity"]
        and row["human_acceptable_core_ids"]
        and row["repaired_transparent_on"]
        and row["repaired_bge_on"]
    ]
    bge_difference = _bootstrap_difference(
        [
            int(bool(row["repaired_transparent_top1_acceptable"]))
            for row in paired
        ],
        [
            int(bool(row["repaired_bge_top1_acceptable"]))
            for row in paired
        ],
    )
    select_bge = (
        bge_difference["difference"] >= 0.05
        and bge_difference["ci95_low"] >= 0.0
    )
    decision = {
        "selected_development_reranker": (
            "repaired_bge" if select_bge else "repaired_transparent"
        ),
        "bge_selected": select_bge,
        "rule": (
            "Select BGE only if paired Top-1 improves by at least 0.05 and "
            "the bootstrap 95% lower bound is nonnegative; H2 remains required."
        ),
        "plain_language": (
            "BAAI is retained as the H2 challenger because it materially "
            "improved H1 ranking under the frozen safety tiers."
            if select_bge
            else
            "BAAI did not clear the development improvement rule. Keep the "
            "transparent tier-plus-lexical reranker for H2 and do not make "
            "BAAI a prerequisite."
        ),
    }
    summary = {
        "protocol": PROTOCOL,
        "repair_protocol": REPAIR_PROTOCOL,
        "status": "DEVELOPMENT_DIAGNOSTIC_NOT_FORMAL_QUALIFICATION",
        "items": len(item_rows),
        "method_scope": (
            "Same H1 human-reviewed Top-5 candidate sets; no unseen-card "
            "judgments are imputed."
        ),
        "methods": methods_summary,
        "paired_bge_minus_transparent": bge_difference,
        "paired_items": len(paired),
        "decision": decision,
        "interpretation_limits": [
            "H1 was used to define the repair and cannot qualify it.",
            "Top-1 is evaluated only when H1 named at least one acceptable candidate.",
            "True opportunity states with no acceptable H1 Top-5 card remain Bank coverage gaps.",
            "BAAI is a within-safety-tier ranker only, never a hard-off or PM labeler.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "item_diagnostics.jsonl"
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.html"
    _write_jsonl(rows_path, item_rows)
    _write_json(summary_path, summary)
    report_path.write_text(_render_report(summary, item_rows), encoding="utf-8")
    manifest = {
        "protocol": PROTOCOL,
        "inputs": {
            "h1_packet": str(args.h1_packet.relative_to(ROOT)),
            "annotations": str(args.annotations.relative_to(ROOT)),
            "bge_model": str(args.bge_model),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (rows_path, summary_path, report_path)
        },
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": summary["status"],
            "decision": decision["selected_development_reranker"],
            "methods": methods_summary,
            "bge_difference": bge_difference,
        }
    )


if __name__ == "__main__":
    main()
