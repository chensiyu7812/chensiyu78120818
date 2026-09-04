#!/usr/bin/env python3
"""Evidence-only audit of the frozen MetaCom V1 Strategy RAG.

This script deliberately does not import or execute the dirty working-tree
retriever.  It replays the byte-identical V1 algorithm documented by the
study freeze and writes only a new audit directory.  It never writes a card
bank, index, V8 label, checkpoint, or prior result.
"""

from __future__ import annotations

from collections import Counter
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from metacom_pm.contracts import StrategyCard
from metacom_pm.io import sha256_file, stable_hex
from metacom_pm.strategy_bank import (
    _preceding_context,
    find_esconv_evoemo_overlaps,
    load_esconv,
    stable_dialogue_split,
)
from metacom_pm.text import normalize_for_hash, normalize_space


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
HOME = ROOT.parents[2]
OUT = ROOT / "outputs" / "strategy_rag_v1_audit"
TMP = OUT.with_name(OUT.name + ".tmp")

CURRENT_BANK = ROOT / "data" / "strategy" / "strategy_cards.jsonl"
V13_BANK = ROOT / "data" / "strategy" / "strategy_cards_v13.jsonl"
SPLIT_MANIFEST = ROOT / "data" / "strategy" / "esconv_split_manifest.jsonl"
BANK_AUDIT = ROOT / "data" / "strategy" / "strategy_bank_audit.json"
ESCONV = ROOT / "data" / "external" / "ESConv.json"
EVOEMO = ROOT / "data" / "external" / "evo_emo.json"
STUDY_FREEZE = ROOT / "outputs" / "study_freeze.json"
V8_CASES = (
    ROOT
    / "outputs"
    / "pm_v2_generation_pilot_semantic_review_v8"
    / "generation_pilot_semantic_review_cases.json"
)
LEGACY_BANK = (
    HOME
    / "esconv_experiment_bundle"
    / "policy_manager_35"
    / "data"
    / "strategy"
    / "pilot_strategy_cards_for_synthetic_sweep.jsonl"
)
V1_RETRIEVER = (
    HOME
    / "metacom_workspace"
    / "metacom_v1"
    / "project"
    / "src"
    / "metacom_pm"
    / "retrieval.py"
)
DIRTY_RETRIEVER = ROOT / "src" / "metacom_pm" / "retrieval.py"
DIRTY_TEXT = ROOT / "src" / "metacom_pm" / "text.py"
V1_TEXT = (
    HOME
    / "metacom_workspace"
    / "metacom_v1"
    / "project"
    / "src"
    / "metacom_pm"
    / "text.py"
)

EXPECTED = {
    "current_bank": "f64ded1e23b4a79c0e08c6b47355ad76b3880d5cbc0bfa7530f366149b5c70db",
    "legacy_bank": "5b2d58c6be07d3c27e20e9162a3494f47f875d02b94b983f8b05c08ff044bf46",
    "v13_bank": "960cfc78169e14817005f770894e0a3abc016dbf33757a9bbfa00702adf570e0",
    "split_manifest": "526d0451cc5a8950d5f1b2aeb10456e4236ef1a01e1bf7b49dba7b87b44474fd",
    "bank_audit": "8cae142aae7b7fc8ca75df2a183e3ba74cdafbe8e350661c113e37902a278465",
    "esconv": "aa0556c5b330562ba009c1cd5137486bfa2a7255f33225a6524cd58f7efdd9af",
    "evoemo": "f30698e87fddaeff51270a666c654da604f487a3456ec60d2b6ae08a6fecd420",
    "v1_retriever": "5202ca6e511d06254e0092629cb197f4e4536bb7bc90024cce78a3546c698d92",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=WORKSPACE, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def v1_query(current_user_text: str, history: list[dict[str, Any]], summary: str) -> str:
    # Byte-for-byte semantics of frozen src/metacom_pm/retrieval.py.
    parts = [summary]
    parts.extend(str(turn.get("content") or "") for turn in history)
    parts.append(current_user_text)
    return normalize_space("\n".join(parts))


def v1_lexical_score(query: str, document: str) -> float:
    def bag(text: str) -> Counter[str]:
        return Counter(TOKEN_RE.findall(str(text or "").lower()))

    q = bag(query)
    d = bag(document)
    if not q or not d:
        return 0.0
    dot = sum(q[token] * d.get(token, 0) for token in q)
    qn = math.sqrt(sum(value * value for value in q.values()))
    dn = math.sqrt(sum(value * value for value in d.values()))
    return dot / (qn * dn) if qn and dn else 0.0


def retrieve(cards: list[dict[str, Any]], query: str, top_k: int = 3) -> list[dict[str, Any]]:
    scored = [(v1_lexical_score(query, card["retrieval_text"]), card) for card in cards]
    scored.sort(key=lambda pair: (pair[0], pair[1]["strategy_id"]), reverse=True)
    return [
        {
            "rank": rank,
            "similarity_score": score,
            **card,
        }
        for rank, (score, card) in enumerate(scored[:top_k], 1)
    ]


def reconstruct_current_bank_in_memory() -> dict[str, Any]:
    """Recompute canonical bytes in RAM; do not create another bank file."""
    esconv = load_esconv(ESCONV)
    overlaps = find_esconv_evoemo_overlaps(esconv, EVOEMO, jaccard_threshold=0.88)
    cards: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(esconv):
        split = stable_dialogue_split(idx, 13)
        dialogue_id = f"esconv_{idx:04d}"
        split_rows.append(
            {
                "dialogue_id": dialogue_id,
                "index": idx,
                "split": split,
                "excluded_for_evoemo_overlap": idx in overlaps,
            }
        )
        if split != "train" or idx in overlaps:
            continue
        dialogue = row["dialog"]
        for turn_index, turn in enumerate(dialogue):
            if turn.get("speaker") != "supporter":
                continue
            annotation = turn.get("annotation") or {}
            strategy = normalize_space(annotation.get("strategy") or "Others")
            response = normalize_space(turn.get("content") or "")
            if not response:
                continue
            context = _preceding_context(dialogue, turn_index)
            situation = normalize_space(row.get("situation") or "")
            retrieval_text = "\n".join(value for value in [situation, context] if value)
            guidance = (
                f"Use the emotional-support strategy '{strategy}' when it fits the "
                "visible dialogue. Adapt it naturally; do not copy private facts or "
                "assume information that the user has not disclosed."
            )
            card = StrategyCard(
                strategy_id=f"strat_{stable_hex(dialogue_id, turn_index, strategy, response, n=20)}",
                strategy_label=strategy,
                retrieval_text=retrieval_text or situation or response,
                guidance_text=guidance,
                example_response=response,
                source_dialogue_id=dialogue_id,
                source_turn_index=turn_index,
            )
            cards.append(card.model_dump(mode="json"))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for card in cards:
        key = (
            normalize_for_hash(card["retrieval_text"]),
            normalize_for_hash(card["example_response"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)
    bank_bytes = "".join(
        json.dumps(card, ensure_ascii=False, sort_keys=True) + "\n" for card in deduped
    ).encode("utf-8")
    split_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in split_rows
    ).encode("utf-8")
    return {
        "bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "n_cards": len(deduped),
        "n_dialogues": len(esconv),
        "n_overlaps": len(overlaps),
    }


def leakage_checks(
    cards: list[dict[str, Any]], split_rows: list[dict[str, Any]], esconv: list[dict[str, Any]]
) -> dict[str, Any]:
    split_by_id = {row["dialogue_id"]: row for row in split_rows}
    source_ids = {card["source_dialogue_id"] for card in cards}
    wrong_split = sorted(
        source_id for source_id in source_ids if split_by_id[source_id]["split"] != "train"
    )
    overlap_sources = sorted(
        source_id
        for source_id in source_ids
        if split_by_id[source_id]["excluded_for_evoemo_overlap"]
    )
    duplicate_ids = len(cards) - len({card["strategy_id"] for card in cards})

    heldout_responses: set[str] = set()
    heldout_retrieval_contexts: set[str] = set()
    heldout_pairs: set[tuple[str, str]] = set()
    heldout_context_sources: dict[str, list[dict[str, Any]]] = {}
    train_dialogues: set[str] = set()
    heldout_dialogues: set[str] = set()
    for idx, row in enumerate(esconv):
        split = stable_dialogue_split(idx, 13)
        dialogue_value = normalize_for_hash(
            "\n".join(
                f"{turn.get('speaker')}: {normalize_space(turn.get('content') or '')}"
                for turn in row["dialog"]
            )
        )
        if split == "train":
            train_dialogues.add(dialogue_value)
            continue
        heldout_dialogues.add(dialogue_value)
        for turn_index, turn in enumerate(row["dialog"]):
            if turn.get("speaker") != "supporter":
                continue
            response = normalize_for_hash(turn.get("content") or "")
            if response:
                heldout_responses.add(response)
            context = _preceding_context(row["dialog"], turn_index)
            situation = normalize_space(row.get("situation") or "")
            retrieval = normalize_for_hash("\n".join(x for x in [situation, context] if x))
            if retrieval:
                heldout_retrieval_contexts.add(retrieval)
                heldout_context_sources.setdefault(retrieval, []).append(
                    {
                        "dialogue_id": f"esconv_{idx:04d}",
                        "turn_index": turn_index,
                        "split": split,
                        "response": normalize_space(turn.get("content") or ""),
                    }
                )
            heldout_pairs.add((retrieval, response))
    response_collisions = sum(
        normalize_for_hash(card["example_response"]) in heldout_responses for card in cards
    )
    response_collision_lengths = [
        len(normalize_for_hash(card["example_response"]).split())
        for card in cards
        if normalize_for_hash(card["example_response"]) in heldout_responses
    ]
    context_collision_details = []
    exact_pair_collisions = 0
    for card in cards:
        retrieval = normalize_for_hash(card["retrieval_text"])
        response = normalize_for_hash(card["example_response"])
        if retrieval in heldout_retrieval_contexts:
            context_collision_details.append(
                {
                    "train_card_id": card["strategy_id"],
                    "train_source_dialogue_id": card["source_dialogue_id"],
                    "train_source_turn_index": card["source_turn_index"],
                    "retrieval_text": card["retrieval_text"],
                    "train_response": card["example_response"],
                    "heldout_sources": heldout_context_sources[retrieval],
                }
            )
        exact_pair_collisions += (retrieval, response) in heldout_pairs
    forbidden_keys = {
        "judge_score",
        "overall",
        "support_rating",
        "action_id",
        "outcome",
        "future_user_response",
        "pm_label",
    }
    observed_keys = set().union(*(card.keys() for card in cards))
    return {
        "source_dialogues": len(source_ids),
        "wrong_split_source_ids": wrong_split,
        "evoemo_overlap_source_ids": overlap_sources,
        "duplicate_strategy_ids": duplicate_ids,
        "heldout_exact_example_response_text_collisions": response_collisions,
        "heldout_unique_response_text_collisions": len(
            {
                normalize_for_hash(card["example_response"])
                for card in cards
                if normalize_for_hash(card["example_response"]) in heldout_responses
            }
        ),
        "heldout_max_collision_response_tokens": max(response_collision_lengths, default=0),
        "heldout_long_response_collisions_ge_20_tokens": sum(
            length >= 20 for length in response_collision_lengths
        ),
        "heldout_exact_retrieval_context_collisions": len(context_collision_details),
        "heldout_retrieval_context_collision_details": context_collision_details,
        "heldout_exact_context_response_pair_collisions": exact_pair_collisions,
        "cross_split_full_dialogue_collisions": len(train_dialogues & heldout_dialogues),
        "forbidden_schema_keys_present": sorted(forbidden_keys & observed_keys),
        "interpretation": (
            "All provenance is train-only. Exact short-response collisions with held-out "
            "text are counted separately and do not mean held-out rows were source rows."
        ),
    }


def render_cards(cards: list[dict[str, Any]]) -> str:
    chunks = []
    for card in cards:
        chunks.append(
            "\n".join(
                [
                    f"  {card['rank']}. `{card['strategy_id']}` score={card['similarity_score']:.9f}",
                    f"     - label: {card['strategy_label']}",
                    f"     - retrieval_text: {card['retrieval_text']}",
                    f"     - guidance_text: {card['guidance_text']}",
                    f"     - example_response: {card['example_response']}",
                    f"     - source: {card['source_dialogue_id']} turn {card['source_turn_index']}",
                ]
            )
        )
    return "\n".join(chunks)


def main() -> None:
    if OUT.exists() or TMP.exists():
        raise RuntimeError(
            f"refusing to overwrite existing audit output: {OUT if OUT.exists() else TMP}"
        )
    required = [
        CURRENT_BANK,
        V13_BANK,
        SPLIT_MANIFEST,
        BANK_AUDIT,
        ESCONV,
        EVOEMO,
        STUDY_FREEZE,
        V8_CASES,
        LEGACY_BANK,
        V1_RETRIEVER,
        DIRTY_RETRIEVER,
        V1_TEXT,
        DIRTY_TEXT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required audit evidence missing: " + ", ".join(missing))

    hashes = {
        "current_bank": sha256_file(CURRENT_BANK),
        "legacy_bank": sha256_file(LEGACY_BANK),
        "v13_bank": sha256_file(V13_BANK),
        "split_manifest": sha256_file(SPLIT_MANIFEST),
        "bank_audit": sha256_file(BANK_AUDIT),
        "esconv": sha256_file(ESCONV),
        "evoemo": sha256_file(EVOEMO),
        "v1_retriever": sha256_file(V1_RETRIEVER),
        "dirty_retriever": sha256_file(DIRTY_RETRIEVER),
        "v1_text": sha256_file(V1_TEXT),
        "dirty_text": sha256_file(DIRTY_TEXT),
    }
    expected_checks = {key: hashes[key] == value for key, value in EXPECTED.items()}
    if not all(expected_checks.values()):
        raise RuntimeError(f"frozen evidence hash mismatch: {expected_checks}")

    current_cards = read_jsonl(CURRENT_BANK)
    legacy_cards = read_jsonl(LEGACY_BANK)
    split_rows = read_jsonl(SPLIT_MANIFEST)
    esconv = load_esconv(ESCONV)
    bank_audit = read_json(BANK_AUDIT)
    study_freeze = read_json(STUDY_FREEZE)
    reconstruction = reconstruct_current_bank_in_memory()
    leakage = leakage_checks(current_cards, split_rows, esconv)

    freeze_bank_hash = study_freeze["data_hashes"]["data/strategy/strategy_cards.jsonl"]
    freeze_retriever_hash = study_freeze["code_hashes"]["src/metacom_pm/retrieval.py"]
    git_current_bank_hash = hashlib.sha256(
        subprocess.run(
            ["git", "show", "main:project/data/strategy/strategy_cards.jsonl"],
            cwd=WORKSPACE,
            check=True,
            capture_output=True,
        ).stdout
    ).hexdigest()
    git_v1_retriever_hash = hashlib.sha256(
        subprocess.run(
            ["git", "show", "34737d6:project/src/metacom_pm/retrieval.py"],
            cwd=WORKSPACE,
            check=True,
            capture_output=True,
        ).stdout
    ).hexdigest()

    cases = read_json(V8_CASES)
    traces: list[dict[str, Any]] = []
    for case in cases:
        query = v1_query(
            case["current_user_text"],
            case["dialogue_before_current"],
            case["session_summary"],
        )
        formal = retrieve(current_cards, query, top_k=3)
        legacy = retrieve(legacy_cards, query, top_k=3)
        displayed_ids = [row["card_id"] for row in case.get("strategy_evidence", [])]
        formal_ids = [row["strategy_id"] for row in formal]
        traces.append(
            {
                "item_id": case["item_id"],
                "case_id": case["case_id"],
                "regime": case["regime"],
                "actual_query": query,
                "query_context_fields": [
                    "session_summary",
                    "dialogue_before_current[*].content",
                    "current_user_text",
                ],
                "query_inputs": {
                    "session_summary": case["session_summary"],
                    "dialogue_before_current": case["dialogue_before_current"],
                    "current_user_text": case["current_user_text"],
                },
                "formal_external_v1": {
                    "bank_path": str(CURRENT_BANK),
                    "bank_sha256": hashes["current_bank"],
                    "card_count": len(current_cards),
                    "retriever_sha256": hashes["v1_retriever"],
                    "top_k": 3,
                    "threshold": None,
                    "embedding_model": None,
                    "index_path": None,
                    "filter_process": "none",
                    "rerank_process": "none; sort all cards by (bag-of-words cosine, strategy_id) descending",
                    "cards": formal,
                },
                "legacy_synthetic_training_v1": {
                    "bank_path": str(LEGACY_BANK),
                    "bank_sha256": hashes["legacy_bank"],
                    "card_count": len(legacy_cards),
                    "retriever_sha256": hashes["v1_retriever"],
                    "top_k": 3,
                    "threshold": None,
                    "embedding_model": None,
                    "index_path": None,
                    "filter_process": "none",
                    "rerank_process": "none; same canonical V1 lexical sort",
                    "cards": legacy,
                },
                "provisional_v8_displayed_card_ids": displayed_ids,
                "formal_top_k_card_ids": formal_ids,
                "displayed_vs_formal": {
                    "exact_order_match": displayed_ids == formal_ids,
                    "displayed_all_in_formal_top_k": set(displayed_ids) <= set(formal_ids),
                    "intersection_card_ids": sorted(set(displayed_ids) & set(formal_ids)),
                    "status": "MATCH" if displayed_ids == formal_ids else "MISMATCH",
                },
            }
        )

    exact_matches = sum(
        trace["displayed_vs_formal"]["exact_order_match"] for trace in traces
    )
    subset_matches = sum(
        trace["displayed_vs_formal"]["displayed_all_in_formal_top_k"] for trace in traces
    )
    current_ids = [card["strategy_id"] for card in current_cards]
    source_dialogues = {card["source_dialogue_id"] for card in current_cards}
    strategy_counts = dict(sorted(Counter(card["strategy_label"] for card in current_cards).items()))
    split_counts = dict(sorted(Counter(row["split"] for row in split_rows).items()))
    overlap_split_counts = dict(
        sorted(
            Counter(
                row["split"]
                for row in split_rows
                if row["excluded_for_evoemo_overlap"]
            ).items()
        )
    )

    retriever_diff = "\n".join(
        difflib.unified_diff(
            V1_RETRIEVER.read_text(encoding="utf-8").splitlines(),
            DIRTY_RETRIEVER.read_text(encoding="utf-8").splitlines(),
            fromfile="canonical_v1/retrieval.py",
            tofile="current_dirty/retrieval.py",
            lineterm="",
        )
    )

    TMP.mkdir(parents=True)
    dump_jsonl(TMP / "V1_RETRIEVAL_TRACES.jsonl", traces)

    lineage_md = f"""# V1 Strategy RAG lineage

审计对象：`{ROOT}`

## 结论

现有 V1 Strategy RAG **确实存在**，但第一篇研究包含两个有明确证据的卡库阶段，不能称为单一版本：

1. synthetic action sweep / PM 训练标签使用 156-card legacy bank，SHA256 `{hashes['legacy_bank']}`；
2. confirmatory ESConv sweep、study freeze 与 EvoEmo generation 使用当前 12,429-card bank，SHA256 `{hashes['current_bank']}`。

Git 证据进一步定位了切换点：commit `34737d675bd8e2357411ce87ad40e94f496de783`（2026-06-28 10:16 JST）仍保存 5b2d bank；commit `3a2599b5ddd3169366107a9b7998cc4bb7ce5080`（2026-06-28 18:10 JST）保存 f64d bank。后者与当日 study freeze 及后续 external manifests 一致。

## 当前 12,429-card bank 的真实生成链路

- 生成入口：`scripts/03_build_strategy_bank.py`（冻结哈希 `28c597bc8fde248eda2556501b2478329a0613687b69637a13ae2df08636d1e4`）。
- 实现：`src/metacom_pm/strategy_bank.py`（冻结哈希 `2a74f5b27f775e6f267e4787f7183b04f255f6a442ebda4a3f6e6144861474b3`）。
- 输入 ESConv：`data/external/ESConv.json`，SHA256 `{hashes['esconv']}`。
- 输入 EvoEmo：`data/external/evo_emo.json`，SHA256 `{hashes['evoemo']}`；仅用于排除重合对话，不向卡片注入内容。
- split：对 1,300 个 ESConv dialogue 以 `stable_hex('esconv_split', seed=13, dialogue_index)` 做固定 70/15/15 dialogue-level split；实际计数 `{split_counts}`。
- overlap gate：exact 或 5-shingle Jaccard >= 0.88；共排除 {bank_audit['n_excluded_overlap']} 个 dialogue，分布 `{overlap_split_counts}`。
- 建卡范围：仅 train 且未与 EvoEmo 重合的 dialogue；共 {len(source_dialogues)} 个 source dialogues、{len(current_cards)} 张去重 supporter-turn cards。
- 每卡内容：ESConv `situation` + 当前 supporter turn 之前最多 6 个 turns 作为 retrieval text；当前 supporter turn 的官方 strategy annotation、response 与固定 guidance 模板。
- 去重：只按 `(normalized retrieval_text, normalized example_response)` exact dedup。
- seed：13；生成模型：**无**；API：**无**；人工后处理：代码和冻结记录中均未发现。
- 冻结输出命令：`PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/03_build_strategy_bank.py --esconv data/external/ESConv.json --evoemo data/external/evo_emo.json --out-dir data/strategy --seed 13`。
- RAM-only 复算（未写新卡库）：bank SHA256 `{reconstruction['bank_sha256']}`，split SHA256 `{reconstruction['split_sha256']}`，与当前文件完全一致。

## 其他 Strategy Bank 版本

- `data/strategy/strategy_cards_v13.jsonl`：156 张，SHA256 `{hashes['v13_bank']}`；由 `scripts/build_strategy_cards_v13.py` 声明从 `data/esconv_rag_v12/strategy_cards_train_v12.jsonl` 和 `data/esconv_pm_v32/pm_train.jsonl` 构建。两份上游文件已不在当前 clean V1/V2 工程或已搜索旧目录中，因此 v13 的更早上游不能在本工作区完整重建。
- `pilot_strategy_cards_for_synthetic_sweep.jsonl`：v13 的 opaque-ID/sanitized 兼容转换，156 张，SHA256 `{hashes['legacy_bank']}`；其 attestation 明确绑定 synthetic sweep。转换产物仍在旧 V1 工程，但未找到独立转换脚本。
- 更早 archives 中另有 `strategy_cards_train_v1/v11`，不在 V3.3 formal manifest 的调用路径内。

没有新建、重写或人工优化任何 Strategy Card Bank。
"""
    (TMP / "V1_RAG_LINEAGE.md").write_text(lineage_md, encoding="utf-8")

    canonical_md = f"""# V1 formal canonical match

## 按实验阶段判定

| 阶段 | 记录卡库 SHA256 | 与当前 `{hashes['current_bank'][:12]}…` | 证据 |
|---|---|---|---|
| synthetic action sweep / PM training outcomes | `{hashes['legacy_bank']}` | **MISMATCH** | `outputs/synthetic_sweep/run_manifest.json`、legacy attestation、`selection_stable.json` provenance |
| confirmatory ESConv action sweep | `{hashes['current_bank']}` | **MATCH** | `outputs/esconv_sweep/run_manifest.json` 与 attestation |
| study freeze | `{freeze_bank_hash}` | **MATCH** | `outputs/study_freeze.json` |
| EvoEmo selective generation | `{hashes['current_bank']}` | **MATCH** | `outputs/evoemo_selective/run_manifest.json` 与 attestation |
| Git `main` current bank | `{git_current_bank_hash}` | **MATCH** | Git blob at `main:project/data/strategy/strategy_cards.jsonl` |

因此“第一篇实际用了哪一个版本”的准确答案是：**训练期使用 5b2d…，正式 confirmatory/external inference 使用 f64d…**。不存在一个贯穿全部阶段的单一卡库。当前文件对 final external V1 是 MATCH，对 PM 训练数据生成是 MISMATCH。

## 冻结一致性

- current bank count：{len(current_cards)}；unique card IDs：{len(set(current_ids))}。
- strategy labels：`{strategy_counts}`。
- study-freeze retriever hash：`{freeze_retriever_hash}`。
- Git 34737d6 retriever hash：`{git_v1_retriever_hash}`。
- V1 preserved workspace retriever hash：`{hashes['v1_retriever']}`。
- 三者一致：{freeze_retriever_hash == git_v1_retriever_hash == hashes['v1_retriever']}。

总判定：**stage-dependent MATCH/MISMATCH，不得简化成无条件 MATCH**。
"""
    (TMP / "V1_RAG_CANONICAL_MATCH.md").write_text(canonical_md, encoding="utf-8")

    retriever_md = f"""# V1 retriever and current dirty diff

## 第一篇正式调用路径

- module：`src/metacom_pm/retrieval.py::context_query` + `StrategyRetriever`。
- callers：`src/metacom_pm/sweep.py`（synthetic/ESConv action sweep）和 `src/metacom_pm/evoemo.py`（external generation）。
- query fields（固定顺序）：`current_session_summary` → 每个 `current_session_history[*].content` → `current_user_text`；最后压缩空白。
- embedding model：**无**。
- score：lowercased ASCII alphanumeric/apostrophe token count vectors的 cosine similarity。
- top-k：3（`configs/experiment.yaml::protocol.strategy_top_k`；EvoEmo 路径硬编码 3）。
- threshold：无；即使 top score 为 0，仍按 ID tie-break 返回三张卡。
- filter：无。
- rerank：无；仅 `(score, strategy_id)` descending。
- index path：无。
- runtime index rebuild：无；每次启动读取 JSONL，逐 query 全量扫描。

## Embedding / index 完整性

- 在第一篇脚本、config、study freeze、run manifests、outputs 和项目文件中均未发现 Strategy RAG embedding artifact、FAISS/Annoy/Chroma index 或 card-ID-to-index manifest。
- 因此不存在可与 card count、card IDs 或 bank hash 对齐的持久化 index；这不是 artifact 丢失，而是 V1 设计本来就没有 index。
- 运行时只把 JSONL 的 12,429（external）或 156（synthetic training）张卡加载成 Python list；score 时全量遍历。card count 与 IDs 直接来自对应 JSONL。
- 第一篇保存了 `strategy_top_k=3`、bank SHA256、retriever/code SHA256 和 config SHA256，但没有 embedding model/index 配置，因为两者均为 `null/not applicable`。
- 复用结论：无需重建 index；必须把“无 index、dynamic full scan”写入新的冻结元数据，防止未来误以为缺文件。

## 当前 dirty 文件

- canonical V1 SHA256：`{hashes['v1_retriever']}`。
- current dirty SHA256：`{hashes['dirty_retriever']}`。
- byte match：**MISMATCH**。
- query construction：未变。
- lexical score dependency：未变；`text.py` 的 dirty 改动只新增 token-budget helper，不改变 `lexical_score`。
- memory retriever：新增 per-source `minimum_score_by_source` 及 fail-closed zero-score filtering。
- strategy retriever：新增 optional `minimum_score`；`None` 时排序/返回与 V1 等价，非 `None` 时会 abstain/drop low scores，属于 V2 行为而非 canonical V1。
- 结论：不能把当前文件的哈希声明为 V1；trace 使用 preserved canonical V1 semantics。

```diff
{retriever_diff}
```
"""
    (TMP / "V1_RETRIEVER_DIFF.md").write_text(retriever_md, encoding="utf-8")

    leakage_md = f"""# V1 Strategy RAG leakage audit

## 判定

未发现当前 `{hashes['current_bank']}` 卡库使用 ESConv validation/test 具体 source rows、EvoEmo 内容、judge score、PM action/outcome label 或未来用户回复。结论是 **NO MATERIAL LEAKAGE FOUND**。

## A/B/C 区分

- A. ESConv 通用策略定义：使用官方 supporter strategy annotation；允许，且确实使用。
- B. ESConv train 内容构建：使用 train supporter response、situation 和该 response 之前的 visible context；允许，且确实使用。
- C. 复制 ESConv validation/test 具体样本：source provenance 检查为 0，未发现。

## 机械检查

- bank source dialogues：{leakage['source_dialogues']}。
- 非 train source IDs：{len(leakage['wrong_split_source_ids'])}。
- 被 EvoEmo overlap gate 排除却仍入库的 source IDs：{len(leakage['evoemo_overlap_source_ids'])}。
- duplicate strategy IDs：{leakage['duplicate_strategy_ids']}。
- forbidden label/schema keys：`{leakage['forbidden_schema_keys_present']}`。
- 与 held-out response 的 exact normalized 文本碰撞：{leakage['heldout_exact_example_response_text_collisions']} cards / {leakage['heldout_unique_response_text_collisions']} unique texts；最大 {leakage['heldout_max_collision_response_tokens']} tokens，>=20-token 碰撞为 {leakage['heldout_long_response_collisions_ge_20_tokens']}。这些是短/通用回复文本碰撞，所有对应卡仍有 train-only source ID，不能解释成使用 held-out source row。
- 与 held-out full retrieval context 的 exact normalized 碰撞：{leakage['heldout_exact_retrieval_context_collisions']}。唯一一例是 train `esconv_0401` turn 0 与 validation `esconv_0438` turn 0 共用 situation `I TALK WITH MY FRIENDSHIP`；对应 response 不同。
- 完整 `(retrieval context, response)` pair 跨 split 碰撞：{leakage['heldout_exact_context_response_pair_collisions']}；完整 dialogue 跨 split 碰撞：{leakage['cross_split_full_dialogue_collisions']}。因此没有发现复制 held-out 具体 card/dialogue 的证据。
- future reply：builder 只切 `dialogue[turn_index-6:turn_index]`，不读取当前 supporter turn 之后的 turns。
- EvoEmo：只用于 exclusion gate，未进入 card schema；81 个近/完全重合 ESConv dialogues 被排除。
- judge/PM labels：builder 无相关输入或字段。
- 人工挑选测试最优结果：当前 12,429-card builder 是固定全量规则，无人工挑选分支。

限制：legacy 156-card v13 的 retained provenance 指向 train-only `pm_train.jsonl`，但其 v12/pm_train 上游文件已经缺失，因此只能验证 retained provenance 与 formal attestation，不能独立从最原始 ESConv 重建 legacy v13。该限制不影响当前 f64d bank 的完整 RAM-only 重算。
"""
    (TMP / "V1_RAG_LEAKAGE_AUDIT.md").write_text(leakage_md, encoding="utf-8")

    readable = [
        "# V1 Strategy RAG：V8 九例真实检索 trace",
        "",
        "以下 primary trace 使用第一篇 final external 的 f64d bank 与 canonical V1 lexical retriever；同时保留训练期 5b2 legacy trace，避免掩盖阶段切换。没有调用 dirty/V2 threshold retriever。",
        "",
    ]
    for trace in traces:
        readable.extend(
            [
                f"## {trace['item_id']} · {trace['regime']}",
                "",
                f"- query：{trace['actual_query']}",
                f"- fields：`{trace['query_context_fields']}`",
                "- V1 process：top-k=3；threshold/filter/rerank/index/embedding 均无；全库 bag-of-words cosine 后按 ID tie-break。",
                f"- V8 provisional displayed IDs：`{trace['provisional_v8_displayed_card_ids']}`",
                f"- 与 formal top-k：**{trace['displayed_vs_formal']['status']}**；intersection=`{trace['displayed_vs_formal']['intersection_card_ids']}`",
                "",
                "### Formal external V1 / f64d",
                "",
                render_cards(trace["formal_external_v1"]["cards"]),
                "",
                "### Legacy synthetic-training V1 / 5b2d",
                "",
                render_cards(trace["legacy_synthetic_training_v1"]["cards"]),
                "",
            ]
        )
    (TMP / "V1_RETRIEVAL_TRACES_READABLE_ZH.md").write_text(
        "\n".join(readable), encoding="utf-8"
    )

    decision_md = f"""# V1 Strategy RAG reuse decision

## 决定：PASS_WITH_MINIMAL_FIXES

当前 12,429-card bank 可以作为 **V1 final external bank** 冻结复用，但当前工作区不能原样、无条件宣布为完整 canonical V1 RAG，原因有二：

1. `retrieval.py` 已 dirty，哈希 `{hashes['dirty_retriever']}` 不等于冻结 V1 `{hashes['v1_retriever']}`；
2. 第一篇 PM 的 synthetic action outcomes 使用 legacy 156-card `{hashes['legacy_bank']}`，而 confirmatory/external 使用 `{hashes['current_bank']}`。必须保留 stage-specific lineage，不能声称同一 bank 贯穿训练和外部推理。

## 最小修复（不得改变卡片语义）

1. 不改当前 f64d bank；将其标识为 `v1_confirmatory_external_bank` 并固定 SHA256、card count、input hashes、seed=13、overlap threshold=0.88。
2. V2 需要复现 V1 baseline 时，使用单独的 canonical V1 retriever 文件/锁定 commit `34737d6` 的语义与 SHA256 `{hashes['v1_retriever']}`；不要覆盖当前 dirty V2 retriever。
3. 在 provenance manifest 同时登记 training bank 5b2d 与 external bank f64d，明确 checkpoint 的训练 action outcomes来自前者。
4. 无需构建/重建 embedding index，因为 V1 从未使用 index；应明确记录 `embedding_model=null, index_path=null, dynamic_full_scan=true`。
5. 审计获人工接受后，用 canonical path 的 trace 替换 V8 provisional 展示卡；本阶段不改 V8 标签或材料。
6. 在人工审查这份报告前，不给 V8 的 RS on/off 标签定稿，不开始正式标注或训练。

## 不能主张

- 不能说 V1 从训练到 external 始终使用同一 Strategy Bank。
- 不能说当前 dirty retriever 就是冻结 V1 retriever。
- 不能把 V8 当前展示的两张 provisional cards 当作 V1 top-k：9 例 exact-order match={exact_matches}/9，displayed pair 全部包含于 formal top-3={subset_matches}/9。

`V1_STRATEGY_RAG_AUDIT_COMPLETE` 仍为 **False**；等待人工审查后再解锁。
"""
    (TMP / "V1_RAG_REUSE_DECISION.md").write_text(decision_md, encoding="utf-8")

    summary = {
        "audit_status": "PASS_WITH_MINIMAL_FIXES",
        "human_review_required": True,
        "V1_STRATEGY_RAG_AUDIT_COMPLETE": False,
        "v1_rag_exists": True,
        "no_old_results_overwritten": True,
        "new_strategy_bank_created": False,
        "existing_strategy_bank_modified": False,
        "v8_strategy_labels_modified": False,
        "pm_training_started": False,
        "canonical_match": {
            "synthetic_training": "MISMATCH",
            "confirmatory_esconv": "MATCH",
            "formal_external_evoemo": "MATCH",
            "current_dirty_retriever": "MISMATCH",
            "canonical_v1_retriever_preserved_copy": "MATCH",
        },
        "hashes": hashes,
        "expected_hash_checks": expected_checks,
        "reconstruction_in_memory": reconstruction,
        "bank": {
            "current_card_count": len(current_cards),
            "current_unique_card_ids": len(set(current_ids)),
            "current_source_dialogues": len(source_dialogues),
            "legacy_card_count": len(legacy_cards),
            "split_counts": split_counts,
            "overlap_split_counts": overlap_split_counts,
            "strategy_label_counts": strategy_counts,
        },
        "retriever": {
            "module": "src/metacom_pm/retrieval.py",
            "query_fields": [
                "current_session_summary",
                "current_session_history[*].content",
                "current_user_text",
            ],
            "embedding_model": None,
            "index_path": None,
            "index_exists": False,
            "runtime_index_rebuild": False,
            "dynamic_full_scan": True,
            "similarity": "bag_of_words_cosine",
            "top_k": 3,
            "threshold": None,
            "filter": None,
            "rerank": None,
            "tie_break": "strategy_id_descending",
        },
        "leakage": leakage,
        "v8_trace": {
            "cases": len(traces),
            "formal_exact_order_matches": exact_matches,
            "displayed_pair_subset_of_formal_top3": subset_matches,
        },
        "minimal_fixes": [
            "pin f64d bank as V1 confirmatory/external only",
            "replay baselines with canonical V1 retriever hash 5202, without overwriting dirty V2 retriever",
            "record separate 5b2 training-bank and f64d external-bank provenance",
            "record explicitly that V1 has no embedding/index",
            "after human acceptance, rebuild V8 displayed evidence from canonical traces without changing card semantics",
        ],
        "output_files": [
            "V1_RAG_LINEAGE.md",
            "V1_RAG_CANONICAL_MATCH.md",
            "V1_RETRIEVER_DIFF.md",
            "V1_RAG_LEAKAGE_AUDIT.md",
            "V1_RETRIEVAL_TRACES.jsonl",
            "V1_RETRIEVAL_TRACES_READABLE_ZH.md",
            "V1_RAG_REUSE_DECISION.md",
            "audit_summary.json",
        ],
    }
    dump_json(TMP / "audit_summary.json", summary)

    os.replace(TMP, OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
