"""V9 smoke-review preparation using the real frozen-candidate Strategy RAG.

V9 is deliberately non-training and non-confirmatory.  It compares both V1
banks, pins a provisional external-bank candidate, retrieves real top-3 cards,
and plans/runs a strictly paired R0/RS response-generation smoke test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .api import OpenAICompatibleClient, require_reported_usage
from .artifacts import create_artifact_attestation
from .attempt_ledger import (
    PersistentAttemptLedger,
    physical_call_key,
    reported_prompt_token_error,
)
from .config import endpoint_from_config, load_config
from .contracts import StrategyCard
from .generation_contract import SupporterGenerationContract
from .io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
    write_jsonl,
)
from .prompts import BASE_SUPPORTER_SYSTEM
from .retrieval_v1_canonical import (
    CANONICAL_V1_RETRIEVER_SHA256,
    CANONICAL_V1_TOP_K,
    StrategyRetrieverV1Canonical,
    context_query,
)


V9_PROTOCOL = (
    "pm-v2-generation-semantic-review-v9-real-rag-paired-smoke-"
    "v2-treatment-bound"
)
V9_STATUS_PENDING = "PAIR_GENERATION_PENDING_NOT_READY_FOR_REVIEW"
V9_STATUS_CANDIDATE = "CANDIDATE_FOR_SMOKE_REVIEW_ONLY"
CURRENT_BANK_SHA256 = (
    "f64ded1e23b4a79c0e08c6b47355ad76b3880d5cbc0bfa7530f366149b5c70db"
)
LEGACY_BANK_SHA256 = (
    "5b2d58c6be07d3c27e20e9162a3494f47f875d02b94b983f8b05c08ff044bf46"
)
CURRENT_BANK_COUNT = 12429
LEGACY_BANK_COUNT = 156
PAIR_STAGE = "pm_v2_v9_r0_rs_paired_generation_smoke_v2_treatment_bound"

BASE_REVIEW_FIELDS = (
    "semantic_family_match",
    "regime_match",
    "memory_sources_marginal_value_match",
    "memory_item_utility_match",
    "source_type_match",
    "dialogue_temporal_order_match",
    "context_grounding_match",
    "memory_age_design_match",
    "surface_naturalness_match",
)
PART_A_FIELDS = (
    "strategy_retrieval_relevance_match",
    "strategy_retrieval_set_usable",
    "strategy_label_fidelity_match",
    "strategy_context_stage_fit_match",
    "strategy_safety_match",
)
PART_B_FIELDS = (
    "pair_is_diagnostic",
    "response_preference",
    "emotional_support_better",
    "contextual_fit_better",
    "non_intrusiveness_better",
    "advice_readiness_match",
)
CARD_UTILITY_FIELDS = (
    "strategy_card_1_utility",
    "strategy_card_2_utility",
    "strategy_card_3_utility",
)
REVIEW_FIELDS = (
    *BASE_REVIEW_FIELDS,
    *PART_A_FIELDS,
    *CARD_UTILITY_FIELDS,
    *PART_B_FIELDS,
)


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)


def _load_cards(path: Path, expected_hash: str, expected_count: int) -> list[StrategyCard]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"strategy bank hash mismatch at {path}: {actual_hash}")
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(path)]
    if len(cards) != expected_count:
        raise RuntimeError(
            f"strategy bank count mismatch at {path}: {len(cards)} != {expected_count}"
        )
    ids = [card.strategy_id for card in cards]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"strategy bank has duplicate IDs: {path}")
    return cards


def _card_trace(score: float, card: StrategyCard, rank: int) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "card_id": card.strategy_id,
        "similarity_score": float(score),
        "strategy_type": card.strategy_label,
        "retrieval_text": card.retrieval_text,
        "guidance_text": card.guidance_text,
        "example_response": card.example_response,
        "source_dialogue_id": card.source_dialogue_id,
        "source_turn_index": int(card.source_turn_index),
    }


def _retrieve(
    retriever: StrategyRetrieverV1Canonical, query: str
) -> list[dict[str, Any]]:
    return [
        _card_trace(score, card, rank)
        for rank, (score, card) in enumerate(
            retriever.retrieve_with_scores(query), 1
        )
    ]


def _duplicate_information(
    legacy: Sequence[dict[str, Any]], current: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    def duplicates(rows: Sequence[dict[str, Any]], key: str) -> list[str]:
        values: dict[str, list[str]] = {}
        for row in rows:
            normalized = " ".join(str(row[key]).lower().split())
            values.setdefault(normalized, []).append(str(row["card_id"]))
        return sorted(
            card_id
            for ids in values.values()
            if len(ids) > 1
            for card_id in ids
        )

    exact_cross: list[dict[str, str]] = []
    for left in legacy:
        for right in current:
            shared = []
            for key in ("retrieval_text", "guidance_text", "example_response"):
                if " ".join(str(left[key]).lower().split()) == " ".join(
                    str(right[key]).lower().split()
                ):
                    shared.append(key)
            if shared:
                exact_cross.append(
                    {
                        "bank_156_card_id": str(left["card_id"]),
                        "bank_12429_card_id": str(right["card_id"]),
                        "exact_shared_fields": ",".join(shared),
                    }
                )
    return {
        "bank_156_duplicate_retrieval_card_ids": duplicates(
            legacy, "retrieval_text"
        ),
        "bank_156_duplicate_response_card_ids": duplicates(
            legacy, "example_response"
        ),
        "bank_12429_duplicate_retrieval_card_ids": duplicates(
            current, "retrieval_text"
        ),
        "bank_12429_duplicate_response_card_ids": duplicates(
            current, "example_response"
        ),
        "cross_bank_exact_text_matches": exact_cross,
        "shared_card_ids": sorted(
            {str(row["card_id"]) for row in legacy}
            & {str(row["card_id"]) for row in current}
        ),
    }


def _render_trace_cards(cards: Sequence[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for card in cards:
        rows.extend(
            [
                f"{card['rank']}. `{card['card_id']}` · score={card['similarity_score']:.9f} · {card['strategy_type']}",
                f"   - retrieval: {card['retrieval_text']}",
                f"   - guidance: {card['guidance_text']}",
                f"   - example: {card['example_response']}",
                f"   - provenance: {card['source_dialogue_id']} / turn {card['source_turn_index']}",
            ]
        )
    return rows


def _render_memory_items(
    title: str, items: Sequence[Mapping[str, Any]]
) -> list[str]:
    rows = [f"#### {title}", ""]
    if not items:
        return [*rows, "- none", ""]
    for index, item in enumerate(items, 1):
        rows.extend(
            [
                f"{index}. `{item['memory_id']}` · source=`{item['source']}` · "
                f"design utility=`{item['utility']}`",
                f"   - text: {item['text']}",
                f"   - created session: {item['created_session']} · "
                f"age sessions: {item['age_sessions']}",
                f"   - stale: `{item['stale']}` · conflicts with current state: "
                f"`{item['conflicts_with_current_state']}` · sensitivity: "
                f"`{item['private_sensitivity']}`",
                f"   - provenance: `{item['provenance_type']}` / "
                f"`{item['provenance_source']}` / session "
                f"{item['provenance_session']}",
            ]
        )
    rows.append("")
    return rows


def _render_memory_evidence(case: Mapping[str, Any]) -> list[str]:
    useful = list(case.get("materially_useful_memory_sources") or [])
    correction = case.get("current_correction") or "none"
    return [
        "### Memory and provenance evidence",
        "",
        f"- semantic family: `{case['semantic_family']}`",
        f"- semantic subtopics: `{canonical_json(case['semantic_subtopics'])}`",
        f"- authorized user context: {case['authorized_user_context']}",
        f"- current correction: {correction}",
        f"- candidate materially useful sources: `{canonical_json(useful)}`",
        f"- memory coverage rationale: {case['memory_coverage_rationale']}",
        f"- context provenance: `{canonical_json(case['context_provenance'])}`",
        "",
        *_render_memory_items("MP · profile memories", case["profile_memories"]),
        *_render_memory_items("MS · summary memories", case["summary_memories"]),
        *_render_memory_items("ME · event memories", case["event_memories"]),
    ]


def annotation_guidelines_text(*, pair_complete: bool) -> str:
    status = (
        "PAIR GENERATION COMPLETE / READY FOR BLIND SMOKE REVIEW / "
        "NOT APPROVED FOR FORMAL ANNOTATION OR TRAINING"
        if pair_complete
        else "PAIR GENERATION PENDING / PART A PREVIEW ONLY"
    )
    part_b_gate = (
        "配对生成已经完成，可以填写盲态 Part B。"
        if pair_complete
        else "匿名 Response A/B 尚未生成，不得填写 Part B。"
    )
    return f"""# PM V2 V9 双人独立 smoke-review 指南

状态：`{status}`

{part_b_gate} 这仍是九例 smoke review，不是正式标注放行。两名审查者必须独立填写各自 CSV，不得查看 `private_condition_mapping.jsonl`，不得复用 V7/V8 或他人的答案。

## Part A：Memory 与真实 retrieval 审查

Memory 字段依据同一 reviewer packet 中展示的 MP/MS/ME、age、utility、provenance、authorized context 与 correction 填写 `1/0`。不能只看当前话语或凭印象填写。

Strategy 字段：

- `strategy_retrieval_relevance_match`：top-3 中至少有一张卡与当前问题直接相关；主题词偶然重合不算相关；
- `strategy_retrieval_set_usable`：top-3 中至少一张 `helpful`，且没有 `harmful`；
- `strategy_label_fidelity_match`：三张卡的 strategy type 是否准确描述其 guidance/example 内容；只检查标签忠实度，不检查时机；
- `strategy_context_stage_fit_match`：检索结果的直接性、建议时机和支持阶段是否适合当前对话；
- `strategy_safety_match`：检索结果是否避免边界侵犯、敏感上下文污染、错误信息和高风险建议。

逐卡填写 `strategy_card_1_utility`、`strategy_card_2_utility`、`strategy_card_3_utility`：

- `helpful`：与当前问题相关、时机合适，并有合理机会改善下一条回复；
- `irrelevant`：缺乏边际帮助，但本身不太可能把回复推向有害方向；
- `harmful`：可能因过早/过度指令、违背明确边界、敏感上下文污染、误导或侵入而降低回复质量或安全性；
- `uncertain`：现有材料不足以可靠归类。

## Part B：匿名 R0/RS 成对回复审查

- `response_preference`、`emotional_support_better`、`contextual_fit_better`、`non_intrusiveness_better`：填 `A / B / tie / uncertain`；
- `pair_is_diagnostic`：只有 A/B 存在足以支持稳定边际判断的实质差异时填 `1`；近乎相同、差异仅是措辞波动或无法可靠判断时填 `0`；
- `advice_readiness_match`：回复的直接性阶段是否符合当前对话，填 `1/0`。

盲审 CSV **不包含** `strategy_rag_marginal_value`。该字段只能在两名审查者完成、分歧处理并揭盲后由脚本派生：诊断充分且偏好 RS 为 `positive`，偏好 R0 为 `negative`，诊断充分且实质相当为 `neutral`；无诊断力或无法判断为 `uncertain`。不要在 notes 中猜测 A/B 的条件身份。
"""


def derive_strategy_rag_marginal_value(
    *, response_preference: str, pair_is_diagnostic: int, blind_mapping: Mapping[str, str]
) -> str:
    """Derive RAG direction after unblinding; never expose it in blind packets."""

    preference = str(response_preference).strip()
    if int(pair_is_diagnostic) not in {0, 1}:
        raise ValueError("pair_is_diagnostic must be 0 or 1")
    if preference not in {"A", "B", "tie", "uncertain"}:
        raise ValueError("response_preference must be A/B/tie/uncertain")
    if set(blind_mapping) != {"A", "B"} or set(blind_mapping.values()) != {
        "R0",
        "RS",
    }:
        raise ValueError("blind_mapping must be a bijection between A/B and R0/RS")
    if int(pair_is_diagnostic) == 0 or preference == "uncertain":
        return "uncertain"
    if preference == "tie":
        return "neutral"
    return "positive" if blind_mapping[preference] == "RS" else "negative"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def prepare_bank_comparison(
    *,
    v8_cases_path: Path,
    legacy_bank_path: Path,
    current_bank_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    _require_new_directory(out_dir)
    cases = read_json(v8_cases_path)
    if not isinstance(cases, list) or len(cases) != 9:
        raise RuntimeError("bank comparison requires exactly nine V8 smoke cases")
    legacy_cards = _load_cards(
        legacy_bank_path, LEGACY_BANK_SHA256, LEGACY_BANK_COUNT
    )
    current_cards = _load_cards(
        current_bank_path, CURRENT_BANK_SHA256, CURRENT_BANK_COUNT
    )
    legacy_retriever = StrategyRetrieverV1Canonical(legacy_cards)
    current_retriever = StrategyRetrieverV1Canonical(current_cards)
    traces: list[dict[str, Any]] = []
    for case in cases:
        query = context_query(
            case["current_user_text"],
            case["dialogue_before_current"],
            case["session_summary"],
        )
        legacy = _retrieve(legacy_retriever, query)
        current = _retrieve(current_retriever, query)
        traces.append(
            {
                "item_id": case["item_id"],
                "case_id": case["case_id"],
                "regime": case["regime"],
                "current_user_text": case["current_user_text"],
                "dialogue_history": case["dialogue_before_current"],
                "session_summary": case["session_summary"],
                "actual_query": query,
                "query_fields": [
                    "session_summary",
                    "dialogue_history[*].content",
                    "current_user_text",
                ],
                "retriever_behavior": {
                    "canonical_source_sha256": CANONICAL_V1_RETRIEVER_SHA256,
                    "similarity": "runtime_bag_of_words_cosine",
                    "top_k": 3,
                    "threshold": None,
                    "reranker": None,
                    "persistent_index": None,
                },
                "v1_training_legacy_bank": {
                    "sha256": LEGACY_BANK_SHA256,
                    "card_count": LEGACY_BANK_COUNT,
                    "top_3": legacy,
                },
                "v1_confirmatory_external_bank": {
                    "sha256": CURRENT_BANK_SHA256,
                    "card_count": CURRENT_BANK_COUNT,
                    "top_3": current,
                },
                "duplicate_information": _duplicate_information(legacy, current),
                "human_review": {
                    "bank_156_relevance": "",
                    "bank_12429_relevance": "",
                    "bank_156_noise": "",
                    "bank_12429_noise": "",
                    "preferred_bank": "",
                    "notes": "",
                },
            }
        )
    trace_path = out_dir / "bank_comparison_traces.jsonl"
    write_jsonl(trace_path, traces)
    summary_fields = (
        "item_id",
        "case_id",
        "regime",
        "bank_156_top3_ids",
        "bank_156_strategy_types",
        "bank_156_scores",
        "bank_12429_top3_ids",
        "bank_12429_strategy_types",
        "bank_12429_scores",
        "cross_bank_exact_match_count",
        "bank_156_relevance",
        "bank_12429_relevance",
        "bank_156_noise",
        "bank_12429_noise",
        "preferred_bank",
        "notes",
    )
    summary_rows = []
    for trace in traces:
        legacy = trace["v1_training_legacy_bank"]["top_3"]
        current = trace["v1_confirmatory_external_bank"]["top_3"]
        summary_rows.append(
            {
                "item_id": trace["item_id"],
                "case_id": trace["case_id"],
                "regime": trace["regime"],
                "bank_156_top3_ids": "|".join(row["card_id"] for row in legacy),
                "bank_156_strategy_types": "|".join(
                    row["strategy_type"] for row in legacy
                ),
                "bank_156_scores": "|".join(
                    f"{row['similarity_score']:.9f}" for row in legacy
                ),
                "bank_12429_top3_ids": "|".join(
                    row["card_id"] for row in current
                ),
                "bank_12429_strategy_types": "|".join(
                    row["strategy_type"] for row in current
                ),
                "bank_12429_scores": "|".join(
                    f"{row['similarity_score']:.9f}" for row in current
                ),
                "cross_bank_exact_match_count": len(
                    trace["duplicate_information"]["cross_bank_exact_text_matches"]
                ),
                "bank_156_relevance": "",
                "bank_12429_relevance": "",
                "bank_156_noise": "",
                "bank_12429_noise": "",
                "preferred_bank": "",
                "notes": "",
            }
        )
    summary_path = out_dir / "bank_comparison_summary.csv"
    _write_csv(summary_path, summary_rows, summary_fields)
    readable = [
        "# V1 两套 Strategy Bank 并排检索",
        "",
        "本材料不自动判断哪套更好。两边使用完全相同的 canonical V1 5202ca lexical retriever。",
        "",
    ]
    for trace in traces:
        readable.extend(
            [
                f"## {trace['item_id']} · {trace['regime']}",
                "",
                f"- current user: {trace['current_user_text']}",
                f"- dialogue history: `{canonical_json(trace['dialogue_history'])}`",
                f"- actual query: {trace['actual_query']}",
                "",
                "### A. 156-card · v1_training_legacy_bank",
                "",
                *_render_trace_cards(
                    trace["v1_training_legacy_bank"]["top_3"]
                ),
                "",
                "### B. 12,429-card · v1_confirmatory_external_bank",
                "",
                *_render_trace_cards(
                    trace["v1_confirmatory_external_bank"]["top_3"]
                ),
                "",
                "### Duplicate information",
                "",
                f"`{canonical_json(trace['duplicate_information'])}`",
                "",
                "### 人工填写",
                "",
                "- bank_156_relevance:",
                "- bank_12429_relevance:",
                "- bank_156_noise:",
                "- bank_12429_noise:",
                "- preferred_bank:",
                "- notes:",
                "",
            ]
        )
    readable_path = out_dir / "bank_comparison_readable_ZH.md"
    readable_path.write_text("\n".join(readable), encoding="utf-8")
    decision_path = out_dir / "BANK_COMPARISON_DECISION_TEMPLATE.md"
    decision_path.write_text(
        """# Bank comparison decision template

状态：`AWAITING_INDEPENDENT_HUMAN_REVIEW`

- reviewer:
- review date:
- preferred bank (`v1_training_legacy_bank` / `v1_confirmatory_external_bank` / `undecided`):
- relevance evidence:
- noise evidence:
- safety concerns:
- decision rationale:
- unresolved disagreements:

本模板为空，不构成自动 bank 选择。
""",
        encoding="utf-8",
    )
    attestation = create_artifact_attestation(
        out_dir / "artifact_attestation.json",
        stage="strategy_rag_v1_bank_comparison",
        inputs={
            "v8_cases": v8_cases_path,
            "legacy_bank": legacy_bank_path,
            "current_bank": current_bank_path,
            "canonical_wrapper": Path(__file__).with_name(
                "retrieval_v1_canonical.py"
            ),
        },
        outputs={
            "traces": (trace_path, True),
            "readable": (readable_path, False),
            "summary": (summary_path, False),
            "decision_template": (decision_path, False),
        },
        parameters={
            "automatic_preferred_bank": None,
            "top_k": 3,
            "threshold": None,
            "reranker": None,
        },
        expected={"cases": 9},
    )
    return {"cases": 9, "attestation": attestation}


def prepare_frozen_candidate(
    *,
    current_bank_path: Path,
    split_manifest_path: Path,
    bank_audit_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    _require_new_directory(out_dir)
    _load_cards(current_bank_path, CURRENT_BANK_SHA256, CURRENT_BANK_COUNT)
    wrapper_path = Path(__file__).with_name("retrieval_v1_canonical.py")
    manifest = {
        "status": "PROVISIONAL_FROZEN_CANDIDATE_AWAITING_HUMAN_APPROVAL",
        "gate_unlocked": False,
        "bank": {
            "logical_name": "v1_confirmatory_external_bank",
            "path": str(current_bank_path.resolve()),
            "sha256": CURRENT_BANK_SHA256,
            "card_count": CURRENT_BANK_COUNT,
            "source": "ESConv train-only with EvoEmo overlap exclusion",
        },
        "legacy_bank": {
            "logical_name": "v1_training_legacy_bank",
            "sha256": LEGACY_BANK_SHA256,
            "card_count": LEGACY_BANK_COUNT,
            "mixed_with_candidate": False,
        },
        "retriever": {
            "canonical_source_sha256": CANONICAL_V1_RETRIEVER_SHA256,
            "compatibility_wrapper_path": str(wrapper_path.resolve()),
            "compatibility_wrapper_sha256": sha256_file(wrapper_path),
            "similarity": "runtime_bag_of_words_cosine",
            "top_k": 3,
            "threshold": None,
            "reranker": None,
            "embedding_model": None,
            "persistent_index": None,
        },
        "inputs": {
            "split_manifest": {
                "path": str(split_manifest_path.resolve()),
                "sha256": sha256_file(split_manifest_path),
            },
            "bank_audit": {
                "path": str(bank_audit_path.resolve()),
                "sha256": sha256_file(bank_audit_path),
            },
        },
        "prohibitions": [
            "not approved for formal annotation",
            "not approved for 27-case generation",
            "not approved for PM training",
            "does not modify V1_STRATEGY_RAG_AUDIT_COMPLETE",
        ],
    }
    manifest["manifest_sha256"] = sha256_text(canonical_json(manifest))
    manifest_path = out_dir / "strategy_rag_manifest.json"
    write_json(manifest_path, manifest)
    manifest_md = out_dir / "strategy_rag_manifest.md"
    manifest_md.write_text(
        f"""# Provisional V2 Strategy RAG frozen candidate

- status: `{manifest['status']}`
- bank: `v1_confirmatory_external_bank`
- bank SHA256: `{CURRENT_BANK_SHA256}`
- cards: {CURRENT_BANK_COUNT}
- retriever source behavior: `{CANONICAL_V1_RETRIEVER_SHA256}`
- wrapper SHA256: `{manifest['retriever']['compatibility_wrapper_sha256']}`
- top-k: 3
- threshold/reranker/embedding/index: none
- gate unlocked: **False**

`v1_training_legacy_bank` 仅保留为训练期 lineage，不与候选卡库混用。
""",
        encoding="utf-8",
    )
    hash_list = out_dir / "bank_file_list.sha256"
    hash_list.write_text(
        "\n".join(
            [
                f"{sha256_file(current_bank_path)}  {current_bank_path.resolve()}",
                f"{sha256_file(split_manifest_path)}  {split_manifest_path.resolve()}",
                f"{sha256_file(bank_audit_path)}  {bank_audit_path.resolve()}",
                f"{sha256_file(wrapper_path)}  {wrapper_path.resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    behavior = out_dir / "retriever_behavior_spec.md"
    behavior.write_text(
        f"""# Canonical V1 retriever behavior specification

- authority: frozen source SHA256 `{CANONICAL_V1_RETRIEVER_SHA256}`
- query order: summary, every visible history content, current user text
- whitespace: collapse all runs to one space
- tokenizer: regex `[A-Za-z0-9']+`, lowercase
- score: cosine of token-count vectors
- ranking: `(score, strategy_id)` descending
- top-k: exactly 3
- threshold: none
- filter/reranker: none
- embedding/index: none
- execution: runtime JSONL load and full scan

The compatibility wrapper exposes no threshold or abstention parameter.
""",
        encoding="utf-8",
    )
    reproduction = out_dir / "reproduction_command.sh"
    reproduction.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd {current_bank_path.resolve().parents[2]}
sha256sum {current_bank_path.resolve()} {split_manifest_path.resolve()} {bank_audit_path.resolve()} {wrapper_path.resolve()}
PYTHONNOUSERSITE=1 PYTHONPATH=src /home/tokkio/miniconda3/envs/sim_eval/bin/python - <<'PY'
from metacom_pm.retrieval_v1_canonical import CANONICAL_V1_RETRIEVER_SHA256, CANONICAL_V1_TOP_K
assert CANONICAL_V1_RETRIEVER_SHA256 == "{CANONICAL_V1_RETRIEVER_SHA256}"
assert CANONICAL_V1_TOP_K == 3
print("PROVISIONAL_CANDIDATE_REPRODUCTION_CHECK_PASS")
PY
""",
        encoding="utf-8",
    )
    status = out_dir / "PROVISIONAL_FREEZE_STATUS.md"
    status.write_text(
        """# PROVISIONAL FREEZE STATUS

`PROVISIONAL / HUMAN APPROVAL REQUIRED / GATE LOCKED`

This candidate is not a formal study freeze. It does not authorize V9 formal
annotation, 27-case generation, or PM training.
""",
        encoding="utf-8",
    )
    return manifest


def _memory_only_rationale(case: Mapping[str, Any]) -> str:
    sources = list(case.get("materially_useful_memory_sources") or [])
    if sources:
        return (
            "Candidate memory marginal value is restricted to the listed sources: "
            + ", ".join(str(source) for source in sources)
            + ". Strategy-resource value is intentionally unlabelled."
        )
    return (
        "No candidate memory source is assigned material marginal value. "
        "Strategy-resource value is intentionally unlabelled."
    )


def _generator_messages(
    case: Mapping[str, Any],
    strategy_cards: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = BASE_SUPPORTER_SYSTEM,
) -> list[dict[str, str]]:
    history = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in case["dialogue_before_current"]
    )
    sections = [
        f"Current-session summary:\n{case['session_summary']}\n\n"
        f"Recent dialogue:\n{history}\n\n"
        f"Current user message:\n{case['current_user_text']}"
    ]
    if strategy_cards:
        lines = [
            f"- {card['guidance_text']}\n  Example style (adapt, do not copy): {card['example_response']}"
            for card in strategy_cards
        ]
        sections.append(
            "Potential emotional-support guidance. Use only when fitting:\n"
            + "\n".join(lines)
        )
    sections.append("Write only the counselor's next response.")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def _pair_seed(base_seed: int, case_id: str) -> int:
    value = hashlib.sha256(
        f"pm-v2-v9-paired-seed\x1f{base_seed}\x1f{case_id}".encode("utf-8")
    ).hexdigest()
    return int(value[:8], 16) % (2**31 - 1)


def _blind_order(case_id: str, seed: int) -> dict[str, str]:
    value = hashlib.sha256(
        f"pm-v2-v9-blind-order\x1f{seed}\x1f{case_id}".encode("utf-8")
    ).digest()[0]
    return (
        {"A": "R0", "B": "RS"}
        if value % 2 == 0
        else {"A": "RS", "B": "R0"}
    )


def _reviewer_rows(cases: Sequence[Mapping[str, Any]], annotator: str) -> list[dict[str, str]]:
    rows = []
    for case in cases:
        row = {"item_id": str(case["item_id"])}
        row.update({field: "" for field in REVIEW_FIELDS})
        row.update({"annotator_id": annotator, "notes": ""})
        rows.append(row)
    return rows


def prepare_v9(
    *,
    v8_cases_path: Path,
    current_bank_path: Path,
    experiment_config_path: Path,
    pm_v2_config_path: Path,
    frozen_manifest_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    _require_new_directory(out_dir)
    frozen = read_json(frozen_manifest_path)
    if frozen.get("status") != "PROVISIONAL_FROZEN_CANDIDATE_AWAITING_HUMAN_APPROVAL":
        raise RuntimeError("V9 requires the locked provisional frozen candidate")
    if frozen.get("gate_unlocked") is not False:
        raise RuntimeError("provisional Strategy RAG gate unexpectedly unlocked")
    cards = _load_cards(current_bank_path, CURRENT_BANK_SHA256, CURRENT_BANK_COUNT)
    retriever = StrategyRetrieverV1Canonical(cards)
    v8_cases = read_json(v8_cases_path)
    if not isinstance(v8_cases, list) or len(v8_cases) != 9:
        raise RuntimeError("V9 smoke review requires exactly nine V8 cases")
    v9_cases: list[dict[str, Any]] = []
    for index, source in enumerate(v8_cases, 1):
        query = context_query(
            source["current_user_text"],
            source["dialogue_before_current"],
            source["session_summary"],
        )
        retrieved = _retrieve(retriever, query)
        case = {
            key: value
            for key, value in source.items()
            if key not in {"strategy_target", "strategy_evidence", "coverage_rationale"}
        }
        case["item_id"] = f"v9_item_{index:02d}"
        case["source_v8_item_id"] = source["item_id"]
        case["memory_coverage_rationale"] = _memory_only_rationale(source)
        case["advice_readiness_candidate"] = source["strategy_target"][
            "advice_readiness"
        ]
        case["actual_retrieval_query"] = query
        case["strategy_retrieval"] = {
            "bank_logical_name": "v1_confirmatory_external_bank",
            "bank_sha256": CURRENT_BANK_SHA256,
            "canonical_retriever_source_sha256": CANONICAL_V1_RETRIEVER_SHA256,
            "top_k": 3,
            "threshold": None,
            "reranker": None,
            "persistent_index": None,
            "actual_order": [row["card_id"] for row in retrieved],
            "cards": retrieved,
            "automatic_item_utility_labels": None,
            "automatic_resource_need_label": None,
        }
        v9_cases.append(case)
    cases_path = out_dir / "generation_pilot_semantic_review_cases.json"
    write_json(cases_path, v9_cases)

    experiment = load_config(experiment_config_path)
    pm_v2 = load_config(pm_v2_config_path)
    supporter_generation_contract = SupporterGenerationContract.from_config(pm_v2)
    sweep = pm_v2["development_sweep"]
    endpoint_name = supporter_generation_contract.generator_endpoint
    endpoint = endpoint_from_config(experiment, endpoint_name)
    temperature = supporter_generation_contract.temperature
    max_tokens = supporter_generation_contract.max_output_tokens
    base_seed = int(sweep["seed"])
    safety_factor = float(pm_v2["api_cost_planning"]["input_token_safety_factor"])
    pricing = sweep["pricing_usd_per_mtok"]
    call_plan: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for case in v9_cases:
        pair_seed = _pair_seed(base_seed, case["case_id"])
        mappings.append(
            {
                "item_id": case["item_id"],
                "case_id": case["case_id"],
                "blind_mapping": _blind_order(case["case_id"], base_seed),
                "mapping_seed": base_seed,
            }
        )
        for condition in ("R0", "RS"):
            strategy_cards = (
                [] if condition == "R0" else case["strategy_retrieval"]["cards"]
            )
            messages = _generator_messages(
                case,
                strategy_cards,
                system_prompt=supporter_generation_contract.system_prompt,
            )
            prompt_sha256 = sha256_text(canonical_json(messages))
            record_ids = {
                "item_id": case["item_id"],
                "condition": condition,
                "supporter_generation_treatment_sha256": (
                    supporter_generation_contract.digest()
                ),
            }
            params = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": pair_seed,
                "response_schema": None,
                "supporter_generation_treatment_sha256": (
                    supporter_generation_contract.digest()
                ),
            }
            estimated = max(1, math.ceil(len(canonical_json(messages)) / 4))
            upper = max(1, math.ceil(estimated * safety_factor))
            call_plan.append(
                {
                    **record_ids,
                    "case_id": case["case_id"],
                    "messages": messages,
                    "prompt_sha256": prompt_sha256,
                    "call_key": physical_call_key(
                        stage=PAIR_STAGE,
                        record_ids=record_ids,
                        prompt_sha256=prompt_sha256,
                        endpoint=endpoint,
                        request_parameters=params,
                    ),
                    "endpoint": {
                        "name": endpoint_name,
                        "base_url": endpoint.base_url,
                        "model": endpoint.model,
                        "family": endpoint.family,
                    },
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "seed": pair_seed,
                    "estimated_input_tokens": estimated,
                    "maximum_input_tokens": upper,
                    "strategy_card_ids": [
                        row["card_id"] for row in strategy_cards
                    ],
                    "supporter_generation_treatment": (
                        supporter_generation_contract.payload()
                    ),
                    "supporter_generation_treatment_sha256": (
                        supporter_generation_contract.digest()
                    ),
                }
            )
    if len({row["call_key"] for row in call_plan}) != 18:
        raise RuntimeError("V9 paired plan must contain 18 unique physical calls")
    plan_path = out_dir / "paired_generation_call_plan.jsonl"
    write_jsonl(plan_path, call_plan)
    mapping_path = out_dir / "private_condition_mapping.jsonl"
    write_jsonl(mapping_path, mappings)
    total_input = sum(row["maximum_input_tokens"] for row in call_plan)
    total_output = len(call_plan) * max_tokens
    estimate = {
        "protocol": "pm-v2-v9-paired-generation-cost-v2-treatment-bound",
        "stage": PAIR_STAGE,
        "expected_api_calls": len(call_plan),
        "maximum_physical_api_attempts": len(call_plan),
        "input_token_safety_factor": safety_factor,
        "maximum_input_tokens_per_call": max(
            row["maximum_input_tokens"] for row in call_plan
        ),
        "maximum_total_input_tokens": total_input,
        "maximum_total_output_tokens": total_output,
        "pricing_usd_per_mtok": {
            "input": float(pricing["input"]),
            "output": float(pricing["output"]),
        },
        "maximum_estimated_cost_usd": total_input / 1_000_000
        * float(pricing["input"])
        + total_output / 1_000_000 * float(pricing["output"]),
        "call_plan_sha256": sha256_file(plan_path),
        "supporter_generation_treatment": (
            supporter_generation_contract.payload()
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "generator": {
            "endpoint_name": endpoint_name,
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "paired_seed_rule": "same deterministic per-case seed for R0 and RS",
        },
    }
    estimate["cost_estimate_sha256"] = sha256_text(canonical_json(estimate))
    estimate_path = out_dir / "paired_generation_cost_estimate.json"
    write_json(estimate_path, estimate)
    config_path = out_dir / "paired_generation_config.json"
    write_json(
        config_path,
        {
            "protocol": V9_PROTOCOL,
            "status": V9_STATUS_PENDING,
            "only_strategy_rag_input_differs": True,
            "generator": estimate["generator"],
            "supporter_generation_treatment": (
                supporter_generation_contract.payload()
            ),
            "supporter_generation_treatment_sha256": (
                supporter_generation_contract.digest()
            ),
            "historical_v9_base_300_artifacts_reused": False,
            "formal_authorization": False,
            "bank_sha256": CURRENT_BANK_SHA256,
            "retriever_source_sha256": CANONICAL_V1_RETRIEVER_SHA256,
            "formal_gate_unlocked": False,
        },
    )

    retrieval_readable = [
        "# PM V2 V9 real-retrieval smoke cases",
        "",
        "状态：`PAIR GENERATION PENDING / NOT READY FOR HUMAN REVIEW`",
        "",
        "Strategy cards are actual canonical top-3. No utility or RS-need label is prefilled.",
        "",
    ]
    for case in v9_cases:
        retrieval_readable.extend(
            [
                f"## {case['item_id']} · {case['regime']}",
                "",
                f"- current user: {case['current_user_text']}",
                f"- query: {case['actual_retrieval_query']}",
                f"- advice readiness candidate: `{case['advice_readiness_candidate']}`（与 RS on/off 解耦）",
                "",
                *_render_trace_cards(case["strategy_retrieval"]["cards"]),
                "",
            ]
        )
    (out_dir / "retrieval_semantic_review_readable_ZH.md").write_text(
        "\n".join(retrieval_readable), encoding="utf-8"
    )
    packet_fields = (
        "item_id",
        "case_id",
        "regime",
        "semantic_family",
        "semantic_subtopics_json",
        "current_user_text",
        "dialogue_before_current_json",
        "session_summary",
        "authorized_user_context",
        "current_correction",
        "context_provenance_json",
        "materially_useful_memory_sources_json",
        "profile_memories_json",
        "summary_memories_json",
        "event_memories_json",
        "memory_coverage_rationale",
        "advice_readiness_candidate",
        "actual_retrieval_query",
        "strategy_retrieval_json",
    )
    packet_rows = []
    for case in v9_cases:
        packet_rows.append(
            {
                "item_id": case["item_id"],
                "case_id": case["case_id"],
                "regime": case["regime"],
                "semantic_family": case["semantic_family"],
                "semantic_subtopics_json": canonical_json(
                    case["semantic_subtopics"]
                ),
                "current_user_text": case["current_user_text"],
                "dialogue_before_current_json": canonical_json(
                    case["dialogue_before_current"]
                ),
                "session_summary": case["session_summary"],
                "authorized_user_context": case["authorized_user_context"],
                "current_correction": case.get("current_correction") or "",
                "context_provenance_json": canonical_json(
                    case["context_provenance"]
                ),
                "materially_useful_memory_sources_json": canonical_json(
                    case["materially_useful_memory_sources"]
                ),
                "profile_memories_json": canonical_json(case["profile_memories"]),
                "summary_memories_json": canonical_json(case["summary_memories"]),
                "event_memories_json": canonical_json(case["event_memories"]),
                "memory_coverage_rationale": case["memory_coverage_rationale"],
                "advice_readiness_candidate": case["advice_readiness_candidate"],
                "actual_retrieval_query": case["actual_retrieval_query"],
                "strategy_retrieval_json": canonical_json(
                    case["strategy_retrieval"]
                ),
            }
        )
    _write_csv(out_dir / "retrieval_review_packet.csv", packet_rows, packet_fields)
    reviewer_fields = ("item_id", *REVIEW_FIELDS, "annotator_id", "notes")
    _write_csv(
        out_dir / "reviewer_a.csv",
        _reviewer_rows(v9_cases, "reviewer_a"),
        reviewer_fields,
    )
    _write_csv(
        out_dir / "reviewer_b.csv",
        _reviewer_rows(v9_cases, "reviewer_b"),
        reviewer_fields,
    )
    (out_dir / "README.md").write_text(
        """# PM V2 generation semantic review V9

V9 uses real canonical top-3 retrieval and deletes V8's assumed Strategy card
selection/utility ground truth. Reviewer files contain no R0/RS mapping and no
prefilled scores. Part B must not be completed until paired generation succeeds.

See `annotation_guidelines_ZH.md` for the Part A/Part B field contracts.
""",
        encoding="utf-8",
    )
    (out_dir / "V9_STATUS.md").write_text(
        """# V9 status

`PAIR_GENERATION_PENDING / NOT READY FOR HUMAN REVIEW`

V9 is not approved for formal annotation, 27-case generation, or PM training.
""",
        encoding="utf-8",
    )
    return {
        "status": V9_STATUS_PENDING,
        "cases": len(v9_cases),
        "calls": len(call_plan),
        "cost_estimate_sha256": estimate["cost_estimate_sha256"],
        "maximum_estimated_cost_usd": estimate["maximum_estimated_cost_usd"],
    }


def validate_v9_dry_run(
    *,
    out_dir: Path,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    require_existing: bool = False,
) -> dict[str, Any]:
    estimate = read_json(out_dir / "paired_generation_cost_estimate.json")
    checks = {
        "api_calls": int(estimate["maximum_physical_api_attempts"])
        <= int(max_api_calls),
        "estimated_cost": float(estimate["maximum_estimated_cost_usd"])
        <= float(max_estimated_usd),
        "input_tokens_per_call": int(estimate["maximum_input_tokens_per_call"])
        <= int(max_input_tokens_per_call),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
        "cost_estimate": estimate,
    }
    if result["status"] != "PASS":
        raise RuntimeError(f"V9 paired-generation budget gate failed: {checks}")
    dry_run_path = out_dir / "paired_generation_dry_run.json"
    if require_existing:
        if not dry_run_path.is_file():
            raise RuntimeError("V9 API run requires a saved matching --dry-run")
        if read_json(dry_run_path) != result:
            raise RuntimeError("saved V9 paired-generation dry-run does not match")
    elif dry_run_path.exists():
        if read_json(dry_run_path) != result:
            raise RuntimeError("refusing to overwrite a different V9 dry-run")
    else:
        write_json(dry_run_path, result)
    return result


def _load_unique_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    for row in iter_jsonl(path):
        call_key = str(row["call_key"])
        if call_key in results:
            raise RuntimeError("paired generation output has duplicate call_key")
        results[call_key] = row
    return results


def _reviewers_are_blank(path: Path) -> bool:
    if not path.exists():
        return True
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field, value in row.items():
            if field in {"item_id", "annotator_id", "notes"}:
                if field == "notes" and str(value or "").strip():
                    return False
                continue
            if str(value or "").strip():
                return False
    return True


def _enrich_blinded_rows(
    cases: Mapping[str, Mapping[str, Any]],
    blinded: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for source in blinded:
        item_id = str(source["item_id"])
        case = cases[item_id]
        enriched.append(
            {
                "item_id": item_id,
                "case_id": case["case_id"],
                "regime": case["regime"],
                "semantic_family": case["semantic_family"],
                "semantic_subtopics": case["semantic_subtopics"],
                "current_user_text": case["current_user_text"],
                "dialogue_before_current": case["dialogue_before_current"],
                "session_summary": case["session_summary"],
                "authorized_user_context": case["authorized_user_context"],
                "current_correction": case.get("current_correction"),
                "context_provenance": case["context_provenance"],
                "materially_useful_memory_sources": case[
                    "materially_useful_memory_sources"
                ],
                "profile_memories": case["profile_memories"],
                "summary_memories": case["summary_memories"],
                "event_memories": case["event_memories"],
                "memory_coverage_rationale": case["memory_coverage_rationale"],
                "advice_readiness_candidate": case[
                    "advice_readiness_candidate"
                ],
                "actual_retrieval_query": case["actual_retrieval_query"],
                "retrieved_cards": case["strategy_retrieval"]["cards"],
                "response_A": source["response_A"],
                "response_B": source["response_B"],
                "mapping_hidden_from_reviewer": True,
            }
        )
    if len(enriched) != 9 or {row["item_id"] for row in enriched} != set(cases):
        raise RuntimeError("blind review material must contain exactly all nine cases")
    return sorted(enriched, key=lambda row: row["item_id"])


def _write_completed_review_materials(
    *, review_dir: Path, cases: Mapping[str, Mapping[str, Any]], blinded: Sequence[Mapping[str, Any]]
) -> dict[str, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    for reviewer in ("reviewer_a.csv", "reviewer_b.csv"):
        if not _reviewers_are_blank(review_dir / reviewer):
            raise RuntimeError(
                f"refusing to overwrite non-empty human annotations: {review_dir / reviewer}"
            )
    enriched = _enrich_blinded_rows(cases, blinded)
    blinded_path = review_dir / "paired_responses_blinded.jsonl"
    write_jsonl(blinded_path, enriched)

    readable = [
        "# PM V2 V9 anonymous R0/RS smoke review",
        "",
        "状态：`PAIR GENERATION COMPLETE / READY FOR BLIND SMOKE REVIEW / NOT APPROVED FOR FORMAL ANNOTATION OR TRAINING`",
        "",
        "Response A/B 顺序经过确定性随机化；本材料不包含 R0/RS 映射。Memory、provenance、真实 top-3 与 A/B 已合并在同一份材料中。",
        "",
    ]
    retrieval_only = [
        "# PM V2 V9 real-retrieval Part A smoke review",
        "",
        "状态：`PAIR GENERATION COMPLETE / READY FOR BLIND SMOKE REVIEW / NOT APPROVED FOR FORMAL ANNOTATION OR TRAINING`",
        "",
        "Strategy cards are actual canonical top-3. No utility or RS-need label is prefilled.",
        "",
    ]
    for row in enriched:
        shared = [
            f"## {row['item_id']} · {row['regime']}",
            "",
            f"- current user: {row['current_user_text']}",
            f"- dialogue: `{canonical_json(row['dialogue_before_current'])}`",
            f"- summary: {row['session_summary']}",
            f"- advice readiness candidate: `{row['advice_readiness_candidate']}`（不等于 RS on/off 标签）",
            f"- actual query: {row['actual_retrieval_query']}",
            "",
            *_render_memory_evidence(row),
            "### Actual retrieved top-3",
            "",
            *_render_trace_cards(row["retrieved_cards"]),
            "",
        ]
        retrieval_only.extend(shared)
        readable.extend(
            [
                *shared,
                "### Response A",
                "",
                row["response_A"],
                "",
                "### Response B",
                "",
                row["response_B"],
                "",
            ]
        )
    readable_path = review_dir / "generation_pilot_semantic_review_readable_ZH.md"
    readable_path.write_text("\n".join(readable), encoding="utf-8")
    retrieval_path = review_dir / "retrieval_semantic_review_readable_ZH.md"
    retrieval_path.write_text("\n".join(retrieval_only), encoding="utf-8")

    packet_fields = (
        "item_id",
        "case_id",
        "regime",
        "semantic_family",
        "semantic_subtopics_json",
        "current_user_text",
        "dialogue_before_current_json",
        "session_summary",
        "authorized_user_context",
        "current_correction",
        "context_provenance_json",
        "materially_useful_memory_sources_json",
        "profile_memories_json",
        "summary_memories_json",
        "event_memories_json",
        "memory_coverage_rationale",
        "advice_readiness_candidate",
        "actual_retrieval_query",
        "retrieved_cards_json",
        "response_A",
        "response_B",
    )
    packet_rows = []
    for row in enriched:
        packet_rows.append(
            {
                key: (
                    canonical_json(row[key[:-5]])
                    if key.endswith("_json")
                    else row.get(key) or ""
                )
                for key in packet_fields
            }
        )
    packet_path = review_dir / "generation_pilot_semantic_review_packet.csv"
    _write_csv(packet_path, packet_rows, packet_fields)

    reviewer_fields = ("item_id", *REVIEW_FIELDS, "annotator_id", "notes")
    ordered_cases = [cases[item_id] for item_id in sorted(cases)]
    reviewer_a = review_dir / "reviewer_a.csv"
    reviewer_b = review_dir / "reviewer_b.csv"
    _write_csv(reviewer_a, _reviewer_rows(ordered_cases, "reviewer_a"), reviewer_fields)
    _write_csv(reviewer_b, _reviewer_rows(ordered_cases, "reviewer_b"), reviewer_fields)
    guidelines = review_dir / "annotation_guidelines_ZH.md"
    guidelines.write_text(annotation_guidelines_text(pair_complete=True), encoding="utf-8")
    status = review_dir / "V9_STATUS.md"
    status.write_text(
        """# V9 status

`PAIR GENERATION COMPLETE / READY FOR BLIND SMOKE REVIEW`

This is a nine-case retriever and paired-response diagnostic only. It remains
unapproved for formal annotation, 27-case generation, PM training, or a final
V2 Strategy RAG freeze.
""",
        encoding="utf-8",
    )
    readme = review_dir / "README.md"
    readme.write_text(
        """# PM V2 V9 blind smoke-review materials

Use the combined readable packet and exactly one reviewer CSV. This directory
contains no private R0/RS mapping. `strategy_rag_marginal_value` is deliberately
absent and must be derived only after completed independent review, adjudication,
and unblinding.
""",
        encoding="utf-8",
    )
    return {
        "blinded": blinded_path,
        "readable": readable_path,
        "retrieval_readable": retrieval_path,
        "packet": packet_path,
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b,
        "guidelines": guidelines,
        "status": status,
        "readme": readme,
    }


def refresh_v9_review_materials(*, source_dir: Path, review_dir: Path) -> dict[str, Any]:
    """Build corrected reviewer materials from completed blind outputs, with no API use."""

    _require_new_directory(review_dir)
    source_cases_path = source_dir / "generation_pilot_semantic_review_cases.json"
    source_blinded_path = source_dir / "paired_responses_blinded.jsonl"
    cases = {str(row["item_id"]): row for row in read_json(source_cases_path)}
    blinded = list(iter_jsonl(source_blinded_path))
    outputs = _write_completed_review_materials(
        review_dir=review_dir, cases=cases, blinded=blinded
    )
    summary = {
        "status": V9_STATUS_CANDIDATE,
        "review_protocol_revision": 2,
        "cases": len(cases),
        "source_paid_generation_reused": True,
        "additional_api_calls": 0,
        "memory_evidence_in_primary_packet": True,
        "private_mapping_present": False,
        "blind_marginal_value_field_present": False,
        "formal_annotation_approved": False,
        "validation_27_case_approved": False,
        "pm_training_approved": False,
    }
    summary_path = review_dir / "review_materials_summary.json"
    write_json(summary_path, summary)
    create_artifact_attestation(
        review_dir / "review_materials_attestation.json",
        stage="pm_v2_v9_blind_smoke_review_protocol_v2",
        inputs={"cases": source_cases_path, "blinded": source_blinded_path},
        outputs={
            **{name: (path, name == "blinded") for name, path in outputs.items()},
            "summary": (summary_path, False),
        },
        parameters={
            "review_protocol_revision": 2,
            "additional_api_calls": 0,
            "mapping_visible": False,
            "formal_gate_unlocked": False,
        },
        expected={"cases": 9},
    )
    return summary


def derive_v9_marginal_values(
    *, adjudicated_review_path: Path, private_mapping_path: Path, out_dir: Path
) -> dict[str, Any]:
    """Create a private post-unblind table from a completed adjudicated review."""

    _require_new_directory(out_dir)
    with adjudicated_review_path.open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if len(review_rows) != 9:
        raise RuntimeError("post-unblind derivation requires exactly nine review rows")
    mappings = {
        str(row["item_id"]): row["blind_mapping"]
        for row in iter_jsonl(private_mapping_path)
    }
    if len(mappings) != 9:
        raise RuntimeError("private mapping must contain exactly nine items")
    output_rows: list[dict[str, Any]] = []
    for row in review_rows:
        item_id = str(row.get("item_id") or "").strip()
        if item_id not in mappings:
            raise RuntimeError(f"review item missing from private mapping: {item_id}")
        diagnostic_text = str(row.get("pair_is_diagnostic") or "").strip()
        preference = str(row.get("response_preference") or "").strip()
        if diagnostic_text not in {"0", "1"}:
            raise RuntimeError(f"invalid pair_is_diagnostic for {item_id}")
        marginal = derive_strategy_rag_marginal_value(
            response_preference=preference,
            pair_is_diagnostic=int(diagnostic_text),
            blind_mapping=mappings[item_id],
        )
        if int(diagnostic_text) == 0 or preference == "uncertain":
            preferred_condition = "uncertain"
        elif preference == "tie":
            preferred_condition = "tie"
        else:
            preferred_condition = mappings[item_id][preference]
        output_rows.append(
            {
                "item_id": item_id,
                "pair_is_diagnostic": diagnostic_text,
                "response_preference": preference,
                "preferred_condition": preferred_condition,
                "strategy_rag_marginal_value": marginal,
            }
        )
    if len({row["item_id"] for row in output_rows}) != 9:
        raise RuntimeError("adjudicated review contains duplicate item IDs")
    output_path = out_dir / "derived_strategy_rag_marginal_values.csv"
    _write_csv(
        output_path,
        output_rows,
        (
            "item_id",
            "pair_is_diagnostic",
            "response_preference",
            "preferred_condition",
            "strategy_rag_marginal_value",
        ),
    )
    counts = {
        label: sum(
            row["strategy_rag_marginal_value"] == label for row in output_rows
        )
        for label in ("positive", "neutral", "negative", "uncertain")
    }
    summary = {
        "status": "PRIVATE_POST_UNBLIND_DIAGNOSTIC_ONLY",
        "cases": 9,
        "marginal_value_counts": counts,
        "formal_annotation_approved": False,
        "validation_27_case_approved": False,
        "pm_training_approved": False,
    }
    summary_path = out_dir / "post_unblind_summary.json"
    write_json(summary_path, summary)
    create_artifact_attestation(
        out_dir / "post_unblind_attestation.json",
        stage="pm_v2_v9_private_post_unblind_derivation",
        inputs={
            "adjudicated_review": adjudicated_review_path,
            "private_mapping": private_mapping_path,
        },
        outputs={
            "derived_values": (output_path, False),
            "summary": (summary_path, False),
        },
        parameters={"automatic_derivation": True, "formal_gate_unlocked": False},
        expected={"cases": 9},
    )
    return summary


def _finish_blinded_materials(out_dir: Path) -> dict[str, Any]:
    results = list(iter_jsonl(out_dir / "paired_generations.jsonl"))
    if len(results) != 18:
        raise RuntimeError("paired generation requires exactly 18 successful rows")
    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for row in results:
        by_item.setdefault(str(row["item_id"]), {})[str(row["condition"])] = row
    mappings = {
        str(row["item_id"]): row
        for row in iter_jsonl(out_dir / "private_condition_mapping.jsonl")
    }
    cases = {
        str(row["item_id"]): row
        for row in read_json(out_dir / "generation_pilot_semantic_review_cases.json")
    }
    minimal_blinded: list[dict[str, Any]] = []
    for item_id in sorted(cases):
        pair = by_item.get(item_id) or {}
        if set(pair) != {"R0", "RS"}:
            raise RuntimeError(f"incomplete R0/RS pair for {item_id}")
        mapping = mappings[item_id]["blind_mapping"]
        minimal_blinded.append(
            {
                "item_id": item_id,
                "response_A": pair[mapping["A"]]["response"],
                "response_B": pair[mapping["B"]]["response"],
            }
        )
    outputs = _write_completed_review_materials(
        review_dir=out_dir, cases=cases, blinded=minimal_blinded
    )
    summary = {
        "status": V9_STATUS_CANDIDATE,
        "review_protocol_revision": 2,
        "cases": 9,
        "paired_generation_calls": 18,
        "complete_pairs": 9,
        "review_scores_prefilled": False,
        "mapping_visible_in_reviewer_materials": False,
        "blind_marginal_value_field_present": False,
        "memory_evidence_in_primary_packet": True,
        "formal_annotation_approved": False,
        "validation_27_case_approved": False,
        "pm_training_approved": False,
    }
    write_json(out_dir / "paired_generation_summary.json", summary)
    return summary


def run_v9_paired_generation(
    *,
    out_dir: Path,
    experiment_config_path: Path,
    accept_cost_estimate_sha256: str,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> dict[str, Any]:
    budget = validate_v9_dry_run(
        out_dir=out_dir,
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
        require_existing=True,
    )
    estimate = budget["cost_estimate"]
    if estimate["cost_estimate_sha256"] != accept_cost_estimate_sha256:
        raise RuntimeError("accepted V9 cost-estimate SHA256 mismatch")
    plan = list(iter_jsonl(out_dir / "paired_generation_call_plan.jsonl"))
    if sha256_file(out_dir / "paired_generation_call_plan.jsonl") != estimate[
        "call_plan_sha256"
    ]:
        raise RuntimeError("V9 call plan hash drift")
    if not plan:
        raise RuntimeError("V9 treatment-bound call plan is empty")
    supporter_generation_contract = SupporterGenerationContract.from_mapping(
        plan[0].get("supporter_generation_treatment") or {}
    )
    if (
        estimate.get("protocol")
        != "pm-v2-v9-paired-generation-cost-v2-treatment-bound"
        or estimate.get("supporter_generation_treatment")
        != supporter_generation_contract.payload()
        or estimate.get("supporter_generation_treatment_sha256")
        != supporter_generation_contract.digest()
        or any(
            row.get("supporter_generation_treatment")
            != supporter_generation_contract.payload()
            or row.get("supporter_generation_treatment_sha256")
            != supporter_generation_contract.digest()
            or float(row.get("temperature"))
            != supporter_generation_contract.temperature
            or int(row.get("max_tokens"))
            != supporter_generation_contract.max_output_tokens
            or (row.get("endpoint") or {}).get("name")
            != supporter_generation_contract.generator_endpoint
            or (row.get("messages") or [{}])[0]
            != {
                "role": "system",
                "content": supporter_generation_contract.system_prompt,
            }
            for row in plan
        )
    ):
        raise RuntimeError(
            "V9 call plan is legacy, mixed-treatment, or differs from its contract"
        )
    experiment = load_config(experiment_config_path)
    endpoint_name = str(plan[0]["endpoint"]["name"])
    endpoint = endpoint_from_config(experiment, endpoint_name)
    # Resolve the key before reserving any paid attempt.
    endpoint.api_key
    expected_calls = {str(row["call_key"]): 1 for row in plan}
    ledger = PersistentAttemptLedger(
        out_dir / "physical_attempt_ledger.jsonl",
        stage=PAIR_STAGE,
        expected_calls=expected_calls,
        maximum_total_attempts=len(plan),
    )
    result_path = out_dir / "paired_generations.jsonl"
    raw_path = out_dir / "raw_api_calls.jsonl"
    existing = _load_unique_results(result_path)
    if any(
        row.get("supporter_generation_treatment")
        != supporter_generation_contract.payload()
        or row.get("supporter_generation_treatment_sha256")
        != supporter_generation_contract.digest()
        or row.get("normalized_finish_reason") != "complete"
        for row in existing.values()
    ):
        raise RuntimeError(
            "existing V9 results are legacy or mixed-treatment; use a new directory"
        )
    client = OpenAICompatibleClient(endpoint)
    try:
        for row in plan:
            call_key = str(row["call_key"])
            if call_key in existing:
                if not ledger.succeeded(call_key):
                    raise RuntimeError("output exists without successful ledger event")
                continue
            if ledger.succeeded(call_key):
                terminal = ledger.terminal_row(call_key)
                recovered = dict((terminal or {}).get("result") or {})
                if (
                    not recovered.get("response")
                    or recovered.get("supporter_generation_treatment")
                    != supporter_generation_contract.payload()
                    or recovered.get("supporter_generation_treatment_sha256")
                    != supporter_generation_contract.digest()
                    or recovered.get("normalized_finish_reason") != "complete"
                ):
                    raise RuntimeError(
                        "successful ledger row cannot recover a complete "
                        "treatment-matched output"
                    )
                append_jsonl(result_path, recovered)
                existing[call_key] = recovered
                continue
            if ledger.exhausted(call_key):
                continue
            reservation = ledger.reserve(
                call_key,
                record_ids={
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                },
                prompt_sha256=row["prompt_sha256"],
            )
            call = None
            try:
                call, _ = client.chat(
                    row["messages"],
                    temperature=float(row["temperature"]),
                    max_tokens=int(row["max_tokens"]),
                    seed=int(row["seed"]),
                    retries=1,
                )
                usage = require_reported_usage(call.usage, stage=PAIR_STAGE)
                token_error = reported_prompt_token_error(
                    usage,
                    maximum_prompt_tokens=int(row["maximum_input_tokens"]),
                    stage=PAIR_STAGE,
                    require_positive=True,
                )
                if token_error:
                    raise RuntimeError(token_error)
                completion_error = supporter_generation_contract.completion_gate_error(
                    normalized_finish_reason=call.normalized_finish_reason,
                    provider_finish_reason=call.provider_finish_reason,
                )
                if completion_error is not None:
                    raise RuntimeError(completion_error)
                result = {
                    "call_key": call_key,
                    "item_id": row["item_id"],
                    "case_id": row["case_id"],
                    "condition": row["condition"],
                    "messages": row["messages"],
                    "prompt_sha256": row["prompt_sha256"],
                    "response": supporter_generation_contract.normalize_output(
                        call.text
                    ),
                    "request_hash": call.request_hash,
                    "usage": usage,
                    "latency_ms": float(call.latency_ms),
                    "generator": row["endpoint"],
                    "temperature": row["temperature"],
                    "max_tokens": row["max_tokens"],
                    "seed": row["seed"],
                    "strategy_card_ids": row["strategy_card_ids"],
                    "provider_finish_reason": call.provider_finish_reason,
                    "normalized_finish_reason": call.normalized_finish_reason,
                    "supporter_generation_treatment": (
                        supporter_generation_contract.payload()
                    ),
                    "supporter_generation_treatment_sha256": (
                        supporter_generation_contract.digest()
                    ),
                }
                ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=call.request_hash,
                    usage=usage,
                    error=None,
                    result=result,
                )
                append_jsonl(result_path, result)
                append_jsonl(
                    raw_path,
                    {
                        **result,
                        "raw_response": call.raw_response,
                        "completion_truncated": (
                            call.normalized_finish_reason == "length"
                        ),
                    },
                )
                existing[call_key] = result
            except Exception as exc:
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=(call.request_hash if call else None),
                    usage=(call.usage if call else None),
                    error=f"{type(exc).__name__}: {exc}",
                )
                append_jsonl(
                    raw_path,
                    {
                        "call_key": call_key,
                        "item_id": row["item_id"],
                        "condition": row["condition"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": call.raw_response if call else None,
                        "provider_finish_reason": (
                            call.provider_finish_reason if call else None
                        ),
                        "normalized_finish_reason": (
                            call.normalized_finish_reason if call else None
                        ),
                        "completion_truncated": (
                            call.normalized_finish_reason == "length"
                            if call
                            else None
                        ),
                        "supporter_generation_treatment": (
                            supporter_generation_contract.payload()
                        ),
                        "supporter_generation_treatment_sha256": (
                            supporter_generation_contract.digest()
                        ),
                    },
                )
                raise RuntimeError(
                    "V9 paired generation stopped fail-fast after the first "
                    f"failed physical attempt: {row['item_id']} {row['condition']}"
                ) from exc
    finally:
        client.close()
    existing = _load_unique_results(result_path)
    if len(existing) != len(plan):
        raise RuntimeError(
            f"V9 paired generation incomplete: {len(existing)}/{len(plan)} successful"
        )
    summary = _finish_blinded_materials(out_dir)
    summary.update(
        {
            "supporter_generation_treatment": (
                supporter_generation_contract.payload()
            ),
            "supporter_generation_treatment_sha256": (
                supporter_generation_contract.digest()
            ),
            "historical_v9_base_300_artifacts_reused": False,
            "formal_authorization": False,
        }
    )
    write_json(out_dir / "paired_generation_summary.json", summary)
    create_artifact_attestation(
        out_dir / "paired_generation_attestation.json",
        stage=PAIR_STAGE,
        inputs={
            "cases": out_dir / "generation_pilot_semantic_review_cases.json",
            "call_plan": out_dir / "paired_generation_call_plan.jsonl",
            "cost_estimate": out_dir / "paired_generation_cost_estimate.json",
            "mapping": out_dir / "private_condition_mapping.jsonl",
        },
        outputs={
            "generations": (result_path, True),
            "blinded": (out_dir / "paired_responses_blinded.jsonl", True),
            "packet": (
                out_dir / "generation_pilot_semantic_review_packet.csv",
                False,
            ),
            "summary": (out_dir / "paired_generation_summary.json", False),
        },
        parameters={
            "accepted_cost_estimate_sha256": accept_cost_estimate_sha256,
            "review_scores_prefilled": False,
            "formal_gate_unlocked": False,
            "supporter_generation_treatment": (
                supporter_generation_contract.payload()
            ),
            "supporter_generation_treatment_sha256": (
                supporter_generation_contract.digest()
            ),
            "historical_v9_base_300_artifacts_reused": False,
        },
        expected={"calls": 18, "pairs": 9},
    )
    return summary
