#!/usr/bin/env python3
"""Prepare the one-shot H1 review of all V1.5 RAG cards and retrieval."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, sha256_text, stable_hex
from metacom_pm.v1_5_strategy_rag_v4 import (
    observable_opportunity_flags,
    rank_applicable_v4_cards,
    validate_v4_candidate_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h1-complete-rag-human-review-v1"
POSITIVE_RETRIEVAL_ITEMS = 64
NEGATIVE_RETRIEVAL_TARGETS = {
    "pure_phatic": 6,
    "explicit_stop": 4,
    "active_high_stakes": 6,
}
MINIMUM_SCORE = 0.05


def _load_natural_helpers():
    path = ROOT / "scripts/v1_5/24l_prepare_rs_natural_retrieval_review_v1_5.py"
    spec = importlib.util.spec_from_file_location("pm15_natural_helpers", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _nearest_core(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        left = _normalize_tokens(
            " ".join(
                [
                    str(row["support_move"]),
                    str(row["when_to_use"]),
                    str(row["when_not_to_use"]),
                ]
            )
        )
        best_id = ""
        best = -1.0
        for other in rows:
            if other["submove_id"] == row["submove_id"]:
                continue
            right = _normalize_tokens(
                " ".join(
                    [
                        str(other["support_move"]),
                        str(other["when_to_use"]),
                        str(other["when_not_to_use"]),
                    ]
                )
            )
            score = len(left & right) / max(1, len(left | right))
            if score > best:
                best = score
                best_id = str(other["submove_id"])
        result[str(row["submove_id"])] = {
            "nearest_submove_id": best_id,
            "token_jaccard": round(best, 4),
        }
    return result


def _collect_prior_dialogues() -> tuple[set[str], list[str]]:
    patterns = (
        "outputs/pm_v1_5_strategy_rag_v4_direct_effect_v*/selected_states.jsonl",
        "outputs/pm_v1_5_rs_natural_retrieval*/private_audit.jsonl",
        "outputs/pm_v1_5_rs_natural_retrieval*/selected_states.jsonl",
        "outputs/pm_v1_5_rs_six_card*/selected_states.jsonl",
        "outputs/pm_v1_5_rs_corrected*/selected_states.jsonl",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in ROOT.glob(pattern)
            if path.is_file()
        }
    )
    identifiers: set[str] = set()
    for path in paths:
        for row in iter_jsonl(path):
            value = (
                row.get("source_dialogue_id")
                or row.get("user_id")
                or row.get("dialogue_id")
            )
            if value:
                identifiers.add(str(value))
    return identifiers, [str(path.relative_to(ROOT)) for path in paths]


def _select_positive(
    candidates: list[dict[str, Any]],
    target: int,
) -> list[dict[str, Any]]:
    by_core: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_core[str(row["top1_core_submove_id"])].append(row)
    for rows in by_core.values():
        rows.sort(
            key=lambda row: (
                -float(row["top1_score"]),
                -float(row["top1_margin"]),
                str(row["source_dialogue_id"]),
                int(row["source_turn_index"]),
            )
        )

    selected: list[dict[str, Any]] = []
    used_dialogues: set[str] = set()
    used_states: set[str] = set()
    # Rare cores are reserved first; at most two states per top-1 core.
    for core in sorted(by_core, key=lambda key: (len(by_core[key]), key)):
        taken = 0
        for row in by_core[core]:
            dialogue = str(row["source_dialogue_id"])
            state = str(row["visible_dialogue_sha256"])
            if dialogue in used_dialogues or state in used_states:
                continue
            selected.append(row)
            used_dialogues.add(dialogue)
            used_states.add(state)
            taken += 1
            if taken == 2:
                break

    family_counts = Counter(str(row["top1_strategy_family"]) for row in selected)
    profile_counts = Counter(str(row["execution_profile"]) for row in selected)
    core_counts = Counter(str(row["top1_core_submove_id"]) for row in selected)
    remaining = [
        row
        for row in candidates
        if str(row["source_dialogue_id"]) not in used_dialogues
        and str(row["visible_dialogue_sha256"]) not in used_states
    ]
    while len(selected) < target:
        if not remaining:
            raise RuntimeError(f"only selected {len(selected)}/{target} positives")
        remaining.sort(
            key=lambda row: (
                family_counts[str(row["top1_strategy_family"])],
                profile_counts[str(row["execution_profile"])],
                core_counts[str(row["top1_core_submove_id"])],
                -float(row["top1_score"]),
                str(row["source_dialogue_id"]),
            )
        )
        row = remaining.pop(0)
        dialogue = str(row["source_dialogue_id"])
        state = str(row["visible_dialogue_sha256"])
        if dialogue in used_dialogues or state in used_states:
            continue
        selected.append(row)
        used_dialogues.add(dialogue)
        used_states.add(state)
        family_counts[str(row["top1_strategy_family"])] += 1
        profile_counts[str(row["execution_profile"])] += 1
        core_counts[str(row["top1_core_submove_id"])] += 1
    return selected[:target]


def _select_negative(
    pools: dict[str, list[dict[str, Any]]],
    targets: dict[str, int],
    excluded_dialogues: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = set(excluded_dialogues)
    for stratum, target in targets.items():
        taken = 0
        for row in sorted(
            pools[stratum],
            key=lambda item: (
                str(item["source_dialogue_id"]),
                int(item["source_turn_index"]),
            ),
        ):
            dialogue = str(row["source_dialogue_id"])
            if dialogue in used:
                continue
            selected.append(row)
            used.add(dialogue)
            taken += 1
            if taken == target:
                break
        if taken != target:
            raise RuntimeError(f"negative stratum {stratum}: {taken}/{target}")
    return selected


def _visible_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        key: card[key]
        for key in (
            "card_id",
            "core_submove_id",
            "strategy_family",
            "execution_profile",
            "support_move",
            "when_to_use",
            "when_not_to_use",
            "compatible_support_modes",
            "compatible_dialogue_phases",
            "goal_types",
            "directive_burden",
            "risk_flags",
            "prompt_guidance",
            "variant_rationale",
        )
    }


def _render_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 H1 · RAG 最终总审核</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:auto;padding:22px;background:#f3f5f7;color:#17202a;line-height:1.48}}
.note,.item{{background:white;border:1px solid #d8dde5;border-radius:10px;padding:17px;margin:15px 0}}
.note{{border-left:5px solid #245f9e}}.warn{{border-left-color:#a85b00}}.dialogue{{white-space:pre-wrap;background:#eef2f6;padding:11px;border-radius:7px}}
.card{{background:#fafafa;border:1px solid #ccd4dd;border-radius:8px;padding:11px;margin:9px 0}}
.source{{border-left:3px solid #9aa8b6;padding-left:10px;margin:12px 0}}.tag{{display:inline-block;background:#e7edf4;border-radius:999px;padding:3px 9px;margin:2px;font-size:12px}}
select,textarea,input[type=text]{{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 9px}}textarea{{min-height:58px}}
.sticky{{position:sticky;top:0;background:#f3f5f7;padding:9px 0;z-index:3}}button{{padding:9px 14px;margin-right:6px}}
.tabs button.active{{background:#245f9e;color:white}}details{{margin:8px 0}}h2{{margin-bottom:5px}}
</style></head><body>
<h1>PM V1.5 H1：RAG 最终总审核</h1>
<div class="note"><b>这是唯一 H1 总包。</b>第一部分用 50 个 core judgments 同时审核
100 张 minimal/dialogic variants；第二部分审核 64 个自然 Top‑5 检索和 16 个应当
abstain 的负例。完成后冻结最终 Bank。不要根据“它听起来不错”批准，要核对可见证据、
when-to-use、when-not-to-use、负担和风险。</div>
<div class="note warn">来源回复只用于判断 atomic move 是否真的出现，不能因为原回复
整体写得好/坏而替卡片背书。检索部分完全不含下一条 supporter response、native strategy
或生成 outcome。</div>
<div class="sticky"><span class="tabs">
<button id="bcard" class="active" onclick="tab('card')">卡片 50</button>
<button id="bretrieval" onclick="tab('retrieval')">检索 80</button></span>
<button onclick="download()">导出完整 JSONL</button> <span id="progress"></span></div>
<div id="root"></div>
<script>
const DATA={data}; const KEY="pm15_h1_complete_rag_v1";
let S=JSON.parse(localStorage.getItem(KEY)||"{{}}"), MODE="card";
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
function setv(id,k,v){{S[id]??={{}};S[id][k]=v;localStorage.setItem(KEY,JSON.stringify(S));progress();}}
function sel(id,k,opts){{const v=(S[id]||{{}})[k]||"";return `<select onchange="setv('${{id}}','${{k}}',this.value)"><option value="">请选择</option>${{opts.map(x=>`<option value="${{x}}" ${{v===x?"selected":""}}>${{x}}</option>`).join("")}}</select>`;}}
function text(id,k,placeholder=""){{const v=(S[id]||{{}})[k]||"";return `<textarea placeholder="${{esc(placeholder)}}" onchange="setv('${{id}}','${{k}}',this.value)">${{esc(v)}}</textarea>`;}}
function dialogue(turns){{return (turns||[]).map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");}}
function card(c){{return `<div class="card"><b>${{esc(c.strategy_family)}} · ${{esc(c.core_submove_id)}} · ${{esc(c.execution_profile)}}</b>
<p><b>Move:</b> ${{esc(c.support_move)}}</p><p><b>Use:</b> ${{esc(c.when_to_use)}}</p>
<p><b>Not use:</b> ${{esc(c.when_not_to_use)}}</p><p><b>Guidance:</b> ${{esc(c.prompt_guidance)}}</p>
<span class="tag">burden=${{esc(c.directive_burden)}}</span><span class="tag">${{esc((c.risk_flags||[]).join(", "))}}</span></div>`;}}
function source(x){{return `<div class="source"><div class="dialogue">${{esc(dialogue(x.recent_context))}}</div>
<p><b>目标 supporter response：</b>${{esc(x.target_response)}}</p></div>`;}}
function cardItems(){{return DATA.card_items.map((x,i)=>{{const id=x.review_item_id;S[id]??={{}};return `<section class="item">
<h2>${{i+1}}/50 · ${{esc(x.strategy_family)}} · ${{esc(x.core_submove_id)}}</h2>
<span class="tag">source dialogues=${{x.source_support.confident_weak_assigned_dialogues}}</span>
<span class="tag">${{x.source_support.provisional_source_support_pass?"旧弱来源门通过":"旧弱来源门未通过"}}</span>
<span class="tag">nearest=${{esc(x.nearest_core.nearest_submove_id)}} / J=${{x.nearest_core.token_jaccard}}</span>
<p><b>Move：</b>${{esc(x.support_move)}}</p><p><b>When to use：</b>${{esc(x.when_to_use)}}</p>
<p><b>When not to use：</b>${{esc(x.when_not_to_use)}}</p>
<details><summary>查看 ${{x.source_examples.length}} 个 train-only 来源例</summary>${{x.source_examples.map(source).join("")}}</details>
<details><summary>查看 minimal/dialogic 两个 runtime variant</summary>${{x.variants.map(card).join("")}}</details>
<label>Atomic move 是否清楚且有意义？${{sel(id,"support_move_clear",["yes","no","uncertain"])}}</label>
<label>when_to_use 是否可执行？${{sel(id,"when_to_use_valid",["yes","no","uncertain"])}}</label>
<label>when_not_to_use、burden、risk 是否足够？${{sel(id,"boundary_risk_valid",["yes","no","uncertain"])}}</label>
<label>来源例是否支持这个 move（不是判断回复整体质量）？${{sel(id,"source_support_valid",["yes","no","uncertain"])}}</label>
<label>minimal variant${{sel(id,"minimal_variant",["approve","revise","reject"])}}</label>
<label>dialogic variant${{sel(id,"dialogic_variant",["approve","revise","reject"])}}</label>
<label>是否与另一 core 实质重复？${{sel(id,"distinctness",["distinct","duplicate","uncertain"])}}</label>
<input type="text" value="${{esc((S[id]||{{}}).duplicate_of||"")}}" placeholder="若重复，填写另一 core_submove_id" onchange="setv('${{id}}','duplicate_of',this.value)">
${{text(id,"required_corrections","拒绝/revise 时写必改项；通过可留空")}}</section>`;}}).join("");}}
function retrievalItems(){{return DATA.retrieval_items.map((x,i)=>{{const id=x.review_item_id;S[id]??={{}};if(x.expected_behavior==="abstain")return `<section class="item">
<h2>${{i+1}}/80 · 负例 · ${{esc(x.negative_stratum)}}</h2><div class="dialogue">${{esc(dialogue(x.visible_dialogue))}}</div>
<p><b>固定规则结果：</b>RS opportunity=false，不返回卡片。</p>
<label>该 abstention 是否正确？${{sel(id,"abstention_correct",["yes","no","uncertain"])}}</label>
${{text(id,"notes","若不正确，说明应允许哪类技术")}}</section>`;
return `<section class="item"><h2>${{i+1}}/80 · 自然 Top‑5</h2>
<span class="tag">${{esc(x.execution_profile)}}</span><span class="tag">Top1 score=${{x.top1_score.toFixed(3)}}</span>
<div class="dialogue">${{esc(dialogue(x.visible_dialogue))}}</div>
${{x.top5.map((c,j)=>`<details ${{j===0?"open":""}}><summary>Rank ${{j+1}} · score=${{c.score.toFixed(3)}}</summary>${{card(c.card)}}</details>`).join("")}}
<label>当前是否存在值得 RS 调用的技术机会？${{sel(id,"rs_opportunity",["yes","no","uncertain"])}}</label>
<label>Top‑1 是否合适？${{sel(id,"top1_fit",["yes","no","uncertain"])}}</label>
<label>Top‑1 是否触发 hard exclusion？${{sel(id,"hard_exclusion_triggered",["yes","no","uncertain"])}}</label>
<label>最佳候选${{sel(id,"best_candidate",["rank1","rank2","rank3","rank4","rank5","no_safe_card","uncertain"])}}</label>
<input type="text" value="${{esc((S[id]||{{}}).acceptable_ranks||"")}}" placeholder="所有可接受 ranks，例如 1,3,4" onchange="setv('${{id}}','acceptable_ranks',this.value)">
${{text(id,"notes","拒绝或改选时写可见依据")}}</section>`;}}).join("");}}
function tab(x){{MODE=x;document.getElementById("bcard").className=x==="card"?"active":"";document.getElementById("bretrieval").className=x==="retrieval"?"active":"";render();}}
function complete(x){{if(x.item_type==="card"){{const s=S[x.review_item_id]||{{}};return ["support_move_clear","when_to_use_valid","boundary_risk_valid","source_support_valid","minimal_variant","dialogic_variant","distinctness"].every(k=>s[k]);}}
const s=S[x.review_item_id]||{{}};return x.expected_behavior==="abstain"?!!s.abstention_correct:["rs_opportunity","top1_fit","hard_exclusion_triggered","best_candidate"].every(k=>s[k]);}}
function rows(){{return [...DATA.card_items,...DATA.retrieval_items].map(x=>({{protocol:DATA.manifest.protocol,review_item_id:x.review_item_id,item_type:x.item_type,...(S[x.review_item_id]||{{}}),annotator_id:""}}));}}
function progress(){{const all=[...DATA.card_items,...DATA.retrieval_items],n=all.filter(complete).length;document.getElementById("progress").textContent=`完成 ${{n}}/${{all.length}}`;}}
function render(){{document.getElementById("root").innerHTML=MODE==="card"?cardItems():retrievalItems();progress();}}
function download(){{const b=new Blob([rows().map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="h1_complete_rag_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
render();
</script></body></html>"""


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cards",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_rag_v4_candidate_v1/strategy_cards_v4_candidate.jsonl",
    )
    parser.add_argument(
        "--core-review",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_strategy_bank_v3_core_review_candidate_v1/human_review_packet.jsonl",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/clean_train_strategy_universe.jsonl",
    )
    parser.add_argument(
        "--formal-test",
        type=Path,
        default=ROOT / "data/esconv_test_v1_5_visible_v2_candidate/runtime_states.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate",
    )
    args = parser.parse_args()
    helpers = _load_natural_helpers()

    cards = validate_v4_candidate_cards([dict(row) for row in iter_jsonl(args.cards)])
    active_cards = [
        card
        for card in cards
        if card["source_support"]["provisional_source_support_pass"]
    ]
    core_rows = [dict(row) for row in iter_jsonl(args.core_review)]
    if len(cards) != 100 or len(core_rows) != 50 or len(active_cards) != 62:
        raise ValueError("unexpected candidate/core/source-pass cardinality")
    card_by_core: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        card_by_core[str(card["core_submove_id"])].append(card)
    if set(card_by_core) != {str(row["submove_id"]) for row in core_rows}:
        raise RuntimeError("core review and V4 candidate submoves differ")
    nearest = _nearest_core(core_rows)

    source_dialogues = {
        str(example["source_dialogue_id"])
        for row in core_rows
        for example in row["source_examples"]
    }
    prior_dialogues, prior_paths = _collect_prior_dialogues()
    formal_test_dialogues = {
        str(row["user_id"]) for row in iter_jsonl(args.formal_test)
    }
    excluded_dialogues = source_dialogues | prior_dialogues | formal_test_dialogues

    raw_rows = [dict(row) for row in iter_jsonl(args.universe)]
    if len(raw_rows) != 9148:
        raise ValueError(f"expected 9148 ESConv-train states, found {len(raw_rows)}")
    visible_rows = [helpers._visible_projection(row) for row in raw_rows]
    del raw_rows

    candidates: list[dict[str, Any]] = []
    negative_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_visible: set[tuple[str, str]] = set()
    for row in visible_rows:
        dialogue_id = str(row["source_dialogue_id"])
        if dialogue_id in excluded_dialogues:
            continue
        latest = helpers._latest_seeker(row)
        if not latest:
            continue
        visible_text = helpers._visible_dialogue_text(row)
        visible_digest = sha256_text(visible_text)
        key = (dialogue_id, visible_digest)
        if key in seen_visible:
            continue
        seen_visible.add(key)
        flags = observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=helpers._recent_seeker(row),
        )
        expanded_high_stakes = helpers._expanded_high_stakes(row)
        negative_stratum = None
        if flags["explicit_stop"]:
            negative_stratum = "explicit_stop"
        elif flags["active_high_stakes"] or expanded_high_stakes:
            negative_stratum = "active_high_stakes"
        elif flags["pure_phatic"]:
            negative_stratum = "pure_phatic"
        if negative_stratum:
            negative_pools[negative_stratum].append(
                {
                    "source_dialogue_id": dialogue_id,
                    "source_turn_index": int(row["source_turn_index"]),
                    "visible_dialogue": row["recent_dialogue"],
                    "visible_dialogue_sha256": visible_digest,
                    "negative_stratum": negative_stratum,
                    "observable_flags": flags,
                }
            )
            continue
        if not flags["substantive"]:
            continue
        query = helpers._natural_query(row, flags)
        ranked = rank_applicable_v4_cards(
            query=query,
            current_user_text=latest,
            recent_user_text=helpers._recent_seeker(row),
            flags=flags,
            cards=active_cards,
        )
        if not ranked or float(ranked[0]["score"]) < MINIMUM_SCORE:
            continue
        second = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
        candidates.append(
            {
                "source_dialogue_id": dialogue_id,
                "source_turn_index": int(row["source_turn_index"]),
                "visible_dialogue": row["recent_dialogue"],
                "visible_dialogue_sha256": visible_digest,
                "observable_flags": flags,
                "execution_profile": str(ranked[0]["execution_profile"]),
                "natural_query": query,
                "top1_core_submove_id": str(ranked[0]["core_submove_id"]),
                "top1_strategy_family": str(ranked[0]["strategy_family"]),
                "top1_score": float(ranked[0]["score"]),
                "top1_margin": float(ranked[0]["score"]) - second,
                "ranked": ranked[:5],
            }
        )

    selected_positive = _select_positive(candidates, POSITIVE_RETRIEVAL_ITEMS)
    selected_negative = _select_negative(
        negative_pools,
        NEGATIVE_RETRIEVAL_TARGETS,
        {str(row["source_dialogue_id"]) for row in selected_positive},
    )
    active_by_id = {str(card["card_id"]): card for card in active_cards}

    card_items: list[dict[str, Any]] = []
    for row in sorted(core_rows, key=lambda item: (str(item["strategy_family"]), str(item["submove_id"]))):
        core = str(row["submove_id"])
        variants = sorted(
            card_by_core[core],
            key=lambda card: str(card["execution_profile"]),
        )
        card_items.append(
            {
                "protocol": PROTOCOL,
                "item_type": "card",
                "review_item_id": "h1_card_" + stable_hex(PROTOCOL, core, n=20),
                "core_submove_id": core,
                "strategy_family": str(row["strategy_family"]),
                "support_move": str(row["support_move"]),
                "when_to_use": str(row["when_to_use"]),
                "when_not_to_use": str(row["when_not_to_use"]),
                "source_support": variants[0]["source_support"],
                "nearest_core": nearest[core],
                "source_examples": row["source_examples"],
                "variants": [_visible_card(card) for card in variants],
            }
        )

    retrieval_items: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for row in selected_positive:
        review_id = "h1_retrieval_" + stable_hex(
            PROTOCOL,
            str(row["source_dialogue_id"]),
            str(row["source_turn_index"]),
            str(row["visible_dialogue_sha256"]),
            n=20,
        )
        top5 = [
            {
                "score": float(item["score"]),
                "compatibility_tier": int(item["compatibility_tier"]),
                "card": _visible_card(active_by_id[str(item["card_id"])]),
            }
            for item in row["ranked"]
        ]
        retrieval_items.append(
            {
                "protocol": PROTOCOL,
                "item_type": "retrieval",
                "review_item_id": review_id,
                "expected_behavior": "rank_cards",
                "visible_dialogue": row["visible_dialogue"],
                "execution_profile": row["execution_profile"],
                "top1_score": row["top1_score"],
                "top1_margin": row["top1_margin"],
                "top5": top5,
            }
        )
        private_rows.append({**row, "review_item_id": review_id})
    for row in selected_negative:
        review_id = "h1_abstain_" + stable_hex(
            PROTOCOL,
            str(row["source_dialogue_id"]),
            str(row["source_turn_index"]),
            str(row["visible_dialogue_sha256"]),
            n=20,
        )
        retrieval_items.append(
            {
                "protocol": PROTOCOL,
                "item_type": "retrieval",
                "review_item_id": review_id,
                "expected_behavior": "abstain",
                "negative_stratum": row["negative_stratum"],
                "visible_dialogue": row["visible_dialogue"],
            }
        )
        private_rows.append({**row, "review_item_id": review_id})

    selected_dialogues = {
        str(row["source_dialogue_id"])
        for row in selected_positive + selected_negative
    }
    positive_core_counts = Counter(
        str(row["top1_core_submove_id"]) for row in selected_positive
    )
    positive_family_counts = Counter(
        str(row["top1_strategy_family"]) for row in selected_positive
    )
    profile_counts = Counter(
        str(row["execution_profile"]) for row in selected_positive
    )
    checks = {
        "candidate_variants_100": len(cards) == 100,
        "core_review_items_50": len(card_items) == 50,
        "every_core_has_two_profiles": all(
            {str(card["execution_profile"]) for card in variants}
            == {"minimal", "dialogic"}
            for variants in card_by_core.values()
        ),
        "active_source_pass_variants_62": len(active_cards) == 62,
        "positive_retrieval_items_64": len(selected_positive) == 64,
        "negative_retrieval_items_16": len(selected_negative) == 16,
        "retrieval_dialogues_unique": len(selected_dialogues) == 80,
        "no_core_source_dialogue_overlap": not selected_dialogues & source_dialogues,
        "no_prior_rs_review_overlap": not selected_dialogues & prior_dialogues,
        "no_formal_esconv_test_overlap": not selected_dialogues & formal_test_dialogues,
        "positive_top5_nonempty": all(row["top5"] for row in retrieval_items if row["expected_behavior"] == "rank_cards"),
        "hidden_next_response_fields_absent_from_retrieval": all(
            not (
                {"target_response", "supporter_response", "strategy_label", "problem_type", "emotion_type"}
                & set(row)
            )
            for row in retrieval_items
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))

    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_SHOT_H1_HUMAN_REVIEW",
        "card_core_items": len(card_items),
        "candidate_variants_covered": len(cards),
        "active_source_pass_variants": len(active_cards),
        "active_source_pass_core_submoves": len(
            {str(card["core_submove_id"]) for card in active_cards}
        ),
        "retrieval_items": len(retrieval_items),
        "positive_retrieval_items": len(selected_positive),
        "negative_retrieval_items": len(selected_negative),
        "positive_top1_core_coverage": len(positive_core_counts),
        "positive_top1_core_counts": dict(sorted(positive_core_counts.items())),
        "positive_family_counts": dict(sorted(positive_family_counts.items())),
        "positive_profile_counts": dict(sorted(profile_counts.items())),
        "negative_stratum_counts": dict(
            sorted(Counter(str(row["negative_stratum"]) for row in selected_negative).items())
        ),
        "selection": "ESConv train only, visible dialogue only, no next response/native strategy/generation outcome",
        "source_examples_role": "card action evidence only; not retrieval selection or PM labels",
        "checks": checks,
        "prior_exclusion_paths": prior_paths,
        "lineage": {
            "cards_sha256": sha256_file(args.cards),
            "core_review_sha256": sha256_file(args.core_review),
            "universe_sha256": sha256_file(args.universe),
            "formal_test_sha256": sha256_file(args.formal_test),
        },
        "post_review_rule": (
            "Freeze approved/revised nonduplicate variants; rejected variants "
            "do not enter the Bank. Freeze the lexical/applicability retriever "
            "only if Top-1 suitable >=0.80, Top-5 acceptable recall >=0.90, "
            "and zero hard-exclusion injection; otherwise repair once using "
            "only H1 corrections, then freeze or disclose failure."
        ),
    }
    template: list[dict[str, Any]] = []
    for row in card_items:
        template.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": row["review_item_id"],
                "item_type": "card",
                "support_move_clear": None,
                "when_to_use_valid": None,
                "boundary_risk_valid": None,
                "source_support_valid": None,
                "minimal_variant": None,
                "dialogic_variant": None,
                "distinctness": None,
                "duplicate_of": "",
                "required_corrections": "",
                "annotator_id": "",
            }
        )
    for row in retrieval_items:
        base = {
            "protocol": PROTOCOL,
            "review_item_id": row["review_item_id"],
            "item_type": "retrieval",
            "annotator_id": "",
        }
        if row["expected_behavior"] == "abstain":
            base.update({"abstention_correct": None, "notes": ""})
        else:
            base.update(
                {
                    "rs_opportunity": None,
                    "top1_fit": None,
                    "hard_exclusion_triggered": None,
                    "best_candidate": None,
                    "acceptable_ranks": "",
                    "notes": "",
                }
            )
        template.append(base)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "card_items": card_items,
        "retrieval_items": retrieval_items,
    }
    write_json(args.out_dir / "h1_review_packet.json", payload)
    write_json(args.out_dir / "h1_manifest.json", manifest)
    write_jsonl(args.out_dir / "human_annotation_template.jsonl", template)
    write_jsonl(args.out_dir / "private_retrieval_audit.jsonl", private_rows)
    (args.out_dir / "human_review.html").write_text(
        _render_html(payload),
        encoding="utf-8",
    )
    print(manifest)


if __name__ == "__main__":
    main()
