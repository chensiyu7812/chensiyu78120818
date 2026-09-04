#!/usr/bin/env python3
"""Prepare the one-shot blinded H2 transparent-vs-BGE qualification page."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, stable_hex
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_strategy_rag_repair import (
    REPAIR_PROTOCOL,
    repaired_observable_opportunity_flags,
    repaired_rank_applicable_v4_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h2-retrieval-human-qualification-v1"
DEFAULT_BGE = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
TOP_K = 3


def _load_bakeoff_module():
    path = ROOT / "scripts/v1_5/24am_compare_post_h1_retrieval_repairs_v1_5.py"
    spec = importlib.util.spec_from_file_location("h1_bakeoff_helpers", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seeker_texts(dialogue: list[dict[str, Any]]) -> list[str]:
    return [
        normalize_space(turn.get("content", ""))
        for turn in dialogue
        if str(turn.get("speaker")) == "seeker"
        and normalize_space(turn.get("content", ""))
    ]


def _query(dialogue: list[dict[str, Any]], flags: dict[str, Any]) -> str:
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool)
        and value
        and key
        not in {
            "substantive",
            "pure_phatic",
            "routine_closing",
            "active_high_stakes",
            "explicit_stop",
            "ordinary_rag_hard_off",
        }
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


def _public_card(card: dict[str, Any]) -> dict[str, Any]:
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
            "directive_burden",
            "risk_flags",
            "prompt_guidance",
        )
    }


def _union_candidates(
    *,
    state_id: str,
    transparent: list[dict[str, Any]],
    bge: list[dict[str, Any]],
    cards_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ids = {
        str(row["card_id"])
        for row in transparent[:TOP_K] + bge[:TOP_K]
    }
    ordered_ids = sorted(
        ids,
        key=lambda card_id: stable_hex(
            PROTOCOL, state_id, card_id, n=32
        ),
    )
    public: list[dict[str, Any]] = []
    number_by_id: dict[str, int] = {}
    for number, card_id in enumerate(ordered_ids, start=1):
        number_by_id[card_id] = number
        public.append(
            {
                "candidate_number": number,
                "candidate_id": "h2_candidate_"
                + stable_hex(PROTOCOL, state_id, card_id, n=20),
                "card": _public_card(cards_by_id[card_id]),
            }
        )
    return public, number_by_id


def _render_html(packet: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    meta = json.dumps(
        {
            "protocol": manifest["protocol"],
            "opportunity_definition": manifest["opportunity_definition"],
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 H2 检索资格审核</title><style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:20px;background:#f3f5f7;color:#17202a;line-height:1.5}}
.note,.item{{background:#fff;border:1px solid #d6dde5;border-radius:10px;padding:16px;margin:14px 0}}
.note{{border-left:5px solid #245f9e}}.warn{{border-left-color:#a85b00}}
.dialogue{{white-space:pre-wrap;background:#edf2f6;border-radius:7px;padding:11px}}
.candidate{{border-left:4px solid #8ca0b3;background:#fafafa;padding:11px 14px;margin:14px 0}}
.card{{background:#eef3f8;padding:10px;border-radius:7px}}select,textarea{{width:100%;box-sizing:border-box;padding:7px;margin:5px 0 10px}}
textarea{{min-height:55px}}button{{padding:9px 14px;margin-right:8px}}.sticky{{position:sticky;top:0;background:#f3f5f7;padding:9px 0;z-index:2}}
.tag{{font-size:12px;background:#e3eaf1;border-radius:999px;padding:3px 8px}}
</style></head><body><h1>PM V1.5 H2：最终 Bank 检索资格审核</h1>
<div class="note"><b>这是 RAG 建库后的最后一份检索人评。</b>判断当前冻结的
topic-agnostic technique Bank 是否存在合格调用机会，以及展示的卡是否适用。
“用户需要帮助”不自动等于 RS opportunity；事实资源、医疗法律、高危安全、寒暄和结束
应由普通 LLM、其他资源或独立流程处理。</div>
<div class="note warn">候选是透明检索与 BGE 各 Top‑3 的盲化并集，页面不显示方法、
排名或分数。acceptable 表示这张卡现在可以安全、非冗余地注入；hard exclusion 表示
违反明确边界、未经许可建议、无依据推断、错误负担或超出普通技术库范围。两者不能同时勾。</div>
<div class="sticky"><button onclick="download()">导出 JSONL</button><span id="progress"></span></div>
<div id="root"></div><script>
const DATA={data};const META={meta};const KEY="pm15_h2_retrieval_v1";
let S=JSON.parse(localStorage.getItem(KEY)||"{{}}");
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const dlg=x=>(x||[]).map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");
function state(id){{S[id]??={{opportunity:"",acceptable:[],hard:[],reviewed:false,notes:""}};return S[id];}}
function save(){{localStorage.setItem(KEY,JSON.stringify(S));progress();}}
function setv(id,k,v){{state(id)[k]=v;save();}}
function toggle(id,k,n,on){{const s=state(id);s[k]=s[k].filter(x=>x!==n);if(on)s[k].push(n);s[k].sort((a,b)=>a-b);if(k==="acceptable"&&on)s.hard=s.hard.filter(x=>x!==n);if(k==="hard"&&on)s.acceptable=s.acceptable.filter(x=>x!==n);save();render();}}
function select(id,k,opts){{const v=state(id)[k];return `<select onchange="setv('${{id}}','${{k}}',this.value)"><option value="">请选择</option>${{opts.map(x=>`<option value="${{x}}" ${{v===x?"selected":""}}>${{x}}</option>`).join("")}}</select>`;}}
function render(){{document.getElementById("root").innerHTML=DATA.map((x,i)=>{{const s=state(x.review_item_id);return `<section class="item">
<h2>${{i+1}}/48</h2><div class="dialogue">${{esc(dlg(x.visible_dialogue))}}</div>
<label><b>当前冻结的普通技术 Bank 是否存在合格 RS opportunity？</b>${{select(x.review_item_id,"opportunity",["yes","no","uncertain"])}}</label>
${{x.candidates.length?x.candidates.map(c=>`<div class="candidate"><b>候选 ${{c.candidate_number}}</b>
<div class="card"><b>${{esc(c.card.strategy_family)}} · ${{esc(c.card.execution_profile)}}</b><br>
<b>Move:</b> ${{esc(c.card.support_move)}}<br><b>Use:</b> ${{esc(c.card.when_to_use)}}<br>
<b>Do not:</b> ${{esc(c.card.when_not_to_use)}}<br><b>Guidance:</b> ${{esc(c.card.prompt_guidance)}}</div>
<label><input type="checkbox" ${{s.acceptable.includes(c.candidate_number)?"checked":""}} onchange="toggle('${{x.review_item_id}}','acceptable',${{c.candidate_number}},this.checked)"> acceptable</label>
<label><input type="checkbox" ${{s.hard.includes(c.candidate_number)?"checked":""}} onchange="toggle('${{x.review_item_id}}','hard',${{c.candidate_number}},this.checked)"> hard exclusion</label></div>`).join(""):`<p class="tag">两个检索器均 abstain；仍需独立判断该 abstention 是否正确。</p>`}}
<label><input type="checkbox" ${{s.reviewed?"checked":""}} onchange="setv('${{x.review_item_id}}','reviewed',this.checked)"> 本项已完成</label>
<textarea placeholder="可选：机会、覆盖缺口或 hard exclusion 的简短理由" onchange="setv('${{x.review_item_id}}','notes',this.value)">${{esc(s.notes)}}</textarea>
</section>`;}}).join("");progress();}}
function collect(){{return DATA.map(x=>{{const s=state(x.review_item_id);return {{protocol:META.protocol,review_item_id:x.review_item_id,h2_state_id:x.h2_state_id,rs_opportunity:s.opportunity,acceptable_candidate_numbers:s.acceptable,hard_exclusion_candidate_numbers:s.hard,reviewed:s.reviewed,notes:s.notes,annotator_id:""}};}});}}
function progress(){{const r=collect();document.getElementById("progress").textContent=`${{r.filter(x=>x.reviewed&&x.rs_opportunity).length}} / ${{r.length}} complete`;}}
function download(){{const text=collect().map(x=>JSON.stringify(x)).join("\\n")+"\\n";const b=new Blob([text],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="pm_v1_5_h2_retrieval_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
render();</script></body></html>"""


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
        "--h2-states",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h2_retrieval_state_freeze_v1/"
        "h2_states_private.jsonl",
    )
    parser.add_argument(
        "--h2-state-manifest",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h2_retrieval_state_freeze_v1/manifest.json",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--bank-manifest",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/manifest.json",
    )
    parser.add_argument("--bge-model", type=Path, default=DEFAULT_BGE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate",
    )
    args = parser.parse_args()

    state_manifest = json.loads(
        args.h2_state_manifest.read_text(encoding="utf-8")
    )
    bank_manifest = json.loads(
        args.bank_manifest.read_text(encoding="utf-8")
    )
    if state_manifest["status"] != (
        "H2_STATES_FROZEN_BEFORE_FINAL_BANK_AND_RERANKER"
    ):
        raise RuntimeError("H2 state freeze is not valid")
    if bank_manifest["status"] != "BANK_CONTENT_FROZEN_H2_RETRIEVAL_PENDING":
        raise RuntimeError("final Bank content is not frozen")
    states = [dict(row) for row in iter_jsonl(args.h2_states)]
    cards = [dict(row) for row in iter_jsonl(args.bank)]
    if len(states) != 48 or len(cards) != 80:
        raise RuntimeError("expected 48 H2 states and 80 final cards")
    cards_by_id = {str(row["card_id"]): row for row in cards}

    prepared: list[dict[str, Any]] = []
    all_texts: list[str] = []
    text_index: dict[str, int] = {}

    def bind_text(text: str) -> int:
        if text not in text_index:
            text_index[text] = len(all_texts)
            all_texts.append(text)
        return text_index[text]

    for state in states:
        dialogue = list(state["visible_dialogue"])
        seekers = _seeker_texts(dialogue)
        latest = seekers[-1]
        recent = " ".join(seekers[-3:])
        flags = repaired_observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=recent,
            visible_dialogue=dialogue,
        )
        query = _query(dialogue, flags)
        transparent = repaired_rank_applicable_v4_cards(
            query=query,
            current_user_text=latest,
            recent_user_text=recent,
            visible_dialogue=dialogue,
            cards=cards,
        )
        prepared.append(
            {
                "state": state,
                "flags": flags,
                "query": query,
                "query_text_index": bind_text(query),
                "transparent": transparent,
            }
        )
    for card in cards:
        bind_text(_card_text(card))

    bakeoff = _load_bakeoff_module()
    embeddings = bakeoff._encode_bge(
        all_texts,
        model_path=args.bge_model,
        batch_size=args.batch_size,
    )
    card_vector_index = {
        card_id: text_index[_card_text(card)]
        for card_id, card in cards_by_id.items()
    }

    packet: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for row in prepared:
        state = row["state"]
        query_vector = embeddings[row["query_text_index"]]
        bge = []
        for candidate in row["transparent"]:
            value = dict(candidate)
            value["bge_cosine"] = round(
                float(
                    query_vector
                    @ embeddings[
                        card_vector_index[str(candidate["card_id"])]
                    ]
                ),
                8,
            )
            bge.append(value)
        bge.sort(
            key=lambda value: (
                int(value["compatibility_tier"]),
                float(value["bge_cosine"]),
                str(value["card_id"]),
            ),
            reverse=True,
        )
        public_candidates, number_by_id = _union_candidates(
            state_id=str(state["h2_state_id"]),
            transparent=row["transparent"],
            bge=bge,
            cards_by_id=cards_by_id,
        )
        review_id = "h2_retrieval_" + stable_hex(
            PROTOCOL, state["h2_state_id"], n=20
        )
        packet.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "h2_state_id": state["h2_state_id"],
                "visible_dialogue": state["visible_dialogue"],
                "candidates": public_candidates,
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "h2_state_id": state["h2_state_id"],
                "source_dialogue_id": state["source_dialogue_id"],
                "selection_stratum": state["selection_stratum"],
                "ordinary_rag_hard_off": row["flags"][
                    "ordinary_rag_hard_off"
                ],
                "ordinary_rag_hard_off_reasons": row["flags"][
                    "ordinary_rag_hard_off_reasons"
                ],
                "transparent_ranked_card_ids": [
                    value["card_id"]
                    for value in row["transparent"][:TOP_K]
                ],
                "transparent_ranked_candidate_numbers": [
                    number_by_id[value["card_id"]]
                    for value in row["transparent"][:TOP_K]
                ],
                "bge_ranked_card_ids": [
                    value["card_id"] for value in bge[:TOP_K]
                ],
                "bge_ranked_candidate_numbers": [
                    number_by_id[value["card_id"]] for value in bge[:TOP_K]
                ],
                "candidate_number_to_card_id": {
                    str(number): card_id
                    for card_id, number in number_by_id.items()
                },
                "transparent_full_candidate_count": len(
                    row["transparent"]
                ),
            }
        )

    packet.sort(key=lambda row: str(row["review_item_id"]))
    private.sort(key=lambda row: str(row["review_item_id"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.out_dir / "h2_review_packet.jsonl"
    private_path = args.out_dir / "private_ranker_audit.jsonl"
    template_path = args.out_dir / "human_annotation_template.jsonl"
    _write_jsonl(packet_path, packet)
    _write_jsonl(private_path, private)
    _write_jsonl(
        template_path,
        [
            {
                "protocol": PROTOCOL,
                "review_item_id": row["review_item_id"],
                "h2_state_id": row["h2_state_id"],
                "rs_opportunity": "",
                "acceptable_candidate_numbers": [],
                "hard_exclusion_candidate_numbers": [],
                "reviewed": False,
                "notes": "",
                "annotator_id": "",
            }
            for row in packet
        ],
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_SHOT_H2_HUMAN_QUALIFICATION",
        "items": len(packet),
        "states_with_no_candidates": sum(
            not row["candidates"] for row in packet
        ),
        "maximum_candidates_per_state": max(
            len(row["candidates"]) for row in packet
        ),
        "opportunity_definition": (
            "yes only when the frozen topic-agnostic technique Bank contains "
            "at least one safe, nonredundant card whose injection is justified "
            "now; generic need for help is insufficient"
        ),
        "blinding": {
            "ranker_identity_hidden": True,
            "ranks_hidden": True,
            "scores_hidden": True,
            "selection_stratum_hidden": True,
            "candidate_order": "stable protocol hash",
        },
        "qualification_gates": {
            "opportunity_uncertain_rate_maximum": 0.10,
            "opportunity_balanced_accuracy_minimum": 0.90,
            "top1_acceptable_minimum": 0.80,
            "top3_acceptable_recall_minimum": 0.90,
            "hard_exclusion_in_method_top3_maximum": 0,
        },
        "ranker_selection_rule": (
            "Select BGE only if its H2 Top-1 acceptable rate exceeds "
            "transparent by at least 0.05, its Top-3 recall is no lower, "
            "and it introduces no additional hard exclusions. Otherwise "
            "retain transparent tier+lexical."
        ),
        "bank_content_change_after_h2_forbidden": True,
        "inputs": {
            "h2_states": str(args.h2_states.relative_to(ROOT)),
            "h2_states_sha256": sha256_file(args.h2_states),
            "h2_state_manifest": str(
                args.h2_state_manifest.relative_to(ROOT)
            ),
            "bank": str(args.bank.relative_to(ROOT)),
            "bank_sha256": sha256_file(args.bank),
            "bank_manifest": str(args.bank_manifest.relative_to(ROOT)),
            "bge_model": str(args.bge_model),
            "repair_protocol": REPAIR_PROTOCOL,
        },
    }
    html_path = args.out_dir / "human_retrieval_review.html"
    html_path.write_text(_render_html(packet, manifest), encoding="utf-8")
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (packet_path, private_path, template_path, html_path)
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": manifest["status"],
            "items": len(packet),
            "no_candidate_states": manifest["states_with_no_candidates"],
            "maximum_candidates": manifest["maximum_candidates_per_state"],
            "human_review": str(html_path),
        }
    )


if __name__ == "__main__":
    main()
