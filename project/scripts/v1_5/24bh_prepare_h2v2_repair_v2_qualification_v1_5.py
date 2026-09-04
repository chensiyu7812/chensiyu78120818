#!/usr/bin/env python3
"""INVALIDATED historical H2v2 packet builder.

The original implementation passed ``query=""`` to a lexical ranker.  Its
82-item output therefore does not represent runtime retrieval and must not be
human-reviewed.  repair_v2 also changes the treatment used by the already
completed RS effect experiments, so it is parked for a future version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, stable_hex
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_strategy_rag_repair import repair_v2_rank_applicable_cards


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h2v2-repair-v2-qualification-v1"
TOP_K = 3

# Frozen before any label exists, same as 24ap's gates.
QUALIFICATION_GATES = {
    "opportunity_uncertain_rate_maximum": 0.10,
    "opportunity_balanced_accuracy_minimum": 0.90,
    "top1_acceptable_minimum": 0.80,
    "top3_acceptable_recall_minimum": 0.90,
    "hard_exclusion_in_top3_maximum": 0,
}


def _seeker_texts(dialogue: list[dict[str, Any]]) -> list[str]:
    return [
        normalize_space(turn.get("content", ""))
        for turn in dialogue
        if str(turn.get("speaker")) == "seeker"
        and normalize_space(turn.get("content", ""))
    ]


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


def main() -> None:
    raise RuntimeError(
        "INVALIDATED_NOT_FOR_REVIEW: the historical H2v2 packet used an "
        "empty lexical query and repair_v2 is not part of the frozen V1.5 "
        "treatment. Do not regenerate or annotate this packet."
    )
    states = list(
        iter_jsonl(
            ROOT / "outputs/pm_v1_5_h2v2_retrieval_state_freeze_v1/"
            "h2v2_states_private.jsonl"
        )
    )
    cards = list(
        iter_jsonl(
            ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
            "strategy_cards_v4_final.jsonl"
        )
    )
    cards_by_id = {str(card["card_id"]): card for card in cards}

    packet: list[dict[str, Any]] = []
    for state in states:
        dialogue = state["visible_dialogue"]
        seeker = _seeker_texts(dialogue)
        current = seeker[-1] if seeker else ""
        recent = " ".join(seeker[-3:])
        ranked = repair_v2_rank_applicable_cards(
            query="",
            current_user_text=current,
            recent_user_text=recent,
            visible_dialogue=dialogue,
            cards=cards,
        )
        top = ranked[:TOP_K]
        candidates = [
            {
                "candidate_number": number,
                "candidate_id": "h2v2_candidate_"
                + stable_hex(
                    PROTOCOL, state["h2v2_state_id"], row["card_id"], n=20
                ),
                "card": _public_card(cards_by_id[str(row["card_id"])]),
            }
            for number, row in enumerate(top, start=1)
        ]
        packet.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": "h2v2_repair_v2_"
                + stable_hex(PROTOCOL, state["h2v2_state_id"], n=20),
                "h2v2_state_id": state["h2v2_state_id"],
                "selection_stratum": state["selection_stratum"],
                "visible_dialogue": dialogue,
                "candidates": candidates,
                "candidate_count": len(candidates),
            }
        )

    out_dir = ROOT / "outputs/pm_v1_5_h2v2_repair_v2_qualification_v1_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = out_dir / "h2v2_review_packet.jsonl"
    packet_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in packet
        ),
        encoding="utf-8",
    )

    no_candidate_items = sum(1 for row in packet if row["candidate_count"] == 0)

    html = _render_html(packet)
    html_path = out_dir / "human_review.html"
    html_path.write_text(html, encoding="utf-8")

    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_H2V2_REPAIR_V2_HUMAN_QUALIFICATION",
        "purpose": (
            "One-shot fresh qualification of repair_v2_rank_applicable_cards "
            "on 82 states disjoint from H1, H2v1, the RS effect study, and "
            "the ESConv frozen validation. BGE is not retested; it already "
            "lost to transparent in H2v1 and repair_v2 does not change it."
        ),
        "review_items": len(packet),
        "items_with_zero_candidates": no_candidate_items,
        "top_k_shown": TOP_K,
        "method": "repair_v2_rank_applicable_cards (transparent tier+lexical, "
        "post-H2 evidence-gated overlay)",
        "scores_hidden_from_reviewer": True,
        "qualification_gates": QUALIFICATION_GATES,
        "bank_content_frozen": True,
        "states_frozen_before_this_scoring": True,
        "inputs": {
            "states": "outputs/pm_v1_5_h2v2_retrieval_state_freeze_v1/"
            "h2v2_states_private.jsonl",
            "bank": "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
            "strategy_cards_v4_final.jsonl",
        },
        "outputs": {packet_path.name: sha256_file(packet_path)},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "status": manifest["status"],
            "review_items": len(packet),
            "items_with_zero_candidates": no_candidate_items,
            "out_dir": str(out_dir),
        }
    )


def _render_html(packet: list[dict[str, Any]]) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 H2v2 · repair_v2 一次性资格审核</title><style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:20px;background:#f3f5f7;color:#17202a;line-height:1.5}}
.note,.item{{background:#fff;border:1px solid #d6dde5;border-radius:10px;padding:16px;margin:14px 0}}
.note{{border-left:5px solid #245f9e}}.warn{{border-left-color:#a85b00}}
.dialogue{{white-space:pre-wrap;background:#edf2f6;border-radius:7px;padding:11px}}
.candidate{{border-left:4px solid #8ca0b3;background:#fafafa;padding:11px 14px;margin:14px 0}}
.card{{background:#eef3f8;padding:10px;border-radius:7px}}select,textarea{{width:100%;box-sizing:border-box;padding:7px;margin:5px 0 10px}}
textarea{{min-height:55px}}button{{padding:9px 14px;margin-right:8px}}.sticky{{position:sticky;top:0;background:#f3f5f7;padding:9px 0;z-index:2}}
.tag{{font-size:12px;background:#e3eaf1;border-radius:999px;padding:3px 8px}}
</style></head><body><h1>PM V1.5 H2v2：repair_v2 一次性检索资格审核</h1>
<div class="note"><b>这是repair_v2修复后的一次性检索确认，82条state，全部与H1、H2v1、
RS效应研究(32组)、ESConv冻结验证(40对)不重叠。</b>只展示repair_v2这一个方法的Top-3
结果（不再重测BGE，H2v1已确认BGE更差且repair_v2不改变BGE本身）。页面不显示排名分数。
判断当前是否存在合格RS opportunity；acceptable表示这张卡现在可以安全、非冗余地注入；
hard exclusion表示违反明确边界、未经许可建议、无依据推断、错误负担或超出普通技术库
范围。两者不能同时勾。</div>
<div class="note warn">这一批"候选为空"的state，代表repair_v2判定为hard-off（不应该
检索），仍需人工确认opportunity是否真的应该是no。</div>
<div class="sticky"><button onclick="download()">导出 JSONL</button><span id="progress"></span></div>
<div id="root"></div><script>
const DATA={data};const KEY="pm15_h2v2_repair_v2_v1";
let S=JSON.parse(localStorage.getItem(KEY)||"{{}}");
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const dlg=x=>(x||[]).map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");
function state(id){{S[id]??={{opportunity:"",acceptable:[],hard:[],reviewed:false,notes:""}};return S[id];}}
function save(){{localStorage.setItem(KEY,JSON.stringify(S));progress();}}
function setv(id,k,v){{state(id)[k]=v;save();}}
function toggle(id,k,n,on){{const s=state(id);s[k]=s[k].filter(x=>x!==n);if(on)s[k].push(n);s[k].sort((a,b)=>a-b);if(k==="acceptable"&&on)s.hard=s.hard.filter(x=>x!==n);if(k==="hard"&&on)s.acceptable=s.acceptable.filter(x=>x!==n);save();render();}}
function select(id,k,opts){{const v=state(id)[k];return `<select onchange="setv('${{id}}','${{k}}',this.value)"><option value="">请选择</option>${{opts.map(x=>`<option value="${{x}}" ${{v===x?"selected":""}}>${{x}}</option>`).join("")}}</select>`;}}
function render(){{document.getElementById("root").innerHTML=DATA.map((x,i)=>{{const s=state(x.review_item_id);return `<section class="item">
<h2>${{i+1}}/${{DATA.length}} <span class="tag">${{esc(x.selection_stratum)}}</span></h2><div class="dialogue">${{esc(dlg(x.visible_dialogue))}}</div>
<label><b>当前修复后的检索是否存在合格 RS opportunity？</b>${{select(x.review_item_id,"opportunity",["yes","no","uncertain"])}}</label>
${{x.candidates.length?x.candidates.map(c=>`<div class="candidate"><b>候选 ${{c.candidate_number}}</b>
<div class="card"><b>${{esc(c.card.strategy_family)}} · ${{esc(c.card.core_submove_id)}}</b><br>
${{esc(c.card.support_move)}}<br><i>Use:</i> ${{esc(c.card.when_to_use)}}<br><i>Not use:</i> ${{esc(c.card.when_not_to_use)}}</div>
<label><input type="checkbox" ${{s.acceptable.includes(c.candidate_number)?"checked":""}} onchange="toggle('${{x.review_item_id}}','acceptable',${{c.candidate_number}},this.checked)"> acceptable</label>
<label><input type="checkbox" ${{s.hard.includes(c.candidate_number)?"checked":""}} onchange="toggle('${{x.review_item_id}}','hard',${{c.candidate_number}},this.checked)"> hard exclusion</label>
</div>`).join(""):'<p><i>repair_v2 判定为 hard-off，无候选卡展示。</i></p>'}}
<label><input type="checkbox" ${{s.reviewed?"checked":""}} onchange="setv('${{x.review_item_id}}','reviewed',this.checked)"> 已完成本条</label>
<textarea placeholder="可选说明" onchange="setv('${{x.review_item_id}}','notes',this.value)">${{esc(s.notes)}}</textarea>
</section>`;}}).join("");progress();}}
function rows(){{return DATA.map(x=>{{const s=state(x.review_item_id);return {{protocol:x.protocol,review_item_id:x.review_item_id,h2v2_state_id:x.h2v2_state_id,...s,annotator_id:""}};}});}}
function progress(){{const r=rows();document.getElementById("progress").textContent=` 已完成 ${{r.filter(x=>x.reviewed).length}} / ${{r.length}}`;}}
function download(){{const b=new Blob([rows().map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="h2v2_repair_v2_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
