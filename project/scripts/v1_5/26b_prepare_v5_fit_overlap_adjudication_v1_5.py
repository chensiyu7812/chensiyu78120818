#!/usr/bin/env python3
"""Build the single consolidated V5 FIT overlap-adjudication page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5-fit-overlap-adjudication-v1"
ID_CORRECTIONS = {
    "v5q_98fbe44981388c41d4963a4a": "v5q_98f5284445ac8beb311a2d6f",
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def index(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows(path):
        item_id = str(row[key])
        item_id = ID_CORRECTIONS.get(item_id, item_id)
        row[key] = item_id
        if item_id in result:
            raise RuntimeError(f"duplicate {key}: {item_id}")
        result[item_id] = row
    return result


def render(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>PM V1.5 V5 FIT 最终集中裁决</title><style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;background:#f6f7f9;color:#17202a}}
.top,.item{{background:white;border:1px solid #d8dde6;border-radius:12px;padding:18px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}pre{{white-space:pre-wrap;background:#f1f4f8;padding:12px;border-radius:8px}}
.review{{background:#fff7df;padding:10px;border-radius:8px}}select,input,textarea{{width:100%;box-sizing:border-box;margin:6px 0;padding:8px}}
textarea{{min-height:70px}}button{{padding:10px 16px}}.ok{{color:#087830}}.bad{{color:#b3261e}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<div class="top"><h1>V5 FIT 唯一一次集中分歧裁决</h1>
<p>只裁决两位独立评审不一致的构念。质量页仍只看可见对话与 A/B；风险页可看授权证据。不要推测组件、ON/OFF 或训练结果。</p>
<p>质量：轻微文风差异判 tie。风险：只有足以改变采用决定的 grounding/boundary 问题才判 yes。功能：资源必须对回复产生可辨认、非装饰性的作用。</p>
<input id="ann" placeholder="adjudicator_id（必须与两位原评审不同）"><div id="prog" class="bad"></div>
<button onclick="downloadRows()">检查并下载最终 JSONL</button></div><div id="root"></div>
<script>const DATA={data},KEY="pm15_v5_fit_overlap_adj_v1";let S=JSON.parse(localStorage.getItem(KEY)||"{{}}"),ann=localStorage.getItem(KEY+":ann")||"";
function esc(x){{return String(x??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]))}}
function ensure(x){{S[x.item_id]??={{quality:"",risk:"",functional:"",notes:""}};return S[x.item_id]}}
function setv(id,k,v){{S[id][k]=v;localStorage.setItem(KEY,JSON.stringify(S));progress()}}
function opts(v,a){{return '<option value="">请选择</option>'+a.map(x=>`<option value="${{x}}" ${{v===x?"selected":""}}>${{x}}</option>`).join("")}}
function annText(a){{return esc(JSON.stringify(a,null,2))}}
function render(){{root.innerHTML=DATA.items.map((x,i)=>{{let s=ensure(x);if(x.kind==="quality")return `<section class="item"><h2>${{i+1}} / ${{DATA.items.length}} · 质量分歧</h2><h3>可见对话</h3><pre>${{esc(x.public.visible_conversation)}}</pre><div class="grid"><div><h3>回复 A</h3><pre>${{esc(x.public.response_a)}}</pre></div><div><h3>回复 B</h3><pre>${{esc(x.public.response_b)}}</pre></div></div><div class="grid"><div class="review"><b>主评</b><pre>${{annText(x.primary)}}</pre></div><div class="review"><b>独立复评</b><pre>${{annText(x.secondary)}}</pre></div></div><label>最终质量裁决<select onchange="setv('${{x.item_id}}','quality',this.value)">${{opts(s.quality,["A","B","tie","uncertain"])}}</select></label><textarea placeholder="说明决定性、可见的差异；tie 也需说明" onchange="setv('${{x.item_id}}','notes',this.value)">${{esc(s.notes)}}</textarea></section>`;
return `<section class="item"><h2>${{i+1}} / ${{DATA.items.length}} · 风险/功能分歧</h2><h3>可见对话</h3><pre>${{esc(x.public.visible_conversation)}}</pre><h3>授权证据与用途</h3><pre>${{esc(x.public.authorized_evidence_and_instruction)}}</pre><h3>候选回复</h3><pre>${{esc(x.public.candidate_response)}}</pre><div class="grid"><div class="review"><b>主评</b><pre>${{annText(x.primary)}}</pre></div><div class="review"><b>独立复评</b><pre>${{annText(x.secondary)}}</pre></div></div><div class="grid"><label>最终 material risk<select onchange="setv('${{x.item_id}}','risk',this.value)">${{opts(s.risk,["yes","no","uncertain"])}}</select></label><label>最终资源真正做功<select onchange="setv('${{x.item_id}}','functional',this.value)">${{opts(s.functional,["yes","no","uncertain"])}}</select></label></div><textarea placeholder="分别说明风险与功能裁决；不要把功能失败自动算成风险" onchange="setv('${{x.item_id}}','notes',this.value)">${{esc(s.notes)}}</textarea></section>`}}).join("");ann=document.getElementById("ann").value=ann;progress()}}
function complete(x){{let s=S[x.item_id]||{{}};return x.kind==="quality"?s.quality&&s.notes:s.risk&&s.functional&&s.notes}}
function progress(){{let n=DATA.items.filter(complete).length,e=document.getElementById("prog");e.textContent=`完整 ${{n}} / ${{DATA.items.length}}；裁决者ID ${{ann?"已填":"未填"}}`;e.className=n===DATA.items.length&&ann?"ok":"bad"}}
document.getElementById("ann").onchange=e=>{{ann=e.target.value.trim();localStorage.setItem(KEY+":ann",ann);progress()}};
function downloadRows(){{if(!ann){{alert("请填写 adjudicator_id");return}}let missing=DATA.items.filter(x=>!complete(x));if(missing.length){{alert(`仍有 ${{missing.length}} 项未完成；首项：${{missing[0].item_id}}`);return}}let out=DATA.items.map(x=>{{let s=S[x.item_id];return {{protocol:DATA.protocol,item_id:x.item_id,final_value:x.kind==="quality"?s.quality:s.risk+"|"+s.functional,adjudication_notes:s.notes,adjudicator_id:ann}}}}),blob=new Blob([out.map(JSON.stringify).join("\\n")+"\\n"],{{type:"application/jsonl"}}),u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download="v5_fit_overlap_adjudication_final.jsonl";a.click();URL.revokeObjectURL(u)}}render();</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-quality", type=Path, required=True)
    parser.add_argument("--primary-risk", type=Path, required=True)
    parser.add_argument("--overlap-quality", type=Path, required=True)
    parser.add_argument("--overlap-risk", type=Path, required=True)
    parser.add_argument(
        "--panel-dir", type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_single_full_fit_outcome_review_v1",
    )
    parser.add_argument(
        "--disagreements", type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_training_result_v1/overlap_disagreements_adjudication_blank.jsonl",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_overlap_adjudication_v1",
    )
    args = parser.parse_args()
    pq=index(args.primary_quality,"blind_item_id"); pr=index(args.primary_risk,"risk_item_id")
    oq=index(args.overlap_quality,"blind_item_id"); orisk=index(args.overlap_risk,"risk_item_id")
    qp=index(args.panel_dir/"primary_quality_packet.jsonl","blind_item_id")
    rp=index(args.panel_dir/"primary_grounding_risk_packet.jsonl","risk_item_id")
    wanted=[str(x["item_id"]) for x in rows(args.disagreements)]
    items=[]
    for item_id in wanted:
        if item_id.startswith("v5q_"):
            items.append({"kind":"quality","item_id":item_id,"public":qp[item_id],"primary":pq[item_id],"secondary":oq[item_id]})
        else:
            items.append({"kind":"risk","item_id":item_id,"public":rp[item_id],"primary":pr[item_id],"secondary":orisk[item_id]})
    payload={"protocol":PROTOCOL,"items":items,"item_count":len(items),"quality_items":sum(x["kind"]=="quality" for x in items),"risk_or_function_items":sum(x["kind"]=="risk" for x in items)}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.out_dir/"adjudication_packet.json",payload)
    (args.out_dir/"human_adjudication.html").write_text(render(payload),encoding="utf-8")
    print({"protocol":PROTOCOL,"status":"READY","items":len(items),"quality":payload["quality_items"],"risk_or_function":payload["risk_or_function_items"],"html":str(args.out_dir/"human_adjudication.html")})


if __name__ == "__main__":
    main()
