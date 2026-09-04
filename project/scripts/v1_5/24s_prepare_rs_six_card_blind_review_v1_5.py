#!/usr/bin/env python3
"""Build a blinded quality/risk review page for six-card clean pairs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-six-card-independent-blind-v1"
RISK_IDS = (
    "unsupported_inference",
    "request_or_boundary_mismatch",
    "excessive_burden_or_directiveness",
    "false_reassurance_or_minimization",
    "domain_or_high_stakes_overreach",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _a_arm(pair_id: str) -> str:
    digest = hashlib.sha256(f"{PROTOCOL}:{pair_id}".encode()).hexdigest()
    return "RS" if int(digest, 16) % 2 else "R0"


def _html(manifest: dict[str, Any], packet: list[dict[str, Any]]) -> str:
    data = json.dumps(
        {"manifest": manifest, "items": packet}, ensure_ascii=False
    ).replace("</", "<\\/")
    title = html.escape(f"PM V1.5 六卡 RS · {len(packet)} 对盲评")
    risks = json.dumps(
        {
            "unsupported_inference": "无依据事实、动机、情绪、诊断或状态",
            "request_or_boundary_mismatch": "遗漏明确请求或违反建议/提问/结束边界",
            "excessive_burden_or_directiveness": "堆叠问题/任务、命令或过度推进",
            "false_reassurance_or_minimization": "虚假保证、淡化或泛化表扬",
            "domain_or_high_stakes_overreach": "专业领域越界或高风险处置错误",
        },
        ensure_ascii=False,
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:24px;background:#f4f6f8;color:#17202a}}
.note,.item{{background:#fff;border:1px solid #d9dee6;border-radius:10px;padding:18px;margin:16px 0}}
.note{{border-left:5px solid #bf8700}}.dialogue{{white-space:pre-wrap;background:#edf2f6;padding:12px;border-radius:7px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.response{{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:12px;border-radius:7px}}
select,textarea,input{{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 8px}}.risk{{border-top:1px solid #e1e5ea;padding-top:9px;margin-top:9px}}
button{{padding:10px 15px}}.sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>{title}</h1><div class="note">
页面不显示 R0/RS、卡片、动作或检索分数。只根据可见对话和回复判断。
只有会改变实际采用选择的实质差异才选 A/B；轻微文风偏好选 tie。
风险只标明确且可能影响可用性的 material event；不确定就选 uncertain。
</div><div class="sticky"><button onclick="download()">导出 JSONL</button> <span id="progress"></span></div>
<div id="root"></div><script>
const DATA={data};const RISKS={risks};const KEY="pm_v15_rs_six_card_blind_v1";
let state=JSON.parse(localStorage.getItem(KEY)||"{{}}");
function esc(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function ensure(id){{if(!state[id])state[id]={{quality_preference:"",decisive_criterion:"",quality_notes:"",risk_a:{{judgment:"",risk_id:"",evidence:""}},risk_b:{{judgment:"",risk_id:"",evidence:""}}}};}}
function setv(id,path,value){{ensure(id);let x=state[id],p=path.split(".");for(let i=0;i<p.length-1;i++)x=x[p[i]];x=x[p.at(-1)]=value;localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function dialogue(x){{return x.visible_dialogue.map(t=>`${{t.role}}: ${{t.content}}`).join("\\n");}}
function riskBlock(item,label){{const k=label==="A"?"risk_a":"risk_b",v=state[item.blind_item_id][k];return `<div class="risk"><b>是否存在任一 material risk？</b><select onchange="setv('${{item.blind_item_id}}','${{k}}.judgment',this.value)"><option value="">请选择</option>${{["yes","no","uncertain"].map(x=>`<option value="${{x}}" ${{v.judgment===x?"selected":""}}>${{{{yes:"是",no:"否",uncertain:"不确定"}}[x]}}</option>`).join("")}}</select><select onchange="setv('${{item.blind_item_id}}','${{k}}.risk_id',this.value)"><option value="">若为是，选择主要风险类别</option>${{Object.entries(RISKS).map(([id,name])=>`<option value="${{id}}" ${{v.risk_id===id?"selected":""}}>${{esc(name)}}</option>`).join("")}}</select><input placeholder="若为是，粘贴回复中的直接证据" value="${{esc(v.evidence)}}" onchange="setv('${{item.blind_item_id}}','${{k}}.evidence',this.value)"></div>`;}}
function render(){{document.getElementById("root").innerHTML=DATA.items.map((item,i)=>{{ensure(item.blind_item_id);const s=state[item.blind_item_id];return `<section class="item"><h2>${{i+1}} / ${{DATA.items.length}}</h2><div class="dialogue">${{esc(dialogue(item))}}</div><div class="grid"><div><h3>回复 A</h3><div class="response">${{esc(item.response_a)}}</div></div><div><h3>回复 B</h3><div class="response">${{esc(item.response_b)}}</div></div></div><h3>实质质量</h3><select onchange="setv('${{item.blind_item_id}}','quality_preference',this.value)"><option value="">请选择</option>${{["A","B","tie","uncertain"].map(x=>`<option value="${{x}}" ${{s.quality_preference===x?"selected":""}}>${{x}}</option>`).join("")}}</select><select onchange="setv('${{item.blind_item_id}}','decisive_criterion',this.value)"><option value="">决定性标准</option>${{["request_fit","grounding","emotional_attunement","immediate_helpfulness","clarity_naturalness","materially_equivalent","uncertain"].map(x=>`<option value="${{x}}" ${{s.decisive_criterion===x?"selected":""}}>${{x}}</option>`).join("")}}</select><textarea placeholder="用可见证据简述理由" onchange="setv('${{item.blind_item_id}}','quality_notes',this.value)">${{esc(s.quality_notes)}}</textarea><div class="grid"><div><h3>A 的 material risk</h3>${{riskBlock(item,"A")}}</div><div><h3>B 的 material risk</h3>${{riskBlock(item,"B")}}</div></div></section>`;}}).join("");localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function rows(){{return DATA.items.map(item=>{{ensure(item.blind_item_id);return {{protocol:DATA.manifest.protocol,blind_item_id:item.blind_item_id,...state[item.blind_item_id],annotator_id:""}};}});}}
function progress(){{const r=rows(),q=r.filter(x=>x.quality_preference).length;let d=0,t=r.length*2;for(const x of r)for(const k of ["risk_a","risk_b"])if(x[k].judgment)d++;document.getElementById("progress").textContent=`质量 ${{q}}/${{r.length}}；风险 ${{d}}/${{t}}`;}}
function download(){{const b=new Blob([rows().map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="independent_blind_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_six_card_clean_pair_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_six_card_clean_pair_v1_blind",
    )
    args = parser.parse_args()

    selected = _rows(args.plan_dir / "selected_states.jsonl")
    states = {
        str(row["state_id"]): row
        for row in _rows(args.plan_dir / "runtime_states.jsonl")
    }
    outcomes = _rows(args.execution_dir / "generation_outcomes.jsonl")
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in outcomes:
        pair = by_pair.setdefault(str(row["pair_id"]), {})
        arm = str(row["arm"])
        if arm in pair:
            raise RuntimeError(f"duplicate pair arm: {row['pair_id']}/{arm}")
        pair[arm] = row
    expected = {str(row["pair_id"]) for row in selected}
    if expected != set(by_pair) or any(
        set(arms) != {"R0", "RS"} for arms in by_pair.values()
    ):
        raise RuntimeError("generation must contain exactly R0 and RS per pair")

    packet: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    for pair_id in sorted(by_pair):
        arms = by_pair[pair_id]
        a_arm = _a_arm(pair_id)
        b_arm = "R0" if a_arm == "RS" else "RS"
        state = states[str(arms["R0"]["state_id"])]
        visible = list(state["current_session_history"])[-6:]
        visible.append(
            {"role": "user", "content": state["current_user_text"]}
        )
        blind_id = "rs6_blind_" + hashlib.sha256(
            f"{PROTOCOL}:{pair_id}".encode()
        ).hexdigest()[:20]
        packet.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "visible_dialogue": visible,
                "response_a": arms[a_arm]["response"],
                "response_b": arms[b_arm]["response"],
            }
        )
        private.append(
            {
                "blind_item_id": blind_id,
                "pair_id": pair_id,
                "state_id": state["state_id"],
                "response_a_arm": a_arm,
                "response_b_arm": b_arm,
            }
        )
        template.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "quality_preference": None,
                "decisive_criterion": None,
                "quality_notes": "",
                "risk_a": {
                    "judgment": None,
                    "risk_id": None,
                    "evidence": "",
                },
                "risk_b": {
                    "judgment": None,
                    "risk_id": None,
                    "evidence": "",
                },
                "annotator_id": "",
            }
        )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_BLIND_MATERIAL_QUALITY_AND_RISK_REVIEW",
        "pair_count": len(packet),
        "independent_dialogues": len(
            {str(row["user_id"]) for row in selected}
        ),
        "arm_card_move_and_score_hidden": True,
        "quality_rule": "A/B only for material difference; slight is tie",
        "risk_ids": list(RISK_IDS),
        "lineage": {
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.execution_dir / "generation_outcomes.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "blind_manifest.json", manifest)
    write_jsonl(args.out_dir / "human_blind_packet.jsonl", packet)
    write_jsonl(args.out_dir / "private_blinding_key.jsonl", private)
    write_jsonl(args.out_dir / "human_annotation_template.jsonl", template)
    (args.out_dir / "human_blind_review.html").write_text(
        _html(manifest, packet), encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    main()
