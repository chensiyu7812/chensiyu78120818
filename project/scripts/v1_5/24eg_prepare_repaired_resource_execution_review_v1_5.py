#!/usr/bin/env python3
"""Prepare the repaired Step2 execution review with separated constructs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text, stable_hex, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-repaired-resource-execution-human-check-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _render(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    title = "PM V1.5 · 修复后 generator 资源执行核验（10条）"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1080px;margin:auto;padding:22px;background:#f4f6f8;color:#17202a}}.note,.item{{background:#fff;border:1px solid #d6dce4;border-radius:10px;padding:17px;margin:15px 0}}.note{{border-left:5px solid #0969da}}.context,.resource,.response,.trace{{white-space:pre-wrap;padding:12px;border-radius:7px;line-height:1.5}}.context{{background:#edf2f6}}.resource{{background:#fff8c5;border:1px solid #d4a72c}}.response{{background:#f4f0ff;border:1px solid #a99bd4}}.trace{{background:#eefbf3;border:1px solid #85c79f}}select,textarea,input{{width:100%;box-sizing:border-box;padding:8px;margin:5px 0 10px}}textarea{{min-height:70px}}button{{padding:9px 14px}}.sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}.ok{{color:#1a7f37}}.bad{{color:#cf222e}}</style></head><body><h1>{html.escape(title)}</h1>
<div class="note"><b>本页只测 Step 2（generator），不改 PM、检索或回复质量标签。</b><br>
先判该条能否从可见材料裁决；再把四件事分开：①声明是否匹配正文；②use/ignore 决定是否合理；③资源是否真正发挥功能；④是否有足以阻止采用的 interaction-and-grounding misuse。<br>
历史事实只能作为“过去曾发生/曾奏效”的证据，不能无对冲地升级为当前事实或原因。提到同一主题、泛泛说“类似”、或在当前消息已经给出同一信息时，不算资源做功。<br>
这 10 条是新的正向机会开发态，全部由 generator 声明 use；因此本页不能单独证明 ignore 路径。</div>
<div class="sticky"><input id="ann" placeholder="annotator_id"><button onclick="downloadRows()">导出 JSONL</button> <span id="prog"></span></div><div id="root"></div><script>
const DATA={payload},KEY="pm_v15_repaired_resource_exec_v1";let state=JSON.parse(localStorage.getItem(KEY)||"{{}}"),ann=localStorage.getItem(KEY+":ann")||"";
function esc(s){{return String(s??"").replace(/[&<>\"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;","'":"&#39;"}}[c]));}}function blank(){{return {{adjudicable:"",declaration_matches_response:"",resource_decision_appropriate:"",functional_execution:"",material_misuse:"",misuse_categories:"",literal_response_excerpt:"",review_notes:""}}}}function ensure(id){{if(!state[id])state[id]=blank()}}function setv(id,k,v){{ensure(id);state[id][k]=v;localStorage.setItem(KEY,JSON.stringify(state));progress();}}function opts(v,vals){{return vals.map(x=>`<option value="${{x}}" ${{v===x?"selected":""}}>${{x}}</option>`).join("")}}function context(x){{return (x.history||[]).map(t=>t.role+": "+t.content).join("\n")+"\nuser: "+x.current_user_text}}
function render(){{document.getElementById("root").innerHTML=DATA.map((x,i)=>{{ensure(x.review_item_id);const s=state[x.review_item_id],yn=["yes","no","uncertain"];return `<section class="item"><h2>${{i+1}} / ${{DATA.length}} · ${{esc(x.component)}} · ${{esc(x.resource_subtype)}}</h2><h3>可见当前语境</h3><div class="context">${{esc(context(x))}}</div><h3>已选资源</h3><div class="resource">${{esc(x.selected_resource)}}</div><h3>generator 声明</h3><div class="trace">decision: ${{esc(x.declared_decision)}}\nfunction: ${{esc(x.declared_function)}}\nreason（仅审计，不作 gold）: ${{esc(x.declared_reason)}}</div><h3>用户实际回复</h3><div class="response">${{esc(x.response)}}</div><label>0. 仅凭以上材料能否可靠裁决？<select onchange="setv('${{x.review_item_id}}','adjudicable',this.value)"><option value="">请选择</option>${{opts(s.adjudicable,yn)}}</select></label><label>1. 声明 use/ignore 是否与回复正文一致？<select onchange="setv('${{x.review_item_id}}','declaration_matches_response',this.value)"><option value="">请选择</option>${{opts(s.declaration_matches_response,yn)}}</select></label><label>2. 在当前语境下，use/ignore 决定本身是否合理？<select onchange="setv('${{x.review_item_id}}','resource_decision_appropriate',this.value)"><option value="">请选择</option>${{opts(s.resource_decision_appropriate,yn)}}</select></label><label>3. 若 use，资源是否真正、非表面地发挥声明功能？<select onchange="setv('${{x.review_item_id}}','functional_execution',this.value)"><option value="">请选择</option>${{opts(s.functional_execution,["yes","no","not_applicable","uncertain"])}}</select></label><label>4. 是否存在足以阻止采用的 material misuse？<select onchange="setv('${{x.review_item_id}}','material_misuse',this.value)"><option value="">请选择</option>${{opts(s.material_misuse,yn)}}</select></label><input placeholder="若 misuse：unsupported_personal_claim / stale_or_conflicting_use / overgeneralized_pattern_or_cause / boundary_violation / excessive_directiveness" value="${{esc(s.misuse_categories)}}" onchange="setv('${{x.review_item_id}}','misuse_categories',this.value)"><textarea placeholder="粘贴支持判断的回复原文；无则填 [none]" onchange="setv('${{x.review_item_id}}','literal_response_excerpt',this.value)">${{esc(s.literal_response_excerpt)}}</textarea><textarea placeholder="一句话说明决定是否合理、资源如何做功或如何误用" onchange="setv('${{x.review_item_id}}','review_notes',this.value)">${{esc(s.review_notes)}}</textarea></section>`}}).join("");document.getElementById("ann").value=ann;progress()}}document.getElementById("ann").onchange=e=>{{ann=e.target.value.trim();localStorage.setItem(KEY+":ann",ann);progress()}};function complete(s){{return s.adjudicable&&s.declaration_matches_response&&s.resource_decision_appropriate&&s.functional_execution&&s.material_misuse&&s.literal_response_excerpt&&s.review_notes}}function rows(){{return DATA.map(x=>{{ensure(x.review_item_id);return {{protocol:"{PROTOCOL}",review_item_id:x.review_item_id,...state[x.review_item_id],annotator_id:ann}}}})}}function progress(){{const n=DATA.filter(x=>{{ensure(x.review_item_id);return complete(state[x.review_item_id])}}).length,e=document.getElementById("prog");e.className=n===DATA.length&&ann?"ok":"bad";e.textContent=`完整 ${{n}} / ${{DATA.length}}；标注者ID ${{ann?"已填":"未填"}}`}}function downloadRows(){{const r=rows(),bad=r.filter(x=>!complete(x));if((bad.length||!ann)&&!confirm("尚未全部完成，仍导出吗？"))return;const b=new Blob([r.map(JSON.stringify).join("\n")+"\n"],{{type:"application/jsonl"}}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="repaired_resource_execution_human_check.jsonl";a.click();URL.revokeObjectURL(a.href)}}render();</script></body></html>'''


def main() -> None:
    source_dir = ROOT / "outputs/pm_v1_5_repaired_resource_execution_v1"
    plan_dir = ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1"
    out_dir = ROOT / "outputs/pm_v1_5_repaired_resource_execution_human_check_v1_candidate"
    plan = {str(row["call_id"]): row for row in _rows(source_dir / "execution_call_plan.jsonl")}
    resources = {str(row["state_id"]): row for row in _rows(plan_dir / "private_selected_resources.jsonl")}
    outcomes = _rows(source_dir / "execution_outcomes.jsonl")
    public, private = [], []
    for row in sorted(outcomes, key=lambda x: stable_hex(PROTOCOL, "display", str(x["state_id"]), n=32)):
        call = plan[str(row["call_id"])]
        state = resources[str(row["state_id"])]["runtime_state"]
        review_id = "resource_repair_review_" + stable_hex(PROTOCOL, str(row["call_id"]), n=24)
        public.append({"protocol": PROTOCOL, "review_item_id": review_id, "component": row["component"], "resource_subtype": row["resource_subtype"], "history": state["current_session_history"], "current_user_text": state["current_user_text"], "selected_resource": call["selected_resource"], "declared_decision": row["execution"]["resource_decision"], "declared_function": row["execution"]["resource_function"], "declared_reason": row["execution"]["concise_decision_reason"], "response": row["execution"]["response"]})
        private.append({"protocol": PROTOCOL, "review_item_id": review_id, "state_id": row["state_id"], "call_id": row["call_id"], "machine_checks": row["machine_checks"], "selection_rule": call["selection_rule"], "selection_uses_response_judge_risk_or_external_outcome": False})
    if len(public) != 10:
        raise RuntimeError(f"expected 10 repaired outcomes, got {len(public)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "human_review_packet.jsonl", public)
    write_jsonl(out_dir / "private_review_key.jsonl", private)
    (out_dir / "human_review.html").write_text(_render(public), encoding="utf-8")
    report = {"protocol": PROTOCOL, "status": "READY_FOR_SEPARATED_CONSTRUCT_HUMAN_CHECK", "items": 10, "declared_decisions": {"use": 10, "ignore": 0}, "scope_limit": "positive-opportunity execution check; ignore-path qualification is not established", "selection_uses_response_judge_risk_or_external_outcome": False, "inputs": {"outcomes_sha256": sha256_file(source_dir / "execution_outcomes.jsonl")}, "packet_sha256": sha256_text(canonical_json(public))}
    write_json(out_dir / "preparation_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
