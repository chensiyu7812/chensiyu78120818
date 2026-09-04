#!/usr/bin/env python3
"""Prepare the minimum 32-item V1.5b functional resource-use review."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text, stable_hex, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-resource-application-human-functional-check-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _render(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    title = "PM V1.5b · generator 资源真正做功核验（32条）"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1080px;margin:auto;padding:22px;background:#f4f6f8;color:#17202a}}.note,.item{{background:#fff;border:1px solid #d6dce4;border-radius:10px;padding:17px;margin:15px 0}}.note{{border-left:5px solid #0969da}}.context,.resource,.plan,.response,.fallback{{white-space:pre-wrap;padding:12px;border-radius:7px;line-height:1.5}}.context{{background:#edf2f6}}.resource{{background:#fff8c5;border:1px solid #d4a72c}}.plan{{background:#eefbf3;border:1px solid #85c79f}}.response{{background:#f4f0ff;border:1px solid #a99bd4}}.fallback{{background:#fff1f0;border:1px solid #e5a7a2}}select,textarea,input{{width:100%;box-sizing:border-box;padding:8px;margin:5px 0 10px}}textarea{{min-height:70px}}button{{padding:9px 14px}}.sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}.ok{{color:#1a7f37}}.bad{{color:#cf222e}}</style></head><body><h1>{html.escape(title)}</h1>
<div class="note"><b>只核验 Step2，不评价 PM 开关是否正确，也不做 A/B 回复质量比较。</b><br>“做功=yes”要求资源改变回复的内容、结构、连续性或支持动作；仅复述历史、贴一句个性化开头后给无关通用建议、或只满足当前消息本来就要求的行为，均为 no。MS/ME 必须保留过去来源并把当前适用性说成暂定；过去事实升级成当前事实/原因属于 misuse。页面不显示机器校验结果。若出现回退回复，只判断它是否可采用，不能用回退替原资源记功。</div>
<div class="sticky"><input id="ann" placeholder="annotator_id"><button onclick="downloadRows()">导出 JSONL</button> <span id="prog"></span></div><div id="root"></div><script>
const DATA={payload},KEY="pm_v15b_resource_application_review_v1";let state=JSON.parse(localStorage.getItem(KEY)||"{{}}"),ann=localStorage.getItem(KEY+":ann")||"";
function esc(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}function blank(){{return {{adjudicable:"",functional_execution:"",material_misuse:"",misuse_categories:"",fallback_usable:"",literal_response_excerpt:"",review_notes:""}}}}function ensure(id){{if(!state[id])state[id]=blank()}}function setv(id,k,v){{ensure(id);state[id][k]=v;localStorage.setItem(KEY,JSON.stringify(state));progress();}}function opts(v,vals){{return vals.map(x=>`<option value="${{x}}" ${{v===x?"selected":""}}>${{x}}</option>`).join("")}}function context(x){{return (x.history||[]).map(t=>t.role+": "+t.content).join("\n")+"\nuser: "+x.current_user_text}}
function render(){{document.getElementById("root").innerHTML=DATA.map((x,i)=>{{ensure(x.review_item_id);const s=state[x.review_item_id],yn=["yes","no","uncertain"];return `<section class="item"><h2>${{i+1}} / ${{DATA.length}} · ${{esc(x.component)}} · ${{esc(x.resource_subtype)}}</h2><h3>可见当前语境</h3><div class="context">${{esc(context(x))}}</div><h3>已选资源</h3><div class="resource">${{esc(x.selected_resource)}}</div><h3>必须执行的功能</h3><div class="plan">function: ${{esc(x.expected_function)}}\nrequired contribution: ${{esc(x.required_contribution)}}\nattribution required: ${{esc(x.attribution_required)}}</div><h3>资源版回复（审核对象）</h3><div class="response">${{esc(x.application_response)}}</div>${{x.fallback_response?`<h3>机器拦截后的无资源回退（不能替资源记功）</h3><div class="fallback">${{esc(x.fallback_response)}}</div>`:""}}<label>0. 能否可靠裁决？<select onchange="setv('${{x.review_item_id}}','adjudicable',this.value)"><option value="">请选择</option>${{opts(s.adjudicable,yn)}}</select></label><label>1. 资源是否真正、非表面地执行了上述功能？<select onchange="setv('${{x.review_item_id}}','functional_execution',this.value)"><option value="">请选择</option>${{opts(s.functional_execution,yn)}}</select></label><label>2. 资源版回复是否有足以阻止采用的 material misuse？<select onchange="setv('${{x.review_item_id}}','material_misuse',this.value)"><option value="">请选择</option>${{opts(s.material_misuse,yn)}}</select></label><input placeholder="若 misuse：unsupported_personal_claim / stale_or_conflicting_use / overgeneralized_pattern_or_cause / boundary_violation / excessive_directiveness" value="${{esc(s.misuse_categories)}}" onchange="setv('${{x.review_item_id}}','misuse_categories',this.value)"><label>3. 若显示回退，回退回复本身是否可采用？无回退选 not_applicable。<select onchange="setv('${{x.review_item_id}}','fallback_usable',this.value)"><option value="">请选择</option>${{opts(s.fallback_usable,["yes","no","not_applicable","uncertain"])}}</select></label><textarea placeholder="粘贴支持判断的回复原文；无则填 [none]" onchange="setv('${{x.review_item_id}}','literal_response_excerpt',this.value)">${{esc(s.literal_response_excerpt)}}</textarea><textarea placeholder="一句话说明资源具体如何做功、空转或误用" onchange="setv('${{x.review_item_id}}','review_notes',this.value)">${{esc(s.review_notes)}}</textarea></section>`}}).join("");document.getElementById("ann").value=ann;progress()}}document.getElementById("ann").onchange=e=>{{ann=e.target.value.trim();localStorage.setItem(KEY+":ann",ann);progress()}};function complete(s){{return s.adjudicable&&s.functional_execution&&s.material_misuse&&s.fallback_usable&&s.literal_response_excerpt&&s.review_notes}}function rows(){{return DATA.map(x=>{{ensure(x.review_item_id);return {{protocol:"{PROTOCOL}",review_item_id:x.review_item_id,...state[x.review_item_id],annotator_id:ann}}}})}}function progress(){{const n=DATA.filter(x=>{{ensure(x.review_item_id);return complete(state[x.review_item_id])}}).length,e=document.getElementById("prog");e.className=n===DATA.length&&ann?"ok":"bad";e.textContent=`完整 ${{n}} / ${{DATA.length}}；标注者ID ${{ann?"已填":"未填"}}`}}function downloadRows(){{const r=rows(),bad=r.filter(x=>!complete(x));if((bad.length||!ann)&&!confirm("尚未全部完成，仍导出吗？"))return;const b=new Blob([r.map(JSON.stringify).join("\n")+"\n"],{{type:"application/jsonl"}}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="v1_5b_resource_application_review.jsonl";a.click();URL.revokeObjectURL(a.href)}}render();</script></body></html>'''


def main() -> None:
    execution_dir = ROOT / "outputs/pm_v1_5b_resource_application_v1"
    source_dir = ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1"
    out_dir = ROOT / "outputs/pm_v1_5b_resource_application_human_check_v1_candidate"
    plans = {str(row["call_id"]): row for row in _rows(execution_dir / "execution_call_plan.jsonl")}
    states = {str(row["state_id"]): row["runtime_state"] for row in _rows(source_dir / "private_selected_resources.jsonl")}
    outcomes = _rows(execution_dir / "execution_outcomes_revalidated_v2.jsonl")
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for row in sorted(outcomes, key=lambda value: stable_hex(PROTOCOL, "display", str(value["call_id"]), n=32)):
        plan_row = plans[str(row["call_id"])]
        state = states[str(row["state_id"])]
        plan = plan_row["execution_plan"]
        review_id = "v15b_resource_review_" + stable_hex(PROTOCOL, str(row["call_id"]), n=24)
        public.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "component": row["component"],
                "resource_subtype": row["resource_subtype"],
                "history": state["current_session_history"],
                "current_user_text": state["current_user_text"],
                "selected_resource": plan_row["selected_resource"],
                "expected_function": plan["expected_function"],
                "required_contribution": plan["required_contribution"],
                "attribution_required": plan["attribution_required"],
                "application_response": row["application"]["response"],
                "fallback_response": None if row["fallback"] is None else row["fallback"]["response"],
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "call_id": row["call_id"],
                "state_id": row["state_id"],
                "machine_checks_private": row["machine_checks"],
                "fallback_executed_private": row["fallback_executed"],
                "selection_uses_response_judge_risk_or_external_outcome": False,
            }
        )
    if len(public) != 32 or CounterLike(public) != {"ME": 8, "MP": 8, "MS": 8, "RS": 8}:
        raise RuntimeError("expected exactly 8 review items per component")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "human_review_packet.jsonl", public)
    write_jsonl(out_dir / "private_review_key.jsonl", private)
    (out_dir / "human_review.html").write_text(_render(public), encoding="utf-8")
    report = {
        "protocol": PROTOCOL,
        "status": "READY_MINIMUM_32_FUNCTIONAL_CHECK",
        "items": len(public),
        "items_per_component": CounterLike(public),
        "quality_comparison_requested": False,
        "pm_or_retrieval_correctness_requested": False,
        "machine_results_visible_to_reviewer": False,
        "input_outcomes_sha256": sha256_file(execution_dir / "execution_outcomes_revalidated_v2.jsonl"),
        "packet_sha256": sha256_text(canonical_json(public)),
    }
    write_json(out_dir / "preparation_report.json", report)
    print(report)


def CounterLike(rows: list[dict[str, Any]]) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        component = str(row["component"])
        values[component] = values.get(component, 0) + 1
    return dict(sorted(values.items()))


if __name__ == "__main__":
    main()
