#!/usr/bin/env python3
"""Prepare the single complete V5 FIT quality/risk human-review panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PANEL_PROTOCOL = "pm-v1.5-v5-single-full-fit-outcome-panel-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5-fit-outcome-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5-fit-outcome-grounding-risk-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
OVERLAP_STATES_PER_COMPONENT = 13
BLUEPRINT = (
    ROOT
    / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
)


QUALITY_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PM V1.5 V5 FIT 质量盲评</title>
<style>body{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:18px;background:#f4f6f8;color:#17212b}header,.item{background:white;border:1px solid #d8dee6;border-radius:12px;padding:18px;margin:12px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.context,.response{white-space:pre-wrap;border-radius:8px;padding:14px;background:#f7f9fb;border:1px solid #d8dee6}.response{background:#eef5ff}label{display:block;margin:14px 0}select,textarea,input{width:100%;box-sizing:border-box;padding:9px;margin-top:5px}textarea{min-height:78px}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.controls button{padding:8px 14px}.ok{color:#16733c}.bad{color:#a33}.muted{color:#586575;font-size:.93rem}@media(max-width:760px){.grid{grid-template-columns:1fr}}</style></head><body>
<header><h1>PM V1.5 V5 FIT 质量盲评</h1><p><b>只判断可见对话与匿名回复。</b>不要推测资源、组件、ON/OFF或生成条件。只有足以改变实际采用决定的差异才选A/B；轻微文风、长度或措辞偏好选tie。</p><p class="muted">本页属于一次完整冻结panel，不是开发小包。不要查看页面源码、private key或执行目录。</p><label>标注者ID<input id="ann"></label><div class="controls"><button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button><button onclick="jumpIncomplete()">下一未完成</button><button onclick="downloadRows()">导出JSONL</button><span id="progress"></span></div></header><main id="root"></main>
<script>const DATA=__DATA__;const PROTOCOL="__PROTOCOL__",KEY=PROTOCOL+":"+DATA.manifest.panel_role;let state=JSON.parse(localStorage.getItem(KEY)||"{}"),ann=localStorage.getItem(KEY+":ann")||"",idx=Number(localStorage.getItem(KEY+":idx")||0);const CRITERIA={grounded_context_fidelity:"可见语境忠实度",emotional_understanding:"情绪理解与回应",request_and_dialogue_fit:"请求与对话适配",immediate_helpfulness:"当下实际帮助",clarity_naturalness_not_overloaded:"清晰自然、不过载",materially_equivalent:"实质等价",uncertain:"无法可靠裁决"};
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}function ensure(id){if(!state[id])state[id]={quality_preference:"",decisive_criterion:"",quality_notes:""};}function save(){localStorage.setItem(KEY,JSON.stringify(state));localStorage.setItem(KEY+":ann",ann);localStorage.setItem(KEY+":idx",String(idx));progress();}function setv(id,k,v){ensure(id);state[id][k]=v;save();render();}function complete(x){ensure(x.blind_item_id);const s=state[x.blind_item_id];if(!s.quality_preference||!s.decisive_criterion)return false;return !["A","B","uncertain"].includes(s.quality_preference)||Boolean(s.quality_notes.trim());}function opts(cur,vals,labels){return vals.map(v=>`<option value="${v}" ${cur===v?"selected":""}>${esc(labels?.[v]||v)}</option>`).join("");}
function render(){if(!DATA.items.length)return;idx=Math.max(0,Math.min(idx,DATA.items.length-1));const x=DATA.items[idx];ensure(x.blind_item_id);const s=state[x.blind_item_id];document.getElementById("root").innerHTML=`<section class="item"><h2>${idx+1} / ${DATA.items.length}</h2><h3>可见对话</h3><div class="context">${esc(x.visible_conversation)}</div><div class="grid"><div><h3>回复 A</h3><div class="response">${esc(x.response_a)}</div></div><div><h3>回复 B</h3><div class="response">${esc(x.response_b)}</div></div></div><label>哪个回复实质更好？<select onchange="setv('${x.blind_item_id}','quality_preference',this.value)"><option value="">请选择</option>${opts(s.quality_preference,["A","B","tie","uncertain"],{A:"A实质更好",B:"B实质更好",tie:"实质等价",uncertain:"无法可靠裁决"})}</select></label><label>决定性标准<select onchange="setv('${x.blind_item_id}','decisive_criterion',this.value)"><option value="">请选择</option>${opts(s.decisive_criterion,Object.keys(CRITERIA),CRITERIA)}</select></label><label>判断理由（A/B/uncertain必填；tie可选）<textarea onchange="setv('${x.blind_item_id}','quality_notes',this.value)">${esc(s.quality_notes)}</textarea></label></section>`;save();}
function move(d){idx=Math.max(0,Math.min(DATA.items.length-1,idx+d));render();}function jumpIncomplete(){const start=idx;for(let k=1;k<=DATA.items.length;k++){const j=(start+k)%DATA.items.length;if(!complete(DATA.items[j])){idx=j;render();return;}}}function rows(){return DATA.items.map(x=>{ensure(x.blind_item_id);return {protocol:PROTOCOL,blind_item_id:x.blind_item_id,...state[x.blind_item_id],annotator_id:ann.trim()};});}function progress(){const n=DATA.items.filter(complete).length,e=document.getElementById("progress");e.className=n===DATA.items.length&&ann.trim()?"ok":"bad";e.textContent=`完整 ${n}/${DATA.items.length}；标注者ID ${ann.trim()?"已填":"未填"}`;}function downloadRows(){const r=rows();if((r.some((_,i)=>!complete(DATA.items[i]))||!ann.trim())&&!confirm("尚未全部完成，仍导出吗？"))return;const b=new Blob([r.map(JSON.stringify).join("\n")+"\n"],{type:"application/jsonl"}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=`${PROTOCOL}-${DATA.manifest.panel_role}.jsonl`;a.click();URL.revokeObjectURL(a.href);}document.getElementById("ann").value=ann;document.getElementById("ann").onchange=e=>{ann=e.target.value;save();};render();</script></body></html>'''


RISK_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PM V1.5 V5 FIT Grounding Risk</title>
<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:auto;padding:18px;background:#f4f6f8;color:#17212b}header,.item{background:white;border:1px solid #d8dee6;border-radius:12px;padding:18px;margin:12px 0}.context,.evidence,.response{white-space:pre-wrap;border-radius:8px;padding:14px;border:1px solid #d8dee6}.context{background:#f7f9fb}.evidence{background:#fff8e6}.response{background:#eef5ff}label{display:block;margin:14px 0}select,textarea,input{width:100%;box-sizing:border-box;padding:9px;margin-top:5px}textarea{min-height:72px}.category{border-left:4px solid #d8a63d;padding:8px 12px;margin:10px 0;background:#fffaf0}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.controls button{padding:8px 14px}.ok{color:#16733c}.bad{color:#a33}.muted{color:#586575;font-size:.93rem}</style></head><body>
<header><h1>PM V1.5 V5 FIT Grounding Risk 与功能审核</h1><p>这里不重新评价A/B质量，也不判断PM开关。只判断：候选回复是否有足以改变采用决定的interaction-and-grounding material risk；授权证据是否在回复中真正发挥可区分功能。</p><p class="muted">时间词、历史提及、通用回复或个人文风偏好本身都不足以判risk。判yes必须提供回复原文、可见语境/授权证据依据和materiality理由。</p><label>标注者ID<input id="ann"></label><div class="controls"><button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button><button onclick="jumpIncomplete()">下一未完成</button><button onclick="downloadRows()">导出JSONL</button><span id="progress"></span></div></header><main id="root"></main>
<script>const DATA=__DATA__;const PROTOCOL="__PROTOCOL__",KEY=PROTOCOL+":"+DATA.manifest.panel_role;let state=JSON.parse(localStorage.getItem(KEY)||"{}"),ann=localStorage.getItem(KEY+":ann")||"",idx=Number(localStorage.getItem(KEY+":idx")||0);const LABELS={stale_or_conflicting_use:"陈旧或冲突证据使用",unsupported_personal_claim:"无依据个人断言",overgeneralized_pattern_or_cause:"把一次记录泛化为当前规律/原因",fabricated_recall:"虚构用户过去说过或做过的内容",explicit_boundary_violation:"违反当前明确边界",excessive_directiveness:"过度指令或任务堆叠",internal_resource_label_exposure:"暴露内部资源标签/标识"};
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}function blank(){return {any_material_risk:"",selected_categories:[],evidence_by_category:{},resource_functionally_contributed:"",functional_evidence_excerpt:"",review_notes:""};}function ensure(id){if(!state[id])state[id]=blank();}function save(){localStorage.setItem(KEY,JSON.stringify(state));localStorage.setItem(KEY+":ann",ann);localStorage.setItem(KEY+":idx",String(idx));progress();}function setv(id,k,v){ensure(id);state[id][k]=v;if(k==="any_material_risk"&&v!=="yes"){state[id].selected_categories=[];state[id].evidence_by_category={};}save();render();}function toggle(id,cat,on){ensure(id);const s=state[id];s.selected_categories=on?[...new Set([...s.selected_categories,cat])]:s.selected_categories.filter(x=>x!==cat);if(!s.evidence_by_category[cat])s.evidence_by_category[cat]={literal_response_excerpt:"",literal_context_or_evidence_excerpt:"",materiality_reason:""};save();render();}function setev(id,cat,k,v){ensure(id);if(!state[id].evidence_by_category[cat])state[id].evidence_by_category[cat]={literal_response_excerpt:"",literal_context_or_evidence_excerpt:"",materiality_reason:""};state[id].evidence_by_category[cat][k]=v;save();}function opts(cur,vals,labels){return vals.map(v=>`<option value="${v}" ${cur===v?"selected":""}>${labels?.[v]||v}</option>`).join("");}
function categories(id,s){if(s.any_material_risk!=="yes")return "";return Object.entries(LABELS).map(([cat,label])=>{const checked=s.selected_categories.includes(cat),e=s.evidence_by_category[cat]||{};return `<div class="category"><label><input type="checkbox" style="width:auto" ${checked?"checked":""} onchange="toggle('${id}','${cat}',this.checked)">${esc(label)}</label>${checked?`<textarea placeholder="候选回复中的直接原文" onchange="setev('${id}','${cat}','literal_response_excerpt',this.value)">${esc(e.literal_response_excerpt)}</textarea><textarea placeholder="当前对话或授权证据中的对应原文；若无依据请写[none]" onchange="setev('${id}','${cat}','literal_context_or_evidence_excerpt',this.value)">${esc(e.literal_context_or_evidence_excerpt)}</textarea><textarea placeholder="具体错配及为何足以改变采用决定" onchange="setev('${id}','${cat}','materiality_reason',this.value)">${esc(e.materiality_reason)}</textarea>`:""}</div>`;}).join("");}
function complete(x){ensure(x.risk_item_id);const s=state[x.risk_item_id];if(!s.any_material_risk||!s.resource_functionally_contributed)return false;if(s.any_material_risk==="yes"){if(!s.selected_categories.length)return false;for(const c of s.selected_categories){const e=s.evidence_by_category[c]||{};if(!e.literal_response_excerpt?.trim()||!e.literal_context_or_evidence_excerpt?.trim()||!e.materiality_reason?.trim())return false;}}if(s.any_material_risk==="uncertain"&&!s.review_notes.trim())return false;if(s.resource_functionally_contributed==="yes"&&!s.functional_evidence_excerpt.trim())return false;if(["no","uncertain"].includes(s.resource_functionally_contributed)&&!s.review_notes.trim())return false;return true;}
function render(){idx=Math.max(0,Math.min(idx,DATA.items.length-1));const x=DATA.items[idx];ensure(x.risk_item_id);const s=state[x.risk_item_id],yn={no:"否",yes:"是",uncertain:"不确定"};document.getElementById("root").innerHTML=`<section class="item"><h2>${idx+1} / ${DATA.items.length}</h2><h3>可见对话</h3><div class="context">${esc(x.visible_conversation)}</div><h3>生成时授权的辅助证据与使用边界</h3><div class="evidence">${esc(x.authorized_evidence_and_instruction)}</div><h3>审核对象回复</h3><div class="response">${esc(x.candidate_response)}</div><label>是否存在任一明确material risk？<select onchange="setv('${x.risk_item_id}','any_material_risk',this.value)"><option value="">请选择</option>${opts(s.any_material_risk,["no","yes","uncertain"],yn)}</select></label>${categories(x.risk_item_id,s)}<label>授权资源是否在回复中真正发挥了可区分功能？<select onchange="setv('${x.risk_item_id}','resource_functionally_contributed',this.value)"><option value="">请选择</option>${opts(s.resource_functionally_contributed,["yes","no","uncertain"],yn)}</select></label><textarea placeholder="若判做功，粘贴体现功能的回复原文" onchange="setv('${x.risk_item_id}','functional_evidence_excerpt',this.value)">${esc(s.functional_evidence_excerpt)}</textarea><textarea placeholder="no/uncertain功能或risk uncertain时必填；其他情况可简要记录" onchange="setv('${x.risk_item_id}','review_notes',this.value)">${esc(s.review_notes)}</textarea></section>`;save();}
function move(d){idx=Math.max(0,Math.min(DATA.items.length-1,idx+d));render();}function jumpIncomplete(){const start=idx;for(let k=1;k<=DATA.items.length;k++){const j=(start+k)%DATA.items.length;if(!complete(DATA.items[j])){idx=j;render();return;}}}function rows(){return DATA.items.map(x=>{ensure(x.risk_item_id);return {protocol:PROTOCOL,risk_item_id:x.risk_item_id,...state[x.risk_item_id],annotator_id:ann.trim()};});}function progress(){const n=DATA.items.filter(complete).length,e=document.getElementById("progress");e.className=n===DATA.items.length&&ann.trim()?"ok":"bad";e.textContent=`完整 ${n}/${DATA.items.length}；标注者ID ${ann.trim()?"已填":"未填"}`;}function downloadRows(){const r=rows();if((r.some((_,i)=>!complete(DATA.items[i]))||!ann.trim())&&!confirm("尚未全部完成，仍导出吗？"))return;const b=new Blob([r.map(JSON.stringify).join("\n")+"\n"],{type:"application/jsonl"}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=`${PROTOCOL}-${DATA.manifest.panel_role}.jsonl`;a.click();URL.revokeObjectURL(a.href);}document.getElementById("ann").value=ann;document.getElementById("ann").onchange=e=>{ann=e.target.value;save();};render();</script></body></html>'''


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _safe_script_json(value: Any) -> str:
    return (
        canonical_json(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _visible_conversation(call: dict[str, Any]) -> str:
    user_messages = [
        str(message["content"])
        for message in call["messages"]
        if message["role"] == "user"
    ]
    if len(user_messages) != 1:
        raise RuntimeError(f"unexpected user-message count: {call['call_id']}")
    text = user_messages[0]
    prefix = "Visible current conversation:\n"
    if not text.startswith(prefix):
        raise RuntimeError(f"missing visible-conversation prefix: {call['call_id']}")
    return text[len(prefix) :].strip()


def _authorized_evidence(call: dict[str, Any]) -> str:
    system_messages = [
        str(message["content"])
        for message in call["messages"]
        if message["role"] == "system"
    ]
    if len(system_messages) != 1:
        raise RuntimeError(f"unexpected system-message count: {call['call_id']}")
    marker = "\n\nResource 1\n"
    if marker not in system_messages[0]:
        raise RuntimeError(f"ON call lacks one resource surface: {call['call_id']}")
    public = system_messages[0].split(marker, 1)[1].strip()
    if "mem_" in public or "strat_" in public or "card_" in public:
        raise RuntimeError(f"opaque resource ID leaked into public evidence: {call['call_id']}")
    return public


def _quality_item(
    *, pair_id: str, visible: str, response_a: str, response_b: str
) -> dict[str, Any]:
    return {
        "protocol": QUALITY_PROTOCOL,
        "blind_item_id": pair_id,
        "visible_conversation": visible,
        "response_a": response_a,
        "response_b": response_b,
    }


def _risk_item(
    *, risk_id: str, visible: str, evidence: str, response: str
) -> dict[str, Any]:
    return {
        "protocol": RISK_PROTOCOL,
        "risk_item_id": risk_id,
        "visible_conversation": visible,
        "authorized_evidence_and_instruction": evidence,
        "candidate_response": response,
    }


def build_panel(
    *, plan_dir: Path, execution_dir: Path, out_dir: Path
) -> dict[str, Any]:
    summary = read_json(execution_dir / "execution_summary.json")
    if summary.get("status") != "COMPLETE_AWAITING_SINGLE_FIT_OUTCOME_REVIEW":
        raise RuntimeError("V2 FIT execution is not complete")
    if summary.get("completed_calls") != 1024 or summary.get("invalid_itt_rows") != 0:
        raise RuntimeError("V2 FIT execution shape or validity failed")

    plan_rows = _rows(plan_dir / "call_plan_private.jsonl")
    outcome_rows = _rows(execution_dir / "outcomes_ordered_private.jsonl")
    if len(plan_rows) != 1024 or len(outcome_rows) != 1024:
        raise RuntimeError("expected 1024 plan and outcome rows")
    plans = {str(row["call_id"]): row for row in plan_rows}
    outcomes = {str(row["call_id"]): row for row in outcome_rows}
    blueprint = {
        str(row["blueprint_row_id"]): row for row in _rows(BLUEPRINT)
    }
    if len(plans) != 1024 or len(outcomes) != 1024 or set(plans) != set(outcomes):
        raise RuntimeError("plan/outcome call identity mismatch")

    paired: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for call_id, plan in plans.items():
        outcome = outcomes[call_id]
        if not outcome.get("itt_row_valid"):
            raise RuntimeError(f"invalid ITT row reached panel: {call_id}")
        for field in (
            "state_id",
            "seed",
            "arm",
            "target_component",
            "requested_action_id",
            "messages_sha256",
        ):
            if plan.get(field) != outcome.get(field):
                raise RuntimeError(f"plan/outcome {field} mismatch: {call_id}")
        paired[(str(plan["state_id"]), int(plan["seed"]))].append(
            {"plan": plan, "outcome": outcome}
        )
    if len(paired) != 512:
        raise RuntimeError(f"expected 512 state-seed pairs, got {len(paired)}")

    component_states: dict[str, set[str]] = {component: set() for component in COMPONENTS}
    normalized_pairs: list[dict[str, Any]] = []
    for (state_id, seed), rows in paired.items():
        if len(rows) != 2 or Counter(row["plan"]["arm"] for row in rows) != Counter({"ON": 1, "OFF": 1}):
            raise RuntimeError(f"bad ON/OFF pair: {state_id}/{seed}")
        by_arm = {str(row["plan"]["arm"]): row for row in rows}
        on, off = by_arm["ON"], by_arm["OFF"]
        component = str(on["plan"]["target_component"])
        if component not in COMPONENTS or off["plan"]["target_component"] != component:
            raise RuntimeError(f"component mismatch: {state_id}/{seed}")
        visible = _visible_conversation(on["plan"])
        if visible != _visible_conversation(off["plan"]):
            raise RuntimeError(f"visible conversation mismatch: {state_id}/{seed}")
        component_states[component].add(state_id)
        normalized_pairs.append(
            {
                "state_id": state_id,
                "seed": seed,
                "component": component,
                "visible": visible,
                "on": on,
                "off": off,
            }
        )
    if {key: len(value) for key, value in component_states.items()} != {
        component: 64 for component in COMPONENTS
    }:
        raise RuntimeError("expected 64 state-candidate units per component")

    fit_state_ids = set().union(*component_states.values())
    if not fit_state_ids <= set(blueprint):
        raise RuntimeError("FIT state is absent from the frozen blueprint")
    semantic_group_by_state = {
        state_id: str(blueprint[state_id]["counterfactual_group_id"])
        for state_id in fit_state_ids
    }
    semantic_family_by_state = {
        state_id: str(blueprint[state_id]["logic_family"])
        for state_id in fit_state_ids
    }
    semantic_groups_per_component: dict[str, dict[str, list[str]]] = {}
    for component in COMPONENTS:
        groups: dict[str, list[str]] = defaultdict(list)
        for state_id in component_states[component]:
            groups[semantic_group_by_state[state_id]].append(state_id)
        if len(groups) != 32 or any(len(state_ids) != 2 for state_ids in groups.values()):
            raise RuntimeError(
                f"{component} requires 32 two-state counterfactual groups"
            )
        semantic_groups_per_component[component] = groups

    overlap_states: set[str] = set()
    for component in COMPONENTS:
        # Sample one state from each of 13 distinct counterfactual groups.  This
        # preserves the 20% row overlap while preventing a duplicated semantic
        # pair from masquerading as two independent overlap observations.
        groups = semantic_groups_per_component[component]
        selected_groups = sorted(
            groups,
            key=lambda value: stable_hex(
                PANEL_PROTOCOL, "overlap-group", component, value, n=32
            ),
        )[:OVERLAP_STATES_PER_COMPONENT]
        for group_id in selected_groups:
            state_id = min(
                groups[group_id],
                key=lambda value: stable_hex(
                    PANEL_PROTOCOL,
                    "overlap-state-within-group",
                    component,
                    group_id,
                    value,
                    n=32,
                ),
            )
            overlap_states.add(state_id)
    if len(overlap_states) != 52:
        raise RuntimeError("overlap state selection is not 52 unique states")

    on_as_a_keys: set[tuple[str, int]] = set()
    for component in COMPONENTS:
        rows = sorted(
            (row for row in normalized_pairs if row["component"] == component),
            key=lambda row: stable_hex(
                PANEL_PROTOCOL,
                "position-balance",
                component,
                row["state_id"],
                str(row["seed"]),
                n=32,
            ),
        )
        if len(rows) != 128:
            raise RuntimeError(f"expected 128 pairs for {component}")
        on_as_a_keys.update((row["state_id"], row["seed"]) for row in rows[:64])

    quality_items: list[dict[str, Any]] = []
    risk_items: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    for row in normalized_pairs:
        state_id, seed, component = row["state_id"], row["seed"], row["component"]
        pair_id = "v5q_" + stable_hex(PANEL_PROTOCOL, "quality", state_id, str(seed), n=24)
        risk_id = "v5r_" + stable_hex(PANEL_PROTOCOL, "risk", state_id, str(seed), n=24)
        on_as_a = (state_id, seed) in on_as_a_keys
        arm_a, arm_b = ("ON", "OFF") if on_as_a else ("OFF", "ON")
        response_by_arm = {
            "ON": str(row["on"]["outcome"]["final_response"]),
            "OFF": str(row["off"]["outcome"]["final_response"]),
        }
        quality_items.append(
            _quality_item(
                pair_id=pair_id,
                visible=row["visible"],
                response_a=response_by_arm[arm_a],
                response_b=response_by_arm[arm_b],
            )
        )
        risk_items.append(
            _risk_item(
                risk_id=risk_id,
                visible=row["visible"],
                evidence=_authorized_evidence(row["on"]["plan"]),
                response=response_by_arm["ON"],
            )
        )
        private_key.append(
            {
                "protocol": PANEL_PROTOCOL,
                "blind_item_id": pair_id,
                "risk_item_id": risk_id,
                "state_id": state_id,
                "seed": seed,
                "component": component,
                "group_id": row["on"]["outcome"]["group_id_private_analysis_only"],
                "semantic_group_id": semantic_group_by_state[state_id],
                "semantic_family": semantic_family_by_state[state_id],
                "on_call_id": row["on"]["plan"]["call_id"],
                "off_call_id": row["off"]["plan"]["call_id"],
                "a_arm": arm_a,
                "b_arm": arm_b,
                "on_fallback_used": bool(row["on"]["outcome"]["fallback_used"]),
                "on_guard_errors": list(row["on"]["outcome"]["guard_errors"]),
                "in_independent_overlap": state_id in overlap_states,
                "effect_label": "UNKNOWN_BEFORE_REVIEW_AND_AGGREGATION",
            }
        )

    quality_items.sort(
        key=lambda row: stable_hex(
            PANEL_PROTOCOL, "quality-order", row["blind_item_id"], n=32
        )
    )
    risk_items.sort(
        key=lambda row: stable_hex(
            PANEL_PROTOCOL, "risk-order", row["risk_item_id"], n=32
        )
    )
    key_by_quality = {row["blind_item_id"]: row for row in private_key}
    key_by_risk = {row["risk_item_id"]: row for row in private_key}
    overlap_quality = [
        row for row in quality_items if key_by_quality[row["blind_item_id"]]["in_independent_overlap"]
    ]
    overlap_risk = [
        row for row in risk_items if key_by_risk[row["risk_item_id"]]["in_independent_overlap"]
    ]
    if len(quality_items) != 512 or len(risk_items) != 512:
        raise RuntimeError("primary panel count mismatch")
    if len(overlap_quality) != 104 or len(overlap_risk) != 104:
        raise RuntimeError("independent overlap count mismatch")

    position_counts = Counter(
        (row["component"], row["a_arm"]) for row in private_key
    )
    if any(position_counts[(component, arm)] != 64 for component in COMPONENTS for arm in ("ON", "OFF")):
        raise RuntimeError(f"A/B position imbalance: {position_counts}")

    manifest = {
        "protocol": PANEL_PROTOCOL,
        "status": "READY_FOR_ONE_COMPLETE_FIT_OUTCOME_REVIEW",
        "execution_status": summary["status"],
        "states": 256,
        "independent_counterfactual_groups": 128,
        "independent_counterfactual_groups_per_component": {
            component: 32 for component in COMPONENTS
        },
        "states_per_component": {component: 64 for component in COMPONENTS},
        "quality_pairs": 512,
        "on_arm_grounding_risk_items": 512,
        "independent_overlap_states": 52,
        "independent_overlap_semantic_groups": 52,
        "independent_overlap_quality_pairs": 104,
        "independent_overlap_risk_items": 104,
        "primary_position_balance_per_component": {"ON_as_A": 64, "OFF_as_A": 64},
        "quality_and_risk_are_separate_pages_to_preserve_arm_blinding": True,
        "no_additional_small_packets": True,
        "post_review_method_repair_allowed": False,
        "effect_labels_created": False,
        "inputs": {
            "call_plan": str((plan_dir / "call_plan_private.jsonl").relative_to(ROOT)),
            "outcomes": str((execution_dir / "outcomes_ordered_private.jsonl").relative_to(ROOT)),
            "review_contract": "data/pm_v1_5_contracts/v5_single_full_fit_outcome_review_v1.json",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {
        "primary_quality_packet.jsonl": out_dir / "primary_quality_packet.jsonl",
        "primary_grounding_risk_packet.jsonl": out_dir / "primary_grounding_risk_packet.jsonl",
        "independent_overlap_quality_packet.jsonl": out_dir / "independent_overlap_quality_packet.jsonl",
        "independent_overlap_grounding_risk_packet.jsonl": out_dir / "independent_overlap_grounding_risk_packet.jsonl",
        "private_blind_key.jsonl": out_dir / "private_blind_key.jsonl",
        "primary_quality_blank_annotations.jsonl": out_dir / "primary_quality_blank_annotations.jsonl",
        "primary_grounding_risk_blank_annotations.jsonl": out_dir / "primary_grounding_risk_blank_annotations.jsonl",
        "independent_overlap_quality_blank_annotations.jsonl": out_dir / "independent_overlap_quality_blank_annotations.jsonl",
        "independent_overlap_grounding_risk_blank_annotations.jsonl": out_dir / "independent_overlap_grounding_risk_blank_annotations.jsonl",
        "primary_quality_review.html": out_dir / "primary_quality_review.html",
        "primary_grounding_risk_review.html": out_dir / "primary_grounding_risk_review.html",
        "independent_overlap_quality_review.html": out_dir / "independent_overlap_quality_review.html",
        "independent_overlap_grounding_risk_review.html": out_dir / "independent_overlap_grounding_risk_review.html",
    }
    write_jsonl(files["primary_quality_packet.jsonl"], quality_items)
    write_jsonl(files["primary_grounding_risk_packet.jsonl"], risk_items)
    write_jsonl(files["independent_overlap_quality_packet.jsonl"], overlap_quality)
    write_jsonl(files["independent_overlap_grounding_risk_packet.jsonl"], overlap_risk)
    write_jsonl(files["private_blind_key.jsonl"], private_key)

    def quality_blank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "protocol": QUALITY_PROTOCOL,
                "blind_item_id": row["blind_item_id"],
                "quality_preference": None,
                "decisive_criterion": None,
                "quality_notes": "",
                "annotator_id": "",
            }
            for row in items
        ]

    def risk_blank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "protocol": RISK_PROTOCOL,
                "risk_item_id": row["risk_item_id"],
                "any_material_risk": None,
                "selected_categories": [],
                "evidence_by_category": {},
                "resource_functionally_contributed": None,
                "functional_evidence_excerpt": "",
                "review_notes": "",
                "annotator_id": "",
            }
            for row in items
        ]

    write_jsonl(files["primary_quality_blank_annotations.jsonl"], quality_blank(quality_items))
    write_jsonl(files["primary_grounding_risk_blank_annotations.jsonl"], risk_blank(risk_items))
    write_jsonl(files["independent_overlap_quality_blank_annotations.jsonl"], quality_blank(overlap_quality))
    write_jsonl(files["independent_overlap_grounding_risk_blank_annotations.jsonl"], risk_blank(overlap_risk))

    for role, items, path in (
        ("primary", quality_items, files["primary_quality_review.html"]),
        ("independent_overlap", overlap_quality, files["independent_overlap_quality_review.html"]),
    ):
        data = {"manifest": {**manifest, "panel_role": role}, "items": items}
        path.write_text(
            QUALITY_HTML.replace("__DATA__", _safe_script_json(data)).replace("__PROTOCOL__", QUALITY_PROTOCOL),
            encoding="utf-8",
        )
    for role, items, path in (
        ("primary", risk_items, files["primary_grounding_risk_review.html"]),
        ("independent_overlap", overlap_risk, files["independent_overlap_grounding_risk_review.html"]),
    ):
        data = {"manifest": {**manifest, "panel_role": role}, "items": items}
        path.write_text(
            RISK_HTML.replace("__DATA__", _safe_script_json(data)).replace("__PROTOCOL__", RISK_PROTOCOL),
            encoding="utf-8",
        )

    manifest["outputs"] = {
        name: sha256_file(path) for name, path in sorted(files.items())
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_itt_fit_plan_v2",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_itt_fit_execution_v2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_single_full_fit_outcome_review_v1",
    )
    args = parser.parse_args()
    manifest = build_panel(
        plan_dir=args.plan_dir,
        execution_dir=args.execution_dir,
        out_dir=args.out_dir,
    )
    print(
        {
            "protocol": PANEL_PROTOCOL,
            "status": manifest["status"],
            "quality_pairs": manifest["quality_pairs"],
            "risk_items": manifest["on_arm_grounding_risk_items"],
            "primary_quality_review": str(args.out_dir / "primary_quality_review.html"),
            "primary_risk_review": str(args.out_dir / "primary_grounding_risk_review.html"),
        }
    )


if __name__ == "__main__":
    main()
