#!/usr/bin/env python3
"""Prepare the single V5.1 confirmation quality and two-arm risk review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, stable_hex, write_json, write_jsonl
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.1-confirmation-review-packet-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5.1-confirmation-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5.1-confirmation-grounding-risk-v1"
FIT_REVIEW_HELPER = ROOT / "scripts/v1_5/25zy_prepare_v5_full_fit_outcome_review_v1_5.py"


RISK_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PM V1.5 V5.1 Confirmation Risk</title>
<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:auto;padding:18px;background:#f4f6f8;color:#17212b}header,.item{background:white;border:1px solid #d8dee6;border-radius:12px;padding:18px;margin:12px 0}.context,.evidence,.response{white-space:pre-wrap;border-radius:8px;padding:14px;border:1px solid #d8dee6}.context{background:#f7f9fb}.evidence{background:#fff8e6}.response{background:#eef5ff}label{display:block;margin:14px 0}select,textarea,input{width:100%;box-sizing:border-box;padding:9px;margin-top:5px}textarea{min-height:72px}.category{border-left:4px solid #d8a63d;padding:8px 12px;margin:10px 0;background:#fffaf0}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.controls button{padding:8px 14px}.ok{color:#16733c}.bad{color:#a33}.muted{color:#586575;font-size:.93rem}</style></head><body>
<header><h1>PM V1.5 V5.1 Confirmation Grounding Risk</h1><p>审核ON与OFF双方回复。这里只判断足以改变采用决定的interaction-and-grounding material risk，不判断哪边质量更好，也不判断资源是否做功。</p><p class="muted">授权证据可见是为了区分真实记忆与虚构记忆；它不表示该回复应获奖励。判yes必须选择类别并粘贴回复原文、授权/语境原文和materiality理由。</p><label>标注者ID<input id="ann"></label><div class="controls"><button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button><button onclick="jumpIncomplete()">下一未完成</button><button onclick="downloadRows()">导出JSONL</button><span id="progress"></span></div></header><main id="root"></main>
<script>const DATA=__DATA__;const PROTOCOL="__PROTOCOL__",KEY=PROTOCOL+":"+DATA.manifest.panel_role;let state=JSON.parse(localStorage.getItem(KEY)||"{}"),ann=localStorage.getItem(KEY+":ann")||"",idx=Number(localStorage.getItem(KEY+":idx")||0);const LABELS={stale_or_conflicting_use:"陈旧或冲突证据使用",unsupported_personal_claim:"无依据个人断言",overgeneralized_pattern_or_cause:"把一次记录泛化为当前规律/原因",fabricated_recall:"虚构用户过去说过或做过的内容",explicit_boundary_violation:"违反当前明确边界",excessive_directiveness:"过度指令或任务堆叠",internal_resource_label_exposure:"暴露内部资源标签/标识"};
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}function blank(){return {any_material_risk:"",selected_categories:[],evidence_by_category:{},risk_notes:""};}function ensure(id){if(!state[id])state[id]=blank();}function save(){localStorage.setItem(KEY,JSON.stringify(state));localStorage.setItem(KEY+":ann",ann);localStorage.setItem(KEY+":idx",String(idx));progress();}function setv(id,k,v){ensure(id);state[id][k]=v;if(k==="any_material_risk"&&v!=="yes"){state[id].selected_categories=[];state[id].evidence_by_category={};}save();render();}function toggle(id,cat,on){ensure(id);const s=state[id];s.selected_categories=on?[...new Set([...s.selected_categories,cat])]:s.selected_categories.filter(x=>x!==cat);if(!s.evidence_by_category[cat])s.evidence_by_category[cat]={literal_response_excerpt:"",literal_context_or_evidence_excerpt:"",materiality_reason:""};save();render();}function setev(id,cat,k,v){ensure(id);if(!state[id].evidence_by_category[cat])state[id].evidence_by_category[cat]={literal_response_excerpt:"",literal_context_or_evidence_excerpt:"",materiality_reason:""};state[id].evidence_by_category[cat][k]=v;save();}function opts(cur,vals,labels){return vals.map(v=>`<option value="${v}" ${cur===v?"selected":""}>${labels?.[v]||v}</option>`).join("");}
function categories(id,s){if(s.any_material_risk!=="yes")return "";return Object.entries(LABELS).map(([cat,label])=>{const checked=s.selected_categories.includes(cat),e=s.evidence_by_category[cat]||{};return `<div class="category"><label><input type="checkbox" style="width:auto" ${checked?"checked":""} onchange="toggle('${id}','${cat}',this.checked)">${esc(label)}</label>${checked?`<textarea placeholder="候选回复中的直接原文" onchange="setev('${id}','${cat}','literal_response_excerpt',this.value)">${esc(e.literal_response_excerpt)}</textarea><textarea placeholder="当前对话或授权证据中的对应原文；若无依据写[none]" onchange="setev('${id}','${cat}','literal_context_or_evidence_excerpt',this.value)">${esc(e.literal_context_or_evidence_excerpt)}</textarea><textarea placeholder="错配及为何足以改变采用决定" onchange="setev('${id}','${cat}','materiality_reason',this.value)">${esc(e.materiality_reason)}</textarea>`:""}</div>`;}).join("");}
function complete(x){ensure(x.risk_item_id);const s=state[x.risk_item_id];if(!s.any_material_risk)return false;if(s.any_material_risk==="yes"){if(!s.selected_categories.length)return false;for(const c of s.selected_categories){const e=s.evidence_by_category[c]||{};if(!e.literal_response_excerpt?.trim()||!e.literal_context_or_evidence_excerpt?.trim()||!e.materiality_reason?.trim())return false;}}if(s.any_material_risk==="uncertain"&&!s.risk_notes.trim())return false;return true;}
function render(){idx=Math.max(0,Math.min(idx,DATA.items.length-1));const x=DATA.items[idx];ensure(x.risk_item_id);const s=state[x.risk_item_id],yn={no:"否",yes:"是",uncertain:"不确定"};document.getElementById("root").innerHTML=`<section class="item"><h2>${idx+1} / ${DATA.items.length}</h2><h3>可见对话</h3><div class="context">${esc(x.visible_conversation)}</div><h3>本回复允许使用的证据与边界</h3><div class="evidence">${esc(x.authorized_evidence_and_instruction)}</div><h3>审核对象回复</h3><div class="response">${esc(x.candidate_response)}</div><label>是否存在任一明确material risk？<select onchange="setv('${x.risk_item_id}','any_material_risk',this.value)"><option value="">请选择</option>${opts(s.any_material_risk,["no","yes","uncertain"],yn)}</select></label>${categories(x.risk_item_id,s)}<textarea placeholder="risk uncertain时必填；其他情况可记录边界判断" onchange="setv('${x.risk_item_id}','risk_notes',this.value)">${esc(s.risk_notes)}</textarea></section>`;save();}
function move(d){idx=Math.max(0,Math.min(DATA.items.length-1,idx+d));render();}function jumpIncomplete(){const start=idx;for(let k=1;k<=DATA.items.length;k++){const j=(start+k)%DATA.items.length;if(!complete(DATA.items[j])){idx=j;render();return;}}}function rows(){return DATA.items.map(x=>{ensure(x.risk_item_id);return {protocol:PROTOCOL,risk_item_id:x.risk_item_id,...state[x.risk_item_id],annotator_id:ann.trim()};});}function progress(){const n=DATA.items.filter(complete).length,e=document.getElementById("progress");e.className=n===DATA.items.length&&ann.trim()?"ok":"bad";e.textContent=`完整 ${n}/${DATA.items.length}；标注者ID ${ann.trim()?"已填":"未填"}`;}function downloadRows(){const r=rows();if((r.some((_,i)=>!complete(DATA.items[i]))||!ann.trim())&&!confirm("尚未全部完成，仍导出吗？"))return;const b=new Blob([r.map(JSON.stringify).join("\n")+"\n"],{type:"application/jsonl"}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=`${PROTOCOL}-${DATA.manifest.panel_role}.jsonl`;a.click();URL.revokeObjectURL(a.href);}document.getElementById("ann").value=ann;document.getElementById("ann").onchange=e=>{ann=e.target.value;save();};render();</script></body></html>'''


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def fit_helper() -> Any:
    spec = importlib.util.spec_from_file_location("v5_fit_review_helper", FIT_REVIEW_HELPER)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load frozen V5 review helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def visible_conversation(call: dict[str, Any]) -> str:
    user = [str(message["content"]) for message in call["messages"] if message["role"] == "user"]
    if len(user) != 1 or not user[0].startswith("Visible current conversation:\n"):
        raise RuntimeError(f"unexpected visible prompt: {call['call_id']}")
    return user[0].split("Visible current conversation:\n", 1)[1].strip()


def authorization(call: dict[str, Any]) -> str:
    system = [str(message["content"]) for message in call["messages"] if message["role"] == "system"]
    if len(system) != 1:
        raise RuntimeError(f"unexpected system prompt: {call['call_id']}")
    text = system[0]
    if call["arm"] == "OFF":
        return "Authorized evidence: visible current conversation only. No prior user-specific memory or strategy resource was authorized for this reply."
    marker = "\n\nResource 1\n"
    if marker not in text:
        raise RuntimeError(f"ON call lacks resource surface: {call['call_id']}")
    return text.split(marker, 1)[1].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v5_1_confirmation_plan_v1")
    parser.add_argument("--execution-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v5_1_confirmation_execution_v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v5_1_confirmation_review_v1_candidate")
    args = parser.parse_args()
    manifest = read_json(args.plan_dir / "freeze_manifest.json")
    summary = read_json(args.execution_dir / "execution_summary.json")
    if manifest["status"] != "READY_FOR_SINGLE_V5_1_CONFIRMATION_EXECUTION":
        raise RuntimeError("confirmation plan is not frozen")
    if summary["status"] != "COMPLETE_AWAITING_SINGLE_V5_1_CONFIRMATION_REVIEW":
        raise RuntimeError("confirmation execution is incomplete")
    if summary["invalid_itt_rows"] != 0 or summary["completed_calls"] != 512:
        raise RuntimeError("confirmation execution has invalid or missing rows")

    calls = {str(row["call_id"]): row for row in rows(args.plan_dir / "call_plan_private.jsonl")}
    outcomes = {str(row["call_id"]): row for row in rows(args.execution_dir / "outcomes_ordered_private.jsonl")}
    bindings = {str(row["state_id"]): row for row in rows(args.plan_dir / "state_policy_bindings_private.jsonl")}
    if len(calls) != 512 or set(calls) != set(outcomes) or len(bindings) != 128:
        raise RuntimeError("confirmation plan/outcome/binding identities differ")
    paired: dict[tuple[str, int], list[str]] = defaultdict(list)
    for call_id, call in calls.items():
        paired[(str(call["state_id"]), int(call["seed"]))].append(call_id)
    if len(paired) != 256:
        raise RuntimeError("expected 256 confirmation ON/OFF pairs")

    quality_rows: list[dict[str, Any]] = []
    quality_key: list[dict[str, Any]] = []
    by_component: dict[str, list[tuple[str, int]]] = {component: [] for component in COMPONENTS}
    for key, call_ids in paired.items():
        component = str(calls[call_ids[0]]["target_component"])
        by_component[component].append(key)
    on_as_a: set[tuple[str, int]] = set()
    for component, keys in by_component.items():
        ordered = sorted(keys, key=lambda key: stable_hex(PROTOCOL, "position", component, key[0], str(key[1]), n=32))
        if len(ordered) != 64:
            raise RuntimeError(f"expected 64 quality pairs for {component}")
        on_as_a.update(ordered[:32])

    for state_seed in sorted(paired):
        call_ids = paired[state_seed]
        by_arm = {str(calls[call_id]["arm"]): call_id for call_id in call_ids}
        if set(by_arm) != {"ON", "OFF"}:
            raise RuntimeError(f"bad arm pair: {state_seed}")
        on_id, off_id = by_arm["ON"], by_arm["OFF"]
        first, second = (on_id, off_id) if state_seed in on_as_a else (off_id, on_id)
        blind_id = "v51q_" + stable_hex(PROTOCOL, "quality", state_seed[0], str(state_seed[1]), n=24)
        visible = visible_conversation(calls[on_id])
        quality_rows.append({
            "protocol": QUALITY_PROTOCOL,
            "blind_item_id": blind_id,
            "visible_conversation": visible,
            "response_a": str(outcomes[first]["final_response"]),
            "response_b": str(outcomes[second]["final_response"]),
        })
        quality_key.append({
            "protocol": PROTOCOL,
            "blind_item_id": blind_id,
            "state_id": state_seed[0],
            "seed": state_seed[1],
            "component": calls[on_id]["target_component"],
            "counterfactual_group_id": calls[on_id]["counterfactual_group_id_private_analysis_only"],
            "response_a_call_id": first,
            "response_a_arm": calls[first]["arm"],
            "response_b_call_id": second,
            "response_b_arm": calls[second]["arm"],
            "policy_decisions": bindings[state_seed[0]]["policy_decisions"],
        })

    risk_rows: list[dict[str, Any]] = []
    risk_key: list[dict[str, Any]] = []
    for call_id in sorted(calls):
        call = calls[call_id]
        risk_id = "v51r_" + stable_hex(PROTOCOL, "risk", call_id, n=24)
        risk_rows.append({
            "protocol": RISK_PROTOCOL,
            "risk_item_id": risk_id,
            "visible_conversation": visible_conversation(call),
            "authorized_evidence_and_instruction": authorization(call),
            "candidate_response": str(outcomes[call_id]["final_response"]),
        })
        risk_key.append({
            "protocol": PROTOCOL,
            "risk_item_id": risk_id,
            "call_id": call_id,
            "state_id": call["state_id"],
            "seed": call["seed"],
            "component": call["target_component"],
            "arm": call["arm"],
            "counterfactual_group_id": call["counterfactual_group_id_private_analysis_only"],
            "policy_decisions": bindings[str(call["state_id"])]["policy_decisions"],
        })

    helper = fit_helper()
    quality_packet = {"manifest": {"protocol": QUALITY_PROTOCOL, "panel_role": "primary", "items": len(quality_rows)}, "items": quality_rows}
    risk_packet = {"manifest": {"protocol": RISK_PROTOCOL, "panel_role": "primary", "items": len(risk_rows)}, "items": risk_rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "quality_packet.json", quality_packet)
    write_json(args.out_dir / "risk_packet.json", risk_packet)
    write_jsonl(args.out_dir / "private_quality_key.jsonl", quality_key)
    write_jsonl(args.out_dir / "private_risk_key.jsonl", risk_key)
    (args.out_dir / "human_quality_blind.html").write_text(
        helper.QUALITY_HTML.replace("__DATA__", helper._safe_script_json(quality_packet)).replace("__PROTOCOL__", QUALITY_PROTOCOL), encoding="utf-8"
    )
    (args.out_dir / "human_grounding_risk.html").write_text(
        RISK_HTML.replace("__DATA__", helper._safe_script_json(risk_packet)).replace("__PROTOCOL__", RISK_PROTOCOL), encoding="utf-8"
    )
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_CONFIRMATION_HUMAN_REVIEW",
        "quality_pairs": len(quality_rows),
        "risk_arm_reviews": len(risk_rows),
        "quality_position_distribution": dict(Counter(row["response_a_arm"] for row in quality_key)),
        "risk_arm_distribution": dict(Counter(row["arm"] for row in risk_key)),
        "function_review_required": False,
        "independent_overlap_required": False,
        "reason_no_new_overlap": "The same frozen instrument already received a FIT overlap and centralized adjudication; confirmation consumes one complete primary proxy panel without another development loop.",
        "input_sha256": {
            "freeze_manifest": sha256_file(args.plan_dir / "freeze_manifest.json"),
            "call_plan": sha256_file(args.plan_dir / "call_plan_private.jsonl"),
            "policy_bindings": sha256_file(args.plan_dir / "state_policy_bindings_private.jsonl"),
            "execution_summary": sha256_file(args.execution_dir / "execution_summary.json"),
            "outcomes": sha256_file(args.execution_dir / "outcomes_ordered_private.jsonl"),
        },
        "outputs_sha256": {
            "quality_packet": sha256_file(args.out_dir / "quality_packet.json"),
            "risk_packet": sha256_file(args.out_dir / "risk_packet.json"),
            "private_quality_key": sha256_file(args.out_dir / "private_quality_key.jsonl"),
            "private_risk_key": sha256_file(args.out_dir / "private_risk_key.jsonl"),
        },
        "method_or_policy_changed_after_confirmation": False,
    }
    write_json(args.out_dir / "review_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
