#!/usr/bin/env python3
"""Prepare the one complete V5.2 FIT quality, risk, and function panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, stable_hex, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.2-full-fit-outcome-panel-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5.2-fit-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5.2-fit-grounding-risk-v1"
FUNCTION_PROTOCOL = "pm-v1.5-v5.2-fit-resource-function-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
QUALITY_HELPER = ROOT / "scripts/v1_5/25zy_prepare_v5_full_fit_outcome_review_v1_5.py"
RISK_HELPER = ROOT / "scripts/v1_5/26i_prepare_v5_1_confirmation_review_v1_5.py"


FUNCTION_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PM V1.5 V5.2 Resource Function</title><style>body{font-family:system-ui,sans-serif;max-width:980px;margin:auto;padding:18px;background:#f4f6f8;color:#17212b}header,.item{background:white;border:1px solid #d8dee6;border-radius:12px;padding:18px;margin:12px 0}.context,.evidence,.response{white-space:pre-wrap;border-radius:8px;padding:14px;border:1px solid #d8dee6}.context{background:#f7f9fb}.evidence{background:#fff8e6}.response{background:#eef5ff}label{display:block;margin:14px 0}select,textarea,input{width:100%;box-sizing:border-box;padding:9px;margin-top:5px}textarea{min-height:72px}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.ok{color:#16733c}.bad{color:#a33}</style></head><body><header><h1>V5.2 资源是否真正做功</h1><p>这里只审核Step2功能执行，不评价总体回复质量、不判断PM是否该开，也不评价risk。提到资源不等于做功；资源必须对回复内容、约束或支持动作产生可区分作用。</p><label>标注者ID<input id="ann"></label><div class="controls"><button onclick="move(-1)">上一条</button><button onclick="move(1)">下一条</button><button onclick="jumpIncomplete()">下一未完成</button><button onclick="downloadRows()">导出JSONL</button><span id="progress"></span></div></header><main id="root"></main><script>const DATA=__DATA__,PROTOCOL="__PROTOCOL__",KEY=PROTOCOL+":"+DATA.manifest.panel_role;let state=JSON.parse(localStorage.getItem(KEY)||"{}"),ann=localStorage.getItem(KEY+":ann")||"",idx=Number(localStorage.getItem(KEY+":idx")||0);function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}function ensure(id){if(!state[id])state[id]={resource_functionally_contributed:"",functional_evidence_excerpt:"",function_notes:""};}function save(){localStorage.setItem(KEY,JSON.stringify(state));localStorage.setItem(KEY+":ann",ann);localStorage.setItem(KEY+":idx",String(idx));progress();}function setv(id,k,v){ensure(id);state[id][k]=v;save();render();}function complete(x){ensure(x.function_item_id);const s=state[x.function_item_id];if(!s.resource_functionally_contributed)return false;if(s.resource_functionally_contributed==="yes"&&!s.functional_evidence_excerpt.trim())return false;if(["no","uncertain"].includes(s.resource_functionally_contributed)&&!s.function_notes.trim())return false;return true;}function render(){idx=Math.max(0,Math.min(idx,DATA.items.length-1));const x=DATA.items[idx];ensure(x.function_item_id);const s=state[x.function_item_id];document.getElementById("root").innerHTML=`<section class="item"><h2>${idx+1} / ${DATA.items.length}</h2><h3>可见对话</h3><div class="context">${esc(x.visible_conversation)}</div><h3>授权资源与执行边界</h3><div class="evidence">${esc(x.authorized_evidence_and_instruction)}</div><h3>审核对象回复</h3><div class="response">${esc(x.candidate_response)}</div><label>资源是否真正产生可区分功能？<select onchange="setv('${x.function_item_id}','resource_functionally_contributed',this.value)"><option value="">请选择</option><option value="yes" ${s.resource_functionally_contributed==='yes'?'selected':''}>是</option><option value="no" ${s.resource_functionally_contributed==='no'?'selected':''}>否</option><option value="uncertain" ${s.resource_functionally_contributed==='uncertain'?'selected':''}>不确定</option></select></label><textarea placeholder="yes时粘贴体现资源功能的回复原文" onchange="setv('${x.function_item_id}','functional_evidence_excerpt',this.value)">${esc(s.functional_evidence_excerpt)}</textarea><textarea placeholder="no/uncertain时说明资源为何只是表面出现或无法裁决" onchange="setv('${x.function_item_id}','function_notes',this.value)">${esc(s.function_notes)}</textarea></section>`;save();}function move(d){idx=Math.max(0,Math.min(DATA.items.length-1,idx+d));render();}function jumpIncomplete(){for(let k=1;k<=DATA.items.length;k++){const j=(idx+k)%DATA.items.length;if(!complete(DATA.items[j])){idx=j;render();return;}}}function progress(){const n=DATA.items.filter(complete).length,e=document.getElementById("progress");e.className=n===DATA.items.length&&ann.trim()?"ok":"bad";e.textContent=`完整 ${n}/${DATA.items.length}；标注者ID ${ann.trim()?"已填":"未填"}`;}function downloadRows(){const rows=DATA.items.map(x=>{ensure(x.function_item_id);return {protocol:PROTOCOL,function_item_id:x.function_item_id,...state[x.function_item_id],annotator_id:ann.trim()};});const b=new Blob([rows.map(JSON.stringify).join("\n")+"\n"],{type:"application/jsonl"}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=`${PROTOCOL}-${DATA.manifest.panel_role}.jsonl`;a.click();URL.revokeObjectURL(a.href);}document.getElementById("ann").value=ann;document.getElementById("ann").onchange=e=>{ann=e.target.value;save();};render();</script></body></html>'''


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load review helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def visible(call: dict[str, Any]) -> str:
    user = [str(item["content"]) for item in call["messages"] if item["role"] == "user"]
    prefix = "Visible current conversation:"
    if len(user) != 1 or not user[0].startswith(prefix):
        raise RuntimeError(f"invalid visible prompt: {call['call_id']}")
    return user[0][len(prefix):].strip()


def authorization(call: dict[str, Any]) -> str:
    if call["arm"] == "OFF":
        return "Only the visible current conversation was authorized. No user-specific memory or strategy resource was authorized."
    plan = dict(call["composition_plan"])
    parts: list[str] = []
    for clause in plan["locked_clauses"]:
        parts.append(
            "Backend-locked authorized content (must remain source-grounded): "
            + str(clause["text"])
        )
    if str(plan["response_preference"]).strip():
        parts.append("Authorized active response-format preference: " + str(plan["response_preference"]))
    if str(plan["strategy_instruction"]).strip():
        parts.append("Authorized atomic support instruction: " + str(plan["strategy_instruction"]))
    if not parts:
        raise RuntimeError(f"ON call has no public authorization: {call['call_id']}")
    text = "\n".join(parts)
    if any(marker in text for marker in ("mem_", "strategy_v", "card_")):
        raise RuntimeError(f"opaque ID leaked into public authorization: {call['call_id']}")
    return text


def deduplicate(
    records: list[dict[str, Any]],
    *,
    key: Callable[[dict[str, Any]], str],
    id_field: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    representatives: list[dict[str, Any]] = []
    representative_by_id: dict[str, str] = {}
    for digest, members in sorted(groups.items()):
        representative = min(members, key=lambda row: str(row[id_field]))
        representatives.append(representative)
        for member in members:
            representative_by_id[str(member[id_field])] = str(representative[id_field])
    return representatives, representative_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v5_2_locked_fit_plan_v2")
    parser.add_argument("--execution-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v5_2_locked_fit_execution_v2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v5_2_full_fit_outcome_review_v1_candidate")
    args = parser.parse_args()
    summary = read_json(args.execution_dir / "execution_summary.json")
    if summary["status"] != "COMPLETE_AWAITING_ONE_FULL_V5_2_FIT_OUTCOME_PANEL" or summary["completed_calls"] != 1024:
        raise RuntimeError("V5.2 FIT execution is incomplete")
    if summary["fallback_calls"] != 0 or summary["invalid_itt_rows"] != 0:
        raise RuntimeError("invalid or fallback V5.2 outcomes cannot enter the panel")

    calls = {str(row["call_id"]): row for row in rows(args.plan_dir / "call_plan_private.jsonl")}
    outcomes = {str(row["call_id"]): row for row in rows(args.execution_dir / "outcomes_ordered_private.jsonl")}
    if len(calls) != 1024 or set(calls) != set(outcomes):
        raise RuntimeError("plan/outcome identity mismatch")
    paired: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for call_id, call in calls.items():
        paired[(str(call["state_id"]), int(call["seed"]))][str(call["arm"])] = call_id
    if len(paired) != 512 or any(set(value) != {"ON", "OFF"} for value in paired.values()):
        raise RuntimeError("V5.2 requires 512 complete ON/OFF pairs")

    keys_by_component: dict[str, list[tuple[str, int]]] = {component: [] for component in COMPONENTS}
    for state_seed, arms in paired.items():
        keys_by_component[str(calls[arms["ON"]]["target_component"])].append(state_seed)
    on_as_a: set[tuple[str, int]] = set()
    for component, values in keys_by_component.items():
        ordered = sorted(values, key=lambda value: stable_hex(PROTOCOL, "position", component, value[0], str(value[1]), n=32))
        if len(ordered) != 128:
            raise RuntimeError(f"expected 128 pairs for {component}")
        on_as_a.update(ordered[:64])

    quality_all: list[dict[str, Any]] = []
    quality_key: list[dict[str, Any]] = []
    risk_all: list[dict[str, Any]] = []
    risk_key: list[dict[str, Any]] = []
    function_all: list[dict[str, Any]] = []
    function_key: list[dict[str, Any]] = []
    for state_seed, arms in sorted(paired.items()):
        on_id, off_id = arms["ON"], arms["OFF"]
        on_call, off_call = calls[on_id], calls[off_id]
        component = str(on_call["target_component"])
        group_id = str(on_call["group_id_private_analysis_only"])
        conversation = visible(on_call)
        if conversation != visible(off_call):
            raise RuntimeError(f"pair visible mismatch: {state_seed}")
        a_id, b_id = (on_id, off_id) if state_seed in on_as_a else (off_id, on_id)
        blind_id = "v52q_" + stable_hex(PROTOCOL, "quality", state_seed[0], str(state_seed[1]), n=24)
        quality_all.append({
            "protocol": QUALITY_PROTOCOL,
            "blind_item_id": blind_id,
            "visible_conversation": conversation,
            "response_a": str(outcomes[a_id]["final_response"]),
            "response_b": str(outcomes[b_id]["final_response"]),
            "dedup_signature_private": canonical_json([
                conversation,
                str(outcomes[on_id]["final_response"]),
                str(outcomes[off_id]["final_response"]),
            ]),
        })
        quality_key.append({
            "protocol": PROTOCOL, "blind_item_id": blind_id, "state_id": state_seed[0], "seed": state_seed[1],
            "component": component, "semantic_group_id": group_id, "on_call_id": on_id, "off_call_id": off_id,
            "a_arm": calls[a_id]["arm"], "b_arm": calls[b_id]["arm"],
        })
        for arm, call_id in (("ON", on_id), ("OFF", off_id)):
            risk_id = "v52r_" + stable_hex(PROTOCOL, "risk", call_id, n=24)
            risk_all.append({
                "protocol": RISK_PROTOCOL, "risk_item_id": risk_id, "visible_conversation": conversation,
                "authorized_evidence_and_instruction": authorization(calls[call_id]),
                "candidate_response": str(outcomes[call_id]["final_response"]),
            })
            risk_key.append({
                "protocol": PROTOCOL, "risk_item_id": risk_id, "call_id": call_id, "state_id": state_seed[0],
                "seed": state_seed[1], "component": component, "semantic_group_id": group_id, "arm": arm,
            })
        function_id = "v52f_" + stable_hex(PROTOCOL, "function", on_id, n=24)
        function_all.append({
            "protocol": FUNCTION_PROTOCOL, "function_item_id": function_id, "visible_conversation": conversation,
            "authorized_evidence_and_instruction": authorization(on_call),
            "candidate_response": str(outcomes[on_id]["final_response"]),
        })
        function_key.append({
            "protocol": PROTOCOL, "function_item_id": function_id, "call_id": on_id, "state_id": state_seed[0],
            "seed": state_seed[1], "component": component, "semantic_group_id": group_id, "arm": "ON",
        })

    quality_manual, quality_rep = deduplicate(
        quality_all,
        key=lambda row: str(row["dedup_signature_private"]),
        id_field="blind_item_id",
    )
    risk_manual, risk_rep = deduplicate(
        risk_all,
        key=lambda row: canonical_json([row["visible_conversation"], row["authorized_evidence_and_instruction"], row["candidate_response"]]),
        id_field="risk_item_id",
    )
    function_manual, function_rep = deduplicate(
        function_all,
        key=lambda row: canonical_json([row["visible_conversation"], row["authorized_evidence_and_instruction"], row["candidate_response"]]),
        id_field="function_item_id",
    )
    quality_key_by_id = {row["blind_item_id"]: row for row in quality_key}
    for row in quality_key:
        row["manual_representative_id"] = quality_rep[row["blind_item_id"]]
        row["manual_representative_a_arm"] = quality_key_by_id[
            row["manual_representative_id"]
        ]["a_arm"]
    for row in risk_key:
        row["manual_representative_id"] = risk_rep[row["risk_item_id"]]
    for row in function_key:
        row["manual_representative_id"] = function_rep[row["function_item_id"]]
    for row in quality_all:
        row.pop("dedup_signature_private", None)

    # Select complete semantic groups for a roughly 20% independent overlap.
    groups_by_component: dict[str, set[str]] = {component: set() for component in COMPONENTS}
    for row in quality_key:
        groups_by_component[row["component"]].add(row["semantic_group_id"])
    overlap_groups: set[str] = set()
    for component, group_ids in groups_by_component.items():
        expected = 64 if component == "RS" else 32
        if len(group_ids) != expected:
            raise RuntimeError(f"semantic group count mismatch for {component}: {len(group_ids)}")
        count = 13 if component == "RS" else 6
        overlap_groups.update(sorted(group_ids, key=lambda value: stable_hex(PROTOCOL, "overlap", component, value, n=32))[:count])
    quality_rep_groups = {row["manual_representative_id"]: row["semantic_group_id"] for row in quality_key}
    risk_rep_groups = {row["manual_representative_id"]: row["semantic_group_id"] for row in risk_key}
    function_rep_groups = {row["manual_representative_id"]: row["semantic_group_id"] for row in function_key}
    overlap_quality = [row for row in quality_manual if quality_rep_groups[row["blind_item_id"]] in overlap_groups]
    overlap_risk = [row for row in risk_manual if risk_rep_groups[row["risk_item_id"]] in overlap_groups]
    overlap_function = [row for row in function_manual if function_rep_groups[row["function_item_id"]] in overlap_groups]

    quality_helper = load_module(QUALITY_HELPER, "v52_quality_helper")
    risk_helper = load_module(RISK_HELPER, "v52_risk_helper")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packets = {
        "primary_quality_packet.jsonl": quality_manual,
        "primary_risk_packet.jsonl": risk_manual,
        "primary_function_packet.jsonl": function_manual,
        "overlap_quality_packet.jsonl": overlap_quality,
        "overlap_risk_packet.jsonl": overlap_risk,
        "overlap_function_packet.jsonl": overlap_function,
        "private_quality_key.jsonl": quality_key,
        "private_risk_key.jsonl": risk_key,
        "private_function_key.jsonl": function_key,
    }
    for name, data in packets.items():
        write_jsonl(args.out_dir / name, data)

    def html(path: Path, template: str, protocol: str, items: list[dict[str, Any]], role: str) -> None:
        payload = {"manifest": {"protocol": PROTOCOL, "panel_role": role}, "items": items}
        path.write_text(template.replace("__DATA__", quality_helper._safe_script_json(payload)).replace("__PROTOCOL__", protocol), encoding="utf-8")

    html(args.out_dir / "human_quality_primary.html", quality_helper.QUALITY_HTML, QUALITY_PROTOCOL, quality_manual, "primary")
    html(args.out_dir / "human_risk_primary.html", risk_helper.RISK_HTML, RISK_PROTOCOL, risk_manual, "primary")
    html(args.out_dir / "human_function_primary.html", FUNCTION_HTML, FUNCTION_PROTOCOL, function_manual, "primary")
    html(args.out_dir / "human_quality_overlap.html", quality_helper.QUALITY_HTML, QUALITY_PROTOCOL, overlap_quality, "independent_overlap")
    html(args.out_dir / "human_risk_overlap.html", risk_helper.RISK_HTML, RISK_PROTOCOL, overlap_risk, "independent_overlap")
    html(args.out_dir / "human_function_overlap.html", FUNCTION_HTML, FUNCTION_PROTOCOL, overlap_function, "independent_overlap")

    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_COMPLETE_V5_2_FIT_OUTCOME_REVIEW",
        "underlying_units": {"quality_pairs": 512, "risk_arm_responses": 1024, "on_arm_function_responses": 512},
        "manual_after_exact_deduplication": {
            "quality": len(quality_manual), "risk": len(risk_manual), "function": len(function_manual)
        },
        "exact_duplicate_rows_propagated": {
            "quality": 512 - len(quality_manual), "risk": 1024 - len(risk_manual), "function": 512 - len(function_manual)
        },
        "independent_overlap_manual": {
            "semantic_groups": len(overlap_groups), "quality": len(overlap_quality),
            "risk": len(overlap_risk), "function": len(overlap_function)
        },
        "quality_position_distribution_underlying": dict(Counter(row["a_arm"] for row in quality_key)),
        "risk_arm_distribution_underlying": dict(Counter(row["arm"] for row in risk_key)),
        "constructs_kept_separate": True,
        "no_outcome_used_to_modify_executor_or_routing": True,
        "inputs_sha256": {
            "plan": sha256_file(args.plan_dir / "call_plan_private.jsonl"),
            "execution": sha256_file(args.execution_dir / "outcomes_ordered_private.jsonl"),
        },
    }
    write_json(args.out_dir / "review_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
