#!/usr/bin/env python3
"""Prepare the fixed minimum human gate for the corrected Step2 executor."""

from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-corrected-step2-minimum-human-gate-v1"
PLAN_DIR = ROOT / "outputs/pm_v1_5b_corrected_step2_execution_gate_v3_candidate"
EXEC_DIR = ROOT / "outputs/pm_v1_5b_corrected_step2_execution_gate_v3_execution"
OUT_DIR = ROOT / "outputs/pm_v1_5b_corrected_step2_minimum_human_gate_v1_candidate"
SINGLE_ACTIONS = ("MP+R0", "MS+R0", "ME+R0", "M0+RS")
MULTI_ACTIONS = ("MPE+RS", "MSE+RS", "MPMSME+RS")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _select(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed action-stratified selection; never inspect response quality."""

    selected: list[dict[str, Any]] = []
    for action in SINGLE_ACTIONS:
        rows = [row for row in outcomes if row["requested_action_id"] == action]
        rows.sort(key=lambda row: stable_hex(PROTOCOL, action, row["call_id"], n=32))
        if len(rows) != 8:
            raise RuntimeError(f"expected eight internal rows for {action}, got {len(rows)}")
        selected.extend(rows[:2])
    for action in MULTI_ACTIONS:
        rows = [row for row in outcomes if row["requested_action_id"] == action]
        rows.sort(key=lambda row: stable_hex(PROTOCOL, action, row["call_id"], n=32))
        if len(rows) != 4:
            raise RuntimeError(f"expected four development rows for {action}, got {len(rows)}")
        if action == "MPMSME+RS":
            selected.extend(rows)
        else:
            selected.extend(rows[:2])
    if len(selected) != 16 or len({row["call_id"] for row in selected}) != 16:
        raise RuntimeError("corrected Step2 human gate must contain 16 unique calls")
    if not any(row["fallback_executed"] for row in selected):
        raise RuntimeError("the fixed all-component stratum must include the safety fallback")
    return sorted(
        selected,
        key=lambda row: stable_hex(PROTOCOL, "display", row["call_id"], n=32),
    )


def _visible_context(call: dict[str, Any]) -> str:
    content = str(call["messages"][1]["content"])
    marker = "\n\nEXECUTION PLAN"
    if marker not in content:
        raise RuntimeError(f"{call['call_id']}: visible-context marker missing")
    return content.split(marker, 1)[0].strip()


def _render(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    title = "PM V1.5b · 修复后 Step2 最小功能与误用核验（16条）"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:22px;background:#f4f6f8;color:#17202a}}.note,.item{{background:#fff;border:1px solid #d6dce4;border-radius:10px;padding:17px;margin:15px 0}}.note{{border-left:5px solid #0969da}}.context,.resource,.plan,.response,.fallback{{white-space:pre-wrap;padding:12px;border-radius:7px;line-height:1.5}}.context{{background:#edf2f6}}.resource{{background:#fff8c5;border:1px solid #d4a72c}}.plan{{background:#eefbf3;border:1px solid #85c79f}}.response{{background:#f4f0ff;border:1px solid #a99bd4}}.fallback{{background:#fff1f0;border:1px solid #e5a7a2}}select,textarea,input{{width:100%;box-sizing:border-box;padding:8px;margin:5px 0 10px}}textarea{{min-height:70px}}button{{padding:9px 14px}}.sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}.ok{{color:#1a7f37}}.bad{{color:#cf222e}}</style></head><body><h1>{html.escape(title)}</h1>
<div class="note"><b>只核验 Step2，不评价 PM 的开关选择，也不重做 A/B 回复质量。</b><br>对每个被 Step1 开启的组件分别判断它是否在回复中真正做功；仅提到过去、套模板、或复述当前消息不算。MS/ME 要明确保留过去出处，并把当前相关性保持为待核实；虚构“上次有效”、把旧事实升级为当前事实/原因、或越过当前边界，均计 material misuse。若显示无资源回退，只判断回退是否可采用，不能替原资源版记功。`M0+R0` 始终是合法动作；本页只抽查已请求资源的执行。</div>
<div class="sticky"><input id="ann" placeholder="annotator_id"><button onclick="downloadRows()">导出 JSONL</button> <span id="prog"></span></div><div id="root"></div><script>
const DATA={payload},KEY="pm_v15b_corrected_step2_human_gate_v1";let state=JSON.parse(localStorage.getItem(KEY)||"{{}}"),ann=localStorage.getItem(KEY+":ann")||"";
function esc(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}function opts(v,vals){{return vals.map(x=>`<option value="${{x}}" ${{v===x?"selected":""}}>${{x}}</option>`).join("")}}function blank(x){{const c={{}};x.requested_components.forEach(k=>c[k]="");return {{adjudicable:"",component_functional:c,material_misuse:"",misuse_categories:"",fallback_usable:"",literal_response_excerpt:"",review_notes:""}}}}function ensure(x){{if(!state[x.review_item_id])state[x.review_item_id]=blank(x)}}function save(){{localStorage.setItem(KEY,JSON.stringify(state));progress()}}function setv(x,k,v){{ensure(x);state[x.review_item_id][k]=v;save()}}function setc(x,k,v){{ensure(x);state[x.review_item_id].component_functional[k]=v;save()}}
function render(){{document.getElementById("root").innerHTML=DATA.map((x,i)=>{{ensure(x);const s=state[x.review_item_id],yn=["yes","no","uncertain"],plans=x.requested_components.map(c=>`${{c}} · function=${{x.plans[c].expected_function}}\nrequired=${{x.plans[c].required_contribution}}`).join("\n\n"),resources=x.requested_components.map(c=>`${{c}}:\n${{x.resources[c]}}`).join("\n\n"),componentFields=x.requested_components.map(c=>`<label>${{c}} 是否真正、非表面地做功？<select onchange='setc(DATA[${{i}}],"${{c}}",this.value)'><option value="">请选择</option>${{opts(s.component_functional[c],yn)}}</select></label>`).join("");return `<section class="item"><h2>${{i+1}} / ${{DATA.length}} · 请求动作 ${{esc(x.requested_action_id)}}</h2><h3>可见语境</h3><div class="context">${{esc(x.visible_context)}}</div><h3>已选资源</h3><div class="resource">${{esc(resources)}}</div><h3>逐组件功能合同</h3><div class="plan">${{esc(plans)}}</div><h3>资源版回复（审核对象）</h3><div class="response">${{esc(x.application_response)}}</div>${{x.fallback_response?`<h3>机器拦截后的 M0+R0 回退</h3><div class="fallback">${{esc(x.fallback_response)}}</div>`:""}}<label>能否可靠裁决？<select onchange='setv(DATA[${{i}}],"adjudicable",this.value)'><option value="">请选择</option>${{opts(s.adjudicable,yn)}}</select></label>${{componentFields}}<label>资源版是否有足以阻止采用的 material misuse？<select onchange='setv(DATA[${{i}}],"material_misuse",this.value)'><option value="">请选择</option>${{opts(s.material_misuse,yn)}}</select></label><input placeholder="若 misuse：fabricated_recall / stale_or_conflicting_use / unsupported_personal_claim / overgeneralized_pattern_or_cause / boundary_violation / excessive_directiveness" value="${{esc(s.misuse_categories)}}" onchange='setv(DATA[${{i}}],"misuse_categories",this.value)'><label>若显示回退，回退是否可采用？无回退选 not_applicable。<select onchange='setv(DATA[${{i}}],"fallback_usable",this.value)'><option value="">请选择</option>${{opts(s.fallback_usable,["yes","no","not_applicable","uncertain"])}}</select></label><textarea placeholder="粘贴支持判断的资源版回复原文；无则填 [none]" onchange='setv(DATA[${{i}}],"literal_response_excerpt",this.value)'>${{esc(s.literal_response_excerpt)}}</textarea><textarea placeholder="一句话说明哪些组件做功、空转或误用" onchange='setv(DATA[${{i}}],"review_notes",this.value)'>${{esc(s.review_notes)}}</textarea></section>`}}).join("");document.getElementById("ann").value=ann;progress()}}document.getElementById("ann").onchange=e=>{{ann=e.target.value.trim();localStorage.setItem(KEY+":ann",ann);progress()}};function complete(x){{ensure(x);const s=state[x.review_item_id];return s.adjudicable&&x.requested_components.every(c=>s.component_functional[c])&&s.material_misuse&&s.fallback_usable&&s.literal_response_excerpt&&s.review_notes}}function rows(){{return DATA.map(x=>{{ensure(x);return {{protocol:"{PROTOCOL}",review_item_id:x.review_item_id,...state[x.review_item_id],annotator_id:ann}}}})}}function progress(){{const n=DATA.filter(complete).length,e=document.getElementById("prog");e.className=n===DATA.length&&ann?"ok":"bad";e.textContent=`完整 ${{n}} / ${{DATA.length}}；标注者ID ${{ann?"已填":"未填"}}`}}function downloadRows(){{const r=rows();if((DATA.some(x=>!complete(x))||!ann)&&!confirm("尚未全部完成，仍导出吗？"))return;const b=new Blob([r.map(JSON.stringify).join("\n")+"\n"],{{type:"application/jsonl"}}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="corrected_step2_human_gate.jsonl";a.click();URL.revokeObjectURL(a.href)}}render();</script></body></html>'''


def main() -> None:
    calls = {str(row["call_id"]): row for row in _rows(PLAN_DIR / "call_plan_private.jsonl")}
    outcomes = _rows(EXEC_DIR / "generation_outcomes_ordered_private.jsonl")
    selected = _select(outcomes)
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for row in selected:
        call = calls[str(row["call_id"])]
        review_id = "step2_human_" + stable_hex(PROTOCOL, row["call_id"], n=24)
        application = row.get("bundle_application")
        if not application:
            raise RuntimeError(f"{row['call_id']}: V3 gate unexpectedly lacks a valid primary object")
        public.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "requested_action_id": row["requested_action_id"],
                "requested_components": call["requested_components"],
                "visible_context": _visible_context(call),
                "resources": call["selected_resources_private"],
                "plans": call["execution_plans"],
                "application_response": application["response"],
                "fallback_response": None if row["fallback"] is None else row["fallback"]["response"],
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "call_id": row["call_id"],
                "state_id": row["state_id"],
                "machine_checks_private": row["bundle_machine_checks"],
                "fallback_executed_private": row["fallback_executed"],
                "selection_rule": "2_per_single_action_2_per_three_component_action_all_4_all_component_action",
                "selection_uses_quality_risk_or_external_outcome": False,
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "human_review_packet.jsonl", public)
    write_jsonl(OUT_DIR / "private_review_key.jsonl", private)
    (OUT_DIR / "human_review.html").write_text(_render(public), encoding="utf-8")
    report = {
        "protocol": PROTOCOL,
        "status": "READY_MINIMUM_16_STEP2_FUNCTIONAL_AND_MISUSE_GATE",
        "items": len(public),
        "requested_action_counts": dict(sorted(Counter(row["requested_action_id"] for row in public).items())),
        "fallback_items": sum(row["fallback_response"] is not None for row in public),
        "pm_routing_correctness_requested": False,
        "pairwise_quality_requested": False,
        "machine_results_visible_to_reviewer": False,
        "m0_r0_remains_legal": True,
        "nonempty_invariant_scope": "only_components_activated_by_step1",
        "inputs": {
            "call_plan_sha256": sha256_file(PLAN_DIR / "call_plan_private.jsonl"),
            "outcomes_sha256": sha256_file(EXEC_DIR / "generation_outcomes_ordered_private.jsonl"),
        },
        "packet_sha256": sha256_text(canonical_json(public)),
    }
    write_json(OUT_DIR / "preparation_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
