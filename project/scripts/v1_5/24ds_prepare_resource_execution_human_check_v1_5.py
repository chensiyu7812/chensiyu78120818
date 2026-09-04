#!/usr/bin/env python3
"""Prepare the minimum 10-item human check of generator execution traces."""

from __future__ import annotations

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
PROTOCOL = "pm-v1.5-resource-execution-minimum-human-check-v1"
TRACE_PROTOCOL = "pm-v1.5-auditable-resource-execution-v2"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _resource_surface(row: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in row["selected_memory_items"]:
        lines.append(f"[{item['source']}] {item['text']}")
    for card in row["selected_strategy_cards"]:
        for key in ("support_move", "when_to_use", "when_not_to_use"):
            if card.get(key):
                lines.append(f"{key}: {card[key]}")
    return "\n".join(lines)


def _render(items: list[dict[str, Any]]) -> str:
    payload = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    title = "PM V1.5 · generator 资源执行最小人工核验（10条）"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1050px;margin:auto;padding:22px;background:#f4f6f8;color:#17202a}}
.note,.item{{background:#fff;border:1px solid #d6dce4;border-radius:10px;padding:17px;margin:15px 0}}
.note{{border-left:5px solid #0969da}}.context,.resource,.response,.trace{{white-space:pre-wrap;padding:12px;border-radius:7px;line-height:1.5}}
.context{{background:#edf2f6}}.resource{{background:#fff8c5;border:1px solid #d4a72c}}.response{{background:#f4f0ff;border:1px solid #a99bd4}}.trace{{background:#eefbf3;border:1px solid #85c79f}}
select,textarea,input{{width:100%;box-sizing:border-box;padding:8px;margin:5px 0 10px}}textarea{{min-height:72px}}
button{{padding:9px 14px}}.sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}
.ok{{color:#1a7f37}}.bad{{color:#cf222e}}.muted{{color:#57606a}}
</style></head><body><h1>{html.escape(title)}</h1>
<div class="note"><b>只核验执行，不重做回复质量评审，也不重判 PM 或检索。</b>
这 10 条由固定规则选出：每组件 2 个 use，加全部 2 个 ignore；没有依据回复好坏、judge结果或风险挑样本。
请问声明是否与回复正文一致、资源是否真正发挥了所声明的功能、是否出现足以阻止采用的明显误用。
“提到同一主题”不等于使用；MP偏好可通过行为隐式使用；轻微文风问题不算 material misuse。</div>
<div class="sticky"><input id="ann" placeholder="annotator_id"><button onclick="downloadRows()">导出 JSONL</button> <span id="prog"></span></div>
<div id="root"></div><script>
const DATA={payload}, KEY="pm_v15_resource_exec_human_check_v1";
let state=JSON.parse(localStorage.getItem(KEY)||"{{}}"),ann=localStorage.getItem(KEY+":ann")||"";
function esc(s){{return String(s??"").replace(/[&<>\"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;","'":"&#39;"}}[c]));}}
function blank(){{return {{declaration_supported:"",functionally_used:"",material_misuse:"",literal_response_excerpt:"",review_notes:""}}}}
function ensure(id){{if(!state[id])state[id]=blank()}}
function setv(id,k,v){{ensure(id);state[id][k]=v;localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function context(x){{return (x.history||[]).map(t=>t.role+": "+t.content).join("\n")+"\nuser: "+x.current_user_text}}
function opts(value){{return ["yes","no","uncertain"].map(v=>`<option value="${{v}}" ${{value===v?"selected":""}}>${{v}}</option>`).join("")}}
function render(){{document.getElementById("root").innerHTML=DATA.map((x,i)=>{{ensure(x.review_item_id);const s=state[x.review_item_id];return `<section class="item"><h2>${{i+1}} / ${{DATA.length}} · ${{esc(x.component)}} · ${{esc(x.resource_subtype)}}</h2><h3>可见当前语境</h3><div class="context">${{esc(context(x))}}</div><h3>已选资源</h3><div class="resource">${{esc(x.selected_resource)}}</div><h3>generator 执行声明</h3><div class="trace">decision: ${{esc(x.declared_decision)}}\nfunction: ${{esc(x.declared_function)}}\nreason: ${{esc(x.declared_reason)}}</div><h3>用户实际会看到的回复</h3><div class="response">${{esc(x.response)}}</div><label>1. use/ignore 声明是否得到回复正文支持？<select onchange="setv('${{x.review_item_id}}','declaration_supported',this.value)"><option value="">请选择</option>${{opts(s.declaration_supported)}}</select></label><label>2. 资源是否真正发挥所声明的功能？（ignore且确实应忽略也选 yes）<select onchange="setv('${{x.review_item_id}}','functionally_used',this.value)"><option value="">请选择</option>${{opts(s.functionally_used)}}</select></label><label>3. 是否存在明确 material misuse？<select onchange="setv('${{x.review_item_id}}','material_misuse',this.value)"><option value="">请选择</option>${{opts(s.material_misuse)}}</select></label><textarea placeholder="若判 use 或 misuse，请粘贴回复中的直接原文；否则可填 [none]" onchange="setv('${{x.review_item_id}}','literal_response_excerpt',this.value)">${{esc(s.literal_response_excerpt)}}</textarea><textarea placeholder="一句话说明资源做了什么、没做什么或如何误用" onchange="setv('${{x.review_item_id}}','review_notes',this.value)">${{esc(s.review_notes)}}</textarea></section>`}}).join("");document.getElementById("ann").value=ann;progress()}}
document.getElementById("ann").onchange=e=>{{ann=e.target.value.trim();localStorage.setItem(KEY+":ann",ann);progress()}};
function complete(s){{return s.declaration_supported&&s.functionally_used&&s.material_misuse&&s.literal_response_excerpt&&s.review_notes}}
function rows(){{return DATA.map(x=>{{ensure(x.review_item_id);return {{protocol:"{PROTOCOL}",review_item_id:x.review_item_id,...state[x.review_item_id],annotator_id:ann}}}})}}
function progress(){{const n=DATA.filter(x=>{{ensure(x.review_item_id);return complete(state[x.review_item_id])}}).length;const e=document.getElementById("prog");e.className=n===DATA.length&&ann?"ok":"bad";e.textContent=`完整 ${{n}} / ${{DATA.length}}；标注者ID ${{ann?"已填":"未填"}}`}}
function downloadRows(){{const r=rows(),bad=r.filter(x=>!complete(x));if((bad.length||!ann)&&!confirm("尚未全部完成，仍导出吗？"))return;const b=new Blob([r.map(JSON.stringify).join("\n")+"\n"],{{type:"application/jsonl"}}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="resource_execution_human_check.jsonl";a.click();URL.revokeObjectURL(a.href)}}
render();</script></body></html>'''


def main() -> None:
    trace_dir = ROOT / "outputs/pm_v1_5_auditable_resource_execution_v2"
    plan_dir = ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1"
    out_dir = ROOT / "outputs/pm_v1_5_resource_execution_human_check_v1_candidate"
    outcomes = _rows(trace_dir / "execution_outcomes.jsonl")
    resources = {
        str(row["state_id"]): row
        for row in _rows(plan_dir / "private_selected_resources.jsonl")
    }
    ignores = [
        row for row in outcomes if row["execution"]["resource_decision"] == "ignore"
    ]
    uses: list[dict[str, Any]] = []
    for component in ("RS", "MP", "MS", "ME"):
        candidates = [
            row
            for row in outcomes
            if row["component"] == component
            and row["execution"]["resource_decision"] == "use"
        ]
        candidates.sort(
            key=lambda row: stable_hex(PROTOCOL, "use", row["state_id"], n=32)
        )
        uses.extend(candidates[:2])
    selected = ignores + uses
    if len(ignores) != 2 or len(uses) != 8 or len(selected) != 10:
        raise RuntimeError(
            f"expected 2 ignores plus 8 stratified uses, got {len(ignores)}+{len(uses)}"
        )
    selected.sort(key=lambda row: stable_hex(PROTOCOL, "display", row["state_id"], n=32))
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for row in selected:
        selected_resource = resources[str(row["state_id"])]
        state = selected_resource["runtime_state"]
        review_id = "resource_exec_review_" + stable_hex(
            PROTOCOL, row["state_id"], n=24
        )
        public.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "component": row["component"],
                "resource_subtype": row["resource_subtype"],
                "history": state["current_session_history"],
                "current_user_text": state["current_user_text"],
                "selected_resource": _resource_surface(selected_resource),
                "declared_decision": row["execution"]["resource_decision"],
                "declared_function": row["execution"]["resource_function"],
                "declared_reason": row["execution"]["concise_decision_reason"],
                "response": row["execution"]["response"],
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "state_id": row["state_id"],
                "trace_call_id": row["call_id"],
                "selection_rule": "all_2_ignore_plus_stable_hash_2_use_per_component",
                "selection_uses_quality_judge_or_risk": False,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "human_review_packet.jsonl", public)
    write_jsonl(out_dir / "private_review_key.jsonl", private)
    (out_dir / "human_review.html").write_text(_render(public), encoding="utf-8")
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_MINIMUM_10_ITEM_HUMAN_CHECK",
        "items": len(public),
        "declared_use_items": len(uses),
        "declared_ignore_items": len(ignores),
        "component_counts": {
            component: sum(row["component"] == component for row in public)
            for component in ("RS", "MP", "MS", "ME")
        },
        "selection_uses_quality_judge_or_risk": False,
        "inputs": {
            "trace_outcomes_sha256": sha256_file(trace_dir / "execution_outcomes.jsonl"),
            "selected_resources_sha256": sha256_file(plan_dir / "private_selected_resources.jsonl"),
        },
        "outputs": {
            name: sha256_file(out_dir / name)
            for name in ("human_review_packet.jsonl", "private_review_key.jsonl", "human_review.html")
        },
        "packet_sha256": sha256_text(canonical_json(public)),
    }
    write_json(out_dir / "preparation_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
