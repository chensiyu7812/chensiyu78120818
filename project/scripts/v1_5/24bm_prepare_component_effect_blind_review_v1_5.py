#!/usr/bin/env python3
"""Build one blinded quality packet for all 256 clean component contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metacom_pm.contracts import RuntimeState
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-four-component-effect-human-quality-blind-v1"
GENERATION_PROTOCOL = (
    "pm-v1.5-four-component-contrast-generation-execution-v1"
)
REPEAT_COUNT = 52


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 四组件效应盲评</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:18px;background:#f4f6f8;color:#18212b}
.top{position:sticky;top:0;z-index:2;background:#fff;border:1px solid #ccd5df;border-radius:10px;padding:12px;margin-bottom:16px}
.item{background:#fff;border:1px solid #ccd5df;border-radius:10px;padding:18px;margin:16px 0}
.summary,.dialogue,.response{white-space:pre-wrap;line-height:1.55;border-radius:8px;padding:12px}
.summary{background:#f1f4f7}.dialogue{background:#f8fafc;margin-top:8px}.response{background:#f7f4ff;min-height:82px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
select,textarea,input{font:inherit;padding:8px;margin:5px 8px 5px 0}
textarea{width:96%;min-height:60px}.hint{color:#52606d;font-size:.94rem}
button{padding:9px 14px;margin-right:8px}.done{border-color:#3a8f5b}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><b>PM V1.5 四组件同状态盲评</b><span id="progress"></span>
<button onclick="downloadRows()">导出 JSONL</button>
<button onclick="clearAll()">清空本页缓存</button>
<div class="hint">只按可见对话判断。A/B 位置不代表任何条件。只有足以改变采用决定的差异才选 A/B；轻微措辞偏好选 tie。不要猜测资源、记忆或策略身份。包内含少量重复项用于一致性检查，请把每项当作独立题目。</div></div>
<div id="root"></div>
<script>
const DATA=__DATA__;
const KEY="pm_v1_5_four_component_effect_human_quality_blind_v1";
const CRITERIA={
request_and_dialogue_fit:"请求与对话适配",
visible_context_fidelity:"可见语境忠实度",
emotional_understanding:"情绪理解与回应",
immediate_helpfulness:"当下实际帮助",
clarity_naturalness:"清晰自然",
materially_equivalent:"实质等价",
uncertain:"无法可靠判断"
};
let state=JSON.parse(localStorage.getItem(KEY)||"{}");
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function ensure(id){if(!state[id])state[id]={quality_preference:"",decisive_criterion:"",quality_notes:"",annotator_id:""};}
function setv(id,k,v){ensure(id);state[id][k]=v;localStorage.setItem(KEY,JSON.stringify(state));render();}
function dialogue(item){return item.recent_dialogue.map(x=>(x.role==="user"?"用户":"支持者")+": "+x.content).join("\n")+"\n用户: "+item.current_user_text;}
function rows(){return DATA.items.map(item=>{ensure(item.blind_item_id);return {protocol:DATA.manifest.protocol,blind_item_id:item.blind_item_id,...state[item.blind_item_id]};});}
function progress(){const r=rows(),n=r.filter(x=>x.quality_preference&&x.decisive_criterion).length;document.getElementById("progress").textContent=`　已完成 ${n} / ${r.length}`;}
function render(){document.getElementById("root").innerHTML=DATA.items.map((item,i)=>{ensure(item.blind_item_id);const s=state[item.blind_item_id],done=s.quality_preference&&s.decisive_criterion;return `<section class="item ${done?"done":""}"><h2>${i+1} / ${DATA.items.length}</h2>${item.current_session_summary?`<div class="summary">背景摘要：${esc(item.current_session_summary)}</div>`:""}<div class="dialogue">${esc(dialogue(item))}</div><div class="grid"><div><h3>回复 A</h3><div class="response">${esc(item.response_a)}</div></div><div><h3>回复 B</h3><div class="response">${esc(item.response_b)}</div></div></div><h3>哪个回复实质更好？</h3><select onchange="setv('${item.blind_item_id}','quality_preference',this.value)"><option value="">请选择</option>${["A","B","tie","uncertain"].map(v=>`<option value="${v}" ${s.quality_preference===v?"selected":""}>${v}</option>`).join("")}</select><select onchange="setv('${item.blind_item_id}','decisive_criterion',this.value)"><option value="">请选择决定性标准</option>${Object.entries(CRITERIA).map(([k,v])=>`<option value="${k}" ${s.decisive_criterion===k?"selected":""}>${esc(v)}</option>`).join("")}</select><textarea placeholder="可选：用可见文本简述决定性差异" onchange="setv('${item.blind_item_id}','quality_notes',this.value)">${esc(s.quality_notes)}</textarea><input placeholder="annotator_id" value="${esc(s.annotator_id)}" onchange="setv('${item.blind_item_id}','annotator_id',this.value)"></section>`;}).join("");progress();}
function downloadRows(){const r=rows(),missing=r.filter(x=>!x.quality_preference||!x.decisive_criterion);if(missing.length&&!confirm(`还有 ${missing.length} 条未完成，仍然导出吗？`))return;const b=new Blob([r.map(x=>JSON.stringify(x)).join("\n")+"\n"],{type:"application/jsonl"}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="four_component_effect_quality_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}
function clearAll(){if(confirm("确定清空本页全部标注缓存？")){localStorage.removeItem(KEY);state={};render();}}
render();
</script></body></html>"""


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _public_item(
    *,
    state: RuntimeState,
    blind_item_id: str,
    response_a: str,
    response_b: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "blind_item_id": blind_item_id,
        "current_session_summary": state.current_session_summary,
        "recent_dialogue": [
            {"role": turn.role, "content": turn.content}
            for turn in state.current_session_history
        ],
        "current_user_text": state.current_user_text,
        "response_a": response_a,
        "response_b": response_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_contrast_blueprint_v1",
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_contrast_generation_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_contrast_generation_v1_execution",
    )
    parser.add_argument(
        "--runtime-states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "runtime_states.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_effect_human_quality_blind_v1",
    )
    args = parser.parse_args()

    summary = read_json(args.execution_dir / "generation_summary.json")
    if (
        summary.get("protocol") != GENERATION_PROTOCOL
        or summary.get("status") != "COMPLETE"
        or summary.get("completed_calls") != 256
    ):
        raise RuntimeError("component generation is not complete")
    blueprint = {
        str(row["contrast_slot_id"]): dict(row)
        for row in iter_jsonl(
            args.blueprint_dir / "contrast_blueprint.jsonl"
        )
    }
    registry = _rows(args.plan_dir / "response_arm_registry.jsonl")
    new_outcomes = {
        (str(row["contrast_slot_id"]), str(row["arm"])): dict(row)
        for row in iter_jsonl(
            args.execution_dir / "generation_outcomes.jsonl"
        )
    }
    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(args.runtime_states)
        )
    }
    arms: dict[tuple[str, str], dict[str, Any]] = {}
    for row in registry:
        key = (str(row["contrast_slot_id"]), str(row["arm"]))
        response = row.get("response")
        if response is None:
            response = new_outcomes[key]["response"]
        enriched = dict(row)
        enriched["response"] = str(response)
        enriched["response_sha256"] = sha256_text(str(response))
        arms[key] = enriched
    if len(arms) != 512:
        raise RuntimeError("expected exactly 512 realized response arms")

    repeat_slots = {
        slot_id
        for slot_id in sorted(
            blueprint,
            key=lambda value: stable_hex(
                PROTOCOL, "reliability-repeat", value, n=32
            ),
        )[:REPEAT_COUNT]
    }
    presentations: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    for slot_id, contrast in blueprint.items():
        for presentation in (
            ["primary", "repeat"] if slot_id in repeat_slots else ["primary"]
        ):
            blind_id = "component_blind_" + stable_hex(
                PROTOCOL, slot_id, presentation, n=24
            )
            swap = (
                int(
                    stable_hex(
                        PROTOCOL, "position", slot_id, presentation, n=8
                    ),
                    16,
                )
                % 2
                == 1
            )
            role_a, role_b = (
                ("treatment", "control")
                if swap
                else ("control", "treatment")
            )
            state = states[str(contrast["state_id"])]
            item = _public_item(
                state=state,
                blind_item_id=blind_id,
                response_a=arms[(slot_id, role_a)]["response"],
                response_b=arms[(slot_id, role_b)]["response"],
            )
            presentations.append(item)
            private_key.append(
                {
                    "protocol": PROTOCOL,
                    "blind_item_id": blind_id,
                    "contrast_slot_id": slot_id,
                    "repeat_group_id": "repeat_group_"
                    + stable_hex(PROTOCOL, slot_id, n=24),
                    "is_reliability_repeat": presentation == "repeat",
                    "component": contrast["component"],
                    "split": contrast["split"],
                    "state_id": contrast["state_id"],
                    "control_action": contrast["control_action"],
                    "treatment_action": contrast["treatment_action"],
                    "a_role": role_a,
                    "b_role": role_b,
                    "a_response_sha256": arms[(slot_id, role_a)][
                        "response_sha256"
                    ],
                    "b_response_sha256": arms[(slot_id, role_b)][
                        "response_sha256"
                    ],
                    "effect_label": "UNKNOWN_BEFORE_HUMAN_ANNOTATION",
                }
            )
    order = sorted(
        range(len(presentations)),
        key=lambda index: stable_hex(
            PROTOCOL,
            "presentation-order",
            presentations[index]["blind_item_id"],
            n=32,
        ),
    )
    presentations = [presentations[index] for index in order]
    if len(presentations) != 256 + REPEAT_COUNT:
        raise RuntimeError("unexpected presentation count")

    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_BOUNDED_BLIND_QUALITY_REVIEW",
        "unique_contrasts": 256,
        "reliability_repeats": REPEAT_COUNT,
        "review_presentations": len(presentations),
        "repeat_fraction_of_unique": REPEAT_COUNT / 256,
        "blind_fields_excluded": [
            "component",
            "split",
            "control/treatment identity",
            "action ids",
            "memory ids and content",
            "strategy card",
            "retrieval score",
            "generation source",
        ],
        "quality_rule": (
            "Choose A/B only for a material quality difference under the "
            "visible dialogue; slight stylistic preference is tie."
        ),
        "tie_policy_after_unblinding": (
            "tie maps to component off because it provides no material "
            "quality gain to justify added cost"
        ),
        "risk_stage": (
            "A separate minimal review is generated only for component-on "
            "quality winners; it is not mixed into this quality judgment."
        ),
        "effect_labels_created": False,
        "inputs": {
            "blueprint": str(
                (
                    args.blueprint_dir / "contrast_blueprint.jsonl"
                ).relative_to(ROOT)
            ),
            "response_registry": str(
                (
                    args.plan_dir / "response_arm_registry.jsonl"
                ).relative_to(ROOT)
            ),
            "new_generation_outcomes": str(
                (
                    args.execution_dir / "generation_outcomes.jsonl"
                ).relative_to(ROOT)
            ),
            "runtime_states": str(args.runtime_states.relative_to(ROOT)),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.out_dir / "human_blind_packet.jsonl"
    key_path = args.out_dir / "private_blind_key.jsonl"
    manifest_path = args.out_dir / "manifest.json"
    html_path = args.out_dir / "human_blind_review.html"
    template_path = args.out_dir / "blank_quality_annotations.jsonl"
    write_jsonl(packet_path, presentations)
    write_jsonl(key_path, private_key)
    write_jsonl(
        template_path,
        [
            {
                "protocol": PROTOCOL,
                "blind_item_id": row["blind_item_id"],
                "quality_preference": None,
                "decisive_criterion": None,
                "quality_notes": "",
                "annotator_id": "",
            }
            for row in presentations
        ],
    )
    write_json(manifest_path, manifest)
    html_data = {
        "manifest": manifest,
        "items": presentations,
    }
    html_path.write_text(
        HTML_TEMPLATE.replace("__DATA__", canonical_json(html_data)),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (packet_path, key_path, template_path, html_path)
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": manifest["status"],
                "unique_contrasts": manifest["unique_contrasts"],
                "reliability_repeats": manifest["reliability_repeats"],
                "review_presentations": manifest[
                    "review_presentations"
                ],
                "human_review": str(html_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
