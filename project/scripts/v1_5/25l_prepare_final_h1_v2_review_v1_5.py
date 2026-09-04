#!/usr/bin/env python3
"""Render the corrected H1-v2 two-gate candidate review packet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, stable_hex, write_json, write_jsonl
from metacom_pm.v1_5_final_candidate_contract import FINAL_CANDIDATE_CONTRACT_PROTOCOL


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-candidate-gold-human-review-v2"
OVERLAP_PROTOCOL = "pm-v1.5-final-candidate-gold-independent-overlap-v2"
STATIC_GATE_PROTOCOL = "pm-v1.5-final-h1-v2-pre-human-static-gate-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")


def _surface(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "component": str(value["component"]),
        "candidate_present": bool(value["candidate_present"]),
        "candidate_text": value["candidate_text"],
        "candidate_age_sessions": value["candidate_age_sessions"],
    }


def _counts(items: list[dict[str, Any]]) -> tuple[int, int]:
    present = sum(
        int(component["candidate_present"])
        for item in items
        for component in item["components"]
    )
    return present, len(items) * 4 - present


_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 H1 v2 双门审核</title><style>
body{font-family:system-ui,sans-serif;max-width:1180px;margin:auto;padding:20px;background:#f4f6f8;color:#18212b;line-height:1.5}
.note,.state,.candidate{background:white;border:1px solid #d5dce5;border-radius:10px;padding:15px;margin:13px 0}.note{border-left:5px solid #235f9c}.warn{border-left-color:#a45a00}.dialogue{background:#edf2f7;border-radius:7px;padding:11px;white-space:pre-wrap}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:11px}.candidate{margin:0;background:#fbfcfd}.absent{opacity:.72}select,textarea,input[type=text]{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 9px}textarea{min-height:58px}.sticky{position:sticky;top:0;background:#f4f6f8;padding:8px 0;z-index:2}button{padding:9px 14px}.tag{display:inline-block;background:#e6edf5;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px}label.check{display:block;font-size:13px}h2{margin-bottom:6px}
</style></head><body><h1>PM V1.5 H1 v2：Step 1 候选双门审核</h1>
<div class="note"><b>这不是回复质量评测。</b>只看当前可见对话和 exact Rank‑1 候选，分别判断两个问题：①候选在当前状态下是否相关、可用且安全；②相对于不注入资源的 R0，它是否提供了非冗余、功能明确的增量。只有两项都 yes 才形成 ON；任一 no 为 OFF；其余为 ABSTAIN。</div>
<div class="note warn"><b>RS 特别规则：</b>“用户需要支持”或“卡片动作看起来合适”只足以支持第一门，不自动等于 ON。第二门还要求该卡明确增加一个当前需要、尚未由可见对话满足的动作或约束。当前消息已直接规定同一动作、上一轮已经执行、普通寒暄/结束、事实回忆请求，均不应因卡片存在而 ON。是否真的改善生成回复留给后续同状态配对实验，不在本页猜测。</div>
<div class="note"><b>记忆特别规则：</b>owner、时间、当前目标必须匹配。MP 要有当前未明说的稳定偏好或实际约束；MS 要让会话摘要中的区分/目标对当前有用；ME 要包含可迁移的过去动作/选择及结果或机制。只同主题、纯背景、未解决事件、当前已重复、错实体或陈旧冲突均不能通过双门。为减少无必要工作，ON/OFF只需选字段和证据码；只有任一门为uncertain时才强制写notes。</div>
<div class="sticky"><input id="annotator" type="text" placeholder="annotator_id（导出前必填）" style="max-width:320px"> <button onclick="downloadRows()">导出 JSONL</button> <span id="progress"></span></div><div id="root"></div>
<script>const DATA=__PACKET__;const KEY=__KEY__;let S=JSON.parse(localStorage.getItem(KEY)||'{}');
const TYPES={MP:['MP_PREFERENCE','MP_PROFILE'],MS:['MS_SESSION'],ME:['ME_REUSABLE_OUTCOME','ME_CONTEXT_EVENT','ME_UNRESOLVED_EVENT'],RS:['RS_ATOMIC_MOVE']};
const CODES=['CURRENT_GOAL_FIT','OWNER_AND_TIME_VALID','SAFE_AND_BOUNDARY_COMPATIBLE','SPECIFIC_FUNCTIONAL_INCREMENT','CANDIDATE_ABSENT','CURRENTLY_REDUNDANT','WRONG_ENTITY_OR_GOAL','STALE_OR_CONFLICTING','BOUNDARY_OR_BURDEN_CONFLICT','WRONG_SUBTYPE_OR_FUNCTION','GENERIC_OR_NO_INCREMENT','AMBIGUOUS_VISIBLE_EVIDENCE'];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function id(b,c){return b+'::'+c}function save(k,f,v){S[k]??={};S[k][f]=v;localStorage.setItem(KEY,JSON.stringify(S));progressNow()}
function selectBox(k,f,opts){let v=(S[k]||{})[f]||'';return `<select onchange="save('${k}','${f}',this.value)"><option value="">请选择</option>${opts.map(x=>`<option ${v===x?'selected':''}>${x}</option>`).join('')}</select>`}
function checks(k){let values=new Set((S[k]||{}).evidence_codes||[]);return CODES.map(x=>`<label class="check"><input type="checkbox" ${values.has(x)?'checked':''} onchange="let a=new Set((S['${k}']||{}).evidence_codes||[]);this.checked?a.add('${x}'):a.delete('${x}');save('${k}','evidence_codes',[...a])">${x}</label>`).join('')}
function card(item,c){let k=id(item.blind_state_id,c.component);if(!c.candidate_present)return `<div class="candidate absent"><h3>${c.component}</h3><span class="tag">CANDIDATE_ABSENT → 自动 OFF</span></div>`;let age=c.candidate_age_sessions==null?'不适用':c.candidate_age_sessions+' sessions ago';let z=S[k]||{};return `<div class="candidate"><h3>${c.component}</h3><span class="tag">exact Rank‑1</span><span class="tag">age=${age}</span><p>${esc(c.candidate_text)}</p><label>独立 subtype 裁定${selectBox(k,'adjudicated_subtype',TYPES[c.component])}</label><label>门 1：applicability / safety${selectBox(k,'applicability_safe',['yes','no','uncertain'])}</label><label>门 2：incremental over R0${selectBox(k,'incremental_over_r0',['yes','no','uncertain'])}</label><b>证据码（至少一个）</b>${checks(k)}<textarea placeholder="写明候选如何产生独特用途，或具体为何不该注入" onchange="save('${k}','notes',this.value)">${esc(z.notes||'')}</textarea></div>`}
function render(){root.innerHTML=DATA.items.map((x,i)=>`<section class="state"><h2>${i+1}/${DATA.items.length} · ${x.blind_state_id}</h2><div class="dialogue">${esc(x.visible_dialogue.map(t=>(t.role==='user'?'用户':'助手')+': '+t.content).join('\n'))}\n用户（当前）: ${esc(x.current_user_text)}</div><div class="grid">${x.components.map(c=>card(x,c)).join('')}</div></section>`).join('');progressNow()}
function ready(z){let uncertain=z.applicability_safe==='uncertain'||z.incremental_over_r0==='uncertain';return z.adjudicated_subtype&&z.applicability_safe&&z.incremental_over_r0&&(z.evidence_codes||[]).length&&(!uncertain||String(z.notes||'').trim())}
function progressNow(){let total=0,done=0;for(const x of DATA.items)for(const c of x.components)if(c.candidate_present){total++;if(ready(S[id(x.blind_state_id,c.component)]||{}))done++}progress.textContent=`已完成 ${done} / ${total} 个有候选裁决（无候选自动 OFF）`}
function derived(z){if(z.applicability_safe==='no'||z.incremental_over_r0==='no')return 'off';if(z.applicability_safe==='yes'&&z.incremental_over_r0==='yes')return 'on';return 'abstain'}
function downloadRows(){let a=annotator.value.trim();if(!a){alert('请填写 annotator_id');return}let missing=[];for(const x of DATA.items)for(const c of x.components)if(c.candidate_present&&!ready(S[id(x.blind_state_id,c.component)]||{}))missing.push(x.blind_state_id+'::'+c.component);if(missing.length){alert('仍有 '+missing.length+' 个裁决未完成；首项：'+missing[0]);return}let rows=DATA.items.map(x=>({protocol:DATA.protocol,blind_state_id:x.blind_state_id,component_decisions:Object.fromEntries(x.components.map(c=>{if(!c.candidate_present)return [c.component,{adjudicated_subtype:'CANDIDATE_ABSENT',applicability_safe:'no',incremental_over_r0:'no',derived_decision:'off',evidence_codes:['CANDIDATE_ABSENT'],notes:''}];let z=S[id(x.blind_state_id,c.component)];return [c.component,{...z,derived_decision:derived(z)}]})),annotator_id:a}));let blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'}),u=URL.createObjectURL(blob),q=document.createElement('a');q.href=u;q.download='h1_v2_candidate_gold_annotations.jsonl';q.click();URL.revokeObjectURL(u)}render();</script></body></html>'''


def _render(packet: dict[str, Any], key: str) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__PACKET__", data).replace("__KEY__", json.dumps(key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_h1_v2_exact_rank1_v10",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_final_candidate_first_v9_h1_v2/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--static-gate",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_v2_pre_human_static_gate_v1/static_gate_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_v2_candidate",
    )
    args = parser.parse_args()

    materialization = read_json(args.candidate_dir / "materialization_report.json")
    realization = read_json(args.candidate_dir / "construction_realization_private.json")
    gate = read_json(args.static_gate)
    rows = [dict(row) for row in iter_jsonl(args.candidate_dir / "candidate_rows_private.jsonl")]
    blueprints = {row["state_id"]: dict(row) for row in iter_jsonl(args.blueprint)}
    candidate_path = args.candidate_dir / "candidate_rows_private.jsonl"
    if not (
        materialization.get("status") == "PASS_COMPLETE"
        and realization.get("status") == "PASS"
        and gate.get("protocol") == STATIC_GATE_PROTOCOL
        and gate.get("status") == "PASS"
        and gate.get("candidate_rows_sha256") == sha256_file(candidate_path)
        and len(rows) == len(blueprints) == 256
    ):
        raise RuntimeError("H1-v2 human packet requires the frozen static gate")

    items: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for row in rows:
        state_id = str(row["state_id"])
        blind_id = "h1v2_state_" + stable_hex(PROTOCOL, state_id, n=24)
        surfaces = [_surface(row["candidate_surfaces"][name]) for name in COMPONENTS]
        for name in COMPONENTS:
            original = row["candidate_surfaces"][name]
            if original["protocol"] != FINAL_CANDIDATE_CONTRACT_PROTOCOL:
                raise RuntimeError("candidate contract mismatch")
            if original["candidate_present"] and original["selected_rank"] != 1:
                raise RuntimeError("candidate is not exact Rank-1")
        items.append(
            {
                "blind_state_id": blind_id,
                "visible_dialogue": row["visible_dialogue"],
                "current_user_text": row["current_user_text"],
                "components": surfaces,
            }
        )
        blueprint = blueprints[state_id]
        bindings.append(
            {
                "blind_state_id": blind_id,
                "state_id": state_id,
                "split": blueprint["split"],
                "construction_action_private_not_gold": blueprint[
                    "private_construction_intent"
                ]["intended_action"],
                "candidate_bindings": {
                    name: {
                        "candidate_id": row["candidate_surfaces"][name]["candidate_id"],
                        "candidate_text_sha256": row["candidate_surfaces"][name]["candidate_text_sha256"],
                    }
                    for name in COMPONENTS
                },
            }
        )

    strata: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for binding in bindings:
        strata[(binding["construction_action_private_not_gold"], binding["split"])].append(binding)
    overlap_ids: set[str] = set()
    for action in sorted({binding["construction_action_private_not_gold"] for binding in bindings}):
        for split, quota in (("FIT", 2), ("FRESH_CONFIRMATION", 1), ("SEALED_INTERNAL_TEST", 1)):
            pool = sorted(
                strata[(action, split)],
                key=lambda row: stable_hex(OVERLAP_PROTOCOL, row["state_id"], n=24),
            )
            if len(pool) < quota:
                raise RuntimeError(f"insufficient overlap stratum: {action}/{split}")
            overlap_ids.update(row["blind_state_id"] for row in pool[:quota])
    if len(overlap_ids) != 64:
        raise RuntimeError("independent overlap must contain 64 states")

    present, absent = _counts(items)
    packet = {
        "protocol": PROTOCOL,
        "items": items,
        "candidate_count": present,
        "absent_count": absent,
        "decision_rule": "ON iff applicability_safe=yes AND incremental_over_r0=yes; any no=>OFF; otherwise ABSTAIN",
        "actual_paired_response_benefit_is_h2_not_h1": True,
        "private_construction_intent_visible": False,
        "split_visible": False,
        "retrieval_scores_visible": False,
        "next_response_or_outcome_visible": False,
    }
    overlap_items = [item for item in items if item["blind_state_id"] in overlap_ids]
    overlap_present, overlap_absent = _counts(overlap_items)
    overlap = {
        **packet,
        "protocol": OVERLAP_PROTOCOL,
        "items": overlap_items,
        "candidate_count": overlap_present,
        "absent_count": overlap_absent,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "human_review_packet.json", packet)
    write_json(args.out_dir / "independent_overlap_packet.json", overlap)
    write_jsonl(args.out_dir / "private_binding.jsonl", bindings)
    (args.out_dir / "human_review.html").write_text(
        _render(packet, "pm15_final_h1_v2_primary"), encoding="utf-8"
    )
    (args.out_dir / "independent_overlap_review.html").write_text(
        _render(overlap, "pm15_final_h1_v2_overlap"), encoding="utf-8"
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_FINAL_H1_V2_REVIEW",
        "states": 256,
        "component_judgments": 1024,
        "human_required_present_candidate_judgments": present,
        "deterministic_absent_off_judgments": absent,
        "independent_overlap_states": 64,
        "independent_overlap_present_candidate_judgments": overlap_present,
        "overlap_split_counts": dict(
            Counter(
                row["split"] for row in bindings if row["blind_state_id"] in overlap_ids
            )
        ),
        "static_gate_sha256": sha256_file(args.static_gate),
        "candidate_rows_sha256": sha256_file(candidate_path),
        "packet_sha256": sha256_file(args.out_dir / "human_review_packet.json"),
        "overlap_packet_sha256": sha256_file(args.out_dir / "independent_overlap_packet.json"),
        "private_binding_sha256": sha256_file(args.out_dir / "private_binding.jsonl"),
        "construction_intent_is_gold": False,
        "actual_response_benefit_used_as_h1_gold": False,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
