#!/usr/bin/env python3
"""Build the single final 96-state H1R packet plus 24-state overlap."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, stable_hex, write_json, write_jsonl
from metacom_pm.v1_5_final_candidate_contract import FINAL_CANDIDATE_CONTRACT_PROTOCOL


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-h1r-semantic-candidate-review-v1"
OVERLAP_PROTOCOL = "pm-v1.5-final-h1r-independent-overlap-v1"
STATIC_GATE_PROTOCOL = "pm-v1.5-final-h1r-pre-human-data-quality-gate-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
SPLITS = ("FIT", "FRESH_CONFIRMATION", "SEALED_INTERNAL_TEST")


def _surface(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "component": str(value["component"]),
        "candidate_present": bool(value["candidate_present"]),
        "candidate_text": value["candidate_text"],
        "candidate_age_sessions": value["candidate_age_sessions"],
    }


_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PM V1.5 H1R 最终语义候选审核</title><style>
body{font-family:system-ui,sans-serif;max-width:1180px;margin:auto;padding:20px;background:#f4f6f8;color:#18212b;line-height:1.5}.note,.state,.candidate{background:white;border:1px solid #d5dce5;border-radius:10px;padding:15px;margin:13px 0}.note{border-left:5px solid #235f9c}.warn{border-left-color:#a45a00}.dialogue{background:#edf2f7;border-radius:7px;padding:11px;white-space:pre-wrap}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:11px}.candidate{margin:0;background:#fbfcfd}select,textarea,input[type=text]{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 9px}textarea{min-height:54px}.sticky{position:sticky;top:0;background:#f4f6f8;padding:8px 0;z-index:2}button{padding:9px 14px}.tag{display:inline-block;background:#e6edf5;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px}label.check{display:block;font-size:13px}h2{margin-bottom:6px}</style></head><body><h1>PM V1.5 H1R：最终 Step 1 语义候选审核</h1>
<div class="note"><b>只判断候选，不评价下一条回复。</b>门1：这条 exact Rank‑1 在当前状态下是否对象/时间正确、目标匹配、且不违反边界；门2：相对不注入资源，它是否提供当前未明说的、功能明确的增量。两门都 yes 才 ON；任一 no 即 OFF；证据不足才 ABSTAIN。</div>
<div class="note warn"><b>不要奖励“同主题”。</b>MP 必须增加未明说的稳定偏好或实际约束；MS 必须让过去会话中的区分/事实服务当前目标；ME 必须是当前可迁移的过去动作/选择及结果或机制；RS 必须增加当前需要且上一轮尚未完成的原子支持动作。卡片写得好、记忆真实、或用户一般需要支持，都不自动等于 ON。</div>
<div class="note"><b>证据码（勾选所有直接成立项）：</b>OWNER_AND_TIME_VALID=对象与时间成立；CURRENT_GOAL_FIT=服务当前目标；SAFE_AND_BOUNDARY_COMPATIBLE=边界/负担相容；SPECIFIC_FUNCTIONAL_INCREMENT=存在明确非冗余用途。否定码分别记录错对象、陈旧冲突、错目标、边界冲突、当前已重复、错 subtype/功能、或过于泛化。普通确定项无需写 notes；仅 uncertain 强制说明。</div>
<div class="sticky"><input id="annotator" type="text" placeholder="annotator_id（导出前必填）" style="max-width:330px"> <button onclick="downloadRows()">导出 JSONL</button> <span id="progress"></span></div><div id="root"></div>
<script>const DATA=__PACKET__;const KEY=__KEY__;let S=JSON.parse(localStorage.getItem(KEY)||'{}');
const TYPES={MP:['MP_PREFERENCE','MP_PROFILE'],MS:['MS_SESSION'],ME:['ME_REUSABLE_OUTCOME','ME_CONTEXT_EVENT','ME_UNRESOLVED_EVENT'],RS:['RS_ATOMIC_MOVE']};
const CODES=['OWNER_AND_TIME_VALID','CURRENT_GOAL_FIT','SAFE_AND_BOUNDARY_COMPATIBLE','SPECIFIC_FUNCTIONAL_INCREMENT','WRONG_OWNER_OR_ENTITY','TIME_STALE_OR_CONFLICTING','WRONG_GOAL_OR_FUNCTION','BOUNDARY_OR_BURDEN_CONFLICT','CURRENTLY_REDUNDANT','WRONG_SUBTYPE_OR_FUNCTION','GENERIC_OR_NO_INCREMENT','AMBIGUOUS_VISIBLE_EVIDENCE'];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function id(b,c){return b+'::'+c}function save(k,f,v){S[k]??={};S[k][f]=v;localStorage.setItem(KEY,JSON.stringify(S));progressNow()}
function selectBox(k,f,opts){let v=(S[k]||{})[f]||'';return `<select onchange="save('${k}','${f}',this.value)"><option value="">请选择</option>${opts.map(x=>`<option ${v===x?'selected':''}>${x}</option>`).join('')}</select>`}
function checks(k){let values=new Set((S[k]||{}).evidence_codes||[]);return CODES.map(x=>`<label class="check"><input type="checkbox" ${values.has(x)?'checked':''} onchange="let a=new Set((S['${k}']||{}).evidence_codes||[]);this.checked?a.add('${x}'):a.delete('${x}');save('${k}','evidence_codes',[...a])">${x}</label>`).join('')}
function card(item,c){let k=id(item.blind_state_id,c.component),z=S[k]||{},age=c.candidate_age_sessions==null?'不适用':c.candidate_age_sessions+' sessions ago';return `<div class="candidate"><h3>${c.component}</h3><span class="tag">exact Rank‑1</span><span class="tag">age=${age}</span><p>${esc(c.candidate_text)}</p><label>独立 subtype 裁定${selectBox(k,'adjudicated_subtype',TYPES[c.component])}</label><label>门1：对象/时间/目标/边界适用${selectBox(k,'applicability_safe',['yes','no','uncertain'])}</label><label>门2：相对 R0 有明确非冗余增量${selectBox(k,'incremental_over_r0',['yes','no','uncertain'])}</label><b>证据码（勾选所有直接成立项）</b>${checks(k)}<textarea placeholder="仅 uncertain 必填；确定项可选" onchange="save('${k}','notes',this.value)">${esc(z.notes||'')}</textarea></div>`}
function render(){root.innerHTML=DATA.items.map((x,i)=>`<section class="state"><h2>${i+1}/${DATA.items.length} · ${x.blind_state_id}</h2><div class="dialogue">${esc(x.visible_dialogue.map(t=>(t.role==='user'?'用户':'助手')+': '+t.content).join('\n'))}\n用户（当前）: ${esc(x.current_user_text)}</div><div class="grid">${x.components.map(c=>card(x,c)).join('')}</div></section>`).join('');progressNow()}
function ready(z){let u=z.applicability_safe==='uncertain'||z.incremental_over_r0==='uncertain';return z.adjudicated_subtype&&z.applicability_safe&&z.incremental_over_r0&&(z.evidence_codes||[]).length&&(!u||String(z.notes||'').trim())}
function progressNow(){let total=DATA.items.length*4,done=0;for(const x of DATA.items)for(const c of x.components)if(ready(S[id(x.blind_state_id,c.component)]||{}))done++;progress.textContent=`已完成 ${done} / ${total} 个裁决`}
function derived(z){if(z.applicability_safe==='no'||z.incremental_over_r0==='no')return 'off';if(z.applicability_safe==='yes'&&z.incremental_over_r0==='yes')return 'on';return 'abstain'}
function downloadRows(){let a=annotator.value.trim();if(!a){alert('请填写 annotator_id');return}let missing=[];for(const x of DATA.items)for(const c of x.components)if(!ready(S[id(x.blind_state_id,c.component)]||{}))missing.push(x.blind_state_id+'::'+c.component);if(missing.length){alert('仍有 '+missing.length+' 个裁决未完成；首项：'+missing[0]);return}let rows=DATA.items.map(x=>({protocol:DATA.protocol,blind_state_id:x.blind_state_id,component_decisions:Object.fromEntries(x.components.map(c=>{let z=S[id(x.blind_state_id,c.component)];return [c.component,{...z,derived_decision:derived(z)}]})),annotator_id:a}));let blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'}),u=URL.createObjectURL(blob),q=document.createElement('a');q.href=u;q.download=DATA.export_filename;q.click();URL.revokeObjectURL(u)}render();</script></body></html>'''


def _render(packet: dict[str, Any], key: str) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__PACKET__", data).replace("__KEY__", json.dumps(key))


def _overlap_ids(bindings: list[dict[str, Any]]) -> set[str]:
    strata: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bindings:
        strata[(row["construction_action_private_not_gold"], row["split"])].append(row)
    selected: set[str] = set()
    actions = sorted({row["construction_action_private_not_gold"] for row in bindings})
    for index, action in enumerate(actions):
        split = SPLITS[index % len(SPLITS)]
        pool = sorted(
            strata[(action, split)],
            key=lambda row: stable_hex(OVERLAP_PROTOCOL, row["state_id"], n=24),
        )
        selected.add(pool[0]["blind_state_id"])
    for split in SPLITS:
        need = 8 - sum(
            row["split"] == split and row["blind_state_id"] in selected
            for row in bindings
        )
        pool = sorted(
            [
                row for row in bindings
                if row["split"] == split and row["blind_state_id"] not in selected
            ],
            key=lambda row: stable_hex(OVERLAP_PROTOCOL, "extra", row["state_id"], n=24),
        )
        selected.update(row["blind_state_id"] for row in pool[:need])
    if len(selected) != 24:
        raise RuntimeError("H1R overlap must contain exactly 24 states")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "outputs/pm_v1_5_final_h1r_exact_rank1_v1")
    parser.add_argument("--blueprint", type=Path, default=ROOT / "data/pm_v1_5_final_h1r_v1/private/construction_blueprint.jsonl")
    parser.add_argument("--static-gate", type=Path, default=ROOT / "outputs/pm_v1_5_final_h1r_pre_human_gate_v1/static_gate_report.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_final_h1r_review_candidate")
    args = parser.parse_args()
    candidate_path = args.candidate_dir / "candidate_rows_private.jsonl"
    rows = [dict(row) for row in iter_jsonl(candidate_path)]
    blueprints = {str(row["state_id"]): dict(row) for row in iter_jsonl(args.blueprint)}
    gate = read_json(args.static_gate)
    materialization = read_json(args.candidate_dir / "materialization_report.json")
    realization = read_json(args.candidate_dir / "construction_realization_private.json")
    if not (
        gate.get("protocol") == STATIC_GATE_PROTOCOL
        and gate.get("status") == "PASS"
        and gate.get("candidate_rows_sha256") == sha256_file(candidate_path)
        and materialization.get("status") == "PASS_COMPLETE"
        and realization.get("status") == "PASS"
        and len(rows) == len(blueprints) == 96
    ):
        raise RuntimeError("H1R packet requires the frozen passing static gate")

    items = []
    bindings = []
    for row in rows:
        state_id = str(row["state_id"])
        blind_id = "h1r_" + stable_hex(PROTOCOL, state_id, n=24)
        surfaces = []
        for component in COMPONENTS:
            surface = row["candidate_surfaces"][component]
            if (
                surface["protocol"] != FINAL_CANDIDATE_CONTRACT_PROTOCOL
                or not surface["candidate_present"]
                or surface["selected_rank"] != 1
            ):
                raise RuntimeError("H1R exact Rank-1 candidate contract mismatch")
            surfaces.append(_surface(surface))
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
                "construction_action_private_not_gold": blueprint["private_construction_intent"]["intended_action"],
                "candidate_bindings": {
                    component: {
                        "candidate_id": row["candidate_surfaces"][component]["candidate_id"],
                        "candidate_text_sha256": row["candidate_surfaces"][component]["candidate_text_sha256"],
                    }
                    for component in COMPONENTS
                },
            }
        )
    items = sorted(items, key=lambda row: stable_hex(PROTOCOL, "order", row["blind_state_id"], n=24))
    overlap_ids = _overlap_ids(bindings)
    overlap_items = sorted(
        [row for row in items if row["blind_state_id"] in overlap_ids],
        key=lambda row: stable_hex(OVERLAP_PROTOCOL, "order", row["blind_state_id"], n=24),
    )
    common = {
        "decision_rule": "ON iff both gates are yes; any no=>OFF; otherwise ABSTAIN",
        "construction_intent_visible": False,
        "split_visible": False,
        "retrieval_scores_visible": False,
        "response_or_outcome_visible": False,
    }
    packet = {"protocol": PROTOCOL, "export_filename": "h1r_primary_annotations.jsonl", "items": items, **common}
    overlap = {"protocol": OVERLAP_PROTOCOL, "export_filename": "h1r_independent_overlap_annotations.jsonl", "items": overlap_items, **common}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "human_review_packet.json", packet)
    write_json(args.out_dir / "independent_overlap_packet.json", overlap)
    write_jsonl(args.out_dir / "private_binding.jsonl", bindings)
    (args.out_dir / "human_review.html").write_text(_render(packet, "pm15_h1r_primary"), encoding="utf-8")
    (args.out_dir / "independent_overlap_review.html").write_text(_render(overlap, "pm15_h1r_overlap"), encoding="utf-8")
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_FINAL_H1R_REVIEW",
        "states": 96,
        "component_judgments": 384,
        "independent_overlap_states": 24,
        "independent_overlap_component_judgments": 96,
        "overlap_split_counts": dict(Counter(row["split"] for row in bindings if row["blind_state_id"] in overlap_ids)),
        "overlap_action_coverage": len({row["construction_action_private_not_gold"] for row in bindings if row["blind_state_id"] in overlap_ids}),
        "static_gate_sha256": sha256_file(args.static_gate),
        "candidate_rows_sha256": sha256_file(candidate_path),
        "packet_sha256": sha256_file(args.out_dir / "human_review_packet.json"),
        "overlap_packet_sha256": sha256_file(args.out_dir / "independent_overlap_packet.json"),
        "private_binding_sha256": sha256_file(args.out_dir / "private_binding.jsonl"),
        "construction_intent_is_gold": False,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
