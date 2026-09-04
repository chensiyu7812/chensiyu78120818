#!/usr/bin/env python3
"""Freeze the one 96-primary + 24-overlap Observation/Eligibility review packet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, stable_hex, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-eligibility-primary-v1"
OVERLAP_PROTOCOL = "pm-v1.5-v3-observation-eligibility-overlap-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")

_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PM V1.5 V3 Observation</title><style>
body{font-family:system-ui,sans-serif;max-width:1080px;margin:auto;padding:20px;background:#f4f6f8;color:#18212b;line-height:1.55}.note,.item{background:#fff;border:1px solid #d5dce5;border-radius:10px;padding:15px;margin:13px 0}.note{border-left:5px solid #235f9c}.warn{border-left-color:#a45a00}.dialogue,.candidate{white-space:pre-wrap;border-radius:7px;padding:11px}.dialogue{background:#edf2f7}.candidate{background:#fff8df;border:1px solid #ead79a}select,textarea,input[type=text]{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 9px}textarea{min-height:55px}.sticky{position:sticky;top:0;background:#f4f6f8;padding:8px 0;z-index:2}button{padding:9px 14px}.tag{display:inline-block;background:#e6edf5;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px}.fixed{background:#eaf6ed;padding:8px;border-radius:6px;margin:4px 0 9px}</style></head><body><h1>PM V1.5 V3：Observation / Eligibility 审核</h1>
<div class="note"><b>本页只判断实际 Rank‑1 候选的四项可观察资格。</b>不评价回复质量，不判断开资源后是否会赢，不生成 Step1 worth_opening 标签。候选相关但当前已经说过，仍应把“具体增量”判 no。</div>
<div class="note warn"><b>四门相互独立。</b>wrong owner 不自动使“候选是否具体”变 no；目标匹配不等于边界允许；资源真实也不等于当前有增量。MP/MS/ME 四门全 yes 才 eligible；RS 的 owner/time 是共享 Bank 的结构真值，不要求人工判断，其余三门全 yes 才 eligible。</div>
<div class="sticky"><input id="annotator" type="text" placeholder="annotator_id（导出前必填）" style="max-width:340px"> <button onclick="downloadRows()">导出 JSONL</button> <span id="progress"></span></div><div id="root"></div>
<script>const DATA=__PACKET__;const KEY=__KEY__;let S=JSON.parse(localStorage.getItem(KEY)||'{}');
const SUB={MP:['MP_PREFERENCE','MP_PROFILE'],MS:['MS_SESSION'],ME:['ME_REUSABLE_OUTCOME','ME_CONTEXT_EVENT','ME_UNRESOLVED_EVENT'],RS:['RS_ATOMIC_MOVE']};
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function save(k,f,v){S[k]??={};S[k][f]=v;localStorage.setItem(KEY,JSON.stringify(S));progressNow()}
function sel(k,f,opts){let v=(S[k]||{})[f]||'';return `<select onchange="save('${k}','${f}',this.value)"><option value="">请选择</option>${opts.map(x=>`<option ${v===x?'selected':''}>${x}</option>`).join('')}</select>`}
function render(){root.innerHTML=DATA.items.map((x,i)=>{let k=x.blind_item_id,z=S[k]||{},age=x.candidate_age_sessions==null?'不适用':x.candidate_age_sessions+' sessions ago';let owner=x.owner_gate_applicable?`<label>正确 owner / entity 且时间仍有效${sel(k,'owner_time_entity_valid',['yes','no','uncertain'])}</label>`:`<div class="fixed">owner/time：structural yes（共享 Strategy Bank 无私人 owner 或时间事实，本项不人工评分）</div>`;return `<section class="item"><h2>${i+1}/${DATA.items.length} · ${x.component}</h2><span class="tag">actual Rank‑1</span><span class="tag">age=${age}</span><div class="dialogue">${esc(x.visible_dialogue.map(t=>(t.role==='user'?'用户':'助手')+': '+t.content).join('\n'))}\n用户（当前）: ${esc(x.current_user_text)}</div><h3>候选</h3><div class="candidate">${esc(x.candidate_text)}</div><label>独立 subtype 裁定${sel(k,'adjudicated_subtype',SUB[x.component])}</label>${owner}<label>服务当前 goal / 所需功能${sel(k,'goal_function_fit',['yes','no','uncertain'])}</label><label>符合明确边界与互动负担${sel(k,'boundary_burden_compatible',['yes','no','uncertain'])}</label><label>提供具体、当前未明说且尚未执行的增量${sel(k,'specific_increment',['yes','no','uncertain'])}</label><textarea placeholder="仅 uncertain 必填；确定项可留空" onchange="save('${k}','notes',this.value)">${esc(z.notes||'')}</textarea></section>`}).join('');progressNow()}
function fields(x){let f=['adjudicated_subtype','goal_function_fit','boundary_burden_compatible','specific_increment'];if(x.owner_gate_applicable)f.push('owner_time_entity_valid');return f}
function ready(x,z){let fs=fields(x),u=fs.slice(1).some(f=>z[f]==='uncertain');return fs.every(f=>z[f])&&(!u||String(z.notes||'').trim())}
function decision(x,z){let vals=[z.goal_function_fit,z.boundary_burden_compatible,z.specific_increment];if(x.owner_gate_applicable)vals.push(z.owner_time_entity_valid);if(vals.some(v=>v==='no'))return 'ineligible';if(vals.every(v=>v==='yes'))return 'eligible';return 'uncertain'}
function progressNow(){let done=DATA.items.filter(x=>ready(x,S[x.blind_item_id]||{})).length;progress.textContent=`已完成 ${done} / ${DATA.items.length}`}
function downloadRows(){let a=annotator.value.trim();if(!a){alert('请填写 annotator_id');return}let miss=DATA.items.filter(x=>!ready(x,S[x.blind_item_id]||{}));if(miss.length){alert('仍有 '+miss.length+' 项未完成；首项：'+miss[0].blind_item_id);return}let rows=DATA.items.map(x=>{let z={...S[x.blind_item_id]};if(!x.owner_gate_applicable)z.owner_time_entity_valid='structural_yes_not_rated';return {protocol:DATA.protocol,blind_item_id:x.blind_item_id,component:x.component,...z,derived_eligibility:decision(x,z),annotator_id:a}});let blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'}),u=URL.createObjectURL(blob),q=document.createElement('a');q.href=u;q.download=DATA.export_filename;q.click();URL.revokeObjectURL(u)}render();</script></body></html>'''


def _render(packet: dict[str, Any], key: str) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__PACKET__", data).replace("__KEY__", json.dumps(key))


def _overlap_ids(bindings: list[dict[str, Any]]) -> set[str]:
    chosen: set[str] = set()
    by_component: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bindings:
        by_component[row["component"]].append(row)
    for component in COMPONENTS:
        fit = [row for row in by_component[component] if row["track"] == "FACTOR_FIT"]
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in fit:
            groups[row["counterfactual_group_id"]].append(row)
        pairs = [members for members in groups.values() if len(members) == 2]
        pairs.sort(key=lambda members: stable_hex(OVERLAP_PROTOCOL, component, members[0]["counterfactual_group_id"], n=24))
        for pair in pairs[:2]:
            chosen.update(row["blind_item_id"] for row in pair)
        confirmation = [row for row in by_component[component] if row["track"] == "ELIGIBILITY_CONFIRMATION"]
        for eligible in (True, False):
            pool = [row for row in confirmation if row["private_composed_eligibility_not_gold"] is eligible]
            pool.sort(key=lambda row: stable_hex(OVERLAP_PROTOCOL, component, eligible, row["blind_item_id"], n=24))
            chosen.add(pool[0]["blind_item_id"])
    return chosen


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_exact_rank1_v1")
    parser.add_argument("--blueprint", type=Path, default=ROOT / "data/pm_v1_5_v3_observation_orthogonal_v1/private/construction_blueprint.jsonl")
    parser.add_argument("--static-gate", type=Path, default=ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_pre_human_gate_v1/static_gate_report.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_v3_observation_review_candidate_v1")
    args = parser.parse_args()
    gate = read_json(args.static_gate)
    candidate_path = args.candidate_dir / "candidate_rows_private.jsonl"
    materialization_path = args.candidate_dir / "materialization_report.json"
    materialization = read_json(materialization_path)
    if not (
        gate.get("status") == "PASS"
        and gate.get("review_packet_allowed") is True
        and gate.get("candidate_rows_sha256") == sha256_file(candidate_path)
        and materialization.get("status") == "PASS"
    ):
        raise RuntimeError("review packet requires the frozen passing pre-human gate")
    candidates = [dict(row) for row in iter_jsonl(candidate_path)]
    blueprint = {str(row["blueprint_row_id"]): dict(row) for row in iter_jsonl(args.blueprint)}
    if not (len(candidates) == len(blueprint) == 96):
        raise RuntimeError("review packet requires exactly 96 states")
    items: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for row in candidates:
        state_id = str(row["state_id"])
        private = blueprint[state_id]
        component = str(row["target_component_private_not_model_input"])
        surface = row["exact_rank1_candidate"]
        blind = "obs_elig_" + stable_hex(PROTOCOL, state_id, component, n=24)
        items.append(
            {
                "blind_item_id": blind,
                "component": component,
                "owner_gate_applicable": component != "RS",
                "visible_dialogue": row["visible_dialogue"],
                "current_user_text": row["current_user_text"],
                "candidate_text": surface["candidate_text"],
                "candidate_age_sessions": surface["candidate_age_sessions"],
            }
        )
        bindings.append(
            {
                "blind_item_id": blind,
                "state_id": state_id,
                "component": component,
                "track": private["track"],
                "counterfactual_group_id": private["counterfactual_group_id"],
                "candidate_id": surface["candidate_id"],
                "candidate_text_sha256": surface["candidate_text_sha256"],
                "private_factor_plan_not_gold": private["private_factor_plan"],
                "private_composed_eligibility_not_gold": private["private_composed_eligibility"],
            }
        )
    items.sort(key=lambda row: stable_hex(PROTOCOL, "order", row["blind_item_id"], n=24))
    overlap_ids = _overlap_ids(bindings)
    overlap_items = [row for row in items if row["blind_item_id"] in overlap_ids]
    overlap_items.sort(key=lambda row: stable_hex(OVERLAP_PROTOCOL, "order", row["blind_item_id"], n=24))
    if len(items) != 96 or len(overlap_items) != 24:
        raise RuntimeError("packet must be exactly 96 primary + 24 overlap")
    common = {
        "decision_rule": "all applicable observable gates yes; RS owner/time is structural yes",
        "is_step1_worth_opening_gold": False,
        "construction_intent_visible": False,
        "track_visible": False,
        "topic_scale_prefix_length_visible_as_metadata": False,
        "retrieval_score_visible": False,
        "response_or_outcome_visible": False,
    }
    primary = {"protocol": PROTOCOL, "export_filename": "observation_eligibility_primary.jsonl", "items": items, **common}
    overlap = {"protocol": OVERLAP_PROTOCOL, "export_filename": "observation_eligibility_overlap.jsonl", "items": overlap_items, **common}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "human_review_packet.json", primary)
    write_json(args.out_dir / "independent_overlap_packet.json", overlap)
    write_jsonl(args.out_dir / "private_binding.jsonl", bindings)
    (args.out_dir / "human_review.html").write_text(_render(primary, "pm15_v3_observation_primary_v1"), encoding="utf-8")
    (args.out_dir / "independent_overlap_review.html").write_text(_render(overlap, "pm15_v3_observation_overlap_v1"), encoding="utf-8")
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_OBSERVATION_ELIGIBILITY_REVIEW",
        "primary_items": len(items),
        "primary_per_component": dict(Counter(row["component"] for row in items)),
        "primary_per_track": dict(Counter(row["track"] for row in bindings)),
        "overlap_items": len(overlap_items),
        "overlap_per_component": dict(Counter(row["component"] for row in overlap_items)),
        "static_gate_sha256": sha256_file(args.static_gate),
        "materialization_report_sha256": sha256_file(materialization_path),
        "candidate_rows_sha256": sha256_file(candidate_path),
        "blueprint_sha256": sha256_file(args.blueprint),
        "primary_packet_sha256": sha256_file(args.out_dir / "human_review_packet.json"),
        "overlap_packet_sha256": sha256_file(args.out_dir / "independent_overlap_packet.json"),
        "private_binding_sha256": sha256_file(args.out_dir / "private_binding.jsonl"),
        "construction_intent_is_gold": False,
        "step1_worth_opening_gold": False,
        "responses_generated": 0,
        "external_lockbox_read": False,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
