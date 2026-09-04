#!/usr/bin/env python3
"""Prepare the single V3 H-Eligibility packet and its frozen 25% overlap."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, stable_hex, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v3-h-eligibility-primary-v1"
OVERLAP_PROTOCOL = "pm-v1.5-v3-h-eligibility-overlap-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")

_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PM V1.5 V3 H-Eligibility</title><style>
body{font-family:system-ui,sans-serif;max-width:1050px;margin:auto;padding:20px;background:#f4f6f8;color:#18212b;line-height:1.5}.note,.item{background:#fff;border:1px solid #d5dce5;border-radius:10px;padding:15px;margin:13px 0}.note{border-left:5px solid #235f9c}.warn{border-left-color:#a45a00}.dialogue,.candidate{white-space:pre-wrap;border-radius:7px;padding:11px}.dialogue{background:#edf2f7}.candidate{background:#fff8df;border:1px solid #ead79a}select,textarea,input[type=text]{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 9px}textarea{min-height:55px}.sticky{position:sticky;top:0;background:#f4f6f8;padding:8px 0;z-index:2}button{padding:9px 14px}.tag{display:inline-block;background:#e6edf5;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px}label.check{display:block;font-size:13px}</style></head><body><h1>PM V1.5 V3：H‑Eligibility 候选资格审核</h1>
<div class="note"><b>本页只审核 exact Rank‑1 候选是否具备注入资格。</b>不看、不猜下一条回复，不判断资源能否提升质量，也不训练“哪次 A/B 会赢”。四个条件都 yes 才 eligible；任一 no 即 ineligible；证据不足才 uncertain。</div>
<div class="note warn"><b>不要因“同主题”就通过。</b>候选必须属于正确的人和时间，服务当前目标与功能，符合边界/负担，并且提供当前对话尚未明说的具体增量。ME 还必须独立裁定是可复用结果、背景事件还是未解决事件。</div>
<div class="sticky"><input id="annotator" type="text" placeholder="annotator_id（导出前必填）" style="max-width:340px"> <button onclick="downloadRows()">导出 JSONL</button> <span id="progress"></span></div><div id="root"></div>
<script>const DATA=__PACKET__;const KEY=__KEY__;let S=JSON.parse(localStorage.getItem(KEY)||'{}');
const SUB={MP:['MP_PREFERENCE','MP_PROFILE'],MS:['MS_SESSION'],ME:['ME_REUSABLE_OUTCOME','ME_CONTEXT_EVENT','ME_UNRESOLVED_EVENT'],RS:['RS_ATOMIC_MOVE']};
const CODES=['OWNER_VALID','TIME_VALID','CURRENT_GOAL_FIT','FUNCTION_FIT','BOUNDARY_BURDEN_COMPATIBLE','SPECIFIC_INCREMENT','WRONG_OWNER_OR_ENTITY','TIME_STALE_OR_CONFLICTING','WRONG_GOAL_OR_FUNCTION','BOUNDARY_OR_BURDEN_CONFLICT','CURRENTLY_REDUNDANT','GENERIC_OR_NO_INCREMENT','WRONG_SUBTYPE_OR_FUNCTION','AMBIGUOUS_VISIBLE_EVIDENCE'];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function save(k,f,v){S[k]??={};S[k][f]=v;localStorage.setItem(KEY,JSON.stringify(S));progressNow()}
function sel(k,f,opts){let v=(S[k]||{})[f]||'';return `<select onchange="save('${k}','${f}',this.value)"><option value="">请选择</option>${opts.map(x=>`<option ${v===x?'selected':''}>${x}</option>`).join('')}</select>`}
function checks(k){let v=new Set((S[k]||{}).evidence_codes||[]);return CODES.map(x=>`<label class="check"><input type="checkbox" ${v.has(x)?'checked':''} onchange="let a=new Set((S['${k}']||{}).evidence_codes||[]);this.checked?a.add('${x}'):a.delete('${x}');save('${k}','evidence_codes',[...a])">${x}</label>`).join('')}
function render(){root.innerHTML=DATA.items.map((x,i)=>{let k=x.blind_item_id,z=S[k]||{},age=x.candidate_age_sessions==null?'不适用':x.candidate_age_sessions+' sessions ago';return `<section class="item"><h2>${i+1}/${DATA.items.length} · ${x.component}</h2><span class="tag">exact Rank‑1</span><span class="tag">age=${age}</span><div class="dialogue">${esc(x.visible_dialogue.map(t=>(t.role==='user'?'用户':'助手')+': '+t.content).join('\n'))}\n用户（当前）: ${esc(x.current_user_text)}</div><h3>候选</h3><div class="candidate">${esc(x.candidate_text)}</div><label>独立 subtype 裁定${sel(k,'adjudicated_subtype',SUB[x.component])}</label><label>正确 owner 且时间仍有效${sel(k,'owner_time_valid',['yes','no','uncertain'])}</label><label>服务当前 goal / 所需功能${sel(k,'goal_function_fit',['yes','no','uncertain'])}</label><label>符合明确边界与互动负担${sel(k,'boundary_burden_fit',['yes','no','uncertain'])}</label><label>提供具体、当前未明说的增量${sel(k,'specific_nonredundant_increment',['yes','no','uncertain'])}</label><b>证据码（勾选所有直接成立项）</b>${checks(k)}<textarea placeholder="仅 uncertain 必填；确定项可选" onchange="save('${k}','notes',this.value)">${esc(z.notes||'')}</textarea></section>`}).join('');progressNow()}
function ready(z){let fs=['adjudicated_subtype','owner_time_valid','goal_function_fit','boundary_burden_fit','specific_nonredundant_increment'];let u=fs.slice(1).some(f=>z[f]==='uncertain');return fs.every(f=>z[f])&&(z.evidence_codes||[]).length&&(!u||String(z.notes||'').trim())}
function decision(z){let fs=['owner_time_valid','goal_function_fit','boundary_burden_fit','specific_nonredundant_increment'];if(fs.some(f=>z[f]==='no'))return 'ineligible';if(fs.every(f=>z[f]==='yes'))return 'eligible';return 'uncertain'}
function progressNow(){let done=DATA.items.filter(x=>ready(S[x.blind_item_id]||{})).length;progress.textContent=`已完成 ${done} / ${DATA.items.length}`}
function downloadRows(){let a=annotator.value.trim();if(!a){alert('请填写 annotator_id');return}let miss=DATA.items.filter(x=>!ready(S[x.blind_item_id]||{}));if(miss.length){alert('仍有 '+miss.length+' 项未完成；首项：'+miss[0].blind_item_id);return}let rows=DATA.items.map(x=>({protocol:DATA.protocol,blind_item_id:x.blind_item_id,component:x.component,...S[x.blind_item_id],derived_eligibility:decision(S[x.blind_item_id]),annotator_id:a}));let blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'}),u=URL.createObjectURL(blob),q=document.createElement('a');q.href=u;q.download=DATA.export_filename;q.click();URL.revokeObjectURL(u)}render();</script></body></html>'''


def _render(packet: dict[str, Any], key: str) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    return _HTML.replace("__PACKET__", data).replace("__KEY__", json.dumps(key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v1",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_review_candidate",
    )
    args = parser.parse_args()
    candidate_path = args.candidate_dir / "candidate_rows_private.jsonl"
    report = read_json(args.candidate_dir / "materialization_report.json")
    rows = [dict(row) for row in iter_jsonl(candidate_path)]
    blueprint = {
        str(row["blueprint_row_id"]): dict(row) for row in iter_jsonl(args.blueprint)
    }
    if not (
        report.get("status") == "PASS"
        and report.get("candidate_rows_sha256") == sha256_file(candidate_path)
        and len(rows) == len(blueprint) == 640
    ):
        raise RuntimeError("H-Eligibility requires the frozen passing V3 P2 materialization")
    selected = [row for row in rows if row["track_private_not_model_input"] == "ELIGIBILITY_AUDIT"]
    if len(selected) != 128:
        raise RuntimeError("H-Eligibility requires exactly 128 audit rows")
    items: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for row in selected:
        state_id = str(row["state_id"])
        component = str(row["target_component_private_not_model_input"])
        surface = row["exact_rank1_candidate"]
        if not surface["candidate_present"] or surface["selected_rank"] != 1:
            raise RuntimeError("H-Eligibility cannot expose an absent/non-rank1 candidate")
        blind = "h_elig_" + stable_hex(PROTOCOL, state_id, component, n=24)
        items.append(
            {
                "blind_item_id": blind,
                "component": component,
                "visible_dialogue": row["visible_dialogue"],
                "current_user_text": row["current_user_text"],
                "candidate_text": surface["candidate_text"],
                "candidate_age_sessions": surface["candidate_age_sessions"],
            }
        )
        construction = blueprint[state_id]
        bindings.append(
            {
                "blind_item_id": blind,
                "state_id": state_id,
                "component": component,
                "candidate_id": surface["candidate_id"],
                "candidate_text_sha256": surface["candidate_text_sha256"],
                "private_coverage_intent_not_gold": construction["private_eligibility_intent"],
            }
        )
    items.sort(key=lambda row: stable_hex(PROTOCOL, "order", row["blind_item_id"], n=24))
    overlap_ids: set[str] = set()
    for component in COMPONENTS:
        for intent in ("ELIGIBLE", "INELIGIBLE"):
            pool = [
                row
                for row in bindings
                if row["component"] == component
                and row["private_coverage_intent_not_gold"] == intent
            ]
            pool.sort(key=lambda row: stable_hex(OVERLAP_PROTOCOL, row["blind_item_id"], n=24))
            overlap_ids.update(row["blind_item_id"] for row in pool[:4])
    overlap_items = [row for row in items if row["blind_item_id"] in overlap_ids]
    overlap_items.sort(key=lambda row: stable_hex(OVERLAP_PROTOCOL, "order", row["blind_item_id"], n=24))
    if len(overlap_items) != 32:
        raise RuntimeError("H-Eligibility overlap must be exactly 32 items")
    common = {
        "decision_rule": "eligible iff all four observable gates are yes",
        "construction_intent_visible": False,
        "split_visible": False,
        "retrieval_score_visible": False,
        "response_or_outcome_visible": False,
    }
    packet = {"protocol": PROTOCOL, "export_filename": "h_eligibility_primary.jsonl", "items": items, **common}
    overlap = {"protocol": OVERLAP_PROTOCOL, "export_filename": "h_eligibility_overlap.jsonl", "items": overlap_items, **common}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "human_review_packet.json", packet)
    write_json(args.out_dir / "independent_overlap_packet.json", overlap)
    write_jsonl(args.out_dir / "private_binding.jsonl", bindings)
    (args.out_dir / "human_review.html").write_text(_render(packet, "pm15_v3_h_eligibility_primary"), encoding="utf-8")
    (args.out_dir / "independent_overlap_review.html").write_text(_render(overlap, "pm15_v3_h_eligibility_overlap"), encoding="utf-8")
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_H_ELIGIBILITY",
        "items": len(items),
        "per_component": dict(Counter(row["component"] for row in items)),
        "overlap_items": len(overlap_items),
        "overlap_per_component": dict(Counter(row["component"] for row in overlap_items)),
        "materialization_report_sha256": sha256_file(args.candidate_dir / "materialization_report.json"),
        "candidate_rows_sha256": sha256_file(candidate_path),
        "primary_packet_sha256": sha256_file(args.out_dir / "human_review_packet.json"),
        "overlap_packet_sha256": sha256_file(args.out_dir / "independent_overlap_packet.json"),
        "private_binding_sha256": sha256_file(args.out_dir / "private_binding.jsonl"),
        "construction_intent_is_gold": False,
        "responses_generated": 0,
        "external_lockbox_read": False,
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
