#!/usr/bin/env python3
"""Freeze and render the one complete H1 candidate/gold review packet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    iter_jsonl,
    sha256_file,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_final_candidate_contract import (
    FINAL_CANDIDATE_CONTRACT_PROTOCOL,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-final-candidate-gold-human-review-v1"
OVERLAP_PROTOCOL = "pm-v1.5-final-candidate-gold-independent-overlap-v1"


def _public_component(surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "component": str(surface["component"]),
        "candidate_present": bool(surface["candidate_present"]),
        "candidate_text": surface["candidate_text"],
        "candidate_age_sessions": surface["candidate_age_sessions"],
    }


def _render_html(packet: dict[str, Any], *, storage_key: str) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 H1 Candidate/Gold 总包</title><style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:auto;padding:20px;background:#f4f6f8;color:#18212b;line-height:1.5}}
.note,.state,.candidate{{background:white;border:1px solid #d5dce5;border-radius:10px;padding:15px;margin:13px 0}}
.note{{border-left:5px solid #235f9c}}.warn{{border-left-color:#a45a00}}.dialogue{{background:#edf2f7;border-radius:7px;padding:11px;white-space:pre-wrap}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:11px}}.candidate{{margin:0;background:#fbfcfd}}
.absent{{opacity:.72}}select,textarea,input[type=text]{{width:100%;box-sizing:border-box;padding:7px;margin:4px 0 9px}}textarea{{min-height:58px}}
.sticky{{position:sticky;top:0;background:#f4f6f8;padding:8px 0;z-index:2}}button{{padding:9px 14px}}.tag{{display:inline-block;background:#e6edf5;border-radius:999px;padding:2px 8px;margin:2px;font-size:12px}}
label.check{{display:block;font-size:13px}} h2{{margin-bottom:6px}}
</style></head><body><h1>PM V1.5 H1：Candidate / Gold 一次性总审核</h1>
<div class="note"><b>只审核 Step 1。</b>对每个组件，只看当前可见对话与将被实际注入的 exact Rank‑1 candidate。判断它现在是否有安全、非冗余、功能明确的增量用途。不要评价下一条回复质量，也不要猜构造条件、动作或 split。</div>
<div class="note warn"><b>ON：</b>owner/时间正确、当前目标匹配、非冗余、无边界冲突，并有可说明的用途。<b>OFF：</b>无候选、错实体/目标、陈旧冲突、当前已说出、功能不对或边界不允许。<b>ABSTAIN：</b>可见证据不足，不能硬判。ME 只有“过去动作/选择 + 结果或机制”才可 ON；纯背景与未解决事件必须 OFF。</div>
<div class="sticky"><input id="annotator" type="text" placeholder="annotator_id（导出前必填）" style="max-width:310px"> <button onclick="download()">导出完整 JSONL</button> <span id="progress"></span></div><div id="root"></div>
<script>const DATA={data};const KEY={json.dumps(storage_key)};let S=JSON.parse(localStorage.getItem(KEY)||"{{}}");
const E=["CURRENT_GOAL_FIT","INCREMENTAL_USE","OWNER_AND_TIME_VALID","SAFE_ALLOWED_FUNCTION","CANDIDATE_ABSENT","CURRENTLY_REDUNDANT","WRONG_ENTITY_OR_GOAL","STALE_OR_CONFLICTING","BOUNDARY_OR_BURDEN_CONFLICT","WRONG_SUBTYPE_OR_FUNCTION","NO_SAFE_INCREMENTAL_USE","AMBIGUOUS_VISIBLE_EVIDENCE"];
const TYPES={{MP:["MP_PREFERENCE","MP_PROFILE"],MS:["MS_SESSION"],ME:["ME_REUSABLE_OUTCOME","ME_CONTEXT_EVENT","ME_UNRESOLVED_EVENT"],RS:["RS_ATOMIC_MOVE"]}};
const esc=x=>String(x??"").replace(/[&<>\"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'\"':"&quot;","'":"&#39;"}}[c]));
function id(b,c){{return b+"::"+c}}function save(k,f,v){{S[k]??={{}};S[k][f]=v;localStorage.setItem(KEY,JSON.stringify(S));prog();}}
function sel(k,f,opts){{let v=(S[k]||{{}})[f]||"";return `<select onchange="save('${{k}}','${{f}}',this.value)"><option value="">请选择</option>${{opts.map(x=>`<option ${{v===x?'selected':''}}>${{x}}</option>`).join('')}}</select>`}}
function checks(k){{let v=new Set((S[k]||{{}}).evidence_codes||[]);return E.map(x=>`<label class="check"><input type="checkbox" ${{v.has(x)?'checked':''}} onchange="let a=new Set((S['${{k}}']||{{}}).evidence_codes||[]);this.checked?a.add('${{x}}'):a.delete('${{x}}');save('${{k}}','evidence_codes',[...a])">${{x}}</label>`).join('')}}
function comp(b,c){{let k=id(b.blind_state_id,c.component);if(!c.candidate_present)return `<div class="candidate absent"><h3>${{c.component}}</h3><span class="tag">CANDIDATE_ABSENT</span><p>正式检索器没有返回候选；合同确定为 OFF。</p></div>`;let age=c.candidate_age_sessions==null?'不适用':c.candidate_age_sessions+' sessions ago';return `<div class="candidate"><h3>${{c.component}}</h3><span class="tag">exact Rank‑1</span><span class="tag">age=${{age}}</span><p>${{esc(c.candidate_text)}}</p><label>独立裁定 subtype${{sel(k,'adjudicated_subtype',TYPES[c.component])}}</label><label>机会决策${{sel(k,'decision',['on','off','abstain'])}}</label><b>证据码（至少一个）</b>${{checks(k)}}<textarea placeholder="简短理由；指出候选怎样做功，或为什么不该用" onchange="save('${{k}}','notes',this.value)">${{esc((S[k]||{{}}).notes||'')}}</textarea></div>`}}
function render(){{root.innerHTML=DATA.items.map((x,i)=>`<section class="state"><h2>${{i+1}}/${{DATA.items.length}} · ${{x.blind_state_id}}</h2><div class="dialogue">${{esc(x.visible_dialogue.map(t=>(t.role==='user'?'用户':'助手')+': '+t.content).join('\\n'))}}\\n用户（当前）: ${{esc(x.current_user_text)}}</div><div class="grid">${{x.components.map(c=>comp(x,c)).join('')}}</div></section>`).join('');prog()}}
function prog(){{let total=0,done=0;for(const x of DATA.items)for(const c of x.components)if(c.candidate_present){{total++;let z=S[id(x.blind_state_id,c.component)]||{{}};if(z.decision&&z.adjudicated_subtype&&(z.evidence_codes||[]).length)done++}}progress.textContent=`已完成 ${{done}} / ${{total}} 个有候选裁决（无候选自动 OFF）`;}}
function download(){{let a=document.getElementById('annotator').value.trim();if(!a){{alert('请填写 annotator_id');return}}let missing=[];for(const x of DATA.items)for(const c of x.components)if(c.candidate_present){{let z=S[id(x.blind_state_id,c.component)]||{{}};if(!z.decision||!z.adjudicated_subtype||!(z.evidence_codes||[]).length)missing.push(x.blind_state_id+'::'+c.component)}}if(missing.length){{alert('仍有 '+missing.length+' 个有候选裁决未完成；首项：'+missing[0]);return}}let rows=DATA.items.map(x=>({{protocol:DATA.protocol,blind_state_id:x.blind_state_id,component_decisions:Object.fromEntries(x.components.map(c=>{{if(!c.candidate_present)return [c.component,{{adjudicated_subtype:'CANDIDATE_ABSENT',decision:'off',evidence_codes:['CANDIDATE_ABSENT'],notes:''}}];let z=S[id(x.blind_state_id,c.component)]||{{}};return [c.component,z]}})),annotator_id:a}}));let blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{{type:'application/jsonl'}}),u=URL.createObjectURL(blob),q=document.createElement('a');q.href=u;q.download='h1_candidate_gold_annotations.jsonl';q.click();URL.revokeObjectURL(u)}}render();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_p2_zero_api_mechanical_rank1_v9_age_bound",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_final_candidate_first_v8/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_final_h1_candidate_gold_v1_candidate",
    )
    args = parser.parse_args()
    report = json.loads(
        (args.candidate_dir / "materialization_report.json").read_text(encoding="utf-8")
    )
    realization = json.loads(
        (args.candidate_dir / "construction_realization_private.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_rows = [
        dict(row) for row in iter_jsonl(args.candidate_dir / "candidate_rows_private.jsonl")
    ]
    blueprints = {str(row["state_id"]): dict(row) for row in iter_jsonl(args.blueprint)}
    if not (
        report.get("status") == "PASS_COMPLETE"
        and report.get("states") == 256
        and report.get("exact_rank1_binding_rate") == 1.0
        and realization.get("status") == "PASS"
        and realization.get("mechanically_checkable_rate") == 1.0
        and len(candidate_rows) == len(blueprints) == 256
    ):
        raise RuntimeError("H1 requires a complete mechanically realized candidate pool")

    public_items = []
    private_bindings = []
    for row in candidate_rows:
        state_id = str(row["state_id"])
        blueprint = blueprints[state_id]
        blind_id = "h1_state_" + stable_hex(PROTOCOL, state_id, n=24)
        surfaces = [
            _public_component(dict(row["candidate_surfaces"][component]))
            for component in ("MP", "MS", "ME", "RS")
        ]
        for component, surface in zip(("MP", "MS", "ME", "RS"), surfaces, strict=True):
            original = row["candidate_surfaces"][component]
            if original["protocol"] != FINAL_CANDIDATE_CONTRACT_PROTOCOL:
                raise RuntimeError("H1 candidate protocol mismatch")
            if surface["candidate_present"] and original["selected_rank"] != 1:
                raise RuntimeError("H1 surface is not exact rank-1")
        public_items.append(
            {
                "blind_state_id": blind_id,
                "visible_dialogue": row["visible_dialogue"],
                "current_user_text": row["current_user_text"],
                "components": surfaces,
            }
        )
        private_bindings.append(
            {
                "blind_state_id": blind_id,
                "state_id": state_id,
                "split": blueprint["split"],
                "construction_action_private_not_gold": blueprint[
                    "private_construction_intent"
                ]["intended_action"],
                "candidate_bindings": {
                    component: {
                        "candidate_id": row["candidate_surfaces"][component]["candidate_id"],
                        "candidate_text_sha256": row["candidate_surfaces"][component][
                            "candidate_text_sha256"
                        ],
                    }
                    for component in ("MP", "MS", "ME", "RS")
                },
            }
        )

    bindings_by_action_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for binding in private_bindings:
        bindings_by_action_split[(binding["construction_action_private_not_gold"], binding["split"])].append(binding)
    overlap_ids: set[str] = set()
    for action in sorted({row["construction_action_private_not_gold"] for row in private_bindings}):
        for split, quota in (("FIT", 2), ("FRESH_CONFIRMATION", 1), ("SEALED_INTERNAL_TEST", 1)):
            pool = sorted(
                bindings_by_action_split[(action, split)],
                key=lambda row: stable_hex(OVERLAP_PROTOCOL, row["state_id"], n=24),
            )
            if len(pool) < quota:
                raise RuntimeError(f"insufficient overlap stratum: {action}/{split}")
            overlap_ids.update(row["blind_state_id"] for row in pool[:quota])
    if len(overlap_ids) != 64:
        raise RuntimeError("independent overlap must contain exactly 64 states")

    packet = {
        "protocol": PROTOCOL,
        "items": public_items,
        "candidate_count": sum(
            int(component["candidate_present"])
            for item in public_items
            for component in item["components"]
        ),
        "absent_count": sum(
            int(not component["candidate_present"])
            for item in public_items
            for component in item["components"]
        ),
        "private_construction_intent_visible": False,
        "split_visible": False,
        "retrieval_scores_visible": False,
        "next_response_or_outcome_visible": False,
    }
    overlap_packet = {
        **packet,
        "protocol": OVERLAP_PROTOCOL,
        "items": [item for item in public_items if item["blind_state_id"] in overlap_ids],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "human_review_packet.json", packet)
    write_json(args.out_dir / "independent_overlap_packet.json", overlap_packet)
    write_jsonl(args.out_dir / "private_binding.jsonl", private_bindings)
    (args.out_dir / "human_review.html").write_text(
        _render_html(packet, storage_key="pm15_final_h1_candidate_gold_primary_v1"),
        encoding="utf-8",
    )
    (args.out_dir / "independent_overlap_review.html").write_text(
        _render_html(
            overlap_packet,
            storage_key="pm15_final_h1_candidate_gold_overlap_v1",
        ),
        encoding="utf-8",
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_COMPLETE_H1_REVIEW",
        "states": 256,
        "component_judgments": 1024,
        "human_required_present_candidate_judgments": packet["candidate_count"],
        "deterministic_absent_off_judgments": packet["absent_count"],
        "independent_overlap_states": 64,
        "overlap_split_counts": dict(
            Counter(
                binding["split"]
                for binding in private_bindings
                if binding["blind_state_id"] in overlap_ids
            )
        ),
        "candidate_rows_sha256": sha256_file(
            args.candidate_dir / "candidate_rows_private.jsonl"
        ),
        "model_feature_rows_sha256": sha256_file(
            args.candidate_dir / "model_feature_rows.jsonl"
        ),
        "blueprint_sha256": sha256_file(args.blueprint),
        "packet_sha256": sha256_file(args.out_dir / "human_review_packet.json"),
        "overlap_packet_sha256": sha256_file(
            args.out_dir / "independent_overlap_packet.json"
        ),
        "private_binding_sha256": sha256_file(args.out_dir / "private_binding.jsonl"),
        "construction_intent_is_gold": False,
        "construction_intent_visible_to_reviewer": False,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
