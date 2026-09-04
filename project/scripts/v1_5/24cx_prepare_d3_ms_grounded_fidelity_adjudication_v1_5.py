#!/usr/bin/env python3
"""Prepare a five-item grounding-aware adjudication for D3 MS quality.

The original quality reviewer correctly followed a visible-dialogue-only
instrument.  That instrument cannot determine whether a statement grounded in
authorized private memory is supported.  This packet selects all and only the
five rows whose original decisive criterion was visible-context fidelity,
shows the union of verified prior memory available across the two arms, and
keeps arm/component identity and the original decision hidden.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from metacom_pm.contracts import MemoryBackendRecord, RuntimeState
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
PROTOCOL = "pm-v1.5-d3-ms-grounded-fidelity-adjudication-v1"
SOURCE_PROTOCOL = "pm-v1.5-d3-ms-replacement-human-quality-blind-v1"
STATUS = "READY_FOR_FIVE_ITEM_GROUNDING_AWARE_ADJUDICATION"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 MS 证据感知补充裁决</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:18px;background:#f4f6f8;color:#18212b}
.top{position:sticky;top:0;z-index:2;background:#fff;border:1px solid #ccd5df;border-radius:10px;padding:12px;margin-bottom:16px}
.item{background:#fff;border:1px solid #ccd5df;border-radius:10px;padding:18px;margin:16px 0}
.dialogue,.evidence,.response{white-space:pre-wrap;line-height:1.55;border-radius:8px;padding:12px}
.dialogue{background:#f8fafc}.evidence{background:#fff7dc;border:1px solid #e0c66b;margin-top:10px}.response{background:#f7f4ff;min-height:82px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
select,textarea,input{font:inherit;padding:8px;margin:5px 8px 5px 0}
textarea{width:96%;min-height:70px}.hint{color:#52606d;font-size:.94rem;line-height:1.45}
button{padding:9px 14px;margin-right:8px}.done{border-color:#3a8f5b}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><b>PM V1.5 MS 证据感知补充裁决</b><span id="progress"></span>
<button onclick="downloadRows()">导出 JSONL</button>
<button onclick="clearAll()">清空本页缓存</button>
<div class="hint">本页只有 5 条，且是原评审中全部以“可见语境忠实度”决胜的条目。黄色区域是已经核验为同一用户、严格早于当前会话的真实历史证据；两条回复都用同一证据集合判断。不要因为提到历史就奖励，也不要把证据明确支持的事实判成无依据。继续只选足以改变实际采用决定的 A/B；轻微差异选 tie。</div></div>
<div id="root"></div>
<script>
const DATA=__DATA__;
const KEY="pm_v1_5_d3_ms_grounded_fidelity_adjudication_v1";
const CRITERIA={grounded_context_fidelity:"证据感知语境忠实度",request_and_dialogue_fit:"请求与对话适配",emotional_understanding:"情绪理解与回应",immediate_helpfulness:"当下实际帮助",clarity_naturalness:"清晰自然",materially_equivalent:"实质等价",uncertain:"无法可靠判断"};
let state=JSON.parse(localStorage.getItem(KEY)||"{}");
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function ensure(id){if(!state[id])state[id]={quality_preference:"",decisive_criterion:"",quality_notes:"",annotator_id:""};}
function setv(id,k,v){ensure(id);state[id][k]=v;localStorage.setItem(KEY,JSON.stringify(state));render();}
function dialogue(item){return item.recent_dialogue.map(x=>(x.role==="user"?"用户":"支持者")+": "+x.content).join("\n")+"\n用户: "+item.current_user_text;}
function evidence(item){return item.verified_prior_user_context.map((x,i)=>`${i+1}. ${x}`).join("\n");}
function rows(){return DATA.items.map(item=>{ensure(item.blind_item_id);return {protocol:DATA.manifest.protocol,blind_item_id:item.blind_item_id,...state[item.blind_item_id]};});}
function progress(){const r=rows(),n=r.filter(x=>x.quality_preference&&x.decisive_criterion&&x.annotator_id).length;document.getElementById("progress").textContent=`　已完成 ${n} / ${r.length}`;}
function render(){document.getElementById("root").innerHTML=DATA.items.map((item,i)=>{ensure(item.blind_item_id);const s=state[item.blind_item_id],done=s.quality_preference&&s.decisive_criterion&&s.annotator_id;return `<section class="item ${done?"done":""}"><h2>${i+1} / ${DATA.items.length}</h2><div class="dialogue">${esc(dialogue(item))}</div><h3>已核验的同一用户既往证据</h3><div class="evidence">${esc(evidence(item))}</div><div class="grid"><div><h3>回复 A</h3><div class="response">${esc(item.response_a)}</div></div><div><h3>回复 B</h3><div class="response">${esc(item.response_b)}</div></div></div><h3>在已核验证据下，哪个回复实质更好？</h3><select onchange="setv('${item.blind_item_id}','quality_preference',this.value)"><option value="">请选择</option>${["A","B","tie","uncertain"].map(v=>`<option value="${v}" ${s.quality_preference===v?"selected":""}>${v}</option>`).join("")}</select><select onchange="setv('${item.blind_item_id}','decisive_criterion',this.value)"><option value="">请选择决定性标准</option>${Object.entries(CRITERIA).map(([k,v])=>`<option value="${k}" ${s.decisive_criterion===k?"selected":""}>${esc(v)}</option>`).join("")}</select><textarea placeholder="请说明已核验证据如何改变或不改变判断" onchange="setv('${item.blind_item_id}','quality_notes',this.value)">${esc(s.quality_notes)}</textarea><input placeholder="annotator_id" value="${esc(s.annotator_id)}" onchange="setv('${item.blind_item_id}','annotator_id',this.value)"></section>`;}).join("");progress();}
function downloadRows(){const r=rows(),missing=r.filter(x=>!x.quality_preference||!x.decisive_criterion||!x.annotator_id);if(missing.length&&!confirm(`还有 ${missing.length} 条未完成，仍然导出吗？`))return;const b=new Blob([r.map(x=>JSON.stringify(x)).join("\n")+"\n"],{type:"application/jsonl"}),a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="d3_ms_grounded_fidelity_adjudication.jsonl";a.click();URL.revokeObjectURL(a.href);}
function clearAll(){if(confirm("确定清空本页全部标注缓存？")){localStorage.removeItem(KEY);state={};render();}}
render();
</script></body></html>"""


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def build(
    *,
    source_annotations_path: Path,
    blind_dir: Path,
    blueprint_dir: Path,
    execution_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    annotations = _rows(source_annotations_path)
    public_by_id = {
        str(row["blind_item_id"]): row
        for row in _rows(blind_dir / "human_blind_packet.jsonl")
    }
    private_by_id = {
        str(row["blind_item_id"]): row
        for row in _rows(blind_dir / "private_blind_key.jsonl")
    }
    if (
        len(annotations) != 40
        or len({str(row["blind_item_id"]) for row in annotations}) != 40
        or {str(row["blind_item_id"]) for row in annotations}
        != set(public_by_id)
        or set(public_by_id) != set(private_by_id)
        or any(row.get("protocol") != SOURCE_PROTOCOL for row in annotations)
    ):
        raise RuntimeError("source MS quality annotations are incomplete or mismatched")
    selected = [
        row
        for row in annotations
        if row.get("decisive_criterion") == "visible_context_fidelity"
    ]
    if len(selected) != 5:
        raise RuntimeError(
            f"expected all and only five fidelity rows, found {len(selected)}"
        )

    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "runtime_states.jsonl")
        )
    }
    backends = {
        record.card_id: {item.memory_id: item for item in record.items}
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "memory_backend.jsonl")
        )
    }
    outcomes_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _rows(execution_dir / "generation_outcomes.jsonl"):
        outcomes_by_pair[str(row["pair_id"])][str(row["arm"])] = row

    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for source_annotation in selected:
        old_id = str(source_annotation["blind_item_id"])
        old_public = public_by_id[old_id]
        old_key = private_by_id[old_id]
        pair_id = str(old_key["pair_id"])
        arms = outcomes_by_pair[pair_id]
        if set(arms) != {"control", "treatment"}:
            raise RuntimeError(f"incomplete pair: {pair_id}")
        state = states[str(old_key["state_id"])]
        catalog = backends[state.card_id]
        evidence_ids = sorted(
            {
                str(memory_id)
                for arm in arms.values()
                for memory_id in arm["selected_memory_ids"]
            }
        )
        evidence_items = [catalog[memory_id] for memory_id in evidence_ids]
        if not evidence_items or any(
            int(item.created_session) >= int(state.session_index)
            for item in evidence_items
        ):
            raise RuntimeError(f"invalid authorized evidence: {old_id}")

        new_id = "d3ms_grounded_" + stable_hex(PROTOCOL, old_id, n=24)
        swap = int(stable_hex(PROTOCOL, "position", old_id, n=8), 16) % 2 == 1
        new_a_old_role, new_b_old_role = (
            ("b", "a") if swap else ("a", "b")
        )
        response_by_old_position = {
            "a": str(old_public["response_a"]),
            "b": str(old_public["response_b"]),
        }
        old_role_by_position = {
            "a": str(old_key["a_role"]),
            "b": str(old_key["b_role"]),
        }
        public.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": new_id,
                "recent_dialogue": old_public["recent_dialogue"],
                "current_user_text": old_public["current_user_text"],
                "verified_prior_user_context": [
                    item.text for item in evidence_items
                ],
                "response_a": response_by_old_position[new_a_old_role],
                "response_b": response_by_old_position[new_b_old_role],
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": new_id,
                "source_blind_item_id": old_id,
                "pair_id": pair_id,
                "pair_role": old_key["pair_role"],
                "contrast_slot_id": old_key["contrast_slot_id"],
                "state_id": state.state_id,
                "user_id": state.user_id,
                "new_a_role": old_role_by_position[new_a_old_role],
                "new_b_role": old_role_by_position[new_b_old_role],
                "authorized_memory_ids": evidence_ids,
                "authorized_evidence_sha256": sha256_text(
                    canonical_json([item.text for item in evidence_items])
                ),
                "selection_rule": (
                    "all_and_only_source_rows_with_"
                    "decisive_criterion_visible_context_fidelity"
                ),
                "source_annotation_preference_hidden": True,
                "effect_label": "UNKNOWN_PENDING_GROUNDED_ADJUDICATION",
            }
        )

    order = sorted(
        range(len(public)),
        key=lambda index: stable_hex(
            PROTOCOL, "display", public[index]["blind_item_id"], n=32
        ),
    )
    public = [public[index] for index in order]
    private = [private[index] for index in order]
    public_keys = set().union(*(row.keys() for row in public))
    forbidden_public = {
        "source_blind_item_id",
        "pair_id",
        "pair_role",
        "contrast_slot_id",
        "state_id",
        "user_id",
        "new_a_role",
        "new_b_role",
        "authorized_memory_ids",
        "selection_rule",
        "source_annotation_preference",
        "component",
    }
    checks = {
        "source_annotations_40_complete": len(annotations) == 40,
        "all_and_only_five_fidelity_rows": len(public) == len(private) == 5,
        "new_blind_ids_unique": len({row["blind_item_id"] for row in public})
        == 5,
        "original_decisions_hidden": all(
            row["source_annotation_preference_hidden"] for row in private
        ),
        "authorized_evidence_nonempty": all(
            row["verified_prior_user_context"] for row in public
        ),
        "private_fields_absent_from_public": not bool(
            public_keys & forbidden_public
        ),
        "responses_nonempty_and_distinct": all(
            row["response_a"].strip()
            and row["response_b"].strip()
            and row["response_a"] != row["response_b"]
            for row in public
        ),
        "no_effect_label_created": all(
            row["effect_label"].startswith("UNKNOWN") for row in private
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"grounded adjudication checks failed: {checks}")

    manifest = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "items": 5,
        "source_protocol": SOURCE_PROTOCOL,
        "selection_rule": (
            "all_and_only_source_rows_with_decisive_criterion_"
            "visible_context_fidelity"
        ),
        "why_selected": (
            "Visible-dialogue fidelity cannot distinguish authorized private "
            "memory from unsupported inference. Other 35 decisions remain frozen."
        ),
        "evidence_policy": (
            "Union of same-user strictly-prior memory selected across both arms; "
            "shown only for grounding, never as a reason to reward mention."
        ),
        "original_decision_and_arm_identity_hidden": True,
        "original_35_non_fidelity_decisions_frozen": True,
        "checks": checks,
        "effect_labels_created": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    source_copy = out_dir / "source_visible_only_annotations.jsonl"
    packet_path = out_dir / "human_grounded_packet.jsonl"
    key_path = out_dir / "private_grounded_key.jsonl"
    blank_path = out_dir / "blank_grounded_adjudications.jsonl"
    html_path = out_dir / "human_grounded_review.html"
    write_jsonl(source_copy, annotations)
    write_jsonl(packet_path, public)
    write_jsonl(key_path, private)
    write_jsonl(
        blank_path,
        [
            {
                "protocol": PROTOCOL,
                "blind_item_id": row["blind_item_id"],
                "quality_preference": None,
                "decisive_criterion": None,
                "quality_notes": "",
                "annotator_id": "",
            }
            for row in public
        ],
    )
    html_path.write_text(
        HTML_TEMPLATE.replace(
            "__DATA__", canonical_json({"manifest": manifest, "items": public})
        ),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in (source_copy, packet_path, key_path, blank_path, html_path)
    }
    manifest["source_annotations_sha256"] = sha256_file(source_annotations_path)
    manifest["generation_outcomes_sha256"] = sha256_file(
        execution_dir / "generation_outcomes.jsonl"
    )
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_replacement_blind_v1",
    )
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d3_ms_replacement_step0_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d3_ms_replacement_generation_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d3_ms_grounded_fidelity_adjudication_v1",
    )
    args = parser.parse_args()
    manifest = build(
        source_annotations_path=args.source_annotations,
        blind_dir=args.blind_dir,
        blueprint_dir=args.blueprint_dir,
        execution_dir=args.execution_dir,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("protocol", "status", "items", "selection_rule")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
