#!/usr/bin/env python3
"""Prepare the one bounded quality-only review for frozen ESConv RS pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PLAN_PROTOCOL = "pm-v1.5-rs-esconv-frozen-validation-plan-v1"
EXECUTION_PROTOCOL = "pm-v1.5-rs-esconv-frozen-validation-execution-v1"
PROTOCOL = "pm-v1.5-rs-esconv-frozen-quality-blind-v1"
CRITERIA = {
    "request_and_conversation_fit": "请求与对话适配",
    "emotional_attunement": "情绪理解与回应",
    "visible_context_fidelity": "忠实于可见语境",
    "immediate_helpfulness": "当下实际帮助",
    "clarity_and_naturalness": "清晰、自然、不过载",
    "materially_equivalent": "实质等价",
    "uncertain": "证据不足/无法判断",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _order(pair_id: str) -> str:
    return hashlib.sha256(f"{PROTOCOL}:order:{pair_id}".encode()).hexdigest()


def _render_html(manifest: dict[str, Any], items: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"manifest": manifest, "items": items}, ensure_ascii=False
    ).replace("</", "<\\/")
    criteria = json.dumps(CRITERIA, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(
        f"PM V1.5 · ESConv 冻结验证 {len(items)} 对质量盲评"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:24px;background:#f4f6f8;color:#17202a}}
.note,.item{{background:#fff;border:1px solid #d9dee6;border-radius:10px;padding:18px;margin:16px 0}}
.note{{border-left:5px solid #0969da}}.summary,.dialogue{{white-space:pre-wrap;background:#edf2f6;padding:12px;border-radius:7px}}
.summary{{background:#fff8c5;margin-bottom:8px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.response{{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:12px;border-radius:7px}}
select,textarea{{width:100%;box-sizing:border-box;padding:8px;margin:5px 0 10px}}
button{{padding:10px 15px}}.sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>{title}</h1>
<div class="note"><b>这是当前 RS 方案唯一一次冻结外部质量审核，共 {len(items)} 对。</b><br>
页面不显示 R0/RS、PM 决策、卡片、策略族、检索分数或任何自动 judge 结果。
只根据同一可见语境中的 A/B 回复判断。<br><br>
只有部署相关的真实质量差异才选 A/B；轻微措辞、长度或个人风格偏好选 tie。
如果优劣互有抵消且没有明确净胜者，也选 tie。确实证据不足才选 uncertain。
本页不评风险；只有解盲后 RS 实质获胜的回复才进入一次最小风险审核。</div>
<div class="sticky"><button onclick="downloadRows()">导出 JSONL</button>
<span id="progress"></span></div><div id="root"></div>
<script>
const DATA={payload};const CRITERIA={criteria};const KEY="pm_v15_rs_esconv_frozen_quality_v1";
let state=JSON.parse(localStorage.getItem(KEY)||"{{}}");
function esc(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function ensure(id){{if(!state[id])state[id]={{quality_preference:"",decisive_criterion:"",quality_notes:"",annotator_id:""}};}}
function setv(id,key,value){{ensure(id);state[id][key]=value;localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function dialogue(item){{return item.visible_dialogue.map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");}}
function render(){{document.getElementById("root").innerHTML=DATA.items.map((item,i)=>{{ensure(item.blind_item_id);const s=state[item.blind_item_id];return `<section class="item"><h2>${{i+1}} / ${{DATA.items.length}}</h2>${{item.current_session_summary?`<div class="summary">背景摘要：${{esc(item.current_session_summary)}}</div>`:""}}<div class="dialogue">${{esc(dialogue(item))}}</div><div class="grid"><div><h3>回复 A</h3><div class="response">${{esc(item.response_a)}}</div></div><div><h3>回复 B</h3><div class="response">${{esc(item.response_b)}}</div></div></div><h3>哪个回复实质更好？</h3><select onchange="setv('${{item.blind_item_id}}','quality_preference',this.value)"><option value="">请选择</option>${{["A","B","tie","uncertain"].map(v=>`<option value="${{v}}" ${{s.quality_preference===v?"selected":""}}>${{v}}</option>`).join("")}}</select><select onchange="setv('${{item.blind_item_id}}','decisive_criterion',this.value)"><option value="">请选择决定性标准</option>${{Object.entries(CRITERIA).map(([k,v])=>`<option value="${{k}}" ${{s.decisive_criterion===k?"selected":""}}>${{esc(v)}}</option>`).join("")}}</select><textarea placeholder="可选：简述决定性差异" onchange="setv('${{item.blind_item_id}}','quality_notes',this.value)">${{esc(s.quality_notes)}}</textarea></section>`;}}).join("");localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function rows(){{return DATA.items.map(item=>{{ensure(item.blind_item_id);return {{protocol:DATA.manifest.protocol,blind_item_id:item.blind_item_id,...state[item.blind_item_id]}};}});}}
function progress(){{const r=rows(),n=r.filter(x=>x.quality_preference&&x.decisive_criterion).length;document.getElementById("progress").textContent=` 已完成 ${{n}} / ${{r.length}}`;}}
function downloadRows(){{const r=rows();const missing=r.filter(x=>!x.quality_preference||!x.decisive_criterion);if(missing.length&&!confirm(`还有 ${{missing.length}} 条未完成，仍然导出吗？`))return;const b=new Blob([r.map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="independent_quality_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
render();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_quality_blind_v1_candidate",
    )
    args = parser.parse_args()

    plan_report = read_json(args.plan_dir / "plan_report.json")
    generation_summary = read_json(
        args.execution_dir / "generation_summary.json"
    )
    selected_list = _rows(args.plan_dir / "selected_states.jsonl")
    selected = {str(row["pair_id"]): row for row in selected_list}
    outcomes_list = _rows(
        args.execution_dir / "generation_outcomes.jsonl"
    )
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for row in outcomes_list:
        pair_id = str(row["pair_id"])
        arm = str(row["arm"])
        if arm in outcomes.setdefault(pair_id, {}):
            raise RuntimeError(f"duplicate arm: {pair_id}/{arm}")
        outcomes[pair_id][arm] = row
    if (
        plan_report.get("protocol") != PLAN_PROTOCOL
        or plan_report.get("status") != "FROZEN_READY_FOR_PAIRED_GENERATION"
        or generation_summary.get("protocol") != EXECUTION_PROTOCOL
        or generation_summary.get("status") != "COMPLETE"
        or len(selected) != 40
        or len(outcomes_list) != 80
        or set(selected) != set(outcomes)
        or any(set(arms) != {"R0", "RS"} for arms in outcomes.values())
        or any(
            row.get("normalized_finish_reason") != "complete"
            for row in outcomes_list
        )
    ):
        raise RuntimeError("frozen plan or paired generation is incomplete")

    items: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    arm_a_counts: Counter[str] = Counter()
    ordered_pair_ids = sorted(selected, key=_order)
    for position, pair_id in enumerate(ordered_pair_ids):
        # The item order is hash-randomized, then treatment placement
        # alternates. This preserves deterministic blinding while guaranteeing
        # exact 20/20 A-position balance for the fixed 40-pair panel.
        a_arm = "R0" if position % 2 == 0 else "RS"
        b_arm = "RS" if a_arm == "R0" else "R0"
        arm_a_counts[a_arm] += 1
        blind_id = "rs_esconv_quality_" + hashlib.sha256(
            f"{PROTOCOL}:{pair_id}".encode()
        ).hexdigest()[:20]
        state = selected[pair_id]
        items.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "current_session_summary": state[
                    "current_session_summary"
                ],
                "visible_dialogue": state["visible_dialogue"],
                "response_a": outcomes[pair_id][a_arm]["response"],
                "response_b": outcomes[pair_id][b_arm]["response"],
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "pair_id": pair_id,
                "state_id": state["state_id"],
                "user_id": state["user_id"],
                "response_a_arm": a_arm,
                "response_b_arm": b_arm,
                "selected_card_id": state["selected_card_id"],
                "selected_core_submove_id": state[
                    "selected_core_submove_id"
                ],
                "selected_strategy_family": state[
                    "selected_strategy_family"
                ],
                "pm_rs_probability": state["pm_rs_probability"],
                "frozen_pm_action": state["frozen_pm_action"],
            }
        )
        template.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "quality_preference": None,
                "decisive_criterion": None,
                "quality_notes": "",
                "annotator_id": "",
            }
        )

    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_FROZEN_QUALITY_ONLY_HUMAN_REVIEW",
        "pair_count": len(items),
        "independent_dialogue_count": len(
            {str(row["user_id"]) for row in selected_list}
        ),
        "questions_per_pair": 1,
        "structured_risk_questions_in_this_phase": 0,
        "all_frozen_pairs_included": True,
        "automatic_judge_used": False,
        "hidden_fields": [
            "R0/RS identity",
            "frozen PM action and probability",
            "strategy card, core move, and family",
            "retrieval score",
        ],
        "quality_rule": (
            "A/B only for a material deployment-relevant advantage; minor "
            "style/length preference and offsetting tradeoffs are tie"
        ),
        "next_phase": (
            "Unblind once; risk-review only RS material wins; compute frozen "
            "PM, always-off, and always-on utility without retuning."
        ),
        "private_randomization_diagnostic": {
            "a_arm_counts": dict(sorted(arm_a_counts.items()))
        },
        "lineage": {
            "plan_freeze_manifest_sha256": sha256_file(
                args.plan_dir / "freeze_manifest.json"
            ),
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_summary_sha256": sha256_file(
                args.execution_dir / "generation_summary.json"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.execution_dir / "generation_outcomes.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "blind_manifest.json", manifest)
    write_jsonl(args.out_dir / "human_blind_packet.jsonl", items)
    write_jsonl(args.out_dir / "private_blinding_key.jsonl", private)
    write_jsonl(args.out_dir / "human_annotation_template.jsonl", template)
    (args.out_dir / "human_blind_review.html").write_text(
        _render_html(manifest, items), encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    main()
