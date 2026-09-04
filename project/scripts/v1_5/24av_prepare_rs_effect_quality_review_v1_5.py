#!/usr/bin/env python3
"""Prepare one bounded, quality-only blind review for all RS effect pairs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-effect-human-quality-blind-v1"
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


def _a_arm(pair_id: str) -> str:
    digest = hashlib.sha256(f"{PROTOCOL}:{pair_id}".encode()).hexdigest()
    return "RS" if int(digest, 16) % 2 else "R0"


def _item_order(pair_id: str) -> str:
    return hashlib.sha256(f"{PROTOCOL}:order:{pair_id}".encode()).hexdigest()


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_html(manifest: dict[str, Any], items: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"manifest": manifest, "items": items}, ensure_ascii=False
    ).replace("</", "<\\/")
    criteria = json.dumps(CRITERIA, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"PM V1.5 RS 直接效应 · {len(items)} 对质量盲评")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:24px;background:#f4f6f8;color:#17202a}}
.note,.item{{background:#fff;border:1px solid #d9dee6;border-radius:10px;padding:18px;margin:16px 0}}
.note{{border-left:5px solid #0969da}}.dialogue{{white-space:pre-wrap;background:#edf2f6;padding:12px;border-radius:7px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.response{{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:12px;border-radius:7px}}
select,textarea,input{{width:100%;box-sizing:border-box;padding:8px;margin:5px 0 10px}}
button{{padding:10px 15px}}.sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style></head><body>
<h1>{title}</h1>
<div class="note"><b>这次只做 32 个质量选择，不做风险矩阵。</b><br>
页面不显示 R0/RS、卡片、策略族、检索分数或自动 judge 结果。请只看可见对话与两个回复。<br><br>
“实质更好”指：如果实际部署只能采用一个回复，你会因为请求适配、情绪理解、语境忠实、
当下帮助或清晰自然方面的真实差异而选它；仅仅更长、更短、措辞稍顺或个人文风偏好，
一律选 tie。若优劣互有抵消且没有明确净胜者，也选 tie。确实无法判断才选 uncertain。<br><br>
明显胡编或越界会降低实际可用质量，但本页无需分类风险。只有 RS 实质胜出的候选，
下一阶段才做一次小规模原子风险审核。</div>
<div class="sticky"><button onclick="downloadRows()">导出 JSONL</button>
<span id="progress"></span></div><div id="root"></div>
<script>
const DATA={payload}; const CRITERIA={criteria};
const KEY="pm_v15_rs_effect_quality_blind_v1";
let state=JSON.parse(localStorage.getItem(KEY)||"{{}}");
function esc(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function ensure(id){{if(!state[id])state[id]={{quality_preference:"",decisive_criterion:"",quality_notes:"",annotator_id:""}};}}
function setv(id,key,value){{ensure(id);state[id][key]=value;localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function dialogue(item){{return item.visible_dialogue.map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");}}
function render(){{document.getElementById("root").innerHTML=DATA.items.map((item,i)=>{{ensure(item.blind_item_id);const s=state[item.blind_item_id];return `<section class="item"><h2>${{i+1}} / ${{DATA.items.length}}</h2><div class="dialogue">${{esc(dialogue(item))}}</div><div class="grid"><div><h3>回复 A</h3><div class="response">${{esc(item.response_a)}}</div></div><div><h3>回复 B</h3><div class="response">${{esc(item.response_b)}}</div></div></div><h3>哪个回复实质更好？</h3><select onchange="setv('${{item.blind_item_id}}','quality_preference',this.value)"><option value="">请选择</option>${{["A","B","tie","uncertain"].map(v=>`<option value="${{v}}" ${{s.quality_preference===v?"selected":""}}>${{v}}</option>`).join("")}}</select><select onchange="setv('${{item.blind_item_id}}','decisive_criterion',this.value)"><option value="">请选择决定性标准</option>${{Object.entries(CRITERIA).map(([k,v])=>`<option value="${{k}}" ${{s.decisive_criterion===k?"selected":""}}>${{esc(v)}}</option>`).join("")}}</select><textarea placeholder="可选：用可见文本简述决定性差异" onchange="setv('${{item.blind_item_id}}','quality_notes',this.value)">${{esc(s.quality_notes)}}</textarea></section>`;}}).join("");localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function rows(){{return DATA.items.map(item=>{{ensure(item.blind_item_id);return {{protocol:DATA.manifest.protocol,blind_item_id:item.blind_item_id,...state[item.blind_item_id]}};}});}}
function progress(){{const r=rows(),n=r.filter(x=>x.quality_preference&&x.decisive_criterion).length;document.getElementById("progress").textContent=` 已完成 ${{n}} / ${{r.length}}`;}}
function downloadRows(){{const r=rows();const missing=r.filter(x=>!x.quality_preference||!x.decisive_criterion);if(missing.length&&!confirm(`还有 ${{missing.length}} 条未完成，仍然导出吗？`))return;const b=new Blob([r.map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="independent_quality_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
render();</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_rs_paired_effect_v1_execution",
    )
    parser.add_argument(
        "--judge-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_paired_effect_v1_judging"
        / "measurement_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_effect_human_quality_blind_v1_candidate",
    )
    args = parser.parse_args()

    selected = {
        str(row["pair_id"]): row
        for row in _rows(args.plan_dir / "selected_states.jsonl")
    }
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _rows(args.execution_dir / "generation_outcomes.jsonl"):
        pair_id = str(row["pair_id"])
        arm = str(row["arm"])
        if arm in outcomes.setdefault(pair_id, {}):
            raise RuntimeError(f"duplicate arm: {pair_id}/{arm}")
        outcomes[pair_id][arm] = row
    if set(selected) != set(outcomes) or any(
        set(arms) != {"R0", "RS"} for arms in outcomes.values()
    ):
        raise RuntimeError("expected exactly one R0 and RS response per pair")

    judge = json.loads(args.judge_report.read_text(encoding="utf-8"))
    if judge["quality"]["measurement_gate_passed"]:
        raise RuntimeError("human fallback is only for a failed judge gate")

    items: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    arm_a_counts: Counter[str] = Counter()
    for pair_id in sorted(selected, key=_item_order):
        a_arm = _a_arm(pair_id)
        b_arm = "RS" if a_arm == "R0" else "R0"
        arm_a_counts[a_arm] += 1
        blind_id = "rs_effect_quality_" + hashlib.sha256(
            f"{PROTOCOL}:{pair_id}".encode()
        ).hexdigest()[:20]
        state = selected[pair_id]
        items.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
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
        "status": "READY_FOR_BOUNDED_QUALITY_ONLY_HUMAN_REVIEW",
        "pair_count": len(items),
        "independent_dialogue_count": len(
            {str(row["user_id"]) for row in selected.values()}
        ),
        "questions_per_pair": 1,
        "structured_risk_questions_in_this_phase": 0,
        "all_pairs_included": True,
        "automatic_judge_used_for_selection": False,
        "hidden_fields": [
            "R0/RS identity",
            "strategy card",
            "strategy family",
            "retrieval score",
            "automatic judge result",
        ],
        "quality_rule": (
            "A/B only for a material deployment-relevant advantage; "
            "minor style/length preference and offsetting tradeoffs are tie"
        ),
        "next_phase": (
            "Only human-adjudicated RS material wins receive atomic risk "
            "review; tie/R0/uncertain pairs do not create RS-positive labels."
        ),
        "lineage": {
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.execution_dir / "generation_outcomes.jsonl"
            ),
            "failed_judge_measurement_sha256": sha256_file(args.judge_report),
        },
    }
    diagnostics: dict[str, Any] = {
        "protocol": PROTOCOL,
        "pair_count": len(selected),
        "automatic_judge_quality_measurement_gate_passed": False,
        "automatic_judge_ab_ba_consistency_rate": judge["quality"][
            "ab_ba_consistency_rate"
        ],
        "automatic_judge_minimum_consistency_rate": judge["quality"][
            "qualification_thresholds"
        ]["minimum_ab_ba_consistency_rate"],
        "automatic_judge_results_are_training_labels": False,
        "a_arm_counts_private_diagnostic": dict(sorted(arm_a_counts.items())),
        "overall_by_arm": {},
        "by_strategy_family": {},
        "interpretation": (
            "RS was realized and changed every response, but added prompt "
            "tokens and produced shorter outputs. This is a treatment "
            "mechanism diagnostic, not a quality conclusion."
        ),
    }
    for arm in ("R0", "RS"):
        arm_rows = [outcomes[pair_id][arm] for pair_id in selected]
        diagnostics["overall_by_arm"][arm] = {
            "responses": len(arm_rows),
            "mean_prompt_tokens": _mean(
                [int(row["usage"]["prompt_tokens"]) for row in arm_rows]
            ),
            "mean_completion_tokens": _mean(
                [
                    int(row["usage"]["completion_tokens"])
                    for row in arm_rows
                ]
            ),
            "mean_total_tokens": _mean(
                [int(row["usage"]["total_tokens"]) for row in arm_rows]
            ),
        }
    families = sorted(
        {
            str(row["selected_strategy_family"])
            for row in selected.values()
        }
    )
    for family in families:
        family_pairs = [
            pair_id
            for pair_id, row in selected.items()
            if str(row["selected_strategy_family"]) == family
        ]
        diagnostics["by_strategy_family"][family] = {
            arm: {
                "responses": len(family_pairs),
                "mean_completion_tokens": _mean(
                    [
                        int(
                            outcomes[pair_id][arm]["usage"][
                                "completion_tokens"
                            ]
                        )
                        for pair_id in family_pairs
                    ]
                ),
            }
            for arm in ("R0", "RS")
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "blind_manifest.json", manifest)
    write_json(args.out_dir / "treatment_diagnostic.json", diagnostics)
    write_jsonl(args.out_dir / "human_blind_packet.jsonl", items)
    write_jsonl(args.out_dir / "private_blinding_key.jsonl", private)
    write_jsonl(args.out_dir / "human_annotation_template.jsonl", template)
    (args.out_dir / "human_blind_review.html").write_text(
        _render_html(manifest, items), encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    main()
