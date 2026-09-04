#!/usr/bin/env python3
"""Prepare a fresh, train-only, outcome-blind SupportNeed annotation packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.io import write_json, write_jsonl
from metacom_pm.v1_5_support_need_packet import (
    build_support_need_annotation_packet,
)


ROOT = Path(__file__).resolve().parents[2]


def _render_html(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><title>SupportNeed 人工盲评</title>
<style>
body{{font:16px sans-serif;max-width:1000px;margin:24px auto;line-height:1.5}}
.card{{border:1px solid #bbb;padding:16px;margin:18px 0;border-radius:8px}}
pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}
label{{display:block;margin:7px 0}} select,input,textarea{{font:inherit}}
textarea{{width:100%;height:60px}} button{{padding:10px 16px}}
.rubric{{background:#f4f8ff;border-left:5px solid #3367aa;padding:12px 18px}}
</style>
<h1>SupportNeedObservation train-only 人工盲评</h1>
<p>只判断当前最合适的支持方式，不判断应否开启 RS/记忆，也不要猜数据来源。</p>
<section class="rubric">
<p><b>判定顺序：</b>先看用户明确边界，再看当前目标，再看对话阶段，最后判断非临床紧迫度。
不要把“用户愿意接受建议”直接等同于“必须开启 RS”。</p>
<ul>
<li><b>listen</b>：主要需要被听见，暂不推进问题解决。</li>
<li><b>explore</b>：用一个低负担问题澄清感受、事实或目标。</li>
<li><b>comfort_reassure</b>：先稳定、安慰、正常化感受，不急于分析或计划。</li>
<li><b>light_guidance</b>：可给一个低门槛、可拒绝的小建议。</li>
<li><b>structured_planning</b>：用户明确准备行动，可共同拆解多步计划。</li>
<li><b>abstain</b>：可见信息不足或多个 mode 无法可靠区分；不是“随便选一个”。</li>
</ul>
<p><b>urgency</b> 只表示当前非临床支持紧迫度，不做疾病/危机诊断；真实危机由独立
safety policy 处理。<b>question_or_task_burden_limit</b> 表示此时回复最多应提出一个
简短问题或一个小任务。若非 abstain，至少选择一个 goal。</p>
</section>
<div id="root"></div><button onclick="download()">导出 JSONL</button>
<script>
const rows={payload};
const modes=["listen","explore","comfort_reassure","light_guidance","structured_planning","abstain"];
const phases=["exploration","comforting","action","abstain"];
const urgencies=["routine","elevated","acute","abstain"];
const goals=["be_heard","make_sense","stabilize","decide","act"];
const bools=["unknown","true","false"];
function sel(options, cls){{return `<select class="${{cls}}"><option value="">--请选择--</option>${{options.map(x=>`<option>${{x}}</option>`).join("")}}</select>`}}
document.querySelector("#root").innerHTML=rows.map((r,i)=>`<section class="card" data-i="${{i}}">
<h2>${{i+1}} / ${{rows.length}}</h2>
<pre>${{JSON.stringify(r.visible_state,null,2)}}</pre>
<label>support mode ${{sel(modes,"mode")}}</label>
<label>goals ${{goals.map(g=>`<input type="checkbox" class="goal" value="${{g}}">${{g}} `).join("")}}</label>
<label>phase ${{sel(phases,"phase")}}</label>
<label>urgency ${{sel(urgencies,"urgency")}}</label>
${{["advice_rejected","advice_requested","one_small_step_requested","listen_first_requested","question_or_task_burden_limit"].map(k=>`<label>${{k}} ${{sel(bools,k)}}</label>`).join("")}}
<label>confidence <input class="confidence" type="number" min="1" max="5" value="3"></label>
<label>notes<textarea class="notes"></textarea></label></section>`).join("");
function download(){{
 const out=[];
 for(let i=0;i<rows.length;i++){{const r=rows[i]; const c=document.querySelector(`[data-i="${{i}}"]`); const mode=c.querySelector(".mode").value;
 const phase=c.querySelector(".phase").value; const urgency=c.querySelector(".urgency").value;
 const selectedGoals=[...c.querySelectorAll(".goal:checked")].map(x=>x.value);
 if(!mode || !phase || !urgency){{alert(`第 ${{i+1}} 条尚未完成 mode/phase/urgency`);return;}}
 if(mode!=="abstain" && selectedGoals.length===0){{alert(`第 ${{i+1}} 条非 abstain，至少选择一个 goal`);return;}}
 const val=k=>{{const x=c.querySelector("."+k).value; return x==="unknown"?null:x==="true"}};
 out.push({{blind_item_id:r.blind_item_id,support_mode:mode==="abstain"?null:mode,
 goals:selectedGoals,
 dialogue_phase:phase==="abstain"?null:phase,
 nonclinical_urgency:urgency==="abstain"?null:urgency,
 advice_rejected:val("advice_rejected"),advice_requested:val("advice_requested"),
 one_small_step_requested:val("one_small_step_requested"),listen_first_requested:val("listen_first_requested"),
 question_or_task_burden_limit:val("question_or_task_burden_limit"),abstain:mode==="abstain",
 confidence:Number(c.querySelector(".confidence").value),notes:c.querySelector(".notes").value}});
 }}
 const blob=new Blob([out.map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="support_need_human_annotations.jsonl";a.click();
}}
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--esconv", type=Path, default=ROOT / "data/external/ESConv.json"
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data/strategy/esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--existing-seed-sources",
        type=Path,
        default=ROOT / "data/strategy/pm_v1_5_selected_seed_sources.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data/strategy/strategy_cards_v1_5.jsonl",
    )
    parser.add_argument("--packet-size", type=int, default=75)
    parser.add_argument("--human-anchor-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=4311)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_support_need_annotation_packet(
        esconv_path=args.esconv,
        split_manifest_path=args.split_manifest,
        existing_seed_sources_path=args.existing_seed_sources,
        strategy_bank_path=args.strategy_bank,
        packet_size=args.packet_size,
        human_anchor_size=args.human_anchor_size,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "qualification_contract.json", result["contract"])
    write_jsonl(args.out_dir / "support_need_packet.jsonl", result["blind_rows"])
    write_jsonl(
        args.out_dir / "human_blind_packet.jsonl", result["human_anchor_rows"]
    )
    write_jsonl(
        args.out_dir / "human_annotation_template.jsonl",
        result["annotation_template_rows"],
    )
    write_jsonl(args.out_dir / "private_lineage.jsonl", result["lineage_rows"])
    (args.out_dir / "human_blind_review.html").write_text(
        _render_html(result["human_anchor_rows"]), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
