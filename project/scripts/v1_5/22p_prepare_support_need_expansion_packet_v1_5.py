#!/usr/bin/env python3
"""Prepare the revised fresh SupportNeed human expansion packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.io import write_json, write_jsonl
from metacom_pm.v1_5_support_need_packet import (
    HUMAN_BOUNDARY_FIELDS,
    build_support_need_expansion_packet,
)


ROOT = Path(__file__).resolve().parents[2]


def _render_html(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><title>SupportNeed 扩展人工盲评</title>
<style>
body{{font:16px sans-serif;max-width:1050px;margin:24px auto;line-height:1.5}}
.card{{border:1px solid #bbb;padding:16px;margin:18px 0;border-radius:8px}}
pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}
label{{display:block;margin:7px 0}} select,input,textarea{{font:inherit}}
textarea{{width:100%;height:60px}} .quote{{width:70%}} button{{padding:10px 16px}}
.rubric{{background:#f4f8ff;border-left:5px solid #3367aa;padding:12px 18px}}
</style>
<h1>SupportNeedObservation 扩展人工盲评（修订语义）</h1>
<section class="rubric">
<p>仍只判断用户此刻需要怎样被支持，不判断 RS/记忆动作。先选 mode、goal、phase、
非临床 urgency，再单独判断下一条回复的负担。</p>
<p><b>负担：</b>minimal_presence=只承接、不加问题/任务；one_focus=最多一个聚焦问题或
可拒绝小步骤；multi_step_ok=用户已准备共同拆解多个步骤。</p>
<p><b>明确边界：</b>只有用户可见文本确实说出边界时才粘贴精确原句；留空表示
“未找到明确证据”，不是 false。不要粘贴 assistant 的话。</p>
</section><div id="root"></div><button onclick="download()">导出 JSONL</button>
<script>
const rows={payload};
const modes=["listen","explore","comfort_reassure","light_guidance","structured_planning","abstain"];
const phases=["exploration","comforting","action","abstain"];
const urgencies=["routine","elevated","acute","abstain"];
const burdens=["minimal_presence","one_focus","multi_step_ok","abstain"];
const goals=["be_heard","make_sense","stabilize","decide","act"];
const boundaryTypes={json.dumps(list(HUMAN_BOUNDARY_FIELDS), ensure_ascii=False)};
function sel(options,cls){{return `<select class="${{cls}}"><option value="">--请选择--</option>${{options.map(x=>`<option>${{x}}</option>`).join("")}}</select>`}}
document.querySelector("#root").innerHTML=rows.map((r,i)=>`<section class="card" data-i="${{i}}">
<h2>${{i+1}} / ${{rows.length}}</h2><pre>${{JSON.stringify(r.visible_state,null,2)}}</pre>
<label>support mode ${{sel(modes,"mode")}}</label>
<label>goals ${{goals.map(g=>`<input type="checkbox" class="goal" value="${{g}}">${{g}} `).join("")}}</label>
<label>phase ${{sel(phases,"phase")}}</label><label>urgency ${{sel(urgencies,"urgency")}}</label>
<label>recommended response burden ${{sel(burdens,"burden")}}</label>
<p><b>可选：明确边界精确引文</b></p>
${{boundaryTypes.map(k=>`<label>${{k}} <input class="quote ${{k}}" placeholder="没有明确原句就留空"></label>`).join("")}}
<label>confidence <input class="confidence" type="number" min="1" max="5" value="3"></label>
<label>notes<textarea class="notes"></textarea></label></section>`).join("");
function download(){{
 const out=[];
 for(let i=0;i<rows.length;i++){{const r=rows[i],c=document.querySelector(`[data-i="${{i}}"]`);
  const mode=c.querySelector(".mode").value,phase=c.querySelector(".phase").value;
  const urgency=c.querySelector(".urgency").value,burden=c.querySelector(".burden").value;
  const goalsSelected=[...c.querySelectorAll(".goal:checked")].map(x=>x.value);
  if(!mode||!phase||!urgency||!burden){{alert(`第 ${{i+1}} 条尚未完成核心字段`);return;}}
  const abstain=mode==="abstain";
  if(!abstain && goalsSelected.length===0){{alert(`第 ${{i+1}} 条至少选择一个 goal`);return;}}
  if(abstain && (phase!=="abstain"||urgency!=="abstain"||burden!=="abstain")){{alert(`第 ${{i+1}} 条 abstain 字段必须一致`);return;}}
  if(!abstain && (phase==="abstain"||urgency==="abstain"||burden==="abstain")){{alert(`第 ${{i+1}} 条非 abstain 不能混用 abstain`);return;}}
  const evidence=boundaryTypes.flatMap(k=>{{const q=c.querySelector("."+k).value.trim();return q?[{{boundary_type:k,exact_user_quote:q}}]:[];}});
  out.push({{blind_item_id:r.blind_item_id,support_mode:abstain?null:mode,goals:abstain?[]:goalsSelected,
   dialogue_phase:abstain?null:phase,nonclinical_urgency:abstain?null:urgency,
   recommended_response_burden:abstain?null:burden,active_explicit_boundary_evidence:abstain?[]:evidence,
   abstain,confidence:Number(c.querySelector(".confidence").value),notes:c.querySelector(".notes").value}});
 }}
 const blob=new Blob([out.map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="support_need_human_annotations_v2.jsonl";a.click();
}}
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-packet-dir", type=Path, required=True)
    parser.add_argument("--packet-size", type=int, default=24)
    parser.add_argument("--fit-anchor-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7727)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_support_need_expansion_packet(
        source_packet_dir=args.source_packet_dir,
        packet_size=args.packet_size,
        fit_anchor_count=args.fit_anchor_count,
        seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "qualification_contract.json", result["contract"])
    write_jsonl(
        args.out_dir / "human_blind_packet.jsonl", result["packet_rows"]
    )
    write_jsonl(
        args.out_dir / "human_annotation_template.jsonl",
        result["template_rows"],
    )
    write_jsonl(
        args.out_dir / "private_lineage.jsonl", result["private_rows"]
    )
    (args.out_dir / "human_blind_review.html").write_text(
        _render_html(result["packet_rows"]), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
