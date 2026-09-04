#!/usr/bin/env python3
"""Build a zero-API, technique-only Strategy Bank V2 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metacom_pm.io import write_json, write_jsonl
from metacom_pm.v1_5_strategy_bank import build_strategy_bank_v2_candidate


ROOT = Path(__file__).resolve().parents[2]


def _render_html(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><title>Strategy Bank V2 五卡人评</title>
<style>
body{{font:16px sans-serif;max-width:1000px;margin:24px auto;line-height:1.5}}
.card{{border:1px solid #bbb;padding:16px;margin:18px 0;border-radius:8px}}
pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}
label{{display:block;margin:7px 0}} select,input,textarea{{font:inherit}}
textarea{{width:100%;height:60px}} button{{padding:10px 16px}}
</style>
<h1>Strategy Bank V2 technique-only 五卡审查</h1>
<p>只审核卡片定义是否清楚、安全、适合 train-only pilot。看不到原 supporter 回复；
不得根据来源 survey 把卡片当成“已证明有益”。</p>
<div id="root"></div><button onclick="download()">导出 JSONL</button>
<script>
const rows={payload};
const checks=["support_move_clear","when_to_use_valid","when_not_to_use_valid",
"mode_phase_goal_fit_valid","burden_and_risk_flags_valid","safe_general_technique",
"approve_for_train_only_pilot"];
function choice(cls){{return `<select class="${{cls}}"><option value="">--请选择--</option><option>true</option><option>false</option></select>`}}
document.querySelector("#root").innerHTML=rows.map((r,i)=>`<section class="card" data-i="${{i}}">
<h2>${{i+1}} / ${{rows.length}} — ${{r.strategy_family}}</h2>
<pre>${{JSON.stringify(r,null,2)}}</pre>
${{checks.map(k=>`<label>${{k}} ${{choice(k)}}</label>`).join("")}}
<label>confidence <input class="confidence" type="number" min="1" max="5" value="3"></label>
<label>required corrections<textarea class="corrections"></textarea></label>
<label>notes<textarea class="notes"></textarea></label></section>`).join("");
function download(){{
 const out=[];
 for(let i=0;i<rows.length;i++){{const r=rows[i],c=document.querySelector(`[data-i="${{i}}"]`),row={{card_id:r.card_id}};
  for(const k of checks){{const v=c.querySelector("."+k).value;if(!v){{alert(`第 ${{i+1}} 条未完成 ${{k}}`);return;}}row[k]=v==="true";}}
  row.confidence=Number(c.querySelector(".confidence").value);
  row.required_corrections=c.querySelector(".corrections").value;
  row.notes=c.querySelector(".notes").value;out.push(row);
 }}
 const blob=new Blob([out.map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="strategy_bank_v2_human_annotations.jsonl";a.click();
}}
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-strategy-bank",
        type=Path,
        default=ROOT / "data/strategy/strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--esconv",
        type=Path,
        default=ROOT / "data/external/ESConv.json",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "data/strategy/esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument("--support-need-lineage", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    result = build_strategy_bank_v2_candidate(
        raw_strategy_bank_path=args.raw_strategy_bank,
        esconv_path=args.esconv,
        split_manifest_path=args.split_manifest,
        support_need_lineage_path=args.support_need_lineage,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "strategy_cards_v2_candidate.jsonl", result["cards"])
    write_jsonl(args.out_dir / "strategy_bank_v2_lineage.jsonl", result["lineage_rows"])
    write_jsonl(
        args.out_dir / "human_review_packet.jsonl",
        result["human_review_rows"],
    )
    write_jsonl(
        args.out_dir / "human_annotation_template.jsonl",
        result["human_annotation_template"],
    )
    (args.out_dir / "human_review.html").write_text(
        _render_html(result["human_review_rows"]),
        encoding="utf-8",
    )
    write_json(args.out_dir / "build_report.json", result["report"])


if __name__ == "__main__":
    main()
