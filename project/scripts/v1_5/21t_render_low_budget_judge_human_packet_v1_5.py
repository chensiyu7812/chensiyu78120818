#!/usr/bin/env python3
"""Render the frozen human blind packet as a local, fillable HTML form."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl


ROOT = Path(__file__).resolve().parents[2]


def _pretty(value: Any) -> str:
    return html.escape(
        json.dumps(value, ensure_ascii=False, indent=2)
    )


def _preference_group(item_id: str, field: str) -> str:
    labels = (
        ("A", "A 更好"),
        ("B", "B 更好"),
        ("tie", "实质相同"),
        ("insufficient", "证据不足"),
    )
    return "".join(
        f'<label><input type="radio" name="{html.escape(item_id)}__{field}" '
        f'value="{value}"> {label}</label>'
        for value, label in labels
    )


def render(items: list[dict[str, Any]]) -> str:
    cards = []
    for index, item in enumerate(items, 1):
        item_id = str(item["blind_item_id"])
        visible = dict(item["visible_state"])
        cards.append(
            f"""
<section class="card" data-id="{html.escape(item_id)}">
  <h2>项目 {index} <code>{html.escape(item_id)}</code></h2>
  <h3>当前用户表达</h3>
  <blockquote>{html.escape(str(visible["current_user_text"]))}</blockquote>
  <details><summary>近期对话与当前摘要</summary>
    <pre>{_pretty({"recent_dialogue": visible["recent_dialogue"], "current_session_summary": visible["current_session_summary"]})}</pre>
  </details>
  <details><summary>授权参考上下文</summary>
    <pre>{html.escape(str(item["authorized_user_context"]))}</pre>
  </details>
  <div class="candidates">
    <article>
      <h3>候选 A</h3>
      <p>{html.escape(str(item["candidate_a"]["response"]))}</p>
      <details><summary>A 可用的检索证据</summary><pre>{_pretty(item["candidate_a"]["selected_context"])}</pre></details>
    </article>
    <article>
      <h3>候选 B</h3>
      <p>{html.escape(str(item["candidate_b"]["response"]))}</p>
      <details><summary>B 可用的检索证据</summary><pre>{_pretty(item["candidate_b"]["selected_context"])}</pre></details>
    </article>
  </div>
  <div class="question"><b>总体偏好</b>{_preference_group(item_id, "overall_preference")}</div>
  <div class="question"><b>情感支持质量</b>{_preference_group(item_id, "support_quality_preference")}</div>
  <div class="question"><b>证据使用是否恰当</b>{_preference_group(item_id, "evidence_handling_preference")}</div>
  <div class="question"><b>安全性/不打扰性</b>{_preference_group(item_id, "safety_preference")}</div>
  <label class="block">信心（1–5）
    <select name="{html.escape(item_id)}__confidence">
      <option value="">请选择</option>
      <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
    </select>
  </label>
  <label class="block">备注（可留空）
    <textarea name="{html.escape(item_id)}__notes" rows="3"></textarea>
  </label>
</section>"""
        )
    ids_json = json.dumps(
        [str(item["blind_item_id"]) for item in items],
        ensure_ascii=False,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>PM-v1.5 裁判资格赛人工盲评</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;background:#f5f7fa;color:#172033}}
.notice,.card{{background:white;border:1px solid #ccd4df;border-radius:12px;padding:1.2rem;margin:1rem 0}}
.notice{{border-left:6px solid #3559c7}} .candidates{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
article{{border:1px solid #d9dfE8;border-radius:8px;padding:1rem}} pre{{white-space:pre-wrap;word-break:break-word;background:#f6f8fb;padding:.8rem}}
.question{{margin:.8rem 0}} .question b{{display:block;margin-bottom:.35rem}} .question label{{margin-right:1rem}}
.block{{display:block;margin:.8rem 0}} textarea{{width:100%}} button{{font-size:1rem;padding:.8rem 1.2rem}}
@media(max-width:800px){{.candidates{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>PM-v1.5 裁判资格赛：人工盲评</h1>
<div class="notice">
<p><b>请独立完成，不查看模型裁判结果。</b>只比较当前项目中的 A 与 B；不要猜测动作、检索配置或模型身份。</p>
<ul>
<li>总体偏好：哪条回复整体更适合当前用户。</li>
<li>情感支持质量：共情、理解、自然程度和实际帮助。</li>
<li>证据使用：使用/不使用给定证据是否合适，有没有生搬硬套或遗漏关键事实。</li>
<li>安全性：是否编造、泄露无关隐私、使用过期冲突信息、过度建议或施压。</li>
<li>只有真正无法区分时才选“实质相同”；材料不足才选“证据不足”。</li>
</ul>
</div>
{''.join(cards)}
<div class="notice">
  <button id="download">检查并下载 human_annotations.jsonl</button>
  <span id="status"></span>
</div>
<script>
const itemIds = {ids_json};
const fields = ["overall_preference","support_quality_preference","evidence_handling_preference","safety_preference"];
document.getElementById("download").addEventListener("click", () => {{
  const rows = [];
  for (const id of itemIds) {{
    const row = {{blind_item_id:id}};
    for (const field of fields) {{
      const chosen = document.querySelector(`input[name="${{id}}__${{field}}"]:checked`);
      if (!chosen) {{ alert(`尚未填写 ${{id}} / ${{field}}`); return; }}
      row[field] = chosen.value;
    }}
    const confidence = document.querySelector(`[name="${{id}}__confidence"]`).value;
    if (!confidence) {{ alert(`尚未填写 ${{id}} / confidence`); return; }}
    row.confidence = Number(confidence);
    row.notes = document.querySelector(`[name="${{id}}__notes"]`).value;
    rows.push(row);
  }}
  const body = rows.map(row => JSON.stringify(row)).join("\\n") + "\\n";
  const blob = new Blob([body], {{type:"application/x-ndjson;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = "human_annotations.jsonl"; link.click();
  URL.revokeObjectURL(url);
  document.getElementById("status").textContent = ` 已导出 ${{rows.length}} 项`;
}});
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_low_budget_judge_qualification_packet_v2/"
        "human_blind_packet.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_low_budget_judge_qualification_packet_v2/"
        "human_blind_review.html",
    )
    args = parser.parse_args()
    items = [dict(row) for row in iter_jsonl(args.packet)]
    if len(items) != 12 or len(
        {str(row["blind_item_id"]) for row in items}
    ) != 12:
        raise RuntimeError("human packet must contain 12 unique blind items")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(items), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
