#!/usr/bin/env python3
"""Prepare a fixed, outcome-blind review packet for 50 V3 core cards."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from metacom_pm.strategy_bank import load_esconv
from metacom_pm.text import normalize_space


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-bank-v3-core-human-review-v1"


def _select_examples(
    rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select high, median, and boundary score examples without outcomes."""

    unique: list[dict[str, Any]] = []
    seen_dialogues: set[str] = set()
    for row in sorted(
        rows,
        key=lambda item: (
            -float(item["top_cosine_score"]),
            str(item["source_dialogue_id"]),
            int(item["source_turn_index"]),
        ),
    ):
        dialogue_id = str(row["source_dialogue_id"])
        if dialogue_id not in seen_dialogues:
            unique.append(row)
            seen_dialogues.add(dialogue_id)
    if len(unique) <= count:
        return unique
    confident = [row for row in unique if row["assignment_confident"]]
    pool = confident if len(confident) >= count else unique
    positions = sorted({0, len(pool) // 2, len(pool) - 1})
    selected = [pool[index] for index in positions]
    if len(selected) < count:
        selected_ids = {
            (row["source_dialogue_id"], row["source_turn_index"])
            for row in selected
        }
        selected.extend(
            row
            for row in pool
            if (row["source_dialogue_id"], row["source_turn_index"])
            not in selected_ids
        )
    return selected[:count]


def _recent_context(
    dialogue: dict[str, Any], *, target_index: int, max_turns: int = 4
) -> list[dict[str, str]]:
    prior = list(dialogue.get("dialog") or [])[:target_index]
    return [
        {
            "speaker": str(turn.get("speaker", "")),
            "content": str(turn.get("content", "")),
        }
        for turn in prior[-max_turns:]
    ]


def _render_html(
    *, manifest: dict[str, Any], packet: list[dict[str, Any]]
) -> str:
    packet_json = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Bank V3 Core Review</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1180px; margin: 24px auto; padding: 0 18px; color: #17202a; }}
.card {{ border: 1px solid #ccd6dd; border-radius: 10px; margin: 18px 0; padding: 18px; }}
.meta {{ color: #52616b; font-size: 14px; }}
.box {{ background: #f6f8fa; border-radius: 6px; padding: 10px; margin: 8px 0; white-space: pre-wrap; }}
.example {{ border-left: 4px solid #829ab1; padding-left: 12px; margin: 14px 0; }}
label {{ display: inline-block; margin: 4px 12px 4px 0; }}
textarea {{ width: 100%; min-height: 70px; }}
button {{ padding: 10px 16px; margin: 8px 8px 8px 0; }}
.warning {{ background: #fff4ce; padding: 12px; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Strategy Bank V3：50 张核心卡审核</h1>
<p class="warning">卡片定义与来源映射必须分开判断。来源例子由 BGE outcome-blind 地弱匹配，
可能明显配错；配错不自动否定卡定义。不要依据例子质量、survey 或既有回复结果判断卡是否有效。</p>
<p>每张卡先审核定义，再分别判断三个来源例子是否真的体现该 submove。页面不展示
problem/emotion、survey、judge、生成结果或 test/external 信息。</p>
<button onclick="downloadResults()">导出 JSONL</button>
<span id="progress"></span>
<div id="cards"></div>
<script>
const manifest = {manifest_json};
const packet = {packet_json};
const fields = [
  "support_move_distinct","when_to_use_valid","when_not_to_use_valid",
  "mode_phase_goal_valid","burden_and_risk_valid","safe_general_technique",
  "approve_core_card"
];
function radio(name, value) {{
  return `<label><input type="radio" name="${{name}}" value="${{value}}" onchange="progress()"> ${{value}}</label>`;
}}
function render() {{
  const root = document.getElementById("cards");
  packet.forEach((row, i) => {{
    const div = document.createElement("section");
    div.className = "card";
    let html = `<h2>${{i+1}}. ${{row.strategy_family}} / ${{row.submove_id}}</h2>`;
    html += `<div class="box"><b>Move</b> ${{row.support_move}}\\n<b>Use</b> ${{row.when_to_use}}\\n<b>Do not</b> ${{row.when_not_to_use}}\\n<b>Mode/phase/goal</b> ${{row.compatible_support_modes.join(", ")}} / ${{row.compatible_dialogue_phases.join(", ")}} / ${{row.goal_types.join(", ")}}\\n<b>Burden/risk</b> ${{row.directive_burden}} / ${{row.risk_flags.join(", ")}}</div>`;
    fields.forEach(field => {{
      html += `<div><b>${{field}}</b> ${{radio(`c_${{i}}_${{field}}`,"yes")}} ${{radio(`c_${{i}}_${{field}}`,"no")}} ${{radio(`c_${{i}}_${{field}}`,"uncertain")}}</div>`;
    }});
    row.source_examples.forEach((ex, j) => {{
      const context = ex.recent_context.map(t => `${{t.speaker}}: ${{t.content}}`).join("\\n");
      html += `<div class="example"><div class="meta">弱映射例 ${{j+1}}；score=${{ex.top_cosine_score}}；margin=${{ex.top_vs_second_margin}}；confident=${{ex.assignment_confident}}</div><div class="box">${{context}}\\n\\nTARGET SUPPORTER RESPONSE:\\n${{ex.target_response}}</div><b>这个 target response 是否真的体现本 submove？</b> ${{radio(`e_${{i}}_${{j}}`,"yes")}} ${{radio(`e_${{i}}_${{j}}`,"partial")}} ${{radio(`e_${{i}}_${{j}}`,"no")}} ${{radio(`e_${{i}}_${{j}}`,"uncertain")}}</div>`;
    }});
    html += `<label>required_corrections / notes<textarea id="notes_${{i}}"></textarea></label>`;
    div.innerHTML = html;
    root.appendChild(div);
  }});
  progress();
}}
function checked(name) {{
  const item = document.querySelector(`input[name="${{name}}"]:checked`);
  return item ? item.value : null;
}}
function collect() {{
  return packet.map((row,i) => ({{
    protocol: manifest.protocol,
    card_id: row.card_id,
    submove_id: row.submove_id,
    ...Object.fromEntries(fields.map(f => [f, checked(`c_${{i}}_${{f}}`)])),
    source_example_fit: row.source_examples.map((ex,j) => ({{
      strategy_id: ex.strategy_id,
      judgment: checked(`e_${{i}}_${{j}}`)
    }})),
    required_corrections: document.getElementById(`notes_${{i}}`).value
  }}));
}}
function progress() {{
  const rows = collect();
  const complete = rows.filter(row =>
    fields.every(f => row[f] !== null) &&
    row.source_example_fit.every(ex => ex.judgment !== null)
  ).length;
  document.getElementById("progress").textContent = `${{complete}} / ${{packet.length}} cards complete`;
}}
function downloadResults() {{
  const text = collect().map(row => JSON.stringify(row)).join("\\n") + "\\n";
  const blob = new Blob([text], {{type:"application/jsonl"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "strategy_bank_v3_core_human_annotations.jsonl";
  a.click();
  URL.revokeObjectURL(a.href);
}}
render();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1",
    )
    parser.add_argument(
        "--raw-cards",
        type=Path,
        default=ROOT / "data/strategy/strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--esconv",
        type=Path,
        default=ROOT / "data/external/ESConv.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v3_core_review_candidate_v1",
    )
    parser.add_argument("--examples-per-card", type=int, default=3)
    args = parser.parse_args()

    cards_path = args.candidate_dir / "strategy_cards_v3_core_candidate.jsonl"
    mapping_path = (
        args.candidate_dir / "strategy_cards_v3_weak_source_mapping.jsonl"
    )
    cards = [dict(row) for row in iter_jsonl(cards_path)]
    mappings_by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(mapping_path):
        mappings_by_card[str(row["assigned_card_id"])].append(dict(row))
    raw_by_id = {
        str(row["strategy_id"]): dict(row)
        for row in iter_jsonl(args.raw_cards)
    }
    esconv = load_esconv(args.esconv)
    esconv_by_id = {
        f"esconv_{index:04d}": row for index, row in enumerate(esconv)
    }

    packet: list[dict[str, Any]] = []
    annotation_template: list[dict[str, Any]] = []
    for card in sorted(cards, key=lambda row: str(row["submove_id"])):
        selected = _select_examples(
            mappings_by_card[str(card["card_id"])],
            args.examples_per_card,
        )
        examples: list[dict[str, Any]] = []
        for row in selected:
            strategy_id = str(row["strategy_id"])
            raw = raw_by_id[strategy_id]
            dialogue_id = str(row["source_dialogue_id"])
            dialogue = esconv_by_id[dialogue_id]
            target_index = int(row["source_turn_index"])
            target = list(dialogue.get("dialog") or [])[target_index]
            if normalize_space(target.get("content", "")) != normalize_space(
                raw["example_response"]
            ):
                raise RuntimeError(f"raw/ESConv response drift: {strategy_id}")
            examples.append(
                {
                    "strategy_id": strategy_id,
                    "source_dialogue_id": dialogue_id,
                    "source_turn_index": target_index,
                    "recent_context": _recent_context(
                        dialogue, target_index=target_index
                    ),
                    "target_response": str(target["content"]),
                    "top_cosine_score": row["top_cosine_score"],
                    "top_vs_second_margin": row["top_vs_second_margin"],
                    "assignment_confident": row["assignment_confident"],
                }
            )
        packet_row = {
            "protocol": PROTOCOL,
            "card_id": card["card_id"],
            "submove_id": card["submove_id"],
            "strategy_family": card["strategy_family"],
            "support_move": card["support_move"],
            "when_to_use": card["when_to_use"],
            "when_not_to_use": card["when_not_to_use"],
            "compatible_support_modes": card["compatible_support_modes"],
            "compatible_dialogue_phases": card[
                "compatible_dialogue_phases"
            ],
            "goal_types": card["goal_types"],
            "directive_burden": card["directive_burden"],
            "risk_flags": card["risk_flags"],
            "source_examples": examples,
        }
        packet.append(packet_row)
        annotation_template.append(
            {
                "protocol": PROTOCOL,
                "card_id": card["card_id"],
                "submove_id": card["submove_id"],
                "support_move_distinct": None,
                "when_to_use_valid": None,
                "when_not_to_use_valid": None,
                "mode_phase_goal_valid": None,
                "burden_and_risk_valid": None,
                "safe_general_technique": None,
                "approve_core_card": None,
                "source_example_fit": [
                    {
                        "strategy_id": example["strategy_id"],
                        "judgment": None,
                    }
                    for example in examples
                ],
                "required_corrections": "",
            }
        )

    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_OUTCOME_BLIND_CORE_CARD_AND_SOURCE_SAMPLE_REVIEW",
        "card_count": len(packet),
        "source_examples_per_card": args.examples_per_card,
        "source_example_count": sum(
            len(row["source_examples"]) for row in packet
        ),
        "outcomes_or_surveys_exposed": False,
        "problem_or_emotion_metadata_exposed": False,
        "raw_examples_exposed_to_generator": False,
        "review_roles": {
            "card_definition": "intrinsic core-card qualification",
            "source_examples": (
                "audit weak BGE mapping only; example quality is not card utility"
            ),
        },
        "lineage": {
            "cards_sha256": sha256_file(cards_path),
            "mapping_sha256": sha256_file(mapping_path),
            "raw_cards_sha256": sha256_file(args.raw_cards),
            "esconv_sha256": sha256_file(args.esconv),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "review_manifest.json", manifest)
    write_jsonl(args.out_dir / "human_review_packet.jsonl", packet)
    write_jsonl(
        args.out_dir / "human_annotation_template.jsonl",
        annotation_template,
    )
    (args.out_dir / "human_review.html").write_text(
        _render_html(manifest=manifest, packet=packet),
        encoding="utf-8",
    )
    print(canonical_json({"output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
