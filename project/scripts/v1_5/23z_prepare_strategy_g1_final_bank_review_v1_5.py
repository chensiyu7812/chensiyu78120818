#!/usr/bin/env python3
"""Prepare the bounded human review for the data-derived G1 Bank."""

from __future__ import annotations

import hashlib
from html import escape
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-strategy-g1-final-bank-human-review-v1"
CANDIDATE_MOVES = [
    "AM01_invite_open_expression",
    "AM02_ask_one_focused_clarification",
    "AM04_tentative_paraphrase_check",
    "AM05_grounded_validation",
    "AM07_acknowledge_effort_strength_or_resource",
    "AM10_offer_one_optional_micro_step",
    "AM14_supportive_transition",
    "AM15_explore_interpersonal_boundary",
]
CARD_SPECS: dict[str, dict[str, Any]] = {
    "AM01_invite_open_expression": {
        "when_to_use": (
            "Use when the user's concern, need, or dilemma is not yet clear "
            "and an open invitation would let them choose what to share."
        ),
        "when_not_to_use": (
            "Do not repeat the invitation after the concern is already clear, "
            "stack it with other questions, or use it to delay a direct "
            "low-risk request that can already be answered."
        ),
        "compatible_support_modes": ["listen", "explore"],
        "compatible_dialogue_phases": ["opening", "exploration"],
        "goal_types": ["be_heard", "make_sense"],
        "directive_burden": "light",
        "risk_flags": ["question_burden", "repeated_question"],
    },
    "AM02_ask_one_focused_clarification": {
        "when_to_use": (
            "Use during exploration when one missing feeling, fact, timing "
            "detail, or goal would materially improve understanding."
        ),
        "when_not_to_use": (
            "Do not stack questions, repeat already answered facts, or probe "
            "when the user explicitly asks only to be heard."
        ),
        "compatible_support_modes": ["explore"],
        "compatible_dialogue_phases": ["exploration"],
        "goal_types": ["make_sense", "decide"],
        "directive_burden": "light",
        "risk_flags": ["question_burden", "repeated_question"],
    },
    "AM04_tentative_paraphrase_check": {
        "when_to_use": (
            "Use when accurate understanding and room for correction matter "
            "more than adding advice or new content."
        ),
        "when_not_to_use": (
            "Do not introduce motives, diagnoses, facts, or stronger "
            "conclusions than the user stated; do not present the paraphrase "
            "as certain."
        ),
        "compatible_support_modes": ["listen", "explore"],
        "compatible_dialogue_phases": ["exploration", "comforting"],
        "goal_types": ["be_heard", "make_sense"],
        "directive_burden": "none",
        "risk_flags": ["unsupported_inference"],
    },
    "AM05_grounded_validation": {
        "when_to_use": (
            "Use when the visible dialogue directly supports a feeling, "
            "difficulty, loss, or tension that can be acknowledged briefly."
        ),
        "when_not_to_use": (
            "Do not invent or intensify emotion, diagnose the user, minimize "
            "the concern, promise an outcome, or validate a harmful factual "
            "claim as true."
        ),
        "compatible_support_modes": ["listen", "comfort_reassure", "explore"],
        "compatible_dialogue_phases": ["exploration", "comforting"],
        "goal_types": ["be_heard", "stabilize", "make_sense"],
        "directive_burden": "none",
        "risk_flags": [
            "emotion_overreach",
            "unsupported_inference",
            "false_reassurance",
        ],
    },
    "AM07_acknowledge_effort_strength_or_resource": {
        "when_to_use": (
            "Use when the visible dialogue shows a specific effort, coping "
            "resource, constructive choice, or strength worth naming."
        ),
        "when_not_to_use": (
            "Do not use generic praise, infer an unshown trait, claim that an "
            "effort will succeed, or praise risky or harmful behavior."
        ),
        "compatible_support_modes": [
            "comfort_reassure",
            "explore",
            "light_guidance",
        ],
        "compatible_dialogue_phases": [
            "exploration",
            "comforting",
            "action",
        ],
        "goal_types": ["stabilize", "reinforce_agency"],
        "directive_burden": "none",
        "risk_flags": [
            "unsupported_inference",
            "generic_praise",
            "false_reassurance",
        ],
    },
    "AM10_offer_one_optional_micro_step": {
        "when_to_use": (
            "Use when advice or action help is explicitly welcomed, the goal "
            "is clear, and one low-risk, low-burden step fits that goal."
        ),
        "when_not_to_use": (
            "Do not use under a listen-only boundary, stack multiple tasks, "
            "prescribe a high-stakes action, or present the step as mandatory "
            "or guaranteed to work."
        ),
        "compatible_support_modes": ["light_guidance"],
        "compatible_dialogue_phases": ["action"],
        "goal_types": ["take_one_step"],
        "directive_burden": "light",
        "risk_flags": [
            "unsolicited_advice",
            "excessive_directiveness",
            "domain_claim",
        ],
    },
    "AM14_supportive_transition": {
        "when_to_use": (
            "Use when the user signals fatigue, completion, a wish to change "
            "focus, or a need to choose whether to continue or close."
        ),
        "when_not_to_use": (
            "Do not close abruptly while the user is still engaging, ignore "
            "an active high-stakes cue, or promise permanent availability."
        ),
        "compatible_support_modes": ["listen", "comfort_reassure", "explore"],
        "compatible_dialogue_phases": ["transition", "closing"],
        "goal_types": ["respect_boundary", "reduce_burden"],
        "directive_burden": "none",
        "risk_flags": [
            "premature_closure",
            "availability_overclaim",
            "unhandled_high_stakes_cue",
        ],
    },
    "AM15_explore_interpersonal_boundary": {
        "when_to_use": (
            "Use when the user raises a specific interpersonal limit and "
            "welcomes help clarifying what boundary would fit their goal."
        ),
        "when_not_to_use": (
            "Do not decide the relationship for the user, infer another "
            "person's motives, prescribe separation, or use ordinary boundary "
            "guidance in an abuse, violence, or other high-stakes situation."
        ),
        "compatible_support_modes": ["explore", "light_guidance"],
        "compatible_dialogue_phases": ["exploration", "action"],
        "goal_types": ["clarify_boundary", "decide"],
        "directive_burden": "light",
        "risk_flags": [
            "values_imposition",
            "excessive_directiveness",
            "high_stakes_boundary",
        ],
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _render_html(manifest: dict[str, Any], packet: list[dict[str, Any]]) -> str:
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    packet_json = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 G1 最终 Bank 候选审核</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1160px;margin:24px auto;padding:0 18px;color:#17202a}}
.card{{border:1px solid #ccd6dd;border-radius:10px;margin:20px 0;padding:18px}}
.box{{background:#f6f8fa;border-radius:6px;padding:10px;margin:8px 0;white-space:pre-wrap}}
.warn{{background:#fff4ce;padding:12px;border-radius:6px}} .meta{{color:#52616b;font-size:14px}}
.example{{border-left:4px solid #829ab1;padding-left:12px;margin:14px 0}}
label{{display:inline-block;margin:4px 12px 4px 0}} textarea{{width:100%;min-height:64px}}
button{{padding:10px 16px;margin:8px 8px 8px 0}}
</style></head><body>
<h1>PM V1.5 G1：8 张最终 Bank 候选审核</h1>
<p class="warn">这是来源与卡定义审核，不是回复质量评测，也不是 PM 开关标签。页面未展示
弱模型置信度、ESConv 原标签、problem/emotion、survey、judge 或外部结果。每个例子必须
同时判断 move 是否真实出现、是否漏掉硬排除。AM07/AM15 的来源数量例外必须单独接受，
不能因为想覆盖类别而默认批准。</p>
<button onclick="downloadResults()">导出 JSONL</button> <span id="progress"></span>
<div id="cards"></div>
<script>
const manifest={manifest_json}; const packet={packet_json};
const fields=["support_move_clear","when_to_use_valid","when_not_to_use_valid",
"eligibility_metadata_valid","burden_and_risk_valid","safe_topic_agnostic_technique",
"source_gate_or_exception_acceptable","approve_for_g2_candidate"];
function radio(name,value){{return `<label><input type="radio" name="${{name}}" value="${{value}}" onchange="progress()"> ${{value}}</label>`}}
function checked(name){{const x=document.querySelector(`input[name="${{name}}"]:checked`);return x?x.value:null}}
function render(){{const root=document.getElementById("cards");packet.forEach((row,i)=>{{
 const d=document.createElement("section");d.className="card";
 let h=`<h2>${{i+1}}. ${{row.move_id}} — ${{row.name}}</h2>`;
 h+=`<div class="meta">source gate: ${{row.source_gate_status}}；clean dialogues=${{row.clean_weak_source_dialogues}}；problem types=${{row.problem_type_count_audit_only}}；emotion types=${{row.emotion_type_count_audit_only}}；max problem share=${{row.maximum_single_problem_type_share_audit_only}}</div>`;
 if(row.exception_rationale)h+=`<p class="warn"><b>Narrow exception:</b> ${{row.exception_rationale}}</p>`;
 h+=`<div class="box"><b>Move</b> ${{row.support_move}}\\n<b>Use</b> ${{row.when_to_use}}\\n<b>Do not</b> ${{row.when_not_to_use}}\\n<b>Mode / phase / goal</b> ${{row.compatible_support_modes.join(", ")}} / ${{row.compatible_dialogue_phases.join(", ")}} / ${{row.goal_types.join(", ")}}\\n<b>Burden / risk</b> ${{row.directive_burden}} / ${{row.risk_flags.join(", ")}}</div>`;
 fields.forEach(f=>h+=`<div><b>${{f}}</b> ${{radio(`c_${{i}}_${{f}}`,"yes")}} ${{radio(`c_${{i}}_${{f}}`,"no")}} ${{radio(`c_${{i}}_${{f}}`,"uncertain")}}</div>`);
 row.source_examples.forEach((ex,j)=>{{const ctx=ex.recent_visible_dialogue.map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");
 h+=`<div class="example"><div class="meta">固定 literal source ${{j+1}} / dialogue=${{ex.source_dialogue_id}}</div><div class="box">${{ctx}}\\n\\nTARGET SUPPORTER RESPONSE:\\n${{ex.supporter_response_to_label}}</div><div><b>move_fit</b> ${{radio(`m_${{i}}_${{j}}`,"yes")}} ${{radio(`m_${{i}}_${{j}}`,"partial")}} ${{radio(`m_${{i}}_${{j}}`,"no")}} ${{radio(`m_${{i}}_${{j}}`,"uncertain")}}</div><div><b>missed_hard_exclusion</b> ${{radio(`x_${{i}}_${{j}}`,"yes")}} ${{radio(`x_${{i}}_${{j}}`,"no")}} ${{radio(`x_${{i}}_${{j}}`,"uncertain")}}</div></div>`}});
 h+=`<label>required_corrections / notes<textarea id="notes_${{i}}"></textarea></label>`;d.innerHTML=h;root.appendChild(d)}});progress()}}
function collect(){{return packet.map((row,i)=>({{protocol:manifest.protocol,card_id:row.card_id,move_id:row.move_id,...Object.fromEntries(fields.map(f=>[f,checked(`c_${{i}}_${{f}}`)])),source_example_review:row.source_examples.map((ex,j)=>({{blind_item_id:ex.blind_item_id,move_fit:checked(`m_${{i}}_${{j}}`),missed_hard_exclusion:checked(`x_${{i}}_${{j}}`)}})),required_corrections:document.getElementById(`notes_${{i}}`).value}}))}}
function progress(){{const r=collect();const n=r.filter(x=>fields.every(f=>x[f]!==null)&&x.source_example_review.every(e=>e.move_fit!==null&&e.missed_hard_exclusion!==null)).length;document.getElementById("progress").textContent=`${{n}} / ${{packet.length}} cards complete`}}
function downloadResults(){{const text=collect().map(x=>JSON.stringify(x)).join("\\n")+"\\n";const b=new Blob([text],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="pm_v1_5_strategy_g1_final_bank_human_annotations.jsonl";a.click();URL.revokeObjectURL(a.href)}}
render();
</script></body></html>"""


def main() -> None:
    codebook = _read_json(
        ROOT / "data/strategy/pm_v1_5_strategy_atomic_move_codebook_v1.json"
    )
    moves = {str(row["move_id"]): row for row in codebook["moves"]}
    coverage = _read_json(
        ROOT
        / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1"
        / "coverage_report.json"
    )
    fixed_examples = _read_jsonl(
        ROOT
        / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1"
        / "fixed_literal_source_human_audit.jsonl"
    )
    example_by_move: dict[str, list[dict[str, Any]]] = {
        move_id: [] for move_id in CANDIDATE_MOVES
    }
    for row in fixed_examples:
        move_id = str(row["move_id"])
        if move_id in example_by_move:
            example_by_move[move_id].append(row)

    public_by_id: dict[str, dict[str, Any]] = {}
    for path in [
        ROOT
        / "outputs/pm_v1_5_strategy_g1_weak_label_pilot_v1"
        / "public_packet.jsonl",
        ROOT
        / "outputs/pm_v1_5_strategy_g1_stage1_weak_labels_v1"
        / "public_packet.jsonl",
        ROOT
        / "outputs/pm_v1_5_strategy_g1_adaptive_wave2_v1"
        / "public_packet.jsonl",
        ROOT
        / "outputs/pm_v1_5_strategy_g1_coverage_completion_wave_v1"
        / "public_packet.jsonl",
    ]:
        for row in _read_jsonl(path):
            public_by_id[str(row["blind_item_id"])] = row

    cards: list[dict[str, Any]] = []
    packet: list[dict[str, Any]] = []
    template: list[dict[str, Any]] = []
    for move_id in CANDIDATE_MOVES:
        report = coverage["per_move"][move_id]
        clean_count = int(
            report["clean_weak_source_independent_dialogues_not_human_gold"]
        )
        is_exception = move_id in {
            "AM07_acknowledge_effort_strength_or_resource",
            "AM15_explore_interpersonal_boundary",
        }
        if not is_exception and clean_count < 20:
            raise ValueError(f"{move_id}: standard candidate misses source gate")
        if is_exception and clean_count < 10:
            raise ValueError(f"{move_id}: narrow candidate lacks even 10 sources")
        examples = example_by_move[move_id]
        if len(examples) != 5:
            raise ValueError(f"{move_id}: expected five fixed exemplars")
        source_examples = []
        for example in examples:
            item_id = str(example["blind_item_id"])
            public = public_by_id[item_id]
            source_examples.append(
                {
                    "blind_item_id": item_id,
                    "source_dialogue_id": example["source_dialogue_id"],
                    "source_turn_index": example["source_turn_index"],
                    "recent_visible_dialogue": public["recent_visible_dialogue"],
                    "supporter_response_to_label": public[
                        "supporter_response_to_label"
                    ],
                }
            )
        move = moves[move_id]
        spec = CARD_SPECS[move_id]
        card_id = "strategy_g1_" + hashlib.sha256(
            f"{PROTOCOL}|{move_id}".encode()
        ).hexdigest()[:20]
        exception_rationale = ""
        if is_exception:
            exception_rationale = (
                f"Narrow move with {clean_count} independent clean weak-source "
                f"dialogues, {report['problem_type_count_audit_only']} problem "
                f"types, {report['emotion_type_count_audit_only']} emotion "
                "types, and max single-problem share "
                f"{report['maximum_single_problem_type_share_audit_only']}. "
                "Count is below the standard 20-dialogue gate; approve only "
                "if the narrow definition and fixed source examples justify "
                "the exception."
            )
        card = {
            "protocol": PROTOCOL,
            "card_id": card_id,
            "move_id": move_id,
            "name": move["name"],
            "support_move": move["definition"],
            **spec,
            "source_gate_status": (
                "NARROW_EXCEPTION_REQUIRES_HUMAN_APPROVAL"
                if is_exception
                else "STANDARD_20_DIALOGUE_GATE_PASS"
            ),
            "exception_rationale": exception_rationale,
            "clean_weak_source_dialogues": clean_count,
            "problem_type_count_audit_only": report[
                "problem_type_count_audit_only"
            ],
            "emotion_type_count_audit_only": report[
                "emotion_type_count_audit_only"
            ],
            "maximum_single_problem_type_share_audit_only": report[
                "maximum_single_problem_type_share_audit_only"
            ],
            "eligible_for_g2": False,
            "raw_source_responses_exposed_to_generator": False,
        }
        cards.append(card)
        packet.append({**card, "source_examples": source_examples})
        template.append(
            {
                "protocol": PROTOCOL,
                "card_id": card_id,
                "move_id": move_id,
                "support_move_clear": None,
                "when_to_use_valid": None,
                "when_not_to_use_valid": None,
                "eligibility_metadata_valid": None,
                "burden_and_risk_valid": None,
                "safe_topic_agnostic_technique": None,
                "source_gate_or_exception_acceptable": None,
                "approve_for_g2_candidate": None,
                "source_example_review": [
                    {
                        "blind_item_id": row["blind_item_id"],
                        "move_fit": None,
                        "missed_hard_exclusion": None,
                    }
                    for row in source_examples
                ],
                "required_corrections": "",
            }
        )

    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_BOUNDED_OUTCOME_BLIND_HUMAN_REVIEW",
        "candidate_card_count": len(cards),
        "standard_gate_card_count": 6,
        "narrow_exception_card_count": 2,
        "fixed_source_examples_per_card": 5,
        "fixed_source_example_count": 40,
        "weak_labels_are_gold": False,
        "problem_emotion_labels_or_values_visible_to_human": False,
        "aggregate_problem_emotion_diversity_counts_visible_to_human": True,
        "survey_quality_judge_or_external_outcomes_visible_to_human": False,
        "raw_source_responses_exposed_to_generator": False,
        "post_review_rule": (
            "Only human-approved cards with no unresolved missed hard "
            "exclusion enter G2. Rejected cards are not replaced by new "
            "weak-source expansion."
        ),
    }
    out_dir = ROOT / "outputs/pm_v1_5_strategy_g1_final_bank_review_candidate_v1"
    _write_json(out_dir / "review_manifest.json", manifest)
    _write_jsonl(out_dir / "strategy_cards_g1_candidate.jsonl", cards)
    _write_jsonl(out_dir / "human_review_packet.jsonl", packet)
    _write_jsonl(out_dir / "human_annotation_template.jsonl", template)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "human_review.html").write_text(
        _render_html(manifest, packet),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out_dir), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
