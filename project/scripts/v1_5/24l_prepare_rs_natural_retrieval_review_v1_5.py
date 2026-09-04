#!/usr/bin/env python3
"""Prepare the outcome-blind natural Top-1 retrieval review for PM V1.5 RS.

The source universe contains the next ESConv supporter response and native
strategy label for other analyses.  This script deliberately projects every
row to pre-response dialogue fields before any eligibility, ranking, sampling,
or serialization step.  Hidden next-response fields are never read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import lexical_score, normalize_space
from metacom_pm.v1_5_strategy_rag_v4 import (
    eligible_families,
    observable_opportunity_flags,
    rank_applicable_v4_cards,
    selected_execution_profile,
    validate_v4_candidate_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-natural-retrieval-fit-review-v1"
MINIMUM_SCORE = 0.05
MAX_SELECTED_PER_CORE_SUBMOVE = 4
TARGETS = {
    "focused_clarification": 12,
    "grounded_reflection": 10,
    "welcomed_microstep": 8,
    "redundancy_negative_control": 2,
}
# The prior plan requested ten minimal states before checking the natural
# source capacity.  Visible-only feasibility found that the preregistered
# opportunity strata support seven safe, distinct minimal-profile candidates.
# This amendment occurs before generation or response outcomes exist.
MINIMAL_TARGETS = {
    "focused_clarification": 0,
    "grounded_reflection": 3,
    "welcomed_microstep": 1,
    "redundancy_negative_control": 2,
}
MINIMUM_MINIMAL_PROFILE_COUNT = 6

_EXPANDED_HIGH_STAKES_RE = re.compile(
    r"\b(?:suicid(?:e|al)|self[- ]?harm|kill(?:ing)?|"
    r"(?:want|wish|hope|feel like|ready to|going to)"
    r"(?:\s+\w+){0,5}\s+(?:die|dead|kill|hurt|harm)|"
    r"(?:want|wish|hope)(?:\s+\w+){0,5}\s+"
    r"(?:him|her|them|someone)\s+to\s+die|"
    r"can(?:not|'t)\s+go\s+on|giv(?:e|ing)\s+up|"
    r"immediate\s+danger|emergency|overdose|"
    r"(?:hit|beat|threaten|abuse|violence|weapon))\b",
    flags=re.IGNORECASE,
)
_DOMAIN_SPECIFIC_REQUEST_RE = re.compile(
    r"\b(?:doctor|medical|medicine|medication|dose|diagnos|symptom|"
    r"headaches?|lawyer|legal|court|police|tax|benefit|unemployment|"
    r"rent|eviction|sanitiz\w*|infection|covid|diet|protein)\b",
    flags=re.IGNORECASE,
)
_FACTUAL_QUESTION_RE = re.compile(
    r"\b(?:how\s+accurate|what\s+(?:are\s+)?the\s+rules|"
    r"how\s+much|how\s+many|qualif(?:y|ication)|"
    r"is\s+it\s+(?:true|safe|legal)|what\s+does\s+.+\s+mean)\b",
    flags=re.IGNORECASE,
)
_VALUE_OBSTACLE_RE = re.compile(
    r"\b(?:but|though|yet|part of me|on the one hand|"
    r"catch[- ]?22|want .{0,80} but|trying .{0,80} but)\b",
    flags=re.IGNORECASE,
)
_PRIOR_ATTEMPT_RE = re.compile(
    r"\b(?:already|have tried|i tried|kept trying|"
    r"reached out|talked to|spoke to|visited|office hours|"
    r"didn't help|did not help|not working|still)\b",
    flags=re.IGNORECASE,
)
_THIRD_PARTY_RE = re.compile(
    r"\b(?:he|she|they|him|her|them|his|hers|their|"
    r"boyfriend|girlfriend|husband|wife|partner|boss|"
    r"coworker|professor|mother|father|friend)\b",
    flags=re.IGNORECASE,
)


def _visible_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Return the only fields allowed to influence retrieval or selection."""

    dialogue = [
        {
            "speaker": str(turn["speaker"]),
            "content": normalize_space(turn["content"]),
        }
        for turn in row["recent_dialogue"]
        if str(turn.get("speaker")) in {"seeker", "supporter"}
        and normalize_space(turn.get("content"))
    ]
    return {
        "source_dialogue_id": str(row["source_dialogue_id"]),
        "source_turn_index": int(row["source_turn_index"]),
        "recent_dialogue": dialogue,
    }


def _seeker_texts(row: dict[str, Any]) -> list[str]:
    return [
        normalize_space(turn["content"])
        for turn in row["recent_dialogue"]
        if turn["speaker"] == "seeker"
    ]


def _latest_seeker(row: dict[str, Any]) -> str:
    rows = _seeker_texts(row)
    return rows[-1] if rows else ""


def _recent_seeker(row: dict[str, Any]) -> str:
    return " ".join(_seeker_texts(row)[-3:])


def _visible_dialogue_text(row: dict[str, Any]) -> str:
    return "\n".join(
        f"{turn['speaker']}: {turn['content']}"
        for turn in row["recent_dialogue"]
    )


def _expanded_high_stakes(row: dict[str, Any]) -> bool:
    return bool(_EXPANDED_HIGH_STAKES_RE.search(_recent_seeker(row)))


def _third_party_complexity(row: dict[str, Any]) -> bool:
    return len(_THIRD_PARTY_RE.findall(_recent_seeker(row))) >= 4


def _natural_query(
    row: dict[str, Any],
    flags: dict[str, bool],
) -> str:
    cue_names = [
        key.replace("_", " ")
        for key, value in flags.items()
        if value
        and key
        not in {
            "substantive",
            "pure_phatic",
            "routine_closing",
            "active_high_stakes",
            "explicit_stop",
        }
    ]
    return (
        "Choose one safe, topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cue_names) or 'none'}. "
        f"Recent seeker context: {_recent_seeker(row)}"
    )


def _stratum(
    *,
    row: dict[str, Any],
    flags: dict[str, bool],
    selected_card: dict[str, Any],
    applicability_v2: bool = False,
) -> str | None:
    latest = _latest_seeker(row)
    family = str(selected_card["strategy_family"])
    domain_specific = bool(_DOMAIN_SPECIFIC_REQUEST_RE.search(latest))
    factual_question = bool(_FACTUAL_QUESTION_RE.search(latest))
    third_party_complex = _third_party_complexity(row)

    if (
        flags["advice_welcome"]
        and family == "Providing Suggestions"
        and not domain_specific
        and not factual_question
    ):
        return "welcomed_microstep"
    if (
        flags["uncertainty_or_multi_concern"]
        and not flags["advice_welcome"]
        and (
            family == "Question"
            or (
                applicability_v2
                and family == "Restatement or Paraphrasing"
            )
        )
        and not factual_question
        and not third_party_complex
    ):
        return "focused_clarification"
    if (
        flags["emotion_visible"]
        and (
            family == "Reflection of feelings"
            or (
                applicability_v2
                and family
                in {
                    "Affirmation and Reassurance",
                    "Restatement or Paraphrasing",
                }
            )
        )
        and not third_party_complex
        and (
            bool(_VALUE_OBSTACLE_RE.search(_recent_seeker(row)))
            or bool(_PRIOR_ATTEMPT_RE.search(_recent_seeker(row)))
            or re.search(
                r"\b(?:i feel|i felt|i am|i'm|im)\b",
                latest,
                flags=re.IGNORECASE,
            )
        )
    ):
        return "grounded_reflection"
    if (
        family == "Restatement or Paraphrasing"
        and not flags["advice_welcome"]
        and not flags["uncertainty_or_multi_concern"]
        and not flags["emotion_visible"]
        and not flags["effort_or_progress_visible"]
        and len(latest.split()) >= 5
    ):
        return "redundancy_negative_control"
    return None


def _rank_cards(
    *,
    query: str,
    flags: dict[str, bool],
    cards: list[dict[str, Any]],
    row: dict[str, Any] | None = None,
    applicability_v2: bool = False,
) -> list[dict[str, Any]]:
    if applicability_v2:
        if row is None:
            raise ValueError("applicability_v2 ranking requires a visible row")
        return rank_applicable_v4_cards(
            query=query,
            current_user_text=_latest_seeker(row),
            recent_user_text=_recent_seeker(row),
            flags=flags,
            cards=cards,
        )
    profile = selected_execution_profile(flags)
    families = set(eligible_families(flags))
    candidates = [
        card
        for card in cards
        if card["execution_profile"] == profile
        and card["strategy_family"] in families
    ]
    ranked = sorted(
        (
            {
                "card_id": str(card["card_id"]),
                "core_submove_id": str(card["core_submove_id"]),
                "strategy_family": str(card["strategy_family"]),
                "execution_profile": str(card["execution_profile"]),
                "score": round(
                    lexical_score(query, str(card["retrieval_text"])),
                    8,
                ),
            }
            for card in candidates
        ),
        key=lambda item: (float(item["score"]), str(item["card_id"])),
        reverse=True,
    )
    return ranked


def _candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["retrieval_score"]),
        -float(row["top_vs_second_margin"]),
        str(row["source_dialogue_id"]),
        int(row["source_turn_index"]),
    )


def _select(
    candidates: list[dict[str, Any]],
    *,
    targets: dict[str, int] | None = None,
    minimal_targets: dict[str, int] | None = None,
    maximum_per_core: int = MAX_SELECTED_PER_CORE_SUBMOVE,
) -> list[dict[str, Any]]:
    targets = dict(targets or TARGETS)
    minimal_targets = dict(minimal_targets or MINIMAL_TARGETS)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_stratum[str(row["stratum"])].append(row)
    for rows in by_stratum.values():
        rows.sort(key=_candidate_sort_key)

    selected: list[dict[str, Any]] = []
    selected_dialogues: set[str] = set()
    selected_visible: set[str] = set()
    selected_core_counts: Counter[str] = Counter()

    def take(stratum: str, count: int, profile: str | None) -> None:
        if count <= 0:
            return
        taken = 0
        for row in by_stratum[stratum]:
            if profile is not None and row["execution_profile"] != profile:
                continue
            dialogue_id = str(row["source_dialogue_id"])
            visible_digest = str(row["visible_dialogue_sha256"])
            core_submove_id = str(row["selected_core_submove_id"])
            if dialogue_id in selected_dialogues or visible_digest in selected_visible:
                continue
            if selected_core_counts[core_submove_id] >= maximum_per_core:
                continue
            selected.append(row)
            selected_dialogues.add(dialogue_id)
            selected_visible.add(visible_digest)
            selected_core_counts[core_submove_id] += 1
            taken += 1
            if taken == count:
                return
        raise RuntimeError(
            f"insufficient visible-only candidates for {stratum}/{profile}: "
            f"needed {count}, found {taken}"
        )

    # Reserve scarce natural minimal-profile opportunities first.
    for stratum in (
        "welcomed_microstep",
        "grounded_reflection",
        "redundancy_negative_control",
        "focused_clarification",
    ):
        take(stratum, minimal_targets[stratum], "minimal")
    for stratum in (
        "focused_clarification",
        "welcomed_microstep",
        "grounded_reflection",
        "redundancy_negative_control",
    ):
        already = sum(row["stratum"] == stratum for row in selected)
        take(stratum, targets[stratum] - already, None)

    selected.sort(
        key=lambda row: (
            list(targets).index(str(row["stratum"])),
            str(row["source_dialogue_id"]),
        )
    )
    return selected


def _render_html(
    *,
    manifest: dict[str, Any],
    packet: list[dict[str, Any]],
) -> str:
    payload = json.dumps(
        {"manifest": manifest, "items": packet},
        ensure_ascii=False,
    ).replace("</", "<\\/")
    title = html.escape(
        f"PM V1.5 · {len(packet)} 组自然 Top‑1 召回适用性复核"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:24px;background:#f4f6f8;color:#17202a}}
.note,.item{{background:#fff;border:1px solid #d9dee6;border-radius:10px;padding:18px;margin:16px 0}}
.note{{border-left:5px solid #2463a8}} .dialogue{{white-space:pre-wrap;background:#edf2f6;padding:12px;border-radius:7px}}
.card{{background:#fafafa;border:1px solid #ccd4df;padding:12px;border-radius:8px;margin:10px 0}}
.meta{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}} .tag{{background:#e8eef6;border-radius:999px;padding:3px 9px;font-size:12px}}
label{{display:block;margin:9px 0}} select,textarea{{width:100%;box-sizing:border-box;padding:8px;margin:4px 0 8px}}
button{{padding:10px 15px}} .sticky{{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:2}}
details{{margin:10px 0}} h3{{margin-bottom:6px}} .warn{{color:#8a4b00}}
</style></head><body>
<h1>{title}</h1>
<div class="note">
本页只检查“检索器自然召回的 Top‑1 卡是否适用于当前可见对话”，不评价原始
ESConv 下一条回复，也不判断 RS 最终应开还是关。请依次判断：
Top‑1 是否符合 when_to_use；是否触发 when_not_to_use；Top‑2/3 是否明显更好。
不要因为卡片写得漂亮而批准，也不要推测生成结果。只有 top1_fit=yes 且
hard_exclusion_triggered=no 且 better_candidate=none 的项目才会进入 R0/RS 生成。
</div>
<div class="sticky"><button onclick="download()">导出 JSONL</button> <span id="progress"></span></div>
<div id="root"></div>
<script>
const DATA={payload};
const KEY="pm_v15_"+String(DATA.manifest.protocol).replace(/[^a-zA-Z0-9_]/g,"_");
let state=JSON.parse(localStorage.getItem(KEY)||"{{}}");
function esc(s){{return String(s??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));}}
function ensure(id){{if(!state[id])state[id]={{top1_fit:"",hard_exclusion_triggered:"",better_candidate:"",notes:""}};}}
function setv(id,k,v){{ensure(id);state[id][k]=v;localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function dialogue(item){{return item.visible_dialogue.map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");}}
function card(c){{return `<div class="card"><b>${{esc(c.strategy_family)}} · ${{esc(c.core_submove_id)}}</b>
<p><b>Support move:</b> ${{esc(c.support_move)}}</p>
<p><b>When to use:</b> ${{esc(c.when_to_use)}}</p>
<p><b>When not to use:</b> ${{esc(c.when_not_to_use)}}</p>
<p><b>Prompt guidance:</b> ${{esc(c.prompt_guidance)}}</p>
<p><b>Risk flags:</b> ${{esc((c.risk_flags||[]).join(", "))}}</p></div>`;}}
function render(){{document.getElementById("root").innerHTML=DATA.items.map((item,i)=>{{ensure(item.review_item_id);const s=state[item.review_item_id];return `<section class="item">
<h2>${{i+1}} / ${{DATA.items.length}}</h2>
<div class="meta"><span class="tag">${{esc(item.stratum)}}</span><span class="tag">${{esc(item.execution_profile)}}</span><span class="tag">score ${{item.retrieval_score.toFixed(3)}}</span><span class="tag">margin ${{item.top_vs_second_margin.toFixed(3)}}</span></div>
<div class="dialogue">${{esc(dialogue(item))}}</div>
<h3>自然召回 Top‑1</h3>${{card(item.top1_card)}}
<details><summary>查看 Top‑2/3 审计候选</summary>${{item.audit_alternatives.map((x,j)=>`<h4>Rank ${{j+2}} · score ${{x.score.toFixed(3)}}</h4>${{card(x.card)}}`).join("")}}</details>
<label>Top‑1 是否适用于当前状态？
<select onchange="setv('${{item.review_item_id}}','top1_fit',this.value)"><option value="">请选择</option>${{["yes","no","uncertain"].map(x=>`<option value="${{x}}" ${{s.top1_fit===x?"selected":""}}>${{x}}</option>`).join("")}}</select></label>
<label>是否触发卡片 hard exclusion / when_not_to_use？
<select onchange="setv('${{item.review_item_id}}','hard_exclusion_triggered',this.value)"><option value="">请选择</option>${{["yes","no","uncertain"].map(x=>`<option value="${{x}}" ${{s.hard_exclusion_triggered===x?"selected":""}}>${{x}}</option>`).join("")}}</select></label>
<label>是否存在明显更好的可见候选？
<select onchange="setv('${{item.review_item_id}}','better_candidate',this.value)"><option value="">请选择</option>${{["none","rank2","rank3","no_safe_card","uncertain"].map(x=>`<option value="${{x}}" ${{s.better_candidate===x?"selected":""}}>${{x}}</option>`).join("")}}</select></label>
<textarea placeholder="若拒绝或改选，请写一句可见证据" onchange="setv('${{item.review_item_id}}','notes',this.value)">${{esc(s.notes)}}</textarea>
</section>`;}}).join("");localStorage.setItem(KEY,JSON.stringify(state));progress();}}
function rows(){{return DATA.items.map(item=>{{ensure(item.review_item_id);return {{protocol:DATA.manifest.protocol,review_item_id:item.review_item_id,...state[item.review_item_id],annotator_id:""}};}});}}
function progress(){{const r=rows(),done=r.filter(x=>x.top1_fit&&x.hard_exclusion_triggered&&x.better_candidate).length;document.getElementById("progress").textContent=`完成 ${{done}}/${{r.length}}`;}}
function download(){{const b=new Blob([rows().map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="natural_retrieval_fit_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument(
        "--ranker",
        choices=("legacy_lexical", "card_applicability_v2"),
        default="legacy_lexical",
    )
    parser.add_argument(
        "--targets-json",
        default=json.dumps(TARGETS),
        help="JSON mapping of stratum to selected-state count.",
    )
    parser.add_argument(
        "--minimal-targets-json",
        default=json.dumps(MINIMAL_TARGETS),
        help="JSON mapping of stratum to reserved minimal-profile count.",
    )
    parser.add_argument(
        "--minimum-minimal-profile-count",
        type=int,
        default=MINIMUM_MINIMAL_PROFILE_COUNT,
    )
    parser.add_argument(
        "--maximum-selected-per-core-submove",
        type=int,
        default=MAX_SELECTED_PER_CORE_SUBMOVE,
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
            / "clean_train_strategy_universe.jsonl"
        ),
    )
    parser.add_argument(
        "--cards",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_strategy_rag_v4_candidate_v1"
            / "strategy_cards_v4_candidate.jsonl"
        ),
    )
    parser.add_argument(
        "--formal-esconv-test",
        type=Path,
        default=(
            ROOT
            / "data/esconv_test_v1_5_visible_v2_candidate"
            / "runtime_states.jsonl"
        ),
    )
    parser.add_argument(
        "--exclude-selected-states",
        nargs="*",
        type=Path,
        default=[
            ROOT
            / f"outputs/pm_v1_5_strategy_rag_v4_direct_effect_v{i}"
            / "selected_states.jsonl"
            for i in (1, 2, 3, 4)
        ],
    )
    parser.add_argument(
        "--exclude-dialogue-files",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Additional JSONL files containing source_dialogue_id or user_id; "
            "used to make a fresh holdout disjoint from development audits."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/pm_v1_5_rs_natural_retrieval_review_v1_candidate"
        ),
    )
    args = parser.parse_args()
    protocol = str(args.protocol)
    targets = {
        str(key): int(value)
        for key, value in json.loads(args.targets_json).items()
    }
    minimal_targets = {
        str(key): int(value)
        for key, value in json.loads(args.minimal_targets_json).items()
    }
    if set(targets) != set(TARGETS) or set(minimal_targets) != set(TARGETS):
        raise ValueError("target mappings must contain the four frozen strata")
    if any(value < 0 for value in targets.values()) or any(
        minimal_targets[key] < 0
        or minimal_targets[key] > targets[key]
        for key in targets
    ):
        raise ValueError("invalid target or minimal-target count")
    applicability_v2 = args.ranker == "card_applicability_v2"

    raw_rows = [dict(row) for row in iter_jsonl(args.universe)]
    if len(raw_rows) != 9148:
        raise ValueError(f"expected 9148 ESConv-train rows, found {len(raw_rows)}")
    # Critical firewall: from this point onward only visible projections exist.
    visible_rows = [_visible_projection(row) for row in raw_rows]
    del raw_rows

    cards = validate_v4_candidate_cards(
        [dict(row) for row in iter_jsonl(args.cards)]
    )
    active_cards = [
        card
        for card in cards
        if card["source_support"]["provisional_source_support_pass"]
    ]
    excluded_dialogues = {
        str(row["user_id"])
        for path in args.exclude_selected_states
        for row in iter_jsonl(path)
    }
    excluded_dialogues.update(
        str(row.get("source_dialogue_id") or row.get("user_id"))
        for path in args.exclude_dialogue_files
        for row in iter_jsonl(path)
        if row.get("source_dialogue_id") or row.get("user_id")
    )
    formal_test_dialogues = {
        str(row["user_id"]) for row in iter_jsonl(args.formal_esconv_test)
    }
    train_dialogues = {
        str(row["source_dialogue_id"]) for row in visible_rows
    }
    if train_dialogues & formal_test_dialogues:
        raise RuntimeError("ESConv train universe overlaps formal ESConv test")

    candidates: list[dict[str, Any]] = []
    audit_counts = Counter()
    seen_visible: set[tuple[str, str]] = set()
    for row in visible_rows:
        dialogue_id = str(row["source_dialogue_id"])
        if dialogue_id in excluded_dialogues:
            audit_counts["excluded_prior_dialogue"] += 1
            continue
        latest = _latest_seeker(row)
        if not latest:
            audit_counts["no_visible_seeker"] += 1
            continue
        visible_text = _visible_dialogue_text(row)
        visible_key = (dialogue_id, sha256_text(visible_text))
        if visible_key in seen_visible:
            audit_counts["duplicate_visible_state"] += 1
            continue
        seen_visible.add(visible_key)
        flags = observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=_recent_seeker(row),
        )
        if (
            flags["pure_phatic"]
            or flags["explicit_stop"]
            or flags["active_high_stakes"]
            or _expanded_high_stakes(row)
            or not flags["substantive"]
        ):
            audit_counts["hard_off"] += 1
            continue
        families = eligible_families(flags)
        if not families:
            audit_counts["no_eligible_family"] += 1
            continue
        query = _natural_query(row, flags)
        ranked = _rank_cards(
            query=query,
            flags=flags,
            cards=active_cards,
            row=row,
            applicability_v2=applicability_v2,
        )
        if not ranked or float(ranked[0]["score"]) < MINIMUM_SCORE:
            audit_counts["score_floor_abstention"] += 1
            continue
        top = ranked[0]
        stratum = _stratum(
            row=row,
            flags=flags,
            selected_card=next(
                card
                for card in active_cards
                if card["card_id"] == top["card_id"]
            ),
            applicability_v2=applicability_v2,
        )
        if stratum is None:
            audit_counts["outside_positive_expansion_strata"] += 1
            continue
        second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
        state_id = "state_" + stable_hex(
            protocol,
            dialogue_id,
            str(row["source_turn_index"]),
            visible_text,
            n=24,
        )
        candidates.append(
            {
                "state_id": state_id,
                "source_dialogue_id": dialogue_id,
                "source_turn_index": int(row["source_turn_index"]),
                "visible_dialogue": row["recent_dialogue"],
                "visible_dialogue_sha256": sha256_text(visible_text),
                "current_user_text": latest,
                "observable_flags": flags,
                "expanded_high_stakes": False,
                "multi_entity_third_party_complexity": _third_party_complexity(
                    row
                ),
                "value_obstacle_cue": bool(
                    _VALUE_OBSTACLE_RE.search(_recent_seeker(row))
                ),
                "prior_attempt_cue": bool(
                    _PRIOR_ATTEMPT_RE.search(_recent_seeker(row))
                ),
                "eligible_families": list(families),
                "execution_profile": selected_execution_profile(flags),
                "natural_query": query,
                "stratum": stratum,
                "selected_card_id": str(top["card_id"]),
                "selected_core_submove_id": str(top["core_submove_id"]),
                "selected_strategy_family": str(top["strategy_family"]),
                "retrieval_score": float(top["score"]),
                "top_vs_second_margin": round(
                    float(top["score"]) - second_score,
                    8,
                ),
                "top3": ranked[:3],
            }
        )

    selected = _select(
        candidates,
        targets=targets,
        minimal_targets=minimal_targets,
        maximum_per_core=int(args.maximum_selected_per_core_submove),
    )
    selected_dialogues = {
        str(row["source_dialogue_id"]) for row in selected
    }
    selected_profiles = Counter(
        str(row["execution_profile"]) for row in selected
    )
    selected_strata = Counter(str(row["stratum"]) for row in selected)
    expected_selected_count = sum(targets.values())
    checks = {
        "exact_target_state_count": len(selected) == expected_selected_count,
        "one_state_per_dialogue": len(selected_dialogues) == len(selected),
        "target_strata_met": dict(selected_strata) == targets,
        "minimum_minimal_profile_met": (
            selected_profiles["minimal"]
            >= int(args.minimum_minimal_profile_count)
        ),
        "all_selected_cards_pass_weak_source_gate": all(
            next(
                card
                for card in active_cards
                if card["card_id"] == row["selected_card_id"]
            )["source_support"]["provisional_source_support_pass"]
            for row in selected
        ),
        "no_prior_direct_effect_dialogue_overlap": not (
            selected_dialogues & excluded_dialogues
        ),
        "no_formal_esconv_test_overlap": not (
            selected_dialogues & formal_test_dialogues
        ),
        "query_contains_no_target_submove_id": all(
            "target_submove" not in row["natural_query"].casefold()
            and row["selected_core_submove_id"].casefold()
            not in row["natural_query"].casefold()
            for row in selected
        ),
        "hidden_next_response_fields_absent": all(
            not (
                {
                    "supporter_response",
                    "strategy_label",
                    "strategy_id",
                    "problem_type",
                    "emotion_type",
                    "experience_type",
                }
                & set(row)
            )
            for row in selected
        ),
        "top1_only_treatment_candidate": all(
            bool(row["selected_card_id"]) for row in selected
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(canonical_json(checks))

    card_by_id = {str(card["card_id"]): card for card in active_cards}
    packet: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    private_audit: list[dict[str, Any]] = []
    for row in selected:
        review_id = "rs_retrieval_" + stable_hex(
            protocol,
            str(row["state_id"]),
            str(row["selected_card_id"]),
            n=20,
        )

        def visible_card(card_id: str) -> dict[str, Any]:
            card = card_by_id[card_id]
            return {
                "card_id": str(card["card_id"]),
                "core_submove_id": str(card["core_submove_id"]),
                "strategy_family": str(card["strategy_family"]),
                "execution_profile": str(card["execution_profile"]),
                "support_move": str(card["support_move"]),
                "when_to_use": str(card["when_to_use"]),
                "when_not_to_use": str(card["when_not_to_use"]),
                "prompt_guidance": str(card["prompt_guidance"]),
                "risk_flags": list(card["risk_flags"]),
                "source_support_dialogues": int(
                    card["source_support"]["confident_weak_assigned_dialogues"]
                ),
                "source_assignment_is_gold": False,
            }

        alternatives = [
            {
                "score": float(item["score"]),
                "card": visible_card(str(item["card_id"])),
            }
            for item in row["top3"][1:3]
        ]
        packet.append(
            {
                "protocol": protocol,
                "review_item_id": review_id,
                "state_id": row["state_id"],
                "stratum": row["stratum"],
                "execution_profile": row["execution_profile"],
                "visible_dialogue": row["visible_dialogue"],
                "observable_flags": row["observable_flags"],
                "retrieval_score": row["retrieval_score"],
                "top_vs_second_margin": row["top_vs_second_margin"],
                "top1_card": visible_card(str(row["selected_card_id"])),
                "audit_alternatives": alternatives,
            }
        )
        templates.append(
            {
                "protocol": protocol,
                "review_item_id": review_id,
                "top1_fit": None,
                "hard_exclusion_triggered": None,
                "better_candidate": None,
                "notes": "",
                "annotator_id": "",
            }
        )
        private_audit.append(
            {
                **row,
                "review_item_id": review_id,
            }
        )

    manifest = {
        "protocol": protocol,
        "status": "READY_FOR_LIGHTWEIGHT_TOP1_CARD_FIT_REVIEW",
        "scope": "ESConv_train_only_outcome_blind_natural_retrieval",
        "review_item_count": len(packet),
        "independent_dialogue_count": len(selected_dialogues),
        "stratum_counts": dict(sorted(selected_strata.items())),
        "profile_counts": dict(sorted(selected_profiles.items())),
        "ranker": (
            "family_and_card_applicability_then_term_frequency_cosine"
            if applicability_v2
            else "family_eligible_term_frequency_cosine"
        ),
        "ranker_uses_card_level_applicability": applicability_v2,
        "minimum_score": MINIMUM_SCORE,
        "injection_top_k_after_approval": 1,
        "top_k_visible_for_audit": 3,
        "maximum_selected_states_per_core_submove": (
            int(args.maximum_selected_per_core_submove)
        ),
        "candidate_card_count": len(cards),
        "active_weak_source_pass_variants": len(active_cards),
        "active_core_submoves": len(
            {str(card["core_submove_id"]) for card in active_cards}
        ),
        "visible_only_projection": True,
        "hidden_next_response_or_native_strategy_used": False,
        "selection_uses_generation_or_judge_outcomes": False,
        "minimal_profile_amendment": {
            "prior_unaudited_target": 10,
            "revised_target": sum(minimal_targets.values()),
            "minimum_gate": int(args.minimum_minimal_profile_count),
            "observed_selected": selected_profiles["minimal"],
            "reason": (
                "Visible-only natural-source feasibility before generation: "
                "the fixed positive-opportunity strata did not contain ten "
                "safe, distinct minimal-profile candidates."
            ),
        },
        "checks": checks,
        "source_lineage": {
            "universe": str(args.universe.relative_to(ROOT)),
            "universe_sha256": sha256_file(args.universe),
            "cards": str(args.cards.relative_to(ROOT)),
            "cards_sha256": sha256_file(args.cards),
            "formal_esconv_test": str(args.formal_esconv_test.relative_to(ROOT)),
            "formal_esconv_test_sha256": sha256_file(args.formal_esconv_test),
            "excluded_selected_state_files": [
                str(path.relative_to(ROOT))
                for path in args.exclude_selected_states
            ],
            "excluded_dialogue_files": [
                str(path.resolve().relative_to(ROOT))
                for path in args.exclude_dialogue_files
            ],
        },
        "next_gate": (
            "Only top1_fit=yes, hard_exclusion_triggered=no, and "
            "better_candidate=none proceed to matched R0/RS generation."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "review_manifest.json", manifest)
    write_jsonl(args.out_dir / "retrieval_review_packet.jsonl", packet)
    write_jsonl(args.out_dir / "human_annotation_template.jsonl", templates)
    write_jsonl(args.out_dir / "private_selection_audit.jsonl", private_audit)
    (args.out_dir / "human_retrieval_review.html").write_text(
        _render_html(manifest=manifest, packet=packet),
        encoding="utf-8",
    )
    write_json(
        args.out_dir / "selection_report.json",
        {
            **manifest,
            "visible_universe_rows": len(visible_rows),
            "visible_universe_dialogues": len(train_dialogues),
            "excluded_prior_dialogues": len(excluded_dialogues),
            "eligible_stratified_candidate_rows": len(candidates),
            "eligible_stratified_candidate_dialogues": len(
                {str(row["source_dialogue_id"]) for row in candidates}
            ),
            "candidate_stratum_rows": dict(
                sorted(Counter(row["stratum"] for row in candidates).items())
            ),
            "candidate_profile_rows": dict(
                sorted(
                    Counter(
                        row["execution_profile"] for row in candidates
                    ).items()
                )
            ),
            "audit_exclusion_counts": dict(sorted(audit_counts.items())),
            "packet_sha256": sha256_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in packet
                )
                + "\n"
            ),
        },
    )
    print(
        {
            "protocol": protocol,
            "status": manifest["status"],
            "review_items": len(packet),
            "strata": dict(selected_strata),
            "profiles": dict(selected_profiles),
            "checks_passed": all(checks.values()),
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
