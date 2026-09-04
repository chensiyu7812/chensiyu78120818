#!/usr/bin/env python3
"""Prepare the single bounded H1 source-relink review for 36 RAG cores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, stable_hex
from metacom_pm.text import normalize_space


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h1-source-relink-human-review-v1"
DEFAULT_NLI = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--cross-encoder--nli-deberta-v3-base/snapshots/"
    "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "has", "have", "if", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "use", "user", "when", "with", "would",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_space(text).casefold())
        if len(token) > 2 and token not in STOPWORDS
    }


def _lexical_score(response: str, core: dict[str, Any]) -> float:
    left = _tokens(response)
    right = _tokens(
        " ".join(
            [
                str(core["support_move"]),
                str(core["when_to_use"]),
                str(core["when_not_to_use"]),
            ]
        )
    )
    return len(left & right) / max(1, len(left | right))


def _normalized_response(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_space(text).casefold()).strip()


def _latest_visible_is_seeker(row: dict[str, Any]) -> bool:
    dialogue = list(row.get("recent_dialogue") or [])
    return bool(dialogue) and str(dialogue[-1].get("speaker")) == "seeker"


def _eligible_source(
    row: dict[str, Any],
    *,
    family: str,
    excluded_dialogues: set[str],
) -> bool:
    response = normalize_space(row.get("supporter_response", ""))
    return (
        str(row.get("strategy_label")) == family
        and str(row.get("source_dialogue_id")) not in excluded_dialogues
        and _latest_visible_is_seeker(row)
        and not bool(row.get("domain_claim_keyword_flag"))
        and 4 <= int(row.get("word_count") or 0) <= 80
        and len(_tokens(response)) >= 2
    )


def _load_h1_status(
    *,
    packet_path: Path,
    annotations_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    annotations = {
        str(row["review_item_id"]): dict(row)
        for row in iter_jsonl(annotations_path)
        if row.get("item_type") == "card"
    }
    approved: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    all_cards: list[dict[str, Any]] = []
    for item in packet["card_items"]:
        item = dict(item)
        all_cards.append(item)
        annotation = annotations[str(item["review_item_id"])]
        variants = {
            str(variant["execution_profile"]): variant
            for variant in item["variants"]
        }
        decisions = {
            "minimal": str(annotation["minimal_variant"]),
            "dialogic": str(annotation["dialogic_variant"]),
        }
        item["h1_annotation"] = annotation
        item["variant_decisions"] = decisions
        item["variants"] = [
            variants[profile] for profile in ("minimal", "dialogic")
        ]
        if (
            annotation["distinctness"] == "distinct"
            and decisions == {"minimal": "approve", "dialogic": "approve"}
            and annotation["source_support_valid"] == "yes"
        ):
            approved.append(item)
        elif (
            annotation["distinctness"] == "distinct"
            and decisions == {"minimal": "revise", "dialogic": "revise"}
            and annotation["source_support_valid"] != "yes"
        ):
            pending.append(item)
        else:
            rejected.append(item)
    if (len(approved), len(pending), len(rejected)) != (11, 36, 3):
        raise RuntimeError(
            "unexpected H1 card partition: "
            f"{len(approved)}/{len(pending)}/{len(rejected)}"
        )
    return approved, pending, rejected, all_cards


def _excluded_dialogues(
    *,
    h1_packet_path: Path,
    private_retrieval_path: Path,
) -> tuple[set[str], dict[str, int]]:
    packet = json.loads(h1_packet_path.read_text(encoding="utf-8"))
    old_sources = {
        str(example["source_dialogue_id"])
        for item in packet["card_items"]
        for example in item["source_examples"]
    }
    retrieval = {
        str(row["source_dialogue_id"])
        for row in iter_jsonl(private_retrieval_path)
        if row.get("source_dialogue_id")
    }
    return old_sources | retrieval, {
        "old_h1_source_dialogues": len(old_sources),
        "h1_retrieval_dialogues": len(retrieval),
        "union_excluded_dialogues": len(old_sources | retrieval),
    }


def _nli_scores(
    pairs: list[tuple[str, str]],
    *,
    model_path: Path,
    batch_size: int,
) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if torch.cuda.is_available():
        device_index = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.get_device_properties(index).total_memory,
        )
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    labels = {
        str(label).casefold(): int(index)
        for index, label in model.config.id2label.items()
    }
    entailment_id = labels["entailment"]
    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=160,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
            scores.extend(
                float(value)
                for value in probabilities[:, entailment_id].cpu().tolist()
            )
    return scores


def _select_candidates(
    scored_by_core: dict[str, list[dict[str, Any]]],
    *,
    examples_per_core: int,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_dialogues: set[str] = set()
    used_responses: set[str] = set()
    problem_counts: dict[str, Counter[str]] = defaultdict(Counter)
    ordered_cores = sorted(
        scored_by_core,
        key=lambda core: (len(scored_by_core[core]), core),
    )
    for rank_slot in range(examples_per_core):
        for core_id in ordered_cores:
            pool = sorted(
                scored_by_core[core_id],
                key=lambda row: (
                    -float(row["nli_entailment"]),
                    -float(row["lexical_score"]),
                    str(row["source_dialogue_id"]),
                    int(row["source_turn_index"]),
                ),
            )
            choice = None
            for row in pool:
                dialogue_id = str(row["source_dialogue_id"])
                response_key = _normalized_response(
                    str(row["supporter_response"])
                )
                problem = str(row.get("problem_type") or "unknown")
                if dialogue_id in used_dialogues or response_key in used_responses:
                    continue
                if problem_counts[core_id][problem] >= 2:
                    continue
                choice = row
                break
            if choice is None:
                raise RuntimeError(
                    f"could not select slot {rank_slot + 1} for {core_id}"
                )
            selected[core_id].append(choice)
            used_dialogues.add(str(choice["source_dialogue_id"]))
            used_responses.add(
                _normalized_response(str(choice["supporter_response"]))
            )
            problem_counts[core_id][
                str(choice.get("problem_type") or "unknown")
            ] += 1
    return dict(selected)


def _preselection_pool(
    scored_by_core: dict[str, list[dict[str, Any]]],
    *,
    weak_mapping_path: Path,
    maximum_per_core: int = 30,
) -> dict[str, list[dict[str, Any]]]:
    weak_scores: dict[tuple[str, str], float] = {}
    for row in iter_jsonl(weak_mapping_path):
        weak_scores[
            (str(row["assigned_submove_id"]), str(row["strategy_id"]))
        ] = float(row["top_cosine_score"])
    result: dict[str, list[dict[str, Any]]] = {}
    for core_id, rows in scored_by_core.items():
        enriched = []
        for row in rows:
            value = dict(row)
            value["old_bge_assignment_score"] = weak_scores.get(
                (core_id, str(row["strategy_id"]))
            )
            enriched.append(value)
        rankings = [
            sorted(
                enriched,
                key=lambda row: (
                    -(
                        float(row["old_bge_assignment_score"])
                        if row["old_bge_assignment_score"] is not None
                        else -1.0
                    ),
                    str(row["source_dialogue_id"]),
                ),
            ),
            sorted(
                enriched,
                key=lambda row: (
                    -float(row["nli_entailment"]),
                    str(row["source_dialogue_id"]),
                ),
            ),
            sorted(
                enriched,
                key=lambda row: (
                    -float(row["lexical_score"]),
                    str(row["source_dialogue_id"]),
                ),
            ),
        ]
        chosen: list[dict[str, Any]] = []
        used: set[str] = set()
        # Round-robin across the three outcome-blind rankers.  The downstream
        # LLM is only a search assistant; all five finalists remain human-blind
        # reviewed.
        cursor = [0, 0, 0]
        while len(chosen) < maximum_per_core:
            progressed = False
            for rank_index, ranking in enumerate(rankings):
                while cursor[rank_index] < len(ranking):
                    row = ranking[cursor[rank_index]]
                    cursor[rank_index] += 1
                    strategy_id = str(row["strategy_id"])
                    if strategy_id in used:
                        continue
                    chosen.append(row)
                    used.add(strategy_id)
                    progressed = True
                    break
                if len(chosen) == maximum_per_core:
                    break
            if not progressed:
                break
        if len(chosen) < maximum_per_core:
            raise RuntimeError(
                f"only {len(chosen)}/{maximum_per_core} preselection rows "
                f"for {core_id}"
            )
        result[core_id] = chosen
    return result


def _public_candidate(row: dict[str, Any], candidate_number: int) -> dict[str, Any]:
    return {
        "candidate_number": candidate_number,
        "candidate_id": stable_hex(
            "h1-source-relink-candidate",
            str(row["strategy_id"]),
            n=20,
        ),
        "source_dialogue_id": row["source_dialogue_id"],
        "source_turn_index": row["source_turn_index"],
        "recent_dialogue": row["recent_dialogue"],
        "supporter_response": row["supporter_response"],
    }


def _render_html(packet: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    data = json.dumps(packet, ensure_ascii=False).replace("</", "<\\/")
    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PM V1.5 H1 来源重绑审核</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:20px;background:#f3f5f7;color:#17202a;line-height:1.5}}
.note,.core{{background:#fff;border:1px solid #d7dde5;border-radius:10px;padding:16px;margin:14px 0}}
.note{{border-left:5px solid #245f9e}}.warn{{border-left-color:#a85b00}}
.definition{{background:#eef3f8;border-radius:7px;padding:11px}}.candidate{{border-left:4px solid #8ca0b3;padding:10px 14px;margin:15px 0;background:#fafafa}}
.dialogue{{white-space:pre-wrap;background:#edf1f5;border-radius:6px;padding:10px}}
textarea{{width:100%;box-sizing:border-box;min-height:55px}}button{{padding:9px 14px;margin-right:8px}}
.sticky{{position:sticky;top:0;background:#f3f5f7;padding:9px 0;z-index:2}}.tag{{font-size:12px;background:#e4ebf2;border-radius:999px;padding:3px 8px}}
</style></head><body>
<h1>PM V1.5 H1：36 个 core 的一次性来源重绑</h1>
<div class="note"><b>只判断来源证据，不重审定义。</b>H1 已确认这些 atomic move、
when-to-use 和边界本身清楚。请勾选目标 supporter response 是否<b>直接、字面地</b>
执行了该 move；不要因为回复整体听起来友善就勾选，也不要因为它另有小缺点就否定一个
清楚出现的 atomic move。每个 core 至少 2 个不同对话的 yes 才能进入冻结 Bank。</div>
<div class="note warn">候选只来自 ESConv train，排除了旧 H1 来源例、H1 检索题、
domain-claim 标记和非 seeker 结尾的上下文。页面不显示 NLI/词法分数；这些分数只负责
候选排序，不是标签。若 5 个均不合适，全部不勾并写一句原因即可。</div>
<div class="sticky"><button onclick="download()">导出 JSONL</button>
<span id="progress"></span></div><div id="root"></div>
<script>
const DATA={data}; const MANIFEST={manifest_json};
const KEY="pm15_h1_source_relink_v1"; let S=JSON.parse(localStorage.getItem(KEY)||"{{}}");
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const dlg=x=>(x||[]).map(t=>`${{t.speaker}}: ${{t.content}}`).join("\\n");
function state(id){{S[id]??={{accepted:[],reviewed:false,notes:""}};return S[id];}}
function save(){{localStorage.setItem(KEY,JSON.stringify(S));progress();}}
function toggle(id,n,on){{const s=state(id);s.accepted=s.accepted.filter(x=>x!==n);if(on)s.accepted.push(n);s.accepted.sort((a,b)=>a-b);save();}}
function reviewed(id,on){{state(id).reviewed=on;save();}}
function notes(id,v){{state(id).notes=v;save();}}
function render(){{document.getElementById("root").innerHTML=DATA.map((x,i)=>{{const s=state(x.review_item_id);return `<section class="core">
<h2>${{i+1}}/36 · ${{esc(x.strategy_family)}} · ${{esc(x.core_submove_id)}}</h2>
<div class="definition"><b>Move:</b> ${{esc(x.support_move)}}<br><b>Use:</b> ${{esc(x.when_to_use)}}<br><b>Do not:</b> ${{esc(x.when_not_to_use)}}</div>
${{x.candidates.map(c=>`<div class="candidate"><b>候选 ${{c.candidate_number}}</b>
<div class="dialogue">${{esc(dlg(c.recent_dialogue))}}</div><p><b>目标 supporter response：</b> ${{esc(c.supporter_response)}}</p>
<label><input type="checkbox" ${{s.accepted.includes(c.candidate_number)?"checked":""}} onchange="toggle('${{x.review_item_id}}',${{c.candidate_number}},this.checked)"> 直接体现该 atomic move</label></div>`).join("")}}
<label><input type="checkbox" ${{s.reviewed?"checked":""}} onchange="reviewed('${{x.review_item_id}}',this.checked)"> 我已检查完本 core 的 5 个候选（可一个都不选）</label>
<p><span class="tag">当前选择 ${{s.accepted.length}} 个；≥2 才通过来源门</span></p>
<textarea placeholder="可选：若不足 2 个，简述共同问题" onchange="notes('${{x.review_item_id}}',this.value)">${{esc(s.notes)}}</textarea>
</section>`;}}).join("");progress();}}
function collect(){{return DATA.map(x=>{{const s=state(x.review_item_id);return {{protocol:MANIFEST.protocol,review_item_id:x.review_item_id,core_submove_id:x.core_submove_id,accepted_candidate_numbers:s.accepted,reviewed:s.reviewed,notes:s.notes,annotator_id:""}};}});}}
function progress(){{const rows=collect();const n=rows.filter(x=>x.reviewed).length;document.getElementById("progress").textContent=`${{n}} / ${{rows.length}} cores reviewed`;}}
function download(){{const text=collect().map(x=>JSON.stringify(x)).join("\\n")+"\\n";const blob=new Blob([text],{{type:"application/jsonl"}});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="pm_v1_5_h1_source_relink_annotations.jsonl";a.click();URL.revokeObjectURL(a.href);}}
render();
</script></body></html>"""


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h1-packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/h1_review_packet.json",
    )
    parser.add_argument(
        "--h1-annotations",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_analysis/h1_annotations_frozen.jsonl",
    )
    parser.add_argument(
        "--private-retrieval-audit",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_complete_rag_review_v1_candidate/private_retrieval_audit.jsonl",
    )
    parser.add_argument(
        "--source-universe",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/clean_train_strategy_universe.jsonl",
    )
    parser.add_argument(
        "--weak-mapping",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1/"
        "strategy_cards_v3_weak_source_mapping.jsonl",
    )
    parser.add_argument("--nli-model", type=Path, default=DEFAULT_NLI)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--examples-per-core", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h1_source_relink_review_v1_candidate",
    )
    args = parser.parse_args()

    approved, pending, rejected, _ = _load_h1_status(
        packet_path=args.h1_packet,
        annotations_path=args.h1_annotations,
    )
    excluded, exclusion_counts = _excluded_dialogues(
        h1_packet_path=args.h1_packet,
        private_retrieval_path=args.private_retrieval_audit,
    )
    universe = [dict(row) for row in iter_jsonl(args.source_universe)]
    eligible_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in universe:
        family = str(row.get("strategy_label"))
        if _eligible_source(
            row, family=family, excluded_dialogues=excluded
        ):
            eligible_by_family[family].append(row)

    scored_by_core: dict[str, list[dict[str, Any]]] = {}
    pairs: list[tuple[str, str]] = []
    pair_bindings: list[tuple[str, dict[str, Any], float]] = []
    for core in pending:
        core_id = str(core["core_submove_id"])
        family = str(core["strategy_family"])
        hypothesis = "The response " + str(core["support_move"]).rstrip(".") + "."
        for row in eligible_by_family[family]:
            lexical = _lexical_score(str(row["supporter_response"]), core)
            pairs.append((str(row["supporter_response"]), hypothesis))
            pair_bindings.append((core_id, row, lexical))
    nli = _nli_scores(
        pairs,
        model_path=args.nli_model,
        batch_size=args.batch_size,
    )
    for (core_id, row, lexical), score in zip(
        pair_bindings, nli, strict=True
    ):
        enriched = dict(row)
        enriched["lexical_score"] = round(lexical, 8)
        enriched["nli_entailment"] = round(score, 8)
        scored_by_core.setdefault(core_id, []).append(enriched)

    preselection = _preselection_pool(
        scored_by_core,
        weak_mapping_path=args.weak_mapping,
    )
    selected = _select_candidates(
        scored_by_core,
        examples_per_core=args.examples_per_core,
    )
    packet: list[dict[str, Any]] = []
    private_scores: list[dict[str, Any]] = []
    for core in pending:
        core_id = str(core["core_submove_id"])
        candidates = [
            _public_candidate(row, index)
            for index, row in enumerate(selected[core_id], start=1)
        ]
        packet.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": stable_hex(
                    "h1-source-relink-review", core_id, n=24
                ),
                "core_submove_id": core_id,
                "strategy_family": core["strategy_family"],
                "support_move": core["support_move"],
                "when_to_use": core["when_to_use"],
                "when_not_to_use": core["when_not_to_use"],
                "candidates": candidates,
            }
        )
        for index, row in enumerate(selected[core_id], start=1):
            private_scores.append(
                {
                    "core_submove_id": core_id,
                    "candidate_number": index,
                    "candidate_id": candidates[index - 1]["candidate_id"],
                    "strategy_id": row["strategy_id"],
                    "source_dialogue_id": row["source_dialogue_id"],
                    "problem_type": row["problem_type"],
                    "lexical_score": row["lexical_score"],
                    "nli_entailment": row["nli_entailment"],
                }
            )

    approved_variants = [
        dict(variant) for core in approved for variant in core["variants"]
    ]
    pending_variants = [
        dict(variant) for core in pending for variant in core["variants"]
    ]
    rejected_variants = [
        dict(variant) for core in rejected for variant in core["variants"]
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preselection_rows: list[dict[str, Any]] = []
    for core in pending:
        core_id = str(core["core_submove_id"])
        for index, row in enumerate(preselection[core_id], start=1):
            preselection_rows.append(
                {
                    "core_submove_id": core_id,
                    "strategy_family": core["strategy_family"],
                    "support_move": core["support_move"],
                    "when_to_use": core["when_to_use"],
                    "when_not_to_use": core["when_not_to_use"],
                    "preselection_candidate_number": index,
                    "strategy_id": row["strategy_id"],
                    "source_dialogue_id": row["source_dialogue_id"],
                    "source_turn_index": row["source_turn_index"],
                    "recent_dialogue": row["recent_dialogue"],
                    "supporter_response": row["supporter_response"],
                    "problem_type": row["problem_type"],
                    "nli_entailment": row["nli_entailment"],
                    "lexical_score": row["lexical_score"],
                    "old_bge_assignment_score": row[
                        "old_bge_assignment_score"
                    ],
                }
            )
    _write_jsonl(
        args.out_dir / "private_preselection_pool.jsonl",
        preselection_rows,
    )
    _write_jsonl(args.out_dir / "source_relink_packet.jsonl", packet)
    _write_jsonl(args.out_dir / "private_candidate_scores.jsonl", private_scores)
    _write_jsonl(args.out_dir / "frozen_approved_variants.jsonl", approved_variants)
    _write_jsonl(args.out_dir / "pending_source_relink_variants.jsonl", pending_variants)
    _write_jsonl(args.out_dir / "h1_rejected_variants.jsonl", rejected_variants)
    template = [
        {
            "protocol": PROTOCOL,
            "review_item_id": row["review_item_id"],
            "core_submove_id": row["core_submove_id"],
            "accepted_candidate_numbers": [],
            "reviewed": False,
            "notes": "",
            "annotator_id": "",
        }
        for row in packet
    ]
    _write_jsonl(args.out_dir / "human_annotation_template.jsonl", template)
    manifest = {
        "protocol": PROTOCOL,
        "purpose": (
            "One bounded source-evidence relink for the 36 H1-revised, "
            "definition-valid, nonduplicate RAG cores."
        ),
        "decision_rule": (
            "A core passes source qualification only with at least two "
            "human-accepted literal examples from distinct ESConv-train dialogues."
        ),
        "human_review_units": len(packet),
        "candidates_per_core": args.examples_per_core,
        "candidate_judgments_visible": len(packet) * args.examples_per_core,
        "h1_partition": {
            "approved_cores": len(approved),
            "pending_source_relink_cores": len(pending),
            "rejected_or_duplicate_cores": len(rejected),
            "approved_variants_frozen_now": len(approved_variants),
            "pending_variants": len(pending_variants),
            "rejected_variants": len(rejected_variants),
        },
        "exclusions": exclusion_counts,
        "selection": {
            "source_split": "ESConv train only",
            "family_label_match_required": True,
            "latest_visible_turn_must_be_seeker": True,
            "domain_claim_keyword_flag_must_be_false": True,
            "word_count_range": [4, 80],
            "old_h1_sources_excluded": True,
            "h1_retrieval_queries_excluded": True,
            "global_unique_source_dialogues": True,
            "global_unique_normalized_responses": True,
            "maximum_same_problem_per_core": 2,
            "ranking": (
                "response-only DeBERTa-v3 NLI entailment, lexical overlap "
                "tie-break; scores hidden from human reviewer"
            ),
        },
        "inputs": {
            "h1_packet": str(args.h1_packet.relative_to(ROOT)),
            "h1_annotations": str(args.h1_annotations.relative_to(ROOT)),
            "private_retrieval_audit": str(
                args.private_retrieval_audit.relative_to(ROOT)
            ),
            "source_universe": str(args.source_universe.relative_to(ROOT)),
            "weak_mapping": str(args.weak_mapping.relative_to(ROOT)),
            "nli_model": str(args.nli_model),
        },
        "intermediate_warning": (
            "The initially rendered five candidates are an offline diagnostic "
            "only. Run 24al before human review; generic NLI alone is not "
            "qualified for fine-grained source relinking."
        ),
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    (args.out_dir / "human_review.html").write_text(
        _render_html(packet, manifest),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        path.name: sha256_file(path)
        for path in sorted(args.out_dir.iterdir())
        if path.is_file()
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": "READY_FOR_ONE_BOUNDED_HUMAN_REVIEW",
            "cores": len(packet),
            "candidate_examples": len(private_scores),
            "frozen_approved_variants": len(approved_variants),
            "pending_variants": len(pending_variants),
            "rejected_variants": len(rejected_variants),
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
