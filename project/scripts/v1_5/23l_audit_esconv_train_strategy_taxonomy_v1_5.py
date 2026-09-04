#!/usr/bin/env python3
"""Audit the train-only ESConv strategy universe before defining V1.5 cards.

This is an inductive discovery diagnostic, not a card builder.  It keeps all
eight native ESConv labels, removes only objective low-information/lineage
failures, tests whether raw response wording exhibits stable label/cluster
structure, and creates a label-blind open-coding packet.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    write_json,
    write_jsonl,
)
from metacom_pm.strategy_bank import load_esconv
from metacom_pm.text import normalize_for_hash, normalize_space


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-esconv-train-inductive-strategy-audit-v1"
NATIVE_LABELS = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
)
CLUSTER_COUNTS = (5, 8, 12, 20)
CLUSTER_SEEDS = (1701, 4311, 9001)

_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|good (?:morning|afternoon|evening)|"
    r"how are you|how'?s (?:it going|things))[\s!,.?]*$",
    flags=re.IGNORECASE,
)
_CLOSING_RE = re.compile(
    r"^(?:thank you|thanks|you'?re welcome|good luck|take care|bye|"
    r"goodbye|have a good (?:day|night))[\s!,.?]*$",
    flags=re.IGNORECASE,
)
_META_RE = re.compile(
    r"\b(?:survey|questionnaire|platform|mechanical turk|mturk|"
    r"rate (?:this|the)|end (?:this|the) (?:chat|session)|"
    r"task is complete)\b",
    flags=re.IGNORECASE,
)
_DOMAIN_CLAIM_RE = re.compile(
    r"\b(?:dose|dosage|milligram|medication|prescription|diagnos(?:e|is)|"
    r"lawyer|legal advice|lawsuit|interest rate|investment|stock market|"
    r"tax advice)\b",
    flags=re.IGNORECASE,
)
_SELF_DISCLOSURE_RE = re.compile(
    r"\b(?:i|i'm|i've|i had|my|me personally|in my experience)\b",
    flags=re.IGNORECASE,
)
_LOW_INFORMATION = {
    "ok",
    "okay",
    "yes",
    "yeah",
    "yep",
    "no",
    "nope",
    "right",
    "sure",
    "i see",
    "i understand",
    "understood",
    "that makes sense",
    "exactly",
    "true",
    "great",
    "good",
    "nice",
    "wow",
}


def _blind_id(source_dialogue_id: str, source_turn_index: int) -> str:
    digest = hashlib.sha256(
        f"{PROTOCOL}|{source_dialogue_id}|{source_turn_index}".encode()
    ).hexdigest()
    return "strategy_discovery_" + digest[:20]


def _exclusion_reason(response: str, has_prior_seeker: bool) -> str | None:
    cleaned = normalize_space(response)
    normalized = normalize_for_hash(cleaned)
    if not has_prior_seeker:
        return "no_prior_visible_seeker_turn"
    if not cleaned:
        return "empty_response"
    if _GREETING_RE.fullmatch(cleaned):
        return "greeting_only"
    if _CLOSING_RE.fullmatch(cleaned):
        return "closing_only"
    if _META_RE.search(cleaned):
        return "platform_or_survey_meta"
    if normalized in _LOW_INFORMATION:
        return "low_information_acknowledgement"
    # A very short turn may still keep a conversation moving, but it does not
    # contain enough observable behavior to justify a reusable support card.
    # This gate is for taxonomy induction only, not a judgment of response
    # quality or usefulness in the original dialogue.
    if len(re.findall(r"\b\w+\b", cleaned)) < 5:
        return "insufficient_semantic_content_for_card_induction"
    return None


def _recent_dialogue(
    dialogue: list[dict[str, Any]], turn_index: int, limit: int = 6
) -> list[dict[str, str]]:
    return [
        {
            "speaker": str(turn.get("speaker") or ""),
            "content": normalize_space(turn.get("content") or ""),
        }
        for turn in dialogue[max(0, turn_index - limit) : turn_index]
        if normalize_space(turn.get("content") or "")
    ]


def _counter_profile(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, int]:
    counter = Counter(str(row[key]) for row in rows)
    return {label: int(counter.get(label, 0)) for label in NATIVE_LABELS}


def _dialogue_profile(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    by_label: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_label[str(row["strategy_label"])].add(
            str(row["source_dialogue_id"])
        )
    return {label: len(by_label[label]) for label in NATIVE_LABELS}


def _cluster_diagnostics(
    matrix: Any, native_labels: list[str], seed: int
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for count in CLUSTER_COUNTS:
        runs: list[np.ndarray] = []
        for cluster_seed in CLUSTER_SEEDS:
            runs.append(
                MiniBatchKMeans(
                    n_clusters=count,
                    random_state=cluster_seed,
                    n_init=5,
                    batch_size=256,
                ).fit_predict(matrix)
            )
        stability = [
            adjusted_rand_score(runs[left], runs[right])
            for left in range(len(runs))
            for right in range(left + 1, len(runs))
        ]
        diagnostics[str(count)] = {
            "cosine_silhouette": round(
                float(
                    silhouette_score(
                        matrix,
                        runs[0],
                        metric="cosine",
                        sample_size=min(2500, matrix.shape[0]),
                        random_state=seed,
                    )
                ),
                4,
            ),
            "mean_seed_stability_ARI": round(float(np.mean(stability)), 4),
            "NMI_with_native_ESConv_label": round(
                float(
                    normalized_mutual_info_score(native_labels, runs[0])
                ),
                4,
            ),
        }
    return diagnostics


def _grouped_label_diagnostic(
    matrix: Any,
    labels: list[str],
    groups: list[str],
    seed: int,
) -> dict[str, Any]:
    ordered_labels = list(NATIVE_LABELS)
    predictions = np.empty(len(labels), dtype=object)
    splitter = GroupKFold(n_splits=5)
    for train_index, test_index in splitter.split(matrix, labels, groups):
        model = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(matrix[train_index], np.asarray(labels)[train_index])
        predictions[test_index] = model.predict(matrix[test_index])
    report = classification_report(
        labels,
        predictions.tolist(),
        labels=ordered_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix_counts = confusion_matrix(
        labels, predictions.tolist(), labels=ordered_labels
    )
    return {
        "role": (
            "Diagnostic of native-label wording consistency only; not card "
            "quality, utility, or a proposed PM model."
        ),
        "grouping": "five-fold source-dialogue-grouped out-of-fold",
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_label": {
            label: {
                metric: round(float(report[label][metric]), 4)
                for metric in ("precision", "recall", "f1-score")
            }
            for label in ordered_labels
        },
        "confusion_labels": ordered_labels,
        "confusion_matrix": matrix_counts.tolist(),
    }


def _diverse_sample_indices(
    matrix: Any,
    rows: list[dict[str, Any]],
    per_label: int,
    seed: int,
) -> list[int]:
    selected: list[int] = []
    for label in NATIVE_LABELS:
        candidates = [
            index
            for index, row in enumerate(rows)
            if row["strategy_label"] == label
        ]
        label_matrix = matrix[candidates]
        count = min(per_label, len(candidates))
        model = MiniBatchKMeans(
            n_clusters=count,
            random_state=seed,
            n_init=10,
            batch_size=128,
        ).fit(label_matrix)
        distances = model.transform(label_matrix)
        used_dialogues: set[str] = set()
        used_indices: set[int] = set()
        for cluster_index in range(count):
            order = np.argsort(distances[:, cluster_index])
            choice = None
            for local_index in order:
                source_index = candidates[int(local_index)]
                dialogue_id = str(rows[source_index]["source_dialogue_id"])
                if (
                    source_index not in used_indices
                    and dialogue_id not in used_dialogues
                ):
                    choice = source_index
                    break
            if choice is None:
                for local_index in order:
                    source_index = candidates[int(local_index)]
                    if source_index not in used_indices:
                        choice = source_index
                        break
            if choice is not None:
                selected.append(choice)
                used_indices.add(choice)
                used_dialogues.add(
                    str(rows[choice]["source_dialogue_id"])
                )
    return sorted(
        selected,
        key=lambda index: _blind_id(
            str(rows[index]["source_dialogue_id"]),
            int(rows[index]["source_turn_index"]),
        ),
    )


def _render_review_html(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh"><meta charset="utf-8">
<title>ESConv train-only 策略开放编码</title>
<style>
body{{font:16px sans-serif;max-width:1050px;margin:24px auto;line-height:1.55}}
.card{{border:1px solid #bbb;border-radius:8px;padding:16px;margin:18px 0}}
.context{{background:#f5f5f5;padding:12px;white-space:pre-wrap}}
textarea{{width:100%;min-height:60px;font:inherit}} select,input{{font:inherit}}
label{{display:block;margin:8px 0}} button{{padding:10px 16px}}
</style>
<h1>ESConv train-only 策略开放编码</h1>
<p>页面有意隐藏 ESConv 原策略标签、现有五族和 50 张 taxonomy。请先用自己的话描述
回复实际执行了什么支持动作；不要猜原标签。样本按原标签做隐藏分层并提高文本多样性，
所以不能用来估计真实类别比例。</p>
<div id="root"></div><button onclick="download()">导出 JSONL</button>
<script>
const rows={payload};
function yn(cls){{return `<select class="${{cls}}"><option value="">--选择--</option>
<option value="true">是</option><option value="false">否</option></select>`}}
document.querySelector("#root").innerHTML=rows.map((r,i)=>`<section class="card" data-i="${{i}}">
<h2>${{i+1}} / ${{rows.length}} — ${{r.blind_item_id}}</h2>
<div class="context">${{r.recent_dialogue.map(x=>`${{x.speaker}}: ${{x.content}}`).join("\\n")}}</div>
<h3>待编码 supporter response</h3><div class="context">${{r.supporter_response}}</div>
<label>是否包含有意义的支持动作？ ${{yn("meaningful_support_action")}}</label>
<label>主要动作（自由描述；例如“询问一个缺失事实”，不要填写现有类别名）
<textarea class="primary_action"></textarea></label>
<label>次要动作（没有则留空）<textarea class="secondary_action"></textarea></label>
<label>是否主要是事实/资源信息？ ${{yn("mainly_information")}}</label>
<label>是否主要依赖支持者自我披露？ ${{yn("mainly_self_disclosure")}}</label>
<label>是否可以抽象成不含话题细节的通用卡？ ${{yn("reusable_as_general_card")}}</label>
<label>是否存在明显风险、越界或不恰当假设？ ${{yn("clear_risk_or_boundary_problem")}}</label>
<label>备注<textarea class="notes"></textarea></label>
</section>`).join("");
function download(){{
 const out=[];
 for(let i=0;i<rows.length;i++){{const c=document.querySelector(`[data-i="${{i}}"]`),r=rows[i],
 row={{protocol:"pm-v1.5-esconv-train-inductive-open-coding-v1",blind_item_id:r.blind_item_id}};
 for(const k of ["meaningful_support_action","mainly_information","mainly_self_disclosure",
 "reusable_as_general_card","clear_risk_or_boundary_problem"]){{
  const v=c.querySelector("."+k).value;if(!v){{alert(`第 ${{i+1}} 条未完成 ${{k}}`);return;}}
  row[k]=v==="true";
 }}
 row.primary_action=c.querySelector(".primary_action").value.trim();
 if(row.meaningful_support_action && !row.primary_action){{alert(`第 ${{i+1}} 条缺主要动作`);return;}}
 row.secondary_action=c.querySelector(".secondary_action").value.trim();
 row.notes=c.querySelector(".notes").value.trim();out.push(row);
 }}
 const blob=new Blob([out.map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);
 a.download="esconv_strategy_open_coding_annotations.jsonl";a.click();
}}
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--split-manifest",
        type=Path,
        default=ROOT / "data/strategy/esconv_split_manifest_v1_5.jsonl",
    )
    parser.add_argument(
        "--support-need-lineage",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_support_need_packet_v3_lineage_candidate"
        / "private_lineage.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1",
    )
    parser.add_argument("--open-coding-per-native-label", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4311)
    args = parser.parse_args()

    raw_rows = [dict(row) for row in iter_jsonl(args.raw_cards)]
    split_rows = [dict(row) for row in iter_jsonl(args.split_manifest)]
    split_by_id = {str(row["dialogue_id"]): row for row in split_rows}
    esconv = load_esconv(args.esconv)
    esconv_by_id = {
        f"esconv_{index:04d}": dict(dialogue)
        for index, dialogue in enumerate(esconv)
    }
    packet_sources = {
        str(row["dialogue_id"])
        for row in iter_jsonl(args.support_need_lineage)
    }

    unknown_labels = sorted(
        {str(row["strategy_label"]) for row in raw_rows} - set(NATIVE_LABELS)
    )
    if unknown_labels:
        raise RuntimeError(f"unknown native ESConv labels: {unknown_labels}")

    raw_source_ids = {
        str(row["source_dialogue_id"]) for row in raw_rows
    }
    ineligible_sources = [
        source_id
        for source_id in raw_source_ids
        if source_id not in split_by_id
        or split_by_id[source_id].get("split") != "train"
        or bool(split_by_id[source_id].get("excluded_for_evoemo_overlap"))
    ]
    if ineligible_sources:
        raise RuntimeError(
            "raw bank contains validation/test/EvoEmo-overlap sources"
        )

    exclusion_counts: Counter[str] = Counter()
    duplicate_label_sets: Counter[str] = Counter()
    first_by_normalized: dict[str, dict[str, Any]] = {}
    prepared: list[dict[str, Any]] = []
    for raw in raw_rows:
        source_id = str(raw["source_dialogue_id"])
        if source_id in packet_sources:
            exclusion_counts["support_need_packet_source"] += 1
            continue
        turn_index = int(raw["source_turn_index"])
        dialogue = list(esconv_by_id[source_id].get("dialog") or [])
        if (
            turn_index < 0
            or turn_index >= len(dialogue)
            or dialogue[turn_index].get("speaker") != "supporter"
        ):
            raise RuntimeError("invalid source turn lineage")
        has_prior_seeker = any(
            turn.get("speaker") == "seeker"
            and normalize_space(turn.get("content") or "")
            for turn in dialogue[:turn_index]
        )
        response = normalize_space(raw.get("example_response") or "")
        reason = _exclusion_reason(response, has_prior_seeker)
        if reason:
            exclusion_counts[reason] += 1
            continue
        normalized = normalize_for_hash(response)
        prior = first_by_normalized.get(normalized)
        if prior is not None:
            exclusion_counts["normalized_duplicate_response"] += 1
            labels = {
                str(prior["strategy_label"]),
                str(raw["strategy_label"]),
            }
            if len(labels) > 1:
                duplicate_label_sets[" | ".join(sorted(labels))] += 1
            continue
        source = esconv_by_id[source_id]
        row = {
            "strategy_id": str(raw["strategy_id"]),
            "source_dialogue_id": source_id,
            "source_turn_index": turn_index,
            "strategy_label": str(raw["strategy_label"]),
            "supporter_response": response,
            "recent_dialogue": _recent_dialogue(dialogue, turn_index),
            "problem_type": normalize_space(
                source.get("problem_type") or "unknown"
            ),
            "emotion_type": normalize_space(
                source.get("emotion_type") or "unknown"
            ),
            "experience_type": normalize_space(
                source.get("experience_type") or "unknown"
            ),
            "word_count": len(re.findall(r"\b\w+\b", response)),
            "has_question_mark": "?" in response,
            "domain_claim_keyword_flag": bool(_DOMAIN_CLAIM_RE.search(response)),
            "first_person_language_flag": bool(
                _SELF_DISCLOSURE_RE.search(response)
            ),
        }
        first_by_normalized[normalized] = row
        prepared.append(row)

    if len(prepared) < 1000:
        raise RuntimeError("unexpectedly small inductive source universe")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=10000,
        sublinear_tf=True,
    )
    texts = [str(row["supporter_response"]) for row in prepared]
    tfidf = vectorizer.fit_transform(texts)
    labels = [str(row["strategy_label"]) for row in prepared]
    groups = [str(row["source_dialogue_id"]) for row in prepared]

    cluster_diagnostic = _cluster_diagnostics(
        tfidf, native_labels=labels, seed=args.seed
    )
    label_diagnostic = _grouped_label_diagnostic(
        tfidf, labels=labels, groups=groups, seed=args.seed
    )
    sample_indices = _diverse_sample_indices(
        tfidf,
        prepared,
        per_label=args.open_coding_per_native_label,
        seed=args.seed,
    )
    public_packet: list[dict[str, Any]] = []
    private_lineage: list[dict[str, Any]] = []
    for index in sample_indices:
        row = prepared[index]
        blind_item_id = _blind_id(
            str(row["source_dialogue_id"]),
            int(row["source_turn_index"]),
        )
        public_packet.append(
            {
                "protocol": "pm-v1.5-esconv-train-inductive-open-coding-v1",
                "blind_item_id": blind_item_id,
                "recent_dialogue": row["recent_dialogue"],
                "supporter_response": row["supporter_response"],
            }
        )
        private_lineage.append(
            {
                "blind_item_id": blind_item_id,
                "source_dialogue_id": row["source_dialogue_id"],
                "source_turn_index": row["source_turn_index"],
                "native_ESConv_strategy_label": row["strategy_label"],
                "selection": (
                    "hidden-label-stratified TF-IDF diversity sample; "
                    "not prevalence representative"
                ),
            }
        )

    word_counts = [int(row["word_count"]) for row in prepared]
    report = {
        "protocol": PROTOCOL,
        "status": "INDUCTIVE_SOURCE_AUDIT_COMPLETE_OPEN_CODING_PENDING",
        "intended_use": (
            "Decide the V1.5 card taxonomy from train-only evidence before "
            "comparing it with the existing top-down five-family candidate."
        ),
        "lineage": {
            "split_rule": "custom reproducible 70/15/15 dialogue split",
            "raw_bank_sources_all_train": True,
            "raw_bank_sources_all_non_EvoEmo_overlap": True,
            "split_train_non_overlap_dialogues": sum(
                row["split"] == "train"
                and not bool(row["excluded_for_evoemo_overlap"])
                for row in split_rows
            ),
            "raw_bank_source_dialogues": len(raw_source_ids),
            "support_need_packet_sources_removed": len(
                raw_source_ids & packet_sources
            ),
            "validation_or_test_rows_read_for_taxonomy": 0,
            "external_outcomes_read_for_taxonomy": 0,
        },
        "grain": "one supporter turn with prior visible dialogue",
        "raw_rows": len(raw_rows),
        "eligible_rows_after_objective_cleaning": len(prepared),
        "eligible_source_dialogues": len(set(groups)),
        "exclusion_counts_first_reason": dict(sorted(exclusion_counts.items())),
        "native_label_rows_before_cleaning": _counter_profile(
            raw_rows, "strategy_label"
        ),
        "native_label_rows_after_cleaning": _counter_profile(
            prepared, "strategy_label"
        ),
        "native_label_dialogues_after_cleaning": _dialogue_profile(prepared),
        "normalized_duplicate_rows_removed": int(
            exclusion_counts["normalized_duplicate_response"]
        ),
        "cross_label_normalized_duplicate_rows_removed": int(
            sum(duplicate_label_sets.values())
        ),
        "most_common_cross_label_duplicate_combinations": dict(
            duplicate_label_sets.most_common(15)
        ),
        "descriptive_flags_after_cleaning": {
            "domain_claim_keyword_rows": sum(
                bool(row["domain_claim_keyword_flag"]) for row in prepared
            ),
            "first_person_language_rows": sum(
                bool(row["first_person_language_flag"]) for row in prepared
            ),
            "question_mark_rows": sum(
                bool(row["has_question_mark"]) for row in prepared
            ),
            "word_count_p10_p25_p50_p75_p90": [
                round(float(value), 1)
                for value in np.percentile(
                    np.asarray(word_counts), [10, 25, 50, 75, 90]
                )
            ],
        },
        "unsupervised_raw_wording_diagnostic": cluster_diagnostic,
        "native_label_wording_diagnostic": label_diagnostic,
        "interpretation_rules": {
            "native_labels_are_not_card_gold": True,
            "clusters_are_not_automatically_cards": True,
            "low_silhouette_or_low_seed_stability": (
                "Raw wording does not support automatic card taxonomy."
            ),
            "high_grouped_label_f1": (
                "Native labels have wording signal, but still do not prove "
                "safety, utility, or submove boundaries."
            ),
            "open_coding_required": (
                "Blindly code atomic support actions, merge synonyms, then "
                "compare the induced codebook with the existing 50 cards."
            ),
        },
        "open_coding_packet": {
            "rows": len(public_packet),
            "hidden_native_label_quota": (
                args.open_coding_per_native_label
            ),
            "representative_of_prevalence": False,
            "current_five_family_names_visible": False,
            "current_50_submoves_visible": False,
        },
        "decision": {
            "existing_five_family_50_card_candidate_status": (
                "TOP_DOWN_REFERENCE_ONLY_PENDING_INDUCTIVE_COMPARISON"
            ),
            "human_review_of_existing_50_should_continue_now": False,
            "automatic_expansion_to_80_100": False,
            "next_gate": (
                "Complete independent open coding, derive atomic moves from "
                "agreement, and retain only source-supported reusable moves."
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "clean_train_strategy_universe.jsonl", prepared)
    write_jsonl(args.out_dir / "open_coding_packet.jsonl", public_packet)
    write_jsonl(args.out_dir / "private_lineage.jsonl", private_lineage)
    (args.out_dir / "human_open_coding.html").write_text(
        _render_review_html(public_packet), encoding="utf-8"
    )
    write_json(args.out_dir / "audit_report.json", report)
    print(
        canonical_json(
            {
                "output": str(args.out_dir),
                "status": report["status"],
                "eligible_rows": len(prepared),
                "open_coding_rows": len(public_packet),
                "macro_f1": label_diagnostic["macro_f1"],
                "cluster_diagnostic": cluster_diagnostic,
            }
        )
    )


if __name__ == "__main__":
    main()
