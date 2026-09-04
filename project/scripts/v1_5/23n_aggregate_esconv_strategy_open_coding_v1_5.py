#!/usr/bin/env python3
"""Aggregate two blind open-coding passes and prepare minimal human review."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score
import torch
from transformers import AutoModel, AutoTokenizer

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-esconv-strategy-open-coding-aggregation-v1"
BOOLEAN_FIELDS = (
    "meaningful_support_action",
    "mainly_information",
    "mainly_self_disclosure",
    "reusable_as_general_technique",
    "clear_risk_or_boundary_problem",
)
CORE_FIELDS = (
    "meaningful_support_action",
    "mainly_information",
    "mainly_self_disclosure",
    "clear_risk_or_boundary_problem",
)
DEFAULT_MODEL = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)


def _encode(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            chunks.append(hidden.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _hash_fraction(item_id: str, salt: str) -> float:
    value = hashlib.sha256(f"{salt}|{item_id}".encode()).hexdigest()
    return int(value[:12], 16) / float(16**12)


def _load_endpoint_rows(
    path: Path, endpoint_name: str
) -> dict[str, dict[str, Any]]:
    rows = [
        dict(row)
        for row in iter_jsonl(path)
        if str(row.get("endpoint_name")) == endpoint_name
    ]
    by_id = {str(row["blind_item_id"]): row for row in rows}
    if len(rows) != 160 or len(by_id) != 160:
        raise RuntimeError(
            f"{endpoint_name}: expected exact 160 unique rows, got "
            f"{len(rows)}/{len(by_id)}"
        )
    return by_id


def _render_html(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh"><meta charset="utf-8">
<title>ESConv Strategy open coding 最小人工归并</title>
<style>
body{{font:16px sans-serif;max-width:1120px;margin:24px auto;line-height:1.5}}
.card{{border:1px solid #bbb;border-radius:8px;padding:16px;margin:18px 0}}
.context,.response{{white-space:pre-wrap;background:#f5f5f5;padding:10px}}
.models{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.model{{border:1px solid #ddd;padding:10px}}
textarea{{width:100%;min-height:55px;font:inherit}}select,input{{font:inherit}}
label{{display:block;margin:7px 0}}button{{padding:10px 16px}}
.tier{{font-weight:bold;color:#8b0000}}
</style>
<h1>ESConv Strategy open coding 最小人工归并</h1>
<p>原 ESConv 标签、现有五族和 50 张 taxonomy 仍不可见。Tier 1 是两模型在核心语义、
information/self-disclosure/risk 上的全部分歧或任一风险；Tier 2 是
reusability-only 分歧的固定样本；Tier 3 是完全一致池的固定 20% 校准样本。
模型输出只是候选描述，请以可见对话和 supporter response 为准。</p>
<div id="root"></div><button onclick="download()">导出 JSONL</button>
<script>
const rows={payload};
function tri(cls){{return `<select class="${{cls}}"><option value="">--选择--</option>
<option value="true">是</option><option value="false">否</option>
<option value="uncertain">不确定</option></select>`}}
function modelBox(name,x){{return `<div class="model"><b>${{name}}</b><pre>${{
JSON.stringify(x,null,2)}}</pre></div>`}}
document.querySelector("#root").innerHTML=rows.map((r,i)=>`<section class="card" data-i="${{i}}">
<h2>${{i+1}} / ${{rows.length}} — ${{r.blind_item_id}}</h2>
<div class="tier">Tier ${{r.review_tier}}：${{r.selection_reasons.join("；")}}</div>
<div class="context">${{r.recent_dialogue.map(x=>`${{x.speaker}}: ${{x.content}}`).join("\\n")}}</div>
<h3>supporter response</h3><div class="response">${{r.supporter_response}}</div>
<div class="models">${{modelBox("Coder A",r.coder_a)}}${{modelBox("Coder B",r.coder_b)}}</div>
<label>存在有意义的支持动作？ ${{tri("meaningful_support_action")}}</label>
<label>Canonical primary atomic action（英文短语，topic-agnostic）
<textarea class="primary_action"></textarea></label>
<label>Secondary action（无则留空）<textarea class="secondary_action"></textarea></label>
<label>主要是 factual information？ ${{tri("mainly_information")}}</label>
<label>主要依赖 supporter self-disclosure？ ${{tri("mainly_self_disclosure")}}</label>
<label>能抽象为通用 support technique？ ${{tri("reusable_as_general_technique")}}</label>
<label>有明确 risk/boundary problem？ ${{tri("clear_risk_or_boundary_problem")}}</label>
<label>Risk/boundary 说明<textarea class="risk_description"></textarea></label>
<label>备注<textarea class="notes"></textarea></label>
</section>`).join("");
function download(){{
 const out=[];
 for(let i=0;i<rows.length;i++){{const r=rows[i],c=document.querySelector(`[data-i="${{i}}"]`),
 row={{protocol:"pm-v1.5-esconv-strategy-open-coding-human-adjudication-v1",
 blind_item_id:r.blind_item_id,review_tier:r.review_tier}};
 for(const k of ["meaningful_support_action","mainly_information","mainly_self_disclosure",
 "reusable_as_general_technique","clear_risk_or_boundary_problem"]){{
  const v=c.querySelector("."+k).value;if(!v){{alert(`第 ${{i+1}} 条未完成 ${{k}}`);return;}}
  row[k]=v==="uncertain"?"uncertain":v==="true";
 }}
 row.primary_action=c.querySelector(".primary_action").value.trim();
 row.secondary_action=c.querySelector(".secondary_action").value.trim();
 row.risk_description=c.querySelector(".risk_description").value.trim();
 row.notes=c.querySelector(".notes").value.trim();
 if(row.meaningful_support_action===true && !row.primary_action){{
  alert(`第 ${{i+1}} 条缺 primary action`);return;
 }}
 out.push(row);
 }}
 const blob=new Blob([out.map(x=>JSON.stringify(x)).join("\\n")+"\\n"],{{type:"application/jsonl"}});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);
 a.download="esconv_strategy_open_coding_human_adjudication.jsonl";a.click();
}}
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1"
        / "open_coding_packet.jsonl",
    )
    parser.add_argument(
        "--coder-a",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_strategy_open_coding_v1"
        / "model_open_codes.jsonl",
    )
    parser.add_argument("--coder-a-endpoint", default="generator")
    parser.add_argument(
        "--coder-b",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_strategy_open_coding_deepseek_v1"
        / "model_open_codes.jsonl",
    )
    parser.add_argument(
        "--coder-b-endpoint", default="training_judge_deepseek_flash"
    )
    parser.add_argument("--embedding-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_esconv_strategy_open_coding_aggregation_v1",
    )
    parser.add_argument(
        "--reusability-disagreement-sample", type=int, default=20
    )
    parser.add_argument("--agreement-sample-rate", type=float, default=0.20)
    args = parser.parse_args()

    packet = {
        str(row["blind_item_id"]): dict(row) for row in iter_jsonl(args.packet)
    }
    if len(packet) != 160:
        raise RuntimeError("public packet must contain exactly 160 unique IDs")
    coder_a = _load_endpoint_rows(args.coder_a, args.coder_a_endpoint)
    coder_b = _load_endpoint_rows(args.coder_b, args.coder_b_endpoint)
    if set(packet) != set(coder_a) or set(packet) != set(coder_b):
        raise RuntimeError("packet and two coder ID sets differ")

    paired_ids = sorted(packet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.embedding_model, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.embedding_model, local_files_only=True
    ).to(device)
    phrases_a = [
        str(coder_a[item_id]["primary_action"] or "no meaningful action")
        for item_id in paired_ids
    ]
    phrases_b = [
        str(coder_b[item_id]["primary_action"] or "no meaningful action")
        for item_id in paired_ids
    ]
    embeddings = _encode(
        phrases_a + phrases_b,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    paired_cosines = np.sum(
        embeddings[: len(paired_ids)] * embeddings[len(paired_ids) :],
        axis=1,
    )

    agreement: dict[str, Any] = {}
    for field in BOOLEAN_FIELDS:
        left = [bool(coder_a[item_id][field]) for item_id in paired_ids]
        right = [bool(coder_b[item_id][field]) for item_id in paired_ids]
        agreement[field] = {
            "coder_a_positive": sum(left),
            "coder_b_positive": sum(right),
            "exact_agreement": sum(a == b for a, b in zip(left, right))
            / len(left),
            "cohen_kappa": float(cohen_kappa_score(left, right)),
            "disagreement_count": sum(a != b for a, b in zip(left, right)),
        }

    paired_rows: list[dict[str, Any]] = []
    core_ids: set[str] = set()
    reusability_only: list[str] = []
    for index, item_id in enumerate(paired_ids):
        a = coder_a[item_id]
        b = coder_b[item_id]
        field_disagreements = [
            field for field in BOOLEAN_FIELDS if bool(a[field]) != bool(b[field])
        ]
        risk_positive = bool(
            a["clear_risk_or_boundary_problem"]
            or b["clear_risk_or_boundary_problem"]
        )
        validation_flags = [
            str(flag)
            for flag in (
                list(a.get("validation_flags") or [])
                + list(b.get("validation_flags") or [])
            )
        ]
        serious_validation_flag = any(
            flag != "evidence_excerpt_not_literal"
            for flag in validation_flags
        )
        nonliteral_evidence = (
            "evidence_excerpt_not_literal" in validation_flags
        )
        core = bool(
            risk_positive
            or serious_validation_flag
            or any(field in field_disagreements for field in CORE_FIELDS)
        )
        if core:
            core_ids.add(item_id)
        elif "reusable_as_general_technique" in field_disagreements:
            reusability_only.append(item_id)
        paired_rows.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": item_id,
                "primary_action_cosine": round(
                    float(paired_cosines[index]), 6
                ),
                "boolean_disagreements": field_disagreements,
                "any_coder_risk_positive": risk_positive,
                "any_serious_model_validation_flag": (
                    serious_validation_flag
                ),
                "any_nonliteral_evidence_excerpt": nonliteral_evidence,
                "coder_a": {
                    field: a[field]
                    for field in (
                        *BOOLEAN_FIELDS,
                        "primary_action",
                        "secondary_action",
                        "risk_description",
                        "response_evidence_excerpt",
                        "confidence",
                        "validation_flags",
                    )
                },
                "coder_b": {
                    field: b[field]
                    for field in (
                        *BOOLEAN_FIELDS,
                        "primary_action",
                        "secondary_action",
                        "risk_description",
                        "response_evidence_excerpt",
                        "confidence",
                        "validation_flags",
                    )
                },
            }
        )

    reusability_selected = set(
        sorted(
            reusability_only,
            key=lambda item_id: (
                _hash_fraction(item_id, "reusability-v1"),
                item_id,
            ),
        )[: args.reusability_disagreement_sample]
    )
    remaining_agreement = [
        item_id
        for item_id in paired_ids
        if item_id not in core_ids and item_id not in reusability_selected
    ]
    agreement_selected = {
        item_id
        for item_id in remaining_agreement
        if _hash_fraction(item_id, "agreement-calibration-v1")
        < args.agreement_sample_rate
    }

    paired_by_id = {row["blind_item_id"]: row for row in paired_rows}
    review_rows: list[dict[str, Any]] = []
    for item_id in paired_ids:
        reasons: list[str] = []
        if item_id in core_ids:
            reasons.append("core semantic/information/self-disclosure/risk disagreement")
            tier = 1
        elif item_id in reusability_selected:
            reasons.append("fixed sample of reusability-only disagreement")
            tier = 2
        elif item_id in agreement_selected:
            reasons.append("fixed 20% calibration sample from remaining pool")
            tier = 3
        else:
            continue
        pair = paired_by_id[item_id]
        review_rows.append(
            {
                "blind_item_id": item_id,
                "review_tier": tier,
                "selection_reasons": reasons,
                "recent_dialogue": packet[item_id]["recent_dialogue"],
                "supporter_response": packet[item_id]["supporter_response"],
                "primary_action_cosine": pair["primary_action_cosine"],
                "boolean_disagreements": pair["boolean_disagreements"],
                "coder_a": pair["coder_a"],
                "coder_b": pair["coder_b"],
            }
        )
    review_rows.sort(key=lambda row: (row["review_tier"], row["blind_item_id"]))

    report = {
        "protocol": PROTOCOL,
        "status": "TWO_PASS_AGGREGATED_HUMAN_ADJUDICATION_PENDING",
        "packet_rows": len(packet),
        "coder_a": args.coder_a_endpoint,
        "coder_b": args.coder_b_endpoint,
        "exact_two_pass_coverage": True,
        "agreement": agreement,
        "primary_action_cosine_role": (
            "BGE phrase-similarity diagnostic only; not semantic gold or "
            "automatic adjudication."
        ),
        "primary_action_cosine_quantiles": {
            key: round(float(value), 6)
            for key, value in zip(
                ("p10", "p25", "p50", "p75", "p90"),
                np.quantile(paired_cosines, [0.10, 0.25, 0.50, 0.75, 0.90]),
                strict=True,
            )
        },
        "evidence_surface_diagnostic": {
            "coder_a_nonliteral_excerpt_rows": sum(
                "evidence_excerpt_not_literal"
                in (coder_a[item_id].get("validation_flags") or [])
                for item_id in paired_ids
            ),
            "coder_b_nonliteral_excerpt_rows": sum(
                "evidence_excerpt_not_literal"
                in (coder_b[item_id].get("validation_flags") or [])
                for item_id in paired_ids
            ),
            "role": (
                "Provider-surface quality diagnostic only. The human page "
                "shows the complete source response, so nonliteral excerpts "
                "do not by themselves expand semantic adjudication."
            ),
        },
        "human_review": {
            "tier_1_all_core_disagreement_or_risk": len(core_ids),
            "tier_2_reusability_only_fixed_sample": len(
                reusability_selected
            ),
            "tier_3_remaining_fixed_rate_sample": len(agreement_selected),
            "total_rows": len(review_rows),
            "full_160_human_review_required": False,
            "unreviewed_model_consensus_is_gold": False,
        },
        "incomplete_sensitivity_not_used": {
            "GLM_complete_rows": 16,
            "reason": "Endpoint was too slow and was interrupted; incomplete rows do not enter the two-pass primary aggregation.",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "paired_model_codes.jsonl", paired_rows)
    write_jsonl(args.out_dir / "human_review_packet.jsonl", review_rows)
    (args.out_dir / "human_review.html").write_text(
        _render_html(review_rows), encoding="utf-8"
    )
    core_review_rows = [
        row for row in review_rows if int(row["review_tier"]) == 1
    ]
    optional_review_rows = [
        row for row in review_rows if int(row["review_tier"]) > 1
    ]
    write_jsonl(
        args.out_dir / "human_review_core_packet.jsonl", core_review_rows
    )
    write_jsonl(
        args.out_dir / "human_review_optional_calibration_packet.jsonl",
        optional_review_rows,
    )
    (args.out_dir / "human_review_core.html").write_text(
        _render_html(core_review_rows), encoding="utf-8"
    )
    (args.out_dir / "human_review_optional_calibration.html").write_text(
        _render_html(optional_review_rows), encoding="utf-8"
    )
    write_json(args.out_dir / "aggregation_report.json", report)
    print(canonical_json({"output": str(args.out_dir), **report}))


if __name__ == "__main__":
    main()
