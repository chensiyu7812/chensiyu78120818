#!/usr/bin/env python3
"""Reinterpret consumed H2 labels under the corrected RS effect construct."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h2-rs-construct-reinterpretation-v1"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    ranked_key = f"{method}_ranked_candidate_numbers"
    opportunity = [r for r in rows if r["human_opportunity"] == "yes"]
    on = [r for r in rows if r[ranked_key]]
    acceptable = [
        r
        for r in opportunity
        if r[ranked_key]
        and r[ranked_key][0] in r["acceptable_candidate_numbers"]
    ]
    broad_prohibited = [
        r
        for r in rows
        if r[ranked_key]
        and r[ranked_key][0] in r["hard_exclusion_candidate_numbers"]
    ]
    opportunity_nonacceptable_realized = [
        r
        for r in opportunity
        if r[ranked_key]
        and r[ranked_key][0] not in r["acceptable_candidate_numbers"]
    ]
    top3_hit = [
        r
        for r in opportunity
        if set(r[ranked_key][:3]) & set(r["acceptable_candidate_numbers"])
    ]
    top3_broad_prohibited = [
        r
        for r in rows
        if set(r[ranked_key][:3])
        & set(r["hard_exclusion_candidate_numbers"])
    ]
    return {
        "retrieval_on_items": len(on),
        "human_opportunity_items": len(opportunity),
        "top1_acceptable_on_human_opportunity": len(acceptable),
        "top1_acceptable_rate_on_human_opportunity": round(
            len(acceptable) / max(1, len(opportunity)), 8
        ),
        "top1_broad_must_not_inject_items_all_states": len(
            broad_prohibited
        ),
        "top1_broad_must_not_inject_rate_when_retrieval_on": round(
            len(broad_prohibited) / max(1, len(on)), 8
        ),
        "realized_top1_nonacceptable_on_human_opportunity": len(
            opportunity_nonacceptable_realized
        ),
        "top3_acceptable_recall_diagnostic": round(
            len(top3_hit) / max(1, len(opportunity)), 8
        ),
        "top3_broad_must_not_inject_items_diagnostic": len(
            top3_broad_prohibited
        ),
        "top1_broad_must_not_inject_review_ids": [
            r["review_item_id"] for r in broad_prohibited
        ],
    }


def _render(summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(method)}</td>"
        f"<td>{value['top1_acceptable_rate_on_human_opportunity']:.1%}</td>"
        f"<td>{value['top1_broad_must_not_inject_items_all_states']}</td>"
        f"<td>{value['top3_acceptable_recall_diagnostic']:.1%}</td>"
        f"<td>{value['top3_broad_must_not_inject_items_diagnostic']}</td>"
        "</tr>"
        for method, value in summary["retrieval_diagnostics"].items()
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>H2 RS Construct Reinterpretation</title><style>
body{{font-family:system-ui,sans-serif;max-width:920px;margin:auto;padding:24px;line-height:1.55}}
.box{{background:#eef3f8;border-left:5px solid #245f9e;padding:14px}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}th,td{{border:1px solid #ccd5df;padding:8px}}
</style></head><body><h1>H2：检索诊断，不是 RS 开关标签</h1>
<div class="box"><b>{html.escape(summary['status'])}</b><br>
H2 保留为 ranker、scope gate 与候选适用性诊断；RS/R0 成对效果生成不再被平均
Top‑1/Top‑3 门阻断。</div>
<table><thead><tr><th>方法</th><th>Top‑1 acceptable</th>
<th>Top‑1 broad must-not</th><th>Top‑3 recall（诊断）</th>
<th>Top‑3 broad must-not（诊断）</th></tr></thead><tbody>{rows}</tbody></table>
<p>运行时只注入 Top‑1。Top‑3 只用于覆盖诊断，不能作为 material response risk，
也不能代替同状态 RS/R0 回复对。透明方法固定为 V1.5 paired-effect treatment；
BGE 不采用。</p></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h2-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h2_retrieval_qualification_v1",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/rs_open_close_construct_v2.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h2_rs_construct_reinterpretation_v1",
    )
    args = parser.parse_args()

    old_summary = json.loads(
        (args.h2_dir / "summary.json").read_text(encoding="utf-8")
    )
    rows = [
        dict(row)
        for row in iter_jsonl(args.h2_dir / "h2_bound_diagnostics.jsonl")
    ]
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (
        len(rows) != 48
        or old_summary["decision"]["selected_ranker"] != "transparent"
        or contract["status"]
        != "ACTIVE_CORRECTED_BEFORE_NEW_PAIRED_OUTCOMES"
    ):
        raise RuntimeError("H2/construct inputs are not the frozen expected set")

    diagnostics = {
        method: _method(rows, method) for method in ("transparent", "bge")
    }
    strata = Counter(str(row["selection_stratum"]) for row in rows)
    transparent = diagnostics["transparent"]
    capacity_supported = (
        transparent["top1_acceptable_on_human_opportunity"]
        >= int(
            contract["minimum_before_RS_head_fit"][
                "independent_material_benefit_groups"
            ]
        )
        and transparent[
            "realized_top1_nonacceptable_on_human_opportunity"
        ]
        >= int(
            contract["minimum_before_RS_head_fit"][
                "independent_nonbenefit_groups_with_realized_treatment"
            ]
        )
    )
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "H2_REINTERPRETED_PAIRED_EFFECT_GENERATION_AUTHORIZED"
            if capacity_supported
            else "H2_REINTERPRETED_CAPACITY_SAMPLE_INSUFFICIENT"
        ),
        "selected_fixed_ranker": "transparent",
        "bge_adopted": False,
        "h2_scientific_role": [
            "ranker_comparison",
            "structural_scope_gate_diagnostic",
            "candidate_suitability_and_coverage_diagnostic",
        ],
        "h2_forbidden_roles": [
            "RS_response_benefit_gold",
            "material_response_risk_gold",
            "always_off_conclusion",
        ],
        "old_formal_rs_enabled_field_withdrawn_as_scientific_conclusion": True,
        "runtime_inject_top_k": 1,
        "retrieval_diagnostics": diagnostics,
        "selection_strata": dict(sorted(strata.items())),
        "capacity_evidence_only": {
            "enough_top1_acceptable_candidates_to_attempt_new_pairs": (
                transparent[
                    "top1_acceptable_on_human_opportunity"
                ]
            ),
            "enough_realized_nonacceptable_candidates_to_expect_off_examples": (
                transparent[
                    "realized_top1_nonacceptable_on_human_opportunity"
                ]
            ),
            "supported": capacity_supported,
            "not_a_pair_outcome_label": True,
        },
        "next_stage": (
            "Select fresh outcome-blind ESConv-train states with zero H1/H2/"
            "Bank-source overlap, freeze transparent Top-1, and generate same-"
            "stack R0/RS pairs. Obtain actual quality/risk/cost labels before "
            "fitting the RS head."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "report.html").write_text(
        _render(summary), encoding="utf-8"
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": summary["status"],
        "inputs": {
            "old_h2_summary": sha256_file(args.h2_dir / "summary.json"),
            "h2_bound_diagnostics": sha256_file(
                args.h2_dir / "h2_bound_diagnostics.jsonl"
            ),
            "construct": sha256_file(args.contract),
        },
        "outputs": {
            "summary.json": sha256_file(args.out_dir / "summary.json"),
            "report.html": sha256_file(args.out_dir / "report.html"),
        },
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": summary["status"],
            "transparent_top1_acceptable": transparent[
                "top1_acceptable_on_human_opportunity"
            ],
            "transparent_realized_nonacceptable": transparent[
                "realized_top1_nonacceptable_on_human_opportunity"
            ],
        }
    )


if __name__ == "__main__":
    main()
