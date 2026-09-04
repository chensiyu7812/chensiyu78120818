#!/usr/bin/env python3
"""Analyze sealed H2 labels and promote at most one qualified RS ranker."""

from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-h2-retrieval-qualification-analysis-v1"
REVIEW_PROTOCOL = "pm-v1.5-h2-retrieval-human-qualification-v1"


def _confusion(
    rows: list[dict[str, Any]],
    *,
    prediction_field: str,
) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        actual = row["human_opportunity"] == "yes"
        predicted = bool(row[prediction_field])
        key = (
            "tp" if actual and predicted
            else "tn" if not actual and not predicted
            else "fp" if not actual and predicted
            else "fn"
        )
        counts[key] += 1
    sensitivity = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    specificity = counts["tn"] / max(1, counts["tn"] + counts["fp"])
    return {
        "tp": counts["tp"],
        "tn": counts["tn"],
        "fp": counts["fp"],
        "fn": counts["fn"],
        "accuracy": round(
            (counts["tp"] + counts["tn"]) / max(1, len(rows)), 8
        ),
        "balanced_accuracy": round((sensitivity + specificity) / 2, 8),
        "sensitivity": round(sensitivity, 8),
        "specificity": round(specificity, 8),
    }


def _method_metrics(
    rows: list[dict[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    opportunities = [
        row for row in rows if row["human_opportunity"] == "yes"
    ]
    top1_correct = 0
    top3_hit = 0
    hard_items = sum(
        bool(
            set(row["hard_exclusion_candidate_numbers"])
            & set(row[f"{method}_ranked_candidate_numbers"][:3])
        )
        for row in rows
    )
    for row in opportunities:
        ranked = list(row[f"{method}_ranked_candidate_numbers"])
        acceptable = set(row["acceptable_candidate_numbers"])
        if ranked and ranked[0] in acceptable:
            top1_correct += 1
        if acceptable & set(ranked[:3]):
            top3_hit += 1
    denominator = len(opportunities)
    return {
        "opportunity_items": denominator,
        "top1_correct": top1_correct,
        "top1_acceptable_rate": round(
            top1_correct / max(1, denominator), 8
        ),
        "top3_hit_items": top3_hit,
        "top3_acceptable_recall": round(
            top3_hit / max(1, denominator), 8
        ),
        "hard_exclusion_items_in_top3": hard_items,
        "coverage_gap_items": sum(
            not row["acceptable_candidate_numbers"]
            for row in opportunities
        ),
    }


def _bind_annotations(
    *,
    annotations_path: Path,
    packet_path: Path,
    private_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations = [dict(row) for row in iter_jsonl(annotations_path)]
    packet = [dict(row) for row in iter_jsonl(packet_path)]
    private = {
        str(row["review_item_id"]): dict(row)
        for row in iter_jsonl(private_path)
    }
    packet_by_id = {str(row["review_item_id"]): row for row in packet}
    annotation_by_id = {
        str(row["review_item_id"]): row for row in annotations
    }
    expected = set(packet_by_id)
    if (
        len(annotations) != 48
        or len(annotation_by_id) != 48
        or set(annotation_by_id) != expected
        or set(private) != expected
    ):
        raise RuntimeError("H2 annotations must exactly cover 48 packet ids")
    bound: list[dict[str, Any]] = []
    for review_id in sorted(expected):
        annotation = annotation_by_id[review_id]
        public = packet_by_id[review_id]
        audit = private[review_id]
        if annotation.get("protocol") != REVIEW_PROTOCOL:
            raise RuntimeError(f"wrong H2 review protocol: {review_id}")
        if annotation.get("reviewed") is not True:
            raise RuntimeError(f"unreviewed H2 item: {review_id}")
        if str(annotation.get("h2_state_id")) != str(
            public["h2_state_id"]
        ):
            raise RuntimeError(f"H2 state binding mismatch: {review_id}")
        opportunity = str(annotation.get("rs_opportunity") or "")
        if opportunity not in {"yes", "no", "uncertain"}:
            raise RuntimeError(f"invalid H2 opportunity label: {review_id}")
        candidate_numbers = {
            int(row["candidate_number"]) for row in public["candidates"]
        }
        acceptable = list(
            annotation.get("acceptable_candidate_numbers") or []
        )
        hard = list(annotation.get("hard_exclusion_candidate_numbers") or [])
        if (
            len(acceptable) != len(set(acceptable))
            or len(hard) != len(set(hard))
            or not set(acceptable) <= candidate_numbers
            or not set(hard) <= candidate_numbers
            or set(acceptable) & set(hard)
        ):
            raise RuntimeError(f"invalid H2 candidate judgments: {review_id}")
        if opportunity == "no" and acceptable:
            raise RuntimeError(
                f"RS opportunity=no cannot accept a candidate: {review_id}"
            )
        bound.append(
            {
                "review_item_id": review_id,
                "h2_state_id": public["h2_state_id"],
                "human_opportunity": opportunity,
                "acceptable_candidate_numbers": acceptable,
                "hard_exclusion_candidate_numbers": hard,
                "notes": annotation.get("notes", ""),
                "transparent_ranked_candidate_numbers": audit[
                    "transparent_ranked_candidate_numbers"
                ],
                "bge_ranked_candidate_numbers": audit[
                    "bge_ranked_candidate_numbers"
                ],
                "transparent_on": bool(
                    audit["transparent_ranked_candidate_numbers"]
                ),
                "bge_on": bool(audit["bge_ranked_candidate_numbers"]),
                "selection_stratum": audit["selection_stratum"],
            }
        )
    return annotations, bound


def _render_report(summary: dict[str, Any]) -> str:
    rows = ""
    for method, value in summary["methods"].items():
        rows += (
            "<tr>"
            f"<td>{html.escape(method)}</td>"
            f"<td>{value['top1_acceptable_rate']:.1%}</td>"
            f"<td>{value['top3_acceptable_recall']:.1%}</td>"
            f"<td>{value['hard_exclusion_items_in_top3']}</td>"
            f"<td>{value['coverage_gap_items']}</td>"
            "</tr>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>PM V1.5 H2 Retrieval Qualification</title><style>
body{{font-family:system-ui,sans-serif;max-width:920px;margin:auto;padding:24px;line-height:1.55}}
.box{{background:#eef3f8;border-left:5px solid #245f9e;padding:14px}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}th,td{{border:1px solid #ccd5df;padding:8px}}
</style></head><body><h1>PM V1.5 H2 Retrieval Qualification</h1>
<div class="box"><b>{html.escape(summary['status'])}</b><br>
{html.escape(summary['decision']['plain_language'])}</div>
<p>Opportunity definitive={summary['opportunity']['definitive_items']}/48，
uncertain={summary['opportunity']['uncertain_items']}；
gate balanced accuracy={summary['opportunity']['balanced_accuracy']:.1%}。</p>
<table><thead><tr><th>Ranker</th><th>Top‑1 acceptable</th>
<th>Top‑3 recall</th><th>Hard exclusions</th><th>Coverage gaps</th>
</tr></thead><tbody>{rows}</tbody></table>
<p>H2 只选择检索器；冻结 Bank 内容不因结果改变。未过门时 RS 保持 formal disabled，
结果作为第一篇的限制，不再在同一 H2 上修规则。</p></body></html>"""


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
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate/"
        "h2_review_packet.jsonl",
    )
    parser.add_argument(
        "--private-audit",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate/"
        "private_ranker_audit.jsonl",
    )
    parser.add_argument(
        "--review-manifest",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate/"
        "manifest.json",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_h2_retrieval_qualification_v1",
    )
    args = parser.parse_args()

    manifest = json.loads(args.review_manifest.read_text(encoding="utf-8"))
    gates = dict(manifest["qualification_gates"])
    annotations, bound = _bind_annotations(
        annotations_path=args.annotations,
        packet_path=args.packet,
        private_path=args.private_audit,
    )
    definitive = [
        row for row in bound if row["human_opportunity"] != "uncertain"
    ]
    uncertain_items = len(bound) - len(definitive)
    uncertain_rate = uncertain_items / len(bound)
    transparent_confusion = _confusion(
        definitive, prediction_field="transparent_on"
    )
    bge_confusion = _confusion(definitive, prediction_field="bge_on")
    if transparent_confusion != bge_confusion:
        raise RuntimeError(
            "transparent/BGE opportunity gates unexpectedly differ"
        )
    methods = {
        method: _method_metrics(definitive, method=method)
        for method in ("transparent", "bge")
    }
    bge_gain = (
        methods["bge"]["top1_acceptable_rate"]
        - methods["transparent"]["top1_acceptable_rate"]
    )
    bge_selected = (
        bge_gain >= 0.05
        and methods["bge"]["top3_acceptable_recall"]
        >= methods["transparent"]["top3_acceptable_recall"]
        and methods["bge"]["hard_exclusion_items_in_top3"]
        <= methods["transparent"]["hard_exclusion_items_in_top3"]
    )
    selected = "bge" if bge_selected else "transparent"
    selected_metrics = methods[selected]
    opportunity_pass = (
        uncertain_rate
        <= float(gates["opportunity_uncertain_rate_maximum"])
        and transparent_confusion["balanced_accuracy"]
        >= float(gates["opportunity_balanced_accuracy_minimum"])
    )
    retrieval_pass = (
        selected_metrics["top1_acceptable_rate"]
        >= float(gates["top1_acceptable_minimum"])
        and selected_metrics["top3_acceptable_recall"]
        >= float(gates["top3_acceptable_recall_minimum"])
        and selected_metrics["hard_exclusion_items_in_top3"]
        <= int(gates["hard_exclusion_in_method_top3_maximum"])
    )
    qualified = opportunity_pass and retrieval_pass
    status = (
        "H2_QUALIFIED_FORMAL_RS_ENABLED"
        if qualified
        else "H2_NOT_QUALIFIED_FORMAL_RS_REMAINS_DISABLED"
    )
    summary = {
        "protocol": PROTOCOL,
        "status": status,
        "opportunity": {
            "definitive_items": len(definitive),
            "uncertain_items": uncertain_items,
            "uncertain_rate": round(uncertain_rate, 8),
            **transparent_confusion,
            "gate_passed": opportunity_pass,
        },
        "methods": methods,
        "bge_minus_transparent_top1": round(bge_gain, 8),
        "decision": {
            "selected_ranker": selected,
            "bge_selected": bge_selected,
            "retrieval_gate_passed": retrieval_pass,
            "formal_rs_enabled": qualified,
            "plain_language": (
                f"Selected {selected}. "
                + (
                    "Opportunity and retrieval gates passed."
                    if qualified
                    else
                    "At least one preregistered gate failed; do not tune on H2."
                )
            ),
        },
        "qualification_gates": gates,
        "bank_content_changed_after_h2": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = args.out_dir / "h2_annotations_frozen.jsonl"
    bound_path = args.out_dir / "h2_bound_diagnostics.jsonl"
    _write_jsonl(
        annotations_path,
        sorted(annotations, key=lambda row: str(row["review_item_id"])),
    )
    _write_jsonl(bound_path, bound)
    _write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "report.html").write_text(
        _render_report(summary), encoding="utf-8"
    )

    bank = [dict(row) for row in iter_jsonl(args.bank)]
    promoted_path = args.out_dir / "strategy_cards_v4_h2_qualified.jsonl"
    if qualified:
        for row in bank:
            row["eligible_for_formal_rs"] = True
            row["quality_status"] = "H2_RETRIEVAL_QUALIFIED_FORMAL_RS"
            row["selected_ranker"] = selected
            row["h2_qualification_protocol"] = PROTOCOL
        _write_jsonl(promoted_path, bank)
    elif promoted_path.exists():
        raise RuntimeError(
            "stale promoted Bank exists in a failed H2 output directory; "
            "use a fresh output directory"
        )
    output_paths = [
        annotations_path,
        bound_path,
        args.out_dir / "summary.json",
        args.out_dir / "report.html",
    ]
    if promoted_path.is_file():
        output_paths.append(promoted_path)
    final_manifest = {
        "protocol": PROTOCOL,
        "status": status,
        "formal_rs_enabled": qualified,
        "selected_ranker": selected,
        "inputs": {
            "annotations_original_path": str(args.annotations),
            "annotations_sha256": sha256_file(args.annotations),
            "packet": str(args.packet.relative_to(ROOT)),
            "private_audit": str(args.private_audit.relative_to(ROOT)),
            "review_manifest": str(args.review_manifest.relative_to(ROOT)),
            "bank": str(args.bank.relative_to(ROOT)),
            "bank_sha256": sha256_file(args.bank),
        },
        "outputs": {
            path.name: sha256_file(path) for path in output_paths
        },
    }
    _write_json(args.out_dir / "manifest.json", final_manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": status,
            "selected_ranker": selected,
            "opportunity_ba": transparent_confusion["balanced_accuracy"],
            "methods": methods,
            "formal_rs_enabled": qualified,
        }
    )


if __name__ == "__main__":
    main()
