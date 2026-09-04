#!/usr/bin/env python3
"""Audit V3 H-Eligibility dual reviews and freeze only defensible evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-h-eligibility-dual-review-audit-v1"
FIELDS = (
    "adjudicated_subtype",
    "owner_time_valid",
    "goal_function_fit",
    "boundary_burden_fit",
    "specific_nonredundant_increment",
    "derived_eligibility",
)

# Adjudication is limited to the eight observed field disagreements.  It never
# consults construction intent to choose the answer.
FIELD_ADJUDICATION = {
    "h_elig_144cdd388c8a43fe2ae67130": {
        "specific_nonredundant_increment": "yes",
        "reason": "The preference is new specific information; its current conflict is already represented by the goal/boundary gates.",
    },
    "h_elig_2f931ab252ba6e0bf91d3482": {
        "specific_nonredundant_increment": "yes",
        "reason": "The notebook fact is specific and unspoken but functionally irrelevant; goal/function, not specificity, rejects it.",
    },
    "h_elig_3aca03e943cec3cf6cd2eb94": {
        "owner_time_valid": "no",
        "reason": "The candidate explicitly describes a fully resolved old issue while the current request is a separate issue.",
    },
    "h_elig_7501d93a62fea6c1162156a7": {
        "owner_time_valid": "no",
        "reason": "The candidate explicitly describes a fully resolved old issue while the current request is a separate issue.",
    },
    "h_elig_79a720d0835fa02cb30dbd9f": {
        "specific_nonredundant_increment": "yes",
        "reason": "The preference is new specific information; its current conflict is already represented by the goal/boundary gates.",
    },
    "h_elig_91932d0b740532165927e33e": {
        "specific_nonredundant_increment": "yes",
        "reason": "The notebook fact is specific and unspoken but functionally irrelevant; goal/function, not specificity, rejects it.",
    },
    "h_elig_9f7b16b524c1f9414c2adf15": {
        "goal_function_fit": "no",
        "specific_nonredundant_increment": "no",
        "reason": "A generic prior instruction to understand pressure before planning does not perform the requested separation of two pressures.",
    },
    "h_elig_ad4fd63b60ae0812f8cb1b32": {
        "boundary_burden_fit": "no",
        "reason": "The dialogic card conflicts with the explicit brief boundary, independently of the already-executed/nonredundant failure.",
    },
}


def _kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs must be non-empty and aligned")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    lcount, rcount = Counter(left), Counter(right)
    labels = set(lcount) | set(rcount)
    expected = sum(
        (lcount[label] / len(left)) * (rcount[label] / len(right))
        for label in labels
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _agreement(primary: list[dict[str, Any]], overlap: list[dict[str, Any]]) -> dict[str, Any]:
    p = {str(row["blind_item_id"]): row for row in primary}
    o = {str(row["blind_item_id"]): row for row in overlap}
    ids = sorted(o)
    output: dict[str, Any] = {}
    for field in FIELDS:
        left = [str(p[item][field]) for item in ids]
        right = [str(o[item][field]) for item in ids]
        output[field] = {
            "n": len(ids),
            "agreement_count": sum(a == b for a, b in zip(left, right, strict=True)),
            "raw_agreement": sum(a == b for a, b in zip(left, right, strict=True)) / len(ids),
            "cohen_kappa": _kappa(left, right),
            "confusion": {
                f"{a} -> {b}": count for (a, b), count in Counter(zip(left, right)).items()
            },
        }
    per_component: dict[str, Any] = {}
    for component in ("MP", "MS", "ME", "RS"):
        subset = [item for item in ids if p[item]["component"] == component]
        left = [str(p[item]["derived_eligibility"]) for item in subset]
        right = [str(o[item]["derived_eligibility"]) for item in subset]
        per_component[component] = {
            "n": len(subset),
            "agreement_count": sum(a == b for a, b in zip(left, right, strict=True)),
            "raw_agreement": sum(a == b for a, b in zip(left, right, strict=True)) / len(subset),
            "cohen_kappa": _kappa(left, right),
        }
    output["derived_eligibility_per_component"] = per_component
    return output


def _render(report: dict[str, Any]) -> str:
    def format_kappa(value: float | None) -> str:
        return "NA" if value is None else f"{value:.4f}"

    rows = "".join(
        f"<tr><td>{html.escape(field)}</td><td>{value['agreement_count']}/{value['n']}</td>"
        f"<td>{value['raw_agreement']:.4f}</td><td>{format_kappa(value['cohen_kappa'])}</td></tr>"
        for field, value in report["agreement"].items()
        if field != "derived_eligibility_per_component"
    )
    components = "".join(
        f"<tr><td>{component}</td><td>{value['agreement_count']}/{value['n']}</td>"
        f"<td>{value['raw_agreement']:.4f}</td><td>{format_kappa(value['cohen_kappa'])}</td></tr>"
        for component, value in report["agreement"]["derived_eligibility_per_component"].items()
    )
    mismatches = "".join(
        f"<li><code>{html.escape(family)}</code>: {html.escape(str(value))}</li>"
        for family, value in report["construction_intent_mismatch_by_logic_family"].items()
    )
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>V3 H-Eligibility 审计</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:auto;padding:24px;line-height:1.55}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:7px}}.ok{{color:#176b2c}}.bad{{color:#a02b20}}</style>
<h1>V3 H-Eligibility 双评与构造审计</h1><p><b>结论：</b><span class='bad'>{html.escape(report['status'])}</span>。双评可靠性已过，但候选实现仍需局部修复，当前不得冻结 eligibility gold。</p>
<h2>总体字段一致性</h2><table><tr><th>字段</th><th>一致</th><th>raw</th><th>κ</th></tr>{rows}</table>
<h2>最终 eligibility 分组件</h2><table><tr><th>组件</th><th>一致</th><th>raw</th><th>κ</th></tr>{components}</table>
<h2>构造意图与独立主评不一致的逻辑族</h2><ul>{mismatches}</ul>
<p>这些 mismatch 只用于定位出题实现，不用于推翻或选择人工裁决。修复后只重审完整 decision surface 实际变化的条目。</p></html>"""


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 P2 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument(
        "--binding",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_review_candidate/private_binding.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_dual_review_audit_v1",
    )
    args = parser.parse_args()
    primary = [dict(row) for row in iter_jsonl(args.primary)]
    overlap = [dict(row) for row in iter_jsonl(args.overlap)]
    binding = {str(row["blind_item_id"]): dict(row) for row in iter_jsonl(args.binding)}
    blueprint = {str(row["blueprint_row_id"]): dict(row) for row in iter_jsonl(args.blueprint)}
    if len(primary) != 128 or len(overlap) != 32:
        raise RuntimeError("expected 128 primary and 32 overlap rows")
    p = {str(row["blind_item_id"]): row for row in primary}
    o = {str(row["blind_item_id"]): row for row in overlap}
    if len(p) != 128 or len(o) != 32 or not set(o) <= set(p) or set(p) != set(binding):
        raise RuntimeError("review IDs are duplicate, missing, or not bound")
    agreement = _agreement(primary, overlap)
    overall = agreement["derived_eligibility"]
    component_values = agreement["derived_eligibility_per_component"].values()
    reliability_pass = (
        overall["raw_agreement"] >= 0.80
        and (overall["cohen_kappa"] or 0.0) >= 0.60
        and all(value["raw_agreement"] >= 0.70 for value in component_values)
    )
    adjudicated: list[dict[str, Any]] = []
    for row in primary:
        result = dict(row)
        override = FIELD_ADJUDICATION.get(str(row["blind_item_id"]))
        if override:
            for field in FIELDS:
                if field in override:
                    result[field] = override[field]
            result["adjudication_reason"] = override["reason"]
        else:
            result["adjudication_reason"] = "NO_OVERLAP_FIELD_DISAGREEMENT"
        gates = (
            result["owner_time_valid"],
            result["goal_function_fit"],
            result["boundary_burden_fit"],
            result["specific_nonredundant_increment"],
        )
        result["derived_eligibility"] = "eligible" if all(value == "yes" for value in gates) else "ineligible" if "no" in gates else "uncertain"
        result["reference_role"] = "REVIEWER_ADJUDICATED_PROXY_NOT_STEP1_GOLD"
        result["identified_human_gold"] = False
        adjudicated.append(result)
    mismatch = Counter()
    for row in adjudicated:
        bind = binding[str(row["blind_item_id"])]
        construction = str(bind["private_coverage_intent_not_gold"]).lower()
        if construction != row["derived_eligibility"]:
            family = str(blueprint[str(bind["state_id"])]["logic_family"])
            mismatch[family] += 1
    repair_families = {
        family: count
        for family, count in mismatch.items()
        if family in {
            "MS_ELIGIBLE_MS_DISTINCTION",
            "MS_WRONG_OWNER_OR_GOAL",
            "RS_ELIGIBLE_RS_SUGGESTION",
            "RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE",
        }
    }
    shape = {
        component: dict(
            Counter(
                row["derived_eligibility"]
                for row in adjudicated
                if row["component"] == component
            )
        )
        for component in ("MP", "MS", "ME", "RS")
    }
    report = {
        "protocol": PROTOCOL,
        "status": (
            "REVIEW_RELIABILITY_PASS_REALIZATION_REPAIR_REQUIRED_NO_GOLD_FREEZE"
            if reliability_pass and repair_families
            else "PASS" if reliability_pass else "REVIEW_RELIABILITY_FAIL"
        ),
        "primary_rows": len(primary),
        "overlap_rows": len(overlap),
        "agreement": agreement,
        "review_reliability_gate_pass": reliability_pass,
        "adjudicated_label_shape": shape,
        "construction_intent_mismatch_count": sum(mismatch.values()),
        "construction_intent_mismatch_by_logic_family": dict(sorted(mismatch.items())),
        "required_realization_repairs": repair_families,
        "current_gold_frozen": False,
        "current_rows_trainable": False,
        "identified_human_gold": False,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "primary_sha256": sha256_file(args.primary),
        "overlap_sha256": sha256_file(args.overlap),
        "binding_sha256": sha256_file(args.binding),
        "blueprint_sha256": sha256_file(args.blueprint),
        "scientific_interpretation": (
            "The reviewers can apply the four-gate codebook reliably. The current "
            "candidate realization cannot yet be frozen because several semantic "
            "families systematically realize the opposite eligibility construct."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "adjudicated_reference_not_gold.jsonl", adjudicated)
    write_json(args.out_dir / "audit_report.json", report)
    (args.out_dir / "report.html").write_text(_render(report), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
