#!/usr/bin/env python3
"""Validate V5.2 confirmation reviews and prepare the sole adjudication packet."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.2-confirmation-overlap-adjudication-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5.2-confirmation-quality-blind-v1"
RISK_PROTOCOL = "pm-v1.5-v5.2-confirmation-grounding-risk-v1"
REVIEW_DIR = ROOT / "outputs/pm_v1_5_v5_2_confirmation_review_v1_candidate"
OUT_DIR = ROOT / "outputs/pm_v1_5_v5_2_confirmation_adjudication_v1_candidate"
QUALITY_HELPER = ROOT / "scripts/v1_5/25zy_prepare_v5_full_fit_outcome_review_v1_5.py"
RISK_HELPER = ROOT / "scripts/v1_5/26i_prepare_v5_1_confirmation_review_v1_5.py"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load review helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_const_ann(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Parse the JSON-compatible object assigned to `const ANN` in a fill script."""
    text = path.read_text()
    marker = "const ANN ="
    marker_at = text.index(marker)
    start = text.index("{", marker_at)
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError(f"unterminated ANN object: {path}")
    duplicate_keys: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate_keys.append(key)
            result[key] = value
        return result

    value = json.loads(text[start:end], object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise ValueError(f"ANN is not an object: {path}")
    return value, duplicate_keys


def cohen_kappa(
    left: dict[str, str], right: dict[str, str], labels: tuple[str, ...]
) -> dict[str, float | int]:
    if set(left) != set(right) or not left:
        raise ValueError("kappa inputs must have identical nonempty IDs")
    ids = sorted(left)
    agreement = sum(left[item_id] == right[item_id] for item_id in ids)
    observed = agreement / len(ids)
    expected = sum(
        (sum(left[item_id] == label for item_id in ids) / len(ids))
        * (sum(right[item_id] == label for item_id in ids) / len(ids))
        for label in labels
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return {
        "n": len(ids),
        "agreement_count": agreement,
        "agreement_rate": observed,
        "chance_expected_agreement": expected,
        "cohen_kappa": kappa,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-primary", type=Path, required=True)
    parser.add_argument("--quality-overlap-script", type=Path, required=True)
    parser.add_argument("--risk-primary", type=Path, required=True)
    parser.add_argument("--risk-overlap-script", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--overlap-annotator-id", default="Claude_independent_overlap_sighted"
    )
    args = parser.parse_args()

    primary_quality_rows = jsonl(args.quality_primary)
    primary_risk_rows = jsonl(args.risk_primary)
    primary_quality = keyed(primary_quality_rows, "blind_item_id")
    primary_risk = keyed(primary_risk_rows, "risk_item_id")
    expected_primary_quality = keyed(
        jsonl(args.review_dir / "primary_quality_packet.jsonl"), "blind_item_id"
    )
    expected_primary_risk = keyed(
        jsonl(args.review_dir / "primary_risk_packet.jsonl"), "risk_item_id"
    )
    expected_overlap_quality = keyed(
        jsonl(args.review_dir / "overlap_quality_packet.jsonl"), "blind_item_id"
    )
    expected_overlap_risk = keyed(
        jsonl(args.review_dir / "overlap_risk_packet.jsonl"), "risk_item_id"
    )
    if set(primary_quality) != set(expected_primary_quality):
        raise ValueError("primary quality IDs do not match the frozen packet")
    if set(primary_risk) != set(expected_primary_risk):
        raise ValueError("primary risk IDs do not match the frozen packet")

    quality_ann, quality_duplicate_keys = extract_const_ann(args.quality_overlap_script)
    risk_ann, risk_duplicate_keys = extract_const_ann(args.risk_overlap_script)
    if set(quality_ann) != set(expected_overlap_quality):
        raise ValueError("overlap quality IDs do not match the frozen packet")
    if set(risk_ann) != {"yes", "notes"}:
        raise ValueError("overlap risk script must contain yes and notes maps")
    if set(risk_ann["yes"]) - set(expected_overlap_risk):
        raise ValueError("overlap risk yes map contains unknown IDs")
    if set(risk_ann["notes"]) - set(expected_overlap_risk):
        raise ValueError("overlap risk notes map contains unknown IDs")

    overlap_quality_rows: list[dict[str, Any]] = []
    for blind_id in sorted(expected_overlap_quality):
        label = dict(quality_ann[blind_id])
        overlap_quality_rows.append(
            {
                "protocol": QUALITY_PROTOCOL,
                "blind_item_id": blind_id,
                "quality_preference": str(label["quality_preference"]),
                "decisive_criterion": str(label["decisive_criterion"]),
                "quality_notes": str(label["quality_notes"]),
                "annotator_id": args.overlap_annotator_id,
            }
        )
    overlap_risk_rows: list[dict[str, Any]] = []
    for risk_id in sorted(expected_overlap_risk):
        evidence = dict(risk_ann["yes"].get(risk_id, {}))
        is_yes = bool(evidence)
        overlap_risk_rows.append(
            {
                "protocol": RISK_PROTOCOL,
                "risk_item_id": risk_id,
                "any_material_risk": "yes" if is_yes else "no",
                "selected_categories": list(evidence),
                "evidence_by_category": evidence,
                "risk_notes": str(risk_ann["notes"].get(risk_id, "")),
                "annotator_id": args.overlap_annotator_id,
            }
        )
    overlap_quality = keyed(overlap_quality_rows, "blind_item_id")
    overlap_risk = keyed(overlap_risk_rows, "risk_item_id")

    allowed_quality = {"A", "B", "tie"}
    allowed_risk = {"yes", "no"}
    if {str(row["quality_preference"]) for row in primary_quality_rows} - allowed_quality:
        raise ValueError("primary quality contains invalid or unresolved labels")
    if {str(row["quality_preference"]) for row in overlap_quality_rows} - allowed_quality:
        raise ValueError("overlap quality contains invalid or unresolved labels")
    if {str(row["any_material_risk"]) for row in primary_risk_rows} - allowed_risk:
        raise ValueError("primary risk contains invalid or unresolved labels")

    quality_disagreement_ids = sorted(
        blind_id
        for blind_id, overlap in overlap_quality.items()
        if str(primary_quality[blind_id]["quality_preference"])
        != str(overlap["quality_preference"])
    )
    quality_direct_reversals = [
        blind_id
        for blind_id in quality_disagreement_ids
        if {
            str(primary_quality[blind_id]["quality_preference"]),
            str(overlap_quality[blind_id]["quality_preference"]),
        }
        == {"A", "B"}
    ]
    risk_disagreement_ids: list[str] = []
    risk_decision_disagreement_ids: list[str] = []
    risk_category_disagreement_ids: list[str] = []
    for risk_id, overlap in overlap_risk.items():
        primary = primary_risk[risk_id]
        decision_differs = str(primary["any_material_risk"]) != str(
            overlap["any_material_risk"]
        )
        category_differs = (
            str(primary["any_material_risk"]) == "yes"
            and str(overlap["any_material_risk"]) == "yes"
            and set(primary["selected_categories"]) != set(overlap["selected_categories"])
        )
        if decision_differs:
            risk_decision_disagreement_ids.append(risk_id)
        if category_differs:
            risk_category_disagreement_ids.append(risk_id)
        if decision_differs or category_differs:
            risk_disagreement_ids.append(risk_id)

    quality_packet = [expected_overlap_quality[item_id] for item_id in quality_disagreement_ids]
    risk_packet = [expected_overlap_risk[item_id] for item_id in risk_disagreement_ids]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "verified_quality_primary.jsonl", primary_quality_rows)
    write_jsonl(args.out_dir / "verified_quality_overlap.jsonl", overlap_quality_rows)
    write_jsonl(args.out_dir / "verified_risk_primary.jsonl", primary_risk_rows)
    write_jsonl(args.out_dir / "verified_risk_overlap.jsonl", overlap_risk_rows)
    write_jsonl(args.out_dir / "quality_disagreement_packet.jsonl", quality_packet)
    write_jsonl(args.out_dir / "risk_disagreement_packet.jsonl", risk_packet)

    quality_helper = load_module(QUALITY_HELPER, "v52_confirmation_adjudication_quality")
    risk_helper = load_module(RISK_HELPER, "v52_confirmation_adjudication_risk")

    def html(path: Path, template: str, protocol: str, items: list[dict[str, Any]]) -> None:
        payload = {
            "manifest": {
                "protocol": PROTOCOL,
                "panel_role": "adjudication",
                "items": len(items),
            },
            "items": items,
        }
        path.write_text(
            template.replace("__DATA__", quality_helper._safe_script_json(payload)).replace(
                "__PROTOCOL__", protocol
            ),
            encoding="utf-8",
        )

    html(
        args.out_dir / "human_quality_adjudication.html",
        quality_helper.QUALITY_HTML,
        QUALITY_PROTOCOL,
        quality_packet,
    )
    html(
        args.out_dir / "human_risk_adjudication.html",
        risk_helper.RISK_HTML,
        RISK_PROTOCOL,
        risk_packet,
    )

    quality_left = {
        item_id: str(primary_quality[item_id]["quality_preference"])
        for item_id in overlap_quality
    }
    quality_right = {
        item_id: str(overlap_quality[item_id]["quality_preference"])
        for item_id in overlap_quality
    }
    risk_left = {
        item_id: str(primary_risk[item_id]["any_material_risk"])
        for item_id in overlap_risk
    }
    risk_right = {
        item_id: str(overlap_risk[item_id]["any_material_risk"])
        for item_id in overlap_risk
    }
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_CONCENTRATED_CONFIRMATION_ADJUDICATION",
        "review_data_quality": {
            "primary_quality_rows": len(primary_quality_rows),
            "overlap_quality_rows": len(overlap_quality_rows),
            "primary_risk_rows": len(primary_risk_rows),
            "overlap_risk_rows": len(overlap_risk_rows),
            "missing_or_extra_ids": 0,
            "uncertain": 0,
            "overlap_script_duplicate_json_keys": {
                "quality": quality_duplicate_keys,
                "risk": risk_duplicate_keys,
                "impact": (
                    "last-value JSON/JavaScript semantics affected one quality criterion field; "
                    "the quality preference and item identity were unchanged"
                    if quality_duplicate_keys
                    else "none"
                ),
            },
        },
        "quality_reliability": {
            **cohen_kappa(quality_left, quality_right, ("A", "B", "tie")),
            "disagreements": len(quality_disagreement_ids),
            "direct_a_b_reversals": len(quality_direct_reversals),
            "primary_distribution_on_overlap": dict(Counter(quality_left.values())),
            "independent_distribution": dict(Counter(quality_right.values())),
        },
        "risk_reliability": {
            **cohen_kappa(risk_left, risk_right, ("yes", "no")),
            "yes_no_disagreements": len(risk_decision_disagreement_ids),
            "category_disagreements_when_both_yes": len(
                risk_category_disagreement_ids
            ),
            "adjudication_union": len(risk_disagreement_ids),
            "primary_distribution_on_overlap": dict(Counter(risk_left.values())),
            "independent_distribution": dict(Counter(risk_right.values())),
        },
        "adjudication": {
            "quality_items": len(quality_packet),
            "risk_items": len(risk_packet),
            "prior_labels_visible_to_adjudicator": False,
            "only_one_adjudication_allowed": True,
        },
        "blinding_limitation": {
            "independent_reviewer_was_design_aware": True,
            "resource_arm_often_inferable_from_source_attribution_text": True,
            "learned_policy_selection_remained_private": True,
            "disposition": (
                "retain as a different-reviewer overlap sensitivity analysis, disclose as sighted/design-aware, "
                "and do not rewrite or regenerate frozen responses"
            ),
        },
        "template_findings_disposition": {
            "volunteer_role_or_topic_reversal": (
                "real frozen response-level quality/risk outcome; retain, do not delete as invalid data"
            ),
            "low_information_prompt_families": (
                "retain ties and report family diagnostics; do not remove after seeing outcomes"
            ),
        },
        "input_sha256": {
            "quality_primary": sha256_file(args.quality_primary),
            "quality_overlap_script": sha256_file(args.quality_overlap_script),
            "risk_primary": sha256_file(args.risk_primary),
            "risk_overlap_script": sha256_file(args.risk_overlap_script),
            "frozen_review_manifest": sha256_file(args.review_dir / "review_manifest.json"),
        },
    }
    write_json(args.out_dir / "adjudication_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
