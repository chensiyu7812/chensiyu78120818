#!/usr/bin/env python3
"""Prepare the one blinded V5.2 overlap-disagreement adjudication."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.2-fit-disagreement-adjudication-v1"
QUALITY_PROTOCOL = "pm-v1.5-v5.2-fit-quality-adjudication-v1"
RISK_PROTOCOL = "pm-v1.5-v5.2-fit-risk-adjudication-v1"
FUNCTION_PROTOCOL = "pm-v1.5-v5.2-fit-function-adjudication-v1"
PANEL = ROOT / "outputs/pm_v1_5_v5_2_full_fit_outcome_review_v1_candidate"
IMPORT = PANEL / "imported_annotations"
OUT = ROOT / "outputs/pm_v1_5_v5_2_fit_disagreement_adjudication_v1_candidate"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def index(path: Path, field: str) -> dict[str, dict[str, Any]]:
    data = rows(path)
    result = {str(row[field]): row for row in data}
    if len(result) != len(data):
        raise RuntimeError(f"duplicate {field}: {path}")
    return result


def module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main() -> None:
    report = read_json(IMPORT / "data_quality_and_agreement_report.json")
    quality_ids = list(report["agreement"]["quality"]["disagreement_ids"])
    risk_ids = list(report["agreement"]["risk"]["disagreement_ids"])
    function_ids = list(report["agreement"]["function"]["disagreement_ids"])
    if (len(quality_ids), len(risk_ids), len(function_ids)) != (35, 30, 13):
        raise RuntimeError("frozen disagreement shape changed")
    quality_all = index(PANEL / "primary_quality_packet.jsonl", "blind_item_id")
    risk_all = index(PANEL / "primary_risk_packet.jsonl", "risk_item_id")
    function_all = index(PANEL / "primary_function_packet.jsonl", "function_item_id")
    quality = [quality_all[item] for item in quality_ids]
    risk = [risk_all[item] for item in risk_ids]
    function = [function_all[item] for item in function_ids]

    quality_helper = module(ROOT / "scripts/v1_5/25zy_prepare_v5_full_fit_outcome_review_v1_5.py", "v52_adj_quality")
    risk_helper = module(ROOT / "scripts/v1_5/26i_prepare_v5_1_confirmation_review_v1_5.py", "v52_adj_risk")
    function_helper = module(ROOT / "scripts/v1_5/27c_prepare_v5_2_full_fit_outcome_panel_v1_5.py", "v52_adj_function")
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "quality_disagreements.jsonl", quality)
    write_jsonl(OUT / "risk_disagreements.jsonl", risk)
    write_jsonl(OUT / "function_disagreements.jsonl", function)

    def html(name: str, template: str, protocol: str, items: list[dict[str, Any]]) -> None:
        payload = {
            "manifest": {
                "protocol": PROTOCOL,
                "panel_role": "third_reviewer_final_adjudication",
                "instruction": "Judge the displayed construct from evidence; prior reviewer labels and component/arm identity are hidden.",
            },
            "items": items,
        }
        (OUT / name).write_text(
            template.replace("__DATA__", quality_helper._safe_script_json(payload)).replace("__PROTOCOL__", protocol),
            encoding="utf-8",
        )

    html("human_quality_adjudication.html", quality_helper.QUALITY_HTML, QUALITY_PROTOCOL, quality)
    html("human_risk_adjudication.html", risk_helper.RISK_HTML, RISK_PROTOCOL, risk)
    html("human_function_adjudication.html", function_helper.FUNCTION_HTML, FUNCTION_PROTOCOL, function)
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_ONE_BLINDED_THIRD_REVIEWER_ADJUDICATION",
        "items": {"quality": 35, "risk": 30, "function": 13, "total": 78},
        "frozen_rules": {
            "quality": "Only adoption-changing differences are A/B; slight preference is tie.",
            "risk": "Authorized evidence does not immunize user-facing record labels; yes still requires material adoption impact.",
            "function": "A backend-locked clause may itself be the resource contribution; free-form generator causation is not required, but irrelevant or redundant attachment is no.",
        },
        "post_adjudication_executor_change_allowed": False,
        "input_sha256": {
            "agreement_report": sha256_file(IMPORT / "data_quality_and_agreement_report.json"),
            "quality_packet": sha256_file(PANEL / "primary_quality_packet.jsonl"),
            "risk_packet": sha256_file(PANEL / "primary_risk_packet.jsonl"),
            "function_packet": sha256_file(PANEL / "primary_function_packet.jsonl"),
        },
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
