#!/usr/bin/env python3
"""Import the six V5.2 review exports and audit panel agreement."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5.2-fit-review-import-and-agreement-v1"
DEFAULT_PANEL = ROOT / "outputs/pm_v1_5_v5_2_full_fit_outcome_review_v1_candidate"
ATTACHMENTS = {
    "primary_quality": Path("/home/tokkio/.codex/attachments/f06464af-78c8-4ac7-8802-e41b70730608/pasted-text.txt"),
    "overlap_quality": Path("/home/tokkio/.codex/attachments/8754a4e5-ee39-4b7d-857f-0da5c5c7ee91/pasted-text.txt"),
    "primary_risk": Path("/home/tokkio/.codex/attachments/8dc2aabb-b6e6-4f18-8dc9-7f2534bb7bd0/pasted-text.txt"),
    "overlap_risk_script": Path("/home/tokkio/.codex/attachments/f6fa63b8-7005-46e6-a06a-3e2c318941b6/pasted-text.txt"),
    "primary_function": Path("/home/tokkio/.codex/attachments/436ba2ee-5217-485f-ba48-6c3e79c31dd7/pasted-text.txt"),
    "overlap_function_script": Path("/home/tokkio/.codex/attachments/ba0285a5-9dd3-4b44-98ec-98efc77445f9/pasted-text.txt"),
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def json_lines(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            result.append(dict(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL {path}:{number}: {exc}") from exc
    return result


def js_ann(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    start_marker = "const ANN = "
    end_marker = "\n\n(function()"
    if start_marker not in text or end_marker not in text:
        raise RuntimeError(f"cannot find ANN object in {path}")
    raw = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    if not raw.endswith(";"):
        raise RuntimeError(f"ANN object lacks terminator in {path}")
    return dict(json.loads(raw[:-1]))


def id_map(data: list[dict[str, Any]], field: str, *, name: str) -> dict[str, dict[str, Any]]:
    result = {str(row[field]): row for row in data}
    if len(result) != len(data):
        raise RuntimeError(f"duplicate IDs in {name}")
    return result


def kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("kappa label shape mismatch")
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a, counts_b = Counter(labels_a), Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = sum(
        counts_a[label] / len(labels_a) * counts_b[label] / len(labels_b)
        for label in labels
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def agreement(
    primary: dict[str, dict[str, Any]],
    overlap: dict[str, dict[str, Any]],
    *,
    label_field: str,
) -> dict[str, Any]:
    if not set(overlap) <= set(primary):
        raise RuntimeError("overlap IDs are not a subset of primary IDs")
    ids = sorted(overlap)
    a = [str(primary[item][label_field]) for item in ids]
    b = [str(overlap[item][label_field]) for item in ids]
    disagreements = [item for item in ids if primary[item][label_field] != overlap[item][label_field]]
    return {
        "n": len(ids),
        "exact_agreement": round((len(ids) - len(disagreements)) / len(ids), 6),
        "cohen_kappa": None if (value := kappa(a, b)) is None else round(value, 6),
        "primary_distribution": dict(Counter(a)),
        "independent_distribution": dict(Counter(b)),
        "disagreement_count": len(disagreements),
        "disagreement_ids": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL)
    args = parser.parse_args()
    for name, path in ATTACHMENTS.items():
        if not path.is_file():
            raise RuntimeError(f"attachment missing: {name}: {path}")

    expected = {
        "primary_quality": id_map(rows(args.panel_dir / "primary_quality_packet.jsonl"), "blind_item_id", name="primary quality packet"),
        "overlap_quality": id_map(rows(args.panel_dir / "overlap_quality_packet.jsonl"), "blind_item_id", name="overlap quality packet"),
        "primary_risk": id_map(rows(args.panel_dir / "primary_risk_packet.jsonl"), "risk_item_id", name="primary risk packet"),
        "overlap_risk": id_map(rows(args.panel_dir / "overlap_risk_packet.jsonl"), "risk_item_id", name="overlap risk packet"),
        "primary_function": id_map(rows(args.panel_dir / "primary_function_packet.jsonl"), "function_item_id", name="primary function packet"),
        "overlap_function": id_map(rows(args.panel_dir / "overlap_function_packet.jsonl"), "function_item_id", name="overlap function packet"),
    }
    imported: dict[str, list[dict[str, Any]]] = {
        "primary_quality": json_lines(ATTACHMENTS["primary_quality"]),
        "overlap_quality": json_lines(ATTACHMENTS["overlap_quality"]),
        "primary_risk": json_lines(ATTACHMENTS["primary_risk"]),
        "primary_function": json_lines(ATTACHMENTS["primary_function"]),
    }

    risk_ann = js_ann(ATTACHMENTS["overlap_risk_script"])
    yes = dict(risk_ann.get("yes", {}))
    notes = dict(risk_ann.get("notes", {}))
    overlap_risk: list[dict[str, Any]] = []
    for item_id in expected["overlap_risk"]:
        categories = dict(yes.get(item_id, {}))
        overlap_risk.append({
            "protocol": "pm-v1.5-v5.2-fit-grounding-risk-v1",
            "risk_item_id": item_id,
            "any_material_risk": "yes" if categories else "no",
            "selected_categories": list(categories),
            "evidence_by_category": categories,
            "risk_notes": str(notes.get(item_id, "")),
            "annotator_id": "claude_independent_overlap",
        })
    imported["overlap_risk"] = overlap_risk

    function_ann = js_ann(ATTACHMENTS["overlap_function_script"])
    imported["overlap_function"] = [
        {
            "protocol": "pm-v1.5-v5.2-fit-resource-function-v1",
            "function_item_id": item_id,
            **dict(function_ann[item_id]),
            "annotator_id": "claude_independent_overlap",
        }
        for item_id in expected["overlap_function"]
        if item_id in function_ann
    ]

    id_fields = {
        "primary_quality": "blind_item_id", "overlap_quality": "blind_item_id",
        "primary_risk": "risk_item_id", "overlap_risk": "risk_item_id",
        "primary_function": "function_item_id", "overlap_function": "function_item_id",
    }
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for name, data in imported.items():
        maps[name] = id_map(data, id_fields[name], name=name)
        expected_ids = set(expected[name])
        observed_ids = set(maps[name])
        if expected_ids != observed_ids:
            raise RuntimeError(
                f"{name} ID coverage mismatch: missing={sorted(expected_ids-observed_ids)}, extra={sorted(observed_ids-expected_ids)}"
            )

    quality_allowed = {"A", "B", "tie", "uncertain"}
    risk_allowed = {"yes", "no", "uncertain"}
    function_allowed = {"yes", "no", "uncertain"}
    for name in ("primary_quality", "overlap_quality"):
        if any(str(row.get("quality_preference")) not in quality_allowed for row in imported[name]):
            raise RuntimeError(f"invalid quality label in {name}")
    for name in ("primary_risk", "overlap_risk"):
        if any(str(row.get("any_material_risk")) not in risk_allowed for row in imported[name]):
            raise RuntimeError(f"invalid risk label in {name}")
    for name in ("primary_function", "overlap_function"):
        if any(str(row.get("resource_functionally_contributed")) not in function_allowed for row in imported[name]):
            raise RuntimeError(f"invalid function label in {name}")

    out_dir = args.panel_dir / "imported_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in imported.items():
        write_jsonl(out_dir / f"{name}.jsonl", data)

    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_WITH_DISAGREEMENTS_REQUIRING_ONE_CENTRAL_ADJUDICATION",
        "coverage": {name: len(data) for name, data in imported.items()},
        "summary_count_corrections": {
            "overlap_quality": "94 records parsed; wc showed 93 because the final JSON line lacked a newline",
            "overlap_risk": "167 records reconstructed; the script defaults every packet ID to no and overrides 10 yes, so the prose claim of 164 was a summary error",
        },
        "agreement": {
            "quality": agreement(maps["primary_quality"], maps["overlap_quality"], label_field="quality_preference"),
            "risk": agreement(maps["primary_risk"], maps["overlap_risk"], label_field="any_material_risk"),
            "function": agreement(maps["primary_function"], maps["overlap_function"], label_field="resource_functionally_contributed"),
        },
    }
    write_json(out_dir / "data_quality_and_agreement_report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
