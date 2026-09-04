#!/usr/bin/env python3
"""Prepare only the changed V2 Observation surfaces for bounded re-review."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-semantic-repair-delta-v2"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_review_candidate_v1",
    )
    parser.add_argument(
        "--v2-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_review_candidate_v2",
    )
    parser.add_argument(
        "--working-reference",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_review_audit_v1/primary_working_reference.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_semantic_repair_delta_v2_candidate",
    )
    args = parser.parse_args()

    v1 = json.loads((args.v1_dir / "human_review_packet.json").read_text(encoding="utf-8"))
    v2 = json.loads((args.v2_dir / "human_review_packet.json").read_text(encoding="utf-8"))
    old = {str(row["blind_item_id"]): row for row in v1["items"]}
    new = {str(row["blind_item_id"]): row for row in v2["items"]}
    if set(old) != set(new) or len(old) != 96:
        raise RuntimeError("V1/V2 packet membership changed; bounded carry-forward is unsafe")
    public_fields = (
        "component",
        "owner_gate_applicable",
        "visible_dialogue",
        "current_user_text",
        "candidate_text",
        "candidate_age_sessions",
    )
    changed_ids = sorted(
        item_id
        for item_id in old
        if any(old[item_id][field] != new[item_id][field] for field in public_fields)
    )
    unchanged_ids = sorted(set(old) - set(changed_ids))
    if len(changed_ids) != 16:
        raise RuntimeError(f"expected exactly 16 semantic repair surfaces, got {len(changed_ids)}")

    reference = {str(row["blind_item_id"]): row for row in _rows(args.working_reference)}
    if set(reference) != set(old):
        raise RuntimeError("working reference does not cover the frozen V1 packet")
    v2_binding = {
        str(row["blind_item_id"]): row
        for row in _rows(args.v2_dir / "private_binding.jsonl")
    }
    items = [new[item_id] for item_id in changed_ids]
    items.sort(key=lambda row: (str(row["component"]), str(row["blind_item_id"])))
    packet = {
        "protocol": PROTOCOL,
        "export_filename": "observation_semantic_repair_delta_v2.jsonl",
        "items": items,
        "decision_rule": "all applicable observable gates yes; RS owner/time is structural yes",
        "repair_scope": [
            "MP_PREFERENCE owner-negative now uses an explicitly superseded preference",
            "RS positive goal now satisfies the actual card-level prerequisite, not only a coarse family",
        ],
        "is_step1_worth_opening_gold": False,
        "response_or_outcome_visible": False,
    }

    render_namespace = runpy.run_path(
        str(ROOT / "scripts/v1_5/25zg_prepare_v3_observation_review_v1_5.py")
    )
    html = render_namespace["_render"](packet, "pm15_v3_observation_semantic_delta_v2")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "human_review_packet.json", packet)
    (args.out_dir / "human_review.html").write_text(html, encoding="utf-8")
    write_jsonl(
        args.out_dir / "private_binding.jsonl",
        [v2_binding[item_id] for item_id in changed_ids],
    )
    write_jsonl(
        args.out_dir / "carry_forward_reference.jsonl",
        [reference[item_id] for item_id in unchanged_ids],
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_BOUNDED_PRIMARY_DELTA_REVIEW",
        "changed_items": len(changed_ids),
        "changed_per_component": dict(Counter(new[item_id]["component"] for item_id in changed_ids)),
        "carry_forward_items": len(unchanged_ids),
        "full_re_review_required": False,
        "independent_overlap_repeated": False,
        "independent_overlap_reason": "The frozen V1 overlap already passed all factor kappas >= .80 and final eligibility agreement was 24/24; this delta repairs explicit audited semantics rather than introducing a new construct.",
        "v1_packet_sha256": sha256_file(args.v1_dir / "human_review_packet.json"),
        "v2_packet_sha256": sha256_file(args.v2_dir / "human_review_packet.json"),
        "v2_binding_sha256": sha256_file(args.v2_dir / "private_binding.jsonl"),
        "working_reference_sha256": sha256_file(args.working_reference),
        "delta_packet_sha256": sha256_file(args.out_dir / "human_review_packet.json"),
        "carry_forward_reference_sha256": sha256_file(args.out_dir / "carry_forward_reference.jsonl"),
        "api_calls": 0,
        "responses_generated": 0,
        "external_lockbox_read": False,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
