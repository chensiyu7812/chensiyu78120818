#!/usr/bin/env python3
"""Prepare the versioned V3 H-Eligibility repair delta for dual review."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-h-eligibility-repair-delta-primary-v2"
INDEPENDENT_PROTOCOL = "pm-v1.5-v3-h-eligibility-repair-delta-independent-v2"
EXPECTED_CHANGED_FAMILIES = {
    "MS_ELIGIBLE_MS_DISTINCTION": 4,
    "MS_ELIGIBLE_MS_RECALL": 4,
    "MS_WRONG_OWNER_OR_GOAL": 4,
    "RS_CURRENT_REQUEST_ALREADY_SPECIFIES_MOVE": 4,
    "RS_ELIGIBLE_RS_SUGGESTION": 4,
    "RS_WRONG_FAMILY_BURDEN_OR_HIGH_STAKES": 4,
}


def _review_renderer():
    source = ROOT / "scripts/v1_5/25x_prepare_v3_eligibility_review_v1_5.py"
    spec = importlib.util.spec_from_file_location("v3_eligibility_packet_v1", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen H-Eligibility renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._render


def _decision_surface(row: Mapping[str, Any]) -> dict[str, Any]:
    surface = row["exact_rank1_candidate"]
    return {
        "visible_dialogue": row["visible_dialogue"],
        "current_user_text": row["current_user_text"],
        "component": row["target_component_private_not_model_input"],
        "candidate_text": surface["candidate_text"],
        "candidate_age_sessions": surface["candidate_age_sessions"],
    }


def _surface_hash(row: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(_decision_surface(row)))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 P2 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v1",
    )
    parser.add_argument(
        "--new-candidate-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v2",
    )
    parser.add_argument(
        "--old-binding",
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
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_repair_delta_v2_candidate",
    )
    args = parser.parse_args()

    old_path = args.old_candidate_dir / "candidate_rows_private.jsonl"
    new_path = args.new_candidate_dir / "candidate_rows_private.jsonl"
    old_report = read_json(args.old_candidate_dir / "materialization_report.json")
    new_report = read_json(args.new_candidate_dir / "materialization_report.json")
    if any(
        report.get("status") != "PASS"
        or report.get("candidate_rows_sha256") != sha256_file(path)
        for report, path in ((old_report, old_path), (new_report, new_path))
    ):
        raise RuntimeError("both candidate versions must be frozen passing materializations")
    old = {str(row["state_id"]): row for row in _rows(old_path)}
    new = {str(row["state_id"]): row for row in _rows(new_path)}
    blueprint = {
        str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)
    }
    old_binding_by_state = {
        str(row["state_id"]): row for row in _rows(args.old_binding)
    }
    if set(old) != set(new) or set(old) != set(blueprint) or len(old) != 640:
        raise RuntimeError("candidate versions and blueprint are not aligned")

    changed = [
        state_id
        for state_id in old
        if blueprint[state_id]["track"] == "ELIGIBILITY_AUDIT"
        and _surface_hash(old[state_id]) != _surface_hash(new[state_id])
    ]
    family_counts = Counter(str(blueprint[state_id]["logic_family"]) for state_id in changed)
    if len(changed) != 24 or dict(family_counts) != EXPECTED_CHANGED_FAMILIES:
        raise RuntimeError(
            f"repair delta changed an unexpected decision surface: {dict(family_counts)}"
        )

    items: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for state_id in changed:
        row = new[state_id]
        component = str(row["target_component_private_not_model_input"])
        surface = row["exact_rank1_candidate"]
        blind_id = "h_elig_v2_" + stable_hex(PROTOCOL, state_id, component, n=24)
        items.append(
            {
                "blind_item_id": blind_id,
                "component": component,
                "visible_dialogue": row["visible_dialogue"],
                "current_user_text": row["current_user_text"],
                "candidate_text": surface["candidate_text"],
                "candidate_age_sessions": surface["candidate_age_sessions"],
            }
        )
        bindings.append(
            {
                "blind_item_id": blind_id,
                "state_id": state_id,
                "component": component,
                "logic_family_private_not_reviewer_input": blueprint[state_id]["logic_family"],
                "private_coverage_intent_not_gold": blueprint[state_id]["private_eligibility_intent"],
                "old_blind_item_id": old_binding_by_state[state_id]["blind_item_id"],
                "old_decision_surface_sha256": _surface_hash(old[state_id]),
                "new_decision_surface_sha256": _surface_hash(row),
                "candidate_id": surface["candidate_id"],
                "candidate_text_sha256": surface["candidate_text_sha256"],
            }
        )
    items.sort(key=lambda row: stable_hex(PROTOCOL, "order", row["blind_item_id"], n=24))
    independent_items = sorted(
        items,
        key=lambda row: stable_hex(INDEPENDENT_PROTOCOL, "order", row["blind_item_id"], n=24),
    )
    common = {
        "decision_rule": "eligible iff all four observable gates are yes",
        "repair_scope_visible": False,
        "construction_intent_visible": False,
        "retrieval_score_visible": False,
        "response_or_outcome_visible": False,
    }
    primary = {
        "protocol": PROTOCOL,
        "export_filename": "h_eligibility_repair_delta_primary.jsonl",
        "items": items,
        **common,
    }
    independent = {
        "protocol": INDEPENDENT_PROTOCOL,
        "export_filename": "h_eligibility_repair_delta_independent.jsonl",
        "items": independent_items,
        **common,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "primary_packet.json", primary)
    write_json(args.out_dir / "independent_packet.json", independent)
    write_jsonl(args.out_dir / "private_binding.jsonl", bindings)
    render = _review_renderer()
    (args.out_dir / "human_review.html").write_text(
        render(primary, "pm15_v3_h_eligibility_repair_delta_primary_v2"),
        encoding="utf-8",
    )
    (args.out_dir / "independent_review.html").write_text(
        render(independent, "pm15_v3_h_eligibility_repair_delta_independent_v2"),
        encoding="utf-8",
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_DUAL_REVIEW_OF_CHANGED_DECISION_SURFACES_ONLY",
        "changed_items": len(items),
        "per_component": dict(Counter(row["component"] for row in items)),
        "changed_logic_families_private_not_reviewer_input": dict(family_counts),
        "unchanged_eligibility_rows_eligible_for_exact_surface_carry_forward": 104,
        "primary_and_independent_review_all_changed_items": True,
        "old_candidate_rows_sha256": sha256_file(old_path),
        "new_candidate_rows_sha256": sha256_file(new_path),
        "blueprint_sha256": sha256_file(args.blueprint),
        "binding_sha256": sha256_file(args.out_dir / "private_binding.jsonl"),
        "construction_intent_is_gold": False,
        "old_labels_auto_copied_for_changed_rows": False,
        "responses_generated": 0,
        "external_lockbox_read": False,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(manifest)


if __name__ == "__main__":
    main()
