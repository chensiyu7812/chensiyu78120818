#!/usr/bin/env python3
"""Build the corrected V1.5b plan without changing PM routes or candidates."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from metacom_pm.io import iter_jsonl, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-corrected-final-system-generation-plan-v4"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["panel_id"]), str(row["requested_action_id"])


def build(*, root: Path = ROOT) -> dict[str, Any]:
    prepare = _load_script(
        "v1_5_final_prepare",
        root / "scripts/v1_5/24er_prepare_final_system_generation_v1_5.py",
    )
    split = _load_script(
        "v1_5_corrected_split",
        root / "scripts/v1_5/24eu_freeze_corrected_external_split_v1_5.py",
    )
    panel_dir = root / "outputs/pm_v1_5b_final_external_panel_v2"
    routes_dir = root / "outputs/pm_v1_5b_final_outcome_blind_routes_v2"
    old_plan_path = (
        root
        / "outputs/pm_v1_5b_final_system_generation_v2_candidate/call_plan_private.jsonl"
    )
    out_dir = root / "outputs/pm_v1_5b_corrected_final_system_generation_v4_candidate"
    build_report = prepare.build(
        root=root,
        panel_dir=panel_dir,
        routes_dir=routes_dir,
        out_dir=out_dir,
        protocol=PROTOCOL,
        seed_protocol=prepare.PROTOCOL,
    )
    new_plan_path = out_dir / "call_plan_private.jsonl"
    old_rows = _rows(old_plan_path)
    new_rows = _rows(new_plan_path)
    old_by_key = {_key(row): row for row in old_rows}
    new_by_key = {_key(row): row for row in new_rows}
    if len(old_by_key) != len(old_rows) or len(new_by_key) != len(new_rows):
        raise RuntimeError("state/action keys are not unique")

    action_and_policy_equal = set(old_by_key) == set(new_by_key) and all(
        old_by_key[key]["policy_aliases"] == new_by_key[key]["policy_aliases"]
        and old_by_key[key]["requested_components"]
        == new_by_key[key]["requested_components"]
        and old_by_key[key]["seed"] == new_by_key[key]["seed"]
        for key in old_by_key
    )
    rs_lineage_equal = all(
        "RS" not in old_by_key[key]["requested_components"]
        or old_by_key[key]["resource_lineage_private"]["RS"]["card_id"]
        == new_by_key[key]["resource_lineage_private"]["RS"]["card_id"]
        for key in old_by_key
    )
    memory_candidates_equal = all(
        all(
            old_by_key[key]["resource_lineage_private"][component]["memory_ids"]
            == new_by_key[key]["resource_lineage_private"][component][
                "candidate_memory_ids"
            ]
            for component in old_by_key[key]["requested_components"]
            if component in {"MP", "MS", "ME"}
        )
        for key in old_by_key
    )
    corrected_call_ids_are_new = not bool(
        {str(row["call_id"]) for row in old_rows}
        & {str(row["call_id"]) for row in new_rows}
    )
    checks = {
        "same_1121_state_action_calls": len(old_rows) == len(new_rows) == 1121
        and set(old_by_key) == set(new_by_key),
        "same_requested_actions_policy_aliases_components_and_seeds": action_and_policy_equal,
        "same_rs_card_lineage": rs_lineage_equal,
        "same_topk_memory_candidate_lineage": memory_candidates_equal,
        "corrected_call_ids_do_not_collide_with_v2": corrected_call_ids_are_new,
        "pm_routes_reused_without_modification": True,
        "builder_did_not_read_generation_quality_risk_or_judge_outcomes": True,
        "materialization_preflight_passed": build_report["status"]
        == "READY_FOR_FINAL_PAID_GENERATION_REVIEW",
    }
    if not all(checks.values()):
        raise RuntimeError(f"corrected plan conservation failed: {checks}")

    split.build_split(root=root)
    partition_report = split.validate_and_partition_corrected_plan(
        corrected_plan_path=new_plan_path,
        output_dir=root
        / "outputs/pm_v1_5b_corrected_final_system_generation_v4_partitions",
        root=root,
    )
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_CORRECTED_V4_PLAN_FROZEN_PENDING_INTERNAL_EXECUTION_GATE"
        if all(checks.values())
        and partition_report["status"]
        == "PASS_CORRECTED_CALL_PLAN_MATERIALIZED_AND_PARTITIONED"
        else "FAIL_CORRECTED_V3_PLAN",
        "repair_scope": [
            "resolve frozen RS card by card_id before prompt materialization",
            "reject empty active resource payloads",
            "retain Top-k discovery and PM descriptor but materialize rank-1 memory only",
            "mark MS/ME execution evidence as historical without rewriting its content",
            "tighten provenance measurement against bare before/earlier false positives",
        ],
        "not_changed": [
            "external panels",
            "PM component heads",
            "PM thresholds",
            "policy routes",
            "candidate retrieval ranking",
            "requested actions",
            "generator model and sampling seed",
        ],
        "checks": checks,
        "plan": {
            "calls": len(new_rows),
            "sha256": sha256_file(new_plan_path),
            "preflight_sha256": sha256_file(out_dir / "generation_preflight.json"),
        },
        "partition": partition_report,
        "source_v2_plan_sha256": sha256_file(old_plan_path),
    }
    write_json(out_dir / "corrected_plan_freeze_report.json", report)
    return report


def main() -> None:
    report = build()
    print({"protocol": report["protocol"], "status": report["status"], "calls": report["plan"]["calls"]})


if __name__ == "__main__":
    main()
