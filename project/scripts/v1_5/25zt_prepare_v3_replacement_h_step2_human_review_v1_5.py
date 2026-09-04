#!/usr/bin/env python3
"""Prepare the only 32-item replacement V3 H-Step2 human gate."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v3-replacement-h-step2-human-review-v1"
PLAN_DIR = ROOT / "outputs/pm_v1_5_v3_replacement_h_step2_plan_v1_candidate"
EXEC_DIR = ROOT / "outputs/pm_v1_5_v3_replacement_h_step2_execution_v1"
OUT_DIR = ROOT / "outputs/pm_v1_5_v3_replacement_h_step2_human_review_v1_candidate"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _visible_context(call: dict[str, Any]) -> str:
    content = str(call["messages"][1]["content"])
    marker = "\n\nEXECUTION PLAN"
    if marker not in content:
        raise RuntimeError(f"{call['call_id']}: visible-context marker missing")
    return content.split(marker, 1)[0].strip()


def _render(items: list[dict[str, Any]]) -> str:
    source = ROOT / "scripts/v1_5/24ex_prepare_corrected_step2_human_gate_v1_5.py"
    spec = importlib.util.spec_from_file_location("step2_review_renderer", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared Step2 review renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL = PROTOCOL
    page = module._render(items)
    return (
        page.replace(
            "修复后 Step2 最小功能与误用核验（16条）",
            "V3 replacement H-Step2 最终功能与误用资格门（32条）",
        )
        .replace("对每个被 Step1 开启的组件分别判断", "对每个预先授权且已确认eligible的组件分别判断")
        .replace("pm_v15b_corrected_step2_human_gate_v1", "pm_v15_v3_replacement_h_step2_review_v1")
        .replace("corrected_step2_human_gate.jsonl", "v3_replacement_h_step2_review_v1.jsonl")
    )


def main() -> None:
    calls = {
        str(row["call_id"]): row
        for row in _rows(PLAN_DIR / "call_plan_private.jsonl")
    }
    outcomes = _rows(EXEC_DIR / "generation_outcomes_ordered_private.jsonl")
    if len(calls) != 32 or len(outcomes) != 32 or set(calls) != {
        str(row["call_id"]) for row in outcomes
    }:
        raise RuntimeError("replacement H-Step2 requires exact 32 aligned calls/outcomes")
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for outcome in outcomes:
        call = calls[str(outcome["call_id"])]
        review_id = "v3h2r_review_" + stable_hex(PROTOCOL, outcome["call_id"], n=24)
        application = outcome.get("bundle_application")
        schema_failure = outcome.get("bundle_schema_failure_private")
        if application:
            application_response = str(application["response"])
        elif schema_failure and isinstance(schema_failure.get("invalid_payload_private"), dict):
            application_response = str(schema_failure["invalid_payload_private"].get("response") or "")
        else:
            raise RuntimeError(f"{outcome['call_id']}: primary response unavailable")
        if not application_response.strip():
            raise RuntimeError(f"{outcome['call_id']}: empty primary response")
        public.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "requested_action_id": outcome["requested_action_id"],
                "requested_components": call["requested_components"],
                "visible_context": _visible_context(call),
                "resources": call["selected_resources_private"],
                "plans": call["execution_plans"],
                "application_response": application_response,
                "fallback_response": None if outcome["fallback"] is None else outcome["fallback"]["response"],
            }
        )
        private.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "call_id": outcome["call_id"],
                "state_id": outcome["state_id"],
                "call_kind": "single" if len(call["requested_components"]) == 1 else "multi",
                "machine_checks_private": outcome["bundle_machine_checks"],
                "fallback_executed_private": outcome["fallback_executed"],
                "schema_failure_private": bool(schema_failure),
                "selection_rule": "all_24_eligible_singles_and_all_8_joint_feasible_multi_frozen_before_generation",
                "selection_uses_quality_risk_or_external_outcome": False,
            }
        )
    order = sorted(
        range(32),
        key=lambda index: stable_hex(PROTOCOL, "display", public[index]["review_item_id"], n=32),
    )
    public = [public[index] for index in order]
    private_by_id = {row["review_item_id"]: row for row in private}
    private = [private_by_id[row["review_item_id"]] for row in public]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    packet_path = OUT_DIR / "human_review_packet.jsonl"
    key_path = OUT_DIR / "private_review_key.jsonl"
    write_jsonl(packet_path, public)
    write_jsonl(key_path, private)
    (OUT_DIR / "human_review.html").write_text(_render(public), encoding="utf-8")
    report = {
        "protocol": PROTOCOL,
        "status": "READY_ONE_SHOT_32_ITEM_REPLACEMENT_H_STEP2_REVIEW",
        "items": 32,
        "single_items": sum(len(row["requested_components"]) == 1 for row in public),
        "multi_items": sum(len(row["requested_components"]) > 1 for row in public),
        "requested_action_counts": dict(sorted(Counter(row["requested_action_id"] for row in public).items())),
        "fallback_items": sum(row["fallback_response"] is not None for row in public),
        "pm_step1_routing_evaluated": False,
        "retrieval_accuracy_evaluated": False,
        "general_pairwise_quality_evaluated": False,
        "step2_function_and_material_misuse_only": True,
        "selection_uses_quality_risk_or_external_outcome": False,
        "inputs": {
            "call_plan_sha256": sha256_file(PLAN_DIR / "call_plan_private.jsonl"),
            "outcomes_sha256": sha256_file(EXEC_DIR / "generation_outcomes_ordered_private.jsonl"),
        },
        "packet_sha256": sha256_text(canonical_json(public)),
    }
    write_json(OUT_DIR / "preparation_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
