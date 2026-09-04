#!/usr/bin/env python3
"""Run 32 structured Step2 resource executions with machine-checkable traces."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import RuntimeState
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    utc_now,
    write_json,
    write_jsonl,
)
from metacom_pm.prompts import common_context
from metacom_pm.v1_5_generator_alignment_audit import (
    COMPONENT_FUNCTION,
    ResourceExecutionOutput,
    resource_execution_messages,
    validate_resource_execution,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-auditable-resource-execution-v2"
PLAN_PROTOCOL = "pm-v1.5-generator-alignment-qualification-plan-v1"
EXPECTED_CALLS = 32


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _resource_surface(row: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in row["selected_memory_items"]:
        lines.append(f"[{item['source']}] {item['text']}")
    for card in row["selected_strategy_cards"]:
        for key in ("support_move", "when_to_use", "when_not_to_use"):
            if card.get(key):
                lines.append(f"{key}: {card[key]}")
    return "\n".join(lines)


def _expected_function(component: str, subtype: str) -> str:
    if subtype == "MP_PREFERENCE":
        return "style_or_pacing"
    if subtype == "MP_PROFILE":
        return "profile_context"
    return COMPONENT_FUNCTION[component]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir", type=Path,
        default=ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "outputs/pm_v1_5_auditable_resource_execution_v2",
    )
    parser.add_argument(
        "--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml"
    )
    parser.add_argument(
        "--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml"
    )
    parser.add_argument("--max-new-calls", type=int, default=EXPECTED_CALLS)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    plan_report = read_json(args.plan_dir / "generation_plan_report.json")
    resources = _rows(args.plan_dir / "private_selected_resources.jsonl")
    base_calls = _rows(args.plan_dir / "call_plan.jsonl")
    source_matched = {
        str(row["state_id"]): row
        for row in base_calls
        if row["prompt_variant"] == "source_matched"
    }
    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)

    plan: list[dict[str, Any]] = []
    for selected in resources:
        state = RuntimeState.model_validate(selected["runtime_state"])
        component = str(selected["component"])
        subtype = next(
            str(row["resource_subtype"])
            for row in base_calls
            if row["state_id"] == state.state_id
        )
        resource_label = "R1" if component == "RS" else "E1"
        resource = _resource_surface(selected)
        expected = _expected_function(component, subtype)
        messages = resource_execution_messages(
            component=component,
            resource_subtype=subtype,
            visible_context=common_context(state),
            selected_resource=resource,
            resource_label=resource_label,
            system_prompt=generation.system_prompt,
        )
        plan.append(
            {
                "protocol": PROTOCOL,
                "call_id": "resource_exec_" + stable_hex(
                    PROTOCOL, state.state_id, n=24
                ),
                "state_id": state.state_id,
                "component": component,
                "resource_subtype": subtype,
                "resource_label": resource_label,
                "selected_resource": resource,
                "expected_resource_function": expected,
                "messages": messages,
                "messages_sha256": sha256_text(canonical_json(messages)),
                "seed": int(source_matched[state.state_id]["generation"]["seed"])
                % (2**31 - 1),
                "selection_uses_response_or_judge_outcome": False,
            }
        )
    plan_ok = (
        plan_report.get("protocol") == PLAN_PROTOCOL
        and len(plan) == EXPECTED_CALLS
        and len({row["state_id"] for row in plan}) == EXPECTED_CALLS
        and Counter(row["component"] for row in plan)
        == Counter({"RS": 8, "MP": 8, "MS": 8, "ME": 8})
        and all(not row["selection_uses_response_or_judge_outcome"] for row in plan)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "execution_call_plan.jsonl"
    write_jsonl(plan_path, plan)
    outcomes_path = args.out_dir / "execution_outcomes.jsonl"
    existing_rows = _rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["call_id"]): row for row in existing_rows}
    expected_ids = {str(row["call_id"]) for row in plan}
    existing_ok = (
        len(existing) == len(existing_rows)
        and set(existing) <= expected_ids
        and all(row.get("protocol") == PROTOCOL for row in existing_rows)
    )
    pending = [row for row in plan if row["call_id"] not in existing]
    key_present = bool(os.environ.get(endpoint.api_key_env, ""))
    preflight = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_EXECUTION" if plan_ok and existing_ok and key_present else "BLOCKED",
        "plan_ok": plan_ok,
        "existing_outputs_ok": existing_ok,
        "planned_calls": len(plan),
        "completed_calls": len(existing),
        "remaining_calls": len(pending),
        "api_key_present": key_present,
        "run_requested": args.run,
    }
    write_json(args.out_dir / "execution_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if not plan_ok or not existing_ok or not key_present:
        raise RuntimeError(canonical_json(preflight))
    if len(pending) > args.max_new_calls:
        raise RuntimeError(
            f"{len(pending)} calls remain but --max-new-calls={args.max_new_calls}"
        )

    client = make_client(endpoint)
    for row in pending:
        result, parsed = client.chat(
            row["messages"],
            temperature=float(generation.temperature),
            max_tokens=max(500, int(generation.max_output_tokens) + 200),
            seed=int(row["seed"]),
            response_schema=ResourceExecutionOutput,
            retries=3,
        )
        if parsed is None:
            raise RuntimeError("resource execution returned no parsed output")
        checks = validate_resource_execution(
            output=parsed,
            resource_label=row["resource_label"],
            expected_function=row["expected_resource_function"],
            selected_resource=row["selected_resource"],
        )
        usage = require_reported_usage(result.usage, stage="auditable_resource_execution")
        saved = {
            "protocol": PROTOCOL,
            "call_id": row["call_id"],
            "state_id": row["state_id"],
            "component": row["component"],
            "resource_subtype": row["resource_subtype"],
            "messages_sha256": row["messages_sha256"],
            "execution": parsed.model_dump(mode="json"),
            "machine_checks": checks,
            "usage": usage,
            "normalized_finish_reason": result.normalized_finish_reason,
            "provider_finish_reason": result.provider_finish_reason,
            "completed_at": utc_now(),
        }
        append_jsonl(outcomes_path, saved)
        existing[row["call_id"]] = saved

    final = [existing[row["call_id"]] for row in plan]
    cells: list[dict[str, Any]] = []
    for component in ("RS", "MP", "MS", "ME"):
        subset = [row for row in final if row["component"] == component]
        checks_pass = sum(
            int(row["machine_checks"]["all_machine_checks_pass"])
            for row in subset
        )
        use_n = sum(
            int(row["execution"]["resource_decision"] == "use")
            for row in subset
        )
        cells.append(
            {
                "component": component,
                "n": len(subset),
                "declared_use_n": use_n,
                "declared_use_rate": use_n / len(subset),
                "machine_valid_n": checks_pass,
                "machine_valid_rate": checks_pass / len(subset),
            }
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_EXECUTION_TRACE_NOT_YET_HUMAN_QUALIFIED",
        "response_count": len(final),
        "all_machine_checks_pass_n": sum(
            int(row["machine_checks"]["all_machine_checks_pass"])
            for row in final
        ),
        "component_cells": cells,
        "generator_identity": {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
            "transport": endpoint.transport,
        },
        "interpretation_contract": {
            "self_declaration_is_human_gold": False,
            "routing_or_retrieval_rejudged": False,
            "user_sees_only_execution_response_field": True,
            "minimum_human_check_next": "stratified 8-use plus every ignore or machine-invalid output",
        },
        "inputs": {
            "plan_freeze_manifest_sha256": sha256_file(args.plan_dir / "freeze_manifest.json"),
            "execution_call_plan_sha256": sha256_file(plan_path),
        },
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
