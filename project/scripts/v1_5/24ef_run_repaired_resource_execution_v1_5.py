#!/usr/bin/env python3
"""Run the single repaired Step2 prompt on ten fresh development states."""

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
from metacom_pm.io import append_jsonl, canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, stable_hex, utc_now, write_json, write_jsonl
from metacom_pm.prompts import common_context
from metacom_pm.v1_5_generator_alignment_audit import ResourceExecutionOutput, resource_execution_messages_v3, validate_resource_execution


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-repaired-resource-execution-v1"
EXPECTED_BY_COMPONENT = {"RS": 2, "MP": 2, "MS": 3, "ME": 3}


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
    return {"RS": "support_technique", "MS": "session_continuity", "ME": "event_or_outcome"}[component]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1")
    parser.add_argument("--prior-review-dir", type=Path, default=ROOT / "outputs/pm_v1_5_resource_execution_human_check_v1_candidate")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_repaired_resource_execution_v1")
    parser.add_argument("--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml")
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    resources = _rows(args.plan_dir / "private_selected_resources.jsonl")
    calls = _rows(args.plan_dir / "call_plan.jsonl")
    source_matched = {str(row["state_id"]): row for row in calls if row["prompt_variant"] == "source_matched"}
    prior_ids = {str(row["state_id"]) for row in _rows(args.prior_review_dir / "private_review_key.jsonl")}
    selected: list[dict[str, Any]] = []
    for component, count in EXPECTED_BY_COMPONENT.items():
        eligible = [row for row in resources if row["component"] == component and str(row["runtime_state"]["state_id"]) not in prior_ids]
        eligible.sort(key=lambda row: stable_hex(PROTOCOL, "selection", str(row["runtime_state"]["state_id"]), n=32))
        selected.extend(eligible[:count])
    if len(selected) != 10 or Counter(row["component"] for row in selected) != Counter(EXPECTED_BY_COMPONENT):
        raise RuntimeError("fresh repaired execution selection is incomplete")

    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    plan: list[dict[str, Any]] = []
    for selected_resource in selected:
        state = RuntimeState.model_validate(selected_resource["runtime_state"])
        component = str(selected_resource["component"])
        subtype = str(source_matched[state.state_id]["resource_subtype"])
        label = "R1" if component == "RS" else "E1"
        resource = _resource_surface(selected_resource)
        expected = _expected_function(component, subtype)
        messages = resource_execution_messages_v3(
            component=component,
            resource_subtype=subtype,
            visible_context=common_context(state),
            selected_resource=resource,
            resource_label=label,
            system_prompt=generation.system_prompt,
        )
        plan.append(
            {
                "protocol": PROTOCOL,
                "call_id": "resource_repair_" + stable_hex(PROTOCOL, state.state_id, n=24),
                "state_id": state.state_id,
                "component": component,
                "resource_subtype": subtype,
                "resource_label": label,
                "selected_resource": resource,
                "expected_resource_function": expected,
                "messages": messages,
                "messages_sha256": sha256_text(canonical_json(messages)),
                "seed": int(source_matched[state.state_id]["generation"]["seed"]) % (2**31 - 1),
                "selection_rule": "fresh_from_prior_10_stable_hash_2RS_2MP_3MS_3ME",
                "selection_uses_response_judge_risk_or_external_outcome": False,
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "execution_call_plan.jsonl"
    write_jsonl(plan_path, plan)
    outcomes_path = args.out_dir / "execution_outcomes.jsonl"
    existing_rows = _rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["call_id"]): row for row in existing_rows}
    expected_ids = {str(row["call_id"]) for row in plan}
    existing_ok = len(existing) == len(existing_rows) and set(existing) <= expected_ids and all(row.get("protocol") == PROTOCOL for row in existing_rows)
    pending = [row for row in plan if row["call_id"] not in existing]
    key_present = bool(os.environ.get(endpoint.api_key_env, ""))
    preflight = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_EXECUTION" if existing_ok and key_present else "BLOCKED",
        "fresh_from_prior_review": not ({row["state_id"] for row in plan} & prior_ids),
        "planned_calls": len(plan),
        "completed_calls": len(existing),
        "remaining_calls": len(pending),
        "component_counts": dict(sorted(Counter(row["component"] for row in plan).items())),
        "api_key_present": key_present,
        "run_requested": args.run,
    }
    write_json(args.out_dir / "execution_preflight.json", preflight)
    if not args.run:
        print(preflight)
        return
    if preflight["status"] != "READY_FOR_EXECUTION":
        raise RuntimeError(canonical_json(preflight))
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
            raise RuntimeError("repaired resource execution returned no parsed output")
        checks = validate_resource_execution(output=parsed, resource_label=row["resource_label"], expected_function=row["expected_resource_function"], selected_resource=row["selected_resource"])
        saved = {
            "protocol": PROTOCOL,
            "call_id": row["call_id"],
            "state_id": row["state_id"],
            "component": row["component"],
            "resource_subtype": row["resource_subtype"],
            "messages_sha256": row["messages_sha256"],
            "execution": parsed.model_dump(mode="json"),
            "machine_checks": checks,
            "usage": require_reported_usage(result.usage, stage="repaired_resource_execution"),
            "normalized_finish_reason": result.normalized_finish_reason,
            "provider_finish_reason": result.provider_finish_reason,
            "completed_at": utc_now(),
        }
        append_jsonl(outcomes_path, saved)
        existing[row["call_id"]] = saved
    final = [existing[row["call_id"]] for row in plan]
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_REPAIRED_EXECUTION_AWAITING_HUMAN_CHECK",
        "responses": len(final),
        "declared_decisions": dict(sorted(Counter(row["execution"]["resource_decision"] for row in final).items())),
        "structured_outputs_parsed": len(final),
        "decision_bookkeeping_valid": sum(bool(row["machine_checks"]["decision_bookkeeping_valid"]) for row in final),
        "exact_span_copy_telemetry_pass": sum(bool(row["machine_checks"]["all_machine_checks_pass"]) for row in final),
        "exact_span_copy_is_scientific_gate": False,
        "selection_uses_response_judge_risk_or_external_outcome": False,
        "inputs": {"call_plan_sha256": sha256_file(plan_path)},
    }
    write_json(args.out_dir / "execution_summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
