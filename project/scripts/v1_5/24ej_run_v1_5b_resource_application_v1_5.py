#!/usr/bin/env python3
"""Run the V1.5b deterministic resource-application contract on 32 use cases.

The selected resources and states are reused from the already frozen 8-per-
component qualification pool.  This run changes only the Step1-to-Step2
execution interface.  A cannot-apply result or failed machine validation
realizes the component as OFF and triggers at most one resource-free fallback;
both calls remain in cost accounting.
"""

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
from metacom_pm.io import append_jsonl, canonical_json, iter_jsonl, sha256_file, sha256_text, stable_hex, utc_now, write_json, write_jsonl
from metacom_pm.prompts import common_context, generation_messages
from metacom_pm.v1_5_generator_alignment_audit import (
    ResourceApplicationOutput,
    build_resource_execution_plan,
    resource_application_messages_v1_5b,
    validate_resource_application,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-resource-application-execution-v1"
SOURCE_PROTOCOL = "pm-v1.5-generator-alignment-qualification-v1"


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


def build_plan(
    *, source_dir: Path, out_dir: Path, pm_config_path: Path, experiment_config_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resources = _rows(source_dir / "private_selected_resources.jsonl")
    source_calls = _rows(source_dir / "call_plan.jsonl")
    source_matched = {
        str(row["state_id"]): row
        for row in source_calls
        if row["prompt_variant"] == "source_matched"
    }
    if len(resources) != 32 or Counter(row["component"] for row in resources) != Counter({"MP": 8, "MS": 8, "ME": 8, "RS": 8}):
        raise RuntimeError("expected the frozen 8-per-component resource pool")
    pm_config = load_config(pm_config_path)
    experiment = load_config(experiment_config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)

    calls: list[dict[str, Any]] = []
    for row in sorted(resources, key=lambda value: (str(value["component"]), str(value["state_id"]))):
        state = RuntimeState.model_validate(row["runtime_state"])
        source_call = source_matched[state.state_id]
        component = str(row["component"])
        subtype = str(source_call["resource_subtype"])
        resource_label = "R1" if component == "RS" else "E1"
        resource = _resource_surface(row)
        plan = build_resource_execution_plan(component=component, resource_subtype=subtype)
        messages = resource_application_messages_v1_5b(
            plan=plan,
            visible_context=common_context(state),
            selected_resource=resource,
            resource_label=resource_label,
            system_prompt=generation.system_prompt,
        )
        fallback_messages = generation_messages(
            state, [], [], system_prompt=generation.system_prompt
        )
        calls.append(
            {
                "protocol": PROTOCOL,
                "call_id": "v15b_apply_" + stable_hex(PROTOCOL, state.state_id, component, n=24),
                "state_id": state.state_id,
                "component": component,
                "resource_subtype": subtype,
                "resource_label": resource_label,
                "selected_resource": resource,
                "current_user_text": state.current_user_text,
                "visible_context": common_context(state),
                "execution_plan": plan.model_dump(mode="json"),
                "messages": messages,
                "messages_sha256": sha256_text(canonical_json(messages)),
                "fallback_messages": fallback_messages,
                "fallback_messages_sha256": sha256_text(canonical_json(fallback_messages)),
                "seed": int(source_call["generation"]["seed"]) % (2**31 - 1),
                "selection_rule": "reuse_frozen_8_per_component_resource_execution_pool",
                "selection_uses_response_judge_risk_or_external_outcome": False,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "execution_call_plan.jsonl"
    write_jsonl(plan_path, calls)
    preflight = {
        "protocol": PROTOCOL,
        "status": "READY_32_USE_CASES",
        "planned_primary_calls": 32,
        "maximum_fallback_calls": 32,
        "component_counts": dict(sorted(Counter(row["component"] for row in calls).items())),
        "same_frozen_states_and_resources_as_prior_diagnostic": True,
        "only_execution_interface_changed": True,
        "external_outcome_read": False,
        "generator_endpoint": endpoint.model,
        "api_key_env": endpoint.api_key_env,
        "api_key_present": bool(os.environ.get(endpoint.api_key_env, "")),
        "inputs": {
            str((source_dir / "private_selected_resources.jsonl").relative_to(ROOT)): sha256_file(source_dir / "private_selected_resources.jsonl"),
            str((source_dir / "call_plan.jsonl").relative_to(ROOT)): sha256_file(source_dir / "call_plan.jsonl"),
        },
    }
    write_json(out_dir / "execution_preflight.json", preflight)
    return calls, preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "outputs/pm_v1_5_generator_alignment_qualification_v1")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5b_resource_application_v1")
    parser.add_argument("--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml")
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    calls, preflight = build_plan(
        source_dir=args.source_dir,
        out_dir=args.out_dir,
        pm_config_path=args.pm_config,
        experiment_config_path=args.experiment_config,
    )
    if not args.run:
        print(preflight)
        return

    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")
    client = make_client(endpoint)
    outcomes_path = args.out_dir / "execution_outcomes.jsonl"
    existing_rows = _rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["call_id"]): row for row in existing_rows}
    expected_ids = {str(row["call_id"]) for row in calls}
    if len(existing) != len(existing_rows) or not set(existing) <= expected_ids:
        raise RuntimeError("existing execution outcomes do not match the frozen plan")

    for row in calls:
        if row["call_id"] in existing:
            continue
        result, parsed = client.chat(
            row["messages"],
            temperature=float(generation.temperature),
            max_tokens=max(500, int(generation.max_output_tokens) + 200),
            seed=int(row["seed"]),
            response_schema=ResourceApplicationOutput,
            retries=3,
        )
        if parsed is None:
            raise RuntimeError("V1.5b application returned no parsed output")
        plan = build_resource_execution_plan(
            component=row["component"], resource_subtype=row["resource_subtype"]
        )
        checks = validate_resource_application(
            output=parsed,
            plan=plan,
            resource_label=row["resource_label"],
            selected_resource=row["selected_resource"],
            current_user_text=row["visible_context"],
        )
        fallback_needed = bool(checks["fallback_required"] or not checks["all_machine_checks_pass"])
        fallback: dict[str, Any] | None = None
        if fallback_needed:
            fallback_result, _ = client.chat(
                row["fallback_messages"],
                temperature=float(generation.temperature),
                max_tokens=int(generation.max_output_tokens),
                seed=int(row["seed"]),
                retries=3,
            )
            fallback = {
                "response": generation.normalize_output(fallback_result.text),
                "usage": require_reported_usage(fallback_result.usage, stage="v1_5b_resource_free_fallback"),
                "normalized_finish_reason": fallback_result.normalized_finish_reason,
                "provider_finish_reason": fallback_result.provider_finish_reason,
            }
        saved = {
            "protocol": PROTOCOL,
            "call_id": row["call_id"],
            "state_id": row["state_id"],
            "component": row["component"],
            "resource_subtype": row["resource_subtype"],
            "messages_sha256": row["messages_sha256"],
            "application": parsed.model_dump(mode="json"),
            "machine_checks": checks,
            "primary_usage": require_reported_usage(result.usage, stage="v1_5b_resource_application"),
            "fallback": fallback,
            "fallback_executed": fallback_needed,
            "final_response": fallback["response"] if fallback is not None else parsed.response,
            "realized_component_on": bool(checks["realized_component_on"]),
            "completed_at": utc_now(),
        }
        append_jsonl(outcomes_path, saved)
        existing[row["call_id"]] = saved

    final = [existing[row["call_id"]] for row in calls]
    corrected_final: list[dict[str, Any]] = []
    call_by_id = {str(row["call_id"]): row for row in calls}
    for row in final:
        call = call_by_id[str(row["call_id"])]
        plan = build_resource_execution_plan(
            component=row["component"], resource_subtype=row["resource_subtype"]
        )
        parsed = ResourceApplicationOutput.model_validate(row["application"])
        corrected_checks = validate_resource_application(
            output=parsed,
            plan=plan,
            resource_label="R1" if row["component"] == "RS" else "E1",
            selected_resource=call["selected_resource"],
            current_user_text=call["visible_context"],
        )
        corrected_final.append(
            {
                **row,
                "machine_checks": corrected_checks,
                "realized_component_on": bool(
                    corrected_checks["realized_component_on"]
                ),
            }
        )
    final = corrected_final
    write_jsonl(
        args.out_dir / "execution_outcomes_revalidated_v2.jsonl", final
    )
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_AWAITING_MINIMUM_FUNCTIONAL_HUMAN_CHECK",
        "planned_primary_calls": len(calls),
        "completed_primary_calls": len(final),
        "fallback_calls": sum(bool(row["fallback_executed"]) for row in final),
        "realized_component_on": sum(bool(row["realized_component_on"]) for row in final),
        "application_status_counts": dict(sorted(Counter(row["application"]["application_status"] for row in final).items())),
        "machine_checks_pass": sum(bool(row["machine_checks"]["all_machine_checks_pass"]) for row in final),
        "component_realized_on": {
            component: sum(bool(row["realized_component_on"]) for row in final if row["component"] == component)
            for component in ("MP", "MS", "ME", "RS")
        },
        "response_quality_or_external_outcome_used": False,
        "input_call_plan_sha256": sha256_file(args.out_dir / "execution_call_plan.jsonl"),
    }
    write_json(args.out_dir / "execution_summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
