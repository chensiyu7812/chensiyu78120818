#!/usr/bin/env python3
"""Execute the frozen, deduplicated V1.5b final system generation plan."""

from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
from typing import Any

from metacom_pm.api import (
    StructuredOutputValidationError,
    make_client,
    require_reported_usage,
)
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    utc_now,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_generator_alignment_audit import (
    ResourceExecutionPlan,
    resource_bundle_output_schema_for_components,
    validate_resource_bundle_application,
)
from metacom_pm.v1_5b_policy_runtime import realize_guarded_action


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-final-system-generation-execution-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _verify_t0_freeze(plan_dir: Path, calls: list[dict[str, Any]]) -> None:
    """Fail closed if any T0 implementation or exact-action schema changed."""

    manifest_path = plan_dir / "t0_freeze_manifest.json"
    if not manifest_path.is_file():
        return  # Historical plans predate the finite T0/T1 protocol.
    manifest = read_json(manifest_path)
    expected_paths = {
        "component_binding_and_step2_contract": ROOT
        / "src/metacom_pm/v1_5_generator_alignment_audit.py",
        "replacement_plan_builder": ROOT
        / "scripts/v1_5/25zs_prepare_v3_replacement_h_step2_execution_v1_5.py",
        "generation_runner": Path(__file__).resolve(),
        "review_packet_builder": ROOT
        / "scripts/v1_5/25zt_prepare_v3_replacement_h_step2_human_review_v1_5.py",
        "review_aggregator": ROOT
        / "scripts/v1_5/25zu_aggregate_v3_replacement_h_step2_review_v1_5.py",
        "joint_feasibility_runtime": ROOT / "src/metacom_pm/v1_5b_policy_runtime.py",
    }
    actual_implementation = {
        name: sha256_file(path) for name, path in expected_paths.items()
    }
    if actual_implementation != manifest.get("implementation_sha256"):
        raise RuntimeError("T0 implementation identity changed after freeze")
    actual_schemas = {
        action_id: sha256_text(
            canonical_json(
                resource_bundle_output_schema_for_components(components).model_json_schema()
            )
        )
        for action_id, components in sorted(
            {
                str(call["requested_action_id"]): tuple(call["requested_components"])
                for call in calls
            }.items()
        )
    }
    if actual_schemas != manifest.get("exact_action_schema_sha256"):
        raise RuntimeError("T0 exact-action schema identity changed after freeze")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5b_final_system_generation_v1_candidate",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5b_final_system_generation_v1_execution",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    plan_path = args.plan_dir / "call_plan_private.jsonl"
    preflight_path = args.plan_dir / "generation_preflight.json"
    preflight = read_json(preflight_path)
    if preflight["status"] not in {
        "READY_FOR_FINAL_PAID_GENERATION_REVIEW",
        "READY_FOR_FINAL_V2_PAID_GENERATION_REVIEW",
        "READY_FOR_T1_REPLACEMENT_PAID_GENERATION_REVIEW",
    }:
        raise RuntimeError("final generation preflight is not ready")
    if sha256_file(plan_path) != preflight["call_plan_sha256"]:
        raise RuntimeError("final call plan hash changed after review")
    calls = _rows(plan_path)
    _verify_t0_freeze(args.plan_dir, calls)
    dry = {
        "protocol": PROTOCOL,
        "status": "READY" if not args.run else "RUN_REQUESTED",
        "planned_primary_calls": len(calls),
        "maximum_fallback_calls": preflight["maximum_fallback_calls"],
        "estimated_primary_generation_usd": preflight["estimated_primary_generation_usd"],
        "estimated_generation_usd_with_all_fallbacks": preflight["estimated_generation_usd_with_all_fallbacks"],
        "call_plan_sha256": preflight["call_plan_sha256"],
    }
    if not args.run:
        print(dry)
        return

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")
    client = make_client(endpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.out_dir / "generation_outcomes_private.jsonl"
    existing_rows = _rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["call_id"]): row for row in existing_rows}
    expected = {str(row["call_id"]) for row in calls}
    if len(existing) != len(existing_rows) or not set(existing) <= expected:
        raise RuntimeError("existing final outcomes do not match the frozen call plan")

    for call in calls:
        call_id = str(call["call_id"])
        if call_id in existing:
            continue
        requested_action = str(call["requested_action_id"])
        if call["response_schema"] == "plain_text":
            result, _ = client.chat(
                call["messages"],
                temperature=float(generation.temperature),
                max_tokens=int(call["primary_output_token_cap"]),
                seed=int(call["seed"]),
                retries=3,
            )
            usage = require_reported_usage(result.usage, stage="v1_5b_final_context_only")
            if int(usage["prompt_tokens"]) > int(call["input_token_upper_bound"]):
                raise RuntimeError(f"reported input overrun for {call_id}")
            response = generation.normalize_output(result.text)
            saved = {
                "protocol": PROTOCOL,
                "call_id": call_id,
                "panel_id": call["panel_id"],
                "domain": call["domain"],
                "state_id": call["state_id"],
                "group_id_private_analysis_only": call["group_id_private_analysis_only"],
                "requested_action_id": requested_action,
                "realized_action_id": "M0+R0",
                "policy_aliases": call["policy_aliases"],
                "messages_sha256": call["messages_sha256"],
                "primary_usage": usage,
                "primary_finish_reason": result.normalized_finish_reason,
                "bundle_application": None,
                "bundle_machine_checks": None,
                "fallback_executed": False,
                "fallback": None,
                "final_response": response,
                "completed_at": utc_now(),
            }
        else:
            schema_failure = None
            response_schema = resource_bundle_output_schema_for_components(
                list(call["requested_components"])
            )
            try:
                result, parsed = client.chat(
                    call["messages"],
                    temperature=float(generation.temperature),
                    max_tokens=int(call["primary_output_token_cap"]),
                    seed=int(call["seed"]),
                    response_schema=response_schema,
                    retries=3,
                )
            except StructuredOutputValidationError as exc:
                # The paid provider call completed, but its trace was internally
                # contradictory.  This is an execution failure for this logical
                # action, not a reason to abort the frozen panel or silently
                # repair/retry the resource response.  Preserve its usage and
                # private diagnostics, realize every requested component OFF,
                # and execute the already-frozen resource-free fallback once.
                result = exc.call
                parsed = None
                schema_failure = {
                    "error_type": type(exc).__name__,
                    "validation_errors": exc.validation_errors,
                    "invalid_payload_private": exc.parsed_payload,
                }
            if parsed is None and schema_failure is None:
                raise RuntimeError(f"bundle application returned no parsed object: {call_id}")
            usage = require_reported_usage(result.usage, stage="v1_5b_final_resource_bundle")
            if int(usage["prompt_tokens"]) > int(call["input_token_upper_bound"]):
                raise RuntimeError(f"reported input overrun for {call_id}")
            plans = {
                component: ResourceExecutionPlan.model_validate(value)
                for component, value in dict(call["execution_plans"]).items()
            }
            if schema_failure is None:
                checks = validate_resource_bundle_application(
                    output=parsed,
                    plans=plans,
                    resource_labels=dict(call["resource_labels"]),
                    selected_resources=dict(call["selected_resources_private"]),
                    current_user_text=str(call["current_user_text"]),
                )
            else:
                checks = {
                    "exact_component_bookkeeping": False,
                    "requested_component_bookkeeping_valid": False,
                    "extraneous_component_statuses_ignored_for_routing": [],
                    "duplicate_identical_requested_status_count": 0,
                    "component_checks": {},
                    "unsafe_applied_component": False,
                    "component_realized_on": {
                        component: False for component in plans
                    },
                    "realized_component_count": 0,
                    "fallback_required": True,
                    "structured_output_valid": False,
                    "fallback_reason": "structured_output_validation_error",
                }
            fallback = None
            if checks["fallback_required"]:
                fallback_result, _ = client.chat(
                    call["fallback_messages"],
                    temperature=float(generation.temperature),
                    max_tokens=int(call["maximum_fallback_output_token_cap"]),
                    seed=int(call["seed"]),
                    retries=3,
                )
                fallback_usage = require_reported_usage(
                    fallback_result.usage, stage="v1_5b_final_resource_free_fallback"
                )
                if int(fallback_usage["prompt_tokens"]) > int(call["maximum_fallback_input_token_upper_bound"]):
                    raise RuntimeError(f"reported fallback input overrun for {call_id}")
                fallback = {
                    "response": generation.normalize_output(fallback_result.text),
                    "usage": fallback_usage,
                    "normalized_finish_reason": fallback_result.normalized_finish_reason,
                    "provider_finish_reason": fallback_result.provider_finish_reason,
                }
                final_response = fallback["response"]
                realized_action = "M0+R0"
            else:
                final_response = generation.normalize_output(parsed.response)
                realized_action = realize_guarded_action(
                    requested_action_id=requested_action,
                    component_passed=checks["component_realized_on"],
                )
            saved = {
                "protocol": PROTOCOL,
                "call_id": call_id,
                "panel_id": call["panel_id"],
                "domain": call["domain"],
                "state_id": call["state_id"],
                "group_id_private_analysis_only": call["group_id_private_analysis_only"],
                "requested_action_id": requested_action,
                "realized_action_id": realized_action,
                "policy_aliases": call["policy_aliases"],
                "messages_sha256": call["messages_sha256"],
                "primary_usage": usage,
                "primary_finish_reason": result.normalized_finish_reason,
                "bundle_application": (
                    parsed.model_dump(mode="json") if parsed is not None else None
                ),
                "bundle_schema_failure_private": schema_failure,
                "bundle_machine_checks": checks,
                "fallback_executed": bool(fallback),
                "fallback": fallback,
                "final_response": final_response,
                "completed_at": utc_now(),
            }
        append_jsonl(outcomes_path, saved)
        existing[call_id] = saved

    final = [existing[str(call["call_id"])] for call in calls]
    write_jsonl(args.out_dir / "generation_outcomes_ordered_private.jsonl", final)
    total_prompt = sum(
        int(row["primary_usage"]["prompt_tokens"])
        + int((row.get("fallback") or {}).get("usage", {}).get("prompt_tokens", 0))
        for row in final
    )
    total_completion = sum(
        int(row["primary_usage"]["completion_tokens"])
        + int((row.get("fallback") or {}).get("usage", {}).get("completion_tokens", 0))
        for row in final
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "COMPLETE_AWAITING_T1_REPLACEMENT_H_STEP2_HUMAN_GATE"
            if preflight["status"]
            == "READY_FOR_T1_REPLACEMENT_PAID_GENERATION_REVIEW"
            else "COMPLETE_AWAITING_FROZEN_QUALITY_RISK_COST_EVALUATION"
        ),
        "planned_primary_calls": len(calls),
        "completed_primary_calls": len(final),
        "fallback_calls": sum(bool(row["fallback_executed"]) for row in final),
        "structured_output_validation_fallbacks": sum(
            bool(row.get("bundle_schema_failure_private")) for row in final
        ),
        "requested_action_distribution": dict(sorted(Counter(row["requested_action_id"] for row in final).items())),
        "realized_action_distribution": dict(sorted(Counter(row["realized_action_id"] for row in final).items())),
        "reported_prompt_tokens_including_fallbacks": total_prompt,
        "reported_completion_tokens_including_fallbacks": total_completion,
        "call_plan_sha256": preflight["call_plan_sha256"],
        "ordered_outcomes_sha256": sha256_file(args.out_dir / "generation_outcomes_ordered_private.jsonl"),
        "external_outcome_used_to_change_routing_or_generation": False,
    }
    write_json(args.out_dir / "execution_summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
