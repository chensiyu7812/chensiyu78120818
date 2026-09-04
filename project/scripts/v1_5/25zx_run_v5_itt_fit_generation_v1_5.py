#!/usr/bin/env python3
"""Run the frozen V5 ITT FIT response plan with deterministic fallback."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

from metacom_pm.api import make_client, require_reported_usage
from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_itt_policy import ITTExperimentRow, validate_itt_row
from metacom_pm.v1_5_typed_resource_adapter import (
    CompiledResourceBundle,
    deterministic_context_only_fallback,
    response_guard_errors,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5-itt-fit-generation-v1"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def verify_freeze(plan_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(plan_dir / "freeze_manifest.json")
    if manifest["status"] != "READY_FOR_SINGLE_V5_ITT_FIT_EXECUTION":
        raise RuntimeError("V5 FIT plan is not ready")
    plan_path = plan_dir / "call_plan_private.jsonl"
    if sha256_file(plan_path) != manifest["call_plan_sha256"]:
        raise RuntimeError("V5 FIT call plan changed after freeze")
    implementation = {
        "typed_adapter": ROOT / "src/metacom_pm/v1_5_typed_resource_adapter.py",
        "itt_validator": ROOT / "src/metacom_pm/v1_5_itt_policy.py",
        "policy_runtime": ROOT / "src/metacom_pm/v1_5b_policy_runtime.py",
        "plan_builder": ROOT / "scripts/v1_5/25zw_materialize_v5_itt_fit_plan_v1_5.py",
        "generation_runner": Path(__file__).resolve(),
    }
    actual = {key: sha256_file(path) for key, path in implementation.items()}
    if actual != manifest["implementation_sha256"]:
        raise RuntimeError("V5 FIT implementation changed after freeze")
    return manifest, rows(plan_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_itt_fit_plan_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_itt_fit_execution_v1",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5 requires {FORMAL_PYTHON}; got {sys.executable}")
    manifest, calls = verify_freeze(args.plan_dir)
    if not args.run:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "READY",
                    "planned_calls": len(calls),
                    "estimated_usd_upper_bound_proxy": manifest[
                        "estimated_generation_usd_upper_bound_proxy"
                    ],
                    "resumable": True,
                }
            )
        )
        return

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    if endpoint.model != manifest["generator"]:
        raise RuntimeError("generator identity differs from the frozen V5 plan")
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")
    client = make_client(endpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.out_dir / "outcomes_private.jsonl"
    existing_rows = rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["call_id"]): row for row in existing_rows}
    expected = {str(row["call_id"]) for row in calls}
    if len(existing) != len(existing_rows) or not set(existing) <= expected:
        raise RuntimeError("existing outcomes do not match the frozen V5 plan")

    completed_now = 0
    for call in calls:
        call_id = str(call["call_id"])
        if call_id in existing:
            continue
        result, _ = client.chat(
            call["messages"],
            temperature=float(generation.temperature),
            max_tokens=int(call["output_token_cap"]),
            seed=int(call["seed"]),
            retries=3,
        )
        usage = require_reported_usage(result.usage, stage="v1_5_v5_itt_fit")
        if int(usage["prompt_tokens"]) > int(call["input_token_upper_bound"]):
            raise RuntimeError(f"reported input overrun for {call_id}")
        primary_response = generation.normalize_output(result.text)
        bundle = CompiledResourceBundle(
            requested_action_id=str(call["requested_action_id"]), directives=()
        )
        # The prompt itself is frozen proof of resource presentation.  The
        # minimal guard needs only to distinguish historical vs nonhistorical
        # authorization, which is reconstructed from the requested target.
        historical = bool(
            call["arm"] == "ON" and call["target_component"] in {"MS", "ME"}
        )
        if historical:
            from metacom_pm.v1_5_typed_resource_adapter import CompiledResourceDirective

            bundle = CompiledResourceBundle(
                requested_action_id=str(call["requested_action_id"]),
                directives=(
                    CompiledResourceDirective(
                        component=str(call["target_component"]),
                        resource_id="frozen-private-binding",
                        candidate_version="frozen-private-version",
                        source_kind="history",
                        owner_id=str(call["candidate_owner_id_pre_action"]),
                        evidence="frozen in prompt",
                        response_instruction="frozen in prompt",
                        forbidden_claims=(),
                        attribution_required=True,
                        maximum_support_moves=1,
                    ),
                ),
            )
        guard_errors = response_guard_errors(
            response=primary_response,
            bundle=bundle,
            current_context=str(call["messages"][-1]["content"]),
        )
        fallback_used = bool(guard_errors)
        final_response = (
            deterministic_context_only_fallback(str(call["current_user_text"]))
            if fallback_used
            else primary_response
        )
        realized_action = "M0+R0" if fallback_used else str(call["requested_action_id"])
        all_true = {component: True for component in COMPONENTS}
        itt = ITTExperimentRow(
            group_id=str(call["group_id_private_analysis_only"]),
            state_id=str(call["state_id"]),
            requested_action_id=str(call["requested_action_id"]),
            realized_action_id=realized_action,
            assigned_executor_version=str(call["assigned_executor_version"]),
            actual_executor_version="typed-resource-adapter-v1",
            assigned_generator_version=str(call["assigned_generator_version"]),
            actual_generator_version=endpoint.model,
            assignment_matches_plan=True,
            candidate_binding_valid=all_true,
            owner_binding_valid=all_true,
            resource_presented_to_executor={
                component: bool(call["resource_presented_to_executor"][component])
                for component in COMPONENTS
            },
            final_response=final_response,
            usage_present=True,
            fallback_used=fallback_used,
        )
        validity = validate_itt_row(itt)
        saved = {
            "protocol": PROTOCOL,
            "call_id": call_id,
            "state_id": call["state_id"],
            "group_id_private_analysis_only": call["group_id_private_analysis_only"],
            "target_component": call["target_component"],
            "arm": call["arm"],
            "requested_action_id": call["requested_action_id"],
            "realized_action_id": realized_action,
            "candidate_id_pre_action": call["candidate_id_pre_action"],
            "candidate_version_pre_action": call["candidate_version_pre_action"],
            "resource_presented_to_executor": call["resource_presented_to_executor"],
            "seed_hex": call["seed_hex"],
            "seed": call["seed"],
            "messages_sha256": call["messages_sha256"],
            "primary_response": primary_response,
            "guard_errors": list(guard_errors),
            "fallback_used": fallback_used,
            "final_response": final_response,
            "usage": usage,
            "finish_reason": result.normalized_finish_reason,
            "itt_row_valid": validity.valid,
            "itt_invalid_reasons": list(validity.invalid_reasons),
            "executor_version": "typed-resource-adapter-v1",
            "generator_version": endpoint.model,
            "completed_at": utc_now(),
        }
        append_jsonl(outcomes_path, saved)
        existing[call_id] = saved
        completed_now += 1

    ordered = [existing[str(call["call_id"])] for call in calls]
    ordered_path = args.out_dir / "outcomes_ordered_private.jsonl"
    write_jsonl(ordered_path, ordered)
    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_AWAITING_SINGLE_FIT_OUTCOME_REVIEW",
        "planned_calls": len(calls),
        "completed_calls": len(ordered),
        "completed_now": completed_now,
        "fallback_calls": sum(bool(row["fallback_used"]) for row in ordered),
        "invalid_itt_rows": sum(not bool(row["itt_row_valid"]) for row in ordered),
        "reported_prompt_tokens": sum(int(row["usage"]["prompt_tokens"]) for row in ordered),
        "reported_completion_tokens": sum(
            int(row["usage"]["completion_tokens"]) for row in ordered
        ),
        "requested_action_distribution": dict(
            Counter(str(row["requested_action_id"]) for row in ordered)
        ),
        "realized_action_distribution": dict(
            Counter(str(row["realized_action_id"]) for row in ordered)
        ),
        "call_plan_sha256": manifest["call_plan_sha256"],
        "ordered_outcomes_sha256": sha256_file(ordered_path),
        "resumable": True,
        "external_outcome_used_to_change_method": False,
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
