#!/usr/bin/env python3
"""Execute the sole frozen V5.2 content-disjoint confirmation plan."""

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
    iter_jsonl,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_itt_policy import ITTExperimentRow, validate_itt_row
from metacom_pm.v1_5_v5_2_locked_composer import (
    LockedClause,
    LockedCompositionPlan,
    compose_locked_response,
    locked_response_guard_errors,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-content-disjoint-confirmation-generation-v1"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def composition_plan(raw: dict[str, Any]) -> LockedCompositionPlan:
    return LockedCompositionPlan(
        requested_action_id=str(raw["requested_action_id"]),
        locked_clauses=tuple(
            LockedClause(**dict(item)) for item in raw["locked_clauses"]
        ),
        response_preference=str(raw["response_preference"]),
        strategy_instruction=str(raw["strategy_instruction"]),
        maximum_generator_support_moves=int(raw["maximum_generator_support_moves"]),
    )


def verify_freeze(plan_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(plan_dir / "freeze_manifest.json")
    if (
        manifest.get("status")
        != "READY_FOR_SINGLE_V5_2_CONTENT_DISJOINT_CONFIRMATION_EXECUTION"
    ):
        raise RuntimeError("V5.2 confirmation plan is not ready")
    plan_path = plan_dir / "call_plan_private.jsonl"
    if sha256_file(plan_path) != manifest["call_plan_sha256"]:
        raise RuntimeError("V5.2 confirmation call plan changed after freeze")
    seal = read_json(plan_dir / "execution_seal.json")
    if seal.get("status") != "SEALED_BEFORE_CONFIRMATION_EXECUTION":
        raise RuntimeError("confirmation execution is not sealed")
    if seal["call_plan_sha256"] != manifest["call_plan_sha256"]:
        raise RuntimeError("execution seal references a different call plan")
    for relative, expected in seal["implementation_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"sealed implementation changed: {relative}")
    return manifest, rows(plan_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_execution_v1",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.2 requires {FORMAL_PYTHON}; got {sys.executable}")
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
        raise RuntimeError("generator identity differs from frozen confirmation plan")
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")
    client = make_client(endpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.out_dir / "outcomes_private.jsonl"
    existing_rows = rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["call_id"]): row for row in existing_rows}
    expected = {str(row["call_id"]) for row in calls}
    if len(existing) != len(existing_rows) or not set(existing) <= expected:
        raise RuntimeError("existing outcomes do not match the frozen confirmation")

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
        usage = require_reported_usage(result.usage, stage="v1_5_v5_2_confirmation")
        if int(usage["prompt_tokens"]) > int(call["input_token_upper_bound"]):
            raise RuntimeError(f"reported input overrun for {call_id}")
        primary = generation.normalize_output(result.text)
        plan = composition_plan(dict(call["composition_plan"]))
        final = compose_locked_response(base_response=primary, plan=plan)
        guard_errors = locked_response_guard_errors(response=final, plan=plan)
        if guard_errors:
            raise RuntimeError(
                f"backend-locked confirmation response failed: {call_id}: {guard_errors}"
            )
        all_true = {component: True for component in COMPONENTS}
        itt = ITTExperimentRow(
            group_id=str(call["counterfactual_group_id_private_analysis_only"]),
            state_id=str(call["state_id"]),
            requested_action_id=str(call["requested_action_id"]),
            realized_action_id=str(call["requested_action_id"]),
            assigned_executor_version=str(call["assigned_executor_version"]),
            actual_executor_version="v5.2-backend-locked-composer-v1",
            assigned_generator_version=str(call["assigned_generator_version"]),
            actual_generator_version=endpoint.model,
            assignment_matches_plan=True,
            candidate_binding_valid=all_true,
            owner_binding_valid=all_true,
            resource_presented_to_executor={
                component: bool(call["resource_presented_to_executor"][component])
                for component in COMPONENTS
            },
            final_response=final,
            usage_present=True,
            fallback_used=False,
        )
        validity = validate_itt_row(itt)
        saved = {
            "protocol": PROTOCOL,
            "call_id": call_id,
            "state_id": call["state_id"],
            "counterfactual_group_id_private_analysis_only": call[
                "counterfactual_group_id_private_analysis_only"
            ],
            "target_component": call["target_component"],
            "arm": call["arm"],
            "requested_action_id": call["requested_action_id"],
            "realized_action_id": call["requested_action_id"],
            "candidate_id_pre_action": call["candidate_id_pre_action"],
            "candidate_version_pre_action": call["candidate_version_pre_action"],
            "resource_presented_to_executor": call["resource_presented_to_executor"],
            "seed_hex": call["seed_hex"],
            "seed": call["seed"],
            "messages_sha256": call["messages_sha256"],
            "primary_response": primary,
            "locked_clauses": call["composition_plan"]["locked_clauses"],
            "guard_errors": [],
            "fallback_used": False,
            "final_response": final,
            "usage": usage,
            "finish_reason": result.normalized_finish_reason,
            "itt_row_valid": validity.valid,
            "itt_invalid_reasons": list(validity.invalid_reasons),
            "executor_version": "v5.2-backend-locked-composer-v1",
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
        "status": "COMPLETE_AWAITING_SINGLE_CONFIRMATION_OUTCOME_PANEL",
        "planned_calls": len(calls),
        "completed_calls": len(ordered),
        "completed_now": completed_now,
        "fallback_calls": 0,
        "invalid_itt_rows": sum(not bool(row["itt_row_valid"]) for row in ordered),
        "reported_prompt_tokens": sum(
            int(row["usage"]["prompt_tokens"]) for row in ordered
        ),
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
        "confirmation_or_external_outcome_used_to_change_method": False,
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
