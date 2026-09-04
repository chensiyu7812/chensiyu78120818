#!/usr/bin/env python3
"""Run the single frozen V5.1 T5 generation plan resumably."""

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
from metacom_pm.v1_5_typed_resource_adapter import (
    CompiledResourceBundle,
    CompiledResourceDirective,
    deterministic_context_only_fallback,
    response_guard_errors,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.1-t5-generation-v1"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def verify_freeze(plan_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(plan_dir / "freeze_manifest.json")
    if manifest.get("status") != "READY_FOR_SINGLE_T5_GENERATION":
        raise RuntimeError("T5 plan is not ready")
    artifacts = {
        "call_plan_sha256": plan_dir / "call_plan_private.jsonl",
        "policy_bindings_sha256": plan_dir / "policy_bindings_private.jsonl",
        "llm_judge_design_sha256": plan_dir / "llm_judge_design_private.jsonl",
        "human_review_design_sha256": plan_dir / "human_review_design_private.jsonl",
    }
    for field, path in artifacts.items():
        if sha256_file(path) != manifest[field]:
            raise RuntimeError(f"T5 frozen artifact changed: {path.name}")
    implementations = {
        "typed_adapter": ROOT / "src/metacom_pm/v1_5_typed_resource_adapter.py",
        "policy_runtime": ROOT / "src/metacom_pm/v1_5b_policy_runtime.py",
        "surface_builder": ROOT / "scripts/v1_5/26l_materialize_v5_1_t5_surfaces_v1_5.py",
        "plan_builder": ROOT / "scripts/v1_5/26m_freeze_v5_1_t5_plan_v1_5.py",
        "runner": Path(__file__).resolve(),
    }
    actual = {name: sha256_file(path) for name, path in implementations.items()}
    if actual != manifest["implementation_sha256"]:
        raise RuntimeError("T5 implementation changed after freeze")
    return manifest, rows(artifacts["call_plan_sha256"])


def guard_bundle(call: dict[str, Any]) -> CompiledResourceBundle:
    directives = []
    subtype_by_component = dict(call["resource_subtypes"])
    ids = dict(call["candidate_ids_pre_action"])
    for component in COMPONENTS:
        if not call["resource_presented_to_executor"][component]:
            continue
        subtype = str(subtype_by_component[component])
        directives.append(
            CompiledResourceDirective(
                component=component,
                resource_id=str(ids[component]),
                candidate_version="frozen-in-call-plan",
                source_kind="strategy" if component == "RS" else "history",
                owner_id=(
                    None if component == "RS" else str(call["user_id_private_analysis_only"])
                ),
                evidence="frozen in prompt",
                response_instruction="frozen in prompt",
                forbidden_claims=(),
                attribution_required=subtype in {
                    "MS_SESSION_OBSERVATION",
                    "ME_REUSABLE_OUTCOME",
                },
                maximum_support_moves=1,
            )
        )
    return CompiledResourceBundle(
        requested_action_id=str(call["requested_action_id"]),
        directives=tuple(directives),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_1_t5_plan_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_1_t5_execution_v1",
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.1 requires {FORMAL_PYTHON}; got {sys.executable}")
    manifest, calls = verify_freeze(args.plan_dir)
    if not args.run:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "status": "READY",
                    "planned_calls": len(calls),
                    "estimated_usd_upper_bound_proxy": manifest["generation_cost"]["estimated_usd_upper_bound_proxy"],
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
        raise RuntimeError("generator identity differs from frozen T5 plan")
    if not os.environ.get(endpoint.api_key_env, ""):
        raise RuntimeError(f"{endpoint.api_key_env} is not set")
    client = make_client(endpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = args.out_dir / "outcomes_private.jsonl"
    existing_rows = rows(outcomes_path) if outcomes_path.is_file() else []
    existing = {str(row["call_id"]): row for row in existing_rows}
    expected = {str(row["call_id"]) for row in calls}
    if len(existing) != len(existing_rows) or not set(existing) <= expected:
        raise RuntimeError("existing T5 outcomes do not match frozen plan")

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
        usage = require_reported_usage(result.usage, stage="v1_5_v5_1_t5")
        if int(usage["prompt_tokens"]) > int(call["input_token_upper_bound"]):
            raise RuntimeError(f"reported input overrun for {call_id}")
        primary_response = generation.normalize_output(result.text)
        errors = response_guard_errors(
            response=primary_response,
            bundle=guard_bundle(call),
            current_context=str(call["messages"][-1]["content"]),
        )
        fallback_used = bool(errors)
        final_response = (
            deterministic_context_only_fallback(str(call["current_user_text"]))
            if fallback_used
            else primary_response
        )
        realized_action = "M0+R0" if fallback_used else str(call["requested_action_id"])
        saved = {
            "protocol": PROTOCOL,
            "call_id": call_id,
            "domain": call["domain"],
            "partition": call["partition"],
            "state_id": call["state_id"],
            "group_id_private_analysis_only": call["group_id_private_analysis_only"],
            "user_id_private_analysis_only": call["user_id_private_analysis_only"],
            "seed_index": call["seed_index"],
            "seed_hex": call["seed_hex"],
            "seed": call["seed"],
            "requested_action_id": call["requested_action_id"],
            "realized_action_id": realized_action,
            "resource_presented_to_executor": call["resource_presented_to_executor"],
            "resource_subtypes": call["resource_subtypes"],
            "candidate_ids_pre_action": call["candidate_ids_pre_action"],
            "messages_sha256": call["messages_sha256"],
            "primary_response": primary_response,
            "guard_errors": list(errors),
            "fallback_used": fallback_used,
            "final_response": final_response,
            "usage": usage,
            "finish_reason": result.normalized_finish_reason,
            "executor_version": "typed-resource-adapter-v1",
            "generator_version": endpoint.model,
            "completed_at": utc_now(),
            "external_outcome_used_to_change_method": False,
        }
        append_jsonl(outcomes_path, saved)
        existing[call_id] = saved
        completed_now += 1

    ordered = [existing[str(call["call_id"])] for call in calls]
    ordered_path = args.out_dir / "outcomes_ordered_private.jsonl"
    write_jsonl(ordered_path, ordered)
    summary = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_AWAITING_FROZEN_AUTOMATIC_JUDGE_AND_HUMAN_EVALUATION",
        "planned_calls": len(calls),
        "completed_calls": len(ordered),
        "completed_now": completed_now,
        "fallback_calls": sum(bool(row["fallback_used"]) for row in ordered),
        "reported_prompt_tokens": sum(int(row["usage"]["prompt_tokens"]) for row in ordered),
        "reported_completion_tokens": sum(int(row["usage"]["completion_tokens"]) for row in ordered),
        "requested_action_distribution": dict(Counter(str(row["requested_action_id"]) for row in ordered)),
        "realized_action_distribution": dict(Counter(str(row["realized_action_id"]) for row in ordered)),
        "partition_calls": dict(Counter(str(row["partition"]) for row in ordered)),
        "partition_fallbacks": dict(Counter(str(row["partition"]) for row in ordered if row["fallback_used"])),
        "call_plan_sha256": manifest["call_plan_sha256"],
        "ordered_outcomes_sha256": sha256_file(ordered_path),
        "resumable": True,
        "sealed_or_external_outcome_used_to_change_method": False,
    }
    write_json(args.out_dir / "execution_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
