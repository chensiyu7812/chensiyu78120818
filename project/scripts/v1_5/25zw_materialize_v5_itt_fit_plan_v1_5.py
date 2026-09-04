#!/usr/bin/env python3
"""Materialize the complete V5 ITT FIT call plan without API calls."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_typed_resource_adapter import (
    TypedResourceCandidate,
    compile_typed_bundle,
    response_only_messages,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5-itt-fit-plan-v1"
BLUEPRINT = ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
CANDIDATES = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
DESIGN = ROOT / "data/pm_v1_5_contracts/v5_itt_fit_design_v1.json"
RUNNER = ROOT / "scripts/v1_5/25zx_run_v5_itt_fit_generation_v1_5.py"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def current_context(row: dict[str, Any]) -> str:
    lines = ["Visible current conversation:"]
    for turn in row["visible_dialogue"]:
        lines.append(f"{str(turn['role']).capitalize()}: {str(turn['content']).strip()}")
    lines.append(f"User: {str(row['current_user_text']).strip()}")
    return "\n".join(lines)


def parse_candidate(row: dict[str, Any]) -> TypedResourceCandidate:
    exact = row["exact_rank1_candidate"]
    component = str(row["target_component_private_not_model_input"])
    hint = str(exact["compiler_subtype_hint"])
    text = str(exact["candidate_text"]).strip()
    common = {
        "component": component,
        "resource_id": str(exact["candidate_id"]),
        "candidate_version": str(exact["candidate_text_sha256"]),
    }
    owner = str(row["structured_candidate_metadata"]["candidate_owner_id"])
    if component == "MP" and hint == "MP_PREFERENCE":
        return TypedResourceCandidate(
            **common,
            subtype="MP_PREFERENCE",
            source_kind="profile",
            owner_id=owner,
            preference=text.split(":", 1)[-1].strip(),
        )
    if component == "MP" and hint == "MP_PROFILE":
        return TypedResourceCandidate(
            **common,
            subtype="MP_PROFILE",
            source_kind="profile",
            owner_id=owner,
            profile_fact=text.split(":", 1)[-1].strip(),
        )
    if component == "MS" and hint == "MS_SESSION":
        age = int(exact["candidate_age_sessions"])
        return TypedResourceCandidate(
            **common,
            subtype="MS_SESSION_OBSERVATION",
            source_kind="session",
            owner_id=owner,
            strictly_prior=True,
            age_sessions=age,
            prior_observation=text,
        )
    if component == "ME" and hint == "ME_REUSABLE_OUTCOME":
        age = int(exact["candidate_age_sessions"])
        match = re.match(r"^For .+? issue, I (.+?), and (.+?)\.?$", text)
        if match:
            past_action = match.group(1).strip()
            observed_outcome = re.sub(r"^it\s+", "", match.group(2).strip())
        else:
            helped = re.match(
                r"^For .+? issue, I (.+?); what helped was (.+?)\.?$", text
            )
            switched = re.match(
                r"^For .+? issue, I tried a long list; switching to one item "
                r"(.+?)\.?$",
                text,
            )
            if helped:
                past_action = (
                    f"{helped.group(1).strip()} and used "
                    f"{helped.group(2).strip()}"
                )
                observed_outcome = "that bounded approach was described as helpful"
            elif switched:
                past_action = "switched from a long list to one item"
                observed_outcome = switched.group(1).strip()
            else:
                raise RuntimeError(f"cannot parse ME action/outcome: {row['state_id']}")
        return TypedResourceCandidate(
            **common,
            subtype="ME_REUSABLE_OUTCOME",
            source_kind="event",
            owner_id=owner,
            strictly_prior=True,
            age_sessions=age,
            past_action=past_action,
            observed_outcome=observed_outcome,
        )
    if component == "RS" and hint == "RS_ATOMIC_MOVE":
        fields: dict[str, str] = {}
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        return TypedResourceCandidate(
            **common,
            subtype="RS_ATOMIC_MOVE",
            source_kind="strategy",
            support_move=fields.get("support_move", ""),
            when_to_use=fields.get("when_to_use", ""),
            when_not_to_use=fields.get("when_not_to_use", ""),
        )
    raise RuntimeError(f"unsupported component/subtype: {component}/{hint}")


def seed_int(seed_hex: str) -> int:
    value = int(seed_hex, 16) % 2_147_483_647
    return value or 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_itt_fit_plan_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5 requires {FORMAL_PYTHON}; got {sys.executable}")
    if not RUNNER.is_file():
        raise RuntimeError("V5 FIT runner must exist before the call plan is frozen")

    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    blueprint = {str(row["blueprint_row_id"]): row for row in rows(BLUEPRINT)}
    candidates = [
        row
        for row in rows(CANDIDATES)
        if row["track_private_not_model_input"] == "COMPONENT_EFFECT"
        and row["split_private_not_model_input"] == "EFFECT_FIT"
    ]
    expected = Counter({component: 64 for component in COMPONENTS})
    observed = Counter(str(row["target_component_private_not_model_input"]) for row in candidates)
    if observed != expected or len(candidates) != 256:
        raise RuntimeError(f"FIT candidate shape mismatch: {observed}")

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    if endpoint.model != design["frozen_generator_candidate"]["model"]:
        raise RuntimeError("configured generator does not match V5 design")
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])
    pricing = {"input": 0.15, "output": 0.60}

    calls: list[dict[str, Any]] = []
    state_bindings: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: str(item["state_id"])):
        state_id = str(row["state_id"])
        source = blueprint[state_id]
        target = str(row["target_component_private_not_model_input"])
        treatments = source["effect_treatments"]
        if treatments is None or not treatments["single_component_change_only"]:
            raise RuntimeError(f"missing one-bit treatment: {state_id}")
        if str(treatments["control"]) != "M0+R0":
            raise RuntimeError(f"unexpected control action: {state_id}")
        typed = parse_candidate(row)
        owner = str(row["user_id_private_not_model_input"])
        context = current_context(row)
        on_bundle = compile_typed_bundle(
            requested_action_id=str(treatments["treatment"]),
            candidates={target: typed},
            current_user_id=owner,
        )
        off_bundle = compile_typed_bundle(
            requested_action_id="M0+R0", candidates={}, current_user_id=owner
        )
        state_bindings.append(
            {
                "state_id": state_id,
                "group_id": str(row["group_id_private_not_model_input"]),
                "target_component": target,
                "candidate": asdict(typed),
                "control_action_id": "M0+R0",
                "treatment_action_id": str(treatments["treatment"]),
                "generation_seed_hex": list(treatments["generation_seeds"]),
            }
        )
        for seed_hex in treatments["generation_seeds"]:
            for arm, action, bundle in (
                ("OFF", "M0+R0", off_bundle),
                ("ON", str(treatments["treatment"]), on_bundle),
            ):
                messages = response_only_messages(current_context=context, bundle=bundle)
                bound = conservative_token_bound(
                    canonical_json(messages), safety_factor=safety_factor
                )
                call_id = "v5fit_" + stable_hex(
                    PROTOCOL, state_id, target, arm, str(seed_hex), n=24
                )
                calls.append(
                    {
                        "protocol": PROTOCOL,
                        "call_id": call_id,
                        "state_id": state_id,
                        "group_id_private_analysis_only": str(
                            row["group_id_private_not_model_input"]
                        ),
                        "target_component": target,
                        "arm": arm,
                        "requested_action_id": action,
                        "candidate_id_pre_action": str(
                            row["exact_rank1_candidate"]["candidate_id"]
                        ),
                        "candidate_version_pre_action": str(
                            row["exact_rank1_candidate"]["candidate_text_sha256"]
                        ),
                        "candidate_owner_id_pre_action": owner,
                        "resource_presented_to_executor": {
                            component: bool(arm == "ON" and component == target)
                            for component in COMPONENTS
                        },
                        "seed_hex": str(seed_hex),
                        "seed": seed_int(str(seed_hex)),
                        "current_user_text": str(row["current_user_text"]),
                        "messages": messages,
                        "messages_sha256": sha256_text(canonical_json(messages)),
                        "input_token_upper_bound": bound,
                        "output_token_cap": int(generation.max_output_tokens),
                        "assigned_executor_version": "typed-resource-adapter-v1",
                        "assigned_generator_version": endpoint.model,
                    }
                )

    calls.sort(key=lambda row: str(row["call_id"]))
    if len(calls) != 1024 or len({str(row["call_id"]) for row in calls}) != 1024:
        raise RuntimeError("V5 FIT must contain 1024 unique response calls")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in calls:
        grouped.setdefault((str(call["state_id"]), str(call["seed_hex"])), []).append(call)
    if any({str(item["arm"]) for item in group} != {"ON", "OFF"} for group in grouped.values()):
        raise RuntimeError("every state/seed requires paired ON and OFF calls")
    if any(len({str(item["seed"]) for item in group}) != 1 for group in grouped.values()):
        raise RuntimeError("paired arms must use the same numeric seed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "call_plan_private.jsonl"
    bindings_path = args.out_dir / "state_candidate_bindings_private.jsonl"
    write_jsonl(plan_path, calls)
    write_jsonl(bindings_path, state_bindings)
    total_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    total_output = sum(int(row["output_token_cap"]) for row in calls)
    estimated = total_input / 1_000_000 * pricing["input"] + total_output / 1_000_000 * pricing["output"]
    implementation_paths = {
        "typed_adapter": ROOT / "src/metacom_pm/v1_5_typed_resource_adapter.py",
        "itt_validator": ROOT / "src/metacom_pm/v1_5_itt_policy.py",
        "policy_runtime": ROOT / "src/metacom_pm/v1_5b_policy_runtime.py",
        "plan_builder": Path(__file__).resolve(),
        "generation_runner": RUNNER,
    }
    freeze = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_V5_ITT_FIT_EXECUTION",
        "method_version": "V1_5_V5_ITT_SEPARATED_POLICY",
        "generator": endpoint.model,
        "temperature": float(generation.temperature),
        "response_schema": "plain_text_only",
        "planned_calls": len(calls),
        "states": len(candidates),
        "states_per_component": dict(observed),
        "calls_per_component": dict(Counter(str(row["target_component"]) for row in calls)),
        "arms": dict(Counter(str(row["arm"]) for row in calls)),
        "seeds_per_arm": 2,
        "input_token_upper_bound_total": total_input,
        "output_token_cap_total": total_output,
        "estimated_generation_usd_upper_bound_proxy": round(estimated, 8),
        "pricing_usd_per_mtok_proxy": pricing,
        "call_plan_sha256": sha256_file(plan_path),
        "state_candidate_bindings_sha256": sha256_file(bindings_path),
        "input_sha256": {
            str(BLUEPRINT.relative_to(ROOT)): sha256_file(BLUEPRINT),
            str(CANDIDATES.relative_to(ROOT)): sha256_file(CANDIDATES),
            str(DESIGN.relative_to(ROOT)): sha256_file(DESIGN),
            "configs/pm_v1_5.yaml": sha256_file(ROOT / "configs/pm_v1_5.yaml"),
            "configs/experiment.yaml": sha256_file(ROOT / "configs/experiment.yaml"),
        },
        "implementation_sha256": {
            key: sha256_file(path) for key, path in implementation_paths.items()
        },
        "checks": {
            "256_independent_fit_states": True,
            "64_states_each_component": True,
            "1024_unique_calls": True,
            "same_seed_within_each_on_off_pair": True,
            "single_requested_component_change_only": True,
            "all_exact_rank1_candidates_typed_and_owner_bound": True,
            "m0_r0_control_contains_no_resource": True,
            "no_generator_self_report_schema": True,
            "no_response_quality_risk_or_external_outcome_read": True,
        },
        "api_calls": 0,
        "human_labels_read": 0,
        "external_lockbox_read": False,
        "formal_python_executable": str(FORMAL_PYTHON),
    }
    write_json(args.out_dir / "freeze_manifest.json", freeze)
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "planned_calls": freeze["planned_calls"],
                "estimated_usd_upper_bound_proxy": freeze[
                    "estimated_generation_usd_upper_bound_proxy"
                ],
                "api_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
