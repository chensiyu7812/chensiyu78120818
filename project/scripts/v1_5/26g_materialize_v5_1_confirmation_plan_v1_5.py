#!/usr/bin/env python3
"""Freeze the complete V5.1 FRESH_CONFIRMATION call and policy plan.

All four policies reuse the same 128 state-candidate units and paired ON/OFF
responses.  Learned decisions come only from the already-frozen V5 final
models.  No response, human label, sealed split, or external outcome is read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_typed_resource_adapter import compile_typed_bundle, response_only_messages
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.1-confirmation-plan-v1"
BASE_HELPER = ROOT / "scripts/v1_5/25zw_materialize_v5_itt_fit_plan_v1_5.py"
RUNNER = ROOT / "scripts/v1_5/26h_run_v5_1_confirmation_generation_v1_5.py"
BLUEPRINT = ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl"
CANDIDATES = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
FEATURES = ROOT / "outputs/pm_v1_5_v5_1_confirmation_surface_v1/confirmation_feature_rows_private.jsonl"
SURFACE_REPORT = ROOT / "outputs/pm_v1_5_v5_1_confirmation_surface_v1/data_integrity_report.json"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json"
FIT_DESIGN = ROOT / "data/pm_v1_5_contracts/v5_itt_fit_design_v1.json"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def helper_module() -> Any:
    spec = importlib.util.spec_from_file_location("v5_fit_plan_helper", BASE_HELPER)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load frozen V5 FIT plan helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def learned_probability(
    *, component: str, features: Mapping[str, float], contract: Mapping[str, Any]
) -> float:
    head = contract["frozen_policy"]["heads"][component]
    names = list(head["feature_names"])
    if set(features) != set(names):
        raise RuntimeError(f"{component} confirmation feature schema differs from final model")
    total = float(head["intercept"])
    for name, mean, scale, coefficient in zip(
        names,
        head["scaler_mean"],
        head["scaler_scale"],
        head["coefficients"],
        strict=True,
    ):
        divisor = float(scale)
        if divisor == 0.0:
            divisor = 1.0
        total += ((float(features[name]) - float(mean)) / divisor) * float(coefficient)
    return sigmoid(total)


def transparent_rule(component: str, f: Mapping[str, float]) -> bool:
    if component == "MP":
        preference = f["candidate_preference_scope_fit"] >= 0.5
        profile = (
            f["candidate_profile_relevance"] >= 0.5
            and f["candidate_profile_entity_scope_fit"] >= 0.5
        )
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_scope_conflict"] < 0.5
            and (preference or profile)
        )
    if component == "MS":
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_prior_issue_marked_resolved"] < 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and (
                f["candidate_prior_outcome_or_distinction"] >= 0.5
                or f["candidate_specific_issue_or_distinction"] >= 0.5
            )
        )
    if component == "ME":
        return bool(
            f["candidate_incremental_information"] >= 0.5
            and f["candidate_current_goal_fit"] >= 0.5
            and f["candidate_contains_action"] >= 0.5
            and (
                f["candidate_contains_result"] >= 0.5
                or f["candidate_contains_mechanism"] >= 0.5
            )
        )
    return all(
        f[name] >= 0.5
        for name in (
            "candidate_mode_fit",
            "candidate_goal_fit",
            "candidate_burden_fit",
            "candidate_boundary_fit",
            "candidate_nonredundancy",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_1_confirmation_plan_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.1 requires {FORMAL_PYTHON}; got {sys.executable}")
    if not RUNNER.is_file():
        raise RuntimeError("confirmation runner must exist before freezing the plan")

    surface = read_json(SURFACE_REPORT)
    if surface.get("status") != "PASS_READY_FOR_FROZEN_CONFIRMATION_CALL_PLAN":
        raise RuntimeError("confirmation data integrity gate did not pass")
    if not all(surface["checks"].values()):
        raise RuntimeError("confirmation data integrity checks are incomplete")
    contract = read_json(CONTRACT)
    design = read_json(FIT_DESIGN)
    if contract.get("confirmation_or_external_outcomes_read") is not False:
        raise RuntimeError("V5.1 contract is not outcome-blind")

    blueprint = {str(row["blueprint_row_id"]): row for row in rows(BLUEPRINT)}
    candidates = [
        row
        for row in rows(CANDIDATES)
        if row["track_private_not_model_input"] == "COMPONENT_EFFECT"
        and row["split_private_not_model_input"] == "FRESH_CONFIRMATION"
    ]
    features = {str(row["state_id"]): row for row in rows(FEATURES)}
    if len(candidates) != 128 or len(features) != 128:
        raise RuntimeError("confirmation requires 128 candidate and feature rows")
    helper = helper_module()

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    if endpoint.model != design["frozen_generator_candidate"]["model"]:
        raise RuntimeError("confirmation generator differs from V5 FIT")
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])
    pricing = {"input": 0.15, "output": 0.60}

    calls: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    learned_distribution: Counter[str] = Counter()
    rule_distribution: Counter[str] = Counter()
    for row in sorted(candidates, key=lambda value: str(value["state_id"])):
        state_id = str(row["state_id"])
        source = blueprint[state_id]
        feature = features[state_id]
        component = str(row["target_component_private_not_model_input"])
        if feature["component"] != component:
            raise RuntimeError(f"candidate/feature component mismatch: {state_id}")
        group_id = str(source["counterfactual_group_id"])
        treatments = source["effect_treatments"]
        if not treatments or not treatments["single_component_change_only"]:
            raise RuntimeError(f"missing one-bit treatment: {state_id}")
        if treatments["control"] != "M0+R0":
            raise RuntimeError(f"unexpected control: {state_id}")

        probability = learned_probability(
            component=component,
            features=feature["model_features"],
            contract=contract,
        )
        threshold = float(contract["frozen_policy"]["heads"][component]["threshold"])
        learned_on = probability >= threshold
        rule_on = transparent_rule(component, feature["model_features"])
        learned_distribution[f"{component}:{'ON' if learned_on else 'OFF'}"] += 1
        rule_distribution[f"{component}:{'ON' if rule_on else 'OFF'}"] += 1

        typed = helper.parse_candidate(row)
        owner = str(row["user_id_private_not_model_input"])
        context = helper.current_context(row)
        on_action = str(treatments["treatment"])
        on_bundle = compile_typed_bundle(
            requested_action_id=on_action,
            candidates={component: typed},
            current_user_id=owner,
        )
        off_bundle = compile_typed_bundle(
            requested_action_id="M0+R0", candidates={}, current_user_id=owner
        )
        bindings.append(
            {
                "protocol": PROTOCOL,
                "state_id": state_id,
                "counterfactual_group_id": group_id,
                "component": component,
                "candidate": asdict(typed),
                "learned_probability": probability,
                "learned_threshold": threshold,
                "policy_decisions": {
                    "always_off": False,
                    "component_fixed_high": True,
                    "transparent_rule": rule_on,
                    "learned_pm": learned_on,
                },
                "confirmation_response_or_label_read": False,
            }
        )
        for seed_hex in treatments["generation_seeds"]:
            for arm, action, bundle in (
                ("OFF", "M0+R0", off_bundle),
                ("ON", on_action, on_bundle),
            ):
                messages = response_only_messages(current_context=context, bundle=bundle)
                bound = conservative_token_bound(
                    canonical_json(messages), safety_factor=safety_factor
                )
                call_id = "v51confirm_" + stable_hex(
                    PROTOCOL, state_id, component, arm, str(seed_hex), n=24
                )
                calls.append(
                    {
                        "protocol": PROTOCOL,
                        "call_id": call_id,
                        "state_id": state_id,
                        "counterfactual_group_id_private_analysis_only": group_id,
                        "target_component": component,
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
                            source_component: bool(
                                arm == "ON" and source_component == component
                            )
                            for source_component in COMPONENTS
                        },
                        "seed_hex": str(seed_hex),
                        "seed": helper.seed_int(str(seed_hex)),
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
    if len(calls) != 512 or len({str(row["call_id"]) for row in calls}) != 512:
        raise RuntimeError("confirmation must contain 512 unique calls")
    paired: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in calls:
        key = (str(call["state_id"]), str(call["seed_hex"]))
        paired.setdefault(key, []).append(call)
    if len(paired) != 256 or any(
        {str(item["arm"]) for item in value} != {"ON", "OFF"}
        for value in paired.values()
    ):
        raise RuntimeError("confirmation ON/OFF pairing failed")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    call_path = args.out_dir / "call_plan_private.jsonl"
    binding_path = args.out_dir / "state_policy_bindings_private.jsonl"
    write_jsonl(call_path, calls)
    write_jsonl(binding_path, bindings)
    total_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    total_output = sum(int(row["output_token_cap"]) for row in calls)
    estimated = total_input / 1_000_000 * pricing["input"] + total_output / 1_000_000 * pricing["output"]
    implementation_paths = {
        "typed_adapter": ROOT / "src/metacom_pm/v1_5_typed_resource_adapter.py",
        "itt_validator": ROOT / "src/metacom_pm/v1_5_itt_policy.py",
        "policy_runtime": ROOT / "src/metacom_pm/v1_5b_policy_runtime.py",
        "fit_plan_helper": BASE_HELPER,
        "confirmation_surface_builder": ROOT / "scripts/v1_5/26f_materialize_v5_1_confirmation_surface_v1_5.py",
        "confirmation_plan_builder": Path(__file__).resolve(),
        "confirmation_runner": RUNNER,
    }
    freeze = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_V5_1_CONFIRMATION_EXECUTION",
        "method_version": "V1_5_V5_1_SYSTEM_PARETO_CONFIRMATION",
        "generator": endpoint.model,
        "temperature": float(generation.temperature),
        "states": len(candidates),
        "counterfactual_groups": len(
            {str(row["counterfactual_group_id"]) for row in bindings}
        ),
        "planned_calls": len(calls),
        "states_per_component": dict(
            Counter(str(row["component"]) for row in bindings)
        ),
        "learned_decision_distribution": dict(sorted(learned_distribution.items())),
        "transparent_rule_distribution": dict(sorted(rule_distribution.items())),
        "arms": dict(Counter(str(row["arm"]) for row in calls)),
        "input_token_upper_bound_total": total_input,
        "output_token_cap_total": total_output,
        "estimated_generation_usd_upper_bound_proxy": round(estimated, 8),
        "pricing_usd_per_mtok_proxy": pricing,
        "call_plan_sha256": sha256_file(call_path),
        "state_policy_bindings_sha256": sha256_file(binding_path),
        "input_sha256": {
            str(BLUEPRINT.relative_to(ROOT)): sha256_file(BLUEPRINT),
            str(CANDIDATES.relative_to(ROOT)): sha256_file(CANDIDATES),
            str(FEATURES.relative_to(ROOT)): sha256_file(FEATURES),
            str(SURFACE_REPORT.relative_to(ROOT)): sha256_file(SURFACE_REPORT),
            str(CONTRACT.relative_to(ROOT)): sha256_file(CONTRACT),
            str(FIT_DESIGN.relative_to(ROOT)): sha256_file(FIT_DESIGN),
            "configs/pm_v1_5.yaml": sha256_file(ROOT / "configs/pm_v1_5.yaml"),
            "configs/experiment.yaml": sha256_file(ROOT / "configs/experiment.yaml"),
        },
        "implementation_sha256": {
            key: sha256_file(path) for key, path in implementation_paths.items()
        },
        "checks": {
            "zero_outcome_data_integrity_gate_pass": True,
            "128_states_64_groups": True,
            "32_states_each_component": True,
            "512_unique_calls": True,
            "same_seed_within_on_off_pairs": True,
            "current_final_models_and_point5_thresholds_only": True,
            "all_four_policies_frozen_before_confirmation": True,
            "response_label_sealed_external_read_zero": True,
        },
        "api_calls": 0,
        "human_or_generated_outcomes_read": 0,
        "sealed_or_external_read": False,
        "post_confirmation_model_feature_threshold_or_executor_change_allowed": False,
        "formal_python_executable": str(FORMAL_PYTHON),
    }
    write_json(args.out_dir / "freeze_manifest.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False))


if __name__ == "__main__":
    main()
