#!/usr/bin/env python3
"""Materialize the frozen V5.2 confirmation features, policies, and calls.

No response, quality/risk/function label, external result, or human judgment is
read.  The output contains paired ON/OFF calls so all four frozen policies can
be replayed on exactly the same stochastic outcomes.
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
from metacom_pm.contracts import MemorySource
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
from metacom_pm.retrieval import source_specific_memory_queries
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_candidate_discovery import discover_final_typed_memory_candidates
from metacom_pm.v1_5_final_candidate_contract import (
    FINAL_FEATURE_NAMES,
    FINAL_FEATURE_CONTRACT_PROTOCOL,
    IndependentFeatureRecord,
    final_model_feature_projection,
    final_rs_model_features,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_source_specific_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import compile_bounded_memory_with_metadata
from metacom_pm.v1_5_v5_2_locked_composer import (
    base_generation_messages,
    build_locked_composition_plan,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.2-content-disjoint-confirmation-plan-v1"
DATA_DIR = ROOT / "data/pm_v1_5_v5_2_confirmation_v1/private"
BLUEPRINT = DATA_DIR / "construction_blueprint.jsonl"
STATES = DATA_DIR / "raw_states.jsonl"
CANDIDATES = DATA_DIR / "candidate_rows_private.jsonl"
CONTRACT = (
    ROOT / "data/pm_v1_5_contracts/v5_2_content_disjoint_confirmation_v1.json"
)
FREEZE_REPORT = (
    ROOT
    / "outputs/pm_v1_5_v5_2_confirmation_freeze_v1/freeze_and_data_quality_report.json"
)
CARDS = ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
FIT_HELPER = ROOT / "scripts/v1_5/27a_materialize_v5_2_locked_fit_plan_v1_5.py"
RUNNER = ROOT / "scripts/v1_5/27b_run_v5_2_locked_fit_generation_v1_5.py"
RS_FEATURE_PROTOCOL = "pm-v1.5-final-rs-transparent-features-v1"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def helper_module() -> Any:
    spec = importlib.util.spec_from_file_location("v52_fit_plan_helper", FIT_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen V5.2 FIT helper")
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
    head = contract["frozen_method"]["heads"][component]
    names = list(head["feature_names"])
    if set(features) != set(names):
        raise RuntimeError(f"{component} feature schema differs from frozen head")
    total = float(head["intercept"])
    for name, mean, scale, coefficient in zip(
        names,
        head["scaler_mean"],
        head["scaler_scale"],
        head["coefficients"],
        strict=True,
    ):
        divisor = float(scale) or 1.0
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


def build_features(
    *,
    state: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    cards_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    component = str(candidate_row["target_component_private_not_model_input"])
    exact = dict(candidate_row["exact_rank1_candidate"])
    if component in {"MP", "MS", "ME"}:
        items, _docs, metadata = compile_bounded_memory_with_metadata(state["user"])
        queries = source_specific_memory_queries(
            str(state["current_user_text"]), list(state["visible_dialogue"]), ""
        )
        discovered = discover_final_typed_memory_candidates(
            queries=queries,
            items=items,
            source_metadata=metadata,
            session_index=int(state["current_session_index"]),
        )
        candidate = discovered[MemorySource(component)]
        if not candidate.selected_items:
            raise RuntimeError(f"candidate disappeared: {state['state_id']}")
        if str(candidate.selected_items[0].memory_id) != str(exact["candidate_id"]):
            raise RuntimeError(f"actual Rank-1 drift: {state['state_id']}")
        observation = build_source_specific_memory_opportunity_observation(
            candidate=candidate,
            catalog_items=items,
            catalog_user_id=str(state["user_id"]),
            current_user_id=str(state["user_id"]),
            current_session_index=int(state["current_session_index"]),
            current_user_text=str(state["current_user_text"]),
            visible_dialogue=list(state["visible_dialogue"]),
            source_metadata=metadata,
            background_action="M0+R0",
        )
        values = final_model_feature_projection(
            component=component, source_features=observation["model_features"]
        )
    else:
        card = cards_by_id.get(str(exact["candidate_id"]))
        if card is None:
            raise RuntimeError(f"RS card missing: {state['state_id']}")
        audit = dict(candidate_row["private_retrieval_audit_not_model_input"])
        values = final_rs_model_features(
            current_user_text=str(state["current_user_text"]),
            visible_dialogue=list(state["visible_dialogue"]),
            observable_flags=dict(audit["observable_flags"]),
            ranked_card={"card_id": str(exact["candidate_id"])},
            bank_card=dict(card),
        )
    if set(values) != set(FINAL_FEATURE_NAMES[component]):
        raise RuntimeError(f"feature schema drift: {state['state_id']}")
    return {key: float(value) for key, value in values.items()}


def _write_once_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    if path.exists():
        old = path.read_text(encoding="utf-8")
        candidate = "".join(canonical_json(value) + "\n" for value in values)
        if old != candidate:
            raise RuntimeError(f"frozen artifact would change: {path}")
        return
    write_jsonl(path, values)


def _write_once_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"frozen artifact would change: {path}")
        return
    write_json(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_2_confirmation_plan_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.2 requires {FORMAL_PYTHON}; got {sys.executable}")
    for path in (BLUEPRINT, STATES, CANDIDATES, CONTRACT, FREEZE_REPORT, CARDS, FIT_HELPER, RUNNER):
        if not path.is_file():
            raise RuntimeError(f"required frozen input missing: {path}")
    contract = read_json(CONTRACT)
    report = read_json(FREEZE_REPORT)
    if contract.get("status") != "FROZEN_BEFORE_ANY_V5_2_CONFIRMATION_GENERATION":
        raise RuntimeError("confirmation contract is not frozen")
    if report.get("status") != "PASS_V5_2_CONTENT_DISJOINT_CONFIRMATION_IDENTITY_FROZEN":
        raise RuntimeError("confirmation data quality gate did not pass")
    if not all(report["checks"].values()):
        raise RuntimeError("confirmation data quality checks are incomplete")
    if contract.get("confirmation_external_or_human_outcomes_read") is not False:
        raise RuntimeError("confirmation contract is not outcome-blind")

    blueprint = {str(row["blueprint_row_id"]): row for row in rows(BLUEPRINT)}
    states = {str(row["state_id"]): row for row in rows(STATES)}
    candidates = rows(CANDIDATES)
    cards_by_id = {str(row["card_id"]): row for row in rows(CARDS)}
    if len(blueprint) != 128 or len(states) != 128 or len(candidates) != 128:
        raise RuntimeError("confirmation requires 128 blueprint/state/candidate rows")
    helper = helper_module()

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])
    pricing = {"input": 0.15, "output": 0.60}

    features_out: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    learned_distribution: Counter[str] = Counter()
    rule_distribution: Counter[str] = Counter()
    for row in sorted(candidates, key=lambda item: str(item["state_id"])):
        state_id = str(row["state_id"])
        source = blueprint[state_id]
        state = states[state_id]
        component = str(row["target_component_private_not_model_input"])
        values = build_features(state=state, candidate_row=row, cards_by_id=cards_by_id)
        feature_record = IndependentFeatureRecord(
            protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
            state_id=state_id,
            component=component,  # type: ignore[arg-type]
            feature_builder_protocol=(
                RS_FEATURE_PROTOCOL
                if component == "RS"
                else SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
            ),
            model_features=values,
        )
        probability = learned_probability(
            component=component, features=values, contract=contract
        )
        threshold = float(
            contract["frozen_method"]["heads"][component]["threshold"]
        )
        learned_on = probability >= threshold
        rule_on = transparent_rule(component, values)
        learned_distribution[f"{component}:{'ON' if learned_on else 'OFF'}"] += 1
        rule_distribution[f"{component}:{'ON' if rule_on else 'OFF'}"] += 1
        features_out.append(
            {
                **asdict(feature_record),
                "counterfactual_group_id_private_analysis_only": str(
                    source["counterfactual_group_id"]
                ),
                "semantic_family_private_analysis_only": str(source["logic_family"]),
                "candidate_text_sha256": str(
                    row["exact_rank1_candidate"]["candidate_text_sha256"]
                ),
                "response_label_external_or_human_outcome_read": False,
            }
        )

        candidate, strategy_guidance = helper.parse_candidate(
            row, strategy_by_id=cards_by_id
        )
        treatment = source["effect_treatments"]
        on_action = str(treatment["treatment"])
        on_plan = build_locked_composition_plan(
            requested_action_id=on_action,
            candidates={component: candidate},
            strategy_prompt_guidance=strategy_guidance,
        )
        off_plan = build_locked_composition_plan(
            requested_action_id="M0+R0", candidates={}
        )
        group_id = str(source["counterfactual_group_id"])
        bindings.append(
            {
                "protocol": PROTOCOL,
                "state_id": state_id,
                "counterfactual_group_id": group_id,
                "component": component,
                "candidate": asdict(candidate),
                "learned_probability": probability,
                "learned_threshold": threshold,
                "policy_decisions": {
                    "always_off": False,
                    "component_fixed_high": True,
                    "transparent_rule": rule_on,
                    "learned_pm": learned_on,
                },
                "response_label_external_or_human_outcome_read": False,
            }
        )
        context = helper.current_context(row)
        for seed_hex in treatment["generation_seeds"]:
            for arm, action, plan in (
                ("OFF", "M0+R0", off_plan),
                ("ON", on_action, on_plan),
            ):
                messages = base_generation_messages(current_context=context, plan=plan)
                message_blob = canonical_json(messages)
                if arm == "ON" and component in {"MS", "ME"}:
                    literal = plan.locked_clauses[0].literal_source_span
                    if literal and literal in message_blob:
                        raise RuntimeError(f"history exposed to generator: {state_id}")
                call_id = "v52confirm_" + stable_hex(
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
                        "candidate_owner_id_pre_action": str(
                            row["user_id_private_not_model_input"]
                        ),
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
                        "messages_sha256": sha256_text(message_blob),
                        "composition_plan": asdict(plan),
                        "input_token_upper_bound": conservative_token_bound(
                            message_blob, safety_factor=safety_factor
                        ),
                        "output_token_cap": int(generation.max_output_tokens),
                        "assigned_executor_version": "v5.2-backend-locked-composer-v1",
                        "assigned_generator_version": endpoint.model,
                        "selection_or_prompt_uses_confirmation_human_external_outcome": False,
                    }
                )

    calls.sort(key=lambda item: str(item["call_id"]))
    if len(features_out) != 128 or len(bindings) != 128 or len(calls) != 512:
        raise RuntimeError("confirmation output shape mismatch")
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in calls:
        pairs.setdefault((str(call["state_id"]), str(call["seed_hex"])), []).append(call)
    if len(pairs) != 256 or any(
        {str(item["arm"]) for item in pair} != {"ON", "OFF"}
        for pair in pairs.values()
    ):
        raise RuntimeError("confirmation ON/OFF pairing failed")
    learned_on = sum(
        bool(row["policy_decisions"]["learned_pm"]) for row in bindings
    )
    if not 0.10 <= learned_on / len(bindings) <= 0.90:
        raise RuntimeError("frozen learned policy is degenerate on the new confirmation")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / "feature_rows_private.jsonl"
    binding_path = args.out_dir / "state_policy_bindings_private.jsonl"
    call_path = args.out_dir / "call_plan_private.jsonl"
    _write_once_jsonl(feature_path, features_out)
    _write_once_jsonl(binding_path, bindings)
    _write_once_jsonl(call_path, calls)
    total_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    total_output = sum(int(row["output_token_cap"]) for row in calls)
    estimated = (
        total_input / 1_000_000 * pricing["input"]
        + total_output / 1_000_000 * pricing["output"]
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_V5_2_CONTENT_DISJOINT_CONFIRMATION_EXECUTION",
        "method_version": "V1_5_V5_2_BACKEND_LOCKED_EXECUTOR_AND_FROZEN_PM",
        "states": len(bindings),
        "counterfactual_groups": len(
            {str(row["counterfactual_group_id"]) for row in bindings}
        ),
        "planned_calls": len(calls),
        "generator": endpoint.model,
        "temperature": float(generation.temperature),
        "learned_decision_distribution": dict(sorted(learned_distribution.items())),
        "transparent_rule_distribution": dict(sorted(rule_distribution.items())),
        "learned_requested_on_fraction": learned_on / len(bindings),
        "arms": dict(Counter(str(row["arm"]) for row in calls)),
        "input_token_upper_bound_total": total_input,
        "output_token_cap_total": total_output,
        "estimated_generation_usd_upper_bound_proxy": round(estimated, 8),
        "pricing_usd_per_mtok_proxy": pricing,
        "feature_rows_sha256": sha256_file(feature_path),
        "state_policy_bindings_sha256": sha256_file(binding_path),
        "call_plan_sha256": sha256_file(call_path),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (BLUEPRINT, STATES, CANDIDATES, CONTRACT, FREEZE_REPORT, CARDS)
        },
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                FIT_HELPER,
                Path(__file__).resolve(),
                ROOT / "src/metacom_pm/v1_5_v5_2_atomic_memory.py",
                ROOT / "src/metacom_pm/v1_5_v5_2_locked_composer.py",
                ROOT / "src/metacom_pm/v1_5_itt_policy.py",
            )
        },
        "checks": {
            "data_quality_freeze_passed": True,
            "128_states_64_groups_32_each_component": True,
            "512_unique_same_seed_paired_calls": True,
            "all_four_policies_frozen_before_generation": True,
            "learned_policy_non_degenerate": True,
            "same_v5_2_locked_executor_as_fit": True,
            "ms_me_literal_history_absent_from_generator_messages": True,
            "response_label_human_and_external_outcomes_read_zero": True,
        },
        "api_calls": 0,
        "human_or_generated_outcomes_read": 0,
        "external_outcomes_read": 0,
        "post_confirmation_model_feature_threshold_executor_or_prompt_change_allowed": False,
        "formal_python_executable": str(FORMAL_PYTHON),
    }
    _write_once_json(args.out_dir / "freeze_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "planned_calls": manifest["planned_calls"],
                "learned_on_fraction": manifest["learned_requested_on_fraction"],
                "estimated_usd_upper_bound_proxy": manifest[
                    "estimated_generation_usd_upper_bound_proxy"
                ],
                "api_calls": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
