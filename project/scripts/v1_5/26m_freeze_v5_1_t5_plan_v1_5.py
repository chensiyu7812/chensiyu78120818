#!/usr/bin/env python3
"""Freeze the complete V5.1 T5 routing, generation and evaluation plan.

The script consumes only the response-free T5 surfaces.  It freezes five
policy comparators, two generation seeds, exact prompt messages, cost bounds,
the independent LLM-judge design and the one-shot stratified human-review
sample.  It never reads a generated reply or any human/LLM outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import ALL_ACTION_IDS
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
from metacom_pm.v1_5_typed_resource_adapter import (
    TypedResourceCandidate,
    compile_typed_bundle,
    response_only_messages,
)
from metacom_pm.v1_5b_policy_runtime import (
    COMPONENTS,
    action_component_bits,
    compile_component_bits,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v5.1-t5-frozen-plan-v1"
SURFACE_DIR = ROOT / "outputs/pm_v1_5_v5_1_t5_surfaces_v1"
SURFACES = SURFACE_DIR / "t5_state_surfaces_private.jsonl"
SURFACE_REPORT = SURFACE_DIR / "surface_gate_report.json"
CONTRACT = ROOT / "data/pm_v1_5_contracts/v5_1_system_pareto_confirmation_v1.json"
RUNNER = ROOT / "scripts/v1_5/26n_run_v5_1_t5_generation_v1_5.py"
POLICIES = (
    "always_off",
    "fixed_high_eligible",
    "transparent_rule",
    "cost_matched_fixed",
    "learned_pm",
)
COMPARATORS = (
    "fixed_high_eligible",
    "always_off",
    "transparent_rule",
    "cost_matched_fixed",
)
SEED_PROTOCOL = "pm-v1.5-v5.1-t5-two-frozen-seeds-v1"
HUMAN_SAMPLE_PROTOCOL = "pm-v1.5-v5.1-t5-stratified-human-final-v1"
LLM_JUDGE_PROTOCOL = "pm-v1.5-v5.1-t5-independent-judge-v1"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def seed_int(seed_hex: str) -> int:
    value = int(seed_hex, 16) % 2_147_483_647
    return value or 1


def current_context(row: Mapping[str, Any]) -> str:
    lines = ["Visible current conversation:"]
    for turn in row["visible_dialogue"]:
        lines.append(f"{str(turn['role']).capitalize()}: {str(turn['content']).strip()}")
    lines.append(f"User: {str(row['current_user_text']).strip()}")
    return "\n".join(lines)


def executable_candidates(row: Mapping[str, Any]) -> dict[str, TypedResourceCandidate]:
    result: dict[str, TypedResourceCandidate] = {}
    for component in COMPONENTS:
        item = row["components"][component]
        raw = item.get("typed_candidate")
        if item["structurally_executable"]:
            if raw is None:
                raise RuntimeError(f"executable {component} has no typed candidate")
            result[component] = TypedResourceCandidate(**dict(raw))
        # A candidate can be mechanically parseable yet still fail a separate
        # grounding/currentness hard gate.  Keep it in the audit surface but do
        # not make it executable or inject it.
    return result


def project_bits(bits: Mapping[str, bool], executable: Mapping[str, Any]) -> dict[str, bool]:
    return {
        component: bool(bits[component] and component in executable)
        for component in COMPONENTS
    }


def requested_policy_bits(row: Mapping[str, Any], policy: str) -> dict[str, bool]:
    if policy == "always_off":
        return {component: False for component in COMPONENTS}
    if policy == "fixed_high_eligible":
        return {
            component: bool(row["components"][component]["structurally_executable"])
            for component in COMPONENTS
        }
    if policy == "transparent_rule":
        return {
            component: bool(
                row["components"][component]["transparent_rule_requested_on_before_feasibility"]
            )
            for component in COMPONENTS
        }
    if policy == "learned_pm":
        return {
            component: bool(
                row["components"][component]["learned_requested_on_before_feasibility"]
            )
            for component in COMPONENTS
        }
    raise ValueError(f"unsupported direct policy: {policy}")


def action_messages_and_bound(
    *,
    row: Mapping[str, Any],
    requested_bits: Mapping[str, bool],
    safety_factor: float,
) -> tuple[str, dict[str, bool], list[dict[str, str]], int, dict[str, TypedResourceCandidate]]:
    executable = executable_candidates(row)
    feasible_bits = project_bits(requested_bits, executable)
    action = compile_component_bits(feasible_bits)
    selected = {
        component: executable[component]
        for component in COMPONENTS
        if feasible_bits[component]
    }
    bundle = compile_typed_bundle(
        requested_action_id=action,
        candidates=selected,
        current_user_id=str(row["user_id_private_analysis_only"]),
    )
    messages = response_only_messages(
        current_context=current_context(row), bundle=bundle
    )
    bound = conservative_token_bound(
        canonical_json(messages), safety_factor=safety_factor
    )
    return action, feasible_bits, messages, bound, selected


def cost_selection_populations(surface_rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in surface_rows:
        partition = str(row["partition"])
        if partition == "sealed_internal_test":
            result["sealed_internal"].append(row)
        elif partition == "esconv_corrected_test_panel":
            result["esconv"].append(row)
        elif partition == "evoemo_qualification_p7_p12":
            result["evoemo_qualification"].append(row)
    return result


def select_cost_matched_actions(
    *, surface_rows: list[dict[str, Any]], safety_factor: float
) -> tuple[dict[str, str], dict[str, Any]]:
    selections: dict[str, str] = {}
    audit: dict[str, Any] = {}
    for population, selected in cost_selection_populations(surface_rows).items():
        learned_costs = []
        for row in selected:
            _, _, _, bound, _ = action_messages_and_bound(
                row=row,
                requested_bits=requested_policy_bits(row, "learned_pm"),
                safety_factor=safety_factor,
            )
            learned_costs.append(bound)
        learned_mean = sum(learned_costs) / len(learned_costs)
        actions = ("M0+R0", "M0+RS") if population == "esconv" else ALL_ACTION_IDS
        summaries: dict[str, Any] = {}
        for fixed_action in actions:
            fixed_bits = action_component_bits(fixed_action)
            costs = []
            realized = Counter()
            for row in selected:
                action, _, _, bound, _ = action_messages_and_bound(
                    row=row,
                    requested_bits=fixed_bits,
                    safety_factor=safety_factor,
                )
                costs.append(bound)
                realized[action] += 1
            mean = sum(costs) / len(costs)
            summaries[fixed_action] = {
                "mean_input_token_upper_bound": mean,
                "absolute_difference_from_learned": abs(mean - learned_mean),
                "relative_difference_from_learned": abs(mean - learned_mean) / max(1.0, learned_mean),
                "realized_action_distribution": dict(sorted(realized.items())),
            }
        chosen = min(
            summaries,
            key=lambda action: (
                summaries[action]["absolute_difference_from_learned"],
                summaries[action]["mean_input_token_upper_bound"],
                action,
            ),
        )
        selections[population] = chosen
        audit[population] = {
            "selection_states": len(selected),
            "learned_mean_input_token_upper_bound": learned_mean,
            "selected_action": chosen,
            "selected_summary": summaries[chosen],
            "all_fixed_actions": summaries,
            "selection_read_response_quality_risk_or_judge": False,
        }
    # The lockbox inherits the action selected on p7-p12.  Its own text/cost
    # surface is never used to pick the comparator.
    selections["evoemo_lockbox"] = selections["evoemo_qualification"]
    audit["evoemo_lockbox"] = {
        "selected_action": selections["evoemo_lockbox"],
        "selection_source": "evoemo_qualification_p7_p12_only",
        "lockbox_surface_used_for_selection": False,
    }
    return selections, audit


def cost_population_for_partition(partition: str) -> str:
    return {
        "sealed_internal_test": "sealed_internal",
        "esconv_corrected_test_panel": "esconv",
        "evoemo_qualification_p7_p12": "evoemo_qualification",
        "evoemo_lockbox_p13_p18": "evoemo_lockbox",
    }[partition]


def secondary_human_selected(row: Mapping[str, Any]) -> bool:
    # Exactly one outcome-blind quartile within every final partition.  EvoEmo
    # remains clustered by user in analysis even though sampling is at state.
    value = int(
        stable_hex(
            HUMAN_SAMPLE_PROTOCOL,
            row["partition"],
            row["state_id"],
            "secondary-comparators",
            n=8,
        ),
        16,
    )
    return value % 4 == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_1_t5_plan_v1",
    )
    args = parser.parse_args()
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V5.1 requires {FORMAL_PYTHON}; got {sys.executable}")
    if not RUNNER.is_file():
        raise RuntimeError("T5 runner must exist before plan freeze")

    surface_report = read_json(SURFACE_REPORT)
    if surface_report.get("status") != "PASS_READY_FOR_T5_CALL_PLAN":
        raise RuntimeError("T5 response-free surface gate did not pass")
    if not all(surface_report["checks"].values()):
        raise RuntimeError("T5 response-free surface checks are incomplete")
    if not surface_report["evoemo_static_memory_qualification"]["all_memory_components_pass"]:
        raise RuntimeError("EvoEmo static transport qualification did not pass")
    surface_rows = rows(SURFACES)
    if len(surface_rows) != 388:
        raise RuntimeError("T5 plan requires 388 states")

    pm_config = load_config(ROOT / "configs/pm_v1_5.yaml")
    experiment = load_config(ROOT / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    generator = endpoint_from_config(experiment, generation.generator_endpoint)
    judge = endpoint_from_config(experiment, "final_judge")
    if generator.model != "meta/llama-3.1-8b-instruct":
        raise RuntimeError("T5 generator differs from the frozen V5/V5.1 generator")
    if judge.family == generator.family:
        raise RuntimeError("final judge must be a different model family from generator")
    safety_factor = float(pm_config["api_cost_planning"]["input_token_safety_factor"])
    pricing_proxy = {"input": 0.15, "output": 0.60}

    cost_actions, cost_audit = select_cost_matched_actions(
        surface_rows=surface_rows, safety_factor=safety_factor
    )
    call_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    judge_design: list[dict[str, Any]] = []
    human_design: list[dict[str, Any]] = []
    policy_action_distributions: dict[str, Counter[str]] = {
        policy: Counter() for policy in POLICIES
    }

    for row in sorted(surface_rows, key=lambda value: str(value["state_id"])):
        state_id = str(row["state_id"])
        partition = str(row["partition"])
        seeds = [
            stable_hex(SEED_PROTOCOL, state_id, str(index), n=16)
            for index in (1, 2)
        ]
        routes: dict[str, Any] = {}
        route_material: dict[str, tuple[list[dict[str, str]], int, dict[str, TypedResourceCandidate]]] = {}
        for policy in POLICIES:
            if policy == "cost_matched_fixed":
                frozen_action = cost_actions[cost_population_for_partition(partition)]
                requested = action_component_bits(frozen_action)
                selection_source = frozen_action
            else:
                requested = requested_policy_bits(row, policy)
                selection_source = None
            action, feasible, messages, bound, selected = action_messages_and_bound(
                row=row,
                requested_bits=requested,
                safety_factor=safety_factor,
            )
            routes[policy] = {
                "requested_bits_before_feasibility": dict(requested),
                "feasible_bits": feasible,
                "action_id": action,
                "cost_matched_fixed_action_before_projection": selection_source,
                "input_token_upper_bound": bound,
            }
            route_material[policy] = (messages, bound, selected)
            policy_action_distributions[policy][action] += 1

        for seed_index, seed_hex in enumerate(seeds, start=1):
            for policy in POLICIES:
                action = str(routes[policy]["action_id"])
                key = (state_id, seed_hex, action)
                if key in call_by_key:
                    continue
                messages, bound, selected = route_material[policy]
                call_id = "v51t5_" + stable_hex(
                    PROTOCOL, partition, state_id, seed_hex, action, n=24
                )
                call_by_key[key] = {
                    "protocol": PROTOCOL,
                    "call_id": call_id,
                    "domain": row["domain"],
                    "partition": partition,
                    "state_id": state_id,
                    "group_id_private_analysis_only": row["group_id_private_analysis_only"],
                    "user_id_private_analysis_only": row["user_id_private_analysis_only"],
                    "seed_index": seed_index,
                    "seed_hex": seed_hex,
                    "seed": seed_int(seed_hex),
                    "requested_action_id": action,
                    "resource_presented_to_executor": {
                        component: component in selected for component in COMPONENTS
                    },
                    "resource_subtypes": {
                        component: candidate.subtype
                        for component, candidate in selected.items()
                    },
                    "candidate_ids_pre_action": {
                        component: candidate.resource_id
                        for component, candidate in selected.items()
                    },
                    "current_user_text": row["current_user_text"],
                    "messages": messages,
                    "messages_sha256": sha256_text(canonical_json(messages)),
                    "input_token_upper_bound": bound,
                    "output_token_cap": int(generation.max_output_tokens),
                    "assigned_executor_version": "typed-resource-adapter-v1",
                    "assigned_generator_version": generator.model,
                }

            learned_action = str(routes["learned_pm"]["action_id"])
            for comparator in COMPARATORS:
                comparator_action = str(routes[comparator]["action_id"])
                comparison_id = "t5judge_" + stable_hex(
                    LLM_JUDGE_PROTOCOL,
                    partition,
                    state_id,
                    seed_hex,
                    comparator,
                    n=24,
                )
                reverse_repeat = int(
                    stable_hex(LLM_JUDGE_PROTOCOL, comparison_id, "reverse", n=8), 16
                ) % 5 == 0
                judge_design.append(
                    {
                        "protocol": LLM_JUDGE_PROTOCOL,
                        "comparison_id": comparison_id,
                        "domain": row["domain"],
                        "partition": partition,
                        "state_id": state_id,
                        "group_id_private_analysis_only": row["group_id_private_analysis_only"],
                        "seed_index": seed_index,
                        "seed_hex": seed_hex,
                        "policy_a": "learned_pm",
                        "policy_b": comparator,
                        "action_a": learned_action,
                        "action_b": comparator_action,
                        "semantic_identity_known_before_generation": learned_action == comparator_action,
                        "judge_required_if_actions_differ": learned_action != comparator_action,
                        "ab_position_key": stable_hex(LLM_JUDGE_PROTOCOL, comparison_id, "position", n=16),
                        "position_reversal_robustness_repeat": reverse_repeat,
                        "judge_endpoint": "final_judge",
                        "judge_model": judge.model,
                        "judge_family": judge.family,
                        "generator_family_different": judge.family != generator.family,
                        "quality_and_grounding_risk_constructs_separate": True,
                    }
                )

        # Human final uses one frozen seed.  The learned-vs-fixed-high primary
        # is complete; three secondary comparator contrasts use one fixed 25%
        # outcome-blind sample in every domain/partition.
        seed_hex = seeds[0]
        selected_comparators = ["fixed_high_eligible"]
        if secondary_human_selected(row):
            selected_comparators.extend(
                ["always_off", "transparent_rule", "cost_matched_fixed"]
            )
        for comparator in selected_comparators:
            comparison_id = "t5human_" + stable_hex(
                HUMAN_SAMPLE_PROTOCOL,
                partition,
                state_id,
                seed_hex,
                comparator,
                n=24,
            )
            overlap = int(
                stable_hex(HUMAN_SAMPLE_PROTOCOL, comparison_id, "overlap", n=8), 16
            ) % 5 == 0
            human_design.append(
                {
                    "protocol": HUMAN_SAMPLE_PROTOCOL,
                    "comparison_id": comparison_id,
                    "domain": row["domain"],
                    "partition": partition,
                    "state_id": state_id,
                    "group_id_private_analysis_only": row["group_id_private_analysis_only"],
                    "seed_index": 1,
                    "seed_hex": seed_hex,
                    "policy_a": "learned_pm",
                    "policy_b": comparator,
                    "action_a": routes["learned_pm"]["action_id"],
                    "action_b": routes[comparator]["action_id"],
                    "semantic_identity_known_before_generation": routes["learned_pm"]["action_id"] == routes[comparator]["action_id"],
                    "primary_full_comparison": comparator == "fixed_high_eligible",
                    "secondary_stratified_quarter_sample": comparator != "fixed_high_eligible",
                    "independent_second_reviewer_overlap": overlap,
                    "quality_review_blind_to_policy_resource_identity": True,
                    "risk_review_sees_only_authorized_evidence_needed_for_grounding": True,
                }
            )

        bindings.append(
            {
                "protocol": PROTOCOL,
                "domain": row["domain"],
                "partition": partition,
                "state_id": state_id,
                "group_id_private_analysis_only": row["group_id_private_analysis_only"],
                "user_id_private_analysis_only": row["user_id_private_analysis_only"],
                "target_component_private_analysis_only": row["target_component_private_analysis_only"],
                "generation_seed_hex": seeds,
                "policy_routes": routes,
                "component_surface_audit": {
                    component: {
                        "candidate_present": row["components"][component]["surface"]["candidate_present"],
                        "candidate_subtype": str(row["components"][component]["surface"]["compiler_subtype_hint"]),
                        "structurally_executable": row["components"][component]["structurally_executable"],
                        "learned_probability": row["components"][component]["learned_probability"],
                        "learned_threshold": row["components"][component]["learned_threshold"],
                        "support_diagnostics": row["components"][component]["support_diagnostics"],
                    }
                    for component in COMPONENTS
                },
                "response_quality_risk_judge_or_external_outcome_read": False,
            }
        )

    calls = sorted(call_by_key.values(), key=lambda row: str(row["call_id"]))
    if len({str(row["call_id"]) for row in calls}) != len(calls):
        raise RuntimeError("T5 call IDs are not unique")
    for binding in bindings:
        state_id = str(binding["state_id"])
        for seed_hex in binding["generation_seed_hex"]:
            for route in binding["policy_routes"].values():
                key = (state_id, str(seed_hex), str(route["action_id"]))
                if key not in call_by_key:
                    raise RuntimeError(f"missing generated action alias: {key}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    call_path = args.out_dir / "call_plan_private.jsonl"
    binding_path = args.out_dir / "policy_bindings_private.jsonl"
    judge_path = args.out_dir / "llm_judge_design_private.jsonl"
    human_path = args.out_dir / "human_review_design_private.jsonl"
    write_jsonl(call_path, calls)
    write_jsonl(binding_path, bindings)
    write_jsonl(judge_path, judge_design)
    write_jsonl(human_path, human_design)

    total_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    total_output = sum(int(row["output_token_cap"]) for row in calls)
    estimated = (
        total_input / 1_000_000 * pricing_proxy["input"]
        + total_output / 1_000_000 * pricing_proxy["output"]
    )
    comparisons_by_partition = Counter(
        str(row["partition"]) for row in human_design
    )
    judge_needed = sum(bool(row["judge_required_if_actions_differ"]) for row in judge_design)
    judge_reversals = sum(
        bool(row["judge_required_if_actions_differ"] and row["position_reversal_robustness_repeat"])
        for row in judge_design
    )
    checks = {
        "388_state_bindings": len(bindings) == 388,
        "two_frozen_seeds_each_state": all(len(row["generation_seed_hex"]) == 2 for row in bindings),
        "all_five_policies_each_state": all(set(row["policy_routes"]) == set(POLICIES) for row in bindings),
        "unique_action_calls_deduplicated_with_policy_aliases": len(calls) < 388 * len(POLICIES) * 2,
        "cost_match_never_reads_response_outcomes": all(not row.get("selection_read_response_quality_risk_or_judge", False) for key, row in cost_audit.items() if key != "evoemo_lockbox"),
        "evo_lockbox_did_not_select_cost_comparator": cost_audit["evoemo_lockbox"]["lockbox_surface_used_for_selection"] is False,
        "independent_final_judge_family": judge.family != generator.family,
        "llm_judge_all_state_seed_comparators_predeclared": len(judge_design) == 388 * 2 * len(COMPARATORS),
        "human_primary_fixed_high_all_states": sum(bool(row["primary_full_comparison"]) for row in human_design) == 388,
        "human_secondary_sample_outcome_blind": True,
        "no_response_human_or_llm_outcome_read": True,
    }
    implementation_paths = {
        "typed_adapter": ROOT / "src/metacom_pm/v1_5_typed_resource_adapter.py",
        "policy_runtime": ROOT / "src/metacom_pm/v1_5b_policy_runtime.py",
        "surface_builder": ROOT / "scripts/v1_5/26l_materialize_v5_1_t5_surfaces_v1_5.py",
        "plan_builder": Path(__file__).resolve(),
        "runner": RUNNER,
    }
    manifest = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_SINGLE_T5_GENERATION" if all(checks.values()) else "FAIL_T5_PLAN_FREEZE",
        "method_version": "V1_5_V5_1_FROZEN_T5",
        "generator": generator.model,
        "generator_family": generator.family,
        "temperature": float(generation.temperature),
        "states": len(bindings),
        "planned_unique_action_calls": len(calls),
        "policy_alias_action_slots_before_deduplication": 388 * len(POLICIES) * 2,
        "partition_call_counts": dict(Counter(str(row["partition"]) for row in calls)),
        "policy_action_distributions": {
            policy: dict(sorted(values.items()))
            for policy, values in policy_action_distributions.items()
        },
        "cost_matched_fixed": {
            "selected_actions": cost_actions,
            "selection_audit": cost_audit,
        },
        "generation_cost": {
            "input_token_upper_bound_total": total_input,
            "output_token_cap_total": total_output,
            "estimated_usd_upper_bound_proxy": round(estimated, 8),
            "pricing_usd_per_mtok_proxy": pricing_proxy,
        },
        "automatic_full_evaluation": {
            "all_states_and_policies": True,
            "metrics": [
                "candidate_present_and_executable_coverage",
                "requested_feasible_realized_action_distribution",
                "component_on_fraction",
                "prompt_completion_total_tokens",
                "fallback_and_guard_rate",
                "empirical_feature_range_support",
            ],
        },
        "independent_llm_judge": {
            "protocol": LLM_JUDGE_PROTOCOL,
            "endpoint": "final_judge",
            "model": judge.model,
            "family": judge.family,
            "all_predeclared_comparisons": len(judge_design),
            "nonidentical_action_comparisons_requiring_calls": judge_needed,
            "position_reversal_robustness_repeats": judge_reversals,
            "role": "secondary sensitivity analysis; never PM gold and never used to modify method",
        },
        "human_final": {
            "protocol": HUMAN_SAMPLE_PROTOCOL,
            "quality_pairs": len(human_design),
            "pairs_by_partition": dict(sorted(comparisons_by_partition.items())),
            "primary_learned_vs_fixed_high_all_states_one_seed": 388,
            "secondary_comparators": ["always_off", "transparent_rule", "cost_matched_fixed"],
            "secondary_sampling": "one outcome-blind hash quartile per partition",
            "independent_overlap_pairs": sum(bool(row["independent_second_reviewer_overlap"]) for row in human_design),
            "constructs": {
                "quality": "blind pairwise material adoption difference",
                "risk": "interaction-and-grounding material risk with authorized evidence visible",
                "cost": "objective provider usage, no human judgment",
            },
        },
        "analysis_contract": {
            "sealed_cluster": "counterfactual_group_id",
            "esconv_cluster": "dialogue_id",
            "evoemo_cluster": "user_id; qualification and lockbox reported separately",
            "quality_noninferiority_margin": -0.05,
            "risk_noninferiority_margin": 0.05,
            "minimum_prompt_cost_reduction_vs_fixed_high": 0.05,
            "primary_comparator": "fixed_high_eligible",
            "resource_value_comparator": "always_off",
            "learning_value_comparator": "transparent_rule",
            "cost_control_comparator": "cost_matched_fixed",
            "external_outcomes_may_modify_method": False,
        },
        "checks": checks,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (SURFACES, SURFACE_REPORT, CONTRACT)
        },
        "implementation_sha256": {
            name: sha256_file(path) for name, path in implementation_paths.items()
        },
        "call_plan_sha256": sha256_file(call_path),
        "policy_bindings_sha256": sha256_file(binding_path),
        "llm_judge_design_sha256": sha256_file(judge_path),
        "human_review_design_sha256": sha256_file(human_path),
        "api_calls": 0,
        "human_or_llm_outcomes_read": 0,
        "sealed_or_external_response_outcomes_read": 0,
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": manifest["status"],
                "states": manifest["states"],
                "planned_unique_action_calls": manifest["planned_unique_action_calls"],
                "estimated_generation_usd_upper_bound_proxy": manifest["generation_cost"]["estimated_usd_upper_bound_proxy"],
                "cost_matched_actions": cost_actions,
                "human_quality_pairs": manifest["human_final"]["quality_pairs"],
                "llm_judge_calls_before_position_reversals": judge_needed,
            }
        )
    )


if __name__ == "__main__":
    main()
