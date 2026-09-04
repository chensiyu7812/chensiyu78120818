#!/usr/bin/env python3
"""Prepare the corrected single- and multi-component Step2 execution gate."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import MemoryItem, MemorySource, RuntimeState
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
from metacom_pm.prompts import common_context, generation_messages
from metacom_pm.text import conservative_token_bound
from metacom_pm.v1_5_candidate_discovery import (
    materialize_rank1_memory_for_execution,
)
from metacom_pm.v1_5_generator_alignment_audit import (
    build_resource_execution_plan,
    materialize_strategy_card_for_execution,
    resource_bundle_application_messages_v1_5b,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-corrected-step2-internal-and-development-gate-v3"
INPUT_USD_PER_MTOK = 0.15
OUTPUT_USD_PER_MTOK = 0.60
SAFETY_FACTOR = 1.5
MULTI_ACTIONS = ("MPE+RS", "MSE+RS", "MPMSME+RS")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * INPUT_USD_PER_MTOK / 1_000_000
        + output_tokens * OUTPUT_USD_PER_MTOK / 1_000_000
    )


def _single_component_calls(
    *, root: Path, generation: SupporterGenerationContract
) -> list[dict[str, Any]]:
    source_dir = root / "outputs/pm_v1_5_generator_alignment_qualification_v1"
    resources = _rows(source_dir / "private_selected_resources.jsonl")
    source_calls = {
        str(row["state_id"]): row
        for row in _rows(source_dir / "call_plan.jsonl")
        if row["prompt_variant"] == "source_matched"
    }
    expected = Counter({"MP": 8, "MS": 8, "ME": 8, "RS": 8})
    if Counter(str(row["component"]) for row in resources) != expected:
        raise RuntimeError("expected frozen 8-per-component qualification pool")
    calls: list[dict[str, Any]] = []
    for row in sorted(
        resources, key=lambda value: (str(value["component"]), str(value["state_id"]))
    ):
        state = RuntimeState.model_validate(row["runtime_state"])
        component = str(row["component"])
        source_call = source_calls[state.state_id]
        subtype = str(source_call["resource_subtype"])
        if component == "RS":
            cards = list(row["selected_strategy_cards"])
            if len(cards) != 1:
                raise RuntimeError("RS internal gate requires exactly one card")
            resource = materialize_strategy_card_for_execution(cards[0])
            lineage = {"card_id": str(cards[0]["card_id"])}
            action_id = "M0+RS"
        else:
            source = MemorySource(component)
            items = tuple(
                MemoryItem.model_validate(item) for item in row["selected_memory_items"]
            )
            resource, execution_item = materialize_rank1_memory_for_execution(
                source=source,
                selected_items=items,
                session_index=state.session_index,
            )
            lineage = {
                "candidate_memory_ids": [item.memory_id for item in items],
                "execution_memory_id": execution_item.memory_id,
                "execution_rank": 1,
                "execution_item_count": 1,
            }
            action_id = f"{component}+R0"
        plan = build_resource_execution_plan(
            component=component, resource_subtype=subtype
        )
        label = component + "1"
        selected = {component: resource}
        messages = resource_bundle_application_messages_v1_5b(
            plans={component: plan},
            visible_context=common_context(state),
            selected_resources=selected,
            resource_labels={component: label},
            system_prompt=generation.system_prompt,
        )
        fallback_messages = generation_messages(
            state, [], [], system_prompt=generation.system_prompt
        )
        calls.append(
            {
                "protocol": PROTOCOL,
                "call_id": "step2gate_"
                + stable_hex(PROTOCOL, "single", state.state_id, component, n=24),
                "panel_id": "internal_" + state.state_id,
                "domain": "InternalQualification",
                "state_id": state.state_id,
                "group_id_private_analysis_only": state.user_id,
                "requested_action_id": action_id,
                "policy_aliases": ["corrected_single_component_execution_gate"],
                "requested_components": [component],
                "execution_plans": {component: plan.model_dump(mode="json")},
                "resource_labels": {component: label},
                "selected_resources_private": selected,
                "resource_lineage_private": {component: lineage},
                "current_user_text": state.current_user_text,
                "messages": messages,
                "messages_sha256": sha256_text(canonical_json(messages)),
                "response_schema": "ResourceBundleApplicationOutput",
                "fallback_messages": fallback_messages,
                "fallback_messages_sha256": sha256_text(
                    canonical_json(fallback_messages)
                ),
                "input_token_upper_bound": conservative_token_bound(
                    canonical_json(messages), safety_factor=SAFETY_FACTOR
                ),
                "primary_output_token_cap": max(
                    500, int(generation.max_output_tokens) + 200
                ),
                "maximum_fallback_input_token_upper_bound": conservative_token_bound(
                    canonical_json(fallback_messages), safety_factor=SAFETY_FACTOR
                ),
                "maximum_fallback_output_token_cap": int(
                    generation.max_output_tokens
                ),
                "seed": int(source_call["generation"]["seed"]) % (2**31 - 1),
                "generation_deduplication_key": f"internal::{state.state_id}::{action_id}",
                "selection_or_prompt_uses_external_response_quality_risk_or_judge": False,
            }
        )
    return calls


def _multi_component_calls(*, root: Path) -> list[dict[str, Any]]:
    source_path = (
        root
        / "outputs/pm_v1_5b_corrected_final_system_generation_v4_partitions"
        / "evoemo_diagnostic_call_plan_private.jsonl"
    )
    source = _rows(source_path)
    selected: list[dict[str, Any]] = []
    for action in MULTI_ACTIONS:
        candidates = sorted(
            (row for row in source if row["requested_action_id"] == action),
            key=lambda row: stable_hex(
                PROTOCOL, "multi", action, str(row["panel_id"]), n=32
            ),
        )
        if len(candidates) < 4:
            raise RuntimeError(f"not enough diagnostic calls for {action}")
        for original in candidates[:4]:
            row = dict(original)
            row["call_id"] = "step2gate_" + stable_hex(
                PROTOCOL, "multi", str(original["call_id"]), n=24
            )
            row["protocol"] = PROTOCOL
            row["policy_aliases"] = ["corrected_multi_component_execution_gate"]
            row["generation_deduplication_key"] = (
                f"development::{row['panel_id']}::{action}"
            )
            selected.append(row)
    return selected


def build(*, root: Path = ROOT) -> dict[str, Any]:
    pm_config = load_config(root / "configs/pm_v1_5.yaml")
    experiment = load_config(root / "configs/experiment.yaml")
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    calls = [
        *_single_component_calls(root=root, generation=generation),
        *_multi_component_calls(root=root),
    ]
    if len(calls) != 44 or len({str(row["call_id"]) for row in calls}) != 44:
        raise RuntimeError("corrected Step2 gate must contain 44 unique calls")
    if any(
        not str(row["selected_resources_private"][component]).strip()
        for row in calls
        for component in row["requested_components"]
    ):
        raise RuntimeError("corrected Step2 gate contains an empty resource")
    out_dir = root / "outputs/pm_v1_5b_corrected_step2_execution_gate_v3_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "call_plan_private.jsonl"
    write_jsonl(plan_path, calls)
    primary_input = sum(int(row["input_token_upper_bound"]) for row in calls)
    primary_output = sum(int(row["primary_output_token_cap"]) for row in calls)
    fallback_input = sum(
        int(row["maximum_fallback_input_token_upper_bound"]) for row in calls
    )
    fallback_output = sum(
        int(row["maximum_fallback_output_token_cap"]) for row in calls
    )
    checks = {
        "32_single_component_calls": sum(
            row["policy_aliases"] == ["corrected_single_component_execution_gate"]
            for row in calls
        )
        == 32,
        "12_multi_component_calls": sum(
            row["policy_aliases"] == ["corrected_multi_component_execution_gate"]
            for row in calls
        )
        == 12,
        "single_component_balance_is_8_each": Counter(
            row["requested_components"][0]
            for row in calls
            if len(row["requested_components"]) == 1
        )
        == Counter({"MP": 8, "MS": 8, "ME": 8, "RS": 8}),
        "multi_action_balance_is_4_each": Counter(
            row["requested_action_id"]
            for row in calls
            if len(row["requested_components"]) > 1
        )
        == Counter({action: 4 for action in MULTI_ACTIONS}),
        "every_step1_active_resource_nonempty_and_materialized": all(
            all(
                str(row["selected_resources_private"][component]).strip()
                and any(
                    row["selected_resources_private"][component]
                    in str(message.get("content") or "")
                    for message in row["messages"]
                )
                for component in row["requested_components"]
            )
            for row in calls
        ),
        "no_external_outcome_used": all(
            not row["selection_or_prompt_uses_external_response_quality_risk_or_judge"]
            for row in calls
        ),
    }
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_FINAL_PAID_GENERATION_REVIEW"
        if all(checks.values())
        else "FAIL_CORRECTED_STEP2_GATE_PLAN",
        "planned_primary_calls": len(calls),
        "maximum_fallback_calls": len(calls),
        "estimated_primary_generation_usd": round(
            _usd(primary_input, primary_output), 6
        ),
        "estimated_generation_usd_with_all_fallbacks": round(
            _usd(primary_input + fallback_input, primary_output + fallback_output), 6
        ),
        "call_plan_sha256": sha256_file(plan_path),
        "generator": endpoint.model,
        "checks": checks,
        "qualification_gates": {
            "single_component_schema_and_bookkeeping": "32/32",
            "single_component_full_realization_minimum": "7/8 per component",
            "multi_component_exact_bookkeeping": "12/12",
            "multi_component_full_action_realization_minimum": "9/12 overall and at least 3/4 per action",
            "fabricated_recall": 0,
            "material_misuse": "at most 2/44 overall and at most 1 per component/action family",
        },
        "inputs": {
            "single_component_pool": sha256_file(
                root
                / "outputs/pm_v1_5_generator_alignment_qualification_v1/private_selected_resources.jsonl"
            ),
            "multi_component_development_plan": sha256_file(
                root
                / "outputs/pm_v1_5b_corrected_final_system_generation_v4_partitions/evoemo_diagnostic_call_plan_private.jsonl"
            ),
        },
    }
    write_json(out_dir / "generation_preflight.json", report)
    return report


def main() -> None:
    report = build()
    print(report)


if __name__ == "__main__":
    main()
