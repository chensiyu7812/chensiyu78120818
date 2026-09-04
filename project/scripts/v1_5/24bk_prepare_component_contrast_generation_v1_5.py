#!/usr/bin/env python3
"""Freeze the 256 genuinely new response calls for four-component contrasts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import (
    MemoryBackendRecord,
    MemoryItem,
    RuntimeState,
    StrategyMode,
    parse_action_id,
)
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
from metacom_pm.prompts import common_context, generation_messages
from metacom_pm.text import estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-four-component-contrast-generation-plan-v1"
BLUEPRINT_PROTOCOL = "pm-v1.5-four-component-clean-contrast-blueprint-v1"
SEED = 20260730
PRICING_USD_PER_MTOK = {"input": 0.15, "output": 0.60}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _identity(endpoint: Any) -> dict[str, Any]:
    return {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }


def _messages(
    *,
    state: RuntimeState,
    memories: list[MemoryItem],
    system_prompt: str,
    card: dict[str, Any] | None,
) -> list[dict[str, str]]:
    if card is None:
        return generation_messages(
            state,
            memories,
            [],
            system_prompt=system_prompt,
        )
    sections = [common_context(state)]
    if memories:
        lines = []
        for item in memories:
            when = item.timestamp or f"session {item.created_session}"
            lines.append(f"- [{item.source.value}; {when}] {item.text}")
        sections.append(
            "Potentially useful past information. Use selectively:\n"
            + "\n".join(lines)
        )
    sections.append(
        "Potential emotional-support technique. Treat it as the primary "
        "move, not the whole reply. Use it only when it fits; otherwise "
        "ignore it. Ground every factual, emotional, and temporal claim "
        "in the visible dialogue. Ask at most one question OR give at "
        "most one suggestion, never both or a list. You may add one short "
        "natural continuation when needed. Compose new wording and never "
        "mention this guidance:\n"
        f"- {card['prompt_guidance']}"
    )
    sections.append("Write only the counselor's next response.")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def build_plan(
    *,
    blueprint_dir: Path,
    runtime_states_path: Path,
    memory_backend_path: Path,
    historical_outcomes_path: Path,
    strategy_cards_path: Path,
    pm_config_path: Path,
    experiment_config_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    blueprint_report = read_json(blueprint_dir / "blueprint_report.json")
    blueprint = _rows(blueprint_dir / "contrast_blueprint.jsonl")
    if (
        blueprint_report.get("protocol") != BLUEPRINT_PROTOCOL
        or blueprint_report.get("status")
        != "READY_256_CONTRAST_BLUEPRINT_ZERO_NEW_API"
        or len(blueprint) != 256
    ):
        raise RuntimeError("component contrast blueprint is not frozen-ready")

    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(runtime_states_path)
        )
    }
    backend_records = [
        MemoryBackendRecord.model_validate(row)
        for row in iter_jsonl(memory_backend_path)
    ]
    memories_by_card = {
        record.card_id: {item.memory_id: item for item in record.items}
        for record in backend_records
    }
    outcomes = {
        (str(row["state_id"]), str(row["requested_action_id"])): dict(row)
        for row in iter_jsonl(historical_outcomes_path)
    }
    cards = {
        str(row["card_id"]): dict(row)
        for row in iter_jsonl(strategy_cards_path)
    }

    pm_config = load_config(pm_config_path)
    experiment = load_config(experiment_config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(
        experiment, generation.generator_endpoint
    )
    identity = _identity(endpoint)

    calls: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    expected_new_keys: set[tuple[str, str]] = set()
    for contrast in blueprint:
        slot_id = str(contrast["contrast_slot_id"])
        state_id = str(contrast["state_id"])
        state = states[state_id]
        if state.card_id != str(contrast["card_id"]):
            raise RuntimeError(f"state/card mismatch for {slot_id}")
        item_map = memories_by_card[state.card_id]
        strategy_candidate = contrast.get("current_strategy_candidate")

        for arm in ("control", "treatment"):
            action_id = str(contrast[f"{arm}_action"])
            generation_status = str(
                contrast[f"{arm}_generation_status"]
            )
            historical = outcomes[(state_id, action_id)]
            if (
                historical["requested_action_id"] != action_id
                or historical["effective_action_id"] != action_id
                or historical["realized_action_id"] != action_id
            ):
                raise RuntimeError(
                    f"historical action was not realized for {slot_id}/{arm}"
                )
            selected_memory_ids = [
                str(value)
                for value in historical.get("selected_memory_ids", [])
            ]
            memories = [item_map[value] for value in selected_memory_ids]
            requested_sources, strategy_mode = parse_action_id(action_id)
            realized_sources = {item.source for item in memories}
            if realized_sources != requested_sources:
                raise RuntimeError(
                    f"memory realization mismatch for {slot_id}/{arm}: "
                    f"requested={sorted(x.value for x in requested_sources)}, "
                    f"realized={sorted(x.value for x in realized_sources)}"
                )

            card: dict[str, Any] | None = None
            if strategy_mode is StrategyMode.RS:
                if not isinstance(strategy_candidate, dict):
                    raise RuntimeError(
                        f"missing current strategy for {slot_id}/{arm}"
                    )
                card_id = str(strategy_candidate["card_id"])
                card = cards[card_id]
            messages = _messages(
                state=state,
                memories=memories,
                system_prompt=generation.system_prompt,
                card=card,
            )
            messages_sha256 = sha256_text(canonical_json(messages))
            registry_row = {
                "protocol": PROTOCOL,
                "contrast_slot_id": slot_id,
                "component": contrast["component"],
                "split": contrast["split"],
                "state_id": state_id,
                "user_id": contrast["user_id"],
                "arm": arm,
                "action_id": action_id,
                "generation_status": generation_status,
                "selected_memory_ids": selected_memory_ids,
                "selected_strategy_card_id": (
                    None if card is None else card["card_id"]
                ),
                "selected_strategy_core_submove_id": (
                    None if card is None else card["core_submove_id"]
                ),
                "messages_sha256": messages_sha256,
                "response_source": (
                    "historical_frozen_outcome"
                    if generation_status == "REUSE_EXISTING_OUTCOME"
                    else "new_current_stack_generation"
                ),
                "response": (
                    str(historical["response"])
                    if generation_status == "REUSE_EXISTING_OUTCOME"
                    else None
                ),
                "response_sha256": (
                    sha256_text(str(historical["response"]))
                    if generation_status == "REUSE_EXISTING_OUTCOME"
                    else None
                ),
                "effect_label": "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW",
            }
            registry.append(registry_row)
            if generation_status == "REUSE_EXISTING_OUTCOME":
                continue
            if not generation_status.startswith("GENERATE_"):
                raise RuntimeError(
                    f"unknown generation status {generation_status!r}"
                )
            call_key = (slot_id, arm)
            if call_key in expected_new_keys:
                raise RuntimeError(f"duplicate new call {call_key}")
            expected_new_keys.add(call_key)
            calls.append(
                {
                    "protocol": PROTOCOL,
                    "call_id": "component_call_"
                    + stable_hex(PROTOCOL, slot_id, arm, n=24),
                    "contrast_slot_id": slot_id,
                    "component": contrast["component"],
                    "split": contrast["split"],
                    "state_id": state_id,
                    "user_id": contrast["user_id"],
                    "arm": arm,
                    "action_id": action_id,
                    "selected_memory_ids": selected_memory_ids,
                    "selected_strategy_card_id": (
                        None if card is None else card["card_id"]
                    ),
                    "selected_strategy_core_submove_id": (
                        None if card is None else card["core_submove_id"]
                    ),
                    "messages": messages,
                    "messages_sha256": messages_sha256,
                    "prompt_sha256": messages_sha256,
                    "estimated_input_tokens": estimate_tokens(
                        canonical_json(messages)
                    ),
                    "generator_identity": identity,
                    "generation": {**generation.payload(), "seed": SEED},
                    "effect_label": "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW",
                }
            )

    if len(registry) != 512 or len(calls) != 256:
        raise RuntimeError(
            f"expected 512 registry arms and 256 calls, got "
            f"{len(registry)} and {len(calls)}"
        )
    new_by_component = Counter(row["component"] for row in calls)
    new_by_split = Counter(row["split"] for row in calls)
    input_tokens = sum(row["estimated_input_tokens"] for row in calls)
    output_tokens = len(calls) * generation.max_output_tokens
    cost_upper = (
        input_tokens * PRICING_USD_PER_MTOK["input"]
        + output_tokens * PRICING_USD_PER_MTOK["output"]
    ) / 1_000_000
    report = {
        "protocol": PROTOCOL,
        "status": "FROZEN_READY_FOR_256_NEW_RESPONSE_CALLS",
        "effect_labels_created": False,
        "human_labels_read": False,
        "api_calls_made": 0,
        "contrast_groups": len(blueprint),
        "response_arms_total": len(registry),
        "historical_response_arms_reused": sum(
            row["response_source"] == "historical_frozen_outcome"
            for row in registry
        ),
        "new_response_calls": len(calls),
        "new_calls_by_component": dict(sorted(new_by_component.items())),
        "new_calls_by_split": dict(sorted(new_by_split.items())),
        "generator_identity": identity,
        "supporter_generation_treatment": generation.payload(),
        "supporter_generation_treatment_sha256": generation.digest(),
        "seed": SEED,
        "pricing_usd_per_mtok": PRICING_USD_PER_MTOK,
        "estimated_input_tokens": input_tokens,
        "max_output_tokens_upper_bound": output_tokens,
        "estimated_generation_cost_usd_upper_bound": round(cost_upper, 6),
        "checks": {
            "one_bit_contrast_count_256": len(blueprint) == 256,
            "two_arms_per_contrast": len(registry) == 512,
            "new_call_count_256": len(calls) == 256,
            "all_effect_labels_unknown": all(
                row["effect_label"]
                == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
                for row in registry + calls
            ),
            "no_historical_RS_response_reused": all(
                not (
                    row["response_source"] == "historical_frozen_outcome"
                    and row["action_id"].endswith("+RS")
                )
                for row in registry
            ),
            "all_new_RS_arms_use_current_card": all(
                row["selected_strategy_card_id"] is not None
                for row in calls
                if row["action_id"].endswith("+RS")
            ),
            "all_R0_arms_have_no_strategy": all(
                row["selected_strategy_card_id"] is None
                for row in registry
                if row["action_id"].endswith("+R0")
            ),
        },
        "inputs": {
            "blueprint_report": str(
                (blueprint_dir / "blueprint_report.json").relative_to(ROOT)
            ),
            "blueprint": str(
                (blueprint_dir / "contrast_blueprint.jsonl").relative_to(ROOT)
            ),
            "runtime_states": str(runtime_states_path.relative_to(ROOT)),
            "memory_backend": str(memory_backend_path.relative_to(ROOT)),
            "historical_outcomes": str(
                historical_outcomes_path.relative_to(ROOT)
            ),
            "current_strategy_cards": str(
                strategy_cards_path.relative_to(ROOT)
            ),
            "pm_config": str(pm_config_path.relative_to(ROOT)),
            "experiment_config": str(
                experiment_config_path.relative_to(ROOT)
            ),
        },
    }
    manifest = {
        "protocol": PROTOCOL,
        "status": "FROZEN_READY_FOR_256_NEW_RESPONSE_CALLS",
        "input_sha256": {
            value: sha256_file(ROOT / value)
            for value in report["inputs"].values()
        },
        "generator_identity": identity,
        "supporter_generation_treatment_sha256": generation.digest(),
        "seed": SEED,
    }
    return report, calls, registry, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_contrast_blueprint_v1",
    )
    parser.add_argument(
        "--runtime-states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "runtime_states.jsonl",
    )
    parser.add_argument(
        "--memory-backend",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "memory_backend.jsonl",
    )
    parser.add_argument(
        "--historical-outcomes",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_longitudinal_action_sweep_v8_19_2_"
        "continuation_v2_dry_run/action_outcomes.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--pm-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_four_component_contrast_generation_v1",
    )
    args = parser.parse_args()
    report, calls, registry, manifest = build_plan(
        blueprint_dir=args.blueprint_dir,
        runtime_states_path=args.runtime_states,
        memory_backend_path=args.memory_backend,
        historical_outcomes_path=args.historical_outcomes,
        strategy_cards_path=args.strategy_cards,
        pm_config_path=args.pm_config,
        experiment_config_path=args.experiment_config,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calls_path = args.out_dir / "call_plan.jsonl"
    registry_path = args.out_dir / "response_arm_registry.jsonl"
    report_path = args.out_dir / "plan_report.json"
    write_jsonl(calls_path, calls)
    write_jsonl(registry_path, registry)
    write_json(report_path, report)
    manifest["outputs"] = {
        calls_path.name: sha256_file(calls_path),
        registry_path.name: sha256_file(registry_path),
        report_path.name: sha256_file(report_path),
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": report["status"],
                "contrast_groups": report["contrast_groups"],
                "historical_response_arms_reused": report[
                    "historical_response_arms_reused"
                ],
                "new_response_calls": report["new_response_calls"],
                "estimated_generation_cost_usd_upper_bound": report[
                    "estimated_generation_cost_usd_upper_bound"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
