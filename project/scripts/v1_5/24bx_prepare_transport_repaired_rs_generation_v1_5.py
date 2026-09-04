#!/usr/bin/env python3
"""Freeze 128 RS response calls on the repaired memory backend."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import (
    MemoryBackendRecord,
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
from metacom_pm.prompts import generation_messages
from metacom_pm.text import estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-transport-repaired-rs-generation-plan-v1"
BLUEPRINT_PROTOCOL = (
    "pm-v1.5-transport-repaired-rs-contrast-blueprint-v1"
)
STATUS = "FROZEN_READY_FOR_128_REPAIRED_RS_CALLS"
SEED = 20260730
PRICING_USD_PER_MTOK = {"input": 0.15, "output": 0.60}


def build_plan(
    *,
    blueprint_dir: Path,
    backend_dir: Path,
    strategy_cards_path: Path,
    pm_config_path: Path,
    experiment_config_path: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    blueprint_report = read_json(
        blueprint_dir / "blueprint_report.json"
    )
    blueprint = [
        dict(row)
        for row in iter_jsonl(
            blueprint_dir / "rs_contrast_blueprint.jsonl"
        )
    ]
    if (
        blueprint_report.get("protocol") != BLUEPRINT_PROTOCOL
        or blueprint_report.get("status")
        != "READY_64_RS_PAIRS_ON_TRANSPORT_REPAIRED_STACK"
        or len(blueprint) != 64
    ):
        raise RuntimeError("repaired RS blueprint is not ready")
    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(
                backend_dir / "runtime_states.jsonl"
            )
        )
    }
    backends = {
        record.card_id: {
            item.memory_id: item for item in record.items
        }
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(
                backend_dir / "memory_backend.jsonl"
            )
        )
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
    identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }
    calls: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for contrast in blueprint:
        slot_id = str(contrast["contrast_slot_id"])
        state = states[str(contrast["state_id"])]
        item_map = backends[state.card_id]
        for arm, action_field in (
            ("control", "control_action"),
            ("treatment", "treatment_action"),
        ):
            action = str(contrast[action_field])
            memory_ids = list(
                contrast[
                    "selected_memory_ids_by_action_generation_only"
                ][action]
            )
            memories = [item_map[str(value)] for value in memory_ids]
            _, strategy_mode = parse_action_id(action)
            strategy_cards: list[dict[str, Any]] = []
            if strategy_mode is StrategyMode.RS:
                candidate = dict(
                    contrast["current_strategy_candidate"]
                )
                strategy_cards = [cards[str(candidate["card_id"])]]
            messages = generation_messages(
                state,
                memories,
                strategy_cards,
                system_prompt=generation.system_prompt,
            )
            prompt_sha256 = sha256_text(canonical_json(messages))
            common = {
                "protocol": PROTOCOL,
                "call_id": "transport_rs_call_"
                + stable_hex(PROTOCOL, slot_id, arm, n=24),
                "contrast_slot_id": slot_id,
                "component": "RS",
                "split": contrast["split"],
                "state_id": state.state_id,
                "user_id": state.user_id,
                "arm": arm,
                "action_id": action,
                "selected_memory_ids": memory_ids,
                "selected_strategy_card_id": (
                    strategy_cards[0]["card_id"]
                    if strategy_cards
                    else None
                ),
                "selected_strategy_core_submove_id": (
                    strategy_cards[0]["core_submove_id"]
                    if strategy_cards
                    else None
                ),
                "messages_sha256": prompt_sha256,
                "prompt_sha256": prompt_sha256,
                "effect_label": (
                    "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
                ),
            }
            registry.append(
                {
                    **common,
                    "generation_status": (
                        "FROZEN_PENDING_CURRENT_STACK_GENERATION"
                    ),
                    "response": None,
                    "response_sha256": None,
                }
            )
            calls.append(
                {
                    **common,
                    "messages": messages,
                    "estimated_input_tokens": estimate_tokens(
                        canonical_json(messages)
                    ),
                    "generator_identity": identity,
                    "generation": {
                        **generation.payload(),
                        "seed": SEED,
                    },
                }
            )
    if len(calls) != 128 or len(registry) != 128:
        raise RuntimeError("expected exactly 128 repaired RS arms")
    input_tokens = sum(
        int(row["estimated_input_tokens"]) for row in calls
    )
    output_tokens = len(calls) * generation.max_output_tokens
    estimated_cost = (
        input_tokens * PRICING_USD_PER_MTOK["input"]
        + output_tokens * PRICING_USD_PER_MTOK["output"]
    ) / 1_000_000
    report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "effect_labels_created": False,
        "human_labels_read": False,
        "api_calls_made": 0,
        "contrast_groups": 64,
        "response_arms_total": 128,
        "new_response_calls": 128,
        "calls_by_split": dict(
            sorted(Counter(row["split"] for row in calls).items())
        ),
        "generator_identity": identity,
        "supporter_generation_treatment": generation.payload(),
        "supporter_generation_treatment_sha256": generation.digest(),
        "seed": SEED,
        "pricing_usd_per_mtok": PRICING_USD_PER_MTOK,
        "estimated_input_tokens": input_tokens,
        "max_output_tokens_upper_bound": output_tokens,
        "estimated_generation_cost_usd_upper_bound": round(
            estimated_cost, 6
        ),
        "checks": {
            "64_one_bit_rs_contrasts": len(blueprint) == 64,
            "two_new_arms_per_contrast": len(calls) == 128,
            "all_effect_labels_unknown": all(
                row["effect_label"]
                == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
                for row in calls + registry
            ),
            "treatment_has_strategy": all(
                row["selected_strategy_card_id"] is not None
                for row in calls
                if row["arm"] == "treatment"
            ),
            "control_has_no_strategy": all(
                row["selected_strategy_card_id"] is None
                for row in calls
                if row["arm"] == "control"
            ),
        },
        "inputs": {
            "blueprint_report": str(
                (blueprint_dir / "blueprint_report.json").relative_to(
                    ROOT
                )
            ),
            "blueprint": str(
                (
                    blueprint_dir / "rs_contrast_blueprint.jsonl"
                ).relative_to(ROOT)
            ),
            "runtime_states": str(
                (backend_dir / "runtime_states.jsonl").relative_to(ROOT)
            ),
            "memory_backend": str(
                (backend_dir / "memory_backend.jsonl").relative_to(ROOT)
            ),
            "strategy_cards": str(
                strategy_cards_path.relative_to(ROOT)
            ),
            "pm_config": str(pm_config_path.relative_to(ROOT)),
            "experiment_config": str(
                experiment_config_path.relative_to(ROOT)
            ),
        },
    }
    if not all(report["checks"].values()):
        report["status"] = "BLOCKED_BY_RS_PLAN_CHECK"
    manifest = {
        "protocol": PROTOCOL,
        "status": report["status"],
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
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1",
    )
    parser.add_argument(
        "--backend-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate",
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
        / "outputs/pm_v1_5_transport_repaired_rs_generation_v1",
    )
    args = parser.parse_args()
    report, calls, registry, manifest = build_plan(
        blueprint_dir=args.blueprint_dir,
        backend_dir=args.backend_dir,
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
        path.name: sha256_file(path)
        for path in (calls_path, registry_path, report_path)
    }
    write_json(args.out_dir / "freeze_manifest.json", manifest)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "contrast_groups": 64,
            "new_response_calls": 128,
            "estimated_generation_cost_usd_upper_bound": report[
                "estimated_generation_cost_usd_upper_bound"
            ],
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
