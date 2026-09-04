#!/usr/bin/env python3
"""Freeze the 144 response calls for the D2 measurement qualification."""

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
PROTOCOL = "pm-v1.5-d2-measurement-generation-plan-v1"
BLUEPRINT_PROTOCOL = "pm-v1.5-d2-measurement-blueprint-v1"
BLUEPRINT_STATUS = "FROZEN_READY_FOR_144_D2_RESPONSE_CALLS"
STATUS = "FROZEN_READY_FOR_144_D2_RESPONSE_CALLS"
PRICING_USD_PER_MTOK = {"input": 0.15, "output": 0.60}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blueprint-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d2_measurement_blueprint_v1",
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
        default=ROOT / "outputs/pm_v1_5_d2_measurement_generation_v1",
    )
    args = parser.parse_args()

    blueprint_report = read_json(
        args.blueprint_dir / "blueprint_report.json"
    )
    blueprint_manifest = read_json(
        args.blueprint_dir / "freeze_manifest.json"
    )
    blueprint_path = (
        args.blueprint_dir / "d2_measurement_blueprint.jsonl"
    )
    blueprint = _rows(blueprint_path)
    if (
        blueprint_report.get("protocol") != BLUEPRINT_PROTOCOL
        or blueprint_report.get("status") != BLUEPRINT_STATUS
        or blueprint_manifest.get("protocol") != BLUEPRINT_PROTOCOL
        or blueprint_manifest.get("status") != BLUEPRINT_STATUS
        or blueprint_manifest.get("blueprint_sha256")
        != sha256_file(blueprint_path)
        or len(blueprint) != 32
    ):
        raise RuntimeError("D2 blueprint is stale or unqualified")

    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(
                args.backend_dir / "runtime_states.jsonl"
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
                args.backend_dir / "memory_backend.jsonl"
            )
        )
    }
    cards = {
        str(row["card_id"]): row for row in _rows(args.strategy_cards)
    }
    pm_config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
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
    for row in blueprint:
        state = states[str(row["state_id"])]
        if state.current_session_summary.strip():
            raise RuntimeError(
                f"D2 state violates current-session policy: {state.state_id}"
            )
        item_map = backends[state.card_id]
        for replicate_index, seed in enumerate(
            row["new_pair_seeds"], start=1
        ):
            pair_id = "d2_pair_" + stable_hex(
                PROTOCOL,
                str(row["d2_state_id"]),
                str(replicate_index),
                str(seed),
                n=24,
            )
            for arm in ("control", "treatment"):
                action = str(row[f"{arm}_action"])
                memory_ids = [
                    str(value)
                    for value in row[
                        "selected_memory_ids_by_action_generation_only"
                    ][action]
                ]
                memories = [item_map[value] for value in memory_ids]
                requested_sources, strategy_mode = parse_action_id(action)
                if {item.source for item in memories} != set(
                    requested_sources
                ):
                    raise RuntimeError(
                        f"memory realization mismatch: {pair_id}/{arm}"
                    )
                strategy_cards: list[dict[str, Any]] = []
                if strategy_mode is StrategyMode.RS:
                    candidate = row.get("current_strategy_candidate")
                    if not isinstance(candidate, dict):
                        raise RuntimeError(
                            f"missing frozen RS candidate: {pair_id}/{arm}"
                        )
                    strategy_cards = [
                        cards[str(candidate["card_id"])]
                    ]
                messages = generation_messages(
                    state,
                    memories,
                    strategy_cards,
                    system_prompt=generation.system_prompt,
                )
                prompt_sha256 = sha256_text(
                    canonical_json(messages)
                )
                common = {
                    "protocol": PROTOCOL,
                    "call_id": "d2_call_"
                    + stable_hex(PROTOCOL, pair_id, arm, n=24),
                    "d2_state_id": row["d2_state_id"],
                    "pair_id": pair_id,
                    "replicate_index": replicate_index,
                    "component": row["component"],
                    "state_origin": row["state_origin"],
                    "state_id": state.state_id,
                    "user_id": state.user_id,
                    "split": row["split"],
                    "arm": arm,
                    "action_id": action,
                    "selected_memory_ids": memory_ids,
                    "selected_strategy_card_id": (
                        strategy_cards[0]["card_id"]
                        if strategy_cards
                        else None
                    ),
                    "prompt_sha256": prompt_sha256,
                    "messages_sha256": prompt_sha256,
                    "effect_label": (
                        "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
                    ),
                }
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
                            "seed": int(seed),
                        },
                    }
                )
                registry.append(
                    {
                        **common,
                        "generation_status": (
                            "FROZEN_PENDING_D2_GENERATION"
                        ),
                        "response": None,
                        "response_sha256": None,
                    }
                )

    pair_ids = {str(row["pair_id"]) for row in calls}
    input_tokens = sum(
        int(row["estimated_input_tokens"]) for row in calls
    )
    output_tokens = len(calls) * generation.max_output_tokens
    cost_upper = (
        input_tokens * PRICING_USD_PER_MTOK["input"]
        + output_tokens * PRICING_USD_PER_MTOK["output"]
    ) / 1_000_000
    checks = {
        "32_states": len(
            {str(row["d2_state_id"]) for row in calls}
        )
        == 32,
        "72_pairs": len(pair_ids) == 72,
        "144_calls": len(calls) == 144,
        "two_arms_per_pair": Counter(
            str(row["pair_id"]) for row in calls
        )
        == Counter({pair_id: 2 for pair_id in pair_ids}),
        "all_state_summaries_empty": all(
            not states[str(row["state_id"])].current_session_summary.strip()
            for row in calls
        ),
        "all_labels_unknown": all(
            row["effect_label"]
            == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
            for row in calls
        ),
        "replicate_seed_varies_within_state": all(
            len(
                {
                    int(call["generation"]["seed"])
                    for call in calls
                    if call["d2_state_id"] == state_id
                }
            )
            == (
                3
                if next(
                    row
                    for row in blueprint
                    if row["d2_state_id"] == state_id
                )["component"]
                == "MS"
                else 2
            )
            for state_id in {
                str(row["d2_state_id"]) for row in calls
            }
        ),
    }
    status = STATUS if all(checks.values()) else (
        "BLOCKED_BY_D2_GENERATION_PLAN_CHECK"
    )
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "effect_labels_created": False,
        "human_labels_read": False,
        "api_calls_made": 0,
        "states": 32,
        "pairs": len(pair_ids),
        "new_response_calls": len(calls),
        "calls_by_component": dict(
            sorted(Counter(row["component"] for row in calls).items())
        ),
        "generator_identity": identity,
        "supporter_generation_treatment": generation.payload(),
        "supporter_generation_treatment_sha256": generation.digest(),
        "pricing_usd_per_mtok": PRICING_USD_PER_MTOK,
        "estimated_input_tokens": input_tokens,
        "max_output_tokens_upper_bound": output_tokens,
        "estimated_generation_cost_usd_upper_bound": round(
            cost_upper, 6
        ),
        "checks": checks,
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                blueprint_path,
                args.blueprint_dir / "blueprint_report.json",
                args.blueprint_dir / "freeze_manifest.json",
                args.backend_dir / "runtime_states.jsonl",
                args.backend_dir / "memory_backend.jsonl",
                args.strategy_cards,
                args.pm_config,
                args.experiment_config,
            )
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    calls_path = args.out_dir / "call_plan.jsonl"
    registry_path = args.out_dir / "response_arm_registry.jsonl"
    report_path = args.out_dir / "plan_report.json"
    write_jsonl(calls_path, calls)
    write_jsonl(registry_path, registry)
    write_json(report_path, report)
    write_json(
        args.out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "generator_identity": identity,
            "supporter_generation_treatment_sha256": generation.digest(),
            "outputs": {
                path.name: sha256_file(path)
                for path in (calls_path, registry_path, report_path)
            },
        },
    )
    print(
        {
            "protocol": PROTOCOL,
            "status": status,
            "states": 32,
            "pairs": len(pair_ids),
            "calls": len(calls),
            "estimated_cost_usd_upper_bound": report[
                "estimated_generation_cost_usd_upper_bound"
            ],
        }
    )


if __name__ == "__main__":
    main()
