#!/usr/bin/env python3
"""Freeze the 320 response calls for PM V1.5 D3."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import MemoryBackendRecord, RuntimeState, StrategyMode, parse_action_id
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, iter_jsonl, read_json, sha256_file, sha256_text, stable_hex, write_json, write_jsonl
from metacom_pm.prompts import generation_messages
from metacom_pm.text import estimate_tokens


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d3-generation-plan-v1"
BLUEPRINT_PROTOCOL = "pm-v1.5-d3-step0-blueprint-v1"
BLUEPRINT_STATUS = "PASS_READY_TO_PREPARE_320_FROZEN_RESPONSE_CALLS"
STATUS = "FROZEN_READY_FOR_320_D3_RESPONSE_CALLS"
PRICING_USD_PER_MTOK = {"input": 0.15, "output": 0.60}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def prepare(
    *,
    blueprint_dir: Path,
    strategy_cards_path: Path,
    pm_config_path: Path,
    experiment_config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    report = read_json(blueprint_dir / "preflight_report.json")
    manifest = read_json(blueprint_dir / "freeze_manifest.json")
    if (
        report.get("protocol") != BLUEPRINT_PROTOCOL
        or report.get("status") != BLUEPRINT_STATUS
        or not all(report.get("checks", {}).values())
        or manifest.get("protocol") != BLUEPRINT_PROTOCOL
        or manifest.get("status") != BLUEPRINT_STATUS
        or manifest.get("preflight_report_sha256") != sha256_file(blueprint_dir / "preflight_report.json")
        or manifest.get("strategy_cards_sha256") != sha256_file(strategy_cards_path)
    ):
        raise RuntimeError("D3 Step-0 blueprint is stale or unqualified")

    pairs = _rows(blueprint_dir / "d3_primary_pairs.jsonl") + _rows(
        blueprint_dir / "d3_repeat_pairs.jsonl"
    )
    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "runtime_states.jsonl")
        )
    }
    backends = {
        record.card_id: {item.memory_id: item for item in record.items}
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "memory_backend.jsonl")
        )
    }
    cards = {str(row["card_id"]): row for row in _rows(strategy_cards_path)}
    pm_config = load_config(pm_config_path)
    experiment = load_config(experiment_config_path)
    generation = SupporterGenerationContract.from_config(pm_config)
    endpoint = endpoint_from_config(experiment, generation.generator_endpoint)
    identity = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "transport": endpoint.transport,
    }

    calls: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for pair in pairs:
        state = states[str(pair["state_id"])]
        pair_id = "pair_" + stable_hex(
            PROTOCOL,
            str(pair["contrast_slot_id"]),
            str(pair["pair_role"]),
            str(pair["pair_seed"]),
            n=24,
        )
        for arm in ("control", "treatment"):
            action = str(pair[f"{arm}_action"])
            memory_ids = [
                str(value)
                for value in pair["selected_memory_ids_by_action_generation_only"][action]
            ]
            memories = [backends[state.card_id][memory_id] for memory_id in memory_ids]
            requested_sources, strategy_mode = parse_action_id(action)
            if {item.source for item in memories} != set(requested_sources):
                raise RuntimeError(f"memory realization mismatch: {pair_id}/{arm}")
            strategy_cards: list[dict[str, Any]] = []
            if strategy_mode is StrategyMode.RS:
                candidate = pair.get("current_strategy_candidate")
                if not isinstance(candidate, dict):
                    raise RuntimeError(f"missing RS candidate: {pair_id}/{arm}")
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
                "call_id": "d3_call_" + stable_hex(PROTOCOL, pair_id, arm, n=24),
                "pair_id": pair_id,
                "pair_role": pair["pair_role"],
                "contrast_slot_id": pair["contrast_slot_id"],
                "component": pair["component"],
                "state_id": state.state_id,
                "user_id": state.user_id,
                "arm": arm,
                "action_id": action,
                "selected_memory_ids": memory_ids,
                "selected_strategy_card_id": strategy_cards[0]["card_id"] if strategy_cards else None,
                "prompt_sha256": prompt_sha256,
                "messages_sha256": prompt_sha256,
                "effect_label": "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW",
            }
            calls.append(
                {
                    **common,
                    "messages": messages,
                    "estimated_input_tokens": estimate_tokens(canonical_json(messages)),
                    "generator_identity": identity,
                    "generation": {**generation.payload(), "seed": int(pair["pair_seed"])},
                }
            )
            registry.append(
                {
                    **common,
                    "generation_status": "FROZEN_PENDING_D3_GENERATION",
                    "response": None,
                    "response_sha256": None,
                }
            )

    pair_counts = Counter(row["pair_id"] for row in calls)
    component_pair_counts = Counter(
        row["component"] for row in calls if row["arm"] == "control"
    )
    input_tokens = sum(int(row["estimated_input_tokens"]) for row in calls)
    output_tokens = len(calls) * generation.max_output_tokens
    cost_upper = (
        input_tokens * PRICING_USD_PER_MTOK["input"]
        + output_tokens * PRICING_USD_PER_MTOK["output"]
    ) / 1_000_000
    checks = {
        "160_pairs": len(pair_counts) == 160,
        "320_calls": len(calls) == 320,
        "two_arms_per_pair": set(pair_counts.values()) == {2},
        "component_pairs_40_each": component_pair_counts == Counter({"RS": 40, "MP": 40, "MS": 40, "ME": 40}),
        "same_seed_within_pair": all(
            len({row["generation"]["seed"] for row in calls if row["pair_id"] == pair_id}) == 1
            for pair_id in pair_counts
        ),
        "distinct_prompts_within_pair": all(
            len({row["prompt_sha256"] for row in calls if row["pair_id"] == pair_id}) == 2
            for pair_id in pair_counts
        ),
        "no_response_or_outcome": all(row["response"] is None and row["effect_label"].startswith("UNKNOWN") for row in registry),
    }
    if not all(checks.values()):
        raise RuntimeError(f"D3 generation-plan checks failed: {checks}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "call_plan.jsonl", calls)
    write_jsonl(out_dir / "generation_registry.jsonl", registry)
    plan_report = {
        "protocol": PROTOCOL,
        "status": STATUS,
        "api_calls_made": 0,
        "pairs": len(pair_counts),
        "planned_calls": len(calls),
        "component_pair_counts": dict(sorted(component_pair_counts.items())),
        "estimated_input_tokens": input_tokens,
        "maximum_output_tokens": output_tokens,
        "pricing_usd_per_mtok": PRICING_USD_PER_MTOK,
        "cost_upper_bound_usd": round(cost_upper, 6),
        "generator_identity": identity,
        "checks": checks,
        "call_plan_sha256": sha256_file(out_dir / "call_plan.jsonl"),
        "registry_sha256": sha256_file(out_dir / "generation_registry.jsonl"),
        "blueprint_manifest_sha256": sha256_file(blueprint_dir / "freeze_manifest.json"),
    }
    write_json(out_dir / "generation_plan_report.json", plan_report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": STATUS,
            "generator_identity": identity,
            "supporter_generation_treatment_sha256": generation.digest(),
            "outputs": {
                "call_plan.jsonl": sha256_file(out_dir / "call_plan.jsonl"),
                "generation_registry.jsonl": sha256_file(out_dir / "generation_registry.jsonl"),
                "generation_plan_report.json": sha256_file(out_dir / "generation_plan_report.json"),
            },
            "outcome_blind": True,
            "api_calls_made": 0,
        },
    )
    return plan_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blueprint-dir", type=Path, default=ROOT / "outputs/pm_v1_5_d3_step0_blueprint_v1")
    parser.add_argument("--strategy-cards", type=Path, default=ROOT / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl")
    parser.add_argument("--pm-config", type=Path, default=ROOT / "configs/pm_v1_5.yaml")
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs/experiment.yaml")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/pm_v1_5_d3_generation_v1")
    args = parser.parse_args()
    report = prepare(
        blueprint_dir=args.blueprint_dir,
        strategy_cards_path=args.strategy_cards,
        pm_config_path=args.pm_config,
        experiment_config_path=args.experiment_config,
        out_dir=args.out_dir,
    )
    print(json.dumps({key: report[key] for key in ("protocol", "status", "pairs", "planned_calls", "cost_upper_bound_usd")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
