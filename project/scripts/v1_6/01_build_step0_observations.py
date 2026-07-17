#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.contracts import MemoryBackendRecord, RuntimeState, StrategyCard
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.pm_v1_6_step0 import FrozenStep0Config, build_step0_observation

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pm_v1_6.yaml")
    parser.add_argument("--runtime-states", type=Path, required=True)
    parser.add_argument("--memory-backend", type=Path, required=True)
    parser.add_argument("--strategy-bank", type=Path, default=ROOT / "data" / "strategy" / "strategy_cards_v1_5.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--catalog-build-amortized-ms", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.get("version") != "pm-v1.6":
        raise RuntimeError("Step-0 builder requires PM-v1.6")
    step_cfg = config["step0"]
    frozen = FrozenStep0Config(
        dimension=int(step_cfg["dimension"]),
        word_features=int(step_cfg["word_features"]),
        char_features=int(step_cfg["char_features"]),
    )
    backends = {
        row.card_id: row
        for row in (
            MemoryBackendRecord.model_validate(raw)
            for raw in iter_jsonl(args.memory_backend)
        )
    }
    strategies = [StrategyCard.model_validate(raw) for raw in iter_jsonl(args.strategy_bank)]
    observations = []
    audits = []
    seen_states = set()
    for raw in iter_jsonl(args.runtime_states):
        state = RuntimeState.model_validate(raw)
        if state.state_id in seen_states:
            raise RuntimeError(f"duplicate runtime state: {state.state_id}")
        seen_states.add(state.state_id)
        if state.card_id not in backends:
            raise RuntimeError(f"missing memory backend for {state.card_id}")
        observation, audit = build_step0_observation(
            runtime_state=state,
            memory_items=backends[state.card_id].items,
            strategy_cards=strategies,
            config=frozen,
            catalog_build_amortized_ms=args.catalog_build_amortized_ms,
        )
        observations.append(observation.model_dump(mode="json"))
        audits.append(audit.model_dump(mode="json"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    observation_path = args.out_dir / "step0_observations.jsonl"
    audit_path = args.out_dir / "step0_audit_bindings.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(observation_path, observations)
    write_jsonl(audit_path, audits)
    write_json(
        summary_path,
        {
            "status": "COMPLETE",
            "protocol": step_cfg["protocol"],
            "states": len(observations),
            "step0_config": frozen.payload(),
            "step0_config_sha256": frozen.digest(),
            "runtime_states_sha256": sha256_file(args.runtime_states),
            "memory_backend_sha256": sha256_file(args.memory_backend),
            "strategy_bank_sha256": sha256_file(args.strategy_bank),
            "observations_sha256": sha256_file(observation_path),
            "audit_bindings_sha256": sha256_file(audit_path),
            "pm_visible_contains_raw_representations": False,
        },
    )


if __name__ == "__main__":
    main()
