#!/usr/bin/env python3
"""Create the frozen, evaluator-balanced PM-v2 API compatibility pilot matrix.

This script is strictly no-API.  Development-only regime annotations are used
only to balance the diagnostic sample; they never enter PM state features.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from metacom_pm.config import load_config
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import canonical_json, sha256_file, sha256_text, write_json
from metacom_pm.pm_v2_contracts import ResourceNeedRegime
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_audit import audit_deployable_feature_observability
from metacom_pm.pm_v2_semantic_audit import (
    require_pmv2_runtime_state_lineage,
    require_semantic_sanity_pass,
)


ROOT = Path(__file__).resolve().parents[1]


def _rank(seed: int, regime: str, state_id: str) -> str:
    return sha256_text(f"{seed}|{regime}|{state_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pm-v2-config", type=Path, default=ROOT / "configs" / "pm_v2.yaml"
    )
    parser.add_argument(
        "--states", type=Path, default=ROOT / "data" / "pm_v2" / "pm_v2_states.jsonl"
    )
    parser.add_argument(
        "--runtime", type=Path, default=ROOT / "data" / "pm_v2" / "runtime_states.jsonl"
    )
    parser.add_argument(
        "--backend", type=Path, default=ROOT / "data" / "pm_v2" / "memory_backend.jsonl"
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v2" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--strategy-bank",
        type=Path,
        default=ROOT / "data" / "strategy" / "strategy_cards.jsonl",
    )
    parser.add_argument(
        "--semantic-sanity-report",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "semantic_sanity_report.json",
    )
    parser.add_argument(
        "--semantic-sanity-attestation",
        type=Path,
        default=ROOT
        / "outputs"
        / "pm_v2_semantic_sanity"
        / "artifact_attestation.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "pm_v2_development_pilot" / "pilot_plan.json",
    )
    args = parser.parse_args()

    config = load_config(args.pm_v2_config)
    if config.get("version") != "pm-v2.2":
        raise RuntimeError("development pilot requires PM-v2.2 config")
    supporter_generation_contract = SupporterGenerationContract.from_config(config)
    pilot = dict(config["development_judging"]["compatibility_pilot"])
    states = load_states(args.states)
    contexts = load_evaluator_context_index(
        args.evaluator_contexts, states=states, require_exact=True
    )
    runtime_state_lineage = require_pmv2_runtime_state_lineage(
        args.runtime, states
    )
    semantic_sanity = require_semantic_sanity_pass(
        report_path=args.semantic_sanity_report,
        attestation_path=args.semantic_sanity_attestation,
        config=config,
        config_path=args.pm_v2_config,
        states_path=args.states,
        backend_path=args.backend,
        evaluator_contexts_path=args.evaluator_contexts,
    )
    deployable_feature_observability = audit_deployable_feature_observability(
        states,
        evaluator_contexts=contexts,
        settings=dict(pilot["deployable_feature_observability"]),
    )
    if deployable_feature_observability["status"] != "PASS":
        raise RuntimeError(
            "deployable PM-v2 features do not pass the frozen train-only "
            "observability gate: " + str(deployable_feature_observability)
        )
    state_by_id = {state.state_id: state for state in states}
    by_regime: dict[str, list] = defaultdict(list)
    for state_id, context in contexts.by_state.items():
        by_regime[str(context["regime"])].append(state_by_id[state_id])

    expected_regimes = {regime.value for regime in ResourceNeedRegime}
    if set(by_regime) != expected_regimes:
        raise RuntimeError(
            "pilot source does not exactly cover all regimes: "
            f"missing={sorted(expected_regimes-set(by_regime))}, "
            f"extra={sorted(set(by_regime)-expected_regimes)}"
        )
    count = int(pilot["states_per_regime"])
    seed = int(pilot["sample_seed"])
    actions = [str(value) for value in pilot["actions"]]
    if count < 1 or not actions or len(actions) != len(set(actions)):
        raise ValueError("invalid compatibility-pilot sample size or action list")

    selected = []
    for regime in sorted(expected_regimes):
        candidates = sorted(
            by_regime[regime], key=lambda state: _rank(seed, regime, state.state_id)
        )
        # Compatibility/feasibility results decide whether the expensive full
        # development run is allowed to proceed.  Restrict the whole pilot to
        # train states so calibration and internal-holdout labels cannot leak
        # into that decision.
        train_candidates = [
            state for state in candidates if state.split.value == "train"
        ]
        if len(train_candidates) < count:
            raise RuntimeError(
                f"regime {regime} has only {len(train_candidates)} train states "
                f"for a {count}-state compatibility pilot"
            )
        chosen = train_candidates[:count]
        if len(chosen) != count:
            raise RuntimeError(f"regime {regime} has only {len(chosen)} pilot states")
        for state in chosen:
            missing = sorted(set(actions) - set(state.allowed_actions))
            if missing:
                raise RuntimeError(
                    f"pilot state {state.state_id} lacks configured actions: {missing}"
                )
            selected.append(
                {
                    "regime": regime,
                    "state_id": state.state_id,
                    "card_id": state.card_id,
                    "split": state.split.value,
                    "actions": actions,
                }
            )

    expected_keys = sorted(
        [row["card_id"], action]
        for row in selected
        for action in row["actions"]
    )
    payload = {
        "status": "READY",
        "protocol": "pm_v2_development_compatibility_pilot_v2_treatment_bound",
        "pm_v2_config_sha256": sha256_file(args.pm_v2_config),
        "supporter_generation_treatment": supporter_generation_contract.payload(),
        "supporter_generation_treatment_sha256": (
            supporter_generation_contract.digest()
        ),
        "states_sha256": sha256_file(args.states),
        "runtime_sha256": sha256_file(args.runtime),
        "backend_sha256": sha256_file(args.backend),
        "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
        "evaluator_contexts_map_sha256": contexts.map_sha256,
        "strategy_bank_sha256": sha256_file(args.strategy_bank),
        "semantic_sanity": semantic_sanity,
        "runtime_state_lineage": runtime_state_lineage,
        "deployable_feature_observability": deployable_feature_observability,
        "sample_seed": seed,
        "states_per_regime": count,
        "actions": actions,
        "selected_states": selected,
        "expected_state_count": len(selected),
        "expected_outcome_count": len(expected_keys),
        "expected_keys_sha256": sha256_text(canonical_json(expected_keys)),
    }
    payload["pilot_plan_sha256"] = sha256_text(canonical_json(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    print(payload)


if __name__ == "__main__":
    main()
