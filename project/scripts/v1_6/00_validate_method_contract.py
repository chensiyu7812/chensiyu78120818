#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.io import canonical_json, sha256_text, write_json
from metacom_pm.pm_v1_6_contracts import (
    ALGORITHM_SELECTION_PROTOCOL,
    CLAIM_PROTOCOL,
    COST_VECTOR_PROTOCOL,
    PROTOCOL_VERSION,
    STEP0_PROTOCOL,
    STRATEGY_FAMILIES,
)
from metacom_pm.pm_v1_6_judge_isolation import (
    JudgeIsolationPolicy,
    require_development_judge_isolation,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pm_v1_6.yaml")
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "pm_v1_6_contract_validation.json")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = load_config(args.experiment_config)
    if config.get("version") != PROTOCOL_VERSION:
        raise RuntimeError("method contract is not PM-v1.6")
    if config["step0"]["protocol"] != STEP0_PROTOCOL:
        raise RuntimeError("Step-0 protocol mismatch")
    if config["step0"]["strategy_family_order"] != list(STRATEGY_FAMILIES):
        raise RuntimeError("strategy-family order differs from code")
    if config["cost_vector"]["protocol"] != COST_VECTOR_PROTOCOL:
        raise RuntimeError("cost-vector protocol mismatch")
    if config["algorithm_competition"]["protocol"] != ALGORITHM_SELECTION_PROTOCOL:
        raise RuntimeError("algorithm-selection protocol mismatch")
    if config["external_claims"]["protocol"] != CLAIM_PROTOCOL:
        raise RuntimeError("claim protocol mismatch")

    isolation_cfg = config["judge_isolation"]
    aliases = list(isolation_cfg["development_endpoints"])
    endpoints = {alias: endpoint_from_config(experiment, alias) for alias in aliases}
    isolation = require_development_judge_isolation(
        endpoints,
        aliases,
        policy=JudgeIsolationPolicy(
            development_families=frozenset(isolation_cfg["development_families"]),
            final_families=frozenset(isolation_cfg["final_families"]),
            final_model_markers=tuple(isolation_cfg["final_model_markers"]),
        ),
    )
    payload = {
        "status": "PASS",
        "version": PROTOCOL_VERSION,
        "config": str(args.config),
        "config_sha256": sha256_text(canonical_json(config)),
        "judge_isolation": isolation,
        "external_condition_matrix": config["external_evaluation"]["condition_matrix"],
        "old_v1_5_artifact_status": config["supersedes"]["old_artifact_status"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)


if __name__ == "__main__":
    main()
