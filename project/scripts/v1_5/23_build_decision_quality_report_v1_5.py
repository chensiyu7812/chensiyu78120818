#!/usr/bin/env python3
"""Build the no-API PM-v1.5 internal decision-quality diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from metacom_pm.artifacts import create_artifact_attestation
from metacom_pm.config import load_config
from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json
from metacom_pm.pm_v2_contracts import ActionLabel, PMV2Split
from metacom_pm.pm_v2_data import load_evaluator_context_index, load_states
from metacom_pm.pm_v2_decision_quality import build_decision_quality_report
from metacom_pm.pm_v2_judging import composite_spec_from_config
from metacom_pm.pm_v2_model import PMV2Model


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-internal-decision-quality-v1"


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, replicates: int, seed: int, confidence: float
) -> dict[str, Any]:
    users = sorted({str(row["user_id"]) for row in rows})
    if len(users) < 2:
        raise RuntimeError("decision-quality bootstrap requires at least two users")
    by_user = {user: [row for row in rows if str(row["user_id"]) == user] for user in users}
    metrics = {
        "mean_regret": ("regret", False),
        "oracle_hit_rate": ("oracle_hit", False),
        "quality_acceptable_rate": ("chosen_quality_acceptable", False),
        "mean_source_f1": ("source_f1", False),
        "m0_accuracy": ("m0_correct", True),
        "rs_accuracy": ("rs_correct", True),
        "mean_excess_cost_if_quality_acceptable": (
            "excess_cost_if_quality_acceptable",
            True,
        ),
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        selected = rng.choice(users, size=len(users), replace=True)
        sampled = [row for user in selected for row in by_user[str(user)]]
        for metric, (field, skip_none) in metrics.items():
            values = [
                float(row[field])
                for row in sampled
                if not (skip_none and row.get(field) is None)
            ]
            if values:
                samples[metric].append(float(np.mean(values)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "protocol": "user-cluster-bootstrap-v1",
        "cluster": "user_id",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence,
        "intervals": {
            metric: {
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
            }
            for metric, values in samples.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pm-v1-5-config", type=Path, default=ROOT / "configs" / "pm_v1_5.yaml"
    )
    parser.add_argument(
        "--states", type=Path, default=ROOT / "data" / "pm_v1_5" / "pm_v2_states.jsonl"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_judging" / "action_labels.jsonl",
    )
    parser.add_argument(
        "--evaluator-contexts",
        type=Path,
        default=ROOT / "data" / "pm_v1_5" / "evaluator_contexts.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "pm_v1_5.joblib",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_model" / "training_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pm_v1_5_decision_quality",
    )
    args = parser.parse_args()

    config = load_config(args.pm_v1_5_config)
    if config.get("version") != "pm-v1.5":
        raise RuntimeError("decision-quality report requires PM-v1.5")
    settings = dict(config["decision_quality"])
    if settings.get("protocol") != PROTOCOL or settings.get("split") != "internal_test":
        raise RuntimeError("PM-v1.5 decision-quality contract is invalid")

    training_report = read_json(args.training_report)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if (
        training_report.get("status") != "COMPLETE"
        or training_report.get("checkpoint_sha256") != checkpoint_sha256
        or training_report.get("pm_v2_config_sha256")
        != sha256_file(args.pm_v1_5_config)
        or training_report.get("learned_routing_advantage_verified") is not True
    ):
        raise RuntimeError("decision-quality report requires the frozen reportable policy")

    all_states = load_states(args.states)
    states = [state for state in all_states if state.split is PMV2Split.INTERNAL_TEST]
    if not states:
        raise RuntimeError("decision-quality report has no internal-test states")
    evaluator_index = load_evaluator_context_index(
        args.evaluator_contexts, states=all_states, require_exact=True
    )
    labels = [ActionLabel.model_validate(row) for row in iter_jsonl(args.labels)]
    labels_by_state: dict[str, dict[str, ActionLabel]] = defaultdict(dict)
    for label in labels:
        if label.action_id in labels_by_state[label.state_id]:
            raise RuntimeError("decision-quality labels contain duplicate state/action")
        labels_by_state[label.state_id][label.action_id] = label
    for state in states:
        observed = set(labels_by_state.get(state.state_id, {}))
        if observed != set(state.allowed_actions):
            raise RuntimeError(
                f"decision-quality action matrix is incomplete for {state.state_id}"
            )

    model = PMV2Model.load(args.checkpoint)
    if model.selection_config.digest() != training_report.get("selection_config_hash"):
        raise RuntimeError("decision-quality checkpoint selection config is stale")
    chosen = {state.state_id: model.choose(state).chosen_action for state in states}
    raw_costs = {
        state.state_id: {
            action: float(label.observed_input_tokens)
            for action, label in labels_by_state[state.state_id].items()
        }
        for state in states
    }
    normalized_costs = {}
    for state_id, costs in raw_costs.items():
        denominator = max(max(costs.values(), default=0.0), 1.0)
        normalized_costs[state_id] = {
            action: cost / denominator for action, cost in costs.items()
        }
    evaluator = evaluator_index.by_state
    core = build_decision_quality_report(
        states=[state.state_id for state in states],
        labels_by_state=labels_by_state,
        chosen_action_by_state=chosen,
        cost_by_state_action=normalized_costs,
        resource_cost_by_state_action=raw_costs,
        regime_by_state={state.state_id: str(evaluator[state.state_id]["regime"]) for state in states},
        needed_memory_sources_by_state={
            state.state_id: list(evaluator[state.state_id]["needed_memory_sources"])
            for state in states
        },
        user_id_by_state={state.state_id: state.user_id for state in states},
        composite_spec=composite_spec_from_config(config),
        risk_weight=float(config["selection"]["risk_weight"]),
        cost_weight=float(config["selection"]["cost_weight"]),
        epsilon=float(settings["utility_oracle_epsilon"]),
        quality_epsilon=float(settings["quality_acceptability_epsilon"]),
    )
    bootstrap = _cluster_bootstrap(
        list(core["rows"]),
        replicates=int(settings["bootstrap_replicates"]),
        seed=int(settings["bootstrap_seed"]),
        confidence=float(settings["bootstrap_confidence_level"]),
    )
    report = {
        "status": "COMPLETE",
        "protocol": PROTOCOL,
        "track": "pm-v1.5",
        "claim_boundary": (
            "synthetic_internal_decision_quality_diagnostic_not_human_validated_"
            "clinical_or_real_world_resource_correctness"
        ),
        "split": "internal_test",
        "pm_v1_5_config_sha256": sha256_file(args.pm_v1_5_config),
        "states_sha256": sha256_file(args.states),
        "labels_sha256": sha256_file(args.labels),
        "evaluator_contexts_sha256": sha256_file(args.evaluator_contexts),
        "checkpoint_sha256": checkpoint_sha256,
        "training_report_sha256": sha256_file(args.training_report),
        "settings": settings,
        "bootstrap": bootstrap,
        **core,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "decision_quality_report.json"
    attestation_path = args.out_dir / "artifact_attestation.json"
    write_json(report_path, report)
    create_artifact_attestation(
        attestation_path,
        stage="pm_v1_5_decision_quality",
        inputs={
            "pm_v1_5_config": args.pm_v1_5_config,
            "states": args.states,
            "labels": args.labels,
            "evaluator_contexts": args.evaluator_contexts,
            "checkpoint": args.checkpoint,
            "training_report": args.training_report,
        },
        outputs={"report": (report_path, False)},
        parameters={"protocol": PROTOCOL, "settings": settings},
        expected={"split": "internal_test", "states": len(states)},
    )
    print(report)


if __name__ == "__main__":
    main()
