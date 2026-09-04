#!/usr/bin/env python3
"""Audit a fit-only dual-domain candidate before any holdout is opened."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.config import load_config
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from metacom_pm.pm_v2_contracts import ActionLabel, PMV2Split
from metacom_pm.pm_v2_data import load_states
from metacom_pm.pm_v2_model import (
    PMV2Model,
    applicable_risk_fields,
    estimated_action_cost_profile,
)
from metacom_pm.v1_5_dual_domain_training import (
    ESCONV_AUXILIARY_DOMAIN,
    LONGITUDINAL_DOMAIN,
    calibration_action_viability,
    training_domain_for_state,
)


PROTOCOL = "pm-v1.5-dual-domain-fit-only-candidate-audit-v1"


def _artifact(candidate: dict[str, Any], name: str) -> Path:
    row = dict(candidate["artifacts"][name])
    path = Path(str(row["path"]))
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"candidate artifact drifted: {name}")
    return path


def _labels(path: Path) -> list[ActionLabel]:
    return [ActionLabel.model_validate(row) for row in iter_jsonl(path)]


def _oracle_opportunity(
    model: PMV2Model,
    states,
    labels,
    *,
    objective_risk_weight: float,
    objective_cost_weight: float,
) -> dict[str, Any]:
    label_map = {(label.state_id, label.action_id): label for label in labels}
    expected = {
        (state.state_id, action_id)
        for state in states
        for action_id in state.allowed_actions
    }
    if set(label_map) != expected:
        raise RuntimeError("candidate audit requires the exact calibration matrix")
    oracle_distribution: Counter[str] = Counter()
    predicted_quality_argmax: Counter[str] = Counter()
    predicted_utility_argmax: Counter[str] = Counter()
    headroom: list[float] = []
    for state in sorted(states, key=lambda row: row.state_id):
        predictions = model.predict_actions(state)
        predicted_quality_argmax[
            max(
                state.allowed_actions,
                key=lambda action: (
                    predictions[action].quality_mean,
                    action,
                ),
            )
        ] += 1
        predicted_utility_argmax[
            max(
                state.allowed_actions,
                key=lambda action: (predictions[action].utility, action),
            )
        ] += 1
        costs = estimated_action_cost_profile(model.feature_builder, state)
        realized: dict[str, float] = {}
        for action_id in state.allowed_actions:
            label = label_map[(state.state_id, action_id)]
            quality = model.selection_config.composite_spec.score(label.response)
            risk = max(
                float(getattr(label.risk, name)) / 3.0
                for name in applicable_risk_fields(action_id)
            )
            realized[action_id] = (
                quality
                - float(objective_risk_weight) * risk
                - float(objective_cost_weight)
                * costs[action_id]["normalized_estimated_resource_cost"]
            )
        oracle = max(realized, key=lambda action: (realized[action], action))
        oracle_distribution[oracle] += 1
        baseline = realized.get("M0+R0")
        if baseline is None:
            raise RuntimeError("candidate audit requires M0+R0 in every state")
        headroom.append(realized[oracle] - baseline)
    return {
        "state_count": len(states),
        "objective_risk_weight": float(objective_risk_weight),
        "objective_cost_weight": float(objective_cost_weight),
        "label_oracle_action_distribution": dict(
            sorted(oracle_distribution.items())
        ),
        "label_oracle_non_m0_r0_count": int(
            len(states) - oracle_distribution.get("M0+R0", 0)
        ),
        "label_oracle_non_m0_r0_rate": float(
            1.0 - oracle_distribution.get("M0+R0", 0) / len(states)
        ),
        "mean_label_oracle_headroom_vs_m0_r0": float(
            sum(headroom) / len(headroom)
        ),
        "positive_label_oracle_headroom_count": int(
            sum(value > 1e-12 for value in headroom)
        ),
        "label_oracle_headroom_above_0_01_count": int(
            sum(value > 0.01 for value in headroom)
        ),
        "maximum_label_oracle_headroom_vs_m0_r0": float(max(headroom)),
        "predicted_quality_argmax_distribution": dict(
            sorted(predicted_quality_argmax.items())
        ),
        "predicted_utility_argmax_distribution": dict(
            sorted(predicted_utility_argmax.items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    fit = read_json(args.fit_report)
    if (
        fit.get("status") != "CANDIDATE_FROZEN_BEFORE_INTERNAL_TEST"
        or fit.get("fit_only") is not True
        or fit.get("internal_test_outcomes_opened") is not False
    ):
        raise RuntimeError("candidate audit requires an outcome-blind fit-only report")
    candidate_path = Path(str(fit["candidate_manifest"]))
    if sha256_file(candidate_path) != fit["candidate_manifest_sha256"]:
        raise RuntimeError("fit-only candidate manifest drifted")
    candidate = read_json(candidate_path)
    if (
        candidate.get("status") != "FROZEN_BEFORE_INTERNAL_TEST"
        or candidate.get("run_identity") != fit.get("run_identity")
        or candidate["parameters"].get("internal_test_outcomes_opened") is not False
    ):
        raise RuntimeError("candidate manifest is not an outcome-blind frozen family")

    config_path = _artifact(candidate, "pm_v1_5_config")
    config = load_config(config_path)
    model_path = _artifact(candidate, "primary_checkpoint")
    model = PMV2Model.load(model_path)
    long_states_path = _artifact(candidate, "longitudinal_states")
    long_labels_path = _artifact(
        candidate, "longitudinal_train_calibration_labels"
    )
    auxiliary_states_path = _artifact(candidate, "auxiliary_calibration_states")
    auxiliary_labels_path = _artifact(
        candidate, "auxiliary_calibration_labels"
    )
    states = [
        state
        for state in [
            *load_states(long_states_path),
            *load_states(auxiliary_states_path),
        ]
        if state.split is PMV2Split.CALIBRATION
    ]
    labels = [*_labels(long_labels_path), *_labels(auxiliary_labels_path)]
    action_preflight = config["external_evaluation"]["action_preflight"]
    calibration_rows = fit["calibration_comparators"]
    grid = config["calibration_grid"]
    domains: dict[str, Any] = {}
    for domain in (LONGITUDINAL_DOMAIN, ESCONV_AUXILIARY_DOMAIN):
        domain_states = [
            state for state in states if training_domain_for_state(state) == domain
        ]
        state_ids = {state.state_id for state in domain_states}
        domain_labels = [
            label for label in labels if label.state_id in state_ids
        ]
        domains[domain] = {
            "action_viability": calibration_action_viability(
                calibration_rows[domain]["policy"],
                action_preflight=action_preflight,
            ),
            "label_and_prediction_opportunity": _oracle_opportunity(
                model,
                domain_states,
                domain_labels,
                objective_risk_weight=float(grid["objective_risk_weight"]),
                objective_cost_weight=float(grid["objective_cost_weight"]),
            ),
        }
    supported = all(
        row["action_viability"]["status"] == "PASS"
        for row in domains.values()
    )
    binding = {
        "protocol": PROTOCOL,
        "fit_report": {
            "path": str(args.fit_report),
            "sha256": sha256_file(args.fit_report),
        },
        "candidate_manifest": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "primary_checkpoint": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    report = {
        "protocol": PROTOCOL,
        "status": (
            "SUPPORTED_FOR_ONE_TIME_INTERNAL_CONSUMPTION"
            if supported
            else "NOT_SUPPORTED_FOR_INTERNAL_TEST_CONSUMPTION"
        ),
        "supervision_status": fit["supervision_status"],
        "internal_test_outcomes_opened": False,
        "run_identity": fit["run_identity"],
        "binding": binding,
        "binding_sha256": sha256_text(canonical_json(binding)),
        "domains": domains,
        "stopping_rule": (
            "do_not_generate_or_open_internal outcomes unless every domain "
            "passes the frozen action-preflight guardrails"
        ),
        "interpretation_boundary": (
            "This is a pre-holdout deployability/learnability screen, not an "
            "efficacy claim and not evidence of optimal routing."
        ),
    }
    core = dict(report)
    report["report_sha256"] = sha256_text(canonical_json(core))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "candidate_viability_report.json", report)
    print(report["status"])


if __name__ == "__main__":
    main()
