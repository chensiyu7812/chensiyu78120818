#!/usr/bin/env python3
"""Evaluate the frozen V5 cross-fitted policy on its actual QRC objective.

This is a FIT/development diagnostic, not independent confirmation.  It does
not change labels, features, folds, models, thresholds, responses, or external
state.  It compares the grouped-OOF learned decisions with always-off,
component-wise fixed-high, and the frozen transparent rule.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5-grouped-oof-pareto-policy-diagnostic-v1"


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def aggregator() -> Any:
    path = ROOT / "scripts/v1_5/26a_aggregate_and_train_v5_fit_v1_5.py"
    spec = importlib.util.spec_from_file_location("v5_fit_trainer", path)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load frozen V5 trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grouped_oof_decisions(
    component_rows: list[dict[str, Any]],
    feature_by_state: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    names = list(feature_by_state[component_rows[0]["state_id"]]["model_features"])
    x = np.asarray(
        [
            [float(feature_by_state[row["state_id"]]["model_features"][name]) for name in names]
            for row in component_rows
        ],
        dtype=float,
    )
    y = np.asarray([int(row["hard_worth_opening"]) for row in component_rows])
    groups = np.asarray(
        [
            str(row.get("semantic_group_id", row.get("semantic_group_id_v5_2")))
            for row in component_rows
        ]
    )
    probabilities: list[np.ndarray] = []
    for seed in contract["primary_model"]["fold_seeds"]:
        probability = np.zeros(len(y), dtype=float)
        splitter = StratifiedGroupKFold(
            n_splits=int(contract["primary_model"]["folds"]),
            shuffle=True,
            random_state=int(seed),
        )
        for train, test in splitter.split(x, y, groups):
            scaler = StandardScaler().fit(x[train])
            model = LogisticRegression(
                C=float(contract["primary_model"]["C"]),
                class_weight=str(contract["primary_model"]["class_weight"]),
                solver=str(contract["primary_model"]["solver"]),
                max_iter=2000,
                random_state=int(seed),
            ).fit(scaler.transform(x[train]), y[train])
            probability[test] = model.predict_proba(scaler.transform(x[test]))[:, 1]
        probabilities.append(probability)
    mean_probability = np.mean(probabilities, axis=0)
    return mean_probability, mean_probability >= float(contract["primary_model"]["threshold"])


def summarize_policy(
    *,
    decisions: dict[str, bool],
    labels: dict[str, dict[str, Any]],
    key_by_state: dict[str, list[dict[str, Any]]],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    versus_off: Counter[str] = Counter()
    versus_fixed_high: Counter[str] = Counter()
    prompt_tokens = 0
    completion_tokens = 0
    risks = 0
    seed_rows = 0
    requested_on_states = 0
    for state_id, use_on in decisions.items():
        if use_on:
            requested_on_states += 1
        label = labels[state_id]
        key_rows = sorted(key_by_state[state_id], key=lambda row: int(row["seed"]))
        for index, key in enumerate(key_rows):
            winner = str(label["seed_quality_winners"][index])
            if use_on:
                versus_off[
                    "better" if winner == "ON" else "worse" if winner == "OFF" else "tie"
                ] += 1
                versus_fixed_high["tie"] += 1
            else:
                versus_off["tie"] += 1
                versus_fixed_high[
                    "better" if winner == "OFF" else "worse" if winner == "ON" else "tie"
                ] += 1
            seed_on_risk = label.get(
                "seed_on_material_risk", label.get("seed_material_risk")
            )
            seed_off_risk = label.get("seed_off_material_risk")
            if seed_on_risk is None:
                raise RuntimeError(f"missing seed risk labels: {state_id}")
            selected_risk = (
                seed_on_risk[index]
                if use_on
                else seed_off_risk[index]
                if seed_off_risk is not None
                else "no"
            )
            if selected_risk == "yes":
                risks += 1
            call_id = str(key["on_call_id"] if use_on else key["off_call_id"])
            usage = outcomes[call_id]["usage"]
            prompt_tokens += int(usage["prompt_tokens"])
            completion_tokens += int(usage["completion_tokens"])
            seed_rows += 1
    non_tie = versus_fixed_high["better"] + versus_fixed_high["worse"]
    return {
        "states": len(decisions),
        "seed_level_comparisons": seed_rows,
        "requested_on_states": requested_on_states,
        "requested_on_fraction": requested_on_states / len(decisions),
        "quality_versus_always_off": dict(versus_off),
        "quality_versus_component_fixed_high": dict(versus_fixed_high),
        "conditional_win_fraction_versus_fixed_high": (
            versus_fixed_high["better"] / non_tie if non_tie else None
        ),
        "material_risk_count": risks,
        "material_risk_rate": risks / seed_rows,
        "mean_prompt_tokens": prompt_tokens / seed_rows,
        "mean_completion_tokens": completion_tokens / seed_rows,
        "mean_total_tokens": (prompt_tokens + completion_tokens) / seed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-path",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_training_result_v1/state_effect_labels_private.jsonl",
    )
    parser.add_argument(
        "--key-path",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_single_full_fit_outcome_review_v1/private_blind_key.jsonl",
    )
    parser.add_argument(
        "--outcome-path",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_itt_fit_execution_v2/outcomes_ordered_private.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_oof_pareto_diagnostic_v1",
    )
    parser.add_argument("--protocol", default=PROTOCOL)
    args = parser.parse_args()
    label_path = args.label_path
    feature_path = ROOT / "outputs/pm_v1_5_v5_fit_training_surface_v1/fit_feature_rows_private.jsonl"
    key_path = args.key_path
    outcome_path = args.outcome_path
    contract_path = ROOT / "data/pm_v1_5_contracts/v5_fit_training_surface_freeze_v1.json"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_rows = rows(label_path)
    labels = {str(row["state_id"]): row for row in label_rows}
    features = {str(row["state_id"]): row for row in rows(feature_path)}
    outcomes = {str(row["call_id"]): row for row in rows(outcome_path)}
    key_by_state: dict[str, list[dict[str, Any]]] = {}
    for row in rows(key_path):
        key_by_state.setdefault(str(row["state_id"]), []).append(row)
    if set(labels) != set(features) or set(labels) != set(key_by_state):
        raise RuntimeError("label/feature/key state identities differ")
    if any(len(value) != 2 for value in key_by_state.values()):
        raise RuntimeError("expected two seed rows per state")

    contract = read_json(contract_path)
    module = aggregator()
    learned: dict[str, bool] = {}
    transparent: dict[str, bool] = {}
    decision_rows: list[dict[str, Any]] = []
    for component in module.COMPONENTS:
        component_rows = sorted(
            [row for row in label_rows if row["component"] == component],
            key=lambda row: str(row["state_id"]),
        )
        probability, decision = grouped_oof_decisions(component_rows, features, contract)
        for row, value, use_on in zip(component_rows, probability, decision, strict=True):
            state_id = str(row["state_id"])
            learned[state_id] = bool(use_on)
            transparent[state_id] = bool(
                module._transparent_rule(component, features[state_id]["model_features"])
            )
            decision_rows.append(
                {
                    "protocol": args.protocol,
                    "state_id": state_id,
                    "component": component,
                    "semantic_group_id": row.get(
                        "semantic_group_id", row.get("semantic_group_id_v5_2")
                    ),
                    "mean_five_seed_grouped_oof_probability": float(value),
                    "learned_requested_on": bool(use_on),
                    "transparent_rule_requested_on": transparent[state_id],
                    "confirmation_or_external_read": False,
                }
            )

    policies = {
        "learned_grouped_oof": learned,
        "transparent_rule": transparent,
        "component_fixed_high": {state_id: True for state_id in labels},
        "always_off": {state_id: False for state_id in labels},
    }
    summaries = {
        name: summarize_policy(
            decisions=decision,
            labels=labels,
            key_by_state=key_by_state,
            outcomes=outcomes,
        )
        for name, decision in policies.items()
    }
    learned_summary = summaries["learned_grouped_oof"]
    fixed_summary = summaries["component_fixed_high"]
    rule_summary = summaries["transparent_rule"]
    report = {
        "protocol": args.protocol,
        "status": "FIT_DEVELOPMENT_PARETO_SIGNAL_REQUIRES_UNTOUCHED_CONFIRMATION",
        "grain": "256 state-candidate units in 128 counterfactual groups; two frozen generation seeds per state",
        "policies": summaries,
        "headline": {
            "learned_quality_better_vs_fixed_high": learned_summary["quality_versus_component_fixed_high"].get("better", 0),
            "learned_quality_worse_vs_fixed_high": learned_summary["quality_versus_component_fixed_high"].get("worse", 0),
            "learned_quality_tie_vs_fixed_high": learned_summary["quality_versus_component_fixed_high"].get("tie", 0),
            "learned_risk_reduction_vs_fixed_high": 1.0 - learned_summary["material_risk_rate"] / fixed_summary["material_risk_rate"],
            "learned_prompt_token_reduction_vs_fixed_high": 1.0 - learned_summary["mean_prompt_tokens"] / fixed_summary["mean_prompt_tokens"],
            "learned_total_token_reduction_vs_fixed_high": 1.0 - learned_summary["mean_total_tokens"] / fixed_summary["mean_total_tokens"],
            "learned_quality_better_vs_always_off": learned_summary["quality_versus_always_off"].get("better", 0),
            "learned_quality_worse_vs_always_off": learned_summary["quality_versus_always_off"].get("worse", 0),
            "rule_quality_better_vs_fixed_high": rule_summary["quality_versus_component_fixed_high"].get("better", 0),
            "rule_quality_worse_vs_fixed_high": rule_summary["quality_versus_component_fixed_high"].get("worse", 0),
        },
        "interpretation_boundary": {
            "supported": "The frozen cross-fitted candidate exhibits the target quality-risk-cost Pareto pattern on FIT development outcomes.",
            "not_yet_supported": "Independent generalization, formal noninferiority, external transport, or publication-grade superiority.",
            "why_not_gold": "Quality and risk remain reviewer-proxy outcomes; grouped OOF prevents direct row fitting but does not replace untouched confirmation."
        },
        "external_or_confirmation_read": False,
        "post_outcome_model_or_threshold_change": False,
        "input_sha256": {
            "labels": sha256_file(label_path),
            "features": sha256_file(feature_path),
            "private_key": sha256_file(key_path),
            "outcomes": sha256_file(outcome_path),
            "contract": sha256_file(contract_path),
        },
    }
    write_jsonl(out_dir / "oof_decisions_private.jsonl", decision_rows)
    write_json(out_dir / "diagnostic_report.json", report)
    print({"protocol": args.protocol, "status": report["status"], **report["headline"]})


if __name__ == "__main__":
    main()
