#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from metacom_pm.config import load_config
from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json
from metacom_pm.pm_v1_6_contracts import Step0Observation
from metacom_pm.pm_v1_6_internal import (
    finish_internal_test,
    reserve_internal_test_once,
)
from metacom_pm.pm_v1_6_models import OverrideConfig, PMV16OutcomeModel
from metacom_pm.pm_v1_6_router import (
    StrongTransparentRouter,
    TransparentRouterConfig,
)
from metacom_pm.pm_v1_6_training import evaluate_observed_policy
from metacom_pm.pm_v2_contracts import ActionLabel
from metacom_pm.pm_v2_data import load_states
from metacom_pm.pm_v2_model import applicable_risk_fields

ROOT = Path(__file__).resolve().parents[2]


def _load_step0(path: Path) -> dict[str, Step0Observation]:
    rows = [Step0Observation.model_validate(row) for row in iter_jsonl(path)]
    result = {row.state_id: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate Step-0 observation")
    return result


def _load_labels(path: Path) -> list[ActionLabel]:
    return [ActionLabel.model_validate(row) for row in iter_jsonl(path)]


def _cluster_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    seed: int,
    replicates: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap equal-weight user means, preserving all within-user dependence."""

    if not rows:
        raise ValueError("paired bootstrap rows are empty")
    by_user: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_user[str(row["user_id"])].append(float(row[field]))
    user_means = {
        user: float(np.mean(values)) for user, values in sorted(by_user.items())
    }
    users = list(user_means)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=float)
    for index in range(int(replicates)):
        sampled = rng.choice(users, size=len(users), replace=True)
        draws[index] = float(
            np.mean([user_means[str(user)] for user in sampled])
        )
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "estimate": float(np.mean(list(user_means.values()))),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
        "n_user_clusters": len(users),
        "n_state_rows": len(rows),
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "estimand": "equal_weight_user_mean_of_within_user_state_deltas",
    }


def _baseline_rows(
    *,
    states,
    labels: Sequence[ActionLabel],
    step0_by_state: Mapping[str, Step0Observation],
    model: PMV16OutcomeModel,
    router: StrongTransparentRouter,
    override: OverrideConfig,
    baseline: str,
    skip_ineligible: bool = False,
) -> list[dict[str, Any]]:
    label_by_key = {(row.state_id, row.action_id): row for row in labels}
    if len(label_by_key) != len(labels):
        raise RuntimeError("duplicate internal action labels")
    rows = []
    for state in states:
        step0 = step0_by_state[state.state_id]
        if baseline == "strong_rule":
            action = router.choose(step0).action_id
        elif baseline == "me_r0":
            action = "ME+R0"
        else:
            raise ValueError(f"unknown internal baseline: {baseline}")
        if action not in state.allowed_actions:
            if skip_ineligible:
                continue
            raise RuntimeError(
                f"internal baseline {baseline} selected illegal action {action} "
                f"for {state.state_id}"
            )
        label = label_by_key[(state.state_id, action)]
        maximum_cost = max(
            model.feature_builder.estimate_action_cost(candidate, step0)
            for candidate in state.allowed_actions
        )
        estimated_cost = model.feature_builder.estimate_action_cost(action, step0)
        quality = model.composite_spec.score(label.response)
        emotional_support = (float(label.response.emotional_support) - 1.0) / 4.0
        risk = max(
            float(getattr(label.risk, field)) / 3.0
            for field in applicable_risk_fields(action)
        )
        normalized_cost = estimated_cost / max(maximum_cost, 1.0)
        utility = (
            quality
            - override.risk_weight * risk
            - override.cost_weight * normalized_cost
        )
        rows.append(
            {
                "state_id": state.state_id,
                "user_id": state.user_id,
                "action": action,
                "quality": quality,
                "emotional_support": emotional_support,
                "risk": risk,
                "normalized_cost": normalized_cost,
                "utility": utility,
            }
        )
    return rows


def _paired_deltas(
    learned_rows,
    baseline_rows,
    *,
    allow_baseline_subset: bool = False,
) -> list[dict[str, Any]]:
    learned = {row.state_id: row for row in learned_rows}
    baseline = {str(row["state_id"]): row for row in baseline_rows}
    if allow_baseline_subset:
        if not set(baseline) or not set(baseline) <= set(learned):
            raise RuntimeError(
                "baseline subset is empty or outside learned state universe"
            )
        state_ids = sorted(baseline)
    else:
        if set(learned) != set(baseline):
            raise RuntimeError(
                "learned and baseline internal state universes differ"
            )
        state_ids = sorted(learned)
    rows = []
    for state_id in state_ids:
        left = learned[state_id]
        right = baseline[state_id]
        if left.user_id != str(right["user_id"]):
            raise RuntimeError("paired internal user IDs differ")
        rows.append(
            {
                "state_id": state_id,
                "user_id": left.user_id,
                "quality_delta": float(left.quality - float(right["quality"])),
                "emotional_support_delta": float(
                    left.emotional_support - float(right["emotional_support"])
                ),
                "risk_delta": float(left.risk - float(right["risk"])),
                "normalized_cost_delta": float(
                    left.normalized_cost - float(right["normalized_cost"])
                ),
                "utility_delta": float(left.utility - float(right["utility"])),
            }
        )
    return rows


def _comparison_summary(rows, *, seed: int, replicates: int) -> dict[str, Any]:
    return {
        metric: _cluster_bootstrap_delta(
            rows,
            field=f"{metric}_delta",
            seed=seed + index * 97,
            replicates=replicates,
        )
        for index, metric in enumerate(
            ("quality", "emotional_support", "risk", "normalized_cost", "utility")
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "pm_v1_6.yaml"
    )
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--internal-bundle-manifest", type=Path, required=True)
    parser.add_argument("--internal-states", type=Path, required=True)
    parser.add_argument("--internal-labels", type=Path, required=True)
    parser.add_argument("--internal-step0", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=8161)
    args = parser.parse_args()

    reservation = None
    try:
        # Reservation occurs before any internal state, label, or Step-0 row is read.
        reservation = reserve_internal_test_once(
            manifest_path=args.candidate_manifest,
            internal_dataset_path=args.internal_bundle_manifest,
            ledger_path=args.ledger,
        )
        config = load_config(args.config)
        if config.get("version") != "pm-v1.6":
            raise RuntimeError("internal evaluation requires PM-v1.6")

        bundle_manifest = read_json(args.internal_bundle_manifest)
        expected = {
            "protocol": "pm-v1.6-sealed-internal-bundle-v1",
            "states_sha256": sha256_file(args.internal_states),
            "labels_sha256": sha256_file(args.internal_labels),
            "step0_sha256": sha256_file(args.internal_step0),
        }
        if any(
            bundle_manifest.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(
                "sealed internal bundle hashes do not match supplied files"
            )

        bundle = joblib.load(args.checkpoint)
        if (
            not isinstance(bundle, Mapping)
            or bundle.get("protocol") != "pm-v1.6-policy-bundle-v1"
        ):
            raise RuntimeError("checkpoint is not a PM-v1.6 policy bundle")
        model = bundle.get("outcome_model")
        if not isinstance(model, PMV16OutcomeModel):
            raise RuntimeError("policy bundle lacks PM-v1.6 outcome model")
        router_config = TransparentRouterConfig.model_validate(
            bundle["router_config"]
        )
        override = OverrideConfig.model_validate(bundle["override_config"])
        router = StrongTransparentRouter(router_config)

        states = load_states(args.internal_states)
        if any(state.split.value != "internal_test" for state in states):
            raise RuntimeError("internal state file contains a non-internal row")
        labels = _load_labels(args.internal_labels)
        step0 = _load_step0(args.internal_step0)
        if set(step0) != {state.state_id for state in states}:
            raise RuntimeError("internal Step-0 universe differs from states")

        learned_rows = evaluate_observed_policy(
            model=model,
            router=router,
            override_config=override,
            states=states,
            labels=labels,
            step0_by_state=step0,
        )
        rule_rows = _baseline_rows(
            states=states,
            labels=labels,
            step0_by_state=step0,
            model=model,
            router=router,
            override=override,
            baseline="strong_rule",
        )
        # ME+R0 is a legacy regression anchor only where ME is a legal action.
        # The eligible subset is defined by the frozen state action mask before
        # outcomes are read; it is not an outcome-conditioned exclusion.
        me_eligible_states = [
            state for state in states if "ME+R0" in state.allowed_actions
        ]
        me_rows = _baseline_rows(
            states=me_eligible_states,
            labels=labels,
            step0_by_state=step0,
            model=model,
            router=router,
            override=override,
            baseline="me_r0",
            skip_ineligible=False,
        )
        if not me_rows:
            raise RuntimeError(
                "internal test contains no pre-outcome ME+R0-eligible states"
            )

        learned_vs_rule_rows = _paired_deltas(learned_rows, rule_rows)
        learned_vs_me_rows = _paired_deltas(
            learned_rows,
            me_rows,
            allow_baseline_subset=True,
        )
        learned_vs_rule = _comparison_summary(
            learned_vs_rule_rows,
            seed=args.bootstrap_seed,
            replicates=args.bootstrap_replicates,
        )
        learned_vs_me = _comparison_summary(
            learned_vs_me_rows,
            seed=args.bootstrap_seed + 10000,
            replicates=args.bootstrap_replicates,
        )

        gate = config["internal_gate"]
        rule_gate = gate["learned_vs_rule"]
        legacy_gate = gate["legacy_me_r0"]
        maximum_absolute_risk = max(row.risk for row in learned_rows)
        checks = {
            "quality_noninferior_to_rule": learned_vs_rule["quality"]["lower"]
            >= -float(rule_gate["quality_noninferiority_margin"]),
            "emotional_support_noninferior_to_rule": learned_vs_rule[
                "emotional_support"
            ]["lower"]
            >= -float(rule_gate["emotional_support_noninferiority_margin"]),
            "absolute_risk_ceiling": maximum_absolute_risk
            <= float(rule_gate["absolute_risk_ceiling"]),
            "risk_not_increased_vs_rule": learned_vs_rule["risk"]["upper"]
            <= float(rule_gate["maximum_risk_delta_ci_upper"]),
            "utility_strictly_better_than_rule": learned_vs_rule["utility"][
                "lower"
            ]
            > float(rule_gate["require_utility_delta_ci_lower_above"]),
            "quality_not_regressed_vs_me_r0": learned_vs_me["quality"]["lower"]
            >= -float(legacy_gate["quality_noninferiority_margin"]),
            "utility_not_regressed_vs_me_r0": learned_vs_me["utility"]["lower"]
            >= -float(legacy_gate["utility_noninferiority_margin"]),
        }
        status = "PASS" if all(checks.values()) else gate["failure_status"]
        report = {
            "status": status,
            "protocol": "pm-v1.6-one-time-internal-evaluation-v1",
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "internal_bundle_manifest_sha256": sha256_file(
                args.internal_bundle_manifest
            ),
            "consumption_key": reservation.consumption_key,
            "selected_model_mode": model.mode.value,
            "strong_router_config_sha256": router_config.digest(),
            "override_config_sha256": override.digest(),
            "n_states": len(states),
            "n_users": len({state.user_id for state in states}),
            "maximum_learned_absolute_risk": maximum_absolute_risk,
            "checks": checks,
            "learned_vs_strong_rule": learned_vs_rule,
            "learned_vs_me_r0_legacy_guard": learned_vs_me,
            "me_r0_eligible_subset": {
                "selection_timing": "frozen_action_mask_before_outcome_access",
                "n_states": len(me_eligible_states),
                "n_users": len({state.user_id for state in me_eligible_states}),
                "state_coverage": len(me_eligible_states) / len(states),
                "state_ids": sorted(state.state_id for state in me_eligible_states),
            },
            "external_generation_allowed": status == "PASS",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
        finish_internal_test(
            reservation,
            status="COMPLETE",
            report_path=args.out,
        )
    except Exception as exc:
        if reservation is not None:
            finish_internal_test(
                reservation,
                status="FAILED",
                report_path=None,
                error=f"{type(exc).__name__}: {str(exc)[:2000]}",
            )
        raise


if __name__ == "__main__":
    main()
