#!/usr/bin/env python3
"""Evaluate the frozen new RS head on the existing ESConv response panel."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
)

from metacom_pm.contracts import MemorySource, parse_action_id
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-new-rs-head-esconv-frozen-panel-diagnostic-v1"
FEATURE_NAMES = (
    "retrieval_score",
    "execution_profile_minimal",
    "family_is_question",
    "family_is_suggestion",
    "advice_welcome",
    "low_burden_or_listen_only",
    "background_MP_on",
    "background_MS_on",
    "background_ME_on",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _values(state: dict[str, Any]) -> dict[str, float]:
    flags = dict(state["observable_flags"])
    background_sources, _ = parse_action_id("M0+R0")
    family = str(state["selected_strategy_family"])
    return {
        "retrieval_score": float(
            state["retrieval_score_diagnostic_only"]
        ),
        "execution_profile_minimal": float(
            state["selected_execution_profile"] == "minimal"
        ),
        "family_is_question": float(family == "Question"),
        "family_is_suggestion": float(
            family == "Providing Suggestions"
        ),
        "advice_welcome": float(bool(flags["advice_welcome"])),
        "low_burden_or_listen_only": float(
            bool(flags["low_burden"] or flags["listen_only"])
        ),
        "background_MP_on": float(
            MemorySource.MP in background_sources
        ),
        "background_MS_on": float(
            MemorySource.MS in background_sources
        ),
        "background_ME_on": float(
            MemorySource.ME in background_sources
        ),
    }


def _predict(
    x: np.ndarray,
    head: dict[str, Any],
) -> np.ndarray:
    members: list[np.ndarray] = []
    for member in head["ensemble_members"]:
        mean = np.asarray(member["scaler_mean"], dtype=float)
        scale = np.asarray(member["scaler_scale"], dtype=float)
        coefficient = np.asarray(member["coefficient"], dtype=float)
        intercept = float(member["intercept"])
        standardized = (x - mean) / scale
        logit = standardized @ coefficient + intercept
        members.append(1.0 / (1.0 + np.exp(-logit)))
    return np.mean(np.stack(members, axis=0), axis=0)


def _classification(
    y: np.ndarray,
    probability: np.ndarray,
    action: np.ndarray,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y, action)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, action)
        ),
        "positive_recall": float(
            recall_score(
                y,
                action,
                pos_label=1,
                zero_division=0,
            )
        ),
        "negative_recall": float(
            recall_score(
                y,
                action,
                pos_label=0,
                zero_division=0,
            )
        ),
        "brier": float(brier_score_loss(y, probability)),
        "predicted_on": int(action.sum()),
        "predicted_off": int((~action).sum()),
    }


def _policy_quality(
    decisions: list[dict[str, Any]],
    action_by_pair: dict[str, bool],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in decisions:
        selected = "RS" if action_by_pair[str(row["pair_id"])] else "R0"
        result = str(row["quality_result"])
        if result == "tie":
            counts["selected_quality_tie"] += 1
        elif result == selected:
            counts["selected_material_winner"] += 1
        else:
            counts["selected_material_loser"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1",
    )
    parser.add_argument(
        "--panel-plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1",
    )
    parser.add_argument(
        "--panel-execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1_execution",
    )
    parser.add_argument(
        "--panel-quality-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_quality_analysis_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_new_rs_head_esconv_frozen_diagnostic_v1",
    )
    args = parser.parse_args()

    training_report = __import__("json").loads(
        (args.model_dir / "training_report.json").read_text(
            encoding="utf-8"
        )
    )
    model = __import__("json").loads(
        (args.model_dir / "model.json").read_text(encoding="utf-8")
    )
    states = _rows(args.panel_plan_dir / "selected_states.jsonl")
    decisions = _rows(args.panel_quality_dir / "quality_decisions.jsonl")
    outcomes = _rows(
        args.panel_execution_dir / "generation_outcomes.jsonl"
    )
    state_by_pair = {str(row["pair_id"]): row for row in states}
    decision_by_pair = {
        str(row["pair_id"]): row for row in decisions
    }
    if (
        len(states) != 40
        or len(decisions) != 40
        or set(state_by_pair) != set(decision_by_pair)
    ):
        raise RuntimeError("expected aligned 40-pair ESConv panel")
    if len({str(row["user_id"]) for row in states}) != 40:
        raise RuntimeError("ESConv panel must contain independent dialogues")

    values = [_values(row) for row in states]
    x = np.asarray(
        [[row[name] for name in FEATURE_NAMES] for row in values],
        dtype=float,
    )
    y = np.asarray(
        [
            int(decision_by_pair[str(row["pair_id"])]["quality_result"] == "RS")
            for row in states
        ],
        dtype=int,
    )
    head = model["heads"]["RS"]
    if tuple(head["feature_names"]) != FEATURE_NAMES:
        raise RuntimeError("RS feature contract differs from frozen model")
    probability = _predict(x, head)
    minimum = np.asarray(
        [
            head["development_support_minimum"][name]
            for name in FEATURE_NAMES
        ],
        dtype=float,
    )
    maximum = np.asarray(
        [
            head["development_support_maximum"][name]
            for name in FEATURE_NAMES
        ],
        dtype=float,
    )
    in_support = np.all(
        (x >= minimum) & (x <= maximum),
        axis=1,
    )
    raw_action = probability >= float(head["threshold"])
    fail_closed_action = raw_action & in_support
    action_by_pair = {
        str(row["pair_id"]): bool(fail_closed_action[index])
        for index, row in enumerate(states)
    }

    outcomes_by_pair_arm = {
        (str(row["pair_id"]), str(row["arm"])): row for row in outcomes
    }
    selected_total_tokens = 0
    always_off_total_tokens = 0
    always_on_total_tokens = 0
    prediction_rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        pair_id = str(state["pair_id"])
        selected_arm = "RS" if fail_closed_action[index] else "R0"
        selected_total_tokens += int(
            outcomes_by_pair_arm[(pair_id, selected_arm)]["usage"][
                "total_tokens"
            ]
        )
        always_off_total_tokens += int(
            outcomes_by_pair_arm[(pair_id, "R0")]["usage"][
                "total_tokens"
            ]
        )
        always_on_total_tokens += int(
            outcomes_by_pair_arm[(pair_id, "RS")]["usage"][
                "total_tokens"
            ]
        )
        prediction_rows.append(
            {
                "protocol": PROTOCOL,
                "pair_id": pair_id,
                "state_id": state["state_id"],
                "user_id": state["user_id"],
                "quality_target_y_provisional_before_risk": int(y[index]),
                "quality_result": decision_by_pair[pair_id][
                    "quality_result"
                ],
                "probability_on": float(probability[index]),
                "in_development_feature_support": bool(
                    in_support[index]
                ),
                "raw_predicted_on": bool(raw_action[index]),
                "fail_closed_predicted_on": bool(
                    fail_closed_action[index]
                ),
                "selected_action": (
                    "M0+RS"
                    if fail_closed_action[index]
                    else "M0+R0"
                ),
            }
        )

    classification = _classification(
        y,
        probability,
        fail_closed_action,
    )
    development_prior = float(
        training_report["head_reports"]["RS"]["baselines"][
            "always_on"
        ]["group_weighted_accuracy"]
    )
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_POSTHOC_DESCRIPTIVE_EXTERNAL_DIAGNOSTIC",
        "formal_external_promotion_allowed": False,
        "reason": (
            "The source RS head failed its registered grouped-OOF learning "
            "gate; this historically used ESConv panel is diagnostic only."
        ),
        "panel_role": (
            "family-balanced frozen 40-dialogue panel, not an unweighted "
            "natural ESConv population"
        ),
        "quality_labels_are_provisional_before_rs_win_risk_review": True,
        "pair_count": len(states),
        "quality_target_counts": dict(sorted(Counter(y.tolist()).items())),
        "development_feature_support": {
            "in_support": int(in_support.sum()),
            "out_of_support": int((~in_support).sum()),
            "in_support_fraction": float(in_support.mean()),
        },
        "classification_against_quality_plus_lower_cost_tie_target": (
            classification
        ),
        "frozen_development_group_weighted_prevalence_prior": (
            development_prior
        ),
        "prior_brier": float(
            brier_score_loss(
                y,
                np.full(len(y), development_prior, dtype=float),
            )
        ),
        "selected_response_quality": _policy_quality(
            decisions,
            action_by_pair,
        ),
        "token_cost": {
            "selected_total_tokens": selected_total_tokens,
            "always_off_total_tokens": always_off_total_tokens,
            "always_on_total_tokens": always_on_total_tokens,
            "incremental_vs_always_off": (
                selected_total_tokens - always_off_total_tokens
            ),
            "saved_vs_always_on": (
                always_on_total_tokens - selected_total_tokens
            ),
        },
        "source_model_gate_passed": bool(
            training_report["head_reports"]["RS"]["grouped_oof"][
                "gate_passed"
            ]
        ),
        "claim_boundary": (
            "This cannot rescue or promote a head that failed development. "
            "It also does not include final material-risk adjudication for "
            "the 16 ESConv RS quality wins."
        ),
        "lineage": {
            "model_sha256": sha256_file(args.model_dir / "model.json"),
            "training_report_sha256": sha256_file(
                args.model_dir / "training_report.json"
            ),
            "selected_states_sha256": sha256_file(
                args.panel_plan_dir / "selected_states.jsonl"
            ),
            "quality_decisions_sha256": sha256_file(
                args.panel_quality_dir / "quality_decisions.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.panel_execution_dir / "generation_outcomes.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "external_diagnostic_report.json", report)
    write_jsonl(args.out_dir / "predictions.jsonl", prediction_rows)
    print(report)


if __name__ == "__main__":
    main()
