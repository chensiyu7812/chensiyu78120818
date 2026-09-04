#!/usr/bin/env python3
"""Unblind frozen ESConv quality labels and compare PM_RS with baselines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
BLIND_PROTOCOL = "pm-v1.5-rs-esconv-frozen-quality-blind-v1"
PROTOCOL = "pm-v1.5-rs-esconv-frozen-quality-analysis-v1"
PREFERENCES = {"A", "B", "tie", "uncertain"}
CRITERIA = {
    "request_and_conversation_fit",
    "emotional_attunement",
    "visible_context_fidelity",
    "immediate_helpfulness",
    "clarity_and_naturalness",
    "materially_equivalent",
    "uncertain",
}
FROZEN_TRAIN_PRIOR = 13.0 / 32.0
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_REPLICATES = 20000


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _classification(
    y: np.ndarray,
    probability: np.ndarray,
    action: np.ndarray,
) -> dict[str, Any]:
    positive = y == 1
    negative = y == 0
    return {
        "accuracy": float((action == y).mean()),
        "balanced_accuracy": float(
            0.5
            * (
                (action[positive] == 1).mean()
                + (action[negative] == 0).mean()
            )
        ),
        "positive_recall": float((action[positive] == 1).mean()),
        "negative_recall": float((action[negative] == 0).mean()),
        "brier": float(np.mean((probability - y) ** 2)),
        "predicted_on": int(action.sum()),
        "predicted_off": int((1 - action).sum()),
    }


def _quality_selection(
    decisions: list[dict[str, Any]],
    action_by_pair: dict[str, str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in decisions:
        pair_id = str(row["pair_id"])
        result = str(row["quality_result"])
        action_arm = (
            "RS" if action_by_pair[pair_id] == "M0+RS" else "R0"
        )
        if result == "tie":
            counts["selected_quality_tie"] += 1
        elif result == "uncertain":
            counts["selected_quality_unknown"] += 1
        elif result == action_arm:
            counts["selected_material_winner"] += 1
        else:
            counts["selected_material_loser"] += 1
    return {
        key: counts[key]
        for key in (
            "selected_material_winner",
            "selected_quality_tie",
            "selected_material_loser",
            "selected_quality_unknown",
        )
    }


def _token_cost(
    pair_ids: list[str],
    action_by_pair: dict[str, str],
    outcomes: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        outcomes[
            (
                pair_id,
                "RS"
                if action_by_pair[pair_id] == "M0+RS"
                else "R0",
            )
        ]
        for pair_id in pair_ids
    ]
    prompt = sum(int(row["usage"]["prompt_tokens"]) for row in selected)
    completion = sum(
        int(row["usage"]["completion_tokens"]) for row in selected
    )
    total = sum(int(row["usage"]["total_tokens"]) for row in selected)
    return {
        "responses": len(selected),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "mean_total_tokens": total / len(selected),
    }


def _bootstrap_accuracy_differences(
    *,
    y: np.ndarray,
    pm_action: np.ndarray,
    families: list[str],
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    family_indices = {
        family: np.asarray(
            [index for index, value in enumerate(families) if value == family],
            dtype=int,
        )
        for family in sorted(set(families))
    }
    pm_minus_off = np.zeros(BOOTSTRAP_REPLICATES)
    pm_minus_on = np.zeros(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in family_indices.values()
            ]
        )
        sampled_y = y[sampled]
        sampled_pm = pm_action[sampled]
        pm_accuracy = float((sampled_pm == sampled_y).mean())
        pm_minus_off[replicate] = pm_accuracy - float(
            (sampled_y == 0).mean()
        )
        pm_minus_on[replicate] = pm_accuracy - float(
            (sampled_y == 1).mean()
        )

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
        }

    return {
        "protocol": "family-stratified-bootstrap-v1",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "pm_accuracy_minus_always_off": interval(pm_minus_off),
        "pm_accuracy_minus_always_on": interval(pm_minus_on),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blind-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_quality_blind_v1_candidate",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1",
    )
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_validation_v1_execution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_esconv_frozen_quality_analysis_v1",
    )
    args = parser.parse_args()

    key_rows = _rows(args.blind_dir / "private_blinding_key.jsonl")
    key = {str(row["blind_item_id"]): row for row in key_rows}
    annotation_rows = _rows(args.annotations)
    annotations = {
        str(row["blind_item_id"]): row for row in annotation_rows
    }
    selected_rows = _rows(args.plan_dir / "selected_states.jsonl")
    selected = {str(row["pair_id"]): row for row in selected_rows}
    outcome_rows = _rows(
        args.execution_dir / "generation_outcomes.jsonl"
    )
    outcomes = {
        (str(row["pair_id"]), str(row["arm"])): row
        for row in outcome_rows
    }
    expected_outcomes = {
        (pair_id, arm)
        for pair_id in selected
        for arm in ("R0", "RS")
    }
    if (
        len(key) != len(key_rows)
        or len(annotations) != len(annotation_rows)
        or len(selected) != len(selected_rows)
        or len(outcomes) != len(outcome_rows)
        or len(key) != 40
        or set(key) != set(annotations)
        or expected_outcomes != set(outcomes)
    ):
        raise RuntimeError("annotation or frozen pair lineage is incomplete")

    errors: list[str] = []
    decisions: list[dict[str, Any]] = []
    for blind_id in sorted(key):
        private = key[blind_id]
        annotation = annotations[blind_id]
        if annotation.get("protocol") != BLIND_PROTOCOL:
            errors.append(f"{blind_id}: protocol mismatch")
        preference = annotation.get("quality_preference")
        criterion = annotation.get("decisive_criterion")
        annotator = str(annotation.get("annotator_id") or "").strip()
        if preference not in PREFERENCES:
            errors.append(f"{blind_id}: invalid quality_preference")
            continue
        if criterion not in CRITERIA:
            errors.append(f"{blind_id}: invalid decisive_criterion")
            continue
        if not annotator:
            errors.append(f"{blind_id}: annotator_id is empty")
        if preference == "tie" and criterion != "materially_equivalent":
            errors.append(f"{blind_id}: tie criterion mismatch")
        if preference == "uncertain" and criterion != "uncertain":
            errors.append(f"{blind_id}: uncertain criterion mismatch")
        if preference in {"A", "B"} and criterion in {
            "materially_equivalent",
            "uncertain",
        }:
            errors.append(f"{blind_id}: A/B criterion mismatch")
        quality_result = (
            str(preference)
            if preference in {"tie", "uncertain"}
            else str(
                private[
                    "response_a_arm"
                    if preference == "A"
                    else "response_b_arm"
                ]
            )
        )
        pair_id = str(private["pair_id"])
        state = selected[pair_id]
        decisions.append(
            {
                "protocol": PROTOCOL,
                "blind_item_id": blind_id,
                "pair_id": pair_id,
                "state_id": private["state_id"],
                "user_id": private["user_id"],
                "quality_result": quality_result,
                "decisive_criterion": criterion,
                "quality_notes": str(
                    annotation.get("quality_notes") or ""
                ).strip(),
                "annotator_id": annotator,
                "selected_card_id": private["selected_card_id"],
                "selected_core_submove_id": private[
                    "selected_core_submove_id"
                ],
                "selected_strategy_family": private[
                    "selected_strategy_family"
                ],
                "pm_rs_probability": float(private["pm_rs_probability"]),
                "frozen_pm_action": private["frozen_pm_action"],
                "provisional_target_y": (
                    None
                    if quality_result == "uncertain"
                    else int(quality_result == "RS")
                ),
                "risk_review_required": quality_result == "RS",
                "provisional_only_until_rs_risk_review": (
                    quality_result == "RS"
                ),
                "current_session_summary": state[
                    "current_session_summary"
                ],
            }
        )
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
    if any(row["quality_result"] == "uncertain" for row in decisions):
        raise RuntimeError(
            "uncertain external outcomes require a separate missing-data "
            "analysis before frozen policy scoring"
        )

    decisions.sort(key=lambda row: str(row["pair_id"]))
    y = np.asarray(
        [int(row["provisional_target_y"]) for row in decisions],
        dtype=int,
    )
    pm_probability = np.asarray(
        [float(row["pm_rs_probability"]) for row in decisions],
        dtype=float,
    )
    pm_action = np.asarray(
        [int(row["frozen_pm_action"] == "M0+RS") for row in decisions],
        dtype=int,
    )
    always_off = np.zeros(len(y), dtype=int)
    always_on = np.ones(len(y), dtype=int)
    frozen_prior = np.full(len(y), FROZEN_TRAIN_PRIOR, dtype=float)
    families = [
        str(row["selected_strategy_family"]) for row in decisions
    ]
    pair_ids = [str(row["pair_id"]) for row in decisions]
    action_maps = {
        "frozen_pm": {
            str(row["pair_id"]): str(row["frozen_pm_action"])
            for row in decisions
        },
        "always_off": {pair_id: "M0+R0" for pair_id in pair_ids},
        "always_on": {pair_id: "M0+RS" for pair_id in pair_ids},
    }
    policy_actions = {
        "frozen_pm": pm_action,
        "always_off": always_off,
        "always_on": always_on,
    }
    policy_probabilities = {
        "frozen_pm": pm_probability,
        "always_off": always_off.astype(float),
        "always_on": always_on.astype(float),
    }
    policy_comparison: dict[str, Any] = {}
    off_cost = _token_cost(pair_ids, action_maps["always_off"], outcomes)
    for name in ("frozen_pm", "always_off", "always_on"):
        cost = _token_cost(pair_ids, action_maps[name], outcomes)
        policy_comparison[name] = {
            "classification_against_quality_plus_lower_cost_tie_target": (
                _classification(
                    y, policy_probabilities[name], policy_actions[name]
                )
            ),
            "selected_response_quality": _quality_selection(
                decisions, action_maps[name]
            ),
            "observed_generation_tokens": cost,
            "incremental_total_tokens_vs_always_off": (
                cost["total_tokens"] - off_cost["total_tokens"]
            ),
            "incremental_total_token_fraction_vs_always_off": (
                (cost["total_tokens"] - off_cost["total_tokens"])
                / off_cost["total_tokens"]
            ),
        }

    prior_brier = float(np.mean((frozen_prior - y) ** 2))
    pm_metrics = policy_comparison["frozen_pm"][
        "classification_against_quality_plus_lower_cost_tie_target"
    ]
    formal_checks = {
        "external_brier_beats_frozen_train_prevalence_prior": (
            float(pm_metrics["brier"]) < prior_brier
        ),
        "external_balanced_accuracy_at_least_0_70": (
            float(pm_metrics["balanced_accuracy"]) >= 0.70
        ),
        "external_positive_recall_at_least_0_60": (
            float(pm_metrics["positive_recall"]) >= 0.60
        ),
        "external_on_fraction_at_least_0_20": (
            int(pm_metrics["predicted_on"]) / len(y) >= 0.20
        ),
        "external_off_fraction_at_least_0_20": (
            int(pm_metrics["predicted_off"]) / len(y) >= 0.20
        ),
        "external_accuracy_strictly_beats_always_off": (
            float(pm_metrics["accuracy"])
            > float(
                policy_comparison["always_off"][
                    "classification_against_quality_plus_lower_cost_tie_target"
                ]["accuracy"]
            )
        ),
        "external_accuracy_strictly_beats_always_on": (
            float(pm_metrics["accuracy"])
            > float(
                policy_comparison["always_on"][
                    "classification_against_quality_plus_lower_cost_tie_target"
                ]["accuracy"]
            )
        ),
    }

    quality_counts = Counter(
        str(row["quality_result"]) for row in decisions
    )
    criterion_counts = Counter(
        str(row["decisive_criterion"]) for row in decisions
    )
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in decisions:
        by_family[str(row["selected_strategy_family"])][
            str(row["quality_result"])
        ] += 1
    risk_candidates = [
        row for row in decisions if row["risk_review_required"]
    ]
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_RS_WIN_RISK_REVIEW",
        "quality_only_provisional": True,
        "panel_role": (
            "family-balanced frozen mechanism panel; not an unweighted "
            "natural ESConv population estimate"
        ),
        "pair_count": len(decisions),
        "independent_dialogues": len(
            {str(row["user_id"]) for row in decisions}
        ),
        "quality_result_counts": dict(sorted(quality_counts.items())),
        "decisive_criterion_counts": dict(
            sorted(criterion_counts.items())
        ),
        "quality_results_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(by_family.items())
        },
        "policy_comparison": policy_comparison,
        "frozen_train_prior": FROZEN_TRAIN_PRIOR,
        "frozen_train_prior_brier_on_external_panel": prior_brier,
        "external_gate_checks_before_rs_win_risk_review": formal_checks,
        "all_external_quality_checks_passed_before_risk": all(
            formal_checks.values()
        ),
        "family_stratified_bootstrap": _bootstrap_accuracy_differences(
            y=y,
            pm_action=pm_action,
            families=families,
        ),
        "rs_win_risk_review_candidate_count": len(risk_candidates),
        "interpretation_guardrails": [
            "RS quality wins remain provisional until atomic material-risk review",
            "ties map to R0 only for the quality-plus-lower-cost target",
            "no threshold, feature, Bank, retriever, or panel state may be "
            "changed from these outcomes",
            "observed token totals exclude candidate-lookup overhead",
            "the benchmark was used historically, so this is a frozen "
            "current-lineage external check rather than a pristine benchmark",
        ],
        "lineage": {
            "annotations_sha256": sha256_file(args.annotations),
            "private_blinding_key_sha256": sha256_file(
                args.blind_dir / "private_blinding_key.jsonl"
            ),
            "selected_states_sha256": sha256_file(
                args.plan_dir / "selected_states.jsonl"
            ),
            "generation_outcomes_sha256": sha256_file(
                args.execution_dir / "generation_outcomes.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "quality_decisions.jsonl", decisions)
    write_jsonl(
        args.out_dir / "rs_win_risk_review_candidates.jsonl",
        risk_candidates,
    )
    write_json(args.out_dir / "quality_analysis_report.json", report)
    print(
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "quality_results": dict(sorted(quality_counts.items())),
            "pm_accuracy": pm_metrics["accuracy"],
            "pm_balanced_accuracy": pm_metrics["balanced_accuracy"],
            "risk_candidates": len(risk_candidates),
        }
    )


if __name__ == "__main__":
    main()
