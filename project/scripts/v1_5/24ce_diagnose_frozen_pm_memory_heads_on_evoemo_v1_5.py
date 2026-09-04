#!/usr/bin/env python3
"""Apply the failed frozen memory heads to EvoEmo without using outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-frozen-memory-heads-evoemo-support-diagnostic-v1"
COMPONENTS = ("MP", "MS", "ME")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _probability(
    values: np.ndarray,
    ensemble_members: list[dict[str, Any]],
) -> float:
    probabilities: list[float] = []
    for member in ensemble_members:
        mean = np.asarray(member["scaler_mean"], dtype=float)
        scale = np.asarray(member["scaler_scale"], dtype=float)
        coefficient = np.asarray(member["coefficient"], dtype=float)
        standardized = (values - mean) / scale
        logit = float(
            np.dot(standardized, coefficient) + member["intercept"]
        )
        probabilities.append(_sigmoid(logit))
    return float(np.mean(probabilities))


def _feature_values(
    descriptor: dict[str, Any],
    feature_names: list[str],
) -> np.ndarray:
    model_features = dict(descriptor["model_features"])
    base = {
        "top1_relevance_bucket": float(
            model_features["top1_relevance_bucket"]
        ),
        "relevance_margin_bucket": float(
            model_features["relevance_margin_bucket"]
        ),
        "token_cost_bucket": float(model_features["token_cost_bucket"]),
        "retrieved_fraction": float(model_features["retrieved_fraction"]),
        "minimum_relative_age_bucket": float(
            model_features["minimum_relative_age_bucket"]
        ),
        "maximum_relative_age_bucket": float(
            model_features["maximum_relative_age_bucket"]
        ),
        # EvoEmo is evaluated one component at a time here. These are not
        # inferred from outcomes and match an M0+R0 component contrast.
        "background_other_memory_count": 0.0,
        "background_strategy_on": 0.0,
    }
    return np.asarray([base[name] for name in feature_names], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1/"
        "model.json",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1/"
        "training_report.json",
    )
    parser.add_argument(
        "--external-descriptors",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/"
        "external_candidate_descriptors.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_frozen_memory_heads_evoemo_diagnostic_v1",
    )
    args = parser.parse_args()

    model = read_json(args.model)
    training_report = read_json(args.training_report)
    if training_report["status"] != (
        "FOUR_COMPONENT_PRIMARY_LEARNING_GATE_NOT_PASSED"
    ):
        raise RuntimeError(
            "This script is explicitly a diagnostic for the failed primary fit"
        )

    external_rows = _rows(args.external_descriptors)
    if len(external_rows) != 612:
        raise RuntimeError("expected 204 EvoEmo states × 3 memory sources")
    by_state: dict[str, dict[str, dict[str, Any]]] = {}
    state_metadata: dict[str, dict[str, Any]] = {}
    for row in external_rows:
        state_id = str(row["state_id"])
        source = str(row["source"])
        if source not in COMPONENTS:
            raise RuntimeError(f"unexpected source: {source}")
        by_state.setdefault(state_id, {})[source] = dict(
            row["candidate_descriptor"]
        )
        state_metadata[state_id] = {
            "user_id": str(row["user_id"]),
            "topic_index": int(row["topic_index"]),
            "turn_index": int(row["turn_index"]),
        }
    if len(by_state) != 204 or any(
        set(value) != set(COMPONENTS) for value in by_state.values()
    ):
        raise RuntimeError("incomplete EvoEmo source grid")

    prediction_rows: list[dict[str, Any]] = []
    component_reports: dict[str, Any] = {}
    for component in COMPONENTS:
        head = dict(model["heads"][component])
        feature_names = list(head["feature_names"])
        minimum = np.asarray(
            [
                head["development_support_minimum"][name]
                for name in feature_names
            ],
            dtype=float,
        )
        maximum = np.asarray(
            [
                head["development_support_maximum"][name]
                for name in feature_names
            ],
            dtype=float,
        )
        counts: Counter[str] = Counter()
        selected_tokens = 0
        always_on_tokens = 0
        probabilities: list[float] = []
        for state_id in sorted(by_state):
            descriptor = by_state[state_id][component]
            candidate_present = bool(descriptor["candidate_present"])
            injected_tokens = int(
                descriptor["incremental_injected_tokens"]
            )
            always_on_tokens += injected_tokens
            probability: float | None = None
            if not candidate_present:
                in_support = True
                action_on = False
                decision_reason = "NO_CANDIDATE_DETERMINISTIC_OFF"
                counts["no_candidate"] += 1
            else:
                values = _feature_values(descriptor, feature_names)
                in_support = bool(
                    np.all((values >= minimum) & (values <= maximum))
                )
                if not in_support:
                    action_on = False
                    decision_reason = "OUT_OF_DEVELOPMENT_SUPPORT_OFF"
                    counts["candidate_out_of_support"] += 1
                else:
                    probability = _probability(
                        values,
                        list(head["ensemble_members"]),
                    )
                    probabilities.append(probability)
                    action_on = probability >= float(head["threshold"])
                    decision_reason = (
                        "FROZEN_HEAD_ON"
                        if action_on
                        else "FROZEN_HEAD_OFF"
                    )
                    counts["candidate_in_support"] += 1
                    counts["predicted_on" if action_on else "predicted_off"] += 1
            if action_on:
                selected_tokens += injected_tokens
            prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "state_id": state_id,
                    **state_metadata[state_id],
                    "component": component,
                    "candidate_present": candidate_present,
                    "development_range_in_support": in_support,
                    "probability_on": probability,
                    "action_on": action_on,
                    "decision_reason": decision_reason,
                    "incremental_injected_tokens": injected_tokens,
                }
            )

        candidate_count = (
            counts["candidate_in_support"]
            + counts["candidate_out_of_support"]
        )
        component_reports[component] = {
            "state_count": len(by_state),
            "candidate_present": candidate_count,
            "candidate_absent": counts["no_candidate"],
            "candidate_present_fraction": candidate_count / len(by_state),
            "candidate_in_development_support": counts[
                "candidate_in_support"
            ],
            "candidate_out_of_development_support": counts[
                "candidate_out_of_support"
            ],
            "candidate_support_fraction": (
                counts["candidate_in_support"] / candidate_count
                if candidate_count
                else None
            ),
            "predicted_on": counts["predicted_on"],
            "predicted_off_with_candidate": counts["predicted_off"],
            "predicted_on_fraction_of_all_states": (
                counts["predicted_on"] / len(by_state)
            ),
            "predicted_on_fraction_of_supported_candidates": (
                counts["predicted_on"] / counts["candidate_in_support"]
                if counts["candidate_in_support"]
                else None
            ),
            "mean_probability_on_supported_candidates": (
                float(np.mean(probabilities)) if probabilities else None
            ),
            "selected_incremental_tokens": selected_tokens,
            "always_on_when_available_incremental_tokens": always_on_tokens,
            "tokens_saved_vs_always_on_when_available": (
                always_on_tokens - selected_tokens
            ),
            "response_quality_evaluated": False,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out_dir / "predictions.jsonl"
    write_jsonl(prediction_path, prediction_rows)
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_OUTCOME_BLIND_DIAGNOSTIC_ONLY",
        "formal_external_promotion_allowed": False,
        "source_model_gate_passed": False,
        "external_state_count": len(by_state),
        "external_user_count": len(
            {value["user_id"] for value in state_metadata.values()}
        ),
        "component_reports": component_reports,
        "quality_or_risk_outcome_used": False,
        "api_calls": 0,
        "claim_boundary": (
            "This measures candidate availability, development-range support, "
            "frozen action rates, and token exposure only. It cannot establish "
            "external quality, risk, utility, or rescue a head that failed "
            "development grouped-OOF."
        ),
        "lineage": {
            "model_sha256": sha256_file(args.model),
            "training_report_sha256": sha256_file(args.training_report),
            "external_descriptors_sha256": sha256_file(
                args.external_descriptors
            ),
            "predictions_sha256": sha256_file(prediction_path),
        },
    }
    write_json(args.out_dir / "external_support_report.json", report)
    print(
        {
            "status": report["status"],
            "components": {
                component: {
                    "candidate_support_fraction": values[
                        "candidate_support_fraction"
                    ],
                    "predicted_on": values["predicted_on"],
                    "predicted_on_fraction_of_all_states": values[
                        "predicted_on_fraction_of_all_states"
                    ],
                }
                for component, values in component_reports.items()
            },
        }
    )


if __name__ == "__main__":
    main()
