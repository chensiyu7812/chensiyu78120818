#!/usr/bin/env python3
"""Freeze wave-1 PM_RS predictions before wave-2 outcomes exist."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from metacom_pm.io import (
    iter_jsonl,
    read_json,
    sha256_file,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-pm-rs-wave2-confirmation-freeze-v1"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transparent_pm_rs_v1/pm_rs_model.json",
    )
    parser.add_argument(
        "--wave1-labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_labels_v1"
        / "pm_rs_training_labels.jsonl",
    )
    parser.add_argument(
        "--wave2-plan",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v2_confirmation",
    )
    parser.add_argument(
        "--future-wave2-labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_labels_v2_confirmation"
        / "pm_rs_training_labels.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_pm_rs_wave2_confirmation_freeze_v1",
    )
    args = parser.parse_args()

    if args.future_wave2_labels.exists():
        raise RuntimeError(
            "wave-2 labels already exist; predictions are no longer "
            "prospectively freezable"
        )
    model = read_json(args.model)
    feature_names = list(model["feature_names"])
    means = [float(value) for value in model["standard_scaler_mean"]]
    scales = [float(value) for value in model["standard_scaler_scale"]]
    coefficients = [float(value) for value in model["coefficient_for_RS"]]
    intercept = float(model["intercept_for_RS"])
    threshold = float(model["algorithm"]["open_threshold"])
    if not (
        len(feature_names)
        == len(means)
        == len(scales)
        == len(coefficients)
    ):
        raise RuntimeError("model artifact dimensions differ")

    selected = _rows(args.wave2_plan / "selected_states.jsonl")
    predictions: list[dict[str, Any]] = []
    for row in selected:
        features = row["transparent_pm_features"]
        raw = [float(features.get(name, 0.0)) for name in feature_names]
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(
                raw, means, scales, strict=True
            )
        ]
        logit = intercept + sum(
            value * coefficient
            for value, coefficient in zip(
                standardized, coefficients, strict=True
            )
        )
        probability = 1.0 / (1.0 + math.exp(-logit))
        predictions.append(
            {
                "protocol": PROTOCOL,
                "pair_id": row["pair_id"],
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "selected_strategy_family": row[
                    "selected_strategy_family"
                ],
                "frozen_rs_probability": probability,
                "frozen_action": (
                    "RS" if probability >= threshold else "R0"
                ),
            }
        )
    action_counts = Counter(row["frozen_action"] for row in predictions)
    wave1 = _rows(args.wave1_labels)
    wave1_prevalence = sum(
        int(row["target_y"]) for row in wave1 if row.get("target_y") in {0, 1}
    ) / sum(row.get("target_y") in {0, 1} for row in wave1)
    contract = {
        "protocol": PROTOCOL,
        "status": "FROZEN_BEFORE_WAVE2_GENERATION_AND_LABELS",
        "primary_question": (
            "Does the wave-1 transparent PM_RS candidate generalize to 32 "
            "new same-stack dialogue groups?"
        ),
        "wave2_rows": len(predictions),
        "frozen_action_counts": dict(sorted(action_counts.items())),
        "frozen_threshold": threshold,
        "training_prevalence_baseline_probability": wave1_prevalence,
        "primary_metrics": [
            "wave2 binary log-loss versus the frozen wave1 prevalence baseline",
            "wave2 balanced accuracy at the frozen threshold",
            "whether both R0 and RS are predicted",
        ],
        "pass_rule": (
            "log-loss below the wave1-prevalence baseline, balanced accuracy "
            "> 0.50, and at least one predicted R0 and one predicted RS"
        ),
        "stop_rule": (
            "If the primary confirmation fails, do not collect a third RS "
            "wave for V1.5. Report the learned marginal-benefit head as "
            "unqualified and use the transparent eligibility gate as the "
            "non-learned RS baseline."
        ),
        "secondary_analysis": (
            "Pooling wave1+wave2 to fit a 64-row model is exploratory only "
            "unless evaluated later on the already untouched formal external "
            "sets."
        ),
        "baai_in_primary_model": False,
        "wave2_labels_present_at_freeze": False,
        "lineage": {
            "model_sha256": sha256_file(args.model),
            "wave1_labels_sha256": sha256_file(args.wave1_labels),
            "wave2_plan_report_sha256": sha256_file(
                args.wave2_plan / "plan_report.json"
            ),
            "wave2_selected_states_sha256": sha256_file(
                args.wave2_plan / "selected_states.jsonl"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "frozen_wave2_predictions.jsonl", predictions)
    write_json(args.out_dir / "confirmation_contract.json", contract)
    print(contract)


if __name__ == "__main__":
    main()
