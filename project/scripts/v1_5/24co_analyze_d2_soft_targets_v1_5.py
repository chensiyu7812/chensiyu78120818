#!/usr/bin/env python3
"""Run a bounded pre-D3 diagnostic on D2 soft expected-benefit targets.

This script does not train or promote a PM.  With only eight states per
component, the frozen capacity rule permits one outcome-facing feature.  The
diagnostic therefore asks only whether the already-frozen candidate match
score has the expected positive direction and improves leave-one-user-out
squared error over a training-mean baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-d2-soft-target-directional-diagnostic-v1"
COMPONENTS = ("RS", "MP", "MS", "ME")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _leave_one_user_out_one_feature(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    prediction = np.zeros(len(y), dtype=float)
    baseline = np.zeros(len(y), dtype=float)
    for group in sorted(set(groups.tolist())):
        test = groups == group
        train = ~test
        if not train.any() or not test.any():
            raise RuntimeError("invalid leave-one-user-out split")
        x_train = x[train]
        mean = float(x_train.mean())
        scale = float(x_train.std(ddof=0))
        if scale == 0.0:
            x_train_scaled = np.zeros(len(x_train), dtype=float)
            x_test_scaled = np.zeros(int(test.sum()), dtype=float)
        else:
            x_train_scaled = (x_train - mean) / scale
            x_test_scaled = (x[test] - mean) / scale
        design = np.column_stack(
            [np.ones(len(x_train_scaled)), x_train_scaled]
        )
        coefficient = np.linalg.lstsq(design, y[train], rcond=None)[0]
        prediction[test] = (
            coefficient[0] + coefficient[1] * x_test_scaled
        )
        baseline[test] = float(y[train].mean())
    model_mse = float(np.mean((y - prediction) ** 2))
    baseline_mse = float(np.mean((y - baseline) ** 2))
    return {
        "prediction": prediction,
        "baseline": baseline,
        "mse": model_mse,
        "training_mean_baseline_mse": baseline_mse,
        "mse_gain_vs_training_mean": baseline_mse - model_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soft-targets",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d2_measurement_review_analysis_v1/"
        "state_soft_targets.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_d2_measurement_blueprint_v1/"
        "d2_measurement_blueprint.jsonl",
    )
    parser.add_argument(
        "--descriptors",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "candidate_descriptors.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_d2_soft_target_diagnostic_v1",
    )
    args = parser.parse_args()

    targets = _rows(args.soft_targets)
    blueprints = {
        str(row["d2_state_id"]): row for row in _rows(args.blueprint)
    }
    descriptors = {
        (
            str(row["state_id"]),
            str(row["candidate_descriptor"]["source"]),
        ): dict(row["candidate_descriptor"])
        for row in _rows(args.descriptors)
    }
    checks = {
        "target_rows_32": len(targets) == 32,
        "unique_d2_state_ids": len(
            {str(row["d2_state_id"]) for row in targets}
        )
        == len(targets),
        "exact_component_balance": Counter(
            str(row["component"]) for row in targets
        )
        == Counter({component: 8 for component in COMPONENTS}),
        "three_independent_pairs_per_state": all(
            int(row["independent_generated_pairs"]) == 3
            for row in targets
        ),
        "all_targets_observed_and_bounded": all(
            row["soft_expected_benefit"] is not None
            and 0.0 <= float(row["soft_expected_benefit"]) <= 1.0
            for row in targets
        ),
        "no_hard_label_claim": all(
            bool(row["not_a_hard_effect_label"]) for row in targets
        ),
        "blueprint_identity_complete": {
            str(row["d2_state_id"]) for row in targets
        }
        == set(blueprints),
    }
    if not all(checks.values()):
        raise RuntimeError(f"soft-target input checks failed: {checks}")

    feature_rows: list[dict[str, Any]] = []
    component_reports: dict[str, Any] = {}
    for component in COMPONENTS:
        rows = sorted(
            (
                row
                for row in targets
                if str(row["component"]) == component
            ),
            key=lambda row: str(row["d2_state_id"]),
        )
        values: list[float] = []
        target_values: list[float] = []
        groups: list[str] = []
        for row in rows:
            blueprint = blueprints[str(row["d2_state_id"])]
            if component == "RS":
                feature_name = "strategy_top1_lexical_retrieval_score"
                feature_value = float(
                    blueprint["current_strategy_candidate"][
                        "retrieval_score"
                    ]
                )
            else:
                feature_name = "memory_top1_lexical_relevance"
                descriptor = descriptors[
                    (str(row["state_id"]), component)
                ]
                feature_value = float(
                    descriptor["top1_lexical_relevance"]
                )
            target = float(row["soft_expected_benefit"])
            values.append(feature_value)
            target_values.append(target)
            groups.append(str(row["user_id"]))
            feature_rows.append(
                {
                    "protocol": PROTOCOL,
                    "d2_state_id": row["d2_state_id"],
                    "component": component,
                    "state_id": row["state_id"],
                    "user_id": row["user_id"],
                    "feature_name": feature_name,
                    "candidate_match_score": feature_value,
                    "soft_expected_benefit": target,
                    "outcome_use": "pre-D3 diagnostic only",
                }
            )
        x = np.asarray(values, dtype=float)
        y = np.asarray(target_values, dtype=float)
        group_array = np.asarray(groups, dtype=object)
        if len(set(groups)) != len(groups):
            raise RuntimeError(
                f"{component}: expected eight independent user groups"
            )
        correlation = spearmanr(x, y)
        loo = _leave_one_user_out_one_feature(x, y, group_array)
        median = float(np.median(x))
        high = y[x >= median]
        low = y[x < median]
        component_reports[component] = {
            "states": len(rows),
            "independent_user_groups": len(set(groups)),
            "capacity_floor_groups_div_5": len(set(groups)) // 5,
            "feature_count": 1,
            "feature_name": feature_name,
            "target_mean": float(y.mean()),
            "target_std": float(y.std(ddof=1)),
            "target_distribution": {
                str(value): int(count)
                for value, count in sorted(Counter(y.tolist()).items())
            },
            "candidate_score_min": float(x.min()),
            "candidate_score_max": float(x.max()),
            "spearman_rho": float(correlation.statistic),
            "spearman_p_value_descriptive_only": float(correlation.pvalue),
            "above_or_equal_median_target_mean": float(high.mean()),
            "below_median_target_mean": float(low.mean()),
            "median_split_target_difference": float(
                high.mean() - low.mean()
            ),
            "leave_one_user_out": {
                key: value
                for key, value in loo.items()
                if key not in {"prediction", "baseline"}
            },
            "positive_direction_on_both_diagnostics": bool(
                float(correlation.statistic) > 0.0
                and loo["mse_gain_vs_training_mean"] > 0.0
            ),
        }
        for index, row in enumerate(feature_rows[-len(rows) :]):
            row["leave_one_user_out_prediction"] = float(
                loo["prediction"][index]
            )
            row["leave_one_user_out_mean_baseline"] = float(
                loo["baseline"][index]
            )

    positive = [
        component
        for component, report in component_reports.items()
        if report["positive_direction_on_both_diagnostics"]
    ]
    report = {
        "protocol": PROTOCOL,
        "status": "PRE_D3_DIAGNOSTIC_COMPLETE_NOT_MODEL_PROMOTION",
        "formal_D2_gate_remains_failed": True,
        "api_calls": 0,
        "new_human_decisions": 0,
        "data_and_grain": {
            "state_rows": len(targets),
            "states_per_component": 8,
            "independent_pairs_per_state": 3,
            "soft_target_definition": (
                "Within-pair reviewer mean of treatment-win=1 and "
                "control-win/tie=0, followed by an equal-weight mean over "
                "three independent generated pairs per state."
            ),
        },
        "checks": checks,
        "component_reports": component_reports,
        "components_with_positive_candidate_match_direction": positive,
        "candidate_match_only_supported_for_all_four_heads": len(positive)
        == 4,
        "interpretation_rule": (
            "A positive Spearman direction plus lower leave-one-user-out MSE "
            "than a training-mean baseline is descriptive evidence that the "
            "single candidate match score carries direction. This is not a "
            "formal pass gate and cannot promote a PM with eight states."
        ),
        "decision_boundary": {
            "if_all_four_positive": (
                "A minimal D3 may retain candidate match as the single "
                "primary score, while pre-registering grounding and explicit "
                "background features only when group capacity permits."
            ),
            "otherwise": (
                "Do not expand the same match-score-only design. Implement "
                "the already-frozen grounding/nonredundancy and explicit "
                "background features before any D3 outcome generation."
            ),
        },
        "limitations": [
            "Eight states per component cannot estimate a deployable head.",
            "RS/MP/ME states were selected partly by historical realized-pair direction for measurement balance, so this is not a prevalence estimate.",
            "P-values are descriptive and are not corrected for four components.",
            "The diagnostic tests candidate match alone, not the complete corrected feature contract."
        ],
        "source_hashes": {
            "soft_targets": sha256_file(args.soft_targets),
            "blueprint": sha256_file(args.blueprint),
            "descriptors": sha256_file(args.descriptors),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / "state_features_and_soft_targets.jsonl"
    write_jsonl(feature_path, feature_rows)
    report["state_features_and_soft_targets_sha256"] = sha256_file(
        feature_path
    )
    write_json(args.out_dir / "diagnostic.json", report)
    print(
        {
            "status": report["status"],
            "positive_components": positive,
            "all_four_positive": len(positive) == 4,
        }
    )


if __name__ == "__main__":
    main()
