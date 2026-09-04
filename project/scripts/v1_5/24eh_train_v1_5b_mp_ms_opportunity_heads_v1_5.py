#!/usr/bin/env python3
"""Train and qualify V1.5b MP/MS heads on the frozen anti-shortcut data.

The sealed confirmation split is evaluated once and is not used to select the
feature schema, C, threshold, or seeds.  Passing this script is an internal
qualification only; final promotion still requires outcome-blind transport
coverage on the untouched external memory catalogs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5b-mp-ms-opportunity-head-fit-v1"
CONTRACT = "data/pm_v1_5_contracts/v1_5b_mp_ms_generator_root_repair_v1.json"
COMPONENTS = ("MP", "MS")
SEEDS = (20260802, 20260803, 20260804, 20260805, 20260806)
C = 0.3
THRESHOLD = 0.5

COUNTERFACTUAL_PAIRS = {
    "MP": (
        ("preference_direct_current_scope", "preference_high_lexical_scope_mismatch"),
        ("profile_direct_incremental", "profile_already_visible"),
        ("profile_direct_incremental", "profile_high_lexical_wrong_entity"),
        ("preference_low_lexical_paraphrase", "preference_realistic_irrelevant"),
    ),
    "MS": (
        ("same_unresolved_issue_direct", "high_lexical_no_incremental_value"),
        ("same_issue_low_lexical_paraphrase", "same_topic_high_lexical_wrong_goal"),
        ("prior_outcome_requested", "old_issue_resolved_new_issue_current"),
        ("prior_distinction_matches_current_goal", "summary_already_visible"),
    ),
}


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="liblinear",
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )


def _metrics(y: np.ndarray, probability: np.ndarray, prior: np.ndarray) -> dict[str, Any]:
    prediction = probability >= THRESHOLD
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "positive_recall": float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "prevalence_prior_brier": float(brier_score_loss(y, prior)),
        "brier_gain_vs_prevalence_prior": float(
            brier_score_loss(y, prior) - brier_score_loss(y, probability)
        ),
        "predicted_on": int(prediction.sum()),
        "predicted_off": int((~prediction).sum()),
    }


def _grouped_oof(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros(len(y), dtype=float)
    prior = np.zeros(len(y), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for train, test in splitter.split(x, y, groups):
        model = _pipeline(seed)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
        prior[test] = float(y[train].mean())
    return probability, prior


def _seed_agreement(decisions: list[np.ndarray]) -> float:
    matrix = np.stack(decisions, axis=0)
    return float(np.mean(np.logical_or(np.all(matrix, axis=0), np.all(~matrix, axis=0))))


def _counterfactual_accuracy(
    *, component: str, rows: list[dict[str, Any]], probability: np.ndarray
) -> dict[str, Any]:
    lookup = {
        (
            str(row["semantic_family_private_not_model_input"]),
            str(row["condition_family_private_not_model_input"]),
        ): float(probability[index])
        for index, row in enumerate(rows)
    }
    comparisons: list[bool] = []
    for family in sorted({key[0] for key in lookup}):
        for positive, negative in COUNTERFACTUAL_PAIRS[component]:
            comparisons.append(lookup[(family, positive)] > lookup[(family, negative)])
    return {
        "correct": int(sum(comparisons)),
        "total": len(comparisons),
        "accuracy": float(np.mean(comparisons)),
    }


def _fit_component(
    component: str,
    fit: list[dict[str, Any]],
    confirmation: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    feature_names = tuple(sorted(fit[0]["model_features"]))
    if any(tuple(sorted(row["model_features"])) != feature_names for row in [*fit, *confirmation]):
        raise RuntimeError(f"{component} feature schema changed")
    x = np.asarray([[float(row["model_features"][name]) for name in feature_names] for row in fit])
    y = np.asarray([int(row["opportunity_target"]) for row in fit])
    groups = np.asarray([row["semantic_family_private_not_model_input"] for row in fit], dtype=object)
    x_confirm = np.asarray([[float(row["model_features"][name]) for name in feature_names] for row in confirmation])
    y_confirm = np.asarray([int(row["opportunity_target"]) for row in confirmation])

    nuisance_names = tuple(name for name in feature_names if name.startswith("background_"))
    nuisance_x = np.asarray([[float(row["model_features"][name]) for name in nuisance_names] for row in fit])

    reports: list[dict[str, Any]] = []
    oof_decisions: list[np.ndarray] = []
    confirm_decisions: list[np.ndarray] = []
    final_models: list[Pipeline] = []
    for seed in SEEDS:
        oof_probability, oof_prior = _grouped_oof(x, y, groups, seed)
        nuisance_probability, nuisance_prior = _grouped_oof(nuisance_x, y, groups, seed)
        model = _pipeline(seed)
        model.fit(x, y)
        confirm_probability = model.predict_proba(x_confirm)[:, 1]
        confirm_prior = np.full(len(y_confirm), float(y.mean()))
        reports.append(
            {
                "seed": seed,
                "semantic_family_grouped_oof": _metrics(y, oof_probability, oof_prior),
                "nuisance_only_grouped_oof": _metrics(y, nuisance_probability, nuisance_prior),
                "sealed_confirmation": _metrics(y_confirm, confirm_probability, confirm_prior),
                "sealed_counterfactual_direction": _counterfactual_accuracy(
                    component=component,
                    rows=confirmation,
                    probability=confirm_probability,
                ),
            }
        )
        oof_decisions.append(oof_probability >= THRESHOLD)
        confirm_decisions.append(confirm_probability >= THRESHOLD)
        final_models.append(model)

    oof = [row["semantic_family_grouped_oof"] for row in reports]
    nuisance = [row["nuisance_only_grouped_oof"] for row in reports]
    confirm = [row["sealed_confirmation"] for row in reports]
    directions = [row["sealed_counterfactual_direction"] for row in reports]
    checks = {
        "64_fit_rows_64_users_32_on_32_off": len(fit) == 64 and len({row["user_id_private_not_model_input"] for row in fit}) == 64 and Counter(y.tolist()) == Counter({0: 32, 1: 32}),
        "32_confirmation_rows_32_users_16_on_16_off": len(confirmation) == 32 and len({row["user_id_private_not_model_input"] for row in confirmation}) == 32 and Counter(y_confirm.tolist()) == Counter({0: 16, 1: 16}),
        "fit_confirmation_users_disjoint": not ({row["user_id_private_not_model_input"] for row in fit} & {row["user_id_private_not_model_input"] for row in confirmation}),
        "fit_confirmation_semantic_families_disjoint": not ({row["semantic_family_private_not_model_input"] for row in fit} & {row["semantic_family_private_not_model_input"] for row in confirmation}),
        "worst_seed_grouped_oof_ba_min_0_80": min(row["balanced_accuracy"] for row in oof) >= 0.80,
        "worst_seed_grouped_oof_recall_specificity_min_0_75": min(row["positive_recall"] for row in oof) >= 0.75 and min(row["specificity"] for row in oof) >= 0.75,
        "every_seed_oof_brier_beats_prior": min(row["brier_gain_vs_prevalence_prior"] for row in oof) > 0.0,
        "five_seed_oof_decision_agreement_min_0_90": _seed_agreement(oof_decisions) >= 0.90,
        "worst_seed_confirmation_ba_min_0_80": min(row["balanced_accuracy"] for row in confirm) >= 0.80,
        "worst_seed_confirmation_recall_specificity_min_0_75": min(row["positive_recall"] for row in confirm) >= 0.75 and min(row["specificity"] for row in confirm) >= 0.75,
        "every_seed_confirmation_brier_beats_prior": min(row["brier_gain_vs_prevalence_prior"] for row in confirm) > 0.0,
        "worst_seed_counterfactual_direction_min_0_80": min(row["accuracy"] for row in directions) >= 0.80,
        "five_seed_confirmation_decision_agreement_min_0_90": _seed_agreement(confirm_decisions) >= 0.90,
        "nuisance_only_grouped_ba_max_0_55": max(row["balanced_accuracy"] for row in nuisance) <= 0.55,
        "no_outcome_fields_read": all(not row["response_quality_risk_or_external_outcome_read"] for row in fit),
    }
    model_path = out_dir / f"{component.lower()}_opportunity_router_v1_5b.joblib"
    joblib.dump(final_models[0], model_path)
    return {
        "status": "PASS_INTERNAL_AWAITING_EXTERNAL_TRANSPORT" if all(checks.values()) else "FAIL_INTERNAL_NOT_ELIGIBLE_FOR_EXTERNAL_TRANSPORT",
        "feature_names": list(feature_names),
        "fixed_hyperparameters": {
            "model": "standardized_l2_logistic_regression",
            "C": C,
            "threshold": THRESHOLD,
            "seeds": list(SEEDS),
        },
        "five_seed_oof_decision_agreement": _seed_agreement(oof_decisions),
        "five_seed_confirmation_decision_agreement": _seed_agreement(confirm_decisions),
        "seed_reports": reports,
        "checks": checks,
        "model_path": str(model_path.relative_to(ROOT)),
    }


def build(*, root: Path = ROOT) -> dict[str, Any]:
    source_dir = root / "outputs/pm_v1_5b_mp_ms_antishortcut_v1"
    source_report = source_dir / "preflight_report.json"
    rows = [dict(row) for row in iter_jsonl(source_dir / "opportunity_rows.jsonl")]
    out_dir = root / "outputs/pm_v1_5b_mp_ms_opportunity_heads_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        component: _fit_component(
            component,
            [row for row in rows if row["component"] == component and row["evidence_role"] == "development_fit"],
            [row for row in rows if row["component"] == component and row["evidence_role"] == "sealed_confirmation"],
            out_dir,
        )
        for component in COMPONENTS
    }
    internal_pass = all(row["status"].startswith("PASS_INTERNAL") for row in reports.values())
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_INTERNAL_MP_MS_AWAITING_OUTCOME_BLIND_EXTERNAL_TRANSPORT" if internal_pass else "FAIL_INTERNAL_MP_MS_ROOT_REPAIR",
        "scientific_scope": "Source-specific post-candidate opportunity routing; not retrieval correctness, generator uptake, or response utility.",
        "components": reports,
        "external_transport_gate_evaluated": False,
        "external_quality_risk_or_response_outcome_read": False,
        "contract": CONTRACT,
        "inputs": {
            "anti_shortcut_preflight_sha256": sha256_file(source_report),
            "opportunity_rows_sha256": sha256_file(source_dir / "opportunity_rows.jsonl"),
            "contract_sha256": sha256_file(root / CONTRACT),
        },
    }
    write_jsonl(out_dir / "fit_and_confirmation_rows_audit.jsonl", rows)
    write_json(out_dir / "fit_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "fit_report_sha256": sha256_file(out_dir / "fit_report.json"),
            "rows_sha256": sha256_file(out_dir / "fit_and_confirmation_rows_audit.jsonl"),
            "models": {
                component: sha256_file(out_dir / f"{component.lower()}_opportunity_router_v1_5b.joblib")
                for component in COMPONENTS
            },
            "confirmation_used_for_feature_or_hyperparameter_selection": False,
            "external_outcome_used": False,
        },
    )
    return report


def main() -> None:
    report = build()
    print({
        "protocol": report["protocol"],
        "status": report["status"],
        "components": {component: value["status"] for component, value in report["components"].items()},
    })


if __name__ == "__main__":
    main()
