#!/usr/bin/env python3
"""Train a small MS opportunity router without response-outcome labels."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.contracts import MemoryBackendRecord, MemorySource
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.text import normalize_space


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-ms-opportunity-router-fit-v1"
SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
FEATURES = (
    "candidate_state_match_score",
    "candidate_not_exactly_visible",
    "candidate_contains_prior_response_or_outcome",
    "current_turn_can_use_prior_continuity",
    "candidate_not_duplicated_by_enabled_resources",
    "background_MP_on",
    "background_ME_on",
    "background_RS_on",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=0.3,
                    penalty="l2",
                    solver="liblinear",
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = probability >= 0.5
    prevalence = float(y.mean())
    prior = np.full(len(y), prevalence, dtype=float)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "positive_recall": float(
            recall_score(y, prediction, pos_label=1, zero_division=0)
        ),
        "brier": float(brier_score_loss(y, probability)),
        "prevalence_prior_brier": float(brier_score_loss(y, prior)),
        "brier_gain_vs_prevalence_prior": float(
            brier_score_loss(y, prior) - brier_score_loss(y, probability)
        ),
        "predicted_on": int(prediction.sum()),
        "predicted_off": int((~prediction).sum()),
        "predicted_on_fraction": float(prediction.mean()),
        "predicted_off_fraction": float((~prediction).mean()),
    }


def _extract(
    contrasts: list[dict[str, Any]],
    design_by_slot: dict[str, dict[str, Any]],
    states: dict[str, dict[str, Any]],
    backends: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_rows: list[list[float]] = []
    y: list[int] = []
    users: list[str] = []
    themes: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    for row in contrasts:
        incremental = row["component_candidate_observation"]["incremental_value"]
        checks = incremental["checks"]
        descriptor = row["component_candidate_observation"]["descriptor"]
        state = states[str(row["state_id"])]
        catalog = backends[str(state["card_id"])]
        control_items = [
            catalog[str(memory_id)]
            for memory_id in row[
                "selected_memory_ids_by_action_generation_only"
            ][row["control_action"]]
        ]
        treatment_ms = [
            catalog[str(memory_id)]
            for memory_id in row[
                "selected_memory_ids_by_action_generation_only"
            ][row["treatment_action"]]
            if catalog[str(memory_id)].source is MemorySource.MS
        ]
        cross_resource_exact_redundancy = any(
            (
                normalize_space(ms.text).lower()
                in normalize_space(other.text).lower()
                or normalize_space(other.text).lower()
                in normalize_space(ms.text).lower()
            )
            for ms in treatment_ms
            for other in control_items
        )
        target = int(
            incremental["candidate_incremental_alignment_score"]
            and not cross_resource_exact_redundancy
        )
        values = {
            "candidate_state_match_score": float(
                row["model_features"]["candidate_state_match_score"]
            ),
            "candidate_not_exactly_visible": float(
                checks["not_exactly_visible_to_seeker"]
            ),
            "candidate_contains_prior_response_or_outcome": float(
                checks["contains_prior_response_or_outcome"]
            ),
            "current_turn_can_use_prior_continuity": float(
                checks["current_turn_can_use_prior_continuity"]
            ),
            "candidate_not_duplicated_by_enabled_resources": float(
                not cross_resource_exact_redundancy
            ),
            "background_MP_on": float(row["model_features"]["background_MP_on"]),
            "background_ME_on": float(row["model_features"]["background_ME_on"]),
            "background_RS_on": float(row["model_features"]["background_RS_on"]),
        }
        if "candidate_incremental_alignment_score" in values:
            raise RuntimeError("composite opportunity target leaked into features")
        slot = str(row["contrast_slot_id"])
        design = design_by_slot[slot]
        x_rows.append([values[name] for name in FEATURES])
        y.append(target)
        users.append(str(row["user_id"]))
        themes.append(str(design["theme_key"]))
        audit_rows.append(
            {
                "protocol": PROTOCOL,
                "contrast_slot_id": slot,
                "user_id": row["user_id"],
                "evaluation_theme_group_private_not_model_input": design[
                    "theme_key"
                ],
                "target": target,
                "pre_background_opportunity_target": int(
                    incremental["candidate_incremental_alignment_score"]
                ),
                "cross_resource_exact_redundancy": (
                    cross_resource_exact_redundancy
                ),
                "features": values,
                "response_or_judge_outcome_read": False,
            }
        )
    return (
        np.asarray(x_rows, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(users, dtype=object),
        np.asarray(themes, dtype=object),
        audit_rows,
    )


def _stratified_oof(
    x: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    probability = np.zeros(len(y), dtype=float)
    for train, test in splitter.split(x, y):
        model = _pipeline(seed)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return probability


def _leave_theme_out(
    x: np.ndarray, y: np.ndarray, themes: np.ndarray, seed: int
) -> np.ndarray:
    probability = np.zeros(len(y), dtype=float)
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(x, y, themes):
        model = _pipeline(seed)
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return probability


def build(*, blueprint_dir: Path, out_dir: Path) -> dict[str, Any]:
    contrasts = _rows(blueprint_dir / "d3_ms_replacement_contrasts.jsonl")
    design_rows = _rows(blueprint_dir / "private_design_strata.jsonl")
    design_by_slot = {
        str(row["contrast_slot_id"]): row for row in design_rows
    }
    states = {
        str(row["state_id"]): row
        for row in _rows(blueprint_dir / "runtime_states.jsonl")
    }
    backends = {
        record.card_id: {item.memory_id: item for item in record.items}
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(blueprint_dir / "memory_backend.jsonl")
        )
    }
    x, y, users, themes, audit_rows = _extract(
        contrasts, design_by_slot, states, backends
    )
    if len(set(users.tolist())) != len(users):
        raise RuntimeError("MS opportunity rows must use independent users")
    if min(Counter(y.tolist()).values()) < 8:
        raise RuntimeError("insufficient support for both opportunity decisions")

    seed_reports: list[dict[str, Any]] = []
    all_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        ordinary = _stratified_oof(x, y, seed)
        theme_holdout = _leave_theme_out(x, y, themes, seed)
        all_predictions.append(theme_holdout >= 0.5)
        seed_reports.append(
            {
                "seed": seed,
                "user_independent_stratified_oof": _metrics(y, ordinary),
                "content_theme_holdout": _metrics(y, theme_holdout),
            }
        )
    prediction_matrix = np.stack(all_predictions, axis=0)
    prediction_agreement = float(
        np.mean(
            np.logical_or(
                np.all(prediction_matrix, axis=0),
                np.all(~prediction_matrix, axis=0),
            )
        )
    )
    theme_bas = [
        row["content_theme_holdout"]["balanced_accuracy"]
        for row in seed_reports
    ]
    theme_recalls = [
        row["content_theme_holdout"]["positive_recall"]
        for row in seed_reports
    ]
    theme_brier_gains = [
        row["content_theme_holdout"]["brier_gain_vs_prevalence_prior"]
        for row in seed_reports
    ]
    final_model = _pipeline(SEEDS[0])
    final_model.fit(x, y)

    checks = {
        "32_independent_users": len(users) == 32 and len(set(users)) == 32,
        "at_least_8_on_and_8_off_targets": min(Counter(y.tolist()).values()) >= 8,
        "composite_target_not_in_features": (
            "candidate_incremental_alignment_score" not in FEATURES
        ),
        "no_response_or_judge_outcome_read": all(
            not row["response_or_judge_outcome_read"] for row in audit_rows
        ),
        "five_seed_theme_holdout_ba_min_0_65": min(theme_bas) >= 0.65,
        "five_seed_positive_recall_min_0_60": min(theme_recalls) >= 0.60,
        "five_seed_brier_beats_prior": min(theme_brier_gains) > 0.0,
        "both_decisions_each_seed": all(
            row["content_theme_holdout"]["predicted_on"] > 0
            and row["content_theme_holdout"]["predicted_off"] > 0
            for row in seed_reports
        ),
        "seed_prediction_agreement_min_0_95": prediction_agreement >= 0.95,
    }
    status = (
        "PASS_CONTROLLED_MS_ROUTING_LEARNABILITY"
        if all(checks.values())
        else "FAIL_CONTROLLED_MS_ROUTING_LEARNABILITY"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "training_rows_audit.jsonl", audit_rows)
    joblib.dump(final_model, out_dir / "ms_opportunity_router.joblib")
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "scientific_scope": (
            "controlled outcome-blind MS opportunity routing; not retrieval, "
            "generator uptake, or realized response-quality prediction"
        ),
        "rows": len(y),
        "independent_users": len(set(users.tolist())),
        "class_counts": {
            str(key): value for key, value in sorted(Counter(y.tolist()).items())
        },
        "feature_names": list(FEATURES),
        "forbidden_composite_feature": "candidate_incremental_alignment_score",
        "evaluation": seed_reports,
        "five_seed_theme_holdout_ba_mean": float(np.mean(theme_bas)),
        "five_seed_theme_holdout_ba_std": float(np.std(theme_bas)),
        "five_seed_prediction_agreement": prediction_agreement,
        "checks": checks,
        "claim_boundary": [
            "This demonstrates learnability on content-disjoint controlled themes.",
            "It does not demonstrate that every enabled MS changes a stochastic reply or wins a human preference comparison.",
            "Real-catalog retrieval, execution, grounded use, risk, cost, and external transport remain separate gates.",
        ],
    }
    write_json(out_dir / "fit_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": status,
            "fit_report_sha256": sha256_file(out_dir / "fit_report.json"),
            "training_rows_audit_sha256": sha256_file(
                out_dir / "training_rows_audit.jsonl"
            ),
            "model_sha256": sha256_file(out_dir / "ms_opportunity_router.joblib"),
            "measurement_layers_contract_sha256": sha256_file(
                ROOT
                / "data/pm_v1_5_contracts/resource_routing_measurement_layers_v1.json"
            ),
        },
    )
    return report


def main() -> None:
    report = build(
        blueprint_dir=ROOT / "outputs/pm_v1_5_ms_incremental_value_step0_v1",
        out_dir=ROOT / "outputs/pm_v1_5_ms_opportunity_router_fit_v1",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "protocol",
                    "status",
                    "rows",
                    "independent_users",
                    "class_counts",
                    "five_seed_theme_holdout_ba_mean",
                    "five_seed_theme_holdout_ba_std",
                    "five_seed_prediction_agreement",
                    "checks",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
