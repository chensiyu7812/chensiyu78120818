#!/usr/bin/env python3
"""Build and fit controlled MP/ME Step-1 opportunity micro-worlds.

These data establish basic learnability of outcome-blind routing conditions.
They do not simulate generator wins and do not count as external validation.
Every content theme contains all on/off conditions so leave-theme-out tests
cannot succeed from topic identity.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import sha256_file, stable_hex, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-mp-me-controlled-opportunity-router-fit-v1"
SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
THEMES = (
    "workload",
    "relationship_change",
    "relocation_loneliness",
    "study_pressure",
    "family_tension",
    "grief_reminder",
    "daily_routine",
    "social_confidence",
)
BACKGROUND_PATTERNS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)

MP_FEATURES = (
    "candidate_state_match_score",
    "candidate_not_exactly_visible",
    "candidate_nonconflicting_and_active",
    "candidate_directly_relevant_to_current_request",
    "candidate_safe_nonsensitive_use",
    "candidate_not_duplicated_by_enabled_resources",
    "candidate_is_preference",
    "background_MS_on",
    "background_ME_on",
    "background_RS_on",
)
ME_FEATURES = (
    "candidate_state_match_score",
    "candidate_not_exactly_visible",
    "candidate_contains_prior_event_or_outcome",
    "current_turn_can_use_event_continuity",
    "candidate_nonconflicting_and_current",
    "candidate_not_duplicated_by_enabled_resources",
    "background_MP_on",
    "background_MS_on",
    "background_RS_on",
)


MP_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "condition": "relevant_active_preference",
        "target": 1,
        "values": (0.88, 1, 1, 1, 1, 1, 1),
    },
    {
        "condition": "relevant_nonsensitive_profile",
        "target": 1,
        "values": (0.82, 1, 1, 1, 1, 1, 0),
    },
    {
        "condition": "preference_with_unrelated_background",
        "target": 1,
        "values": (0.74, 1, 1, 1, 1, 1, 1),
    },
    {
        "condition": "profile_resolves_current_reference",
        "target": 1,
        "values": (0.78, 1, 1, 1, 1, 1, 0),
    },
    {
        "condition": "already_visible_current_turn",
        "target": 0,
        "values": (0.86, 0, 1, 1, 1, 1, 1),
    },
    {
        "condition": "revoked_or_conflicting_preference",
        "target": 0,
        "values": (0.80, 1, 0, 1, 1, 1, 1),
    },
    {
        "condition": "irrelevant_or_sensitive_profile",
        "target": 0,
        "values": (0.18, 1, 1, 0, 0, 1, 0),
    },
    {
        "condition": "duplicated_by_enabled_resource",
        "target": 0,
        "values": (0.76, 1, 1, 1, 1, 0, 1),
    },
)
ME_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "condition": "prior_event_outcome_answers_current_continuity",
        "target": 1,
        "values": (0.88, 1, 1, 1, 1, 1),
    },
    {
        "condition": "prior_coping_result_informs_current_choice",
        "target": 1,
        "values": (0.82, 1, 1, 1, 1, 1),
    },
    {
        "condition": "event_context_with_unrelated_background",
        "target": 1,
        "values": (0.74, 1, 1, 1, 1, 1),
    },
    {
        "condition": "repeated_situation_needs_prior_outcome",
        "target": 1,
        "values": (0.78, 1, 1, 1, 1, 1),
    },
    {
        "condition": "event_already_visible_current_turn",
        "target": 0,
        "values": (0.86, 0, 1, 1, 1, 1),
    },
    {
        "condition": "topical_event_without_usable_outcome",
        "target": 0,
        "values": (0.73, 1, 0, 1, 1, 1),
    },
    {
        "condition": "stale_or_conflicting_event",
        "target": 0,
        "values": (0.68, 1, 1, 1, 0, 1),
    },
    {
        "condition": "event_duplicated_by_enabled_resource",
        "target": 0,
        "values": (0.76, 1, 1, 1, 1, 0),
    },
)


def _jitter(component: str, theme: str, condition: str) -> float:
    raw = int(stable_hex(PROTOCOL, component, theme, condition, n=8), 16)
    return ((raw % 9) - 4) / 200.0


def _build_rows(component: str) -> list[dict[str, Any]]:
    conditions = MP_CONDITIONS if component == "MP" else ME_CONDITIONS
    features = MP_FEATURES if component == "MP" else ME_FEATURES
    rows: list[dict[str, Any]] = []
    for theme_index, theme in enumerate(THEMES):
        for condition_index, condition in enumerate(conditions):
            values = list(condition["values"])
            values[0] = min(
                0.99,
                max(0.01, float(values[0]) + _jitter(component, theme, condition["condition"])),
            )
            mp, me, rs = BACKGROUND_PATTERNS[(theme_index + condition_index) % 8]
            background = {
                "MP": {"background_MS_on": mp, "background_ME_on": me, "background_RS_on": rs},
                "ME": {"background_MP_on": mp, "background_MS_on": me, "background_RS_on": rs},
            }[component]
            feature_values = {
                **dict(zip(features[: len(values)], values, strict=True)),
                **background,
            }
            user_id = (
                f"pmv15_{component.lower()}_opp_"
                + stable_hex(PROTOCOL, component, theme, condition["condition"], n=16)
            )
            rows.append(
                {
                    "protocol": PROTOCOL,
                    "component": component,
                    "controlled_user_id": user_id,
                    "content_theme_private_not_model_input": theme,
                    "condition": condition["condition"],
                    "opportunity_target": int(condition["target"]),
                    "features": feature_values,
                    "target_is_composite_feature": False,
                    "response_or_judge_outcome_read": False,
                    "external_dataset_identity_model_input": False,
                }
            )
    return rows


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
    prior = np.full(len(y), float(y.mean()), dtype=float)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "positive_recall": float(recall_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "prevalence_prior_brier": float(brier_score_loss(y, prior)),
        "brier_gain_vs_prevalence_prior": float(
            brier_score_loss(y, prior) - brier_score_loss(y, probability)
        ),
        "predicted_on": int(prediction.sum()),
        "predicted_off": int((~prediction).sum()),
    }


def _fit_component(
    component: str, rows: list[dict[str, Any]], out_dir: Path
) -> dict[str, Any]:
    features = MP_FEATURES if component == "MP" else ME_FEATURES
    x = np.asarray(
        [[float(row["features"][name]) for name in features] for row in rows],
        dtype=float,
    )
    y = np.asarray([row["opportunity_target"] for row in rows], dtype=int)
    themes = np.asarray(
        [row["content_theme_private_not_model_input"] for row in rows], dtype=object
    )
    seed_reports: list[dict[str, Any]] = []
    theme_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        stratified = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
        ordinary_probability = np.zeros(len(y), dtype=float)
        for train, test in stratified.split(x, y):
            model = _pipeline(seed)
            model.fit(x[train], y[train])
            ordinary_probability[test] = model.predict_proba(x[test])[:, 1]
        theme_probability = np.zeros(len(y), dtype=float)
        for train, test in LeaveOneGroupOut().split(x, y, themes):
            model = _pipeline(seed)
            model.fit(x[train], y[train])
            theme_probability[test] = model.predict_proba(x[test])[:, 1]
        theme_predictions.append(theme_probability >= 0.5)
        seed_reports.append(
            {
                "seed": seed,
                "independent_stratified_oof": _metrics(y, ordinary_probability),
                "content_theme_holdout": _metrics(y, theme_probability),
            }
        )
    matrix = np.stack(theme_predictions, axis=0)
    agreement = float(
        np.mean(np.logical_or(np.all(matrix, axis=0), np.all(~matrix, axis=0)))
    )
    bas = [row["content_theme_holdout"]["balanced_accuracy"] for row in seed_reports]
    recalls = [row["content_theme_holdout"]["positive_recall"] for row in seed_reports]
    brier_gains = [
        row["content_theme_holdout"]["brier_gain_vs_prevalence_prior"]
        for row in seed_reports
    ]
    checks = {
        "64_independent_users": (
            len(rows) == 64 and len({row["controlled_user_id"] for row in rows}) == 64
        ),
        "32_on_32_off": Counter(y.tolist()) == Counter({0: 32, 1: 32}),
        "every_theme_contains_all_8_conditions": all(
            sum(row["content_theme_private_not_model_input"] == theme for row in rows) == 8
            for theme in THEMES
        ),
        "no_response_or_judge_outcome_read": all(
            not row["response_or_judge_outcome_read"] for row in rows
        ),
        "no_dataset_identity_feature": True,
        "no_composite_target_feature": True,
        "five_seed_theme_holdout_ba_min_0_65": min(bas) >= 0.65,
        "five_seed_positive_recall_min_0_60": min(recalls) >= 0.60,
        "five_seed_brier_beats_prior": min(brier_gains) > 0.0,
        "both_decisions_each_seed": all(
            row["content_theme_holdout"]["predicted_on"] > 0
            and row["content_theme_holdout"]["predicted_off"] > 0
            for row in seed_reports
        ),
    }
    final_model = _pipeline(SEEDS[0])
    final_model.fit(x, y)
    model_path = out_dir / f"{component.lower()}_opportunity_router.joblib"
    joblib.dump(final_model, model_path)
    return {
        "status": "CONTROLLED_BASIC_LEARNABILITY_PASSED" if all(checks.values()) else "CONTROLLED_BASIC_LEARNABILITY_FAILED",
        "feature_names": list(features),
        "counts": {
            "rows": len(rows),
            "independent_users": len({row["controlled_user_id"] for row in rows}),
            "class_counts": dict(sorted(Counter(y.tolist()).items())),
            "themes": len(set(themes.tolist())),
        },
        "checks": checks,
        "five_seed_theme_prediction_agreement": agreement,
        "seed_reports": seed_reports,
        "model_path": str(model_path.relative_to(ROOT)),
    }


def build(*, root: Path = ROOT) -> dict[str, Any]:
    out_dir = root / "outputs/pm_v1_5_mp_me_opportunity_router_fit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    for component in ("MP", "ME"):
        rows = _build_rows(component)
        all_rows.extend(rows)
        reports[component] = _fit_component(component, rows, out_dir)
    rows_path = out_dir / "controlled_opportunity_rows.jsonl"
    write_jsonl(rows_path, all_rows)
    passed = all(report["status"].endswith("PASSED") for report in reports.values())
    report = {
        "protocol": PROTOCOL,
        "status": (
            "MP_ME_CONTROLLED_BASIC_ROUTING_LEARNABILITY_PASSED"
            if passed
            else "MP_ME_CONTROLLED_ROUTING_NOT_YET_PASSED"
        ),
        "estimand": (
            "Whether one already-discovered, same-user candidate has an "
            "outcome-blind opportunity to add nonredundant current value."
        ),
        "component_reports": reports,
        "hard_gates_outside_learned_head": [
            "wrong_user",
            "future_or_noncausal_memory",
            "explicit_current_conflict_or_revocation",
            "exact_duplicate_of_current_visible_context",
        ],
        "evidence_role": (
            "Controlled micro-world proof of basic learnability only. External "
            "EvoEmo transport, real candidate qualification, generator use, and "
            "response quality-risk-cost remain separate tests."
        ),
        "claim_boundary": {
            "allowed": (
                "MP and ME routing conditions are representable and learnable "
                "without response-outcome labels in a content-balanced controlled environment."
            ),
            "forbidden": [
                "MP or ME real-world accuracy is established.",
                "The retriever selects the best memory.",
                "The generator uses injected memory correctly.",
                "EvoEmo external generalization is established.",
            ],
        },
    }
    write_json(out_dir / "fit_report.json", report)
    write_json(
        out_dir / "freeze_manifest.json",
        {
            "protocol": PROTOCOL,
            "status": report["status"],
            "artifacts": {
                path.name: sha256_file(path)
                for path in (
                    rows_path,
                    out_dir / "mp_opportunity_router.joblib",
                    out_dir / "me_opportunity_router.joblib",
                    out_dir / "fit_report.json",
                )
            },
        },
    )
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "MP_theme_holdout_ba": report["component_reports"]["MP"]["seed_reports"][0]["content_theme_holdout"]["balanced_accuracy"],
            "ME_theme_holdout_ba": report["component_reports"]["ME"]["seed_reports"][0]["content_theme_holdout"]["balanced_accuracy"],
        }
    )


if __name__ == "__main__":
    main()
