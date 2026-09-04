#!/usr/bin/env python3
"""Fit the RS Step-1 opportunity head and test it on human H2 labels.

The development target is an outcome-blind transparent opportunity rule.  Its
internal fit is only a controlled learnability check.  The primary diagnostic
reported here is transfer to the existing H2 human judgement of whether the
fixed Bank contains at least one safe, nonredundant card worth considering.
Card ranking and response quality are deliberately outside this estimand.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_strategy_rag_runtime import observable_flags


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-rs-opportunity-router-fit-v1"
SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
FEATURES = (
    "active_high_stakes",
    "explicit_stop",
    "listen_only",
    "question_repetition_block",
    "open_expression_opportunity",
    "focused_clarification_opportunity",
    "paraphrase_check_opportunity",
    "grounded_validation_opportunity",
    "explicit_advice_welcome",
    "one_low_risk_step_available",
    "transition_opportunity",
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


def _vector(flags: dict[str, Any]) -> list[float]:
    return [float(bool(flags.get(name, False))) for name in FEATURES]


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
    }


def build(*, root: Path = ROOT) -> dict[str, Any]:
    development_path = (
        root / "outputs/pm_v1_5_rs_opportunity_layer_v1/opportunity_states.jsonl"
    )
    h2_dir = root / "outputs/pm_v1_5_h2_retrieval_qualification_v1_candidate"
    h2_annotations_path = (
        root / "outputs/pm_v1_5_h2_retrieval_qualification_v1/h2_annotations_frozen.jsonl"
    )
    out_dir = root / "outputs/pm_v1_5_rs_opportunity_router_fit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    packet = {row["review_item_id"]: row for row in _rows(h2_dir / "h2_review_packet.jsonl")}
    private = {
        row["review_item_id"]: row
        for row in _rows(h2_dir / "private_ranker_audit.jsonl")
    }
    annotations = _rows(h2_annotations_path)
    if set(packet) != set(private) or set(packet) != {
        row["review_item_id"] for row in annotations
    }:
        raise RuntimeError("H2 packet, private lineage, and annotations do not align")

    h2_dialogues = {str(row["source_dialogue_id"]) for row in private.values()}
    development_all = _rows(development_path)
    development = [
        row
        for row in development_all
        if str(row["source_dialogue_id"]) not in h2_dialogues
    ]
    removed_overlap = len(development_all) - len(development)
    if removed_overlap == 0:
        raise RuntimeError("expected the overlap audit to be active")

    x_train: list[list[float]] = []
    y_train: list[int] = []
    training_audit: list[dict[str, Any]] = []
    flag_replay_mismatches = 0
    for row in development:
        replay = observable_flags(list(row["recent_dialogue"]))
        if replay != row["observable_flags"]:
            flag_replay_mismatches += 1
        x_train.append(_vector(replay))
        y_train.append(int(row["opportunity_y"]))
        training_audit.append(
            {
                "protocol": PROTOCOL,
                "source_dialogue_id": row["source_dialogue_id"],
                "target": int(row["opportunity_y"]),
                "features": {name: bool(replay[name]) for name in FEATURES},
                "label_source": "transparent_outcome_blind_rule",
                "response_or_judge_outcome_read": False,
                "excluded_from_h2": True,
            }
        )
    if flag_replay_mismatches:
        raise RuntimeError(f"runtime flag replay mismatch: {flag_replay_mismatches}")

    x_test: list[list[float]] = []
    y_test: list[int] = []
    h2_audit: list[dict[str, Any]] = []
    for annotation in annotations:
        review_id = str(annotation["review_item_id"])
        label = str(annotation["rs_opportunity"])
        if label not in {"yes", "no"}:
            raise RuntimeError(f"non-definitive H2 label: {review_id}")
        flags = observable_flags(list(packet[review_id]["visible_dialogue"]))
        x_test.append(_vector(flags))
        y_test.append(int(label == "yes"))
        h2_audit.append(
            {
                "protocol": PROTOCOL,
                "review_item_id": review_id,
                "h2_state_id": annotation["h2_state_id"],
                "source_dialogue_id_private_not_model_input": private[review_id][
                    "source_dialogue_id"
                ],
                "human_opportunity_target": int(label == "yes"),
                "features": {name: bool(flags[name]) for name in FEATURES},
                "response_or_judge_outcome_read": False,
            }
        )

    x_train_a = np.asarray(x_train, dtype=float)
    y_train_a = np.asarray(y_train, dtype=int)
    x_test_a = np.asarray(x_test, dtype=float)
    y_test_a = np.asarray(y_test, dtype=int)
    if min(Counter(y_train).values()) < 8 or min(Counter(y_test).values()) < 8:
        raise RuntimeError("insufficient support for both RS opportunity decisions")

    seed_reports: list[dict[str, Any]] = []
    h2_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        train_oof = np.zeros(len(y_train_a), dtype=float)
        for fit_idx, test_idx in splitter.split(x_train_a, y_train_a):
            fold_model = _pipeline(seed)
            fold_model.fit(x_train_a[fit_idx], y_train_a[fit_idx])
            train_oof[test_idx] = fold_model.predict_proba(x_train_a[test_idx])[:, 1]
        model = _pipeline(seed)
        model.fit(x_train_a, y_train_a)
        h2_probability = model.predict_proba(x_test_a)[:, 1]
        h2_predictions.append(h2_probability >= 0.5)
        seed_reports.append(
            {
                "seed": seed,
                "controlled_rule_replay_oof": _metrics(y_train_a, train_oof),
                "human_h2_transfer": _metrics(y_test_a, h2_probability),
            }
        )

    prediction_matrix = np.stack(h2_predictions, axis=0)
    seed_agreement = float(
        np.mean(
            np.logical_or(
                np.all(prediction_matrix, axis=0),
                np.all(~prediction_matrix, axis=0),
            )
        )
    )
    h2_bas = [row["human_h2_transfer"]["balanced_accuracy"] for row in seed_reports]
    h2_recalls = [row["human_h2_transfer"]["positive_recall"] for row in seed_reports]
    h2_brier_gains = [
        row["human_h2_transfer"]["brier_gain_vs_prevalence_prior"]
        for row in seed_reports
    ]
    both_decisions = all(
        row["human_h2_transfer"]["predicted_on"] > 0
        and row["human_h2_transfer"]["predicted_off"] > 0
        for row in seed_reports
    )
    checks = {
        "h2_dialogue_overlap_removed_before_fit": removed_overlap > 0,
        "no_h2_dialogue_in_training": not (
            {str(row["source_dialogue_id"]) for row in development} & h2_dialogues
        ),
        "outcome_blind_features_only": True,
        "no_response_or_judge_outcome_read": True,
        "at_least_8_per_class_train_and_h2": (
            min(Counter(y_train).values()) >= 8 and min(Counter(y_test).values()) >= 8
        ),
        "five_seed_h2_balanced_accuracy_min_0_65": min(h2_bas) >= 0.65,
        "five_seed_h2_positive_recall_min_0_60": min(h2_recalls) >= 0.60,
        "five_seed_h2_brier_beats_prior": min(h2_brier_gains) > 0.0,
        "both_decisions_each_seed": both_decisions,
    }
    passed = all(checks.values())
    final_model = _pipeline(SEEDS[0])
    final_model.fit(x_train_a, y_train_a)
    joblib.dump(final_model, out_dir / "rs_opportunity_router.joblib")
    write_jsonl(out_dir / "training_rows_audit.jsonl", training_audit)
    write_jsonl(out_dir / "h2_transfer_rows_audit.jsonl", h2_audit)

    report = {
        "protocol": PROTOCOL,
        "status": (
            "BASIC_HUMAN_ALIGNED_RS_ROUTING_LEARNABILITY_PASSED"
            if passed
            else "RS_ROUTING_LEARNABILITY_NOT_YET_PASSED"
        ),
        "estimand": (
            "Whether the fixed Strategy Bank contains a safe, nonredundant "
            "resource opportunity now; not which card wins and not whether a "
            "single stochastic RS response beats R0."
        ),
        "development_label_role": (
            "Transparent rule supervision. OOF measures controlled rule replay "
            "only and is not claimed as independent human accuracy."
        ),
        "primary_validation_role": (
            "Pre-existing human H2 opportunity labels, after removing all H2 "
            "dialogue overlap from model fitting. H2 is development-informed, "
            "not an untouched external test."
        ),
        "feature_names": list(FEATURES),
        "counts": {
            "development_rows_before_overlap_filter": len(development_all),
            "development_rows_after_overlap_filter": len(development),
            "h2_overlap_rows_removed": removed_overlap,
            "development_classes": dict(sorted(Counter(y_train).items())),
            "h2_human_classes": dict(sorted(Counter(y_test).items())),
            "h2_states": len(y_test),
        },
        "checks": checks,
        "five_seed_h2_prediction_agreement": seed_agreement,
        "seed_reports": seed_reports,
        "selected_model_seed": SEEDS[0],
        "separate_unresolved_layer": {
            "candidate_ranking": (
                "H2 transparent Top-1 acceptable rate was 19/31; this router "
                "result does not repair or hide that item-ranking limitation."
            ),
            "generator_use": "Not evaluated by this router.",
            "response_quality_risk_cost": "Must remain a downstream paired evaluation.",
        },
        "claim_boundary": {
            "allowed": (
                "A small outcome-blind RS opportunity head can reproduce a "
                "basic human-aligned open/close distinction on H2."
            ),
            "forbidden": [
                "RS card selection is solved.",
                "RS improves every response.",
                "H2 is an untouched external test.",
                "The model learned latent psychological need diagnosis.",
            ],
        },
        "inputs": {
            "development": {
                "path": str(development_path.relative_to(root)),
                "sha256": sha256_file(development_path),
            },
            "h2_packet": {
                "path": str((h2_dir / "h2_review_packet.jsonl").relative_to(root)),
                "sha256": sha256_file(h2_dir / "h2_review_packet.jsonl"),
            },
            "h2_annotations": {
                "path": str(h2_annotations_path.relative_to(root)),
                "sha256": sha256_file(h2_annotations_path),
            },
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
                    out_dir / "rs_opportunity_router.joblib",
                    out_dir / "training_rows_audit.jsonl",
                    out_dir / "h2_transfer_rows_audit.jsonl",
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
            "h2_balanced_accuracy": report["seed_reports"][0][
                "human_h2_transfer"
            ]["balanced_accuracy"],
            "h2_predicted_on": report["seed_reports"][0][
                "human_h2_transfer"
            ]["predicted_on"],
            "h2_predicted_off": report["seed_reports"][0][
                "human_h2_transfer"
            ]["predicted_off"],
        }
    )


if __name__ == "__main__":
    main()
