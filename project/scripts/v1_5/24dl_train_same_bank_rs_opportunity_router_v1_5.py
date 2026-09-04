#!/usr/bin/env python3
"""Train RS opportunity routing against the same frozen 80-card Bank as H2."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, stable_hex, write_json, write_jsonl
from metacom_pm.text import normalize_space
from metacom_pm.v1_5_strategy_rag_repair import (
    repaired_observable_opportunity_flags,
    repaired_rank_applicable_v4_cards,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-same-bank-rs-opportunity-router-fit-v1"
SELECTION_SEED = "pm-v1.5-same-bank-rs-opportunity-selection-v1"
SEEDS = (20260801, 20260802, 20260803, 20260804, 20260805)
FEATURES = (
    "active_high_stakes",
    "advice_welcome",
    "effort_or_progress_visible",
    "emotion_visible",
    "expanded_routine_closing",
    "explicit_stop",
    "factual_or_resource_request",
    "latest_visible_turn_not_seeker",
    "legal_occupational_misconduct",
    "listen_only",
    "low_burden",
    "pure_phatic",
    "routine_closing",
    "substance_dependent_safety",
    "substantive",
    "uncertainty_or_multi_concern",
    "violence_or_harm",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _seekers(dialogue: list[dict[str, Any]]) -> list[str]:
    return [
        normalize_space(row.get("content", ""))
        for row in dialogue
        if str(row.get("speaker")) == "seeker"
        and normalize_space(row.get("content", ""))
    ]


def _query(dialogue: list[dict[str, Any]], flags: dict[str, Any]) -> str:
    cues = [
        key.replace("_", " ")
        for key, value in flags.items()
        if isinstance(value, bool)
        and value
        and key
        not in {
            "substantive",
            "pure_phatic",
            "routine_closing",
            "active_high_stakes",
            "explicit_stop",
            "ordinary_rag_hard_off",
        }
    ]
    return (
        "Represent this sentence for searching relevant passages: "
        "Choose one safe, topic-agnostic emotional-support technique. "
        f"Observable cues: {', '.join(cues) or 'none'}. "
        f"Recent seeker context: {' '.join(_seekers(dialogue)[-3:])}"
    )


def _observe(dialogue: list[dict[str, Any]], cards: list[dict[str, Any]]) -> dict[str, Any]:
    seekers = _seekers(dialogue)
    if not seekers:
        raise ValueError("dialogue has no seeker text")
    latest = seekers[-1]
    recent = " ".join(seekers[-3:])
    flags = repaired_observable_opportunity_flags(
        current_user_text=latest,
        recent_user_text=recent,
        visible_dialogue=dialogue,
    )
    ranked = repaired_rank_applicable_v4_cards(
        query=_query(dialogue, flags),
        current_user_text=latest,
        recent_user_text=recent,
        visible_dialogue=dialogue,
        cards=cards,
    )
    return {
        "flags": flags,
        "opportunity_y": int(bool(ranked)),
        "candidate_count": len(ranked),
    }


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
        "positive_recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(
            ((prediction == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
        ),
        "brier": float(brier_score_loss(y, probability)),
        "prevalence_prior_brier": float(brier_score_loss(y, prior)),
        "brier_gain_vs_prevalence_prior": float(
            brier_score_loss(y, prior) - brier_score_loss(y, probability)
        ),
        "predicted_on": int(prediction.sum()),
        "predicted_off": int((~prediction).sum()),
    }


def _choose_balanced(rows: list[dict[str, Any]], per_class: int = 80) -> list[dict[str, Any]]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dialogue[str(row["source_dialogue_id"])].append(row)
    one_per_dialogue: list[dict[str, Any]] = []
    for dialogue_id, values in by_dialogue.items():
        values.sort(
            key=lambda row: stable_hex(
                SELECTION_SEED,
                dialogue_id,
                row["source_turn_index"],
                n=32,
            )
        )
        one_per_dialogue.append(values[0])

    positives = sorted(
        [row for row in one_per_dialogue if row["opportunity_y"] == 1],
        key=lambda row: stable_hex(
            SELECTION_SEED, "on", row["source_dialogue_id"], n=32
        ),
    )
    negatives_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in one_per_dialogue:
        if row["opportunity_y"]:
            continue
        reasons = list(row["flags"].get("ordinary_rag_hard_off_reasons") or [])
        key = "+".join(sorted(reasons)) or "no_applicable_card"
        negatives_by_reason[key].append(row)
    for reason, values in negatives_by_reason.items():
        values.sort(
            key=lambda row: stable_hex(
                SELECTION_SEED, "off", reason, row["source_dialogue_id"], n=32
            )
        )
    negatives: list[dict[str, Any]] = []
    while len(negatives) < per_class:
        progressed = False
        for reason in sorted(negatives_by_reason):
            if negatives_by_reason[reason]:
                negatives.append(negatives_by_reason[reason].pop(0))
                progressed = True
                if len(negatives) == per_class:
                    break
        if not progressed:
            break
    if len(positives) < per_class or len(negatives) < per_class:
        raise RuntimeError(
            f"insufficient balanced same-Bank rows: on={len(positives)}, off={len(negatives)}"
        )
    selected = positives[:per_class] + negatives
    selected.sort(key=lambda row: str(row["source_dialogue_id"]))
    return selected


def build(*, root: Path = ROOT) -> dict[str, Any]:
    universe_path = (
        root
        / "outputs/pm_v1_5_esconv_train_strategy_inductive_audit_v1/clean_train_strategy_universe.jsonl"
    )
    bank_path = root / "outputs/pm_v1_5_strategy_bank_v4_final_v1/strategy_cards_v4_final.jsonl"
    h2_state_path = root / "outputs/pm_v1_5_h2_retrieval_state_freeze_v1/h2_states_private.jsonl"
    h2_annotation_path = (
        root / "outputs/pm_v1_5_h2_retrieval_qualification_v1/h2_annotations_frozen.jsonl"
    )
    formal_path = root / "data/esconv_test_v1_5_visible_v2_candidate/runtime_states.jsonl"
    accepted_source_path = (
        root
        / "outputs/pm_v1_5_strategy_g1_stage1_aggregation_v1/combined_private_weak_sources.jsonl"
    )
    out_dir = root / "outputs/pm_v1_5_same_bank_rs_opportunity_router_fit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = _rows(bank_path)
    if len(cards) != 80:
        raise RuntimeError("same-Bank RS fit requires the frozen 80-card V4 Bank")
    h2_states = _rows(h2_state_path)
    annotations = {
        str(row["h2_state_id"]): row for row in _rows(h2_annotation_path)
    }
    if len(h2_states) != 48 or set(annotations) != {
        str(row["h2_state_id"]) for row in h2_states
    }:
        raise RuntimeError("H2 state/annotation mismatch")
    h2_dialogues = {str(row["source_dialogue_id"]) for row in h2_states}
    formal_dialogues = {str(row["user_id"]) for row in _rows(formal_path)}
    bank_source_dialogues = {
        str(row["source_dialogue_id"]) for row in _rows(accepted_source_path)
    }
    blocked = h2_dialogues | formal_dialogues | bank_source_dialogues

    candidate_rows: list[dict[str, Any]] = []
    for source in _rows(universe_path):
        dialogue_id = str(source["source_dialogue_id"])
        dialogue = list(source.get("recent_dialogue") or [])
        if dialogue_id in blocked or not dialogue:
            continue
        observation = _observe(dialogue, cards)
        candidate_rows.append(
            {
                "protocol": PROTOCOL,
                "source_dialogue_id": dialogue_id,
                "source_turn_index": int(source["source_turn_index"]),
                **observation,
            }
        )
    selected = _choose_balanced(candidate_rows)
    x_train = np.asarray([_vector(row["flags"]) for row in selected], dtype=float)
    y_train = np.asarray([row["opportunity_y"] for row in selected], dtype=int)

    h2_audit: list[dict[str, Any]] = []
    for state in h2_states:
        observation = _observe(list(state["visible_dialogue"]), cards)
        annotation = annotations[str(state["h2_state_id"])]
        human = str(annotation["rs_opportunity"])
        if human not in {"yes", "no"}:
            raise RuntimeError("H2 human opportunity must be definitive")
        h2_audit.append(
            {
                "protocol": PROTOCOL,
                "h2_state_id": state["h2_state_id"],
                "source_dialogue_id_private_not_model_input": state["source_dialogue_id"],
                "human_opportunity_target": int(human == "yes"),
                "transparent_same_bank_opportunity": observation["opportunity_y"],
                "candidate_count_private_not_model_input": observation["candidate_count"],
                "flags": observation["flags"],
            }
        )
    x_h2 = np.asarray([_vector(row["flags"]) for row in h2_audit], dtype=float)
    y_h2 = np.asarray([row["human_opportunity_target"] for row in h2_audit], dtype=int)
    rule_h2 = np.asarray(
        [row["transparent_same_bank_opportunity"] for row in h2_audit], dtype=float
    )

    seed_reports: list[dict[str, Any]] = []
    h2_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y_train), dtype=float)
        for train_idx, test_idx in splitter.split(x_train, y_train):
            model = _pipeline(seed)
            model.fit(x_train[train_idx], y_train[train_idx])
            oof[test_idx] = model.predict_proba(x_train[test_idx])[:, 1]
        model = _pipeline(seed)
        model.fit(x_train, y_train)
        h2_probability = model.predict_proba(x_h2)[:, 1]
        h2_predictions.append(h2_probability >= 0.5)
        seed_reports.append(
            {
                "seed": seed,
                "controlled_same_bank_oof": _metrics(y_train, oof),
                "human_h2_transfer": _metrics(y_h2, h2_probability),
            }
        )
    matrix = np.stack(h2_predictions, axis=0)
    agreement = float(
        np.mean(np.logical_or(np.all(matrix, axis=0), np.all(~matrix, axis=0)))
    )
    h2_bas = [row["human_h2_transfer"]["balanced_accuracy"] for row in seed_reports]
    h2_recalls = [row["human_h2_transfer"]["positive_recall"] for row in seed_reports]
    h2_brier_gains = [
        row["human_h2_transfer"]["brier_gain_vs_prevalence_prior"]
        for row in seed_reports
    ]
    checks = {
        "same_80_card_bank_for_development_and_h2": True,
        "h2_dialogues_excluded_from_fit": not (
            {str(row["source_dialogue_id"]) for row in selected} & h2_dialogues
        ),
        "formal_esconv_test_excluded_from_fit": not (
            {str(row["source_dialogue_id"]) for row in selected} & formal_dialogues
        ),
        "bank_source_examples_excluded_from_fit": not (
            {str(row["source_dialogue_id"]) for row in selected} & bank_source_dialogues
        ),
        "80_on_80_off_independent_dialogues": (
            len(selected) == 160
            and len({row["source_dialogue_id"] for row in selected}) == 160
            and Counter(y_train.tolist()) == Counter({0: 80, 1: 80})
        ),
        "composite_hard_off_not_used_as_feature": "ordinary_rag_hard_off" not in FEATURES,
        "no_response_quality_risk_or_judge_outcome_read": True,
        "five_seed_h2_balanced_accuracy_min_0_65": min(h2_bas) >= 0.65,
        "five_seed_h2_positive_recall_min_0_60": min(h2_recalls) >= 0.60,
        "five_seed_h2_brier_beats_prior": min(h2_brier_gains) > 0.0,
        "both_decisions_each_seed": all(
            row["human_h2_transfer"]["predicted_on"] > 0
            and row["human_h2_transfer"]["predicted_off"] > 0
            for row in seed_reports
        ),
    }
    passed = all(checks.values())
    final_model = _pipeline(SEEDS[0])
    final_model.fit(x_train, y_train)
    joblib.dump(final_model, out_dir / "rs_opportunity_router.joblib")
    write_jsonl(out_dir / "training_rows_audit.jsonl", selected)
    write_jsonl(out_dir / "h2_transfer_rows_audit.jsonl", h2_audit)
    report = {
        "protocol": PROTOCOL,
        "status": (
            "BASIC_SAME_BANK_HUMAN_ALIGNED_RS_ROUTING_PASSED"
            if passed
            else "SAME_BANK_RS_ROUTING_NOT_YET_PASSED"
        ),
        "estimand": (
            "Outcome-blind Step1 RS opportunity for the frozen 80-card V4 Bank. "
            "Candidate ranking, prompt execution, and response effect are separate layers."
        ),
        "feature_names": list(FEATURES),
        "counts": {
            "development_rows": len(selected),
            "development_classes": dict(sorted(Counter(y_train.tolist()).items())),
            "h2_human_classes": dict(sorted(Counter(y_h2.tolist()).items())),
            "blocked_h2_dialogues": len(h2_dialogues),
            "blocked_formal_test_dialogues": len(formal_dialogues),
            "blocked_bank_source_dialogues": len(bank_source_dialogues),
        },
        "transparent_same_bank_rule_on_h2": _metrics(y_h2, rule_h2),
        "five_seed_h2_prediction_agreement": agreement,
        "seed_reports": seed_reports,
        "checks": checks,
        "separate_layers": {
            "candidate_ranking": "H2 Top-1 acceptable remains 19/31 and is not repaired here.",
            "generator_execution": "Not evaluated here.",
            "system_quality_risk_cost": "Retained as downstream paired evidence, not training gold.",
        },
        "claim_boundary": {
            "allowed": (
                "Under the same frozen Bank, a small transparent head learns a "
                "basic human-aligned RS opportunity distinction on development-informed H2."
            ),
            "forbidden": [
                "The card ranker is qualified.",
                "RS is always useful.",
                "H2 is an untouched external test.",
                "The head predicts single-sample generator wins.",
            ],
        },
        "inputs": {
            "bank": {"path": str(bank_path.relative_to(root)), "sha256": sha256_file(bank_path)},
            "universe": {"path": str(universe_path.relative_to(root)), "sha256": sha256_file(universe_path)},
            "h2_states": {"path": str(h2_state_path.relative_to(root)), "sha256": sha256_file(h2_state_path)},
            "h2_annotations": {"path": str(h2_annotation_path.relative_to(root)), "sha256": sha256_file(h2_annotation_path)},
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
    first = report["seed_reports"][0]["human_h2_transfer"]
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "h2_balanced_accuracy": first["balanced_accuracy"],
            "h2_positive_recall": first["positive_recall"],
            "h2_predicted_on": first["predicted_on"],
            "h2_predicted_off": first["predicted_off"],
        }
    )


if __name__ == "__main__":
    main()
