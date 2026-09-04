#!/usr/bin/env python3
"""Run the registered BGE-small candidate-semantic challenger."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from metacom_pm.contracts import StrategyMode, parse_action_id
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-transport-repaired-bge-small-challenger-v1"
MODEL_PATH = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
COMPONENTS = ("RS", "MP", "MS", "ME")
MEMORY_COMPONENTS = ("MP", "MS", "ME")
SEEDS = (20260730, 20260731, 20260732, 20260733, 20260734)
N_SPLITS = 5
C_VALUE = 0.03
THRESHOLD = 0.5
MIN_DECISION_FRACTION = 0.20


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups.tolist())
    result = np.asarray(
        [1.0 / counts[value] for value in groups],
        dtype=float,
    )
    return result * (len(result) / result.sum())


def _pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=C_VALUE,
                    penalty="l2",
                    solver="liblinear",
                    class_weight=None,
                    random_state=seed,
                    max_iter=2000,
                ),
            ),
        ]
    )


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Pipeline:
    model = _pipeline(seed)
    model.fit(x, y, logistic__sample_weight=_weights(groups))
    return model


def _metrics(
    y: np.ndarray,
    probability: np.ndarray,
    prior: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    weights = _weights(groups)
    action = probability >= THRESHOLD
    brier = float(
        brier_score_loss(y, probability, sample_weight=weights)
    )
    prior_brier = float(
        brier_score_loss(y, prior, sample_weight=weights)
    )
    return {
        "group_weighted_brier": brier,
        "group_weighted_prior_brier": prior_brier,
        "group_weighted_brier_gain_vs_prevalence_prior": (
            prior_brier - brier
        ),
        "group_weighted_balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                action,
                sample_weight=weights,
            )
        ),
        "group_weighted_positive_recall": float(
            recall_score(
                y,
                action,
                pos_label=1,
                sample_weight=weights,
                zero_division=0,
            )
        ),
        "predicted_on": int(action.sum()),
        "predicted_off": int((~action).sum()),
        "predicted_on_fraction": float(action.mean()),
        "predicted_off_fraction": float((~action).mean()),
    }


def _oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    probability = np.zeros(len(y), dtype=float)
    prior = np.zeros(len(y), dtype=float)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(x, y, groups),
        start=1,
    ):
        train_groups = groups[train_index]
        if set(train_groups) & set(groups[test_index]):
            raise RuntimeError("user group leaked across fold")
        model = _fit(
            x[train_index],
            y[train_index],
            train_groups,
            seed + fold,
        )
        probability[test_index] = model.predict_proba(
            x[test_index]
        )[:, 1]
        prior[test_index] = float(
            np.average(
                y[train_index],
                weights=_weights(train_groups),
            )
        )
    return {
        "probability": probability,
        "prior": prior,
        "metrics": _metrics(y, probability, prior, groups),
    }


def _visible_state_text(state: dict[str, Any]) -> str:
    dialogue = "\n".join(
        f"{turn['role']}: {turn['content']}"
        for turn in state["current_session_history"]
    )
    return (
        "Emotional-support dialogue before the next response.\n"
        f"Summary: {state.get('current_session_summary') or '(none)'}\n"
        f"{dialogue}\nuser: {state['current_user_text']}"
    )


def _encode(texts: list[str], model_path: Path) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            batch = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            batch = {
                key: value.to(device) for key, value in batch.items()
            }
            embedding = model(**batch).last_hidden_state[:, 0]
            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1,
            )
            chunks.append(embedding.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _gate(
    metrics: dict[str, Any],
    seed_std: float,
    primary_ba: float,
    primary_brier: float,
) -> dict[str, bool]:
    return {
        "balanced_accuracy_at_least_0_70": (
            metrics["group_weighted_balanced_accuracy"] >= 0.70
        ),
        "positive_recall_at_least_0_60": (
            metrics["group_weighted_positive_recall"] >= 0.60
        ),
        "brier_beats_prevalence_prior": (
            metrics["group_weighted_brier"]
            < metrics["group_weighted_prior_brier"]
        ),
        "on_and_off_each_at_least_0_20": (
            metrics["predicted_on_fraction"] >= MIN_DECISION_FRACTION
            and metrics["predicted_off_fraction"]
            >= MIN_DECISION_FRACTION
        ),
        "five_seed_ba_std_at_most_0_03": seed_std <= 0.03,
        "balanced_accuracy_beats_primary": (
            metrics["group_weighted_balanced_accuracy"] > primary_ba
        ),
        "brier_beats_primary": (
            metrics["group_weighted_brier"] < primary_brier
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_final_effect_labels_v1/"
        "component_effect_labels.jsonl",
    )
    parser.add_argument(
        "--memory-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1/"
        "memory_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--rs-blueprint",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1/"
        "rs_contrast_blueprint.jsonl",
    )
    parser.add_argument(
        "--candidate-descriptors",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "candidate_descriptors.jsonl",
    )
    parser.add_argument(
        "--runtime-states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
        "runtime_states.jsonl",
    )
    parser.add_argument(
        "--strategy-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v4_final_v1/"
        "strategy_cards_v4_final.jsonl",
    )
    parser.add_argument(
        "--primary-report",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_pm_fit_v1/"
        "training_report.json",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_bge_challenger_v1",
    )
    args = parser.parse_args()

    if not args.model.is_dir():
        raise RuntimeError(f"local BGE model missing: {args.model}")
    labels = _rows(args.labels)
    blueprints = {
        str(row["contrast_slot_id"]): row
        for row in (
            _rows(args.memory_blueprint) + _rows(args.rs_blueprint)
        )
    }
    descriptors = {
        (
            str(row["state_id"]),
            str(row["candidate_descriptor"]["source"]),
        ): dict(row["candidate_descriptor"])
        for row in _rows(args.candidate_descriptors)
    }
    states = {
        str(row["state_id"]): row for row in _rows(args.runtime_states)
    }
    cards = {
        str(row["card_id"]): row for row in _rows(args.strategy_cards)
    }
    primary = __import__("json").loads(
        args.primary_report.read_text(encoding="utf-8")
    )
    if primary["all_primary_gates_pass"]:
        raise RuntimeError(
            "BGE challenger is only authorized after primary failure"
        )

    rs_rows = [row for row in labels if row["component"] == "RS"]
    rs_state_texts = [
        _visible_state_text(states[str(row["state_id"])]) for row in rs_rows
    ]
    rs_candidate_texts = [
        str(
            cards[
                str(
                    blueprints[str(row["contrast_slot_id"])][
                        "current_strategy_candidate"
                    ]["card_id"]
                )
            ]["retrieval_text"]
        )
        for row in rs_rows
    ]
    embeddings = _encode(
        rs_state_texts + rs_candidate_texts,
        args.model,
    )
    rs_similarity = np.sum(
        embeddings[: len(rs_rows)]
        * embeddings[len(rs_rows) :],
        axis=1,
    )
    rs_similarity_by_contrast = {
        str(row["contrast_slot_id"]): float(rs_similarity[index])
        for index, row in enumerate(rs_rows)
    }

    reports: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for component in COMPONENTS:
        component_rows = [
            row for row in labels if row["component"] == component
        ]
        values: list[list[float]] = []
        if component == "RS":
            feature_names = (
                "candidate_state_bge_cosine",
                "execution_profile_minimal",
                "background_MP_on",
                "background_MS_on",
                "background_ME_on",
            )
            for row in component_rows:
                blueprint = blueprints[str(row["contrast_slot_id"])]
                candidate = blueprint["current_strategy_candidate"]
                sources, _ = parse_action_id(
                    str(blueprint["control_action"])
                )
                values.append(
                    [
                        rs_similarity_by_contrast[
                            str(row["contrast_slot_id"])
                        ],
                        float(
                            candidate["execution_profile"] == "minimal"
                        ),
                        float(any(x.value == "MP" for x in sources)),
                        float(any(x.value == "MS" for x in sources)),
                        float(any(x.value == "ME" for x in sources)),
                    ]
                )
        else:
            feature_names = (
                "candidate_state_bge_cosine",
                "source_catalog_bge_cosine",
                "background_other_memory_count",
                "background_strategy_on",
            )
            for row in component_rows:
                descriptor = descriptors[
                    (str(row["state_id"]), component)
                ]
                blueprint = blueprints[str(row["contrast_slot_id"])]
                sources, strategy = parse_action_id(
                    str(blueprint["control_action"])
                )
                values.append(
                    [
                        float(
                            descriptor[
                                "candidate_state_bge_similarity_optional"
                            ]
                        ),
                        float(
                            descriptor[
                                "source_catalog_bge_similarity_optional"
                            ]
                        ),
                        float(len(sources)),
                        float(strategy is StrategyMode.RS),
                    ]
                )
        x = np.asarray(values, dtype=float)
        y = np.asarray(
            [int(row["target_y"]) for row in component_rows],
            dtype=int,
        )
        groups = np.asarray(
            [str(row["user_id"]) for row in component_rows],
            dtype=object,
        )
        split = np.asarray(
            [str(row["split"]) for row in component_rows],
            dtype=object,
        )
        dev = np.flatnonzero(np.isin(split, ["train", "calibration"]))
        internal = np.flatnonzero(split == "internal_test")
        runs = [
            _oof(x[dev], y[dev], groups[dev], seed) for seed in SEEDS
        ]
        probability = np.mean(
            np.stack([run["probability"] for run in runs], axis=0),
            axis=0,
        )
        prior = np.mean(
            np.stack([run["prior"] for run in runs], axis=0),
            axis=0,
        )
        metrics = _metrics(
            y[dev],
            probability,
            prior,
            groups[dev],
        )
        seed_ba = [
            run["metrics"]["group_weighted_balanced_accuracy"]
            for run in runs
        ]
        seed_std = float(np.std(seed_ba, ddof=1))
        primary_metrics = primary["head_reports"][component][
            "grouped_oof"
        ]["metrics"]
        gate_checks = _gate(
            metrics,
            seed_std,
            float(
                primary_metrics["group_weighted_balanced_accuracy"]
            ),
            float(primary_metrics["group_weighted_brier"]),
        )
        models = [
            _fit(x[dev], y[dev], groups[dev], seed) for seed in SEEDS
        ]
        internal_probability = np.mean(
            np.stack(
                [
                    model.predict_proba(x[internal])[:, 1]
                    for model in models
                ],
                axis=0,
            ),
            axis=0,
        )
        internal_prior = np.full(
            len(internal),
            float(
                np.average(
                    y[dev],
                    weights=_weights(groups[dev]),
                )
            ),
        )
        internal_metrics = _metrics(
            y[internal],
            internal_probability,
            internal_prior,
            groups[internal],
        )
        reports[component] = {
            "feature_names": list(feature_names),
            "grouped_oof_metrics": metrics,
            "balanced_accuracy_by_seed": seed_ba,
            "balanced_accuracy_seed_std": seed_std,
            "internal_test_metrics": internal_metrics,
            "promotion_checks": gate_checks,
            "promoted": all(gate_checks.values()),
        }
        for index, row_index in enumerate(dev):
            row = component_rows[int(row_index)]
            prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "component": component,
                    "stage": "development_grouped_oof",
                    "contrast_slot_id": row["contrast_slot_id"],
                    "target_y": int(y[row_index]),
                    "probability_on": float(probability[index]),
                }
            )
        for index, row_index in enumerate(internal):
            row = component_rows[int(row_index)]
            prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "component": component,
                    "stage": "internal_test_frozen_threshold",
                    "contrast_slot_id": row["contrast_slot_id"],
                    "target_y": int(y[row_index]),
                    "probability_on": float(
                        internal_probability[index]
                    ),
                }
            )

    promoted = [
        component
        for component, report in reports.items()
        if report["promoted"]
    ]
    report = {
        "protocol": PROTOCOL,
        "status": (
            "BGE_CHALLENGER_PROMOTED_FOR_SOME_HEADS"
            if promoted
            else "BGE_CHALLENGER_NOT_PROMOTED"
        ),
        "model": "BAAI/bge-small-en-v1.5",
        "model_path": str(args.model),
        "candidate_semantics_only": True,
        "raw_memory_text_not_used_as_logistic_feature": True,
        "development_and_internal_user_groups_disjoint": True,
        "external_results_used": False,
        "C": C_VALUE,
        "threshold": THRESHOLD,
        "head_reports": reports,
        "promoted_heads": promoted,
        "claim_boundary": (
            "This is a registered semantic challenger, not a new label "
            "source and not proof of correct memory retrieval."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "primary_report_sha256": sha256_file(args.primary_report),
            "model_config_sha256": sha256_file(
                args.model / "config.json"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "bge_challenger_report.json", report)
    write_jsonl(args.out_dir / "predictions.jsonl", prediction_rows)
    print(
        {
            "status": report["status"],
            "promoted_heads": promoted,
            "heads": {
                component: {
                    "oof_ba": reports[component][
                        "grouped_oof_metrics"
                    ]["group_weighted_balanced_accuracy"],
                    "oof_brier_gain": reports[component][
                        "grouped_oof_metrics"
                    ][
                        "group_weighted_brier_gain_vs_prevalence_prior"
                    ],
                    "internal_ba": reports[component][
                        "internal_test_metrics"
                    ]["group_weighted_balanced_accuracy"],
                }
                for component in COMPONENTS
            },
        }
    )


if __name__ == "__main__":
    main()
