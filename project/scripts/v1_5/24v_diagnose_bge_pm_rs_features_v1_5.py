#!/usr/bin/env python3
"""Diagnose whether BGE adds OOF signal to the transparent PM_RS head.

This is a post-label development diagnostic, not a promotion search.  Every
PCA/scaling transform is fitted inside its training fold.  The frozen labels
and 0.60 open threshold are unchanged.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from metacom_pm.contracts import RuntimeState, StrategyCard
from metacom_pm.io import iter_jsonl, sha256_file, write_json
from metacom_pm.prompts import common_context


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-bge-pm-rs-feature-diagnostic-v1"
MODEL = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
OPEN_THRESHOLD = 0.60
RANDOM_SEED = 20260730


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _encode(texts: list[str], model_path: Path) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            encoded = tokenizer(
                texts[start : start + 32],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            cls = model(**encoded).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            chunks.append(cls.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _logistic(c: float) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        penalty="l2",
        solver="liblinear",
        class_weight=None,
        random_state=RANDOM_SEED,
        max_iter=2000,
    )


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    action = (probability >= OPEN_THRESHOLD).astype(int)
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "balanced_accuracy_at_0_60": float(
            balanced_accuracy_score(y, action)
        ),
        "predicted_R0": int((action == 0).sum()),
        "predicted_RS": int((action == 1).sum()),
        "open_rate_at_0_60": float(action.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_labels_v1"
        / "pm_rs_training_labels.jsonl",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v1/runtime_states.jsonl",
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_rs_six_card_clean_pair_v1/selected_states.jsonl",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT / "data/strategy/strategy_cards_v1_5_minimal.jsonl",
    )
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transparent_pm_rs_v1"
        / "bge_feature_diagnostic.json",
    )
    args = parser.parse_args()

    labels = [
        row for row in _rows(args.labels) if row.get("target_y") in {0, 1}
    ]
    states = {
        str(row["state_id"]): RuntimeState.model_validate(row)
        for row in _rows(args.states)
    }
    selected = {
        str(row["pair_id"]): row for row in _rows(args.selected)
    }
    cards = {
        card.strategy_id: card
        for card in (
            StrategyCard.model_validate(row) for row in _rows(args.bank)
        )
    }
    feature_names = sorted(
        {
            str(key)
            for row in labels
            for key in row["transparent_pm_features"]
        }
    )
    transparent = np.asarray(
        [
            [
                float(row["transparent_pm_features"].get(name, 0.0))
                for name in feature_names
            ]
            for row in labels
        ],
        dtype=float,
    )
    texts: list[str] = []
    for row in labels:
        state = states[str(row["state_id"])]
        choice = selected[str(row["pair_id"])]
        card = cards[str(choice["selected_strategy_card_id"])]
        texts.append(
            "Emotional-support dialogue state:\n"
            + common_context(state)
            + "\n\nCandidate support guidance:\n"
            + card.guidance_text
        )
    embeddings = _encode(texts, args.model)
    y = np.asarray([int(row["target_y"]) for row in labels], dtype=int)
    folds = StratifiedKFold(
        n_splits=4, shuffle=True, random_state=RANDOM_SEED
    )
    variants: dict[str, tuple[np.ndarray, Pipeline]] = {
        "transparent_frozen_C0.3": (
            transparent,
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("logistic", _logistic(0.3)),
                ]
            ),
        ),
        "transparent_strong_shrinkage_C0.03_diagnostic": (
            transparent,
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("logistic", _logistic(0.03)),
                ]
            ),
        ),
        "bge_PCA4_C0.03": (
            embeddings,
            Pipeline(
                [
                    ("pca", PCA(n_components=4, random_state=RANDOM_SEED)),
                    ("scale", StandardScaler()),
                    ("logistic", _logistic(0.03)),
                ]
            ),
        ),
    }
    combined = np.concatenate([transparent, embeddings], axis=1)
    n_transparent = transparent.shape[1]
    variants["transparent_plus_bge_PCA4_C0.03"] = (
        combined,
        Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            (
                                "transparent",
                                StandardScaler(),
                                slice(0, n_transparent),
                            ),
                            (
                                "bge",
                                Pipeline(
                                    [
                                        (
                                            "pca",
                                            PCA(
                                                n_components=4,
                                                random_state=RANDOM_SEED,
                                            ),
                                        ),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                slice(n_transparent, combined.shape[1]),
                            ),
                        ]
                    ),
                ),
                ("logistic", _logistic(0.03)),
            ]
        ),
    )
    results: dict[str, Any] = {}
    for name, (x, estimator) in variants.items():
        probability = cross_val_predict(
            estimator, x, y, cv=folds, method="predict_proba"
        )[:, 1]
        results[name] = _metrics(y, probability)
    prevalence = float(y.mean())
    results["prevalence_only"] = _metrics(
        y, np.full(len(y), prevalence, dtype=float)
    )
    base = results["transparent_frozen_C0.3"]
    bge = results["bge_PCA4_C0.03"]
    combined_result = results["transparent_plus_bge_PCA4_C0.03"]
    bge_promotable = any(
        candidate["log_loss"] < results["prevalence_only"]["log_loss"]
        and candidate["log_loss"] < base["log_loss"]
        and candidate["balanced_accuracy_at_0_60"] > 0.50
        and candidate["predicted_R0"] > 0
        and candidate["predicted_RS"] > 0
        for candidate in (bge, combined_result)
    )
    report = {
        "protocol": PROTOCOL,
        "status": "COMPLETE_POST_LABEL_DIAGNOSTIC",
        "rows": len(labels),
        "class_counts": dict(sorted(Counter(y.tolist()).items())),
        "model_path": str(args.model),
        "embedding_dimension": int(embeddings.shape[1]),
        "visible_input": (
            "current-session dialogue plus the already retrieved candidate "
            "card guidance; no generated response or outcome text"
        ),
        "cross_validation": (
            "same fixed four stratified folds; PCA and scaling fit inside "
            "each training fold"
        ),
        "results": results,
        "bge_feature_promotion_gate_passed": bge_promotable,
        "decision": (
            "BAAI may enter the V1.5 PM_RS head"
            if bge_promotable
            else "Do not add BAAI to the V1.5 PM_RS head"
        ),
        "caveat": (
            "This post-label diagnostic can reject BGE but cannot by itself "
            "establish a publishable model advantage."
        ),
        "lineage": {
            "labels_sha256": sha256_file(args.labels),
            "states_sha256": sha256_file(args.states),
            "selected_sha256": sha256_file(args.selected),
            "bank_sha256": sha256_file(args.bank),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
