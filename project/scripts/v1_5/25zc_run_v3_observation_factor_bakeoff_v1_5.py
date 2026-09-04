#!/usr/bin/env python3
"""Run the frozen V3 Observation factor-head bakeoff without external outcomes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import canonical_json, iter_jsonl, sha256_file, sha256_text, write_json, write_jsonl
from metacom_pm.v1_5_observation_contract import transparent_semantic_observation


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-factor-bakeoff-execution-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
FACTORS = {
    "owner_time_entity_valid": "owner_time_valid",
    "goal_function_fit": "goal_function_fit",
    "boundary_burden_compatible": "boundary_burden_fit",
    "specific_increment": "specific_nonredundant_increment",
}
CANDIDATES = ("transparent_v1", "lexical_logistic_v1", "bge_small_hybrid_v1")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _metric(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    pred = (probabilities >= 0.5).astype(int)
    tp = int(np.sum((labels == 1) & (pred == 1)))
    tn = int(np.sum((labels == 0) & (pred == 0)))
    fp = int(np.sum((labels == 0) & (pred == 1)))
    fn = int(np.sum((labels == 1) & (pred == 0)))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "n": int(labels.size),
        "positive": int(labels.sum()),
        "negative": int(labels.size - labels.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def _manual_one_hot(values: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    index = {value: idx for idx, value in enumerate(categories)}
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row, value in enumerate(values):
        if value in index:
            result[row, index[value]] = 1.0
    return result


def _fit_logistic_dense(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if len(np.unique(train_y)) < 2:
        return np.full(test_x.shape[0], float(np.mean(train_y)), dtype=np.float64)
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=0.3,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    ).fit(scaler.transform(train_x), train_y)
    return model.predict_proba(scaler.transform(test_x))[:, 1]


def _fit_lexical(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    relation_texts: Sequence[str],
    schema: np.ndarray,
    transparent: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if len(np.unique(labels[train])) < 2:
        return np.full(test.size, float(np.mean(labels[train])), dtype=np.float64)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=256,
        sublinear_tf=True,
    )
    train_text = vectorizer.fit_transform([relation_texts[idx] for idx in train])
    test_text = vectorizer.transform([relation_texts[idx] for idx in test])
    train_aux = sparse.csr_matrix(np.hstack([schema[train], transparent[train]]))
    test_aux = sparse.csr_matrix(np.hstack([schema[test], transparent[test]]))
    model = LogisticRegression(
        C=0.3,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    ).fit(sparse.hstack([train_text, train_aux]), labels[train])
    return model.predict_proba(sparse.hstack([test_text, test_aux]))[:, 1]


def _fit_bge(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    relation_embeddings: np.ndarray,
    schema: np.ndarray,
    transparent: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if len(np.unique(labels[train])) < 2:
        return np.full(test.size, float(np.mean(labels[train])), dtype=np.float64)
    pca = PCA(n_components=min(8, train.size - 1), random_state=seed).fit(
        relation_embeddings[train]
    )
    train_x = np.hstack([pca.transform(relation_embeddings[train]), schema[train], transparent[train]])
    test_x = np.hstack([pca.transform(relation_embeddings[test]), schema[test], transparent[test]])
    return _fit_logistic_dense(train_x, labels[train], test_x, seed=seed)


def _grouped_oof(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    seeds: Sequence[int],
    predictor: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> tuple[np.ndarray, list[float]]:
    seed_probabilities: list[np.ndarray] = []
    seed_balanced: list[float] = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        probabilities = np.full(labels.size, np.nan, dtype=np.float64)
        for train, test in splitter.split(np.zeros(labels.size), labels, groups):
            probabilities[test] = predictor(train, test, seed)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("grouped OOF did not cover every row")
        seed_probabilities.append(probabilities)
        seed_balanced.append(float(balanced_accuracy_score(labels, probabilities >= 0.5)))
    return np.mean(seed_probabilities, axis=0), seed_balanced


def _family_oof(
    *,
    labels: np.ndarray,
    families: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> np.ndarray:
    probabilities = np.full(labels.size, np.nan, dtype=np.float64)
    for fold, (train, test) in enumerate(
        LeaveOneGroupOut().split(np.zeros(labels.size), labels, families)
    ):
        probabilities[test] = predictor(train, test, 1000 + fold)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("leave-family-out did not cover every row")
    return probabilities


def _counterfactual_direction(
    records: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        grouped[(str(row["component"]), str(row["counterfactual_group_id"]))].append(idx)
    comparisons: list[dict[str, Any]] = []
    for (component, group_id), indices in grouped.items():
        positives = [idx for idx in indices if labels[idx] == 1]
        negatives = [idx for idx in indices if labels[idx] == 0]
        for positive in positives:
            for negative in negatives:
                comparisons.append(
                    {
                        "component": component,
                        "counterfactual_group_id": group_id,
                        "positive_family": records[positive]["logic_family"],
                        "negative_family": records[negative]["logic_family"],
                        "correct": bool(probabilities[positive] > probabilities[negative]),
                    }
                )
    correct = sum(row["correct"] for row in comparisons)
    return {
        "n": len(comparisons),
        "correct": correct,
        "rate": correct / len(comparisons) if comparisons else None,
        "failed_family_pairs": dict(
            Counter(
                f"{row['positive_family']} -> {row['negative_family']}"
                for row in comparisons
                if not row["correct"]
            )
        ),
    }


def _encode_bge(state_texts: Sequence[str], candidate_texts: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).eval()
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))

    def encode(texts: Sequence[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), 32):
            batch = tokenizer(
                list(texts[start : start + 32]),
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            with torch.no_grad():
                values = model(**batch).last_hidden_state[:, 0]
                values = torch.nn.functional.normalize(values, p=2, dim=1)
            chunks.append(values.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    state = encode(state_texts)
    candidate = encode(candidate_texts)
    relation = np.hstack([state, candidate, np.abs(state - candidate), state * candidate])
    return relation, {
        "model": model_name,
        "hidden_size": int(state.shape[1]),
        "relation_size": int(relation.shape[1]),
        "pooling": "normalized_cls",
        "device": "cpu_transformers_fallback",
        "local_files_only": True,
    }


def _nuisance_probe(records: Sequence[Mapping[str, Any]], labels: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    prefix_values = [str(row["prefix_bucket"]) for row in records]
    topic_values = [str(row["topic_family"]) for row in records]
    scale_values = [str(row["history_scale"]) for row in records]
    component_values = [str(row["component"]) for row in records]
    blocks = []
    for values in (prefix_values, topic_values, scale_values, component_values):
        blocks.append(_manual_one_hot(values, sorted(set(values))))
    lengths = np.asarray(
        [
            [
                min(5, len(str(row["state_text"]).split()) // 20),
                min(5, len(str(row["candidate_text"]).split()) // 20),
            ]
            for row in records
        ],
        dtype=np.float32,
    )
    features = np.hstack([*blocks, lengths])

    def predictor(train: np.ndarray, test: np.ndarray, seed: int) -> np.ndarray:
        return _fit_logistic_dense(features[train], labels[train], features[test], seed=seed)

    probabilities, seeds = _grouped_oof(
        labels=labels,
        groups=groups,
        seeds=(17, 29, 43, 71, 113),
        predictor=predictor,
    )
    result = _metric(labels, probabilities)
    result["five_seed_balanced_accuracy"] = seeds
    result["five_seed_balanced_accuracy_sd"] = float(np.std(seeds))
    return result


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 P2 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "data/pm_v1_5_contracts/v3_observation_factor_bakeoff_v1.json",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_h_eligibility_reference_v3/eligibility_reference.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v3/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "data/pm_v1_5_v3_effect_blueprint_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_factor_bakeoff_v1",
    )
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if sha256_file(args.reference) != contract["reference_sha256"]:
        raise RuntimeError("frozen Observation reference hash changed")
    reference = sorted(_rows(args.reference), key=lambda row: str(row["state_id"]))
    candidate_map = {str(row["state_id"]): row for row in _rows(args.candidates)}
    blueprint = {str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)}
    if len(reference) != 128:
        raise RuntimeError("bakeoff requires exactly 128 frozen Eligibility rows")

    component_categories = list(COMPONENTS)
    subtype_categories = sorted({str(row["adjudicated_subtype"]) for row in reference})
    records: list[dict[str, Any]] = []
    transparent_matrix: list[list[float]] = []
    state_texts: list[str] = []
    candidate_texts: list[str] = []
    for label in reference:
        state_id = str(label["state_id"])
        candidate_row = candidate_map[state_id]
        exact = candidate_row["exact_rank1_candidate"]
        if exact["candidate_text_sha256"] != label["candidate_text_sha256"]:
            raise RuntimeError(f"candidate/reference hash mismatch: {state_id}")
        dialogue = " ".join(
            f"{turn.get('role', turn.get('speaker', ''))}: {turn.get('content', '')}"
            for turn in candidate_row["visible_dialogue"]
        )
        state_text = f"{dialogue} current user: {candidate_row['current_user_text']}"
        candidate_text = str(exact["candidate_text"])
        observation = transparent_semantic_observation(
            state_id=state_id,
            component=str(label["component"]),
            current_user_text=str(candidate_row["current_user_text"]),
            visible_dialogue=candidate_row["visible_dialogue"],
            candidate_present=True,
            candidate_text=candidate_text,
            candidate_subtype=str(exact["compiler_subtype_hint"]),
        )
        current = str(candidate_row["current_user_text"])
        prefix = re.split(r"\bi am dealing with\b", current, flags=re.IGNORECASE)[0]
        construction = blueprint[state_id]
        records.append(
            {
                "state_id": state_id,
                "component": str(label["component"]),
                "subtype": str(label["adjudicated_subtype"]),
                "counterfactual_group_id": str(construction["counterfactual_group_id"]),
                "logic_family": str(construction["logic_family"]),
                "logic_pair": str(construction["logic_pair"]),
                "topic_family": str(construction["topic_family"]),
                "history_scale": str(construction["history_scale"]),
                "prefix_bucket": sha256_text(prefix.lower())[:12],
                "state_text": state_text,
                "candidate_text": candidate_text,
            }
        )
        transparent_matrix.append(
            [float(observation.factor_scores[factor]) for factor in FACTORS]
        )
        state_texts.append(state_text)
        candidate_texts.append(candidate_text)

    transparent = np.asarray(transparent_matrix, dtype=np.float32)
    component_one_hot = _manual_one_hot(
        [row["component"] for row in records], component_categories
    )
    subtype_one_hot = _manual_one_hot(
        [row["subtype"] for row in records], subtype_categories
    )
    schema = np.hstack([component_one_hot, subtype_one_hot]).astype(np.float32)
    relation_texts = [
        f"STATE: {state}\nCANDIDATE: {candidate}"
        for state, candidate in zip(state_texts, candidate_texts, strict=True)
    ]
    relation_embeddings, embedding_meta = _encode_bge(state_texts, candidate_texts)
    groups = np.asarray([row["counterfactual_group_id"] for row in records])
    families = np.asarray([row["logic_pair"] for row in records])
    seeds = tuple(int(value) for value in contract["validation"]["seeds"])
    gate = contract["gates"]

    labels_by_factor = {
        factor: np.asarray(
            [int(label[gold_field] == "yes") for label in reference], dtype=np.int64
        )
        for factor, gold_field in FACTORS.items()
    }
    nuisance = {
        factor: _nuisance_probe(records, labels, groups)
        for factor, labels in labels_by_factor.items()
    }
    nuisance_pass = all(
        value["balanced_accuracy"] < gate["nuisance_probe_balanced_accuracy_max_exclusive"]
        for value in nuisance.values()
    )

    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATES:
        factor_results: dict[str, Any] = {}
        candidate_pass = nuisance_pass
        for factor_index, (factor, labels) in enumerate(labels_by_factor.items()):
            if candidate_id == "transparent_v1":
                probabilities = transparent[:, factor_index].astype(np.float64)
                seed_scores = [float(balanced_accuracy_score(labels, probabilities >= 0.5))] * len(seeds)
                family_probabilities = probabilities.copy()
            else:
                if candidate_id == "lexical_logistic_v1":
                    predictor = lambda train, test, seed: _fit_lexical(
                        train,
                        test,
                        labels,
                        relation_texts,
                        schema,
                        transparent,
                        seed=seed,
                    )
                else:
                    predictor = lambda train, test, seed: _fit_bge(
                        train,
                        test,
                        labels,
                        relation_embeddings,
                        schema,
                        transparent,
                        seed=seed,
                    )
                probabilities, seed_scores = _grouped_oof(
                    labels=labels,
                    groups=groups,
                    seeds=seeds,
                    predictor=predictor,
                )
                family_probabilities = _family_oof(
                    labels=labels,
                    families=families,
                    predictor=predictor,
                )
            metric = _metric(labels, probabilities)
            family_metric = _metric(labels, family_probabilities)
            direction = _counterfactual_direction(records, labels, probabilities)
            seed_sd = float(np.std(seed_scores))
            passed = bool(
                metric["balanced_accuracy"] >= gate["each_factor_balanced_accuracy_min"]
                and metric["recall"] >= gate["each_factor_recall_min"]
                and metric["specificity"] >= gate["each_factor_specificity_min"]
                and direction["rate"] >= gate["each_factor_minimal_counterfactual_direction_min"]
                and family_metric["balanced_accuracy"]
                >= gate["each_factor_leave_semantic_family_out_balanced_accuracy_min"]
                and seed_sd <= gate["five_seed_balanced_accuracy_sd_max"]
            )
            metric.update(
                {
                    "minimal_counterfactual_direction": direction,
                    "leave_semantic_family_out": family_metric,
                    "five_seed_balanced_accuracy": seed_scores,
                    "five_seed_balanced_accuracy_sd": seed_sd,
                    "status": "PASS" if passed else "FAIL",
                }
            )
            factor_results[factor] = metric
            candidate_pass = candidate_pass and passed
            for idx, row in enumerate(records):
                prediction_rows.append(
                    {
                        "protocol": PROTOCOL,
                        "candidate": candidate_id,
                        "state_id": row["state_id"],
                        "component": row["component"],
                        "factor": factor,
                        "gold": int(labels[idx]),
                        "grouped_oof_probability": float(probabilities[idx]),
                        "leave_family_out_probability": float(family_probabilities[idx]),
                    }
                )
        results[candidate_id] = {
            "status": "PASS_ALL_GATES" if candidate_pass else "FAIL_ONE_OR_MORE_GATES",
            "factors": factor_results,
            "nuisance_probe_shared_data_gate_pass": nuisance_pass,
        }

    selected = next(
        (
            candidate_id
            for candidate_id in contract["selection"]["order_if_all_gates_pass"]
            if results[candidate_id]["status"] == "PASS_ALL_GATES"
        ),
        None,
    )
    report = {
        "protocol": PROTOCOL,
        "status": "PASS_OBSERVATION_CANDIDATE_SELECTED" if selected else "FAIL_NO_OBSERVATION_CANDIDATE_QUALIFIED",
        "selected_candidate": selected,
        "role": contract["role"],
        "contract_sha256": sha256_file(args.contract),
        "reference_sha256": sha256_file(args.reference),
        "candidate_rows_sha256": sha256_file(args.candidates),
        "rows": len(records),
        "counterfactual_groups": len(set(groups)),
        "logic_pairs": len(set(families)),
        "data_quality": {
            "unique_states": len({row["state_id"] for row in records}),
            "per_component": dict(Counter(row["component"] for row in records)),
            "forbidden_metadata_used_as_feature": False,
            "labels_used_only_as_factor_supervision": True,
            "nuisance_probe": nuisance,
            "nuisance_probe_pass": nuisance_pass,
        },
        "embedding": embedding_meta,
        "results": results,
        "selection_order": contract["selection"]["order_if_all_gates_pass"],
        "final_model_fit": False,
        "step1_training_authorized": False,
        "step2_qualification_authorized": selected is not None,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "oof_predictions.jsonl", prediction_rows)
    report["oof_predictions_sha256"] = sha256_file(args.out_dir / "oof_predictions.jsonl")
    write_json(args.out_dir / "qualification_report.json", report)
    print(
        {
            "status": report["status"],
            "selected_candidate": selected,
            "candidate_statuses": {
                key: value["status"] for key, value in results.items()
            },
            "out": str(args.out_dir / "qualification_report.json"),
        }
    )


if __name__ == "__main__":
    main()
