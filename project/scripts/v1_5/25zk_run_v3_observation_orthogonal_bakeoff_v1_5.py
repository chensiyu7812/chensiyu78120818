#!/usr/bin/env python3
"""Run the frozen V3 Observation bakeoff on 64 FIT, then one-shot 32 confirmation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import runpy
import sys
from typing import Any, Callable

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_observation_contract import transparent_semantic_observation
from metacom_pm.v1_5_v3_observation_realization import (
    NEUTRAL_FILLER_ELEVEN,
    NEUTRAL_FILLER_REMAINDERS,
    NEUTRAL_FILLER_TEN,
    PREFIX_TEXT,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-orthogonal-bakeoff-execution-v2"
COMPONENTS = ("MP", "MS", "ME", "RS")
FACTORS = (
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "specific_increment",
)
CANDIDATES = ("transparent_v1", "lexical_logistic_v1", "bge_small_hybrid_v1")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _strip_padding(text: str) -> str:
    neutral = [
        *PREFIX_TEXT.values(),
        NEUTRAL_FILLER_TEN,
        NEUTRAL_FILLER_ELEVEN,
        *NEUTRAL_FILLER_REMAINDERS.values(),
        "Earlier today, I chose a fresh wording for this present request.",
    ]
    result = str(text)
    for phrase in sorted(neutral, key=len, reverse=True):
        result = result.replace(phrase, " ")
    return re.sub(r"\s+", " ", result).strip()


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


def _one_hot(values: list[str], categories: list[str]) -> np.ndarray:
    index = {value: idx for idx, value in enumerate(categories)}
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row, value in enumerate(values):
        if value in index:
            result[row, index[value]] = 1.0
    return result


def _grouped_oof(
    labels: np.ndarray,
    groups: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, list[float]]:
    legacy = runpy.run_path(str(ROOT / "scripts/v1_5/25zc_run_v3_observation_factor_bakeoff_v1_5.py"))
    return legacy["_grouped_oof"](
        labels=labels, groups=groups, seeds=seeds, predictor=predictor
    )


def _family_oof(
    labels: np.ndarray,
    families: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> np.ndarray:
    legacy = runpy.run_path(str(ROOT / "scripts/v1_5/25zc_run_v3_observation_factor_bakeoff_v1_5.py"))
    return legacy["_family_oof"](labels=labels, families=families, predictor=predictor)


def _fit_dense(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int
) -> np.ndarray:
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
    texts: list[str],
    schema: np.ndarray,
    transparent: np.ndarray,
    seed: int,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=256,
        sublinear_tf=True,
    )
    train_text = vectorizer.fit_transform([texts[index] for index in train])
    test_text = vectorizer.transform([texts[index] for index in test])
    train_x = sparse.hstack(
        [train_text, sparse.csr_matrix(np.hstack([schema[train], transparent[train]]))]
    )
    test_x = sparse.hstack(
        [test_text, sparse.csr_matrix(np.hstack([schema[test], transparent[test]]))]
    )
    model = LogisticRegression(
        C=0.3,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=seed,
    ).fit(train_x, labels[train])
    return model.predict_proba(test_x)[:, 1]


def _fit_bge(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    relation: np.ndarray,
    schema: np.ndarray,
    transparent: np.ndarray,
    seed: int,
) -> np.ndarray:
    pca = PCA(n_components=min(8, len(train) - 1), random_state=seed).fit(relation[train])
    train_x = np.hstack([pca.transform(relation[train]), schema[train], transparent[train]])
    test_x = np.hstack([pca.transform(relation[test]), schema[test], transparent[test]])
    return _fit_dense(train_x, labels[train], test_x, seed)


def _direction(
    records: list[dict[str, Any]], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[(row["component"], row["counterfactual_group_id"])].append(index)
    comparisons = []
    for (component, group_id), indices in grouped.items():
        positive = [index for index in indices if labels[index] == 1]
        negative = [index for index in indices if labels[index] == 0]
        for pos in positive:
            for neg in negative:
                comparisons.append(
                    {
                        "component": component,
                        "group": group_id,
                        "correct": bool(probabilities[pos] > probabilities[neg]),
                    }
                )
    return {
        "n": len(comparisons),
        "correct": sum(row["correct"] for row in comparisons),
        "rate": (
            sum(row["correct"] for row in comparisons) / len(comparisons)
            if comparisons
            else None
        ),
    }


def main() -> None:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError(f"formal V3 requires {FORMAL_PYTHON}; got {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_human_reference_v2/observation_reference.jsonl",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_orthogonal_exact_rank1_v2/candidate_rows_private.jsonl",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_v3_observation_orthogonal_v1/private/construction_blueprint.jsonl",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_contracts/v3_observation_orthogonal_bakeoff_v2.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_orthogonal_bakeoff_v2",
    )
    args = parser.parse_args()

    reference = _rows(args.reference)
    candidates = {str(row["state_id"]): row for row in _rows(args.candidates)}
    blueprint = {str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)}
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    frozen_inputs = {
        "reference_sha256": sha256_file(args.reference),
        "candidate_rows_sha256": sha256_file(args.candidates),
        "blueprint_sha256": sha256_file(args.blueprint),
    }
    for key, observed in frozen_inputs.items():
        expected = str(contract.get(key, ""))
        if observed != expected:
            raise RuntimeError(
                f"frozen input hash mismatch for {key}: expected {expected}; got {observed}"
            )
    if len(reference) != 96 or len(candidates) != 96 or len(blueprint) != 96:
        raise RuntimeError("orthogonal bakeoff requires 96 aligned rows")

    component_categories = list(COMPONENTS)
    subtype_categories = sorted({str(row["adjudicated_subtype"]) for row in reference})
    records: list[dict[str, Any]] = []
    transparent_rows: list[list[float]] = []
    state_texts: list[str] = []
    candidate_texts: list[str] = []
    for label in reference:
        state_id = str(label["state_id"])
        candidate_row = candidates[state_id]
        exact = candidate_row["exact_rank1_candidate"]
        if exact["candidate_text_sha256"] != label["candidate_text_sha256"]:
            raise RuntimeError(f"candidate/reference hash mismatch: {state_id}")
        construction = blueprint[state_id]
        current_raw = str(candidate_row["current_user_text"])
        candidate_raw = str(exact["candidate_text"])
        current = _strip_padding(current_raw)
        candidate = _strip_padding(candidate_raw)
        observation = transparent_semantic_observation(
            state_id=state_id,
            component=str(label["component"]),
            current_user_text=current,
            visible_dialogue=[],
            candidate_present=True,
            candidate_text=candidate,
            candidate_subtype=str(exact["compiler_subtype_hint"]),
        )
        records.append(
            {
                "state_id": state_id,
                "component": str(label["component"]),
                "subtype": str(label["adjudicated_subtype"]),
                "track": str(label["track"]),
                "counterfactual_group_id": str(label["counterfactual_group_id"]),
                "topic_family": str(construction["topic_family"]),
                "history_scale": str(construction["history_scale"]),
                "prefix_bucket": str(construction["prefix_family"]),
                "semantic_families": construction["private_semantic_surface_family"],
                "state_text": current,
                "candidate_text": candidate,
                "human": label,
            }
        )
        transparent_rows.append(
            [float(observation.factor_scores[factor]) for factor in FACTORS]
        )
        state_texts.append(current)
        candidate_texts.append(candidate)

    transparent = np.asarray(transparent_rows, dtype=np.float32)
    schema = np.hstack(
        [
            _one_hot([row["component"] for row in records], component_categories),
            _one_hot([row["subtype"] for row in records], subtype_categories),
        ]
    )
    relation_texts = [
        f"STATE: {state}\nCANDIDATE: {candidate}"
        for state, candidate in zip(state_texts, candidate_texts, strict=True)
    ]
    legacy = runpy.run_path(str(ROOT / "scripts/v1_5/25zc_run_v3_observation_factor_bakeoff_v1_5.py"))
    relation_embeddings, embedding_meta = legacy["_encode_bge"](state_texts, candidate_texts)
    gate = contract["gates"]
    seeds = tuple(int(value) for value in contract["validation"]["seeds"])

    fit_indices = np.asarray([i for i, row in enumerate(records) if row["track"] == "FACTOR_FIT"])
    confirmation_indices = np.asarray(
        [i for i, row in enumerate(records) if row["track"] == "ELIGIBILITY_CONFIRMATION"]
    )
    results: dict[str, Any] = {candidate: {"factors": {}} for candidate in CANDIDATES}
    oof_rows: list[dict[str, Any]] = []
    nuisance = {}

    for factor_index, factor in enumerate(FACTORS):
        applicable_fit = np.asarray(
            [
                index
                for index in fit_indices
                if not (factor == "owner_time_entity_valid" and records[index]["component"] == "RS")
            ]
        )
        labels = np.asarray(
            [int(records[index]["human"][factor] == "yes") for index in applicable_fit],
            dtype=np.int64,
        )
        groups = np.asarray([records[index]["counterfactual_group_id"] for index in applicable_fit])
        families = np.asarray(
            [records[index]["semantic_families"][factor] for index in applicable_fit]
        )
        local_records = [records[index] for index in applicable_fit]
        local_transparent = transparent[applicable_fit]
        local_schema = schema[applicable_fit]
        local_relation = relation_embeddings[applicable_fit]
        local_texts = [relation_texts[index] for index in applicable_fit]

        nuisance_features = np.hstack(
            [
                _one_hot([row["component"] for row in local_records], component_categories),
                _one_hot(
                    [row["topic_family"] for row in local_records],
                    sorted({row["topic_family"] for row in local_records}),
                ),
                _one_hot(
                    [row["history_scale"] for row in local_records],
                    sorted({row["history_scale"] for row in local_records}),
                ),
                _one_hot(
                    [row["prefix_bucket"] for row in local_records],
                    sorted({row["prefix_bucket"] for row in local_records}),
                ),
                np.asarray(
                    [
                        [len(row["state_text"].split()), len(row["candidate_text"].split())]
                        for row in local_records
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        nuisance_predictor = lambda train, test, seed: _fit_dense(
            nuisance_features[train], labels[train], nuisance_features[test], seed
        )
        nuisance_prob, nuisance_seed = _grouped_oof(labels, groups, nuisance_predictor, seeds)
        nuisance[factor] = _metric(labels, nuisance_prob)
        nuisance[factor]["five_seed_balanced_accuracy"] = nuisance_seed
        nuisance[factor]["pass"] = nuisance[factor]["balanced_accuracy"] < gate[
            "nuisance_probe_balanced_accuracy_max_exclusive"
        ]

        for candidate_id in CANDIDATES:
            if candidate_id == "transparent_v1":
                probabilities = local_transparent[:, factor_index].astype(np.float64)
                family_probabilities = probabilities.copy()
                seed_scores = [
                    float(balanced_accuracy_score(labels, probabilities >= 0.5))
                ] * len(seeds)
            elif candidate_id == "lexical_logistic_v1":
                predictor = lambda train, test, seed: _fit_lexical(
                    train,
                    test,
                    labels,
                    local_texts,
                    local_schema,
                    local_transparent,
                    seed,
                )
                probabilities, seed_scores = _grouped_oof(labels, groups, predictor, seeds)
                family_probabilities = _family_oof(labels, families, predictor)
            else:
                predictor = lambda train, test, seed: _fit_bge(
                    train,
                    test,
                    labels,
                    local_relation,
                    local_schema,
                    local_transparent,
                    seed,
                )
                probabilities, seed_scores = _grouped_oof(labels, groups, predictor, seeds)
                family_probabilities = _family_oof(labels, families, predictor)
            metric = _metric(labels, probabilities)
            family_metric = _metric(labels, family_probabilities)
            direction = _direction(local_records, labels, probabilities)
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
            results[candidate_id]["factors"][factor] = metric
            for local_index, record in enumerate(local_records):
                oof_rows.append(
                    {
                        "protocol": PROTOCOL,
                        "candidate": candidate_id,
                        "state_id": record["state_id"],
                        "component": record["component"],
                        "factor": factor,
                        "gold": int(labels[local_index]),
                        "grouped_oof_probability": float(probabilities[local_index]),
                        "leave_family_out_probability": float(family_probabilities[local_index]),
                    }
                )

    nuisance_pass = all(row["pass"] for row in nuisance.values())
    for candidate_id in CANDIDATES:
        passed = nuisance_pass and all(
            row["status"] == "PASS" for row in results[candidate_id]["factors"].values()
        )
        results[candidate_id]["status"] = (
            "PASS_ALL_FIT_GATES" if passed else "FAIL_ONE_OR_MORE_FIT_GATES"
        )
    selected = next(
        (
            candidate
            for candidate in contract["selection"]["order_if_all_gates_pass"]
            if results[candidate]["status"] == "PASS_ALL_FIT_GATES"
        ),
        None,
    )

    confirmation = None
    confirmation_rows: list[dict[str, Any]] = []
    if selected is not None:
        per_factor_probabilities: dict[str, dict[int, float]] = defaultdict(dict)
        per_factor_metrics = {}
        for factor_index, factor in enumerate(FACTORS):
            train = np.asarray(
                [
                    index
                    for index in fit_indices
                    if not (factor == "owner_time_entity_valid" and records[index]["component"] == "RS")
                ]
            )
            test = np.asarray(
                [
                    index
                    for index in confirmation_indices
                    if not (factor == "owner_time_entity_valid" and records[index]["component"] == "RS")
                ]
            )
            full_labels = np.asarray(
                [
                    0
                    if row["human"][factor] == "structural_yes_not_rated"
                    else int(row["human"][factor] == "yes")
                    for row in records
                ],
                dtype=np.int64,
            )
            if selected == "transparent_v1":
                probabilities = transparent[test, factor_index].astype(np.float64)
            elif selected == "lexical_logistic_v1":
                probabilities = _fit_lexical(
                    train,
                    test,
                    full_labels,
                    relation_texts,
                    schema,
                    transparent,
                    20260803,
                )
            else:
                probabilities = _fit_bge(
                    train,
                    test,
                    full_labels,
                    relation_embeddings,
                    schema,
                    transparent,
                    20260803,
                )
            labels = full_labels[test]
            per_factor_metrics[factor] = _metric(labels, probabilities)
            for index, probability in zip(test, probabilities, strict=True):
                per_factor_probabilities[factor][int(index)] = float(probability)

        composite_labels = []
        composite_probabilities = []
        per_component = defaultdict(lambda: {"gold": [], "prob": []})
        for index in confirmation_indices:
            row = records[index]
            probabilities = [
                per_factor_probabilities[factor][int(index)]
                for factor in FACTORS
                if not (factor == "owner_time_entity_valid" and row["component"] == "RS")
            ]
            composite_probability = float(min(probabilities))
            gold = int(row["human"]["derived_eligibility"] == "eligible")
            composite_labels.append(gold)
            composite_probabilities.append(composite_probability)
            per_component[row["component"]]["gold"].append(gold)
            per_component[row["component"]]["prob"].append(composite_probability)
            confirmation_rows.append(
                {
                    "protocol": PROTOCOL,
                    "candidate": selected,
                    "state_id": row["state_id"],
                    "component": row["component"],
                    "gold_eligibility": gold,
                    "composed_probability_min_gate": composite_probability,
                    "predicted_eligibility": int(composite_probability >= 0.5),
                }
            )
        confirmation = {
            "status": "ONE_SHOT_CONSUMED",
            "selected_candidate": selected,
            "per_factor": per_factor_metrics,
            "composite_global": _metric(
                np.asarray(composite_labels), np.asarray(composite_probabilities)
            ),
            "composite_per_component": {
                component: _metric(np.asarray(values["gold"]), np.asarray(values["prob"]))
                for component, values in per_component.items()
            },
            "used_for_model_or_threshold_selection": False,
        }

    report = {
        "protocol": PROTOCOL,
        "status": (
            "PASS_FIT_CANDIDATE_SELECTED_CONFIRMATION_CONSUMED"
            if selected is not None
            else "FAIL_NO_FIT_CANDIDATE_QUALIFIED_CONFIRMATION_NOT_CONSUMED"
        ),
        "selected_candidate": selected,
        "fit_rows": int(len(fit_indices)),
        "confirmation_rows": int(len(confirmation_indices)),
        "fit_data_quality": {
            "per_component": dict(Counter(records[index]["component"] for index in fit_indices)),
            "nuisance_probe": nuisance,
            "nuisance_probe_pass": nuisance_pass,
            "padding_removed_before_model_input": True,
            "construction_intent_used_as_feature_or_gold": False,
        },
        "fit_results": results,
        "one_shot_confirmation": confirmation,
        "embedding": embedding_meta,
        "step1_training_authorized": False,
        "step2_qualification_authorized": selected is not None,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        **frozen_inputs,
        "contract_sha256": sha256_file(args.contract),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "fit_oof_predictions.jsonl", oof_rows)
    write_jsonl(args.out_dir / "confirmation_predictions.jsonl", confirmation_rows)
    report["fit_oof_predictions_sha256"] = sha256_file(
        args.out_dir / "fit_oof_predictions.jsonl"
    )
    report["confirmation_predictions_sha256"] = sha256_file(
        args.out_dir / "confirmation_predictions.jsonl"
    )
    write_json(args.out_dir / "qualification_report.json", report)
    print(
        {
            "status": report["status"],
            "selected_candidate": selected,
            "candidate_statuses": {
                candidate: results[candidate]["status"] for candidate in CANDIDATES
            },
            "confirmation_composite": (
                confirmation["composite_global"] if confirmation else None
            ),
            "out": str(args.out_dir / "qualification_report.json"),
        }
    )


if __name__ == "__main__":
    main()
