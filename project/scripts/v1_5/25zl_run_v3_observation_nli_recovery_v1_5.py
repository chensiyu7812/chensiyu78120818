#!/usr/bin/env python3
"""Run the single preauthorized NLI recovery candidate, then one-shot confirmation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import runpy
import sys
from typing import Any

import numpy as np

from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.v1_5_v3_observation_realization import (
    NEUTRAL_FILLER_ELEVEN,
    NEUTRAL_FILLER_REMAINDERS,
    NEUTRAL_FILLER_TEN,
    PREFIX_TEXT,
)


ROOT = Path(__file__).resolve().parents[2]
FORMAL_PYTHON = ROOT.parent / ".venv-pm-v1-5/bin/python"
PROTOCOL = "pm-v1.5-v3-observation-nli-recovery-execution-v1"
COMPONENTS = ("MP", "MS", "ME", "RS")
FACTORS = (
    "owner_time_entity_valid",
    "goal_function_fit",
    "boundary_burden_compatible",
    "specific_increment",
)
MODEL_DIR = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--cross-encoder--nli-deberta-v3-base/snapshots/"
    "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _semantic_text(text: str) -> str:
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


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _score_nli(
    records: list[dict[str, Any]],
    factor: str,
    hypotheses: dict[str, dict[str, str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    if torch.cuda.is_available():
        device_index = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.get_device_properties(index).total_memory,
        )
        device = torch.device(f"cuda:{device_index}")
    else:
        torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
        device = torch.device("cpu")
    model.to(device).eval()
    contradiction_id = int(model.config.label2id["contradiction"])
    entailment_id = int(model.config.label2id["entailment"])
    pairs: list[tuple[str, str]] = []
    for row in records:
        premise = (
            f"Current user message: {row['semantic_state_text']}\n"
            f"Retrieved candidate: {row['semantic_candidate_text']}"
        )
        pairs.append((premise, hypotheses[factor]["positive"]))
        pairs.append((premise, hypotheses[factor]["negative"]))
    supports: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), 16):
            batch = pairs[start : start + 16]
            encoded = tokenizer(
                [row[0] for row in batch],
                [row[1] for row in batch],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            supports.extend(
                (
                    logits[:, entailment_id] - logits[:, contradiction_id]
                ).detach().cpu().numpy().astype(float).tolist()
            )
    pair_support = np.asarray(supports, dtype=np.float64).reshape(len(records), 2)
    probabilities = _softmax(pair_support)[:, 0]
    return probabilities, {
        "device": str(device),
        "pairs": len(pairs),
        "max_length": 384,
        "label2id": dict(model.config.label2id),
    }


def _metric(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = (probabilities >= 0.5).astype(int)
    tp = int(np.sum((labels == 1) & (prediction == 1)))
    tn = int(np.sum((labels == 0) & (prediction == 0)))
    fp = int(np.sum((labels == 0) & (prediction == 1)))
    fn = int(np.sum((labels == 1) & (prediction == 0)))
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
                        "counterfactual_group_id": group_id,
                        "correct": bool(probabilities[pos] > probabilities[neg]),
                    }
                )
    correct = sum(row["correct"] for row in comparisons)
    return {
        "n": len(comparisons),
        "correct": correct,
        "rate": correct / len(comparisons) if comparisons else None,
    }


def _factor_report(
    records: list[dict[str, Any]],
    factor: str,
    probabilities: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray([int(row["human"][factor] == "yes") for row in records])
    metric = _metric(labels, probabilities)
    direction = _direction(records, labels, probabilities)
    family_metrics = {}
    for family in sorted({row["semantic_families"][factor] for row in records}):
        indices = np.asarray(
            [
                index
                for index, row in enumerate(records)
                if row["semantic_families"][factor] == family
            ]
        )
        family_metrics[family] = _metric(labels[indices], probabilities[indices])
    worst_family = min(
        value["balanced_accuracy"] for value in family_metrics.values()
    )
    gates = contract["gates"]
    passed = bool(
        metric["balanced_accuracy"] >= gates["each_factor_balanced_accuracy_min"]
        and metric["recall"] >= gates["each_factor_recall_min"]
        and metric["specificity"] >= gates["each_factor_specificity_min"]
        and direction["rate"]
        >= gates["each_factor_minimal_counterfactual_direction_min"]
        and worst_family
        >= gates["each_factor_worst_semantic_family_balanced_accuracy_min"]
    )
    metric.update(
        {
            "minimal_counterfactual_direction": direction,
            "semantic_family_metrics": family_metrics,
            "worst_semantic_family_balanced_accuracy": worst_family,
            "status": "PASS" if passed else "FAIL",
        }
    )
    return metric


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
        default=ROOT / "data/pm_v1_5_contracts/v3_observation_nli_recovery_v1.json",
    )
    parser.add_argument(
        "--prior-bakeoff",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_v3_observation_orthogonal_bakeoff_v2/qualification_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v3_observation_nli_recovery_v1",
    )
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    observed_hashes = {
        "reference_sha256": sha256_file(args.reference),
        "candidate_rows_sha256": sha256_file(args.candidates),
        "blueprint_sha256": sha256_file(args.blueprint),
        "prior_failed_bakeoff_sha256": sha256_file(args.prior_bakeoff),
    }
    for key, observed in observed_hashes.items():
        if observed != contract[key]:
            raise RuntimeError(f"frozen hash mismatch for {key}")
    model_hashes = {
        "config_sha256": sha256_file(MODEL_DIR / "config.json"),
        "model_sha256": sha256_file(MODEL_DIR / "model.safetensors"),
        "tokenizer_sha256": sha256_file(MODEL_DIR / "tokenizer.json"),
    }
    for key, observed in model_hashes.items():
        if observed != contract["candidate"][key]:
            raise RuntimeError(f"frozen model hash mismatch for {key}")

    reference = _rows(args.reference)
    candidates = {str(row["state_id"]): row for row in _rows(args.candidates)}
    blueprint = {str(row["blueprint_row_id"]): row for row in _rows(args.blueprint)}
    if len(reference) != 96 or len(candidates) != 96 or len(blueprint) != 96:
        raise RuntimeError("NLI recovery requires 96 aligned rows")

    records = []
    for label in reference:
        state_id = str(label["state_id"])
        candidate_row = candidates[state_id]
        exact = candidate_row["exact_rank1_candidate"]
        if exact["candidate_text_sha256"] != label["candidate_text_sha256"]:
            raise RuntimeError(f"candidate/reference hash mismatch: {state_id}")
        construction = blueprint[state_id]
        records.append(
            {
                "state_id": state_id,
                "component": str(label["component"]),
                "track": str(label["track"]),
                "counterfactual_group_id": str(label["counterfactual_group_id"]),
                "topic_family": str(construction["topic_family"]),
                "history_scale": str(construction["history_scale"]),
                "prefix_bucket": str(construction["prefix_family"]),
                "semantic_families": construction["private_semantic_surface_family"],
                "state_text": str(candidate_row["current_user_text"]),
                "candidate_text": str(exact["candidate_text"]),
                "semantic_state_text": _semantic_text(candidate_row["current_user_text"]),
                "semantic_candidate_text": _semantic_text(exact["candidate_text"]),
                "human": label,
            }
        )

    legacy = runpy.run_path(
        str(ROOT / "scripts/v1_5/25zc_run_v3_observation_factor_bakeoff_v1_5.py")
    )
    fit = [row for row in records if row["track"] == "FACTOR_FIT"]
    confirmation = [row for row in records if row["track"] == "ELIGIBILITY_CONFIRMATION"]
    nuisance = {}
    fit_reports = {}
    fit_prediction_rows = []
    model_runs = {}
    for factor in FACTORS:
        applicable = [
            row
            for row in fit
            if not (factor == "owner_time_entity_valid" and row["component"] == "RS")
        ]
        labels = np.asarray([int(row["human"][factor] == "yes") for row in applicable])
        groups = np.asarray([row["counterfactual_group_id"] for row in applicable])
        nuisance[factor] = legacy["_nuisance_probe"](applicable, labels, groups)
        probabilities, model_run = _score_nli(
            applicable, factor, contract["hypotheses"]
        )
        model_runs[f"fit_{factor}"] = model_run
        fit_reports[factor] = _factor_report(
            applicable, factor, probabilities, contract
        )
        for row, label, probability in zip(
            applicable, labels, probabilities, strict=True
        ):
            fit_prediction_rows.append(
                {
                    "protocol": PROTOCOL,
                    "state_id": row["state_id"],
                    "component": row["component"],
                    "factor": factor,
                    "gold": int(label),
                    "nli_probability": float(probability),
                }
            )
    nuisance_pass = all(
        row["balanced_accuracy"]
        < contract["gates"][
            "raw_surface_nuisance_probe_balanced_accuracy_max_exclusive"
        ]
        for row in nuisance.values()
    )
    fit_pass = nuisance_pass and all(
        row["status"] == "PASS" for row in fit_reports.values()
    )

    confirmation_report = None
    confirmation_rows = []
    confirmation_pass = False
    if fit_pass:
        per_factor_probability: dict[str, dict[str, float]] = defaultdict(dict)
        per_factor_report = {}
        for factor in FACTORS:
            applicable = [
                row
                for row in confirmation
                if not (
                    factor == "owner_time_entity_valid" and row["component"] == "RS"
                )
            ]
            labels = np.asarray(
                [int(row["human"][factor] == "yes") for row in applicable]
            )
            probabilities, model_run = _score_nli(
                applicable, factor, contract["hypotheses"]
            )
            model_runs[f"confirmation_{factor}"] = model_run
            per_factor_report[factor] = _metric(labels, probabilities)
            for row, probability in zip(applicable, probabilities, strict=True):
                per_factor_probability[factor][row["state_id"]] = float(probability)
        composite_gold = []
        composite_probability = []
        per_component: defaultdict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"gold": [], "probability": []}
        )
        for row in confirmation:
            values = [
                per_factor_probability[factor][row["state_id"]]
                for factor in FACTORS
                if not (
                    factor == "owner_time_entity_valid" and row["component"] == "RS"
                )
            ]
            probability = float(min(values))
            gold = int(row["human"]["derived_eligibility"] == "eligible")
            composite_gold.append(gold)
            composite_probability.append(probability)
            per_component[row["component"]]["gold"].append(gold)
            per_component[row["component"]]["probability"].append(probability)
            confirmation_rows.append(
                {
                    "protocol": PROTOCOL,
                    "state_id": row["state_id"],
                    "component": row["component"],
                    "gold_eligibility": gold,
                    "composed_probability_min_gate": probability,
                    "predicted_eligibility": int(probability >= 0.5),
                }
            )
        global_metric = _metric(
            np.asarray(composite_gold), np.asarray(composite_probability)
        )
        gates = contract["gates"]
        confirmation_pass = bool(
            all(
                row["balanced_accuracy"]
                >= gates["confirmation_each_factor_balanced_accuracy_min"]
                for row in per_factor_report.values()
            )
            and global_metric["balanced_accuracy"]
            >= gates["confirmation_composite_balanced_accuracy_min"]
            and global_metric["recall"]
            >= gates["confirmation_composite_recall_min"]
            and global_metric["specificity"]
            >= gates["confirmation_composite_specificity_min"]
        )
        confirmation_report = {
            "status": "PASS" if confirmation_pass else "FAIL",
            "one_shot_consumed": True,
            "used_for_hypothesis_threshold_or_model_selection": False,
            "per_factor": per_factor_report,
            "composite_global": global_metric,
            "composite_per_component": {
                component: _metric(
                    np.asarray(values["gold"]),
                    np.asarray(values["probability"]),
                )
                for component, values in sorted(per_component.items())
            },
        }

    if not fit_pass:
        status = "FAIL_FIT_CONFIRMATION_NOT_CONSUMED"
    elif confirmation_pass:
        status = "PASS_FIT_AND_ONE_SHOT_CONFIRMATION"
    else:
        status = "PASS_FIT_FAIL_ONE_SHOT_CONFIRMATION"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit_path = args.out_dir / "fit_predictions.jsonl"
    confirmation_path = args.out_dir / "confirmation_predictions.jsonl"
    write_jsonl(fit_path, fit_prediction_rows)
    write_jsonl(confirmation_path, confirmation_rows)
    report = {
        "protocol": PROTOCOL,
        "status": status,
        "candidate": contract["candidate"]["id"],
        "fit_rows": len(fit),
        "confirmation_rows": len(confirmation),
        "fit_pass": fit_pass,
        "fit_factor_reports": fit_reports,
        "raw_surface_nuisance_probe": nuisance,
        "raw_surface_nuisance_probe_pass": nuisance_pass,
        "one_shot_confirmation": confirmation_report,
        "step1_training_authorized": False,
        "step2_qualification_authorized": confirmation_pass,
        "response_or_outcome_read": False,
        "external_lockbox_read": False,
        "confirmation_read_before_fit_decision": False,
        "model_runs": model_runs,
        "frozen_input_hashes": observed_hashes,
        "frozen_model_hashes": model_hashes,
        "contract_sha256": sha256_file(args.contract),
        "fit_predictions_sha256": sha256_file(fit_path),
        "confirmation_predictions_sha256": sha256_file(confirmation_path),
        "component_counts": dict(Counter(row["component"] for row in records)),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    write_json(args.out_dir / "qualification_report.json", report)
    print(
        {
            "status": status,
            "fit_pass": fit_pass,
            "fit_factor_statuses": {
                factor: value["status"] for factor, value in fit_reports.items()
            },
            "confirmation": (
                confirmation_report["composite_global"]
                if confirmation_report is not None
                else None
            ),
            "out": str(args.out_dir / "qualification_report.json"),
        }
    )


if __name__ == "__main__":
    main()
