#!/usr/bin/env python3
"""Qualify BGE-small only as a candidate-memory relevance feature.

This challenger reads the 48-user fit partition only.  It does not read the
consumed diagnostic confirmation, response outcomes, human judgments, or any
external outcome.  The comparison replaces the lexical candidate-state score
with max cosine(query, selected item) while keeping every other feature,
splitter, model family, C, threshold, and seed fixed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from metacom_pm.contracts import MemoryBackendRecord, MemorySource, RuntimeState
from metacom_pm.io import iter_jsonl, sha256_file, write_json, write_jsonl
from metacom_pm.retrieval import source_specific_memory_queries


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-bge-memory-candidate-semantic-qualification-v1"
MODEL_PATH = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
COMPONENTS = ("MP", "MS", "ME")
MIN_LOCO_BA_GAIN = 0.05
MAX_OOF_BA_REGRESSION = 0.02


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _trainer():
    path = ROOT / "scripts/v1_5/24dx_train_real_text_memory_opportunity_heads_v1_5.py"
    spec = importlib.util.spec_from_file_location("memory_head_trainer", path)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load frozen memory trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode(texts: list[str], model_path: Path) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
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
            batch = {key: value.to(device) for key, value in batch.items()}
            vectors = model(**batch).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            chunks.append(vectors.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _d3_surfaces(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    source_dir = root / "outputs/pm_v1_5_d3_step0_blueprint_v1"
    blueprints = {
        (str(row["component"]), str(row["user_id"])): row
        for row in _rows(source_dir / "d3_contrast_blueprint.jsonl")
        if row["component"] in COMPONENTS
    }
    states = {
        row.user_id: row
        for row in (
            RuntimeState.model_validate(value)
            for value in _rows(source_dir / "runtime_states.jsonl")
        )
    }
    backends = {
        row.card_id: row
        for row in (
            MemoryBackendRecord.model_validate(value)
            for value in _rows(source_dir / "memory_backend.jsonl")
        )
    }
    surfaces: dict[tuple[str, str], dict[str, Any]] = {}
    for (component, user_id), blueprint in blueprints.items():
        state = states[user_id]
        backend = backends[state.card_id]
        source = MemorySource(component)
        item_by_id = {item.memory_id: item for item in backend.items}
        ids = blueprint["selected_memory_ids_by_action_generation_only"][
            blueprint["treatment_action"]
        ]
        selected = [
            item_by_id[str(memory_id)]
            for memory_id in ids
            if item_by_id[str(memory_id)].source is source
        ]
        queries = source_specific_memory_queries(
            state.current_user_text,
            [turn.model_dump(mode="json") for turn in state.current_session_history],
            state.current_session_summary,
        )
        surfaces[(component, user_id)] = {
            "query": queries[source],
            "candidate_texts": [item.text for item in selected],
        }
    return surfaces


def _supplement_surfaces(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = (
        root
        / "outputs/pm_v1_5_memory_opportunity_supplement_v1/candidate_semantic_surfaces.jsonl"
    )
    return {
        (str(row["component"]), str(row["user_id_private_not_model_input"])): {
            "query": str(row["query_private_semantic_encoder_input"]),
            "candidate_texts": list(
                row["selected_candidate_texts_private_semantic_encoder_input"]
            ),
        }
        for row in _rows(path)
        if row["evidence_role"] == "supplemental_fit"
    }


def _semantic_scores(
    fit_rows: list[dict[str, Any]], surfaces: dict[tuple[str, str], dict[str, Any]], model_path: Path
) -> list[float]:
    texts: list[str] = []
    spans: list[tuple[int, list[int]]] = []
    for row in fit_rows:
        key = (
            str(row["component"]),
            str(row["user_id_private_not_model_input"]),
        )
        surface = surfaces[key]
        query_index = len(texts)
        texts.append(str(surface["query"]))
        candidate_indices = []
        for value in surface["candidate_texts"]:
            candidate_indices.append(len(texts))
            texts.append(str(value))
        if not candidate_indices:
            raise RuntimeError(f"candidate surface absent: {key}")
        spans.append((query_index, candidate_indices))
    vectors = _encode(texts, model_path)
    return [
        max(float(np.dot(vectors[query], vectors[index])) for index in candidates)
        for query, candidates in spans
    ]


def _evaluate_component(
    *, component: str, rows: list[dict[str, Any]], semantic: list[float], baseline: dict[str, Any], trainer: Any
) -> dict[str, Any]:
    feature_names = sorted(rows[0]["model_features"])
    values = []
    for row, score in zip(rows, semantic, strict=True):
        features = dict(row["model_features"])
        features.pop("candidate_state_match_score")
        features["candidate_state_bge_cosine"] = float(score)
        values.append(features)
    challenger_features = sorted(values[0])
    x = np.asarray(
        [[float(row[name]) for name in challenger_features] for row in values],
        dtype=float,
    )
    y = np.asarray([row["opportunity_target"] for row in rows], dtype=int)
    users = np.asarray(
        [row["user_id_private_not_model_input"] for row in rows], dtype=object
    )
    conditions = np.asarray(
        [row["condition_family_private_not_model_input"] for row in rows],
        dtype=object,
    )
    seed_reports = []
    predictions = []
    for seed in trainer.SEEDS:
        probability, prior = trainer._oof(
            x=x,
            y=y,
            groups=users,
            seed=seed,
            leave_one_group_out=False,
        )
        condition_probability, condition_prior = trainer._oof(
            x=x,
            y=y,
            groups=conditions,
            seed=seed,
            leave_one_group_out=True,
        )
        predictions.append(probability >= trainer.DECISION_THRESHOLD)
        seed_reports.append(
            {
                "seed": seed,
                "user_grouped_oof": trainer._metrics(y, probability, prior),
                "leave_condition_family_out": trainer._metrics(
                    y, condition_probability, condition_prior
                ),
            }
        )
    matrix = np.stack(predictions, axis=0)
    agreement = float(
        np.mean(
            np.logical_or(np.all(matrix, axis=0), np.all(~matrix, axis=0))
        )
    )
    oof = [row["user_grouped_oof"] for row in seed_reports]
    loco = [row["leave_condition_family_out"] for row in seed_reports]
    baseline_oof = [row["user_grouped_oof"] for row in baseline["seed_reports"]]
    baseline_loco = [
        row["leave_condition_family_out"] for row in baseline["seed_reports"]
    ]
    worst_oof = min(row["balanced_accuracy"] for row in oof)
    worst_loco = min(row["balanced_accuracy"] for row in loco)
    baseline_worst_oof = min(
        row["balanced_accuracy"] for row in baseline_oof
    )
    baseline_worst_loco = min(
        row["balanced_accuracy"] for row in baseline_loco
    )
    checks = {
        "challenger_worst_oof_ba_min_0_70": worst_oof >= 0.70,
        "challenger_recall_specificity_min_0_65": (
            min(row["positive_recall"] for row in oof) >= 0.65
            and min(row["specificity"] for row in oof) >= 0.65
        ),
        "challenger_every_seed_brier_beats_prior": min(
            row["brier_gain_vs_train_fold_prior"] for row in oof
        )
        > 0.0,
        "challenger_decision_agreement_min_0_90": agreement >= 0.90,
        "challenger_worst_loco_ba_min_0_65": worst_loco >= 0.65,
        "loco_ba_gain_over_lexical_min_0_05": (
            worst_loco - baseline_worst_loco >= MIN_LOCO_BA_GAIN
        ),
        "oof_ba_regression_vs_lexical_at_most_0_02": (
            baseline_worst_oof - worst_oof <= MAX_OOF_BA_REGRESSION
        ),
    }
    return {
        "selected": all(checks.values()),
        "baseline_feature_names": feature_names,
        "challenger_feature_names": challenger_features,
        "baseline_worst_oof_ba": baseline_worst_oof,
        "challenger_worst_oof_ba": worst_oof,
        "baseline_worst_leave_condition_ba": baseline_worst_loco,
        "challenger_worst_leave_condition_ba": worst_loco,
        "five_seed_decision_agreement": agreement,
        "checks": checks,
        "seed_reports": seed_reports,
    }


def _lexical_baseline_on_fit(rows: list[dict[str, Any]], trainer: Any) -> dict[str, Any]:
    feature_names = sorted(rows[0]["model_features"])
    x = np.asarray(
        [
            [float(row["model_features"][name]) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    y = np.asarray([row["opportunity_target"] for row in rows], dtype=int)
    users = np.asarray(
        [row["user_id_private_not_model_input"] for row in rows], dtype=object
    )
    conditions = np.asarray(
        [row["condition_family_private_not_model_input"] for row in rows],
        dtype=object,
    )
    seed_reports = []
    for seed in trainer.SEEDS:
        probability, prior = trainer._oof(
            x=x,
            y=y,
            groups=users,
            seed=seed,
            leave_one_group_out=False,
        )
        condition_probability, condition_prior = trainer._oof(
            x=x,
            y=y,
            groups=conditions,
            seed=seed,
            leave_one_group_out=True,
        )
        seed_reports.append(
            {
                "seed": seed,
                "user_grouped_oof": trainer._metrics(y, probability, prior),
                "leave_condition_family_out": trainer._metrics(
                    y, condition_probability, condition_prior
                ),
            }
        )
    return {"seed_reports": seed_reports}


def build(*, root: Path = ROOT, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    if not model_path.is_dir():
        raise RuntimeError(f"local BGE model unavailable: {model_path}")
    fit_path = (
        root
        / "outputs/pm_v1_5_real_text_memory_opportunity_heads_v1/fit_rows_audit.jsonl"
    )
    fit_rows = _rows(fit_path)
    surfaces = {**_d3_surfaces(root), **_supplement_surfaces(root)}
    semantic = _semantic_scores(fit_rows, surfaces, model_path)
    trainer = _trainer()
    reports: dict[str, Any] = {}
    score_rows: list[dict[str, Any]] = []
    cursor = 0
    for component in COMPONENTS:
        rows = [row for row in fit_rows if row["component"] == component]
        scores = semantic[cursor : cursor + len(rows)]
        cursor += len(rows)
        reports[component] = _evaluate_component(
            component=component,
            rows=rows,
            semantic=scores,
            baseline=_lexical_baseline_on_fit(rows, trainer),
            trainer=trainer,
        )
        score_rows.extend(
            {
                "protocol": PROTOCOL,
                "component": component,
                "user_id_private_not_model_input": row[
                    "user_id_private_not_model_input"
                ],
                "condition_family_private_not_model_input": row[
                    "condition_family_private_not_model_input"
                ],
                "opportunity_target": row["opportunity_target"],
                "candidate_state_bge_cosine": float(score),
                "confirmation_or_external_read": False,
            }
            for row, score in zip(rows, scores, strict=True)
        )
    selected = [component for component in COMPONENTS if reports[component]["selected"]]
    report = {
        "protocol": PROTOCOL,
        "status": (
            "SELECT_BGE_FOR_" + "_".join(selected)
            if selected
            else "REJECT_BGE_KEEP_LEXICAL"
        ),
        "selected_components": selected,
        "reports": reports,
        "model": "BAAI/bge-small-en-v1.5",
        "model_path": str(model_path),
        "selection_data": "48-user fit partition only",
        "diagnostic_confirmation_read": False,
        "fresh_confirmation_read": False,
        "external_outcome_read": False,
        "inputs": {
            str(fit_path.relative_to(root)): sha256_file(fit_path),
            "bge_config.json": sha256_file(model_path / "config.json"),
        },
    }
    out_dir = root / "outputs/pm_v1_5_bge_memory_candidate_qualification_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "qualification_report.json", report)
    write_jsonl(out_dir / "semantic_score_rows.jsonl", score_rows)
    return report


def main() -> None:
    report = build()
    print(
        {
            "protocol": report["protocol"],
            "status": report["status"],
            "selected_components": report["selected_components"],
            "summary": {
                component: {
                    "selected": value["selected"],
                    "lexical_loco": value["baseline_worst_leave_condition_ba"],
                    "bge_loco": value["challenger_worst_leave_condition_ba"],
                    "lexical_oof": value["baseline_worst_oof_ba"],
                    "bge_oof": value["challenger_worst_oof_ba"],
                }
                for component, value in report["reports"].items()
            },
        }
    )


if __name__ == "__main__":
    main()
