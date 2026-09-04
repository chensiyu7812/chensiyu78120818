#!/usr/bin/env python3
"""Run the single predeclared BGE-small relation-feature challenger for V5 FIT."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from metacom_pm.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-v5-bge-small-relation-challenger-v1"
MODEL_PATH = Path(
    "/home/tokkio/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/"
    "snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)


def rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def trainer() -> Any:
    path = ROOT / "scripts/v1_5/26a_aggregate_and_train_v5_fit_v1_5.py"
    spec = importlib.util.spec_from_file_location("v5_fit_trainer", path)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load frozen V5 trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(texts: list[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(
                texts[start : start + 64],
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            batch = {name: value.to(device) for name, value in batch.items()}
            vector = model(**batch).last_hidden_state[:, 0]
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
            pieces.append(vector.cpu().numpy())
    return np.concatenate(pieces, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-path",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_training_result_v1/state_effect_labels_private.jsonl",
    )
    parser.add_argument(
        "--primary-report",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_fit_training_result_v1/fit_report.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/pm_v1_5_v5_bge_challenger_v1",
    )
    parser.add_argument(
        "--protocol",
        default=PROTOCOL,
    )
    args = parser.parse_args()
    if not MODEL_PATH.is_dir():
        raise RuntimeError(f"local BGE model unavailable: {MODEL_PATH}")
    label_path = args.label_path
    feature_path = ROOT / "outputs/pm_v1_5_v5_fit_training_surface_v1/fit_feature_rows_private.jsonl"
    candidate_path = ROOT / "outputs/pm_v1_5_v3_p2_exact_rank1_v4/candidate_rows_private.jsonl"
    contract_path = ROOT / "data/pm_v1_5_contracts/v5_fit_training_surface_freeze_v1.json"
    primary_path = args.primary_report
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {str(row["state_id"]): row for row in rows(label_path)}
    features = {str(row["state_id"]): row for row in rows(feature_path)}
    candidates = {str(row["state_id"]): row for row in rows(candidate_path)}
    if set(labels) != set(features) or not set(labels) <= set(candidates):
        raise RuntimeError("V5 label/feature/candidate identities do not align")

    # The two views are exactly those named in the pre-label challenger contract.
    # They are encoded once; there is no instruction, prompt, pooling, layer, or
    # model search after observing FIT labels.
    unique: dict[str, int] = {}
    texts: list[str] = []
    triples: dict[str, tuple[int, int, int]] = {}
    for state_id in sorted(labels):
        row = candidates[state_id]
        current = str(row["current_user_text"])
        dialogue = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in row["visible_dialogue"]
        )
        full = f"{dialogue}\nuser: {current}"
        candidate = str(row["exact_rank1_candidate"]["candidate_text"])
        indices = []
        for text in (current, full, candidate):
            if text not in unique:
                unique[text] = len(texts)
                texts.append(text)
            indices.append(unique[text])
        triples[state_id] = (indices[0], indices[1], indices[2])
    vectors = encode(texts)

    augmented: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for state_id in sorted(labels):
        current_i, full_i, candidate_i = triples[state_id]
        current_cosine = float(np.dot(vectors[current_i], vectors[candidate_i]))
        full_cosine = float(np.dot(vectors[full_i], vectors[candidate_i]))
        base = dict(features[state_id])
        model_features = dict(base["model_features"])
        model_features["bge_current_goal_candidate_cosine"] = current_cosine
        model_features["bge_full_visible_context_candidate_cosine"] = full_cosine
        base["model_features"] = model_features
        # V5.2 labels preserve the new execution semantic-group identity,
        # while grouped CV must retain the pre-label counterfactual grouping
        # and semantic-family boundary frozen on the feature surface.
        base["semantic_group_id"] = labels[state_id].get(
            "semantic_group_id_v5_2",
            base.get("semantic_group_id", base["semantic_group_id_private_cv_only"]),
        )
        base["semantic_family"] = base.get(
            "semantic_family", base["semantic_family_private_cv_only"]
        )
        augmented.append({**base, **labels[state_id]})
        semantic_rows.append(
            {
                "protocol": args.protocol,
                "state_id": state_id,
                "component": labels[state_id]["component"],
                "bge_current_goal_candidate_cosine": current_cosine,
                "bge_full_visible_context_candidate_cosine": full_cosine,
                "external_or_confirmation_text_read": False,
            }
        )

    module = trainer()
    contract = read_json(contract_path)
    heads = {
        component: module._fit_component(
            rows=[row for row in augmented if row["component"] == component],
            contract=contract,
        )
        for component in module.COMPONENTS
    }
    status = (
        "CHALLENGER_PROMOTED_TO_FRESH_CONFIRMATION"
        if all(row["status"] == "FIT_PROMOTED_TO_FRESH_CONFIRMATION" for row in heads.values())
        else "CHALLENGER_PARTIAL_OR_NOT_LEARNED_ON_FROZEN_FIT"
    )
    report = {
        "protocol": args.protocol,
        "status": status,
        "model": "BAAI/bge-small-en-v1.5",
        "views": [
            "current_user_text to exact Rank-1 candidate cosine",
            "full visible dialogue plus current user text to exact Rank-1 candidate cosine",
        ],
        "integration": "append exactly two frozen relation features to the unchanged transparent primary features",
        "same_trainer_contract": str(contract_path.relative_to(ROOT)),
        "selection_search_performed": False,
        "confirmation_or_external_read": False,
        "timing_disclosure": (
            "The model and two semantic views were frozen before FIT labels; this exact "
            "serialization script was materialized after FIT labels and was executed once "
            "without outcome-guided alternatives. Treat any gain as challenger evidence, "
            "not permission to modify the frozen transparent primary."
        ),
        "heads": heads,
        "input_sha256": {
            "labels": sha256_file(label_path),
            "features": sha256_file(feature_path),
            "candidate_rows": sha256_file(candidate_path),
            "contract": sha256_file(contract_path),
            "primary_report": sha256_file(primary_path),
            "bge_config": sha256_file(MODEL_PATH / "config.json"),
        },
    }
    write_jsonl(out_dir / "semantic_features_private.jsonl", semantic_rows)
    write_json(out_dir / "challenger_report.json", report)
    print({"protocol": args.protocol, "status": status})


if __name__ == "__main__":
    main()
