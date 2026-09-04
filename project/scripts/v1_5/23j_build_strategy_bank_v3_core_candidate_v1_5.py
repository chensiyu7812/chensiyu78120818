#!/usr/bin/env python3
"""Build 50 outcome-blind core Strategy Bank V3 candidate cards."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from metacom_pm.v1_5_strategy_bank_v3 import (
    EXPECTED_FAMILIES,
    card_distinctness_report,
    make_card_id,
    summarize_v3_assignments,
    taxonomy_embedding_text,
    taxonomy_sha256,
    validate_v3_taxonomy,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path(
    "/home/tokkio/.cache/huggingface/hub/"
    "models--BAAI--bge-small-en-v1.5/snapshots/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)


def _encode(
    *,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            chunks.append(hidden.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=ROOT
        / "data/strategy/pm_v1_5_strategy_bank_v3_core_taxonomy_v1.jsonl",
    )
    parser.add_argument(
        "--raw-cards",
        type=Path,
        default=ROOT / "data/strategy/strategy_cards_v1_5.jsonl",
    )
    parser.add_argument(
        "--v2-cards",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
        / "strategy_cards_v2_candidate.jsonl",
    )
    parser.add_argument(
        "--v2-lineage",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v2_review_candidate_v2"
        / "strategy_bank_v2_lineage.jsonl",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v3_core_candidate_v1",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-score", type=float, default=0.45)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--minimum-dialogues", type=int, default=20)
    parser.add_argument("--distinctness-warning-cosine", type=float, default=0.90)
    args = parser.parse_args()

    specs = validate_v3_taxonomy(
        [dict(row) for row in iter_jsonl(args.taxonomy)]
    )
    taxonomy_digest = taxonomy_sha256(specs)
    raw_by_id = {
        str(row["strategy_id"]): dict(row)
        for row in iter_jsonl(args.raw_cards)
    }
    family_by_v2_card = {
        str(row["card_id"]): str(row["strategy_family"])
        for row in iter_jsonl(args.v2_cards)
    }
    source_rows: list[dict[str, Any]] = []
    for row in iter_jsonl(args.v2_lineage):
        source = dict(row)
        strategy_id = str(source["strategy_id"])
        raw = raw_by_id.get(strategy_id)
        if raw is None:
            raise RuntimeError(f"missing raw strategy row: {strategy_id}")
        family = family_by_v2_card[str(source["card_id"])]
        if str(raw["strategy_label"]) != family:
            raise RuntimeError(f"strategy family drift: {strategy_id}")
        source_rows.append(
            {
                **source,
                "strategy_family": family,
                "source_response_for_offline_assignment": str(
                    raw["example_response"]
                ),
            }
        )

    model_id = "BAAI/bge-small-en-v1.5@" + args.model.name
    card_ids = {
        str(spec["submove_id"]): make_card_id(
            spec=spec,
            taxonomy_sha256=taxonomy_digest,
            embedding_model_id=model_id,
        )
        for spec in specs
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True
    )
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to(
        device
    )

    specs_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_indices_by_family: dict[str, list[int]] = defaultdict(list)
    for spec in specs:
        specs_by_family[str(spec["strategy_family"])].append(spec)
    for family in specs_by_family:
        specs_by_family[family].sort(key=lambda row: str(row["submove_id"]))
    for index, row in enumerate(source_rows):
        source_indices_by_family[str(row["strategy_family"])].append(index)

    score_matrices: dict[str, np.ndarray] = {}
    card_embeddings_by_family: dict[str, np.ndarray] = {}
    for family in EXPECTED_FAMILIES:
        family_specs = specs_by_family[family]
        card_embeddings = _encode(
            texts=[taxonomy_embedding_text(row) for row in family_specs],
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
        )
        source_embeddings = _encode(
            texts=[
                str(source_rows[index]["source_response_for_offline_assignment"])
                for index in source_indices_by_family[family]
            ],
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
        )
        card_embeddings_by_family[family] = card_embeddings
        score_matrices[family] = source_embeddings @ card_embeddings.T

    built = summarize_v3_assignments(
        specs=specs,
        card_ids=card_ids,
        source_rows=source_rows,
        score_matrix_by_family=score_matrices,
        source_indices_by_family=source_indices_by_family,
        minimum_score=args.minimum_score,
        minimum_margin=args.minimum_margin,
        minimum_dialogues=args.minimum_dialogues,
    )
    distinctness = card_distinctness_report(
        specs=specs,
        embeddings_by_family=card_embeddings_by_family,
        warning_cosine=args.distinctness_warning_cosine,
    )
    cards = list(built["cards"])
    mapping_rows = list(built["mapping_rows"])
    family_assignment_counts: dict[str, dict[str, int]] = {}
    for family in EXPECTED_FAMILIES:
        family_assignment_counts[family] = {
            str(spec["submove_id"]): sum(
                row["strategy_family"] == family
                and row["assigned_submove_id"] == spec["submove_id"]
                for row in mapping_rows
            )
            for spec in specs_by_family[family]
        }
    report = {
        "protocol": "pm-v1.5-strategy-bank-v3-core-build-report-v1",
        "status": "CORE_50_BUILT_PENDING_TAXONOMY_AND_SOURCE_SAMPLE_REVIEW",
        "card_count": len(cards),
        "family_count": len(EXPECTED_FAMILIES),
        "cards_per_family": 10,
        "source_rows": len(source_rows),
        "source_dialogue_count": len(
            {str(row["source_dialogue_id"]) for row in source_rows}
        ),
        "raw_examples_exposed_to_generator": False,
        "embedding_role": (
            "weak outcome-blind source-to-submove support mapping and "
            "duplicate warning only; not PM input, gold label, or card utility"
        ),
        "embedding_model": model_id,
        "device_type": device.type,
        "assignment_summary": built["assignment_summary"],
        "family_assignment_counts": family_assignment_counts,
        "distinctness": distinctness,
        "automatic_promotion_authorized": False,
        "next_gate": (
            "Review the 50 taxonomy definitions, warning pairs, and a fixed "
            "source sample per card before authoring execution variants."
        ),
        "lineage": {
            "taxonomy_sha256": sha256_file(args.taxonomy),
            "raw_cards_sha256": sha256_file(args.raw_cards),
            "v2_cards_sha256": sha256_file(args.v2_cards),
            "v2_lineage_sha256": sha256_file(args.v2_lineage),
            "model_config_sha256": sha256_file(args.model / "config.json"),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "strategy_cards_v3_core_candidate.jsonl", cards)
    write_jsonl(
        args.out_dir / "strategy_cards_v3_weak_source_mapping.jsonl",
        mapping_rows,
    )
    write_json(args.out_dir / "build_report.json", report)
    print(canonical_json({"output": str(args.out_dir), **report}))


if __name__ == "__main__":
    main()
