#!/usr/bin/env python3
"""Profile within-family source diversity before expanding Strategy Bank V3."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
CLUSTER_COUNTS = (5, 8, 10, 12, 16, 20)


def _percentiles(values: list[int]) -> list[float]:
    return [
        round(float(value), 1)
        for value in np.percentile(np.asarray(values), [25, 50, 75])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_strategy_bank_v3_diversity_profile_v1.json",
    )
    parser.add_argument("--seed", type=int, default=4311)
    args = parser.parse_args()

    raw_by_id = {
        str(row["strategy_id"]): dict(row)
        for row in iter_jsonl(args.raw_cards)
    }
    family_by_card = {
        str(row["card_id"]): str(row["strategy_family"])
        for row in iter_jsonl(args.v2_cards)
    }
    texts: dict[str, list[str]] = defaultdict(list)
    dialogues: dict[str, set[str]] = defaultdict(set)
    problems: dict[str, Counter[str]] = defaultdict(Counter)
    emotions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in iter_jsonl(args.v2_lineage):
        family = family_by_card[str(row["card_id"])]
        raw = raw_by_id[str(row["strategy_id"])]
        texts[family].append(str(raw["example_response"]))
        dialogues[family].add(str(row["source_dialogue_id"]))
        problems[family][str(row["problem_type"])] += 1
        emotions[family][str(row["emotion_type"])] += 1

    family_profiles: list[dict[str, Any]] = []
    for family in sorted(texts):
        family_texts = texts[family]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.90,
            max_features=6000,
        )
        matrix = vectorizer.fit_transform(family_texts)
        cluster_diagnostic: dict[str, float] = {}
        for count in CLUSTER_COUNTS:
            labels = MiniBatchKMeans(
                n_clusters=count,
                random_state=args.seed,
                n_init=5,
                batch_size=256,
            ).fit_predict(matrix)
            cluster_diagnostic[str(count)] = round(
                float(
                    silhouette_score(
                        matrix,
                        labels,
                        metric="cosine",
                        sample_size=min(800, len(family_texts)),
                        random_state=args.seed,
                    )
                ),
                3,
            )
        word_lengths = [
            len(re.findall(r"\b\w+\b", text)) for text in family_texts
        ]
        family_profiles.append(
            {
                "strategy_family": family,
                "eligible_source_rows": len(family_texts),
                "independent_source_dialogues": len(dialogues[family]),
                "word_length_p25_p50_p75": _percentiles(word_lengths),
                "question_mark_rate": round(
                    sum("?" in text for text in family_texts)
                    / len(family_texts),
                    3,
                ),
                "distinct_problem_types": len(problems[family]),
                "distinct_emotion_types": len(emotions[family]),
                "tfidf_cluster_silhouette_diagnostic": cluster_diagnostic,
                "cluster_interpretation": (
                    "near-zero silhouette: raw wording does not justify "
                    "automatic subtype discovery"
                    if max(cluster_diagnostic.values()) < 0.10
                    else "some lexical cluster structure is present"
                ),
            }
        )

    report = {
        "protocol": "pm-v1.5-strategy-bank-v3-diversity-profile-v1",
        "status": "SOURCE_VOLUME_SUFFICIENT_AUTOMATIC_CLUSTERING_UNQUALIFIED",
        "intended_use": (
            "justify candidate-bank capacity and reject raw lexical clustering "
            "as the sole card-authoring method"
        ),
        "eligible_source_rows": sum(
            row["eligible_source_rows"] for row in family_profiles
        ),
        "independent_source_dialogue_union": len(
            {
                dialogue_id
                for family_dialogues in dialogues.values()
                for dialogue_id in family_dialogues
            }
        ),
        "family_profiles": family_profiles,
        "decision": {
            "candidate_bank_target": "80-100 cards",
            "target_is_hard_quota": False,
            "core_submoves_per_family": "8-10",
            "automatic_tfidf_clusters_authorized_as_cards": False,
            "required_authoring_basis": (
                "transparent support-move taxonomy plus source-backed "
                "coverage and independent human content review"
            ),
        },
        "lineage": {
            "raw_cards_sha256": sha256_file(args.raw_cards),
            "v2_cards_sha256": sha256_file(args.v2_cards),
            "v2_lineage_sha256": sha256_file(args.v2_lineage),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(canonical_json({"output": str(args.output), **report}))


if __name__ == "__main__":
    main()
