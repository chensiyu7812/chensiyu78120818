"""Outcome-blind construction helpers for the expanded Strategy Bank V3."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import canonical_json, sha256_text, stable_hex


V3_PROTOCOL = "pm-v1.5-strategy-bank-v3-core-candidate-v1"
EXPECTED_FAMILIES = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Affirmation and Reassurance",
    "Providing Suggestions",
)
REQUIRED_SPEC_FIELDS = (
    "submove_id",
    "strategy_family",
    "support_move",
    "when_to_use",
    "when_not_to_use",
    "compatible_support_modes",
    "compatible_dialogue_phases",
    "goal_types",
    "directive_burden",
    "risk_flags",
)


def validate_v3_taxonomy(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the frozen 10-submove-per-family core taxonomy."""

    specs = [dict(row) for row in rows]
    if len(specs) != 50:
        raise ValueError("V3 core taxonomy must contain exactly 50 submoves")
    ids = [str(row.get("submove_id", "")) for row in specs]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("V3 submove ids must be unique and non-empty")
    family_counts = Counter(str(row.get("strategy_family", "")) for row in specs)
    if set(family_counts) != set(EXPECTED_FAMILIES):
        raise ValueError("V3 taxonomy family set differs from the safe core")
    if any(family_counts[family] != 10 for family in EXPECTED_FAMILIES):
        raise ValueError("V3 core taxonomy requires ten submoves per family")
    for row in specs:
        missing = [field for field in REQUIRED_SPEC_FIELDS if field not in row]
        if missing:
            raise ValueError(
                f"{row.get('submove_id')}: missing taxonomy fields {missing}"
            )
        for field in (
            "compatible_support_modes",
            "compatible_dialogue_phases",
            "goal_types",
            "risk_flags",
        ):
            values = row[field]
            if (
                not isinstance(values, list)
                or not values
                or len(values) != len(set(values))
            ):
                raise ValueError(
                    f"{row['submove_id']}: {field} must be non-empty/unique"
                )
        if row["directive_burden"] not in {"none", "light", "structured"}:
            raise ValueError(
                f"{row['submove_id']}: invalid directive_burden"
            )
    return specs


def taxonomy_embedding_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        (
            str(row["strategy_family"]),
            str(row["support_move"]),
            str(row["when_to_use"]),
            "Do not:",
            str(row["when_not_to_use"]),
        )
    )


def make_card_id(
    *,
    spec: Mapping[str, Any],
    taxonomy_sha256: str,
    embedding_model_id: str,
) -> str:
    return "strategy_v3_" + stable_hex(
        V3_PROTOCOL,
        dict(spec),
        taxonomy_sha256,
        embedding_model_id,
        n=24,
    )


def summarize_v3_assignments(
    *,
    specs: Sequence[Mapping[str, Any]],
    card_ids: Mapping[str, str],
    source_rows: Sequence[Mapping[str, Any]],
    score_matrix_by_family: Mapping[str, np.ndarray],
    source_indices_by_family: Mapping[str, Sequence[int]],
    minimum_score: float,
    minimum_margin: float,
    minimum_dialogues: int,
) -> dict[str, Any]:
    """Bind weak semantic assignments to card-level source support."""

    spec_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in specs:
        spec_by_family[str(row["strategy_family"])].append(dict(row))
    for family in spec_by_family:
        spec_by_family[family].sort(key=lambda row: str(row["submove_id"]))

    mapping_rows: list[dict[str, Any]] = []
    support_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_top_scores: list[float] = []
    all_margins: list[float] = []
    for family in EXPECTED_FAMILIES:
        family_specs = spec_by_family[family]
        scores = np.asarray(score_matrix_by_family[family])
        source_indices = list(source_indices_by_family[family])
        if scores.shape != (len(source_indices), len(family_specs)):
            raise ValueError(f"{family}: assignment score matrix shape mismatch")
        order = np.argsort(-scores, axis=1)
        for local_index, source_index in enumerate(source_indices):
            top = int(order[local_index, 0])
            second = int(order[local_index, 1])
            top_score = float(scores[local_index, top])
            margin = top_score - float(scores[local_index, second])
            spec = family_specs[top]
            confident = top_score >= minimum_score and margin >= minimum_margin
            source = dict(source_rows[source_index])
            row = {
                "protocol": "pm-v1.5-strategy-bank-v3-weak-source-mapping-v1",
                "strategy_id": str(source["strategy_id"]),
                "source_dialogue_id": str(source["source_dialogue_id"]),
                "source_turn_index": int(source["source_turn_index"]),
                "strategy_family": family,
                "assigned_submove_id": str(spec["submove_id"]),
                "assigned_card_id": card_ids[str(spec["submove_id"])],
                "top_cosine_score": round(top_score, 8),
                "top_vs_second_margin": round(margin, 8),
                "assignment_confident": confident,
                "assignment_role": (
                    "weak outcome-blind source-support evidence; not a gold "
                    "submove label or PM target"
                ),
                "raw_example_response_exposed_to_generator": False,
                "problem_type": str(source["problem_type"]),
                "emotion_type": str(source["emotion_type"]),
                "experience_type": str(source["experience_type"]),
            }
            mapping_rows.append(row)
            support_rows[str(spec["submove_id"])].append(row)
            all_top_scores.append(top_score)
            all_margins.append(margin)

    cards: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda row: str(row["submove_id"])):
        submove_id = str(spec["submove_id"])
        assigned = support_rows[submove_id]
        confident = [row for row in assigned if row["assignment_confident"]]
        confident_dialogues = {
            str(row["source_dialogue_id"]) for row in confident
        }
        all_dialogues = {str(row["source_dialogue_id"]) for row in assigned}
        cards.append(
            {
                "protocol": V3_PROTOCOL,
                "card_id": card_ids[submove_id],
                **dict(spec),
                "content_scope": "technique_only",
                "retrieval_text": " ".join(
                    (
                        str(spec["support_move"]),
                        str(spec["when_to_use"]),
                        "Compatible modes:",
                        ", ".join(spec["compatible_support_modes"]),
                        "Goals:",
                        ", ".join(spec["goal_types"]),
                    )
                ),
                "prompt_guidance": " ".join(
                    (
                        str(spec["support_move"]),
                        str(spec["when_not_to_use"]),
                        "Use only information visible in the current prompt.",
                    )
                ),
                "source_support": {
                    "weak_assigned_rows": len(assigned),
                    "weak_assigned_dialogues": len(all_dialogues),
                    "confident_weak_assigned_rows": len(confident),
                    "confident_weak_assigned_dialogues": len(
                        confident_dialogues
                    ),
                    "minimum_dialogues_required": minimum_dialogues,
                    "provisional_source_support_pass": (
                        len(confident_dialogues) >= minimum_dialogues
                    ),
                    "assignment_is_gold": False,
                    "human_source_sample_review_required": True,
                    "raw_examples_exposed_to_generator": False,
                },
                "quality_status": (
                    "CORE_TAXONOMY_CANDIDATE_PENDING_DISTINCTNESS_"
                    "SOURCE_SAMPLE_AND_HUMAN_REVIEW"
                ),
                "eligible_for_formal_rs": False,
            }
        )

    def quantiles(values: Sequence[float]) -> dict[str, float]:
        return {
            key: round(float(value), 6)
            for key, value in zip(
                ("p05", "p25", "p50", "p75", "p95"),
                np.quantile(np.asarray(values), [0.05, 0.25, 0.5, 0.75, 0.95]),
                strict=True,
            )
        }

    return {
        "cards": cards,
        "mapping_rows": mapping_rows,
        "assignment_summary": {
            "source_rows": len(mapping_rows),
            "confident_source_rows": sum(
                bool(row["assignment_confident"]) for row in mapping_rows
            ),
            "minimum_score": minimum_score,
            "minimum_margin": minimum_margin,
            "top_score_quantiles": quantiles(all_top_scores),
            "top_vs_second_margin_quantiles": quantiles(all_margins),
            "cards_passing_provisional_source_support": sum(
                bool(
                    row["source_support"][
                        "provisional_source_support_pass"
                    ]
                )
                for row in cards
            ),
        },
    }


def card_distinctness_report(
    *,
    specs: Sequence[Mapping[str, Any]],
    embeddings_by_family: Mapping[str, np.ndarray],
    warning_cosine: float,
) -> dict[str, Any]:
    spec_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in specs:
        spec_by_family[str(row["strategy_family"])].append(dict(row))
    warnings: list[dict[str, Any]] = []
    maxima: dict[str, float] = {}
    for family in EXPECTED_FAMILIES:
        rows = sorted(
            spec_by_family[family], key=lambda row: str(row["submove_id"])
        )
        matrix = np.asarray(embeddings_by_family[family])
        similarities = matrix @ matrix.T
        maximum = -1.0
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                score = float(similarities[left, right])
                maximum = max(maximum, score)
                if score >= warning_cosine:
                    warnings.append(
                        {
                            "strategy_family": family,
                            "left_submove_id": str(rows[left]["submove_id"]),
                            "right_submove_id": str(rows[right]["submove_id"]),
                            "cosine": round(score, 8),
                            "action": "human_merge_or_justify",
                        }
                    )
        maxima[family] = round(maximum, 8)
    return {
        "warning_cosine": warning_cosine,
        "within_family_maximum_pair_cosine": maxima,
        "warning_pair_count": len(warnings),
        "warning_pairs": warnings,
        "semantic_similarity_is_automatic_deletion_rule": False,
    }


def taxonomy_sha256(specs: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(canonical_json([dict(row) for row in specs]))
