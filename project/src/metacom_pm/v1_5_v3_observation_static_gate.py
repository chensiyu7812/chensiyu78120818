"""Pre-human static gates for realized orthogonal Observation candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io import canonical_json, sha256_text
from .v1_5_v3_observation_orthogonal import FACTORS, LEARNED_FACTORS


PROTOCOL = "pm-v1.5-v3-observation-orthogonal-pre-human-static-gate-v1"
SEEDS = (17, 29, 43, 71, 113)


def _candidate_text(row: Mapping[str, Any]) -> str:
    return str(row["exact_rank1_candidate"]["candidate_text"])


def _length_features(row: Mapping[str, Any]) -> dict[str, float]:
    visible_words = sum(
        len(str(turn["content"]).split()) for turn in row["visible_dialogue"]
    )
    return {
        "current_words": float(len(str(row["current_user_text"]).split())),
        "visible_words": float(visible_words),
        "candidate_words": float(len(_candidate_text(row).split())),
        "total_words": float(
            visible_words
            + len(str(row["current_user_text"]).split())
            + len(_candidate_text(row).split())
        ),
    }


def _feature_sets(
    *, blueprint: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    lengths = _length_features(candidate)
    component = str(blueprint["target_component"])
    subtype = str(candidate["exact_rank1_candidate"]["compiler_subtype_hint"])
    base = {
        "component": component,
        "topic_family": str(blueprint["topic_family"]),
        "history_scale": str(blueprint["history_scale"]),
        "prefix_family": str(blueprint["prefix_family"]),
        "length_direction": str(blueprint["length_direction"]),
        "candidate_subtype": subtype,
        "candidate_function": str(blueprint["required_candidate_function"]),
        **lengths,
    }
    retrieval = candidate["private_retrieval_audit_not_model_input"]
    if component == "RS":
        retrieval_features = {
            "actual_strategy_family": str(retrieval["actual_strategy_family"]),
            "source_catalog_count": float(retrieval["source_catalog_count"]),
        }
    else:
        descriptor = retrieval["descriptor"]
        retrieval_features = {
            "source_catalog_count": float(retrieval["source_catalog_count"]),
            "candidate_age_sessions": float(
                candidate["exact_rank1_candidate"]["candidate_age_sessions"]
            ),
            "top1_lexical_relevance": float(descriptor["top1_lexical_relevance"]),
            "top1_top2_lexical_margin": float(
                descriptor["top1_top2_lexical_margin"]
            ),
            "incremental_injected_tokens": float(
                descriptor["incremental_injected_tokens"]
            ),
        }
    return {
        "frozen_nuisance_gate": base,
        "length_only": lengths,
        "topic_only": {
            "component": component,
            "topic_family": str(blueprint["topic_family"]),
        },
        "retrieval_descriptor_diagnostic": retrieval_features,
    }


def _oof_balanced_accuracy(
    features: Sequence[dict[str, Any]],
    labels: Sequence[int],
    groups: Sequence[str],
) -> dict[str, Any]:
    seed_scores: list[float] = []
    labels_array = np.asarray(labels, dtype=int)
    groups_array = np.asarray(groups)
    for seed in SEEDS:
        predictions = np.zeros(len(labels_array), dtype=int)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for train, test in splitter.split(features, labels_array, groups_array):
            model = Pipeline(
                [
                    ("vectorizer", DictVectorizer(sparse=False)),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=0.3,
                            class_weight="balanced",
                            solver="liblinear",
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            model.fit([features[index] for index in train], labels_array[train])
            predictions[test] = model.predict([features[index] for index in test])
        seed_scores.append(float(balanced_accuracy_score(labels_array, predictions)))
    return {
        "balanced_accuracy_mean": mean(seed_scores),
        "balanced_accuracy_sd": pstdev(seed_scores),
        "per_seed": dict(zip((str(seed) for seed in SEEDS), seed_scores, strict=True)),
    }


def _counterfactual_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for component in ("MP", "MS", "ME", "RS"):
        subset = [row for row in rows if row["blueprint"]["target_component"] == component]
        by_vector = {
            (
                int(row["blueprint"].get("private_factorial_repetition", 0)),
                *tuple(
                    bool(row["blueprint"]["private_factor_plan"][factor]["private_target"])
                    for factor in FACTORS
                ),
            ): row
            for row in subset
        }
        coverage[component] = {}
        for factor in LEARNED_FACTORS[component]:
            position = 1 + FACTORS.index(factor)
            edges: list[dict[str, bool]] = []
            for vector, negative in by_vector.items():
                if vector[position]:
                    continue
                positive_vector = list(vector)
                positive_vector[position] = True
                positive = by_vector.get(tuple(positive_vector))
                if positive is None:
                    continue
                negative_length = len(str(negative["candidate"]["current_user_text"]).split())
                positive_length = len(str(positive["candidate"]["current_user_text"]).split())
                edges.append(
                    {
                        "cross_topic": negative["blueprint"]["topic_family"]
                        != positive["blueprint"]["topic_family"],
                        "cross_scale": negative["blueprint"]["history_scale"]
                        != positive["blueprint"]["history_scale"],
                        "positive_longer": positive_length > negative_length,
                        "negative_longer": negative_length > positive_length,
                    }
                )
            coverage[component][factor] = {
                "edges": len(edges),
                "cross_topic_edges": sum(edge["cross_topic"] for edge in edges),
                "cross_scale_edges": sum(edge["cross_scale"] for edge in edges),
                "positive_longer_edges": sum(edge["positive_longer"] for edge in edges),
                "negative_longer_edges": sum(edge["negative_longer"] for edge in edges),
            }
    return coverage


def audit_pre_human_static_gate(
    *,
    blueprint_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    materialization_report: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    blueprint = {str(row["blueprint_row_id"]): dict(row) for row in blueprint_rows}
    candidates = {str(row["state_id"]): dict(row) for row in candidate_rows}
    if set(blueprint) != set(candidates):
        failures.append("blueprint_candidate_id_mismatch")
    if materialization_report.get("status") != "PASS":
        failures.append("exact_rank1_materialization_not_pass")

    decision_hashes = [
        sha256_text(
            canonical_json(
                {
                    "visible_dialogue": row["visible_dialogue"],
                    "current_user_text": row["current_user_text"],
                    "component": row["target_component_private_not_model_input"],
                    "candidate_text": _candidate_text(row),
                }
            )
        )
        for row in candidate_rows
    ]
    duplicate_surfaces = len(decision_hashes) - len(set(decision_hashes))
    if duplicate_surfaces:
        failures.append("duplicate_complete_decision_surface")

    joined = [
        {"blueprint": blueprint[state_id], "candidate": candidates[state_id]}
        for state_id in sorted(blueprint)
        if blueprint[state_id]["track"] == "FACTOR_FIT"
    ]
    probes: dict[str, Any] = {}
    for factor in FACTORS:
        applicable = [
            row
            for row in joined
            if row["blueprint"]["private_factor_plan"][factor]["is_learning_target"]
        ]
        labels = [
            int(row["blueprint"]["private_factor_plan"][factor]["private_target"])
            for row in applicable
        ]
        groups = [str(row["blueprint"]["counterfactual_group_id"]) for row in applicable]
        probes[factor] = {}
        for feature_set in (
            "frozen_nuisance_gate",
            "length_only",
            "topic_only",
            "retrieval_descriptor_diagnostic",
        ):
            features = [
                _feature_sets(
                    blueprint=row["blueprint"], candidate=row["candidate"]
                )[feature_set]
                for row in applicable
            ]
            probes[factor][feature_set] = _oof_balanced_accuracy(
                features, labels, groups
            )
        if probes[factor]["frozen_nuisance_gate"]["balanced_accuracy_mean"] >= 0.65:
            failures.append(f"nuisance_probe_failed_{factor}")
        if probes[factor]["length_only"]["balanced_accuracy_mean"] >= 0.65:
            failures.append(f"length_probe_failed_{factor}")

    coverage = _counterfactual_coverage(joined)
    for component, factors in coverage.items():
        for factor, values in factors.items():
            if not values["cross_topic_edges"]:
                failures.append(f"missing_cross_topic_edge_{component}_{factor}")
            if not values["cross_scale_edges"]:
                failures.append(f"missing_cross_scale_edge_{component}_{factor}")
            if not values["positive_longer_edges"] or not values["negative_longer_edges"]:
                failures.append(f"missing_both_length_directions_{component}_{factor}")

    return {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "scope": "PRE_HUMAN_REALIZED_TEXT_AND_EXACT_RANK1_STATIC_GATE",
        "states": len(candidate_rows),
        "factor_fit_states": len(joined),
        "unique_complete_decision_surfaces": len(set(decision_hashes)),
        "duplicate_complete_decision_surfaces": duplicate_surfaces,
        "exact_rank1_materialization_status": materialization_report.get("status"),
        "nuisance_probes": probes,
        "counterfactual_coverage": coverage,
        "failures": failures,
        "api_calls": 0,
        "human_labels_read": 0,
        "external_lockbox_read": False,
        "responses_generated": 0,
        "construction_intent_is_gold": False,
        "review_packet_allowed": not failures,
        "next_step_if_pass": "FREEZE_ONE_96_PRIMARY_PLUS_24_OVERLAP_REVIEW_PACKET",
    }
