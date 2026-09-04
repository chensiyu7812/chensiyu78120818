#!/usr/bin/env python3
"""Build the formal content-disjoint Evo-style synthetic memory backend."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from metacom_pm.config import load_config
from metacom_pm.contracts import (
    ACTION_MEMORY_MAP,
    MemoryBackendRecord,
    MemorySource,
    StrategyMode,
    canonical_action_id,
)
from metacom_pm.evoemo import (
    EVO_MEMORY_PROTOCOL,
    evo_memory_builder_contract_hash,
)
from metacom_pm.io import (
    canonical_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v1_5_semantic import (
    FrozenTransformerSemanticEncoder,
    require_semantic_runtime_contract,
    semantic_encoder_spec_from_config,
    semantic_vector_similarity,
)
from metacom_pm.pm_v1_5_step0 import MEMORY_ITEM_FEATURE_TOKEN_CLIP
from metacom_pm.pm_v2_contracts import (
    ObservableSourceSummary,
    PMV2State,
    Step0MemoryObservation,
    Step0Observation,
)
from metacom_pm.pm_v2_data import (
    build_deployable_catalog_statistics,
    load_bundles,
    load_states,
    state_to_v1_runtime,
)
from metacom_pm.retrieval import (
    DEFAULT_MEMORY_TOP_K,
    MemoryRetriever,
    context_query,
)
from metacom_pm.text import estimate_tokens
from metacom_pm.v1_5_candidate_discovery import (
    CANDIDATE_DISCOVERY_PROTOCOL,
    discover_memory_candidates,
)
from metacom_pm.v1_5_memory_transport import (
    TRANSPORT_ADAPTER_PROTOCOL,
    compile_synthetic_longitudinal_catalogs,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-evo-style-synthetic-memory-backend-v1"


def _profile(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "n": len(values),
        "min": min(values),
        "median": median,
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _expected_tokens(catalog: ObservableSourceSummary, source: MemorySource) -> int:
    if not catalog.available:
        return 0
    retrieved = min(catalog.count, int(DEFAULT_MEMORY_TOP_K[source]))
    mean_tokens = float(catalog.estimated_tokens) / float(catalog.count)
    return int(
        round(
            min(
                float(catalog.estimated_tokens),
                mean_tokens * retrieved,
                float(
                    DEFAULT_MEMORY_TOP_K[source]
                    * MEMORY_ITEM_FEATURE_TOKEN_CLIP
                ),
            )
        )
    )


def _centroid(
    vectors_by_text: dict[str, np.ndarray],
    texts: list[str],
    *,
    decimals: int,
) -> tuple[float, ...]:
    matrix = np.stack([vectors_by_text[text] for text in texts])
    centroid = np.mean(matrix, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0.0 or not np.all(np.isfinite(centroid)):
        raise RuntimeError("invalid memory catalog centroid")
    centroid = centroid / norm
    return tuple(round(float(value), decimals) for value in centroid)


def build_backend(
    *,
    bundles_path: Path,
    states_path: Path,
    pm_config_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    bundles = load_bundles(bundles_path)
    old_states = load_states(states_path)
    state_by_case = {
        (state.user_id, state.surface_form_id): state for state in old_states
    }
    if len(state_by_case) != len(old_states):
        raise RuntimeError("source states are not unique by user/surface")

    catalogs = compile_synthetic_longitudinal_catalogs(bundles)
    if len(catalogs) != 416:
        raise RuntimeError(
            f"expected 416 noninitial synthetic states, observed {len(catalogs)}"
        )
    missing = [
        (catalog.user_id, catalog.case.surface_form_id)
        for catalog in catalogs
        if (catalog.user_id, catalog.case.surface_form_id) not in state_by_case
    ]
    if missing:
        raise RuntimeError(f"catalogs lack source states: {missing[:3]}")

    config = load_config(pm_config_path)
    encoder = FrozenTransformerSemanticEncoder.load(
        semantic_encoder_spec_from_config(config)
    )
    semantic_runtime = require_semantic_runtime_contract(config, encoder)

    unique_texts = sorted(
        {item.text for catalog in catalogs for item in catalog.items}
    )
    vectors_by_text: dict[str, np.ndarray] = {}
    batch_size = 64
    for start in range(0, len(unique_texts), batch_size):
        texts = unique_texts[start : start + batch_size]
        matrix = encoder.encode(texts)
        vectors_by_text.update(
            {text: matrix[index] for index, text in enumerate(texts)}
        )

    pm_states: list[PMV2State] = []
    backends: list[MemoryBackendRecord] = []
    lineage_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    catalog_counts: dict[str, list[float]] = defaultdict(list)
    catalog_tokens: dict[str, list[float]] = defaultdict(list)
    selected_counts: dict[str, list[float]] = defaultdict(list)
    retriever = MemoryRetriever(
        minimum_score_by_source={
            source: float((config.get("retrieval") or {})["memory_min_score"])
            for source in MemorySource
        }
    )

    for catalog in catalogs:
        old = state_by_case[(catalog.user_id, catalog.case.surface_form_id)]
        dimension = int(encoder.spec.output_dimension)
        if len(old.text_embedding) != 2 * dimension:
            raise RuntimeError(
                f"source state semantic dimension drift: {old.state_id}"
            )
        query_vector = old.text_embedding[dimension:]
        inventory: dict[MemorySource, ObservableSourceSummary] = {}
        for source in MemorySource:
            items = [item for item in catalog.items if item.source is source]
            texts = [item.text for item in items]
            statistics = build_deployable_catalog_statistics(
                texts=texts,
                created_sessions=[item.created_session for item in items],
                session_index=catalog.chronological_session_index,
            )
            centroid = _centroid(
                vectors_by_text,
                texts,
                decimals=int(encoder.spec.output_round_decimals),
            )
            similarity = semantic_vector_similarity(query_vector, centroid)
            inventory[source] = ObservableSourceSummary(
                available=bool(statistics["available"]),
                count=int(statistics["count"]),
                min_age_sessions=statistics["min_age_sessions"],
                median_age_sessions=statistics["median_age_sessions"],
                max_age_sessions=statistics["max_age_sessions"],
                estimated_tokens=int(statistics["estimated_tokens"]),
                query_similarity_mean=similarity,
                representation_valid=True,
                catalog_embedding=list(statistics["catalog_fingerprint"]),
            )
            catalog_counts[source.value].append(float(len(items)))
            catalog_tokens[source.value].append(
                float(statistics["estimated_tokens"])
            )

        available = {
            source for source, summary in inventory.items() if summary.available
        }
        allowed_actions = sorted(
            {
                canonical_action_id(sources, strategy)
                for sources in ACTION_MEMORY_MAP.values()
                if sources <= available
                for strategy in StrategyMode
            }
        )
        if old.step0_observation is None:
            raise RuntimeError(f"source state lacks Step-0: {old.state_id}")
        memory_observations = {
            source: Step0MemoryObservation(
                available=summary.available,
                count=summary.count,
                min_age_sessions=summary.min_age_sessions,
                median_age_sessions=summary.median_age_sessions,
                max_age_sessions=summary.max_age_sessions,
                expected_retrieval_tokens=_expected_tokens(summary, source),
                representation_valid=summary.representation_valid,
                query_to_source_similarity=summary.query_similarity_mean,
            )
            for source, summary in inventory.items()
        }
        new_state_id = "state_" + stable_hex(
            PROTOCOL, old.state_id, n=24
        )
        new_card_id = "card_" + stable_hex(
            PROTOCOL, old.card_id, n=24
        )
        transport_lineage = {
            "memory_transport_protocol": PROTOCOL,
            "transport_adapter_protocol": TRANSPORT_ADAPTER_PROTOCOL,
            "memory_builder_protocol": EVO_MEMORY_PROTOCOL,
            "memory_builder_contract_sha256": (
                evo_memory_builder_contract_hash()
            ),
            "source_state_id": old.state_id,
            "source_card_id": old.card_id,
            "source_session_index": old.session_index,
            "external_file_read": False,
            "external_content_used": False,
            "external_outcome_used": False,
            "generated_memory_labels_used": False,
        }
        provenance = {
            "backend_record_id": new_card_id,
            "runtime_provenance_sha256": sha256_text(
                canonical_json(transport_lineage)
            ),
        }
        if "data_generation_sha256" in old.provenance:
            provenance["data_generation_sha256"] = old.provenance[
                "data_generation_sha256"
            ]
        if "semantic_observation" in old.provenance:
            provenance["semantic_observation"] = old.provenance[
                "semantic_observation"
            ]
        state = PMV2State(
            state_id=new_state_id,
            card_id=new_card_id,
            user_id=old.user_id,
            split=old.split,
            semantic_family=old.semantic_family,
            surface_form_id=old.surface_form_id,
            current_user_text=old.current_user_text,
            current_session_history=old.current_session_history,
            current_session_summary=old.current_session_summary,
            session_index=catalog.chronological_session_index,
            inventory=inventory,
            strategy_catalog_count=old.strategy_catalog_count,
            strategy_estimated_tokens=old.strategy_estimated_tokens,
            step0_observation=Step0Observation(
                memory_sources=memory_observations,
                strategy=old.step0_observation.strategy,
            ),
            text_embedding=old.text_embedding,
            allowed_actions=allowed_actions,
            provenance=provenance,
        )
        backend = MemoryBackendRecord(
            card_id=new_card_id, items=list(catalog.items)
        )
        query = context_query(
            state.current_user_text,
            [
                turn.model_dump(mode="json")
                for turn in state.current_session_history
            ],
            state.current_session_summary,
        )
        discoveries = discover_memory_candidates(
            query=query,
            items=backend.items,
            retriever=retriever,
            session_index=state.session_index,
        )
        for source in MemorySource:
            source_items = [
                item for item in backend.items if item.source is source
            ]
            discovery = discoveries[source]
            selected = list(discovery.selected_items)
            selected_counts[source.value].append(float(len(selected)))
            selected_texts = [item.text for item in selected]
            candidate_similarity = (
                semantic_vector_similarity(
                    query_vector,
                    _centroid(
                        vectors_by_text,
                        selected_texts,
                        decimals=int(encoder.spec.output_round_decimals),
                    ),
                )
                if selected_texts
                else 0.0
            )
            descriptor = {
                **discovery.descriptor,
                "candidate_state_bge_similarity_optional": (
                    candidate_similarity
                ),
                "source_catalog_bge_similarity_optional": (
                    state.inventory[source].query_similarity_mean
                ),
            }
            retrieval_rows.append(
                {
                    "protocol": PROTOCOL,
                    "state_id": new_state_id,
                    "user_id": state.user_id,
                    "split": state.split.value,
                    "semantic_family": state.semantic_family,
                    "source": source.value,
                    "catalog_count": len(source_items),
                    "selected_count": len(selected),
                    "top1_lexical_score": descriptor[
                        "top1_lexical_relevance"
                    ],
                    "top2_lexical_score": (
                        descriptor["top1_lexical_relevance"]
                        - descriptor["top1_top2_lexical_margin"]
                        if len(source_items) >= 2
                        else None
                    ),
                    "top1_top2_margin": descriptor[
                        "top1_top2_lexical_margin"
                    ],
                    "selected_memory_ids": [
                        item.memory_id for item in selected
                    ],
                    "candidate_descriptor": descriptor,
                }
            )
            descriptor_rows.append(
                {
                    "protocol": PROTOCOL,
                    "state_id": new_state_id,
                    "user_id": state.user_id,
                    "split": state.split.value,
                    "semantic_family": state.semantic_family,
                    "candidate_descriptor": descriptor,
                    "outcome_label": (
                        "UNKNOWN_BEFORE_PAIRED_HUMAN_REVIEW"
                    ),
                }
            )
        pm_states.append(state)
        backends.append(backend)
        lineage_rows.append(
            {
                "protocol": PROTOCOL,
                "state_id": new_state_id,
                "card_id": new_card_id,
                "user_id": state.user_id,
                "split": state.split.value,
                "surface_form_id": state.surface_form_id,
                "semantic_family": state.semantic_family,
                "chronological_session_index": state.session_index,
                "prior_case_ids": list(catalog.prior_case_ids),
                "source_state_id": old.state_id,
                "source_card_id": old.card_id,
                "transport_lineage": transport_lineage,
            }
        )

    split_users: dict[str, set[str]] = defaultdict(set)
    for state in pm_states:
        split_users[state.split.value].add(state.user_id)
    split_names = sorted(split_users)
    split_overlap = sum(
        len(split_users[left] & split_users[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    )
    memory_text_splits: dict[str, set[str]] = defaultdict(set)
    state_by_card = {state.card_id: state for state in pm_states}
    for backend in backends:
        split = state_by_card[backend.card_id].split.value
        for item in backend.items:
            memory_text_splits[item.text].add(split)
    cross_split_texts = {
        text: sorted(splits)
        for text, splits in memory_text_splits.items()
        if len(splits) > 1
    }
    source_selection_present = {
        source.value: sum(
            row["source"] == source.value and row["selected_count"] > 0
            for row in retrieval_rows
        )
        for source in MemorySource
    }
    report = {
        "protocol": PROTOCOL,
        "status": "READY_FOR_OUTCOME_BLIND_192_MEMORY_CONTRAST_BLUEPRINT",
        "api_calls_made": 0,
        "human_labels_read": False,
        "external_file_read": False,
        "external_content_used": False,
        "external_outcome_used": False,
        "generated_memory_labels_used": False,
        "states": len(pm_states),
        "users": len({state.user_id for state in pm_states}),
        "split_counts": dict(
            sorted(Counter(state.split.value for state in pm_states).items())
        ),
        "memory_builder": {
            "protocol": EVO_MEMORY_PROTOCOL,
            "contract_sha256": evo_memory_builder_contract_hash(),
            "transport_adapter_protocol": TRANSPORT_ADAPTER_PROTOCOL,
        },
        "semantic_encoder": {
            "model_id": encoder.spec.model_id,
            "spec_sha256": encoder.binding.spec_sha256,
            "runtime_contract_sha256": semantic_runtime[
                "contract_sha256"
            ],
        },
        "retrieval": {
            "implementation": "metacom_pm.retrieval.MemoryRetriever",
            "score": "lexical_score_not_BGE_cosine",
            "memory_min_score": float(
                (config.get("retrieval") or {})["memory_min_score"]
            ),
            "top_k_by_source": {
                source.value: int(DEFAULT_MEMORY_TOP_K[source])
                for source in MemorySource
            },
            "states_with_nonempty_selection": source_selection_present,
            "selected_count_profiles": {
                source.value: _profile(selected_counts[source.value])
                for source in MemorySource
            },
        },
        "candidate_descriptor_contract": {
            "protocol": CANDIDATE_DISCOVERY_PROTOCOL,
            "decision_timing": (
                "post-candidate-discovery_pre-injection_pre-generation"
            ),
            "primary_transparent_fields": [
                "source",
                "candidate_present",
                "retrieved_count",
                "incremental_injected_tokens",
                "minimum_age_sessions",
                "median_age_sessions",
                "maximum_age_sessions",
                "top1_lexical_relevance",
                "top1_top2_lexical_margin",
                "top_k_capacity_fraction",
                "other-component background bits added at contrast assembly",
            ],
            "optional_bge_challenger_fields": [
                "candidate_state_bge_similarity_optional",
                "source_catalog_bge_similarity_optional",
            ],
            "diagnostic_only_not_primary_pm_feature": [
                "full catalog count",
                "full catalog tokens",
                "dataset or split identity",
            ],
            "forbidden": [
                "raw memory text",
                "memory IDs",
                "generated response",
                "human or judge outcome",
                "old needed_memory_sources",
                "external dataset identity",
            ],
        },
        "catalog_profiles": {
            source.value: {
                "count": _profile(catalog_counts[source.value]),
                "estimated_tokens": _profile(
                    catalog_tokens[source.value]
                ),
            }
            for source in MemorySource
        },
        "leakage_and_integrity": {
            "strictly_prior_sessions_only": all(
                item.created_session < state_by_card[backend.card_id].session_index
                for backend in backends
                for item in backend.items
            ),
            "user_split_overlap": split_overlap,
            "cross_split_exact_memory_text_count": len(cross_split_texts),
            "cross_split_exact_memory_text_examples_sha256": [
                sha256_text(text)
                for text in sorted(cross_split_texts)[:20]
            ],
            "cross_split_exact_text_is_not_outcome_leakage": True,
            "note": (
                "Repeated synthetic topic/style templates are a distribution "
                "limitation to audit as a shortcut; raw text and hashes are "
                "not PM features."
            ),
        },
        "checks": {
            "exactly_416_noninitial_states": len(pm_states) == 416,
            "all_three_sources_available": all(
                all(state.inventory[source].available for source in MemorySource)
                for state in pm_states
            ),
            "all_states_have_16_legal_actions": all(
                len(state.allowed_actions) == 16 for state in pm_states
            ),
            "strictly_prior_sessions_only": all(
                item.created_session < state_by_card[backend.card_id].session_index
                for backend in backends
                for item in backend.items
            ),
            "no_user_overlap_across_splits": split_overlap == 0,
            "all_sources_offer_retrieved_candidates_for_at_least_192_states": all(
                source_selection_present[source.value] >= 192
                for source in MemorySource
            ),
            "no_external_or_outcome_inputs": True,
            "same_evo_memory_builder": True,
        },
        "limitations": [
            "The nine sessions per user are reconstructed from previously independent synthetic cases, not naturally observed longitudinal sessions.",
            "Some synthetic profile and topic surfaces repeat across users; grouped user splits and shortcut audits remain mandatory.",
            "MemoryRetriever currently ranks by lexical overlap. BGE supplies pre-retrieval source-level semantic observations but does not rank memory items.",
        ],
        "inputs": {
            "bundles": str(bundles_path.relative_to(ROOT)),
            "bundles_sha256": sha256_file(bundles_path),
            "source_states": str(states_path.relative_to(ROOT)),
            "source_states_sha256": sha256_file(states_path),
            "pm_config": str(pm_config_path.relative_to(ROOT)),
            "pm_config_sha256": sha256_file(pm_config_path),
        },
    }
    if not all(report["checks"].values()):
        report["status"] = "BLOCKED_BY_TRANSPORT_BACKEND_CHECK"

    out_dir.mkdir(parents=True, exist_ok=True)
    pm_states_path = out_dir / "pm_v2_states.jsonl"
    runtime_path = out_dir / "runtime_states.jsonl"
    backend_path = out_dir / "memory_backend.jsonl"
    lineage_path = out_dir / "state_lineage.jsonl"
    retrieval_path = out_dir / "retrieval_audit.jsonl"
    descriptor_path = out_dir / "candidate_descriptors.jsonl"
    write_jsonl(
        pm_states_path,
        [state.model_dump(mode="json") for state in pm_states],
    )
    write_jsonl(
        runtime_path,
        [
            state_to_v1_runtime(state).model_dump(mode="json")
            for state in pm_states
        ],
    )
    write_jsonl(
        backend_path,
        [backend.model_dump(mode="json") for backend in backends],
    )
    write_jsonl(lineage_path, lineage_rows)
    write_jsonl(retrieval_path, retrieval_rows)
    write_jsonl(descriptor_path, descriptor_rows)
    report["outputs"] = {
        path.name: sha256_file(path)
        for path in (
            pm_states_path,
            runtime_path,
            backend_path,
            lineage_path,
            retrieval_path,
            descriptor_path,
        )
    }
    report["artifact_sha256"] = sha256_text(canonical_json(report))
    write_json(out_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "pm_v2_bundles.jsonl",
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_formal_v8_19_2_runtime_projection_repair_candidate/"
        "pm_v2_states.jsonl",
    )
    parser.add_argument(
        "--pm-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate",
    )
    args = parser.parse_args()
    report = build_backend(
        bundles_path=args.bundles,
        states_path=args.states,
        pm_config_path=args.pm_config,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "status": report["status"],
                "states": report["states"],
                "split_counts": report["split_counts"],
                "retrieved_candidate_states": report["retrieval"][
                    "states_with_nonempty_selection"
                ],
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
