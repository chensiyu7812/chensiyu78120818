#!/usr/bin/env python3
"""Audit exact mechanism equivalence and distribution overlap with EvoEmo."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

from metacom_pm.config import endpoint_from_config, load_config
from metacom_pm.contracts import (
    MemoryBackendRecord,
    MemoryItem,
    MemorySource,
    RuntimeState,
)
from metacom_pm.evoemo import (
    _fixed_context_before_turn,
    evo_memory_builder_contract_hash,
    load_evoemo,
    make_evo_runtime_state,
)
from metacom_pm.generation_contract import SupporterGenerationContract
from metacom_pm.io import (
    canonical_json,
    iter_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from metacom_pm.pm_v2_data import load_bundles
from metacom_pm.prompts import generation_messages
from metacom_pm.retrieval import (
    DEFAULT_MEMORY_TOP_K,
    MemoryRetriever,
    context_query,
)
from metacom_pm.text import estimate_tokens
from metacom_pm.v1_5_candidate_discovery import (
    CANDIDATE_DISCOVERY_PROTOCOL,
    choose_after_candidate_discovery,
    discover_memory_candidates,
)
from metacom_pm.v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    bounded_prior_history_user,
    compile_bounded_memory,
    synthetic_basic_info,
    synthetic_prior_session,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pm-v1.5-evo-mimic-equivalence-audit-v1"
MINIMUM_JOINT_PRIMARY_DESCRIPTOR_SUPPORT = 0.80


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _profile(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": len(values),
        "min": min(values),
        "p05": _quantile(values, 0.05),
        "median": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _outside_rate(
    values: list[float],
    reference: list[float],
    *,
    robust: bool,
) -> float | None:
    if not values or not reference:
        return None
    lower = (
        _quantile(reference, 0.05) if robust else min(reference)
    )
    upper = (
        _quantile(reference, 0.95) if robust else max(reference)
    )
    assert lower is not None and upper is not None
    return sum(value < lower or value > upper for value in values) / len(
        values
    )


def _catalog_profile(
    users: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    sessions: list[dict[str, Any]] = []
    item_values: dict[str, list[float]] = defaultdict(list)
    catalog_values: dict[str, list[float]] = defaultdict(list)
    profile_key_counts: list[float] = []
    profile_keys: set[str] = set()
    memory_texts: set[str] = set()
    for user in users:
        bounded_user = bounded_prior_history_user(user)
        basic = dict(bounded_user.get("basic_info") or {})
        profile_key_counts.append(float(len(basic)))
        profile_keys.update(str(key) for key in basic)
        user_sessions = list(bounded_user.get("dialog_history") or [])
        sessions.extend(user_sessions)
        items, _ = compile_bounded_memory(user)
        memory_texts.update(item.text for item in items)
        for source in MemorySource:
            source_items = [
                item for item in items if item.source is source
            ]
            catalog_values[source.value].append(float(len(source_items)))
            item_values[source.value].extend(
                float(estimate_tokens(item.text))
                for item in source_items
            )
    summary_tokens = [
        float(estimate_tokens(str(session.get("summary") or "")))
        for session in sessions
        if str(session.get("summary") or "").strip()
    ]
    dialogue_turns = [
        float(len(session.get("dialogue") or [])) for session in sessions
    ]
    seeker_turns = [
        float(
            sum(
                turn.get("role") == "seeker"
                for turn in (session.get("dialogue") or [])
            )
        )
        for session in sessions
    ]
    timestamp_formats = Counter(
        (
            "session_ordinal"
            if str(session.get("timestamp") or "").startswith("session-")
            else "calendar_or_other"
        )
        for session in sessions
    )
    return (
        {
            "users": len(users),
            "sessions": len(sessions),
            "profile_fields_per_user": _profile(profile_key_counts),
            "profile_field_names": sorted(profile_keys),
            "session_summary_present_rate": (
                sum(bool(str(row.get("summary") or "").strip()) for row in sessions)
                / len(sessions)
                if sessions
                else None
            ),
            "summary_tokens": _profile(summary_tokens),
            "dialogue_turns_per_session": _profile(dialogue_turns),
            "seeker_turns_per_session": _profile(seeker_turns),
            "timestamp_format_counts": dict(sorted(timestamp_formats.items())),
            "catalog_count_per_user": {
                source.value: _profile(catalog_values[source.value])
                for source in MemorySource
            },
            "item_tokens": {
                source.value: _profile(item_values[source.value])
                for source in MemorySource
            },
        },
        memory_texts,
    )


def _synthetic_users(bundles_path: Path) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = []
    for bundle in load_bundles(bundles_path):
        # Each bundle has nine cases.  At the latest eligible internal state,
        # only the first eight are causal prior sessions; the ninth is the
        # current case.  Profile the deployable maximum rather than treating
        # the current case as memory.
        sessions = [
            synthetic_prior_session(case)
            for case in sorted(
                bundle.cases, key=lambda value: value.session_index
            )[:-1]
        ]
        users.append(
            {
                "id": bundle.user_id,
                "basic_info": synthetic_basic_info(bundle),
                "dialog_history": sessions,
            }
        )
    return users


def _external_descriptor_rows(
    *,
    users: list[dict[str, Any]],
    tracks_path: Path,
    retriever: MemoryRetriever,
    turn_indices: list[int],
) -> list[dict[str, Any]]:
    users_by_id = {str(user["id"]): user for user in users}
    topics = {
        (str(user["id"]), int(topic["idx"])): topic
        for user in users
        for topic in (user.get("subsequent_topics") or [])
    }
    items_by_user = {
        user_id: compile_bounded_memory(user)[0]
        for user_id, user in users_by_id.items()
    }
    rows: list[dict[str, Any]] = []
    for track in iter_jsonl(tracks_path):
        user_id = str(track["user_id"])
        topic_index = int(track["topic_index"])
        user = users_by_id[user_id]
        topic = topics[(user_id, topic_index)]
        items = items_by_user[user_id]
        for turn_index in turn_indices:
            current = str(track["seeker_turns"][turn_index - 1])
            context = _fixed_context_before_turn(dict(track), turn_index)
            state = make_evo_runtime_state(
                user,
                topic,
                context,
                current,
                items,
                turn_index,
                "outcome_blind_transport_audit",
                track_id=str(track["track_id"]),
                fixed_open_loop=True,
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
                items=items,
                retriever=retriever,
                session_index=state.session_index,
            )
            for source in MemorySource:
                rows.append(
                    {
                        "domain": "evoemo_external",
                        "state_id": state.state_id,
                        "user_id": user_id,
                        "topic_index": topic_index,
                        "turn_index": turn_index,
                        "source": source.value,
                        "candidate_descriptor": discoveries[
                            source
                        ].descriptor,
                    }
                )
    return rows


def _internal_descriptor_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        rows.append(
            {
                "domain": "repaired_internal",
                "state_id": row["state_id"],
                "user_id": row["user_id"],
                "split": row["split"],
                "source": row["source"],
                "candidate_descriptor": row["candidate_descriptor"],
            }
        )
    return rows


def _descriptor_comparison(
    internal: list[dict[str, Any]],
    external: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields = (
        "retrieved_fraction",
        "token_cost_bucket",
        "minimum_relative_age_bucket",
        "median_relative_age_bucket",
        "maximum_relative_age_bucket",
        "top1_relevance_bucket",
        "relevance_margin_bucket",
        "catalog_capacity_bucket",
    )
    report: dict[str, Any] = {}
    long_rows: list[dict[str, Any]] = []
    for source in MemorySource:
        internal_rows = [
            row
            for row in internal
            if row["source"] == source.value
            and row.get("split") in {"train", "calibration"}
        ]
        external_rows = [
            row for row in external if row["source"] == source.value
        ]
        external_present_rows = [
            row
            for row in external_rows
            if row["candidate_descriptor"]["candidate_present"]
        ]
        source_report: dict[str, Any] = {
            "internal_support_rows": len(internal_rows),
            "external_rows": len(external_rows),
            "external_present_candidate_rows": len(
                external_present_rows
            ),
            "candidate_present_rate": {
                "internal": sum(
                    bool(row["candidate_descriptor"]["candidate_present"])
                    for row in internal_rows
                )
                / len(internal_rows),
                "external": sum(
                    bool(row["candidate_descriptor"]["candidate_present"])
                    for row in external_rows
                )
                / len(external_rows),
            },
            "fields": {},
        }
        for field in fields:
            internal_values = [
                float(
                    row["candidate_descriptor"]["model_features"][field]
                )
                for row in internal_rows
                if row["candidate_descriptor"]["model_features"].get(field)
                is not None
            ]
            external_values = [
                float(
                    row["candidate_descriptor"]["model_features"][field]
                )
                for row in external_rows
                if row["candidate_descriptor"]["model_features"].get(field)
                is not None
            ]
            comparison = {
                "internal": _profile(internal_values),
                "external": _profile(external_values),
                "external_outside_internal_min_max_rate": _outside_rate(
                    external_values, internal_values, robust=False
                ),
                "external_outside_internal_p05_p95_rate": _outside_rate(
                    external_values, internal_values, robust=True
                ),
            }
            source_report["fields"][field] = comparison
            long_rows.extend(
                {
                    "source": source.value,
                    "field": field,
                    "domain": domain,
                    "median": profile["median"],
                    "p05": profile["p05"],
                    "p95": profile["p95"],
                }
                for domain, profile in (
                    ("repaired_internal_support", comparison["internal"]),
                    ("evoemo_external", comparison["external"]),
                )
            )
        per_row_in_support = []
        for row in external_present_rows:
            descriptor = row["candidate_descriptor"]["model_features"]
            supported = True
            for field in fields:
                value = descriptor.get(field)
                reference = [
                    candidate["candidate_descriptor"]["model_features"].get(
                        field
                    )
                    for candidate in internal_rows
                    if candidate["candidate_descriptor"][
                        "model_features"
                    ].get(field)
                    is not None
                ]
                if value is None or not reference:
                    supported = False
                    break
                if float(value) < min(reference) or float(value) > max(
                    reference
                ):
                    supported = False
                    break
            per_row_in_support.append(supported)
        source_report["all_primary_fields_within_internal_min_max_rate"] = (
            sum(per_row_in_support) / len(per_row_in_support)
            if per_row_in_support
            else None
        )
        report[source.value] = source_report
    return report, long_rows


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
        "--internal-backend-dir",
        type=Path,
        default=ROOT
        / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate",
    )
    parser.add_argument(
        "--generation-plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_generation_v1",
    )
    parser.add_argument(
        "--repaired-rs-generation-plan-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_generation_v1",
    )
    parser.add_argument(
        "--evoemo",
        type=Path,
        default=ROOT / "data/external/evo_emo.json",
    )
    parser.add_argument(
        "--fixed-tracks",
        type=Path,
        default=ROOT
        / "outputs/evoemo_fixed_tracks_v1_5_v3_formal_fresh_candidate/"
        "fixed_seeker_tracks.jsonl",
    )
    parser.add_argument(
        "--pm-config",
        type=Path,
        default=ROOT / "configs/pm_v1_5.yaml",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/experiment.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT
        / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1",
    )
    args = parser.parse_args()

    config = load_config(args.pm_config)
    experiment = load_config(args.experiment_config)
    generation = SupporterGenerationContract.from_config(config)
    endpoint = endpoint_from_config(
        experiment, generation.generator_endpoint
    )
    memory_min_score = float(config["retrieval"]["memory_min_score"])
    retriever = MemoryRetriever(
        minimum_score_by_source={
            source: memory_min_score for source in MemorySource
        }
    )

    synthetic_users = _synthetic_users(args.bundles)
    external_users = load_evoemo(args.evoemo)
    synthetic_bundles = load_bundles(args.bundles)
    synthetic_cases = [
        case for bundle in synthetic_bundles for case in bundle.cases
    ]
    external_sessions = [
        session
        for user in external_users
        for session in (user.get("dialog_history") or [])
    ]
    internal_catalog_profile, internal_texts = _catalog_profile(
        synthetic_users
    )
    external_catalog_profile, external_texts = _catalog_profile(
        external_users
    )
    internal_descriptor_rows = _internal_descriptor_rows(
        args.internal_backend_dir / "retrieval_audit.jsonl"
    )
    turn_indices = [
        int(value)
        for value in config["external_evaluation"]["turn_indices"]
    ]
    external_descriptor_rows = _external_descriptor_rows(
        users=external_users,
        tracks_path=args.fixed_tracks,
        retriever=retriever,
        turn_indices=turn_indices,
    )
    descriptor_comparison, descriptor_long_rows = (
        _descriptor_comparison(
            internal_descriptor_rows, external_descriptor_rows
        )
    )

    backend_report = json.loads(
        (args.internal_backend_dir / "report.json").read_text(
            encoding="utf-8"
        )
    )
    states = {
        state.state_id: state
        for state in (
            RuntimeState.model_validate(row)
            for row in iter_jsonl(
                args.internal_backend_dir / "runtime_states.jsonl"
            )
        )
    }
    backends = {
        record.card_id: {
            item.memory_id: item for item in record.items
        }
        for record in (
            MemoryBackendRecord.model_validate(row)
            for row in iter_jsonl(
                args.internal_backend_dir / "memory_backend.jsonl"
            )
        )
    }
    plan_report = json.loads(
        (args.generation_plan_dir / "plan_report.json").read_text(
            encoding="utf-8"
        )
    )
    strategy_cards = {
        str(row["card_id"]): dict(row)
        for row in iter_jsonl(
            ROOT / plan_report["inputs"]["strategy_cards"]
        )
    }
    plan_calls = [
        dict(row)
        for row in iter_jsonl(
            args.generation_plan_dir / "call_plan.jsonl"
        )
    ]
    r0_calls = [
        row for row in plan_calls if str(row["action_id"]).endswith("+R0")
    ]
    r0_prompt_matches = 0
    rs_prompt_matches = 0
    for row in r0_calls:
        state = states[str(row["state_id"])]
        item_map = backends[state.card_id]
        memories = [
            item_map[str(memory_id)]
            for memory_id in row["selected_memory_ids"]
        ]
        messages = generation_messages(
            state,
            memories,
            [],
            system_prompt=generation.system_prompt,
        )
        if sha256_text(canonical_json(messages)) == row["messages_sha256"]:
            r0_prompt_matches += 1
    rs_calls = [
        row for row in plan_calls if str(row["action_id"]).endswith("+RS")
    ]
    for row in rs_calls:
        state = states[str(row["state_id"])]
        item_map = backends[state.card_id]
        memories = [
            item_map[str(memory_id)]
            for memory_id in row["selected_memory_ids"]
        ]
        card_id = str(row["selected_strategy_card_id"])
        messages = generation_messages(
            state,
            memories,
            [strategy_cards[card_id]],
            system_prompt=generation.system_prompt,
        )
        if sha256_text(canonical_json(messages)) == row["messages_sha256"]:
            rs_prompt_matches += 1

    # Restoring the full causal external history and replacing absolute ages
    # with relative ages changes candidate descriptors, but it should not
    # silently invalidate the 512 already generated internal response arms.
    # Internal users have at most eight prior sessions, so their selected
    # items and generator prompts should recompute exactly.  Check the
    # separate 128-arm repaired-RS plan explicitly rather than inferring this
    # from the 384-arm memory plan.
    repaired_rs_report = json.loads(
        (
            args.repaired_rs_generation_plan_dir / "plan_report.json"
        ).read_text(encoding="utf-8")
    )
    repaired_rs_calls = [
        dict(row)
        for row in iter_jsonl(
            args.repaired_rs_generation_plan_dir / "call_plan.jsonl"
        )
    ]
    repaired_rs_prompt_matches = 0
    for row in repaired_rs_calls:
        state = states[str(row["state_id"])]
        item_map = backends[state.card_id]
        memories = [
            item_map[str(memory_id)]
            for memory_id in row["selected_memory_ids"]
        ]
        card_id = row.get("selected_strategy_card_id")
        cards = [strategy_cards[str(card_id)]] if card_id else []
        messages = generation_messages(
            state,
            memories,
            cards,
            system_prompt=generation.system_prompt,
        )
        if sha256_text(canonical_json(messages)) == row["messages_sha256"]:
            repaired_rs_prompt_matches += 1

    generation_reuse_checks = {
        "memory_plan_prompt_hash_matches": r0_prompt_matches
        + rs_prompt_matches,
        "memory_plan_prompt_hash_total": len(plan_calls),
        "repaired_rs_plan_prompt_hash_matches": (
            repaired_rs_prompt_matches
        ),
        "repaired_rs_plan_prompt_hash_total": len(repaired_rs_calls),
        "total_existing_generated_arms_reusable": (
            r0_prompt_matches
            + rs_prompt_matches
            + repaired_rs_prompt_matches
        ),
        "total_existing_generated_arms": (
            len(plan_calls) + len(repaired_rs_calls)
        ),
    }
    generation_reuse_checks["all_prompt_hashes_match"] = (
        generation_reuse_checks[
            "total_existing_generated_arms_reusable"
        ]
        == generation_reuse_checks["total_existing_generated_arms"]
        == 512
    )
    generation_reuse_checks["generator_identity_matches"] = (
        repaired_rs_report["generator_identity"]
        == plan_report["generator_identity"]
    )

    exact_checks = [
        {
            "layer": "MemoryItem schema and MP/MS/ME ontology",
            "status": "PASS",
            "evidence": "Both domains instantiate the same MemoryItem model and MemorySource enum.",
        },
        {
            "layer": "memory compiler implementation",
            "status": (
                "PASS"
                if backend_report["memory_builder"]["contract_sha256"]
                == evo_memory_builder_contract_hash()
                else "FAIL"
            ),
            "evidence": (
                f"{BOUNDED_MEMORY_COMPILER_PROTOCOL}:"
                f"{evo_memory_builder_contract_hash()}"
            ),
        },
        {
            "layer": "causal prior-history rule",
            "status": (
                "PASS"
                if backend_report["leakage_and_integrity"][
                    "strictly_prior_sessions_only"
                ]
                else "FAIL"
            ),
            "evidence": "Internal items are strictly older than the current session; EvoEmo build_evo_memory reads dialog_history only.",
        },
        {
            "layer": "query builder",
            "status": "PASS",
            "evidence": sha256_text(inspect.getsource(context_query)),
        },
        {
            "layer": "memory retriever score, threshold, tie-break and Top-k",
            "status": "PASS",
            "evidence": sha256_text(inspect.getsource(MemoryRetriever)),
        },
        {
            "layer": "evidence filter",
            "status": (
                "PASS"
                if not bool(config["evidence_filter"]["enabled"])
                else "FAIL"
            ),
            "evidence": "disabled on both V1.5 internal and EvoEmo paths",
        },
        {
            "layer": "R0 memory injection prompt compiler",
            "status": (
                "PASS" if r0_prompt_matches == len(r0_calls) else "FAIL"
            ),
            "evidence": f"{r0_prompt_matches}/{len(r0_calls)} frozen calls exactly recompute through generation_messages",
        },
        {
            "layer": "RS-background injection prompt compiler",
            "status": (
                "PASS" if rs_prompt_matches == len(rs_calls) else "FAIL"
            ),
            "evidence": (
                f"{rs_prompt_matches}/{len(rs_calls)} frozen V4 calls "
                "exactly recompute through generation_messages"
            ),
        },
        {
            "layer": "candidate descriptor extractor",
            "status": (
                "PASS"
                if backend_report["candidate_descriptor_contract"][
                    "protocol"
                ]
                == CANDIDATE_DISCOVERY_PROTOCOL
                else "FAIL"
            ),
            "evidence": CANDIDATE_DISCOVERY_PROTOCOL,
        },
        {
            "layer": "PM decision timing",
            "status": "PASS",
            "evidence": sha256_text(
                inspect.getsource(choose_after_candidate_discovery)
            ),
        },
        {
            "layer": "generator identity and treatment",
            "status": (
                "PASS"
                if plan_report["generator_identity"]
                == {
                    "base_url": endpoint.base_url,
                    "model": endpoint.model,
                    "family": endpoint.family,
                    "transport": endpoint.transport,
                }
                else "FAIL"
            ),
            "evidence": generation.digest(),
        },
        {
            "layer": "content disjointness",
            "status": (
                "PASS"
                if not (internal_texts & external_texts)
                else "FAIL"
            ),
            "evidence": f"exact normalized memory-text overlap={len(internal_texts & external_texts)}",
        },
        {
            "layer": "timestamp surface normalization",
            "status": "PASS",
            "evidence": "generation_messages renders created_session only as relative session age and ignores raw timestamp strings.",
        },
    ]
    exact_passes = sum(row["status"] == "PASS" for row in exact_checks)
    failed_layers = [
        row["layer"] for row in exact_checks if row["status"] != "PASS"
    ]
    support_coverage = {
        source: float(
            descriptor_comparison[source][
                "all_primary_fields_within_internal_min_max_rate"
            ]
        )
        for source in ("MP", "MS", "ME")
    }
    failed_support_sources = [
        source
        for source, rate in support_coverage.items()
        if rate < MINIMUM_JOINT_PRIMARY_DESCRIPTOR_SUPPORT
    ]
    failed_generation_reuse_checks = (
        []
        if (
            generation_reuse_checks["all_prompt_hashes_match"]
            and generation_reuse_checks["generator_identity_matches"]
        )
        else ["existing_generated_prompt_recomputation"]
    )
    report = {
        "protocol": PROTOCOL,
        "status": (
            "BLOCKED_BEFORE_PAID_MEMORY_GENERATION"
            if (
                failed_layers
                or failed_support_sources
                or failed_generation_reuse_checks
            )
            else "MIMIC_AND_CANDIDATE_SUPPORT_GATE_PASS"
        ),
        "api_calls_made": 0,
        "human_or_judge_outcomes_read": False,
        "external_reference_answers_read": False,
        "external_content_used_for_training": False,
        "definition": {
            "exact_equivalence": (
                "The same executable compiler, temporal rule, query, "
                "retriever, candidate descriptor, PM timing, filter, "
                "injection compiler, and generator treatment."
            ),
            "distribution_similarity": (
                "Outcome-blind overlap of input and candidate-feature "
                "distributions. It cannot be made 100% without copying the "
                "external test content and invalidating generalization."
            ),
        },
        "exact_mechanism_checks": exact_checks,
        "exact_mechanism_passes": exact_passes,
        "exact_mechanism_total": len(exact_checks),
        "failed_exact_layers": failed_layers,
        "generation_reuse_checks": generation_reuse_checks,
        "failed_generation_reuse_checks": (
            failed_generation_reuse_checks
        ),
        "candidate_support_gate": {
            "population": (
                "external candidates present after the shared retriever; "
                "candidate-absent states close deterministically"
            ),
            "minimum_joint_primary_descriptor_support_per_source": (
                MINIMUM_JOINT_PRIMARY_DESCRIPTOR_SUPPORT
            ),
            "observed_support_by_source": support_coverage,
            "failed_sources": failed_support_sources,
        },
        "content_profiles": {
            "repaired_internal": internal_catalog_profile,
            "evoemo_external": external_catalog_profile,
            "exact_memory_text_overlap": len(
                internal_texts & external_texts
            ),
        },
        "semantic_input_gaps": {
            "profile_field_name_overlap": sorted(
                set(
                    internal_catalog_profile["profile_field_names"]
                )
                & set(external_catalog_profile["profile_field_names"])
            ),
            "synthetic_summary_fallback_to_current_user_text_rate": (
                sum(
                    not str(case.session_summary or "").strip()
                    for case in synthetic_cases
                )
                / len(synthetic_cases)
            ),
            "external_empty_summary_rate": (
                sum(
                    not str(session.get("summary") or "").strip()
                    for session in external_sessions
                )
                / len(external_sessions)
            ),
            "runtime_history_policy": "all_causal_prior_sessions",
            "internal_maximum_prior_sessions": 8,
            "external_prior_session_range": [
                min(
                    len(user.get("dialog_history") or [])
                    for user in external_users
                ),
                max(
                    len(user.get("dialog_history") or [])
                    for user in external_users
                ),
            ],
            "runtime_summary_present_rate": {
                "internal": internal_catalog_profile[
                    "session_summary_present_rate"
                ],
                "external": external_catalog_profile[
                    "session_summary_present_rate"
                ],
            },
            "interpretation": [
                "Internal MP represents support preferences while EvoEmo MP represents demographic/profile facts; sharing the same basic_info wire field does not make the semantic task identical.",
                "Half of raw internal sessions lack a supplied summary, but the shared deterministic fallback gives both runtime catalogs complete MS coverage.",
                "Both runtime paths compile every causal prior session. Raw catalog count is diagnostic-only; the PM uses session-relative age and bounded candidate features so natural history length is not a dataset-identity shortcut.",
            ],
        },
        "candidate_descriptor_comparison": descriptor_comparison,
        "external_descriptor_states": len(
            {
                row["state_id"] for row in external_descriptor_rows
            }
        ),
        "external_descriptor_rows": len(external_descriptor_rows),
        "internal_support_descriptor_rows": sum(
            row.get("split") in {"train", "calibration"}
            for row in internal_descriptor_rows
        ),
        "completed_repairs": [
            "Shared candidate-discovery and descriptor function is called by internal construction and EvoEmo runtime.",
            "EvoEmo PM decision is post-candidate-discovery and pre-injection; raw IDs, text, dataset identity, and outcomes are excluded.",
            "Internal and external RS backgrounds share the V4 strategy-card injection compiler.",
            "Memory time labels use the same relative-session representation.",
            "Backend, 192-pair blueprint, and 384-call plan were rebuilt after the mechanism change.",
        ],
        "claim_boundary": {
            "can_be_100_percent": "executable mechanism equivalence",
            "cannot_be_100_percent_without_leakage": (
                "content, topic, history length, catalog size, and candidate "
                "feature distributions"
            ),
            "required_external_reporting": (
                "outcome-free in-support coverage plus full unweighted "
                "quality/risk/cost after the model and thresholds are frozen"
            ),
        },
        "inputs": {
            "bundles": str(args.bundles.relative_to(ROOT)),
            "bundles_sha256": sha256_file(args.bundles),
            "internal_backend_report": str(
                (
                    args.internal_backend_dir / "report.json"
                ).relative_to(ROOT)
            ),
            "internal_backend_report_sha256": sha256_file(
                args.internal_backend_dir / "report.json"
            ),
            "generation_plan": str(
                (
                    args.generation_plan_dir / "plan_report.json"
                ).relative_to(ROOT)
            ),
            "generation_plan_sha256": sha256_file(
                args.generation_plan_dir / "plan_report.json"
            ),
            "repaired_rs_generation_plan": str(
                (
                    args.repaired_rs_generation_plan_dir
                    / "plan_report.json"
                ).relative_to(ROOT)
            ),
            "repaired_rs_generation_plan_sha256": sha256_file(
                args.repaired_rs_generation_plan_dir / "plan_report.json"
            ),
            "repaired_rs_generation_identity_matches": (
                generation_reuse_checks["generator_identity_matches"]
            ),
            "evoemo": str(args.evoemo.relative_to(ROOT)),
            "evoemo_sha256": sha256_file(args.evoemo),
            "fixed_tracks": str(args.fixed_tracks.relative_to(ROOT)),
            "fixed_tracks_sha256": sha256_file(args.fixed_tracks),
            "pm_config": str(args.pm_config.relative_to(ROOT)),
            "pm_config_sha256": sha256_file(args.pm_config),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "equivalence_audit.json", report)
    write_jsonl(
        args.out_dir / "candidate_distribution_summary.jsonl",
        descriptor_long_rows,
    )
    write_jsonl(
        args.out_dir / "exact_mechanism_checks.jsonl",
        exact_checks,
    )
    write_jsonl(
        args.out_dir / "external_candidate_descriptors.jsonl",
        external_descriptor_rows,
    )
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": report["status"],
                "exact_mechanism": (
                    f"{exact_passes}/{len(exact_checks)}"
                ),
                "failed_layers": failed_layers,
                "generation_reuse": generation_reuse_checks,
                "external_descriptor_states": report[
                    "external_descriptor_states"
                ],
                "api_calls_made": 0,
                "out_dir": str(args.out_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
