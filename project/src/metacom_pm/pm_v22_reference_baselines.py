from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .api import Endpoint, make_client, request_log
from .artifacts import create_artifact_attestation
from .attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
    reported_prompt_token_error,
)
from .contracts import (
    CostRecord,
    MemoryItem,
    MemorySource,
    StrategyCard,
    StrategyMode,
    canonical_action_id,
)
from .evidence_filter import EvidenceFilterConfig, filter_evidence
from .evoemo import (
    _fixed_context_before_turn,
    _load_fixed_tracks,
    _session_rag,
    _track_key,
    build_evo_memory,
    evo_memory_global_catalog_digest,
    load_evoemo,
    make_evo_runtime_state,
)
from .generation_contract import SupporterGenerationContract
from .io import (
    append_jsonl,
    canonical_json,
    ensure_run_manifest,
    index_jsonl_unique,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .prompts import generation_messages
from .retrieval import MemoryRetriever, StrategyRetriever, context_query
from .text import conservative_token_bound, estimate_tokens


PMV22_REFERENCE_BASELINE_STAGE = "evoemo_pmv22_reference_baselines"
PMV22_REFERENCE_BASELINE_PROTOCOL = "pm-v2.2-reference-baselines-v1"
PMV22_REFERENCE_EVIDENCE_PROTOCOL = (
    "pm-v2.2-reference-baseline-evidence-processing-v1"
)
PMV22_EXTERNAL_UNIT_PROTOCOL = (
    "pm-v2-external-fixed-context-evaluation-universe-v1"
)
REFERENCE_BASELINE_CONDITIONS = (
    "no_memory_r0",
    "best_fixed",
    "session_rag_rs",
    "full_history_rs",
)
POLICY_LOCK_TIMING = "before_first_reference_baseline_api_call"
POST_GENERATION_POLICY_TUNING_PROHIBITED = True
POLICY_LOCK_FIELDS = (
    "policy_checkpoint_sha256",
    "policy_training_report_sha256",
    "policy_lock_timing",
    "post_generation_policy_tuning_prohibited",
)


def build_raw_generation_contract_gate(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    call_plan: Sequence[Mapping[str, Any]],
    supporter_generation_treatment: Mapping[str, Any],
    supporter_generation_treatment_sha256: str,
    fixed_seeker_generation_treatment: Mapping[str, Any],
    fixed_seeker_generation_treatment_sha256: str,
    evidence_processing_contracts: Mapping[str, Mapping[str, Any]],
    policy_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the explicit all-raw-row gate required before COMPLETE."""

    expected_keys = {str(row.get("call_key") or "") for row in call_plan}
    observed_keys = {
        str(row.get("physical_call_key") or "") for row in raw_rows
    }
    expected_rows = len(call_plan)
    observed_rows = len(raw_rows)
    normalized_evidence = {
        str(key): dict(value) for key, value in evidence_processing_contracts.items()
    }
    checks = {
        "row_count_exact": observed_rows == expected_rows,
        "call_keys_exact": (
            "" not in expected_keys
            and "" not in observed_keys
            and len(expected_keys) == expected_rows
            and len(observed_keys) == observed_rows
            and observed_keys == expected_keys
        ),
        "finish_reasons_complete": all(
            row.get("normalized_finish_reason") == "complete" for row in raw_rows
        ),
        "errors_absent": all(row.get("error") in (None, "") for row in raw_rows),
        "supporter_generation_treatment_exact": all(
            row.get("supporter_generation_treatment")
            == dict(supporter_generation_treatment)
            and row.get("supporter_generation_treatment_sha256")
            == supporter_generation_treatment_sha256
            for row in raw_rows
        ),
        "fixed_seeker_generation_treatment_exact": all(
            row.get("fixed_seeker_generation_treatment")
            == dict(fixed_seeker_generation_treatment)
            and row.get("fixed_seeker_generation_treatment_sha256")
            == fixed_seeker_generation_treatment_sha256
            for row in raw_rows
        ),
        "evidence_processing_contract_exact": all(
            str(row.get("condition") or "") in normalized_evidence
            and row.get("evidence_processing_contract")
            == normalized_evidence[str(row.get("condition") or "")]
            and row.get("evidence_processing_contract_sha256")
            == sha256_text(
                canonical_json(
                    normalized_evidence[str(row.get("condition") or "")]
                )
            )
            for row in raw_rows
        ),
        "policy_lock_exact": all(
            all(row.get(key) == policy_lock.get(key) for key in POLICY_LOCK_FIELDS)
            for row in raw_rows
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        **checks,
    }


def require_reference_policy_lock(
    *,
    policy_checkpoint_path: str | Path,
    policy_training_report_path: str | Path,
) -> dict[str, Any]:
    """Bind the already-selected PM before any reference response can exist."""

    checkpoint = Path(policy_checkpoint_path)
    report_path = Path(policy_training_report_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "reference-baseline planning requires the final PM-v2 policy "
            f"checkpoint: {checkpoint}"
        )
    if not report_path.is_file():
        raise FileNotFoundError(
            "reference-baseline planning requires the final PM-v2 training "
            f"report: {report_path}"
        )
    report = read_json(report_path)
    if (
        report.get("status") != "COMPLETE"
        or report.get("require_learned_routing_advantage_before_external")
        is not True
        or report.get("learned_routing_advantage_verified") is not True
    ):
        raise RuntimeError(
            "reference baselines require a final policy whose training report "
            "passed the frozen learned-routing gate"
        )
    reported_checkpoint = str(report.get("checkpoint") or "")
    if not reported_checkpoint or Path(reported_checkpoint).name != checkpoint.name:
        raise RuntimeError(
            "policy training report does not identify the selected checkpoint"
        )
    checkpoint_sha256 = sha256_file(checkpoint)
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(
            "policy training report does not bind the selected checkpoint SHA-256"
        )
    return {
        "policy_checkpoint_sha256": checkpoint_sha256,
        "policy_training_report_sha256": sha256_file(report_path),
        "policy_lock_timing": POLICY_LOCK_TIMING,
        "post_generation_policy_tuning_prohibited": (
            POST_GENERATION_POLICY_TUNING_PROHIBITED
        ),
    }


def generator_endpoint_payload(endpoint: Endpoint) -> dict[str, str]:
    return {
        "model": endpoint.model,
        "family": str(endpoint.family or ""),
        "base_url": endpoint.base_url,
    }


def build_reference_evidence_processing_contracts(
    *,
    evidence_filter_config: EvidenceFilterConfig,
    evidence_filter_model_binding: Mapping[str, Any],
    memory_min_score: float,
    strategy_min_score: float,
    strategy_top_k: int,
    session_rag_top_k: int,
) -> dict[str, dict[str, Any]]:
    """Freeze what each fixed comparator retrieves and what may be filtered.

    Raw-session comparators intentionally preserve their authorized raw-memory
    payload. Applying the structured ME item cap to those whole-session
    documents would silently turn ``full_history_rs`` into another top-k
    condition. They still use the same contextual strategy-card fail-safe.
    """

    filter_payload = evidence_filter_config.payload()
    filter_sha256 = evidence_filter_config.digest()
    model_binding = dict(evidence_filter_model_binding)
    common = {
        "protocol": PMV22_REFERENCE_EVIDENCE_PROTOCOL,
        "evidence_filter": filter_payload,
        "evidence_filter_config_sha256": filter_sha256,
        "evidence_filter_model": model_binding,
        "strategy_retrieval": {
            "retriever": "deterministic_lexical_strategy_retriever_v1",
            "top_k": int(strategy_top_k),
            "minimum_score": float(strategy_min_score),
        },
    }
    values = {
        "no_memory_r0": {
            **common,
            "requested_action_id": "M0+R0",
            "memory_candidate_policy": "none",
            "memory_processing": "not_applicable_no_memory_candidates",
            "strategy_candidate_policy": "none",
            "strategy_processing": "not_applicable_r0",
        },
        "best_fixed": {
            **common,
            "requested_action_id": "MPMSME+RS",
            "memory_candidate_policy": {
                "retriever": "deterministic_structured_memory_retriever_v1",
                "sources": ["MP", "MS", "ME"],
                "minimum_score_per_source": float(memory_min_score),
                "top_k_per_source": {"MP": 2, "MS": 2, "ME": 3},
            },
            "memory_processing": "full_shared_supervised_evidence_filter",
            "strategy_candidate_policy": "shared_strategy_retrieval",
            "strategy_processing": "shared_contextual_strategy_fail_safe",
        },
        "session_rag_rs": {
            **common,
            "requested_action_id": "SESSION_RAG+RS",
            "memory_candidate_policy": {
                "retriever": "deterministic_raw_session_lexical_retriever_v1",
                "top_k": int(session_rag_top_k),
                "authorized_content": "past_dialog_history_only",
            },
            "memory_processing": (
                "preserve_all_retrieved_raw_session_documents_no_structured_"
                "item_filter"
            ),
            "strategy_candidate_policy": "shared_strategy_retrieval",
            "strategy_processing": "shared_contextual_strategy_fail_safe",
        },
        "full_history_rs": {
            **common,
            "requested_action_id": "FULL_HISTORY+RS",
            "memory_candidate_policy": {
                "retriever": "all_authorized_past_session_documents_v1",
                "authorized_content": "past_dialog_history_only",
            },
            "memory_processing": (
                "preserve_all_authorized_raw_session_documents_no_structured_"
                "item_filter"
            ),
            "strategy_candidate_policy": "shared_strategy_retrieval",
            "strategy_processing": "shared_contextual_strategy_fail_safe",
        },
    }
    if tuple(values) != REFERENCE_BASELINE_CONDITIONS:
        raise RuntimeError("reference-baseline evidence contract order changed")
    return values


def evidence_processing_contracts_sha256(
    contract: Mapping[str, Mapping[str, Any]],
) -> str:
    return sha256_text(canonical_json(dict(contract)))


@dataclass(frozen=True)
class PreparedReferenceCall:
    plan: dict[str, Any]
    messages: list[dict[str, str]]
    runtime: Any
    query: str
    context_before_turn: list[dict[str, str]]
    seeker_message: str
    candidate_memory: list[MemoryItem]
    candidate_strategy: list[StrategyCard]
    selected_memory: list[MemoryItem]
    selected_strategy: list[StrategyCard]
    evidence_filter_decision: dict[str, Any] | None
    retrieval_calls: int
    evidence_filter_calls: int


def _normalized_seeds(seeds: Sequence[int]) -> list[int]:
    values = [int(value) for value in seeds]
    if not values or len(values) != len(set(values)):
        raise ValueError("reference-baseline seeds must be non-empty and unique")
    return values


def _normalized_turn_indices(
    turn_indices: Sequence[int], *, max_turns: int
) -> list[int]:
    values = [int(value) for value in turn_indices]
    if (
        not values
        or values != sorted(set(values))
        or any(value < 1 or value > int(max_turns) for value in values)
    ):
        raise ValueError(
            "reference-baseline turn indices must be sorted, unique, and within max_turns"
        )
    return values


def _scenarios(
    evoemo_path: str | Path,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (user, topic)
        for user in load_evoemo(evoemo_path)
        for topic in (user.get("subsequent_topics") or [])
    ]


def build_reference_evaluation_unit_contract(
    scenarios: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    simulator_id: str,
    turn_indices: Sequence[int],
) -> tuple[dict[str, Any], list[tuple[str, int, int, str, int]]]:
    units = sorted(
        {
            (
                str(user["id"]),
                int(topic["idx"]),
                int(seed),
                str(simulator_id),
                int(turn_index),
            )
            for user, topic in scenarios
            for seed in seeds
            for turn_index in turn_indices
        }
    )
    if not units:
        raise ValueError("reference-baseline evaluation universe is empty")
    return (
        {
            "protocol": PMV22_EXTERNAL_UNIT_PROTOCOL,
            "evaluation_turn_indices": [int(value) for value in turn_indices],
            "expected_unit_count": len(units),
            "expected_units_sha256": sha256_text(canonical_json(units)),
        },
        units,
    )


def _validate_track(
    track: Mapping[str, Any],
    *,
    max_turns: int,
    fixed_seeker_generation_treatment: Mapping[str, Any],
    fixed_seeker_generation_treatment_sha256: str,
) -> None:
    if len(track.get("seeker_turns") or []) != int(max_turns):
        raise RuntimeError("fixed seeker track turn count differs from the contract")
    if (
        track.get("fixed_seeker_generation_contract")
        != dict(fixed_seeker_generation_treatment)
        or track.get("fixed_seeker_generation_contract_sha256")
        != fixed_seeker_generation_treatment_sha256
    ):
        raise RuntimeError(
            "reference baselines require PM-v2.2 treatment-bound fixed tracks"
        )
    provenance = list(track.get("turn_provenance") or [])
    if len(provenance) != int(max_turns) or any(
        row.get("normalized_finish_reason") != "complete"
        or row.get("fixed_seeker_generation_contract_sha256")
        != fixed_seeker_generation_treatment_sha256
        for row in provenance
    ):
        raise RuntimeError("fixed seeker track has a non-complete or stale turn")


def _serialize_evidence(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in rows]


def _prepare_condition_evidence(
    *,
    condition: str,
    query: str,
    runtime: Any,
    items: Sequence[MemoryItem],
    session_docs: list[dict[str, Any]],
    memory_retriever: MemoryRetriever,
    strategy_retriever: StrategyRetriever,
    evidence_filter_config: EvidenceFilterConfig,
    evidence_filter_model: Any,
    session_rag_top_k: int,
) -> tuple[
    str,
    list[MemoryItem],
    list[StrategyCard],
    list[MemoryItem],
    list[StrategyCard],
    dict[str, Any] | None,
    int,
    int,
]:
    if condition == "no_memory_r0":
        return "M0+R0", [], [], [], [], None, 0, 0

    candidate_strategy = strategy_retriever.retrieve(query)
    if condition == "best_fixed":
        action_id = "MPMSME+RS"
        candidate_memory = memory_retriever.retrieve(
            query, items, frozenset(MemorySource)
        )
        filtered = filter_evidence(
            requested_action_id=action_id,
            current_user_text=runtime.current_user_text,
            context_query_text=query,
            memory_candidates=candidate_memory,
            strategy_candidates=candidate_strategy,
            config=evidence_filter_config,
            session_index=runtime.session_index,
            memory_helpfulness_model=evidence_filter_model,
        )
        return (
            action_id,
            candidate_memory,
            candidate_strategy,
            filtered.memory_view,
            filtered.strategy_view,
            filtered.decision.model_dump(mode="json"),
            4,
            1,
        )

    if condition == "session_rag_rs":
        action_id = "SESSION_RAG+RS"
        candidate_memory = _session_rag(
            query, session_docs, top_k=int(session_rag_top_k)
        )
    elif condition == "full_history_rs":
        action_id = "FULL_HISTORY+RS"
        candidate_memory = _session_rag(
            query, session_docs, top_k=len(session_docs)
        )
    else:
        raise ValueError(f"unknown PM-v2.2 reference condition: {condition}")

    strategy_filtered = filter_evidence(
        requested_action_id="M0+RS",
        current_user_text=runtime.current_user_text,
        context_query_text=query,
        memory_candidates=[],
        strategy_candidates=candidate_strategy,
        config=evidence_filter_config,
        session_index=runtime.session_index,
        memory_helpfulness_model=None,
    )
    selected_strategy = list(strategy_filtered.strategy_view)
    effective_action_id = canonical_action_id(
        {item.source for item in candidate_memory},
        StrategyMode.RS if selected_strategy else StrategyMode.R0,
    )
    decision = {
        "protocol": PMV22_REFERENCE_EVIDENCE_PROTOCOL,
        "scope": "strategy_only_raw_memory_preserved",
        "raw_memory_preserved": True,
        "strategy_filter_decision": strategy_filtered.decision.model_dump(
            mode="json"
        ),
        "effective_action_id": effective_action_id,
    }
    return (
        action_id,
        candidate_memory,
        candidate_strategy,
        list(candidate_memory),
        selected_strategy,
        decision,
        2,
        1,
    )


def iter_prepared_reference_calls(
    *,
    evoemo_path: str | Path,
    strategy_cards: Sequence[StrategyCard],
    tracks: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    generator_endpoint: Endpoint,
    supporter_generation_contract: SupporterGenerationContract,
    fixed_seeker_generation_treatment: Mapping[str, Any],
    fixed_seeker_generation_treatment_sha256: str,
    evidence_processing_contracts: Mapping[str, Mapping[str, Any]],
    evidence_processing_contracts_sha256: str,
    evidence_filter_config: EvidenceFilterConfig,
    evidence_filter_model: Any,
    policy_lock: Mapping[str, Any],
    simulator_id: str,
    max_turns: int,
    seeds: Sequence[int],
    turn_indices: Sequence[int],
    memory_min_score: float,
    strategy_min_score: float,
    strategy_top_k: int,
    session_rag_top_k: int,
    input_token_safety_factor: float,
) -> Iterator[PreparedReferenceCall]:
    supporter_treatment = supporter_generation_contract.payload()
    supporter_treatment_sha256 = supporter_generation_contract.digest()
    memory_retriever = MemoryRetriever(
        minimum_score_by_source={
            source: float(memory_min_score) for source in MemorySource
        }
    )
    strategy_retriever = StrategyRetriever(
        strategy_cards,
        top_k=int(strategy_top_k),
        minimum_score=float(strategy_min_score),
    )
    for user, topic in _scenarios(evoemo_path):
        items, session_docs = build_evo_memory(user)
        for seed in seeds:
            track_key = _track_key(
                str(user["id"]), int(topic["idx"]), int(seed), simulator_id
            )
            track = tracks.get(track_key)
            if track is None:
                raise RuntimeError(f"missing PM-v2.2 fixed seeker track: {track_key}")
            _validate_track(
                track,
                max_turns=max_turns,
                fixed_seeker_generation_treatment=(
                    fixed_seeker_generation_treatment
                ),
                fixed_seeker_generation_treatment_sha256=(
                    fixed_seeker_generation_treatment_sha256
                ),
            )
            track_id = str(track["track_id"])
            for turn_index in turn_indices:
                seeker_message = supporter_generation_contract.normalize_output(
                    str(track["seeker_turns"][int(turn_index) - 1])
                )
                if not seeker_message:
                    raise RuntimeError("fixed seeker message is empty")
                context_before_turn = _fixed_context_before_turn(
                    dict(track), int(turn_index)
                )
                for condition in REFERENCE_BASELINE_CONDITIONS:
                    runtime = make_evo_runtime_state(
                        user,
                        topic,
                        context_before_turn,
                        seeker_message,
                        items,
                        int(turn_index),
                        condition,
                        track_id=track_id,
                        fixed_open_loop=True,
                    )
                    query = context_query(
                        runtime.current_user_text,
                        [
                            row.model_dump(mode="json")
                            for row in runtime.current_session_history
                        ],
                        runtime.current_session_summary,
                    )
                    (
                        requested_action_id,
                        candidate_memory,
                        candidate_strategy,
                        selected_memory,
                        selected_strategy,
                        filter_decision,
                        retrieval_calls,
                        filter_calls,
                    ) = _prepare_condition_evidence(
                        condition=condition,
                        query=query,
                        runtime=runtime,
                        items=items,
                        session_docs=session_docs,
                        memory_retriever=memory_retriever,
                        strategy_retriever=strategy_retriever,
                        evidence_filter_config=evidence_filter_config,
                        evidence_filter_model=evidence_filter_model,
                        session_rag_top_k=session_rag_top_k,
                    )
                    messages = generation_messages(
                        runtime,
                        selected_memory,
                        selected_strategy,
                        system_prompt=supporter_generation_contract.system_prompt,
                    )
                    messages_json = canonical_json(messages)
                    prompt_hash = sha256_text(messages_json)
                    evidence_contract = dict(
                        evidence_processing_contracts[condition]
                    )
                    evidence_contract_sha256 = sha256_text(
                        canonical_json(evidence_contract)
                    )
                    record_ids = {
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "condition": condition,
                        "seed": int(seed),
                        "simulator_id": str(simulator_id),
                        "turn_index": int(turn_index),
                        "interaction_mode": "fixed",
                        "track_id": track_id,
                    }
                    call_key = physical_call_key(
                        stage=PMV22_REFERENCE_BASELINE_STAGE,
                        record_ids=record_ids,
                        prompt_sha256=prompt_hash,
                        endpoint=generator_endpoint,
                        request_parameters={
                            "temperature": (
                                supporter_generation_contract.temperature
                            ),
                            "max_tokens": (
                                supporter_generation_contract.max_output_tokens
                            ),
                            "seed": int(seed) + int(turn_index),
                            "response_schema": None,
                            "supporter_generation_treatment": supporter_treatment,
                            "supporter_generation_treatment_sha256": (
                                supporter_treatment_sha256
                            ),
                            "fixed_seeker_generation_treatment": dict(
                                fixed_seeker_generation_treatment
                            ),
                            "fixed_seeker_generation_treatment_sha256": (
                                fixed_seeker_generation_treatment_sha256
                            ),
                            "evidence_processing_contracts": dict(
                                evidence_processing_contracts
                            ),
                            "evidence_processing_contracts_sha256": (
                                evidence_processing_contracts_sha256
                            ),
                            "evidence_processing_contract": evidence_contract,
                            "evidence_processing_contract_sha256": (
                                evidence_contract_sha256
                            ),
                            **dict(policy_lock),
                        },
                    )
                    selected_memory_rows = _serialize_evidence(selected_memory)
                    selected_strategy_rows = _serialize_evidence(selected_strategy)
                    candidate_memory_rows = _serialize_evidence(candidate_memory)
                    candidate_strategy_rows = _serialize_evidence(
                        candidate_strategy
                    )
                    plan = {
                        **record_ids,
                        "requested_action_id": requested_action_id,
                        "effective_action_id": canonical_action_id(
                            {item.source for item in selected_memory},
                            (
                                StrategyMode.RS
                                if selected_strategy
                                else StrategyMode.R0
                            ),
                        ),
                        "turn_seed": int(seed) + int(turn_index),
                        "input_tokens_est": conservative_token_bound(
                            messages_json,
                            safety_factor=float(input_token_safety_factor),
                        ),
                        "raw_input_tokens_est": estimate_tokens(messages_json),
                        "max_output_tokens": (
                            supporter_generation_contract.max_output_tokens
                        ),
                        "max_http_attempts": 1,
                        "prompt_hash": prompt_hash,
                        "call_key": call_key,
                        "candidate_memory_count": len(candidate_memory),
                        "kept_memory_count": len(selected_memory),
                        "candidate_strategy_count": len(candidate_strategy),
                        "kept_strategy_count": len(selected_strategy),
                        "candidate_memory_sha256": sha256_text(
                            canonical_json(candidate_memory_rows)
                        ),
                        "selected_memory_sha256": sha256_text(
                            canonical_json(selected_memory_rows)
                        ),
                        "candidate_strategy_sha256": sha256_text(
                            canonical_json(candidate_strategy_rows)
                        ),
                        "selected_strategy_sha256": sha256_text(
                            canonical_json(selected_strategy_rows)
                        ),
                        "supporter_generation_treatment": supporter_treatment,
                        "supporter_generation_treatment_sha256": (
                            supporter_treatment_sha256
                        ),
                        "fixed_seeker_generation_treatment": dict(
                            fixed_seeker_generation_treatment
                        ),
                        "fixed_seeker_generation_treatment_sha256": (
                            fixed_seeker_generation_treatment_sha256
                        ),
                        "evidence_processing_contract": evidence_contract,
                        "evidence_processing_contract_sha256": (
                            evidence_contract_sha256
                        ),
                        **dict(policy_lock),
                    }
                    yield PreparedReferenceCall(
                        plan=plan,
                        messages=messages,
                        runtime=runtime,
                        query=query,
                        context_before_turn=context_before_turn,
                        seeker_message=seeker_message,
                        candidate_memory=list(candidate_memory),
                        candidate_strategy=list(candidate_strategy),
                        selected_memory=list(selected_memory),
                        selected_strategy=list(selected_strategy),
                        evidence_filter_decision=filter_decision,
                        retrieval_calls=retrieval_calls,
                        evidence_filter_calls=filter_calls,
                    )


def plan_reference_baselines(
    *,
    evoemo_path: str | Path,
    strategy_bank_path: str | Path,
    fixed_tracks_path: str | Path,
    policy_checkpoint_path: str | Path,
    policy_training_report_path: str | Path,
    generator_endpoint: Endpoint,
    supporter_generation_contract: SupporterGenerationContract,
    fixed_seeker_generation_treatment: Mapping[str, Any],
    fixed_seeker_generation_treatment_sha256: str,
    evidence_filter_config: EvidenceFilterConfig,
    evidence_filter_model: Any,
    evidence_filter_model_binding: Mapping[str, Any],
    simulator_id: str,
    max_turns: int,
    seeds: Sequence[int],
    turn_indices: Sequence[int],
    memory_min_score: float,
    strategy_min_score: float,
    strategy_top_k: int,
    session_rag_top_k: int,
    input_token_safety_factor: float,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seeds = _normalized_seeds(seeds)
    turn_indices = _normalized_turn_indices(
        turn_indices, max_turns=int(max_turns)
    )
    if not str(simulator_id).strip():
        raise ValueError("reference-baseline simulator_id must be non-empty")
    prices = {
        "input": float(input_usd_per_mtok),
        "output": float(output_usd_per_mtok),
    }
    if any(value <= 0.0 for value in prices.values()):
        raise ValueError("reference-baseline planning requires positive prices")
    if float(input_token_safety_factor) < 1.0:
        raise ValueError("input-token safety factor must be at least one")
    policy_lock = require_reference_policy_lock(
        policy_checkpoint_path=policy_checkpoint_path,
        policy_training_report_path=policy_training_report_path,
    )

    scenarios = _scenarios(evoemo_path)
    # evoemo_sha256 below only pins the raw input file, not what
    # build_evo_memory actually constructs from it (MP/MS/ME item content,
    # chunking, ids) -- record that separately so this cost estimate (and
    # everything derived from it) is auditable against the memory builder
    # that actually produced the retrieval catalog used in planning. Derive
    # the user list from `scenarios` (not a fresh load_evoemo(evoemo_path)
    # call) so this goes through the same _scenarios seam every caller and
    # test fixture already uses, instead of a second, independent read.
    seen_user_ids: set[str] = set()
    unique_scenario_users: list[dict[str, Any]] = []
    for scenario_user, _scenario_topic in scenarios:
        scenario_user_id = str(scenario_user["id"])
        if scenario_user_id not in seen_user_ids:
            seen_user_ids.add(scenario_user_id)
            unique_scenario_users.append(scenario_user)
    evo_memory_digest = evo_memory_global_catalog_digest(unique_scenario_users)
    evaluation_unit_contract, units = build_reference_evaluation_unit_contract(
        scenarios,
        seeds=seeds,
        simulator_id=simulator_id,
        turn_indices=turn_indices,
    )
    tracks = _load_fixed_tracks(fixed_tracks_path)
    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)
    ]
    if not strategy_cards:
        raise RuntimeError("PM-v2.2 reference Strategy Bank is empty")
    evidence_contract = build_reference_evidence_processing_contracts(
        evidence_filter_config=evidence_filter_config,
        evidence_filter_model_binding=evidence_filter_model_binding,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        strategy_top_k=strategy_top_k,
        session_rag_top_k=session_rag_top_k,
    )
    evidence_contract_sha256 = (
        evidence_processing_contracts_sha256(evidence_contract)
    )
    prepared = iter_prepared_reference_calls(
        evoemo_path=evoemo_path,
        strategy_cards=strategy_cards,
        tracks=tracks,
        generator_endpoint=generator_endpoint,
        supporter_generation_contract=supporter_generation_contract,
        fixed_seeker_generation_treatment=fixed_seeker_generation_treatment,
        fixed_seeker_generation_treatment_sha256=(
            fixed_seeker_generation_treatment_sha256
        ),
        evidence_processing_contracts=evidence_contract,
        evidence_processing_contracts_sha256=evidence_contract_sha256,
        evidence_filter_config=evidence_filter_config,
        evidence_filter_model=evidence_filter_model,
        policy_lock=policy_lock,
        simulator_id=simulator_id,
        max_turns=max_turns,
        seeds=seeds,
        turn_indices=turn_indices,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        strategy_top_k=strategy_top_k,
        session_rag_top_k=session_rag_top_k,
        input_token_safety_factor=input_token_safety_factor,
    )
    call_plan = [row.plan for row in prepared]
    call_keys = [str(row["call_key"]) for row in call_plan]
    if len(call_keys) != len(set(call_keys)):
        raise RuntimeError("reference-baseline call plan has duplicate keys")
    expected_calls = len(units) * len(REFERENCE_BASELINE_CONDITIONS)
    if len(call_plan) != expected_calls:
        raise RuntimeError("reference-baseline call plan matrix is incomplete")
    for condition in REFERENCE_BASELINE_CONDITIONS:
        observed = sorted(
            (
                str(row["user_id"]),
                int(row["topic_index"]),
                int(row["seed"]),
                str(row["simulator_id"]),
                int(row["turn_index"]),
            )
            for row in call_plan
            if row["condition"] == condition
        )
        if observed != units:
            raise RuntimeError(
                f"reference-baseline call plan is incomplete for {condition}"
            )

    input_counts = [int(row["input_tokens_est"]) for row in call_plan]
    total_input_tokens = sum(input_counts)
    total_output_tokens = sum(
        int(row["max_output_tokens"]) for row in call_plan
    )
    supporter_treatment = supporter_generation_contract.payload()
    supporter_treatment_sha256 = supporter_generation_contract.digest()
    call_plan_sha256 = sha256_text(canonical_json(call_plan))
    cost_payload = {
        "protocol": PMV22_REFERENCE_BASELINE_PROTOCOL,
        "stage": PMV22_REFERENCE_BASELINE_STAGE,
        "conditions": sorted(REFERENCE_BASELINE_CONDITIONS),
        "evoemo_sha256": sha256_file(evoemo_path),
        "evo_memory_builder_contract_sha256": evo_memory_digest[
            "builder_contract_sha256"
        ],
        "evo_memory_global_catalog_sha256": evo_memory_digest[
            "global_catalog_sha256"
        ],
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "fixed_tracks_sha256": sha256_file(fixed_tracks_path),
        "generator_endpoint": generator_endpoint_payload(generator_endpoint),
        "supporter_generation_treatment": supporter_treatment,
        "supporter_generation_treatment_sha256": supporter_treatment_sha256,
        "fixed_seeker_generation_treatment": dict(
            fixed_seeker_generation_treatment
        ),
        "fixed_seeker_generation_treatment_sha256": (
            fixed_seeker_generation_treatment_sha256
        ),
        "evidence_processing_contracts": evidence_contract,
        "evidence_processing_contracts_sha256": (
            evidence_contract_sha256
        ),
        "evidence_filter_config_sha256": evidence_filter_config.digest(),
        "evidence_filter_model": dict(evidence_filter_model_binding),
        **policy_lock,
        "simulator_id": simulator_id,
        "max_turns": int(max_turns),
        "seeds": seeds,
        "evaluation_unit_contract": evaluation_unit_contract,
        "paid_generation_scope": "frozen_evaluation_turns_only",
        "input_token_safety_factor": float(input_token_safety_factor),
        "fail_on_reported_input_overrun": True,
        "generator_pricing_usd_per_mtok": prices,
        "expected_api_calls": len(call_plan),
        "maximum_physical_http_attempts": len(call_plan),
        "maximum_physical_attempts_per_logical_call": 1,
        "total_input_tokens_est": total_input_tokens,
        "max_input_tokens_per_call_est": max(input_counts, default=0),
        "total_output_tokens_est": total_output_tokens,
        "estimated_cost_usd": (
            total_input_tokens / 1_000_000 * prices["input"]
            + total_output_tokens / 1_000_000 * prices["output"]
        ),
        "call_plan_sha256": call_plan_sha256,
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }
    cost_estimate = {
        **cost_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(cost_payload)),
    }
    checks = {
        "api_calls": len(call_plan) <= int(max_api_calls),
        "estimated_cost_usd": cost_estimate["estimated_cost_usd"]
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": cost_estimate[
            "max_input_tokens_per_call_est"
        ]
        <= int(max_input_tokens_per_call),
    }
    saved_estimate = {
        **cost_estimate,
        "budget_gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "limits": dict(cost_payload["budget_limits"]),
        },
    }
    return saved_estimate, call_plan


def persist_reference_baseline_dry_run(
    out_dir: str | Path,
    *,
    cost_estimate: Mapping[str, Any],
    call_plan: Sequence[Mapping[str, Any]],
    overwrite: bool = False,
) -> str:
    out_dir = Path(out_dir)
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=overwrite,
        stage=PMV22_REFERENCE_BASELINE_STAGE,
    )
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    normalized_estimate = dict(cost_estimate)
    normalized_plan = [dict(row) for row in call_plan]
    if estimate_path.is_file() or plan_path.is_file():
        if not estimate_path.is_file() or not plan_path.is_file():
            raise RuntimeError("reference-baseline dry-run bundle is partial")
        if (
            read_json(estimate_path) == normalized_estimate
            and list(iter_jsonl(plan_path)) == normalized_plan
        ):
            return "VALIDATED_EXISTING"
        if not overwrite:
            raise RuntimeError(
                "reference-baseline plan changed; use a new output directory"
            )
    if overwrite:
        for name in (
            "cost_estimate.json",
            "call_plan.jsonl",
            "run_manifest.json",
            "turns.jsonl",
            "raw_api_calls.jsonl",
            "generation_summary.json",
            "artifact_attestation.json",
        ):
            path = out_dir / name
            if path.exists():
                path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, normalized_estimate)
    write_jsonl(plan_path, normalized_plan)
    return "WRITTEN"


def require_accepted_reference_baseline_plan(
    out_dir: str | Path,
    *,
    cost_estimate: Mapping[str, Any],
    call_plan: Sequence[Mapping[str, Any]],
    accepted_cost_estimate_sha256: str,
) -> None:
    out_dir = Path(out_dir)
    estimate_path = out_dir / "cost_estimate.json"
    plan_path = out_dir / "call_plan.jsonl"
    if not estimate_path.is_file() or not plan_path.is_file():
        raise RuntimeError(
            "reference-baseline API mode requires a matching saved --dry-run"
        )
    if read_json(estimate_path) != dict(cost_estimate) or list(
        iter_jsonl(plan_path)
    ) != [dict(row) for row in call_plan]:
        raise RuntimeError("saved reference-baseline dry run is stale")
    expected = str(cost_estimate.get("cost_estimate_sha256") or "")
    if not expected or accepted_cost_estimate_sha256 != expected:
        raise RuntimeError(
            "--accept-cost-estimate-sha256 must exactly equal the current "
            f"reference-baseline estimate hash: {expected}"
        )
    if (cost_estimate.get("budget_gate") or {}).get("status") != "PASS":
        raise RuntimeError("reference-baseline budget gate did not PASS")


def _turn_unit(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["user_id"]),
        int(row["topic_index"]),
        str(row["condition"]),
        int(row["seed"]),
        str(row["simulator_id"]),
        str(row["interaction_mode"]),
        int(row["turn_index"]),
    )


def _reconcile_reference_turns(
    *,
    ledger: PersistentAttemptLedger,
    plan_by_unit: Mapping[tuple[Any, ...], Mapping[str, Any]],
    turns_path: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    turn_fields = (
        "user_id",
        "topic_index",
        "condition",
        "seed",
        "simulator_id",
        "interaction_mode",
        "turn_index",
    )
    turns = (
        index_jsonl_unique(turns_path, turn_fields)
        if turns_path.is_file()
        else {}
    )
    if set(turns) - set(plan_by_unit):
        raise RuntimeError("reference-baseline turns contain unplanned units")
    for unit, planned in plan_by_unit.items():
        call_key = str(planned["call_key"])
        if unit not in turns and ledger.succeeded(call_key):
            terminal = ledger.terminal_row(call_key)
            result = (terminal or {}).get("result") or {}
            stored = result.get("turn_record")
            if not isinstance(stored, Mapping) or _turn_unit(stored) != unit:
                raise RuntimeError(
                    "successful reference-baseline attempt lacks its recoverable turn"
                )
            append_jsonl(turns_path, dict(stored))
            turns[unit] = dict(stored)
        if unit in turns:
            row = turns[unit]
            if (
                row.get("physical_call_key") != call_key
                or not ledger.succeeded(call_key)
                or row.get("supporter_generation_treatment_sha256")
                != planned.get("supporter_generation_treatment_sha256")
                or row.get("fixed_seeker_generation_treatment_sha256")
                != planned.get("fixed_seeker_generation_treatment_sha256")
                or row.get("evidence_processing_contract_sha256")
                != planned.get("evidence_processing_contract_sha256")
                or any(row.get(key) != planned.get(key) for key in POLICY_LOCK_FIELDS)
                or row.get("normalized_finish_reason") != "complete"
            ):
                raise RuntimeError(
                    "persisted reference-baseline turn is not bound to its plan"
                )
    return turns


def run_reference_baselines(
    *,
    out_dir: str | Path,
    evoemo_path: str | Path,
    strategy_bank_path: str | Path,
    fixed_tracks_path: str | Path,
    policy_checkpoint_path: str | Path,
    policy_training_report_path: str | Path,
    fixed_tracks_attestation_path: str | Path,
    pm_v2_config_path: str | Path,
    evidence_filter_checkpoint_path: str | Path | None,
    evidence_filter_report_path: str | Path | None,
    evidence_filter_attestation_path: str | Path | None,
    strategy_bank_approval_path: str | Path,
    generator_endpoint: Endpoint,
    supporter_generation_contract: SupporterGenerationContract,
    fixed_seeker_generation_treatment: Mapping[str, Any],
    fixed_seeker_generation_treatment_sha256: str,
    evidence_filter_config: EvidenceFilterConfig,
    evidence_filter_model: Any,
    evidence_filter_model_binding: Mapping[str, Any],
    strategy_bank_approval: Mapping[str, Any],
    simulator_id: str,
    max_turns: int,
    seeds: Sequence[int],
    turn_indices: Sequence[int],
    memory_min_score: float,
    strategy_min_score: float,
    strategy_top_k: int,
    session_rag_top_k: int,
    input_token_safety_factor: float,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    accepted_cost_estimate_sha256: str,
    client_factory: Callable[[Endpoint], Any] = make_client,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    cost_estimate, call_plan = plan_reference_baselines(
        evoemo_path=evoemo_path,
        strategy_bank_path=strategy_bank_path,
        fixed_tracks_path=fixed_tracks_path,
        policy_checkpoint_path=policy_checkpoint_path,
        policy_training_report_path=policy_training_report_path,
        generator_endpoint=generator_endpoint,
        supporter_generation_contract=supporter_generation_contract,
        fixed_seeker_generation_treatment=fixed_seeker_generation_treatment,
        fixed_seeker_generation_treatment_sha256=(
            fixed_seeker_generation_treatment_sha256
        ),
        evidence_filter_config=evidence_filter_config,
        evidence_filter_model=evidence_filter_model,
        evidence_filter_model_binding=evidence_filter_model_binding,
        simulator_id=simulator_id,
        max_turns=max_turns,
        seeds=seeds,
        turn_indices=turn_indices,
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        strategy_top_k=strategy_top_k,
        session_rag_top_k=session_rag_top_k,
        input_token_safety_factor=input_token_safety_factor,
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
    )
    require_accepted_reference_baseline_plan(
        out_dir,
        cost_estimate=cost_estimate,
        call_plan=call_plan,
        accepted_cost_estimate_sha256=accepted_cost_estimate_sha256,
    )
    turns_path = out_dir / "turns.jsonl"
    raw_path = out_dir / "raw_api_calls.jsonl"
    ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    summary_path = out_dir / "generation_summary.json"
    manifest_path = out_dir / "run_manifest.json"
    attestation_path = out_dir / "artifact_attestation.json"

    common_contract = {
        "conditions": sorted(REFERENCE_BASELINE_CONDITIONS),
        "evo_memory_builder_contract_sha256": cost_estimate[
            "evo_memory_builder_contract_sha256"
        ],
        "evo_memory_global_catalog_sha256": cost_estimate[
            "evo_memory_global_catalog_sha256"
        ],
        "supporter_generation_treatment": cost_estimate[
            "supporter_generation_treatment"
        ],
        "supporter_generation_treatment_sha256": cost_estimate[
            "supporter_generation_treatment_sha256"
        ],
        "fixed_seeker_generation_treatment": cost_estimate[
            "fixed_seeker_generation_treatment"
        ],
        "fixed_seeker_generation_treatment_sha256": cost_estimate[
            "fixed_seeker_generation_treatment_sha256"
        ],
        "evidence_processing_contracts": cost_estimate[
            "evidence_processing_contracts"
        ],
        "evidence_processing_contracts_sha256": cost_estimate[
            "evidence_processing_contracts_sha256"
        ],
        "simulator_id": simulator_id,
        "max_turns": int(max_turns),
        "seeds": [int(value) for value in seeds],
        "evaluation_unit_contract": cost_estimate["evaluation_unit_contract"],
        "paid_generation_scope": "frozen_evaluation_turns_only",
        "policy_checkpoint_sha256": cost_estimate[
            "policy_checkpoint_sha256"
        ],
        "policy_training_report_sha256": cost_estimate[
            "policy_training_report_sha256"
        ],
        "policy_lock_timing": cost_estimate["policy_lock_timing"],
        "post_generation_policy_tuning_prohibited": cost_estimate[
            "post_generation_policy_tuning_prohibited"
        ],
    }
    manifest = ensure_run_manifest(
        manifest_path,
        {
            "stage": PMV22_REFERENCE_BASELINE_STAGE,
            "protocol": PMV22_REFERENCE_BASELINE_PROTOCOL,
            "evoemo_sha256": sha256_file(evoemo_path),
            "strategy_bank_sha256": sha256_file(strategy_bank_path),
            "fixed_tracks_sha256": sha256_file(fixed_tracks_path),
            "fixed_tracks_attestation_sha256": sha256_file(
                fixed_tracks_attestation_path
            ),
            "pm_v2_config_sha256": sha256_file(pm_v2_config_path),
            "generator_model": generator_endpoint.model,
            "generator_family": str(generator_endpoint.family or ""),
            "generator_base_url": generator_endpoint.base_url,
            **common_contract,
            "evidence_filter_config_sha256": evidence_filter_config.digest(),
            "evidence_filter_model": dict(evidence_filter_model_binding),
            "strategy_bank_approval": dict(strategy_bank_approval),
            "retrieval_settings": {
                "memory_min_score": float(memory_min_score),
                "strategy_min_score": float(strategy_min_score),
                "strategy_top_k": int(strategy_top_k),
                "session_rag_top_k": int(session_rag_top_k),
            },
            "generator_retries": 1,
            "generator_pricing_usd_per_mtok": {
                "input": float(input_usd_per_mtok),
                "output": float(output_usd_per_mtok),
            },
            "input_token_safety_factor": float(input_token_safety_factor),
            "fail_on_reported_input_overrun": True,
            "accepted_cost_estimate_sha256": (
                accepted_cost_estimate_sha256
            ),
            "physical_attempt_ledger_protocol": (
                PHYSICAL_ATTEMPT_LEDGER_PROTOCOL
            ),
            "runtime_maximum_physical_http_attempts": len(call_plan),
            "budget_maximum_api_calls": int(max_api_calls),
        },
    )
    expected_calls = {str(row["call_key"]): 1 for row in call_plan}
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=PMV22_REFERENCE_BASELINE_STAGE,
        expected_calls=expected_calls,
        maximum_total_attempts=len(expected_calls),
    )
    if ledger.failures():
        raise RuntimeError(
            "reference-baseline bundle contains a failed spent attempt; use a "
            "new protocol-reviewed output directory"
        )
    if any(
        ledger.attempts_for(call_key)
        and ledger.terminal_row(call_key) is None
        for call_key in expected_calls
    ):
        raise RuntimeError(
            "reference-baseline bundle contains an unresolved spent attempt"
        )
    plan_by_unit = {_turn_unit(row): row for row in call_plan}
    if len(plan_by_unit) != len(call_plan):
        raise RuntimeError("reference-baseline call plan has duplicate units")
    turns = _reconcile_reference_turns(
        ledger=ledger,
        plan_by_unit=plan_by_unit,
        turns_path=turns_path,
    )

    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)
    ]
    tracks = _load_fixed_tracks(fixed_tracks_path)
    evidence_contract = cost_estimate[
        "evidence_processing_contracts"
    ]
    evidence_contract_sha256 = cost_estimate[
        "evidence_processing_contracts_sha256"
    ]
    client = None
    try:
        prepared_calls = iter_prepared_reference_calls(
            evoemo_path=evoemo_path,
            strategy_cards=strategy_cards,
            tracks=tracks,
            generator_endpoint=generator_endpoint,
            supporter_generation_contract=supporter_generation_contract,
            fixed_seeker_generation_treatment=(
                fixed_seeker_generation_treatment
            ),
            fixed_seeker_generation_treatment_sha256=(
                fixed_seeker_generation_treatment_sha256
            ),
            evidence_processing_contracts=evidence_contract,
            evidence_processing_contracts_sha256=(
                evidence_contract_sha256
            ),
            evidence_filter_config=evidence_filter_config,
            evidence_filter_model=evidence_filter_model,
            policy_lock={
                "policy_checkpoint_sha256": cost_estimate[
                    "policy_checkpoint_sha256"
                ],
                "policy_training_report_sha256": cost_estimate[
                    "policy_training_report_sha256"
                ],
                "policy_lock_timing": cost_estimate["policy_lock_timing"],
                "post_generation_policy_tuning_prohibited": cost_estimate[
                    "post_generation_policy_tuning_prohibited"
                ],
            },
            simulator_id=simulator_id,
            max_turns=max_turns,
            seeds=seeds,
            turn_indices=turn_indices,
            memory_min_score=memory_min_score,
            strategy_min_score=strategy_min_score,
            strategy_top_k=strategy_top_k,
            session_rag_top_k=session_rag_top_k,
            input_token_safety_factor=input_token_safety_factor,
        )
        for prepared in prepared_calls:
            planned = plan_by_unit[_turn_unit(prepared.plan)]
            if prepared.plan != planned:
                raise RuntimeError(
                    "reference-baseline execution input differs from accepted plan"
                )
            unit = _turn_unit(planned)
            if unit in turns:
                continue
            call_key = str(planned["call_key"])
            if ledger.attempts_for(call_key):
                raise RuntimeError(
                    "reference-baseline logical call already spent its one attempt"
                )
            # Client construction validates endpoint credentials but performs no
            # HTTP request. Do it before spending the one durable attempt; the
            # ledger reservation still precedes the actual provider call.
            if client is None:
                client = client_factory(generator_endpoint)
            reservation = ledger.reserve(
                call_key,
                record_ids={
                    key: planned[key]
                    for key in (
                        "user_id",
                        "topic_index",
                        "condition",
                        "seed",
                        "simulator_id",
                        "turn_index",
                        "interaction_mode",
                        "track_id",
                    )
                },
                prompt_sha256=str(planned["prompt_hash"]),
            )
            result = None
            try:
                result, _ = client.chat(
                    prepared.messages,
                    temperature=supporter_generation_contract.temperature,
                    max_tokens=supporter_generation_contract.max_output_tokens,
                    seed=int(planned["turn_seed"]),
                    response_schema=None,
                    retries=1,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raw = request_log(
                    stage=PMV22_REFERENCE_BASELINE_STAGE,
                    endpoint=generator_endpoint,
                    messages=prepared.messages,
                    result=None,
                    parsed=None,
                    error=error,
                    prompt_hash=str(planned["prompt_hash"]),
                    record_ids={
                        **{key: planned[key] for key in planned if key in {
                            "user_id", "topic_index", "condition", "seed",
                            "simulator_id", "turn_index", "interaction_mode",
                            "track_id",
                        }},
                        "physical_call_key": call_key,
                        "physical_attempt_index": reservation.attempt_index,
                        "physical_attempt_key": reservation.attempt_key,
                    },
                )
                raw.update(
                    {
                        "supporter_generation_treatment": planned[
                            "supporter_generation_treatment"
                        ],
                        "supporter_generation_treatment_sha256": planned[
                            "supporter_generation_treatment_sha256"
                        ],
                        "fixed_seeker_generation_treatment": planned[
                            "fixed_seeker_generation_treatment"
                        ],
                        "fixed_seeker_generation_treatment_sha256": planned[
                            "fixed_seeker_generation_treatment_sha256"
                        ],
                        "evidence_processing_contract": planned[
                            "evidence_processing_contract"
                        ],
                        "evidence_processing_contract_sha256": planned[
                            "evidence_processing_contract_sha256"
                        ],
                        **{key: planned[key] for key in POLICY_LOCK_FIELDS},
                    }
                )
                append_jsonl(raw_path, raw)
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=None,
                    usage=None,
                    error=error,
                )
                raise

            completion_error = supporter_generation_contract.completion_gate_error(
                normalized_finish_reason=result.normalized_finish_reason,
                provider_finish_reason=result.provider_finish_reason,
            )
            usage_error = reported_prompt_token_error(
                result.usage,
                maximum_prompt_tokens=int(planned["input_tokens_est"]),
                stage="PM-v2.2 reference-baseline supporter generation",
                require_positive=True,
            )
            supporter_message = supporter_generation_contract.normalize_output(
                result.text
            )
            empty_error = (
                None
                if supporter_message
                else "supporter completion is empty after frozen normalization"
            )
            gate_error = completion_error or usage_error or empty_error
            raw = request_log(
                stage=PMV22_REFERENCE_BASELINE_STAGE,
                endpoint=generator_endpoint,
                messages=prepared.messages,
                result=result,
                parsed=None,
                error=gate_error,
                prompt_hash=str(planned["prompt_hash"]),
                record_ids={
                    "user_id": planned["user_id"],
                    "topic_index": planned["topic_index"],
                    "condition": planned["condition"],
                    "seed": planned["seed"],
                    "simulator_id": planned["simulator_id"],
                    "turn_index": planned["turn_index"],
                    "interaction_mode": planned["interaction_mode"],
                    "track_id": planned["track_id"],
                    "physical_call_key": call_key,
                    "physical_attempt_index": reservation.attempt_index,
                    "physical_attempt_key": reservation.attempt_key,
                },
            )
            raw.update(
                {
                    "supporter_generation_treatment": planned[
                        "supporter_generation_treatment"
                    ],
                    "supporter_generation_treatment_sha256": planned[
                        "supporter_generation_treatment_sha256"
                    ],
                    "fixed_seeker_generation_treatment": planned[
                        "fixed_seeker_generation_treatment"
                    ],
                    "fixed_seeker_generation_treatment_sha256": planned[
                        "fixed_seeker_generation_treatment_sha256"
                    ],
                    "evidence_processing_contract": planned[
                        "evidence_processing_contract"
                    ],
                    "evidence_processing_contract_sha256": planned[
                        "evidence_processing_contract_sha256"
                    ],
                    **{key: planned[key] for key in POLICY_LOCK_FIELDS},
                }
            )
            append_jsonl(raw_path, raw)
            if gate_error is not None:
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=result.request_hash,
                    usage=result.usage,
                    error=gate_error,
                    result={
                        "provider_finish_reason": result.provider_finish_reason,
                        "normalized_finish_reason": (
                            result.normalized_finish_reason
                        ),
                        "supporter_generation_treatment_sha256": planned[
                            "supporter_generation_treatment_sha256"
                        ],
                    },
                )
                raise RuntimeError(gate_error)

            memory_tokens = sum(
                estimate_tokens(row.text) for row in prepared.selected_memory
            )
            strategy_tokens = sum(
                estimate_tokens(row.guidance_text + row.example_response)
                for row in prepared.selected_strategy
            )
            candidate_memory_tokens = sum(
                estimate_tokens(row.text) for row in prepared.candidate_memory
            )
            candidate_strategy_tokens = sum(
                estimate_tokens(row.guidance_text + row.example_response)
                for row in prepared.candidate_strategy
            )
            input_price = float(input_usd_per_mtok)
            output_price = float(output_usd_per_mtok)
            api_cost = (
                int(result.usage["prompt_tokens"]) / 1_000_000 * input_price
                + int(result.usage["completion_tokens"]) / 1_000_000
                * output_price
            )
            cost = CostRecord(
                pm_input_tokens_est=0,
                retrieval_calls=prepared.retrieval_calls,
                reranker_calls=0,
                memory_tokens=memory_tokens,
                strategy_tokens=strategy_tokens,
                base_prompt_tokens=max(
                    0, int(planned["raw_input_tokens_est"])
                    - memory_tokens
                    - strategy_tokens,
                ),
                total_input_tokens=int(result.usage["prompt_tokens"]),
                output_tokens=int(result.usage["completion_tokens"]),
                latency_ms=float(result.latency_ms),
                api_cost_usd=api_cost,
                generation_latency_ms=float(result.latency_ms),
                evidence_filter_calls=prepared.evidence_filter_calls,
                candidate_memory_count=len(prepared.candidate_memory),
                kept_memory_count=len(prepared.selected_memory),
                candidate_strategy_count=len(prepared.candidate_strategy),
                kept_strategy_count=len(prepared.selected_strategy),
                candidate_memory_tokens=candidate_memory_tokens,
                candidate_strategy_tokens=candidate_strategy_tokens,
                dropped_memory_tokens=(candidate_memory_tokens - memory_tokens),
                dropped_strategy_tokens=(
                    candidate_strategy_tokens - strategy_tokens
                ),
            )
            turn_record = {
                "user_id": planned["user_id"],
                "topic_index": planned["topic_index"],
                "condition": planned["condition"],
                "seed": planned["seed"],
                "simulator_id": planned["simulator_id"],
                "turn_index": planned["turn_index"],
                "interaction_mode": "fixed",
                "track_id": planned["track_id"],
                "protocol": PMV22_REFERENCE_BASELINE_PROTOCOL,
                "trajectory_comparability": "causal_fixed_context_one_step",
                "state_id": prepared.runtime.state_id,
                "exogenous_state_id": prepared.runtime.provenance[
                    "exogenous_state_id"
                ],
                "card_id": prepared.runtime.card_id,
                "context_before_turn": prepared.context_before_turn,
                "context_sha256": sha256_text(
                    canonical_json(prepared.context_before_turn)
                ),
                "seeker_message": prepared.seeker_message,
                "supporter_message": supporter_message,
                "action_id": planned["requested_action_id"],
                "requested_action_id": planned["requested_action_id"],
                "effective_action_id": planned["effective_action_id"],
                "candidate_memory": _serialize_evidence(
                    prepared.candidate_memory
                ),
                "candidate_strategy": _serialize_evidence(
                    prepared.candidate_strategy
                ),
                "evidence_filter_decision": (
                    prepared.evidence_filter_decision
                ),
                "selected_memory": _serialize_evidence(
                    prepared.selected_memory
                ),
                "selected_strategy": _serialize_evidence(
                    prepared.selected_strategy
                ),
                "cost": cost.model_dump(mode="json"),
                "input_tokens": cost.total_input_tokens,
                "output_tokens": cost.output_tokens,
                "latency_ms": cost.latency_ms,
                "provider_finish_reason": result.provider_finish_reason,
                "normalized_finish_reason": result.normalized_finish_reason,
                "supporter_generation_treatment": planned[
                    "supporter_generation_treatment"
                ],
                "supporter_generation_treatment_sha256": planned[
                    "supporter_generation_treatment_sha256"
                ],
                "fixed_seeker_generation_treatment": planned[
                    "fixed_seeker_generation_treatment"
                ],
                "fixed_seeker_generation_treatment_sha256": planned[
                    "fixed_seeker_generation_treatment_sha256"
                ],
                "evidence_processing_contract": planned[
                    "evidence_processing_contract"
                ],
                "evidence_processing_contract_sha256": planned[
                    "evidence_processing_contract_sha256"
                ],
                **{key: planned[key] for key in POLICY_LOCK_FIELDS},
                "physical_call_key": call_key,
                "physical_attempt_index": reservation.attempt_index,
                "physical_attempt_key": reservation.attempt_key,
            }
            ledger.finish(
                reservation,
                succeeded=True,
                request_hash=result.request_hash,
                usage=result.usage,
                error=None,
                result={"turn_record": turn_record, "request_log": raw},
            )
            append_jsonl(turns_path, turn_record)
            turns[unit] = turn_record
    finally:
        if client is not None:
            client.close()

    if len(turns) != len(call_plan) or not all(
        ledger.succeeded(call_key) for call_key in expected_calls
    ):
        raise RuntimeError("PM-v2.2 reference-baseline generation is incomplete")
    raw_rows = [dict(row) for row in iter_jsonl(raw_path)]
    raw_generation_contract_gate = build_raw_generation_contract_gate(
        raw_rows=raw_rows,
        call_plan=call_plan,
        supporter_generation_treatment=common_contract[
            "supporter_generation_treatment"
        ],
        supporter_generation_treatment_sha256=common_contract[
            "supporter_generation_treatment_sha256"
        ],
        fixed_seeker_generation_treatment=common_contract[
            "fixed_seeker_generation_treatment"
        ],
        fixed_seeker_generation_treatment_sha256=common_contract[
            "fixed_seeker_generation_treatment_sha256"
        ],
        evidence_processing_contracts=common_contract[
            "evidence_processing_contracts"
        ],
        policy_lock={key: common_contract[key] for key in POLICY_LOCK_FIELDS},
    )
    if raw_generation_contract_gate["status"] != "PASS":
        raise RuntimeError(
            "reference-baseline raw-generation contract gate failed: "
            f"{raw_generation_contract_gate}"
        )
    summary = {
        "status": "COMPLETE",
        "stage": PMV22_REFERENCE_BASELINE_STAGE,
        **common_contract,
        "generator_endpoint": generator_endpoint_payload(generator_endpoint),
        "expected_turns": len(call_plan),
        "completed_turns": len(turns),
        "expected_turns_per_condition": int(
            cost_estimate["evaluation_unit_contract"]["expected_unit_count"]
        ),
        "normalized_finish_reason_counts": {"complete": len(turns)},
        "non_complete_finish_reason_count": 0,
        "new_or_historical_physical_http_attempts": ledger.started_attempts,
        "successful_physical_http_attempts": len(turns),
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "maximum_physical_attempts_per_logical_call": 1,
        "cost_estimate_sha256": cost_estimate["cost_estimate_sha256"],
        "call_plan_sha256": cost_estimate["call_plan_sha256"],
        "evidence_filter_config_sha256": evidence_filter_config.digest(),
        "evidence_filter_model": dict(evidence_filter_model_binding),
        "strategy_bank_approval": dict(strategy_bank_approval),
        "raw_generation_contract_gate": raw_generation_contract_gate,
    }
    write_json(summary_path, summary)
    attestation_inputs = {
        "evoemo": evoemo_path,
        "strategy_bank": strategy_bank_path,
        "policy_checkpoint": policy_checkpoint_path,
        "policy_training_report": policy_training_report_path,
        "pm_v2_config": pm_v2_config_path,
        "strategy_bank_approval": strategy_bank_approval_path,
        "fixed_tracks": fixed_tracks_path,
        "fixed_tracks_attestation": fixed_tracks_attestation_path,
        "run_manifest": manifest_path,
        "cost_estimate": out_dir / "cost_estimate.json",
        "call_plan": out_dir / "call_plan.jsonl",
    }
    evidence_filter_paths = {
        "evidence_filter_checkpoint": evidence_filter_checkpoint_path,
        "evidence_filter_report": evidence_filter_report_path,
        "evidence_filter_attestation": evidence_filter_attestation_path,
    }
    if evidence_filter_config.enabled:
        if any(path is None for path in evidence_filter_paths.values()):
            raise RuntimeError(
                "enabled Evidence Filter requires checkpoint, report, and attestation"
            )
        attestation_inputs.update(evidence_filter_paths)
    elif any(path is not None for path in evidence_filter_paths.values()):
        raise RuntimeError(
            "disabled Evidence Filter must not bind nonexistent or unrelated model artifacts"
        )

    create_artifact_attestation(
        attestation_path,
        stage=PMV22_REFERENCE_BASELINE_STAGE,
        inputs=attestation_inputs,
        outputs={
            "turns": (turns_path, True),
            "raw_calls": (raw_path, True),
            "physical_attempt_ledger": (ledger_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            **common_contract,
            "generator_endpoint": generator_endpoint_payload(
                generator_endpoint
            ),
            "generator_endpoint_sha256": sha256_text(
                canonical_json(generator_endpoint_payload(generator_endpoint))
            ),
            "evidence_filter_config_sha256": evidence_filter_config.digest(),
            "evidence_filter_model": dict(evidence_filter_model_binding),
            "strategy_bank_approval": dict(strategy_bank_approval),
            "retrieval_settings": manifest["retrieval_settings"],
            "cost_estimate_sha256": cost_estimate["cost_estimate_sha256"],
            "call_plan_sha256": cost_estimate["call_plan_sha256"],
            "generator_retries": 1,
            "generator_pricing_usd_per_mtok": {
                "input": float(input_usd_per_mtok),
                "output": float(output_usd_per_mtok),
            },
            "input_token_safety_factor": float(input_token_safety_factor),
            "fail_on_reported_input_overrun": True,
            "physical_attempt_ledger_protocol": (
                PHYSICAL_ATTEMPT_LEDGER_PROTOCOL
            ),
            "maximum_physical_http_attempts_authorized": len(expected_calls),
            "raw_generation_contract_gate": raw_generation_contract_gate,
        },
        expected={
            "turns": len(call_plan),
            "turns_per_condition": int(
                cost_estimate["evaluation_unit_contract"]["expected_unit_count"]
            ),
            "conditions": sorted(REFERENCE_BASELINE_CONDITIONS),
            "accepted_normalized_finish_reasons": ["complete"],
            "non_complete_finish_reason_count": 0,
            "maximum_physical_attempts_per_logical_call": 1,
            "raw_generation_contract_gate": raw_generation_contract_gate,
        },
    )
    return summary
