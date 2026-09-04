from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import time

from .api import (
    Endpoint,
    OpenAICompatibleClient,
    request_log,
    require_reported_usage,
)
from .attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
)
from .artifacts import create_artifact_attestation
from .contracts import (
    ActionOutcome,
    CostRecord,
    MemoryBackendRecord,
    MemorySource,
    RuntimeState,
    StrategyCard,
    StrategyMode,
    parse_action_id,
)
from .evidence_filter import EvidenceFilterConfig, filter_evidence
from .evidence_filter_model import PMV2EvidenceFilterModel
from .generation_contract import SupporterGenerationContract
from .io import (
    append_jsonl,
    iter_jsonl,
    load_done_keys,
    sha256_file,
    sha256_text,
    canonical_json,
    write_json,
    utc_now,
    ensure_run_manifest,
)
from .prompts import generation_messages, BASE_SUPPORTER_SYSTEM
from .retrieval import MemoryRetriever, StrategyRetriever, context_query
from .text import conservative_token_bound, estimate_tokens
from .sampling import select_stratified_card_ids


def _resolve_supporter_generation_treatment(
    *,
    supporter_generation_contract: SupporterGenerationContract | None,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Resolve legacy or PM-v2.2 generation settings without silent drift."""

    if supporter_generation_contract is None:
        return system_prompt or BASE_SUPPORTER_SYSTEM, None, None

    contract = supporter_generation_contract
    if system_prompt is not None and system_prompt != contract.system_prompt:
        raise ValueError(
            "supporter system prompt differs from the PM-v2.2 generation contract"
        )
    if float(temperature) != contract.temperature:
        raise ValueError(
            "supporter temperature differs from the PM-v2.2 generation contract"
        )
    if int(max_tokens) != contract.max_output_tokens:
        raise ValueError(
            "supporter max_tokens differs from the PM-v2.2 generation contract"
        )
    return contract.system_prompt, contract.payload(), contract.digest()


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def load_states(path: str | Path) -> dict[str, RuntimeState]:
    return {
        row["card_id"]: RuntimeState.model_validate(row)
        for row in iter_jsonl(path)
    }


def load_backends(path: str | Path) -> dict[str, MemoryBackendRecord]:
    return {
        row["card_id"]: MemoryBackendRecord.model_validate(row)
        for row in iter_jsonl(path)
    }


def load_strategy_cards(path: str | Path) -> list[StrategyCard]:
    return [StrategyCard.model_validate(row) for row in iter_jsonl(path)]


def _materialize_prompt_equivalent_outcome(
    *,
    canonical: ActionOutcome,
    prepared: dict[str, Any],
    action_id: str,
    system_prompt: str,
) -> ActionOutcome:
    """Fan one paid response out to a logically distinct requested action."""

    state = prepared["state"]
    query = prepared["query"]
    sources = prepared["sources"]
    strategy_mode = prepared["strategy_mode"]
    memory_view = prepared["memory_view"]
    strategy_view = prepared["strategy_view"]
    candidate_memory_view = prepared["candidate_memory_view"]
    candidate_strategy_view = prepared["candidate_strategy_view"]
    filter_decision = prepared["evidence_filter_decision"]
    base_tokens = estimate_tokens(
        state.current_user_text
        + state.current_session_summary
        + "\n".join(row.content for row in state.current_session_history)
        + system_prompt
    )
    memory_tokens = sum(estimate_tokens(row.text) for row in memory_view)
    strategy_tokens = sum(
        estimate_tokens(row.guidance_text + row.example_response)
        for row in strategy_view
    )
    candidate_memory_tokens = sum(
        estimate_tokens(row.text) for row in candidate_memory_view
    )
    candidate_strategy_tokens = sum(
        estimate_tokens(row.guidance_text + row.example_response)
        for row in candidate_strategy_view
    )
    cost = CostRecord(
        pm_input_tokens_est=estimate_tokens(query)
        + sum(len(catalog.catalog_fingerprint) for catalog in state.inventory.values()),
        retrieval_calls=len(sources)
        + (1 if strategy_mode is StrategyMode.RS else 0),
        reranker_calls=0,
        memory_tokens=memory_tokens,
        strategy_tokens=strategy_tokens,
        base_prompt_tokens=base_tokens,
        total_input_tokens=canonical.cost.total_input_tokens,
        output_tokens=canonical.cost.output_tokens,
        latency_ms=canonical.cost.latency_ms,
        api_cost_usd=None,
        evidence_filter_calls=(1 if filter_decision is not None else 0),
        candidate_memory_count=len(candidate_memory_view),
        kept_memory_count=len(memory_view),
        candidate_strategy_count=len(candidate_strategy_view),
        kept_strategy_count=len(strategy_view),
        candidate_memory_tokens=candidate_memory_tokens,
        candidate_strategy_tokens=candidate_strategy_tokens,
        dropped_memory_tokens=candidate_memory_tokens - memory_tokens,
        dropped_strategy_tokens=candidate_strategy_tokens - strategy_tokens,
    )
    return ActionOutcome(
        card_id=canonical.card_id,
        state_id=state.state_id,
        user_id=state.user_id,
        action_id=action_id,
        response=canonical.response,
        selected_memory_ids=[row.memory_id for row in memory_view],
        selected_strategy_ids=[row.strategy_id for row in strategy_view],
        memory_view=memory_view,
        strategy_view=strategy_view,
        effective_action_id=prepared["effective_action_id"],
        candidate_memory_view=candidate_memory_view,
        candidate_strategy_view=candidate_strategy_view,
        evidence_filter_decision=filter_decision,
        cost=cost,
        model_name=canonical.model_name,
        prompt_hash=canonical.prompt_hash,
        request_hash=canonical.request_hash,
        provenance={
            **canonical.provenance,
            "prompt_equivalence_alias": action_id != canonical.action_id,
            "prompt_equivalence_canonical_action_id": canonical.action_id,
        },
    )


def plan_action_sweep(
    runtime_path: str | Path,
    backend_path: str | Path,
    strategy_bank_path: str | Path,
    *,
    endpoint: Endpoint,
    max_cards: int | None = None,
    card_filter: set[str] | None = None,
    action_filter: set[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    seed: int | None = 4311,
    request_retries: int = 3,
    fail_fast: bool = False,
    input_token_safety_factor: float = 1.0,
    fail_on_reported_input_overrun: bool = False,
    strategy_top_k: int = 3,
    memory_min_score: float | None = None,
    strategy_min_score: float | None = None,
    evidence_filter_config: EvidenceFilterConfig | None = None,
    memory_helpfulness_model: PMV2EvidenceFilterModel | None = None,
    system_prompt: str | None = None,
    supporter_generation_contract: SupporterGenerationContract | None = None,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    contract_bindings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build an exact no-API action-sweep call plan and immutable cost hash."""

    if request_retries < 1:
        raise ValueError("request_retries must be positive")
    (
        system_prompt,
        generation_treatment,
        generation_treatment_sha256,
    ) = _resolve_supporter_generation_treatment(
        supporter_generation_contract=supporter_generation_contract,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    runtime_path = Path(runtime_path)
    backend_path = Path(backend_path)
    strategy_bank_path = Path(strategy_bank_path)
    states = load_states(runtime_path)
    backends = load_backends(backend_path)
    if set(states) != set(backends):
        raise ValueError("runtime/backend card IDs differ")
    cards = list(states)
    if card_filter is not None:
        if max_cards is not None:
            raise ValueError("card_filter and max_cards are mutually exclusive")
        unknown_cards = sorted(set(card_filter) - set(states))
        if unknown_cards:
            raise ValueError(f"card_filter contains unknown cards: {unknown_cards[:5]}")
        cards = [card_id for card_id in cards if card_id in card_filter]
    if max_cards is not None:
        cards = select_stratified_card_ids(states, max_cards)

    strategies = load_strategy_cards(strategy_bank_path)
    if not strategies:
        raise ValueError("strategy bank is empty")
    strategy_retriever = StrategyRetriever(
        strategies, top_k=strategy_top_k, minimum_score=strategy_min_score
    )
    memory_retriever = MemoryRetriever(
        minimum_score_by_source=(
            {source: float(memory_min_score) for source in MemorySource}
            if memory_min_score is not None
            else None
        )
    )
    rows: list[dict[str, Any]] = []
    for card_id in cards:
        state = states[card_id]
        backend = backends[card_id]
        query = context_query(
            state.current_user_text,
            [item.model_dump(mode="json") for item in state.current_session_history],
            state.current_session_summary,
        )
        for action_id in state.allowed_actions:
            if action_filter is not None and action_id not in action_filter:
                continue
            sources, strategy_mode = parse_action_id(action_id)
            candidate_memory_view = memory_retriever.retrieve(
                query, backend.items, sources
            )
            candidate_strategy_view = (
                strategy_retriever.retrieve(query)
                if strategy_mode is StrategyMode.RS
                else []
            )
            if evidence_filter_config is not None:
                filtered = filter_evidence(
                    requested_action_id=action_id,
                    current_user_text=state.current_user_text,
                    context_query_text=query,
                    memory_candidates=candidate_memory_view,
                    strategy_candidates=candidate_strategy_view,
                    config=evidence_filter_config,
                    session_index=state.session_index,
                    memory_helpfulness_model=memory_helpfulness_model,
                )
                memory_view = filtered.memory_view
                strategy_view = filtered.strategy_view
                filter_decision = filtered.decision
            else:
                memory_view = candidate_memory_view
                strategy_view = candidate_strategy_view
                filter_decision = None
            messages = generation_messages(
                state, memory_view, strategy_view, system_prompt=system_prompt
            )
            prompt_sha256 = sha256_text(canonical_json(messages))
            record_ids = {
                "card_id": card_id,
                "prompt_equivalence_sha256": prompt_sha256,
            }
            if generation_treatment_sha256 is not None:
                record_ids["supporter_generation_treatment_sha256"] = (
                    generation_treatment_sha256
                )
            request_parameters = {
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "seed": seed,
                "response_schema": None,
            }
            if generation_treatment_sha256 is not None:
                request_parameters["supporter_generation_treatment_sha256"] = (
                    generation_treatment_sha256
                )
            call_key = physical_call_key(
                stage="action_sweep_generation",
                record_ids=record_ids,
                prompt_sha256=prompt_sha256,
                endpoint=endpoint,
                request_parameters=request_parameters,
            )
            messages_json = canonical_json(messages)
            raw_input_tokens = estimate_tokens(messages_json)
            input_tokens = conservative_token_bound(
                messages_json, safety_factor=float(input_token_safety_factor)
            )
            plan_row = {
                "card_id": card_id,
                "state_id": state.state_id,
                "action_id": action_id,
                "requested_action_id": action_id,
                "effective_action_id": (
                    filter_decision.effective_action_id
                    if filter_decision is not None
                    else action_id
                ),
                "candidate_memory_count": len(candidate_memory_view),
                "kept_memory_count": len(memory_view),
                "candidate_strategy_count": len(candidate_strategy_view),
                "kept_strategy_count": len(strategy_view),
                "evidence_filter_config_sha256": (
                    filter_decision.config_sha256
                    if filter_decision is not None
                    else None
                ),
                "prompt_sha256": prompt_sha256,
                "call_key": call_key,
                "max_http_attempts": int(request_retries),
                "raw_estimated_input_tokens": raw_input_tokens,
                "estimated_input_tokens": input_tokens,
                "maximum_output_tokens": int(max_tokens),
            }
            if generation_treatment is not None:
                plan_row.update(
                    {
                        "supporter_generation_treatment": generation_treatment,
                        "supporter_generation_treatment_sha256": (
                            generation_treatment_sha256
                        ),
                    }
                )
            rows.append(plan_row)

    equivalence_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        equivalence_groups.setdefault(str(row["call_key"]), []).append(row)
    for group in equivalence_groups.values():
        actions = sorted(str(row["action_id"]) for row in group)
        equivalence_id = sha256_text(
            canonical_json(
                {
                    "card_id": group[0]["card_id"],
                    "prompt_sha256": group[0]["prompt_sha256"],
                    "actions": actions,
                }
            )
        )
        for row in group:
            row["prompt_equivalence_id"] = equivalence_id
            row["prompt_equivalence_actions"] = actions
            row["prompt_equivalence_class_size"] = len(group)

    physical_rows = [group[0] for group in equivalence_groups.values()]
    input_tokens = [int(row["estimated_input_tokens"]) for row in physical_rows]
    undeduplicated_input_tokens = [
        int(row["estimated_input_tokens"]) for row in rows
    ]
    total_input = sum(input_tokens)
    total_output = len(physical_rows) * int(max_tokens)
    logical_cost_usd = (
        total_input / 1_000_000 * float(input_usd_per_mtok)
        + total_output / 1_000_000 * float(output_usd_per_mtok)
    )
    maximum_physical_attempts = len(physical_rows) * int(request_retries)
    payload = {
        "protocol": "pm_v2_action_sweep_cost_v3_persistent_attempt_ledger",
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "endpoint": {
            "base_url": endpoint.base_url,
            "model": endpoint.model,
            "family": endpoint.family,
        },
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "seed": seed,
        "request_retries": int(request_retries),
        "fail_fast": bool(fail_fast),
        "input_token_safety_factor": float(input_token_safety_factor),
        "fail_on_reported_input_overrun": bool(fail_on_reported_input_overrun),
        "strategy_top_k": int(strategy_top_k),
        "memory_min_score": memory_min_score,
        "strategy_min_score": strategy_min_score,
        "evidence_filter": (
            evidence_filter_config.payload()
            if evidence_filter_config is not None
            else None
        ),
        "evidence_filter_config_sha256": (
            evidence_filter_config.digest()
            if evidence_filter_config is not None
            else None
        ),
        "evidence_filter_model": (
            {
                "checkpoint_sha256": memory_helpfulness_model.checkpoint_sha256,
                "model_contract_sha256": memory_helpfulness_model.contract_hash(),
            }
            if memory_helpfulness_model is not None
            else None
        ),
        "max_cards": max_cards,
        "card_filter": sorted(card_filter) if card_filter else None,
        "action_filter": sorted(action_filter) if action_filter else None,
        "system_prompt_sha256": sha256_text(system_prompt),
        "expected_api_calls": len(physical_rows),
        "logical_api_calls": len(rows),
        "prompt_equivalence_protocol": "pm-v2-prompt-equivalence-v1",
        "prompt_equivalence_classes": len(physical_rows),
        "prompt_alias_outcomes": len(rows) - len(physical_rows),
        "prompt_alias_rate": (
            (len(rows) - len(physical_rows)) / len(rows) if rows else 0.0
        ),
        "maximum_physical_api_attempts": maximum_physical_attempts,
        "estimated_total_input_tokens": total_input,
        "maximum_total_output_tokens": total_output,
        "unduplicated_logical_total_input_tokens": sum(
            undeduplicated_input_tokens
        ),
        "unduplicated_logical_maximum_output_tokens": len(rows)
        * int(max_tokens),
        "maximum_physical_total_input_tokens": total_input
        * int(request_retries),
        "maximum_physical_total_output_tokens": total_output
        * int(request_retries),
        "mean_input_tokens": (
            float(total_input / len(input_tokens)) if input_tokens else 0.0
        ),
        "p95_input_tokens": _percentile(input_tokens, 0.95),
        "max_input_tokens": max(input_tokens, default=0),
        "pricing": {
            "input_usd_per_mtok": float(input_usd_per_mtok),
            "output_usd_per_mtok": float(output_usd_per_mtok),
        },
        "logical_estimated_cost_usd": logical_cost_usd,
        "estimated_cost_usd": logical_cost_usd * int(request_retries),
        "cost_basis": "maximum_physical_api_attempts",
        "call_plan_sha256": sha256_text(canonical_json(rows)),
        "contract_bindings": dict(contract_bindings or {}),
    }
    if generation_treatment is not None:
        payload.update(
            {
                "supporter_generation_treatment": generation_treatment,
                "supporter_generation_treatment_sha256": (
                    generation_treatment_sha256
                ),
            }
        )
    payload["cost_estimate_sha256"] = sha256_text(canonical_json(payload))
    return payload, rows


def run_action_sweep(
    runtime_path: str | Path,
    backend_path: str | Path,
    strategy_bank_path: str | Path,
    out_outcomes_path: str | Path,
    out_raw_calls_path: str | Path,
    out_summary_path: str | Path,
    *,
    endpoint: Endpoint,
    max_cards: int | None = None,
    card_filter: set[str] | None = None,
    action_filter: set[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 300,
    seed: int | None = 4311,
    request_retries: int = 3,
    fail_fast: bool = False,
    input_token_safety_factor: float = 1.0,
    fail_on_reported_input_overrun: bool = False,
    strategy_top_k: int = 3,
    memory_min_score: float | None = None,
    strategy_min_score: float | None = None,
    evidence_filter_config: EvidenceFilterConfig | None = None,
    memory_helpfulness_model: PMV2EvidenceFilterModel | None = None,
    overwrite: bool = False,
    system_prompt: str | None = None,
    supporter_generation_contract: SupporterGenerationContract | None = None,
    study_freeze_sha256: str | None = None,
    contract_bindings: dict[str, Any] | None = None,
    max_physical_api_attempts: int | None = None,
    out_attempt_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    if request_retries < 1:
        raise ValueError("request_retries must be positive")
    (
        system_prompt,
        generation_treatment,
        generation_treatment_sha256,
    ) = _resolve_supporter_generation_treatment(
        supporter_generation_contract=supporter_generation_contract,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    runtime_path = Path(runtime_path)
    backend_path = Path(backend_path)
    strategy_bank_path = Path(strategy_bank_path)
    out_outcomes_path = Path(out_outcomes_path)
    out_raw_calls_path = Path(out_raw_calls_path)
    out_summary_path = Path(out_summary_path)
    output_dir = out_summary_path.parent
    manifest_path = output_dir / "run_manifest.json"
    attestation_path = output_dir / "artifact_attestation.json"
    attempt_ledger_path = Path(
        out_attempt_ledger_path or output_dir / "physical_attempt_ledger.jsonl"
    )
    forbid_overwrite_of_spent_attempts(
        attempt_ledger_path,
        overwrite=overwrite,
        stage="action sweep",
    )
    if overwrite:
        for path in (
            out_outcomes_path,
            out_raw_calls_path,
            out_summary_path,
            manifest_path,
            attestation_path,
            attempt_ledger_path,
        ):
            if path.exists():
                path.unlink()

    states = load_states(runtime_path)
    backends = load_backends(backend_path)
    if set(states) != set(backends):
        raise ValueError("runtime/backend card IDs differ")
    cards = list(states)
    if card_filter is not None:
        if max_cards is not None:
            raise ValueError("card_filter and max_cards are mutually exclusive")
        unknown_cards = sorted(set(card_filter) - set(states))
        if unknown_cards:
            raise ValueError(f"card_filter contains unknown cards: {unknown_cards[:5]}")
        cards = [card_id for card_id in cards if card_id in card_filter]
    if max_cards is not None:
        cards = select_stratified_card_ids(states, max_cards)
    strategies = load_strategy_cards(strategy_bank_path)
    if not strategies:
        raise ValueError("strategy bank is empty")
    strategy_retriever = StrategyRetriever(
        strategies, top_k=strategy_top_k, minimum_score=strategy_min_score
    )
    memory_retriever = MemoryRetriever(
        minimum_score_by_source=(
            {source: float(memory_min_score) for source in MemorySource}
            if memory_min_score is not None
            else None
        )
    )

    prepared_calls: dict[tuple[str, str], dict[str, Any]] = {}
    for card_id in cards:
        state = states[card_id]
        backend = backends[card_id]
        query = context_query(
            state.current_user_text,
            [x.model_dump(mode="json") for x in state.current_session_history],
            state.current_session_summary,
        )
        for action_id in state.allowed_actions:
            if action_filter is not None and action_id not in action_filter:
                continue
            sources, strategy_mode = parse_action_id(action_id)
            candidate_memory_view = memory_retriever.retrieve(
                query, backend.items, sources
            )
            candidate_strategy_view = (
                strategy_retriever.retrieve(query)
                if strategy_mode is StrategyMode.RS
                else []
            )
            if evidence_filter_config is not None:
                filtered = filter_evidence(
                    requested_action_id=action_id,
                    current_user_text=state.current_user_text,
                    context_query_text=query,
                    memory_candidates=candidate_memory_view,
                    strategy_candidates=candidate_strategy_view,
                    config=evidence_filter_config,
                    session_index=state.session_index,
                    memory_helpfulness_model=memory_helpfulness_model,
                )
                memory_view = filtered.memory_view
                strategy_view = filtered.strategy_view
                filter_decision = filtered.decision
            else:
                memory_view = candidate_memory_view
                strategy_view = candidate_strategy_view
                filter_decision = None
            messages = generation_messages(
                state, memory_view, strategy_view, system_prompt=system_prompt
            )
            messages_json = canonical_json(messages)
            prompt_hash = sha256_text(messages_json)
            raw_input_tokens = estimate_tokens(messages_json)
            input_token_upper_bound = conservative_token_bound(
                messages_json, safety_factor=float(input_token_safety_factor)
            )
            record_ids = {
                "card_id": card_id,
                "prompt_equivalence_sha256": prompt_hash,
            }
            if generation_treatment_sha256 is not None:
                record_ids["supporter_generation_treatment_sha256"] = (
                    generation_treatment_sha256
                )
            request_parameters = {
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "seed": seed,
                "response_schema": None,
            }
            if generation_treatment_sha256 is not None:
                request_parameters["supporter_generation_treatment_sha256"] = (
                    generation_treatment_sha256
                )
            call_key = physical_call_key(
                stage="action_sweep_generation",
                record_ids=record_ids,
                prompt_sha256=prompt_hash,
                endpoint=endpoint,
                request_parameters=request_parameters,
            )
            prepared_calls[(card_id, action_id)] = {
                "state": state,
                "query": query,
                "sources": sources,
                "strategy_mode": strategy_mode,
                "memory_view": memory_view,
                "strategy_view": strategy_view,
                "candidate_memory_view": candidate_memory_view,
                "candidate_strategy_view": candidate_strategy_view,
                "evidence_filter_decision": filter_decision,
                "effective_action_id": (
                    filter_decision.effective_action_id
                    if filter_decision is not None
                    else action_id
                ),
                "messages": messages,
                "prompt_hash": prompt_hash,
                "raw_estimated_input_tokens": raw_input_tokens,
                "input_token_upper_bound": input_token_upper_bound,
                "record_ids": record_ids,
                "call_key": call_key,
            }
    expected_keys = set(prepared_calls)
    physical_call_keys = {
        str(row["call_key"]) for row in prepared_calls.values()
    }
    planned_maximum_attempts = len(physical_call_keys) * int(request_retries)
    runtime_attempt_cap = int(
        planned_maximum_attempts
        if max_physical_api_attempts is None
        else max_physical_api_attempts
    )
    if runtime_attempt_cap < planned_maximum_attempts:
        raise RuntimeError(
            "runtime physical-attempt cap is below the frozen action-sweep plan"
        )
    run_metadata = {
        "stage": "action_sweep",
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "endpoint_model": endpoint.model,
        "endpoint_base_url": endpoint.base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "request_retries": int(request_retries),
        "fail_fast": bool(fail_fast),
        "input_token_safety_factor": float(input_token_safety_factor),
        "fail_on_reported_input_overrun": bool(fail_on_reported_input_overrun),
        "strategy_top_k": strategy_top_k,
        "memory_min_score": memory_min_score,
        "strategy_min_score": strategy_min_score,
        "evidence_filter": (
            evidence_filter_config.payload()
            if evidence_filter_config is not None
            else None
        ),
        "evidence_filter_config_sha256": (
            evidence_filter_config.digest()
            if evidence_filter_config is not None
            else None
        ),
        "evidence_filter_model": (
            {
                "checkpoint_sha256": memory_helpfulness_model.checkpoint_sha256,
                "model_contract_sha256": memory_helpfulness_model.contract_hash(),
            }
            if memory_helpfulness_model is not None
            else None
        ),
        "action_filter": sorted(action_filter) if action_filter else None,
        "card_filter": sorted(card_filter) if card_filter else None,
        "system_prompt_sha256": sha256_text(system_prompt),
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "planned_maximum_physical_api_attempts": planned_maximum_attempts,
        "runtime_maximum_physical_api_attempts": runtime_attempt_cap,
        "call_key_matrix_sha256": sha256_text(
            canonical_json(sorted(physical_call_keys))
        ),
        "logical_outcomes": len(prepared_calls),
        "prompt_equivalence_classes": len(physical_call_keys),
        "prompt_alias_outcomes": len(prepared_calls) - len(physical_call_keys),
        "contract_bindings": dict(contract_bindings or {}),
    }
    if generation_treatment is not None:
        run_metadata.update(
            {
                "supporter_generation_treatment": generation_treatment,
                "supporter_generation_treatment_sha256": (
                    generation_treatment_sha256
                ),
            }
        )
    if study_freeze_sha256 is not None:
        run_metadata["study_freeze_sha256"] = study_freeze_sha256
    ensure_run_manifest(manifest_path, run_metadata, overwrite=False)

    done: set[tuple[Any, Any]] = set()
    duplicates: list[tuple[Any, Any]] = []
    extras: list[tuple[Any, Any]] = []
    if out_outcomes_path.exists():
        for row in iter_jsonl(out_outcomes_path):
            key = (row.get("card_id"), row.get("action_id"))
            if key in done:
                duplicates.append(key)
            done.add(key)
            if key not in expected_keys:
                extras.append(key)
            expected_call = prepared_calls.get(key)
            if expected_call is not None and (
                (row.get("provenance") or {}).get("physical_call_key")
                != expected_call["call_key"]
            ):
                extras.append(key)
    if duplicates or extras:
        raise RuntimeError(
            "action sweep output is incompatible with immutable run manifest; "
            f"duplicates={duplicates[:5]}, extras={extras[:5]}. "
            "Use a new directory or --overwrite."
        )
    ledger = PersistentAttemptLedger(
        attempt_ledger_path,
        stage="action_sweep_generation",
        expected_calls={
            row["call_key"]: int(request_retries)
            for row in prepared_calls.values()
        },
        maximum_total_attempts=runtime_attempt_cap,
    )
    successful_raw_call_keys: set[str] = set()
    if out_raw_calls_path.exists():
        duplicate_success_raw: list[str] = []
        for row in iter_jsonl(out_raw_calls_path):
            if row.get("error") is not None:
                continue
            call_key = str(row.get("physical_call_key") or "")
            if not call_key:
                raise RuntimeError("successful action-sweep raw row lacks physical_call_key")
            if call_key in successful_raw_call_keys:
                duplicate_success_raw.append(call_key)
            successful_raw_call_keys.add(call_key)
        if duplicate_success_raw:
            raise RuntimeError(
                "action-sweep raw log contains duplicate successful calls: "
                f"{duplicate_success_raw[:5]}"
            )

    # Reconcile the append-only paid-attempt ledger before considering any new
    # HTTP request.  The SUCCEEDED event is the canonical crash-recovery source;
    # derived JSONL files may safely be materialized again without repaying.
    for key, prepared in prepared_calls.items():
        call_key = str(prepared["call_key"])
        if key in done:
            if not ledger.succeeded(call_key):
                raise RuntimeError(
                    f"persisted action outcome lacks a successful ledger event: {key}"
                )
            if call_key not in successful_raw_call_keys:
                terminal = ledger.terminal_row(call_key)
                raw_payload = ((terminal or {}).get("result") or {}).get(
                    "raw_call"
                )
                if not isinstance(raw_payload, dict):
                    raise RuntimeError(
                        "persisted action outcome lacks its successful raw-call "
                        f"recovery payload: {key}"
                    )
                if (
                    raw_payload.get("error") is not None
                    or str(raw_payload.get("physical_call_key") or "")
                    != call_key
                ):
                    raise RuntimeError(
                        f"action-sweep raw recovery payload mismatches plan: {key}"
                    )
                append_jsonl(out_raw_calls_path, raw_payload)
                successful_raw_call_keys.add(call_key)
            continue
        if not ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key)
        terminal_result = (terminal or {}).get("result") or {}
        outcome_payload = terminal_result.get("action_outcome")
        raw_payload = terminal_result.get("raw_call")
        if not isinstance(outcome_payload, dict) or not isinstance(raw_payload, dict):
            raise RuntimeError(
                "successful action-sweep ledger event lacks crash-recovery payload "
                f"for {key}"
            )
        canonical_outcome = ActionOutcome.model_validate(outcome_payload)
        if (
            canonical_outcome.card_id != key[0]
            or canonical_outcome.prompt_hash != prepared["prompt_hash"]
            or (canonical_outcome.provenance or {}).get("physical_call_key")
            != call_key
        ):
            raise RuntimeError(
                f"successful action-sweep recovery payload mismatches plan: {key}"
            )
        outcome = (
            canonical_outcome
            if canonical_outcome.action_id == key[1]
            else _materialize_prompt_equivalent_outcome(
                canonical=canonical_outcome,
                prepared=prepared,
                action_id=str(key[1]),
                system_prompt=system_prompt,
            )
        )
        raw_call_key = str(raw_payload.get("physical_call_key") or "")
        if raw_payload.get("error") is not None or raw_call_key != call_key:
            raise RuntimeError(
                f"successful action-sweep raw recovery payload is invalid: {key}"
            )
        if call_key not in successful_raw_call_keys:
            append_jsonl(out_raw_calls_path, raw_payload)
            successful_raw_call_keys.add(call_key)
        append_jsonl(out_outcomes_path, outcome.model_dump(mode="json"))
        done.add(key)
    historical_attempts = ledger.started_attempts
    client: OpenAICompatibleClient | None = None
    n_calls = 0
    failures: list[dict[str, Any]] = [
        {
            **dict(row.get("record_ids") or {}),
            "call_key": row["call_key"],
            "attempt_index": row["attempt_index"],
            "error": row.get("error"),
            "historical": True,
        }
        for row in ledger.failures()
    ]
    aborted_on_first_failure = False
    aborted_on_input_token_overrun = False
    aborted_on_missing_reported_usage = False
    aborted_on_completion_gate = False
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        for (card_id, action_id), prepared in prepared_calls.items():
            if (card_id, action_id) in done:
                continue
            call_key = str(prepared["call_key"])
            if ledger.succeeded(call_key):
                terminal = ledger.terminal_row(call_key)
                canonical_payload = ((terminal or {}).get("result") or {}).get(
                    "action_outcome"
                )
                if not isinstance(canonical_payload, dict):
                    failures.append(
                        {
                            "card_id": card_id,
                            "action_id": action_id,
                            "call_key": call_key,
                            "error": "successful physical prompt lacks recovery outcome",
                        }
                    )
                    if fail_fast:
                        aborted_on_first_failure = True
                        break
                    continue
                canonical_outcome = ActionOutcome.model_validate(canonical_payload)
                alias_outcome = _materialize_prompt_equivalent_outcome(
                    canonical=canonical_outcome,
                    prepared=prepared,
                    action_id=action_id,
                    system_prompt=system_prompt,
                )
                append_jsonl(
                    out_outcomes_path, alias_outcome.model_dump(mode="json")
                )
                done.add((card_id, action_id))
                continue
            if ledger.exhausted(call_key):
                failures.append(
                    {
                        "card_id": card_id,
                        "action_id": action_id,
                        "call_key": call_key,
                        "error": "physical-attempt bound already exhausted",
                    }
                )
                if fail_fast:
                    aborted_on_first_failure = True
                    break
                continue
            if fail_fast and failures:
                aborted_on_first_failure = True
                break
            # Resolve credentials and construct the HTTP client before the
            # durable STARTED reservation. Missing local configuration is not
            # a physical API attempt and must not consume the frozen budget.
            if client is None:
                client = OpenAICompatibleClient(endpoint)
            reservation = ledger.reserve(
                call_key,
                record_ids=prepared["record_ids"],
                prompt_sha256=prepared["prompt_hash"],
            )
            n_calls += 1
            result = None
            try:
                result, _ = client.chat(
                    prepared["messages"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    response_schema=None,
                    retries=1,
                )
                try:
                    reported_usage = require_reported_usage(
                        result.usage, stage="action sweep generation"
                    )
                except RuntimeError as usage_exc:
                    error = f"{type(usage_exc).__name__}: {usage_exc}"
                    append_jsonl(
                        out_raw_calls_path,
                        request_log(
                            stage="generation",
                            endpoint=endpoint,
                            messages=prepared["messages"],
                            result=result,
                            parsed=None,
                            error=error,
                            prompt_hash=prepared["prompt_hash"],
                            record_ids={
                                **prepared["record_ids"],
                                "physical_call_key": call_key,
                                "physical_attempt_index": reservation.attempt_index,
                                "physical_attempt_key": reservation.attempt_key,
                            },
                        ),
                    )
                    ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=error,
                    )
                    failures.append(
                        {
                            "card_id": card_id,
                            "action_id": action_id,
                            "call_key": call_key,
                            "attempt_index": reservation.attempt_index,
                            "error": error,
                            "missing_reported_usage": True,
                        }
                    )
                    aborted_on_missing_reported_usage = True
                    if fail_fast:
                        aborted_on_first_failure = True
                        break
                    continue
                successful_raw_payload = request_log(
                    stage="generation",
                    endpoint=endpoint,
                    messages=prepared["messages"],
                    result=result,
                    parsed=None,
                    error=None,
                    prompt_hash=prepared["prompt_hash"],
                    record_ids={
                        **prepared["record_ids"],
                        "physical_call_key": call_key,
                        "physical_attempt_index": reservation.attempt_index,
                        "physical_attempt_key": reservation.attempt_key,
                    },
                )
                completion_error = (
                    supporter_generation_contract.completion_gate_error(
                        normalized_finish_reason=result.normalized_finish_reason,
                        provider_finish_reason=result.provider_finish_reason,
                    )
                    if supporter_generation_contract is not None
                    else None
                )
                if completion_error is not None:
                    failed_raw_payload = {
                        **successful_raw_payload,
                        "error": completion_error,
                    }
                    append_jsonl(out_raw_calls_path, failed_raw_payload)
                    ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=result.request_hash,
                        usage=reported_usage,
                        error=completion_error,
                    )
                    failures.append(
                        {
                            "card_id": card_id,
                            "action_id": action_id,
                            "call_key": call_key,
                            "attempt_index": reservation.attempt_index,
                            "error": completion_error,
                            "supporter_completion_gate_rejected": True,
                            "provider_finish_reason": result.provider_finish_reason,
                            "normalized_finish_reason": (
                                result.normalized_finish_reason
                            ),
                        }
                    )
                    aborted_on_completion_gate = True
                    if fail_fast:
                        aborted_on_first_failure = True
                        break
                    continue
                reported_prompt_tokens = reported_usage["prompt_tokens"]
                if (
                    fail_on_reported_input_overrun
                    and reported_prompt_tokens
                    > int(prepared["input_token_upper_bound"])
                ):
                    error = (
                        "reported prompt_tokens exceed the frozen conservative "
                        f"bound: reported={reported_prompt_tokens}, "
                        f"bound={prepared['input_token_upper_bound']}"
                    )
                    failed_raw_payload = {
                        **successful_raw_payload,
                        "error": error,
                    }
                    append_jsonl(out_raw_calls_path, failed_raw_payload)
                    ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=result.request_hash,
                        usage=reported_usage,
                        error=error,
                    )
                    failures.append(
                        {
                            "card_id": card_id,
                            "action_id": action_id,
                            "call_key": call_key,
                            "attempt_index": reservation.attempt_index,
                            "error": error,
                            "reported_input_token_overrun": True,
                        }
                    )
                    aborted_on_input_token_overrun = True
                    break
                try:
                    for key in total_usage:
                        total_usage[key] += reported_usage[key]
                    state = prepared["state"]
                    query = prepared["query"]
                    sources = prepared["sources"]
                    strategy_mode = prepared["strategy_mode"]
                    memory_view = prepared["memory_view"]
                    strategy_view = prepared["strategy_view"]
                    candidate_memory_view = prepared["candidate_memory_view"]
                    candidate_strategy_view = prepared["candidate_strategy_view"]
                    filter_decision = prepared["evidence_filter_decision"]
                    response_text = (
                        supporter_generation_contract.normalize_output(result.text)
                        if supporter_generation_contract is not None
                        else result.text
                    )
                    base_tokens = estimate_tokens(
                        state.current_user_text
                        + state.current_session_summary
                        + "\n".join(x.content for x in state.current_session_history)
                        + system_prompt
                    )
                    memory_tokens = sum(estimate_tokens(x.text) for x in memory_view)
                    strategy_tokens = sum(
                        estimate_tokens(x.guidance_text + x.example_response)
                        for x in strategy_view
                    )
                    cost = CostRecord(
                        pm_input_tokens_est=estimate_tokens(query) + sum(
                            len(cat.catalog_fingerprint)
                            for cat in state.inventory.values()
                        ),
                        retrieval_calls=len(sources)
                        + (1 if strategy_mode is StrategyMode.RS else 0),
                        reranker_calls=0,
                        memory_tokens=memory_tokens,
                        strategy_tokens=strategy_tokens,
                        base_prompt_tokens=base_tokens,
                        total_input_tokens=(
                            reported_usage["prompt_tokens"]
                            or base_tokens + memory_tokens + strategy_tokens
                        ),
                        output_tokens=(
                            reported_usage["completion_tokens"]
                            or estimate_tokens(response_text)
                        ),
                        latency_ms=result.latency_ms,
                        api_cost_usd=None,
                        evidence_filter_calls=(1 if filter_decision is not None else 0),
                        candidate_memory_count=len(candidate_memory_view),
                        kept_memory_count=len(memory_view),
                        candidate_strategy_count=len(candidate_strategy_view),
                        kept_strategy_count=len(strategy_view),
                        candidate_memory_tokens=sum(
                            estimate_tokens(x.text) for x in candidate_memory_view
                        ),
                        candidate_strategy_tokens=sum(
                            estimate_tokens(x.guidance_text + x.example_response)
                            for x in candidate_strategy_view
                        ),
                        dropped_memory_tokens=sum(
                            estimate_tokens(x.text) for x in candidate_memory_view
                        )
                        - memory_tokens,
                        dropped_strategy_tokens=sum(
                            estimate_tokens(x.guidance_text + x.example_response)
                            for x in candidate_strategy_view
                        )
                        - strategy_tokens,
                    )
                    outcome = ActionOutcome(
                        card_id=card_id,
                        state_id=state.state_id,
                        user_id=state.user_id,
                        action_id=action_id,
                        response=response_text,
                        selected_memory_ids=[x.memory_id for x in memory_view],
                        selected_strategy_ids=[x.strategy_id for x in strategy_view],
                        memory_view=memory_view,
                        strategy_view=strategy_view,
                        effective_action_id=prepared["effective_action_id"],
                        candidate_memory_view=candidate_memory_view,
                        candidate_strategy_view=candidate_strategy_view,
                        evidence_filter_decision=filter_decision,
                        cost=cost,
                        model_name=endpoint.model,
                        prompt_hash=prepared["prompt_hash"],
                        request_hash=result.request_hash,
                        provenance={
                            "runtime_sha256": sha256_file(runtime_path),
                            "backend_sha256": sha256_file(backend_path),
                            "strategy_bank_sha256": sha256_file(strategy_bank_path),
                            "generated_at": utc_now(),
                            "temperature": temperature,
                            "seed": seed,
                            "request_retries": int(request_retries),
                            "physical_call_key": call_key,
                            "physical_attempt_index": reservation.attempt_index,
                            "physical_attempt_key": reservation.attempt_key,
                            "provider_finish_reason": result.provider_finish_reason,
                            "normalized_finish_reason": (
                                result.normalized_finish_reason
                            ),
                            **(
                                {
                                    "supporter_generation_treatment": (
                                        generation_treatment
                                    ),
                                    "supporter_generation_treatment_sha256": (
                                        generation_treatment_sha256
                                    ),
                                }
                                if generation_treatment is not None
                                else {}
                            ),
                            "contract_bindings_sha256": sha256_text(
                                canonical_json(dict(contract_bindings or {}))
                            ),
                        },
                    )
                    ledger.finish(
                        reservation,
                        succeeded=True,
                        request_hash=result.request_hash,
                        usage=reported_usage,
                        error=None,
                        result={
                            "raw_call": successful_raw_payload,
                            "action_outcome": outcome.model_dump(mode="json"),
                        },
                    )
                    append_jsonl(out_raw_calls_path, successful_raw_payload)
                    successful_raw_call_keys.add(call_key)
                    append_jsonl(out_outcomes_path, outcome.model_dump(mode="json"))
                    done.add((card_id, action_id))
                except Exception as exc:
                    if not ledger.succeeded(call_key):
                        ledger.finish(
                            reservation,
                            succeeded=False,
                            request_hash=result.request_hash,
                            usage=reported_usage,
                            error=(
                                "post-response artifact construction failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    failures.append(
                        {
                            "card_id": card_id,
                            "action_id": action_id,
                            "call_key": call_key,
                            "error": "post-request persistence failed: "
                            f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if fail_fast:
                        aborted_on_first_failure = True
                        break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                append_jsonl(
                    out_raw_calls_path,
                    request_log(
                        stage="generation",
                        endpoint=endpoint,
                        messages=prepared["messages"],
                        result=result,
                        parsed=None,
                        error=error,
                        prompt_hash=prepared["prompt_hash"],
                        record_ids={
                            **prepared["record_ids"],
                            "physical_call_key": call_key,
                            "physical_attempt_index": reservation.attempt_index,
                            "physical_attempt_key": reservation.attempt_key,
                        },
                    ),
                )
                ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=None,
                    usage=None,
                    error=error,
                )
                failures.append(
                    {
                        "card_id": card_id,
                        "action_id": action_id,
                        "call_key": call_key,
                        "attempt_index": reservation.attempt_index,
                        "error": error,
                    }
                )
                if fail_fast:
                    aborted_on_first_failure = True
                    break
    finally:
        if client is not None:
            client.close()

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    counted_call_keys: set[str] = set()
    for prepared in prepared_calls.values():
        call_key = str(prepared["call_key"])
        if call_key in counted_call_keys:
            continue
        counted_call_keys.add(call_key)
        if not ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key)
        usage = require_reported_usage(
            (terminal or {}).get("usage"), stage="action sweep persisted success"
        )
        for name in total_usage:
            total_usage[name] += usage[name]

    expected = len(expected_keys)
    completed = len(load_done_keys(out_outcomes_path, ("card_id", "action_id")) & expected_keys)
    summary = {
        "status": "COMPLETE" if completed == expected else "INCOMPLETE",
        "n_cards": len(cards),
        "expected_outcomes": expected,
        "completed_outcomes": completed,
        "new_api_calls": n_calls,
        "new_physical_http_attempts": n_calls,
        "historical_physical_http_attempts": historical_attempts,
        "total_physical_http_attempts": ledger.started_attempts,
        "remaining_runtime_physical_http_attempts": ledger.remaining_attempts,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "request_retries": int(request_retries),
        "fail_fast": bool(fail_fast),
        "aborted_on_first_failure": aborted_on_first_failure,
        "aborted_on_input_token_overrun": aborted_on_input_token_overrun,
        "aborted_on_missing_reported_usage": aborted_on_missing_reported_usage,
        "aborted_on_completion_gate": aborted_on_completion_gate,
        "maximum_physical_api_attempts_planned": planned_maximum_attempts,
        "maximum_physical_api_attempts_authorized": runtime_attempt_cap,
        "failures": failures,
        "usage": total_usage,
        "endpoint_model": endpoint.model,
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "contract_bindings": dict(contract_bindings or {}),
    }
    if generation_treatment is not None:
        summary.update(
            {
                "supporter_generation_treatment": generation_treatment,
                "supporter_generation_treatment_sha256": (
                    generation_treatment_sha256
                ),
            }
        )
    write_json(out_summary_path, summary)
    if summary["status"] != "COMPLETE":
        raise RuntimeError(f"action sweep incomplete: {len(failures)} failures")
    create_artifact_attestation(
        attestation_path,
        stage="action_sweep",
        inputs={
            "runtime": runtime_path,
            "backend": backend_path,
            "strategy_bank": strategy_bank_path,
            "run_manifest": manifest_path,
        },
        outputs={
            "action_outcomes": (out_outcomes_path, True),
            "raw_calls": (out_raw_calls_path, True),
            "physical_attempt_ledger": (attempt_ledger_path, True),
            "summary": (out_summary_path, False),
        },
        parameters={
            "endpoint_model": endpoint.model,
            "endpoint_family": endpoint.family,
            "endpoint_base_url": endpoint.base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "request_retries": int(request_retries),
            "fail_fast": bool(fail_fast),
            "input_token_safety_factor": float(input_token_safety_factor),
            "fail_on_reported_input_overrun": bool(
                fail_on_reported_input_overrun
            ),
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "maximum_physical_api_attempts_planned": planned_maximum_attempts,
            "maximum_physical_api_attempts_authorized": runtime_attempt_cap,
            "strategy_top_k": strategy_top_k,
            "memory_min_score": memory_min_score,
            "strategy_min_score": strategy_min_score,
            "evidence_filter": (
                evidence_filter_config.payload()
                if evidence_filter_config is not None
                else None
            ),
            "evidence_filter_config_sha256": (
                evidence_filter_config.digest()
                if evidence_filter_config is not None
                else None
            ),
            "evidence_filter_model": (
                {
                    "checkpoint_sha256": memory_helpfulness_model.checkpoint_sha256,
                    "model_contract_sha256": memory_helpfulness_model.contract_hash(),
                }
                if memory_helpfulness_model is not None
                else None
            ),
            "action_filter": sorted(action_filter) if action_filter else None,
            "card_filter": sorted(card_filter) if card_filter else None,
            "max_cards": max_cards,
            **(
                {
                    "supporter_generation_treatment": generation_treatment,
                    "supporter_generation_treatment_sha256": (
                        generation_treatment_sha256
                    ),
                }
                if generation_treatment is not None
                else {}
            ),
            "contract_bindings": dict(contract_bindings or {}),
        },
        expected={
            "cards": len(cards),
            "outcomes": expected,
        },
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary
