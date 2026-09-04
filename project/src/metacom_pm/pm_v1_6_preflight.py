from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .contracts import (
    MemoryBackendRecord,
    MemorySource,
    RuntimeState,
    StrategyCard,
    StrategyMode,
    parse_action_id,
)
from .generation_contract import SupporterGenerationContract
from .io import canonical_json, iter_jsonl, sha256_file, sha256_text
from .pm_v1_6_contracts import (
    ACTION_LINEAGE_PROTOCOL,
    RetrievalAttempt,
    prompt_equivalence_id,
    realized_action_from_evidence,
)
from .pm_v2_data import EvaluatorContextIndex, load_evaluator_context_index
from .prompts import generation_messages
from .retrieval import MemoryRetriever, StrategyRetriever, context_query
from .text import estimate_tokens

PREFLIGHT_PROTOCOL = "pm-v1.6-required-hit-and-alias-preflight-v1"


@dataclass(frozen=True)
class PreflightInputs:
    runtime_path: Path
    backend_path: Path
    evaluator_contexts_path: Path
    strategy_bank_path: Path


def _load_runtime(path: str | Path) -> dict[str, RuntimeState]:
    rows = [RuntimeState.model_validate(row) for row in iter_jsonl(path)]
    result = {row.card_id: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate runtime card_id in V1.6 preflight")
    return result


def _load_backends(path: str | Path) -> dict[str, MemoryBackendRecord]:
    rows = [MemoryBackendRecord.model_validate(row) for row in iter_jsonl(path)]
    result = {row.card_id: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate backend card_id in V1.6 preflight")
    return result


def _load_strategies(path: str | Path) -> list[StrategyCard]:
    rows = [StrategyCard.model_validate(row) for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError("V1.6 preflight requires a non-empty Strategy Bank")
    return rows


def required_sources_for_context(context: Mapping[str, Any]) -> set[str]:
    regime = str(context["regime"])
    needed = {str(value) for value in context.get("needed_memory_sources") or []}
    if regime in {"profile_needed", "summary_needed", "event_needed", "multi_source_needed"}:
        return needed
    if regime == "memory_harmful":
        # The harmful regime is only meaningful if at least one annotated harmful
        # memory can actually be retrieved under the frozen mechanism. It is not
        # required that all three sources hit in every surface form.
        return {"ANY_HARMFUL_MEMORY"}
    return set()


def _harmful_memory_ids(context: Mapping[str, Any]) -> set[str]:
    return {
        str(row["memory_id"])
        for row in context.get("memory_annotations") or []
        if row.get("item_utility") == "harmful"
    }


def _timed_memory_retrieval(
    retriever: MemoryRetriever,
    *,
    query: str,
    items,
    source: MemorySource,
) -> tuple[list, RetrievalAttempt]:
    started = time.perf_counter()
    selected = retriever.retrieve(query, items, frozenset({source}))
    latency_ms = (time.perf_counter() - started) * 1000.0
    return selected, RetrievalAttempt(
        source=source.value,
        call_count=1,
        hit_count=len(selected),
        retrieved_tokens=sum(estimate_tokens(row.text) for row in selected),
        latency_ms=latency_ms,
    )


def _timed_strategy_retrieval(
    retriever: StrategyRetriever,
    *,
    query: str,
) -> tuple[list[StrategyCard], RetrievalAttempt]:
    started = time.perf_counter()
    selected = retriever.retrieve(query)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return selected, RetrievalAttempt(
        source="RS",
        call_count=1,
        hit_count=len(selected),
        retrieved_tokens=sum(
            estimate_tokens(row.guidance_text + " " + row.example_response)
            for row in selected
        ),
        latency_ms=latency_ms,
    )


def build_preflight(
    *,
    runtime_path: str | Path,
    backend_path: str | Path,
    evaluator_contexts_path: str | Path,
    strategy_bank_path: str | Path,
    supporter_contract: SupporterGenerationContract,
    memory_min_score: float,
    strategy_min_score: float,
    strategy_top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runtime_path = Path(runtime_path)
    backend_path = Path(backend_path)
    evaluator_contexts_path = Path(evaluator_contexts_path)
    strategy_bank_path = Path(strategy_bank_path)

    states = _load_runtime(runtime_path)
    backends = _load_backends(backend_path)
    if set(states) != set(backends):
        raise RuntimeError("runtime/backend card universes differ")
    strategies = _load_strategies(strategy_bank_path)
    evaluator_index: EvaluatorContextIndex = load_evaluator_context_index(
        evaluator_contexts_path,
        require_exact=False,
    )
    if set(evaluator_index.by_card) != set(states):
        raise RuntimeError("evaluator-context/runtime card universes differ")

    memory_retriever = MemoryRetriever(
        minimum_score_by_source={
            source: float(memory_min_score) for source in MemorySource
        }
    )
    strategy_retriever = StrategyRetriever(
        strategies,
        top_k=int(strategy_top_k),
        minimum_score=float(strategy_min_score),
    )

    action_rows: list[dict[str, Any]] = []
    state_required_hit: dict[str, dict[str, Any]] = {}
    for card_id in sorted(states):
        state = states[card_id]
        backend = backends[card_id]
        context = evaluator_index.by_card[card_id]
        if context["state_id"] != state.state_id:
            raise RuntimeError(f"evaluator state mismatch for card {card_id}")
        query = context_query(
            state.current_user_text,
            [turn.model_dump(mode="json") for turn in state.current_session_history],
            state.current_session_summary,
        )
        harmful_ids = _harmful_memory_ids(context)
        state_action_rows: list[dict[str, Any]] = []
        source_hit_ids: dict[str, set[str]] = {source.value: set() for source in MemorySource}
        strategy_hit = False

        for action_id in state.allowed_actions:
            requested_sources, strategy_mode = parse_action_id(action_id)
            memory_view = []
            attempts: list[RetrievalAttempt] = []
            for source in MemorySource:
                if source not in requested_sources:
                    continue
                selected, attempt = _timed_memory_retrieval(
                    memory_retriever,
                    query=query,
                    items=backend.items,
                    source=source,
                )
                memory_view.extend(selected)
                attempts.append(attempt)
                source_hit_ids[source.value].update(row.memory_id for row in selected)
            strategy_view: list[StrategyCard] = []
            if strategy_mode is StrategyMode.RS:
                strategy_view, strategy_attempt = _timed_strategy_retrieval(
                    strategy_retriever,
                    query=query,
                )
                attempts.append(strategy_attempt)
                strategy_hit = strategy_hit or bool(strategy_view)

            messages = generation_messages(
                state,
                memory_view,
                strategy_view,
                system_prompt=supporter_contract.system_prompt,
            )
            realized = realized_action_from_evidence(
                [item.source for item in memory_view],
                strategy_card_count=len(strategy_view),
            )
            row = {
                "protocol": ACTION_LINEAGE_PROTOCOL,
                "card_id": card_id,
                "state_id": state.state_id,
                "user_id": state.user_id,
                "split": state.split,
                "regime": context["regime"],
                "requested_action_id": action_id,
                "retrieval_attempts": [
                    attempt.model_dump(mode="json") for attempt in attempts
                ],
                "candidate_memory_ids": [item.memory_id for item in memory_view],
                "candidate_strategy_ids": [item.strategy_id for item in strategy_view],
                "realized_action_id": realized,
                "prompt_sha256": sha256_text(canonical_json(messages)),
                "prompt_equivalence_id": prompt_equivalence_id(messages),
                "prompt_equivalence_class_size": 1,
                "shared_quality_label_weight": 1.0,
                "supporter_generation_treatment_sha256": supporter_contract.digest(),
            }
            state_action_rows.append(row)
            action_rows.append(row)

        aliases: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in state_action_rows:
            aliases[str(row["prompt_equivalence_id"])].append(row)
        for group in aliases.values():
            actions = sorted(str(row["requested_action_id"]) for row in group)
            class_size = len(group)
            for row in group:
                row["prompt_equivalence_class_size"] = class_size
                row["prompt_equivalence_actions"] = actions
                row["shared_quality_label_weight"] = 1.0 / class_size

        required = required_sources_for_context(context)
        checks: dict[str, bool] = {}
        for source in sorted(required - {"ANY_HARMFUL_MEMORY"}):
            checks[f"required_source_{source}_has_hit"] = bool(source_hit_ids[source])
        if "ANY_HARMFUL_MEMORY" in required:
            retrieved_harmful = set().union(*source_hit_ids.values()) & harmful_ids
            checks["harmful_regime_retrieves_annotated_harmful_memory"] = bool(
                retrieved_harmful
            )
        if context["regime"] in {"strategy_helpful", "strategy_harmful"}:
            checks["required_strategy_has_hit"] = strategy_hit
        state_required_hit[state.state_id] = {
            "state_id": state.state_id,
            "card_id": card_id,
            "regime": context["regime"],
            "needed_memory_sources": list(context["needed_memory_sources"]),
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        }

    action_rows.sort(key=lambda row: (row["state_id"], row["requested_action_id"]))
    required_rows = [state_required_hit[key] for key in sorted(state_required_hit)]
    failed = [row for row in required_rows if row["status"] != "PASS"]
    transition_counts = Counter(
        (row["requested_action_id"], row["realized_action_id"])
        for row in action_rows
    )
    zero_hit_rows = [
        row
        for row in action_rows
        if row["requested_action_id"] != row["realized_action_id"]
    ]
    alias_rows = [
        row for row in action_rows if row["prompt_equivalence_class_size"] > 1
    ]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "protocol": PREFLIGHT_PROTOCOL,
        "timing": "before_response_generation_and_judging",
        "outcome_conditioned_exclusion_forbidden": True,
        "runtime_sha256": sha256_file(runtime_path),
        "backend_sha256": sha256_file(backend_path),
        "evaluator_contexts_sha256": sha256_file(evaluator_contexts_path),
        "strategy_bank_sha256": sha256_file(strategy_bank_path),
        "supporter_generation_treatment": supporter_contract.payload(),
        "supporter_generation_treatment_sha256": supporter_contract.digest(),
        "memory_min_score": float(memory_min_score),
        "strategy_min_score": float(strategy_min_score),
        "strategy_top_k": int(strategy_top_k),
        "states": len(states),
        "logical_action_rows": len(action_rows),
        "required_hit_failures": len(failed),
        "required_hit_failed_state_ids": [row["state_id"] for row in failed],
        "requested_realized_mismatch_rows": len(zero_hit_rows),
        "requested_realized_mismatch_rate": (
            len(zero_hit_rows) / len(action_rows) if action_rows else 0.0
        ),
        "prompt_alias_rows": len(alias_rows),
        "prompt_alias_rate": len(alias_rows) / len(action_rows) if action_rows else 0.0,
        "prompt_equivalence_classes": len(
            {(row["state_id"], row["prompt_equivalence_id"]) for row in action_rows}
        ),
        "requested_to_realized_transition_matrix": [
            {
                "requested_action_id": requested,
                "realized_action_id": realized,
                "count": count,
            }
            for (requested, realized), count in sorted(transition_counts.items())
        ],
        "required_hit_states": required_rows,
        "action_plan_sha256": sha256_text(canonical_json(action_rows)),
    }
    return summary, action_rows


def require_preflight_pass(summary: Mapping[str, Any]) -> None:
    if summary.get("protocol") != PREFLIGHT_PROTOCOL or summary.get("status") != "PASS":
        raise RuntimeError("PM-v1.6 required-hit/alias preflight did not PASS")
    if int(summary.get("required_hit_failures", -1)) != 0:
        raise RuntimeError("PM-v1.6 preflight contains required-hit failures")
