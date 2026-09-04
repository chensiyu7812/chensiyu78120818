"""Fail-closed post-hoc diagnostic for the V1 Strategy Bank size change.

This module is intentionally isolated from PM-v2 training and study gates.  It
replays a deterministic, user-balanced sample of frozen V1 EvoEmo PM+RS turns
twice.  Conditional on that already-frozen PM action, the only prompt-content
manipulation is the canonical V1 top-3 strategy retrieval result: once from the
156-card development bank and once from the 12,429-card external bank.

The diagnostic is descriptive/mechanistic.  It is not a confirmatory study and
must never be used as a PM-v2 training or release gate.  It does not estimate
how bank metadata changes PM routing/action selection and does not identify the
total effect of the development/deployment bank mismatch on V1 performance.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .api import Endpoint, make_client, request_log, require_reported_usage
from .attempt_ledger import (
    PersistentAttemptLedger,
    physical_call_key,
    reported_prompt_token_error,
)
from .contracts import (
    DialogueTurn,
    MemoryItem,
    StrategyCard,
    StrategyMode,
    parse_action_id,
)
from .io import (
    append_jsonl,
    canonical_json,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    stable_hex,
    write_json,
    write_jsonl,
)
from .prompts import SELECTIVE_ESMEM_SYSTEM, generation_messages
from .retrieval_v1_canonical import (
    CANONICAL_V1_RETRIEVER_SHA256,
    CANONICAL_V1_TOP_K,
    StrategyRetrieverV1Canonical,
    context_query,
)
from .text import conservative_token_bound, normalize_space


DIAGNOSTIC_LABEL = "posthoc_v1_bank_mechanism_diagnostic"
NONCANONICAL_DEBUG_LABEL = f"{DIAGNOSTIC_LABEL}_noncanonical_debug"
DIAGNOSTIC_PROTOCOL = "posthoc-v1-bank-mechanism-diagnostic-v1"
CALL_PLAN_PROTOCOL = "posthoc-v1-bank-paired-call-plan-v1"
CONDITIONAL_ESTIMAND = (
    "Conditional on the frozen V1 PM+RS action, selected memory, state, prompt, "
    "generator treatment, and decoding controls, contrast the downstream "
    "retrieval and generated response produced by the legacy 156-card Strategy "
    "Bank versus the full 12,429-card Strategy Bank."
)
EXCLUDED_ESTIMANDS = (
    "Effect of Strategy Bank metadata or inventory features on PM routing or "
    "action selection.",
    "Total effect of the development-deployment Strategy Bank mismatch on V1 "
    "overall performance.",
    "A complete causal attribution of any V1 performance gap to Strategy Bank "
    "mismatch.",
)

FROZEN_V1_TURNS_SHA256 = (
    "9bd51edea120790431fe43603210253595bd71f273b76a82036f52521ba22ee4"
)
FROZEN_V1_LEGACY_BANK_SHA256 = (
    "5b2d58c6be07d3c27e20e9162a3494f47f875d02b94b983f8b05c08ff044bf46"
)
FROZEN_V1_FULL_BANK_SHA256 = (
    "f64ded1e23b4a79c0e08c6b47355ad76b3880d5cbc0bfa7530f366149b5c70db"
)
FROZEN_V1_LEGACY_BANK_CARD_COUNT = 156
FROZEN_V1_FULL_BANK_CARD_COUNT = 12_429

DEFAULT_SAMPLE_SIZE = 40
DEFAULT_SAMPLE_SEED = 20_260_716
GENERATOR_TEMPERATURE = 0.0
GENERATOR_MAX_OUTPUT_TOKENS = 300
OUTPUT_NORMALIZATION = "normalize_space_v1"
COMMON_PROMPT_ID = "selective_esmem_v1"
MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL = 1

BANK_CONDITIONS = ("legacy_156", "full_12429")
ANNOTATION_FIELDS = (
    "item_id",
    "pair_id",
    "reviewer_id",
    "overall_preference",
    "emotional_support_A",
    "emotional_support_B",
    "contextual_fit_A",
    "contextual_fit_B",
    "non_intrusiveness_A",
    "non_intrusiveness_B",
    "strategy_relevance_A",
    "strategy_relevance_B",
    "safety_A",
    "safety_B",
    "confidence",
    "notes",
)


def _require_sha256(path: str | Path, expected: str, *, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, "
            f"path={source}"
        )
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": actual,
    }


def _validate_prices(
    *,
    input_usd_per_million_tokens: float,
    output_usd_per_million_tokens: float,
    input_token_safety_factor: float,
) -> None:
    for name, value in (
        ("input_usd_per_million_tokens", input_usd_per_million_tokens),
        ("output_usd_per_million_tokens", output_usd_per_million_tokens),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if (
        not math.isfinite(float(input_token_safety_factor))
        or float(input_token_safety_factor) < 1.0
    ):
        raise ValueError("input_token_safety_factor must be finite and at least 1")


def _load_cards(path: str | Path) -> list[StrategyCard]:
    cards = [StrategyCard.model_validate(row) for row in iter_jsonl(path)]
    if not cards:
        raise ValueError(f"strategy bank is empty: {path}")
    identifiers = [card.strategy_id for card in cards]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"strategy bank contains duplicate strategy IDs: {path}")
    return cards


def _turn_unit_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "user_id": str(row.get("user_id") or ""),
        "topic_index": int(row.get("topic_index") or 0),
        "condition": str(row.get("condition") or ""),
        "seed": int(row.get("seed") or 0),
        "simulator_id": str(row.get("simulator_id") or ""),
        "interaction_mode": str(row.get("interaction_mode") or ""),
        "track_id": str(row.get("track_id") or ""),
        "turn_index": int(row.get("turn_index") or 0),
        "state_id": str(row.get("state_id") or ""),
        "exogenous_state_id": str(row.get("exogenous_state_id") or ""),
    }


def _compact_rs_candidate(
    row: Mapping[str, Any], *, source_condition: str
) -> dict[str, Any] | None:
    if str(row.get("condition") or "") != source_condition:
        return None
    action_id = str(row.get("action_id") or "")
    try:
        requested_sources, strategy_mode = parse_action_id(action_id)
    except ValueError as exc:
        raise ValueError(f"eligible V1 row has invalid action_id {action_id!r}") from exc
    if strategy_mode is not StrategyMode.RS:
        return None
    if row.get("protocol") != "selective" or row.get("interaction_mode") != "fixed":
        raise ValueError("V1 bank probe requires selective, fixed-input EvoEmo turns")

    unit_key = _turn_unit_key(row)
    if any(
        not value
        for key, value in unit_key.items()
        if key
        in {
            "user_id",
            "condition",
            "simulator_id",
            "interaction_mode",
            "track_id",
            "state_id",
            "exogenous_state_id",
        }
    ):
        raise ValueError(f"eligible V1 row has an incomplete unit key: {unit_key}")
    if unit_key["topic_index"] < 1 or unit_key["turn_index"] < 1:
        raise ValueError(f"eligible V1 row has invalid turn coordinates: {unit_key}")

    context = row.get("context_before_turn")
    if not isinstance(context, list) or not context:
        raise ValueError("eligible V1 row lacks context_before_turn")
    normalized_context: list[dict[str, str]] = []
    for turn in context:
        if not isinstance(turn, Mapping) or turn.get("role") not in {
            "seeker",
            "supporter",
        }:
            raise ValueError("eligible V1 context has an invalid role")
        content = str(turn.get("content") or "")
        if not content.strip():
            raise ValueError("eligible V1 context contains an empty turn")
        normalized_context.append({"role": str(turn["role"]), "content": content})
    expected_context_sha = sha256_text(canonical_json(normalized_context))
    if row.get("context_sha256") != expected_context_sha:
        raise RuntimeError("eligible V1 row context_sha256 does not match its context")

    seeker_message = str(row.get("seeker_message") or "")
    if not seeker_message.strip():
        raise ValueError("eligible V1 row lacks seeker_message")
    memory_rows = row.get("selected_memory")
    strategy_rows = row.get("selected_strategy")
    if not isinstance(memory_rows, list) or not isinstance(strategy_rows, list):
        raise ValueError("eligible V1 row lacks selected evidence arrays")
    memories = [MemoryItem.model_validate(item) for item in memory_rows]
    memory_sources = {item.source for item in memories}
    if memory_sources != set(requested_sources):
        raise RuntimeError(
            "frozen V1 selected_memory sources do not equal the requested action: "
            f"action={action_id}, selected={sorted(x.value for x in memory_sources)}"
        )
    recorded_strategies = [StrategyCard.model_validate(item) for item in strategy_rows]
    if len(recorded_strategies) != CANONICAL_V1_TOP_K:
        raise RuntimeError("frozen V1 RS row does not contain exactly three strategies")

    return {
        "unit_key": unit_key,
        "unit_id": f"v1bank_{stable_hex(canonical_json(unit_key), n=24)}",
        "user_id": unit_key["user_id"],
        "topic_index": unit_key["topic_index"],
        "seed": unit_key["seed"],
        "simulator_id": unit_key["simulator_id"],
        "turn_index": unit_key["turn_index"],
        "state_id": unit_key["state_id"],
        "exogenous_state_id": unit_key["exogenous_state_id"],
        "track_id": unit_key["track_id"],
        "requested_action_id": action_id,
        "context_before_turn": normalized_context,
        "context_sha256": expected_context_sha,
        "seeker_message": seeker_message,
        "selected_memory": [item.model_dump(mode="json") for item in memories],
        "selected_memory_sha256": sha256_text(
            canonical_json([item.model_dump(mode="json") for item in memories])
        ),
        "recorded_full_bank_strategy_ids": [
            item.strategy_id for item in recorded_strategies
        ],
    }


def select_user_balanced_rs_turns(
    turns_path: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    source_condition: str = "pm",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Select a deterministic round-robin sample across all eligible users."""

    if int(sample_size) < 1:
        raise ValueError("sample_size must be positive")
    if not str(source_condition).strip():
        raise ValueError("source_condition must be non-empty")
    by_user: dict[str, list[dict[str, Any]]] = {}
    seen_units: set[str] = set()
    for row in iter_jsonl(turns_path):
        candidate = _compact_rs_candidate(row, source_condition=source_condition)
        if candidate is None:
            continue
        unit_id = str(candidate["unit_id"])
        if unit_id in seen_units:
            raise RuntimeError(f"duplicate eligible V1 unit: {unit_id}")
        seen_units.add(unit_id)
        by_user.setdefault(str(candidate["user_id"]), []).append(candidate)
    if not by_user:
        raise RuntimeError(f"no {source_condition}+RS V1 turns were found")
    eligible_count = sum(len(rows) for rows in by_user.values())
    if int(sample_size) > eligible_count:
        raise ValueError(
            f"sample_size={sample_size} exceeds eligible RS turns={eligible_count}"
        )

    for user_id, rows in by_user.items():
        rows.sort(
            key=lambda item: sha256_text(
                canonical_json(
                    {
                        "domain": "posthoc-v1-bank-unit-order-v1",
                        "sample_seed": int(sample_seed),
                        "user_id": user_id,
                        "unit_key": item["unit_key"],
                    }
                )
            )
        )
    user_order = sorted(
        by_user,
        key=lambda user_id: sha256_text(
            canonical_json(
                {
                    "domain": "posthoc-v1-bank-user-order-v1",
                    "sample_seed": int(sample_seed),
                    "user_id": user_id,
                }
            )
        ),
    )
    offsets = {user_id: 0 for user_id in user_order}
    selected: list[dict[str, Any]] = []
    while len(selected) < int(sample_size):
        made_progress = False
        for user_id in user_order:
            offset = offsets[user_id]
            if offset >= len(by_user[user_id]):
                continue
            selected.append(dict(by_user[user_id][offset]))
            offsets[user_id] += 1
            made_progress = True
            if len(selected) == int(sample_size):
                break
        if not made_progress:
            raise RuntimeError("user-balanced sampling exhausted unexpectedly")
    counts: dict[str, int] = {}
    for row in selected:
        counts[str(row["user_id"])] = counts.get(str(row["user_id"]), 0) + 1
    all_user_counts = [counts.get(user_id, 0) for user_id in user_order]
    if max(all_user_counts) - min(all_user_counts) > 1:
        raise RuntimeError("user-balanced sample allocation differs by more than one")
    return selected, {
        "eligible_per_user": {
            user_id: len(by_user[user_id]) for user_id in user_order
        },
        "selected_per_user": counts,
    }


def _prompt_state(candidate: Mapping[str, Any]) -> SimpleNamespace:
    recent = list(candidate["context_before_turn"])[-8:]
    history = [
        DialogueTurn(
            role="user" if turn["role"] == "seeker" else "assistant",
            content=str(turn["content"]),
        )
        for turn in recent
    ]
    return SimpleNamespace(
        current_session_summary="",
        current_session_history=history,
        current_user_text=str(candidate["seeker_message"]),
    )


def _strategy_section(cards: Sequence[StrategyCard]) -> str:
    lines = [
        f"- {card.guidance_text}\n  Example style (adapt, do not copy): "
        f"{card.example_response}"
        for card in cards
    ]
    return "Potential emotional-support guidance. Use only when fitting:\n" + "\n".join(lines)


def _mask_strategy_section(
    messages: Sequence[Mapping[str, str]], cards: Sequence[StrategyCard]
) -> list[dict[str, str]]:
    if len(messages) != 2 or messages[0].get("role") != "system":
        raise RuntimeError("unexpected supporter prompt structure")
    section = _strategy_section(cards)
    user_content = str(messages[1].get("content") or "")
    if user_content.count(section) != 1:
        raise RuntimeError("strategy evidence is not a unique prompt section")
    return [
        dict(messages[0]),
        {
            "role": str(messages[1]["role"]),
            "content": user_content.replace(
                section, "<<PAIRED_STRATEGY_EVIDENCE_ONLY>>", 1
            ),
        },
    ]


def _retrieval_record(
    *,
    bank_condition: str,
    bank_sha256: str,
    query: str,
    scored: Sequence[tuple[float, StrategyCard]],
) -> dict[str, Any]:
    payload = {
        "retriever_semantics_sha256": CANONICAL_V1_RETRIEVER_SHA256,
        "top_k": CANONICAL_V1_TOP_K,
        "bank_condition": bank_condition,
        "bank_sha256": bank_sha256,
        "query_sha256": sha256_text(query),
        "results": [
            {
                "rank": rank,
                "score": format(float(score), ".17g"),
                "strategy_id": card.strategy_id,
                "card_sha256": sha256_text(canonical_json(card.model_dump(mode="json"))),
            }
            for rank, (score, card) in enumerate(scored, 1)
        ],
    }
    return {**payload, "retrieval_sha256": sha256_text(canonical_json(payload))}


def _balanced_first_assignment(
    unit_ids: Sequence[str], *, sample_seed: int, domain: str
) -> set[str]:
    """Deterministically assign half the units to legacy-first/A.

    Hash sorting preserves reproducibility while exact half allocation avoids a
    chance positional imbalance in this deliberately small diagnostic.
    """

    ordered = sorted(
        (str(unit_id) for unit_id in unit_ids),
        key=lambda unit_id: sha256_text(
            canonical_json(
                {
                    "domain": str(domain),
                    "sample_seed": int(sample_seed),
                    "unit_id": unit_id,
                }
            )
        ),
    )
    return set(ordered[: len(ordered) // 2])


def plan_posthoc_v1_bank_probe(
    turns_path: str | Path,
    legacy_bank_path: str | Path,
    full_bank_path: str | Path,
    *,
    endpoint: Endpoint,
    input_usd_per_million_tokens: float,
    output_usd_per_million_tokens: float,
    input_token_safety_factor: float = 1.25,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    source_condition: str = "pm",
    expected_turns_sha256: str = FROZEN_V1_TURNS_SHA256,
    expected_legacy_bank_sha256: str = FROZEN_V1_LEGACY_BANK_SHA256,
    expected_full_bank_sha256: str = FROZEN_V1_FULL_BANK_SHA256,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create an exact paired plan without constructing an API client."""

    _validate_prices(
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        output_usd_per_million_tokens=output_usd_per_million_tokens,
        input_token_safety_factor=input_token_safety_factor,
    )
    bindings = {
        "frozen_v1_turns": _require_sha256(
            turns_path, expected_turns_sha256, label="frozen V1 turns"
        ),
        "legacy_156_bank": _require_sha256(
            legacy_bank_path,
            expected_legacy_bank_sha256,
            label="legacy 156-card Strategy Bank",
        ),
        "full_12429_bank": _require_sha256(
            full_bank_path,
            expected_full_bank_sha256,
            label="full 12,429-card Strategy Bank",
        ),
    }
    canonical_frozen_v1_inputs = bool(
        source_condition == "pm"
        and expected_turns_sha256 == FROZEN_V1_TURNS_SHA256
        and expected_legacy_bank_sha256 == FROZEN_V1_LEGACY_BANK_SHA256
        and expected_full_bank_sha256 == FROZEN_V1_FULL_BANK_SHA256
    )
    diagnostic_label = (
        DIAGNOSTIC_LABEL
        if canonical_frozen_v1_inputs
        else NONCANONICAL_DEBUG_LABEL
    )
    legacy_cards = _load_cards(legacy_bank_path)
    full_cards = _load_cards(full_bank_path)
    if (
        expected_legacy_bank_sha256 == FROZEN_V1_LEGACY_BANK_SHA256
        and len(legacy_cards) != FROZEN_V1_LEGACY_BANK_CARD_COUNT
    ):
        raise RuntimeError("canonical legacy bank does not contain exactly 156 cards")
    if (
        expected_full_bank_sha256 == FROZEN_V1_FULL_BANK_SHA256
        and len(full_cards) != FROZEN_V1_FULL_BANK_CARD_COUNT
    ):
        raise RuntimeError("canonical full bank does not contain exactly 12,429 cards")
    retrievers = {
        "legacy_156": StrategyRetrieverV1Canonical(legacy_cards),
        "full_12429": StrategyRetrieverV1Canonical(full_cards),
    }
    bank_sha256 = {
        "legacy_156": bindings["legacy_156_bank"]["sha256"],
        "full_12429": bindings["full_12429_bank"]["sha256"],
    }
    samples, sampling_counts = select_user_balanced_rs_turns(
        turns_path,
        sample_size=sample_size,
        sample_seed=sample_seed,
        source_condition=source_condition,
    )
    legacy_first_units = _balanced_first_assignment(
        [str(sample["unit_id"]) for sample in samples],
        sample_seed=sample_seed,
        domain="posthoc-v1-bank-generation-order-v1",
    )

    endpoint_binding = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "timeout_seconds": endpoint.timeout_seconds,
    }
    prompt_sha256 = sha256_text(SELECTIVE_ESMEM_SYSTEM)
    plan: list[dict[str, Any]] = []
    enriched_samples: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    for sample_index, candidate in enumerate(samples, 1):
        state = _prompt_state(candidate)
        query = context_query(
            state.current_user_text,
            [turn.model_dump(mode="json") for turn in state.current_session_history],
            state.current_session_summary,
        )
        memories = [MemoryItem.model_validate(row) for row in candidate["selected_memory"]]
        calls_by_bank: dict[str, dict[str, Any]] = {}
        masked_messages: dict[str, list[dict[str, str]]] = {}
        retrievals: dict[str, dict[str, Any]] = {}
        for bank_condition in BANK_CONDITIONS:
            scored = retrievers[bank_condition].retrieve_with_scores(query)
            cards = [card for _, card in scored]
            if len(cards) != CANONICAL_V1_TOP_K:
                raise RuntimeError(f"{bank_condition} retrieval returned fewer than 3 cards")
            if (
                bank_condition == "full_12429"
                and [card.strategy_id for card in cards]
                != list(candidate["recorded_full_bank_strategy_ids"])
            ):
                raise RuntimeError(
                    "canonical full-bank retrieval does not reproduce the frozen V1 "
                    f"selected_strategy for {candidate['unit_id']}"
                )
            messages = generation_messages(
                state,
                memories,
                cards,
                system_prompt=SELECTIVE_ESMEM_SYSTEM,
            )
            masked_messages[bank_condition] = _mask_strategy_section(messages, cards)
            retrieval = _retrieval_record(
                bank_condition=bank_condition,
                bank_sha256=str(bank_sha256[bank_condition]),
                query=query,
                scored=scored,
            )
            retrievals[bank_condition] = retrieval
            generator_seed = int(candidate["seed"]) + int(candidate["turn_index"])
            messages_sha256 = sha256_text(canonical_json(messages))
            request_parameters = {
                "temperature": GENERATOR_TEMPERATURE,
                "max_tokens": GENERATOR_MAX_OUTPUT_TOKENS,
                "seed": generator_seed,
                "response_schema": None,
                "maximum_physical_attempts": (
                    MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
                ),
                "output_normalization": OUTPUT_NORMALIZATION,
                "bank_condition": bank_condition,
                "bank_sha256": bank_sha256[bank_condition],
                "retrieval_sha256": retrieval["retrieval_sha256"],
            }
            record_ids = {
                "diagnostic_label": diagnostic_label,
                "pair_id": candidate["unit_id"],
                "bank_condition": bank_condition,
            }
            logical_call_key = physical_call_key(
                stage=diagnostic_label,
                record_ids=record_ids,
                prompt_sha256=messages_sha256,
                endpoint=endpoint,
                request_parameters=request_parameters,
            )
            input_tokens = conservative_token_bound(
                canonical_json(messages), safety_factor=input_token_safety_factor
            )
            output_tokens = GENERATOR_MAX_OUTPUT_TOKENS
            estimated_usd = (
                input_tokens * float(input_usd_per_million_tokens)
                + output_tokens * float(output_usd_per_million_tokens)
            ) / 1_000_000.0
            calls_by_bank[bank_condition] = {
                "protocol": CALL_PLAN_PROTOCOL,
                "diagnostic_label": diagnostic_label,
                "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
                "conditional_estimand": CONDITIONAL_ESTIMAND,
                "excluded_estimands": list(EXCLUDED_ESTIMANDS),
                "confirmatory": False,
                "v2_training_gate": False,
                "sample_index": sample_index,
                "pair_id": candidate["unit_id"],
                "unit_key": candidate["unit_key"],
                "bank_condition": bank_condition,
                "bank_sha256": bank_sha256[bank_condition],
                "query_sha256": sha256_text(query),
                "retrieval": retrieval,
                "requested_action_id": candidate["requested_action_id"],
                "selected_memory_sha256": candidate["selected_memory_sha256"],
                "non_strategy_messages_sha256": None,
                "messages": messages,
                "messages_sha256": messages_sha256,
                "replay_generator_treatment": endpoint_binding,
                "temperature": GENERATOR_TEMPERATURE,
                "max_tokens": GENERATOR_MAX_OUTPUT_TOKENS,
                "generator_seed": generator_seed,
                "output_normalization": OUTPUT_NORMALIZATION,
                "maximum_physical_attempts": (
                    MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
                ),
                "logical_call_key": logical_call_key,
                "conservative_input_tokens": input_tokens,
                "maximum_output_tokens": output_tokens,
                "estimated_usd": estimated_usd,
            }
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        if masked_messages["legacy_156"] != masked_messages["full_12429"]:
            raise RuntimeError(
                "paired prompts differ outside the strategy evidence section: "
                f"{candidate['unit_id']}"
            )
        non_strategy_sha = sha256_text(
            canonical_json(masked_messages["legacy_156"])
        )
        call_order = list(
            BANK_CONDITIONS
            if str(candidate["unit_id"]) in legacy_first_units
            else reversed(BANK_CONDITIONS)
        )
        for order_index, bank_condition in enumerate(call_order, 1):
            row = calls_by_bank[bank_condition]
            row["within_pair_call_order"] = order_index
            row["non_strategy_messages_sha256"] = non_strategy_sha
            plan.append(row)
        enriched_samples.append(
            {
                **candidate,
                "sample_index": sample_index,
                "sample_seed": int(sample_seed),
                "diagnostic_label": diagnostic_label,
                "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
                "query_sha256": sha256_text(query),
                "non_strategy_messages_sha256": non_strategy_sha,
                "retrievals": retrievals,
                "paired_invariance": {
                    "same_frozen_turn": True,
                    "same_current_seeker_text": True,
                    "same_recent_dialogue": True,
                    "same_requested_action": True,
                    "same_selected_memory": True,
                    "same_replay_generator_treatment": True,
                    "same_system_prompt": True,
                    "same_temperature": True,
                    "same_max_output_tokens": True,
                    "same_generator_seed": True,
                    "same_output_normalization": True,
                    "only_strategy_evidence_section_differs": True,
                },
            }
        )

    logical_keys = [str(row["logical_call_key"]) for row in plan]
    if len(plan) != 2 * int(sample_size) or len(logical_keys) != len(set(logical_keys)):
        raise RuntimeError("paired call plan cardinality or uniqueness is invalid")
    call_plan_sha256 = sha256_text(canonical_json(plan))
    estimated_usd = (
        total_input_tokens * float(input_usd_per_million_tokens)
        + total_output_tokens * float(output_usd_per_million_tokens)
    ) / 1_000_000.0
    estimate_payload = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "diagnostic_label": diagnostic_label,
        "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
        "input_classification": (
            "CANONICAL_FROZEN_V1"
            if canonical_frozen_v1_inputs
            else "NONCANONICAL_DEBUG_ONLY"
        ),
        "claim_boundary": (
            "Conditional downstream mechanism diagnostic after freezing the "
            "V1 PM+RS action; not a total-effect or routing diagnostic; not "
            "confirmatory; not a PM-v2 training, study-freeze, or release gate"
        ),
        "conditional_estimand": CONDITIONAL_ESTIMAND,
        "excluded_estimands": list(EXCLUDED_ESTIMANDS),
        "confirmatory": False,
        "v2_training_gate": False,
        "inputs": bindings,
        "source_condition": source_condition,
        "sample_size": int(sample_size),
        "sample_seed": int(sample_seed),
        "sample_selection_protocol": (
            "hash-ranked within user, hash-ranked user order, round-robin-v1"
        ),
        "eligible_user_count": len(sampling_counts["eligible_per_user"]),
        "eligible_turn_count": sum(sampling_counts["eligible_per_user"].values()),
        "eligible_per_user": sampling_counts["eligible_per_user"],
        "selected_per_user": sampling_counts["selected_per_user"],
        "strategy_bank_card_counts": {
            "legacy_156": len(legacy_cards),
            "full_12429": len(full_cards),
        },
        "canonical_v1_retriever_sha256": CANONICAL_V1_RETRIEVER_SHA256,
        "strategy_top_k": CANONICAL_V1_TOP_K,
        "replay_generator_treatment": endpoint_binding,
        "replay_generator_treatment_note": (
            "A common endpoint/model treatment held fixed across both bank arms; "
            "its binding is not claimed to prove the historical identity of the "
            "original V1 generator."
        ),
        "common_system_prompt_id": COMMON_PROMPT_ID,
        "common_system_prompt_sha256": prompt_sha256,
        "temperature": GENERATOR_TEMPERATURE,
        "max_output_tokens": GENERATOR_MAX_OUTPUT_TOKENS,
        "output_normalization": OUTPUT_NORMALIZATION,
        "maximum_physical_attempts_per_logical_call": (
            MAXIMUM_PHYSICAL_ATTEMPTS_PER_LOGICAL_CALL
        ),
        "planned_logical_calls": len(plan),
        "maximum_physical_api_attempts": len(plan),
        "expected_call_formula": "2 * sample_size",
        "input_token_safety_factor": float(input_token_safety_factor),
        "conservative_total_input_tokens": total_input_tokens,
        "maximum_total_output_tokens": total_output_tokens,
        "pricing_usd_per_million_tokens": {
            "input": float(input_usd_per_million_tokens),
            "output": float(output_usd_per_million_tokens),
        },
        "maximum_estimated_usd": estimated_usd,
        "call_plan_sha256": call_plan_sha256,
    }
    estimate = {
        **estimate_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(estimate_payload)),
    }
    return estimate, enriched_samples, plan


def _write_annotation_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in ANNOTATION_FIELDS
                }
            )
    os.replace(temporary, path)


def _review_plan(
    samples: Sequence[Mapping[str, Any]],
    sample_seed: int,
    *,
    diagnostic_label: str,
    canonical_frozen_v1_inputs: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    legacy_a_units = _balanced_first_assignment(
        [str(sample["unit_id"]) for sample in samples],
        sample_seed=sample_seed,
        domain="posthoc-v1-bank-human-blind-v1",
    )
    for sample in samples:
        pair_id = str(sample["unit_id"])
        first = list(
            BANK_CONDITIONS
            if pair_id in legacy_a_units
            else reversed(BANK_CONDITIONS)
        )
        for reviewer_id, mapping, order_variant in (
            ("reviewer_a", first, 1),
            ("reviewer_b", list(reversed(first)), 2),
        ):
            rows.append(
                {
                    "diagnostic_label": diagnostic_label,
                    "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
                    "confirmatory": False,
                    "v2_training_gate": False,
                    "item_id": f"{pair_id}_order{order_variant}",
                    "pair_id": pair_id,
                    "reviewer_id": reviewer_id,
                    "order_variant": order_variant,
                    "response_A_condition": mapping[0],
                    "response_B_condition": mapping[1],
                }
            )
    return rows


def persist_posthoc_v1_bank_dry_run(
    out_dir: str | Path,
    estimate: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    call_plan: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> str:
    """Persist or validate a complete no-client plan and review template."""

    out = Path(out_dir)
    estimate_path = out / "cost_estimate.json"
    samples_path = out / "sampled_units.jsonl"
    plan_path = out / "paired_call_plan.jsonl"
    ledger_path = out / "physical_attempt_ledger.jsonl"
    paths = (estimate_path, samples_path, plan_path)
    if overwrite and ledger_path.is_file() and ledger_path.stat().st_size > 0:
        raise RuntimeError("cannot overwrite a diagnostic after any physical attempt")
    normalized = (
        dict(estimate),
        [dict(row) for row in samples],
        [dict(row) for row in call_plan],
    )
    if any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise RuntimeError("saved post-hoc diagnostic dry-run bundle is partial")
        existing = (
            read_json(estimate_path),
            list(iter_jsonl(samples_path)),
            list(iter_jsonl(plan_path)),
        )
        if existing == normalized:
            return "VALIDATED_EXISTING"
        if not overwrite:
            raise RuntimeError(
                "current diagnostic plan differs from saved dry-run; use a new "
                "output directory or --overwrite before any physical attempt"
            )
    out.mkdir(parents=True, exist_ok=True)
    write_json(estimate_path, normalized[0])
    write_jsonl(samples_path, normalized[1])
    write_jsonl(plan_path, normalized[2])
    review_plan = _review_plan(
        samples,
        int(estimate["sample_seed"]),
        diagnostic_label=str(estimate["diagnostic_label"]),
        canonical_frozen_v1_inputs=bool(
            estimate["canonical_frozen_v1_inputs"]
        ),
    )
    write_jsonl(out / "private_blind_review_plan.jsonl", review_plan)
    schema = review_annotation_schema(
        diagnostic_label=str(estimate["diagnostic_label"]),
        canonical_frozen_v1_inputs=bool(
            estimate["canonical_frozen_v1_inputs"]
        ),
    )
    write_json(out / "review_annotation_schema.json", schema)
    for reviewer_id in ("reviewer_a", "reviewer_b"):
        reviewer_rows = [
            {
                "item_id": row["item_id"],
                "pair_id": row["pair_id"],
                "reviewer_id": reviewer_id,
            }
            for row in review_plan
            if row["reviewer_id"] == reviewer_id
        ]
        _write_annotation_csv(out / f"{reviewer_id}_template.csv", reviewer_rows)
    return "WRITTEN"


def review_annotation_schema(
    *,
    diagnostic_label: str = DIAGNOSTIC_LABEL,
    canonical_frozen_v1_inputs: bool = True,
) -> dict[str, Any]:
    return {
        "protocol": "posthoc-v1-bank-dual-order-human-review-v1",
        "diagnostic_label": diagnostic_label,
        "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
        "input_classification": (
            "CANONICAL_FROZEN_V1"
            if canonical_frozen_v1_inputs
            else "NONCANONICAL_DEBUG_ONLY"
        ),
        "claim_boundary": (
            "Conditional downstream mechanism diagnostic after freezing the "
            "V1 PM+RS action; not a routing or total-effect estimate; not "
            "confirmatory; not a V2 gate"
        ),
        "conditional_estimand": CONDITIONAL_ESTIMAND,
        "excluded_estimands": list(EXCLUDED_ESTIMANDS),
        "assignment": (
            "Each pair is independently reviewed twice. Reviewer B receives the "
            "exact reverse order of reviewer A. Reviewers must not access the "
            "private blind mapping or paired call plan."
        ),
        "fields": {
            "overall_preference": ["A", "B", "tie", "unable"],
            "emotional_support_A/B": "integer 1..5",
            "contextual_fit_A/B": "integer 1..5",
            "non_intrusiveness_A/B": "integer 1..5; 5 is least intrusive",
            "strategy_relevance_A/B": "integer 1..5",
            "safety_A/B": ["safe", "concern", "unable"],
            "confidence": ["low", "medium", "high"],
            "notes": "optional concise rationale; do not guess condition identity",
        },
        "analysis": {
            "unit": "frozen V1 turn pair",
            "primary_descriptive_outcome": (
                "full-bank preference minus legacy-bank preference after unblinding"
            ),
            "secondary_outcomes": [
                "within-pair score differences for emotional support",
                "within-pair score differences for contextual fit",
                "within-pair score differences for non-intrusiveness",
                "within-pair score differences for strategy relevance",
                "safety-concern discordance",
            ],
            "order_check": (
                "Report underlying-response agreement across reversed A/B order; "
                "do not collapse a pair with unresolved reviewer disagreement."
            ),
            "reporting": (
                "Report counts, paired proportions/differences, uncertainty, and "
                "the post-hoc non-confirmatory label. Do not use the result to "
                "authorize PM-v2 training or external claims."
            ),
        },
    }


def _require_saved_plan(
    out_dir: Path,
    estimate: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    call_plan: Sequence[Mapping[str, Any]],
    *,
    accepted_cost_estimate_sha256: str | None,
) -> None:
    expected_hash = str(estimate["cost_estimate_sha256"])
    if accepted_cost_estimate_sha256 != expected_hash:
        raise RuntimeError(
            "--accept-cost-estimate-sha256 must exactly match the current dry-run: "
            f"{expected_hash}"
        )
    required = {
        "cost_estimate.json": dict(estimate),
        "sampled_units.jsonl": [dict(row) for row in samples],
        "paired_call_plan.jsonl": [dict(row) for row in call_plan],
    }
    for name, expected in required.items():
        path = out_dir / name
        if not path.is_file():
            raise RuntimeError(f"API mode requires a matching saved --dry-run: {path}")
        actual = read_json(path) if path.suffix == ".json" else list(iter_jsonl(path))
        if actual != expected:
            raise RuntimeError(f"saved diagnostic dry-run is stale: {path}")


def _validate_runtime_budget(
    estimate: Mapping[str, Any],
    *,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
) -> None:
    if int(max_api_calls) < 1 or int(max_input_tokens_per_call) < 1:
        raise ValueError("API and input-token caps must be positive")
    if not math.isfinite(float(max_estimated_usd)) or float(max_estimated_usd) <= 0:
        raise ValueError("max_estimated_usd must be finite and positive")
    if int(estimate["maximum_physical_api_attempts"]) > int(max_api_calls):
        raise RuntimeError("diagnostic call plan exceeds --max-api-calls")
    if float(estimate["maximum_estimated_usd"]) > float(max_estimated_usd):
        raise RuntimeError("diagnostic cost estimate exceeds --max-estimated-usd")


def _review_item(
    sample: Mapping[str, Any],
    mapping: Mapping[str, Any],
    responses: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    pair_id = str(sample["unit_id"])
    a_condition = str(mapping["response_A_condition"])
    b_condition = str(mapping["response_B_condition"])
    return {
        "protocol": "posthoc-v1-bank-dual-order-human-review-v1",
        "diagnostic_label": mapping["diagnostic_label"],
        "canonical_frozen_v1_inputs": mapping[
            "canonical_frozen_v1_inputs"
        ],
        "conditional_estimand": CONDITIONAL_ESTIMAND,
        "excluded_estimands": list(EXCLUDED_ESTIMANDS),
        "confirmatory": False,
        "v2_training_gate": False,
        "item_id": mapping["item_id"],
        "pair_id": pair_id,
        "reviewer_id": mapping["reviewer_id"],
        "order_variant": mapping["order_variant"],
        "visible_dialogue_before_current_turn": sample["context_before_turn"],
        "current_seeker_message": sample["seeker_message"],
        "same_authorized_memory_for_both_responses": sample["selected_memory"],
        "response_A": responses[(pair_id, a_condition)],
        "response_B": responses[(pair_id, b_condition)],
        "review_instruction": (
            "Compare only the two responses using the supplied shared context. "
            "Do not infer which Strategy Bank produced A or B."
        ),
    }


def run_posthoc_v1_bank_probe(
    out_dir: str | Path,
    estimate: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    call_plan: Sequence[Mapping[str, Any]],
    *,
    endpoint: Endpoint,
    accepted_cost_estimate_sha256: str,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    """Execute an accepted plan with exactly one attempt and complete-only gate."""

    out = Path(out_dir)
    _require_saved_plan(
        out,
        estimate,
        samples,
        call_plan,
        accepted_cost_estimate_sha256=accepted_cost_estimate_sha256,
    )
    _validate_runtime_budget(
        estimate,
        max_api_calls=max_api_calls,
        max_estimated_usd=max_estimated_usd,
        max_input_tokens_per_call=max_input_tokens_per_call,
    )
    canonical_frozen_v1_inputs = bool(
        estimate.get("canonical_frozen_v1_inputs")
    )
    diagnostic_label = str(estimate.get("diagnostic_label") or "")
    expected_label = (
        DIAGNOSTIC_LABEL
        if canonical_frozen_v1_inputs
        else NONCANONICAL_DEBUG_LABEL
    )
    if diagnostic_label != expected_label:
        raise RuntimeError("diagnostic label/input-classification mismatch")
    replay_treatment = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "family": endpoint.family,
        "timeout_seconds": endpoint.timeout_seconds,
    }
    if estimate.get("replay_generator_treatment") != replay_treatment or any(
        row.get("replay_generator_treatment") != replay_treatment
        for row in call_plan
    ):
        raise RuntimeError("runtime replay generator treatment differs from dry-run")
    if any(
        int(row["conservative_input_tokens"]) > int(max_input_tokens_per_call)
        for row in call_plan
    ):
        raise RuntimeError("a planned diagnostic prompt exceeds the input-token cap")

    ledger_path = out / "physical_attempt_ledger.jsonl"
    raw_path = out / "raw_api_calls.jsonl"
    expected_calls = {str(row["logical_call_key"]): 1 for row in call_plan}
    ledger = PersistentAttemptLedger(
        ledger_path,
        stage=diagnostic_label,
        expected_calls=expected_calls,
        maximum_total_attempts=len(expected_calls),
    )
    pending_rows: list[Mapping[str, Any]] = []
    for plan_row in call_plan:
        call_key = str(plan_row["logical_call_key"])
        terminal = ledger.terminal_row(call_key)
        if ledger.succeeded(call_key):
            result = dict((terminal or {}).get("result") or {})
            if (
                result.get("normalized_finish_reason") != "complete"
                or not str(result.get("response") or "").strip()
            ):
                raise RuntimeError("successful diagnostic ledger result is invalid")
            continue
        if ledger.attempts_for(call_key):
            raise RuntimeError(
                "diagnostic logical call already spent its one physical attempt: "
                f"{call_key}"
            )
        pending_rows.append(plan_row)

    # Client construction validates credentials and endpoint initialization but
    # performs no HTTP request.  It must happen before reserving an attempt so a
    # missing key or local constructor failure cannot consume the sole attempt.
    client = None
    if pending_rows:
        factory = client_factory or make_client
        client = factory(endpoint)
    try:
        for plan_row in pending_rows:
            call_key = str(plan_row["logical_call_key"])
            record_ids = {
                "diagnostic_label": diagnostic_label,
                "pair_id": plan_row["pair_id"],
                "bank_condition": plan_row["bank_condition"],
                "logical_call_key": call_key,
            }
            reservation = ledger.reserve(
                call_key,
                record_ids=record_ids,
                prompt_sha256=str(plan_row["messages_sha256"]),
            )
            call = None
            raw_written = False
            try:
                assert client is not None
                call, _ = client.chat(
                    list(plan_row["messages"]),
                    temperature=GENERATOR_TEMPERATURE,
                    max_tokens=GENERATOR_MAX_OUTPUT_TOKENS,
                    seed=int(plan_row["generator_seed"]),
                    response_schema=None,
                    retries=1,
                )
                usage = require_reported_usage(
                    call.usage, stage=diagnostic_label
                )
                call_gate_errors: list[str] = []
                prompt_token_error = reported_prompt_token_error(
                    usage,
                    maximum_prompt_tokens=int(
                        plan_row["conservative_input_tokens"]
                    ),
                    stage=diagnostic_label,
                    require_positive=True,
                )
                if prompt_token_error is not None:
                    call_gate_errors.append(prompt_token_error)
                if int(usage["completion_tokens"]) > int(
                    plan_row["maximum_output_tokens"]
                ):
                    call_gate_errors.append(
                        f"{diagnostic_label} reported completion_tokens exceed "
                        "the frozen "
                        "per-call output bound: "
                        f"reported={usage['completion_tokens']}, "
                        f"bound={plan_row['maximum_output_tokens']}"
                    )
                if call.normalized_finish_reason != "complete":
                    call_gate_errors.append(
                        "supporter generation rejected by complete-only finish gate: "
                        f"provider={call.provider_finish_reason!r}, "
                        f"normalized={call.normalized_finish_reason!r}"
                    )
                call_gate_error = (
                    " | ".join(call_gate_errors) if call_gate_errors else None
                )
                append_jsonl(
                    raw_path,
                    request_log(
                        stage=diagnostic_label,
                        endpoint=endpoint,
                        messages=list(plan_row["messages"]),
                        result=call,
                        parsed=None,
                        error=call_gate_error,
                        prompt_hash=str(plan_row["messages_sha256"]),
                        record_ids=record_ids,
                    ),
                )
                raw_written = True
                if call_gate_error is not None:
                    ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=call.request_hash,
                        usage=usage,
                        error=call_gate_error,
                        result={
                            "provider_finish_reason": call.provider_finish_reason,
                            "normalized_finish_reason": call.normalized_finish_reason,
                            "reported_prompt_tokens": usage["prompt_tokens"],
                            "maximum_prompt_tokens": plan_row[
                                "conservative_input_tokens"
                            ],
                            "reported_completion_tokens": usage[
                                "completion_tokens"
                            ],
                            "maximum_completion_tokens": plan_row[
                                "maximum_output_tokens"
                            ],
                        },
                    )
                    raise RuntimeError(call_gate_error)
                response = normalize_space(call.text)
                if not response:
                    raise RuntimeError("complete diagnostic call normalized to empty output")
                ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=call.request_hash,
                    usage=usage,
                    error=None,
                    result={
                        "response": response,
                        "provider_finish_reason": call.provider_finish_reason,
                        "normalized_finish_reason": call.normalized_finish_reason,
                    },
                    metadata={
                        "bank_sha256": plan_row["bank_sha256"],
                        "retrieval_sha256": plan_row["retrieval"]["retrieval_sha256"],
                        "non_strategy_messages_sha256": plan_row[
                            "non_strategy_messages_sha256"
                        ],
                    },
                )
            except Exception as exc:
                if not raw_written:
                    append_jsonl(
                        raw_path,
                        request_log(
                            stage=diagnostic_label,
                            endpoint=endpoint,
                            messages=list(plan_row["messages"]),
                            result=call,
                            parsed=None,
                            error=f"{type(exc).__name__}: {exc}",
                            prompt_hash=str(plan_row["messages_sha256"]),
                            record_ids=record_ids,
                        ),
                    )
                if ledger.terminal_row(call_key) is None:
                    ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=call.request_hash if call else None,
                        usage=call.usage if call else None,
                        error=f"{type(exc).__name__}: {exc}",
                        result=(
                            {
                                "provider_finish_reason": call.provider_finish_reason,
                                "normalized_finish_reason": call.normalized_finish_reason,
                            }
                            if call
                            else None
                        ),
                    )
                raise
    finally:
        if client is not None:
            client.close()

    responses: dict[tuple[str, str], str] = {}
    generation_rows: list[dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for plan_row in call_plan:
        call_key = str(plan_row["logical_call_key"])
        terminal = ledger.terminal_row(call_key)
        if terminal is None or terminal.get("event") != "SUCCEEDED":
            raise RuntimeError("diagnostic run ended without every planned success")
        result = dict(terminal.get("result") or {})
        response = str(result.get("response") or "")
        pair_id = str(plan_row["pair_id"])
        bank_condition = str(plan_row["bank_condition"])
        responses[(pair_id, bank_condition)] = response
        usage = dict(terminal.get("usage") or {})
        for key in total_usage:
            total_usage[key] += int(usage.get(key) or 0)
        generation_rows.append(
            {
                "protocol": DIAGNOSTIC_PROTOCOL,
                "diagnostic_label": diagnostic_label,
                "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
                "conditional_estimand": CONDITIONAL_ESTIMAND,
                "excluded_estimands": list(EXCLUDED_ESTIMANDS),
                "confirmatory": False,
                "v2_training_gate": False,
                "pair_id": pair_id,
                "bank_condition": bank_condition,
                "bank_sha256": plan_row["bank_sha256"],
                "retrieval_sha256": plan_row["retrieval"]["retrieval_sha256"],
                "non_strategy_messages_sha256": plan_row[
                    "non_strategy_messages_sha256"
                ],
                "logical_call_key": call_key,
                "response": response,
                "provider_finish_reason": result.get("provider_finish_reason"),
                "normalized_finish_reason": result.get("normalized_finish_reason"),
                "usage": usage,
            }
        )
    prices = estimate["pricing_usd_per_million_tokens"]
    observed_usd = (
        total_usage["prompt_tokens"] * float(prices["input"])
        + total_usage["completion_tokens"] * float(prices["output"])
    ) / 1_000_000.0
    accepted_estimated_usd = float(estimate["maximum_estimated_usd"])
    observed_budget_gate = {
        "diagnostic_label": diagnostic_label,
        "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
        "status": "PASS",
        "observed_usd": observed_usd,
        "accepted_cost_estimate_usd_ceiling": accepted_estimated_usd,
        "cli_max_estimated_usd_ceiling": float(max_estimated_usd),
        "within_accepted_cost_estimate": observed_usd <= accepted_estimated_usd,
        "within_cli_max_estimated_usd": observed_usd <= float(max_estimated_usd),
    }
    if not (
        observed_budget_gate["within_accepted_cost_estimate"]
        and observed_budget_gate["within_cli_max_estimated_usd"]
    ):
        observed_budget_gate["status"] = "FAIL"
        write_json(out / "observed_budget_gate_failure.json", observed_budget_gate)
        raise RuntimeError(
            "reported diagnostic cost exceeded an accepted budget ceiling: "
            f"observed={observed_usd}, accepted={accepted_estimated_usd}, "
            f"cli={float(max_estimated_usd)}"
        )
    write_jsonl(out / "paired_generations.jsonl", generation_rows)

    blind_plan = list(iter_jsonl(out / "private_blind_review_plan.jsonl"))
    sample_index = {str(row["unit_id"]): row for row in samples}
    review_items: list[dict[str, Any]] = []
    for mapping in blind_plan:
        review_items.append(
            _review_item(sample_index[str(mapping["pair_id"])], mapping, responses)
        )
    write_jsonl(out / "dual_order_review_items.jsonl", review_items)
    for reviewer_id in ("reviewer_a", "reviewer_b"):
        items = [row for row in review_items if row["reviewer_id"] == reviewer_id]
        write_jsonl(out / f"{reviewer_id}_items.jsonl", items)
        _write_annotation_csv(
            out / f"{reviewer_id}.csv",
            [
                {
                    "item_id": row["item_id"],
                    "pair_id": row["pair_id"],
                    "reviewer_id": reviewer_id,
                }
                for row in items
            ],
        )
    summary = {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "diagnostic_label": diagnostic_label,
        "canonical_frozen_v1_inputs": canonical_frozen_v1_inputs,
        "status": "COMPLETE",
        "claim_boundary": (
            "Conditional downstream mechanism diagnostic after freezing the "
            "V1 PM+RS action; not a routing or total-effect estimate; not "
            "confirmatory; not a V2 gate"
        ),
        "conditional_estimand": CONDITIONAL_ESTIMAND,
        "excluded_estimands": list(EXCLUDED_ESTIMANDS),
        "confirmatory": False,
        "v2_training_gate": False,
        "sample_size": len(samples),
        "planned_calls": len(call_plan),
        "successful_calls": len(generation_rows),
        "physical_attempts": ledger.started_attempts,
        "completion_truncated_count": 0,
        "normalized_finish_reason_counts": {"complete": len(generation_rows)},
        "reported_usage": total_usage,
        "observed_usd": observed_usd,
        "observed_budget_gate": observed_budget_gate,
        "accepted_cost_estimate_sha256": accepted_cost_estimate_sha256,
        "artifacts": {
            name: sha256_file(out / name)
            for name in (
                "paired_generations.jsonl",
                "dual_order_review_items.jsonl",
                "reviewer_a.csv",
                "reviewer_b.csv",
                "review_annotation_schema.json",
                "private_blind_review_plan.jsonl",
                "physical_attempt_ledger.jsonl",
                "raw_api_calls.jsonl",
            )
        },
    }
    write_json(out / "summary.json", summary)
    return summary
