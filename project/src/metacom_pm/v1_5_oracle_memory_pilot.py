from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .contracts import MemoryBackendRecord, MemorySource
from .io import canonical_json, sha256_text
from .pm_v2_contracts import PMV2State


ORACLE_MEMORY_PILOT_PROTOCOL = (
    "pm-v1.5-train-only-oracle-memory-uptake-upper-bound-pilot-v1"
)
ORACLE_MEMORY_PILOT_SEED = 27191
ORACLE_MEMORY_PILOT_REGIME_COUNTS = {
    "profile_needed": 3,
    "summary_needed": 3,
    "event_needed": 3,
    "multi_source_needed": 3,
    "memory_harmful": 6,
}
CONTROL_ARM = "frozen_m0_r0_control"
HELPFUL_ARM = "oracle_helpful_memory_same_prompt"
HARMFUL_ARM = "oracle_harmful_memory_same_prompt"


def _target_action(regime: str, sources: frozenset[str]) -> str:
    if regime == "memory_harmful":
        if sources != frozenset({"MP", "MS", "ME"}):
            raise RuntimeError("memory-harmful oracle must cover MP/MS/ME")
        return "MPMSME+R0"
    codes = {
        frozenset({"MP"}): "MP",
        frozenset({"MS"}): "MS",
        frozenset({"ME"}): "ME",
        frozenset({"MP", "MS"}): "MPMS",
        frozenset({"MP", "ME"}): "MPE",
        frozenset({"MS", "ME"}): "MSE",
        frozenset({"MP", "MS", "ME"}): "MPMSME",
    }
    if sources not in codes:
        raise RuntimeError(f"unsupported oracle-memory source set: {sorted(sources)}")
    return f"{codes[sources]}+R0"


def select_oracle_memory_pilot_states(
    states: Sequence[PMV2State],
    evaluator_by_state: Mapping[str, Mapping[str, Any]],
    backend_by_card: Mapping[str, MemoryBackendRecord],
    *,
    excluded_state_ids: set[str],
    prefer_unused_user_ids: set[str] | None = None,
    seed: int = ORACLE_MEMORY_PILOT_SEED,
) -> list[dict[str, Any]]:
    """Select fresh train states without consulting response or judge outcomes."""

    prefer_unused_user_ids = set(prefer_unused_user_ids or ())
    by_regime: dict[str, list[PMV2State]] = defaultdict(list)
    for state in states:
        if state.split.value != "train" or state.state_id in excluded_state_ids:
            continue
        evaluator = evaluator_by_state.get(state.state_id)
        if evaluator is None:
            raise RuntimeError(f"missing evaluator context: {state.state_id}")
        regime = str(evaluator["regime"])
        if regime in ORACLE_MEMORY_PILOT_REGIME_COUNTS:
            by_regime[regime].append(state)

    selected: list[dict[str, Any]] = []
    used_users: set[str] = set()
    for regime, required_count in ORACLE_MEMORY_PILOT_REGIME_COUNTS.items():
        candidates = sorted(
            by_regime.get(regime, []),
            key=lambda state: (
                state.user_id not in prefer_unused_user_ids,
                sha256_text(f"{seed}|{regime}|{state.state_id}"),
                state.state_id,
            ),
        )
        chosen = [state for state in candidates if state.user_id not in used_users][
            :required_count
        ]
        if len(chosen) != required_count:
            raise RuntimeError(
                f"insufficient fresh user-disjoint states for {regime}: "
                f"{len(chosen)} != {required_count}"
            )
        for state in chosen:
            evaluator = evaluator_by_state[state.state_id]
            backend = backend_by_card.get(state.card_id)
            if backend is None:
                raise RuntimeError(f"missing memory backend: {state.card_id}")
            desired_utility = (
                "harmful" if regime == "memory_harmful" else "helpful"
            )
            annotations = {
                str(row["memory_id"]): row
                for row in evaluator.get("memory_annotations") or []
                if str(row.get("item_utility")) == desired_utility
            }
            if not annotations:
                raise RuntimeError(
                    f"state lacks annotated {desired_utility} memory: {state.state_id}"
                )
            item_by_id = {item.memory_id: item for item in backend.items}
            missing = sorted(set(annotations) - set(item_by_id))
            if missing:
                raise RuntimeError(
                    f"annotated memories are absent from backend for "
                    f"{state.state_id}: {missing}"
                )
            memory_ids = sorted(
                annotations,
                key=lambda memory_id: (
                    str(annotations[memory_id]["source"]),
                    memory_id,
                ),
            )
            sources = frozenset(
                str(annotations[memory_id]["source"]) for memory_id in memory_ids
            )
            if desired_utility == "helpful":
                expected_sources = frozenset(
                    str(value)
                    for value in evaluator.get("needed_memory_sources") or []
                )
                if sources != expected_sources:
                    raise RuntimeError(
                        f"helpful oracle source coverage drifted for {state.state_id}: "
                        f"{sorted(sources)} != {sorted(expected_sources)}"
                    )
                arm = HELPFUL_ARM
            else:
                if len(memory_ids) != 3 or sources != frozenset(
                    source.value for source in MemorySource
                ):
                    raise RuntimeError(
                        f"harmful oracle must contain one MP/MS/ME item: "
                        f"{state.state_id}"
                    )
                arm = HARMFUL_ARM
            target_action_id = _target_action(regime, sources)
            if target_action_id not in state.allowed_actions:
                raise RuntimeError(
                    f"oracle action is illegal: {state.state_id}/{target_action_id}"
                )
            selected.append(
                {
                    "state_id": state.state_id,
                    "card_id": state.card_id,
                    "user_id": state.user_id,
                    "semantic_family": state.semantic_family,
                    "regime": regime,
                    "treatment_arm": arm,
                    "target_action_id": target_action_id,
                    "target_item_utility": desired_utility,
                    "target_memory_ids": memory_ids,
                    "target_memory_sources": sorted(sources),
                }
            )
            used_users.add(state.user_id)
    return selected


def build_oracle_memory_pilot_contract(
    *,
    selected_states: Sequence[Mapping[str, Any]],
    source_lineage: Mapping[str, Any],
    seed: int = ORACLE_MEMORY_PILOT_SEED,
) -> dict[str, Any]:
    expected = sum(ORACLE_MEMORY_PILOT_REGIME_COUNTS.values())
    if len(selected_states) != expected:
        raise RuntimeError(
            f"oracle-memory pilot state count changed: "
            f"{len(selected_states)} != {expected}"
        )
    state_ids = [str(row["state_id"]) for row in selected_states]
    card_ids = [str(row["card_id"]) for row in selected_states]
    user_ids = [str(row["user_id"]) for row in selected_states]
    if len(state_ids) != len(set(state_ids)):
        raise RuntimeError("oracle-memory pilot repeats a state")
    if len(card_ids) != len(set(card_ids)):
        raise RuntimeError("oracle-memory pilot repeats a card")
    if len(user_ids) != len(set(user_ids)):
        raise RuntimeError("oracle-memory pilot users are not disjoint")
    regime_counts = {
        regime: sum(str(row["regime"]) == regime for row in selected_states)
        for regime in ORACLE_MEMORY_PILOT_REGIME_COUNTS
    }
    if regime_counts != ORACLE_MEMORY_PILOT_REGIME_COUNTS:
        raise RuntimeError(
            f"oracle-memory regime balance changed: {regime_counts}"
        )

    payload = {
        "protocol": ORACLE_MEMORY_PILOT_PROTOCOL,
        "status": "READY_ZERO_API_DESIGN",
        "scope": "longitudinal_train_only",
        "purpose": (
            "separate retrieval failure from generator resource-uptake failure "
            "by exposing only evaluator-annotated memory items"
        ),
        "seed": seed,
        "regime_counts": dict(ORACLE_MEMORY_PILOT_REGIME_COUNTS),
        "arms": {
            CONTROL_ARM: {
                "new_api_calls": 0,
                "action_id": "M0+R0",
                "source": "existing attested longitudinal sweep outcome",
            },
            HELPFUL_ARM: {
                "new_api_calls_per_eligible_state": 1,
                "evidence": "all and only item_utility=helpful memories",
                "retrieval_score_gate": "bypassed for diagnostic upper bound",
                "prompt_and_generator": "frozen and unchanged",
            },
            HARMFUL_ARM: {
                "new_api_calls_per_eligible_state": 1,
                "evidence": "all and only item_utility=harmful memories",
                "retrieval_score_gate": "bypassed for diagnostic upper bound",
                "prompt_and_generator": "frozen and unchanged",
            },
        },
        "selected_states": [dict(row) for row in selected_states],
        "selected_state_ids_sha256": sha256_text(
            canonical_json(sorted(state_ids))
        ),
        "source_lineage": dict(source_lineage),
        "planned_new_logical_calls": expected,
        "evaluation": {
            "api_judges": "forbidden",
            "primary_helpful_rule": {
                "pairs": 12,
                "mean_target_evidence_bge_cosine_delta": ">0",
                "positive_pairs": ">=8",
            },
            "harmful_selectivity_guardrail": {
                "pairs": 6,
                "mean_target_evidence_bge_cosine_delta": "<=0",
                "nonpositive_pairs": ">=4",
            },
            "cluster_unit": "user_id",
            "interpretation": (
                "Report-only mechanism upper bound. Passing establishes "
                "memory-uptake capacity under oracle evidence, not response "
                "quality, deployable routing, judge validity, or PM training labels."
            ),
        },
        "strategy_scope": (
            "excluded because evaluator truth specifies strategy use/skip but "
            "does not identify an item-level helpful strategy card"
        ),
        "stop_rules": [
            "do not change states, arms, thresholds, prompt, generator, or seed after outcomes",
            "do not read calibration or internal-test",
            "do not call API judges or create action labels",
            "do not use evaluator annotations in a deployable PM or external test",
            "if helpful uptake is not established, localize the blocker to generator/prompt integration",
            "if helpful uptake is established, test response quality only with a separately preregistered small validation",
        ],
        "api_judges_used": False,
        "training_labels_created": False,
    }
    return {**payload, "contract_sha256": sha256_text(canonical_json(payload))}


def materialize_oracle_backend(
    *,
    selected_states: Sequence[Mapping[str, Any]],
    backend_by_card: Mapping[str, MemoryBackendRecord],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected in selected_states:
        card_id = str(selected["card_id"])
        wanted = set(str(value) for value in selected["target_memory_ids"])
        backend = backend_by_card.get(card_id)
        if backend is None:
            raise RuntimeError(f"missing backend for oracle materialization: {card_id}")
        items = [item for item in backend.items if item.memory_id in wanted]
        if {item.memory_id for item in items} != wanted:
            raise RuntimeError(f"oracle backend is incomplete: {card_id}")
        items.sort(key=lambda item: (item.source.value, item.memory_id))
        rows.append(
            MemoryBackendRecord(card_id=card_id, items=items).model_dump(mode="json")
        )
    rows.sort(key=lambda row: str(row["card_id"]))
    return rows
