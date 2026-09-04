from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .io import canonical_json, sha256_text
from .pm_v2_contracts import PMV2State


RESPONSE_MECHANISM_PILOT_PROTOCOL = (
    "pm-v1.5-train-only-response-mechanism-uptake-pilot-v1"
)
RESPONSE_MECHANISM_PILOT_REGIMES = (
    "profile_needed",
    "summary_needed",
    "event_needed",
    "multi_source_needed",
    "memory_harmful",
    "strategy_helpful",
    "strategy_harmful",
)
RESPONSE_MECHANISM_PILOT_ARMS = (
    "frozen_current_control",
    "top1_evidence_surface_same_prompt",
)


def _target_action(evaluator_row: Mapping[str, Any]) -> str:
    regime = str(evaluator_row["regime"])
    if regime in {"strategy_helpful", "strategy_harmful"}:
        return "M0+RS"
    if regime == "memory_harmful":
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
    sources = frozenset(str(value) for value in evaluator_row["needed_memory_sources"])
    if sources not in codes:
        raise RuntimeError(
            f"response-mechanism pilot lacks a target action for {sorted(sources)}"
        )
    return f"{codes[sources]}+R0"


def select_response_mechanism_pilot_states(
    states: Sequence[PMV2State],
    evaluator_by_state: Mapping[str, Mapping[str, Any]],
    *,
    states_per_regime: int = 2,
    seed: int = 9187,
) -> list[dict[str, Any]]:
    """Select train-only, cross-regime user-disjoint diagnostic states."""

    if states_per_regime < 1:
        raise ValueError("states_per_regime must be positive")
    by_regime: dict[str, list[PMV2State]] = defaultdict(list)
    for state in states:
        if str(state.split.value) != "train":
            continue
        row = evaluator_by_state.get(state.state_id)
        if row is None:
            raise RuntimeError(f"missing evaluator context for {state.state_id}")
        regime = str(row["regime"])
        if regime in RESPONSE_MECHANISM_PILOT_REGIMES:
            by_regime[regime].append(state)
    selected: list[dict[str, Any]] = []
    used_users: set[str] = set()
    for regime in RESPONSE_MECHANISM_PILOT_REGIMES:
        candidates = sorted(
            by_regime.get(regime, []),
            key=lambda state: (
                sha256_text(f"{seed}|{regime}|{state.state_id}"),
                state.state_id,
            ),
        )
        chosen = [state for state in candidates if state.user_id not in used_users][
            :states_per_regime
        ]
        if len(chosen) != states_per_regime:
            raise RuntimeError(
                f"insufficient user-disjoint train states for {regime}: "
                f"{len(chosen)} != {states_per_regime}"
            )
        for state in chosen:
            used_users.add(state.user_id)
            evaluator = evaluator_by_state[state.state_id]
            selected.append(
                {
                    "state_id": state.state_id,
                    "card_id": state.card_id,
                    "user_id": state.user_id,
                    "semantic_family": state.semantic_family,
                    "regime": regime,
                    "target_action_id": _target_action(evaluator),
                    "needed_memory_sources": [
                        str(value)
                        for value in evaluator.get("needed_memory_sources", [])
                    ],
                }
            )
    return selected


def response_mechanism_pilot_contract(
    *,
    selected_states: Sequence[Mapping[str, Any]],
    source_lineage: Mapping[str, Any],
    states_per_regime: int = 2,
    seed: int = 9187,
) -> dict[str, Any]:
    expected = len(RESPONSE_MECHANISM_PILOT_REGIMES) * states_per_regime
    if len(selected_states) != expected:
        raise RuntimeError(
            f"response-mechanism pilot state count changed: "
            f"{len(selected_states)} != {expected}"
        )
    state_ids = [str(row["state_id"]) for row in selected_states]
    user_ids = [str(row["user_id"]) for row in selected_states]
    if len(state_ids) != len(set(state_ids)):
        raise RuntimeError("response-mechanism pilot has duplicate states")
    if len(user_ids) != len(set(user_ids)):
        raise RuntimeError("response-mechanism pilot users are not disjoint")
    regime_counts = {
        regime: sum(str(row["regime"]) == regime for row in selected_states)
        for regime in RESPONSE_MECHANISM_PILOT_REGIMES
    }
    if any(value != states_per_regime for value in regime_counts.values()):
        raise RuntimeError("response-mechanism pilot regime balance changed")
    payload = {
        "protocol": RESPONSE_MECHANISM_PILOT_PROTOCOL,
        "status": "READY_ZERO_API_DESIGN",
        "scope": "longitudinal_train_only",
        "purpose": (
            "diagnose retrieval dilution and generator resource uptake; this "
            "pilot does not qualify LLM judges or authorize PM training"
        ),
        "seed": seed,
        "states_per_regime": states_per_regime,
        "regimes": list(RESPONSE_MECHANISM_PILOT_REGIMES),
        "arms": {
            "frozen_current_control": {
                "new_api_calls": 0,
                "retrieval": "frozen MP=2, MS=2, ME=3, strategy=3",
                "prompt": "frozen current prompt",
                "source": "existing attested longitudinal sweep outcome",
            },
            "top1_evidence_surface_same_prompt": {
                "new_api_calls_per_state": 1,
                "retrieval": (
                    "unchanged frozen retrievers followed by deterministic "
                    "post-retrieval max-one-per-source evidence filtering"
                ),
                "prompt_surface": "MP<=1, MS<=1, ME<=1, strategy<=1",
                "prompt": "frozen current prompt",
            },
        },
        "selected_states": [dict(row) for row in selected_states],
        "selected_state_ids_sha256": sha256_text(canonical_json(sorted(state_ids))),
        "source_lineage": dict(source_lineage),
        "planned_new_logical_calls": expected,
        "evaluation": {
            "api_judges": "forbidden",
            "primary_diagnostics": [
                "paired selected-resource/response BGE cosine delta versus frozen control",
                "paired lexical resource-uptake delta versus frozen control",
                "helpful-minus-harmful uptake separation",
                "exact-copy and unsupported-detail guardrails",
            ],
            "cluster_unit": "user_id",
            "interpretation": (
                "This is a report-only causal mechanism diagnostic. It may "
                "localize retrieval dilution but cannot qualify judges, create "
                "action labels, select a PM, or authorize training."
            ),
        },
        "stop_rules": [
            "do not change thresholds, states, arms, prompt, or retrieval after outcomes",
            "do not read calibration or internal-test",
            "do not call final external judges",
            "do not promote the mechanism on uptake diagnostics alone",
        ],
        "training_labels_created": False,
    }
    return {
        **payload,
        "contract_sha256": sha256_text(canonical_json(payload)),
    }
