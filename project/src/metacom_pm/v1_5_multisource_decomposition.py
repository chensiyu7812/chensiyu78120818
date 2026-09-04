from __future__ import annotations

from typing import Any, Mapping, Sequence

from .io import canonical_json, sha256_text
from .v1_5_oracle_memory_pilot import (
    CONTROL_ARM,
    HELPFUL_ARM,
    ORACLE_MEMORY_PILOT_PROTOCOL,
)


MULTISOURCE_DECOMPOSITION_PROTOCOL = (
    "pm-v1.5-train-only-oracle-memory-multisource-decomposition-v1"
)
MULTISOURCE_DECOMPOSITION_STAGE = (
    "longitudinal_oracle_memory_multisource_decomposition_pilot"
)
MULTISOURCE_REGIME = "multi_source_needed"
SINGLE_SOURCE_ACTIONS = {
    "MP": "MP+R0",
    "MS": "MS+R0",
    "ME": "ME+R0",
}


def build_multisource_decomposition_contract(
    *,
    oracle_contract: Mapping[str, Any],
    backend_rows: Sequence[Mapping[str, Any]],
    source_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the three-state, nine-call single-source decomposition."""

    if oracle_contract.get("protocol") != ORACLE_MEMORY_PILOT_PROTOCOL:
        raise RuntimeError("unexpected oracle-memory pilot contract")
    without_sha = {
        key: value
        for key, value in oracle_contract.items()
        if key != "contract_sha256"
    }
    if oracle_contract.get("contract_sha256") != sha256_text(
        canonical_json(without_sha)
    ):
        raise RuntimeError("oracle-memory pilot contract hash mismatch")
    selected = [
        dict(row)
        for row in oracle_contract.get("selected_states") or []
        if str(row["regime"]) == MULTISOURCE_REGIME
    ]
    if len(selected) != 3:
        raise RuntimeError("multisource decomposition requires exactly 3 states")
    if len({str(row["state_id"]) for row in selected}) != 3:
        raise RuntimeError("multisource decomposition repeats a state")
    if len({str(row["user_id"]) for row in selected}) != 3:
        raise RuntimeError("multisource decomposition repeats a user")

    backend_by_card = {
        str(row["card_id"]): row for row in backend_rows
    }
    decomposition_states: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda value: str(value["state_id"])):
        if (
            str(row["treatment_arm"]) != HELPFUL_ARM
            or str(row["target_action_id"]) != "MPMSME+R0"
            or set(str(value) for value in row["target_memory_sources"])
            != set(SINGLE_SOURCE_ACTIONS)
        ):
            raise RuntimeError(
                f"multisource oracle treatment drifted: {row['state_id']}"
            )
        backend = backend_by_card.get(str(row["card_id"]))
        if backend is None:
            raise RuntimeError(
                f"multisource backend missing: {row['state_id']}"
            )
        memory_by_source: dict[str, str] = {}
        for item in backend.get("items") or []:
            source = str(item["source"])
            if source in memory_by_source:
                raise RuntimeError(
                    f"multisource backend repeats {source}: {row['state_id']}"
                )
            memory_by_source[source] = str(item["memory_id"])
        if set(memory_by_source) != set(SINGLE_SOURCE_ACTIONS):
            raise RuntimeError(
                f"multisource backend source coverage drifted: {row['state_id']}"
            )
        if set(memory_by_source.values()) != {
            str(value) for value in row["target_memory_ids"]
        }:
            raise RuntimeError(
                f"multisource backend item coverage drifted: {row['state_id']}"
            )
        decomposition_states.append(
            {
                "state_id": str(row["state_id"]),
                "card_id": str(row["card_id"]),
                "user_id": str(row["user_id"]),
                "semantic_family": str(row["semantic_family"]),
                "existing_control_arm": CONTROL_ARM,
                "existing_all_sources_arm": HELPFUL_ARM,
                "existing_control_action_id": "M0+R0",
                "existing_all_sources_action_id": "MPMSME+R0",
                "single_source_arms": [
                    {
                        "source": source,
                        "action_id": SINGLE_SOURCE_ACTIONS[source],
                        "target_memory_id": memory_by_source[source],
                    }
                    for source in sorted(SINGLE_SOURCE_ACTIONS)
                ],
            }
        )

    record = {
        "protocol": MULTISOURCE_DECOMPOSITION_PROTOCOL,
        "stage": MULTISOURCE_DECOMPOSITION_STAGE,
        "status": "READY_ZERO_API_DESIGN",
        "scope": "longitudinal_train_only_report_only",
        "purpose": (
            "localize the three failed multi-source uptake cases by varying "
            "only whether MP, MS, or ME is exposed"
        ),
        "states": decomposition_states,
        "planned_new_logical_calls": 9,
        "reused_zero_api_outcomes": {
            "M0+R0": 3,
            "MPMSME+R0": 3,
        },
        "frozen_factors": [
            "state",
            "visible dialogue",
            "generator endpoint/model",
            "system prompt",
            "temperature",
            "seed",
            "R0 strategy mode",
            "output cap",
        ],
        "only_manipulated_factor": "one exposed oracle memory source",
        "analysis": {
            "per_state_contrasts": [
                "MP-only minus M0",
                "MS-only minus M0",
                "ME-only minus M0",
                "all-three minus best-single-source",
            ],
            "primary_measurement": (
                "target-evidence uptake using the already frozen BGE analysis"
            ),
            "response_quality_requires_separate_blinded_measurement": True,
            "training_labels_created": False,
            "frozen_interference_classification": {
                "per_state_clear_interference": (
                    "at least one single-source arm has positive source-aligned "
                    "BGE delta versus M0 and at least two single-source arms "
                    "outperform all-three on the same source reference"
                ),
                "supported_report_only": (
                    "clear interference in at least 2 of 3 states"
                ),
                "no_single_source_uptake": (
                    "zero states contain a positive source-aligned single-source "
                    "delta"
                ),
                "otherwise": "source-specific-or-inconclusive-report-only",
            },
        },
        "stop_rules": [
            "do not read calibration or internal-test",
            "do not change prompt, generator, state, seed, or source items",
            "do not tune a source combination after observing the nine outcomes",
            "do not create training labels or authorize PM training",
            "do not treat evaluator-selected oracle evidence as deployable retrieval",
        ],
        "source_lineage": dict(source_lineage),
        "api_calls_made_while_preparing_contract": 0,
        "training_labels_created": False,
    }
    return {
        **record,
        "contract_sha256": sha256_text(canonical_json(record)),
    }


def validate_existing_multisource_outcomes(
    *,
    contract: Mapping[str, Any],
    outcome_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact M0/all-source rows reused by the decomposition."""

    wanted: dict[tuple[str, str], Mapping[str, Any]] = {}
    for state in contract.get("states") or []:
        state_id = str(state["state_id"])
        wanted[(state_id, CONTROL_ARM)] = state
        wanted[(state_id, HELPFUL_ARM)] = state
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in outcome_rows:
        arm = str(
            dict(row.get("provenance") or {}).get(
                "oracle_memory_pilot_arm"
            )
            or ""
        )
        key = (str(row["state_id"]), arm)
        if key not in wanted:
            continue
        if key in found:
            raise RuntimeError(f"duplicate multisource reused outcome: {key}")
        expected_action = (
            "M0+R0" if arm == CONTROL_ARM else "MPMSME+R0"
        )
        if str(row["action_id"]) != expected_action:
            raise RuntimeError(f"multisource reused action drifted: {key}")
        if arm == CONTROL_ARM and (
            row.get("memory_view") or row.get("strategy_view")
        ):
            raise RuntimeError(f"multisource M0 exposed evidence: {key}")
        if arm == HELPFUL_ARM and row.get("strategy_view"):
            raise RuntimeError(f"multisource all-source exposed strategy: {key}")
        found[key] = dict(row)
    if set(found) != set(wanted):
        raise RuntimeError(
            "multisource reused outcome coverage mismatch: "
            f"missing={sorted(set(wanted)-set(found))}"
        )
    return [
        found[key]
        for key in sorted(found, key=lambda value: (value[0], value[1]))
    ]
