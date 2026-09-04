from __future__ import annotations

from types import SimpleNamespace

import pytest

from metacom_pm.contracts import MemoryBackendRecord, MemoryItem
from metacom_pm.v1_5_oracle_memory_pilot import (
    HARMFUL_ARM,
    HELPFUL_ARM,
    ORACLE_MEMORY_PILOT_REGIME_COUNTS,
    build_oracle_memory_pilot_contract,
    materialize_oracle_backend,
    select_oracle_memory_pilot_states,
)
from metacom_pm.v1_5_oracle_memory_uptake import _status


def _memory_id(index: int) -> str:
    return f"mem_{index:024x}"


def _fixture():
    source_by_regime = {
        "profile_needed": ["MP"],
        "summary_needed": ["MS"],
        "event_needed": ["ME"],
        "multi_source_needed": ["MP", "MS", "ME"],
        "memory_harmful": ["MP", "MS", "ME"],
    }
    states = []
    evaluator = {}
    backends = {}
    index = 1
    for regime, count in ORACLE_MEMORY_PILOT_REGIME_COUNTS.items():
        for offset in range(count + 1):
            state_id = f"state-{regime}-{offset}"
            card_id = f"card-{regime}-{offset}"
            user_id = f"user-{regime}-{offset}"
            utility = "harmful" if regime == "memory_harmful" else "helpful"
            annotations = []
            items = []
            for source in source_by_regime[regime]:
                memory_id = _memory_id(index)
                index += 1
                annotations.append(
                    {
                        "memory_id": memory_id,
                        "source": source,
                        "item_utility": utility,
                    }
                )
                items.append(
                    MemoryItem(
                        memory_id=memory_id,
                        source=source,
                        created_session=1,
                        text=f"{regime} {source} target",
                    )
                )
            states.append(
                SimpleNamespace(
                    state_id=state_id,
                    card_id=card_id,
                    user_id=user_id,
                    semantic_family=f"family-{regime}",
                    split=SimpleNamespace(value="train"),
                    allowed_actions=[
                        "MP+R0",
                        "MS+R0",
                        "ME+R0",
                        "MPMSME+R0",
                    ],
                )
            )
            evaluator[state_id] = {
                "regime": regime,
                "needed_memory_sources": (
                    source_by_regime[regime]
                    if regime != "memory_harmful"
                    else []
                ),
                "memory_annotations": annotations,
            }
            backends[card_id] = MemoryBackendRecord(card_id=card_id, items=items)
    return states, evaluator, backends


def test_oracle_memory_selection_is_fresh_train_only_and_exact() -> None:
    states, evaluator, backends = _fixture()
    excluded = {"state-profile_needed-0"}
    selected = select_oracle_memory_pilot_states(
        states,
        evaluator,
        backends,
        excluded_state_ids=excluded,
        prefer_unused_user_ids={state.user_id for state in states},
    )
    assert len(selected) == 18
    assert excluded.isdisjoint(row["state_id"] for row in selected)
    assert len({row["user_id"] for row in selected}) == 18
    assert sum(row["treatment_arm"] == HELPFUL_ARM for row in selected) == 12
    assert sum(row["treatment_arm"] == HARMFUL_ARM for row in selected) == 6
    assert all(row["target_memory_ids"] for row in selected)

    backend_rows = materialize_oracle_backend(
        selected_states=selected,
        backend_by_card=backends,
    )
    selected_by_card = {row["card_id"]: row for row in selected}
    for row in backend_rows:
        assert {item["memory_id"] for item in row["items"]} == set(
            selected_by_card[row["card_id"]]["target_memory_ids"]
        )


def test_oracle_memory_contract_freezes_report_only_stopping_rules() -> None:
    states, evaluator, backends = _fixture()
    selected = select_oracle_memory_pilot_states(
        states,
        evaluator,
        backends,
        excluded_state_ids=set(),
    )
    contract = build_oracle_memory_pilot_contract(
        selected_states=selected,
        source_lineage={"fixture": "unit"},
    )
    assert contract["planned_new_logical_calls"] == 18
    assert contract["evaluation"]["api_judges"] == "forbidden"
    assert contract["training_labels_created"] is False
    assert "item-level helpful strategy card" in contract["strategy_scope"]
    assert contract["contract_sha256"]

    duplicated = [dict(row) for row in selected]
    duplicated[1]["user_id"] = duplicated[0]["user_id"]
    with pytest.raises(RuntimeError, match="users are not disjoint"):
        build_oracle_memory_pilot_contract(
            selected_states=duplicated,
            source_lineage={"fixture": "unit"},
        )


def test_oracle_memory_runner_binds_analysis_and_bypasses_only_score_gate() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/v1_5/21j_run_longitudinal_oracle_memory_pilot_v1_5.py"
    ).read_text(encoding="utf-8")
    assert '"analysis_code_sha256"' in source
    assert '"analysis_runner_sha256"' in source
    assert "memory_min_score=None" in source
    assert "evidence_filter_config=None" in source
    assert '"api_judges_used": False' in source
    assert '"training_labels_created": False' in source


def test_oracle_memory_uptake_status_uses_frozen_counts_and_directions() -> None:
    supported = _status(
        {
            "pairs": 12,
            "mean_target_evidence_bge_cosine_delta": 0.01,
            "positive_pairs": 8,
        },
        {
            "pairs": 6,
            "mean_target_evidence_bge_cosine_delta": -0.01,
            "nonpositive_pairs": 4,
        },
    )
    assert supported == (
        "ORACLE_MEMORY_UPTAKE_CAPACITY_SUPPORTED_REPORT_ONLY",
        True,
        True,
    )
    wrong_count = _status(
        {
            "pairs": 11,
            "mean_target_evidence_bge_cosine_delta": 0.01,
            "positive_pairs": 8,
        },
        {
            "pairs": 6,
            "mean_target_evidence_bge_cosine_delta": -0.01,
            "nonpositive_pairs": 4,
        },
    )
    assert wrong_count[0] == (
        "ORACLE_MEMORY_UPTAKE_CAPACITY_NOT_ESTABLISHED_REPORT_ONLY"
    )
