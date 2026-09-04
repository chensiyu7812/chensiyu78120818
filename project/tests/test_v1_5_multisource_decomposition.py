from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from metacom_pm.io import canonical_json, sha256_text
from metacom_pm.v1_5_multisource_decomposition import (
    SINGLE_SOURCE_ACTIONS,
    build_multisource_decomposition_contract,
    validate_existing_multisource_outcomes,
)
from metacom_pm.v1_5_multisource_uptake import (
    classify_multisource_interference,
)
from metacom_pm.v1_5_oracle_memory_pilot import (
    CONTROL_ARM,
    HELPFUL_ARM,
    ORACLE_MEMORY_PILOT_PROTOCOL,
)


def _fixture():
    selected = []
    backends = []
    outcomes = []
    for index in range(3):
        state_id = f"state-{index}"
        card_id = f"card-{index}"
        ids = {source: f"memory-{source}-{index}" for source in ("MP", "MS", "ME")}
        selected.append(
            {
                "state_id": state_id,
                "card_id": card_id,
                "user_id": f"user-{index}",
                "semantic_family": f"family-{index}",
                "regime": "multi_source_needed",
                "treatment_arm": HELPFUL_ARM,
                "target_action_id": "MPMSME+R0",
                "target_item_utility": "helpful",
                "target_memory_ids": sorted(ids.values()),
                "target_memory_sources": ["ME", "MP", "MS"],
            }
        )
        backends.append(
            {
                "card_id": card_id,
                "items": [
                    {
                        "memory_id": ids[source],
                        "source": source,
                        "text": f"{source} target",
                    }
                    for source in ("MP", "MS", "ME")
                ],
            }
        )
        outcomes.extend(
            [
                {
                    "state_id": state_id,
                    "action_id": "M0+R0",
                    "memory_view": [],
                    "strategy_view": [],
                    "provenance": {"oracle_memory_pilot_arm": CONTROL_ARM},
                },
                {
                    "state_id": state_id,
                    "action_id": "MPMSME+R0",
                    "memory_view": [{"memory_id": value} for value in ids.values()],
                    "strategy_view": [],
                    "provenance": {"oracle_memory_pilot_arm": HELPFUL_ARM},
                },
            ]
        )
    oracle = {
        "protocol": ORACLE_MEMORY_PILOT_PROTOCOL,
        "selected_states": selected,
    }
    oracle["contract_sha256"] = sha256_text(canonical_json(oracle))
    return oracle, backends, outcomes


def test_multisource_contract_freezes_nine_single_source_calls() -> None:
    oracle, backends, outcomes = _fixture()
    contract = build_multisource_decomposition_contract(
        oracle_contract=oracle,
        backend_rows=backends,
        source_lineage={"fixture": "unit"},
    )
    assert contract["planned_new_logical_calls"] == 9
    assert contract["reused_zero_api_outcomes"] == {
        "M0+R0": 3,
        "MPMSME+R0": 3,
    }
    assert contract["training_labels_created"] is False
    for state in contract["states"]:
        assert {
            row["action_id"] for row in state["single_source_arms"]
        } == set(SINGLE_SOURCE_ACTIONS.values())
    assert len(validate_existing_multisource_outcomes(
        contract=contract,
        outcome_rows=outcomes,
    )) == 6


def test_multisource_contract_rejects_missing_source() -> None:
    oracle, backends, _ = _fixture()
    bad = deepcopy(backends)
    bad[0]["items"].pop()
    with pytest.raises(RuntimeError, match="source coverage drifted"):
        build_multisource_decomposition_contract(
            oracle_contract=oracle,
            backend_rows=bad,
            source_lineage={"fixture": "unit"},
        )


def test_multisource_reuse_rejects_wrong_action_or_missing_row() -> None:
    oracle, backends, outcomes = _fixture()
    contract = build_multisource_decomposition_contract(
        oracle_contract=oracle,
        backend_rows=backends,
        source_lineage={"fixture": "unit"},
    )
    bad = deepcopy(outcomes)
    bad[0]["action_id"] = "MP+R0"
    with pytest.raises(RuntimeError, match="action drifted"):
        validate_existing_multisource_outcomes(
            contract=contract,
            outcome_rows=bad,
        )
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        validate_existing_multisource_outcomes(
            contract=contract,
            outcome_rows=outcomes[:-1],
        )


def test_multisource_runner_is_report_only_and_budgeted_by_physical_attempts() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "scripts/v1_5/"
        "21n_run_longitudinal_multisource_decomposition_v1_5.py"
    ).read_text(encoding="utf-8")
    assert "TRANSPORT_MAX_ATTEMPTS = 4" in source
    assert "maximum_physical_api_attempts" in source
    assert "require_paid_run_release(" in source
    assert "require_output_directory_not_previously_consumed(" in source
    assert '"api_judges_used": False' in source
    assert '"training_labels_created": False' in source
    assert "memory_min_score=None" in source
    assert "evidence_filter_config=None" in source
    assert 'action_filter={action_id}' in source
    assert '"analysis_code_sha256"' in source
    assert '"analysis_runner_sha256"' in source


def test_tracked_multisource_contract_matches_current_contract_code() -> None:
    from metacom_pm.io import read_json, sha256_file

    root = Path(__file__).resolve().parents[1]
    contract = read_json(
        root
        / "data/pm_v1_5_contracts/"
        "longitudinal_multisource_decomposition_pilot_v1.json"
    )
    lineage = contract["source_lineage"]
    assert lineage["preparation_code_sha256"] == sha256_file(
        root / "src/metacom_pm/v1_5_multisource_decomposition.py"
    )
    assert lineage["preparation_runner_sha256"] == sha256_file(
        root
        / "scripts/v1_5/"
        "21m_prepare_longitudinal_multisource_decomposition_v1_5.py"
    )
    without_sha = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    from metacom_pm.io import canonical_json, sha256_text

    assert contract["contract_sha256"] == sha256_text(
        canonical_json(without_sha)
    )


def test_multisource_classification_uses_frozen_report_only_rule() -> None:
    supported = classify_multisource_interference(
        [
            {
                "clear_multisource_interference": True,
                "positive_single_source_count": 1,
            },
            {
                "clear_multisource_interference": True,
                "positive_single_source_count": 2,
            },
            {
                "clear_multisource_interference": False,
                "positive_single_source_count": 0,
            },
        ]
    )
    assert supported == "MULTISOURCE_INTERFERENCE_SUPPORTED_REPORT_ONLY"
    none = classify_multisource_interference(
        [
            {
                "clear_multisource_interference": False,
                "positive_single_source_count": 0,
            }
            for _ in range(3)
        ]
    )
    assert none == "NO_SINGLE_SOURCE_UPTAKE_REPORT_ONLY"
    inconclusive = classify_multisource_interference(
        [
            {
                "clear_multisource_interference": True,
                "positive_single_source_count": 1,
            },
            {
                "clear_multisource_interference": False,
                "positive_single_source_count": 1,
            },
            {
                "clear_multisource_interference": False,
                "positive_single_source_count": 0,
            },
        ]
    )
    assert inconclusive == "SOURCE_SPECIFIC_OR_INCONCLUSIVE_REPORT_ONLY"
