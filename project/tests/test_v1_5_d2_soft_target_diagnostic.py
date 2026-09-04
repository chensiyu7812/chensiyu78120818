from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_d2_soft_targets_keep_pair_and_reviewer_grain_distinct() -> None:
    directory = ROOT / "outputs/pm_v1_5_d2_measurement_review_analysis_v1"
    pairs = _jsonl(directory / "pair_soft_measurements.jsonl")
    states = _jsonl(directory / "state_soft_targets.jsonl")
    assert len(pairs) == 96
    assert len(states) == 32
    assert Counter(row["component"] for row in states) == Counter(
        {"RS": 8, "MP": 8, "MS": 8, "ME": 8}
    )
    assert all(row["independent_generated_pairs"] == 3 for row in states)
    assert all(row["not_a_hard_effect_label"] for row in states)
    assert all(0.0 <= row["soft_expected_benefit"] <= 1.0 for row in states)


def test_d2_soft_directional_diagnostic_is_not_model_promotion() -> None:
    directory = ROOT / "outputs/pm_v1_5_d2_soft_target_diagnostic_v1"
    report = _json(directory / "diagnostic.json")
    rows = _jsonl(directory / "state_features_and_soft_targets.jsonl")
    assert report["status"] == (
        "PRE_D3_DIAGNOSTIC_COMPLETE_NOT_MODEL_PROMOTION"
    )
    assert report["formal_D2_gate_remains_failed"] is True
    assert report["api_calls"] == 0
    assert report["new_human_decisions"] == 0
    assert len(rows) == 32
    assert all(report["checks"].values())
    assert all(
        component_report["feature_count"]
        <= component_report["capacity_floor_groups_div_5"]
        for component_report in report["component_reports"].values()
    )


def test_d3_minimum_contract_is_bounded_and_capacity_safe() -> None:
    contract = _json(
        ROOT
        / "data/pm_v1_5_contracts/"
        "d3_minimum_stochastic_effect_training_v1.json"
    )
    assert contract["status"] == (
        "FROZEN_BEFORE_ANY_D3_STATE_OR_OUTCOME_GENERATION"
    )
    design = contract["data_design"]
    assert design["new_content_disjoint_users"] == 32
    assert design["state_contrasts"] == 128
    assert design["total_pairs"] == 160
    assert design["response_calls"] == 320
    assert design["primary_human_quality_decisions"] == 160
    assert design["D2_rows_used_for_training"] is False
    features = contract["feature_contract"]
    assert len(features["RS"]) == 5
    assert len(features["MP"]) == 6
    assert len(features["MS"]) == 5
    assert len(features["ME"]) == 5
