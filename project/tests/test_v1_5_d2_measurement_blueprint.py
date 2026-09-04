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


def test_d2_blueprint_has_bounded_balanced_measurement_design() -> None:
    directory = ROOT / "outputs/pm_v1_5_d2_measurement_blueprint_v1"
    report = _json(directory / "blueprint_report.json")
    rows = _jsonl(directory / "d2_measurement_blueprint.jsonl")
    assert report["status"] == "FROZEN_READY_FOR_144_D2_RESPONSE_CALLS"
    assert len(rows) == 32
    assert Counter(row["component"] for row in rows) == Counter(
        {"RS": 8, "MP": 8, "MS": 8, "ME": 8}
    )
    assert report["new_pairs"] == 72
    assert report["new_response_calls"] == 144
    assert report["total_human_decisions"] == 104
    assert all(report["checks"].values())


def test_d2_fresh_ms_is_outcome_disjoint_and_summary_qualified() -> None:
    directory = ROOT / "outputs/pm_v1_5_d2_measurement_blueprint_v1"
    rows = _jsonl(directory / "d2_measurement_blueprint.jsonl")
    fresh = [row for row in rows if row["component"] == "MS"]
    assert len(fresh) == 8
    assert all(row["historical_pair_reference"] is None for row in fresh)
    assert all(row["new_pair_count"] == 3 for row in fresh)
    assert all(
        row["ms_qualification"][
            "state_absent_from_all_first_fit_contrasts"
        ]
        and row["ms_qualification"][
            "all_selected_items_use_supplied_summary"
        ]
        and not row["ms_qualification"]["last_message_fallback_used"]
        for row in fresh
    )
    assert len({row["control_action"] for row in fresh}) == 8


def test_d2_call_plan_is_blind_and_has_two_arms_per_pair() -> None:
    directory = ROOT / "outputs/pm_v1_5_d2_measurement_generation_v1"
    report = _json(directory / "plan_report.json")
    rows = _jsonl(directory / "call_plan.jsonl")
    assert report["status"] == "FROZEN_READY_FOR_144_D2_RESPONSE_CALLS"
    assert len(rows) == 144
    pairs = Counter(row["pair_id"] for row in rows)
    assert len(pairs) == 72
    assert set(pairs.values()) == {2}
    assert all(
        row["effect_label"] == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
        for row in rows
    )
    assert all(report["checks"].values())


def test_d2_execution_and_review_packets_are_complete() -> None:
    execution = (
        ROOT
        / "outputs/pm_v1_5_d2_measurement_generation_v1_execution"
    )
    summary = _json(execution / "execution_summary.json")
    outcomes = _jsonl(execution / "generation_outcomes.jsonl")
    assert summary["status"] == "COMPLETE"
    assert summary["completed_calls"] == 144
    assert len({row["call_id"] for row in outcomes}) == 144
    assert set(Counter(row["pair_id"] for row in outcomes).values()) == {2}

    review = ROOT / "outputs/pm_v1_5_d2_measurement_blind_v1"
    report = _json(review / "report.json")
    assert report["status"] == (
        "READY_FOR_PRIMARY_72_AND_INDEPENDENT_OVERLAP_32"
    )
    primary = _jsonl(review / "primary_72/human_blind_packet.jsonl")
    overlap = _jsonl(
        review / "independent_overlap_32/human_blind_packet.jsonl"
    )
    assert len(primary) == 72
    assert len(overlap) == 32
    assert all("component" not in row for row in primary + overlap)


def test_d2_valid_independent_overlap_fails_frozen_exact_gate() -> None:
    report = _json(
        ROOT
        / "outputs/pm_v1_5_d2_measurement_review_analysis_v1/"
        "analysis.json"
    )
    assert report["status"] == "D2_MEASUREMENT_GATE_FAIL"
    assert report["primary_validation"]["valid"] is True
    assert report["secondary_validation"]["valid"] is True
    assert report["secondary_validation"]["id_overlap_count"] == 32
    assert set(report["primary_validation"]["annotator_counts"]).isdisjoint(
        report["secondary_validation"]["annotator_counts"]
    )
    gate = report["primary_measurement_gate"]
    assert gate["exact_three_way_reproduction_rate"] == 30 / 32
    assert gate["binary_on_off_reproduction_rate_diagnostic"] == 1.0
    assert gate["uncertain_fraction"] == 0.0
    inter_rater = report["inter_rater_gate"]
    assert inter_rater["exact_matches"] == 21
    assert inter_rater["exact_agreement"] == 21 / 32
    assert inter_rater["gate_min"] == 0.75
    assert inter_rater["gate_passed"] is False
    assert inter_rater["binary_on_off_diagnostic"]["matches"] == 26
    assert (
        inter_rater["binary_on_off_diagnostic"]["agreement"] == 26 / 32
    )
    assert inter_rater["exact_disagreement_count"] == 11
    assert inter_rater["binary_disagreement_count"] == 6
    assert report["formal_D2_gate_passed"] is False
