from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from metacom_pm.contracts import ALL_ACTION_IDS, RuntimeState, parse_action_id


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pm_v1_5_d3_step0_blueprint_v1"
GEN = ROOT / "outputs/pm_v1_5_d3_generation_v1"
EXEC = ROOT / "outputs/pm_v1_5_d3_generation_v1_execution"
BLIND = ROOT / "outputs/pm_v1_5_d3_blind_review_v1"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_d3_step0_preflight_passes_without_outcomes_or_api() -> None:
    report = _json(OUT / "preflight_report.json")
    assert report["status"] == "PASS_READY_TO_PREPARE_320_FROZEN_RESPONSE_CALLS"
    assert report["api_calls_made"] == 0
    assert report["human_labels_read"] is False
    assert report["external_outcomes_read"] is False
    assert report["users"] == 32
    assert report["state_contrasts"] == 128
    assert report["total_pairs"] == 160
    assert report["planned_response_calls"] == 320
    assert all(report["checks"].values())
    assert report["content_disjoint_audit"]["exact_normalized_overlap_counts"] == {
        "ESConv": 0,
        "EvoEmo": 0,
    }


def test_d3_step0_keeps_all_actions_and_one_bit_contrasts() -> None:
    states = [RuntimeState.model_validate(row) for row in _jsonl(OUT / "runtime_states.jsonl")]
    rows = _jsonl(OUT / "d3_contrast_blueprint.jsonl")
    assert len(states) == 32
    assert all(set(state.allowed_actions) == set(ALL_ACTION_IDS) for state in states)
    assert Counter(row["component"] for row in rows) == Counter(
        {"RS": 32, "MP": 32, "MS": 32, "ME": 32}
    )
    for row in rows:
        control_sources, control_strategy = parse_action_id(row["control_action"])
        treatment_sources, treatment_strategy = parse_action_id(row["treatment_action"])
        changed = len(set(control_sources) ^ set(treatment_sources)) + int(
            control_strategy != treatment_strategy
        )
        assert changed == 1
        assert row["effect_label"] == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
        assert row["outcome_read"] is False


def test_d3_step0_features_are_bounded_crossed_and_balanced() -> None:
    report = _json(OUT / "preflight_report.json")
    assert report["feature_counts"] == {"RS": 5, "MP": 6, "MS": 5, "ME": 5}
    for counts in report["candidate_strata_counts"].values():
        assert len(counts) == 4
        assert min(counts.values()) >= 4
    for counts in report["background_counts"].values():
        assert len(counts) == 8
        assert set(counts.values()) == {4}


def test_d3_hard_gates_and_repeat_selection_are_frozen_separately() -> None:
    controls = _jsonl(OUT / "hard_gate_controls.jsonl")
    repeats = _jsonl(OUT / "d3_repeat_pairs.jsonl")
    assert Counter(row["control_type"] for row in controls) == Counter(
        {
            "wrong_user": 8,
            "future_memory": 8,
            "stale_request": 8,
            "explicit_conflict": 8,
        }
    )
    assert all(row["observation"]["deterministic_hard_off"] for row in controls)
    assert all(row["generation_allowed"] is False for row in controls)
    assert Counter(row["component"] for row in repeats) == Counter(
        {"RS": 8, "MP": 8, "MS": 8, "ME": 8}
    )
    assert all(row["pair_role"] == "outcome_blind_repeat" for row in repeats)


def test_d3_generation_plan_is_frozen_at_160_pairs_320_calls() -> None:
    report = _json(GEN / "generation_plan_report.json")
    calls = _jsonl(GEN / "call_plan.jsonl")
    assert report["status"] == "FROZEN_READY_FOR_320_D3_RESPONSE_CALLS"
    assert report["api_calls_made"] == 0
    assert report["pairs"] == 160
    assert report["planned_calls"] == 320
    assert report["component_pair_counts"] == {
        "RS": 40,
        "MP": 40,
        "MS": 40,
        "ME": 40,
    }
    assert all(report["checks"].values())
    assert len(calls) == 320
    assert all(row["effect_label"] == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW" for row in calls)


def test_d3_generation_completed_and_blind_packet_hides_private_identity() -> None:
    summary = _json(EXEC / "execution_summary.json")
    outcomes = _jsonl(EXEC / "generation_outcomes.jsonl")
    manifest = _json(BLIND / "manifest.json")
    public = _jsonl(BLIND / "human_blind_packet.jsonl")
    private = _jsonl(BLIND / "private_blind_key.jsonl")
    assert summary["status"] == "COMPLETE"
    assert len(outcomes) == 320
    assert len({row["call_id"] for row in outcomes}) == 320
    assert all(row["normalized_finish_reason"] == "complete" for row in outcomes)
    assert manifest["status"] == "READY_FOR_SINGLE_BOUNDED_160_PAIR_D3_QUALITY_REVIEW"
    assert all(manifest["checks"].values())
    assert len(public) == len(private) == 160
    forbidden = {
        "pair_id",
        "pair_role",
        "contrast_slot_id",
        "component",
        "state_id",
        "user_id",
        "a_role",
        "b_role",
    }
    assert all(not (set(row) & forbidden) for row in public)
