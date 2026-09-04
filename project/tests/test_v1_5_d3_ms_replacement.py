from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from metacom_pm.contracts import MemorySource, RuntimeState, parse_action_id
from metacom_pm.retrieval import seeker_only_context_query


ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT = ROOT / "outputs/pm_v1_5_d3_ms_replacement_step0_v1"
PLAN = ROOT / "outputs/pm_v1_5_d3_ms_replacement_generation_v1"
EXECUTION = ROOT / "outputs/pm_v1_5_d3_ms_replacement_generation_v1_execution"
BLIND = ROOT / "outputs/pm_v1_5_d3_ms_replacement_blind_v1"
GROUNDED = ROOT / "outputs/pm_v1_5_d3_ms_grounded_fidelity_adjudication_v1"
AGGREGATION = ROOT / "outputs/pm_v1_5_d3_ms_grounded_quality_aggregation_v1"
RISK = ROOT / "outputs/pm_v1_5_d3_ms_on_win_risk_review_v1_candidate"
FINAL = ROOT / "outputs/pm_v1_5_d3_ms_final_effect_labels_v1"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_seeker_only_query_excludes_supporter_text() -> None:
    history = [
        {"role": "user", "content": "A deadline is worrying me."},
        {
            "role": "assistant",
            "content": "You already shared the prior session summary.",
        },
        {"speaker": "seeker", "content": "I cannot choose a priority."},
        {"speaker": "supporter", "content": "supporter sentinel"},
    ]
    query = seeker_only_context_query("Where do I begin?", history)
    assert "deadline" in query
    assert "priority" in query
    assert "Where do I begin?" in query
    assert "prior session summary" not in query
    assert "supporter sentinel" not in query


def test_ms_replacement_preflight_qualifies_objective_retrieval() -> None:
    report = _json(BLUEPRINT / "preflight_report.json")
    audit = _jsonl(BLUEPRINT / "retrieval_qualification_audit.jsonl")
    assert report["status"] == (
        "PASS_READY_TO_PREPARE_80_FROZEN_MS_REPLACEMENT_CALLS"
    )
    assert report["api_calls_made"] == 0
    assert report["human_labels_read"] is False
    assert report["external_outcomes_read"] is False
    assert all(report["checks"].values())
    assert len(audit) == 32
    assert all(len(row["selected_ms_ids"]) == 2 for row in audit)
    assert all(row["all_selected_relevant"] for row in audit)
    assert all(row["correct_owner"] and row["strictly_prior"] for row in audit)
    assert all(
        row["supporter_mutation_query_invariant"]
        and row["supporter_mutation_selection_invariant"]
        for row in audit
    )
    assert report["content_disjoint_audit"]["exact_normalized_overlap_counts"] == {
        "ESConv": 0,
        "EvoEmo": 0,
    }


def test_ms_replacement_preserves_actions_backgrounds_and_one_bit_effect() -> None:
    states = [
        RuntimeState.model_validate(row)
        for row in _jsonl(BLUEPRINT / "runtime_states.jsonl")
    ]
    rows = _jsonl(BLUEPRINT / "d3_ms_replacement_contrasts.jsonl")
    assert len(states) == len(rows) == 32
    assert all(len(state.allowed_actions) == 16 for state in states)
    assert set(Counter(row["background_action"] for row in rows).values()) == {4}
    for row in rows:
        control = parse_action_id(row["control_action"])
        treatment = parse_action_id(row["treatment_action"])
        assert set(treatment[0]) - set(control[0]) == {MemorySource.MS}
        assert not (set(control[0]) - set(treatment[0]))
        assert control[1] == treatment[1]
        assert row["effect_label"] == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
        assert row["outcome_read"] is False


def test_ms_replacement_generation_plan_is_frozen_and_pairwise_clean() -> None:
    report = _json(PLAN / "generation_plan_report.json")
    calls = _jsonl(PLAN / "call_plan.jsonl")
    assert report["status"] == "FROZEN_READY_FOR_80_D3_MS_REPLACEMENT_CALLS"
    assert report["pairs"] == 40
    assert report["planned_calls"] == 80
    assert report["api_calls_made"] == 0
    assert all(report["checks"].values())
    assert len(calls) == 80
    assert Counter(row["component"] for row in calls) == Counter({"MS": 80})
    pair_counts = Counter(row["pair_id"] for row in calls)
    assert len(pair_counts) == 40
    assert set(pair_counts.values()) == {2}
    for pair_id in pair_counts:
        pair = [row for row in calls if row["pair_id"] == pair_id]
        assert len({row["generation"]["seed"] for row in pair}) == 1
        assert len({row["prompt_sha256"] for row in pair}) == 2


def test_ms_replacement_generation_and_blind_packet_are_complete() -> None:
    summary = _json(EXECUTION / "execution_summary.json")
    outcomes = _jsonl(EXECUTION / "generation_outcomes.jsonl")
    manifest = _json(BLIND / "manifest.json")
    public = _jsonl(BLIND / "human_blind_packet.jsonl")
    private = _jsonl(BLIND / "private_blind_key.jsonl")
    assert summary["status"] == "COMPLETE"
    assert len(outcomes) == 80
    assert len({row["call_id"] for row in outcomes}) == 80
    assert all(row["normalized_finish_reason"] == "complete" for row in outcomes)
    assert manifest["status"] == (
        "READY_FOR_SINGLE_BOUNDED_40_PAIR_MS_REPLACEMENT_QUALITY_REVIEW"
    )
    assert manifest["exactly_identical_response_pairs"] == 2
    assert all(manifest["checks"].values())
    assert len(public) == len(private) == 40
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


def test_ms_grounded_adjudication_is_exactly_five_and_hides_prior_decisions() -> None:
    manifest = _json(GROUNDED / "manifest.json")
    source = _jsonl(GROUNDED / "source_visible_only_annotations.jsonl")
    public = _jsonl(GROUNDED / "human_grounded_packet.jsonl")
    private = _jsonl(GROUNDED / "private_grounded_key.jsonl")
    assert manifest["status"] == (
        "READY_FOR_FIVE_ITEM_GROUNDING_AWARE_ADJUDICATION"
    )
    assert manifest["selection_rule"] == (
        "all_and_only_source_rows_with_decisive_criterion_"
        "visible_context_fidelity"
    )
    assert all(manifest["checks"].values())
    assert len(source) == 40
    assert Counter(row["decisive_criterion"] for row in source)[
        "visible_context_fidelity"
    ] == 5
    assert len(public) == len(private) == 5
    assert all(row["verified_prior_user_context"] for row in public)
    forbidden = {
        "source_blind_item_id",
        "pair_id",
        "pair_role",
        "contrast_slot_id",
        "state_id",
        "user_id",
        "new_a_role",
        "new_b_role",
        "authorized_memory_ids",
        "selection_rule",
        "source_annotation_preference",
        "component",
    }
    assert all(not (set(row) & forbidden) for row in public)
    assert all(
        row["source_annotation_preference_hidden"] is True
        and row["effect_label"]
        == "UNKNOWN_PENDING_GROUNDED_ADJUDICATION"
        for row in private
    )


def test_ms_grounded_replacement_aggregates_without_double_counting() -> None:
    report = _json(AGGREGATION / "aggregation_report.json")
    pairs = _jsonl(AGGREGATION / "pair_quality_measurements_pre_risk.jsonl")
    states = _jsonl(AGGREGATION / "state_soft_targets_pre_risk.jsonl")
    assert report["status"] == "PASS_ONE_MS_ON_WIN_PENDING_MINIMAL_RISK"
    assert all(report["checks"].values())
    assert report["pair_verdict_counts"] == {
        "control": 1,
        "tie": 38,
        "treatment": 1,
    }
    assert report["replacement_counts"] == {
        "grounded_fidelity_replacement": 5,
        "original_non_fidelity_frozen": 35,
    }
    assert len(pairs) == 40
    assert len({row["pair_id"] for row in pairs}) == 40
    assert len(states) == 32
    assert Counter(
        row["treatment_win_soft_target_pre_risk"] for row in states
    ) == Counter({0.0: 31, 0.5: 1})
    assert all(row["state_training_weight"] == 1.0 for row in states)


def test_ms_minimal_risk_packet_contains_only_the_surviving_on_win() -> None:
    manifest = _json(RISK / "manifest.json")
    public = _jsonl(RISK / "human_risk_packet.jsonl")
    private = _jsonl(RISK / "private_risk_key.jsonl")
    assert manifest["status"] == "READY_FOR_ONE_MS_ON_WIN_MINIMAL_RISK_REVIEW"
    assert manifest["items"] == 1
    assert all(manifest["checks"].values())
    assert len(public) == len(private) == 1
    assert public[0]["authorized_auxiliary_evidence"]
    assert private[0]["component"] == "MS"
    assert private[0]["candidate_arm"] == "treatment"
    assert private[0]["quality_label"] == (
        "ON_QUALITY_WIN_PENDING_MATERIAL_RISK"
    )


def test_ms_final_labels_preserve_conditional_risk_scope_and_do_not_force_train() -> None:
    report = _json(FINAL / "finalization_report.json")
    pairs = _jsonl(FINAL / "final_pair_effect_labels.jsonl")
    states = _jsonl(FINAL / "final_state_soft_targets.jsonl")
    risks = _jsonl(FINAL / "validated_risk_annotations.jsonl")
    assert report["status"] == (
        "FINAL_MS_MEASUREMENT_COMPLETE_NOT_DISCRIMINATIVE_TRAIN_QUALIFIED"
    )
    assert all(report["checks"].values())
    assert len(pairs) == 40
    assert len(states) == 32
    assert len(risks) == 1
    assert Counter(row["treatment_win_soft_target"] for row in states) == Counter(
        {0.0: 31, 0.5: 1}
    )
    assert report["positive_state_count"] == 1
    assert report["confident_positive_state_count"] == 0
    assert report["discriminative_head_training_qualified"] is False
    assert report["risk_result"] == {
        "reviewed_quality_on_wins": 1,
        "material_risk_yes": 0,
        "material_risk_no": 1,
        "material_risk_uncertain": 0,
        "denominator_scope": "quality-winning MS-on responses only",
        "not_an_overall_MS_risk_rate": True,
        "not_a_control_vs_treatment_risk_difference": True,
    }
    assert risks[0]["any_material_risk"] == "no"
    assert risks[0]["candidate_response_lineage_validation"] == "PASS"
