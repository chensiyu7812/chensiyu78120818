from pathlib import Path

from metacom_pm.io import read_json


ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "data/pm_v1_5_contracts/staged_policy_architecture_v2.json"
EXECUTION = ROOT / "data/pm_v1_5_contracts/final_execution_plan_v3.json"


def _stages() -> dict[str, dict]:
    contract = read_json(ARCH)
    return {row["id"]: row for row in contract["stages"]}


def test_only_step1_is_a_learned_pm_routing_stage() -> None:
    stages = _stages()
    assert stages["OBSERVATION"]["may_output_final_on_off"] is False
    assert stages["STEP1_ROUTING"]["is_learned_pm_stage"] is True
    assert stages["STEP2_EXECUTION"]["is_learned_pm_stage"] is False
    assert stages["STEP2_EXECUTION"]["may_reopen_step1_on_off"] is False
    assert stages["STEP1_ROUTING"]["model"] == "four_standardized_l2_logistic_component_effect_heads"


def test_observation_and_router_gold_are_separate() -> None:
    contract = read_json(ARCH)
    separation = contract["gold_separation"]
    assert separation["observation_gold"] == "explicit_factor_and_evidence_code_review"
    assert separation["eligibility_gold"] == "candidate_semantic_and_boundary_review"
    assert separation["step1_gold"] == "repeated_clean_component_effect_worth_opening"
    assert separation["step2_gold"] == "exact_binding_functional_use_and_misuse_review"
    assert separation["one_layer_may_substitute_for_another"] is False
    forbidden = set(_stages()["OBSERVATION"]["may_not_read"])
    assert {"construction_intent", "human_worth_opening_gold", "external_outcome"} <= forbidden


def test_formal_effect_dataset_replaces_old_h1r_splits() -> None:
    execution = read_json(EXECUTION)
    dataset = execution["formal_effect_dataset"]
    assert dataset["old_h1_h1r_role"] == "DEVELOPMENT_EVIDENCE_ONLY"
    assert dataset["per_head"]["EFFECT_FIT"]["states"] == 64
    assert dataset["per_head"]["FRESH_CONFIRMATION"]["states"] == 32
    assert dataset["per_head"]["SEALED_INTERNAL_TEST"]["states"] == 32
    assert dataset["construction_intent_is_gold"] is False
    assert dataset["unexecuted_treatment_is_negative"] is False
    assert dataset["all_effect_candidates_must_pass_objective_hard_eligibility"] is True
    assert dataset["semantic_goal_increment_uncertainty_enters_step1_as_features"] is True
    assert dataset["eligibility_audit_per_head"]["generates_responses"] is False


def test_full_action_space_and_default_off_are_preserved() -> None:
    contract = read_json(ARCH)
    execution = read_json(EXECUTION)
    assert contract["components"] == ["MP", "MS", "ME", "RS"]
    assert contract["legal_action_count"] == 16
    assert execution["default_action"] == "M0+R0"
    assert execution["components"] == ["MP", "MS", "ME", "RS"]


def test_eligibility_is_not_router_gold_and_step2_precedes_effect_data() -> None:
    execution = read_json(EXECUTION)
    eligibility = _stages()["ELIGIBILITY"]
    assert execution["step1_target"]["eligibility_is_step1_gold"] is False
    assert execution["step1_target"]["single_generation_winner_is_gold"] is False
    assert eligibility["role"] == "objective_mask_and_router_input_not_step1_gold"
    assert "goal_function_fit" in eligibility["not_hard_off_when_semantically_uncertain"]
    assert "specific_increment" in eligibility["not_hard_off_when_semantically_uncertain"]
    assert execution["step2_gates_before_effect_generation"]["exact_binding_rate"] == 1.0
    assert execution["step2_gates_before_effect_generation"]["fabricated_recall_count_max"] == 0


def test_external_capability_scope_is_explicit() -> None:
    execution = read_json(EXECUTION)
    matrix = execution["external_capability_matrix"]
    assert "RS" in matrix["ESConv"]["primary_capabilities"]
    assert "ME_longitudinal_memory" in matrix["ESConv"]["not_claimable"]
    assert "MP_PROFILE" in matrix["EvoEmo"]["primary_capabilities"]
    assert "MP_PREFERENCE" in matrix["EvoEmo"]["not_fully_claimable"]
    assert "all_16_actions" in matrix["INTERNAL_SYNTHETIC"]["primary_capabilities"]


def test_basic_learnability_must_beat_transparent_rule() -> None:
    execution = read_json(EXECUTION)
    gates = execution["step1_pass_gates"]
    assert gates["balanced_accuracy_min"] == 0.65
    assert gates["brier_must_beat_transparent_rule"] is True
    assert gates["matching_rule_baseline_counts_as_learned"] is False
    assert gates["all_four_heads_required_for_full_claim"] is True


def test_v3_p2_progress_is_bound_to_the_formal_same_stack_artifacts() -> None:
    execution = read_json(EXECUTION)
    progress = execution["p2_progress"]
    assert progress["visible_states"] == 640
    assert progress["exact_rank1_present_per_component"] == {
        "MP": 160,
        "MS": 160,
        "ME": 160,
        "RS": 160,
    }
    assert progress["formal_rank1_constructed_target_match_rate"] == 1.0
    assert progress["compiler_subtype_match_rate"] == 1.0
    assert progress["complete_step1_decision_surface_duplicates"] == 0
    assert progress["h_eligibility_primary_items"] == 128
    assert progress["h_eligibility_overlap_items"] == 32
    assert progress["responses_generated"] == 28
    assert (
        progress["h_step2_v2_human_status"]
        == "DEVELOPMENT_DIAGNOSTIC_INVALID_FOR_STEP2_PROMOTION"
    )
    assert progress["replacement_h_step2_required"] is True
