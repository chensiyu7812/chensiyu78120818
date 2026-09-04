from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from metacom_pm.contracts import MemorySource, StrategyMode, parse_action_id


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts/v1_5" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_contract_keeps_marginal_benefit_as_gold() -> None:
    contract = json.loads(
        (
            ROOT
            / "data/pm_v1_5_contracts/"
            "canonical_component_marginal_effect_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["status"].startswith("ACTIVE_CANONICAL")
    assert contract["components"] == ["MP", "MS", "ME", "RS"]
    assert contract["legal_action_count"] == 16
    assert contract["applicability_role"]["pm_gold"] is False
    assert contract["contrast_blueprint"]["total_contrast_groups"] == 256
    assert contract["estimand"]["tie_policy"].startswith("off")


def test_invalid_h2v2_packet_is_not_review_authorized() -> None:
    manifest = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_h2v2_repair_v2_qualification_v1_candidate/"
            "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "INVALIDATED_NOT_FOR_HUMAN_REVIEW"
    assert manifest["human_review_authorized"] is False
    assert any("query" in reason for reason in manifest["invalidation_reasons"])


def test_reuse_audit_qualifies_only_R0_memory_contrasts() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_component_contrast_reuse_audit_v1/"
            "audit_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "R0_MEMORY_CONTRASTS_STRUCTURALLY_REUSABLE"
    assert report["reusable_contrasts"] == 468 * 3 * 4
    assert report["training_labels_created"] is False
    assert report["human_quality_labels_read"] is False
    assert report["stack_checks"]["R0_contrasts_inject_no_strategy"] is True
    assert (
        report["stack_checks"][
            "historical_strategy_bank_differs_from_current_bank"
        ]
        is True
    )
    assert "historical RS arms" in report["reuse_boundary"]["not_approved"]


def test_blueprint_has_64_clean_one_bit_contrasts_per_head() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_four_component_contrast_blueprint_v1/"
            "blueprint_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "READY_256_CONTRAST_BLUEPRINT_ZERO_NEW_API"
    assert report["component_counts"] == {
        "ME": 64,
        "MP": 64,
        "MS": 64,
        "RS": 64,
    }
    assert report["split_counts"] == {
        "calibration": 64,
        "internal_test": 64,
        "train": 128,
    }
    assert report["reuse_counts"] == {
        "FULL_PAIR_REUSE_R0_MEMORY_CONTRAST": 96,
        "GENERATE_BOTH_CURRENT_RS_BACKGROUND_ARMS": 96,
        "REUSE_R0_CONTROL_GENERATE_CURRENT_RS_TREATMENT": 64,
    }
    assert report["new_generation_response_calls"] == 256
    assert all(report["checks"].values())

    rows = [
        json.loads(line)
        for line in (
            ROOT
            / "outputs/pm_v1_5_four_component_contrast_blueprint_v1/"
            "contrast_blueprint.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 256
    assert len({row["state_id"] for row in rows}) == 256
    for row in rows:
        control_sources, control_strategy = parse_action_id(
            row["control_action"]
        )
        treatment_sources, treatment_strategy = parse_action_id(
            row["treatment_action"]
        )
        if row["component"] == "RS":
            assert control_sources == treatment_sources
            assert control_strategy is StrategyMode.R0
            assert treatment_strategy is StrategyMode.RS
        else:
            assert control_strategy is treatment_strategy
            assert treatment_sources - control_sources == {
                MemorySource(row["component"])
            }
            assert not (control_sources - treatment_sources)
        assert row["effect_label"] == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"


def test_background_builder_has_eight_backgrounds_per_component() -> None:
    module = _load_script(
        "24bj_build_component_contrast_blueprint_v1_5.py"
    )
    for component in ("MP", "MS", "ME", "RS"):
        assert len(module._backgrounds(component)) == 8


def test_generation_plan_preserves_clean_contrasts_and_unknown_labels() -> None:
    plan_dir = (
        ROOT
        / "outputs/pm_v1_5_four_component_contrast_generation_v1"
    )
    report = json.loads(
        (plan_dir / "plan_report.json").read_text(encoding="utf-8")
    )
    calls = [
        json.loads(line)
        for line in (plan_dir / "call_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    registry = [
        json.loads(line)
        for line in (plan_dir / "response_arm_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert report["status"] == "FROZEN_READY_FOR_256_NEW_RESPONSE_CALLS"
    assert report["contrast_groups"] == 256
    assert report["response_arms_total"] == 512
    assert report["historical_response_arms_reused"] == 256
    assert report["new_response_calls"] == 256
    assert len(calls) == 256
    assert len(registry) == 512
    assert len(
        {(row["contrast_slot_id"], row["arm"]) for row in calls}
    ) == 256
    assert all(report["checks"].values())
    assert all(
        row["effect_label"] == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
        for row in calls + registry
    )
    assert all(
        row["selected_strategy_card_id"] is None
        for row in registry
        if row["action_id"].endswith("+R0")
    )


def test_legacy_mixed_quality_packet_is_blind_but_transport_paused() -> None:
    packet_dir = (
        ROOT
        / "outputs/pm_v1_5_four_component_effect_human_quality_blind_v1"
    )
    manifest = json.loads(
        (packet_dir / "manifest.json").read_text(encoding="utf-8")
    )
    packet = [
        json.loads(line)
        for line in (packet_dir / "human_blind_packet.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    key = [
        json.loads(line)
        for line in (packet_dir / "private_blind_key.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert manifest["status"] == (
        "PAUSED_NOT_AUTHORIZED_MEMORY_TRANSPORT_REBUILD"
    )
    assert manifest["human_review_authorized"] is False
    assert manifest["unique_contrasts"] == 256
    assert manifest["reliability_repeats"] == 52
    assert manifest["review_presentations"] == 308
    assert len(packet) == 308
    assert len(key) == 308
    forbidden = {
        "component",
        "split",
        "control_action",
        "treatment_action",
        "a_role",
        "b_role",
        "selected_memory_ids",
        "selected_strategy_card_id",
    }
    assert all(not (forbidden & set(row)) for row in packet)
    assert sum(row["is_reliability_repeat"] for row in key) == 52
    assert all(
        row["effect_label"] == "UNKNOWN_BEFORE_HUMAN_ANNOTATION"
        for row in key
    )


def test_memory_transport_contract_separates_content_from_pipeline() -> None:
    contract = json.loads(
        (
            ROOT
            / "data/pm_v1_5_contracts/"
            "memory_transport_candidate_contract_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["status"] == (
        "MIMIC_GATE_PASS_ALL_REPAIRED_FOUR_COMPONENT_GENERATION_COMPLETE"
    )
    assert (
        contract["formal_runtime"]["decision_timing"]
        == "post-candidate-discovery, pre-injection, pre-generation"
    )
    assert contract["effect_data_status"][
        "existing_256_contrasts_remain_usable"
    ] == "DIAGNOSTIC_ONLY_NOT_FORMAL_ALL_FOUR_COMPONENTS_REBUILT"
    assert contract["effect_data_status"][
        "no_regeneration_required_for_candidate_aware_training"
    ] is False
    assert contract["selected_internal_repair"]["zero_api_feasibility"][
        "eligible_states"
    ] == 416
    assert contract["current_transport_audit"][
        "same_memory_construction_algorithm"
    ] is False
    final_runtime = contract["final_shared_runtime_repair"]
    assert final_runtime["history_policy"].startswith(
        "Every internal and external state compiles all legitimate causal"
    )
    assert final_runtime["all_sources_pass"] is True
    assert final_runtime["outcome_blind_joint_candidate_support"] == {
        "MP": 1.0,
        "MS": 0.8676470588235294,
        "ME": 0.8921568627450981,
    }
    assert final_runtime["existing_generation_prompt_recomputation"][
        "total"
    ] == "512/512"
    assert contract["ood_and_transport_gate"][
        "external_outcomes_may_not_set_support_thresholds"
    ] is True


def test_memory_transport_audit_detects_real_catalog_shift() -> None:
    audit = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_memory_transport_contract_audit_v1/"
            "memory_transport_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["status"] == (
        "TRANSPORT_MISMATCH_REQUIRES_CLAIM_AND_INPUT_REPAIR"
    )
    assert audit["internal_catalog_records"] == 468
    assert audit["external_users"] == 18
    internal = audit["profiles"]["internal_synthetic"]
    external = audit["profiles"]["evoemo_external"]
    assert internal["ME"]["catalog_count"]["median"] == 2
    assert external["ME"]["catalog_count"]["median"] == 68
    assert internal["ME"]["item_tokens"]["median"] == 37
    assert external["ME"]["item_tokens"]["median"] == 102
    assert audit["contract_checks"][
        "same_memory_construction_algorithm"
    ] is False
    assert audit["contract_checks"]["external_outcomes_read"] is False


def test_evo_style_synthetic_recompile_is_content_disjoint_and_feasible() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_evo_style_synthetic_memory_recompile_audit_v1/"
            "recompile_feasibility_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == (
        "FEASIBLE_WITHOUT_EXTERNAL_CONTENT_OR_OUTCOMES"
    )
    assert report["synthetic_users"] == 52
    assert report["eligible_longitudinal_states"] == 416
    assert report["profiles"]["MP"]["catalog_count"]["median"] == 3
    assert report["profiles"]["MS"]["catalog_count"]["median"] == 4.5
    assert report["profiles"]["ME"]["catalog_count"]["median"] == 5
    assert report["checks"]["external_file_read"] is False
    assert report["checks"]["external_content_used"] is False
    assert report["checks"]["external_outcome_used"] is False
    assert report["checks"]["same_builder_function_as_evoemo"] is True
    assert report["checks"]["strictly_prior_sessions_only"] is True
    assert report["checks"]["at_least_256_eligible_states"] is True
    assert report["checks"][
        "all_sources_have_top_k_competition_somewhere"
    ] is True


def test_transport_repaired_backend_is_causal_and_content_disjoint() -> None:
    report = json.loads(
        (
            ROOT
            / "data/pm_v1_5_evo_style_synthetic_memory_v1_candidate/"
            "report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == (
        "READY_FOR_OUTCOME_BLIND_192_MEMORY_CONTRAST_BLUEPRINT"
    )
    assert report["states"] == 416
    assert report["split_counts"] == {
        "calibration": 96,
        "internal_test": 128,
        "train": 192,
    }
    assert report["external_file_read"] is False
    assert report["external_content_used"] is False
    assert report["external_outcome_used"] is False
    assert report["generated_memory_labels_used"] is False
    assert report["retrieval"]["score"] == "lexical_score_not_BGE_cosine"
    assert report["catalog_profiles"]["MP"]["count"]["median"] == 3
    assert report["catalog_profiles"]["MS"]["count"]["median"] == 4.5
    assert report["catalog_profiles"]["ME"]["count"]["median"] == 5
    assert all(report["checks"].values())


def test_transport_repaired_blueprint_replaces_only_memory_pairs() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1/"
            "blueprint_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == (
        "READY_192_MEMORY_PAIRS_ZERO_API_RETAIN_64_RS"
    )
    assert report["component_counts"] == {"ME": 64, "MP": 64, "MS": 64}
    assert report["split_counts"] == {
        "calibration": 48,
        "internal_test": 48,
        "train": 96,
    }
    assert report["retained_formal_rs_contrasts"] == 64
    assert report["replacement_total_contrasts"] == 256
    assert report["new_response_calls_required"] == 384
    assert all(report["checks"].values())


def test_transport_repaired_generation_plan_is_frozen_and_outcome_blind() -> None:
    plan_dir = (
        ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_generation_v1"
    )
    report = json.loads(
        (plan_dir / "plan_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (plan_dir / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    calls = [
        json.loads(line)
        for line in (plan_dir / "call_plan.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    registry = [
        json.loads(line)
        for line in (plan_dir / "response_arm_registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected_status = "FROZEN_READY_FOR_384_NEW_RESPONSE_CALLS"
    assert report["status"] == expected_status
    assert manifest["status"] == expected_status
    assert report["contrast_groups"] == 192
    assert report["new_response_calls"] == 384
    assert len(calls) == 384
    assert len(registry) == 384
    assert len(
        {(row["contrast_slot_id"], row["arm"]) for row in calls}
    ) == 384
    assert all(report["checks"].values())
    assert all(
        row["effect_label"] == "UNKNOWN_BEFORE_BLIND_HUMAN_REVIEW"
        for row in calls + registry
    )
    assert all(row["response"] is None for row in registry)


def test_evo_mimic_audit_passes_only_after_exact_stack_matches() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_evo_mimic_equivalence_audit_v1/"
            "equivalence_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "MIMIC_AND_CANDIDATE_SUPPORT_GATE_PASS"
    assert report["api_calls_made"] == 0
    assert report["human_or_judge_outcomes_read"] is False
    assert report["external_content_used_for_training"] is False
    assert report["exact_mechanism_passes"] == 13
    assert report["exact_mechanism_total"] == 13
    assert report["failed_exact_layers"] == []
    assert report["semantic_input_gaps"][
        "profile_field_name_overlap"
    ] == []
    assert report["semantic_input_gaps"][
        "synthetic_summary_fallback_to_current_user_text_rate"
    ] == 0.5
    coverage = report["candidate_descriptor_comparison"]
    assert coverage["MP"][
        "all_primary_fields_within_internal_min_max_rate"
    ] == 1.0
    assert coverage["MS"][
        "all_primary_fields_within_internal_min_max_rate"
    ] >= 0.8
    assert coverage["ME"][
        "all_primary_fields_within_internal_min_max_rate"
    ] >= 0.8
    assert report["semantic_input_gaps"]["runtime_history_policy"] == (
        "all_causal_prior_sessions"
    )
    assert report["generation_reuse_checks"][
        "all_prompt_hashes_match"
    ] is True
    assert report["generation_reuse_checks"][
        "total_existing_generated_arms_reusable"
    ] == 512


def test_repaired_rs_blueprint_uses_same_backend_and_fresh_states() -> None:
    rs_dir = (
        ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_contrast_blueprint_v1"
    )
    memory_dir = (
        ROOT
        / "outputs/pm_v1_5_transport_repaired_memory_contrast_blueprint_v1"
    )
    report = json.loads(
        (rs_dir / "blueprint_report.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (rs_dir / "rs_contrast_blueprint.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    memory_rows = [
        json.loads(line)
        for line in (memory_dir / "memory_contrast_blueprint.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert report["status"] == (
        "READY_64_RS_PAIRS_ON_TRANSPORT_REPAIRED_STACK"
    )
    assert len(rows) == 64
    assert report["background_counts"] == {
        "M0+R0": 8,
        "ME+R0": 8,
        "MP+R0": 8,
        "MPE+R0": 8,
        "MPMS+R0": 8,
        "MPMSME+R0": 8,
        "MS+R0": 8,
        "MSE+R0": 8,
    }
    assert report["split_counts"] == {
        "calibration": 16,
        "internal_test": 16,
        "train": 32,
    }
    assert {row["state_id"] for row in rows}.isdisjoint(
        {row["state_id"] for row in memory_rows}
    )


def test_repaired_rs_generation_and_final_blind_packet_are_complete() -> None:
    execution_dir = (
        ROOT
        / "outputs/pm_v1_5_transport_repaired_rs_generation_v1_execution"
    )
    summary = json.loads(
        (execution_dir / "generation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    outcomes = [
        json.loads(line)
        for line in (execution_dir / "generation_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert summary["status"] == "COMPLETE"
    assert summary["completed_calls"] == 128
    assert len(outcomes) == 128
    assert all(row["normalized_finish_reason"] == "complete" for row in outcomes)
    assert all(row["response"].strip() for row in outcomes)

    packet_dir = (
        ROOT
        / "outputs/pm_v1_5_transport_repaired_four_component_blind_v1"
    )
    manifest = json.loads(
        (packet_dir / "manifest.json").read_text(encoding="utf-8")
    )
    private = [
        json.loads(line)
        for line in (packet_dir / "private_blind_key.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    primary = [
        row for row in private if not row["is_reliability_repeat"]
    ]
    assert manifest["status"] == (
        "READY_FOR_ONE_FINAL_BOUNDED_BLIND_QUALITY_REVIEW"
    )
    assert manifest["unique_contrasts"] == 256
    assert manifest["component_counts"] == {
        "ME": 64,
        "MP": 64,
        "MS": 64,
        "RS": 64,
    }
    assert manifest["review_presentations"] == 308
    assert manifest["all_components_on_transport_repaired_stack"] is True
    assert manifest["legacy_component_pairs_included"] is False
    assert len(primary) == 256
    assert {
        component: sum(
            row["component"] == component for row in primary
        )
        for component in ("RS", "MP", "MS", "ME")
    } == {"RS": 64, "MP": 64, "MS": 64, "ME": 64}
    assert {
        (component, role): sum(
            row["component"] == component and row["a_role"] == role
            for row in primary
        )
        for component in ("RS", "MP", "MS", "ME")
        for role in ("control", "treatment")
    } == {
        (component, role): 32
        for component in ("RS", "MP", "MS", "ME")
        for role in ("control", "treatment")
    }
