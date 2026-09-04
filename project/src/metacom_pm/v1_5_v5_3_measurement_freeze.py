"""Pre-outcome V5.3 measurement definitions and sample-allocation principles.

The contract freezes estimands, clustering, comparator roles, and review
provenance before paired outcomes exist.  It intentionally does *not* invent
an item-count threshold while the final P2 candidate blueprint is still being
materialized.  Repeating templates is not allowed to manufacture statistical
power; the eventual counts must be the actual unique user/family/
counterfactual groups available in that blueprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .contracts import StrictModel
from .io import canonical_json, sha256_file, stable_hex
from .v1_5_v5_3_response_baselines import POLICIES


MEASUREMENT_PROTOCOL = "pm-v1.5-v5.3-measurement-freeze-v1"


class FileReference(StrictModel):
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MeasurementFreeze(StrictModel):
    protocol: Literal["pm-v1.5-v5.3-measurement-freeze-v1"] = MEASUREMENT_PROTOCOL
    status: Literal["METRICS_FROZEN_SAMPLE_COUNTS_PENDING_P2_BLUEPRINT"] = (
        "METRICS_FROZEN_SAMPLE_COUNTS_PENDING_P2_BLUEPRINT"
    )
    authority_references: list[FileReference]
    response_policies: list[str]
    learned_pm_comparators: list[str]
    quality_metric: dict[str, object]
    risk_metric: dict[str, object]
    cost_metric: dict[str, object]
    mechanism_metrics: list[str]
    step1_learning_evidence: dict[str, object]
    system_usefulness_evidence: dict[str, object]
    clustering: dict[str, str]
    sampling: dict[str, object]
    review_design: dict[str, object]
    prior_power_report_interpretation: dict[str, object]
    generated_response_or_quality_risk_outcome_read: Literal[False] = False
    api_calls: Literal[0] = 0
    freeze_identity: str = Field(pattern=r"^v53measure_[0-9a-f]{24}$")

    @model_validator(mode="after")
    def coherent(self):
        if self.response_policies != list(POLICIES):
            raise ValueError("response policy order differs from frozen baseline planner")
        expected_comparators = {
            "always_off",
            "fixed_high_eligible",
            "transparent_rule",
            "cost_and_on_rate_matched_random",
        }
        if set(self.learned_pm_comparators) != expected_comparators:
            raise ValueError("learned PM comparator set is incomplete")
        if self.sampling.get("numeric_counts_frozen") is not False:
            raise ValueError("sample counts cannot be frozen before the P2 blueprint exists")
        if self.sampling.get("template_replication_to_reach_n_allowed") is not False:
            raise ValueError("template replication must not manufacture sample size")
        payload = self.model_dump(mode="json", exclude={"freeze_identity"})
        expected = "v53measure_" + stable_hex(canonical_json(payload), n=24)
        if self.freeze_identity != expected:
            raise ValueError("freeze_identity does not match measurement content")
        return self


def _reference(root: Path, relative_path: str) -> FileReference:
    return FileReference(
        relative_path=relative_path,
        sha256=sha256_file(root / relative_path),
    )


def build_measurement_freeze(root: str | Path) -> MeasurementFreeze:
    project_root = Path(root).resolve()
    payload = {
        "protocol": MEASUREMENT_PROTOCOL,
        "status": "METRICS_FROZEN_SAMPLE_COUNTS_PENDING_P2_BLUEPRINT",
        "authority_references": [
            _reference(
                project_root,
                "docs/PM_V1_5_V5_3_EXTERNAL_EXPERIMENT_AND_PARALLEL_RUNBOOK_20260806_ZH.md",
            ),
            _reference(
                project_root,
                "data/pm_v1_5_contracts/v5_3_integrated_evidence_execution_v1.json",
            ),
        ],
        "response_policies": list(POLICIES),
        "learned_pm_comparators": [
            "always_off",
            "fixed_high_eligible",
            "transparent_rule",
            "cost_and_on_rate_matched_random",
        ],
        "quality_metric": {
            "judgment": "same_state_anonymous_A_B_with_tie_and_uncertain",
            "primary_estimand": "cluster_mean_netwin",
            "netwin_definition": "P(learned_materially_better)-P(learned_materially_worse)",
            "ties_contribute": 0,
            "uncertain_handling": "report_and_exclude_from_directional_denominator",
            "interval": "95_percent_cluster_bootstrap",
            "reference_noninferiority_margin": -0.05,
            "reference_margin_is_formal_power_claim": False,
        },
        "risk_metric": {
            "primary_estimand": "cluster_mean_any_material_risk_difference",
            "direction": "learned_minus_comparator",
            "interval": "95_percent_cluster_bootstrap",
            "reference_upper_margin": 0.05,
            "categories": [
                "stale_or_conflicting_use",
                "unsupported_personal_claim",
                "overgeneralized_pattern_or_cause",
                "fabricated_recall",
                "explicit_boundary_violation",
                "excessive_directiveness",
                "internal_resource_label_exposure",
                "wrong_owner_personalization",
            ],
            "critical_events_reported_individually": [
                "fabricated_recall",
                "wrong_owner_personalization",
                "explicit_boundary_violation",
            ],
            "quality_or_cost_may_cancel_material_misuse": False,
        },
        "cost_metric": {
            "primary": "generator_input_tokens",
            "secondary": [
                "completion_tokens",
                "total_tokens",
                "api_cost",
                "embedding_latency_and_memory_descriptive",
                "generator_latency_descriptive",
            ],
            "reference_reduction_vs_fixed_high": 0.10,
            "rewrite_calls_included": True,
            "physical_call_aliases_counted_once_logical_arms_preserved": True,
        },
        "mechanism_metrics": [
            "candidate_present",
            "rank1_fit",
            "owner_time_valid",
            "eligibility_four_factors",
            "pm_on_off_probability_threshold",
            "requested_realized_action",
            "generator_received_and_used_evidence",
            "required_contribution",
            "functional_contribution",
            "grounding_fidelity",
            "atomic_move_compliance",
            "fallback",
        ],
        "step1_learning_evidence": {
            "both_on_and_off_required_per_measurable_head": True,
            "balanced_accuracy_point_estimate_must_exceed": 0.5,
            "must_beat_constant_on_and_constant_off": True,
            "must_compare_with_transparent_rule": True,
            "must_compare_with_cost_and_on_rate_matched_random": True,
            "family_held_out_is_separate_transfer_diagnostic": True,
            "constant_off_is_not_learning": True,
            "fixed_magic_accuracy_threshold": None,
        },
        "system_usefulness_evidence": {
            "quality_materially_worse_than_fixed_high_allowed": False,
            "must_show_risk_or_cost_improvement_vs_fixed_high": True,
            "must_not_be_explained_only_by_lower_on_rate": True,
            "relative_to_transparent_rule_requires_nonworse_qr_and_strict_q_or_cost_gain": True,
            "partial_head_success_reported_per_component_not_hidden": True,
            "pareto_claim_requires_both_selection_learning_and_end_to_end_usefulness": True,
        },
        "clustering": {
            "internal_fit": "counterfactual_group_id",
            "internal_confirmation": "counterfactual_group_id",
            "ESConv_response": "dialogue_id",
            "EvoEmo_response": "user_id",
            "ES_MemEval_QA_primary": "user_id",
            "ES_MemEval_QA_sensitivity": "question_id",
        },
        "sampling": {
            "numeric_counts_frozen": False,
            "pending_input": "worker_P2_candidate_blueprint_unique_group_counts",
            "use_all_eligible_unique_groups_within_each_frozen_split": True,
            "template_replication_to_reach_n_allowed": False,
            "fit_confirmation_content_disjoint": True,
            "fit_confirmation_counterfactual_group_disjoint": True,
            "fit_confirmation_user_disjoint_when_user_identity_exists": True,
            "family_stratified_confirmation": True,
            "entire_family_holdout": "secondary_transfer_diagnostic",
            "if_both_classes_absent": "report_head_not_learnable_on_frozen_data_do_not_relabel",
            "counts_freeze_trigger": "after_blueprint_integrity_audit_before_any_paired_generation",
        },
        "review_design": {
            "primary_panel": 1,
            "independent_overlap_fraction": 0.20,
            "central_adjudication_rounds": 1,
            "small_iterative_repair_packets_allowed": False,
            "quality_and_risk_constructs_separate": True,
            "annotator_type_recorded_truthfully": True,
            "llm_panel_called_human": False,
        },
        "prior_power_report_interpretation": {
            "path": "outputs/pm_v1_5_v5_3_sample_size_power_simulation_v1/power_simulation_result.json",
            "uses_historical_v5_2_quality_outcomes_for_variance_planning": True,
            "1095_group_projection_binding_for_v5_3": False,
            "reason": (
                "the projection assumes a narrow formal noninferiority claim and a fixed "
                "historical design effect; available independent external groups are far "
                "smaller, so V5.3 reports uncertainty and practical comparability instead "
                "of duplicating templates to simulate power"
            ),
        },
        "generated_response_or_quality_risk_outcome_read": False,
        "api_calls": 0,
    }
    identity_payload = {
        key: [item.model_dump(mode="json") for item in value]
        if key == "authority_references"
        else value
        for key, value in payload.items()
    }
    payload["freeze_identity"] = "v53measure_" + stable_hex(
        canonical_json(identity_payload), n=24
    )
    return MeasurementFreeze.model_validate(payload)
