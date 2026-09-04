import pytest

from metacom_pm.v1_5_itt_policy import (
    ITTExperimentRow,
    requested_treatment_bits,
    validate_itt_row,
    validate_pre_injection_feature_names,
)
from metacom_pm.v1_5b_policy_runtime import COMPONENTS


def row(**overrides) -> ITTExperimentRow:
    all_true = {component: True for component in COMPONENTS}
    values = {
        "group_id": "group-1",
        "state_id": "state-1",
        "requested_action_id": "MPMSME+RS",
        "realized_action_id": "MPMSME+RS",
        "assigned_executor_version": "typed-adapter-v1",
        "actual_executor_version": "typed-adapter-v1",
        "assigned_generator_version": "generator-v1",
        "actual_generator_version": "generator-v1",
        "assignment_matches_plan": True,
        "candidate_binding_valid": all_true,
        "owner_binding_valid": all_true,
        "resource_presented_to_executor": all_true,
        "final_response": "A complete response.",
        "usage_present": True,
    }
    values.update(overrides)
    return ITTExperimentRow(**values)


def test_requested_action_not_realized_action_defines_treatment():
    item = row(realized_action_id="M0+R0", fallback_used=True)
    assert all(requested_treatment_bits(item).values())
    assert validate_itt_row(item).valid


def test_nonuse_fallback_and_misuse_are_valid_itt_outcomes():
    item = row(
        realized_action_id="M0+R0",
        fallback_used=True,
        functional_use={component: False for component in COMPONENTS},
        material_misuse=True,
    )
    assert validate_itt_row(item).valid


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"assignment_matches_plan": False}, "ASSIGNMENT_MISMATCH"),
        ({"final_response": ""}, "MISSING_FINAL_RESPONSE"),
        ({"usage_present": False}, "MISSING_USAGE"),
        ({"actual_executor_version": "changed"}, "EXECUTOR_IDENTITY_DRIFT"),
        ({"actual_generator_version": "changed"}, "GENERATOR_IDENTITY_DRIFT"),
    ],
)
def test_only_mechanical_failures_invalidate_rows(overrides, reason):
    validity = validate_itt_row(row(**overrides))
    assert not validity.valid
    assert reason in validity.invalid_reasons


def test_missing_requested_resource_invalidates_but_off_resource_does_not():
    presented = {component: True for component in COMPONENTS}
    presented["ME"] = False
    validity = validate_itt_row(row(resource_presented_to_executor=presented))
    assert "ME_NOT_PRESENTED_TO_EXECUTOR" in validity.invalid_reasons

    off = row(
        requested_action_id="M0+R0",
        realized_action_id="M0+R0",
        resource_presented_to_executor={component: False for component in COMPONENTS},
    )
    assert validate_itt_row(off).valid


def test_post_action_fields_cannot_enter_step1_features():
    validate_pre_injection_feature_names(
        ["candidate_similarity", "boundary_visible", "incremental_tokens"]
    )
    with pytest.raises(ValueError, match="post-action"):
        validate_pre_injection_feature_names(
            ["candidate_similarity", "fallback_used", "response_quality_score"]
        )

