from __future__ import annotations

from pathlib import Path

import pytest

from metacom_pm.contracts import ALL_ACTION_IDS
from metacom_pm.v1_5_v5_3_step2_compatibility_plan import (
    Step2CompatibilityPlan,
    build_step2_compatibility_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def test_plan_materializes_all_16_actions_without_semantic_claim() -> None:
    plan = build_step2_compatibility_plan(ROOT)
    assert [row.action_id for row in plan.rows] == list(ALL_ACTION_IDS)
    assert len(plan.rows) == 16
    assert plan.api_calls == 0
    assert plan.semantic_generator_compatibility_claimed is False
    assert plan.recovery_policy_selected is None
    assert {row.structural_status for row in plan.rows} == {"PASS"}


def test_m0_r0_is_ordinary_generation_and_other_actions_bind_exact_ids() -> None:
    plan = build_step2_compatibility_plan(ROOT)
    m0 = next(row for row in plan.rows if row.action_id == "M0+R0")
    assert m0.pm_selected_m0_r0_is_ordinary_generation is True
    assert m0.evidence_ids == []
    assert m0.evidence_count == 0
    for row in plan.rows:
        if row.action_id == "M0+R0":
            continue
        assert row.pm_selected_m0_r0_is_ordinary_generation is False
        assert row.evidence_count > 0
        assert row.evidence_ids == [
            row.candidate_ids[c] for c in row.requested_components
        ]


def test_json_roundtrip_and_identity_detect_drift() -> None:
    plan = build_step2_compatibility_plan(ROOT)
    data = plan.model_dump_json()
    assert Step2CompatibilityPlan.model_validate_json(data) == plan
    changed = plan.model_dump(mode="json")
    changed["rows"][0]["messages_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="plan_identity"):
        Step2CompatibilityPlan.model_validate(changed)
