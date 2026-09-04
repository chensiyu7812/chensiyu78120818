from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24du_audit_pm_training_transport_stability_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location("training_transport_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_head_data_pass_is_not_misreported_as_external_readiness() -> None:
    report = _module().build(root=ROOT)
    assert report["status"] == (
        "HEAD_DATA_PASSED_EXTERNAL_TRANSPORT_RECOMPILE_REQUIRED"
    )
    assert report["component_evidence"]["RS"]["evidence_grade"].startswith("A_")
    for component in ("MP", "MS", "ME"):
        assert report["component_evidence"][component]["evidence_grade"] == (
            "A_CONTROLLED_REAL_TEXT_CONFIRMED"
        )
        assert report["component_evidence"][component]["fit_rows"] == 48
        assert report["component_evidence"][component]["confirmation_rows"] == 16


def test_legacy_micro_world_and_remaining_transport_gap_are_explicit() -> None:
    report = _module().build(root=ROOT)
    for component in ("MP", "ME"):
        evidence = report["component_evidence"][component]
        legacy = evidence["legacy_micro_world"]
        assert legacy["rows"] == 64
        assert legacy["rows_with_dialogue_or_candidate_text"] == 0
        assert (
            report["external_transport"]["by_component"][component][
                "exact_feature_protocol_match"
            ]
            is False
        )


def test_external_outcomes_are_not_used_and_repair_is_bounded() -> None:
    report = _module().build(root=ROOT)
    assert report["current_gates"]["external_policy_outcomes_used_for_this_audit"] is False
    assert report["minimum_repair_batch"]["additional_full_human_annotation_wave"] is False
    assert report["minimum_repair_batch"]["MP"]["train_rows"] == 48
    assert report["minimum_repair_batch"]["ME"]["confirmation_rows"] == 16
    assert report["current_gates"][
        "three_memory_heads_pass_real_text_fresh_confirmation"
    ]
