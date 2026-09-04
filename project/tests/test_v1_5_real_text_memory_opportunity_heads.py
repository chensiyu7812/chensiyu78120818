from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = (
    ROOT
    / "scripts/v1_5/24dx_train_real_text_memory_opportunity_heads_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("memory_head_trainer", TRAINER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_real_text_memory_heads_pass_frozen_gates() -> None:
    report = _module().build(root=ROOT)
    assert report["status"] == "PASS_THREE_MEMORY_HEADS_PROMOTED"
    for component in ("MP", "MS", "ME"):
        values = report["components"][component]
        assert values["status"] == "PASS_PROMOTED"
        assert all(values["checks"].values())
        assert values["five_seed_confirmation_decision_agreement"] == 1.0


def test_component_specific_floor_and_fresh_confirmation_are_explicit() -> None:
    report = _module().build(root=ROOT)
    assert report["confirmation_status"] == (
        "FRESH_V2_READ_ONCE_AFTER_REPAIR_FREEZE"
    )
    assert (
        report["components"]["MP"]["fixed_hyperparameters"][
            "final_fit_candidate_match_floor"
        ]
        is not None
    )
    assert (
        report["components"]["MS"]["fixed_hyperparameters"][
            "final_fit_candidate_match_floor"
        ]
        is not None
    )
    assert (
        report["components"]["ME"]["fixed_hyperparameters"][
            "final_fit_candidate_match_floor"
        ]
        is None
    )
    manifest = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_real_text_memory_opportunity_heads_v1/freeze_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert not manifest[
        "fresh_v2_confirmation_used_for_feature_threshold_or_hyperparameter_selection"
    ]
    assert not manifest["fresh_v2_confirmation_reusable_after_this_run"]


def test_bge_candidate_semantic_challenger_was_rejected_on_fit_only() -> None:
    report = json.loads(
        (
            ROOT
            / "outputs/pm_v1_5_bge_memory_candidate_qualification_v1/qualification_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "REJECT_BGE_KEEP_LEXICAL"
    assert report["selected_components"] == []
    assert report["diagnostic_confirmation_read"] is False
    assert report["fresh_confirmation_read"] is False
    assert report["external_outcome_read"] is False
