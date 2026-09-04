from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24dw_freeze_memory_opportunity_supplement_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("memory_supplement", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supplement_has_sealed_balanced_fit_and_confirmation() -> None:
    report = _module().build(root=ROOT)
    assert report["status"] == (
        "PASS_FROZEN_16_FIT_16_CONFIRMATION_MEMORY_SUPPLEMENT"
    )
    assert report["checks"]["32_new_users_16_fit_16_confirmation"]
    assert report["checks"]["fit_confirmation_semantic_families_disjoint"]
    assert report["checks"]["96_formal_rows"]
    assert report["checks"][
        "each_memory_head_each_role_16_rows_8_on_8_off"
    ]


def test_private_conditions_survive_production_compilation() -> None:
    report = _module().build(root=ROOT)
    assert report["checks"]["same_production_feature_protocol"]
    assert report["checks"][
        "low_relevance_compiles_below_relevant_each_formal_split"
    ]
    assert report["checks"]["no_hard_off_in_formal_rows"]
    assert report["checks"]["five_condition_families_each_formal_split"]


def test_supplement_is_outcome_blind_and_exactly_external_disjoint() -> None:
    report = _module().build(root=ROOT)
    assert report["api_calls_made"] == 0
    assert report["human_labels_read"] is False
    assert report["external_outcomes_read"] is False
    assert report["checks"]["external_exact_content_disjoint"]
    assert report["checks"]["zero_api_zero_outcome"]
