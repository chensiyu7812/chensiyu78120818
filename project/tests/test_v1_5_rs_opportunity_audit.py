from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24y_build_rs_opportunity_audit_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location("rs_opportunity_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_opportunity_baseline_is_outcome_blind_and_balanced() -> None:
    report, rows = _module().build(ROOT)
    assert report["status"] == "AUDITED_TRANSPARENT_OPPORTUNITY_BASELINE_READY"
    assert len(rows) == 76
    assert len({row["source_dialogue_id"] for row in rows}) == 76
    assert report["class_counts"] == {
        "no_opportunity": 38,
        "opportunity": 38,
    }
    assert report["positive_by_active_move"] == {
        "AM01_invite_open_expression": 8,
        "AM02_ask_one_focused_clarification": 6,
        "AM04_tentative_paraphrase_check": 8,
        "AM05_grounded_validation": 8,
        "AM10_offer_one_optional_micro_step": 8,
    }
    assert (
        report["selection"]["selected_prior_development_overlap"] > 0
    )
    assert all(not row["selection_uses_hidden_next_response"] for row in rows)
    assert all(not row["selection_uses_native_strategy_label"] for row in rows)
    assert all(not row["selection_uses_response_outcome"] for row in rows)
