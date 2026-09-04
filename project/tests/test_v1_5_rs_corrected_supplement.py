from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24z_prepare_rs_corrected_supplement_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "rs_corrected_supplement", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrected_supplement_is_16_fresh_pairs_with_six_features() -> None:
    report, states, selected, calls = _module().build(ROOT)
    assert (
        report["status"]
        == "READY_FOR_CORRECTED_SUPPLEMENT_EXECUTION_PREFLIGHT"
    )
    assert len(states) == len(selected) == 16
    assert len(calls) == 32
    assert report["selected_by_move"] == {
        "AM01_invite_open_expression": 4,
        "AM04_tentative_paraphrase_check": 4,
        "AM05_grounded_validation": 4,
        "AM10_offer_one_optional_micro_step": 4,
    }
    assert report["selection"]["zero_wave1_dialogue_overlap"]
    freeze = report["primary_model_freeze_before_supplement_outcomes"]
    assert freeze["feature_count"] == 6
    assert freeze["maximum_at_48_groups"] == 9
    assert not freeze["baai_features"]
    assert not report["execution_policy"]["old_64_call_wave2_authorized"]
    assert not report["execution_policy"][
        "this_32_call_supplement_authorized_by_plan"
    ]
