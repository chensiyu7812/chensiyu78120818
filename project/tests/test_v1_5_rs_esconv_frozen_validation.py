from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24bc_prepare_rs_esconv_frozen_validation_v1_5.py"
)
RUN_SCRIPT = (
    ROOT
    / "scripts/v1_5/24bd_run_rs_esconv_frozen_validation_v1_5.py"
)
BLIND_SCRIPT = (
    ROOT
    / "scripts/v1_5/24be_prepare_rs_esconv_frozen_blind_review_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "rs_esconv_frozen_validation", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_family_policy_reproduces_development_candidate() -> None:
    module = _load()
    final_fit = {
        "threshold": 0.4,
        "intercept_for_RS_on": -0.23400570673846433,
        "coefficient_for_RS_on_by_family": {
            "Question": 0.17891587389547364,
            "Providing Suggestions": -0.41890315361254066,
            "Affirmation and Reassurance": 0.07251327154880871,
            "Restatement or Paraphrasing": 0.27952122018503417,
            "Reflection of feelings": -0.34605291875524014,
        },
    }
    decisions = {
        family: module._policy_for_family(family, final_fit)[1]
        for family in module.EXPECTED_FAMILIES
    }
    assert decisions == {
        "Question": "M0+RS",
        "Providing Suggestions": "M0+R0",
        "Affirmation and Reassurance": "M0+RS",
        "Restatement or Paraphrasing": "M0+RS",
        "Reflection of feelings": "M0+R0",
    }


def test_external_panel_contract_is_small_and_outcome_blind() -> None:
    module = _load()
    assert module.TARGET_PER_FAMILY == 8
    assert module.EXPECTED_TEST_STATES == 2112
    assert module.EXPECTED_TEST_DIALOGUES == 169
    assert "frozen-validation" in module.PROTOCOL


def test_execution_protocol_is_bound_to_the_frozen_plan() -> None:
    spec = importlib.util.spec_from_file_location(
        "run_rs_esconv_frozen_validation", RUN_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PLAN_PROTOCOL == (
        "pm-v1.5-rs-esconv-frozen-validation-plan-v1"
    )
    assert module.PROTOCOL.endswith("execution-v1")


def test_blind_review_order_is_deterministic() -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_rs_esconv_frozen_blind", BLIND_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pair_id = "rs_external_pair_example"
    assert module._order(pair_id) == module._order(pair_id)
    assert module.PROTOCOL.endswith("quality-blind-v1")
