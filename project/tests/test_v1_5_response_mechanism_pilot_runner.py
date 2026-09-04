from __future__ import annotations

import importlib.util
from pathlib import Path

from metacom_pm.contracts import MemorySource


def _runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/v1_5/21f_run_longitudinal_response_mechanism_pilot_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location("response_mechanism_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top1_surface_is_post_retrieval_and_exactly_bounded() -> None:
    config = _runner_module()._top1_evidence_surface()
    assert config.enabled is True
    assert config.candidate_scope == "post_retrieval_pre_generation"
    assert config.maximum_total_memory_items == 3
    assert config.strategy_rule.maximum_items == 1
    assert config.drop_exact_duplicate_text is False
    for source in MemorySource:
        rule = config.memory_rules[source]
        assert rule.maximum_items == 1
        assert rule.minimum_current_score == 0.0
        assert rule.minimum_context_score == 0.0


def test_pilot_execution_constants_freeze_report_only_scope() -> None:
    module = _runner_module()
    assert module.PILOT_ARM == "top1_evidence_surface_same_prompt"
    assert module.CONTROL_ARM == "frozen_current_control"
    assert module.TRANSPORT_MAX_ATTEMPTS == 4
    assert module.PILOT_STAGE == "longitudinal_response_mechanism_pilot"
    assert module.CANONICAL_PILOT_CONTRACT_PATH.name == (
        "longitudinal_response_mechanism_pilot_v1.json"
    )
    assert module.CANONICAL_UPTAKE_MEASUREMENT_CONTRACT_PATH.name == (
        "longitudinal_response_mechanism_uptake_measurement_v1.json"
    )
