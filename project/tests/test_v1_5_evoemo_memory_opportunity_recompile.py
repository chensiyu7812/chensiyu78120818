from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/v1_5/24ea_recompile_evoemo_memory_opportunity_v1_5.py"
    spec = importlib.util.spec_from_file_location("evo_memory_transport", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evoemo_recompile_is_complete_outcome_blind_and_fail_closed() -> None:
    report = _module().build(root=ROOT)
    assert report["status"] == "PASS_OUTCOME_BLIND_EXTERNAL_FEATURE_TRANSPORT"
    assert all(report["checks"].values())
    assert {
        component: report["components"][component]["states"]
        for component in ("MP", "MS", "ME")
    } == {"MP": 204, "MS": 204, "ME": 204}
    # Mechanism identity passing is deliberately separate from adequate
    # external coverage.  The current raw lexical feature does not transport.
    assert report["components"]["MS"]["development_range_in_support_rate"] < 0.80
    assert report["components"]["ME"]["development_range_in_support_rate"] < 0.80
