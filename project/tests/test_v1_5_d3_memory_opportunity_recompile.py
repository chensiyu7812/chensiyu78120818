from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24dv_recompile_d3_memory_opportunity_rows_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location("d3_opportunity_recompile", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_d3_rows_recompile_through_production_bridge() -> None:
    report = _module().build(root=ROOT)
    assert report["status"] == "PASS_REUSE_32_REAL_TEXT_ROWS_PER_MEMORY_HEAD"
    assert all(report["checks"].values())
    for component in ("MP", "MS", "ME"):
        values = report["component_reports"][component]
        assert values["rows"] == 32
        assert values["independent_users"] == 32
        assert values["class_counts"] == {"0": 16, "1": 16}
        assert values["all_features_match_frozen_step0"] is True


def test_recompile_never_reads_later_outcomes() -> None:
    report = _module().build(root=ROOT)
    assert report["checks"][
        "no_response_quality_risk_or_external_outcome_read"
    ] is True
    assert "generation" not in " ".join(report["inputs"])
    assert "blind" not in " ".join(report["inputs"])
