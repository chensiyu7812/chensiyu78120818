from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24ac_analyze_rs_corrected_supplement_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("rs_corr_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_unblinding() -> None:
    module = _load()
    key = {"response_a_arm": "RS", "response_b_arm": "R0"}
    assert (
        module._quality_result(
            {"quality_preference": "A_materially_better"}, key
        )
        == "RS"
    )
    assert (
        module._quality_result(
            {"quality_preference": "B_materially_better"}, key
        )
        == "R0"
    )
    assert (
        module._quality_result(
            {"quality_preference": "materially_equivalent"}, key
        )
        == "tie"
    )


def test_ordered_ellipsis_excerpt_validation() -> None:
    module = _load()
    response = "Start with one thing. Then choose a topic. Listen carefully."
    assert module._ordered_ellipsis_excerpt(
        "Start with one thing. ... Listen carefully.", response
    )
    assert not module._ordered_ellipsis_excerpt(
        "Listen carefully. ... Start with one thing.", response
    )
