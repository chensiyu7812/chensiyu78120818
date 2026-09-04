from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24am_compare_post_h1_retrieval_repairs_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("post_h1_bakeoff", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptable_ranks_are_bounded_to_displayed_top_five() -> None:
    module = _module()
    assert module._acceptable_ranks("2,4,5") == {2, 4, 5}
    assert module._acceptable_ranks("") == set()


def test_confusion_reports_balanced_accuracy() -> None:
    module = _module()
    rows = [
        {"human_opportunity": True, "m_on": True},
        {"human_opportunity": True, "m_on": False},
        {"human_opportunity": False, "m_on": False},
        {"human_opportunity": False, "m_on": True},
    ]
    result = module._confusion(rows, "m")
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5
