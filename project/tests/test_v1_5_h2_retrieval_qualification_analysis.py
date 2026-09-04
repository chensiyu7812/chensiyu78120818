from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24aq_analyze_h2_retrieval_qualification_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("h2_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_metrics_counts_off_as_top1_and_top3_failure() -> None:
    module = _module()
    rows = [
        {
            "human_opportunity": "yes",
            "acceptable_candidate_numbers": [2],
            "hard_exclusion_candidate_numbers": [],
            "m_ranked_candidate_numbers": [2, 3],
        },
        {
            "human_opportunity": "yes",
            "acceptable_candidate_numbers": [1],
            "hard_exclusion_candidate_numbers": [],
            "m_ranked_candidate_numbers": [],
        },
        {
            "human_opportunity": "no",
            "acceptable_candidate_numbers": [],
            "hard_exclusion_candidate_numbers": [],
            "m_ranked_candidate_numbers": [],
        },
    ]
    result = module._method_metrics(rows, method="m")
    assert result["opportunity_items"] == 2
    assert result["top1_acceptable_rate"] == 0.5
    assert result["top3_acceptable_recall"] == 0.5


def test_confusion_balanced_accuracy() -> None:
    module = _module()
    rows = [
        {"human_opportunity": "yes", "on": True},
        {"human_opportunity": "yes", "on": False},
        {"human_opportunity": "no", "on": False},
        {"human_opportunity": "no", "on": True},
    ]
    result = module._confusion(rows, prediction_field="on")
    assert result["balanced_accuracy"] == 0.5
