from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = (
        ROOT
        / "scripts/v1_5/24ar_reinterpret_h2_rs_construct_v1_5.py"
    )
    spec = importlib.util.spec_from_file_location("h2_reinterpret", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_separates_top1_from_top3_broad_exclusions() -> None:
    module = _module()
    rows = [
        {
            "human_opportunity": "yes",
            "review_item_id": "a",
            "acceptable_candidate_numbers": [1],
            "hard_exclusion_candidate_numbers": [2],
            "transparent_ranked_candidate_numbers": [1, 2],
        },
        {
            "human_opportunity": "yes",
            "review_item_id": "b",
            "acceptable_candidate_numbers": [2],
            "hard_exclusion_candidate_numbers": [1],
            "transparent_ranked_candidate_numbers": [1, 2],
        },
    ]
    result = module._method(rows, "transparent")
    assert result["top1_acceptable_on_human_opportunity"] == 1
    assert result["top1_broad_must_not_inject_items_all_states"] == 1
    assert result["top3_acceptable_recall_diagnostic"] == 1.0
    assert result["top3_broad_must_not_inject_items_diagnostic"] == 2


def test_corrected_contract_keeps_effect_and_opportunity_separate() -> None:
    contract = json.loads(
        (
            ROOT
            / "data/pm_v1_5_contracts/rs_open_close_construct_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert "paired_material_benefit" in contract["construct_layers"]
    assert (
        contract["construct_layers"]["structural_eligibility"]["is_not"]
        == "a response-benefit label"
    )
    assert contract["retrieval_treatment"]["inject_top_k"] == 1
