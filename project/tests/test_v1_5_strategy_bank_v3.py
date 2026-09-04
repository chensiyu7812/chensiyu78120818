from __future__ import annotations

from pathlib import Path

from metacom_pm.io import iter_jsonl
from metacom_pm.v1_5_strategy_bank_v3 import (
    EXPECTED_FAMILIES,
    validate_v3_taxonomy,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v3_core_taxonomy_has_ten_distinct_submoves_per_safe_family():
    rows = [
        dict(row)
        for row in iter_jsonl(
            ROOT
            / "data/strategy/pm_v1_5_strategy_bank_v3_core_taxonomy_v1.jsonl"
        )
    ]
    validated = validate_v3_taxonomy(rows)
    assert len(validated) == 50
    assert {
        row["strategy_family"] for row in validated
    } == set(EXPECTED_FAMILIES)
