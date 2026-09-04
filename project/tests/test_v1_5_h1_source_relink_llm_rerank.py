from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24al_rerank_h1_source_relink_candidates_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("h1_source_llm", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ranking_requires_ten_unique_known_candidates() -> None:
    module = _module()
    valid = module.CoreCandidateRanking(
        ranked_candidate_numbers=list(range(1, 11)),
        estimated_direct_fit_count=4,
        selection_confidence="medium",
        short_reason="bounded search",
    )
    module._validate_ranking(valid, available_numbers=set(range(1, 31)))
    invalid = valid.model_copy(
        update={"ranked_candidate_numbers": [1] * 10}
    )
    with pytest.raises(RuntimeError, match="10 unique"):
        module._validate_ranking(
            invalid, available_numbers=set(range(1, 31))
        )


def test_final_selection_uses_unique_dialogues() -> None:
    module = _module()
    rows = {
        core: [
            {
                "strategy_id": f"{core}-{index}",
                "source_dialogue_id": f"d{index}",
                "supporter_response": f"{core} response {index}",
            }
            for index in range(1, 8)
        ]
        for core in ("a", "b")
    }
    outcomes = {
        core: {"ranked_candidate_numbers": list(range(1, 7))}
        for core in rows
    }
    selected = module._final_selection(
        cores=["a", "b"],
        shuffled_by_core=rows,
        outcomes=outcomes,
        examples_per_core=2,
    )
    dialogues = [
        row["source_dialogue_id"]
        for core_rows in selected.values()
        for row in core_rows
    ]
    assert len(dialogues) == len(set(dialogues)) == 4
