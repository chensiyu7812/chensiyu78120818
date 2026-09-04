from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24ak_prepare_h1_source_relink_review_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location("h1_source_relink", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_eligibility_requires_seeker_and_family() -> None:
    module = _module()
    row = {
        "strategy_label": "Question",
        "source_dialogue_id": "d1",
        "recent_dialogue": [{"speaker": "seeker", "content": "help"}],
        "domain_claim_keyword_flag": False,
        "word_count": 5,
        "supporter_response": "What feels most important right now?",
    }
    assert module._eligible_source(
        row, family="Question", excluded_dialogues=set()
    )
    assert not module._eligible_source(
        row, family="Providing Suggestions", excluded_dialogues=set()
    )
    row["recent_dialogue"][-1]["speaker"] = "supporter"
    assert not module._eligible_source(
        row, family="Question", excluded_dialogues=set()
    )


def test_candidate_selection_enforces_global_dialogue_uniqueness() -> None:
    module = _module()
    pools = {}
    for core in ("a", "b"):
        pools[core] = [
            {
                "source_dialogue_id": f"d{index}",
                "source_turn_index": index,
                "supporter_response": f"{core} response {index}",
                "problem_type": f"p{index % 2}",
                "nli_entailment": 1.0 - index / 100,
                "lexical_score": 0.1,
            }
            for index in range(1, 8)
        ]
    selected = module._select_candidates(pools, examples_per_core=2)
    dialogues = [
        row["source_dialogue_id"]
        for rows in selected.values()
        for row in rows
    ]
    assert len(dialogues) == 4
    assert len(set(dialogues)) == 4
