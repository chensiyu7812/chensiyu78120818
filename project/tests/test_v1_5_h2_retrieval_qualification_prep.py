from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24ap_prepare_h2_retrieval_qualification_v1_5.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("h2_retrieval_prep", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_union_candidates_is_blind_and_deduplicated() -> None:
    module = _module()
    cards = {}
    for index in range(1, 5):
        card_id = f"c{index}"
        cards[card_id] = {
            "card_id": card_id,
            "core_submove_id": f"core{index}",
            "strategy_family": "Question",
            "execution_profile": "minimal",
            "support_move": "Ask one thing.",
            "when_to_use": "When needed.",
            "when_not_to_use": "When not needed.",
            "directive_burden": "light",
            "risk_flags": [],
            "prompt_guidance": "Ask one thing.",
        }
    transparent = [{"card_id": "c1"}, {"card_id": "c2"}, {"card_id": "c3"}]
    bge = [{"card_id": "c2"}, {"card_id": "c3"}, {"card_id": "c4"}]
    public, mapping = module._union_candidates(
        state_id="s1",
        transparent=transparent,
        bge=bge,
        cards_by_id=cards,
    )
    assert len(public) == 4
    assert set(mapping) == {"c1", "c2", "c3", "c4"}
    assert all("score" not in row for row in public)
