from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24ai_prepare_h1_complete_rag_review_v1_5.py"


def _load():
    spec = importlib.util.spec_from_file_location("h1_rag", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_h1_has_one_finite_review_budget() -> None:
    module = _load()
    assert module.POSITIVE_RETRIEVAL_ITEMS == 64
    assert sum(module.NEGATIVE_RETRIEVAL_TARGETS.values()) == 16
    assert module.MINIMUM_SCORE == 0.05


def test_nearest_core_never_points_to_itself() -> None:
    module = _load()
    rows = [
        {
            "submove_id": "a",
            "support_move": "reflect one feeling",
            "when_to_use": "emotion visible",
            "when_not_to_use": "no evidence",
        },
        {
            "submove_id": "b",
            "support_move": "clarify one feeling",
            "when_to_use": "emotion unclear",
            "when_not_to_use": "already clear",
        },
    ]
    result = module._nearest_core(rows)
    assert result["a"]["nearest_submove_id"] == "b"
    assert result["b"]["nearest_submove_id"] == "a"
