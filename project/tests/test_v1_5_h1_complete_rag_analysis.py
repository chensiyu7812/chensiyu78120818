from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24aj_analyze_h1_complete_rag_review_v1_5.py"


def _load():
    spec = importlib.util.spec_from_file_location("h1_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptable_rank_parser_is_bounded() -> None:
    module = _load()
    assert module._acceptable_ranks("1,3,5") == {1, 3, 5}
    assert module._acceptable_ranks("") == set()
    assert module._acceptable_ranks("rank2 and rank4") == {2, 4}


def test_rate_handles_zero_denominator() -> None:
    module = _load()
    assert module._rate(1, 2) == 0.5
    assert module._rate(0, 0) == 0.0
