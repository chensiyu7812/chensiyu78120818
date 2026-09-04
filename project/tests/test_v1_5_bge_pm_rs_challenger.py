from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24ba_run_bge_pm_rs_challenger_v1_5.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "bge_pm_rs_challenger", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bge_challenger_is_single_frozen_low_capacity_variant() -> None:
    module = _load()
    assert module.PCA_COMPONENTS == 4
    assert module.C_VALUE == 0.03
    assert module.OUTER_FOLDS == 5
    assert len(module.SEEDS) == 5


def test_visible_state_text_contains_no_card_or_outcome() -> None:
    module = _load()
    state = {
        "visible_dialogue": [
            {"speaker": "seeker", "content": "I feel stuck."}
        ]
    }
    text = module._visible_state_text(state)
    assert "I feel stuck." in text
    assert "Candidate support guidance" not in text
    assert "Response A" not in text
    assert "selected_strategy" not in text
