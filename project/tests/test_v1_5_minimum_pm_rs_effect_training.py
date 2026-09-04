from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24az_train_minimum_pm_rs_effect_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "minimum_pm_rs_effect", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_contract_is_low_capacity_and_pre_generation() -> None:
    module = _load()
    assert len(module.FEATURE_NAMES) == 6
    assert set(module.FEATURE_NAMES) == {
        "advice_welcome",
        "emotion_visible",
        "uncertainty_or_multi_concern",
        "current_user_token_estimate",
        "visible_dialogue_turn_count",
        "current_user_has_question_mark",
    }
    assert not any(
        token in feature.casefold()
        for feature in module.FEATURE_NAMES
        for token in module.FORBIDDEN_FEATURE_TOKENS
    )


def test_threshold_selection_requires_both_decisions() -> None:
    module = _load()
    assert module.MIN_DECISION_FRACTION == 0.20
    assert min(module.THRESHOLD_GRID) == 0.30
    assert max(module.THRESHOLD_GRID) == 0.70
    assert module.OUTER_FOLDS == 5
    assert len(module.SEEDS) == 5
