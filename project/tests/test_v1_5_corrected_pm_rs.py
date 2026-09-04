from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24ad_train_confirm_corrected_pm_rs_v1_5.py"


def _load():
    spec = importlib.util.spec_from_file_location("pm_rs_corrected", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_feature_count_is_six() -> None:
    module = _load()
    assert len(module.FEATURE_NAMES) == 6
    assert module.OPEN_THRESHOLD == 0.60


def test_metrics_do_not_confuse_all_off_accuracy_with_learning() -> None:
    module = _load()
    y = np.asarray([0] * 14 + [1] * 2)
    probability = np.full(16, 0.4)
    prior = np.full(16, 0.4)
    metrics = module._metrics(y, probability, prior)
    assert metrics["accuracy_at_0_60"] == 0.875
    assert metrics["balanced_accuracy_at_0_60"] == 0.5
    assert metrics["predicted_on"] == 0
