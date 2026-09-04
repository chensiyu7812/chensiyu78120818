from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24bf_analyze_rs_esconv_frozen_quality_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "rs_esconv_frozen_quality_analysis", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classification_metrics_keep_quality_and_cost_target_explicit() -> None:
    module = _load()
    y = np.asarray([1, 1, 0, 0])
    probability = np.asarray([0.8, 0.2, 0.7, 0.1])
    action = np.asarray([1, 0, 1, 0])
    metrics = module._classification(y, probability, action)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["positive_recall"] == 0.5
    assert metrics["negative_recall"] == 0.5
    assert metrics["predicted_on"] == 2
    assert metrics["predicted_off"] == 2


def test_frozen_prior_is_training_prevalence_not_external_outcome() -> None:
    module = _load()
    assert module.FROZEN_TRAIN_PRIOR == 13 / 32
    assert module.BOOTSTRAP_REPLICATES == 20000
