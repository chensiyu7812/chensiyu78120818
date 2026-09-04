from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24af_audit_controlled_memory_routing_shortcuts_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("routing_shortcuts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shortcut_audit_uses_only_preaction_construction_features() -> None:
    module = _load()
    assert "needed_memory_sources" not in module.FEATURES
    assert "regime" not in module.FEATURES
    assert "item_utility" not in module.FEATURES
    assert "estimated_tokens" in module.FEATURES


def test_threshold_audit_checks_both_directions() -> None:
    module = _load()
    result = module._best_threshold(
        module.np.asarray([0.1, 0.2, 0.8, 0.9]),
        module.np.asarray([1, 1, 0, 0]),
    )
    assert result["balanced_accuracy"] == 1.0
    assert result["direction"] == "le"
