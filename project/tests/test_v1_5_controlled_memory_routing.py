from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24ae_train_controlled_memory_routing_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("memory_routing", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_contract_is_low_capacity_and_pre_action() -> None:
    module = _load()
    assert len(module.FEATURE_NAMES) == 6
    assert "regime" not in module.FEATURE_NAMES
    assert "needed_memory_sources" not in module.FEATURE_NAMES
    assert "semantic_family" not in module.FEATURE_NAMES


def test_feature_extraction_does_not_read_evaluator_labels() -> None:
    module = _load()
    state = {
        "inventory": {
            "ME": {
                "query_similarity_mean": 0.7,
                "estimated_tokens": 50,
                "min_age_sessions": 1,
                "max_age_sessions": 3,
            }
        },
        "current_user_text": "I need help with this.",
        "current_session_history": [{"role": "user", "content": "Earlier"}],
    }
    assert module._features(state, "ME") == [0.7, 50.0, 1.0, 3.0, 5.0, 1.0]
