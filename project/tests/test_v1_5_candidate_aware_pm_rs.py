from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/24bb_train_candidate_aware_pm_rs_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "candidate_aware_pm_rs", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_aware_contract_is_coarse_and_low_capacity() -> None:
    module = _load()
    assert len(module.FAMILIES) == 5
    assert module.C_VALUE == 0.3
    assert module.OUTER_FOLDS == 5
    assert len(module.SEEDS) == 5


def test_family_contract_has_no_item_or_outcome_feature() -> None:
    module = _load()
    serialized = " ".join(module.FAMILIES).casefold()
    for forbidden in (
        "card_id",
        "retrieval_score",
        "response",
        "judge",
        "target",
    ):
        assert forbidden not in serialized
