from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24ah_build_reverse_designed_blueprint_v1_5.py"


def _load():
    spec = importlib.util.spec_from_file_location("reverse_blueprint", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_sixteen_actions_are_unique() -> None:
    module = _load()
    assert len({module.action_id(index) for index in range(16)}) == 16


def test_blueprint_is_balanced_and_group_disjoint() -> None:
    module = _load()
    rows = module.build_rows()
    report = module.audit(rows)
    assert report["status"] == "BALANCED_BLUEPRINT_ALLOCATION_FROZEN"
    assert report["total_states"] == 256
    assert report["total_users"] == 64
    assert report["split_users_disjoint"] is True
    assert {
        split: Counter(
            row["action_id"] for row in rows if row["split"] == split
        ).most_common()[0][1]
        for split in ("train", "calibration", "internal_test")
    } == {"train": 10, "calibration": 3, "internal_test": 3}
