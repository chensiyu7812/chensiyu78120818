from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/v1_5/24ab_prepare_rs_corrected_supplement_blind_review_v1_5.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("corrected_blind", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blinding_is_deterministic_and_balanced() -> None:
    module = _load()
    pair_ids = [f"pair_{index}" for index in range(32)]
    first = [module._a_arm(pair_id) for pair_id in pair_ids]
    second = [module._a_arm(pair_id) for pair_id in pair_ids]
    assert first == second
    assert set(first) == {"R0", "RS"}


def test_public_packet_has_no_arm_or_card_identity() -> None:
    packet_path = (
        ROOT
        / "outputs/pm_v1_5_rs_corrected_16_pair_supplement_v1_independent_blind"
        / "human_blind_packet.jsonl"
    )
    if not packet_path.exists():
        return
    for line in packet_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {
            "protocol",
            "blind_item_id",
            "visible_dialogue",
            "response_a",
            "response_b",
        }
        serialized = json.dumps(row)
        assert "selected_strategy_card_id" not in serialized
        assert "response_a_arm" not in serialized


def test_risk_template_is_atomic_and_multiselect() -> None:
    module = _load()
    risk = module._risk_template()
    assert risk == {
        "any_material_risk": None,
        "selected_categories": [],
        "evidence_by_category": {},
    }
    assert len(module.RISK_CATEGORIES) == 5
