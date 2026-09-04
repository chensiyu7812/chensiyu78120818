from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24ao_freeze_final_strategy_bank_v1_5.py"


def _module():
    spec = importlib.util.spec_from_file_location("final_bank_freeze", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_bank_validator_requires_two_profiles_and_sources() -> None:
    module = _module()
    cards = []
    for core_index in range(40):
        for profile in ("minimal", "dialogic"):
            cards.append(
                {
                    "card_id": f"c-{core_index}-{profile}",
                    "core_submove_id": f"core-{core_index}",
                    "execution_profile": profile,
                    "content_scope": "technique_only",
                    "raw_source_response_exposed_to_generator": False,
                    "source_qualification": {
                        "status": "HUMAN_SOURCE_EVIDENCE_QUALIFIED",
                        "accepted_source_dialogues": 2,
                    },
                }
            )
    module._validate_final_bank(cards)
    cards[0]["source_qualification"]["accepted_source_dialogues"] = 1
    with pytest.raises(RuntimeError, match="fewer than two"):
        module._validate_final_bank(cards)
