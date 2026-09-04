from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "scripts/v1_5/23s_run_strategy_g1_weak_label_pilot_v1_5.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("strategy_g1_pilot", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weak_label_contract_has_frozen_move_and_exclusion_sets() -> None:
    module = _load_module()
    codebook = json.loads(
        (
            ROOT
            / "data/strategy/pm_v1_5_strategy_atomic_move_codebook_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert len({move["move_id"] for move in codebook["moves"]}) == 17
    assert len(module.ALLOWED_EXCLUSIONS) == 8


def test_literal_match_normalizes_case_and_space() -> None:
    module = _load_module()
    assert module._is_literal("I feel stuck", "  i FEEL   stuck today")
    assert module._is_literal("can't—really", "I can't really do that")
    assert not module._is_literal("invented excerpt", "actual response")


def test_nonliteral_recent_dialogue_move_is_dropped_not_promoted() -> None:
    module = _load_module()
    module.MoveEvidence.model_rebuild(_types_namespace=module.__dict__)
    module.ExclusionEvidence.model_rebuild(_types_namespace=module.__dict__)
    module.WeakLabelItem.model_rebuild(_types_namespace=module.__dict__)
    module.WeakLabelBatch.model_rebuild(_types_namespace=module.__dict__)
    parsed = module.WeakLabelBatch(
        items=[
            {
                "blind_item_id": "x",
                "recognized_move_ids": [
                    "AM01_invite_open_expression",
                    "AM10_offer_one_optional_micro_step",
                ],
                "move_evidence": [
                    {
                        "move_id": "AM01_invite_open_expression",
                        "response_excerpt": "What would you like to talk about?",
                    },
                    {
                        "move_id": "AM10_offer_one_optional_micro_step",
                        "response_excerpt": "Have you tried a new hobby?",
                    },
                ],
                "hard_exclusion_flags": [],
                "exclusion_evidence": [],
                "confidence": "medium",
                "notes": "",
            }
        ]
    )
    rows = module._validate_batch(
        expected_rows=[
            {
                "blind_item_id": "x",
                "recent_visible_dialogue": [
                    {
                        "speaker": "supporter",
                        "content": "What would you like to talk about?",
                    }
                ],
                "supporter_response_to_label": "Have you tried a new hobby?",
            }
        ],
        parsed=parsed,
        allowed_move_ids={
            "AM01_invite_open_expression",
            "AM10_offer_one_optional_micro_step",
        },
    )
    assert rows[0]["recognized_move_ids"] == [
        "AM10_offer_one_optional_micro_step"
    ]
    assert rows[0]["dropped_nonliteral_target_move_ids"] == [
        "AM01_invite_open_expression"
    ]
    assert rows[0]["target_response_grounding_repair_applied"] is True
