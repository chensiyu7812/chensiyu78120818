from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.v1_5_candidate_discovery import (
    materialize_rank1_memory_for_execution,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v1_5/24er_prepare_final_system_generation_v1_5.py"
SPEC = importlib.util.spec_from_file_location("final_generation_prepare", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _card() -> dict[str, object]:
    return {
        "card_id": "card_1",
        "support_move": "Ask one focused question.",
        "when_to_use": "Use when one missing detail blocks understanding.",
        "when_not_to_use": "Do not use after a stop or no-probing boundary.",
    }


def test_rank_wrapper_resolves_to_full_generator_card() -> None:
    index = MODULE._index_strategy_cards([_card()])
    resolved = MODULE._resolve_ranked_card(
        [{"card_id": "card_1", "score": 0.9}], index
    )
    assert resolved == _card()
    surface = MODULE._card_surface(resolved)
    assert "support_move: Ask one focused question." in surface
    assert "when_to_use:" in surface
    assert "when_not_to_use:" in surface


def test_strategy_card_index_rejects_empty_or_duplicate_ids() -> None:
    with pytest.raises(RuntimeError, match="empty card_id"):
        MODULE._index_strategy_cards([{**_card(), "card_id": " "}])
    with pytest.raises(RuntimeError, match="duplicate card_id"):
        MODULE._index_strategy_cards([_card(), _card()])


def test_strategy_card_surface_requires_all_declared_fields() -> None:
    broken = _card()
    broken["when_not_to_use"] = ""
    with pytest.raises(RuntimeError, match="when_not_to_use"):
        MODULE._card_surface(broken)


def test_memory_execution_surface_is_one_explicitly_historical_item() -> None:
    first = MemoryItem(
        memory_id="mem_111111111111",
        source=MemorySource.ME,
        created_session=3,
        text="A short pause reduced pressure during one earlier episode.",
    )
    second = MemoryItem(
        memory_id="mem_222222222222",
        source=MemorySource.ME,
        created_session=4,
        text="This rank-two text must not be materialized.",
    )
    selected = (first, second)
    surface, execution_item = materialize_rank1_memory_for_execution(
        source=MemorySource.ME, selected_items=selected, session_index=8
    )
    assert execution_item.memory_id == first.memory_id
    assert "ONE PAST SEEKER EPISODE" in surface
    assert "5 sessions ago" in surface
    assert first.text in surface
    assert second.text not in surface


def test_context_only_action_legitimately_requires_no_resource() -> None:
    MODULE._require_active_resources(
        action_id="M0+R0", active=[], resources={}, state_id="state_1"
    )


def test_only_components_turned_on_by_step1_must_be_nonempty() -> None:
    MODULE._require_active_resources(
        action_id="ME+R0",
        active=["ME"],
        resources={"ME": "Past event candidate: one bounded event."},
        state_id="state_1",
    )
    with pytest.raises(RuntimeError, match="empty resources.*RS"):
        MODULE._require_active_resources(
            action_id="ME+RS",
            active=["ME", "RS"],
            resources={"ME": "Past event candidate: one bounded event.", "RS": "  "},
            state_id="state_1",
        )
