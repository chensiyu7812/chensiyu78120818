from __future__ import annotations

import json
from pathlib import Path

from metacom_pm.contracts import DialogueTurn, MemoryItem, MemorySource, RuntimeState
from metacom_pm.prompts import (
    RESOURCE_MATCHED_GENERATION_PROTOCOL,
    RESOURCE_MATCHED_GENERATION_PROTOCOL_V2,
    generation_messages,
    resource_matched_generation_messages,
    resource_matched_generation_messages_v2,
)


ROOT = Path(__file__).resolve().parents[1]


def _state() -> RuntimeState:
    return RuntimeState(
        state_id="state_alignment",
        card_id="card_alignment",
        user_id="user_alignment",
        split="development",
        semantic_family="alignment",
        current_user_text="I want one small optional idea today.",
        current_session_history=[
            DialogueTurn(role="assistant", content="What would feel manageable?")
        ],
        current_session_summary="",
        session_index=5,
        inventory={},
        allowed_actions=["M0+R0", "M0+RS"],
        provenance={},
    )


def _memory(source: MemorySource, suffix: str, text: str) -> MemoryItem:
    return MemoryItem(
        memory_id="mem_" + suffix * 20,
        source=source,
        created_session=2,
        text=text,
    )


def test_resource_matched_prompt_explains_distinct_source_roles() -> None:
    memories = [
        _memory(MemorySource.MP, "a", "The user prefers one idea at a time."),
        _memory(MemorySource.MS, "b", "The prior session ended with one open question."),
        _memory(MemorySource.ME, "c", "A short walk helped once after work."),
    ]
    prompt = resource_matched_generation_messages(_state(), memories, [])[1][
        "content"
    ]
    assert "MP: use an active preference" in prompt
    assert "MS: use a completed prior-session summary" in prompt
    assert "ME: use a prior event or outcome" in prompt
    assert "use the smallest subset" in prompt.lower()
    assert "current user message overrides" in prompt
    assert "always/never pattern" in prompt


def test_legacy_prompt_surface_remains_reproducible() -> None:
    memory = _memory(MemorySource.ME, "d", "A prior event was discussed.")
    legacy = generation_messages(_state(), [memory], [])[1]["content"]
    assert "Source-specific use contract" not in legacy
    assert "Potentially useful past information" in legacy


def test_alignment_contract_matches_prompt_protocol() -> None:
    path = (
        ROOT
        / "data/pm_v1_5_contracts/generator_resource_use_alignment_v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    assert contract["protocol"] == RESOURCE_MATCHED_GENERATION_PROTOCOL
    assert contract["causal_order"][1] == "pm_step1_component_routing"
    assert contract["measurement_layers"][2]["layer"] == "L2_generator_use"
    assert "grounded_functional_use" in contract["measurement_layers"][2]["labels"]


def test_v2_prompt_preserves_temporal_status_and_redundancy_boundary() -> None:
    memories = [
        _memory(MemorySource.MP, "a", "The user recently moved."),
        _memory(MemorySource.MS, "b", "A late shift affected focus last time."),
        _memory(MemorySource.ME, "c", "One external dependency mattered once."),
    ]
    prompt = resource_matched_generation_messages_v2(_state(), memories, [])[1][
        "content"
    ]
    assert RESOURCE_MATCHED_GENERATION_PROTOCOL_V2.endswith(
        "v2-temporal-grounding"
    )
    assert "what was true then" in prompt
    assert "past provenance" in prompt
    assert "already stated" in prompt
    assert "not just X" in prompt
