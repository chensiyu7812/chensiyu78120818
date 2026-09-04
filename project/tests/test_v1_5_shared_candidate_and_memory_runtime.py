from __future__ import annotations

from metacom_pm.contracts import (
    DialogueTurn,
    MemoryItem,
    MemorySource,
    RuntimeState,
)
from metacom_pm.prompts import generation_messages
from metacom_pm.retrieval import MemoryRetriever
from metacom_pm.v1_5_candidate_discovery import (
    CANDIDATE_DISCOVERY_PROTOCOL,
    discover_memory_candidates,
)
from metacom_pm.v1_5_memory_transport import (
    bounded_prior_history_user,
    compile_bounded_memory,
)


def _user(session_count: int = 10) -> dict:
    return {
        "id": "synthetic_user",
        "basic_info": {"job": "teacher"},
        "dialog_history": [
            {
                "id": f"s{index}",
                "timestamp": f"2026-01-{index:02d}",
                "summary": "" if index == 1 else f"Summary {index}",
                "dialogue": [
                    {
                        "role": "seeker",
                        "content": f"Work concern {index}",
                    }
                ],
            }
            for index in range(1, session_count + 1)
        ],
    }


def test_shared_compiler_uses_all_prior_history_and_relative_time_surface() -> None:
    bounded = bounded_prior_history_user(_user())
    assert len(bounded["dialog_history"]) == 10
    assert bounded["dialog_history"][0]["id"] == "s1"
    assert all(not row["timestamp"] for row in bounded["dialog_history"])
    assert all(row["summary"] for row in bounded["dialog_history"])

    items, _ = compile_bounded_memory(_user())
    state = RuntimeState(
        state_id="state",
        card_id="card",
        user_id="synthetic_user",
        split="development",
        semantic_family="test",
        current_user_text="Work is difficult today.",
        current_session_history=[
            DialogueTurn(role="assistant", content="What feels hardest?")
        ],
        current_session_summary="",
        session_index=11,
        inventory={},
        allowed_actions=["M0+R0", "M0+RS"],
        provenance={},
    )
    messages = generation_messages(state, [items[-1]], [])
    prompt = messages[-1]["content"]
    assert "session ago" in prompt
    assert "2026-" not in prompt


def test_shared_candidate_descriptor_exposes_no_text_or_id() -> None:
    items = [
        MemoryItem(
            memory_id="mem_0123456789abcdef0123",
            source=MemorySource.ME,
            created_session=1,
            text="The user previously discussed work pressure.",
        )
    ]
    discoveries = discover_memory_candidates(
        query="work pressure",
        items=items,
        retriever=MemoryRetriever(
            minimum_score_by_source={
                source: 0.0 for source in MemorySource
            }
        ),
        session_index=2,
    )
    descriptor = discoveries[MemorySource.ME].descriptor
    assert descriptor["protocol"] == CANDIDATE_DISCOVERY_PROTOCOL
    assert descriptor["candidate_present"] is True
    assert "mem_0123456789abcdef0123" not in repr(descriptor)
    assert items[0].text not in repr(descriptor)
    assert descriptor["model_features"]["candidate_present"] is True
    assert (
        descriptor["model_features"]["minimum_relative_age_bucket"] == 3
    )
