from __future__ import annotations

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.retrieval import MemoryRetriever
from metacom_pm.v1_5_candidate_discovery import (
    MemoryCandidate,
    describe_memory_candidate,
    discover_memory_candidates_by_source_query_v1_5b,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_source_specific_memory_opportunity_observation,
)


def _candidate(source: MemorySource, text: str, query: str) -> MemoryCandidate:
    prefix = {MemorySource.MP: "aa", MemorySource.MS: "bb", MemorySource.ME: "cc"}[
        source
    ]
    item = MemoryItem(
        memory_id=f"mem_{prefix}0123456789abcdef012345",
        source=source,
        created_session=1,
        text=text,
    )
    return MemoryCandidate(
        source=source,
        selected_items=(item,),
        descriptor=describe_memory_candidate(
            source=source,
            query=query,
            source_items=[item],
            selected_items=[item],
            session_index=3,
        ),
    )


def test_v1_5b_candidate_discovery_removes_exact_duplicate_top_k_slots() -> None:
    duplicate_text = "Earlier, naming one dependency made the deadline easier."
    items = [
        MemoryItem(
            memory_id=f"mem_{index:024x}",
            source=MemorySource.MS,
            created_session=index,
            text=duplicate_text,
        )
        for index in (1, 2)
    ]
    discoveries = discover_memory_candidates_by_source_query_v1_5b(
        queries={source: "deadline dependency" for source in MemorySource},
        items=items,
        retriever=MemoryRetriever(
            minimum_score_by_source={source: 0.0 for source in MemorySource}
        ),
        session_index=4,
    )
    ms = discoveries[MemorySource.MS]
    assert len(ms.selected_items) == 1
    assert ms.descriptor["retrieved_count_before_exact_text_dedup"] == 2
    assert ms.descriptor["retrieved_count_after_exact_text_dedup"] == 1
    assert ms.descriptor["exact_text_duplicates_removed"] == 1


def test_mp_source_fit_separates_current_preference_scope_conflict() -> None:
    candidate = _candidate(
        MemorySource.MP,
        "Stable support preference: when I ask for advice, offer one small option.",
        "choir advice option",
    )
    item = candidate.selected_items[0]
    common = {
        "candidate": candidate,
        "catalog_items": [item],
        "catalog_user_id": "u",
        "current_user_id": "u",
        "current_session_index": 3,
        "visible_dialogue": [],
        "source_metadata": {item.memory_id: {"mp_subtype": "MP_PREFERENCE"}},
        "background_action": "MS+R0",
    }
    welcomed = build_source_specific_memory_opportunity_observation(
        **common,
        current_user_text="Could you give me one small idea about the choir?",
    )
    rejected = build_source_specific_memory_opportunity_observation(
        **common,
        current_user_text="I don't want advice about the choir; just listen.",
    )
    assert welcomed["protocol"] == SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
    assert welcomed["model_features"]["candidate_preference_scope_fit"] == 1.0
    assert welcomed["model_features"]["candidate_current_scope_conflict"] == 0.0
    assert rejected["model_features"]["candidate_preference_scope_fit"] == 0.0
    assert rejected["model_features"]["candidate_current_scope_conflict"] == 1.0
    assert item.text not in repr(welcomed)


def test_ms_source_fit_exposes_continuity_goal_and_prior_value_separately() -> None:
    candidate = _candidate(
        MemorySource.MS,
        (
            "Last session, waiting for another team's input was the main "
            "deadline bottleneck; naming one dependency helped."
        ),
        "deadline pressure helped last time",
    )
    item = candidate.selected_items[0]
    observation = build_source_specific_memory_opportunity_observation(
        candidate=candidate,
        catalog_items=[item],
        catalog_user_id="u",
        current_user_id="u",
        current_session_index=3,
        current_user_text=(
            "The deadline pressure is back. Which part matters most, and what "
            "helped last time?"
        ),
        visible_dialogue=[],
        source_metadata={
            item.memory_id: {"summary_origin": "supplied_strictly_prior_summary"}
        },
        background_action="MPE+RS",
    )
    features = observation["model_features"]
    assert features["current_continuity_invitation"] == 1.0
    assert features["candidate_prior_outcome_or_distinction"] == 1.0
    assert features["candidate_incremental_information"] == 1.0
    assert features["candidate_current_goal_fit"] >= 0.5
