from __future__ import annotations

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.retrieval import MemoryRetriever, source_specific_memory_queries
from metacom_pm.v1_5_candidate_discovery import (
    discover_memory_candidates_by_source_query,
)
from metacom_pm.v1_5_memory_opportunity_features import (
    MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
    build_memory_opportunity_observation,
)
from metacom_pm.v1_5_memory_transport import (
    compile_bounded_memory_with_metadata,
    compile_me_structure_metadata,
)


def _user() -> dict:
    return {
        "id": "user_a",
        "basic_info": {
            "stable_preference_1": "Ask permission before offering advice.",
            "job": "teacher",
        },
        "dialog_history": [
            {
                "id": "s1",
                "summary": "A short walk after work helped reduce tension.",
                "dialogue": [
                    {
                        "role": "seeker",
                        "content": "Work deadlines felt overwhelming, but a walk helped.",
                    }
                ],
            },
            {
                "id": "s2",
                "summary": "",
                "dialogue": [
                    {
                        "role": "seeker",
                        "content": "The user felt lonely after moving.",
                    }
                ],
            },
        ],
    }


def test_shared_transport_metadata_distinguishes_profile_preference_and_ms_origin() -> None:
    items, _, metadata = compile_bounded_memory_with_metadata(_user())
    mp_types = {
        metadata[item.memory_id]["mp_subtype"]
        for item in items
        if item.source is MemorySource.MP
    }
    ms_origins = {
        metadata[item.memory_id]["summary_origin"]
        for item in items
        if item.source is MemorySource.MS
    }
    assert mp_types == {"MP_PREFERENCE", "MP_PROFILE"}
    assert ms_origins == {
        "supplied_strictly_prior_summary",
        "compiled_fallback_last_seeker_text",
    }
    assert set(metadata) == {item.memory_id for item in items}
    assert {metadata[item.memory_id]["owner_id"] for item in items} == {"user_a"}
    assert all(metadata[item.memory_id]["record_active"] for item in items)
    assert all(not metadata[item.memory_id]["record_superseded"] for item in items)


def test_shared_transport_metadata_preserves_explicit_record_status() -> None:
    user = _user()
    user["basic_info_record_status"] = {
        "stable_preference_1": {"active": True, "superseded": True}
    }
    user["dialog_history"][0]["memory_record_status"] = {
        "active": False,
        "superseded": True,
    }
    items, _, metadata = compile_bounded_memory_with_metadata(user)
    preference = next(
        item
        for item in items
        if metadata[item.memory_id].get("profile_field") == "stable_preference_1"
    )
    assert metadata[preference.memory_id]["record_active"] is False
    assert metadata[preference.memory_id]["record_superseded"] is True
    first_session_items = [
        item for item in items if int(item.created_session) == 1
    ]
    assert first_session_items
    assert all(
        metadata[item.memory_id]["record_active"] is False
        and metadata[item.memory_id]["record_superseded"] is True
        for item in first_session_items
    )


def test_production_feature_bridge_is_domain_identity_invariant_and_private() -> None:
    items, _, metadata = compile_bounded_memory_with_metadata(_user())
    current = "Work feels overwhelming again; what helped last time?"
    visible = [{"speaker": "seeker", "content": current}]
    queries = source_specific_memory_queries(current, visible, "")
    discoveries = discover_memory_candidates_by_source_query(
        queries=queries,
        items=items,
        retriever=MemoryRetriever(
            minimum_score_by_source={source: 0.0 for source in MemorySource}
        ),
        session_index=3,
    )
    candidate = discoveries[MemorySource.MS]
    internal = build_memory_opportunity_observation(
        candidate=candidate,
        catalog_items=items,
        catalog_user_id="user_a",
        current_user_id="user_a",
        current_session_index=3,
        current_user_text=current,
        visible_dialogue=visible,
        source_metadata=metadata,
        background_action="MPMSME+RS",
    )
    external = build_memory_opportunity_observation(
        candidate=candidate,
        catalog_items=items,
        catalog_user_id="user_a",
        current_user_id="user_a",
        current_session_index=3,
        current_user_text=current,
        visible_dialogue=visible,
        source_metadata=metadata,
        background_action="MPMSME+RS",
    )
    assert internal == external
    assert internal["protocol"] == MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
    assert internal["outcome_read"] is False
    assert set(internal["model_features"]) == {
        "candidate_state_match_score",
        "candidate_grounding_or_nonredundancy_score",
        "background_MP_on",
        "background_ME_on",
        "background_RS_on",
    }
    rendered = repr(internal)
    assert "user_a" not in rendered
    assert all(item.memory_id not in rendered for item in items)
    assert all(item.text not in rendered for item in items)


def test_wrong_user_and_unqualified_fallback_summary_fail_closed() -> None:
    items, _, metadata = compile_bounded_memory_with_metadata(_user())
    fallback = next(
        item
        for item in items
        if item.source is MemorySource.MS
        and metadata[item.memory_id]["summary_origin"]
        == "compiled_fallback_last_seeker_text"
    )
    from metacom_pm.v1_5_candidate_discovery import MemoryCandidate, describe_memory_candidate

    descriptor = describe_memory_candidate(
        source=MemorySource.MS,
        query="lonely moving",
        source_items=[item for item in items if item.source is MemorySource.MS],
        selected_items=[fallback],
        session_index=3,
    )
    observation = build_memory_opportunity_observation(
        candidate=MemoryCandidate(
            source=MemorySource.MS,
            selected_items=(fallback,),
            descriptor=descriptor,
        ),
        catalog_items=items,
        catalog_user_id="user_a",
        current_user_id="different_user",
        current_session_index=3,
        current_user_text="I feel lonely after moving.",
        visible_dialogue=[],
        source_metadata=metadata,
        background_action="M0+R0",
    )
    assert observation["deterministic_hard_off"] is True
    assert "belongs_to_current_user" in observation["hard_off_reasons"]
    assert "source_specific_eligibility" in observation["hard_off_reasons"]


def test_mp_preference_bit_is_compiled_not_hand_authored() -> None:
    item = MemoryItem(
        memory_id="mem_0123456789abcdef01234567",
        source=MemorySource.MP,
        created_session=0,
        text="Stable Preference 1: Ask permission before offering advice.",
    )
    from metacom_pm.v1_5_candidate_discovery import MemoryCandidate, describe_memory_candidate

    descriptor = describe_memory_candidate(
        source=MemorySource.MP,
        query="advice permission",
        source_items=[item],
        selected_items=[item],
        session_index=2,
    )
    observation = build_memory_opportunity_observation(
        candidate=MemoryCandidate(
            source=MemorySource.MP,
            selected_items=(item,),
            descriptor=descriptor,
        ),
        catalog_items=[item],
        catalog_user_id="u",
        current_user_id="u",
        current_session_index=2,
        current_user_text="Could you offer one idea?",
        visible_dialogue=[],
        source_metadata={item.memory_id: {"mp_subtype": "MP_PREFERENCE"}},
        background_action="M0+R0",
    )
    assert observation["model_features"]["candidate_is_preference"] == 1.0


def test_me_structure_metadata_is_outcome_blind_and_exposed_as_separate_factors() -> None:
    item = MemoryItem(
        memory_id="mem_cc0123456789abcdef012345",
        source=MemorySource.ME,
        created_session=1,
        text="I tried taking a short walk, which helped me feel calmer.",
    )
    from metacom_pm.v1_5_candidate_discovery import MemoryCandidate, describe_memory_candidate
    from metacom_pm.v1_5_memory_opportunity_features import (
        build_source_specific_memory_opportunity_observation,
    )

    descriptor = describe_memory_candidate(
        source=MemorySource.ME,
        query="what helped me feel calmer last time",
        source_items=[item],
        selected_items=[item],
        session_index=3,
    )
    metadata = compile_me_structure_metadata(item.text)
    observation = build_source_specific_memory_opportunity_observation(
        candidate=MemoryCandidate(
            source=MemorySource.ME,
            selected_items=(item,),
            descriptor=descriptor,
        ),
        catalog_items=[item],
        catalog_user_id="u",
        current_user_id="u",
        current_session_index=3,
        current_user_text="What helped me feel calmer last time?",
        visible_dialogue=[],
        source_metadata={item.memory_id: metadata},
        background_action="M0+R0",
    )
    features = observation["model_features"]
    assert metadata["me_subtype_hint"] == "ME_REUSABLE_OUTCOME"
    assert metadata["subtype_hint_is_gold"] is False
    assert features["candidate_contains_action"] == 1.0
    assert features["candidate_contains_result"] == 1.0
    assert features["candidate_contains_mechanism"] == 1.0
    assert observation["outcome_read"] is False
