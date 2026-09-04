from __future__ import annotations

import pytest

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.v1_5_candidate_discovery import (
    MemoryCandidate,
    describe_memory_candidate,
)
from metacom_pm.v1_5_final_candidate_contract import (
    FINAL_CANDIDATE_CONTRACT_PROTOCOL,
    FINAL_FEATURE_CONTRACT_PROTOCOL,
    CandidateOpportunityGold,
    CandidateSubtype,
    ExactCandidateTrainingRow,
    IndependentFeatureRecord,
    audit_exact_candidate_rows,
    candidate_text_sha256,
    exact_rank1_memory_surface,
    exact_rank1_strategy_surface,
    final_model_feature_projection,
    final_rs_model_features,
)
from metacom_pm.v1_5_memory_transport import (
    BOUNDED_MEMORY_COMPILER_PROTOCOL,
    compile_me_structure_metadata,
)


def _me_candidate() -> tuple[MemoryCandidate, MemoryItem]:
    item = MemoryItem(
        memory_id="mem_0123456789abcdef01234567",
        source=MemorySource.ME,
        created_session=1,
        text="I tried a short walk, which helped me feel calmer.",
    )
    candidate = MemoryCandidate(
        source=MemorySource.ME,
        selected_items=(item,),
        descriptor=describe_memory_candidate(
            source=MemorySource.ME,
            query="what helped me feel calmer",
            source_items=[item],
            selected_items=[item],
            session_index=3,
        ),
    )
    return candidate, item


def _features(state_id: str) -> IndependentFeatureRecord:
    values = final_model_feature_projection(
        component="ME",
        source_features={
            "candidate_content_match_level": 1.0,
            "candidate_incremental_information": 1.0,
            "candidate_current_goal_fit": 0.5,
            "candidate_contains_action": 1.0,
            "candidate_contains_result": 1.0,
            "candidate_contains_mechanism": 1.0,
            "current_continuity_invitation": 1.0,
        },
    )
    return IndependentFeatureRecord(
        protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
        state_id=state_id,
        component="ME",
        feature_builder_protocol="outcome-blind-me-compiler-v1",
        model_features=values,
    )


def test_exact_rank1_surface_binds_actual_item_and_compiler_hint() -> None:
    candidate, item = _me_candidate()
    metadata = compile_me_structure_metadata(item.text)
    surface = exact_rank1_memory_surface(
        state_id="state_fit_001",
        candidate=candidate,
        source_metadata={item.memory_id: metadata},
        current_session_index=4,
        compiler_protocol=BOUNDED_MEMORY_COMPILER_PROTOCOL,
    )
    assert surface.protocol == FINAL_CANDIDATE_CONTRACT_PROTOCOL
    assert surface.selected_rank == 1
    assert surface.candidate_id == item.memory_id
    assert surface.candidate_text == item.text
    assert surface.compiler_subtype_hint is CandidateSubtype.ME_REUSABLE_OUTCOME


def test_gold_with_different_candidate_text_cannot_enter_training() -> None:
    candidate, item = _me_candidate()
    surface = exact_rank1_memory_surface(
        state_id="state_fit_001",
        candidate=candidate,
        source_metadata={item.memory_id: compile_me_structure_metadata(item.text)},
        current_session_index=4,
        compiler_protocol=BOUNDED_MEMORY_COMPILER_PROTOCOL,
    )
    wrong_gold = CandidateOpportunityGold(
        state_id=surface.state_id,
        component="ME",
        candidate_id=item.memory_id,
        candidate_text_sha256=candidate_text_sha256("a different imagined event"),
        adjudicated_subtype=CandidateSubtype.ME_REUSABLE_OUTCOME,
        decision="on",
        labeler_protocol="independent-human-adjudication-v1",
        evidence_codes=("ACTION_RESULT_REUSABLE",),
    )
    with pytest.raises(ValueError, match="exact rank-1 candidate text"):
        ExactCandidateTrainingRow(
            surface=surface,
            gold=wrong_gold,
            features=_features(surface.state_id),
            split="FIT",
            group_id="user_group_001",
            semantic_family="coping_reuse",
        )


def test_me_on_rejects_context_only_event() -> None:
    with pytest.raises(ValueError, match="reusable outcome"):
        CandidateOpportunityGold(
            state_id="state_fit_002",
            component="ME",
            candidate_id="mem_abcdefabcdefabcdefabcdef",
            candidate_text_sha256="0" * 64,
            adjudicated_subtype=CandidateSubtype.ME_CONTEXT_EVENT,
            decision="on",
            labeler_protocol="independent-human-adjudication-v1",
            evidence_codes=("CONTEXT_ONLY",),
        )


def test_feature_contract_rejects_shortcut_and_outcome_fields() -> None:
    with pytest.raises(ValueError, match="feature schema mismatch"):
        IndependentFeatureRecord(
            protocol=FINAL_FEATURE_CONTRACT_PROTOCOL,
            state_id="state_fit_003",
            component="MP",
            feature_builder_protocol="feature-builder-v1",
            model_features={
                "candidate_content_match_level": 1.0,
                "candidate_incremental_information": 1.0,
                "candidate_preference_scope_fit": 1.0,
                "candidate_profile_relevance": 0.0,
                "candidate_profile_entity_scope_fit": 0.0,
                "candidate_current_scope_conflict": 0.0,
                "target_action_id": 1.0,
            },
        )


def test_valid_bound_row_passes_static_audit() -> None:
    candidate, item = _me_candidate()
    surface = exact_rank1_memory_surface(
        state_id="state_fit_004",
        candidate=candidate,
        source_metadata={item.memory_id: compile_me_structure_metadata(item.text)},
        current_session_index=4,
        compiler_protocol=BOUNDED_MEMORY_COMPILER_PROTOCOL,
    )
    gold = CandidateOpportunityGold(
        state_id=surface.state_id,
        component="ME",
        candidate_id=surface.candidate_id,
        candidate_text_sha256=surface.candidate_text_sha256,
        adjudicated_subtype=CandidateSubtype.ME_REUSABLE_OUTCOME,
        decision="on",
        labeler_protocol="independent-human-adjudication-v1",
        evidence_codes=("ACTION_RESULT_REUSABLE", "CURRENT_GOAL_FIT"),
    )
    row = ExactCandidateTrainingRow(
        surface=surface,
        gold=gold,
        features=_features(surface.state_id),
        split="FIT",
        group_id="user_group_004",
        semantic_family="coping_reuse",
    )
    report = audit_exact_candidate_rows([row])
    assert report["status"] == "PASS"
    assert report["exact_rank1_binding_enforced"] is True


def test_rs_surface_binds_rank1_to_complete_execution_card() -> None:
    card = {
        "card_id": "card_0123456789abcdef01234567",
        "support_move": "Offer one reversible experiment.",
        "when_to_use": "The user welcomes one small idea.",
        "when_not_to_use": "Do not use after a listen-only boundary.",
    }
    surface = exact_rank1_strategy_surface(
        state_id="state_fit_rs_001",
        ranked_cards=[{"card_id": card["card_id"], "score": 0.9}],
        cards_by_id={card["card_id"]: card},
        compiler_protocol="frozen-rs-ranker-v1",
    )
    assert surface.selected_rank == 1
    assert surface.candidate_id == card["card_id"]
    assert "support_move:" in (surface.candidate_text or "")
    assert surface.compiler_subtype_hint is CandidateSubtype.RS_ATOMIC_MOVE


def test_rs_features_are_candidate_aware_and_outcome_blind() -> None:
    ranked = {"card_id": "card_0123456789abcdef01234567"}
    card = {
        "card_id": ranked["card_id"],
        "strategy_family": "Providing Suggestions",
        "execution_profile": "minimal",
        "retrieval_text": "one small optional suggestion",
        "support_move": "Offer one small optional suggestion.",
        "when_to_use": "The user explicitly welcomes one idea.",
    }
    features = final_rs_model_features(
        current_user_text="Could you give me one small idea?",
        visible_dialogue=[],
        observable_flags={
            "advice_welcome": True,
            "listen_only": False,
            "low_burden": True,
            "ordinary_rag_hard_off": False,
        },
        ranked_card=ranked,
        bank_card=card,
    )
    assert set(features) == {
        "candidate_content_match_level",
        "candidate_mode_fit",
        "candidate_goal_fit",
        "candidate_burden_fit",
        "candidate_boundary_fit",
        "candidate_nonredundancy",
    }
    assert features["candidate_mode_fit"] == 1.0
    assert features["candidate_boundary_fit"] == 1.0
