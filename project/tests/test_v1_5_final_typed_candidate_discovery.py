import numpy as np
import pytest

from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.v1_5_candidate_discovery import (
    FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL,
    discover_final_typed_memory_candidates,
    final_typed_content_match_level,
)
from metacom_pm.v1_5_strategy_rag_repair import (
    pre_pm_strategy_candidate_family,
    pre_pm_strategy_retrieval_query,
    effect_study_observable_flags,
    rank_pre_pm_strategy_candidates,
    repair_v2_rank_applicable_cards,
)
from metacom_pm.io import iter_jsonl
from pathlib import Path


def _queries(text: str):
    return {source: text for source in MemorySource}


class FakeMsEncoder:
    """Deterministic keyword-match encoder, same pattern as
    test_v1_5_v5_3_semantic_ms_retrieval.FakeEncoder -- avoids depending on
    torch/transformers/model weights in this test file."""

    def __init__(self, vocabulary: list[str]) -> None:
        self.vocabulary = vocabulary

    def encode(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([1.0 if word in lowered else 0.0 for word in self.vocabulary])
        matrix = np.array(rows, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def test_final_typed_discovery_filters_function_word_only_matches() -> None:
    ids = {
        MemorySource.MP: "mem_aaaaaaaaaaaaaaaaaaaa",
        MemorySource.MS: "mem_bbbbbbbbbbbbbbbbbbbb",
        MemorySource.ME: "mem_cccccccccccccccccccc",
    }
    items = [
        MemoryItem(
            memory_id=ids[source],
            source=source,
            created_session=1,
            text="The user was there and it was something they had.",
        )
        for source in MemorySource
    ]
    result = discover_final_typed_memory_candidates(
        queries=_queries("I am here and I want to talk about deadlines."),
        items=items,
        source_metadata={
            "mem_aaaaaaaaaaaaaaaaaaaa": {"mp_subtype": "MP_PROFILE"},
            "mem_bbbbbbbbbbbbbbbbbbbb": {
                "summary_origin": "supplied_strictly_prior_summary"
            },
            "mem_cccccccccccccccccccc": {
                "me_subtype_hint": "ME_CONTEXT_EVENT"
            },
        },
        session_index=2,
    )
    assert all(not candidate.selected_items for candidate in result.values())
    assert all(
        candidate.descriptor["protocol"]
        == FINAL_TYPED_CANDIDATE_DISCOVERY_PROTOCOL
        for candidate in result.values()
    )


def test_final_typed_discovery_prefers_reusable_me_within_content_tier() -> None:
    items = [
        MemoryItem(
            memory_id="mem_11111111111111111111",
            source=MemorySource.ME,
            created_session=2,
            text="The team deadline felt uncertain and difficult.",
        ),
        MemoryItem(
            memory_id="mem_22222222222222222222",
            source=MemorySource.ME,
            created_session=1,
            text="I wrote one team deadline question and it helped.",
        ),
    ]
    result = discover_final_typed_memory_candidates(
        queries=_queries("The team deadline is difficult."),
        items=items,
        source_metadata={
            "mem_11111111111111111111": {
                "me_subtype_hint": "ME_CONTEXT_EVENT"
            },
            "mem_22222222222222222222": {
                "me_subtype_hint": "ME_REUSABLE_OUTCOME"
            },
        },
        session_index=3,
    )
    assert (
        result[MemorySource.ME].selected_items[0].memory_id
        == "mem_22222222222222222222"
    )


def test_final_typed_content_match_ignores_support_boilerplate() -> None:
    assert (
        final_typed_content_match_level(
            "I want to talk about my deadlines.",
            "The user discussed feelings about exercise.",
        )
        == 0.0
    )


def test_mp_label_prefix_does_not_create_false_content_match() -> None:
    # Regression: MP items render as "Job: office worker" (see
    # evoemo.build_evo_memory). Matching on the raw text made the structural
    # label "Job" fire on any query mentioning "job", even one about someone
    # else's job -- unrelated to whether "office worker" was ever true here.
    items = [
        MemoryItem(
            memory_id="mem_dddddddddddddddddddd",
            source=MemorySource.MP,
            created_session=0,
            text="Job: office worker",
        )
    ]
    result = discover_final_typed_memory_candidates(
        queries=_queries(
            "My partner's job requires last-minute travel and it's stressful."
        ),
        items=items,
        source_metadata={},
        session_index=1,
    )
    assert len(result[MemorySource.MP].selected_items) == 0


def test_mp_value_content_still_matches_after_label_stripped() -> None:
    items = [
        MemoryItem(
            memory_id="mem_eeeeeeeeeeeeeeeeeeee",
            source=MemorySource.MP,
            created_session=0,
            text="Job: office worker",
        )
    ]
    result = discover_final_typed_memory_candidates(
        queries=_queries(
            "I'm exhausted from my office job, being an office worker is hard."
        ),
        items=items,
        source_metadata={},
        session_index=1,
    )
    assert (
        result[MemorySource.MP].selected_items[0].memory_id
        == "mem_eeeeeeeeeeeeeeeeeeee"
    )


def test_mp_parenthetical_status_word_does_not_create_false_content_match() -> None:
    # Regression: real p11 case -- "Education: high school (in progress)"
    # fired 6/9 times on unrelated therapy-speak sentences via the word
    # "progress" inside "(in progress)", not because the query was ever
    # actually about the user's education status.
    items = [
        MemoryItem(
            memory_id="mem_ffffffffffffffffffff",
            source=MemorySource.MP,
            created_session=0,
            text="Education: high school (in progress)",
        )
    ]
    result = discover_final_typed_memory_candidates(
        queries=_queries(
            "I guess I just need to remind myself that healing is not a linear "
            "process. Setbacks like this encounter with John can happen."
        ),
        items=items,
        source_metadata={},
        session_index=1,
    )
    assert len(result[MemorySource.MP].selected_items) == 0


def test_mp_high_school_still_matches_a_real_school_query() -> None:
    items = [
        MemoryItem(
            memory_id="mem_aabbccddaabbccddaabb",
            source=MemorySource.MP,
            created_session=0,
            text="Education: high school (in progress)",
        )
    ]
    result = discover_final_typed_memory_candidates(
        queries=_queries("I'm putting in a lot of effort to improve my study habits at high school."),
        items=items,
        source_metadata={},
        session_index=1,
    )
    assert (
        result[MemorySource.MP].selected_items[0].memory_id
        == "mem_aabbccddaabbccddaabb"
    )


def test_ms_semantic_encoder_is_opt_in_default_stays_lexical() -> None:
    # Same items/query as the lexical-tiebreak default would rank one way;
    # with no encoder passed, behavior must be byte-for-byte the pre-existing
    # content-match-tier + lexical path (no accidental behavior change for
    # any caller that does not pass ms_semantic_encoder).
    items = [
        MemoryItem(
            memory_id="mem_11111111111111111111",
            source=MemorySource.MS,
            created_session=1,
            text="Talked about the new job and feeling anxious about it.",
        ),
        MemoryItem(
            memory_id="mem_22222222222222222222",
            source=MemorySource.MS,
            created_session=2,
            text="Discussed the upcoming financial planning seminar.",
        ),
    ]
    result = discover_final_typed_memory_candidates(
        queries=_queries("I'm anxious about the seminar."),
        items=items,
        source_metadata={},
        session_index=3,
    )
    assert result[MemorySource.MS].descriptor["ms_semantic_reranked"] is False


def test_ms_semantic_encoder_reranks_ms_only_when_passed() -> None:
    items = [
        MemoryItem(
            memory_id="mem_33333333333333333333",
            source=MemorySource.MS,
            created_session=1,
            text="Talked about the new job.",
        ),
        MemoryItem(
            memory_id="mem_44444444444444444444",
            source=MemorySource.MS,
            created_session=2,
            text="Discussed the upcoming financial planning seminar.",
        ),
    ]
    encoder = FakeMsEncoder(["seminar", "job"])
    result = discover_final_typed_memory_candidates(
        queries=_queries("I'm anxious about the seminar."),
        items=items,
        source_metadata={},
        session_index=3,
        ms_semantic_encoder=encoder,
    )
    assert result[MemorySource.MS].descriptor["ms_semantic_reranked"] is True
    assert (
        result[MemorySource.MS].selected_items[0].memory_id
        == "mem_44444444444444444444"
    )


def test_ms_semantic_encoder_reports_semantic_relevance_not_stale_lexical() -> None:
    # Regression: describe_memory_candidate() computes top1_lexical_relevance
    # from lexical_score over source_items regardless of which mechanism
    # picked `selected` -- correct when lexical_score was part of ranking,
    # stale/misleading once BGE picks a different top-1. The BGE branch must
    # additionally report the score it actually ranked on.
    items = [
        MemoryItem(
            memory_id="mem_88888888888888888888",
            source=MemorySource.MS,
            created_session=1,
            text="Talked about the new job.",
        ),
        MemoryItem(
            memory_id="mem_99999999999999999999",
            source=MemorySource.MS,
            created_session=2,
            text="Discussed the upcoming financial planning seminar.",
        ),
    ]
    encoder = FakeMsEncoder(["seminar", "job"])
    result = discover_final_typed_memory_candidates(
        queries=_queries("I'm anxious about the seminar."),
        items=items,
        source_metadata={},
        session_index=3,
        ms_semantic_encoder=encoder,
    )
    descriptor = result[MemorySource.MS].descriptor
    assert "top1_semantic_relevance" in descriptor
    assert "top1_top2_semantic_margin" in descriptor
    # BGE picked mem_99999999999999999999 (matches "seminar"), which has zero
    # lexical-word overlap with source_items text as scored independently by
    # lexical_score -- the semantic relevance field must reflect the actual
    # winning BGE score (> 0), not get left at a stale/unrelated number.
    assert descriptor["top1_semantic_relevance"] > 0.0


def test_ms_semantic_encoder_still_enforces_causal_boundary() -> None:
    # rank_ms_candidates itself has no session_index awareness; the causal
    # hard-assertion must still come from describe_memory_candidate, which
    # every branch (lexical or BGE) routes through identically.
    items = [
        MemoryItem(
            memory_id="mem_55555555555555555555",
            source=MemorySource.MS,
            created_session=5,
            text="A future or current session summary about the seminar.",
        ),
    ]
    encoder = FakeMsEncoder(["seminar"])
    with pytest.raises(ValueError, match="current or future memory"):
        discover_final_typed_memory_candidates(
            queries=_queries("I'm anxious about the seminar."),
            items=items,
            source_metadata={},
            session_index=5,
            ms_semantic_encoder=encoder,
        )


def test_ms_semantic_encoder_does_not_affect_mp_or_me() -> None:
    items = [
        MemoryItem(
            memory_id="mem_66666666666666666666",
            source=MemorySource.MP,
            created_session=0,
            text="Job: office worker",
        ),
        MemoryItem(
            memory_id="mem_77777777777777777777",
            source=MemorySource.ME,
            created_session=1,
            text="I tried journaling and it helped me feel calmer.",
        ),
    ]
    encoder = FakeMsEncoder(["job"])
    result = discover_final_typed_memory_candidates(
        queries=_queries("My office job and journaling helped."),
        items=items,
        source_metadata={},
        session_index=2,
        ms_semantic_encoder=encoder,
    )
    assert result[MemorySource.MP].descriptor["ms_semantic_reranked"] is False
    assert result[MemorySource.ME].descriptor["ms_semantic_reranked"] is False


def _strategy_cards():
    root = Path(__file__).resolve().parents[1]
    return list(
        iter_jsonl(
            root
            / "outputs/pm_v1_5_strategy_bank_v4_final_v1"
            / "strategy_cards_v4_final.jsonl"
        )
    )


def test_final_rs_recognizes_optional_single_idea_request() -> None:
    current = "Can you give me one small optional idea for dealing with social anxiety?"
    dialogue = [{"speaker": "seeker", "content": current}]
    flags = effect_study_observable_flags(
        current_user_text=current,
        recent_user_text=current,
        visible_dialogue=dialogue,
    )
    assert flags["advice_welcome"] is True
    assert flags["low_burden"] is True
    ranked = repair_v2_rank_applicable_cards(
        query=current,
        current_user_text=current,
        recent_user_text=current,
        visible_dialogue=dialogue,
        cards=_strategy_cards(),
    )
    assert ranked
    assert ranked[0]["strategy_family"] == "Providing Suggestions"


def test_final_rs_reflection_request_resolves_recent_explicit_feeling() -> None:
    current = "Help me put this feeling into words."
    recent = "I'm still struggling to find inspiration. It's just so hard. " + current
    dialogue = [
        {
            "speaker": "seeker",
            "content": "I'm still struggling to find inspiration. It's just so hard.",
        },
        {"speaker": "seeker", "content": current},
    ]
    flags = effect_study_observable_flags(
        current_user_text=current,
        recent_user_text=recent,
        visible_dialogue=dialogue,
    )
    assert flags["reflection_welcome"] is True
    assert flags["emotion_visible"] is True
    ranked = repair_v2_rank_applicable_cards(
        query=recent,
        current_user_text=current,
        recent_user_text=recent,
        visible_dialogue=dialogue,
        cards=_strategy_cards(),
    )
    assert ranked
    assert ranked[0]["strategy_family"] == "Reflection of feelings"


def test_pre_pm_ranker_defers_applicability_but_keeps_state_hard_off() -> None:
    current = "I am overwhelmed. I only want you to listen; no advice."
    dialogue = [{"speaker": "seeker", "content": current}]
    ranked = rank_pre_pm_strategy_candidates(
        query=current,
        current_user_text=current,
        recent_user_text=current,
        visible_dialogue=dialogue,
        cards=_strategy_cards(),
    )
    assert ranked
    assert all(
        row["applicability_reasons"] == ["DEFERRED_TO_STEP1_PM"]
        for row in ranked
    )

    closing = "Thanks for listening."
    assert rank_pre_pm_strategy_candidates(
        query=closing,
        current_user_text=closing,
        recent_user_text=closing,
        visible_dialogue=[{"speaker": "seeker", "content": closing}],
        cards=_strategy_cards(),
    ) == []


def test_pre_pm_query_expansion_identifies_move_type_without_deciding_use() -> None:
    cases = {
        "Several parts feel tangled, and I cannot tell which one is driving the pressure.": "focused question",
        "The reaction is hard to name, and I want help finding accurate words for it.": "reflect",
        "I want something manageable and am open to advice.": "optional concrete microstep",
        "I need space to say what this is like before deciding what to do.": "open invitation",
        "The last focused question already identified it, so do not repeat it.": "focused question",
    }
    for current, expected in cases.items():
        expanded = pre_pm_strategy_retrieval_query(
            query="base query",
            current_user_text=current,
            recent_user_text=current,
        ).lower()
        assert expected in expanded
        assert "applicable" not in expanded
        assert "benefit" not in expanded


def test_pre_pm_family_selection_is_content_routing_not_on_off() -> None:
    assert (
        pre_pm_strategy_candidate_family(
            "Several parts feel tangled, and I cannot tell which one drives the pressure."
        )
        == "Question"
    )
    assert (
        pre_pm_strategy_candidate_family(
            "The reaction is hard to name, and I need accurate words for it."
        )
        == "Reflection of feelings"
    )
    assert (
        pre_pm_strategy_candidate_family(
            "I need space to say what this is like before deciding what to do."
        )
        == "Restatement or Paraphrasing"
    )
    # Redundant cases deliberately still retrieve the already-used family;
    # Step 1, not retrieval, must turn that candidate off.
    assert (
        pre_pm_strategy_candidate_family(
            "That focused question already identified it, so do not repeat it."
        )
        == "Question"
    )
