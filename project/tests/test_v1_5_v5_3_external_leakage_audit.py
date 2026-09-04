from __future__ import annotations

from metacom_pm.v1_5_v5_3_external_leakage_audit import (
    audit_evoemo_projection_and_qa,
    compare_text_surface_overlap,
    extract_internal_text_surfaces,
    project_evoemo_response_source,
    qa_generation_serialization,
    qa_gold_canary_is_invariant,
    qa_query_serialization,
    response_gold_canary_is_invariant,
    serialize_response_projection,
)


def _session(session_id: str, seeker_text: str) -> dict:
    return {
        "id": session_id,
        "timestamp": "2026-01-01",
        "summary": f"summary for {session_id}",
        "dialogue": [
            {"idx": 1, "role": "seeker", "content": seeker_text},
            {"idx": 2, "role": "supporter", "content": "Tell me more."},
        ],
        "observation": [],
    }


def _user(user_id: str, name: str, session_ids: tuple[str, ...]) -> dict:
    return {
        "id": user_id,
        "basic_info": {"name": name, "job": "teacher"},
        "dialog_history": [
            _session(session_id, f"Past turn {index} for {name}.")
            for index, session_id in enumerate(session_ids)
        ],
        "event_experience": [{"event": "forbidden response-side shortcut"}],
        "social_relationship": ["forbidden response-side shortcut"],
        "questions": [
            {
                "id": "facts",
                "questions": [
                    {
                        "idx": 1,
                        "capability": "information extraction",
                        "question": f"What happened to {name}?",
                        "answer": "sealed answer",
                        "evidence": ["sealed evidence"],
                    }
                ],
            }
        ],
        "summaries": [
            {
                "idx": 2,
                "capability": "temporal reasoning",
                "question": f"How did {name} change?",
                "answer": "sealed summary answer",
                "evidence": ["sealed summary evidence"],
                "theme": "sealed evaluator theme",
                "group": ["sealed evaluator group"],
            }
        ],
        "subsequent_topics": [{"topic": "future sealed topic"}],
    }


def test_response_projection_has_only_basic_info_and_strictly_prior_prefix() -> None:
    user = _user("u1", "Alex", ("s1", "s2", "s3"))
    projection = project_evoemo_response_source(
        user, strictly_prior_session_count=1
    )
    assert list(projection) == ["basic_info", "dialog_history"]
    assert [row["id"] for row in projection["dialog_history"]] == ["s1"]
    rendered = serialize_response_projection(projection)
    assert "s2" not in rendered and "s3" not in rendered
    assert "sealed answer" not in rendered
    assert "future sealed topic" not in rendered


def test_response_answer_and_evidence_canary_cannot_change_serialization() -> None:
    user = _user("u1", "Alex", ("s1", "s2"))
    assert response_gold_canary_is_invariant(
        user, strictly_prior_session_count=2
    )


def test_qa_answer_and_evidence_canary_cannot_change_query_or_generation() -> None:
    row = {
        "question": "What happened?",
        "answer": "gold answer",
        "evidence": ["gold evidence"],
        "capability": "gold capability",
    }
    documents = [
        {"session_id": "s1", "date": "2026-01-01", "text": "Alex: A past fact."}
    ]
    assert qa_gold_canary_is_invariant(qa_row=row, session_documents=documents)
    query = qa_query_serialization(question=row["question"])
    generation = qa_generation_serialization(
        question=row["question"], session_documents=documents
    )
    for forbidden in (row["answer"], row["evidence"][0], row["capability"]):
        assert forbidden not in query
        assert forbidden not in generation


def test_projection_audit_uses_composite_session_identity_not_bare_id() -> None:
    # EvoEmo itself contains a raw session id shared across two users. That is
    # not cross-user leakage when the owner is part of the identity.
    users = [
        _user("u1", "Alex", ("shared", "u1-only")),
        _user("u2", "Blair", ("shared", "u2-only")),
    ]
    report = audit_evoemo_projection_and_qa(users)
    assert report["raw_session_ids_shared_across_users"] == 1
    assert report["session_identity_rule"] == "(user_id, session_id), never bare session_id"
    assert report["response_current_or_future_session_exposures"] == 0
    assert report["response_cross_user_session_exposures"] == 0
    assert report["qa_cross_user_session_exposures"] == 0
    assert report["response_forbidden_gold_canary_failures"] == 0
    assert report["qa_answer_evidence_canary_failures"] == 0


def test_overlap_interface_detects_exact_and_normalized_ngram_without_raw_text() -> None:
    external = [
        {
            "surface_id": "external:q1",
            "category": "external_question",
            "text": "Where did the user spend the quiet winter weekend?",
        }
    ]
    internal = [
        {
            "surface_id": "internal:i1",
            "category": "internal_superdomain",
            "text": "Where did the user spend the quiet winter weekend?",
        },
        {
            "surface_id": "internal:i2",
            "category": "internal_superdomain",
            "text": "A copied fragment asks where did the USER spend the quiet winter weekend elsewhere.",
        },
    ]
    report = compare_text_surface_overlap(
        internal_surfaces=internal, external_surfaces=external, ngram_size=5
    )
    assert report["internal_surfaces_with_exact_overlap"] == 1
    assert report["internal_surfaces_with_normalized_ngram_overlap"] == 2
    assert report["raw_external_text_in_report"] is False
    rendered = str(report)
    assert external[0]["text"] not in rendered
    assert all("ngram_sha256" in row for row in report["normalized_ngram_collisions"])


def test_internal_surface_extractor_uses_only_explicit_model_visible_fields() -> None:
    rows = [
        {
            "state_id": "state-1",
            "current_user_text": "This is model-visible.",
            "answer": "This is evaluator-only.",
            "nested": {"candidate_text": "This candidate is model-visible."},
        }
    ]
    surfaces = extract_internal_text_surfaces(rows)
    assert {row["text"] for row in surfaces} == {
        "This is model-visible.",
        "This candidate is model-visible.",
    }
    assert all("evaluator-only" not in row["text"] for row in surfaces)
