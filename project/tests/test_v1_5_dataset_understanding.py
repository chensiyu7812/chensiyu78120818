from __future__ import annotations

from metacom_pm.v1_5_dataset_understanding import (
    build_dataset_understanding_profile,
    profile_esconv,
)


def _esconv_row(strategy: str, feedback: str = "5"):
    return {
        "experience_type": "Current Experience",
        "emotion_type": "anxiety",
        "problem_type": "job crisis",
        "situation": "work is difficult",
        "survey_score": {
            "seeker": {
                "initial_emotion_intensity": "5",
                "final_emotion_intensity": "3",
                "empathy": "4",
                "relevance": "4",
            },
            "supporter": {"relevance": "4"},
        },
        "dialog": [
            {
                "speaker": "seeker",
                "content": "What should I do?",
                "annotation": {},
            },
            {
                "speaker": "supporter",
                "content": "What feels most urgent?",
                "annotation": {"strategy": strategy},
            },
            {
                "speaker": "seeker",
                "content": "The deadline.",
                "annotation": {"feedback": feedback},
            },
        ],
    }


def _evo_user():
    return {
        "id": "u1",
        "basic_info": {"name": "Alex"},
        "event_experience": [
            {
                "id": "event1",
                "date": "2024-01-01",
                "conv_id": "p1_conv_1",
                "event": "A change happened.",
                "influenced_by": [],
            }
        ],
        "social_relationship": [],
        "dialog_history": [
            {
                "id": "p1_conv_1",
                "timestamp": "2024-01-01",
                "emotion": "anxiety",
                "topic": "work",
                "summary": "Alex discussed work.",
                "dialogue": [
                    {"idx": 1, "role": "seeker", "content": "Work is hard."},
                    {"idx": 2, "role": "supporter", "content": "Tell me more."},
                ],
                "observation": [],
            }
        ],
        "questions": [
            {
                "id": "facts",
                "questions": [
                    {
                        "capability": "information extraction",
                        "question": "sealed",
                        "answer": "sealed",
                        "evidence": ["event1"],
                        "idx": 1,
                    }
                ],
            }
        ],
        "summaries": [],
        "subsequent_topics": [
            {
                "idx": 1,
                "topic": "sealed",
                "more_details": "sealed",
                "physical_condition": "sealed",
                "psychological_condition": "sealed",
                "related_sessions": ["p1_conv_1"],
            }
        ],
    }


def test_esconv_strategy_analysis_excludes_nontrain_and_overlap():
    rows = [_esconv_row("Question"), _esconv_row("Providing Suggestions")]
    split = [
        {
            "dialogue_id": "esconv_0000",
            "index": 0,
            "split": "train",
            "excluded_for_evoemo_overlap": False,
        },
        {
            "dialogue_id": "esconv_0001",
            "index": 1,
            "split": "test",
            "excluded_for_evoemo_overlap": False,
        },
    ]
    profile = profile_esconv(rows, split)
    assert profile["train_observed_strategy_decisions"] == 1
    assert profile["strategy_counts"] == {"Question": 1}
    assert profile["analysis_scope"][
        "validation_and_test_strategy_annotations_opened"
    ] is False


def test_profile_is_deterministic_and_keeps_evaluator_text_out_of_output():
    esconv = [_esconv_row("Question")]
    split = [
        {
            "dialogue_id": "esconv_0000",
            "index": 0,
            "split": "train",
            "excluded_for_evoemo_overlap": False,
        }
    ]
    kwargs = {
        "esconv": esconv,
        "split_rows": split,
        "evoemo_users": [_evo_user()],
        "source_hashes": {"a": "0" * 64},
    }
    first = build_dataset_understanding_profile(**kwargs)
    second = build_dataset_understanding_profile(**kwargs)
    assert first == second
    assert first["profile_sha256"] == second["profile_sha256"]
    rendered = str(first)
    assert "A change happened." not in rendered
    assert "'question': 'sealed'" not in rendered
    assert "'answer': 'sealed'" not in rendered
    assert first["evoemo"]["deployable_memory_catalog"]["total_items"] == {
        "ME": 1,
        "MP": 1,
        "MS": 1,
    }
