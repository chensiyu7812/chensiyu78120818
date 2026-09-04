from metacom_pm.v1_5_v5_2_atomic_memory import (
    compile_atomic_reusable_outcome,
    compile_atomic_session_observation,
    me_subtype_hints,
)


def test_rejects_grief_explanation_as_reusable_outcome() -> None:
    text = (
        "I've started having recurring dreams about my late dog. "
        "When I wake up, the loss hits all over again. "
        "I wonder if it's because I've been feeling stressed with work."
    )
    assert compile_atomic_reusable_outcome(text) is None


def test_rejects_unresolved_relationship_narrative() -> None:
    text = (
        "We've grown apart, and it's hard to rely on her. "
        "I have tried, but reconnecting feels like another task because I am exhausted."
    )
    assert compile_atomic_reusable_outcome(text) is None


def test_accepts_same_sentence_action_and_positive_result() -> None:
    item = compile_atomic_reusable_outcome(
        "I tried writing one short note first, and it helped me say what mattered."
    )
    assert item is not None
    assert item.polarity == "positive"
    assert item.sentence_count == 1
    assert item.literal_evidence_span.startswith("I tried")


def test_accepts_immediately_anaphoric_result() -> None:
    item = compile_atomic_reusable_outcome(
        "I paused before replying. That helped me avoid escalating the disagreement."
    )
    assert item is not None
    assert item.sentence_count == 2
    assert item.polarity == "positive"


def test_accepts_negative_result_as_avoidance_evidence() -> None:
    item = compile_atomic_reusable_outcome(
        "I tried making a long checklist, but it did not help me feel less overwhelmed."
    )
    assert item is not None
    assert item.polarity == "negative"


def test_session_observation_requires_specific_bounded_note() -> None:
    assert compile_atomic_session_observation("The seeker discussed the topic.") is None
    item = compile_atomic_session_observation(
        "The seeker felt overloaded after a shift change and wanted to separate sleep loss from workload."
    )
    assert item is not None
    assert "shift change" in item.literal_past_note


# 2026-08-06: real, manually-verified cases found while reverse-engineering
# why ME never fires on real EvoEmo data (see PM_V1_5_V5_3_ME_REGEX_RECALL_
# FIX_20260806_ZH.md) -- the original action-verb list and past-tense-only
# result matching rejected genuinely valid action+result reports.


def test_accepts_present_tense_anaphoric_result() -> None:
    item = compile_atomic_reusable_outcome(
        "I've also started journaling again. It helps sort through these complicated feelings."
    )
    assert item is not None
    assert item.polarity == "positive"


def test_accepts_verb_not_in_original_narrow_list() -> None:
    item = compile_atomic_reusable_outcome(
        "I joined a writing workshop, and it helped a bit."
    )
    assert item is not None


def test_accepts_coping_activity_as_sentence_subject() -> None:
    item = compile_atomic_reusable_outcome(
        "Meditation helped before, and I should get back to it regularly."
    )
    assert item is not None
    assert item.past_action_span == "Meditation"


def test_accepts_gerund_phrase_coping_activity_subject() -> None:
    item = compile_atomic_reusable_outcome(
        "Joining the peer tutoring club has helped a bit."
    )
    assert item is not None


def test_rejects_coping_activity_subject_result_about_third_party() -> None:
    assert compile_atomic_reusable_outcome("Meditation helped my sister a lot.") is None
    assert compile_atomic_reusable_outcome("Meditation helped him a lot.") is None


def test_rejects_purpose_clause_as_result() -> None:
    assert (
        compile_atomic_reusable_outcome(
            "I joined that tutoring club to help academically, but it hasn't been enough."
        )
        is None
    )


def test_rejects_helping_a_third_party_as_self_coping_outcome() -> None:
    assert (
        compile_atomic_reusable_outcome(
            "I tried to help, gave first aid until the ambulance came."
        )
        is None
    )


def test_rejects_hypothetical_result_as_observed() -> None:
    assert (
        compile_atomic_reusable_outcome("Talking to someone who understands might help.")
        is None
    )
    assert (
        compile_atomic_reusable_outcome(
            "I used to be part of one, and that sounds like it would help."
        )
        is None
    )


def test_me_subtype_hints_labels_valid_and_invalid_items() -> None:
    class _Item:
        def __init__(self, memory_id: str, text: str) -> None:
            self.memory_id = memory_id
            self.text = text

    items = [
        _Item("m1", "Meditation helped before, and I should get back to it regularly."),
        _Item("m2", "My dog died last week and I have been sad."),
    ]
    hints = me_subtype_hints(items)
    assert hints["m1"]["me_subtype_hint"] == "ME_REUSABLE_OUTCOME"
    assert hints["m2"]["me_subtype_hint"] == "ME_CONTEXT_EVENT"
