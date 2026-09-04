from metacom_pm.contracts import MemoryItem, MemorySource
from metacom_pm.v1_5_d3_features import ms_incremental_value_observation


def _item(text: str) -> MemoryItem:
    return MemoryItem(
        memory_id="mem_0123456789abcdef",
        source=MemorySource.MS,
        created_session=2,
        text=text,
    )


def test_ms_incremental_value_requires_new_prior_outcome_and_current_use() -> None:
    useful = ms_incremental_value_observation(
        selected_items=[
            _item("Choosing one ten-minute priority review helped reduce the deadline pressure.")
        ],
        current_user_text=(
            "The deadline problem is here again. Could you offer one small optional suggestion?"
        ),
        visible_dialogue=[
            {"speaker": "seeker", "content": "Several deadlines are competing."},
            {"speaker": "supporter", "content": "Tell me what matters today."},
        ],
    )
    redundant = ms_incremental_value_observation(
        selected_items=[_item("Several deadlines are competing.")],
        current_user_text=(
            "The deadline problem is here again. Could you offer one small optional suggestion?"
        ),
        visible_dialogue=[
            {"speaker": "seeker", "content": "Several deadlines are competing."},
            {"speaker": "supporter", "content": "Tell me what matters today."},
        ],
    )
    assert useful["candidate_incremental_alignment_score"] == 1.0
    assert redundant["candidate_incremental_alignment_score"] == 0.0
    assert useful["outcome_read"] is False
    assert useful["not_an_effect_label"] is True


def test_ms_incremental_value_is_invariant_to_supporter_text() -> None:
    kwargs = {
        "selected_items": [
            _item("A smaller study target worked and made it easier to begin.")
        ],
        "current_user_text": (
            "Exam preparation is difficult again. What helped last time?"
        ),
    }
    first = ms_incremental_value_observation(
        **kwargs,
        visible_dialogue=[
            {"speaker": "seeker", "content": "My concentration drops after work."},
            {"speaker": "supporter", "content": "You already shared a prior summary."},
        ],
    )
    second = ms_incremental_value_observation(
        **kwargs,
        visible_dialogue=[
            {"speaker": "seeker", "content": "My concentration drops after work."},
            {"speaker": "supporter", "content": "A completely unrelated sentinel."},
        ],
    )
    assert first == second
