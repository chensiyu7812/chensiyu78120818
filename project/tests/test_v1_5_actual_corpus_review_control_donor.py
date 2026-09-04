"""Regression tests for the context_grounding_match control-donor fix.

A negative control for context_grounding_match works by swapping in a
donor's session_summary from a different semantic family. Found for real:
a donor whose session_summary happens to be empty produces a control that
is no longer distinguishable from a legitimate N/A summary (the
audit-contract fix correctly never treats an empty summary as unsupported
evidence), so it stopped being an effective negative control. Fixed by
requiring the donor's session_summary to be non-empty.
"""

from __future__ import annotations

import pytest

from metacom_pm.v1_5_actual_corpus_review import _corrupt_actual_payload


def _item(semantic_family: str, session_summary: str) -> dict:
    return {
        "payload": {
            "semantic_family": semantic_family,
            "session_summary": session_summary,
        }
    }


def test_context_grounding_control_donor_summary_is_never_empty() -> None:
    target = _item("relationship_uncertainty_target", "original summary text")
    donors = [
        _item("other_family", ""),  # different family, differs, but empty
        _item("other_family_2", "a real substantive donor summary"),
    ]
    _, override = _corrupt_actual_payload(
        target, field="context_grounding_match", donors=donors
    )
    assert override["session_summary"] == "a real substantive donor summary"


def test_context_grounding_control_fails_closed_with_no_eligible_donor() -> None:
    target = _item("family_a", "original summary text")
    donors = [
        _item("other_family", ""),  # only candidate is empty -- ineligible
    ]
    with pytest.raises(StopIteration):
        _corrupt_actual_payload(target, field="context_grounding_match", donors=donors)
