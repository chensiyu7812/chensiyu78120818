import pytest

from metacom_pm.v1_5_v5_3_action_readiness import (
    ActionReadiness,
    observe_action_readiness,
)


@pytest.mark.parametrize(
    "text",
    [
        "I need a manageable place to begin and am open to one optional idea.",
        "I want to avoid repeating what made this harder before.",
        "I want one safer alternative to the approach that did not work before.",
        "I am open to one small reversible experiment that I can evaluate.",
        "I'd be willing to consider a small option, if you have one.",
        "Could you help me find one low-commitment step?",
    ],
)
def test_action_invitation_surfaces(text: str) -> None:
    assert observe_action_readiness(text) is ActionReadiness.INVITES_ACTION


@pytest.mark.parametrize(
    "text",
    [
        "I just need you to listen; I don't want any advice.",
        "Do not offer or reuse an action; I only want the feeling named.",
        "An idea may matter later, but I am not ready for advice now.",
    ],
)
def test_action_decline_has_precedence(text: str) -> None:
    assert observe_action_readiness(text) is ActionReadiness.DECLINES_ACTION


def test_unknown_is_not_collapsed_to_decline() -> None:
    assert (
        observe_action_readiness("This has been weighing on me all week.")
        is ActionReadiness.UNKNOWN
    )
