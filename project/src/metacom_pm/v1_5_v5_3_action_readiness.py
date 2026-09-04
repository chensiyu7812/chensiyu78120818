"""Narrow, transparent action-readiness observation for V5.3.

This is deliberately three-valued.  Absence of an explicit action cue is
UNKNOWN, not a negative label.  A current-turn refusal has precedence over
an invitation.  The observer is a low-capacity auditable primary feature;
it is not claimed to understand arbitrary user intent.

The additional invitation surfaces were developed only on already consumed
V3/V5.2 construction rows.  They therefore require a content-disjoint
qualification before formal V5.3 FIT, and must not be described as external
generalization evidence.
"""

from __future__ import annotations

from enum import Enum
import re


class ActionReadiness(str, Enum):
    INVITES_ACTION = "INVITES_ACTION"
    DECLINES_ACTION = "DECLINES_ACTION"
    UNKNOWN = "UNKNOWN"


_DECLINE_RE = re.compile(
    r"\b(?:do not|don't|dont)\s+(?:want|need)\s+(?:any\s+)?(?:advice|suggestions?)\b|"
    r"\b(?:just|only)\s+(?:want|need)(?:\s+you)?\s+to\s+listen\b|"
    r"\b(?:no|without)\s+(?:advice|suggestions?)\b|"
    r"\b(?:do not|don't|dont)\s+(?:offer|reuse|suggest)\s+(?:an?\s+)?(?:action|step|idea)\b|"
    r"\bonly\s+want\s+(?:the\s+)?(?:feeling|emotion)\s+(?:named|reflected)\b|"
    r"\bnot\s+ready\s+(?:for|to\s+consider)\s+(?:advice|an?\s+idea|a\s+step)\b",
    flags=re.IGNORECASE,
)


_INVITE_RE = re.compile(
    r"\bwhat\s+(?:do|should|can|could)\s+i\s+(?:do|try)\b|"
    r"\bhow\s+(?:do|can|should|could)\s+i\b|"
    r"\bany\s+(?:advice|suggestions?)\b|"
    r"\b(?:can|could|would)\s+you\s+(?:suggest|recommend)\b|"
    r"\b(?:can|could|would)\s+you\s+help\s+me\s+(?:decide|start|find|choose|figure)\b|"
    r"\bi(?:\s+am|(?:\s+would|'d)\s+be)\s+(?:open|ready|willing)\s+to\s+(?:consider|try)\b|"
    r"\b(?:am|'m)\s+open\s+to\s+(?:one|an?|some)\s+(?:small\s+|optional\s+|low[- ]pressure\s+|reversible\s+)*(?:idea|option|step|experiment|suggestion|approach)\b|"
    r"\b(?:need|want)\s+(?:a|one)\s+(?:manageable\s+place\s+to\s+begin|safer\s+alternative|small\s+reversible\s+experiment|low[- ]commitment\s+(?:step|option))\b|"
    r"\bwant\s+to\s+avoid\s+repeating\s+what\s+made\s+this\s+harder\s+before\b|"
    r"\b(?:help|work\s+with)\s+me\s+(?:think\s+of|find|choose)\s+(?:one|a)\s+(?:small\s+|optional\s+|safer\s+)*(?:step|idea|option|alternative|experiment)\b",
    flags=re.IGNORECASE,
)


def observe_action_readiness(current_user_text: str) -> ActionReadiness:
    text = " ".join(str(current_user_text or "").split())
    if _DECLINE_RE.search(text):
        return ActionReadiness.DECLINES_ACTION
    if _INVITE_RE.search(text):
        return ActionReadiness.INVITES_ACTION
    return ActionReadiness.UNKNOWN
