"""Candidate Strategy RAG V4 helpers for direct R0/RS effect testing.

V4 keeps the fifty outcome-blind V3 core submoves and gives each one two
execution profiles:

* ``minimal`` executes only the atomic move and is suitable for low-burden
  interaction;
* ``dialogic`` integrates the same move naturally while still forbidding
  stacked questions, advice, or tasks.

The resulting one-hundred-card catalog is a development candidate, not a
formal Strategy Bank.  Its first role is to test whether a realistically
retrieved technique can improve the same generator over R0.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re
from typing import Any

from .io import canonical_json, stable_hex
from .text import lexical_score, normalize_space
from .v1_5_strategy_bank_v3 import EXPECTED_FAMILIES


V4_PROTOCOL = "pm-v1.5-strategy-rag-v4-100-card-candidate-v1"
EXECUTION_PROFILES = ("minimal", "dialogic")

_ACTIVE_HIGH_STAKES_RE = re.compile(
    r"\b(?:suicid(?:e|al)|self[- ]?harm|"
    r"(?:feel like|want to|ready to|going to)"
    r"(?:\s+\w+){0,4}\s+giv(?:e|ing)\s+up|"
    r"i\s+can(?:not|'t)\s+go\s+on|"
    r"(?:want|plan|going|might|may|could|feel like|thinking about)"
    r"(?:\s+\w+){0,3}\s+(?:kill|hurt|harm)(?:ing)?"
    r"(?:\s+\w+){0,2}\s+(?:myself|someone|him|her|them)|"
    r"(?:immediate|in|real)\s+danger|"
    r"(?:partner|spouse|boyfriend|girlfriend|parent)"
    r"(?:\s+\w+){0,4}\s+(?:hit|beat|threaten|abuse)(?:s|d|ing)?)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_STOP_RE = re.compile(
    r"(?:\b(?:please\s+)?(?:stop|end|close|quit)"
    r"(?:\s+(?:this|the))?\s+(?:chat|conversation|talking|survey)\b|"
    r"^\s*(?:bye|goodbye|good-bye)\W*$)",
    flags=re.IGNORECASE,
)
_LISTEN_ONLY_RE = re.compile(
    r"\b(?:just|only)\s+(?:want|need)(?:\s+you)?\s+to\s+listen\b|"
    r"\b(?:do not|don't|dont)\s+(?:want|need)\s+(?:any\s+)?advice\b|"
    r"\bno\s+advice\b",
    flags=re.IGNORECASE,
)
_LOW_BURDEN_RE = re.compile(
    r"\b(?:one|single|small|simple|brief|quick|manageable|low[- ]pressure)"
    r"\s+(?:optional\s+)?(?:focused\s+|short\s+|gentle\s+)?"
    r"(?:idea|suggestion|step|question|thing)\b|"
    r"\b(?:one thing at a time|keep it (?:brief|simple)|"
    r"not (?:too much|a long list)|overwhelmed|exhausted)\b",
    flags=re.IGNORECASE,
)
_ADVICE_WELCOME_RE = re.compile(
    r"(?:\bwhat\s+(?:do|should|can|could)\s+i\s+(?:do|try)\b|"
    r"\bwhat\s+else\s+(?:(?:can|could|should)\s+i\s+|to\s+)?"
    r"(?:do|try)\b|"
    r"\bwhat\s+else\s+i\s+(?:can|could|should)\s+(?:do|try)\b|"
    r"\bhow\s+(?:do|can|should|could)\s+i\b|"
    r"\bi\s+(?:do not|don't|dont)\s+know\s+how\s+to\b|"
    r"\bany\s+(?:advice|tips?|ideas?|suggestions?)\b|"
    r"\b(?:can|could|would)\s+you\s+(?:suggest|recommend)\b|"
    r"\b(?:can|could|would)\s+you\s+give\s+me\s+"
    r"(?:one|a|some|any)?\s*(?:small|simple|brief)?\s*"
    r"(?:optional\s+)?"
    r"(?:suggestion|idea|tip|step|advice)\b|"
    r"\bwhat\s+would\s+you\s+(?:do|suggest|recommend)\b)",
    flags=re.IGNORECASE,
)
_REFLECTION_WELCOME_RE = re.compile(
    r"\b(?:help|can|could|would)\s+(?:me\s+)?(?:to\s+)?"
    r"(?:put|name|describe|express)\s+(?:this|that|the|my)?\s*"
    r"(?:feeling|emotion|experience)\s+(?:into\s+words|in\s+words)?\b|"
    r"\bhelp\s+me\s+(?:say|name)\s+(?:what|how)\s+i\s+feel\b",
    flags=re.IGNORECASE,
)
_EMOTION_WORD = (
    r"(?:sad|angry|upset|hurt|afraid|scared|anxious|worried|nervous|"
    r"overwhelmed|exhausted|tired|lonely|frustrated|confused|embarrassed|"
    r"ashamed|guilty|disappointed|stressed|depressed|discouraged|lost|"
    r"hopeless|uncertain|unsure)"
)
_EMOTION_RE = re.compile(
    rf"(?:\b(?:i\s+(?:feel|felt|am|was|have been)|i['’]?m|im|"
    rf"feeling|makes?\s+me\s+feel)\s+(?:quite\s+|very\s+|so\s+|"
    rf"really\s+|a\s+bit\s+|pretty\s+|emotionally\s+)*{_EMOTION_WORD}\b|"
    rf"\b(?:getting|became|becoming)\s+(?:quite\s+|very\s+|so\s+|"
    rf"really\s+)*{_EMOTION_WORD}\b|"
    rf"\b(?:very|so|really|quite)\s+{_EMOTION_WORD}\b|"
    r"\b(?:this|that|it)\s+(?:is|feels|was)\s+"
    r"(?:painful|difficult|hard)\b|"
    r"\bit['’]?s\s+(?:just\s+)?(?:so\s+)?(?:painful|difficult|hard)\b|"
    r"\bi(?:['’]?m|\s+am)\s+(?:still\s+)?struggling\b)",
    flags=re.IGNORECASE,
)
_EFFORT_OR_PROGRESS_RE = re.compile(
    r"\b(?:i (?:tried|have tried|managed|started|kept|continued|decided|"
    r"reached out|spoke|talked|set|finished|made progress)|"
    r"i(?:'m| am) trying|i was able to|i finally|my first step|"
    r"getting better|improved|made it through)\b",
    flags=re.IGNORECASE,
)
_UNCERTAINTY_OR_MULTI_CONCERN_RE = re.compile(
    r"\b(?:not sure|don't know|dont know|uncertain|confused|"
    r"where to start|how to start|part of me|on the one hand|"
    r"several things|so many things|everything at once)\b",
    flags=re.IGNORECASE,
)
_PURE_PHATIC_RE = re.compile(
    r"^\s*(?:h+e+l+o+|hi+|hey+|thanks?|thank you(?: so much)?|"
    r"thanks?\s+for\s+listening|"
    r"bye|goodbye|good-bye|good ?night|have a good (?:day|night)|"
    r"you(?:'re| are) welcome)[\s!.?,:;'\"]*$",
    flags=re.IGNORECASE,
)
_ROUTINE_CLOSING_RE = re.compile(
    r"(?:\b(?:i\s+)?(?:hope|wish)(?:\s+you)?(?:\s+have)?\s+"
    r"(?:a\s+)?(?:great|good|wonderful|happy|safe|peaceful)\s+"
    r"(?:day|night|evening|weekend|holiday|christmas|new year)\b|"
    r"\bstay\s+safe\b|"
    r"\btake\s+care(?:\s+of\s+yourself)?\b)",
    flags=re.IGNORECASE,
)
_FACTUAL_IDEA_QUESTION_RE = re.compile(
    r"\bany\s+ideas?\s+how\s+(?:it|this|that|they|would|does|do|"
    r"is|are|can|could)\b",
    flags=re.IGNORECASE,
)
_OPEN_FOCUS_RE = re.compile(
    r"\b(?:do not|don't|dont)\s+know\s+(?:where|how)\s+to\s+"
    r"(?:start|begin)\b|"
    r"\b(?:not sure|unsure)\s+(?:where|how)\s+to\s+(?:start|begin)\b|"
    r"\b(?:want|need|would like)\s+to\s+talk\b|"
    r"\bsomething\s+(?:is\s+)?bothering\s+me\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_GOAL_RE = re.compile(
    r"\b(?:my\s+goal|i\s+(?:want|need|hope|plan|intend|would like|"
    r"am trying|i'm trying|im trying)\s+to|"
    r"trying\s+to|decided\s+to|think\s+i\s+will|"
    r"what\s+i\s+want|the\s+main\s+thing)\b",
    flags=re.IGNORECASE,
)
_PERSISTENCE_RE = re.compile(
    r"\b(?:still|ongoing|keeps?|continues?|persist(?:s|ed|ing)?|"
    r"lately|recently|for\s+(?:days?|weeks?|months?|years?|a while)|"
    r"all\s+(?:the\s+)?time|every\s+(?:day|night|week)|"
    r"have\s+been|has\s+been)\b",
    flags=re.IGNORECASE,
)
_MIXED_OR_TWO_SIDES_RE = re.compile(
    r"\bpart\s+of\s+me\b|"
    r"\bon\s+the\s+one\s+hand\b|"
    r"\bmixed\s+(?:feelings?|emotions?)\b|"
    r"\b(?:want|hope|like|love|care|need|prefer|feel|think|know)"
    r"[^.!?]{1,90}\b(?:but|though|yet|however)\b[^.!?]{1,110}",
    flags=re.IGNORECASE,
)
_VALUE_TENSION_RE = re.compile(
    r"\b(?:responsib(?:le|ility)|important|matter(?:s|ed)?|value|"
    r"care\s+about|have\s+to|need\s+to|want\s+to|trying\s+to|"
    r"should)\b[^.!?]{0,100}\b(?:but|though|yet|however|while)\b|"
    r"\b(?:but|though|yet|however|while)\b[^.!?]{0,100}"
    r"\b(?:responsib(?:le|ility)|important|matter(?:s|ed)?|value|"
    r"care\s+about|have\s+to|need\s+to|want\s+to|trying\s+to)\b",
    flags=re.IGNORECASE,
)
_IMPACT_RE = re.compile(
    r"\b(?:affect(?:s|ed|ing)?|impact(?:s|ed|ing)?|interfere|"
    r"sleep|work|school|study|appetite|concentrat|focus|function|"
    r"unable|can(?:not|'t)|could(?:not|n't)|hard\s+to\s+"
    r"(?:sleep|work|study|focus|function)|through\s+the\s+roof)\b",
    flags=re.IGNORECASE,
)
_PRIORITY_RE = re.compile(
    r"\b(?:biggest|main|primary|most\s+(?:important|pressing)|"
    r"priority|especially|right\s+now|at\s+the\s+moment)\b",
    flags=re.IGNORECASE,
)
_PAST_COPING_RE = re.compile(
    r"\b(?:coping|cope|already\s+tried|have\s+tried|i\s+tried|"
    r"used\s+to|what\s+helped|did(?:n't| not)\s+help|"
    r"not\s+working|reached\s+out|talked\s+to|spoke\s+to)\b",
    flags=re.IGNORECASE,
)
_OVERLOAD_RE = re.compile(
    r"\b(?:overwhelmed|exhausted|too\s+much|can't\s+handle|"
    r"cannot\s+handle|at\s+my\s+limit|need\s+a\s+break|"
    r"not\s+ready|low\s+energy|drained)\b",
    flags=re.IGNORECASE,
)
_SUPPORT_PERSON_RE = re.compile(
    r"\b(?:trusted\s+(?:person|friend)|friend|family|sister|brother|"
    r"mother|father|mom|dad|partner|spouse|coworker|colleague|"
    r"roommate|teacher|professor|counselor|therapist)\b",
    flags=re.IGNORECASE,
)
_COMMUNICATION_GOAL_RE = re.compile(
    r"\b(?:talk|speak|tell|ask|message|text|email|explain|discuss|"
    r"conversation|communicat|bring\s+(?:it|this)\s+up|"
    r"say\s+to)\b",
    flags=re.IGNORECASE,
)
_ENVIRONMENT_OR_ROUTINE_RE = re.compile(
    r"\b(?:sleep|bed|night|routine|schedule|environment|room|desk|"
    r"workspace|phone|screen|social\s+media|notification|"
    r"distraction|noise|light|break|calendar|reminder|focus)\b",
    flags=re.IGNORECASE,
)
_MULTIPLE_TASKS_RE = re.compile(
    r"\b(?:everything\s+at\s+once|so\s+many|several|multiple|"
    r"where\s+to\s+start|one\s+thing\s+at\s+a\s+time|"
    r"work[^.!?]{0,50}school|school[^.!?]{0,50}work)\b",
    flags=re.IGNORECASE,
)
_AGENCY_OR_CHOICE_RE = re.compile(
    r"\b(?:decid|choice|option|prefer|consider|weigh|think\s+i\s+will|"
    r"i\s+will|i'll|plan\s+to|leaning\s+toward)\b",
    flags=re.IGNORECASE,
)
_SERIOUS_PERSISTENT_SUPPORT_RE = re.compile(
    r"\b(?:therapist|counselor|counselling|counseling|professional|"
    r"doctor|depress(?:ed|ion)|anxi(?:ous|ety)|panic|trauma|"
    r"for\s+(?:months?|years?)|all\s+the\s+time)\b",
    flags=re.IGNORECASE,
)
_DOMAIN_DECISION_RE = re.compile(
    r"\b(?:medical|medicine|medication|dose|diagnos|legal|lawyer|"
    r"court|police|cops?|press\s+charges?|tax|investment|loan|"
    r"emergency|weapon|violence|divorc|separat|break\s*up|"
    r"end\s+(?:this|the)\s+(?:relationship|situation)|"
    r"leave\s+(?:my\s+)?(?:partner|spouse|husband|wife|boyfriend|"
    r"girlfriend))\b",
    flags=re.IGNORECASE,
)
_POSITIVE_EVIDENCE_RE = re.compile(
    r"\b(?:managed|made\s+progress|improved|getting\s+better|"
    r"was\s+able|i\s+finally|decided|plan(?:ned)?\s+to|"
    r"going\s+to\s+try|will\s+try|maybe\s+i(?:'ll| will)|"
    r"gives?\s+me\s+hope|there\s+is\s+(?:a\s+)?(?:chance|option|"
    r"possibility)|support(?:ive|ed)?)\b",
    flags=re.IGNORECASE,
)


def _profile_guidance(family: str, profile: str) -> str:
    if profile == "minimal":
        if family == "Question":
            return (
                "Ask exactly one short question as the only main move. "
                "Do not add advice, a second question, or another task."
            )
        return (
            "Execute only this support move in one concise sentence. "
            "Do not append a question, advice, a second move, or another task."
        )
    if family == "Question":
        return (
            "Use at most one brief acknowledgment before exactly one focused "
            "question. Do not add advice or a second question."
        )
    if family == "Restatement or Paraphrasing":
        return (
            "Integrate the paraphrase naturally and leave room for correction "
            "only when ambiguity matters. Do not append advice or a new probe."
        )
    if family == "Reflection of feelings":
        return (
            "Reflect the feeling naturally and leave conversational space. "
            "Do not turn the reflection into advice or a follow-up question."
        )
    if family == "Affirmation and Reassurance":
        return (
            "Tie the acknowledgment to visible evidence and preserve the "
            "user's agency. Do not add generic praise, promises, or advice."
        )
    if family == "Providing Suggestions":
        return (
            "Offer one optional suggestion and at most one brief rationale "
            "linked to the user's stated goal. Make refusal easy and do not "
            "add a second task."
        )
    raise ValueError(f"unsupported family: {family}")


def build_v4_candidate_cards(
    core_cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand fifty V3 core cards into one hundred execution candidates."""

    rows = [dict(row) for row in core_cards]
    if len(rows) != 50:
        raise ValueError("V4 requires exactly fifty V3 core cards")
    family_counts = Counter(str(row["strategy_family"]) for row in rows)
    if set(family_counts) != set(EXPECTED_FAMILIES):
        raise ValueError("V4 family set differs from the five safe families")
    if any(family_counts[family] != 10 for family in EXPECTED_FAMILIES):
        raise ValueError("V4 requires ten core submoves per family")

    cards: list[dict[str, Any]] = []
    for core in sorted(rows, key=lambda row: str(row["submove_id"])):
        family = str(core["strategy_family"])
        for profile in EXECUTION_PROFILES:
            profile_use = (
                "Prefer this profile when the user requests low burden, one "
                "point, or a concise response."
                if profile == "minimal"
                else (
                    "Use this profile when the user is actively engaging and "
                    "has not requested a minimal or one-point response."
                )
            )
            profile_not = (
                "Do not use this profile to add any second support action."
                if profile == "minimal"
                else (
                    "Do not use under an explicit low-burden, one-point, "
                    "listen-only, no-probing, pause, or stop boundary."
                )
            )
            card_id = "strategy_v4_" + stable_hex(
                V4_PROTOCOL,
                str(core["card_id"]),
                profile,
                n=24,
            )
            support_move = str(core["support_move"])
            when_to_use = f"{core['when_to_use']} {profile_use}"
            when_not_to_use = f"{core['when_not_to_use']} {profile_not}"
            profile_guidance = _profile_guidance(family, profile)
            cards.append(
                {
                    "protocol": V4_PROTOCOL,
                    "card_id": card_id,
                    "core_card_id": str(core["card_id"]),
                    "core_submove_id": str(core["submove_id"]),
                    "execution_profile": profile,
                    "strategy_family": family,
                    "support_move": support_move,
                    "when_to_use": when_to_use,
                    "when_not_to_use": when_not_to_use,
                    "compatible_support_modes": list(
                        core["compatible_support_modes"]
                    ),
                    "compatible_dialogue_phases": list(
                        core["compatible_dialogue_phases"]
                    ),
                    "goal_types": list(core["goal_types"]),
                    "directive_burden": str(core["directive_burden"]),
                    "risk_flags": list(core["risk_flags"]),
                    "content_scope": "technique_only",
                    "retrieval_text": " ".join(
                        (
                            family,
                            support_move,
                            when_to_use,
                            "Goals:",
                            ", ".join(core["goal_types"]),
                        )
                    ),
                    "prompt_guidance": " ".join(
                        (
                            support_move,
                            profile_guidance,
                            str(core["when_not_to_use"]),
                            "Use only information visible in the current prompt.",
                        )
                    ),
                    "source_support": dict(core["source_support"]),
                    "variant_rationale": profile_use,
                    "quality_status": (
                        "DEVELOPMENT_CANDIDATE_PENDING_RETRIEVAL_AND_"
                        "DIRECT_EFFECT_QUALIFICATION"
                    ),
                    "eligible_for_formal_rs": False,
                    "raw_source_response_exposed_to_generator": False,
                }
            )
    validate_v4_candidate_cards(cards)
    return cards


def validate_v4_candidate_cards(
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in cards]
    if len(rows) != 100:
        raise ValueError("V4 candidate Bank must contain exactly 100 cards")
    ids = [str(row.get("card_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("V4 card ids must be non-empty and unique")
    core_profile = [
        (str(row["core_submove_id"]), str(row["execution_profile"]))
        for row in rows
    ]
    if len(core_profile) != len(set(core_profile)):
        raise ValueError("V4 repeats a core-submove execution profile")
    if Counter(profile for _, profile in core_profile) != {
        "minimal": 50,
        "dialogic": 50,
    }:
        raise ValueError("V4 requires fifty cards per execution profile")
    for row in rows:
        if row["strategy_family"] not in EXPECTED_FAMILIES:
            raise ValueError("V4 card has an unsupported family")
        if row["content_scope"] != "technique_only":
            raise ValueError("V4 forbids domain-information cards")
        if row["raw_source_response_exposed_to_generator"]:
            raise ValueError("V4 forbids raw source responses in prompts")
    return rows


def observable_opportunity_flags(
    *,
    current_user_text: str,
    recent_user_text: str | None = None,
) -> dict[str, bool]:
    latest = normalize_space(current_user_text)
    recent = normalize_space(recent_user_text or latest)
    routine_closing = bool(_ROUTINE_CLOSING_RE.search(latest))
    pure_phatic = bool(_PURE_PHATIC_RE.fullmatch(latest)) or routine_closing
    listen_only = bool(_LISTEN_ONLY_RE.search(latest))
    advice_welcome = (
        bool(_ADVICE_WELCOME_RE.search(latest))
        and not bool(_FACTUAL_IDEA_QUESTION_RE.search(latest))
        and not listen_only
    )
    reflection_welcome = bool(_REFLECTION_WELCOME_RE.search(latest))
    # A direct request to put a feeling into words may resolve an explicit
    # emotion from the immediately visible seeker context.  This is kept
    # narrow: without the direct request, old emotion words do not silently
    # open the reflection family for the current turn.
    emotion_visible = bool(_EMOTION_RE.search(latest)) or (
        reflection_welcome and bool(_EMOTION_RE.search(recent))
    )
    return {
        "pure_phatic": pure_phatic,
        "routine_closing": routine_closing,
        "active_high_stakes": bool(_ACTIVE_HIGH_STAKES_RE.search(recent)),
        "explicit_stop": bool(_EXPLICIT_STOP_RE.search(latest)),
        "listen_only": listen_only,
        "low_burden": bool(_LOW_BURDEN_RE.search(latest)) or listen_only,
        "advice_welcome": advice_welcome,
        "reflection_welcome": reflection_welcome,
        "emotion_visible": emotion_visible,
        "effort_or_progress_visible": bool(
            _EFFORT_OR_PROGRESS_RE.search(latest)
        ),
        "uncertainty_or_multi_concern": bool(
            _UNCERTAINTY_OR_MULTI_CONCERN_RE.search(latest)
        ),
        "substantive": len(latest.split()) >= 4 and not pure_phatic,
    }


def eligible_families(flags: Mapping[str, bool]) -> tuple[str, ...]:
    """Return broad safe candidate families before semantic ranking."""

    if (
        flags["pure_phatic"]
        or flags["active_high_stakes"]
        or flags["explicit_stop"]
        or not flags["substantive"]
    ):
        return ()
    families: list[str] = []
    if flags["uncertainty_or_multi_concern"] and not flags["listen_only"]:
        families.append("Question")
    if flags["substantive"]:
        families.append("Restatement or Paraphrasing")
    if flags["emotion_visible"] or flags.get("reflection_welcome", False):
        families.append("Reflection of feelings")
    if flags["emotion_visible"] or flags["effort_or_progress_visible"]:
        families.append("Affirmation and Reassurance")
    if flags["advice_welcome"]:
        families.append("Providing Suggestions")
    return tuple(family for family in EXPECTED_FAMILIES if family in families)


def selected_execution_profile(flags: Mapping[str, bool]) -> str:
    return "minimal" if flags["low_burden"] or flags["listen_only"] else "dialogic"


def observable_card_cues(
    *,
    current_user_text: str,
    recent_user_text: str | None = None,
    flags: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    """Extract explicit, reviewable cues used by card-level applicability.

    These cues do not predict response benefit and do not inspect any hidden
    next response.  They only encode prerequisites already written in each
    card's ``when_to_use`` / ``when_not_to_use`` fields.
    """

    latest = normalize_space(current_user_text)
    recent = normalize_space(recent_user_text or latest)
    base = dict(
        flags
        or observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=recent,
        )
    )
    explicit_goal = bool(_EXPLICIT_GOAL_RE.search(recent))
    mixed = bool(_MIXED_OR_TWO_SIDES_RE.search(recent))
    value_tension = bool(_VALUE_TENSION_RE.search(recent))
    return {
        "open_focus": bool(_OPEN_FOCUS_RE.search(latest)),
        "specific_focus_visible": (
            len(latest.split()) >= 7
            and not bool(_OPEN_FOCUS_RE.search(latest))
        ),
        "explicit_goal": explicit_goal,
        "persistence_visible": bool(_PERSISTENCE_RE.search(recent)),
        "mixed_or_two_sides_visible": mixed,
        "value_tension_visible": value_tension,
        "impact_visible": bool(_IMPACT_RE.search(recent)),
        "priority_visible": bool(_PRIORITY_RE.search(recent)),
        "past_coping_visible": bool(_PAST_COPING_RE.search(recent)),
        "overload_visible": bool(_OVERLOAD_RE.search(recent)),
        "support_person_visible": bool(_SUPPORT_PERSON_RE.search(recent)),
        "communication_goal_visible": bool(
            _COMMUNICATION_GOAL_RE.search(recent)
        ),
        "environment_or_routine_visible": bool(
            _ENVIRONMENT_OR_ROUTINE_RE.search(recent)
        ),
        "multiple_tasks_visible": bool(_MULTIPLE_TASKS_RE.search(recent)),
        "agency_or_choice_visible": bool(_AGENCY_OR_CHOICE_RE.search(recent)),
        "serious_persistent_support_visible": bool(
            _SERIOUS_PERSISTENT_SUPPORT_RE.search(recent)
        ),
        "domain_decision_visible": bool(_DOMAIN_DECISION_RE.search(recent)),
        "positive_evidence_visible": bool(_POSITIVE_EVIDENCE_RE.search(latest)),
        "emotion_visible": bool(base.get("emotion_visible")),
        "uncertainty_visible": bool(
            base.get("uncertainty_or_multi_concern")
        ),
        "effort_or_progress_visible": bool(
            base.get("effort_or_progress_visible")
        ),
        "advice_welcome": bool(base.get("advice_welcome")),
        "reflection_welcome": bool(base.get("reflection_welcome")),
        "listen_only": bool(base.get("listen_only")),
        "low_burden": bool(base.get("low_burden")),
    }


def assess_v4_card_applicability(
    card: Mapping[str, Any],
    *,
    current_user_text: str,
    recent_user_text: str | None = None,
    flags: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Apply transparent card-level prerequisites before semantic ranking.

    ``compatibility_tier`` is a coarse, inspectable ordering signal:
    3 means a card-specific visible cue is present, 2 means a safe family
    fallback, and 0 means the card is inapplicable.  Lexical cosine remains a
    tie-breaker inside a tier; it is never treated as a safety probability.
    """

    latest = normalize_space(current_user_text)
    recent = normalize_space(recent_user_text or latest)
    base = dict(
        flags
        or observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=recent,
        )
    )
    cues = observable_card_cues(
        current_user_text=latest,
        recent_user_text=recent,
        flags=base,
    )
    family = str(card["strategy_family"])
    core = str(card["core_submove_id"])
    profile = str(card["execution_profile"])
    reasons: list[str] = []

    if (
        base.get("pure_phatic")
        or base.get("active_high_stakes")
        or base.get("explicit_stop")
        or not base.get("substantive")
    ):
        reasons.append("global_hard_off")
    if profile == "dialogic" and (
        base.get("low_burden") or base.get("listen_only")
    ):
        reasons.append("dialogic_profile_conflicts_with_low_burden")
    if family == "Question" and base.get("listen_only"):
        reasons.append("question_conflicts_with_listen_only")
    if family == "Providing Suggestions" and not base.get("advice_welcome"):
        reasons.append("suggestion_without_visible_permission")
    if family == "Providing Suggestions" and cues["domain_decision_visible"]:
        reasons.append("domain_decision_outside_technique_bank")
    if reasons:
        return {
            "eligible": False,
            "compatibility_tier": 0,
            "reasons": reasons,
            "observable_card_cues": cues,
        }

    tier = 0
    if core == "question_open_invitation":
        tier = 3 if cues["open_focus"] and not cues["specific_focus_visible"] else 0
    elif core == "question_clarify_feeling":
        tier = 3 if not cues["emotion_visible"] and cues["specific_focus_visible"] else 0
    elif core == "question_clarify_goal":
        tier = (
            3
            if not cues["explicit_goal"]
            and not cues["advice_welcome"]
            and cues["specific_focus_visible"]
            else 0
        )
    elif core == "question_clarify_impact":
        tier = 3 if not cues["impact_visible"] else 0
    elif core == "question_clarify_meaning":
        tier = 3 if cues["uncertainty_visible"] or cues["value_tension_visible"] else 0
    elif core == "question_explore_past_coping":
        tier = 3 if cues["persistence_visible"] and not cues["past_coping_visible"] else 0
    elif core == "reflection_explicit_emotion":
        tier = 3 if cues["emotion_visible"] else 0
    elif core == "reflection_tentative_implicit_emotion":
        tier = (
            2
            if cues["reflection_welcome"] and cues["emotion_visible"]
            else 0
        )
    elif core == "reflection_lingering_feeling":
        tier = 3 if cues["emotion_visible"] and cues["persistence_visible"] else 0
    elif core == "reflection_mixed_feelings":
        tier = 3 if cues["emotion_visible"] and cues["mixed_or_two_sides_visible"] else 0
    elif core == "reflection_uncertainty":
        tier = 3 if cues["emotion_visible"] and cues["uncertainty_visible"] else 0
    elif core == "reflection_value_tension":
        tier = 3 if cues["emotion_visible"] and cues["value_tension_visible"] else 0
    elif core == "restatement_boundary_preference":
        tier = 3 if cues["listen_only"] or cues["low_burden"] else 0
    elif core == "restatement_current_impact":
        tier = 3 if cues["impact_visible"] else 0
    elif core == "restatement_current_priority":
        tier = 3 if cues["priority_visible"] else 0
    elif core == "restatement_stated_goal":
        tier = 3 if cues["explicit_goal"] else 0
    elif core == "restatement_two_sides":
        tier = 3 if cues["mixed_or_two_sides_visible"] else 0
    elif core == "affirmation_acknowledge_disclosure":
        tier = 2 if cues["specific_focus_visible"] else 0
    elif core == "affirmation_acknowledge_progress":
        tier = 3 if cues["effort_or_progress_visible"] else 0
    elif core == "affirmation_grounded_hope":
        tier = 3 if cues["positive_evidence_visible"] else 0
    elif core == "affirmation_permission_to_pace":
        tier = 3 if cues["overload_visible"] else 0
    elif core == "affirmation_recognize_persistence":
        tier = (
            3
            if cues["persistence_visible"] and cues["effort_or_progress_visible"]
            else 0
        )
    elif core == "affirmation_support_agency":
        tier = 3 if cues["agency_or_choice_visible"] else 0
    elif core == "affirmation_validate_reaction":
        tier = 3 if cues["emotion_visible"] else 0
    elif core == "suggestion_adjust_environment":
        tier = 3 if cues["environment_or_routine_visible"] else 0
    elif core == "suggestion_break_down_first_step":
        tier = (
            3
            if cues["overload_visible"]
            and (cues["explicit_goal"] or cues["uncertainty_visible"])
            else 0
        )
    elif core == "suggestion_brief_pause_or_grounding":
        tier = 3 if cues["overload_visible"] else 0
    elif core == "suggestion_communication_opening":
        tier = 3 if cues["communication_goal_visible"] else 0
    elif core == "suggestion_consider_qualified_support":
        tier = 3 if cues["serious_persistent_support_visible"] else 0
    elif core == "suggestion_prioritize_one_focus":
        tier = 3 if cues["multiple_tasks_visible"] else 0
    elif core == "suggestion_reach_trusted_support":
        tier = 3 if cues["support_person_visible"] else 0
    elif core == "suggestion_small_experiment":
        tier = (
            3
            if cues["uncertainty_visible"]
            and not cues["domain_decision_visible"]
            else 0
        )
    elif core == "suggestion_single_microstep":
        # The only safe generic suggestion fallback: one optional action whose
        # content must still be grounded in the user's visible goal.
        tier = 2 if cues["advice_welcome"] else 0
    else:
        reasons.append("core_submove_has_no_frozen_applicability_rule")

    if tier == 0:
        reasons.append("required_visible_cue_absent")
    return {
        "eligible": tier > 0,
        "compatibility_tier": tier,
        "reasons": reasons,
        "observable_card_cues": cues,
    }


def rank_applicable_v4_cards(
    *,
    query: str,
    current_user_text: str,
    recent_user_text: str,
    flags: Mapping[str, bool],
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank only cards whose explicit card-level prerequisites are met."""

    profile = selected_execution_profile(flags)
    families = set(eligible_families(flags))
    ranked: list[dict[str, Any]] = []
    for source_card in cards:
        card = dict(source_card)
        if (
            str(card["execution_profile"]) != profile
            or str(card["strategy_family"]) not in families
        ):
            continue
        applicability = assess_v4_card_applicability(
            card,
            current_user_text=current_user_text,
            recent_user_text=recent_user_text,
            flags=flags,
        )
        if not applicability["eligible"]:
            continue
        ranked.append(
            {
                "card_id": str(card["card_id"]),
                "core_submove_id": str(card["core_submove_id"]),
                "strategy_family": str(card["strategy_family"]),
                "execution_profile": str(card["execution_profile"]),
                "compatibility_tier": int(
                    applicability["compatibility_tier"]
                ),
                "applicability_reasons": list(applicability["reasons"]),
                "observable_card_cues": dict(
                    applicability["observable_card_cues"]
                ),
                "score": round(
                    lexical_score(query, str(card["retrieval_text"])),
                    8,
                ),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            int(item["compatibility_tier"]),
            float(item["score"]),
            str(item["card_id"]),
        ),
        reverse=True,
    )


def bank_semantic_digest(cards: Sequence[Mapping[str, Any]]) -> str:
    return stable_hex(
        V4_PROTOCOL,
        canonical_json([dict(row) for row in cards]),
        n=64,
    )
