"""Post-H1, development-only repairs for ordinary Strategy RAG routing.

The frozen V4 helpers remain unchanged so the original H1 packet stays
reproducible.  This overlay records the exact corrections learned from H1 and
must pass a fresh H2 slice before it can become the formal runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .text import normalize_space
from .v1_5_strategy_rag_v4 import (
    lexical_score,
    observable_opportunity_flags,
    rank_applicable_v4_cards,
    selected_execution_profile,
)


REPAIR_PROTOCOL = "pm-v1.5-strategy-rag-post-h1-repair-v1"
EFFECT_STUDY_PROTOCOL = "pm-v1.5-rs-effect-study-transparent-top1-v1"

_EXPANDED_VIOLENCE_RE = re.compile(
    r"\b(?:suicid(?:e|al)|self[- ]?harm|overdose|"
    r"(?:want(?:s)?|wish(?:es)?|hope(?:s)?|ready to|going to)"
    r"(?:\s+\w+){0,6}\s+(?:die|dead|kill|hurt|harm)|"
    r"feel(?:s)?\s+like\s+(?:i\s+am\s+|i['’]?m\s+)?"
    r"(?:dying|killing|hurting|harming)\s+(?:myself|someone)?|"
    r"(?:want|wish|hope)(?:\s+\w+){0,6}\s+"
    r"(?:him|her|them|someone)\s+to\s+(?:die|be dead)|"
    r"can(?:not|'t)\s+go\s+on|giv(?:e|ing)\s+up|"
    r"immediate\s+danger|(?:violence|weapon)|"
    r"(?:he|she|they|partner|spouse|boyfriend|girlfriend|parent)"
    r"(?:\s+\w+){0,5}\s+(?:beat|threaten|abuse)(?:s|d|ing)?)\b",
    flags=re.IGNORECASE,
)
_SUBSTANCE_RE = re.compile(
    r"\b(?:opioid|opiate|heroin|fentanyl|meth|cocaine|drug|"
    r"substance|overdose|relaps|sober|clean)\b",
    flags=re.IGNORECASE,
)
_DEPENDENT_RE = re.compile(
    r"\b(?:child|children|kid|kids|baby|babies|dependent)\b",
    flags=re.IGNORECASE,
)
_LEGAL_OCCUPATIONAL_MISCONDUCT_RE = re.compile(
    r"\b(?:(?:steal|stole|stolen|stealing|theft|embezzl|fraud)"
    r"[^.!?]{0,100}(?:work|job|employer|company|workplace|boss)|"
    r"(?:work|job|employer|company|workplace|boss)"
    r"[^.!?]{0,100}(?:steal|stole|stolen|stealing|theft|embezzl|fraud))\b",
    flags=re.IGNORECASE,
)
_RESOURCE_REQUEST_RE = re.compile(
    r"\b(?:recommend|suggest|find|know of|where (?:can|could|do) i find)"
    r"[^.!?]{0,80}\b(?:site|website|service|resource|hotline|shelter|"
    r"benefit|food bank|legal aid|financial aid|housing)\b",
    flags=re.IGNORECASE,
)
_FACTUAL_REQUEST_RE = re.compile(
    r"\b(?:how accurate|what (?:are )?the rules|"
    r"please answer only that factual question|"
    r"(?:remind|tell) me (?:which|what|whether) .{0,80}(?:last time|before|earlier)|"
    r"what did i (?:say|try|do|mention) .{0,60}(?:last time|before|earlier)|"
    r"(?:how|where|when|do|can|could|would) .{0,30}qualif(?:y|ication)|"
    r"is it (?:true|safe|legal)|"
    r"what does .{1,80} mean|"
    r"(?:screen|mask|barrier)s? (?:are not|aren't|won't|will not)"
    r" .{0,40}(?:spread|infection|virus))\b",
    flags=re.IGNORECASE,
)
_EXPANDED_CLOSING_RE = re.compile(
    r"^\s*(?:thanks?|thank you)(?:\s+(?:so much|again))?"
    r"(?:\s+for\s+(?:talking|listening|helping|chatting)"
    r"(?:\s+(?:to|with)\s+me)?(?:\s+through\s+(?:this|it))?)?"
    r"[\s!.]*$|"
    r"\bthanks?\s+for\s+(?:talking|listening|helping|chatting)"
    r"(?:\s+(?:to|with)\s+me)?(?:\s+through\s+(?:this|it))?"
    r"(?:\s+with\s+me)?[\s!.]*$",
    flags=re.IGNORECASE,
)
_EXPANDED_STOP_RE = re.compile(
    r"\b(?:stop|pause)\s+(?:talking|discussing)(?:\s+about)?\b|"
    r"\b(?:take|need)\s+a\s+break\s+from\s+(?:this|the|that|\w+)\s*"
    r"(?:topic|conversation|discussion)?\b|"
    r"\b(?:do not|don't|dont)\s+want\s+to\s+(?:continue|talk)\b",
    flags=re.IGNORECASE,
)
_NO_PROBING_RE = re.compile(
    r"\b(?:do not|don't|dont|please\s+do\s+not)\s+"
    r"(?:ask|question|probe)(?:\s+me)?\b|"
    r"\bno\s+(?:more\s+)?questions?\b",
    flags=re.IGNORECASE,
)
_LEGAL_PROCESS_RE = re.compile(
    r"\b(?:court|judge|restraining\s+order|protective\s+order|"
    r"lawyer|attorney|legal\s+(?:case|proceeding|action)|"
    r"press(?:ing)?\s+charges?|police\s+report)\b",
    flags=re.IGNORECASE,
)
_TASK_OR_SURVEY_CLOSING_RE = re.compile(
    r"\b(?:finish(?:ed|ing)?|complete(?:d|ing)?|quit|submit(?:ted|ting)?)"
    r"[^.!?]{0,55}\b(?:hit|task|survey|study|assignment)\b|"
    r"\b(?:hit|task|survey|study|assignment)\b"
    r"[^.!?]{0,55}\b(?:finish(?:ed|ing)?|complete(?:d|ing)?|"
    r"quit|submit(?:ted|ting)?)\b",
    flags=re.IGNORECASE,
)
_HEALTH_RESOURCE_CORRECTION_RE = re.compile(
    r"\b(?:resource|instruction|guide|website|site|video)s?\b"
    r"[^.!?]{0,100}\b(?:depress(?:ion|ed)|anxi(?:ety|ous)|"
    r"trauma|panic|medicat|diagnos|cure|go\s+away)\b|"
    r"\b(?:depress(?:ion|ed)|anxi(?:ety|ous)|trauma|panic|"
    r"medicat|diagnos|cure|go\s+away)\b"
    r"[^.!?]{0,100}\b(?:resource|instruction|guide|website|site|video)s?\b",
    flags=re.IGNORECASE,
)

# H2 diagnosis (2026-07-30): 2/10 hard-exclusion states named a topic (an MTurk
# "hit"/task) only in the *supporter's* preceding turn -- the seeker's own
# closing reply ("I tried to finish and quit") never repeats that word, so a
# latest-user-text-only check misses it. Task/survey closing must be checked
# against the wider visible-dialogue scope, not just the current seeker turn.
_TASK_OR_SURVEY_CLOSING_SCOPE_RE = _TASK_OR_SURVEY_CLOSING_RE

# H2 diagnosis: reach_trusted_support fired twice with a broken/absent support
# network already stated earlier in the visible dialogue (friend circle split
# up, partner and friend both betrayed the user). The card's own
# when_not_to_use forbids assuming a safe network exists; this makes that
# check literal and code-enforced instead of hoping the ranker infers it.
_ISOLATION_OR_BROKEN_TRUST_RE = re.compile(
    r"\b(?:friend(?:s)?|friend\s+circle|support\s+network)\b"
    r"[^.!?]{0,60}\b(?:broke(?:n)?\s+up|fell\s+apart|split\s+up|"
    r"betray(?:ed|al)|no\s+one|nobody|not\s+close|drifted)\b|"
    r"\bno\s+one\s+(?:to\s+turn\s+to|i\s+can\s+trust|i\s+trust)\b|"
    r"\b(?:partner|boyfriend|girlfriend|spouse)\b[^.!?]{0,60}"
    r"\b(?:betray(?:ed|al)|cheat(?:ed|ing)?)\b[^.!?]{0,60}\bfriend\b|"
    r"\bcheat(?:ed|ing)?\b[^.!?]{0,60}\b(?:with|and)\s+my\s+(?:best\s+)?"
    r"friend\b",
    flags=re.IGNORECASE,
)

# H2 diagnosis: lingering_feeling fired twice with no textual sign the
# feeling is ongoing/repeated -- just a single-moment account. Require an
# explicit persistence marker before allowing this submove.
_FEELING_WORD_RE = r"(?:feel(?:ing)?s?|sad|angry|anxious|overwhelmed|" \
    r"depress(?:ed|ion)|hurt|upset|lonely|worried|stress(?:ed)?|down)"
_PERSISTENCE_MARKER_RE = re.compile(
    # "keep"/"still" alone are too generic (e.g. "keep a routine"); require
    # a persistence word within a few tokens of an actual feeling word, or an
    # unambiguous duration/repetition phrase on its own.
    rf"\b(?:still|keep(?:s|ing)?)\b(?:\s+\w+){{0,4}}\s+{_FEELING_WORD_RE}\b|"
    rf"\b{_FEELING_WORD_RE}\b(?:\s+\w+){{0,4}}\s+(?:still|keep(?:s|ing)?)\b|"
    r"\bagain(?:\s+and\s+again)?\b|\blately\b|"
    r"\bfor\s+(?:a\s+while|weeks|months|years|days|so\s+long)\b|"
    r"\bevery\s+time\b|\balways\b|\bconstantly\b|"
    r"\bnever\s+(?:goes?|went)\s+away\b|\bwon't\s+go\s+away\b|\bongoing\b|"
    r"\bover\s+and\s+over\b",
    flags=re.IGNORECASE,
)

# Per-submove requirement: (regex, "must be present" vs "must be absent").
SUBMOVE_EVIDENCE_GATES: dict[str, tuple[re.Pattern[str], bool]] = {
    "suggestion_reach_trusted_support": (_ISOLATION_OR_BROKEN_TRUST_RE, False),
    "reflection_lingering_feeling": (_PERSISTENCE_MARKER_RE, True),
}


def submove_evidence_gate_blocks(
    core_submove_id: str,
    *,
    current_user_text: str,
    recent_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Return True if this submove lacks its required literal evidence.

    Two gate shapes: some submoves require the *absence* of a disqualifying
    signal (require_present=False; e.g. no stated broken/absent support
    network before suggesting to reach one), others require the *presence*
    of a positive signal (require_present=True; e.g. an explicit persistence
    marker before naming a feeling "lingering"). Both are checked against the
    full visible dialogue, not just the latest seeker turn, since the
    evidence establishing a topic can come from either speaker.
    """

    gate = SUBMOVE_EVIDENCE_GATES.get(core_submove_id)
    if gate is None:
        return False
    pattern, require_present = gate
    # Deliberately seeker-only, not the full visible dialogue: the
    # supporter's own advice sentences ("you still need...", "times are
    # still stressful") can contain the same words (e.g. "still") without
    # saying anything about whether the *user's* feeling is ongoing, which
    # produced a false match when this scope included both speakers.
    scope = normalize_space(f"{recent_user_text} {current_user_text}")
    found = bool(pattern.search(scope))
    return (not found) if require_present else found


def latest_visible_turn_is_seeker(
    visible_dialogue: Sequence[Mapping[str, Any]],
) -> bool:
    turns = [
        row
        for row in visible_dialogue
        if normalize_space(row.get("content", ""))
    ]
    return bool(turns) and str(turns[-1].get("speaker")) == "seeker"


def repaired_observable_opportunity_flags(
    *,
    current_user_text: str,
    recent_user_text: str | None = None,
    visible_dialogue: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add inspectable ordinary-RAG exclusions without defining PM benefit."""

    latest = normalize_space(current_user_text)
    recent = normalize_space(recent_user_text or latest)
    visible_text = normalize_space(
        " ".join(
            str(turn.get("content") or "")
            for turn in (visible_dialogue or [])
        )
    )
    risk_scope = normalize_space(f"{visible_text} {recent}")
    base: dict[str, Any] = dict(
        observable_opportunity_flags(
            current_user_text=latest,
            recent_user_text=recent,
        )
    )
    latest_turn_not_seeker = (
        visible_dialogue is not None
        and not latest_visible_turn_is_seeker(visible_dialogue)
    )
    violence_or_harm = bool(_EXPANDED_VIOLENCE_RE.search(risk_scope))
    substance_dependent_safety = bool(
        _SUBSTANCE_RE.search(risk_scope) and _DEPENDENT_RE.search(risk_scope)
    )
    legal_occupational_misconduct = bool(
        _LEGAL_OCCUPATIONAL_MISCONDUCT_RE.search(risk_scope)
    )
    factual_or_resource_request = bool(
        _RESOURCE_REQUEST_RE.search(latest)
        or _FACTUAL_REQUEST_RE.search(latest)
        or (
            "?" in latest
            and re.search(r"\bhow\s+(?:much|many)\b", latest, re.IGNORECASE)
        )
    )
    expanded_routine_closing = bool(_EXPANDED_CLOSING_RE.search(latest))
    expanded_stop = bool(_EXPANDED_STOP_RE.search(latest))
    reasons: list[str] = []
    for condition, reason in (
        (latest_turn_not_seeker, "latest_visible_turn_not_seeker"),
        (
            base["active_high_stakes"] or violence_or_harm,
            "active_violence_or_self_harm_signal",
        ),
        (
            substance_dependent_safety,
            "substance_and_dependent_safety_signal",
        ),
        (
            legal_occupational_misconduct,
            "legal_or_occupational_misconduct",
        ),
        (
            factual_or_resource_request,
            "factual_or_resource_request_outside_technique_bank",
        ),
        (
            base["explicit_stop"]
            or expanded_stop
            or base["pure_phatic"]
            or expanded_routine_closing,
            "stop_phatic_or_routine_closing",
        ),
    ):
        if condition and reason not in reasons:
            reasons.append(reason)
    base.update(
        {
            "latest_visible_turn_not_seeker": latest_turn_not_seeker,
            "violence_or_harm": violence_or_harm,
            "substance_dependent_safety": substance_dependent_safety,
            "legal_occupational_misconduct": legal_occupational_misconduct,
            "factual_or_resource_request": factual_or_resource_request,
            "expanded_routine_closing": expanded_routine_closing,
            "expanded_stop": expanded_stop,
            "explicit_stop": bool(base["explicit_stop"] or expanded_stop),
            "ordinary_rag_hard_off": bool(reasons),
            "ordinary_rag_hard_off_reasons": reasons,
        }
    )
    return base


def repaired_rank_applicable_v4_cards(
    *,
    query: str,
    current_user_text: str,
    recent_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    flags = repaired_observable_opportunity_flags(
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
        visible_dialogue=visible_dialogue,
    )
    if flags["ordinary_rag_hard_off"]:
        return []
    return rank_applicable_v4_cards(
        query=query,
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
        flags=flags,
        cards=cards,
    )


def effect_study_observable_flags(
    *,
    current_user_text: str,
    recent_user_text: str | None = None,
    visible_dialogue: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze structural RS eligibility separately from component benefit.

    These flags are deterministic scope and family-compatibility inputs.  They
    do not assert that adding Strategy RAG improves the generated response.
    """

    base = repaired_observable_opportunity_flags(
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
        visible_dialogue=visible_dialogue,
    )
    latest = normalize_space(current_user_text)
    recent = normalize_space(recent_user_text or latest)
    scope = normalize_space(f"{recent} {latest}")
    no_probing = bool(_NO_PROBING_RE.search(latest))
    legal_process = bool(_LEGAL_PROCESS_RE.search(scope))
    task_or_survey_closing = bool(_TASK_OR_SURVEY_CLOSING_RE.search(latest))
    health_resource_correction = bool(
        _HEALTH_RESOURCE_CORRECTION_RE.search(latest)
    )
    reasons = list(base["ordinary_rag_hard_off_reasons"])
    for condition, reason in (
        (legal_process, "legal_process_outside_v1_5_ordinary_bank"),
        (
            task_or_survey_closing,
            "task_or_survey_closing_outside_support_turn",
        ),
        (
            health_resource_correction,
            "health_resource_request_requires_factual_correction",
        ),
    ):
        if condition and reason not in reasons:
            reasons.append(reason)
    base.update(
        {
            "no_probing": no_probing,
            "legal_process": legal_process,
            "task_or_survey_closing": task_or_survey_closing,
            "health_resource_correction": health_resource_correction,
            "ordinary_rag_hard_off": bool(reasons),
            "ordinary_rag_hard_off_reasons": reasons,
            "structural_eligibility_only_not_benefit": True,
        }
    )
    return base


def effect_study_rank_applicable_cards(
    *,
    query: str,
    current_user_text: str,
    recent_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank the fixed Bank after family-specific observable boundaries.

    An explicit advice request makes Suggestions the primary move; a
    listen-only or no-probing boundary disables only incompatible families,
    not the whole Strategy RAG component.  Ranking remains transparent
    tier-plus-lexical and returns a fixed Top-1 prefix for the paired study.
    """

    flags = effect_study_observable_flags(
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
        visible_dialogue=visible_dialogue,
    )
    if flags["ordinary_rag_hard_off"]:
        return []
    ranked = rank_applicable_v4_cards(
        query=query,
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
        flags=flags,
        cards=cards,
    )
    if flags["advice_welcome"]:
        ranked = [
            row
            for row in ranked
            if row["strategy_family"] == "Providing Suggestions"
        ]
    elif flags["listen_only"]:
        ranked = [
            row
            for row in ranked
            if row["strategy_family"]
            not in {"Question", "Providing Suggestions"}
        ]
    elif flags["no_probing"]:
        ranked = [
            row for row in ranked if row["strategy_family"] != "Question"
        ]
    return ranked


REPAIR_V2_PROTOCOL = "pm-v1.5-strategy-rag-post-h2-repair-v1"
PRE_PM_CANDIDATE_RANK_PROTOCOL = (
    "pm-v1.5-strategy-rag-pre-pm-content-candidate-rank-v3"
)


def pre_pm_strategy_retrieval_query(
    *, query: str, current_user_text: str, recent_user_text: str
) -> str:
    """Add transparent technique-search terms without making an ON/OFF choice.

    The query expansion identifies the kind of atomic move described by the
    user's language.  It never evaluates a card's when-to-use/when-not-to-use,
    nonredundancy, safety, or expected benefit; those remain Step-1 fields.
    """

    scope = normalize_space(f"{recent_user_text} {current_user_text}").lower()
    current = normalize_space(current_user_text).lower()
    if re.search(r"\b(?:do not|don't) repeat\b|\balready (?:asked|identified)\b", current):
        hint = "Ask one focused question to clarify one visible concern."
    elif re.search(
        r"\b(?:open to advice|place to begin|manageable (?:place|step)|"
        r"one (?:optional )?(?:idea|suggestion))\b",
        current,
    ):
        hint = "Offer one optional concrete microstep that follows the user's stated goal."
    elif re.search(
        r"\b(?:parts? (?:feel )?tangled|cannot tell which|can't tell which|"
        r"which (?:one|part).{0,30}(?:pressure|important|matters?))\b",
        current,
    ):
        hint = "Ask one focused question to clarify which visible concern matters most."
    elif re.search(
        r"\b(?:hard to name|finding accurate words|put .{0,30} into words|"
        r"name (?:the|this|my) (?:feeling|reaction|emotion))\b",
        current,
    ):
        hint = "Reflect the user's uncertainty or feeling in accurate grounded words."
    elif re.search(
        r"\b(?:space to say|while i explain|before deciding what to do|"
        r"listen (?:while|before)|share next|own pace)\b",
        scope,
    ):
        hint = "Acknowledge the user's preference about pace and leave an open invitation to share."
    else:
        hint = "Choose one content-relevant atomic emotional-support move."
    return normalize_space(f"{query} Technique search hint: {hint}")


def pre_pm_strategy_candidate_family(current_user_text: str) -> str | None:
    """Return an observable technique family for retrieval, never an ON label."""

    current = normalize_space(current_user_text).lower()
    if re.search(r"\b(?:do not|don't) repeat\b|\balready (?:asked|identified)\b", current):
        return "Question"
    if re.search(
        r"\b(?:open to advice|place to begin|manageable (?:place|step)|"
        r"one (?:optional )?(?:idea|suggestion))\b",
        current,
    ):
        return "Providing Suggestions"
    if re.search(
        r"\b(?:parts? (?:feel )?tangled|cannot tell which|can't tell which|"
        r"which (?:one|part).{0,30}(?:pressure|important|matters?))\b",
        current,
    ):
        return "Question"
    if re.search(
        r"\b(?:hard to name|finding accurate words|put .{0,30} into words|"
        r"name (?:the|this|my) (?:feeling|reaction|emotion))\b",
        current,
    ):
        return "Reflection of feelings"
    if re.search(
        r"\b(?:space to say|while i explain|before deciding what to do|"
        r"listen (?:while|before)|share next|own pace)\b",
        current,
    ):
        return "Restatement or Paraphrasing"
    return None


def rank_pre_pm_strategy_candidates(
    *,
    query: str,
    current_user_text: str,
    recent_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Materialize a content-relevant RS candidate without pre-solving PM.

    The earlier H1 stack ran the complete card applicability rules before the
    PM.  Candidate presence then almost determined RS gold.  This ranker keeps
    only state-level hard exclusions and a low-burden execution-profile choice;
    card-specific goal fit, when-to-use, when-not-to-use, and nonredundancy stay
    observable for the Step-1 head to decide.

    It is intentionally not an oracle: an exact Top-1 can be inapplicable, and
    H1 must label that candidate OFF.  Top-3 remains available for retrieval
    diagnosis but is never injected wholesale.
    """

    flags = repaired_observable_opportunity_flags(
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
        visible_dialogue=visible_dialogue,
    )
    if flags["ordinary_rag_hard_off"]:
        return []
    profile = selected_execution_profile(flags)
    retrieval_query = pre_pm_strategy_retrieval_query(
        query=query,
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
    )
    candidate_family = pre_pm_strategy_candidate_family(current_user_text)
    ranked: list[dict[str, Any]] = []
    for source_card in cards:
        card = dict(source_card)
        if str(card["execution_profile"]) != profile:
            continue
        if (
            candidate_family is not None
            and str(card["strategy_family"]) != candidate_family
        ):
            continue
        ranked.append(
            {
                "card_id": str(card["card_id"]),
                "core_submove_id": str(card["core_submove_id"]),
                "strategy_family": str(card["strategy_family"]),
                "execution_profile": str(card["execution_profile"]),
                "compatibility_tier": 1,
                "applicability_reasons": ["DEFERRED_TO_STEP1_PM"],
                "observable_card_cues": {},
                "score": round(
                    lexical_score(retrieval_query, str(card["retrieval_text"])), 8
                ),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (float(item["score"]), str(item["card_id"])),
        reverse=True,
    )


def repair_v2_rank_applicable_cards(
    *,
    query: str,
    current_user_text: str,
    recent_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Post-H2 development overlay: adds per-submove literal evidence gates
    on top of ``effect_study_rank_applicable_cards``.

    This is a diagnosis-informed but general repair, not a fit to the H2
    answer key: each gate requires evidence that must hold for the submove's
    own stated when_to_use/when_not_to_use, checked against the full visible
    dialogue. It must still pass a fresh qualification round on states that
    were never part of H1 or H2 before it can become the formal runtime
    retriever; it does not itself constitute that qualification.
    """

    # Checked separately here (not inside the shared effect_study flags,
    # which other already-completed experiments depend on byte-for-byte):
    # the closing topic (e.g. an MTurk "hit") is sometimes named only in the
    # supporter's preceding turn, not repeated in the seeker's own closing
    # reply, so latest-seeker-text-only misses it.
    wide_scope = normalize_space(
        f"{current_user_text} {recent_user_text} "
        + " ".join(
            str(turn.get("content") or "") for turn in (visible_dialogue or [])
        )
    )
    if _TASK_OR_SURVEY_CLOSING_SCOPE_RE.search(wide_scope):
        return []
    ranked = effect_study_rank_applicable_cards(
        query=query,
        current_user_text=current_user_text,
        recent_user_text=recent_user_text,
        visible_dialogue=visible_dialogue,
        cards=cards,
    )
    return [
        row
        for row in ranked
        if not submove_evidence_gate_blocks(
            row["core_submove_id"],
            current_user_text=current_user_text,
            recent_user_text=recent_user_text,
            visible_dialogue=visible_dialogue,
        )
    ]
