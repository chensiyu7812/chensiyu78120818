"""Auditable pre-generation candidate features for PM V1.5 D3.

The retriever chooses a concrete candidate first.  This module then exposes
only two candidate scalars plus the exact background component bits to the
learned head.  It never reads a response, a judge result, a dataset name, or
an outcome label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .contracts import MemoryItem, MemorySource, parse_action_id
from .text import lexical_score, normalize_space


D3_FEATURE_PROTOCOL = "pm-v1.5-d3-auditable-candidate-features-v1"
MS_INCREMENTAL_VALUE_PROTOCOL = (
    "pm-v1.5-ms-incremental-candidate-value-v1"
)

_STALE_REQUEST_RE = re.compile(
    r"\b(?:please|can you|could you|would you|i want you to|i need you to)\b",
    flags=re.IGNORECASE,
)
_CURRENT_PERMISSION_RE = re.compile(
    r"\b(?:advice|suggest|idea|option|what (?:can|should|could) i do|"
    r"help me (?:decide|plan|figure out))\b",
    flags=re.IGNORECASE,
)
_CONFLICT_MARKER_RE = re.compile(
    r"\b(?:no longer|not anymore|that changed|this changed|"
    r"isn't true now|is not true now|used to,? but)\b",
    flags=re.IGNORECASE,
)
_MS_PRIOR_OUTCOME_RE = re.compile(
    r"\b(?:helped|worked|made [^.!?]{0,80} (?:easier|calmer)|"
    r"reduced|eased|allowed|clarified|became easier|was useful|was not helpful|"
    r"did not help|didn't help)\b",
    flags=re.IGNORECASE,
)
_MS_CURRENT_USE_RE = re.compile(
    r"\b(?:again|last time|previous(?:ly)?|earlier|before|returned|return|"
    r"what helped|what worked|what changed|small optional suggestion|"
    r"offer (?:me )?(?:one )?(?:small )?(?:optional )?(?:suggestion|idea)|"
    r"understand why|which part matters most|help (?:me )?put(?:ting)? this into words)\b",
    flags=re.IGNORECASE,
)
_ALREADY_VISIBLE_BY_SOURCE = {
    MemorySource.MP: re.compile(
        r"\balready (?:said|shared|mentioned|stated)\b[^.!?]{0,90}"
        r"\b(?:support preference|profile context)\b",
        flags=re.IGNORECASE,
    ),
    MemorySource.MS: re.compile(
        r"\balready (?:said|shared|mentioned|stated)\b[^.!?]{0,90}"
        r"\b(?:prior|previous|earlier) session summary\b",
        flags=re.IGNORECASE,
    ),
    MemorySource.ME: re.compile(
        r"\balready (?:said|shared|mentioned|stated|described)\b[^.!?]{0,90}"
        r"\b(?:past|prior|previous|earlier) event\b",
        flags=re.IGNORECASE,
    ),
}


def _visible_text(
    *, current_user_text: str, visible_dialogue: Sequence[Mapping[str, Any]]
) -> str:
    return normalize_space(
        " ".join(
            [
                *(str(row.get("content") or "") for row in visible_dialogue),
                current_user_text,
            ]
        )
    )


def _literal_redundancy(candidate_text: str, visible_text: str) -> bool:
    candidate = normalize_space(candidate_text).lower()
    visible = normalize_space(visible_text).lower()
    # Exact reuse is intentionally conservative.  Lexical similarity is the
    # separate match feature and must not be counted again as redundancy.
    return bool(candidate) and len(candidate) >= 16 and candidate in visible


def memory_grounding_observation(
    *,
    source: MemorySource,
    selected_items: Sequence[MemoryItem],
    catalog_memory_ids: set[str],
    catalog_user_id: str,
    current_user_id: str,
    current_session_index: int,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return six literal checks, a score, and a fail-closed decision.

    Redundancy is deliberately a soft opportunity signal: repeating visible
    context may make a component useless, but it is not a safety violation.
    Ownership, causal order, conflict, stale requests, and source eligibility
    are hard exclusions and never become noisy PM training targets.
    """

    metadata = source_metadata or {}
    visible = _visible_text(
        current_user_text=current_user_text,
        visible_dialogue=visible_dialogue,
    )
    present = bool(selected_items)
    belongs = present and catalog_user_id == current_user_id and all(
        item.memory_id in catalog_memory_ids for item in selected_items
    )
    strictly_prior = present and all(
        int(item.created_session) < int(current_session_index)
        for item in selected_items
    )
    conflict = bool(_CONFLICT_MARKER_RE.search(visible)) and any(
        lexical_score(visible, item.text) >= 0.12 for item in selected_items
    )
    nonconflicting = present and not conflict
    explicit_visible_reuse = bool(_ALREADY_VISIBLE_BY_SOURCE[source].search(visible))
    nonredundant = present and not explicit_visible_reuse and not any(
        _literal_redundancy(item.text, visible) for item in selected_items
    )
    contains_old_request = source is not MemorySource.MP and any(
        _STALE_REQUEST_RE.search(item.text) for item in selected_items
    )
    request_current = not contains_old_request or bool(
        _CURRENT_PERMISSION_RE.search(current_user_text)
    )

    if source is MemorySource.MP:
        eligible = present and all(
            str(metadata.get(item.memory_id, {}).get("mp_subtype", ""))
            in {"MP_PREFERENCE", "MP_PROFILE"}
            for item in selected_items
        )
    elif source is MemorySource.MS:
        eligible = present and all(
            metadata.get(item.memory_id, {}).get("summary_origin")
            == "supplied_strictly_prior_summary"
            for item in selected_items
        )
    else:
        eligible = present and all(item.source is MemorySource.ME for item in selected_items)

    checks = {
        "belongs_to_current_user": bool(belongs),
        "strictly_prior": bool(strictly_prior),
        "nonconflicting": bool(nonconflicting),
        "not_already_visible": bool(nonredundant),
        "request_currentness": bool(request_current),
        "source_specific_eligibility": bool(eligible),
    }
    hard_failures = [
        name
        for name in (
            "belongs_to_current_user",
            "strictly_prior",
            "nonconflicting",
            "request_currentness",
            "source_specific_eligibility",
        )
        if not checks[name]
    ]
    score = sum(int(value) for value in checks.values()) / len(checks)
    return {
        "protocol": D3_FEATURE_PROTOCOL,
        "component": source.value,
        "candidate_present": present,
        "checks": checks,
        "candidate_grounding_or_nonredundancy_score": round(score, 8),
        "deterministic_hard_off": (not present) or bool(hard_failures),
        "hard_off_reasons": (
            ["candidate_absent"] if not present else []
        )
        + hard_failures,
        "outcome_read": False,
    }


def ms_incremental_value_observation(
    *,
    selected_items: Sequence[MemoryItem],
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure whether a valid MS candidate can add useful prior-session context.

    This is an outcome-blind opportunity rubric, not a response-quality label.
    It intentionally reads only seeker-visible current context plus the realized
    strictly-prior summaries.  Supporter-authored metacommunication cannot alter
    the score.  When used as the controlled routing target, its composite value
    must not also be passed to the learned model as an input.  Paired responses
    evaluate downstream system effect; they do not define whether routing itself
    was correct.
    """

    seeker_visible = normalize_space(
        " ".join(
            [
                *(
                    str(row.get("content") or "")
                    for row in visible_dialogue
                    if str(row.get("speaker") or row.get("role") or "").lower()
                    in {"seeker", "user"}
                ),
                current_user_text,
            ]
        )
    )
    present = bool(selected_items)
    exact_visible_repetition = present and any(
        _literal_redundancy(item.text, seeker_visible)
        for item in selected_items
    )
    contains_prior_outcome = present and any(
        _MS_PRIOR_OUTCOME_RE.search(item.text) for item in selected_items
    )
    current_invites_prior_use = bool(_MS_CURRENT_USE_RE.search(current_user_text))
    nonredundant = present and not exact_visible_repetition
    # A compact, auditable opportunity score.  The interaction term matters:
    # prior outcomes are useful only when the current turn can use continuity,
    # and topical repetition alone receives no credit.
    alignment = float(
        present
        and nonredundant
        and contains_prior_outcome
        and current_invites_prior_use
    )
    return {
        "protocol": MS_INCREMENTAL_VALUE_PROTOCOL,
        "candidate_present": present,
        "checks": {
            "not_exactly_visible_to_seeker": bool(nonredundant),
            "contains_prior_response_or_outcome": bool(contains_prior_outcome),
            "current_turn_can_use_prior_continuity": bool(current_invites_prior_use),
        },
        "candidate_incremental_alignment_score": alignment,
        "not_an_effect_label": True,
        "outcome_read": False,
    }


def _strategy_family_redundant(
    family: str, visible_dialogue: Sequence[Mapping[str, Any]]
) -> bool:
    supporter = [
        normalize_space(row.get("content") or "")
        for row in visible_dialogue
        if str(row.get("speaker") or "") == "supporter"
    ]
    last = supporter[-1].lower() if supporter else ""
    if not last:
        return False
    if "already used this same support technique" in last:
        return True
    if family == "Question":
        return "?" in last
    if family == "Providing Suggestions":
        return bool(re.search(r"\b(?:you could|you might|try|consider)\b", last))
    if family == "Reflection of feelings":
        return bool(re.search(r"\b(?:it sounds|you feel|feeling|weighing)\b", last))
    if family == "Affirmation and Reassurance":
        return bool(re.search(r"\b(?:makes sense|acknowledge|effort|right to)\b", last))
    if family == "Restatement or Paraphrasing":
        return bool(re.search(r"\b(?:so,?|what i'm hearing|you've said)\b", last))
    return False


def strategy_grounding_observation(
    *,
    candidate: Mapping[str, Any] | None,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the RS counterpart of the six transparent grounding checks."""

    present = candidate is not None
    card = dict(candidate or {})
    flags = dict(card.get("observable_flags") or {})
    family = str(card.get("strategy_family") or "")
    source_eligible = present and card.get("content_scope", "technique_only") == "technique_only"
    family_supported = present and int(card.get("compatibility_tier", 0)) > 0
    no_hard_exclusion = present and not bool(flags.get("ordinary_rag_hard_off"))
    if family == "Providing Suggestions":
        request_current = bool(flags.get("advice_welcome"))
    elif family == "Question":
        request_current = not bool(flags.get("listen_only") or flags.get("no_probing"))
    else:
        request_current = True
    nonredundant = present and not _strategy_family_redundant(
        family, visible_dialogue
    )
    query_current = present and normalize_space(current_user_text) in normalize_space(
        str(card.get("query") or "")
    )
    checks = {
        "candidate_query_is_current": bool(query_current),
        "source_specific_eligibility": bool(source_eligible),
        "family_visible_cue_supported": bool(family_supported),
        "no_structural_hard_exclusion": bool(no_hard_exclusion),
        "not_already_visible": bool(nonredundant),
        "request_currentness": bool(request_current),
    }
    hard_names = (
        "candidate_query_is_current",
        "source_specific_eligibility",
        "family_visible_cue_supported",
        "no_structural_hard_exclusion",
        "request_currentness",
    )
    hard_failures = [name for name in hard_names if not checks[name]]
    score = sum(int(value) for value in checks.values()) / len(checks)
    return {
        "protocol": D3_FEATURE_PROTOCOL,
        "component": "RS",
        "candidate_present": present,
        "checks": checks,
        "candidate_grounding_or_nonredundancy_score": round(score, 8),
        "deterministic_hard_off": (not present) or bool(hard_failures),
        "hard_off_reasons": (
            ["candidate_absent"] if not present else []
        )
        + hard_failures,
        "outcome_read": False,
    }


def d3_model_features(
    *,
    component: str,
    candidate_state_match_score: float,
    grounding_score: float,
    background_action: str,
    candidate_is_preference: bool = False,
) -> dict[str, float]:
    """Build the exact bounded feature vector frozen for one D3 head."""

    sources, strategy = parse_action_id(background_action)
    bits = {
        "MP": float(MemorySource.MP in sources),
        "MS": float(MemorySource.MS in sources),
        "ME": float(MemorySource.ME in sources),
        "RS": float(strategy.value == "RS"),
    }
    common = {
        "candidate_state_match_score": float(candidate_state_match_score),
        "candidate_grounding_or_nonredundancy_score": float(grounding_score),
    }
    if component == "RS":
        return {**common, "background_MP_on": bits["MP"], "background_MS_on": bits["MS"], "background_ME_on": bits["ME"]}
    if component == "MP":
        return {
            **common,
            "candidate_is_preference": float(candidate_is_preference),
            "background_MS_on": bits["MS"],
            "background_ME_on": bits["ME"],
            "background_RS_on": bits["RS"],
        }
    if component == "MS":
        return {**common, "background_MP_on": bits["MP"], "background_ME_on": bits["ME"], "background_RS_on": bits["RS"]}
    if component == "ME":
        return {**common, "background_MP_on": bits["MP"], "background_MS_on": bits["MS"], "background_RS_on": bits["RS"]}
    raise ValueError(f"unsupported component: {component}")
