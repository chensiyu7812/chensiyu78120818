"""One production feature bridge for PM V1.5 memory opportunity heads.

Internal training, untouched confirmation, and EvoEmo must call this module
after the same candidate discovery step.  It combines the candidate's bounded
retrieval descriptor with source-specific grounding checks and background
component bits.  No response, judge result, target action, dataset identity,
raw memory text, or memory identifier appears in the returned model features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from .contracts import MemoryItem, MemorySource
from .v1_5_candidate_discovery import MemoryCandidate
from .v1_5_d3_features import d3_model_features, memory_grounding_observation
from .text import content_word_match_level, normalize_space


MEMORY_OPPORTUNITY_FEATURE_PROTOCOL = (
    "pm-v1.5-production-memory-opportunity-features-v1"
)
SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL = (
    "pm-v1.5-production-memory-opportunity-features-v2-content-level"
)
SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL = (
    "pm-v1.5b-source-specific-memory-opportunity-features-v1"
)
DEFAULT_MAXIMUM_INCREMENTAL_TOKENS = 512


_ADVICE_REQUEST_RE = re.compile(
    r"\b(?:advice|suggestions?|(?:one|a) (?:small )?(?:idea|option)|"
    r"(?:give|offer|share) (?:me )?(?:one|a)? ?(?:small )?(?:idea|option)|"
    r"what (?:should|can|could) i do|"
    r"help me (?:decide|plan|choose|figure out))\b",
    flags=re.IGNORECASE,
)
_ADVICE_REJECT_RE = re.compile(
    r"\b(?:no advice|don't want advice|do not want advice|not looking for advice|"
    r"just (?:listen|hear me|let me vent)|only want to (?:talk|vent)|"
    r"without (?:advice|suggestions|a plan))\b",
    flags=re.IGNORECASE,
)
_LISTEN_RE = re.compile(
    r"\b(?:listen|hear me|let me (?:finish|vent|talk)|finish my thoughts|"
    r"without interrupt|talk (?:it |this )?through|put .* into words)\b",
    flags=re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"\b(?:one question|focused question|ask .* question|clarify|"
    r"help me understand|which part|what matters)\b",
    flags=re.IGNORECASE,
)
_LOW_BURDEN_RE = re.compile(
    r"\b(?:concise|brief|short|one small|low[- ]burden|not a list|"
    r"not a plan|overwhelmed|one at a time)\b",
    flags=re.IGNORECASE,
)
_REFLECTION_RE = re.compile(
    r"\b(?:reflect|paraphrase|put .* into words|name (?:the )?(?:feeling|tension)|"
    r"say back|summari[sz]e)\b",
    flags=re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(?:plan|steps?|priority|decide|choose|action|experiment|try next)\b",
    flags=re.IGNORECASE,
)
_CONTINUITY_RE = re.compile(
    r"\b(?:again|returned|back|last time|previously|earlier|before|"
    r"still|same (?:issue|problem|feeling)|what (?:helped|worked|changed)|"
    r"which part|why this)\b",
    flags=re.IGNORECASE,
)
_PRIOR_VALUE_RE = re.compile(
    r"\b(?:helped|worked|eased|reduced|made .* easier|clarified|"
    r"one part|rather than|instead of|the main|most difficult|"
    r"wasn't helpful|was not helpful|didn't help|did not help)\b",
    flags=re.IGNORECASE,
)
_RESOLVED_RE = re.compile(
    r"\b(?:resolved|finished|completed|no longer a problem|fully settled|"
    r"closed the issue)\b",
    flags=re.IGNORECASE,
)
_ISSUE_SPECIFIC_RE = re.compile(
    r"\b(?:because of|main (?:pressure|difficulty|bottleneck|part)|"
    r"rather than|instead of|remained (?:open|unresolved)|"
    r"the issue (?:remained|was)|one part)\b",
    flags=re.IGNORECASE,
)
_THIRD_PARTY_FOCUS_RE = re.compile(
    r"\b(?:my|a|the) (?:sibling|brother|sister|friend|coworker|colleague|"
    r"partner|spouse|child|parent)\b",
    flags=re.IGNORECASE,
)


def _interaction_modes(text: str) -> set[str]:
    """Return a small, preregistered set of observable conversational modes."""

    value = normalize_space(text)
    modes: set[str] = set()
    for name, pattern in (
        ("advice", _ADVICE_REQUEST_RE),
        ("listen", _LISTEN_RE),
        ("question", _QUESTION_RE),
        ("low_burden", _LOW_BURDEN_RE),
        ("reflection", _REFLECTION_RE),
        ("plan", _PLAN_RE),
    ):
        if pattern.search(value):
            modes.add(name)
    return modes


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


def source_specific_candidate_fit_factors(
    *,
    source: MemorySource,
    candidate: MemoryCandidate,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Mapping[str, Any]],
    not_already_visible: bool,
) -> dict[str, float]:
    """Derive transparent source-specific factors without outcome labels.

    Generic overlap remains one diagnostic signal, but it can no longer stand
    in for source semantics.  MP separates preference scope from profile
    relevance.  MS separates same-issue continuity, current conversational
    goal, and the presence of a usable prior distinction/outcome.  ME exposes
    independently compiled action/result/mechanism slots rather than treating
    every same-topic historical episode as reusable.
    """

    visible = _visible_text(
        current_user_text=current_user_text,
        visible_dialogue=visible_dialogue,
    )
    selected = tuple(candidate.selected_items)
    content_levels = [
        content_word_match_level(visible, item.text) for item in selected
    ]
    content_level = max(content_levels, default=0.0)
    incremental = float(bool(selected) and not_already_visible)

    if source is MemorySource.MP:
        preference_items = [
            item
            for item in selected
            if source_metadata.get(item.memory_id, {}).get("mp_subtype")
            == "MP_PREFERENCE"
        ]
        profile_items = [
            item
            for item in selected
            if source_metadata.get(item.memory_id, {}).get("mp_subtype")
            == "MP_PROFILE"
        ]
        current_modes = _interaction_modes(current_user_text)
        preference_modes = set().union(
            *(_interaction_modes(item.text) for item in preference_items)
        ) if preference_items else set()
        scope_conflict = bool(
            preference_items
            and _ADVICE_REJECT_RE.search(current_user_text)
            and "advice" in preference_modes
        )
        if not preference_items:
            preference_fit = 0.0
        elif scope_conflict:
            preference_fit = 0.0
        elif preference_modes & current_modes:
            preference_fit = 1.0
        else:
            # A stable pacing preference can still be applicable when the
            # current turn does not explicitly restate it, but receives only
            # partial rather than automatic credit.
            preference_fit = 0.5
        profile_relevance = max(
            (
                content_word_match_level(visible, item.text)
                for item in profile_items
            ),
            default=0.0,
        )
        profile_entity_scope_fit = float(
            bool(profile_items)
            and not bool(_THIRD_PARTY_FOCUS_RE.search(current_user_text))
        )
        return {
            "candidate_content_match_level": float(content_level),
            "candidate_incremental_information": incremental,
            "candidate_preference_scope_fit": float(preference_fit),
            "candidate_profile_relevance": float(profile_relevance),
            "candidate_profile_entity_scope_fit": profile_entity_scope_fit,
            "candidate_current_scope_conflict": float(scope_conflict),
        }

    if source is MemorySource.MS:
        current_modes = _interaction_modes(current_user_text)
        candidate_modes = set().union(
            *(_interaction_modes(item.text) for item in selected)
        ) if selected else set()
        if current_modes and candidate_modes:
            goal_fit = float(bool(current_modes & candidate_modes))
        elif current_modes or candidate_modes:
            goal_fit = 0.5
        else:
            goal_fit = 0.5
        return {
            "candidate_content_match_level": float(content_level),
            "candidate_incremental_information": incremental,
            "candidate_same_issue_level": float(content_level),
            "candidate_current_goal_fit": float(goal_fit),
            "candidate_prior_outcome_or_distinction": float(
                any(_PRIOR_VALUE_RE.search(item.text) for item in selected)
            ),
            "candidate_specific_issue_or_distinction": float(
                any(_ISSUE_SPECIFIC_RE.search(item.text) for item in selected)
            ),
            "current_continuity_invitation": float(
                bool(_CONTINUITY_RE.search(current_user_text))
            ),
            "candidate_prior_issue_marked_resolved": float(
                any(_RESOLVED_RE.search(item.text) for item in selected)
            ),
        }

    if source is MemorySource.ME:
        current_modes = _interaction_modes(current_user_text)
        candidate_modes = set().union(
            *(_interaction_modes(item.text) for item in selected)
        ) if selected else set()
        if current_modes and candidate_modes:
            goal_fit = float(bool(current_modes & candidate_modes))
        elif current_modes or candidate_modes:
            goal_fit = 0.5
        else:
            goal_fit = 0.5
        selected_metadata = [
            source_metadata.get(item.memory_id, {}) for item in selected
        ]
        return {
            "candidate_content_match_level": float(content_level),
            "candidate_incremental_information": incremental,
            "candidate_current_goal_fit": float(goal_fit),
            "candidate_contains_action": float(
                any(bool(row.get("me_contains_action")) for row in selected_metadata)
            ),
            "candidate_contains_result": float(
                any(bool(row.get("me_contains_result")) for row in selected_metadata)
            ),
            "candidate_contains_mechanism": float(
                any(bool(row.get("me_contains_mechanism")) for row in selected_metadata)
            ),
            "current_continuity_invitation": float(
                bool(_CONTINUITY_RE.search(current_user_text))
            ),
        }

    return {
        "candidate_content_match_level": float(content_level),
        "candidate_incremental_information": incremental,
    }


def build_memory_opportunity_observation(
    *,
    candidate: MemoryCandidate,
    catalog_items: Sequence[MemoryItem],
    catalog_user_id: str,
    current_user_id: str,
    current_session_index: int,
    current_user_text: str,
    visible_dialogue: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Mapping[str, Any]],
    background_action: str,
    maximum_incremental_tokens: int = DEFAULT_MAXIMUM_INCREMENTAL_TOKENS,
) -> dict[str, Any]:
    """Return the exact source-specific head input and deterministic gates."""

    if maximum_incremental_tokens < 1:
        raise ValueError("maximum_incremental_tokens must be positive")
    source = candidate.source
    descriptor = dict(candidate.descriptor)
    if str(descriptor.get("source")) != source.value:
        raise ValueError("candidate descriptor source mismatch")
    catalog_ids = {
        item.memory_id for item in catalog_items if item.source is source
    }
    grounding = memory_grounding_observation(
        source=source,
        selected_items=candidate.selected_items,
        catalog_memory_ids=catalog_ids,
        catalog_user_id=catalog_user_id,
        current_user_id=current_user_id,
        current_session_index=current_session_index,
        current_user_text=current_user_text,
        visible_dialogue=visible_dialogue,
        source_metadata=source_metadata,
    )
    selected_metadata = [
        dict(source_metadata.get(item.memory_id) or {})
        for item in candidate.selected_items
    ]
    candidate_is_preference = bool(
        source is MemorySource.MP
        and selected_metadata
        and all(row.get("mp_subtype") == "MP_PREFERENCE" for row in selected_metadata)
    )
    incremental_tokens = int(descriptor.get("incremental_injected_tokens") or 0)
    within_budget = incremental_tokens <= maximum_incremental_tokens
    features = d3_model_features(
        component=source.value,
        candidate_state_match_score=float(
            descriptor.get("top1_lexical_relevance") or 0.0
        ),
        grounding_score=float(
            grounding["candidate_grounding_or_nonredundancy_score"]
        ),
        background_action=background_action,
        candidate_is_preference=candidate_is_preference,
    )
    hard_off_reasons = list(grounding["hard_off_reasons"])
    if not within_budget:
        hard_off_reasons.append("incremental_token_budget_exceeded")
    return {
        "protocol": MEMORY_OPPORTUNITY_FEATURE_PROTOCOL,
        "component": source.value,
        "model_features": features,
        "deterministic_hard_off": bool(hard_off_reasons),
        "hard_off_reasons": hard_off_reasons,
        "candidate_audit": {
            "candidate_present": bool(descriptor.get("candidate_present")),
            "retrieved_count": int(descriptor.get("retrieved_count") or 0),
            "incremental_injected_tokens": incremental_tokens,
            "within_incremental_token_budget": within_budget,
            "candidate_model_features": dict(descriptor.get("model_features") or {}),
            "grounding_checks": dict(grounding["checks"]),
        },
        "forbidden_inputs_read": [],
        "outcome_read": False,
    }


def build_scale_stable_memory_opportunity_observation(
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the one allowed scale-stable V1.5 representation repair.

    Candidate discovery, grounding, hard gates, budgets, and background bits
    remain identical to V1.  Only the raw term-frequency cosine feature is
    replaced by the pre-registered three-level content-word match.
    """

    result = build_memory_opportunity_observation(**kwargs)
    candidate = kwargs["candidate"]
    features = dict(result["model_features"])
    features.pop("candidate_state_match_score")
    features["candidate_content_match_level"] = float(
        candidate.descriptor.get("selected_content_word_match_level") or 0.0
    )
    result["protocol"] = SCALE_STABLE_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
    result["model_features"] = features
    result["candidate_audit"] = {
        **dict(result["candidate_audit"]),
        "selected_content_word_match_level": features[
            "candidate_content_match_level"
        ],
        "raw_lexical_score_used_by_head": False,
    }
    return result


def build_source_specific_memory_opportunity_observation(
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the V1.5b source-specific representation for MP/MS repair.

    The old V1/V2 builders remain untouched for reproducibility.  This bridge
    reuses their deterministic ownership, causal-order, conflict, redundancy,
    eligibility, and cost gates, then replaces the one-dimensional semantic
    proxy with explicit source constructs.  ME can pass through for transport
    diagnostics but is not retuned by this repair.
    """

    result = build_memory_opportunity_observation(**kwargs)
    candidate: MemoryCandidate = kwargs["candidate"]
    source = candidate.source
    grounding_checks = dict(result["candidate_audit"]["grounding_checks"])
    factors = source_specific_candidate_fit_factors(
        source=source,
        candidate=candidate,
        current_user_text=str(kwargs["current_user_text"]),
        visible_dialogue=kwargs["visible_dialogue"],
        source_metadata=kwargs["source_metadata"],
        not_already_visible=bool(grounding_checks["not_already_visible"]),
    )
    features = dict(result["model_features"])
    features.pop("candidate_state_match_score", None)
    features.update(factors)
    result["protocol"] = SOURCE_SPECIFIC_MEMORY_OPPORTUNITY_FEATURE_PROTOCOL
    result["model_features"] = features
    result["candidate_audit"] = {
        **dict(result["candidate_audit"]),
        "source_specific_factor_names": sorted(factors),
        "source_specific_factors": dict(factors),
        "raw_candidate_or_visible_text_returned": False,
        "raw_lexical_score_used_by_head": False,
    }
    return result
