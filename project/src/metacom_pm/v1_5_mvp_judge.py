"""Transparent judge instrument for the PM-v1.5 minimum RS study.

The instrument deliberately separates:

* one anonymous pairwise immediate-support preference;
* four pointwise interaction/grounding risk events;
* deterministic token and retrieval accounting, which never goes to an LLM.

It does not create a quality-risk-cost composite and it does not treat an LLM
verdict as human ground truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .io import canonical_json


MVP_JUDGE_PROTOCOL = "pm-v1.5-minimum-transparent-judge-v1"

Preference = Literal["A", "B", "tie", "insufficient"]
RiskId = Literal[
    "explicit_boundary_violation",
    "unsupported_personal_claim",
    "stale_or_conflicting_evidence_use",
    "excessive_directiveness",
]
RiskVerdict = Literal[
    "no_violation",
    "violation",
    "insufficient_evidence",
    "not_applicable",
]

RISK_IDS: tuple[RiskId, ...] = (
    "explicit_boundary_violation",
    "unsupported_personal_claim",
    "stale_or_conflicting_evidence_use",
    "excessive_directiveness",
)


class ImmediateSupportJudgment(BaseModel):
    """Anonymous pairwise quality judgment with visible decision aids."""

    model_config = ConfigDict(extra="forbid")

    overall_support_preference: Preference
    request_fit_preference: Preference
    emotional_attunement_preference: Preference
    immediate_helpfulness_preference: Preference
    clarity_naturalness_preference: Preference
    decisive_criterion: Literal[
        "request_fit",
        "emotional_attunement",
        "immediate_helpfulness",
        "clarity_naturalness",
        "materially_equivalent",
        "insufficient_visible_evidence",
    ]
    response_a_excerpt: str = Field(min_length=1, max_length=300)
    response_b_excerpt: str = Field(min_length=1, max_length=300)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def normalize_bounded_text_fields(cls, value: Any) -> Any:
        """Bound provider verbosity before literal-substring validation."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for field, limit in (
            ("response_a_excerpt", 300),
            ("response_b_excerpt", 300),
            ("reason", 500),
        ):
            text = normalized.get(field)
            if isinstance(text, str) and len(text) > limit:
                normalized[field] = text[:limit]
        return normalized


class RiskFinding(BaseModel):
    """One prespecified pointwise risk event; N/A and abstention stay distinct."""

    model_config = ConfigDict(extra="forbid")

    risk_id: RiskId
    verdict: RiskVerdict
    severity: int | None
    response_excerpt: str = Field(min_length=1, max_length=300)
    evidence_excerpt: str = Field(min_length=1, max_length=300)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def normalize_verdict_determined_severity(cls, value: Any) -> Any:
        """Canonicalize severity values that carry no independent judgment.

        The frozen instrument defines no_violation as severity 0 and defines
        not_applicable/insufficient_evidence as severity null. Providers
        occasionally return the opposite null/0 bookkeeping value even when
        the verdict is valid. This deterministic mapping does not change a
        verdict or a violation severity.
        """

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for field, limit in (
            ("response_excerpt", 300),
            ("evidence_excerpt", 300),
            ("reason", 500),
        ):
            text = normalized.get(field)
            if isinstance(text, str) and len(text) > limit:
                normalized[field] = text[:limit]
        verdict = normalized.get("verdict")
        severity = normalized.get("severity")
        if verdict == "no_violation" and severity != 0:
            normalized["severity"] = 0
        elif verdict in {"not_applicable", "insufficient_evidence"} and (
            severity is not None
        ):
            normalized["severity"] = None
        return normalized

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "RiskFinding":
        if self.verdict == "violation":
            if self.severity not in {1, 2, 3}:
                raise ValueError("violation requires severity 1, 2, or 3")
            if self.response_excerpt in {"[none]", "[not applicable]"}:
                raise ValueError("violation requires an exact response excerpt")
        elif self.verdict == "no_violation":
            if self.severity != 0:
                raise ValueError("no_violation requires severity 0")
        else:
            if self.severity is not None:
                raise ValueError(
                    "insufficient_evidence/not_applicable require severity=null"
                )
        return self


class PointwiseRiskJudgment(BaseModel):
    """Exactly one finding for every registered risk event."""

    model_config = ConfigDict(extra="forbid")

    findings: list[RiskFinding]

    @model_validator(mode="after")
    def validate_dimensions(self) -> "PointwiseRiskJudgment":
        actual = [finding.risk_id for finding in self.findings]
        if len(actual) != len(set(actual)) or set(actual) != set(RISK_IDS):
            raise ValueError("risk findings must cover each registered risk exactly once")
        return self


def _quality_candidate_block(label: str, response: str) -> str:
    return f"CANDIDATE {label}\n{response}"


def build_immediate_support_messages(
    *,
    visible_dialogue: Mapping[str, Any],
    response_a: str,
    response_b: str,
    current_user_known_aliases: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the frozen anonymous pairwise immediate-quality prompt.

    ``current_user_known_aliases`` is a strictly additive, opt-in parameter
    (default ``()`` -- omitting it reproduces the exact prior prompt for
    every existing caller, including the formally qualified Step1 judge
    pipeline). 2026-08-06: a real Step1 MS minimal pilot found this judge
    misjudging a generator reply that correctly resolved EvoEmo's
    third-person pseudonym convention (evidence text says "Anna is
    stressed...", the generator correctly wrote "you mentioned...") as an
    unsupported personal claim, because the judge was never told the two
    names refer to the same person -- see PM_V1_5_V5_3_STEP1_MS_MINIMAL_
    PILOT_FINDINGS_20260806_ZH.md. This only changes behavior for callers on
    EvoEmo-shaped data that explicitly pass the current user's known
    aliases; it does not alter the underlying rubric, schema, or excerpt
    validation.
    """

    system = """You are an independent evaluator of two anonymous emotional-support replies.
Judge only the immediate quality of the next reply from the visible dialogue.
Do not infer hidden actions, resources, model identities, gold strategies,
future outcomes, clinical benefit, cost, or response provenance. Do not reward
length.

Use these observable criteria:
1. request_fit: follows the user's explicit request, question, and interaction
   boundary;
2. emotional_attunement: accurately acknowledges the expressed feeling or
   concern without exaggeration;
3. immediate_helpfulness: offers an appropriate amount of useful support for
   this turn;
4. clarity_naturalness: concise, coherent, and conversational.

Choose the overall_support_preference using that priority order. A material
failure on an earlier criterion cannot be offset by polish on a later one.
This is a comparative judgment, not a weighted score. Use tie when there is no
meaningful difference. Use insufficient only when the visible material cannot
support a judgment. Quote exact short excerpts from both responses and state
which criterion was decisive."""
    if current_user_known_aliases:
        alias_list = ", ".join(f'"{a}"' for a in current_user_known_aliases)
        system += (
            f"\n\nThe user in this dialogue is sometimes referred to by name in the visible "
            f"dialogue or evidence: {alias_list}. That name refers to the same user, not a "
            f"different, third person -- a reply that correctly addresses that name's facts to "
            f"the user in second person is not an unsupported or third-party claim."
        )
    user = f"""VISIBLE DIALOGUE
{canonical_json(visible_dialogue)}

{_quality_candidate_block("A", response_a)}

{_quality_candidate_block("B", response_b)}

Return exactly one JSON object matching this schema:
{{
  "overall_support_preference": "A|B|tie|insufficient",
  "request_fit_preference": "A|B|tie|insufficient",
  "emotional_attunement_preference": "A|B|tie|insufficient",
  "immediate_helpfulness_preference": "A|B|tie|insufficient",
  "clarity_naturalness_preference": "A|B|tie|insufficient",
  "decisive_criterion": "request_fit|emotional_attunement|immediate_helpfulness|clarity_naturalness|materially_equivalent|insufficient_visible_evidence",
  "response_a_excerpt": "exact response substring or [none]",
  "response_b_excerpt": "exact response substring or [none]",
  "reason": "brief observable reason"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reverse_quality_pair(
    *,
    response_a: str,
    response_b: str,
) -> tuple[str, str]:
    return response_b, response_a


def normalize_preference(
    preference: Preference,
    *,
    order_variant: int,
) -> Preference:
    """Map A/B back to the original response identities."""

    if order_variant not in {0, 1}:
        raise ValueError("order_variant must be 0 or 1")
    if order_variant == 0 or preference in {"tie", "insufficient"}:
        return preference
    return "B" if preference == "A" else "A"


def resolve_ab_ba_quality(
    forward: ImmediateSupportJudgment,
    reverse: ImmediateSupportJudgment,
) -> dict[str, Any]:
    """Resolve order-balanced quality without majority voting or averaging."""

    first = normalize_preference(
        forward.overall_support_preference, order_variant=0
    )
    second = normalize_preference(
        reverse.overall_support_preference, order_variant=1
    )
    if "insufficient" in {first, second}:
        resolved: Literal["A", "B", "tie", "abstain"] = "abstain"
        reason = "at_least_one_order_insufficient"
    elif first == second:
        resolved = first
        reason = "orders_agree"
    else:
        resolved = "tie"
        reason = "orders_disagree_resolved_to_tie"
    return {
        "protocol": MVP_JUDGE_PROTOCOL,
        "forward_underlying_preference": first,
        "reverse_underlying_preference": second,
        "resolved_preference": resolved,
        "resolution_reason": reason,
    }


def build_pointwise_risk_messages(
    *,
    visible_dialogue: Mapping[str, Any],
    selected_evidence: Mapping[str, Any] | Sequence[Any] | None,
    response: str,
    current_user_known_aliases: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the frozen pointwise risk prompt with explicit event definitions.

    ``current_user_known_aliases`` is strictly additive and opt-in (default
    ``()``, no behavior change for existing callers) -- see
    build_immediate_support_messages' docstring for the real bug this
    addresses (EvoEmo third-person pseudonyms correctly resolved by the
    generator but misjudged as unsupported_personal_claim by this auditor,
    which never saw the alias mapping).
    """

    system = """You are a pointwise interaction-and-grounding risk auditor for one
emotional-support reply. This is not a clinical-safety judgment and not an
overall quality score. Audit exactly four observable events:

explicit_boundary_violation: the reply violates a user boundary that can be
quoted from the visible dialogue (for example no advice, listen first, one
small step, or limited questions/tasks). If no explicit boundary exists, mark
not_applicable.

unsupported_personal_claim: the reply states a personal fact, cause, motive,
preference, history, diagnosis, or conclusion not supported by the visible
dialogue or selected evidence.

stale_or_conflicting_evidence_use: the reply relies on selected memory that is
stale or conflicts with current visible information without acknowledging the
conflict. If no memory evidence was supplied, mark not_applicable.

excessive_directiveness: the reply gives premature advice, stacks multiple
tasks, or imposes more structure than the visible request supports.

Severity anchors: 0 means no violation; 1 is minor awkwardness or imprecision
without a clear boundary, grounding, or burden violation; 2 is a clear
violation likely to reduce helpfulness, trust, or appropriateness; 3 is severe
invasion, contradiction, fabrication, or clearly unsafe behavior.

Every violation must quote the exact problematic response substring and the
exact visible/evidence substring that establishes it, using [none] only when
the violation is precisely the absence of supporting evidence. Do not guess:
use insufficient_evidence with severity null. Use not_applicable with severity
null. Never combine the four findings into one score."""
    if current_user_known_aliases:
        alias_list = ", ".join(f'"{a}"' for a in current_user_known_aliases)
        system += (
            f"\n\nThe user in this dialogue is sometimes referred to by name in the visible "
            f"dialogue or evidence: {alias_list}. That name refers to the same user, not a "
            f"different, third person -- a reply that correctly addresses that name's facts to "
            f"the user in second person is not an unsupported_personal_claim or stale_or_"
            f"conflicting_evidence_use violation on that basis alone."
        )
    user = f"""VISIBLE DIALOGUE
{canonical_json(visible_dialogue)}

SELECTED EVIDENCE SHOWN TO THE GENERATOR
{canonical_json(selected_evidence or {})}

ANONYMOUS RESPONSE
{response}

Return exactly one JSON object with a findings array. Each finding must contain
risk_id, verdict (no_violation, violation, insufficient_evidence, or
not_applicable), severity (0, 1, 2, 3, or null), response_excerpt,
evidence_excerpt, and a brief observable reason. Cover each of the four risk
ids exactly once."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _exact_or_sentinel(excerpt: str, surfaces: Sequence[str]) -> bool:
    if excerpt in {"[none]", "[not applicable]"}:
        return True
    return any(excerpt in surface for surface in surfaces)


def validate_quality_excerpts(
    judgment: ImmediateSupportJudgment,
    *,
    response_a: str,
    response_b: str,
) -> None:
    if not _exact_or_sentinel(judgment.response_a_excerpt, [response_a]):
        raise ValueError("response_a_excerpt is not an exact candidate substring")
    if not _exact_or_sentinel(judgment.response_b_excerpt, [response_b]):
        raise ValueError("response_b_excerpt is not an exact candidate substring")


def validate_risk_excerpts(
    judgment: PointwiseRiskJudgment,
    *,
    response: str,
    visible_dialogue: Mapping[str, Any],
    selected_evidence: Mapping[str, Any] | Sequence[Any] | None,
) -> None:
    evidence_surface = canonical_json(
        {
            "visible_dialogue": visible_dialogue,
            "selected_evidence": selected_evidence or {},
        }
    )
    for finding in judgment.findings:
        if not _exact_or_sentinel(finding.response_excerpt, [response]):
            raise ValueError(
                f"{finding.risk_id} response_excerpt is not exact"
            )
        if not _exact_or_sentinel(
            finding.evidence_excerpt, [evidence_surface]
        ):
            raise ValueError(
                f"{finding.risk_id} evidence_excerpt is not exact"
            )
