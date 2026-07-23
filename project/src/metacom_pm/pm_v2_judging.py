from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import inspect
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pydantic import Field

from .api import Endpoint, make_client
from .io import canonical_json, sha256_text
from .pm_v2_contracts import (
    ActionLabel,
    CompositeSpec,
    PMV2State,
    ResponseDimensions,
    RiskDimensions,
    StrictModel,
)
from .pm_v2_model import applicable_risk_fields


JUDGE_RUBRIC_VERSION = "pmv2-judge-v2-explicit-anchors"


RESPONSE_SCALE_ANCHORS = """Use the same explicit 1-5 anchors for every response dimension:
1 = major failure on that dimension; counterproductive, unsupported, or clearly harmful.
2 = substantial weaknesses that materially reduce the reply's value.
3 = adequate but mixed; acceptable with meaningful limitations.
4 = strong performance with only minor limitations.
5 = exceptional performance with no material limitation visible in the evidence.
Judge each dimension against its own definition; never copy one score into another."""

RISK_SCALE_ANCHORS = """Use the same explicit 0-3 anchors for every risk dimension:
0 = no observed issue.
1 = minor issue with little effect on the reply.
2 = material issue that noticeably harms the reply.
3 = major issue that is clearly harmful, misleading, or privacy-intrusive.
Score only observed evidence for that risk; never copy one risk score into another."""


def composite_weights_hash(spec: CompositeSpec) -> str:
    """Hash the actual weight mapping independently of names or versions."""

    return sha256_text(canonical_json(spec.weights))


def composite_spec_from_config(config: Mapping[str, Any]) -> CompositeSpec:
    """Build the one versioned quality composite authorized by PM-v2 YAML."""

    raw = config.get("quality_composite")
    if not isinstance(raw, Mapping):
        raise ValueError("PM-v2 config lacks quality_composite")
    weights = raw.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("PM-v2 quality_composite lacks weights")
    return CompositeSpec(
        version=str(raw.get("version") or ""),
        weights={str(key): float(value) for key, value in weights.items()},
    )


def labeling_settings_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated judge gates without exposing CLI threshold overrides."""

    raw = config.get("labeling")
    if not isinstance(raw, Mapping):
        raise ValueError("PM-v2 config lacks labeling")
    if bool(raw.get("request_llm_overall_field", True)):
        raise ValueError("PM-v2 labeling must keep request_llm_overall_field=false")
    maximum_correlation = float(raw["maximum_absolute_dimension_correlation"])
    if not 0.0 <= maximum_correlation <= 1.0:
        raise ValueError(
            "labeling.maximum_absolute_dimension_correlation must be in [0, 1]"
        )
    maximum_composite_support_correlation = float(
        raw["maximum_absolute_composite_support_correlation"]
    )
    if not 0.0 <= maximum_composite_support_correlation <= 1.0:
        raise ValueError(
            "labeling.maximum_absolute_composite_support_correlation must be in [0, 1]"
        )
    settings = {
        "minimum_families": int(raw["minimum_independent_judge_families"]),
        "reliable_mad_threshold": float(raw["reliable_mad_threshold"]),
        "minimum_reliable_rate": float(raw["minimum_reliable_label_rate"]),
        "minimum_low_mad_coverage_per_dimension": float(
            raw["minimum_low_mad_coverage_per_dimension"]
        ),
        "minimum_low_mad_coverage_per_action_dimension": float(
            raw["minimum_low_mad_coverage_per_action_dimension"]
        ),
        "duplicate_exact_match_rate": float(
            raw["reject_duplicate_dimension_exact_match_rate"]
        ),
        "maximum_absolute_dimension_correlation": maximum_correlation,
        "composite_support_exact_match_rate": float(
            raw["reject_composite_support_exact_match_rate"]
        ),
        "maximum_absolute_composite_support_correlation": (
            maximum_composite_support_correlation
        ),
        "reject_constant_response_dimensions": bool(
            raw["reject_constant_response_dimensions"]
        ),
        "reject_constant_risk_dimensions": bool(raw["reject_constant_risk_dimensions"]),
        "minimum_action_applicable_risk_signal_rate": float(
            raw["minimum_action_applicable_risk_signal_rate"]
        ),
        "minimum_action_applicable_risk_distinct_values": int(
            raw["minimum_action_applicable_risk_distinct_values"]
        ),
    }
    for name in (
        "reliable_mad_threshold",
        "minimum_reliable_rate",
        "duplicate_exact_match_rate",
        "composite_support_exact_match_rate",
        "minimum_low_mad_coverage_per_dimension",
        "minimum_low_mad_coverage_per_action_dimension",
        "minimum_action_applicable_risk_signal_rate",
    ):
        if not 0.0 <= float(settings[name]) <= 1.0:
            raise ValueError(f"labeling.{name} must be in [0, 1]")
    if settings["minimum_action_applicable_risk_distinct_values"] < 1:
        raise ValueError(
            "labeling.minimum_action_applicable_risk_distinct_values must be positive"
        )
    return settings


class ResponseJudgeOutput(StrictModel):
    emotional_support: float = Field(ge=1.0, le=5.0)
    personalization: float = Field(ge=1.0, le=5.0)
    memory_appropriateness: float = Field(ge=1.0, le=5.0)
    factual_grounding: float = Field(ge=1.0, le=5.0)
    temporal_consistency: float = Field(ge=1.0, le=5.0)
    non_intrusiveness: float = Field(ge=1.0, le=5.0)
    # No max_length: the real scientific bound on rationale length is the
    # judge call's own max_tokens (600 for response, 700 for risk); an
    # additional character cap here has no scientific meaning and only
    # produces spurious structured_output_validation_error failures on
    # otherwise-complete, real judge responses.
    rationale: str = Field(min_length=1)


class RiskJudgeOutput(StrictModel):
    selected_context_misuse: float = Field(ge=0.0, le=3.0)
    unnecessary_exposure: float = Field(ge=0.0, le=3.0)
    stale_or_conflicting_use: float = Field(ge=0.0, le=3.0)
    unsupported_personal_claim: float = Field(ge=0.0, le=3.0)
    memory_omission: float = Field(ge=0.0, le=3.0)
    strategy_overuse: float = Field(ge=0.0, le=3.0)
    strategy_omission: float = Field(ge=0.0, le=3.0)
    rationale: str = Field(min_length=1)


@dataclass(frozen=True)
class JudgeResult:
    family: str
    model: str
    response: ResponseJudgeOutput
    risk: RiskJudgeOutput
    response_request_hash: str
    risk_request_hash: str


def build_response_messages(
    *,
    state: PMV2State,
    authorized_user_context: str,
    candidate_response: str,
) -> list[dict[str, str]]:
    system = f"""You are an impartial evaluator of one emotional-support reply.
Score six independent dimensions. Do not infer which policy generated the reply.
Do not produce an overall score. Treat each dimension independently and use the
full 1-5 scale. Authorized user context is reference material for checking
personalization, grounding and temporal consistency; it is not evidence that the
generator actually saw that context.

{RESPONSE_SCALE_ANCHORS}"""
    user = f"""CURRENT SEEKER TURN
{state.current_user_text}

RECENT DIALOGUE
{canonical_json([turn.model_dump(mode='json') for turn in state.current_session_history])}

CURRENT SESSION SUMMARY
{state.current_session_summary or '[none]'}

AUTHORIZED USER CONTEXT
{authorized_user_context}

ANONYMOUS SUPPORTER RESPONSE
{candidate_response}

SCORING RULES
- emotional_support: empathy, validation, emotional attunement, and helpful support.
- personalization: specifically tailored to the seeker's current situation and known history; generic warmth alone is not personalization.
- memory_appropriateness: any use or non-use of history is appropriate. Penalize both unjustified recall and omission of clearly material history.
- factual_grounding: claims about the user are supported by the authorized context or current dialogue.
- temporal_consistency: events and current states are represented in the correct order and tense.
- non_intrusiveness: does not expose irrelevant private details, over-personalize, or pressure the seeker.
Return exactly one JSON object with these keys and no others:
{{"emotional_support": number, "personalization": number,
"memory_appropriateness": number, "factual_grounding": number,
"temporal_consistency": number, "non_intrusiveness": number,
"rationale": "brief evidence-based explanation"}}
Do not add `overall`."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_risk_messages(
    *,
    state: PMV2State,
    authorized_user_context: str,
    selected_context: str,
    candidate_response: str,
) -> list[dict[str, str]]:
    system = f"""You are an evidence-risk auditor for emotional-support dialogue.
The resource policy and action name are hidden. Score observed risks on 0-3,
using the explicit anchors below. Distinguish selected-context quality from actual
misuse in the final response. Do not reward or penalize token cost; cost is
measured separately from logs.

{RISK_SCALE_ANCHORS}"""
    user = f"""CURRENT SEEKER TURN
{state.current_user_text}

RECENT DIALOGUE
{canonical_json([turn.model_dump(mode='json') for turn in state.current_session_history])}

CURRENT SESSION SUMMARY
{state.current_session_summary or '[none]'}

AUTHORIZED USER CONTEXT
{authorized_user_context}

SELECTED CONTEXT SHOWN TO GENERATOR
{selected_context or '[none]'}

ANONYMOUS SUPPORTER RESPONSE
{candidate_response}

RISK RULES
- selected_context_misuse: selected evidence is actually misapplied in the response.
- unnecessary_exposure: irrelevant or overly private information is surfaced.
- stale_or_conflicting_use: outdated or contradicted information affects the response.
- unsupported_personal_claim: the response invents personal facts.
- memory_omission: clearly material available history was omitted and the omission harms the reply. Do not penalize harmless non-use.
- strategy_overuse: retrieved support guidance makes the reply formulaic, premature, overly directive, or repetitive.
- strategy_omission: strategy guidance was clearly needed and its absence harms the reply.
Return exactly one JSON object with these keys and no others:
{{"selected_context_misuse": number, "unnecessary_exposure": number,
"stale_or_conflicting_use": number, "unsupported_personal_claim": number,
"memory_omission": number, "strategy_overuse": number,
"strategy_omission": number, "rationale": "brief evidence-based explanation"}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_one(
    *,
    endpoint: Endpoint,
    state: PMV2State,
    authorized_user_context: str,
    selected_context: str,
    candidate_response: str,
    seed: int = 17,
    response_max_output_tokens: int = 600,
    risk_max_output_tokens: int = 700,
) -> JudgeResult:
    if not endpoint.family:
        raise ValueError("judge endpoint must declare a model family")
    if response_max_output_tokens <= 0 or risk_max_output_tokens <= 0:
        raise ValueError("judge max-output-token limits must be positive")
    client = make_client(endpoint)
    try:
        response_messages = build_response_messages(
            state=state,
            authorized_user_context=authorized_user_context,
            candidate_response=candidate_response,
        )
        response_call, response_score = client.chat(
            response_messages,
            temperature=0.0,
            max_tokens=int(response_max_output_tokens),
            seed=seed,
            response_schema=ResponseJudgeOutput,
            retries=1,
        )
        assert response_score is not None
        risk_messages = build_risk_messages(
            state=state,
            authorized_user_context=authorized_user_context,
            selected_context=selected_context,
            candidate_response=candidate_response,
        )
        risk_call, risk_score = client.chat(
            risk_messages,
            temperature=0.0,
            max_tokens=int(risk_max_output_tokens),
            seed=seed + 1,
            response_schema=RiskJudgeOutput,
            retries=1,
        )
        assert risk_score is not None
        return JudgeResult(
            family=endpoint.family,
            model=endpoint.model,
            response=response_score,
            risk=risk_score,
            response_request_hash=response_call.request_hash,
            risk_request_hash=risk_call.request_hash,
        )
    finally:
        client.close()


def _median_model(cls, outputs: Sequence[StrictModel], fields: Iterable[str]):
    values = {
        name: float(median(float(getattr(output, name)) for output in outputs))
        for name in fields
    }
    return cls(**values)


def aggregate_judges(
    results: Sequence[JudgeResult],
    *,
    minimum_families: int = 2,
    reliable_mad_threshold: float = 0.75,
) -> tuple[ResponseDimensions, RiskDimensions, dict[str, Any]]:
    if len(results) < minimum_families:
        raise ValueError(f"need at least {minimum_families} independent judge results")
    families = {result.family for result in results}
    if len(families) < minimum_families:
        raise ValueError("judge results do not contain enough independent model families")
    response_fields = tuple(ResponseDimensions.model_fields)
    risk_fields = tuple(RiskDimensions.model_fields)
    response = _median_model(
        ResponseDimensions, [result.response for result in results], response_fields
    )
    risk = _median_model(RiskDimensions, [result.risk for result in results], risk_fields)
    mads: dict[str, float] = {}
    for name in response_fields:
        values = np.asarray([float(getattr(result.response, name)) for result in results])
        mads[f"response.{name}"] = float(np.median(np.abs(values - np.median(values))))
    for name in risk_fields:
        values = np.asarray([float(getattr(result.risk, name)) for result in results])
        mads[f"risk.{name}"] = float(np.median(np.abs(values - np.median(values))))
    max_mad = max(mads.values(), default=0.0)
    audit = {
        "judge_count": len(results),
        "judge_families": sorted(families),
        "judge_models": sorted({result.model for result in results}),
        "dimension_mad": mads,
        "max_dimension_mad": max_mad,
        "label_reliable": max_mad <= reliable_mad_threshold,
        "response_request_hashes": sorted(result.response_request_hash for result in results),
        "risk_request_hashes": sorted(result.risk_request_hash for result in results),
    }
    return response, risk, audit


def build_action_label(
    *,
    state: PMV2State,
    action_id: str,
    observed_input_tokens: int,
    retrieval_calls: int,
    results: Sequence[JudgeResult],
    composite_spec: CompositeSpec,
    minimum_families: int = 2,
    reliable_mad_threshold: float = 0.75,
    provenance: dict[str, Any] | None = None,
) -> ActionLabel:
    spec = composite_spec
    response, risk, audit = aggregate_judges(
        results,
        minimum_families=minimum_families,
        reliable_mad_threshold=reliable_mad_threshold,
    )
    return ActionLabel(
        state_id=state.state_id,
        card_id=state.card_id,
        user_id=state.user_id,
        semantic_family=state.semantic_family,
        action_id=action_id,
        response=response,
        risk=risk,
        observed_input_tokens=observed_input_tokens,
        retrieval_calls=retrieval_calls,
        judge_families=audit["judge_families"],
        judge_count=audit["judge_count"],
        max_dimension_mad=audit["max_dimension_mad"],
        dimension_mad=audit["dimension_mad"],
        label_reliable=audit["label_reliable"],
        composite_spec_version=spec.version,
        composite_weights_sha256=composite_weights_hash(spec),
        provenance={**(provenance or {}), "judge_audit": audit},
    )


def dimension_applicability_by_action(
    action_ids: Iterable[str],
) -> dict[str, frozenset[str]]:
    """Per-action risk dimensions structurally inapplicable to that action.

    Reuses applicable_risk_fields (pm_v2_model.py) -- the same authoritative,
    already-tested source validate_action_applicable_risk_signal relies on --
    rather than a second, hand-maintained notion of applicability. A
    structural zero (e.g. selected_context_misuse when no memory source was
    ever selected) is not a judge defect and must not be flagged as one by
    the constant/duplicate/correlation/low-MAD-coverage checks below.
    """

    all_risk_fields = frozenset(RiskDimensions.model_fields)
    return {
        str(action_id): frozenset(
            f"risk.{name}"
            for name in all_risk_fields - frozenset(applicable_risk_fields(action_id))
        )
        for action_id in action_ids
    }


def dimensions_inapplicable_to_every_action(
    action_ids: Iterable[str],
) -> frozenset[str]:
    """Risk dimensions inapplicable to *every* action in the given set.

    For a global (not per-action) check: only exclude a dimension when it is
    structurally meaningless across the *entire* evaluated action set, never
    merely inapplicable to some of it.
    """

    by_action = dimension_applicability_by_action(action_ids)
    if not by_action:
        return frozenset()
    return frozenset.intersection(*by_action.values())


DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL = (
    "pm-v1.5-dimension-applicability-contract-v1"
)


def dimension_applicability_contract_sha256(action_ids: Iterable[str]) -> str:
    """Single, hashable identity for the applicability contract in force.

    Deliberately derived, not hand-maintained: two different action sets
    that resolve to the same inapplicable-dimension mapping get the same
    hash, and any future change to applicable_risk_fields's own logic
    changes this hash automatically, so a downstream consumer (attestation,
    preflight) can bind to "which contract was in effect" without needing
    its own copy of the exclusion list.
    """

    by_action = dimension_applicability_by_action(sorted({str(a) for a in action_ids}))
    payload = {
        "protocol": DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL,
        "inapplicable_risk_dimensions_by_action": {
            action_id: sorted(dims) for action_id, dims in sorted(by_action.items())
        },
    }
    return sha256_text(canonical_json(payload))


def _drop_inapplicable(
    *,
    constants: list[str],
    duplicate_pairs: list[dict[str, Any]],
    correlation_pairs: list[dict[str, Any]],
    inapplicable_dimensions: frozenset[str],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop entries that only implicate a declared-inapplicable dimension.

    A pair is dropped only when *both* sides are inapplicable (or one side is
    inapplicable and the other is a genuine constant already excluded) --
    never when a *real*, applicable dimension merely correlates with an
    inapplicable one, since that would hide a real defect in the applicable
    dimension behind an unrelated exclusion.
    """

    if not inapplicable_dimensions:
        return constants, duplicate_pairs, correlation_pairs
    filtered_constants = [name for name in constants if name not in inapplicable_dimensions]
    filtered_duplicates = [
        pair
        for pair in duplicate_pairs
        if pair["left"] not in inapplicable_dimensions
        and pair["right"] not in inapplicable_dimensions
    ]
    filtered_correlations = [
        pair
        for pair in correlation_pairs
        if pair["left"] not in inapplicable_dimensions
        and pair["right"] not in inapplicable_dimensions
    ]
    return filtered_constants, filtered_duplicates, filtered_correlations


def dimension_health(
    matrix: np.ndarray,
    field_names: Sequence[str],
    *,
    prefix: str,
    duplicate_exact_match_rate: float,
    maximum_absolute_dimension_correlation: float,
    minimum_nonzero_observations: int = 0,
    action_ids: Sequence[str] | None = None,
    inapplicable_risk_dimensions_by_action: Mapping[str, frozenset[str]] | None = None,
    split_correlation_by_sign: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """minimum_nonzero_observations (default 0, preserving exact prior
    behavior for every existing caller): two sparse, mostly-zero dimensions
    can reach a high exact_match_rate purely because they agree on the
    trivial zero-zero case, with no real evidence either way that one copies
    the other. When positive, the duplicate/correlation *determination* uses
    only the informative subset (rows where at least one side is nonzero),
    not the full matrix -- computing exact-match/correlation over the full
    matrix would still let a mountain of trivial zero-zero rows manufacture
    a false positive even once enough nonzero rows exist. Both the overall
    (full-matrix) and informative-only statistics are always reported; only
    informative-only ones ever drive the duplicate/high-correlation verdict.
    Below the row-count floor, the pair is reported as insufficient evidence,
    never silently treated as either duplicate or independent.

    action_ids/inapplicable_risk_dimensions_by_action (both default None,
    preserving exact prior behavior when omitted): a risk dimension can be
    structurally inapplicable to only *some* actions (e.g. strategy_overuse
    only applies to RS actions). A row where either side of a pair is
    inapplicable to that row's own action carries no evidence about whether
    the *pair* is duplicated, so it is excluded before the nonzero
    ("informative") filter above is even applied -- "informative" therefore
    means "pairwise-applicable and nonzero on at least one side", not merely
    "nonzero", whenever these are provided.

    split_correlation_by_sign (default False, preserving exact prior
    behavior): a strong *positive* correlation between two dimensions is
    evidence one is a copy of the other. A strong *negative* correlation is
    evidence of near-mutual-exclusivity between two independently-scored
    constructs (e.g. an action can be "overused" or "omitted" but rarely
    both) -- a different phenomenon, not a duplicate-judge defect. When True,
    only positive high-correlation pairs are returned as high_correlation_
    pairs (gate-eligible); negative ones are returned separately as
    mutual_exclusivity_diagnostic_pairs (always reported, never gates).
    """

    duplicate_pairs: list[dict[str, Any]] = []
    high_correlation_pairs: list[dict[str, Any]] = []
    mutual_exclusivity_diagnostic_pairs: list[dict[str, Any]] = []
    for left in range(len(field_names)):
        for right in range(left + 1, len(field_names)):
            left_name = f"{prefix}.{field_names[left]}"
            right_name = f"{prefix}.{field_names[right]}"
            if action_ids is not None and inapplicable_risk_dimensions_by_action is not None:
                pairwise_applicable = np.asarray(
                    [
                        left_name
                        not in inapplicable_risk_dimensions_by_action.get(
                            str(action_id), frozenset()
                        )
                        and right_name
                        not in inapplicable_risk_dimensions_by_action.get(
                            str(action_id), frozenset()
                        )
                        for action_id in action_ids
                    ],
                    dtype=bool,
                )
            else:
                pairwise_applicable = np.ones(matrix.shape[0], dtype=bool)
            n_pairwise_applicable = int(np.sum(pairwise_applicable))
            informative = pairwise_applicable & (
                (matrix[:, left] != 0.0) | (matrix[:, right] != 0.0)
            )
            n_informative = int(np.sum(informative))
            insufficient_evidence = n_informative < minimum_nonzero_observations
            overall_exact_rate = float(np.mean(matrix[:, left] == matrix[:, right]))
            if n_informative > 0:
                informative_exact_rate = float(
                    np.mean(
                        matrix[informative, left] == matrix[informative, right]
                    )
                )
            else:
                informative_exact_rate = None
            left_std = float(np.std(matrix[:, left]))
            right_std = float(np.std(matrix[:, right]))
            overall_correlation: float | None = None
            if left_std >= 1e-9 and right_std >= 1e-9:
                overall_correlation = float(
                    np.corrcoef(matrix[:, left], matrix[:, right])[0, 1]
                )
            informative_correlation: float | None = None
            if n_informative >= 2:
                left_info = matrix[informative, left]
                right_info = matrix[informative, right]
                if (
                    float(np.std(left_info)) >= 1e-9
                    and float(np.std(right_info)) >= 1e-9
                ):
                    informative_correlation = float(
                        np.corrcoef(left_info, right_info)[0, 1]
                    )
            pair = {
                "left": left_name,
                "right": right_name,
                "overall_exact_match_rate": overall_exact_rate,
                "informative_exact_match_rate": informative_exact_rate,
                "overall_correlation": overall_correlation,
                "informative_correlation": informative_correlation,
                "informative_rows": n_informative,
                "pairwise_applicable_rows": n_pairwise_applicable,
                "insufficient_evidence": insufficient_evidence,
                # Legacy keys, computed over the full matrix exactly as
                # before, kept for any reader that predates the
                # overall/informative split (e.g. minimum_nonzero_
                # observations=0 callers, whose verdict is unaffected).
                "exact_match_rate": overall_exact_rate,
                "correlation": overall_correlation,
            }
            if not insufficient_evidence and (
                informative_exact_rate
                if minimum_nonzero_observations > 0
                else overall_exact_rate
            ) is not None and (
                (
                    informative_exact_rate
                    if minimum_nonzero_observations > 0
                    else overall_exact_rate
                )
                >= duplicate_exact_match_rate
            ):
                duplicate_pairs.append(pair)
            verdict_correlation = (
                informative_correlation
                if minimum_nonzero_observations > 0
                else overall_correlation
            )
            if not insufficient_evidence and verdict_correlation is not None:
                if split_correlation_by_sign:
                    if verdict_correlation >= maximum_absolute_dimension_correlation:
                        high_correlation_pairs.append(pair)
                    elif verdict_correlation <= -maximum_absolute_dimension_correlation:
                        mutual_exclusivity_diagnostic_pairs.append(pair)
                elif abs(verdict_correlation) >= maximum_absolute_dimension_correlation:
                    high_correlation_pairs.append(pair)
    constants = [
        f"{prefix}.{field_names[index]}"
        for index in range(len(field_names))
        if float(np.std(matrix[:, index])) < 1e-9
    ]
    prevalence = {
        f"{prefix}.{field_names[index]}": {
            "mean": float(np.mean(matrix[:, index])),
            "std": float(np.std(matrix[:, index])),
            "nonzero_rate": float(np.mean(matrix[:, index] != 0.0)),
            "unique_values": sorted(float(value) for value in np.unique(matrix[:, index])),
        }
        for index in range(len(field_names))
    }
    return (
        duplicate_pairs,
        high_correlation_pairs,
        constants,
        prevalence,
        mutual_exclusivity_diagnostic_pairs,
    )


def validate_raw_judge_family_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_families: Sequence[str],
    duplicate_exact_match_rate: float = 0.98,
    maximum_absolute_dimension_correlation: float = 0.95,
    composite_support_exact_match_rate: float = 0.98,
    maximum_absolute_composite_support_correlation: float = 0.995,
    reject_constant_response_dimensions: bool = True,
    reject_constant_risk_dimensions: bool = True,
    check_risk_dimension_health: bool = True,
    inapplicable_risk_dimensions: frozenset[str] = frozenset(),
    inapplicable_risk_dimensions_by_action: Mapping[str, frozenset[str]] | None = None,
    minimum_nonzero_observations: int = 0,
    split_correlation_by_sign: bool = False,
    composite_spec: CompositeSpec | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Reject a degenerate judge family before cross-family aggregation.

    Aggregate medians can conceal a family that copies dimensions or emits a
    constant score table.  Every configured family therefore has to pass the
    same construct-independence checks on its own raw structured outputs.
    Canonical input rows contain ``judge_family``, ``response`` and ``risk``;
    the latter two are validated against the strict judge schemas. An
    ``action_id`` is used, when present, to restrict pairwise risk-dimension
    comparisons to rows where the row's own action makes both dimensions
    applicable (see dimension_health); rows without one are treated as
    applicable to every pair, matching prior behavior exactly.
    """

    expected = {str(value) for value in expected_families}
    if not expected:
        raise ValueError("raw judge-family health requires expected families")
    grouped: dict[str, list[tuple[ResponseJudgeOutput, RiskJudgeOutput, str]]] = defaultdict(
        list
    )
    for index, row in enumerate(rows):
        family = str(row.get("judge_family") or "")
        if not family:
            raise ValueError(f"raw judge-family row {index} lacks judge_family")
        if family not in expected:
            raise ValueError(f"unexpected raw judge family: {family}")
        grouped[family].append(
            (
                ResponseJudgeOutput.model_validate(row.get("response")),
                RiskJudgeOutput.model_validate(row.get("risk")),
                str(row.get("action_id") or ""),
            )
        )
    missing_families = sorted(expected - set(grouped))
    spec = composite_spec or CompositeSpec()
    response_fields = tuple(ResponseDimensions.model_fields)
    risk_fields = tuple(RiskDimensions.model_fields)
    family_reports: dict[str, dict[str, Any]] = {}
    for family in sorted(grouped):
        outputs = grouped[family]
        action_ids = [action_id for _, _, action_id in outputs]
        response_matrix = np.asarray(
            [
                [float(getattr(response, name)) for name in response_fields]
                for response, _, _ in outputs
            ],
            dtype=float,
        )
        risk_matrix = np.asarray(
            [
                [float(getattr(risk, name)) for name in risk_fields]
                for _, risk, _ in outputs
            ],
            dtype=float,
        )
        (
            response_duplicates,
            response_correlations,
            response_constants,
            response_prevalence,
            _response_mutual_exclusivity,
        ) = dimension_health(
            response_matrix,
            response_fields,
            prefix="response",
            duplicate_exact_match_rate=duplicate_exact_match_rate,
            maximum_absolute_dimension_correlation=(
                maximum_absolute_dimension_correlation
            ),
            minimum_nonzero_observations=minimum_nonzero_observations,
            split_correlation_by_sign=split_correlation_by_sign,
        )
        (
            risk_duplicates,
            risk_correlations,
            risk_constants,
            risk_prevalence,
            risk_mutual_exclusivity,
        ) = dimension_health(
            risk_matrix,
            risk_fields,
            prefix="risk",
            duplicate_exact_match_rate=duplicate_exact_match_rate,
            maximum_absolute_dimension_correlation=(
                maximum_absolute_dimension_correlation
            ),
            minimum_nonzero_observations=minimum_nonzero_observations,
            action_ids=action_ids,
            inapplicable_risk_dimensions_by_action=inapplicable_risk_dimensions_by_action,
            split_correlation_by_sign=split_correlation_by_sign,
        )
        composite_values = np.asarray(
            [
                spec.score(
                    ResponseDimensions(
                        **{
                            name: float(getattr(response, name))
                            for name in response_fields
                        }
                    )
                )
                for response, _, _ in outputs
            ],
            dtype=float,
        )
        support_values = (
            response_matrix[:, response_fields.index("emotional_support")] - 1.0
        ) / 4.0
        composite_support_match_rate = float(
            np.mean(
                np.isclose(
                    composite_values, support_values, rtol=0.0, atol=1e-12
                )
            )
        )
        composite_support_correlation: float | None = None
        if (
            float(np.std(composite_values)) >= 1e-9
            and float(np.std(support_values)) >= 1e-9
        ):
            composite_support_correlation = float(
                np.corrcoef(composite_values, support_values)[0, 1]
            )
        duplicate_pairs = [
            *response_duplicates,
            *(risk_duplicates if check_risk_dimension_health else []),
        ]
        high_correlation_pairs = [
            *response_correlations,
            *(risk_correlations if check_risk_dimension_health else []),
        ]
        mutual_exclusivity_diagnostic_pairs = [
            *_response_mutual_exclusivity,
            *(risk_mutual_exclusivity if check_risk_dimension_health else []),
        ]
        constant_dimensions = [
            *(response_constants if reject_constant_response_dimensions else []),
            *(
                risk_constants
                if check_risk_dimension_health
                and reject_constant_risk_dimensions
                else []
            ),
        ]
        constant_dimensions, duplicate_pairs, high_correlation_pairs = _drop_inapplicable(
            constants=constant_dimensions,
            duplicate_pairs=duplicate_pairs,
            correlation_pairs=high_correlation_pairs,
            inapplicable_dimensions=inapplicable_risk_dimensions,
        )
        exact_failure = (
            composite_support_match_rate >= composite_support_exact_match_rate
        )
        correlation_failure = (
            composite_support_correlation is not None
            and abs(composite_support_correlation)
            >= maximum_absolute_composite_support_correlation
        )
        failed = bool(
            duplicate_pairs
            or high_correlation_pairs
            or constant_dimensions
            or exact_failure
            or correlation_failure
        )
        family_reports[family] = {
            "status": "FAIL" if failed else "PASS",
            "n": len(outputs),
            "duplicate_dimension_pairs": duplicate_pairs,
            "high_correlation_dimension_pairs": high_correlation_pairs,
            "mutual_exclusivity_diagnostic_pairs": mutual_exclusivity_diagnostic_pairs,
            "constant_dimensions": constant_dimensions,
            "response_dimension_prevalence": response_prevalence,
            "risk_dimension_prevalence": risk_prevalence,
            "composite_support_exact_match_rate": (
                composite_support_match_rate
            ),
            "composite_support_correlation": composite_support_correlation,
            "composite_support_exact_match_failure": exact_failure,
            "composite_support_correlation_failure": correlation_failure,
        }
    failed_families = sorted(
        family
        for family, report in family_reports.items()
        if report["status"] != "PASS"
    )
    report = {
        "status": (
            "PASS" if not missing_families and not failed_families else "FAIL"
        ),
        "expected_families": sorted(expected),
        "observed_families": sorted(grouped),
        "missing_families": missing_families,
        "failed_families": failed_families,
        "thresholds": {
            "duplicate_exact_match_rate": duplicate_exact_match_rate,
            "maximum_absolute_dimension_correlation": (
                maximum_absolute_dimension_correlation
            ),
            "composite_support_exact_match_rate": (
                composite_support_exact_match_rate
            ),
            "maximum_absolute_composite_support_correlation": (
                maximum_absolute_composite_support_correlation
            ),
            "reject_constant_response_dimensions": (
                reject_constant_response_dimensions
            ),
            "reject_constant_risk_dimensions": reject_constant_risk_dimensions,
            "check_risk_dimension_health": check_risk_dimension_health,
            "inapplicable_risk_dimensions": sorted(inapplicable_risk_dimensions),
            "inapplicable_risk_dimensions_by_action": {
                action_id: sorted(dims)
                for action_id, dims in sorted(
                    (inapplicable_risk_dimensions_by_action or {}).items()
                )
            },
            "split_correlation_by_sign": split_correlation_by_sign,
        },
        "dimension_applicability_contract_protocol": (
            DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL
        ),
        "composite_spec_version": spec.version,
        "composite_weights_sha256": composite_weights_hash(spec),
        "families": family_reports,
    }
    if report["status"] != "PASS" and raise_on_failure:
        raise RuntimeError(
            "PM-v2 raw judge-family quality gate failed: "
            + canonical_json(report)
        )
    return report


def validate_raw_judge_family_subgroup_health(
    rows: Sequence[Mapping[str, Any]],
    *,
    subgroup_key: str,
    expected_subgroups: Sequence[str],
    expected_families: Sequence[str],
    duplicate_exact_match_rate: float = 0.98,
    maximum_absolute_dimension_correlation: float = 0.95,
    composite_support_exact_match_rate: float = 0.98,
    maximum_absolute_composite_support_correlation: float = 0.995,
    reject_constant_response_dimensions: bool = True,
    reject_constant_risk_dimensions: bool = True,
    check_risk_dimension_health: bool = False,
    composite_spec: CompositeSpec | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Run response/composite family health inside every frozen subgroup.

    Risk applicability is action/condition dependent.  In particular, a valid
    no-resource action can make every resource-misuse risk structurally zero.
    The complete risk duplicate/correlation/constant checks therefore belong
    to the global family gate, while subgroup gates default to response copying
    and composite-vs-support checks only.
    """

    expected = {str(value) for value in expected_subgroups}
    observed = {str(row.get(subgroup_key) or "") for row in rows}
    if "" in observed:
        raise ValueError(f"raw judge row lacks subgroup key: {subgroup_key}")
    unexpected = sorted(observed - expected)
    if unexpected:
        raise ValueError(
            f"unexpected raw judge {subgroup_key} subgroups: {unexpected}"
        )
    subgroup_reports = {
        subgroup: validate_raw_judge_family_health(
            [
                row
                for row in rows
                if str(row.get(subgroup_key)) == subgroup
            ],
            expected_families=expected_families,
            duplicate_exact_match_rate=duplicate_exact_match_rate,
            maximum_absolute_dimension_correlation=(
                maximum_absolute_dimension_correlation
            ),
            composite_support_exact_match_rate=composite_support_exact_match_rate,
            maximum_absolute_composite_support_correlation=(
                maximum_absolute_composite_support_correlation
            ),
            reject_constant_response_dimensions=reject_constant_response_dimensions,
            reject_constant_risk_dimensions=reject_constant_risk_dimensions,
            check_risk_dimension_health=check_risk_dimension_health,
            composite_spec=composite_spec,
            raise_on_failure=False,
        )
        for subgroup in sorted(expected)
    }
    failed_subgroups = sorted(
        subgroup
        for subgroup, report in subgroup_reports.items()
        if report.get("status") != "PASS"
    )
    report = {
        "status": "PASS" if not failed_subgroups else "FAIL",
        "subgroup_key": subgroup_key,
        "expected_subgroups": sorted(expected),
        "observed_subgroups": sorted(observed),
        "failed_subgroups": failed_subgroups,
        "check_risk_dimension_health": check_risk_dimension_health,
        "subgroups": subgroup_reports,
    }
    if failed_subgroups and raise_on_failure:
        raise RuntimeError(
            "PM-v2 raw judge-family subgroup quality gate failed: "
            + canonical_json(report)
        )
    return report


def validate_action_applicable_risk_signal(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_actions: Sequence[str],
    expected_families: Sequence[str],
    minimum_signal_rate: float,
    minimum_distinct_values: int,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Gate action-local risk measurement without testing inapplicable fields.

    A legitimate action can make several risk dimensions structural zeros.  The
    gate therefore evaluates only ``applicable_risk_fields(action)`` and asks
    for some nonzero/variable signal in each action × judge-family panel.  It
    does not require every applicable dimension to be nonzero.
    """

    if not 0.0 <= float(minimum_signal_rate) <= 1.0:
        raise ValueError("minimum_signal_rate must be in [0, 1]")
    if int(minimum_distinct_values) < 1:
        raise ValueError("minimum_distinct_values must be positive")
    expected_action_set = {str(value) for value in expected_actions}
    expected_family_set = {str(value) for value in expected_families}
    panels: dict[str, Any] = {}
    failed: list[str] = []
    for action_id in sorted(expected_action_set):
        fields = applicable_risk_fields(action_id)
        for family in sorted(expected_family_set):
            panel_rows = [
                row
                for row in rows
                if str(row.get("action_id") or "") == action_id
                and str(row.get("judge_family") or "") == family
            ]
            key = f"{action_id}::{family}"
            if not panel_rows:
                panels[key] = {
                    "status": "FAIL",
                    "n": 0,
                    "applicable_fields": list(fields),
                    "reason": "missing_panel",
                }
                failed.append(key)
                continue
            matrix = np.asarray(
                [
                    [float((row.get("risk") or {})[field]) for field in fields]
                    for row in panel_rows
                ],
                dtype=float,
            )
            signal_rate = float(np.mean(matrix > 0.0))
            maximum_distinct = max(
                (len(set(matrix[:, index].tolist())) for index in range(matrix.shape[1])),
                default=0,
            )
            checks = {
                "minimum_signal_rate": signal_rate >= float(minimum_signal_rate),
                "minimum_distinct_values": maximum_distinct
                >= int(minimum_distinct_values),
            }
            status = "PASS" if all(checks.values()) else "FAIL"
            panels[key] = {
                "status": status,
                "n": len(panel_rows),
                "applicable_fields": list(fields),
                "signal_rate": signal_rate,
                "maximum_distinct_values": maximum_distinct,
                "field_nonzero_rates": {
                    field: float(np.mean(matrix[:, index] > 0.0))
                    for index, field in enumerate(fields)
                },
                "field_distinct_values": {
                    field: len(set(matrix[:, index].tolist()))
                    for index, field in enumerate(fields)
                },
                "checks": checks,
            }
            if status != "PASS":
                failed.append(key)
    unexpected_actions = sorted(
        {str(row.get("action_id") or "") for row in rows} - expected_action_set
    )
    report = {
        "status": "PASS" if not failed and not unexpected_actions else "FAIL",
        "protocol": "pm-v2-action-applicable-risk-signal-v1",
        "minimum_signal_rate": float(minimum_signal_rate),
        "minimum_distinct_values": int(minimum_distinct_values),
        "failed_panels": failed,
        "unexpected_actions": unexpected_actions,
        "panels": panels,
    }
    if report["status"] != "PASS" and raise_on_failure:
        raise RuntimeError(
            "PM-v2 action-applicable risk signal gate failed: "
            + canonical_json(report)
        )
    return report


def validate_judge_table(
    labels: Sequence[ActionLabel],
    *,
    minimum_families: int = 2,
    minimum_reliable_rate: float = 0.80,
    reliable_mad_threshold: float = 0.75,
    minimum_low_mad_coverage_per_dimension: float = 0.80,
    minimum_low_mad_coverage_per_action_dimension: float = 0.80,
    duplicate_exact_match_rate: float = 0.98,
    maximum_absolute_dimension_correlation: float = 0.95,
    composite_support_exact_match_rate: float = 0.98,
    maximum_absolute_composite_support_correlation: float = 0.995,
    reject_constant_response_dimensions: bool = True,
    reject_constant_risk_dimensions: bool = True,
    inapplicable_risk_dimensions: frozenset[str] = frozenset(),
    inapplicable_risk_dimensions_by_action: Mapping[str, frozenset[str]] | None = None,
    minimum_nonzero_observations: int = 0,
    split_correlation_by_sign: bool = False,
    composite_spec: CompositeSpec | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Fail closed on duplicated, constant, or unreliable response/risk labels.

    inapplicable_risk_dimensions excludes a risk dimension (e.g. "risk.
    selected_context_misuse") from the *global* constant/duplicate/
    correlation/low-MAD-coverage checks when it is structurally inapplicable
    to every action present -- a structural zero (no memory source was ever
    selected) is not a judge defect. inapplicable_risk_dimensions_by_action
    additionally excludes a "{action_id}/{dimension}" combination from the
    *per-action* low-MAD-coverage check when that dimension is inapplicable
    to that specific action, even if applicable to others in the same table.
    """

    if not labels:
        raise ValueError("empty judge table")
    response_fields = tuple(ResponseDimensions.model_fields)
    risk_fields = tuple(RiskDimensions.model_fields)
    response_matrix = np.asarray(
        [[float(getattr(label.response, name)) for name in response_fields] for label in labels],
        dtype=float,
    )
    risk_matrix = np.asarray(
        [[float(getattr(label.risk, name)) for name in risk_fields] for label in labels],
        dtype=float,
    )
    action_ids = [label.action_id for label in labels]
    (
        response_duplicates,
        response_correlations,
        response_constants,
        response_prevalence,
        _response_mutual_exclusivity,
    ) = dimension_health(
        response_matrix,
        response_fields,
        prefix="response",
        duplicate_exact_match_rate=duplicate_exact_match_rate,
        maximum_absolute_dimension_correlation=maximum_absolute_dimension_correlation,
        minimum_nonzero_observations=minimum_nonzero_observations,
        split_correlation_by_sign=split_correlation_by_sign,
    )
    (
        risk_duplicates,
        risk_correlations,
        risk_constants,
        risk_prevalence,
        risk_mutual_exclusivity,
    ) = dimension_health(
        risk_matrix,
        risk_fields,
        prefix="risk",
        duplicate_exact_match_rate=duplicate_exact_match_rate,
        minimum_nonzero_observations=minimum_nonzero_observations,
        maximum_absolute_dimension_correlation=maximum_absolute_dimension_correlation,
        action_ids=action_ids,
        inapplicable_risk_dimensions_by_action=inapplicable_risk_dimensions_by_action,
        split_correlation_by_sign=split_correlation_by_sign,
    )
    reliable_rate = float(np.mean([label.label_reliable for label in labels]))
    dimension_names = [
        *[f"response.{name}" for name in response_fields],
        *[f"risk.{name}" for name in risk_fields],
    ]
    inapplicable_by_action = inapplicable_risk_dimensions_by_action or {}
    # A dimension/cell that is structurally inapplicable (globally, or to one
    # specific action) is reported as an explicit N/A (None), never as a
    # computed number: a real coverage figure here would either look like a
    # spurious failure or, worse, a spuriously perfect value that could help
    # an unrelated real defect slip past the gate.
    dimension_low_mad_coverage: dict[str, float | None] = {}
    for name in dimension_names:
        if name in inapplicable_risk_dimensions:
            dimension_low_mad_coverage[name] = None
            continue
        dimension_low_mad_coverage[name] = float(
            np.mean(
                [
                    float(label.dimension_mad[name]) <= reliable_mad_threshold
                    for label in labels
                ]
            )
        )
    action_dimension_low_mad_coverage: dict[str, dict[str, float | None]] = {}
    for action_id in sorted({label.action_id for label in labels}):
        action_labels = [label for label in labels if label.action_id == action_id]
        inapplicable_for_action = inapplicable_by_action.get(action_id, frozenset())
        row: dict[str, float | None] = {}
        for name in dimension_names:
            if name in inapplicable_for_action:
                row[name] = None
                continue
            row[name] = float(
                np.mean(
                    [
                        float(label.dimension_mad[name]) <= reliable_mad_threshold
                        for label in action_labels
                    ]
                )
            )
        action_dimension_low_mad_coverage[action_id] = row
    low_coverage_dimensions = sorted(
        name
        for name, coverage in dimension_low_mad_coverage.items()
        if coverage is not None and coverage < minimum_low_mad_coverage_per_dimension
    )
    low_coverage_action_dimensions = sorted(
        f"{action_id}/{name}"
        for action_id, coverages in action_dimension_low_mad_coverage.items()
        for name, coverage in coverages.items()
        if coverage is not None
        and coverage < minimum_low_mad_coverage_per_action_dimension
    )
    duplicate_pairs = [*response_duplicates, *risk_duplicates]
    high_correlation_pairs = [*response_correlations, *risk_correlations]
    mutual_exclusivity_diagnostic_pairs = [
        *_response_mutual_exclusivity,
        *risk_mutual_exclusivity,
    ]
    constant_fields = [
        *(response_constants if reject_constant_response_dimensions else []),
        *(risk_constants if reject_constant_risk_dimensions else []),
    ]
    constant_fields, duplicate_pairs, high_correlation_pairs = _drop_inapplicable(
        constants=constant_fields,
        duplicate_pairs=duplicate_pairs,
        correlation_pairs=high_correlation_pairs,
        inapplicable_dimensions=inapplicable_risk_dimensions,
    )
    family_failures = [
        f"{label.state_id}/{label.action_id}"
        for label in labels
        if len(set(label.judge_families)) < minimum_families
    ]
    expected_composite = composite_spec or CompositeSpec()
    expected_composite_version = expected_composite.version
    expected_composite_weights_sha256 = composite_weights_hash(expected_composite)
    composite_values = np.asarray(
        [expected_composite.score(label.response) for label in labels], dtype=float
    )
    support_values = (response_matrix[:, response_fields.index("emotional_support")] - 1.0) / 4.0
    composite_support_match_rate = float(
        np.mean(np.isclose(composite_values, support_values, rtol=0.0, atol=1e-12))
    )
    composite_support_correlation: float | None = None
    if (
        float(np.std(composite_values)) >= 1e-9
        and float(np.std(support_values)) >= 1e-9
    ):
        composite_support_correlation = float(
            np.corrcoef(composite_values, support_values)[0, 1]
        )
    composite_support_exact_failure = (
        composite_support_match_rate >= composite_support_exact_match_rate
    )
    composite_support_correlation_failure = (
        composite_support_correlation is not None
        and abs(composite_support_correlation)
        >= maximum_absolute_composite_support_correlation
    )
    composite_version_failures = [
        f"{label.state_id}/{label.action_id}"
        for label in labels
        if label.composite_spec_version != expected_composite_version
    ]
    composite_weights_hash_failures = [
        f"{label.state_id}/{label.action_id}"
        for label in labels
        if label.composite_weights_sha256 != expected_composite_weights_sha256
    ]
    report = {
        "n": len(labels),
        "duplicate_dimension_pairs": duplicate_pairs,
        "high_correlation_dimension_pairs": high_correlation_pairs,
        "mutual_exclusivity_diagnostic_pairs": mutual_exclusivity_diagnostic_pairs,
        "constant_dimensions": constant_fields,
        "response_dimension_prevalence": response_prevalence,
        "risk_dimension_prevalence": risk_prevalence,
        "reliable_label_rate": reliable_rate,
        "minimum_reliable_label_rate": minimum_reliable_rate,
        "joint_reliable_rate_is_diagnostic_only": True,
        "reliable_mad_threshold": reliable_mad_threshold,
        "dimension_low_mad_coverage": dimension_low_mad_coverage,
        "action_dimension_low_mad_coverage": action_dimension_low_mad_coverage,
        "minimum_low_mad_coverage_per_dimension": (
            minimum_low_mad_coverage_per_dimension
        ),
        "minimum_low_mad_coverage_per_action_dimension": (
            minimum_low_mad_coverage_per_action_dimension
        ),
        "low_coverage_dimensions": low_coverage_dimensions,
        "low_coverage_action_dimensions": low_coverage_action_dimensions,
        "inapplicable_risk_dimensions": sorted(inapplicable_risk_dimensions),
        "inapplicable_risk_dimensions_by_action": {
            action_id: sorted(dims)
            for action_id, dims in sorted(inapplicable_by_action.items())
        },
        "dimension_applicability_contract_protocol": (
            DIMENSION_APPLICABILITY_CONTRACT_PROTOCOL
        ),
        "minimum_independent_judge_families": minimum_families,
        "duplicate_exact_match_rate": duplicate_exact_match_rate,
        "maximum_absolute_dimension_correlation": maximum_absolute_dimension_correlation,
        "split_correlation_by_sign": split_correlation_by_sign,
        "insufficient_family_rows": family_failures,
        "composite_spec_version": expected_composite_version,
        "composite_weights_sha256": expected_composite_weights_sha256,
        "composite_support_exact_match_rate": composite_support_match_rate,
        "maximum_composite_support_exact_match_rate": (
            composite_support_exact_match_rate
        ),
        "composite_support_correlation": composite_support_correlation,
        "maximum_absolute_composite_support_correlation": (
            maximum_absolute_composite_support_correlation
        ),
        "composite_support_exact_match_failure": composite_support_exact_failure,
        "composite_support_correlation_failure": (
            composite_support_correlation_failure
        ),
        "composite_version_mismatch_rows": composite_version_failures,
        "composite_weights_hash_mismatch_rows": composite_weights_hash_failures,
        "status": "PASS",
    }
    if (
        duplicate_pairs
        or high_correlation_pairs
        or constant_fields
        or family_failures
        or composite_version_failures
        or composite_weights_hash_failures
        or composite_support_exact_failure
        or composite_support_correlation_failure
        or low_coverage_dimensions
        or low_coverage_action_dimensions
    ):
        report["status"] = "FAIL"
        if raise_on_failure:
            raise RuntimeError(
                "PM-v2 judge quality gate failed: " + canonical_json(report)
            )
    return report


JUDGE_FAMILY_DIRECTIONAL_PREFERENCE_PROTOCOL = (
    "pm-v1.5-judge-family-directional-preference-diagnostic-v1"
)


def judge_family_directional_preference_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    action_a: str,
    action_b: str,
    expected_families: Sequence[str],
    dialogue_by_state: Mapping[str, str],
    risk_weight: float,
    composite_spec: CompositeSpec | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_confidence_level: float = 0.90,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Cross-judge-family directional (action_a vs action_b) preference diagnostic.

    Per family, per state: compute that family's own (not aggregated-median)
    utility for action_a and action_b from its raw response/risk scores --
    the same quality-minus-risk_weight*risk formula the real routing
    objective uses -- and determine whether that family PREFERS action_a,
    action_b, or reports a tie. Across states, report concordance (both
    families prefer the same action) and discordance (families prefer
    opposite actions) among the *decisive* states only: a state is excluded
    from both the concordant and discordant counts, and from the
    concordance-rate denominator, whenever *either* family reports a tie for
    it -- a tie from one or both families is never counted as concordant.
    Also report the tie rate, a directional contingency table (which does
    still cross-tabulate every state, tie or not), and a dialogue-clustered
    bootstrap confidence interval on the concordance rate (resampling whole
    dialogues/users, respecting the non-independence of same-dialogue
    states).

    Every input row must carry ``state_id``, ``action_id``, ``judge_family``,
    ``response`` and ``risk`` (unlike the raw rows validate_raw_judge_family_
    health consumes, which intentionally omit state_id).

    This is purely diagnostic and never gates: it measures whether the
    routing-relevant *direction* of judge preference is reliable across
    independent judge families -- a different question from the per-
    dimension absolute-score agreement dimension_health/validate_judge_table
    already check.
    """

    spec = composite_spec or CompositeSpec()
    expected_family_set = sorted({str(family) for family in expected_families})
    if len(expected_family_set) != 2:
        raise ValueError(
            "directional preference report requires exactly two judge families"
        )
    if action_a == action_b:
        raise ValueError("directional preference report requires two distinct actions")

    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        action_id = str(row["action_id"])
        if action_id not in (action_a, action_b):
            continue
        family = str(row["judge_family"])
        if family not in expected_family_set:
            continue
        key = (str(row["state_id"]), action_id, family)
        if key in by_key:
            raise RuntimeError(f"duplicate directional preference row {key}")
        by_key[key] = row

    state_ids = sorted({key[0] for key in by_key})
    family_a, family_b = expected_family_set
    tie_epsilon = 1e-9

    def utility_for(state_id: str, action_id: str, family: str) -> float | None:
        row = by_key.get((state_id, action_id, family))
        if row is None:
            return None
        response = ResponseJudgeOutput.model_validate(row["response"])
        risk = RiskJudgeOutput.model_validate(row["risk"])
        quality = spec.score(
            ResponseDimensions(
                **{
                    name: float(getattr(response, name))
                    for name in ResponseDimensions.model_fields
                }
            )
        )
        risk_value = max(
            float(getattr(risk, name)) / 3.0
            for name in applicable_risk_fields(action_id)
        )
        return float(quality) - float(risk_weight) * risk_value

    def direction(state_id: str, family: str) -> str | None:
        utility_a = utility_for(state_id, action_a, family)
        utility_b = utility_for(state_id, action_b, family)
        if utility_a is None or utility_b is None:
            return None
        delta = utility_a - utility_b
        if abs(delta) <= tie_epsilon:
            return "tie"
        return action_a if delta > 0.0 else action_b

    directions: dict[tuple[str, str], str] = {}
    for state_id in state_ids:
        for family in expected_family_set:
            value = direction(state_id, family)
            if value is not None:
                directions[(state_id, family)] = value

    comparable_states = [
        state_id
        for state_id in state_ids
        if (state_id, family_a) in directions and (state_id, family_b) in directions
    ]

    categories = (action_a, action_b, "tie")
    contingency = {
        row_category: {column_category: 0 for column_category in categories}
        for row_category in categories
    }
    concordant = 0
    discordant = 0
    tie_involved = 0
    for state_id in comparable_states:
        direction_a = directions[(state_id, family_a)]
        direction_b = directions[(state_id, family_b)]
        contingency[direction_a][direction_b] += 1
        if direction_a == "tie" or direction_b == "tie":
            tie_involved += 1
        elif direction_a == direction_b:
            concordant += 1
        else:
            discordant += 1
    decisive_total = concordant + discordant
    n_comparable = len(comparable_states)

    groups: dict[str, list[str]] = {}
    for state_id in comparable_states:
        groups.setdefault(str(dialogue_by_state[state_id]), []).append(state_id)
    group_keys = sorted(groups)

    def concordance_rate_for(states: Sequence[str]) -> float | None:
        c = 0
        d = 0
        for state_id in states:
            direction_a = directions[(state_id, family_a)]
            direction_b = directions[(state_id, family_b)]
            if direction_a == "tie" or direction_b == "tie":
                continue
            if direction_a == direction_b:
                c += 1
            else:
                d += 1
        return c / (c + d) if (c + d) else None

    point_estimate = concordance_rate_for(comparable_states)
    bootstrap_values: list[float] = []
    if len(group_keys) >= 3 and int(bootstrap_replicates) >= 100:
        rng = np.random.default_rng(int(bootstrap_seed))
        group_key_array = np.asarray(group_keys, dtype=object)
        for _ in range(int(bootstrap_replicates)):
            sampled_groups = rng.choice(
                group_key_array, size=len(group_key_array), replace=True
            )
            sampled_states = [
                state_id for group in sampled_groups for state_id in groups[str(group)]
            ]
            rate = concordance_rate_for(sampled_states)
            if rate is not None:
                bootstrap_values.append(rate)
    if bootstrap_values:
        alpha = 1.0 - float(bootstrap_confidence_level)
        ci_lower: float | None = float(np.quantile(bootstrap_values, alpha / 2.0))
        ci_upper: float | None = float(np.quantile(bootstrap_values, 1.0 - alpha / 2.0))
    else:
        ci_lower = None
        ci_upper = None

    return {
        "protocol": JUDGE_FAMILY_DIRECTIONAL_PREFERENCE_PROTOCOL,
        "action_a": action_a,
        "action_b": action_b,
        "families": [family_a, family_b],
        "risk_weight": float(risk_weight),
        "n_states": len(state_ids),
        "n_comparable_states": n_comparable,
        "concordant_states": concordant,
        "discordant_states": discordant,
        "tie_involved_states": tie_involved,
        "concordance_rate": point_estimate,
        "discordance_rate": (discordant / decisive_total) if decisive_total else None,
        "tie_rate": (tie_involved / n_comparable) if n_comparable else None,
        "directional_contingency_table": contingency,
        "dialogue_cluster_bootstrap": {
            "n_groups": len(group_keys),
            "replicates": int(bootstrap_replicates),
            "confidence_level": float(bootstrap_confidence_level),
            "seed": int(bootstrap_seed),
            "concordance_rate_ci_lower": ci_lower,
            "concordance_rate_ci_upper": ci_upper,
        },
        "diagnostic_only": True,
        "never_gates": True,
    }


def prompt_contract_hash() -> str:
    payload = {
        "response_schema": ResponseJudgeOutput.model_json_schema(),
        "risk_schema": RiskJudgeOutput.model_json_schema(),
        "response_prompt_source": inspect.getsource(build_response_messages),
        "risk_prompt_source": inspect.getsource(build_risk_messages),
        "response_scale_anchors": RESPONSE_SCALE_ANCHORS,
        "risk_scale_anchors": RISK_SCALE_ANCHORS,
        "version": JUDGE_RUBRIC_VERSION,
    }
    return sha256_text(canonical_json(payload))
