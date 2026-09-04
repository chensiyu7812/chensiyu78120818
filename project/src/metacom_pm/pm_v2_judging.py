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
    rationale: str = Field(min_length=1, max_length=800)


class RiskJudgeOutput(StrictModel):
    selected_context_misuse: float = Field(ge=0.0, le=3.0)
    unnecessary_exposure: float = Field(ge=0.0, le=3.0)
    stale_or_conflicting_use: float = Field(ge=0.0, le=3.0)
    unsupported_personal_claim: float = Field(ge=0.0, le=3.0)
    memory_omission: float = Field(ge=0.0, le=3.0)
    strategy_overuse: float = Field(ge=0.0, le=3.0)
    strategy_omission: float = Field(ge=0.0, le=3.0)
    rationale: str = Field(min_length=1, max_length=800)


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


def dimension_health(
    matrix: np.ndarray,
    field_names: Sequence[str],
    *,
    prefix: str,
    duplicate_exact_match_rate: float,
    maximum_absolute_dimension_correlation: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    duplicate_pairs: list[dict[str, Any]] = []
    high_correlation_pairs: list[dict[str, Any]] = []
    for left in range(len(field_names)):
        for right in range(left + 1, len(field_names)):
            exact_rate = float(np.mean(matrix[:, left] == matrix[:, right]))
            left_std = float(np.std(matrix[:, left]))
            right_std = float(np.std(matrix[:, right]))
            correlation: float | None = None
            if left_std >= 1e-9 and right_std >= 1e-9:
                correlation = float(
                    np.corrcoef(matrix[:, left], matrix[:, right])[0, 1]
                )
            pair = {
                "left": f"{prefix}.{field_names[left]}",
                "right": f"{prefix}.{field_names[right]}",
                "exact_match_rate": exact_rate,
                "correlation": correlation,
            }
            if exact_rate >= duplicate_exact_match_rate:
                duplicate_pairs.append(pair)
            if (
                correlation is not None
                and abs(correlation) >= maximum_absolute_dimension_correlation
            ):
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
    return duplicate_pairs, high_correlation_pairs, constants, prevalence


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
    composite_spec: CompositeSpec | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Reject a degenerate judge family before cross-family aggregation.

    Aggregate medians can conceal a family that copies dimensions or emits a
    constant score table.  Every configured family therefore has to pass the
    same construct-independence checks on its own raw structured outputs.
    Canonical input rows contain ``judge_family``, ``response`` and ``risk``;
    the latter two are validated against the strict judge schemas.
    """

    expected = {str(value) for value in expected_families}
    if not expected:
        raise ValueError("raw judge-family health requires expected families")
    grouped: dict[str, list[tuple[ResponseJudgeOutput, RiskJudgeOutput]]] = defaultdict(list)
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
            )
        )
    missing_families = sorted(expected - set(grouped))
    spec = composite_spec or CompositeSpec()
    response_fields = tuple(ResponseDimensions.model_fields)
    risk_fields = tuple(RiskDimensions.model_fields)
    family_reports: dict[str, dict[str, Any]] = {}
    for family in sorted(grouped):
        outputs = grouped[family]
        response_matrix = np.asarray(
            [
                [float(getattr(response, name)) for name in response_fields]
                for response, _ in outputs
            ],
            dtype=float,
        )
        risk_matrix = np.asarray(
            [
                [float(getattr(risk, name)) for name in risk_fields]
                for _, risk in outputs
            ],
            dtype=float,
        )
        (
            response_duplicates,
            response_correlations,
            response_constants,
            response_prevalence,
        ) = dimension_health(
            response_matrix,
            response_fields,
            prefix="response",
            duplicate_exact_match_rate=duplicate_exact_match_rate,
            maximum_absolute_dimension_correlation=(
                maximum_absolute_dimension_correlation
            ),
        )
        (
            risk_duplicates,
            risk_correlations,
            risk_constants,
            risk_prevalence,
        ) = dimension_health(
            risk_matrix,
            risk_fields,
            prefix="risk",
            duplicate_exact_match_rate=duplicate_exact_match_rate,
            maximum_absolute_dimension_correlation=(
                maximum_absolute_dimension_correlation
            ),
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
                for response, _ in outputs
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
        constant_dimensions = [
            *(response_constants if reject_constant_response_dimensions else []),
            *(
                risk_constants
                if check_risk_dimension_health
                and reject_constant_risk_dimensions
                else []
            ),
        ]
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
        },
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
    composite_spec: CompositeSpec | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Fail closed on duplicated, constant, or unreliable response/risk labels."""

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
    (
        response_duplicates,
        response_correlations,
        response_constants,
        response_prevalence,
    ) = dimension_health(
        response_matrix,
        response_fields,
        prefix="response",
        duplicate_exact_match_rate=duplicate_exact_match_rate,
        maximum_absolute_dimension_correlation=maximum_absolute_dimension_correlation,
    )
    (
        risk_duplicates,
        risk_correlations,
        risk_constants,
        risk_prevalence,
    ) = dimension_health(
        risk_matrix,
        risk_fields,
        prefix="risk",
        duplicate_exact_match_rate=duplicate_exact_match_rate,
        maximum_absolute_dimension_correlation=maximum_absolute_dimension_correlation,
    )
    reliable_rate = float(np.mean([label.label_reliable for label in labels]))
    dimension_names = [
        *[f"response.{name}" for name in response_fields],
        *[f"risk.{name}" for name in risk_fields],
    ]
    dimension_low_mad_coverage = {
        name: float(
            np.mean(
                [
                    float(label.dimension_mad[name]) <= reliable_mad_threshold
                    for label in labels
                ]
            )
        )
        for name in dimension_names
    }
    action_dimension_low_mad_coverage: dict[str, dict[str, float]] = {}
    for action_id in sorted({label.action_id for label in labels}):
        action_labels = [label for label in labels if label.action_id == action_id]
        action_dimension_low_mad_coverage[action_id] = {
            name: float(
                np.mean(
                    [
                        float(label.dimension_mad[name]) <= reliable_mad_threshold
                        for label in action_labels
                    ]
                )
            )
            for name in dimension_names
        }
    low_coverage_dimensions = sorted(
        name
        for name, coverage in dimension_low_mad_coverage.items()
        if coverage < minimum_low_mad_coverage_per_dimension
    )
    low_coverage_action_dimensions = sorted(
        f"{action_id}/{name}"
        for action_id, coverages in action_dimension_low_mad_coverage.items()
        for name, coverage in coverages.items()
        if coverage < minimum_low_mad_coverage_per_action_dimension
    )
    duplicate_pairs = [*response_duplicates, *risk_duplicates]
    high_correlation_pairs = [*response_correlations, *risk_correlations]
    constant_fields = [
        *(response_constants if reject_constant_response_dimensions else []),
        *(risk_constants if reject_constant_risk_dimensions else []),
    ]
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
        "minimum_independent_judge_families": minimum_families,
        "duplicate_exact_match_rate": duplicate_exact_match_rate,
        "maximum_absolute_dimension_correlation": maximum_absolute_dimension_correlation,
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
