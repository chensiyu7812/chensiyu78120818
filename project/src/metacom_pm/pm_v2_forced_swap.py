from __future__ import annotations

from collections import Counter, defaultdict
import inspect
import math
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import Field, model_validator

from .api import (
    Endpoint,
    chat_request_payload,
    make_client,
    request_log,
    request_payload_has_schema,
)
from .artifacts import create_artifact_attestation, require_artifact_attestation
from .attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key as make_physical_call_key,
    reported_prompt_token_error,
)
from .io import (
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .pm_v2_contracts import StrictModel
from .pm_v2_external_eval import load_fixed_turns
from .pm_v2_evoemo import compare_observed_cost_matched_turns
from .pm_v2_judging import RESPONSE_SCALE_ANCHORS
from .text import conservative_token_bound, estimate_tokens


class ForcedSwapCandidateScore(StrictModel):
    candidate_id: Literal["C1", "C2"]
    support: int = Field(ge=1, le=5)
    personalization: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1, max_length=500)


class ForcedSwapJudgeOutput(StrictModel):
    candidate_scores: list[ForcedSwapCandidateScore] = Field(
        min_length=2, max_length=2
    )
    preferred_candidate_id: Literal["C1", "C2", "tie"]
    preference_rationale: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def exact_candidate_ids(self):
        if sorted(score.candidate_id for score in self.candidate_scores) != ["C1", "C2"]:
            raise ValueError("candidate_scores must contain C1 and C2 exactly once")
        return self


FORCED_SWAP_RUBRIC = {
    "support": (
        "empathy, validation, emotional attunement, and useful support for the "
        "current seeker turn"
    ),
    "personalization": (
        "specific tailoring grounded in the visible dialogue and authorized context; "
        "generic warmth alone is not personalization"
    ),
    "preference": (
        "choose C1 or C2 only when clearly better for support and grounded, "
        "non-intrusive personalization; otherwise choose tie"
    ),
}


def build_forced_swap_messages(
    *,
    turn: Mapping[str, Any],
    authorized_user_context: str,
    candidate_responses: Sequence[str],
) -> list[dict[str, str]]:
    if len(candidate_responses) != 2:
        raise ValueError("forced-swap prompt requires exactly two candidates")
    system = f"""You are an evaluator-only judge for emotional-support responses.
Two anonymous candidates answered the identical fixed-context seeker turn. Score
each candidate independently before choosing a preference. Do not infer the policy,
resource action, or candidate identity. Do not reward mentioning more personal facts.
Use authorized context only to check relevance, grounding, and non-intrusiveness.
Prefer tie when responses are close or trade off strengths. Return strict JSON.

{RESPONSE_SCALE_ANCHORS}"""
    payload = {
        "task": "blind forced-swap comparison on one fixed-context seeker turn",
        "rubric": FORCED_SWAP_RUBRIC,
        "authorized_user_context": authorized_user_context,
        "fixed_context_case": {
            "context_before_turn": turn.get("context_before_turn") or [],
            "current_seeker_message": str(turn["seeker_message"]),
            "turn_index": int(turn["turn_index"]),
        },
        "anonymous_candidates": [
            {"candidate_id": "C1", "response": str(candidate_responses[0])},
            {"candidate_id": "C2", "response": str(candidate_responses[1])},
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": canonical_json(payload)},
    ]


def forced_swap_prompt_contract_hash() -> str:
    return sha256_text(
        canonical_json(
            {
                "schema": ForcedSwapJudgeOutput.model_json_schema(),
                "prompt_source": inspect.getsource(build_forced_swap_messages),
                "response_scale_anchors": RESPONSE_SCALE_ANCHORS,
                "rubric": FORCED_SWAP_RUBRIC,
                "version": "pmv2-forced-swap-v1",
            }
        )
    )


def select_forced_swap_units(
    units: Sequence[tuple[str, int, int, str, int]],
    *,
    sample_units: int,
    seed: int,
) -> list[tuple[str, int, int, str, int]]:
    if sample_units <= 0:
        raise ValueError("forced-swap sample_units must be positive")
    if sample_units > len(units):
        raise ValueError("forced-swap sample_units exceeds the frozen external matrix")
    ordered = sorted(
        units,
        key=lambda unit: (
            sha256_text(canonical_json({"seed": int(seed), "unit": unit})),
            unit,
        ),
    )
    return sorted(ordered[:sample_units])


def resolve_dual_order_preference(
    first: str,
    second: str,
    *,
    treatment: str,
    baseline: str,
) -> str:
    valid = {treatment, baseline, "tie"}
    if first not in valid or second not in valid:
        raise ValueError("forced-swap preference contains an unknown condition")
    return first if first == second else "tie"


def _endpoint_contract(endpoint: Endpoint) -> dict[str, Any]:
    payload = {
        "model": endpoint.model,
        "family": endpoint.family,
        "base_url": endpoint.base_url,
    }
    return {**payload, "sha256": sha256_text(canonical_json(payload))}


def forced_swap_unit_id(unit: tuple[str, int, int, str, int]) -> str:
    return sha256_text(canonical_json(unit))[:24]


def _authorized_context_map(evoemo_path: str | Path) -> dict[tuple[str, int], str]:
    from .evoemo import evaluator_context, load_evoemo

    result: dict[tuple[str, int], str] = {}
    for user in load_evoemo(evoemo_path):
        for topic in user.get("subsequent_topics") or []:
            result[(str(user["id"]), int(topic["idx"]))] = canonical_json(
                evaluator_context(user, topic)
            )
    return result


def _condition_from_candidate(
    preferred_candidate_id: str, mapping: Mapping[str, str]
) -> str:
    if preferred_candidate_id == "tie":
        return "tie"
    try:
        return str(mapping[preferred_candidate_id])
    except KeyError as exc:
        raise ValueError("judge returned an unknown candidate ID") from exc


def forced_swap_family_direction_agreement(
    family_mean_deltas: Mapping[str, float], *, epsilon: float = 1e-12
) -> bool:
    """Require every judge family to provide a non-zero, same-sign signal."""

    if len(family_mean_deltas) < 2:
        return False
    directions = [
        1 if float(value) > epsilon else -1 if float(value) < -epsilon else 0
        for value in family_mean_deltas.values()
    ]
    return all(direction != 0 for direction in directions) and len(
        set(directions)
    ) == 1


def _require_saved_dry_run(
    estimate_path: Path,
    call_plan_path: Path,
    *,
    current_estimate: Mapping[str, Any],
    current_plan: Sequence[Mapping[str, Any]],
) -> None:
    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("forced-swap API run requires a matching saved dry-run")
    saved = read_json(estimate_path)
    if saved.get("cost_estimate_sha256") != current_estimate.get(
        "cost_estimate_sha256"
    ):
        raise RuntimeError("saved forced-swap cost estimate is stale")
    saved_plan = list(iter_jsonl(call_plan_path))
    if sha256_text(canonical_json(saved_plan)) != sha256_text(
        canonical_json(list(current_plan))
    ):
        raise RuntimeError("saved forced-swap call plan is stale")


def forced_swap_schema_futility_reason(
    *,
    schema_failures: int,
    expected_calls: int,
    minimum_schema_success_rate: float,
) -> str | None:
    allowed_failures = math.floor(
        (1.0 - float(minimum_schema_success_rate)) * int(expected_calls) + 1e-12
    )
    if int(schema_failures) > allowed_failures:
        return (
            "forced-swap schema threshold is unreachable from persisted attempts"
        )
    return None


def _cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    n_resamples: int = 5000,
) -> dict[str, Any]:
    if not values or len(values) != len(clusters):
        raise ValueError("cluster bootstrap requires aligned non-empty values")
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        by_cluster[str(cluster)].append(float(value))
    cluster_ids = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        rows = [value for cluster in sampled for value in by_cluster[str(cluster)]]
        samples[index] = float(np.mean(rows))
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "n_clusters": len(cluster_ids),
        "n_resamples": int(n_resamples),
    }


def _summarize(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    sampled_units: Sequence[tuple[str, int, int, str, int]],
    treatment: str,
    baseline: str,
    endpoint_families: Sequence[str],
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    judgments: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    by_unit_order: dict[tuple[str, int], list[str]] = defaultdict(list)
    score_by_unit_condition: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    score_by_family_unit_condition: dict[
        tuple[str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        parsed = ForcedSwapJudgeOutput.model_validate(row["judgment"])
        mapping = {str(key): str(value) for key, value in row["candidate_mapping"].items()}
        preferred = _condition_from_candidate(parsed.preferred_candidate_id, mapping)
        judgment = {
            "unit_id": str(row["unit_id"]),
            "order_variant": int(row["order_variant"]),
            "judge_family": str(row["judge_family"]),
            "preferred_candidate_id": parsed.preferred_candidate_id,
            "preferred_condition": preferred,
            "candidate_mapping": mapping,
            "preference_rationale": parsed.preference_rationale,
        }
        judgments.append(judgment)
        by_unit_order[(judgment["unit_id"], judgment["order_variant"])].append(
            preferred
        )
        for candidate_score in parsed.candidate_scores:
            condition = mapping[candidate_score.candidate_id]
            score = {
                "unit_id": str(row["unit_id"]),
                "order_variant": int(row["order_variant"]),
                "judge_family": str(row["judge_family"]),
                "condition": condition,
                "candidate_id": candidate_score.candidate_id,
                "position": 1 if candidate_score.candidate_id == "C1" else 2,
                "support": int(candidate_score.support),
                "personalization": int(candidate_score.personalization),
                "rationale": candidate_score.rationale,
            }
            scores.append(score)
            for metric in ("support", "personalization"):
                score_by_unit_condition[(score["unit_id"], condition)][metric].append(
                    float(score[metric])
                )
                score_by_family_unit_condition[
                    (score["judge_family"], score["unit_id"], condition)
                ][metric].append(float(score[metric]))

    resolved_units: list[dict[str, Any]] = []
    resolution_counts = {treatment: 0, baseline: 0, "tie": 0}
    order_disagreement = 0
    family_disagreement = 0
    deltas = {"support": [], "personalization": []}
    for unit in sampled_units:
        unit_id = forced_swap_unit_id(unit)
        order_preferences: dict[int, str] = {}
        for order_variant in (0, 1):
            values = by_unit_order[(unit_id, order_variant)]
            if len(values) != len(endpoint_families):
                raise RuntimeError("forced-swap judgment matrix is incomplete")
            order_preferences[order_variant] = values[0] if len(set(values)) == 1 else "tie"
            family_disagreement += int(len(set(values)) > 1)
        resolved = resolve_dual_order_preference(
            order_preferences[0],
            order_preferences[1],
            treatment=treatment,
            baseline=baseline,
        )
        if order_preferences[0] != order_preferences[1]:
            order_disagreement += 1
        resolution_counts[resolved] += 1
        unit_result = {
            "unit_id": unit_id,
            "order0_preference": order_preferences[0],
            "order1_preference": order_preferences[1],
            "resolved_preference": resolved,
        }
        for metric in ("support", "personalization"):
            treatment_values = score_by_unit_condition[(unit_id, treatment)][metric]
            baseline_values = score_by_unit_condition[(unit_id, baseline)][metric]
            if not treatment_values or not baseline_values:
                raise RuntimeError("forced-swap score matrix is incomplete")
            delta = mean(treatment_values) - mean(baseline_values)
            deltas[metric].append(delta)
            unit_result[f"{metric}_delta_treatment_minus_baseline"] = delta
        resolved_units.append(unit_result)

    condition_summary: dict[str, dict[str, float]] = {}
    for condition in (treatment, baseline):
        condition_summary[condition] = {}
        for metric in ("support", "personalization"):
            values = [
                float(row[metric]) for row in scores if row["condition"] == condition
            ]
            condition_summary[condition][metric] = mean(values)
    support_ci = _cluster_bootstrap_ci(
        deltas["support"],
        [str(unit[0]) for unit in sampled_units],
        seed=bootstrap_seed,
    )
    family_support_deltas: dict[str, list[float]] = {}
    for family in endpoint_families:
        family_support_deltas[family] = []
        for unit in sampled_units:
            unit_id = forced_swap_unit_id(unit)
            treatment_values = score_by_family_unit_condition[
                (family, unit_id, treatment)
            ]["support"]
            baseline_values = score_by_family_unit_condition[
                (family, unit_id, baseline)
            ]["support"]
            if len(treatment_values) != 2 or len(baseline_values) != 2:
                raise RuntimeError("forced-swap family score matrix is incomplete")
            family_support_deltas[family].append(
                mean(treatment_values) - mean(baseline_values)
            )
    family_mean_deltas = {
        family: mean(values) for family, values in family_support_deltas.items()
    }
    family_direction_agreement = forced_swap_family_direction_agreement(
        family_mean_deltas
    )
    left_family, right_family = endpoint_families
    left_deltas = np.asarray(family_support_deltas[left_family], dtype=float)
    right_deltas = np.asarray(family_support_deltas[right_family], dtype=float)
    left_constant = float(np.std(left_deltas)) < 1e-12
    right_constant = float(np.std(right_deltas)) < 1e-12
    cross_family_correlation = (
        None
        if left_constant or right_constant
        else float(np.corrcoef(left_deltas, right_deltas)[0, 1])
    )
    analysis = {
        "condition_summary": condition_summary,
        "dual_order_mean_delta_treatment_minus_baseline": {
            metric: mean(values) for metric, values in deltas.items()
        },
        "dual_order_resolved_preference": resolution_counts,
        "order_disagreement_units": order_disagreement,
        "order_disagreement_rate": order_disagreement / len(sampled_units),
        "family_disagreement_order_cells": family_disagreement,
        "support_delta_cluster_bootstrap_ci": support_ci,
        "family_support_delta_treatment_minus_baseline": family_mean_deltas,
        "family_support_delta_direction_agreement": family_direction_agreement,
        "cross_family_support_delta_correlation": cross_family_correlation,
        "cross_family_correlation_estimable": cross_family_correlation is not None,
        "cross_family_constant_delta": {
            left_family: left_constant,
            right_family: right_constant,
        },
        "all_resolved_preferences_tie": (
            resolution_counts["tie"] == len(sampled_units)
        ),
        "resolved_units": resolved_units,
    }
    return judgments, scores, analysis


def forced_swap_compatibility_gate(
    *,
    analysis: Mapping[str, Any],
    endpoint_families: Sequence[str],
    compatibility_thresholds: Mapping[str, Any],
    schema_success_rate: float,
    schema_successes: int,
    schema_attempts: int,
    schema_expected_calls: int,
    observed_cost_match_pass: bool,
    futility_reason: str | None,
) -> dict[str, Any]:
    """Apply structural and preregistered forced-swap reportability gates.

    Constant deltas and an all-tie panel contain no sensitivity evidence.  They
    must never be promoted to a key-claim PASS merely because a correlation is
    undefined or a direction check is vacuously true.
    """

    correlation = analysis.get("cross_family_support_delta_correlation")
    constant_flags = analysis.get("cross_family_constant_delta") or {}
    direction_agreement = bool(
        analysis.get("family_support_delta_direction_agreement", False)
    )
    expected_family_set = {str(value) for value in endpoint_families}
    delta_variation = (
        set(str(key) for key in constant_flags) == expected_family_set
        and not any(bool(value) for value in constant_flags.values())
    )
    resolved = analysis.get("dual_order_resolved_preference") or {}
    resolved_total = sum(int(value) for value in resolved.values())
    all_tie = bool(analysis.get("all_resolved_preferences_tie", False)) or (
        resolved_total > 0 and int(resolved.get("tie", 0)) == resolved_total
    )
    checks = {
        "schema_success_rate": float(schema_success_rate)
        >= float(compatibility_thresholds["minimum_schema_success_rate"]),
        "exact_two_distinct_families": len(expected_family_set) == 2,
        "cross_family_direction_agreement": (
            not bool(
                compatibility_thresholds[
                    "require_cross_family_direction_agreement"
                ]
            )
            or direction_agreement
        ),
        "cross_family_support_delta_correlation": (
            correlation is not None
            and float(correlation)
            >= float(
                compatibility_thresholds[
                    "minimum_cross_family_support_delta_correlation"
                ]
            )
        ),
        "per_family_support_delta_variation": delta_variation,
        "resolved_preference_sensitivity": resolved_total > 0 and not all_tie,
        "order_disagreement_rate": float(
            analysis.get("order_disagreement_rate", 1.0)
        )
        <= float(compatibility_thresholds["maximum_order_disagreement_rate"]),
        "futility_continuation": float(
            (analysis.get("support_delta_cluster_bootstrap_ci") or {}).get(
                "upper", float("-inf")
            )
        )
        >= float(
            compatibility_thresholds[
                "minimum_support_delta_ci_upper_for_continuation"
            ]
        ),
        "observed_cost_match": bool(observed_cost_match_pass),
    }
    return {
        "status": "PASS" if all(checks.values()) else "NONREPORTABLE",
        "checks": checks,
        "thresholds": dict(compatibility_thresholds),
        "schema_success_rate": float(schema_success_rate),
        "schema_successes": int(schema_successes),
        "schema_attempts": int(schema_attempts),
        "schema_expected_calls": int(schema_expected_calls),
        "futility_triggered": futility_reason is not None,
        "futility_reason": futility_reason,
        "all_resolved_preferences_tie": all_tie,
        "constant_support_delta_families": sorted(
            str(family)
            for family, is_constant in constant_flags.items()
            if bool(is_constant)
        ),
        "cross_family_correlation_constant_waiver": False,
        "cross_family_correlation_note": (
            "undefined correlation is nonreportable; constant per-family deltas "
            "receive no structural waiver"
            if correlation is None
            else None
        ),
    }


def forced_swap_efficacy_gate(
    *,
    analysis: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    compatibility_pass: bool,
) -> dict[str, Any]:
    """Separate measurement/continuation compatibility from PM efficacy."""

    support_ci = analysis.get("support_delta_cluster_bootstrap_ci") or {}
    family_deltas = analysis.get(
        "family_support_delta_treatment_minus_baseline"
    ) or {}
    resolved = analysis.get("dual_order_resolved_preference") or {}
    treatment = str(thresholds["treatment"])
    baseline = str(thresholds["baseline"])
    preference_margin = int(resolved.get(treatment, 0)) - int(
        resolved.get(baseline, 0)
    )
    checks = {
        "compatibility_pass": bool(compatibility_pass),
        "support_ci_lower_advantage": float(
            support_ci.get("lower", float("-inf"))
        )
        > float(thresholds["minimum_support_delta_ci_lower_for_advantage"]),
        "positive_support_delta_every_family": (
            not bool(thresholds["require_positive_support_delta_every_family"])
            or (
                bool(family_deltas)
                and all(float(value) > 0.0 for value in family_deltas.values())
            )
        ),
        "resolved_preference_margin": preference_margin
        >= int(thresholds["minimum_resolved_preference_margin"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "NOT_VERIFIED",
        "checks": checks,
        "support_ci": dict(support_ci),
        "family_support_deltas": {
            str(key): float(value) for key, value in family_deltas.items()
        },
        "resolved_preference_margin": preference_margin,
        "thresholds": {
            "minimum_support_delta_ci_lower_for_advantage": float(
                thresholds["minimum_support_delta_ci_lower_for_advantage"]
            ),
            "require_positive_support_delta_every_family": bool(
                thresholds["require_positive_support_delta_every_family"]
            ),
            "minimum_resolved_preference_margin": int(
                thresholds["minimum_resolved_preference_margin"]
            ),
        },
    }


def require_forced_swap_key_claim(
    summary_path: str | Path,
    attestation_path: str | Path,
    *,
    study_freeze_sha256: str,
    full_expected_units: Sequence[tuple[str, int, int, str, int]],
    forced_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that a PASS pilot is the exact frozen external compatibility gate."""

    verification = require_artifact_attestation(
        attestation_path,
        required_stage="pm_v2_forced_swap_key_claim",
        required_output_paths={"summary": summary_path},
        expected_freeze_sha256=study_freeze_sha256,
    )
    summary = read_json(summary_path)
    treatment = str(forced_contract["treatment"])
    baseline = str(forced_contract["baseline"])
    sample_units = int(forced_contract["sample_units"])
    judge_seed = int(forced_contract["judge_seed"])
    selected = select_forced_swap_units(
        full_expected_units, sample_units=sample_units, seed=judge_seed
    )
    selected_sha256 = sha256_text(canonical_json(selected))
    excluded_unit_ids = [forced_swap_unit_id(unit) for unit in selected]
    endpoint_contracts = {
        str(row["family"]): {
            "model": str(row["model"]),
            "family": str(row["family"]),
            "base_url": str(row["base_url"]),
            "sha256": str(row["sha256"]),
        }
        for row in forced_contract["judge_endpoints"]
    }
    thresholds = {
        key: forced_contract[key]
        for key in (
            "maximum_order_disagreement_rate",
            "minimum_schema_success_rate",
            "minimum_cross_family_support_delta_correlation",
            "require_cross_family_direction_agreement",
            "minimum_support_delta_ci_upper_for_continuation",
        )
    }
    efficacy_thresholds = {
        "treatment": treatment,
        "baseline": baseline,
        "minimum_support_delta_ci_lower_for_advantage": float(
            forced_contract["minimum_support_delta_ci_lower_for_advantage"]
        ),
        "require_positive_support_delta_every_family": bool(
            forced_contract["require_positive_support_delta_every_family"]
        ),
        "minimum_resolved_preference_margin": int(
            forced_contract["minimum_resolved_preference_margin"]
        ),
    }
    selection_contract = {
        "algorithm": "sha256_rank_without_replacement_v1",
        "algorithm_source_sha256": sha256_text(
            inspect.getsource(select_forced_swap_units)
        ),
        "full_expected_units_sha256": sha256_text(
            canonical_json(full_expected_units)
        ),
        "sample_units": sample_units,
        "sample_seed": judge_seed,
        "selected_units_sha256": selected_sha256,
        "excluded_unit_ids": excluded_unit_ids,
    }
    selection_contract_sha256 = sha256_text(canonical_json(selection_contract))
    attestation = read_json(attestation_path)
    attestation_inputs = attestation.get("inputs") or {}
    treatment_turn_record = attestation_inputs.get("turn_treatment") or {}
    baseline_turn_record = attestation_inputs.get("turn_baseline") or {}
    if not treatment_turn_record.get("path") or not baseline_turn_record.get("path"):
        raise RuntimeError("forced-swap attestation lacks cost-match turn inputs")
    observed_cost_match_report = compare_observed_cost_matched_turns(
        treatment_turn_record["path"],
        baseline_turn_record["path"],
        maximum_relative_deviation=float(
            forced_contract["maximum_cost_matched_relative_deviation"]
        ),
        expected_units=full_expected_units,
    )
    expected_efficacy = forced_swap_efficacy_gate(
        analysis=summary,
        thresholds=efficacy_thresholds,
        compatibility_pass=True,
    )
    expected_summary = {
        "status": "PASS",
        "support_key_claim_verified": True,
        "study_freeze_sha256": study_freeze_sha256,
        "treatment": treatment,
        "baseline": baseline,
        "sample_units": sample_units,
        "sample_units_sha256": selected_sha256,
        "sample_selection_contract_sha256": selection_contract_sha256,
        "excluded_unit_ids": excluded_unit_ids,
        "order_variants": [int(value) for value in forced_contract["order_variants"]],
        "endpoint_contracts": endpoint_contracts,
        "judge_seed": judge_seed,
        "full_expected_units_sha256": sha256_text(
            canonical_json(full_expected_units)
        ),
        "prompt_contract_sha256": forced_swap_prompt_contract_hash(),
        "compatibility_thresholds": thresholds,
        "efficacy_thresholds": efficacy_thresholds,
        "efficacy_gate": expected_efficacy,
        "observed_cost_match_report": observed_cost_match_report,
        "judge_retries": 1,
        "pricing_usd_per_mtok": forced_contract["judge_pricing_usd_per_mtok"],
        "api_cost_planning": forced_contract["api_cost_planning"],
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"forced-swap summary contract mismatch: {key}")
    gate = summary.get("compatibility_gate") or {}
    if gate.get("status") != "PASS" or not all(
        bool(value) for value in (gate.get("checks") or {}).values()
    ):
        raise RuntimeError("forced-swap compatibility/value gate did not PASS")
    if (
        summary.get("efficacy_gate") != expected_efficacy
        or expected_efficacy.get("status") != "PASS"
        or summary.get("support_key_claim_verified") is not True
    ):
        raise RuntimeError("forced-swap efficacy gate did not verify PM advantage")
    observed_cost_match = summary.get("observed_cost_match_report") or {}
    if (
        observed_cost_match.get("status") != "PASS"
        or not bool(observed_cost_match.get("check"))
        or float(observed_cost_match.get("maximum_relative_deviation", -1.0))
        != float(forced_contract["maximum_cost_matched_relative_deviation"])
    ):
        raise RuntimeError("forced-swap observed-token cost-match gate did not pass")
    parameters = attestation.get("parameters") or {}
    for key in (
        "treatment",
        "baseline",
        "sample_units",
        "sample_units_sha256",
        "order_variants",
        "endpoint_contracts",
        "judge_seed",
        "prompt_contract_sha256",
        "sample_selection_contract_sha256",
        "excluded_unit_ids",
        "compatibility_thresholds",
        "efficacy_thresholds",
        "efficacy_gate",
        "support_key_claim_verified",
        "observed_cost_match_report",
        "judge_retries",
        "pricing_usd_per_mtok",
        "api_cost_planning",
    ):
        if parameters.get(key) != expected_summary.get(key):
            raise RuntimeError(f"forced-swap attestation contract mismatch: {key}")
    return {
        "summary": summary,
        "attestation_sha256": verification["attestation_sha256"],
        "excluded_units": selected,
        "excluded_unit_ids": excluded_unit_ids,
        "sample_selection_contract_sha256": selection_contract_sha256,
    }


def run_forced_swap_evaluation(
    *,
    evoemo_path: str | Path,
    study_freeze_path: str | Path,
    study_freeze_sha256: str,
    turn_paths: Sequence[str | Path],
    generation_attestation_paths: Sequence[str | Path],
    expected_units: Sequence[tuple[str, int, int, str, int]],
    treatment: str,
    baseline: str,
    endpoints: Sequence[Endpoint],
    sample_units: int,
    order_variants: Sequence[int],
    judge_seed: int,
    estimated_output_tokens_per_call: int,
    maximum_order_disagreement_rate: float,
    minimum_schema_success_rate: float,
    minimum_cross_family_support_delta_correlation: float,
    require_cross_family_direction_agreement: bool,
    minimum_support_delta_ci_upper_for_continuation: float,
    minimum_support_delta_ci_lower_for_advantage: float,
    require_positive_support_delta_every_family: bool,
    minimum_resolved_preference_margin: int,
    maximum_cost_matched_relative_deviation: float,
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
    api_cost_planning: Mapping[str, Any],
    out_dir: str | Path,
    run: bool,
    max_api_calls: int,
    max_estimated_usd: float,
    max_input_tokens_per_call: int,
    accept_cost_estimate_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if run and overwrite:
        raise RuntimeError(
            "paid API runs prohibit overwrite; use a new output directory"
        )
    if list(order_variants) != [0, 1]:
        raise ValueError("reportable forced-swap requires order_variants exactly [0, 1]")
    api_cost_planning = dict(api_cost_planning)
    if set(api_cost_planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("forced-swap api_cost_planning contract is incomplete")
    input_token_safety_factor = float(
        api_cost_planning["input_token_safety_factor"]
    )
    if input_token_safety_factor < 1.0 or not bool(
        api_cost_planning["fail_on_reported_input_overrun"]
    ):
        raise ValueError("forced-swap requires fail-closed cost planning")
    if len(turn_paths) != 2 or len(generation_attestation_paths) != 2:
        raise ValueError("forced-swap requires exactly two attested turn inputs")
    if treatment == baseline:
        raise ValueError("forced-swap treatment and baseline must differ")
    endpoint_families = [str(endpoint.family or "") for endpoint in endpoints]
    if len(endpoints) != 2 or any(not value for value in endpoint_families):
        raise ValueError("forced-swap requires exactly two declared judge families")
    if len(set(endpoint_families)) != len(endpoint_families):
        raise ValueError("forced-swap judges must use distinct model families")
    family_pricing = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in pricing_usd_per_mtok.items()
    }
    if set(family_pricing) != set(endpoint_families) or any(
        set(values) != {"input", "output"}
        or any(value < 0.0 for value in values.values())
        for values in family_pricing.values()
    ):
        raise ValueError("forced-swap pricing must exactly cover judge families")
    if estimated_output_tokens_per_call <= 0:
        raise ValueError("estimated_output_tokens_per_call must be positive")
    for name, value in (
        ("maximum_order_disagreement_rate", maximum_order_disagreement_rate),
        ("minimum_schema_success_rate", minimum_schema_success_rate),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not -1.0 <= float(minimum_cross_family_support_delta_correlation) <= 1.0:
        raise ValueError(
            "minimum_cross_family_support_delta_correlation must be in [-1, 1]"
        )
    if read_json(study_freeze_path).get("freeze_sha256") != study_freeze_sha256:
        raise RuntimeError("forced-swap study freeze file/hash mismatch")
    stage = "pm_v2_forced_swap_key_claim"

    attested_conditions: set[str] = set()
    turn_path_by_condition: dict[str, str | Path] = {}
    attestation_hashes: dict[str, str] = {}
    for turn_path, attestation_path in zip(turn_paths, generation_attestation_paths):
        verification = require_artifact_attestation(
            attestation_path,
            required_stage="evoemo_pm_v2_generation",
            required_output_paths={"turns": turn_path},
            expected_freeze_sha256=study_freeze_sha256,
        )
        attestation = read_json(attestation_path)
        condition = str((attestation.get("parameters") or {}).get("condition") or "")
        if not condition:
            raise RuntimeError("generation attestation lacks its condition")
        attested_conditions.add(condition)
        if condition in turn_path_by_condition:
            raise RuntimeError(f"duplicate forced-swap turn input for {condition}")
        turn_path_by_condition[condition] = turn_path
        attestation_hashes[str(Path(attestation_path).resolve())] = str(
            verification["attestation_sha256"]
        )
    if attested_conditions != {treatment, baseline}:
        raise RuntimeError("forced-swap generation attestations cover the wrong conditions")
    if treatment != "pm_v2" or baseline != "pm_v2_cost_matched_fixed":
        raise RuntimeError("forced-swap cost-match gate requires the frozen PM/fixed pair")
    observed_cost_match_report = compare_observed_cost_matched_turns(
        turn_path_by_condition[treatment],
        turn_path_by_condition[baseline],
        maximum_relative_deviation=maximum_cost_matched_relative_deviation,
        expected_units=expected_units,
    )
    if observed_cost_match_report["status"] != "PASS":
        raise RuntimeError(
            "forced-swap blocked: observed input-token cost match exceeds tolerance"
        )

    matrix = load_fixed_turns(
        turn_paths,
        conditions=[treatment, baseline],
        turn_indices=sorted({unit[4] for unit in expected_units}),
        expected_units=expected_units,
    )
    sampled = select_forced_swap_units(
        list(expected_units), sample_units=sample_units, seed=judge_seed
    )
    sample_sha256 = sha256_text(canonical_json(sampled))
    excluded_unit_ids = [forced_swap_unit_id(unit) for unit in sampled]
    selection_contract = {
        "algorithm": "sha256_rank_without_replacement_v1",
        "algorithm_source_sha256": sha256_text(
            inspect.getsource(select_forced_swap_units)
        ),
        "full_expected_units_sha256": sha256_text(canonical_json(expected_units)),
        "sample_units": int(sample_units),
        "sample_seed": int(judge_seed),
        "selected_units_sha256": sample_sha256,
        "excluded_unit_ids": excluded_unit_ids,
    }
    selection_contract_sha256 = sha256_text(canonical_json(selection_contract))
    compatibility_thresholds = {
        "maximum_order_disagreement_rate": float(
            maximum_order_disagreement_rate
        ),
        "minimum_schema_success_rate": float(minimum_schema_success_rate),
        "minimum_cross_family_support_delta_correlation": float(
            minimum_cross_family_support_delta_correlation
        ),
        "require_cross_family_direction_agreement": bool(
            require_cross_family_direction_agreement
        ),
        "minimum_support_delta_ci_upper_for_continuation": float(
            minimum_support_delta_ci_upper_for_continuation
        ),
    }
    efficacy_thresholds = {
        "treatment": treatment,
        "baseline": baseline,
        "minimum_support_delta_ci_lower_for_advantage": float(
            minimum_support_delta_ci_lower_for_advantage
        ),
        "require_positive_support_delta_every_family": bool(
            require_positive_support_delta_every_family
        ),
        "minimum_resolved_preference_margin": int(
            minimum_resolved_preference_margin
        ),
    }
    authorized = _authorized_context_map(evoemo_path)
    endpoint_contracts = {
        str(endpoint.family): _endpoint_contract(endpoint) for endpoint in endpoints
    }
    order_by_variant = {0: (treatment, baseline), 1: (baseline, treatment)}
    plan_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    cost_rows: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(sampled):
        user_id, topic_index, _seed, _simulator_id, _turn_index = unit
        unit_id = forced_swap_unit_id(unit)
        reference_turn = matrix[(*unit, treatment)]
        for order_variant in order_variants:
            order = order_by_variant[int(order_variant)]
            candidate_mapping = {"C1": order[0], "C2": order[1]}
            messages = build_forced_swap_messages(
                turn=reference_turn,
                authorized_user_context=authorized[(user_id, topic_index)],
                candidate_responses=[
                    str(matrix[(*unit, condition)]["supporter_message"])
                    for condition in order
                ],
            )
            prompt_sha256 = sha256_text(canonical_json(messages))
            for family_index, endpoint in enumerate(endpoints):
                family = str(endpoint.family)
                key = (unit_id, int(order_variant), family)
                call_seed = (
                    int(judge_seed)
                    + int(unit_index) * 100
                    + int(order_variant) * 10
                    + int(family_index)
                )
                request_payload = chat_request_payload(
                    endpoint,
                    messages,
                    temperature=0.0,
                    max_tokens=int(estimated_output_tokens_per_call),
                    seed=call_seed,
                    response_schema=ForcedSwapJudgeOutput,
                )
                if not request_payload_has_schema(request_payload):
                    raise RuntimeError(
                        "forced-swap request payload lacks structured schema"
                    )
                plan_by_key[key] = {
                    "unit": unit,
                    "unit_index": unit_index,
                    "order_variant": int(order_variant),
                    "family_index": family_index,
                    "candidate_mapping": candidate_mapping,
                    "messages": messages,
                    "prompt_sha256": prompt_sha256,
                    "seed": call_seed,
                }
                payload_text = canonical_json(request_payload)
                base_input_tokens_est = estimate_tokens(payload_text)
                input_tokens_est = conservative_token_bound(
                    payload_text, safety_factor=input_token_safety_factor
                )
                pricing = family_pricing[family]
                cost_row = {
                        "unit_id": unit_id,
                        "order_variant": int(order_variant),
                        "judge_family": family,
                        "candidate_order": list(order),
                        "seed": call_seed,
                        "input_tokens_est": input_tokens_est,
                        "base_input_tokens_est": base_input_tokens_est,
                        "max_output_tokens": int(estimated_output_tokens_per_call),
                        "max_http_attempts": 1,
                        "prompt_sha256": prompt_sha256,
                        "request_payload_sha256": sha256_text(
                            canonical_json(request_payload)
                        ),
                        "request_payload_includes_schema": (
                            request_payload_has_schema(request_payload)
                        ),
                        "pricing_usd_per_mtok": pricing,
                        "maximum_cost_usd": input_tokens_est / 1_000_000
                        * pricing["input"]
                        + int(estimated_output_tokens_per_call) / 1_000_000
                        * pricing["output"],
                    }
                cost_row["physical_call_key"] = make_physical_call_key(
                    stage=stage,
                    record_ids={
                        "unit_id": unit_id,
                        "order_variant": int(order_variant),
                        "judge_family": family,
                    },
                    prompt_sha256=prompt_sha256,
                    endpoint=endpoint,
                    request_parameters={
                        "temperature": 0.0,
                        "max_tokens": int(estimated_output_tokens_per_call),
                        "seed": int(call_seed),
                        "response_schema": ForcedSwapJudgeOutput.__name__,
                        "request_payload_sha256": cost_row[
                            "request_payload_sha256"
                        ],
                        "retries": 1,
                        "study_freeze_sha256": study_freeze_sha256,
                    },
                )
                cost_rows.append(cost_row)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_judge_calls.jsonl"
    ledger_path = out_dir / "judge_call_ledger.jsonl"
    judgment_path = out_dir / "judgments.jsonl"
    score_path = out_dir / "scores.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "run_manifest.json"
    estimate_path = out_dir / "cost_estimate.json"
    call_plan_path = out_dir / "call_plan.jsonl"
    attestation_path = out_dir / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=overwrite,
        stage="PM-v2 forced-swap evaluation",
    )
    if overwrite:
        targets = (
            raw_path,
            judgment_path,
            score_path,
            summary_path,
            attestation_path,
        )
        if not run:
            targets = (*targets, manifest_path, estimate_path, call_plan_path)
        for path in targets:
            if path.exists():
                path.unlink()

    plan_by_physical_key = {
        str(row["physical_call_key"]): row for row in cost_rows
    }
    if len(plan_by_physical_key) != len(cost_rows):
        raise RuntimeError("duplicate forced-swap physical-call key")
    attempt_ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={key: 1 for key in plan_by_physical_key},
        maximum_total_attempts=int(max_api_calls),
    )
    ledger_rows = attempt_ledger.event_rows
    pending_cost_rows = [
        row
        for row in cost_rows
        if not attempt_ledger.succeeded(str(row["physical_call_key"]))
        and not attempt_ledger.exhausted(str(row["physical_call_key"]))
    ]
    historical_attempts = attempt_ledger.started_attempts
    maximum_physical_attempts = historical_attempts + len(pending_cost_rows)
    budgeted_cost_rows = [
        *[
            plan_by_physical_key[key]
            for key in sorted(attempt_ledger.started_call_keys)
        ],
        *pending_cost_rows,
    ]
    total_input_tokens = sum(
        int(row["input_tokens_est"]) for row in budgeted_cost_rows
    )
    total_output_tokens = sum(
        int(row["max_output_tokens"]) for row in budgeted_cost_rows
    )
    input_counts = [int(row["input_tokens_est"]) for row in budgeted_cost_rows]
    estimate_payload = {
        "stage": "pm_v2_forced_swap_key_claim",
        "full_logical_api_calls": len(cost_rows),
        "historical_physical_http_attempts": historical_attempts,
        "planned_new_api_calls": len(pending_cost_rows),
        "maximum_physical_http_attempts": maximum_physical_attempts,
        "expected_units": len(sampled),
        "order_variants": list(order_variants),
        "judge_families": endpoint_families,
        "endpoint_contracts": endpoint_contracts,
        "judge_seed": int(judge_seed),
        "full_expected_units_sha256": sha256_text(canonical_json(expected_units)),
        "total_input_tokens_est": total_input_tokens,
        "max_input_tokens_per_call_est": max(input_counts, default=0),
        "total_output_tokens_est": total_output_tokens,
        "estimated_cost_usd": sum(
            float(row["maximum_cost_usd"]) for row in budgeted_cost_rows
        ),
        "pricing_usd_per_mtok": family_pricing,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "call_plan_sha256": sha256_text(canonical_json(pending_cost_rows)),
        "ledger_sha256": sha256_text(canonical_json(ledger_rows)),
        "sample_units_sha256": sample_sha256,
        "sample_selection_contract_sha256": selection_contract_sha256,
        "prompt_contract_sha256": forced_swap_prompt_contract_hash(),
        "study_freeze_sha256": study_freeze_sha256,
        "endpoint_contracts": endpoint_contracts,
        "compatibility_thresholds": compatibility_thresholds,
        "efficacy_thresholds": efficacy_thresholds,
        "observed_cost_match_report": observed_cost_match_report,
        "judge_retries": 1,
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }
    estimate = {
        **estimate_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(estimate_payload)),
    }
    budget_checks = {
        "api_calls": maximum_physical_attempts <= int(max_api_calls),
        "estimated_cost_usd": estimate["estimated_cost_usd"]
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": estimate["max_input_tokens_per_call_est"]
        <= int(max_input_tokens_per_call),
    }
    budget_gate = {
        "status": "PASS" if all(budget_checks.values()) else "FAIL",
        "checks": budget_checks,
        "limits": estimate_payload["budget_limits"],
    }

    manifest = ensure_run_manifest(
        manifest_path,
        {
            "stage": "pm_v2_forced_swap_key_claim",
            "evoemo_sha256": sha256_file(evoemo_path),
            "study_freeze_sha256": study_freeze_sha256,
            "study_freeze_file_sha256": sha256_file(study_freeze_path),
            "turn_sha256": {
                str(Path(path).resolve()): sha256_file(path) for path in turn_paths
            },
            "generation_attestation_sha256": attestation_hashes,
            "treatment": treatment,
            "baseline": baseline,
            "full_expected_units_sha256": sha256_text(canonical_json(expected_units)),
            "sample_units_sha256": sample_sha256,
            "sample_selection_contract": selection_contract,
            "sample_selection_contract_sha256": selection_contract_sha256,
            "sample_units": len(sampled),
            "order_variants": list(order_variants),
            "endpoint_contracts": endpoint_contracts,
            "judge_seed": int(judge_seed),
            "estimated_output_tokens_per_call": int(estimated_output_tokens_per_call),
            "prompt_contract_sha256": forced_swap_prompt_contract_hash(),
            "compatibility_thresholds": compatibility_thresholds,
            "efficacy_thresholds": efficacy_thresholds,
            "judge_retries": 1,
            "pricing_usd_per_mtok": family_pricing,
            "api_cost_planning": api_cost_planning,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "observed_cost_match_report": observed_cost_match_report,
        },
    )
    dry_summary = {
        "status": "DRY_RUN_COMPLETE" if not run else "STARTING",
        "treatment": treatment,
        "baseline": baseline,
        "sample_units": len(sampled),
        "sample_units_sha256": sample_sha256,
        "sample_selection_contract_sha256": selection_contract_sha256,
        "excluded_unit_ids": excluded_unit_ids,
        "order_variants": list(order_variants),
        "judge_families": endpoint_families,
        "endpoint_contracts": endpoint_contracts,
        "judge_seed": int(judge_seed),
        "full_expected_units_sha256": sha256_text(canonical_json(expected_units)),
        "full_logical_api_calls": len(cost_rows),
        "planned_new_api_calls": len(pending_cost_rows),
        "historical_physical_http_attempts": historical_attempts,
        "prompt_contract_sha256": forced_swap_prompt_contract_hash(),
        "compatibility_thresholds": compatibility_thresholds,
        "efficacy_thresholds": efficacy_thresholds,
        "study_freeze_sha256": study_freeze_sha256,
        "run_manifest_sha256": manifest["manifest_sha256"],
        "cost_estimate": estimate,
        "budget_gate": budget_gate,
        "observed_cost_match_report": observed_cost_match_report,
        "judge_retries": 1,
        "pricing_usd_per_mtok": family_pricing,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    }
    if not run:
        write_json(estimate_path, {**estimate, "budget_gate": budget_gate})
        write_jsonl(call_plan_path, pending_cost_rows)
        if budget_gate["status"] != "PASS":
            raise RuntimeError("forced-swap dry-run failed the budget gate")
        return dry_summary
    if budget_gate["status"] != "PASS":
        raise RuntimeError("forced-swap API run failed the budget gate")
    _require_saved_dry_run(
        estimate_path,
        call_plan_path,
        current_estimate=estimate,
        current_plan=pending_cost_rows,
    )
    if accept_cost_estimate_sha256 != estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "forced-swap API run requires exact --accept-cost-estimate-sha256"
        )

    expected_raw_keys = set(plan_by_key)

    def raw_rows_from_ledger() -> dict[tuple[str, int, str], dict[str, Any]]:
        rows: dict[tuple[str, int, str], dict[str, Any]] = {}
        terminal_by_physical_key = {
            str(row["call_key"]): row
            for row in attempt_ledger.event_rows
            if row.get("event") in {"SUCCEEDED", "FAILED"}
        }
        for physical_key in sorted(attempt_ledger.started_call_keys):
            cost_plan = plan_by_physical_key[physical_key]
            key = (
                str(cost_plan["unit_id"]),
                int(cost_plan["order_variant"]),
                str(cost_plan["judge_family"]),
            )
            plan = plan_by_key[key]
            start_row = next(
                row
                for row in attempt_ledger.event_rows
                if row["call_key"] == physical_key and row["event"] == "STARTED"
            )
            expected_record_ids = {
                "unit_id": key[0],
                "order_variant": key[1],
                "judge_family": key[2],
            }
            if (
                start_row.get("record_ids") != expected_record_ids
                or start_row.get("prompt_sha256") != plan["prompt_sha256"]
            ):
                raise RuntimeError(
                    f"stale forced-swap attempt-ledger provenance: {key}"
                )
            contract = endpoint_contracts[key[2]]
            row = {
                "unit_id": key[0],
                "order_variant": key[1],
                "judge_family": key[2],
                "judge_model": contract["model"],
                "endpoint_sha256": contract["sha256"],
                "candidate_mapping": plan["candidate_mapping"],
                "prompt_sha256": plan["prompt_sha256"],
                "prompt_contract_sha256": forced_swap_prompt_contract_hash(),
                "sample_units_sha256": sample_sha256,
                "study_freeze_sha256": study_freeze_sha256,
            }
            terminal = terminal_by_physical_key.get(physical_key)
            if terminal is not None and terminal.get("event") == "SUCCEEDED":
                usage_error = reported_prompt_token_error(
                    terminal.get("usage"),
                    maximum_prompt_tokens=int(cost_plan["input_tokens_est"]),
                    stage="persisted forced-swap judge",
                    require_positive=True,
                )
                if usage_error is not None:
                    raise RuntimeError(
                        f"successful forced-swap usage is invalid for {key}: "
                        f"{usage_error}"
                    )
                result_payload = terminal.get("result") or {}
                parsed = ForcedSwapJudgeOutput.model_validate(
                    result_payload.get("judgment")
                )
                request_log_payload = result_payload.get("request_log")
                if not isinstance(request_log_payload, Mapping):
                    raise RuntimeError(
                        f"successful forced-swap ledger lacks request log: {key}"
                    )
                row.update(
                    {
                        "judgment": parsed.model_dump(mode="json"),
                        "error": None,
                        "request_log": dict(request_log_payload),
                    }
                )
            else:
                row.update(
                    {
                        "judgment": None,
                        "error": (
                            str(terminal.get("error"))
                            if terminal is not None
                            else "UNKNOWN_AFTER_PERSISTED_STARTED_EVENT"
                        ),
                        "request_log": None,
                    }
                )
            rows[key] = row
        return rows

    raw_by_key = raw_rows_from_ledger()
    schema_failures_seen = sum(row.get("error") is not None for row in raw_by_key.values())
    futility_reason = forced_swap_schema_futility_reason(
        schema_failures=schema_failures_seen,
        expected_calls=len(expected_raw_keys),
        minimum_schema_success_rate=minimum_schema_success_rate,
    )
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    clients = (
        {str(endpoint.family): make_client(endpoint) for endpoint in endpoints}
        if pending_cost_rows and futility_reason is None
        else {}
    )
    try:
        for cost_plan in pending_cost_rows:
            if futility_reason is not None:
                break
            key = (
                str(cost_plan["unit_id"]),
                int(cost_plan["order_variant"]),
                str(cost_plan["judge_family"]),
            )
            unit_id, order_variant, family = key
            plan = plan_by_key[key]
            endpoint = endpoint_by_family[family]
            reservation = attempt_ledger.reserve(
                str(cost_plan["physical_call_key"]),
                record_ids={
                    "unit_id": unit_id,
                    "order_variant": order_variant,
                    "judge_family": family,
                },
                prompt_sha256=str(plan["prompt_sha256"]),
            )
            result = None
            try:
                result, parsed = clients[family].chat(
                    plan["messages"],
                    temperature=0.0,
                    max_tokens=int(estimated_output_tokens_per_call),
                    seed=int(plan["seed"]),
                    response_schema=ForcedSwapJudgeOutput,
                    retries=1,
                )
                assert parsed is not None
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:2000]}"
                result_usage = result.usage if result is not None else None
                attempt_ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=(
                        result.request_hash if result is not None else None
                    ),
                    usage=result_usage,
                    error=error,
                    metadata={
                        "plan_sha256": sha256_text(canonical_json(cost_plan)),
                        "endpoint_sha256": endpoint_contracts[family]["sha256"],
                        "study_freeze_sha256": study_freeze_sha256,
                    },
                )
                reported_input_tokens = int(
                    (result_usage or {}).get("prompt_tokens") or 0
                )
                if reported_input_tokens > int(cost_plan["input_tokens_est"]):
                    raise RuntimeError(
                        "reported forced-swap input usage exceeded the frozen "
                        "conservative bound after the terminal event was saved for "
                        f"{key}: {reported_input_tokens} > "
                        f"{cost_plan['input_tokens_est']}"
                    )
                schema_failures_seen += 1
                futility_reason = forced_swap_schema_futility_reason(
                    schema_failures=schema_failures_seen,
                    expected_calls=len(expected_raw_keys),
                    minimum_schema_success_rate=minimum_schema_success_rate,
                )
            else:
                usage_error = reported_prompt_token_error(
                    result.usage,
                    maximum_prompt_tokens=int(cost_plan["input_tokens_est"]),
                    stage="forced-swap judge",
                    require_positive=bool(
                        api_cost_planning["fail_on_reported_input_overrun"]
                    ),
                )
                request_log_payload = request_log(
                    stage=stage,
                    endpoint=endpoint,
                    messages=plan["messages"],
                    result=result,
                    parsed=parsed,
                    error=usage_error,
                    prompt_hash=plan["prompt_sha256"],
                    record_ids={
                        "unit_id": unit_id,
                        "order_variant": order_variant,
                    },
                )
                if usage_error is not None:
                    attempt_ledger.finish(
                        reservation,
                        succeeded=False,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=usage_error,
                        result={
                            "judgment": parsed.model_dump(mode="json"),
                            "request_log": request_log_payload,
                        },
                        metadata={
                            "plan_sha256": sha256_text(canonical_json(cost_plan)),
                            "endpoint_sha256": endpoint_contracts[family]["sha256"],
                            "study_freeze_sha256": study_freeze_sha256,
                        },
                    )
                    schema_failures_seen += 1
                    futility_reason = forced_swap_schema_futility_reason(
                        schema_failures=schema_failures_seen,
                        expected_calls=len(expected_raw_keys),
                        minimum_schema_success_rate=minimum_schema_success_rate,
                    )
                    continue
                attempt_ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=result.request_hash,
                    usage=result.usage,
                    error=None,
                    result={
                        "judgment": parsed.model_dump(mode="json"),
                        "request_log": request_log_payload,
                    },
                    metadata={
                        "plan_sha256": sha256_text(canonical_json(cost_plan)),
                        "endpoint_sha256": endpoint_contracts[family]["sha256"],
                        "study_freeze_sha256": study_freeze_sha256,
                    },
                )
    finally:
        for client in clients.values():
            client.close()
    ledger_rows = attempt_ledger.event_rows
    raw_by_key = raw_rows_from_ledger()
    write_jsonl(raw_path, [raw_by_key[key] for key in sorted(raw_by_key)])

    ordered_raw = [raw_by_key[key] for key in sorted(raw_by_key)]
    successful_raw = [row for row in ordered_raw if row.get("error") is None]
    schema_success_rate = len(successful_raw) / len(expected_raw_keys)
    judgments: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {}
    if len(successful_raw) == len(ordered_raw):
        judgments, scores, analysis = _summarize(
            raw_rows=successful_raw,
            sampled_units=sampled,
            treatment=treatment,
            baseline=baseline,
            endpoint_families=endpoint_families,
            bootstrap_seed=judge_seed + 100_000,
        )
    write_jsonl(judgment_path, judgments)
    write_jsonl(score_path, scores)
    compatibility_gate = forced_swap_compatibility_gate(
        analysis=analysis,
        endpoint_families=endpoint_families,
        compatibility_thresholds=compatibility_thresholds,
        schema_success_rate=schema_success_rate,
        schema_successes=len(successful_raw),
        schema_attempts=attempt_ledger.started_attempts,
        schema_expected_calls=len(expected_raw_keys),
        observed_cost_match_pass=(observed_cost_match_report["status"] == "PASS"),
        futility_reason=futility_reason,
    )
    efficacy_gate = forced_swap_efficacy_gate(
        analysis=analysis,
        thresholds=efficacy_thresholds,
        compatibility_pass=compatibility_gate["status"] == "PASS",
    )
    final_status = compatibility_gate["status"]
    support_key_claim_verified = efficacy_gate["status"] == "PASS"
    summary = {
        **dry_summary,
        "status": final_status,
        "raw_rows": len(ordered_raw),
        "judgment_rows": len(judgments),
        "score_rows": len(scores),
        "physical_http_attempts": attempt_ledger.started_attempts,
        "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
        "support_key_claim_verified": support_key_claim_verified,
        "compatibility_gate": compatibility_gate,
        "efficacy_gate": efficacy_gate,
        **analysis,
    }
    write_json(summary_path, summary)
    create_artifact_attestation(
        attestation_path,
        stage="pm_v2_forced_swap_key_claim",
        inputs={
            "evoemo": evoemo_path,
            "study_freeze": study_freeze_path,
            "run_manifest": manifest_path,
            "cost_estimate": estimate_path,
            "turn_treatment": turn_paths[0],
            "turn_baseline": turn_paths[1],
            "generation_attestation_treatment": generation_attestation_paths[0],
            "generation_attestation_baseline": generation_attestation_paths[1],
        },
        outputs={
            "raw_calls": (raw_path, True),
            "call_ledger": (ledger_path, True),
            "judgments": (judgment_path, True),
            "scores": (score_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "treatment": treatment,
            "baseline": baseline,
            "sample_units": len(sampled),
            "sample_units_sha256": sample_sha256,
            "order_variants": list(order_variants),
            "judge_families": endpoint_families,
            "endpoint_contracts": endpoint_contracts,
            "judge_seed": int(judge_seed),
            "prompt_contract_sha256": forced_swap_prompt_contract_hash(),
            "sample_selection_contract_sha256": selection_contract_sha256,
            "excluded_unit_ids": excluded_unit_ids,
            "compatibility_thresholds": compatibility_thresholds,
            "efficacy_thresholds": efficacy_thresholds,
            "compatibility_gate": compatibility_gate,
            "efficacy_gate": efficacy_gate,
            "support_key_claim_verified": support_key_claim_verified,
            "observed_cost_match_report": observed_cost_match_report,
            "judge_retries": 1,
            "pricing_usd_per_mtok": family_pricing,
            "api_cost_planning": api_cost_planning,
            "physical_http_attempts": attempt_ledger.started_attempts,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
        },
        expected={
            "api_calls": len(cost_rows),
            "physical_http_attempts": attempt_ledger.started_attempts,
            "judgment_rows": len(judgments),
            "score_rows": len(scores),
            "status": final_status,
        },
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary
