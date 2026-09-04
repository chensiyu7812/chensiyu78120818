from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

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
from .evoemo import evaluator_context, load_evoemo
from .contracts import (
    MemorySource,
    StrategyMode,
    canonical_action_id,
    parse_action_id,
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
from .pm_v2_contracts import CompositeSpec, ResponseDimensions, RiskDimensions
from .pm_v2_judging import (
    ResponseJudgeOutput,
    RiskJudgeOutput,
    build_response_messages,
    build_risk_messages,
    composite_weights_hash,
    dimension_health,
    prompt_contract_hash,
    validate_raw_judge_family_health,
    validate_raw_judge_family_subgroup_health,
)
from .pm_v2_model import applicable_risk_fields
from .text import conservative_token_bound, estimate_tokens


DIMENSIONS = tuple(ResponseDimensions.model_fields)
RISK_DIMENSIONS = tuple(RiskDimensions.model_fields)


def _external_action_contract(turn_row: Mapping[str, Any]) -> dict[str, str]:
    """Return canonical requested/effective actions without trusting condition names."""

    raw_requested = str(
        turn_row.get("requested_action_id") or turn_row.get("action_id") or ""
    )
    try:
        parse_action_id(raw_requested)
        requested = raw_requested
    except ValueError:
        requested = ""

    selected_sources: set[MemorySource] = set()
    for item in turn_row.get("selected_memory") or []:
        try:
            selected_sources.add(MemorySource(str(item.get("source") or "")))
        except ValueError:
            continue
    inferred_effective = canonical_action_id(
        selected_sources,
        StrategyMode.RS
        if bool(turn_row.get("selected_strategy") or [])
        else StrategyMode.R0,
    )
    raw_realized = str(
        turn_row.get("realized_action_id")
        or turn_row.get("effective_action_id")
        or ""
    )
    try:
        parse_action_id(raw_realized)
        realized = raw_realized
    except ValueError:
        realized = inferred_effective
    if realized != inferred_effective:
        raise RuntimeError("external turn realized_action_id mismatches selected evidence")
    raw_effective = str(turn_row.get("effective_action_id") or realized)
    if raw_effective != realized:
        raise RuntimeError("legacy effective_action_id mismatches realized_action_id")
    # Legacy fixed baselines may not expose a canonical requested action.  Their
    # realized evidence action is the only defensible applicability contract.
    if not requested:
        requested = realized
    return {
        "requested_action_id": requested,
        "realized_action_id": realized,
        "effective_action_id": realized,
    }


def _unit_key(row: dict[str, Any]) -> tuple[str, int, int, str, int]:
    return (
        str(row["user_id"]),
        int(row["topic_index"]),
        int(row["seed"]),
        str(row["simulator_id"]),
        int(row["turn_index"]),
    )


def load_fixed_turns(
    paths: Sequence[str | Path],
    *,
    conditions: Sequence[str],
    turn_indices: Sequence[int],
    expected_units: Sequence[tuple[str, int, int, str, int]] | None = None,
    full_expected_units: Sequence[tuple[str, int, int, str, int]] | None = None,
    excluded_units: Sequence[tuple[str, int, int, str, int]] = (),
) -> dict[tuple[str, int, int, str, int, str], dict[str, Any]]:
    requested = set(conditions)
    turns = set(int(value) for value in turn_indices)
    expected_unit_set = set(expected_units) if expected_units is not None else None
    full_unit_set = (
        set(full_expected_units)
        if full_expected_units is not None
        else expected_unit_set
    )
    excluded_unit_set = set(excluded_units)
    if excluded_unit_set and full_unit_set is None:
        raise ValueError("excluded_units requires full_expected_units")
    if full_unit_set is not None:
        if not excluded_unit_set <= full_unit_set:
            raise RuntimeError("excluded external units are outside the frozen universe")
        required_scoring = full_unit_set - excluded_unit_set
        if expected_unit_set is None:
            expected_unit_set = required_scoring
        elif expected_unit_set != required_scoring:
            raise RuntimeError(
                "external scoring units are not the exact frozen universe minus "
                "the exact excluded pilot set"
            )
    full_rows: dict[tuple[str, int, int, str, int, str], dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            condition = str(row.get("condition"))
            if condition not in requested or int(row.get("turn_index", -1)) not in turns:
                continue
            if row.get("interaction_mode") != "fixed":
                raise ValueError("PM-v2 external evaluation requires fixed-input turns")
            key = (*_unit_key(row), condition)
            if full_unit_set is not None and key[:-1] not in full_unit_set:
                raise RuntimeError(f"unexpected external unit in turn matrix: {key[:-1]}")
            if key in full_rows:
                raise ValueError(f"duplicate external turn key: {key}")
            full_rows[key] = row
    universe_units = sorted(
        full_unit_set
        if full_unit_set is not None
        else {key[:-1] for key in full_rows}
    )
    missing_full = [
        (*unit, condition)
        for unit in universe_units
        for condition in requested
        if (*unit, condition) not in full_rows
    ]
    if missing_full:
        raise RuntimeError(f"external turn matrix is incomplete: {missing_full[:10]}")
    for unit in universe_units:
        unit_rows = [full_rows[(*unit, condition)] for condition in requested]
        context_hashes = {str(row.get("context_sha256")) for row in unit_rows}
        seeker_messages = {str(row.get("seeker_message")) for row in unit_rows}
        if len(context_hashes) != 1 or len(seeker_messages) != 1:
            raise RuntimeError(
                f"fixed-input mismatch for unit {unit}: "
                f"context_hashes={context_hashes}, seeker_messages={seeker_messages}"
            )
    scoring_units = sorted(
        expected_unit_set
        if expected_unit_set is not None
        else {key[:-1] for key in full_rows}
    )
    rows = {
        (*unit, condition): full_rows[(*unit, condition)]
        for unit in scoring_units
        for condition in requested
    }
    return rows


def expected_external_units(
    evoemo_path: str | Path,
    *,
    seeds: Sequence[int],
    simulator_id: str,
    turn_indices: Sequence[int],
) -> list[tuple[str, int, int, str, int]]:
    units = {
        (
            str(user["id"]),
            int(topic["idx"]),
            int(seed),
            str(simulator_id),
            int(turn_index),
        )
        for user in load_evoemo(evoemo_path)
        for topic in user.get("subsequent_topics") or []
        for seed in seeds
        for turn_index in turn_indices
    }
    if not units:
        raise ValueError("frozen EvoEmo contract produced no expected units")
    return sorted(units)


def _authorized_context_map(evoemo_path: str | Path) -> dict[tuple[str, int], str]:
    mapping: dict[tuple[str, int], str] = {}
    for user in load_evoemo(evoemo_path):
        for topic in user.get("subsequent_topics") or []:
            key = (str(user["id"]), int(topic["idx"]))
            mapping[key] = canonical_json(evaluator_context(user, topic))
    return mapping


def build_external_pointwise_case(
    turn_row: Mapping[str, Any],
    *,
    authorized_user_context: str,
) -> dict[str, Any]:
    """Build the exact anonymous response/risk prompts used by external judging."""

    from .contracts import RuntimeState
    from .pm_v2_data import runtime_to_pmv2_state

    runtime = RuntimeState(
        state_id=str(turn_row["state_id"]),
        card_id=str(turn_row["card_id"]),
        user_id=str(turn_row["user_id"]),
        split="evoemo_test",
        semantic_family="evoemo_dialogue_generation",
        current_user_text=str(turn_row["seeker_message"]),
        current_session_history=[
            {
                "role": "user" if turn["role"] == "seeker" else "assistant",
                "content": turn["content"],
            }
            for turn in turn_row["context_before_turn"][-8:]
        ],
        current_session_summary="",
        session_index=1,
        inventory={
            "MP": {"available": False, "count": 0, "estimated_tokens": 0},
            "MS": {"available": False, "count": 0, "estimated_tokens": 0},
            "ME": {"available": False, "count": 0, "estimated_tokens": 0},
        },
        allowed_actions=["M0+R0", "M0+RS"],
    )
    state = runtime_to_pmv2_state(runtime)
    selected_context = "\n".join(
        [
            f"MEMORY: {item.get('text', '')}"
            for item in turn_row.get("selected_memory") or []
        ]
        + [
            "STRATEGY: "
            + str(item.get("guidance_text") or item.get("example_response") or "")
            for item in turn_row.get("selected_strategy") or []
        ]
    )
    response_messages = build_response_messages(
        state=state,
        authorized_user_context=authorized_user_context,
        candidate_response=str(turn_row["supporter_message"]),
    )
    risk_messages = build_risk_messages(
        state=state,
        authorized_user_context=authorized_user_context,
        selected_context=selected_context,
        candidate_response=str(turn_row["supporter_message"]),
    )
    return {
        "state": state,
        "selected_context": selected_context,
        "response_messages": response_messages,
        "risk_messages": risk_messages,
    }


def _median_response(outputs: Sequence[ResponseJudgeOutput]) -> ResponseDimensions:
    return ResponseDimensions(
        **{
            name: float(median(float(getattr(output, name)) for output in outputs))
            for name in DIMENSIONS
        }
    )


def _median_risk(outputs: Sequence[RiskJudgeOutput]) -> RiskDimensions:
    return RiskDimensions(
        **{
            name: float(median(float(getattr(output, name)) for output in outputs))
            for name in RISK_DIMENSIONS
        }
    )


def _mad_response(outputs: Sequence[ResponseJudgeOutput]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in DIMENSIONS:
        values = np.asarray([float(getattr(output, name)) for output in outputs])
        result[name] = float(np.median(np.abs(values - np.median(values))))
    return result


def _mad_risk(outputs: Sequence[RiskJudgeOutput]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in RISK_DIMENSIONS:
        values = np.asarray([float(getattr(output, name)) for output in outputs])
        result[name] = float(np.median(np.abs(values - np.median(values))))
    return result


def _bootstrap_cluster_delta(
    rows: Sequence[dict[str, Any]],
    *,
    cluster_field: str,
    n_resamples: int = 10000,
    seed: int = 17,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot bootstrap empty paired rows")
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row[cluster_field])].append(float(row["delta"]))
    cluster_ids = sorted(clusters)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        values = [value for cluster in sampled for value in clusters[str(cluster)]]
        samples[index] = float(np.mean(values))
    estimate = float(np.mean([row["delta"] for row in rows]))
    return {
        "estimate": estimate,
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
        "p_delta_lt_0": float(np.mean(samples < 0.0)),
        "n_clusters": len(cluster_ids),
        "n_resamples": n_resamples,
    }


def validate_external_score_table(
    rows: Sequence[dict[str, Any]],
    *,
    duplicate_exact_match_rate: float,
    minimum_reliable_rate: float,
    reliable_mad_threshold: float = 0.75,
    minimum_low_mad_coverage_per_dimension: float = 0.80,
    minimum_low_mad_coverage_per_condition_dimension: float = 0.80,
    reject_constant_response_dimensions: bool,
    reject_constant_risk_dimensions: bool,
    maximum_absolute_dimension_correlation: float = 0.95,
    composite_support_exact_match_rate: float = 0.98,
    maximum_absolute_composite_support_correlation: float = 0.995,
    composite_spec: CompositeSpec | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty external score table")
    matrix = np.asarray([[float(row[name]) for name in DIMENSIONS] for row in rows])
    risk_matrix = np.asarray([[float(row[name]) for name in RISK_DIMENSIONS] for row in rows])
    (
        response_duplicates,
        response_correlations,
        response_constants,
        response_prevalence,
        _response_mutual_exclusivity,
    ) = dimension_health(
        matrix,
        DIMENSIONS,
        prefix="response",
        duplicate_exact_match_rate=duplicate_exact_match_rate,
        maximum_absolute_dimension_correlation=maximum_absolute_dimension_correlation,
    )
    (
        risk_duplicates,
        risk_correlations,
        risk_constants,
        risk_prevalence,
        _risk_mutual_exclusivity,
    ) = dimension_health(
        risk_matrix,
        RISK_DIMENSIONS,
        prefix="risk",
        duplicate_exact_match_rate=duplicate_exact_match_rate,
        maximum_absolute_dimension_correlation=maximum_absolute_dimension_correlation,
    )
    duplicate_pairs = [*response_duplicates, *risk_duplicates]
    high_correlation_pairs = [*response_correlations, *risk_correlations]
    constants = [
        *(response_constants if reject_constant_response_dimensions else []),
        *(risk_constants if reject_constant_risk_dimensions else []),
    ]
    reliable_rate = float(np.mean([bool(row["label_reliable"]) for row in rows]))
    frozen_composite = composite_spec or CompositeSpec()
    composite_values = np.asarray(
        [
            frozen_composite.score(
                ResponseDimensions(
                    **{name: float(row[name]) for name in DIMENSIONS}
                )
            )
            for row in rows
        ],
        dtype=float,
    )
    support_values = (matrix[:, DIMENSIONS.index("emotional_support")] - 1.0) / 4.0
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
    dimension_mad_fields = {
        **{
            f"response.{name}": ("response_dimension_mad", name)
            for name in DIMENSIONS
        },
        **{
            f"risk.{name}": ("risk_dimension_mad", name)
            for name in RISK_DIMENSIONS
        },
    }
    mad_contract_complete = all(
        isinstance(row.get(container), Mapping)
        and name in row[container]
        for row in rows
        for container, name in dimension_mad_fields.values()
    )
    dimension_low_mad_coverage = {
        dimension: float(
            np.mean(
                [
                    float((row.get(container) or {}).get(name, float("inf")))
                    <= reliable_mad_threshold
                    for row in rows
                ]
            )
        )
        for dimension, (container, name) in dimension_mad_fields.items()
    }
    condition_contract_complete = all(bool(row.get("condition")) for row in rows)
    condition_dimension_low_mad_coverage: dict[str, dict[str, float]] = {}
    for condition in sorted(
        {str(row.get("condition") or "__missing__") for row in rows}
    ):
        condition_rows = [
            row
            for row in rows
            if str(row.get("condition") or "__missing__") == condition
        ]
        condition_dimension_low_mad_coverage[condition] = {
            dimension: float(
                np.mean(
                    [
                        float((row.get(container) or {}).get(name, float("inf")))
                        <= reliable_mad_threshold
                        for row in condition_rows
                    ]
                )
            )
            for dimension, (container, name) in dimension_mad_fields.items()
        }
    low_coverage_dimensions = sorted(
        dimension
        for dimension, coverage in dimension_low_mad_coverage.items()
        if coverage < minimum_low_mad_coverage_per_dimension
    )
    low_coverage_condition_dimensions = sorted(
        f"{condition}/{dimension}"
        for condition, coverages in condition_dimension_low_mad_coverage.items()
        for dimension, coverage in coverages.items()
        if coverage < minimum_low_mad_coverage_per_condition_dimension
    )
    report = {
        "n": len(rows),
        "duplicate_dimension_pairs": duplicate_pairs,
        "high_correlation_dimension_pairs": high_correlation_pairs,
        "constant_dimensions": constants,
        "response_constant_dimensions": response_constants,
        "risk_constant_dimensions": risk_constants,
        "response_dimension_prevalence": response_prevalence,
        "risk_dimension_prevalence": risk_prevalence,
        "maximum_absolute_dimension_correlation": maximum_absolute_dimension_correlation,
        "composite_spec": frozen_composite.model_dump(mode="json"),
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
        "reliable_rate": reliable_rate,
        "minimum_reliable_rate": minimum_reliable_rate,
        "joint_reliable_rate_is_diagnostic_only": True,
        "dimension_mad_contract_complete": mad_contract_complete,
        "condition_contract_complete": condition_contract_complete,
        "reliable_mad_threshold": reliable_mad_threshold,
        "dimension_low_mad_coverage": dimension_low_mad_coverage,
        "condition_dimension_low_mad_coverage": (
            condition_dimension_low_mad_coverage
        ),
        "minimum_low_mad_coverage_per_dimension": (
            minimum_low_mad_coverage_per_dimension
        ),
        "minimum_low_mad_coverage_per_condition_dimension": (
            minimum_low_mad_coverage_per_condition_dimension
        ),
        "low_coverage_dimensions": low_coverage_dimensions,
        "low_coverage_condition_dimensions": low_coverage_condition_dimensions,
        "status": "PASS",
    }
    if (
        duplicate_pairs
        or high_correlation_pairs
        or constants
        or not mad_contract_complete
        or not condition_contract_complete
        or low_coverage_dimensions
        or low_coverage_condition_dimensions
        or composite_support_exact_failure
        or composite_support_correlation_failure
    ):
        report["status"] = "FAIL"
        raise RuntimeError("PM-v2 external judge gate failed: " + canonical_json(report))
    return report


def run_external_response_evaluation(
    *,
    evoemo_path: str | Path,
    turn_paths: Sequence[str | Path],
    conditions: Sequence[str],
    treatment: str,
    turn_indices: Sequence[int],
    endpoints: Sequence[Endpoint],
    out_dir: str | Path,
    run: bool,
    max_api_calls: int,
    expected_units: Sequence[tuple[str, int, int, str, int]],
    full_expected_units: Sequence[tuple[str, int, int, str, int]],
    composite_spec: CompositeSpec,
    labeling: Mapping[str, Any],
    minimum_low_mad_coverage_per_dimension: float,
    minimum_low_mad_coverage_per_condition_dimension: float,
    required_conditions: Sequence[str],
    generation_attestation_paths: Sequence[str | Path],
    study_freeze_sha256: str,
    require_key_claim_verification: bool,
    pricing_usd_per_mtok: Mapping[str, Mapping[str, float]],
    api_cost_planning: Mapping[str, Any],
    primary_bootstrap_cluster: str,
    sensitivity_bootstrap_cluster: str,
    pointwise_schema_smoke_summary_path: str | Path,
    pointwise_schema_smoke_attestation_path: str | Path,
    pointwise_schema_smoke_verification: Mapping[str, Any],
    key_claim_verification_path: str | Path | None = None,
    key_claim_verification_attestation_path: str | Path | None = None,
    excluded_unit_ids: Sequence[str] = (),
    pilot_selection_contract_sha256: str | None = None,
    full_expected_units_sha256: str | None = None,
    observed_cost_match_report: Mapping[str, Any] | None = None,
    accept_cost_estimate_sha256: str | None = None,
    max_estimated_usd: float = 10.0,
    max_input_tokens_per_call: int = 12_000,
    estimated_response_output_tokens: int = 600,
    estimated_risk_output_tokens: int = 700,
    overwrite: bool = False,
    seed: int = 3701,
) -> dict[str, Any]:
    if run and overwrite:
        raise RuntimeError(
            "paid API runs prohibit overwrite; use a new output directory"
        )
    if treatment not in conditions:
        raise ValueError("treatment must be included in conditions")
    if (
        pointwise_schema_smoke_verification.get("status") != "PASS"
        or not pointwise_schema_smoke_summary_path
        or not pointwise_schema_smoke_attestation_path
    ):
        raise RuntimeError(
            "full external evaluation requires a passing pointwise schema smoke "
            "before any judge client can be created"
        )
    if str(primary_bootstrap_cluster) != "user_id":
        raise ValueError("external primary bootstrap cluster must be user_id")
    if str(sensitivity_bootstrap_cluster) != "scenario":
        raise ValueError("external sensitivity bootstrap cluster must be scenario")
    bootstrap_cluster_contract = {
        "primary": "user_id",
        "primary_score_field": "user_cluster",
        "sensitivity": "scenario",
        "sensitivity_score_field": "scenario_cluster",
    }
    api_cost_planning = dict(api_cost_planning)
    if set(api_cost_planning) != {
        "input_token_safety_factor",
        "fail_on_reported_input_overrun",
    }:
        raise ValueError("external api_cost_planning contract is incomplete")
    input_token_safety_factor = float(
        api_cost_planning["input_token_safety_factor"]
    )
    if input_token_safety_factor < 1.0 or not bool(
        api_cost_planning["fail_on_reported_input_overrun"]
    ):
        raise ValueError("external evaluation requires fail-closed cost planning")
    if (
        not isinstance(observed_cost_match_report, Mapping)
        or observed_cost_match_report.get("status") != "PASS"
        or not bool(observed_cost_match_report.get("check"))
    ):
        raise RuntimeError(
            "full external evaluation requires a passing observed-token cost-match gate"
        )
    key_claim_attestation_verification = None
    key_claim_verification = None
    if require_key_claim_verification:
        if (
            key_claim_verification_path is None
            or key_claim_verification_attestation_path is None
        ):
            raise RuntimeError(
                "full external evaluation requires the frozen forced-swap summary "
                "and artifact attestation before any API call"
            )
        key_claim_attestation_verification = require_artifact_attestation(
            key_claim_verification_attestation_path,
            required_stage="pm_v2_forced_swap_key_claim",
            required_output_paths={"summary": key_claim_verification_path},
            expected_freeze_sha256=study_freeze_sha256,
        )
        key_claim_verification = read_json(key_claim_verification_path)
        if (
            key_claim_verification.get("status") != "PASS"
            or (key_claim_verification.get("compatibility_gate") or {}).get("status")
            != "PASS"
            or key_claim_verification.get("study_freeze_sha256")
            != study_freeze_sha256
            or not bool(key_claim_verification.get("support_key_claim_verified"))
            or key_claim_verification.get("excluded_unit_ids")
            != list(excluded_unit_ids)
            or key_claim_verification.get("sample_selection_contract_sha256")
            != pilot_selection_contract_sha256
        ):
            raise RuntimeError(
                "forced-swap artifact is incompatible or failed; full external API "
                "evaluation is blocked"
            )
    missing_conditions = sorted(set(required_conditions) - set(conditions))
    if missing_conditions:
        raise ValueError(f"external evaluation lacks required conditions: {missing_conditions}")
    families = {endpoint.family for endpoint in endpoints}
    minimum_families = int(labeling["minimum_families"])
    if (
        None in families
        or len(families) < minimum_families
        or len(families) != len(endpoints)
    ):
        raise ValueError("external evaluation requires distinct independent judge families")
    family_pricing = {
        str(family): {
            "input": float(values["input"]),
            "output": float(values["output"]),
        }
        for family, values in pricing_usd_per_mtok.items()
    }
    if set(family_pricing) != {str(value) for value in families} or any(
        set(values) != {"input", "output"}
        or any(value <= 0.0 for value in values.values())
        for values in family_pricing.values()
    ):
        raise ValueError(
            "external judge pricing must exactly cover judge families with "
            "strictly positive rates"
        )
    full_units = sorted(full_expected_units)
    units = sorted(expected_units)
    excluded_units = sorted(set(full_units) - set(units))
    if set(units) | set(excluded_units) != set(full_units):
        raise RuntimeError("external full/scoring unit partition is invalid")
    derived_excluded_ids = [
        sha256_text(canonical_json(unit))[:24] for unit in excluded_units
    ]
    if sorted(derived_excluded_ids) != sorted(str(value) for value in excluded_unit_ids):
        raise RuntimeError("external excluded unit IDs do not bind the exact pilot set")
    if full_expected_units_sha256 is None or sha256_text(
        canonical_json(full_units)
    ) != str(full_expected_units_sha256):
        raise RuntimeError("full external expected-unit hash is missing or stale")
    matrix = load_fixed_turns(
        turn_paths,
        conditions=conditions,
        turn_indices=turn_indices,
        expected_units=units,
        full_expected_units=full_units,
        excluded_units=excluded_units,
    )
    actual_units_sha256 = sha256_text(canonical_json(units))
    excluded_set = set(str(value) for value in excluded_unit_ids)
    if any(sha256_text(canonical_json(unit))[:24] in excluded_set for unit in units):
        raise RuntimeError("forced-swap pilot units leaked into full external evaluation")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_judge_calls.jsonl"
    ledger_path = out_dir / "judge_call_ledger.jsonl"
    score_path = out_dir / "response_scores.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "run_manifest.json"
    estimate_path = out_dir / "cost_estimate.json"
    call_plan_path = out_dir / "call_plan.jsonl"
    attestation_path = out_dir / "artifact_attestation.json"
    forbid_overwrite_of_spent_attempts(
        ledger_path,
        overwrite=overwrite,
        stage="PM-v2 external response evaluation",
    )
    if overwrite:
        targets = (raw_path, score_path, summary_path, attestation_path)
        if not run:
            targets = (*targets, manifest_path, estimate_path, call_plan_path)
        for path in targets:
            if path.exists():
                path.unlink()

    endpoint_contracts = {
        str(endpoint.family): {
            "model": endpoint.model,
            "family": endpoint.family,
            "base_url": endpoint.base_url,
            "sha256": sha256_text(
                canonical_json(
                    {
                        "model": endpoint.model,
                        "family": endpoint.family,
                        "base_url": endpoint.base_url,
                    }
                )
            ),
        }
        for endpoint in endpoints
    }
    composite_weights_sha256 = composite_weights_hash(composite_spec)
    stage = "pm_v2_external_response_and_risk_evaluation"
    authorized = _authorized_context_map(evoemo_path)

    plan_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    execution_by_call_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    cost_rows: list[dict[str, Any]] = []
    for unit_index, unit in enumerate(units):
        user_id, topic_index, unit_seed, simulator_id, turn_index = unit
        unit_id = sha256_text(canonical_json(unit))[:24]
        context = authorized[(user_id, topic_index)]
        for condition in conditions:
            turn_row = matrix[(*unit, condition)]
            action_contract = _external_action_contract(turn_row)
            pointwise_case = build_external_pointwise_case(
                turn_row,
                authorized_user_context=context,
            )
            response_messages = pointwise_case["response_messages"]
            risk_messages = pointwise_case["risk_messages"]
            observed_tokens = int(
                turn_row.get("input_tokens")
                or (turn_row.get("cost") or {}).get("total_input_tokens")
                or 0
            )
            for family_index, endpoint in enumerate(endpoints):
                family = str(endpoint.family)
                key = (unit_id, condition, family)
                response_prompt_hash = sha256_text(canonical_json(response_messages))
                risk_prompt_hash = sha256_text(canonical_json(risk_messages))
                plan_by_key[key] = {
                    "unit": unit,
                    "unit_index": unit_index,
                    "family_index": family_index,
                    "turn_row": turn_row,
                    "response_messages": response_messages,
                    "risk_messages": risk_messages,
                    "response_prompt_hash": response_prompt_hash,
                    "risk_prompt_hash": risk_prompt_hash,
                    "observed_input_tokens": observed_tokens,
                    "requested_action_id": action_contract["requested_action_id"],
                    "realized_action_id": action_contract["realized_action_id"],
                    "effective_action_id": action_contract["effective_action_id"],
                }
                response_seed = seed + unit_index * 100 + family_index
                for judge_type, messages, schema, max_output_tokens, call_seed in (
                    (
                        "response",
                        response_messages,
                        ResponseJudgeOutput,
                        int(estimated_response_output_tokens),
                        response_seed,
                    ),
                    (
                        "risk",
                        risk_messages,
                        RiskJudgeOutput,
                        int(estimated_risk_output_tokens),
                        response_seed + 1,
                    ),
                ):
                    request_payload = chat_request_payload(
                        endpoint,
                        messages,
                        temperature=0.0,
                        max_tokens=max_output_tokens,
                        seed=call_seed,
                        response_schema=schema,
                    )
                    if not request_payload_has_schema(request_payload):
                        raise RuntimeError(
                            "external judge request payload lacks structured schema"
                        )
                    payload_text = canonical_json(request_payload)
                    base_input_tokens_est = estimate_tokens(payload_text)
                    input_tokens_est = conservative_token_bound(
                        payload_text, safety_factor=input_token_safety_factor
                    )
                    pricing = family_pricing[family]
                    plan_row = {
                            "unit_id": unit_id,
                            "condition": condition,
                            "judge_family": family,
                            "judge_model": endpoint.model,
                            "judge_type": judge_type,
                            "seed": call_seed,
                            "input_tokens_est": input_tokens_est,
                            "base_input_tokens_est": base_input_tokens_est,
                            "max_output_tokens": max_output_tokens,
                            "max_http_attempts": 1,
                            "prompt_hash": sha256_text(canonical_json(messages)),
                            "request_payload_sha256": sha256_text(
                                canonical_json(request_payload)
                            ),
                            "request_payload_includes_schema": (
                                request_payload_has_schema(request_payload)
                            ),
                            "pricing_usd_per_mtok": pricing,
                            "maximum_cost_usd": input_tokens_est / 1_000_000
                            * pricing["input"]
                            + max_output_tokens / 1_000_000 * pricing["output"],
                    }
                    plan_row["physical_call_key"] = make_physical_call_key(
                        stage=stage,
                        record_ids={
                            "unit_id": unit_id,
                            "condition": condition,
                            "judge_family": family,
                            "judge_type": judge_type,
                        },
                        prompt_sha256=str(plan_row["prompt_hash"]),
                        endpoint=endpoint,
                        request_parameters={
                            "temperature": 0.0,
                            "max_tokens": int(max_output_tokens),
                            "seed": int(call_seed),
                            "response_schema": schema.__name__,
                            "request_payload_sha256": plan_row[
                                "request_payload_sha256"
                            ],
                            "retries": 1,
                            "study_freeze_sha256": study_freeze_sha256,
                        },
                    )
                    cost_rows.append(plan_row)
                    execution_by_call_key[
                        (unit_id, condition, family, judge_type)
                    ] = {
                        "messages": messages,
                        "schema": schema,
                        "unit": unit,
                        "observed_input_tokens": observed_tokens,
                    }

    def call_key(row):
        return (
            str(row["unit_id"]),
            str(row["condition"]),
            str(row["judge_family"]),
            str(row["judge_type"]),
        )

    plan_by_call_key = {call_key(row): row for row in cost_rows}
    if len(plan_by_call_key) != len(cost_rows):
        raise RuntimeError("duplicate external judge call-plan key")
    plan_by_physical_key = {
        str(row["physical_call_key"]): row for row in cost_rows
    }
    if len(plan_by_physical_key) != len(cost_rows):
        raise RuntimeError("duplicate external judge physical-call key")
    attempt_ledger = PersistentAttemptLedger(
        ledger_path,
        stage=stage,
        expected_calls={key: 1 for key in plan_by_physical_key},
        maximum_total_attempts=int(max_api_calls),
    )
    ledger_rows = attempt_ledger.event_rows
    successful_call_rows: dict[tuple[str, str, str, str], dict] = {}
    for ledger_row in ledger_rows:
        physical_key = str(ledger_row["call_key"])
        plan = plan_by_physical_key[physical_key]
        key = call_key(plan)
        expected_record_ids = {
            "unit_id": key[0],
            "condition": key[1],
            "judge_family": key[2],
            "judge_type": key[3],
        }
        if (
            ledger_row.get("record_ids") != expected_record_ids
            or ledger_row.get("prompt_sha256") != plan["prompt_hash"]
        ):
            raise RuntimeError(f"stale external attempt-ledger provenance: {key}")
        if ledger_row.get("event") == "SUCCEEDED":
            if key in successful_call_rows:
                raise RuntimeError(f"duplicate successful external judge call: {key}")
            schema = ResponseJudgeOutput if key[3] == "response" else RiskJudgeOutput
            result_payload = ledger_row.get("result") or {}
            parsed = schema.model_validate(result_payload.get("parsed"))
            if not ledger_row.get("request_hash"):
                raise RuntimeError(f"successful external ledger lacks request hash: {key}")
            usage_error = reported_prompt_token_error(
                ledger_row.get("usage"),
                maximum_prompt_tokens=int(plan["input_tokens_est"]),
                stage="persisted external judge",
                require_positive=True,
            )
            if usage_error is not None:
                raise RuntimeError(
                    f"successful external judge usage is invalid for {key}: "
                    f"{usage_error}"
                )
            request_log_payload = result_payload.get("request_log")
            if not isinstance(request_log_payload, Mapping):
                raise RuntimeError(f"successful external ledger lacks request log: {key}")
            successful_call_rows[key] = {
                **plan,
                "parsed": parsed.model_dump(mode="json"),
                "request_hash": str(ledger_row["request_hash"]),
                "request_log": dict(request_log_payload),
            }
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
            plan_by_physical_key[physical_key]
            for physical_key in sorted(attempt_ledger.started_call_keys)
        ],
        *pending_cost_rows,
    ]
    total_input = sum(int(row["input_tokens_est"]) for row in budgeted_cost_rows)
    total_output = sum(int(row["max_output_tokens"]) for row in budgeted_cost_rows)
    input_counts = [int(row["input_tokens_est"]) for row in budgeted_cost_rows]
    estimate_payload = {
        "stage": stage,
        "full_logical_api_calls": len(cost_rows),
        "historical_physical_http_attempts": historical_attempts,
        "planned_new_api_calls": len(pending_cost_rows),
        "maximum_physical_http_attempts": maximum_physical_attempts,
        "expected_judge_pairs": len(plan_by_key),
        "total_input_tokens_est": total_input,
        "max_input_tokens_per_call_est": max(input_counts, default=0),
        "total_output_tokens_est": total_output,
        "estimated_cost_usd": sum(
            float(row["maximum_cost_usd"]) for row in budgeted_cost_rows
        ),
        "pricing_usd_per_mtok": family_pricing,
        "api_cost_planning": api_cost_planning,
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "call_plan_sha256": sha256_text(canonical_json(pending_cost_rows)),
        "ledger_sha256": sha256_text(canonical_json(ledger_rows)),
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
        "study_freeze_sha256": study_freeze_sha256,
        "full_expected_units_sha256": full_expected_units_sha256,
        "excluded_unit_ids": list(excluded_unit_ids),
        "pilot_selection_contract_sha256": pilot_selection_contract_sha256,
        "pointwise_schema_smoke_verification": dict(
            pointwise_schema_smoke_verification
        ),
        "observed_cost_match_report": dict(observed_cost_match_report),
        "bootstrap_cluster_contract": bootstrap_cluster_contract,
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
        "limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
    }
    manifest = ensure_run_manifest(
        manifest_path,
        {
            "stage": "pm_v2_external_response_and_risk_evaluation",
            "evoemo_sha256": sha256_file(evoemo_path),
            "turn_sha256": {str(Path(path).resolve()): sha256_file(path) for path in turn_paths},
            "generation_attestation_sha256": {
                str(Path(path).resolve()): sha256_file(path)
                for path in generation_attestation_paths
            },
            "conditions": list(conditions),
            "required_conditions": list(required_conditions),
            "treatment": treatment,
            "turn_indices": [int(value) for value in turn_indices],
            "expected_units_sha256": sha256_text(canonical_json(units)),
            "full_expected_units_sha256": full_expected_units_sha256,
            "excluded_unit_ids": list(excluded_unit_ids),
            "pilot_selection_contract_sha256": pilot_selection_contract_sha256,
            "observed_cost_match_report": dict(observed_cost_match_report),
            "expected_unit_count": len(units),
            "endpoint_contracts": endpoint_contracts,
            "judge_prompt_contract_hash": prompt_contract_hash(),
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_sha256,
            "labeling": dict(labeling),
            "pricing_usd_per_mtok": family_pricing,
            "api_cost_planning": api_cost_planning,
            "bootstrap_cluster_contract": bootstrap_cluster_contract,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "seed": int(seed),
            "judge_retries": 1,
            "study_freeze_sha256": study_freeze_sha256,
            "key_claim_verification_sha256": (
                sha256_file(key_claim_verification_path)
                if key_claim_verification_path is not None
                else None
            ),
            "pointwise_schema_smoke_summary_sha256": sha256_file(
                pointwise_schema_smoke_summary_path
            ),
            "pointwise_schema_smoke_attestation_sha256": sha256_file(
                pointwise_schema_smoke_attestation_path
            ),
            "pointwise_schema_smoke_verification": dict(
                pointwise_schema_smoke_verification
            ),
            "key_claim_verification_attestation_sha256": (
                sha256_file(key_claim_verification_attestation_path)
                if key_claim_verification_attestation_path is not None
                else None
            ),
        },
    )
    dry = {
        "status": "DRY_RUN_COMPLETE" if not run else "STARTING",
        "n_units": len(units),
        "conditions": list(conditions),
        "treatment": treatment,
        "judge_families": sorted(str(value) for value in families),
        "full_logical_api_calls": len(cost_rows),
        "planned_new_api_calls": len(pending_cost_rows),
        "historical_physical_http_attempts": historical_attempts,
        "expected_judge_pairs": len(plan_by_key),
        "single_candidate_pointwise": True,
        "llm_overall_requested": False,
        "response_and_risk_judging": True,
        "excluded_pilot_unit_count": len(excluded_unit_ids),
        "excluded_unit_ids": list(excluded_unit_ids),
        "pilot_selection_contract_sha256": pilot_selection_contract_sha256,
        "composite_spec": composite_spec.model_dump(mode="json"),
        "composite_weights_sha256": composite_weights_sha256,
        "observed_cost_match_report": dict(observed_cost_match_report),
        "bootstrap_cluster_contract": bootstrap_cluster_contract,
        "run_manifest_sha256": manifest["manifest_sha256"],
        "cost_estimate": estimate,
        "budget_gate": budget_gate,
    }
    if budget_gate["status"] != "PASS":
        write_json(estimate_path, {**estimate, "budget_gate": budget_gate})
        write_jsonl(call_plan_path, pending_cost_rows)
        raise RuntimeError("PM-v2 external evaluation budget gate failed")
    if not run:
        write_json(estimate_path, {**estimate, "budget_gate": budget_gate})
        write_jsonl(call_plan_path, pending_cost_rows)
        return dry
    if not estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("API run requires a matching saved dry-run")
    saved_estimate = read_json(estimate_path)
    if saved_estimate.get("cost_estimate_sha256") != estimate["cost_estimate_sha256"]:
        raise RuntimeError("saved external-evaluation dry-run is stale")
    if list(iter_jsonl(call_plan_path)) != pending_cost_rows:
        raise RuntimeError("saved external-evaluation call plan is stale")
    if accept_cost_estimate_sha256 != estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "API run requires the exact --accept-cost-estimate-sha256 from dry-run"
        )

    expected_raw_keys = set(plan_by_key)
    exhausted_unsuccessful = [
        call_key(plan)
        for plan in cost_rows
        if call_key(plan) not in successful_call_rows
        and attempt_ledger.exhausted(str(plan["physical_call_key"]))
    ]
    if exhausted_unsuccessful:
        raise RuntimeError(
            "external judging contains spent unsuccessful physical calls; the "
            "one-attempt protocol forbids reissuing them: "
            + str(exhausted_unsuccessful[:10])
        )
    endpoint_by_family = {str(endpoint.family): endpoint for endpoint in endpoints}
    clients = {
        family: make_client(endpoint) for family, endpoint in endpoint_by_family.items()
    }
    try:
        for plan in pending_cost_rows:
            key = call_key(plan)
            if key in successful_call_rows:
                continue
            unit_id, condition, family, judge_type = key
            execution = execution_by_call_key[key]
            endpoint = endpoint_by_family[family]
            record_ids = {
                "unit_id": unit_id,
                "condition": condition,
                "judge_family": family,
                "judge_type": judge_type,
            }
            reservation = attempt_ledger.reserve(
                str(plan["physical_call_key"]),
                record_ids=record_ids,
                prompt_sha256=str(plan["prompt_hash"]),
            )
            try:
                result, parsed = clients[family].chat(
                    execution["messages"],
                    temperature=0.0,
                    max_tokens=int(plan["max_output_tokens"]),
                    seed=int(plan["seed"]),
                    response_schema=execution["schema"],
                    retries=1,
                )
                assert parsed is not None
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:2000]}"
                attempt_ledger.finish(
                    reservation,
                    succeeded=False,
                    request_hash=None,
                    usage=None,
                    error=error,
                    metadata={
                        "plan_sha256": sha256_text(canonical_json(plan)),
                        "endpoint_sha256": endpoint_contracts[family]["sha256"],
                        "judge_prompt_contract_hash": prompt_contract_hash(),
                        "study_freeze_sha256": study_freeze_sha256,
                    },
                )
            else:
                parsed_payload = parsed.model_dump(mode="json")
                usage_error = reported_prompt_token_error(
                    result.usage,
                    maximum_prompt_tokens=int(plan["input_tokens_est"]),
                    stage=f"external {judge_type} judge",
                    require_positive=bool(
                        api_cost_planning["fail_on_reported_input_overrun"]
                    ),
                )
                request_log_payload = request_log(
                    stage=f"pm_v2_external_{judge_type}_judge",
                    endpoint=endpoint,
                    messages=execution["messages"],
                    result=result,
                    parsed=parsed,
                    error=usage_error,
                    prompt_hash=plan["prompt_hash"],
                    record_ids={
                        "unit_id": unit_id,
                        "condition": condition,
                        "judge_type": judge_type,
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
                            "parsed": parsed_payload,
                            "request_log": request_log_payload,
                        },
                        metadata={
                            "plan_sha256": sha256_text(canonical_json(plan)),
                            "endpoint_sha256": endpoint_contracts[family]["sha256"],
                            "judge_prompt_contract_hash": prompt_contract_hash(),
                            "study_freeze_sha256": study_freeze_sha256,
                        },
                    )
                    raise RuntimeError(
                        "external judge usage accounting failed after its terminal "
                        f"ledger event was saved for {key}: {usage_error}"
                    )
                attempt_ledger.finish(
                    reservation,
                    succeeded=True,
                    request_hash=result.request_hash,
                    usage=result.usage,
                    error=None,
                    result={
                        "parsed": parsed_payload,
                        "request_log": request_log_payload,
                    },
                    metadata={
                        "plan_sha256": sha256_text(canonical_json(plan)),
                        "endpoint_sha256": endpoint_contracts[family]["sha256"],
                        "judge_prompt_contract_hash": prompt_contract_hash(),
                        "study_freeze_sha256": study_freeze_sha256,
                    },
                )
                successful_call_rows[key] = {
                    **plan,
                    "parsed": parsed_payload,
                    "request_hash": result.request_hash,
                    "request_log": request_log_payload,
                }
            if attempt_ledger.succeeded(str(plan["physical_call_key"])):
                pass
            else:
                raise RuntimeError(
                    "external judge HTTP call failed after its terminal ledger event "
                    "was saved: "
                    + str(key)
                )
    finally:
        for client in clients.values():
            client.close()
    ledger_rows = attempt_ledger.event_rows

    raw_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pair_key in sorted(expected_raw_keys):
        unit_id, condition, family = pair_key
        response_row = successful_call_rows.get(
            (unit_id, condition, family, "response")
        )
        risk_row = successful_call_rows.get((unit_id, condition, family, "risk"))
        if response_row is None or risk_row is None:
            continue
        plan = plan_by_key[pair_key]
        user_id, topic_index, unit_seed, simulator_id, turn_index = plan["unit"]
        raw_by_key[pair_key] = {
            "unit_id": unit_id,
            "user_id": user_id,
            "topic_index": topic_index,
            "seed": unit_seed,
            "simulator_id": simulator_id,
            "turn_index": turn_index,
            "condition": condition,
            "judge_family": family,
            "judge_model": response_row["judge_model"],
            "endpoint_sha256": endpoint_contracts[family]["sha256"],
            "judge_prompt_contract_hash": prompt_contract_hash(),
            "response_prompt_hash": plan["response_prompt_hash"],
            "risk_prompt_hash": plan["risk_prompt_hash"],
            "study_freeze_sha256": study_freeze_sha256,
            "observed_input_tokens": plan["observed_input_tokens"],
            "requested_action_id": plan["requested_action_id"],
            "effective_action_id": plan["effective_action_id"],
            "response_scores": response_row["parsed"],
            "risk_scores": risk_row["parsed"],
            "response_request_log": response_row["request_log"],
            "risk_request_log": risk_row["request_log"],
        }
    write_jsonl(raw_path, [raw_by_key[key] for key in sorted(raw_by_key)])
    if set(raw_by_key) != expected_raw_keys:
        raise RuntimeError("external judge raw matrix is incomplete after API run")
    canonical_raw_rows = [
        {
            "judge_family": row["judge_family"],
            "condition": row["condition"],
            "response": row["response_scores"],
            "risk": row["risk_scores"],
        }
        for row in raw_by_key.values()
    ]
    raw_family_global_gate = validate_raw_judge_family_health(
        canonical_raw_rows,
        expected_families=families,
        composite_spec=composite_spec,
        duplicate_exact_match_rate=float(labeling["duplicate_exact_match_rate"]),
        maximum_absolute_dimension_correlation=float(
            labeling["maximum_absolute_dimension_correlation"]
        ),
        composite_support_exact_match_rate=float(
            labeling["composite_support_exact_match_rate"]
        ),
        maximum_absolute_composite_support_correlation=float(
            labeling["maximum_absolute_composite_support_correlation"]
        ),
        reject_constant_response_dimensions=bool(
            labeling["reject_constant_response_dimensions"]
        ),
        reject_constant_risk_dimensions=bool(
            labeling["reject_constant_risk_dimensions"]
        ),
    )
    raw_family_condition_gate = validate_raw_judge_family_subgroup_health(
        canonical_raw_rows,
        subgroup_key="condition",
        expected_subgroups=conditions,
        expected_families=families,
        composite_spec=composite_spec,
        duplicate_exact_match_rate=float(labeling["duplicate_exact_match_rate"]),
        maximum_absolute_dimension_correlation=float(
            labeling["maximum_absolute_dimension_correlation"]
        ),
        composite_support_exact_match_rate=float(
            labeling["composite_support_exact_match_rate"]
        ),
        maximum_absolute_composite_support_correlation=float(
            labeling["maximum_absolute_composite_support_correlation"]
        ),
        reject_constant_response_dimensions=bool(
            labeling["reject_constant_response_dimensions"]
        ),
        reject_constant_risk_dimensions=bool(
            labeling["reject_constant_risk_dimensions"]
        ),
    )
    raw_family_gate = {
        "status": (
            "PASS"
            if raw_family_global_gate.get("status") == "PASS"
            and raw_family_condition_gate.get("status") == "PASS"
            else "FAIL"
        ),
        "global": raw_family_global_gate,
        "family_by_condition": raw_family_condition_gate,
    }
    grouped_response: dict[tuple[str, str], list[ResponseJudgeOutput]] = defaultdict(list)
    grouped_risk: dict[tuple[str, str], list[RiskJudgeOutput]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    families_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in iter_jsonl(raw_path):
        key = (str(row["unit_id"]), str(row["condition"]))
        grouped_response[key].append(
            ResponseJudgeOutput.model_validate(row["response_scores"])
        )
        grouped_risk[key].append(RiskJudgeOutput.model_validate(row["risk_scores"]))
        metadata[key] = row
        families_by_key[key].add(str(row["judge_family"]))
    score_rows: list[dict[str, Any]] = []
    for key in sorted(grouped_response):
        response_outputs = grouped_response[key]
        risk_outputs = grouped_risk[key]
        output_families = families_by_key[key]
        if len(output_families) < minimum_families:
            raise RuntimeError(f"missing independent judges for {key}")
        dimensions = _median_response(response_outputs)
        risks = _median_risk(risk_outputs)
        response_mad = _mad_response(response_outputs)
        risk_mad = _mad_risk(risk_outputs)
        base = metadata[key]
        requested_action_id = str(base["requested_action_id"])
        applicable_fields = applicable_risk_fields(requested_action_id)
        applicable_risk = max(
            float(getattr(risks, name)) for name in applicable_fields
        ) / 3.0
        all_dimension_risk = max(risks.model_dump().values()) / 3.0
        row = {
            "unit_id": key[0],
            "condition": key[1],
            "user_id": str(base["user_id"]),
            "topic_index": int(base["topic_index"]),
            "scenario_cluster": f"{base['user_id']}::{base['topic_index']}",
            "user_cluster": str(base["user_id"]),
            "seed": int(base["seed"]),
            "turn_index": int(base["turn_index"]),
            **dimensions.model_dump(),
            **risks.model_dump(),
            "quality_composite": composite_spec.score(dimensions),
            "composite_weights_sha256": composite_weights_sha256,
            "requested_action_id": requested_action_id,
            "effective_action_id": str(base["effective_action_id"]),
            "applicable_risk_fields": list(applicable_fields),
            "risk_composite": applicable_risk,
            "risk_composite_all_dimensions_diagnostic": all_dimension_risk,
            "external_estimand": "quality_risk_observed_cost_componentwise_pareto_v1",
            "observed_input_tokens": int(base["observed_input_tokens"]),
            "response_dimension_mad": response_mad,
            "risk_dimension_mad": risk_mad,
            "max_dimension_mad": max(
                [*response_mad.values(), *risk_mad.values()], default=0.0
            ),
            "label_reliable": max(
                [*response_mad.values(), *risk_mad.values()], default=0.0
            )
            <= float(labeling["reliable_mad_threshold"]),
            "judge_families": sorted(output_families),
        }
        score_rows.append(row)
    write_jsonl(score_path, score_rows)
    gate = validate_external_score_table(
        score_rows,
        duplicate_exact_match_rate=float(labeling["duplicate_exact_match_rate"]),
        minimum_reliable_rate=float(labeling["minimum_reliable_rate"]),
        reliable_mad_threshold=float(labeling["reliable_mad_threshold"]),
        minimum_low_mad_coverage_per_dimension=float(
            minimum_low_mad_coverage_per_dimension
        ),
        minimum_low_mad_coverage_per_condition_dimension=float(
            minimum_low_mad_coverage_per_condition_dimension
        ),
        reject_constant_response_dimensions=bool(
            labeling["reject_constant_response_dimensions"]
        ),
        reject_constant_risk_dimensions=bool(
            labeling["reject_constant_risk_dimensions"]
        ),
        maximum_absolute_dimension_correlation=float(
            labeling["maximum_absolute_dimension_correlation"]
        ),
        composite_support_exact_match_rate=float(
            labeling["composite_support_exact_match_rate"]
        ),
        maximum_absolute_composite_support_correlation=float(
            labeling["maximum_absolute_composite_support_correlation"]
        ),
        composite_spec=composite_spec,
    )
    gate["raw_per_family_health"] = raw_family_gate

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_condition[row["condition"]].append(row)
    condition_summary = {
        condition: {
            "n": len(rows),
            **{
                name: float(np.mean([row[name] for row in rows]))
                for name in (
                    *DIMENSIONS,
                    *RISK_DIMENSIONS,
                    "quality_composite",
                    "risk_composite",
                    "risk_composite_all_dimensions_diagnostic",
                    "observed_input_tokens",
                )
            },
        }
        for condition, rows in sorted(by_condition.items())
    }
    treatment_rows = {row["unit_id"]: row for row in by_condition[treatment]}
    comparisons: dict[str, Any] = {}
    for baseline in conditions:
        if baseline == treatment:
            continue
        baseline_rows = {row["unit_id"]: row for row in by_condition[baseline]}
        metric_results: dict[str, Any] = {}
        for metric in (
            *DIMENSIONS,
            *RISK_DIMENSIONS,
            "quality_composite",
            "risk_composite",
            "risk_composite_all_dimensions_diagnostic",
            "observed_input_tokens",
        ):
            paired = []
            for unit_id in sorted(set(treatment_rows) & set(baseline_rows)):
                left = treatment_rows[unit_id]
                right = baseline_rows[unit_id]
                paired.append(
                    {
                        "unit_id": unit_id,
                        "scenario_cluster": left["scenario_cluster"],
                        "user_cluster": left["user_cluster"],
                        "delta": float(left[metric]) - float(right[metric]),
                    }
                )
            metric_results[metric] = {
                "n": len(paired),
                "primary_cluster": "user_id",
                "primary_cluster_ci": _bootstrap_cluster_delta(
                    paired, cluster_field="user_cluster", seed=seed + 1
                ),
                "sensitivity_cluster": "scenario",
                "sensitivity_cluster_ci": _bootstrap_cluster_delta(
                    paired, cluster_field="scenario_cluster", seed=seed
                ),
                "user_cluster_ci": _bootstrap_cluster_delta(
                    paired, cluster_field="user_cluster", seed=seed + 1
                ),
                "scenario_cluster_ci": _bootstrap_cluster_delta(
                    paired, cluster_field="scenario_cluster", seed=seed
                ),
                "wins": sum(row["delta"] > 0 for row in paired),
                "ties": sum(row["delta"] == 0 for row in paired),
                "losses": sum(row["delta"] < 0 for row in paired),
            }
        comparisons[baseline] = metric_results
    key_claim_gate = {
        "required": bool(require_key_claim_verification),
        "status": "PASS",
        "verification_path": (
            str(Path(key_claim_verification_path).resolve())
            if key_claim_verification_path is not None
            else None
        ),
        "attestation_path": (
            str(Path(key_claim_verification_attestation_path).resolve())
            if key_claim_verification_attestation_path is not None
            else None
        ),
        "attestation_sha256": (
            key_claim_attestation_verification.get("attestation_sha256")
            if key_claim_attestation_verification is not None
            else None
        ),
        "excluded_unit_ids": list(excluded_unit_ids),
        "pilot_selection_contract_sha256": pilot_selection_contract_sha256,
    }
    final_status = "COMPLETE"
    summary = {
        **dry,
        "status": final_status,
        "execution_status": "COMPLETE",
        "external_estimand": "quality_risk_observed_cost_componentwise_pareto_v1",
        "risk_composite_basis": "requested_action_applicable_fields",
        "single_external_utility_claim_allowed": False,
        "score_rows": len(score_rows),
        "judge_gate": gate,
        "condition_summary": condition_summary,
        "paired_treatment_deltas": comparisons,
        "bootstrap_cluster_contract": bootstrap_cluster_contract,
        "key_claim_gate": key_claim_gate,
        "scores_path": str(score_path),
        "raw_path": str(raw_path),
        "ledger_path": str(ledger_path),
        "physical_http_attempts": attempt_ledger.started_attempts,
        "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
    }
    write_json(summary_path, summary)
    attestation_inputs = {
        "evoemo": evoemo_path,
        "run_manifest": manifest_path,
        "cost_estimate": estimate_path,
        **{f"turn_{index}": path for index, path in enumerate(turn_paths)},
        **{
            f"generation_attestation_{index}": path
            for index, path in enumerate(generation_attestation_paths)
        },
    }
    if key_claim_verification_path is not None:
        attestation_inputs["key_claim_verification"] = key_claim_verification_path
    if key_claim_verification_attestation_path is not None:
        attestation_inputs["key_claim_verification_attestation"] = (
            key_claim_verification_attestation_path
        )
    attestation_inputs["pointwise_schema_smoke_summary"] = (
        pointwise_schema_smoke_summary_path
    )
    attestation_inputs["pointwise_schema_smoke_attestation"] = (
        pointwise_schema_smoke_attestation_path
    )
    create_artifact_attestation(
        attestation_path,
        stage="pm_v2_external_response_and_risk_evaluation",
        inputs=attestation_inputs,
        outputs={
            "raw_calls": (raw_path, True),
            "call_ledger": (ledger_path, True),
            "scores": (score_path, True),
            "summary": (summary_path, False),
        },
        parameters={
            "conditions": list(conditions),
            "treatment": treatment,
            "expected_unit_count": len(units),
            "judge_families": sorted(str(value) for value in families),
            "composite_spec": composite_spec.model_dump(mode="json"),
            "composite_weights_sha256": composite_weights_sha256,
            "external_estimand": (
                "quality_risk_observed_cost_componentwise_pareto_v1"
            ),
            "risk_composite_basis": "requested_action_applicable_fields",
            "single_external_utility_claim_allowed": False,
            "key_claim_gate": key_claim_gate,
            "judge_gate": gate,
            "full_expected_units_sha256": full_expected_units_sha256,
            "evaluated_units_sha256": actual_units_sha256,
            "observed_cost_match_report": dict(observed_cost_match_report),
            "judge_retries": 1,
            "pricing_usd_per_mtok": family_pricing,
            "api_cost_planning": api_cost_planning,
            "bootstrap_cluster_contract": bootstrap_cluster_contract,
            "physical_http_attempts": attempt_ledger.started_attempts,
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "final_ledger_sha256": sha256_text(canonical_json(ledger_rows)),
            "pointwise_schema_smoke_verification": dict(
                pointwise_schema_smoke_verification
            ),
        },
        expected={"score_rows": len(units) * len(conditions)},
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary
