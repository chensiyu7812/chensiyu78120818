from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import time
from collections import Counter
import math
import joblib

from .api import Endpoint, OpenAICompatibleClient, request_log
from .action_execution import (
    execute_requested_retrievals,
    realized_action_id_from_evidence,
)
from .attempt_ledger import (
    PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
    PersistentAttemptLedger,
    forbid_overwrite_of_spent_attempts,
    physical_call_key,
    reported_prompt_token_error,
)
from .artifacts import (
    create_artifact_attestation,
    require_content_addressed_attestation,
)
from .contracts import (
    CostRecord,
    MemorySource,
    StrategyCard,
    StrategyMode,
    parse_action_id,
)
from .evidence_filter import EvidenceFilterConfig, filter_evidence
from .evidence_filter_model import PMV2EvidenceFilterModel
from .evoemo import (
    FIXED_SEEKER_V22_STAGE,
    FIXED_SEEKER_V23_STAGE,
    NEUTRAL_INITIAL_GREETING,
    _fixed_context_before_turn,
    _load_fixed_tracks,
    _track_key,
    fixed_seeker_cost_planning_contract,
    load_evoemo,
    make_evo_runtime_state,
)
from .io import (
    append_jsonl,
    canonical_json,
    ensure_run_manifest,
    iter_jsonl,
    index_jsonl_unique,
    load_done_keys,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from .generation_contract import SupporterGenerationContract
from .pm_v2_data import (
    compare_external_observable_state_support,
    runtime_to_pmv2_state,
)
from .pm_v2_contracts import PMV2State
from .pm_v1_5_step0 import build_strategy_family_catalog
from .pm_v1_5_semantic import (
    SemanticTextEncoder,
    require_runtime_verification_matches_encoder,
    semantic_centroid,
    summarize_semantic_truncation_audits,
)
from .pm_v1_5_rule_router import (
    DEVELOPMENT_EXTERNAL_SCORE_COMPARISON_PROTOCOL,
    RULE_ROUTER_PROTOCOL,
    TransparentRuleRouter,
    compare_development_external_score_diagnostics,
    transparent_rule_score_diagnostics,
)
from .pm_v2_fixed_model import FixedActionPMV2Model
from .pm_v2_model import PMV2Model, decision_fallback_kind
from .prompts import generation_messages
from .response_mechanism_contract import build_response_mechanism_contract
from .retrieval import (
    MemoryRetriever,
    StrategyRetriever,
    source_specific_memory_queries,
)
from .v1_5_candidate_discovery import (
    choose_after_candidate_discovery,
    discover_memory_candidates_by_source_query,
)
from .v1_5_memory_transport import (
    bounded_memory_global_catalog_digest,
    compile_bounded_memory,
)
from .text import conservative_token_bound, estimate_tokens, normalize_space


ACTION_PREFLIGHT_GATE_KEYS = {
    "maximum_severe_ood_fallback_rate",
    "maximum_no_feasible_fallback_rate",
    "minimum_m0_rate",
    "minimum_r0_rate",
    "minimum_m0_r0_rate",
    "minimum_nonfallback_rate",
    "maximum_action_share",
    "minimum_action_entropy_bits",
}
ACTION_PREFLIGHT_METRIC_SCOPES = {
    "enforced_gate": "exact_frozen_evaluation_turn_states",
    "all_turn_preflight": "diagnostic_only_not_a_paid_entry_gate",
    "fallback_rates": "all_states_within_each_report_scope",
    "learned_action_metrics": "fallback_type=none",
    "learned_action_metric_names": [
        "m0_rate",
        "r0_rate",
        "m0_r0_rate",
        "maximum_action_share",
        "action_entropy_bits",
        "distinct_actions",
    ],
}

COST_MATCH_TREATMENT = "pm_v2"
COST_MATCH_BASELINE = "pm_v2_cost_matched_fixed"
COST_MATCH_CONDITIONS = {COST_MATCH_TREATMENT, COST_MATCH_BASELINE}
EVALUATION_UNIT_CONTRACT_PROTOCOL = (
    "pm-v2-external-fixed-context-evaluation-universe-v1"
)


def bind_external_generation_request_log(
    row: Mapping[str, Any],
    *,
    supporter_generation_treatment: Mapping[str, Any],
    supporter_generation_treatment_sha256: str,
    fixed_seeker_generation_treatment: Mapping[str, Any],
    fixed_seeker_generation_treatment_sha256: str,
) -> dict[str, Any]:
    """Bind every raw external call, including failures, to both treatments."""

    return {
        **dict(row),
        "supporter_generation_treatment": dict(
            supporter_generation_treatment
        ),
        "supporter_generation_treatment_sha256": (
            supporter_generation_treatment_sha256
        ),
        "fixed_seeker_generation_treatment": dict(
            fixed_seeker_generation_treatment
        ),
        "fixed_seeker_generation_treatment_sha256": (
            fixed_seeker_generation_treatment_sha256
        ),
    }


def summarize_external_generation_raw_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    planned_call_keys: Sequence[str],
    supporter_generation_treatment: Mapping[str, Any],
    supporter_generation_treatment_sha256: str,
    fixed_seeker_generation_treatment: Mapping[str, Any],
    fixed_seeker_generation_treatment_sha256: str,
) -> dict[str, Any]:
    """Prove the raw paid-call matrix is exact, complete, and unmixed."""

    normalized_rows = [dict(row) for row in rows]
    expected_keys = [str(value) for value in planned_call_keys]
    observed_keys = [
        str(row.get("physical_call_key") or "") for row in normalized_rows
    ]
    non_complete = sum(
        row.get("normalized_finish_reason") != "complete"
        for row in normalized_rows
    )
    checks = {
        "row_count_exact": len(normalized_rows) == len(expected_keys),
        "planned_call_keys_unique": len(expected_keys) == len(set(expected_keys)),
        "raw_call_keys_unique": (
            "" not in observed_keys
            and len(observed_keys) == len(set(observed_keys))
        ),
        "raw_call_keys_exact": sorted(observed_keys) == sorted(expected_keys),
        "all_finish_reasons_complete": non_complete == 0,
        "all_errors_absent": all(row.get("error") is None for row in normalized_rows),
        "supporter_treatment_exact": all(
            row.get("supporter_generation_treatment")
            == dict(supporter_generation_treatment)
            and row.get("supporter_generation_treatment_sha256")
            == supporter_generation_treatment_sha256
            for row in normalized_rows
        ),
        "fixed_seeker_treatment_exact": all(
            row.get("fixed_seeker_generation_treatment")
            == dict(fixed_seeker_generation_treatment)
            and row.get("fixed_seeker_generation_treatment_sha256")
            == fixed_seeker_generation_treatment_sha256
            for row in normalized_rows
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_rows": len(expected_keys),
        "observed_rows": len(normalized_rows),
        "non_complete_finish_reason_count": int(non_complete),
        "supporter_generation_treatment_sha256": (
            supporter_generation_treatment_sha256
        ),
        "fixed_seeker_generation_treatment_sha256": (
            fixed_seeker_generation_treatment_sha256
        ),
    }


def build_generation_evaluation_units(
    scenarios: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    simulator_id: str,
    evaluation_turn_indices: Sequence[int],
) -> list[tuple[str, int, int, str, int]]:
    """Build the exact condition-independent unit universe paid generation may use."""

    units = {
        (
            str(user["id"]),
            int(topic["idx"]),
            int(seed),
            str(simulator_id),
            int(turn_index),
        )
        for user, topic in scenarios
        for seed in seeds
        for turn_index in evaluation_turn_indices
    }
    if not units:
        raise ValueError("PM-v2 generation evaluation-unit universe cannot be empty")
    return sorted(units)


def require_generation_evaluation_unit_contract(
    contract: Mapping[str, Any],
    *,
    scenarios: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    seeds: Sequence[int],
    simulator_id: str,
    max_turns: int,
) -> tuple[dict[str, Any], list[tuple[str, int, int, str, int]]]:
    """Validate a frozen sparse generation universe against loaded EvoEmo rows."""

    required_keys = {
        "protocol",
        "evaluation_turn_indices",
        "expected_unit_count",
        "expected_units_sha256",
    }
    if set(contract) != required_keys:
        raise ValueError(
            "generation evaluation-unit contract keys do not match the frozen contract"
        )
    if contract.get("protocol") != EVALUATION_UNIT_CONTRACT_PROTOCOL:
        raise ValueError("unknown generation evaluation-unit contract protocol")
    turn_indices = [int(value) for value in contract["evaluation_turn_indices"]]
    if (
        not turn_indices
        or turn_indices != sorted(set(turn_indices))
        or any(value < 1 or value > int(max_turns) for value in turn_indices)
    ):
        raise ValueError(
            "evaluation_turn_indices must be sorted, unique, and within max_turns"
        )
    units = build_generation_evaluation_units(
        scenarios,
        seeds=seeds,
        simulator_id=simulator_id,
        evaluation_turn_indices=turn_indices,
    )
    actual_hash = sha256_text(canonical_json(units))
    if (
        int(contract["expected_unit_count"]) != len(units)
        or str(contract["expected_units_sha256"]) != actual_hash
    ):
        raise RuntimeError(
            "loaded EvoEmo generation universe differs from the frozen exact unit set"
        )
    normalized = {
        "protocol": EVALUATION_UNIT_CONTRACT_PROTOCOL,
        "evaluation_turn_indices": turn_indices,
        "expected_unit_count": len(units),
        "expected_units_sha256": actual_hash,
    }
    return normalized, units


def reconcile_succeeded_generation_turns(
    *,
    ledger: PersistentAttemptLedger,
    expected_turns: Mapping[tuple[Any, ...], Mapping[str, Any]],
    turn_index: dict[tuple[Any, ...], dict[str, Any]],
    turn_path: str | Path,
) -> int:
    """Materialize canonical turn records durably stored in success events.

    The ledger success event is fsynced before the separate turn JSONL append.
    A process death in that narrow interval must not turn a paid, successful
    request into an unrecoverable missing row on resume.
    """

    recovered = 0
    for unit, planned in expected_turns.items():
        if unit in turn_index:
            continue
        call_key = str(planned["call_key"])
        if not ledger.succeeded(call_key):
            continue
        terminal = ledger.terminal_row(call_key)
        usage_error = reported_prompt_token_error(
            terminal.get("usage") if terminal is not None else None,
            maximum_prompt_tokens=int(planned["input_tokens_est"]),
            stage="persisted EvoEmo supporter generation",
            require_positive=True,
        )
        if usage_error is not None:
            raise RuntimeError(
                f"successful PM-v2 generation usage is invalid for {unit}: "
                f"{usage_error}"
            )
        result = terminal.get("result") if terminal is not None else None
        stored_turn = result.get("turn_record") if isinstance(result, Mapping) else None
        if not isinstance(stored_turn, Mapping):
            raise RuntimeError(
                "successful PM-v2 generation ledger entry lacks its canonical "
                f"recoverable turn_record: {unit}"
            )
        turn_record = dict(stored_turn)
        observed_unit = (
            str(turn_record.get("user_id") or ""),
            int(turn_record.get("topic_index") or 0),
            str(turn_record.get("condition") or ""),
            int(turn_record.get("seed") or 0),
            str(turn_record.get("simulator_id") or ""),
            str(turn_record.get("interaction_mode") or ""),
            int(turn_record.get("turn_index") or 0),
        )
        if observed_unit != unit:
            raise RuntimeError(
                "recoverable PM-v2 generation turn unit differs from the frozen "
                f"call plan: expected={unit}, observed={observed_unit}"
            )
        if (
            turn_record.get("physical_call_key") != call_key
            or terminal is None
            or turn_record.get("physical_attempt_index")
            != terminal.get("attempt_index")
            or turn_record.get("physical_attempt_key")
            != terminal.get("attempt_key")
            or not str(turn_record.get("supporter_message") or "").strip()
            or not isinstance(turn_record.get("cost"), Mapping)
            or (
                "supporter_generation_treatment" in planned
                and turn_record.get("supporter_generation_treatment")
                != planned.get("supporter_generation_treatment")
            )
            or (
                "supporter_generation_treatment_sha256" in planned
                and turn_record.get("supporter_generation_treatment_sha256")
                != planned.get("supporter_generation_treatment_sha256")
            )
            or (
                "supporter_generation_treatment" in planned
                and turn_record.get("normalized_finish_reason") != "complete"
            )
            or (
                "fixed_seeker_generation_treatment" in planned
                and turn_record.get("fixed_seeker_generation_treatment")
                != planned.get("fixed_seeker_generation_treatment")
            )
            or (
                "fixed_seeker_generation_treatment_sha256" in planned
                and turn_record.get("fixed_seeker_generation_treatment_sha256")
                != planned.get("fixed_seeker_generation_treatment_sha256")
            )
        ):
            raise RuntimeError(
                "recoverable PM-v2 generation turn violates its successful "
                f"physical-attempt binding: {unit}"
            )
        append_jsonl(turn_path, turn_record)
        turn_index[unit] = turn_record
        recovered += 1
    return recovered


def _token_rows_by_unit(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    token_field: str,
) -> dict[tuple[str, int, int, str, int], float]:
    by_unit: dict[tuple[str, int, int, str, int], float] = {}
    for row in rows:
        if str(row.get("condition")) != condition:
            raise RuntimeError(
                f"cost-match rows contain condition {row.get('condition')!r}; "
                f"expected {condition!r}"
            )
        unit = (
            str(row["user_id"]),
            int(row["topic_index"]),
            int(row["seed"]),
            str(row["simulator_id"]),
            int(row["turn_index"]),
        )
        if unit in by_unit:
            raise RuntimeError(f"duplicate cost-match unit: {unit}")
        value = float(row[token_field])
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"invalid {token_field} for cost-match unit: {unit}")
        by_unit[unit] = value
    if not by_unit:
        raise RuntimeError(f"empty cost-match rows for {condition}")
    return by_unit


def compare_cost_matched_token_rows(
    treatment_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    token_field: str,
    maximum_relative_deviation: float,
    stage: str,
) -> dict[str, Any]:
    """Require the learned and fixed policies to cover one exact token matrix."""

    tolerance = float(maximum_relative_deviation)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("maximum cost-matched relative deviation must be in [0, 1)")
    treatment = _token_rows_by_unit(
        treatment_rows,
        condition=COST_MATCH_TREATMENT,
        token_field=token_field,
    )
    baseline = _token_rows_by_unit(
        baseline_rows,
        condition=COST_MATCH_BASELINE,
        token_field=token_field,
    )
    if set(treatment) != set(baseline):
        missing_treatment = sorted(set(baseline) - set(treatment))[:20]
        missing_baseline = sorted(set(treatment) - set(baseline))[:20]
        raise RuntimeError(
            "cost-match token matrices differ: "
            f"missing_treatment={missing_treatment}, "
            f"missing_baseline={missing_baseline}"
        )
    treatment_mean = sum(treatment.values()) / len(treatment)
    baseline_mean = sum(baseline.values()) / len(baseline)
    ratio = baseline_mean / treatment_mean
    deviation = abs(ratio - 1.0)
    paired_deviations = [
        abs(baseline[unit] / treatment[unit] - 1.0) for unit in sorted(treatment)
    ]
    check = deviation <= tolerance
    return {
        "status": "PASS" if check else "FAIL",
        "stage": stage,
        "token_field": token_field,
        "treatment": COST_MATCH_TREATMENT,
        "baseline": COST_MATCH_BASELINE,
        "n_units": len(treatment),
        "unit_matrix_sha256": sha256_text(canonical_json(sorted(treatment))),
        "treatment_rows_sha256": sha256_text(canonical_json(list(treatment_rows))),
        "baseline_rows_sha256": sha256_text(canonical_json(list(baseline_rows))),
        "treatment_mean_tokens": treatment_mean,
        "baseline_mean_tokens": baseline_mean,
        "baseline_to_treatment_mean_ratio": ratio,
        "relative_deviation": deviation,
        "maximum_relative_deviation": tolerance,
        "maximum_paired_relative_deviation": max(paired_deviations),
        "check": check,
    }


def compare_observed_cost_matched_turns(
    treatment_turn_path: str | Path,
    baseline_turn_path: str | Path,
    *,
    maximum_relative_deviation: float,
    expected_units: Sequence[tuple[str, int, int, str, int]] | None = None,
) -> dict[str, Any]:
    """Gate response judging on observed, not merely estimated, prompt tokens."""

    treatment_rows = list(iter_jsonl(treatment_turn_path))
    baseline_rows = list(iter_jsonl(baseline_turn_path))
    if expected_units is not None:
        expected_set = set(expected_units)

        def include(row: Mapping[str, Any]) -> bool:
            return (
                str(row["user_id"]),
                int(row["topic_index"]),
                int(row["seed"]),
                str(row["simulator_id"]),
                int(row["turn_index"]),
            ) in expected_set

        treatment_rows = [row for row in treatment_rows if include(row)]
        baseline_rows = [row for row in baseline_rows if include(row)]
    report = compare_cost_matched_token_rows(
        treatment_rows,
        baseline_rows,
        token_field="input_tokens",
        maximum_relative_deviation=maximum_relative_deviation,
        stage="observed_generation_tokens",
    )
    report["treatment_turns_sha256"] = sha256_file(treatment_turn_path)
    report["baseline_turns_sha256"] = sha256_file(baseline_turn_path)
    report["expected_units_sha256"] = (
        sha256_text(canonical_json(sorted(expected_units)))
        if expected_units is not None
        else None
    )
    if expected_units is not None and report["n_units"] != len(set(expected_units)):
        raise RuntimeError("observed cost-match rows do not exactly cover expected units")
    return report


def summarize_action_preflight(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: str,
    gates: Mapping[str, float],
) -> dict[str, Any]:
    """Summarize and independently gate the two fallback mechanisms."""

    if not rows:
        raise ValueError("PM-v2 action preflight cannot be empty")
    if set(gates) != ACTION_PREFLIGHT_GATE_KEYS:
        raise ValueError("action preflight gate keys do not match the frozen contract")
    fallback_types = [row.get("fallback_type") for row in rows]
    unknown_fallback_types = sorted(
        {str(value) for value in fallback_types if value not in {None, "severe_ood", "no_feasible"}}
    )
    if unknown_fallback_types:
        raise ValueError(f"unknown PM-v2 fallback types: {unknown_fallback_types}")
    n_preflight = len(rows)
    severe_ood_fallback_rate = fallback_types.count("severe_ood") / n_preflight
    no_feasible_fallback_rate = fallback_types.count("no_feasible") / n_preflight
    fallback_rate = severe_ood_fallback_rate + no_feasible_fallback_rate
    learned_rows = [row for row in rows if row.get("fallback_type") is None]
    n_learned = len(learned_rows)
    overall_action_counts = Counter(str(row["chosen_action"]) for row in rows)
    action_counts = Counter(str(row["chosen_action"]) for row in learned_rows)
    action_probabilities = [
        count / n_learned for count in action_counts.values()
    ] if n_learned else []
    action_entropy = -sum(
        probability * math.log2(probability)
        for probability in action_probabilities
        if probability > 0
    )
    m0_rate = sum(
        not bool(parse_action_id(str(row["chosen_action"]))[0])
        for row in learned_rows
    ) / n_learned if n_learned else 0.0
    r0_rate = sum(
        parse_action_id(str(row["chosen_action"]))[1] is StrategyMode.R0
        for row in learned_rows
    ) / n_learned if n_learned else 0.0
    m0_r0_rate = (
        sum(str(row["chosen_action"]) == "M0+R0" for row in learned_rows)
        / n_learned
        if n_learned
        else 0.0
    )
    max_action_share = (
        max(action_counts.values()) / n_learned if n_learned else 0.0
    )
    nonfallback_rate = 1.0 - fallback_rate
    learned_policy_condition = condition == "pm_v2"
    checks = {
        "severe_ood_fallback_rate": severe_ood_fallback_rate
        <= float(gates["maximum_severe_ood_fallback_rate"]),
        "no_feasible_fallback_rate": no_feasible_fallback_rate
        <= float(gates["maximum_no_feasible_fallback_rate"]),
        "m0_rate": (not learned_policy_condition)
        or m0_rate >= float(gates["minimum_m0_rate"]),
        "r0_rate": (not learned_policy_condition)
        or r0_rate >= float(gates["minimum_r0_rate"]),
        "m0_r0_rate": (not learned_policy_condition)
        or m0_r0_rate >= float(gates["minimum_m0_r0_rate"]),
        "nonfallback_rate": (not learned_policy_condition)
        or nonfallback_rate >= float(gates["minimum_nonfallback_rate"]),
        "maximum_action_share": (not learned_policy_condition)
        or max_action_share <= float(gates["maximum_action_share"]),
        "action_entropy_bits": (not learned_policy_condition)
        or action_entropy >= float(gates["minimum_action_entropy_bits"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "n_states": n_preflight,
        "n_learned_policy_states": n_learned,
        "n_fallback_states": n_preflight - n_learned,
        "severe_ood_fallback_rate": severe_ood_fallback_rate,
        "no_feasible_fallback_rate": no_feasible_fallback_rate,
        "fallback_rate": fallback_rate,
        "m0_rate": m0_rate,
        "r0_rate": r0_rate,
        "m0_r0_rate": m0_r0_rate,
        "nonfallback_rate": nonfallback_rate,
        "maximum_action_share": max_action_share,
        "action_entropy_bits": action_entropy,
        "distinct_actions": len(action_counts),
        "fallback_type_distribution": dict(
            Counter("none" if value is None else str(value) for value in fallback_types)
        ),
        "action_distribution": dict(action_counts),
        "learned_action_distribution": dict(action_counts),
        "overall_action_distribution": dict(overall_action_counts),
        "metric_scopes": ACTION_PREFLIGHT_METRIC_SCOPES,
        "checks": checks,
        "limits": {str(key): float(value) for key, value in gates.items()},
        "diversity_gates_apply": learned_policy_condition,
        "examples": [dict(row) for row in rows[:20]],
    }


def run_pmv2_fixed_evoemo(
    evoemo_path: str | Path,
    strategy_bank_path: str | Path,
    checkpoint_path: str | Path,
    fixed_tracks_path: str | Path,
    out_dir: str | Path,
    *,
    project_root: str | Path,
    generator_endpoint: Endpoint,
    supporter_generation_contract: SupporterGenerationContract,
    fixed_seeker_generation_contract: Mapping[str, Any],
    fixed_seeker_generation_contract_sha256: str,
    simulator_id: str,
    fixed_tracks_attestation_path: str | Path | None = None,
    # Shared by both the original PM-v2.2 track (scripts/24_run_pm_v2_evoemo.py,
    # still V2, unchanged) and the V1.5 track (scripts/v1_5/
    # 24_run_pm_v2_evoemo_v1_5.py, migrated to V3): the caller states which
    # fixed-seeker attestation stage its own frozen bundle must carry, rather
    # than this shared runner silently assuming one track's version for both.
    fixed_seeker_required_stage: str = FIXED_SEEKER_V22_STAGE,
    condition: str = "pm_v2",
    max_turns: int = 10,
    seeds: Sequence[int] = (101,),
    evaluation_unit_contract: Mapping[str, Any],
    max_scenarios: int | None = None,
    overwrite: bool = False,
    strategy_action_tokens: int = 260,
    study_freeze_sha256: str | None = None,
    run: bool = False,
    accept_cost_estimate_sha256: str | None = None,
    max_api_calls: int = 2000,
    max_estimated_usd: float = 5.0,
    max_input_tokens_per_call: int = 12000,
    input_usd_per_mtok: float,
    output_usd_per_mtok: float,
    input_token_safety_factor: float = 1.0,
    fail_on_reported_input_overrun: bool = False,
    strategy_top_k: int = 3,
    memory_min_score: float | None = None,
    strategy_min_score: float | None = None,
    require_complete_strategy_family_catalog: bool = False,
    evidence_filter_config: EvidenceFilterConfig | None = None,
    memory_helpfulness_model: PMV2EvidenceFilterModel | None = None,
    evidence_filter_model_binding: Mapping[str, Any] | None = None,
    action_preflight_gates: Mapping[str, float],
    maximum_cost_matched_relative_deviation: float,
    cost_match_reference_call_plan_path: str | Path | None = None,
    cost_match_reference_cost_estimate_path: str | Path | None = None,
    semantic_encoder: SemanticTextEncoder | None = None,
    semantic_runtime_verification: Mapping[str, Any] | None = None,
    development_training_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate only the frozen PM-v2 condition on policy-independent tracks."""

    if run and overwrite:
        raise RuntimeError(
            "paid API runs prohibit overwrite; use a new output directory"
        )
    if fixed_seeker_required_stage not in {
        FIXED_SEEKER_V22_STAGE,
        FIXED_SEEKER_V23_STAGE,
    }:
        raise ValueError(
            f"unsupported fixed-seeker required stage: {fixed_seeker_required_stage!r}"
        )
    generator_pricing_usd_per_mtok = {
        "input": float(input_usd_per_mtok),
        "output": float(output_usd_per_mtok),
    }
    if any(value <= 0.0 for value in generator_pricing_usd_per_mtok.values()):
        raise ValueError("PM-v2 generator pricing must be strictly positive")
    if semantic_encoder is not None:
        semantic_runtime_lineage = require_runtime_verification_matches_encoder(
            semantic_runtime_verification or {}, semantic_encoder
        )
        semantic_runtime_lineage.update(
            {
                "semantic_encoder_spec_sha256": semantic_encoder.binding.spec_sha256,
                "semantic_encoder_snapshot_tree_sha256": (
                    semantic_encoder.binding.snapshot_tree_sha256
                ),
            }
        )
        if development_training_report_path is None:
            raise RuntimeError(
                "semantic external condition requires the frozen training report"
            )
        development_training_report_path = Path(
            development_training_report_path
        ).resolve()
        development_training_report = read_json(development_training_report_path)
        development_score_diagnostics = development_training_report.get(
            "step0_score_diagnostics_by_split"
        ) or {}
        development_observable_support = development_training_report.get(
            "development_observable_state_support"
        ) or {}
        development_training_report_sha256 = sha256_file(
            development_training_report_path
        )
    else:
        if semantic_runtime_verification is not None:
            raise RuntimeError(
                "fixed-action condition cannot claim an unused semantic runtime"
            )
        semantic_runtime_lineage = {
            "status": "NOT_APPLICABLE_FIXED_ACTION_CONDITION",
            "contract": None,
            "contract_sha256": None,
            "semantic_encoder_spec_sha256": None,
            "semantic_encoder_snapshot_tree_sha256": None,
        }
        if development_training_report_path is not None:
            raise RuntimeError(
                "fixed-action condition cannot claim an unused training-score reference"
            )
        development_score_diagnostics = None
        development_observable_support = None
        development_training_report_sha256 = None
    derived_evidence_filter_model_binding = (
        {
            "checkpoint_sha256": memory_helpfulness_model.checkpoint_sha256,
            "model_contract_sha256": memory_helpfulness_model.contract_hash(),
        }
        if memory_helpfulness_model is not None
        else None
    )
    if (
        evidence_filter_model_binding is not None
        and derived_evidence_filter_model_binding is not None
        and dict(evidence_filter_model_binding)
        != derived_evidence_filter_model_binding
    ):
        raise RuntimeError("explicit Evidence Filter model binding is stale")
    effective_evidence_filter_model_binding = (
        dict(evidence_filter_model_binding)
        if evidence_filter_model_binding is not None
        else derived_evidence_filter_model_binding
    )

    supporter_treatment = supporter_generation_contract.payload()
    supporter_treatment_sha256 = supporter_generation_contract.digest()
    fixed_seeker_treatment = dict(fixed_seeker_generation_contract)
    fixed_seeker_treatment_sha256 = str(
        fixed_seeker_generation_contract_sha256
    )
    if (
        not fixed_seeker_treatment
        or len(fixed_seeker_treatment_sha256) != 64
        or sha256_text(canonical_json(fixed_seeker_treatment))
        != fixed_seeker_treatment_sha256
    ):
        raise ValueError("fixed-seeker generation contract binding is invalid")
    if set(action_preflight_gates) != ACTION_PREFLIGHT_GATE_KEYS:
        raise ValueError("action_preflight_gates do not match the frozen contract")
    cost_match_tolerance = float(maximum_cost_matched_relative_deviation)
    if not 0.0 <= cost_match_tolerance < 1.0:
        raise ValueError("maximum cost-matched relative deviation must be in [0, 1)")
    cost_match_condition = condition in COST_MATCH_CONDITIONS
    expected_reference_condition = (
        COST_MATCH_BASELINE if condition == COST_MATCH_TREATMENT else COST_MATCH_TREATMENT
    ) if cost_match_condition else None
    if cost_match_condition and (
        cost_match_reference_call_plan_path is None
        or cost_match_reference_cost_estimate_path is None
    ):
        raise ValueError("cost-matched conditions require frozen counterpart dry-run paths")
    fixed_tracks_attestation_path = Path(
        fixed_tracks_attestation_path
        or Path(fixed_tracks_path).parent / "artifact_attestation.json"
    )
    fixed_bundle_dir = Path(fixed_tracks_path).resolve().parent
    fixed_verification = require_content_addressed_attestation(
        fixed_tracks_attestation_path,
        required_stage=fixed_seeker_required_stage,
        relocated_inputs={
            "evoemo": evoemo_path,
            "run_manifest": fixed_bundle_dir / "run_manifest.json",
            "cost_estimate": fixed_bundle_dir / "cost_estimate.json",
            "call_plan": fixed_bundle_dir / "call_plan.jsonl",
        },
        relocated_outputs={
            "tracks": fixed_tracks_path,
            "raw_calls": fixed_bundle_dir / "raw_seeker_calls.jsonl",
            "physical_attempt_ledger": fixed_bundle_dir
            / "physical_attempt_ledger.jsonl",
            "summary": fixed_bundle_dir / "summary.json",
        },
    )
    fixed_attestation = read_json(fixed_tracks_attestation_path)
    fixed_parameters = fixed_attestation.get("parameters") or {}
    fixed_cost_planning = fixed_seeker_cost_planning_contract(
        fixed_parameters.get("fixed_seeker_cost_planning") or {}
    )
    fixed_cost_planning_sha256 = sha256_text(
        canonical_json(fixed_cost_planning)
    )
    if fixed_parameters.get(
        "fixed_seeker_cost_planning_sha256"
    ) != fixed_cost_planning_sha256:
        raise RuntimeError("fixed-track cost-planning contract hash mismatch")
    expected_fixed_parameters = {
        "simulator_id": simulator_id,
        "max_turns": int(max_turns),
        "seeds": [int(seed) for seed in seeds],
        "fixed_seeker_generation_contract": fixed_seeker_treatment,
        "fixed_seeker_generation_contract_sha256": (
            fixed_seeker_treatment_sha256
        ),
    }
    for key, expected in expected_fixed_parameters.items():
        if fixed_parameters.get(key) != expected:
            raise RuntimeError(f"fixed-track attestation parameter mismatch: {key}")
    users = load_evoemo(evoemo_path)
    strategy_cards = [
        StrategyCard.model_validate(row) for row in iter_jsonl(strategy_bank_path)
    ]
    if not strategy_cards:
        raise ValueError("strategy bank is empty")
    strategy_catalog_refresh_started = time.perf_counter()
    strategy_family_catalog = build_strategy_family_catalog(
        strategy_cards,
        require_all_families=require_complete_strategy_family_catalog,
        semantic_encoder=semantic_encoder,
    )
    strategy_catalog_refresh_ms = (
        (time.perf_counter() - strategy_catalog_refresh_started) * 1000.0
        if semantic_encoder is not None
        else 0.0
    )
    strategy_retriever = StrategyRetriever(
        strategy_cards, top_k=strategy_top_k, minimum_score=strategy_min_score
    )
    memory_retriever = MemoryRetriever(
        minimum_score_by_source=(
            {source: float(memory_min_score) for source in MemorySource}
            if memory_min_score is not None
            else None
        )
    )
    try:
        model = PMV2Model.load(checkpoint_path)
    except TypeError:
        candidate = joblib.load(checkpoint_path)
        if not isinstance(candidate, TransparentRuleRouter):
            raise
        if candidate.format_version != RULE_ROUTER_PROTOCOL:
            raise RuntimeError("unsupported transparent-rule checkpoint format")
        model = candidate
    requires_step0_observation = not isinstance(model, FixedActionPMV2Model)
    if condition in {"pm_v2", "pm_v1_5_transparent_rule_step0"}:
        if semantic_encoder is None:
            raise RuntimeError(
                "reportable learned/rule external generation requires the frozen "
                "semantic encoder"
            )
        fitted_binding = getattr(
            getattr(model, "feature_builder", None),
            "semantic_encoder_spec_sha256",
            None,
        )
        if isinstance(model, PMV2Model) and (
            fitted_binding != semantic_encoder.binding.spec_sha256
        ):
            raise RuntimeError(
                "PM checkpoint semantic encoder binding differs from runtime"
            )
    tracks = _load_fixed_tracks(fixed_tracks_path)
    fixed_expected = fixed_attestation.get("expected") or {}
    if (
        int(fixed_expected.get("tracks", -1)) != len(tracks)
        or int(fixed_expected.get("turns_per_track", -1)) != int(max_turns)
        or int(fixed_expected.get("completion_truncated_count", -1)) != 0
    ):
        raise RuntimeError("fixed-track attestation expected matrix mismatch")
    fixed_summary = read_json(fixed_bundle_dir / "summary.json")
    fixed_raw_rows = list(iter_jsonl(fixed_bundle_dir / "raw_seeker_calls.jsonl"))
    if (
        fixed_summary.get("status") != "COMPLETE"
        or fixed_summary.get("fixed_seeker_generation_contract")
        != fixed_seeker_treatment
        or fixed_summary.get("fixed_seeker_generation_contract_sha256")
        != fixed_seeker_treatment_sha256
        or fixed_summary.get("fixed_seeker_cost_planning")
        != fixed_cost_planning
        or fixed_summary.get("fixed_seeker_cost_planning_sha256")
        != fixed_cost_planning_sha256
        or (fixed_summary.get("planned_budget_gate") or {}).get("status")
        != "PASS"
        or (fixed_summary.get("observed_budget_gate") or {}).get("status")
        != "PASS"
        or fixed_parameters.get("planned_budget_gate")
        != fixed_summary.get("planned_budget_gate")
        or fixed_parameters.get("observed_budget_gate")
        != fixed_summary.get("observed_budget_gate")
        or (fixed_attestation.get("expected") or {}).get(
            "planned_budget_gate"
        )
        != fixed_summary.get("planned_budget_gate")
        or (fixed_attestation.get("expected") or {}).get(
            "observed_budget_gate"
        )
        != fixed_summary.get("observed_budget_gate")
        or int(fixed_summary.get("completion_truncated_count", -1)) != 0
        or any(
            track.get("fixed_seeker_generation_contract")
            != fixed_seeker_treatment
            or track.get("fixed_seeker_generation_contract_sha256")
            != fixed_seeker_treatment_sha256
            or track.get("fixed_seeker_cost_planning")
            != fixed_cost_planning
            or track.get("fixed_seeker_cost_planning_sha256")
            != fixed_cost_planning_sha256
            or len(track.get("turn_provenance") or []) != int(max_turns)
            or any(
                turn.get("normalized_finish_reason") != "complete"
                or turn.get("fixed_seeker_generation_contract_sha256")
                != fixed_seeker_treatment_sha256
                or turn.get("fixed_seeker_cost_planning_sha256")
                != fixed_cost_planning_sha256
                for turn in (track.get("turn_provenance") or [])
            )
            for track in tracks.values()
        )
        or any(
            row.get("normalized_finish_reason") != "complete"
            or row.get("fixed_seeker_generation_contract_sha256")
            != fixed_seeker_treatment_sha256
            or row.get("fixed_seeker_cost_planning_sha256")
            != fixed_cost_planning_sha256
            for row in fixed_raw_rows
        )
    ):
        raise RuntimeError(
            "fixed seeker V2.2 input is mixed, truncated, or contract-stale"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dialogue_path = out_dir / "dialogues.jsonl"
    turn_path = out_dir / "turns.jsonl"
    raw_path = out_dir / "raw_api_calls.jsonl"
    attempt_ledger_path = out_dir / "physical_attempt_ledger.jsonl"
    summary_path = out_dir / "generation_summary.json"
    preflight_path = out_dir / "pm_v2_preflight.json"
    manifest_path = out_dir / "run_manifest.json"
    attestation_path = out_dir / "artifact_attestation.json"
    comparison_path = out_dir / "semantic_distribution_comparison.json"
    cost_estimate_path = out_dir / "cost_estimate.json"
    call_plan_path = out_dir / "call_plan.jsonl"
    forbid_overwrite_of_spent_attempts(
        attempt_ledger_path,
        overwrite=overwrite,
        stage="PM-v2 EvoEmo generation",
    )
    if overwrite:
        paths = (
            dialogue_path,
            turn_path,
            raw_path,
            summary_path,
            preflight_path,
            manifest_path,
            attestation_path,
            comparison_path,
        )
        if not run:
            paths = (*paths, cost_estimate_path, call_plan_path)
        for path in paths:
            if path.exists():
                path.unlink()

    scenarios: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for user in users:
        for topic in user.get("subsequent_topics") or []:
            scenarios.append((user, topic))
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
    normalized_evaluation_unit_contract, frozen_evaluation_units = (
        require_generation_evaluation_unit_contract(
            evaluation_unit_contract,
            scenarios=scenarios,
            seeds=seeds,
            simulator_id=simulator_id,
            max_turns=max_turns,
        )
    )
    evaluation_turn_indices = normalized_evaluation_unit_contract[
        "evaluation_turn_indices"
    ]

    # evoemo_sha256 above only pins the raw input file, not what
    # build_evo_memory actually constructs from it (MP/MS/ME item content,
    # chunking, ids) -- record that separately so this run's manifest is
    # auditable against the memory builder that actually produced its
    # retrieval catalog, not just the source data.
    evo_memory_digest = bounded_memory_global_catalog_digest(users)

    ensure_run_manifest(
        manifest_path,
        {
            "stage": "evoemo_pm_v2_generation",
            "evoemo_sha256": sha256_file(evoemo_path),
            "evo_memory_builder_contract_sha256": evo_memory_digest[
                "builder_contract_sha256"
            ],
            "evo_memory_global_catalog_sha256": evo_memory_digest[
                "global_catalog_sha256"
            ],
            "strategy_bank_sha256": sha256_file(strategy_bank_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "fixed_tracks_sha256": sha256_file(fixed_tracks_path),
            "fixed_tracks_attestation_sha256": fixed_verification["attestation_sha256"],
            "generator_model": generator_endpoint.model,
            "generator_family": generator_endpoint.family,
            "generator_base_url": generator_endpoint.base_url,
            "supporter_generation_treatment": supporter_treatment,
            "supporter_generation_treatment_sha256": supporter_treatment_sha256,
            "fixed_seeker_generation_treatment": fixed_seeker_treatment,
            "fixed_seeker_generation_treatment_sha256": (
                fixed_seeker_treatment_sha256
            ),
            "simulator_id": simulator_id,
            "protocol": supporter_generation_contract.version,
            "condition": condition,
            "max_turns": int(max_turns),
            "seeds": [int(seed) for seed in seeds],
            "evaluation_unit_contract": normalized_evaluation_unit_contract,
            "paid_generation_scope": "frozen_evaluation_turns_only",
            "all_turn_action_preflight_scope": list(range(1, int(max_turns) + 1)),
            "max_scenarios": max_scenarios,
            "selection_config_hash": model.selection_config.digest(),
            "model_format_version": model.format_version,
            "strategy_action_tokens": int(strategy_action_tokens),
            "strategy_top_k": int(strategy_top_k),
            "memory_min_score": memory_min_score,
            "strategy_min_score": strategy_min_score,
            "evidence_filter": (
                evidence_filter_config.payload()
                if evidence_filter_config is not None
                else None
            ),
            "evidence_filter_config_sha256": (
                evidence_filter_config.digest()
                if evidence_filter_config is not None
                else None
            ),
            "evidence_filter_model": effective_evidence_filter_model_binding,
            "action_preflight_gates": dict(action_preflight_gates),
            "action_preflight_metric_scopes": ACTION_PREFLIGHT_METRIC_SCOPES,
            "maximum_cost_matched_relative_deviation": cost_match_tolerance,
            "cost_match_reference_condition": expected_reference_condition,
            "cost_match_reference_call_plan_path": (
                str(Path(cost_match_reference_call_plan_path).resolve())
                if cost_match_reference_call_plan_path is not None
                else None
            ),
            "cost_match_reference_cost_estimate_path": (
                str(Path(cost_match_reference_cost_estimate_path).resolve())
                if cost_match_reference_cost_estimate_path is not None
                else None
            ),
            "study_freeze_sha256": study_freeze_sha256,
            "generator_retries": 1,
            "generator_pricing_usd_per_mtok": generator_pricing_usd_per_mtok,
            "input_token_safety_factor": float(input_token_safety_factor),
            "fail_on_reported_input_overrun": bool(
                fail_on_reported_input_overrun
            ),
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "runtime_maximum_physical_http_attempts": int(max_api_calls),
            "semantic_runtime_verification": semantic_runtime_lineage,
            "development_training_report_sha256": (
                development_training_report_sha256
            ),
            "development_external_score_comparison_protocol": (
                DEVELOPMENT_EXTERNAL_SCORE_COMPARISON_PROTOCOL
                if semantic_encoder is not None
                else None
            ),
        },
    )

    preflight_rows: list[dict[str, Any]] = []
    preflight_states: list[PMV2State] = []
    evaluation_preflight_states: list[PMV2State] = []
    call_plan: list[dict[str, Any]] = []
    semantic_centroids_by_user: dict[
        str, dict[MemorySource, tuple[float, ...]]
    ] = {}
    semantic_catalog_refresh_ms_by_user: dict[str, float] = {}

    def source_centroids_for_user(
        user_id: str, items
    ) -> dict[MemorySource, tuple[float, ...]] | None:
        if not requires_step0_observation or semantic_encoder is None:
            return None
        if user_id not in semantic_centroids_by_user:
            started = time.perf_counter()
            semantic_centroids_by_user[user_id] = {
                source: semantic_centroid(
                    semantic_encoder,
                    [item.text for item in items if item.source is source],
                )
                for source in MemorySource
            }
            semantic_catalog_refresh_ms_by_user[user_id] = (
                time.perf_counter() - started
            ) * 1000.0
        return semantic_centroids_by_user[user_id]

    for user, topic in scenarios:
        items, _ = compile_bounded_memory(user)
        user_id = str(user["id"])
        source_centroids = source_centroids_for_user(user_id, items)
        for seed in seeds:
            key = _track_key(str(user["id"]), int(topic["idx"]), int(seed), simulator_id)
            track = tracks.get(key)
            if track is None:
                raise RuntimeError(f"missing fixed seeker track: {key}")
            if len(track.get("seeker_turns") or []) != max_turns:
                raise RuntimeError(f"fixed track turn count mismatch: {key}")
            for turn_index in range(1, max_turns + 1):
                seeker_message = normalize_space(track["seeker_turns"][turn_index - 1])
                state_context = _fixed_context_before_turn(track, turn_index)
                runtime = make_evo_runtime_state(
                    user,
                    topic,
                    state_context,
                    seeker_message,
                    items,
                    turn_index,
                    condition,
                    track_id=str(track["track_id"]),
                    fixed_open_loop=True,
                    semantic_encoder=(
                        semantic_encoder if requires_step0_observation else None
                    ),
                    semantic_source_centroids=source_centroids,
                )
                pm_state = runtime_to_pmv2_state(
                    runtime,
                    strategy_catalog_count=len(strategy_cards),
                    strategy_estimated_tokens=strategy_action_tokens,
                    strategy_family_catalog=strategy_family_catalog,
                    include_step0_observation=requires_step0_observation,
                    semantic_encoder=(
                        semantic_encoder if requires_step0_observation else None
                    ),
                )
                queries = source_specific_memory_queries(
                    runtime.current_user_text,
                    [
                        row.model_dump(mode="json")
                        for row in runtime.current_session_history
                    ],
                    runtime.current_session_summary,
                )
                query = queries[MemorySource.ME]
                discoveries = discover_memory_candidates_by_source_query(
                    queries=queries,
                    items=items,
                    retriever=memory_retriever,
                    session_index=runtime.session_index,
                )
                pm_state, decision = choose_after_candidate_discovery(
                    model=model,
                    pm_state=pm_state,
                    discoveries=discoveries,
                )
                if semantic_encoder is not None and isinstance(pm_state, PMV2State):
                    preflight_states.append(pm_state)
                    if turn_index in evaluation_turn_indices:
                        evaluation_preflight_states.append(pm_state)
                fallback_type = (
                    decision_fallback_kind(decision) if condition == "pm_v2" else None
                )
                if condition != "pm_v2" and decision.ood_fallback_used:
                    raise RuntimeError("fixed-action baseline cannot use an OOD fallback")
                preflight_rows.append(
                    {
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "seed": int(seed),
                        "turn_index": turn_index,
                        "chosen_action": decision.chosen_action,
                        "semantic_ood_score": decision.semantic_ood_score,
                        "metadata_ood_score": decision.metadata_ood_score,
                        "ood_fallback_used": decision.ood_fallback_used,
                            "fallback_type": fallback_type,
                            "semantic_observation": (
                                (getattr(pm_state, "provenance", {}) or {}).get(
                                    "semantic_observation"
                                )
                                or {}
                        ),
                    }
                )
                if turn_index not in evaluation_turn_indices:
                    continue
                sources, strategy = parse_action_id(decision.chosen_action)
                (
                    candidate_memory_view,
                    candidate_strategy_view,
                    retrieval_attempts,
                ) = execute_requested_retrievals(
                    requested_action_id=decision.chosen_action,
                    query=query,
                    memory_items=items,
                    memory_retriever=memory_retriever,
                    strategy_retriever=strategy_retriever,
                    memory_queries_by_source=queries,
                )
                expected_memory_ids = [
                    item.memory_id
                    for source in MemorySource
                    if source in sources
                    for item in discoveries[source].selected_items
                ]
                if [
                    item.memory_id for item in candidate_memory_view
                ] != expected_memory_ids:
                    raise RuntimeError(
                        "post-decision retrieval drifted from the candidate "
                        "described to PM"
                    )
                if evidence_filter_config is not None:
                    filtered = filter_evidence(
                        requested_action_id=decision.chosen_action,
                        current_user_text=runtime.current_user_text,
                        context_query_text=query,
                        memory_candidates=candidate_memory_view,
                        strategy_candidates=candidate_strategy_view,
                        config=evidence_filter_config,
                        session_index=runtime.session_index,
                        memory_helpfulness_model=memory_helpfulness_model,
                    )
                    memory_view = filtered.memory_view
                    strategy_view = filtered.strategy_view
                    filter_decision = filtered.decision
                else:
                    memory_view = candidate_memory_view
                    strategy_view = candidate_strategy_view
                    filter_decision = None
                realized_action_id = realized_action_id_from_evidence(
                    memory_view, strategy_view
                )
                system = supporter_generation_contract.system_prompt
                messages = generation_messages(
                    runtime, memory_view, strategy_view, system_prompt=system
                )
                messages_json = canonical_json(messages)
                raw_input_tokens_est = estimate_tokens(messages_json)
                input_tokens_est = conservative_token_bound(
                    messages_json,
                    safety_factor=float(input_token_safety_factor),
                )
                prompt_hash = sha256_text(messages_json)
                record_ids = {
                    "user_id": str(user["id"]),
                    "topic_index": int(topic["idx"]),
                    "condition": condition,
                    "seed": int(seed),
                    "simulator_id": simulator_id,
                    "turn_index": turn_index,
                    "interaction_mode": "fixed",
                    "track_id": str(track["track_id"]),
                    "fixed_seeker_generation_contract_sha256": (
                        fixed_seeker_treatment_sha256
                    ),
                }
                max_output_tokens = supporter_generation_contract.max_output_tokens
                call_key = physical_call_key(
                    stage="evoemo_pm_v2_supporter",
                    record_ids=record_ids,
                    prompt_sha256=prompt_hash,
                    endpoint=generator_endpoint,
                    request_parameters={
                        "temperature": supporter_generation_contract.temperature,
                        "max_tokens": max_output_tokens,
                        "seed": int(seed) + turn_index,
                        "response_schema": None,
                        "supporter_generation_treatment": supporter_treatment,
                        "supporter_generation_treatment_sha256": (
                            supporter_treatment_sha256
                        ),
                        "fixed_seeker_generation_treatment": fixed_seeker_treatment,
                        "fixed_seeker_generation_treatment_sha256": (
                            fixed_seeker_treatment_sha256
                        ),
                    },
                )
                call_plan.append(
                    {
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "seed": int(seed),
                        "simulator_id": simulator_id,
                        "turn_index": turn_index,
                        "condition": condition,
                        "chosen_action": decision.chosen_action,
                        "requested_action_id": decision.chosen_action,
                        "retrieval_attempts": [
                            row.model_dump(mode="json") for row in retrieval_attempts
                        ],
                        "realized_action_id": realized_action_id,
                        "effective_action_id": realized_action_id,
                        "candidate_memory_count": len(candidate_memory_view),
                        "kept_memory_count": len(memory_view),
                        "candidate_strategy_count": len(candidate_strategy_view),
                        "kept_strategy_count": len(strategy_view),
                        "evidence_filter_config_sha256": (
                            filter_decision.config_sha256
                            if filter_decision is not None
                            else None
                        ),
                        "input_tokens_est": input_tokens_est,
                        "raw_input_tokens_est": raw_input_tokens_est,
                        "max_output_tokens": max_output_tokens,
                        "max_http_attempts": 1,
                        "prompt_hash": prompt_hash,
                        "prompt_equivalence_id": prompt_hash,
                        "label_lineage_id": prompt_hash,
                        "call_key": call_key,
                        "supporter_generation_treatment": supporter_treatment,
                        "supporter_generation_treatment_sha256": (
                            supporter_treatment_sha256
                        ),
                        "fixed_seeker_generation_treatment": fixed_seeker_treatment,
                        "fixed_seeker_generation_treatment_sha256": (
                            fixed_seeker_treatment_sha256
                        ),
                    }
                )
    evaluation_preflight_rows = [
        row
        for row in preflight_rows
        if int(row["turn_index"]) in set(evaluation_turn_indices)
    ]
    evaluation_gate = summarize_action_preflight(
        evaluation_preflight_rows,
        condition=condition,
        gates=action_preflight_gates,
    )
    all_turn_diagnostic = summarize_action_preflight(
        preflight_rows,
        condition=condition,
        gates=action_preflight_gates,
    )
    external_score_diagnostics = (
        transparent_rule_score_diagnostics(preflight_states)
        if semantic_encoder is not None and preflight_states
        else {
            "status": (
                "UNAVAILABLE_NONCONTRACT_TEST_DOUBLE"
                if semantic_encoder is not None
                else "NOT_APPLICABLE_FIXED_ACTION_CONDITION"
            ),
            "state_count": len(preflight_states),
        }
    )
    development_external_score_comparison = (
        compare_development_external_score_diagnostics(
            development_score_diagnostics or {}, external_score_diagnostics
        )
        if semantic_encoder is not None
        else {
            "protocol": DEVELOPMENT_EXTERNAL_SCORE_COMPARISON_PROTOCOL,
            "status": "NOT_APPLICABLE_FIXED_ACTION_CONDITION",
            "outcome_labels_used": False,
            "external_threshold_selection_or_retuning_authorized": False,
        }
    )
    observable_state_support = (
        compare_external_observable_state_support(
            development=development_observable_support or {},
            external_states=evaluation_preflight_states,
        )
        if semantic_encoder is not None
        else {
            "protocol": (
                "pm-v1.5-development-external-observable-state-support-v1"
            ),
            "status": "NOT_APPLICABLE_FIXED_ACTION_CONDITION",
            "outcome_labels_used": False,
            "external_threshold_selection_or_retuning_authorized": False,
        }
    )
    write_json(comparison_path, development_external_score_comparison)

    preflight = {
        **evaluation_gate,
        "gate_scope": "frozen_evaluation_turns_only",
        "evaluation_turn_indices": list(evaluation_turn_indices),
        "evaluation_unit_contract": normalized_evaluation_unit_contract,
        "all_turn_diagnostic": {
            **all_turn_diagnostic,
            "gate_enforced": False,
            "scope": "all_fixed_track_turns",
            "turn_indices": list(range(1, int(max_turns) + 1)),
        },
        "semantic_catalog_refresh": {
            "scope": "one_local_source-centroid_refresh_per_external_user",
            "included_in_per_turn_step0_latency": False,
            "user_count": len(semantic_catalog_refresh_ms_by_user),
            "total_latency_ms": float(
                sum(semantic_catalog_refresh_ms_by_user.values())
            ),
            "per_user_latency_ms": semantic_catalog_refresh_ms_by_user,
            "api_calls": 0,
            "api_cost_usd": 0.0,
            "strategy_family_and_readiness_latency_ms": float(
                strategy_catalog_refresh_ms
            ),
        },
        "semantic_truncation": summarize_semantic_truncation_audits(
            [row.get("semantic_observation") or {} for row in preflight_rows]
        ),
        "semantic_runtime_verification": semantic_runtime_lineage,
        "step0_score_diagnostics": external_score_diagnostics,
        "development_external_score_comparison": (
            development_external_score_comparison
        ),
        "observable_state_support": observable_state_support,
    }
    preflight["semantic_truncation"]["current_user_text_gate"] = (
        "PASS"
        if (
            semantic_encoder is None
            or preflight["semantic_truncation"]["current_user_text_complete"]
        )
        else "FAIL"
    )
    preflight["semantic_truncation"]["implicit_visible_state_truncation_gate"] = (
        "PASS"
        if (
            semantic_encoder is None
            or preflight["semantic_truncation"][
                "implicit_visible_state_truncation_complete"
            ]
        )
        else "FAIL"
    )
    section_rows = preflight["semantic_truncation"].get(
        "section_allocation"
    ) or {}
    preflight["semantic_truncation"]["complete_section_allocation_gate"] = (
        "PASS"
        if (
            semantic_encoder is None
            or all(
                int((section_rows.get(name) or {}).get("state_count", -1))
                == len(preflight_states)
                for name in (
                    "current_user",
                    "session_summary",
                    "recent_dialogue",
                )
            )
        )
        else "FAIL"
    )
    if (
        preflight["semantic_truncation"]["current_user_text_gate"] != "PASS"
        or preflight["semantic_truncation"][
            "implicit_visible_state_truncation_gate"
        ]
        != "PASS"
        or preflight["semantic_truncation"][
            "complete_section_allocation_gate"
        ]
        != "PASS"
        or observable_state_support.get("status") not in {
            "PASS",
            "NOT_APPLICABLE_FIXED_ACTION_CONDITION",
        }
    ):
        preflight["status"] = "FAIL"
    write_json(preflight_path, preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError(
            "PM-v2 scoring-turn preflight failed before API calls: " + str(preflight)
        )

    call_plan_units = sorted(
        {
            (
                str(row["user_id"]),
                int(row["topic_index"]),
                int(row["seed"]),
                str(row["simulator_id"]),
                int(row["turn_index"]),
            )
            for row in call_plan
        }
    )
    if len(call_plan_units) != len(call_plan) or call_plan_units != frozen_evaluation_units:
        raise RuntimeError(
            "generation call plan does not exactly equal the frozen external unit universe"
        )

    input_counts = [int(row["input_tokens_est"]) for row in call_plan]
    total_input_tokens = sum(input_counts)
    total_output_tokens = sum(int(row["max_output_tokens"]) for row in call_plan)
    cost_payload = {
        "stage": "evoemo_pm_v2_generation",
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "condition": condition,
        "evo_memory_builder_contract_sha256": evo_memory_digest[
            "builder_contract_sha256"
        ],
        "evo_memory_global_catalog_sha256": evo_memory_digest[
            "global_catalog_sha256"
        ],
        "supporter_generation_treatment": supporter_treatment,
        "supporter_generation_treatment_sha256": supporter_treatment_sha256,
        "fixed_seeker_generation_treatment": fixed_seeker_treatment,
        "fixed_seeker_generation_treatment_sha256": fixed_seeker_treatment_sha256,
        "evaluation_unit_contract": normalized_evaluation_unit_contract,
        "call_plan_exactly_matches_frozen_evaluation_units": True,
        "input_token_safety_factor": float(input_token_safety_factor),
        "fail_on_reported_input_overrun": bool(fail_on_reported_input_overrun),
        "expected_api_calls": len(call_plan),
        "maximum_physical_http_attempts": len(call_plan),
        "total_input_tokens_est": total_input_tokens,
        "max_input_tokens_per_call_est": max(input_counts, default=0),
        "total_output_tokens_est": total_output_tokens,
        "estimated_cost_usd": total_input_tokens / 1_000_000 * float(input_usd_per_mtok)
        + total_output_tokens / 1_000_000 * float(output_usd_per_mtok),
        "generator_pricing_usd_per_mtok": generator_pricing_usd_per_mtok,
        "call_plan_sha256": sha256_text(canonical_json(call_plan)),
        "generator_endpoint": {
            "model": generator_endpoint.model,
            "family": generator_endpoint.family,
            "base_url": generator_endpoint.base_url,
        },
        "budget_limits": {
            "max_api_calls": int(max_api_calls),
            "max_estimated_usd": float(max_estimated_usd),
            "max_input_tokens_per_call": int(max_input_tokens_per_call),
        },
        "study_freeze_sha256": study_freeze_sha256,
        "maximum_cost_matched_relative_deviation": cost_match_tolerance,
        "cost_match_reference_condition": expected_reference_condition,
        "semantic_runtime_verification": semantic_runtime_lineage,
        "development_training_report_sha256": development_training_report_sha256,
        "development_external_score_comparison": (
            development_external_score_comparison
        ),
    }
    cost_estimate = {
        **cost_payload,
        "cost_estimate_sha256": sha256_text(canonical_json(cost_payload)),
    }
    budget_checks = {
        "api_calls": len(call_plan) <= int(max_api_calls),
        "estimated_cost_usd": cost_estimate["estimated_cost_usd"]
        <= float(max_estimated_usd),
        "max_input_tokens_per_call": cost_estimate["max_input_tokens_per_call_est"]
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
    call_keys = [str(row["call_key"]) for row in call_plan]
    if len(call_keys) != len(set(call_keys)):
        raise RuntimeError("PM-v2 generation call plan has duplicate logical call keys")
    ledger = PersistentAttemptLedger(
        attempt_ledger_path,
        stage="evoemo_pm_v2_supporter",
        expected_calls={call_key: 1 for call_key in call_keys},
        maximum_total_attempts=int(max_api_calls),
    )
    saved_dry_run = {**cost_estimate, "budget_gate": budget_gate}
    if ledger.started_attempts:
        if not call_plan_path.is_file() or not cost_estimate_path.is_file():
            raise RuntimeError(
                "spent PM-v2 generation ledger requires its immutable saved dry-run"
            )
        persisted_call_plan = list(iter_jsonl(call_plan_path))
        persisted_cost_estimate = read_json(cost_estimate_path)
        if persisted_call_plan != call_plan or persisted_cost_estimate != saved_dry_run:
            raise RuntimeError(
                "spent PM-v2 generation ledger forbids changing its call plan or "
                "cost estimate"
            )
    elif not run or budget_gate["status"] != "PASS":
        write_json(cost_estimate_path, {**cost_estimate, "budget_gate": budget_gate})
        write_jsonl(call_plan_path, call_plan)
    if budget_gate["status"] != "PASS":
        raise RuntimeError("PM-v2 generation budget gate failed before API calls")
    if not run:
        return {
            "status": "DRY_RUN_COMPLETE",
            "condition": condition,
            "preflight": preflight,
            "cost_estimate": cost_estimate,
            "budget_gate": budget_gate,
            "cost_estimate_path": str(cost_estimate_path),
            "call_plan_path": str(call_plan_path),
            "cost_match_estimated_preflight": None,
            "evaluation_unit_contract": normalized_evaluation_unit_contract,
        }
    if not cost_estimate_path.is_file() or not call_plan_path.is_file():
        raise RuntimeError("API run requires a matching saved generation dry-run")
    saved_estimate = read_json(cost_estimate_path)
    if saved_estimate.get("cost_estimate_sha256") != cost_estimate["cost_estimate_sha256"]:
        raise RuntimeError("saved PM-v2 generation dry-run is stale")
    saved_call_plan = list(iter_jsonl(call_plan_path))
    if sha256_text(canonical_json(saved_call_plan)) != cost_estimate[
        "call_plan_sha256"
    ] or saved_call_plan != call_plan:
        raise RuntimeError("saved PM-v2 generation call plan is stale")
    if accept_cost_estimate_sha256 != cost_estimate["cost_estimate_sha256"]:
        raise RuntimeError(
            "API run requires exact --accept-cost-estimate-sha256 from dry-run"
        )

    cost_match_estimated_preflight = None
    cost_match_preflight_path = out_dir / "cost_match_estimated_preflight.json"
    if cost_match_condition:
        reference_plan_path = Path(str(cost_match_reference_call_plan_path))
        reference_estimate_path = Path(str(cost_match_reference_cost_estimate_path))
        if not reference_plan_path.is_file() or not reference_estimate_path.is_file():
            raise RuntimeError(
                "both frozen cost-match dry-runs must exist before generation API calls"
            )
        reference_rows = list(iter_jsonl(reference_plan_path))
        reference_estimate = read_json(reference_estimate_path)
        reference_payload = {
            key: value
            for key, value in reference_estimate.items()
            if key not in {"cost_estimate_sha256", "budget_gate"}
        }
        if reference_estimate.get("cost_estimate_sha256") != sha256_text(
            canonical_json(reference_payload)
        ):
            raise RuntimeError("counterpart cost-match estimate self-hash mismatch")
        if (
            reference_estimate.get("stage") != "evoemo_pm_v2_generation"
            or reference_estimate.get("condition") != expected_reference_condition
            or reference_estimate.get("supporter_generation_treatment")
            != supporter_treatment
            or reference_estimate.get("supporter_generation_treatment_sha256")
            != supporter_treatment_sha256
            or reference_estimate.get("fixed_seeker_generation_treatment")
            != fixed_seeker_treatment
            or reference_estimate.get(
                "fixed_seeker_generation_treatment_sha256"
            )
            != fixed_seeker_treatment_sha256
            or reference_estimate.get("study_freeze_sha256") != study_freeze_sha256
            or float(
                reference_estimate.get(
                    "maximum_cost_matched_relative_deviation", -1.0
                )
            )
            != cost_match_tolerance
            or reference_estimate.get("cost_match_reference_condition") != condition
            or reference_estimate.get("evaluation_unit_contract")
            != normalized_evaluation_unit_contract
            or reference_estimate.get(
                "call_plan_exactly_matches_frozen_evaluation_units"
            )
            is not True
            or (reference_estimate.get("budget_gate") or {}).get("status") != "PASS"
        ):
            raise RuntimeError("counterpart cost-match estimate violates frozen contract")
        if reference_estimate.get("call_plan_sha256") != sha256_text(
            canonical_json(reference_rows)
        ):
            raise RuntimeError("counterpart cost-match call plan is stale")
        treatment_rows, baseline_rows = (
            (call_plan, reference_rows)
            if condition == COST_MATCH_TREATMENT
            else (reference_rows, call_plan)
        )
        cost_match_estimated_preflight = compare_cost_matched_token_rows(
            treatment_rows,
            baseline_rows,
            token_field="input_tokens_est",
            maximum_relative_deviation=cost_match_tolerance,
            stage="generation_dry_run_estimated_tokens",
        )
        cost_match_estimated_preflight.update(
            {
                "current_cost_estimate_sha256": cost_estimate[
                    "cost_estimate_sha256"
                ],
                "reference_cost_estimate_sha256": reference_estimate[
                    "cost_estimate_sha256"
                ],
            }
        )
        write_json(cost_match_preflight_path, cost_match_estimated_preflight)
        if cost_match_estimated_preflight["status"] != "PASS":
            raise RuntimeError(
                "cost-matched fixed baseline failed the frozen estimated-token gate"
            )

    dialogue_fields = (
        "user_id",
        "topic_index",
        "condition",
        "seed",
        "simulator_id",
        "interaction_mode",
    )
    turn_fields = (*dialogue_fields, "turn_index")
    expected_turns: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in call_plan:
        unit = (
            str(row["user_id"]),
            int(row["topic_index"]),
            condition,
            int(row["seed"]),
            str(row["simulator_id"]),
            "fixed",
            int(row["turn_index"]),
        )
        if unit in expected_turns:
            raise RuntimeError(f"duplicate PM-v2 generation call-plan unit: {unit}")
        expected_turns[unit] = row
    historical_attempts = ledger.started_attempts
    turn_index = (
        index_jsonl_unique(turn_path, turn_fields) if turn_path.exists() else {}
    )
    recovered_turns = reconcile_succeeded_generation_turns(
        ledger=ledger,
        expected_turns=expected_turns,
        turn_index=turn_index,
        turn_path=turn_path,
    )
    dialogue_index = (
        index_jsonl_unique(dialogue_path, dialogue_fields)
        if dialogue_path.exists()
        else {}
    )
    extra_turns = sorted(set(turn_index) - set(expected_turns))
    expected_dialogue_keys = {
        unit[:-1] for unit in expected_turns
    }
    extra_dialogues = sorted(set(dialogue_index) - expected_dialogue_keys)
    if extra_turns or extra_dialogues:
        raise RuntimeError(
            "PM-v2 generation outputs contain units outside the frozen call plan: "
            f"turns={extra_turns[:5]}, dialogues={extra_dialogues[:5]}"
        )
    for unit, row in turn_index.items():
        planned = expected_turns[unit]
        call_key = str(planned["call_key"])
        if (
            row.get("physical_call_key") != call_key
            or not ledger.succeeded(call_key)
            or row.get("supporter_generation_treatment") != supporter_treatment
            or row.get("supporter_generation_treatment_sha256")
            != supporter_treatment_sha256
            or row.get("fixed_seeker_generation_treatment")
            != fixed_seeker_treatment
            or row.get("fixed_seeker_generation_treatment_sha256")
            != fixed_seeker_treatment_sha256
            or row.get("normalized_finish_reason") != "complete"
        ):
            raise RuntimeError(
                "persisted turn lacks its successful physical-attempt/treatment "
                f"binding: {unit}"
            )
    for unit in dialogue_index:
        covered = [key for key in turn_index if key[:-1] == unit]
        covered_turn_indices = sorted(int(key[-1]) for key in covered)
        if covered_turn_indices != list(evaluation_turn_indices):
            raise RuntimeError(
                "persisted dialogue does not have the exact sparse frozen turn "
                f"matrix: {unit}; observed={covered_turn_indices}"
            )

    generator: OpenAICompatibleClient | None = None
    new_attempts = 0
    aborted_on_input_token_overrun = False
    aborted_on_usage_accounting_error = False
    aborted_on_completion_gate_error = False
    failures: list[dict[str, Any]] = [
        {
            **dict(row.get("record_ids") or {}),
            "physical_call_key": row["call_key"],
            "physical_attempt_index": row["attempt_index"],
            "error": row.get("error"),
            "historical": True,
        }
        for row in ledger.failures()
    ]
    try:
        for user, topic in scenarios:
            items, _ = compile_bounded_memory(user)
            source_centroids = source_centroids_for_user(str(user["id"]), items)
            for seed in seeds:
                track_key = _track_key(
                    str(user["id"]), int(topic["idx"]), int(seed), simulator_id
                )
                fixed_track = tracks.get(track_key)
                if fixed_track is None:
                    raise RuntimeError(f"missing fixed seeker track: {track_key}")
                dialogue_key = (
                    str(user["id"]),
                    int(topic["idx"]),
                    condition,
                    int(seed),
                    simulator_id,
                    "fixed",
                )
                track_id = str(fixed_track["track_id"])
                for turn_number in evaluation_turn_indices:
                    unit = (*dialogue_key, turn_number)
                    if unit in turn_index:
                        continue
                    planned = expected_turns[unit]
                    call_key = str(planned["call_key"])
                    if ledger.succeeded(call_key):
                        raise RuntimeError(
                            "successful PM-v2 generation call was not reconciled "
                            f"from its durable terminal result: {unit}"
                        )
                    if ledger.exhausted(call_key):
                        failures.append(
                            {
                                "unit": list(unit),
                                "physical_call_key": call_key,
                                "error": "physical-attempt bound already exhausted",
                            }
                        )
                        continue
                    seeker_message = normalize_space(
                        fixed_track["seeker_turns"][turn_number - 1]
                    )
                    state_conversation = _fixed_context_before_turn(
                        fixed_track, turn_number
                    )
                    step0_start = time.perf_counter()
                    runtime = make_evo_runtime_state(
                        user,
                        topic,
                        state_conversation,
                        seeker_message,
                        items,
                        turn_number,
                        condition,
                        track_id=track_id,
                        fixed_open_loop=True,
                        semantic_encoder=(
                            semantic_encoder if requires_step0_observation else None
                        ),
                        semantic_source_centroids=source_centroids,
                    )
                    pm_state = runtime_to_pmv2_state(
                        runtime,
                        strategy_catalog_count=len(strategy_cards),
                        strategy_estimated_tokens=strategy_action_tokens,
                        strategy_family_catalog=strategy_family_catalog,
                        include_step0_observation=requires_step0_observation,
                        semantic_encoder=(
                            semantic_encoder if requires_step0_observation else None
                        ),
                    )
                    pre_evidence_ms = (
                        (time.perf_counter() - step0_start) * 1000.0
                        if requires_step0_observation
                        else 0.0
                    )
                    queries = source_specific_memory_queries(
                        runtime.current_user_text,
                        [
                            row.model_dump(mode="json")
                            for row in runtime.current_session_history
                        ],
                        runtime.current_session_summary,
                    )
                    query = queries[MemorySource.ME]
                    discoveries = discover_memory_candidates_by_source_query(
                        queries=queries,
                        items=items,
                        retriever=memory_retriever,
                        session_index=runtime.session_index,
                    )
                    pm_start = time.perf_counter()
                    (
                        pm_state,
                        decision,
                    ) = choose_after_candidate_discovery(
                        model=model,
                        pm_state=pm_state,
                        discoveries=discoveries,
                    )
                    pm_ms = (time.perf_counter() - pm_start) * 1000.0
                    action_id = decision.chosen_action
                    sources, strategy = parse_action_id(action_id)
                    (
                        candidate_memory_view,
                        candidate_strategy_view,
                        retrieval_attempts,
                    ) = execute_requested_retrievals(
                        requested_action_id=action_id,
                        query=query,
                        memory_items=items,
                        memory_retriever=memory_retriever,
                        strategy_retriever=strategy_retriever,
                        memory_queries_by_source=queries,
                    )
                    expected_memory_ids = [
                        item.memory_id
                        for source in MemorySource
                        if source in sources
                        for item in discoveries[source].selected_items
                    ]
                    if [
                        item.memory_id for item in candidate_memory_view
                    ] != expected_memory_ids:
                        raise RuntimeError(
                            "post-decision retrieval drifted from the "
                            "candidate described to PM"
                        )
                    retrieval_ms = sum(
                        row.latency_ms for row in retrieval_attempts if row.called
                    )
                    filter_start = time.perf_counter()
                    if evidence_filter_config is not None:
                        filtered = filter_evidence(
                            requested_action_id=action_id,
                            current_user_text=runtime.current_user_text,
                            context_query_text=query,
                            memory_candidates=candidate_memory_view,
                            strategy_candidates=candidate_strategy_view,
                            config=evidence_filter_config,
                            session_index=runtime.session_index,
                            memory_helpfulness_model=memory_helpfulness_model,
                        )
                        memory_view = filtered.memory_view
                        strategy_view = filtered.strategy_view
                        filter_decision = filtered.decision
                    else:
                        memory_view = candidate_memory_view
                        strategy_view = candidate_strategy_view
                        filter_decision = None
                    realized_action_id = realized_action_id_from_evidence(
                        memory_view, strategy_view
                    )
                    filter_ms = (time.perf_counter() - filter_start) * 1000.0
                    system = supporter_generation_contract.system_prompt
                    messages = generation_messages(
                        runtime, memory_view, strategy_view, system_prompt=system
                    )
                    prompt_hash = sha256_text(canonical_json(messages))
                    record_ids = {
                        "user_id": str(user["id"]),
                        "topic_index": int(topic["idx"]),
                        "condition": condition,
                        "seed": int(seed),
                        "simulator_id": simulator_id,
                        "turn_index": turn_number,
                        "interaction_mode": "fixed",
                        "track_id": track_id,
                        "fixed_seeker_generation_contract_sha256": (
                            fixed_seeker_treatment_sha256
                        ),
                    }
                    recomputed_call_key = physical_call_key(
                        stage="evoemo_pm_v2_supporter",
                        record_ids=record_ids,
                        prompt_sha256=prompt_hash,
                        endpoint=generator_endpoint,
                        request_parameters={
                            "temperature": supporter_generation_contract.temperature,
                            "max_tokens": int(planned["max_output_tokens"]),
                            "seed": int(seed) + turn_number,
                            "response_schema": None,
                            "supporter_generation_treatment": supporter_treatment,
                            "supporter_generation_treatment_sha256": (
                                supporter_treatment_sha256
                            ),
                            "fixed_seeker_generation_treatment": (
                                fixed_seeker_treatment
                            ),
                            "fixed_seeker_generation_treatment_sha256": (
                                fixed_seeker_treatment_sha256
                            ),
                        },
                    )
                    if (
                        recomputed_call_key != call_key
                        or prompt_hash != planned["prompt_hash"]
                        or planned.get("supporter_generation_treatment")
                        != supporter_treatment
                        or planned.get("supporter_generation_treatment_sha256")
                        != supporter_treatment_sha256
                        or planned.get("fixed_seeker_generation_treatment")
                        != fixed_seeker_treatment
                        or planned.get(
                            "fixed_seeker_generation_treatment_sha256"
                        )
                        != fixed_seeker_treatment_sha256
                        or int(planned["max_output_tokens"])
                        != supporter_generation_contract.max_output_tokens
                    ):
                        raise RuntimeError(
                            f"runtime PM-v2 call differs from frozen dry-run: {unit}"
                        )
                    if generator is None:
                        generator = OpenAICompatibleClient(generator_endpoint)
                    reservation = ledger.reserve(
                        call_key,
                        record_ids=record_ids,
                        prompt_sha256=prompt_hash,
                    )
                    new_attempts += 1
                    result = None
                    try:
                        result, _ = generator.chat(
                            messages,
                            temperature=supporter_generation_contract.temperature,
                            max_tokens=int(planned["max_output_tokens"]),
                            seed=int(seed) + turn_number,
                            response_schema=None,
                            retries=1,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        failed_request_log = bind_external_generation_request_log(
                            request_log(
                                stage="evoemo_pm_v2_supporter",
                                endpoint=generator_endpoint,
                                messages=messages,
                                result=result,
                                parsed=None,
                                error=error,
                                prompt_hash=prompt_hash,
                                record_ids={
                                    **record_ids,
                                    "physical_call_key": call_key,
                                    "physical_attempt_index": reservation.attempt_index,
                                    "physical_attempt_key": reservation.attempt_key,
                                },
                            ),
                            supporter_generation_treatment=supporter_treatment,
                            supporter_generation_treatment_sha256=(
                                supporter_treatment_sha256
                            ),
                            fixed_seeker_generation_treatment=(
                                fixed_seeker_treatment
                            ),
                            fixed_seeker_generation_treatment_sha256=(
                                fixed_seeker_treatment_sha256
                            ),
                        )
                        append_jsonl(raw_path, failed_request_log)
                        ledger.finish(
                            reservation,
                            succeeded=False,
                            request_hash=None,
                            usage=None,
                            error=error,
                            result={"request_log": failed_request_log},
                        )
                        failures.append(
                            {
                                **record_ids,
                                "physical_call_key": call_key,
                                "physical_attempt_index": reservation.attempt_index,
                                "error": error,
                            }
                        )
                        continue

                    usage_error = reported_prompt_token_error(
                        result.usage,
                        maximum_prompt_tokens=int(planned["input_tokens_est"]),
                        stage="EvoEmo supporter generation",
                        require_positive=bool(fail_on_reported_input_overrun),
                    )
                    completion_error = (
                        supporter_generation_contract.completion_gate_error(
                            normalized_finish_reason=result.normalized_finish_reason,
                            provider_finish_reason=result.provider_finish_reason,
                        )
                    )
                    response_error = completion_error or usage_error
                    generation_request_log = bind_external_generation_request_log(
                        request_log(
                            stage="evoemo_pm_v2_supporter",
                            endpoint=generator_endpoint,
                            messages=messages,
                            result=result,
                            parsed=None,
                            error=response_error,
                            prompt_hash=prompt_hash,
                            record_ids={
                                **record_ids,
                                "physical_call_key": call_key,
                                "physical_attempt_index": reservation.attempt_index,
                                "physical_attempt_key": reservation.attempt_key,
                            },
                        ),
                        supporter_generation_treatment=supporter_treatment,
                        supporter_generation_treatment_sha256=(
                            supporter_treatment_sha256
                        ),
                        fixed_seeker_generation_treatment=fixed_seeker_treatment,
                        fixed_seeker_generation_treatment_sha256=(
                            fixed_seeker_treatment_sha256
                        ),
                    )
                    append_jsonl(raw_path, generation_request_log)
                    reported_prompt_tokens = int(
                        (result.usage or {}).get("prompt_tokens") or 0
                    )
                    if completion_error is not None:
                        error = f"{completion_error}, unit={unit}"
                        ledger.finish(
                            reservation,
                            succeeded=False,
                            request_hash=result.request_hash,
                            usage=result.usage,
                            error=error,
                            result={"request_log": generation_request_log},
                        )
                        failures.append(
                            {
                                **record_ids,
                                "physical_call_key": call_key,
                                "physical_attempt_index": reservation.attempt_index,
                                "provider_finish_reason": result.provider_finish_reason,
                                "normalized_finish_reason": (
                                    result.normalized_finish_reason
                                ),
                                "completion_gate_failed": True,
                                "error": error,
                            }
                        )
                        aborted_on_completion_gate_error = True
                        break
                    if usage_error is not None:
                        error = f"{usage_error}, unit={unit}"
                        ledger.finish(
                            reservation,
                            succeeded=False,
                            request_hash=result.request_hash,
                            usage=result.usage,
                            error=error,
                            result={"request_log": generation_request_log},
                        )
                        failures.append(
                            {
                                **record_ids,
                                "physical_call_key": call_key,
                                "physical_attempt_index": reservation.attempt_index,
                                "reported_input_token_overrun": (
                                    reported_prompt_tokens
                                    > int(planned["input_tokens_est"])
                                ),
                                "missing_or_nonpositive_reported_usage": (
                                    reported_prompt_tokens <= 0
                                ),
                                "error": error,
                            }
                        )
                        aborted_on_input_token_overrun = (
                            reported_prompt_tokens
                            > int(planned["input_tokens_est"])
                        )
                        aborted_on_usage_accounting_error = True
                        break
                    supporter_message = supporter_generation_contract.normalize_output(
                        result.text
                    )
                    memory_tokens = sum(estimate_tokens(row.text) for row in memory_view)
                    strategy_tokens = sum(
                        estimate_tokens(row.guidance_text + row.example_response)
                        for row in strategy_view
                    )
                    base_tokens = estimate_tokens(
                        system
                        + runtime.current_user_text
                        + runtime.current_session_summary
                        + "\n".join(
                            row.content for row in runtime.current_session_history
                        )
                    )
                    visible_state_tokens = estimate_tokens(
                        runtime.current_user_text
                        + runtime.current_session_summary
                        + "\n".join(
                            row.content for row in runtime.current_session_history
                        )
                    )
                    semantic_views = (
                        (
                            (
                                getattr(pm_state, "provenance", {}) or {}
                            ).get("semantic_observation")
                            or {}
                        ).get("tokenization")
                        or {}
                    ).get("views") or {}
                    semantic_visible_tokens = int(
                        (
                            semantic_views.get("visible_dialogue_state")
                            or {}
                        ).get("original_token_count", 0)
                    )
                    semantic_current_tokens = int(
                        (semantic_views.get("current_user_text") or {}).get(
                            "original_token_count", 0
                        )
                    )
                    if semantic_encoder is not None and requires_step0_observation and (
                        semantic_visible_tokens <= 0
                        or semantic_current_tokens <= 0
                    ):
                        raise RuntimeError(
                            "reportable Step-0 cost lacks semantic token telemetry"
                        )
                    cost = CostRecord(
                        pm_input_tokens_est=(
                            visible_state_tokens
                            if requires_step0_observation
                            else 0
                        ),
                        catalog_reads=(
                            len(runtime.inventory)
                            if requires_step0_observation
                            else 0
                        ),
                        step0_memory_comparisons=(
                            len(runtime.inventory)
                            if requires_step0_observation
                            else 0
                        ),
                        step0_strategy_family_comparisons=(
                            len(strategy_family_catalog.vectors)
                            if requires_step0_observation
                            else 0
                        ),
                        step0_advice_readiness_comparisons=(
                            len(strategy_family_catalog.readiness_vectors)
                            if requires_step0_observation
                            else 0
                        ),
                        step0_encoder_input_tokens_est=(
                            2 * semantic_visible_tokens
                            + 2 * semantic_current_tokens
                            if requires_step0_observation
                            and semantic_encoder is not None
                            else 0
                        ),
                        step0_encoder_invocations=(
                            2
                            if requires_step0_observation
                            and semantic_encoder is not None
                            else 0
                        ),
                        step0_latency_ms=pre_evidence_ms,
                        pre_evidence_compute_ms=pre_evidence_ms,
                        pm_inference_ms=pm_ms,
                        retrieval_latency_ms=retrieval_ms,
                        generation_latency_ms=result.latency_ms,
                        retrieval_calls=len(sources)
                        + (1 if strategy is StrategyMode.RS else 0),
                        reranker_calls=0,
                        memory_tokens=memory_tokens,
                        strategy_tokens=strategy_tokens,
                        base_prompt_tokens=base_tokens,
                        total_input_tokens=(
                            result.usage["prompt_tokens"]
                            or base_tokens + memory_tokens + strategy_tokens
                        ),
                        output_tokens=(
                            result.usage["completion_tokens"]
                            or estimate_tokens(supporter_message)
                        ),
                        latency_ms=(
                            pre_evidence_ms
                            + pm_ms
                            + retrieval_ms
                            + filter_ms
                            + result.latency_ms
                        ),
                        api_cost_usd=(
                            (
                                result.usage["prompt_tokens"]
                                or base_tokens + memory_tokens + strategy_tokens
                            )
                            / 1_000_000
                            * float(input_usd_per_mtok)
                            + (
                                result.usage["completion_tokens"]
                                or estimate_tokens(supporter_message)
                            )
                            / 1_000_000
                            * float(output_usd_per_mtok)
                        ),
                        evidence_filter_calls=(1 if filter_decision is not None else 0),
                        evidence_filter_latency_ms=filter_ms,
                        candidate_memory_count=len(candidate_memory_view),
                        kept_memory_count=len(memory_view),
                        candidate_strategy_count=len(candidate_strategy_view),
                        kept_strategy_count=len(strategy_view),
                        candidate_memory_tokens=sum(
                            estimate_tokens(row.text) for row in candidate_memory_view
                        ),
                        candidate_strategy_tokens=sum(
                            estimate_tokens(row.guidance_text + row.example_response)
                            for row in candidate_strategy_view
                        ),
                        dropped_memory_tokens=sum(
                            estimate_tokens(row.text) for row in candidate_memory_view
                        )
                        - memory_tokens,
                        dropped_strategy_tokens=sum(
                            estimate_tokens(row.guidance_text + row.example_response)
                            for row in candidate_strategy_view
                        )
                        - strategy_tokens,
                    )
                    turn_record = {
                        **record_ids,
                        "protocol": supporter_generation_contract.version,
                        "trajectory_comparability": "causal_fixed_context_one_step",
                        "state_id": runtime.state_id,
                        "exogenous_state_id": runtime.provenance["exogenous_state_id"],
                        "card_id": runtime.card_id,
                        "context_before_turn": state_conversation,
                        "context_sha256": sha256_text(
                            canonical_json(state_conversation)
                        ),
                        "seeker_message": seeker_message,
                        "supporter_message": supporter_message,
                        "action_id": action_id,
                        "requested_action_id": action_id,
                        "retrieval_attempts": [
                            row.model_dump(mode="json") for row in retrieval_attempts
                        ],
                        "realized_action_id": realized_action_id,
                        "effective_action_id": realized_action_id,
                        "prompt_equivalence_id": prompt_hash,
                        "label_lineage_id": prompt_hash,
                        "candidate_memory": [
                            row.model_dump(mode="json")
                            for row in candidate_memory_view
                        ],
                        "candidate_strategy": [
                            row.model_dump(mode="json")
                            for row in candidate_strategy_view
                        ],
                        "evidence_filter_decision": (
                            filter_decision.model_dump(mode="json")
                            if filter_decision is not None
                            else None
                        ),
                        "selected_memory": [
                            row.model_dump(mode="json") for row in memory_view
                        ],
                        "selected_strategy": [
                            row.model_dump(mode="json") for row in strategy_view
                        ],
                        "cost": cost.model_dump(mode="json"),
                        "input_tokens": cost.total_input_tokens,
                        "output_tokens": cost.output_tokens,
                        "latency_ms": cost.latency_ms,
                        "pm_v2_decision": decision.model_dump(mode="json"),
                        "provider_finish_reason": result.provider_finish_reason,
                        "normalized_finish_reason": result.normalized_finish_reason,
                        "supporter_generation_treatment": supporter_treatment,
                        "supporter_generation_treatment_sha256": (
                            supporter_treatment_sha256
                        ),
                        "fixed_seeker_generation_treatment": fixed_seeker_treatment,
                        "fixed_seeker_generation_treatment_sha256": (
                            fixed_seeker_treatment_sha256
                        ),
                        "physical_call_key": call_key,
                        "physical_attempt_index": reservation.attempt_index,
                        "physical_attempt_key": reservation.attempt_key,
                    }
                    ledger.finish(
                        reservation,
                        succeeded=True,
                        request_hash=result.request_hash,
                        usage=result.usage,
                        error=None,
                        result={
                            "turn_record": turn_record,
                            "request_log": generation_request_log,
                        },
                    )
                    append_jsonl(turn_path, turn_record)
                    turn_index[unit] = turn_record

                if (
                    aborted_on_usage_accounting_error
                    or aborted_on_completion_gate_error
                ):
                    break

                track_turns = [
                    turn_index[(*dialogue_key, turn_number)]
                    for turn_number in evaluation_turn_indices
                    if (*dialogue_key, turn_number) in turn_index
                ]
                if (
                    len(track_turns) != len(evaluation_turn_indices)
                    or dialogue_key in dialogue_index
                ):
                    continue
                conversation: list[dict[str, str]] = [
                    {"role": "supporter", "content": NEUTRAL_INITIAL_GREETING}
                ]
                evaluation_cases: list[dict[str, Any]] = []
                for turn_record in track_turns:
                    conversation.extend(
                        [
                            {"role": "seeker", "content": turn_record["seeker_message"]},
                            {
                                "role": "supporter",
                                "content": turn_record["supporter_message"],
                            },
                        ]
                    )
                    evaluation_cases.append(
                        {
                            "turn_index": turn_record["turn_index"],
                            "context_before_turn": turn_record["context_before_turn"],
                            "current_seeker_message": turn_record["seeker_message"],
                            "supporter_response": turn_record["supporter_message"],
                            "context_sha256": turn_record["context_sha256"],
                        }
                    )
                dialogue_record = {
                    "user_id": str(user["id"]),
                    "topic_index": int(topic["idx"]),
                    "condition": condition,
                    "protocol": supporter_generation_contract.version,
                    "supporter_generation_treatment": supporter_treatment,
                    "supporter_generation_treatment_sha256": (
                        supporter_treatment_sha256
                    ),
                    "fixed_seeker_generation_treatment": fixed_seeker_treatment,
                    "fixed_seeker_generation_treatment_sha256": (
                        fixed_seeker_treatment_sha256
                    ),
                    "interaction_mode": "fixed",
                    "trajectory_comparability": "causal_fixed_context_one_step",
                    "simulator_id": simulator_id,
                    "track_id": track_id,
                    "seed": int(seed),
                    "initial_greeting": NEUTRAL_INITIAL_GREETING,
                    "dialogue": conversation,
                    "dialogue_semantics": (
                        "display_only_sparse_fixed_context_evaluation_cases_not_"
                        "a_contiguous_trajectory"
                    ),
                    "evaluation_turn_indices": list(evaluation_turn_indices),
                    "evaluation_cases": evaluation_cases,
                    "turns": track_turns,
                }
                append_jsonl(dialogue_path, dialogue_record)
                dialogue_index[dialogue_key] = dialogue_record
            if aborted_on_usage_accounting_error or aborted_on_completion_gate_error:
                break
    finally:
        if generator is not None:
            generator.close()

    expected = len(scenarios) * len(seeds)
    completed = len(dialogue_index)
    raw_rows = list(iter_jsonl(raw_path)) if raw_path.is_file() else []
    raw_generation_contract_gate = summarize_external_generation_raw_matrix(
        raw_rows,
        planned_call_keys=call_keys,
        supporter_generation_treatment=supporter_treatment,
        supporter_generation_treatment_sha256=supporter_treatment_sha256,
        fixed_seeker_generation_treatment=fixed_seeker_treatment,
        fixed_seeker_generation_treatment_sha256=(
            fixed_seeker_treatment_sha256
        ),
    )
    summary = {
        "status": (
            "COMPLETE"
            if (
                completed == expected
                and len(turn_index) == len(expected_turns)
                and raw_generation_contract_gate["status"] == "PASS"
            )
            else "INCOMPLETE"
        ),
        "condition": condition,
        "supporter_generation_treatment": supporter_treatment,
        "supporter_generation_treatment_sha256": supporter_treatment_sha256,
        "fixed_seeker_generation_treatment": fixed_seeker_treatment,
        "fixed_seeker_generation_treatment_sha256": fixed_seeker_treatment_sha256,
        "expected_dialogues": expected,
        "completed_dialogues": completed,
        "expected_turns": len(expected_turns),
        "completed_turns": len(turn_index),
        "evaluation_unit_contract": normalized_evaluation_unit_contract,
        "generated_unit_ids_exactly_match_external_expected_units": (
            sorted(
                (
                    str(unit[0]),
                    int(unit[1]),
                    int(unit[3]),
                    str(unit[4]),
                    int(unit[6]),
                )
                for unit in turn_index
            )
            == frozen_evaluation_units
        ),
        "new_physical_http_attempts": new_attempts,
        "recovered_turns_from_success_ledger": recovered_turns,
        "historical_physical_http_attempts": historical_attempts,
        "total_physical_http_attempts": ledger.started_attempts,
        "remaining_runtime_physical_http_attempts": ledger.remaining_attempts,
        "maximum_physical_http_attempts_authorized": int(max_api_calls),
        "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
        "input_token_safety_factor": float(input_token_safety_factor),
        "fail_on_reported_input_overrun": bool(fail_on_reported_input_overrun),
        "aborted_on_input_token_overrun": aborted_on_input_token_overrun,
        "aborted_on_usage_accounting_error": aborted_on_usage_accounting_error,
        "aborted_on_completion_gate_error": aborted_on_completion_gate_error,
        "raw_generation_contract_gate": raw_generation_contract_gate,
        "non_complete_finish_reason_count": raw_generation_contract_gate[
            "non_complete_finish_reason_count"
        ],
        "failures": failures,
        "preflight": preflight,
        "cost_estimate": cost_estimate,
        "budget_gate": budget_gate,
        "cost_match_estimated_preflight": cost_match_estimated_preflight,
        "semantic_runtime_verification": semantic_runtime_lineage,
        "development_training_report_sha256": development_training_report_sha256,
        "development_external_score_comparison": (
            development_external_score_comparison
        ),
    }
    write_json(summary_path, summary)
    if summary["status"] != "COMPLETE":
        raise RuntimeError("PM-v2 EvoEmo generation incomplete: " + str(summary))
    attestation_inputs = {
        "evoemo": evoemo_path,
        "strategy_bank": strategy_bank_path,
        "checkpoint": checkpoint_path,
        "fixed_tracks": fixed_tracks_path,
        "fixed_tracks_attestation": fixed_tracks_attestation_path,
        "run_manifest": manifest_path,
        "cost_estimate": cost_estimate_path,
        "call_plan": call_plan_path,
    }
    if development_training_report_path is not None:
        attestation_inputs["development_training_report"] = (
            development_training_report_path
        )
    attestation_outputs = {
        "dialogues": (dialogue_path, True),
        "turns": (turn_path, True),
        "raw_calls": (raw_path, True),
        "physical_attempt_ledger": (attempt_ledger_path, True),
        "summary": (summary_path, False),
        "preflight": (preflight_path, False),
        "semantic_distribution_comparison": (comparison_path, False),
    }
    if cost_match_condition:
        attestation_inputs.update(
            {
                "cost_match_reference_call_plan": cost_match_reference_call_plan_path,
                "cost_match_reference_cost_estimate": cost_match_reference_cost_estimate_path,
            }
        )
        attestation_outputs["cost_match_estimated_preflight"] = (
            cost_match_preflight_path,
            False,
        )
    generator_endpoint_sha256 = sha256_text(
        canonical_json(
            {
                "model": generator_endpoint.model,
                "family": generator_endpoint.family,
                "base_url": generator_endpoint.base_url,
            }
        )
    )
    response_mechanism_contract = build_response_mechanism_contract(
        project_root=project_root,
        supporter_generation_contract=supporter_generation_contract,
        generator_endpoint_sha256=generator_endpoint_sha256,
        strategy_bank_sha256=sha256_file(strategy_bank_path),
        memory_min_score=memory_min_score,
        strategy_min_score=strategy_min_score,
        strategy_top_k=int(strategy_top_k),
        evidence_filter_enabled=bool(
            evidence_filter_config.enabled if evidence_filter_config is not None else False
        ),
    )
    create_artifact_attestation(
        attestation_path,
        stage="evoemo_pm_v2_generation",
        inputs=attestation_inputs,
        outputs=attestation_outputs,
        parameters={
            "condition": condition,
            "protocol": supporter_generation_contract.version,
            "evo_memory_builder_contract_sha256": evo_memory_digest[
                "builder_contract_sha256"
            ],
            "evo_memory_global_catalog_sha256": evo_memory_digest[
                "global_catalog_sha256"
            ],
            "response_mechanism_contract": response_mechanism_contract,
            "supporter_generation_treatment": supporter_treatment,
            "supporter_generation_treatment_sha256": supporter_treatment_sha256,
            "fixed_seeker_generation_treatment": fixed_seeker_treatment,
            "fixed_seeker_generation_treatment_sha256": (
                fixed_seeker_treatment_sha256
            ),
            "simulator_id": simulator_id,
            "max_turns": max_turns,
            "seeds": [int(seed) for seed in seeds],
            "evaluation_unit_contract": normalized_evaluation_unit_contract,
            "paid_generation_scope": "frozen_evaluation_turns_only",
            "action_preflight_gate_scope": "frozen_evaluation_turns_only",
            "all_turn_action_preflight_is_diagnostic_only": True,
            "selection_config_hash": model.selection_config.digest(),
            "generator_endpoint_sha256": generator_endpoint_sha256,
            "strategy_action_tokens": int(strategy_action_tokens),
            "strategy_top_k": int(strategy_top_k),
            "memory_min_score": memory_min_score,
            "strategy_min_score": strategy_min_score,
            "evidence_filter": (
                evidence_filter_config.payload()
                if evidence_filter_config is not None
                else None
            ),
            "evidence_filter_config_sha256": (
                evidence_filter_config.digest()
                if evidence_filter_config is not None
                else None
            ),
            "evidence_filter_model": effective_evidence_filter_model_binding,
            "action_preflight_gates": dict(action_preflight_gates),
            "maximum_cost_matched_relative_deviation": cost_match_tolerance,
            "cost_match_reference_condition": expected_reference_condition,
            "cost_match_estimated_preflight": cost_match_estimated_preflight,
            "cost_estimate_sha256": cost_estimate["cost_estimate_sha256"],
            "generator_retries": 1,
            "generator_pricing_usd_per_mtok": generator_pricing_usd_per_mtok,
            "input_token_safety_factor": float(input_token_safety_factor),
            "fail_on_reported_input_overrun": bool(
                fail_on_reported_input_overrun
            ),
            "physical_attempt_ledger_protocol": PHYSICAL_ATTEMPT_LEDGER_PROTOCOL,
            "maximum_physical_http_attempts_authorized": int(max_api_calls),
            "raw_generation_contract_gate": raw_generation_contract_gate,
            "semantic_runtime_verification": semantic_runtime_lineage,
            "development_training_report_sha256": (
                development_training_report_sha256
            ),
            "development_external_score_comparison": (
                development_external_score_comparison
            ),
        },
        expected={
            "dialogues": expected,
            "turns": len(frozen_evaluation_units),
            "raw_calls": len(frozen_evaluation_units),
            "non_complete_finish_reason_count": 0,
            "raw_generation_contract_gate_status": "PASS",
        },
        study_freeze_sha256=study_freeze_sha256,
    )
    return summary
